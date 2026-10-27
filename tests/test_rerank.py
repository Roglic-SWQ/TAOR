import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from mmkgc.rerank import RerankError, main, rerank_run


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeRerankEngine:
    def __init__(self) -> None:
        self.contexts = []
        self.profile = type("Profile", (), {"profile_name": "fake-rerank-profile"})()

    def rank(self, context):
        self.contexts.append(context)
        image_input = context.generated_image_inputs["image_001"]
        assert Path(image_input["image_path"]).is_file()
        return {
            "ranking_status": "completed",
            "profile_name": "fake-rerank-profile",
            "score_version": "stage4_structured_graph_v3",
            "text_ranking": {
                "ranked_candidates": [
                    {"candidate_id": "text_001", "final_score": 0.9}
                ],
                "best_candidate_id": "text_001",
            },
            "image_ranking": {
                "ranked_candidates": [
                    {"candidate_id": "image_001", "final_score": 0.8}
                ],
                "best_candidate_id": "image_001",
            },
            "pair_ranking": {
                "ranked_pairs": [],
                "top_5_pairs": [],
                "best_pair_id": None,
                "best_pair": None,
            },
            "kg_aware_candidate_consistency_ranking": {
                "ground_truth_used": False,
            },
            "ranking_model_calls": [
                {
                    "scorer": "fake",
                    "status": "scored",
                    "candidate_id": "text_001",
                }
            ],
        }


class FailingDiagnosticEngine:
    def __init__(self) -> None:
        self.profile = type("Profile", (), {"profile_name": "fake-rerank-profile"})()

    def rank(self, context):
        del context
        exc = RuntimeError("qwen judge timeout")
        exc.diagnostic_events = [
            {
                "scorer": "gpt4omini_judge",
                "kind": "image",
                "candidate_id": "image_001",
                "status": "failed",
                "details": {"error_type": "TimeoutError"},
            }
        ]
        raise exc


def test_rerank_run_writes_isolated_outputs_and_preserves_source_run(tmp_path):
    run_dir = _write_source_run(tmp_path / "source_run")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"samples": [{"entity_key": "Q1", "reason": "unit"}]}),
        encoding="utf-8",
    )
    engine = FakeRerankEngine()

    summary = rerank_run(
        run_dir,
        sample_manifest=manifest,
        stage4_config="configs/stage4-model-fusion.yaml",
        engine=engine,
    )

    output_dir = run_dir / "stage4_model_fusion_rerank"
    assert summary["processed_entities"] == 1
    assert summary["failed_entities"] == 0
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "sample_manifest.json").is_file()
    assert (output_dir / "results.jsonl").is_file()
    assert (output_dir / "metrics_summary.json").is_file()

    entity_dir = output_dir / "entities" / "Q1"
    assert (entity_dir / "4_kg_aware_candidate_consistency_ranking.json").is_file()
    assert (entity_dir / "final.json").is_file()
    assert (entity_dir / "ranking_model_calls.jsonl").is_file()
    assert not (entity_dir / "0_coarse_relation_filtering.json").exists()

    final = json.loads((entity_dir / "final.json").read_text(encoding="utf-8"))
    assert final["stage4_ranker"] == "model-fusion"
    assert final["stage4_profile_name"] == "fake-rerank-profile"
    assert final["ground_truth_used"] is False
    assert final["best_pair_id"] is None
    assert json.loads((run_dir / "entities" / "Q1" / "final.json").read_text())[
        "stage4_ranker"
    ] == "model-fusion"
    assert engine.contexts[0].entity_key == "Q1"


def test_rerank_run_refuses_existing_output_dir(tmp_path):
    run_dir = _write_source_run(tmp_path / "source_run")
    (run_dir / "stage4_model_fusion_rerank").mkdir()

    with pytest.raises(FileExistsError):
        rerank_run(run_dir, engine=FakeRerankEngine())


def test_rerank_run_rejects_image_paths_outside_entity_dir(tmp_path):
    run_dir = _write_source_run(tmp_path / "source_run", image_path="../outside.png")
    (run_dir / "entities" / "outside.png").write_bytes(_png_bytes())

    with pytest.raises(RerankError, match="image path"):
        rerank_run(run_dir, engine=FakeRerankEngine())


def test_rerank_run_preserves_failure_diagnostics(tmp_path):
    run_dir = _write_source_run(tmp_path / "source_run")

    summary = rerank_run(run_dir, engine=FailingDiagnosticEngine())

    output_dir = run_dir / "stage4_model_fusion_rerank"
    failures = json.loads((output_dir / "failures.json").read_text(encoding="utf-8"))
    assert summary["processed_entities"] == 0
    assert summary["failed_entities"] == 1
    assert failures == [
        {
            "entity_key": "Q1",
            "stage": "rerank",
            "error_type": "RuntimeError",
            "error": "qwen judge timeout",
            "diagnostic_events": [
                {
                    "scorer": "gpt4omini_judge",
                    "kind": "image",
                    "candidate_id": "image_001",
                    "status": "failed",
                    "details": {"error_type": "TimeoutError"},
                }
            ],
        }
    ]


def test_rerank_cli_uses_injected_engine_factory(tmp_path):
    run_dir = _write_source_run(tmp_path / "source_run")
    engine = FakeRerankEngine()

    code = main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(tmp_path / "reranked"),
            "--limit",
            "1",
        ],
        engine_factory=lambda *_args, **_kwargs: engine,
    )

    assert code == 0
    assert (tmp_path / "reranked" / "entities" / "Q1" / "final.json").is_file()


def test_rerank_run_reads_v2_stage_artifact_envelopes(tmp_path):
    run_dir = _write_source_run(tmp_path / "source_run", stage4_envelope=True)
    engine = FakeRerankEngine()

    summary = rerank_run(run_dir, engine=engine)

    assert summary["processed_entities"] == 1
    assert engine.contexts[0].image_candidates[0]["candidate_id"] == "image_001"


def _write_source_run(
    run_dir: Path,
    *,
    image_path: str = "generated_images/image_001.png",
    stage4_envelope: bool = False,
) -> Path:
    entity_dir = run_dir / "entities" / "Q1"
    image_file = entity_dir / "generated_images" / "image_001.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(_png_bytes())
    _write_json(
        run_dir / "config.json",
        {"missing_mode": "image", "candidate_budget": 1, "stage4_ranker": "model-fusion"},
    )
    _write_json(
        entity_dir / "final.json",
        {
            "entity": {
                "entity_key": "Q1",
                "label": "Example Entity",
                "existing_modalities": {"text": None, "image": None},
            },
            "missing_mode": "image",
            "stage4_ranker": "model-fusion",
        },
    )
    _write_json(entity_dir / "1_knowledge_modeling.json", {"entity_profile": {}, "relation_facts": []})
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {
            "supported_facts": [],
            "safe_inferences": [],
            "modality_constraints": {
                "text": {"must_include": [], "should_include": [], "context_only": [], "avoid": []},
                "image": {"must_include": [], "should_include": [], "context_only": [], "avoid": []},
            },
            "forbidden_content": [],
        },
    )
    stage4_data = {
        "target_modalities": ["text", "image"],
        "text_candidates": [
            {
                "candidate_id": "text_001",
                "target_modality": "text",
                "generated_content": {"type": "text", "value": "Example text."},
            }
        ],
        "image_candidates": [
            {
                "candidate_id": "image_001",
                "target_modality": "image",
                "generated_content": {
                    "type": "image",
                    "value": "prompt",
                    "image_path": image_path,
                },
            }
        ],
        "candidate_budget": 1,
    }
    stage4_payload = (
        {
            "schema_version": "mmkgc_stage_artifact_v2",
            "stage": "relation_alternation_generation",
            "summary": {},
            "data": stage4_data,
        }
        if stage4_envelope
        else {"relation_alternation_generation": stage4_data}
    )
    _write_json(entity_dir / "3_relation_alternation_generation.json", stage4_payload)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
