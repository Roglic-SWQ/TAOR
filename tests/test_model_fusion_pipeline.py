from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import sys
from typing import Any

from mmkgc.adapters import GeneratedImageArtifact
from mmkgc.data import EntityRecord, MKGWDataStore
from mmkgc.pipeline import MMKGCPipeline

try:
    from tests.test_pipeline_outputs import make_fixture
except ImportError:  # pragma: no cover
    make_fixture = None


def _png_bytes() -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG_BYTES = _png_bytes()








def test_model_fusion_text_mode_passes_existing_image_input_from_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_fixture(tmp_path / "dataset")
    existing_path = dataset_dir / "images" / "Example Film" / "image.jpg"
    existing_path.write_bytes(PNG_BYTES)
    record = MKGWDataStore(dataset_dir).build_record(0, missing_mode="text")
    engine = CapturingEngine()

    result = MMKGCPipeline(
        ranking_engine=engine,
        dataset_root=dataset_dir,
    ).run(record, "text")

    context = engine.contexts[0]
    assert context.existing_image_input is not None
    assert context.existing_image_input.image_path == existing_path.resolve()
    assert context.existing_image_input.image_bytes is None
    assert context.existing_image_input.image_path.resolve().is_relative_to(dataset_dir.resolve())
    payload = context.to_prompt_payload()
    assert payload["has_existing_image"] is True
    assert payload["existing_modalities"] == {}
    assert str(existing_path) not in str(payload)
    assert result["final"]["stage4_ranker"] == "model-fusion"


def test_model_fusion_text_mode_accepts_existing_image_path_prefixed_by_dataset_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_dir = _make_fixture(Path("dataset"))
    existing_path = dataset_dir / "images" / "Example Film" / "image.jpg"
    existing_path.write_bytes(PNG_BYTES)
    record = MKGWDataStore(dataset_dir).build_record(0, missing_mode="text")
    engine = CapturingEngine()

    MMKGCPipeline(
        ranking_engine=engine,
        dataset_root=dataset_dir,
    ).run(record, "text")

    context = engine.contexts[0]
    assert context.existing_image_input is not None
    assert context.existing_image_input.image_path == existing_path.resolve()


def test_model_fusion_text_mode_skips_unsupported_existing_image_input(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_fixture(tmp_path / "dataset")
    existing_path = dataset_dir / "images" / "Example Film" / "image.jpg"
    existing_path.write_bytes(b"not-a-real-png-or-jpeg")
    record = MKGWDataStore(dataset_dir).build_record(0, missing_mode="text")
    engine = CapturingEngine()

    result = MMKGCPipeline(
        ranking_engine=engine,
        dataset_root=dataset_dir,
    ).run(record, "text")

    context = engine.contexts[0]
    assert context.existing_image_input is None
    assert result["final"]["ranking_status"] == "completed"






def test_factory_import_does_not_import_heavy_model_modules_before_preflight() -> None:
    code = (
        "import sys; "
        "import mmkgc.ranking.factory; "
        "print([m for m in ('sentence_transformers','torch','transformers') if m in sys.modules])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout.strip() == "[]"


class CapturingEngine:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def rank(self, context: Any) -> dict[str, Any]:
        self.contexts.append(context)
        text_candidate = {
            "candidate_id": "text_001",
            "target_modality": "text",
            "score": 0.9,
            "final_score": 0.9,
            "generated_content": {"type": "text", "value": "Grounded description."},
        }
        image_candidate = {
            "candidate_id": "image_001",
            "target_modality": "image",
            "score": 0.8,
            "final_score": 0.8,
            "generated_content": {"type": "image", "value": "Grounded image."},
        }
        image_ranking = {
            "best_candidate_id": "image_001",
            "top_5_candidates": [image_candidate],
            "ranked_candidates": [image_candidate],
        }
        pair_ranking = {
            "best_pair_id": None,
            "best_pair": None,
            "top_5_pairs": [],
            "ranked_pairs": [],
        }
        if context.missing_mode == "text":
            image_ranking = {
                "best_candidate_id": None,
                "top_5_candidates": [],
                "ranked_candidates": [],
            }
            pair_ranking = {
                "best_pair_id": None,
                "best_pair": None,
                "top_5_pairs": [],
                "ranked_pairs": [],
            }
        return {
            "ranking_status": "completed",
            "ranker": "model_fusion",
            "score_version": "stage4_structured_graph_v3",
            "profile_name": "unit-profile",
            "ranking_model_calls": [
                {
                    "scorer": "fake",
                    "kind": "text",
                    "candidate_id": str(context.text_candidates[0]["candidate_id"]),
                    "status": "scored",
                    "details": {"safe": True},
                }
            ],
            "text_ranking": {
                "best_candidate_id": "text_001",
                "top_5_candidates": [text_candidate],
                "ranked_candidates": [text_candidate],
            },
            "image_ranking": image_ranking,
            "pair_ranking": pair_ranking,
            "kg_aware_candidate_consistency_ranking": {"ground_truth_used": False},
        }


class RetryAfterNoValidCandidateEngine:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def rank(self, context: Any) -> dict[str, Any]:
        self.contexts.append(context)
        if len(self.contexts) == 1:
            return {
                "ranking_status": "no_valid_candidate",
                "ranker": "model_fusion",
                "score_version": "stage4_structured_graph_v3",
                "profile_name": "unit-profile",
                "ranking_model_calls": [
                    {
                        "scorer": "fake",
                        "kind": "text",
                        "candidate_id": "text_001",
                        "status": "rejected",
                    }
                ],
                "text_ranking": {
                    "best_candidate_id": None,
                    "ranked_candidates": [
                        {
                            "candidate_id": "text_001",
                            "hard_gate": {
                                "decision": "reject",
                                "hard_violations": ["unsupported_claim"],
                                "soft_violations": [],
                            },
                        }
                    ],
                    "rejected_candidates": [
                        {
                            "candidate_id": "text_001",
                            "hard_gate": {
                                "decision": "reject",
                                "hard_violations": ["unsupported_claim"],
                                "soft_violations": [],
                            },
                        }
                    ],
                },
                "image_ranking": {
                    "best_candidate_id": None,
                    "ranked_candidates": [],
                    "rejected_candidates": [],
                },
                "pair_ranking": {
                    "best_pair_id": None,
                    "best_pair": None,
                    "top_5_pairs": [],
                    "ranked_pairs": [],
                },
                "kg_aware_candidate_consistency_ranking": {"ground_truth_used": False},
            }
        return CapturingEngine().rank(context)


class FakeImageGenerator:
    def generate(
        self,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
    ) -> GeneratedImageArtifact:
        del positive_prompt, negative_prompt
        return GeneratedImageArtifact(
            entity_key=entity_key,
            candidate_id=candidate_id,
            relative_image_path=f"generated_images/{candidate_id}.png",
            relative_metadata_path=f"generated_images/{candidate_id}.json",
            image_bytes=PNG_BYTES,
            metadata={"status": "generated", "generator": "fake-image"},
            model_call={
                "modality": "image",
                "candidate_id": candidate_id,
                "status": "generated",
            },
        )


def _fixture_record(tmp_path: Path) -> EntityRecord:
    if make_fixture is None:
        dataset_dir = _make_fixture(tmp_path / "dataset")
    else:
        dataset_dir = make_fixture(tmp_path / "dataset")
    return MKGWDataStore(dataset_dir).build_record(0, missing_mode="both")


def _make_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "entity2id.txt").write_text(
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 0",
                "http://www.wikidata.org/entity/Q2 1",
                "http://www.wikidata.org/entity/Q3 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "relation2id.txt").write_text(
        "\n".join(
            [
                "http://www.wikidata.org/entity/P31 0",
                "http://www.wikidata.org/entity/P17 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train.txt").write_text(
        "\n".join(
            [
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .",
                "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P17 http://www.wikidata.org/entity/Q3 .",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "valid.txt").write_text("", encoding="utf-8")
    (tmp_path / "test.txt").write_text("", encoding="utf-8")
    (tmp_path / "entity_info.json").write_text(
        (
            '{"Q1":{"label":"Example Film","description":"a fictional film"},'
            '"Q2":{"label":"film","description":"type of creative work"},'
            '"Q3":{"label":"Exampleland","description":"a country"}}'
        ),
        encoding="utf-8",
    )
    (tmp_path / "relation_info.json").write_text(
        '{"P31":"instance of","P17":"country"}',
        encoding="utf-8",
    )
    image_dir = tmp_path / "images" / "Example Film"
    image_dir.mkdir(parents=True)
    (image_dir / "image.jpg").write_bytes(b"fake")
    return tmp_path
