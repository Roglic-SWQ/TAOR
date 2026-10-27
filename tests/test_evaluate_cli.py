from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import mmkgc.evaluate as evaluate_cli
from mmkgc.evaluate import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_run(run_dir: Path, entity_key: str = "Q1") -> None:
    entity_dir = run_dir / "entities" / entity_key
    entity_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    _write_json(
        entity_dir / "1_knowledge_modeling.json",
        {"entity_profile": {"entity_type": "person"}},
    )
    _write_json(entity_dir / "1_knowledge_modeling.json", {"stage": "reasoning"})
    _write_json(
        entity_dir / "4_kg_aware_candidate_consistency_ranking.json",
        {
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text-1",
                        "generated_content": {"value": "Ada Lovelace wrote about computing."},
                        "final_score": 0.8,
                        "relation_coverage_score": 0.7,
                        "must_term_score": 0.6,
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image-1",
                        "positive_prompt": "portrait of Ada Lovelace as a person",
                        "expected_relation_coverage": ["portrait"],
                        "used_constraints": [{"constraint": "person"}],
                        "final_score": 0.75,
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "pair_id": "pair-1",
                        "text_candidate": {"candidate_id": "text-1", "final_score": 0.8},
                        "image_candidate": {"candidate_id": "image-1", "final_score": 0.75},
                        "final_score": 0.7,
                        "pair_similarity_score": 0.6,
                        "kg_anchor_score": 0.5,
                    }
                ]
            },
        },
    )
    _write_json(
        entity_dir / "final.json",
        {
            "entity": {
                "entity_key": entity_key,
                "label": "Ada Lovelace",
                "description": "Ada Lovelace wrote about early computing.",
            }
        },
    )


def _attach_generated_image_to_minimal_run(
    run_dir: Path,
    *,
    entity_key: str = "Q1",
    image_name: str = "image-1.png",
) -> None:
    from PIL import Image

    entity_dir = run_dir / "entities" / entity_key
    image_dir = entity_dir / "generated_images"
    image_dir.mkdir()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_dir / image_name)

    ranking_path = entity_dir / "4_kg_aware_candidate_consistency_ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    image_candidate = ranking["image_ranking"]["ranked_candidates"][0]
    image_candidate["candidate_id"] = "image-1"
    image_candidate["generated_content"] = {
        "image_path": f"generated_images/{image_name}"
    }
    _write_json(ranking_path, ranking)


def test_cli_help_includes_image_quality_options() -> None:
    help_text = evaluate_cli._build_parser().format_help()

    assert "--image-quality-evaluator" in help_text
    assert "--clip-model-name" in help_text
    assert "--clip-model-dir" in help_text
    assert "--clip-device" in help_text


def test_cli_help_includes_stage8_options() -> None:
    help_text = evaluate_cli._build_parser().format_help()

    assert "--blip-itm-evaluator" in help_text
    assert "--blip-model-dir" in help_text
    assert "--blip-device" in help_text
    assert "--vlm-judge-evaluator" in help_text
    assert "--vlm-judge-model-config" in help_text
    assert "--vlm-judge-timeout-seconds" in help_text


def test_cli_help_includes_stage11_options() -> None:
    help_text = evaluate_cli._build_parser().format_help()

    assert "--text-quality-evaluator" in help_text
    assert "--bertscore-model-type" in help_text
    assert "--bertscore-device" in help_text
    assert "--gemini-embedding-evaluator" in help_text
    assert "--gemini-embedding-config" in help_text


def test_cli_requires_run_dir() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_writes_default_evaluation_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    summary_path = run_dir / "evaluation" / "evaluation_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["evaluated_entities"] == 1


def test_cli_default_does_not_require_clip_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["image_quality_evaluator"] == "none"
    assert stdout_summary["clip_evaluated_entities"] == 0
    assert stdout_summary["clip_evaluated_images"] == 0
    assert (run_dir / "evaluation" / "evaluation_summary.json").is_file()
    assert not (run_dir / "evaluation" / "clip_image_quality_summary.json").exists()


def test_cli_default_does_not_run_stage8_evaluators(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["blip_itm_evaluator"] == "none"
    assert stdout_summary["blip_evaluated_entities"] == 0
    assert stdout_summary["blip_evaluated_images"] == 0
    assert stdout_summary["vlm_judge_evaluator"] == "none"
    assert stdout_summary["vlm_judged_entities"] == 0
    assert stdout_summary["vlm_judged_images"] == 0
    assert stdout_summary["image_quality_comparison_written"] is False
    assert stdout_summary["image_quality_comparison_evaluated_images"] == 0
    assert not (run_dir / "evaluation" / "blip_itm_summary.json").exists()
    assert not (run_dir / "evaluation" / "qwen_vl_judge_summary.json").exists()
    assert not (
        run_dir / "evaluation" / "image_quality_model_comparison_summary.json"
    ).exists()


def test_cli_default_does_not_run_stage11_evaluators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    def fail_bertscore_init(*args, **kwargs):
        raise AssertionError("BERTScore should not initialize by default")

    def fail_gemini_config(*args, **kwargs):
        raise AssertionError("Gemini config should not load by default")

    monkeypatch.setattr(evaluate_cli, "BERTScoreLocalScorer", fail_bertscore_init)
    monkeypatch.setattr(evaluate_cli, "load_gemini_embedding_config", fail_gemini_config)

    exit_code = main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["text_quality_evaluator"] == "none"
    assert stdout_summary["bertscore_evaluated_entities"] == 0
    assert stdout_summary["bertscore_evaluated_text_candidates"] == 0
    assert stdout_summary["gemini_embedding_evaluator"] == "none"
    assert stdout_summary["gemini_evaluated_entities"] == 0
    assert stdout_summary["gemini_evaluated_text_candidates"] == 0
    assert stdout_summary["gemini_evaluated_images"] == 0
    assert stdout_summary["gemini_evaluated_pairs"] == 0
    assert not (run_dir / "evaluation" / "bertscore_text_quality_summary.json").exists()
    assert not (run_dir / "evaluation" / "gemini_embedding_summary.json").exists()


def test_cli_rejects_existing_evaluation_dir_without_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    (run_dir / "evaluation").mkdir()

    with pytest.raises(FileExistsError):
        main(["--run-dir", str(run_dir)])


def test_cli_allows_custom_output_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "custom-evaluation"
    _write_minimal_run(run_dir)

    exit_code = main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "evaluation_summary.json").is_file()
    assert not (run_dir / "evaluation").exists()


def test_cli_rejects_output_dir_equal_to_run_dir_with_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    final_json = run_dir / "entities" / "Q1" / "final.json"

    with pytest.raises(ValueError, match="output_dir must not be the run_dir"):
        main(["--run-dir", str(run_dir), "--output-dir", str(run_dir), "--overwrite"])

    assert final_json.is_file()


def test_cli_persists_failed_entity_keys_when_one_entity_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir, "Q_good")
    _write_minimal_run(run_dir, "Q_bad")
    output_dir = tmp_path / "evaluation"

    original_evaluate_entity = evaluate_cli.evaluate_entity

    def fail_one_entity(artifacts):
        if artifacts.entity_key == "Q_bad":
            raise ValueError("synthetic scorer failure")
        return original_evaluate_entity(artifacts)

    monkeypatch.setattr(evaluate_cli, "evaluate_entity", fail_one_entity)

    exit_code = main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])

    assert exit_code == 0
    summary = json.loads(
        (output_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["evaluated_entities"] == 1
    assert summary["failed_entities"] == 1
    assert summary["failed_entity_keys"] == ["Q_bad"]


def test_cli_continues_when_one_entity_fails_to_load(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir, "Q_good")
    _write_minimal_run(run_dir, "Q_bad")
    (run_dir / "entities" / "Q_bad" / "4_kg_aware_candidate_consistency_ranking.json").unlink()
    output_dir = tmp_path / "evaluation"

    exit_code = main(["--run-dir", str(run_dir), "--output-dir", str(output_dir)])

    assert exit_code == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["evaluated_entities"] == 1
    assert stdout_summary["failed_entities"] == 1
    assert stdout_summary["failed_entity_keys"] == ["Q_bad"]
    summary = json.loads(
        (output_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["evaluated_entities"] == 1
    assert summary["failed_entities"] == 1
    assert summary["failed_entity_keys"] == ["Q_bad"]

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}

    assert rows["failed_entities"] == "1"
    assert rows["failed_entity_keys"] == '["Q_bad"]'


def test_cli_clip_local_writes_clip_outputs_with_fake_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"
    configs = []
    calls = []

    class FakeEvaluator:
        def __init__(self, config):
            configs.append(config)

        def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
            calls.append({"image_path": image_path, "texts": texts})
            return [90.0 for _ in texts]

    monkeypatch.setattr(evaluate_cli, "ClipLocalImageQualityEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        evaluate_cli,
        "validate_clip_model_dir",
        lambda model_dir: tmp_path / "CLIP",
    )
    monkeypatch.setattr(
        evaluate_cli,
        "validate_clip_weight_path",
        lambda model_dir, model_name: Path(model_dir) / "ViT-B-32.pt",
    )

    exit_code = main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--image-quality-evaluator",
            "clip-local",
            "--clip-model-name",
            "ViT-B/32",
            "--clip-model-dir",
            "CLIP",
            "--clip-device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert configs[0].model_name == "ViT-B/32"
    assert configs[0].model_dir == Path("CLIP")
    assert configs[0].device == "cpu"
    assert len(calls) == 1
    assert (output_dir / "evaluation_summary.json").is_file()
    summary_path = output_dir / "clip_image_quality_summary.json"
    assert summary_path.is_file()
    clip_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert clip_summary["evaluated_entities"] == 1
    assert clip_summary["evaluated_images"] == 1
    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["image_quality_evaluator"] == "clip-local"
    assert stdout_summary["clip_evaluated_entities"] == 1
    assert stdout_summary["clip_evaluated_images"] == 1


def test_cli_blip_and_qwen_fake_providers_write_stage8_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"
    blip_model_dir = tmp_path / "blip-itm-large-coco"
    qwen_config_path = tmp_path / "qwen.yaml"
    blip_configs = []
    qwen_config_calls = []
    qwen_client_configs = []

    class FakeBlipScorer:
        def __init__(self, config):
            blip_configs.append(config)

        def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
            return [80.0, 70.0, 60.0][: len(texts)]

    class FakeQwenClient:
        def __init__(self, config):
            qwen_client_configs.append(config)

        def judge_image(self, image_path: Path, prompt: str) -> dict:
            return {
                "entity_match_score": 91,
                "prompt_alignment_score": 92,
                "kg_evidence_score": 93,
                "artifact_score": 94,
                "overall_judge_score": 95,
                "rationale": "synthetic judge result",
                "latency_ms": 12,
                "raw_response": {"api_key": "should-not-leak"},
            }

    def fake_load_qwen_config(path: str | Path, timeout_seconds: float = 120):
        qwen_config_calls.append((Path(path), timeout_seconds))
        return SimpleNamespace(
            model_name="fake-qwen-vl",
            base_url="http://localhost:12345/v1",
            api_key="secret-token",
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(
        evaluate_cli,
        "validate_blip_model_dir",
        lambda model_dir: blip_model_dir,
    )
    monkeypatch.setattr(evaluate_cli, "BlipITMLocalScorer", FakeBlipScorer)
    monkeypatch.setattr(evaluate_cli, "load_qwen_vl_judge_config", fake_load_qwen_config)
    monkeypatch.setattr(evaluate_cli, "QwenVLJudgeClient", FakeQwenClient)

    exit_code = main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--blip-itm-evaluator",
            "blip-itm-local",
            "--blip-model-dir",
            str(blip_model_dir),
            "--blip-device",
            "cpu",
            "--vlm-judge-evaluator",
            "qwen-vl-judge",
            "--vlm-judge-model-config",
            str(qwen_config_path),
            "--vlm-judge-timeout-seconds",
            "7.5",
        ]
    )

    assert exit_code == 0
    assert blip_configs[0].model_dir == blip_model_dir
    assert blip_configs[0].device == "cpu"
    assert qwen_config_calls == [(qwen_config_path, 7.5)]
    assert qwen_client_configs[0].model_name == "fake-qwen-vl"
    assert (output_dir / "blip_itm_summary.json").is_file()
    assert (output_dir / "qwen_vl_judge_summary.json").is_file()
    comparison_path = output_dir / "image_quality_model_comparison_summary.json"
    assert comparison_path.is_file()
    assert list((output_dir / "entities").glob("*/blip_itm.json"))
    assert list((output_dir / "entities").glob("*/qwen_vl_judge.json"))

    stdout_text = capsys.readouterr().out
    assert "secret-token" not in stdout_text
    assert "should-not-leak" not in stdout_text
    stdout_summary = json.loads(stdout_text)
    assert stdout_summary["blip_itm_evaluator"] == "blip-itm-local"
    assert stdout_summary["blip_evaluated_entities"] == 1
    assert stdout_summary["blip_evaluated_images"] == 1
    assert stdout_summary["vlm_judge_evaluator"] == "qwen-vl-judge"
    assert stdout_summary["vlm_judged_entities"] == 1
    assert stdout_summary["vlm_judged_images"] == 1
    assert stdout_summary["image_quality_comparison_written"] is True
    assert stdout_summary["image_quality_comparison_evaluated_images"] == 1


def test_cli_bertscore_and_gemini_fake_providers_write_stage11_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"
    bertscore_configs = []
    gemini_config_calls = []
    gemini_client_configs = []

    class FakeBERTScoreScorer:
        def __init__(self, model_type: str, device: str | None):
            bertscore_configs.append((model_type, device))

        def score_pairs(self, candidates: list[str], references: list[str]) -> list[dict]:
            return [
                {"precision": 0.91, "recall": 0.92, "f1": 0.93}
                for _ in candidates
            ]

    class FakeGeminiClient:
        def __init__(self, config):
            gemini_client_configs.append(config)

        def embed_text(self, text: str) -> list[float]:
            return [1.0, 0.0] if "Ada" in text else [0.0, 1.0]

        def embed_image(self, image_path: str | Path) -> list[float]:
            return [1.0, 0.0]

    def fake_load_gemini_config(path: str | Path):
        gemini_config_calls.append(Path(path))
        return SimpleNamespace(model_name="gemini-embedding-2")

    monkeypatch.setattr(evaluate_cli, "BERTScoreLocalScorer", FakeBERTScoreScorer)
    monkeypatch.setattr(evaluate_cli, "load_gemini_embedding_config", fake_load_gemini_config)
    monkeypatch.setattr(evaluate_cli, "GeminiEmbeddingClient", FakeGeminiClient)

    exit_code = main(
        [
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--text-quality-evaluator",
            "bertscore-local",
            "--bertscore-model-type",
            "roberta-large",
            "--bertscore-device",
            "cpu",
            "--gemini-embedding-evaluator",
            "gemini-api",
            "--gemini-embedding-config",
            "configs/gemini-embedding.yaml",
        ]
    )

    assert exit_code == 0
    assert bertscore_configs == [("roberta-large", "cpu")]
    assert gemini_config_calls == [Path("configs/gemini-embedding.yaml")]
    assert gemini_client_configs[0].model_name == "gemini-embedding-2"
    assert (output_dir / "bertscore_text_quality_summary.json").is_file()
    assert (output_dir / "gemini_embedding_summary.json").is_file()
    comparison = json.loads(
        (output_dir / "image_quality_model_comparison_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["bertscore"]["bertscore_overall_text_score"] == 0.93
    assert comparison["gemini_embedding"]["gemini_overall_multimodal_score"] == 100.0

    stdout_summary = json.loads(capsys.readouterr().out)
    assert stdout_summary["text_quality_evaluator"] == "bertscore-local"
    assert stdout_summary["bertscore_evaluated_entities"] == 1
    assert stdout_summary["bertscore_evaluated_text_candidates"] == 1
    assert stdout_summary["gemini_embedding_evaluator"] == "gemini-api"
    assert stdout_summary["gemini_evaluated_entities"] == 1
    assert stdout_summary["gemini_evaluated_text_candidates"] == 1
    assert stdout_summary["gemini_evaluated_images"] == 1
    assert stdout_summary["gemini_evaluated_pairs"] == 1
    assert stdout_summary["image_quality_comparison_written"] is True


def test_cli_clip_local_validate_failure_is_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"

    def fail_validate_clip_model_dir(model_dir):
        raise RuntimeError("missing local CLIP directory")

    monkeypatch.setattr(
        evaluate_cli,
        "validate_clip_model_dir",
        fail_validate_clip_model_dir,
    )

    with pytest.raises(RuntimeError, match="missing local CLIP directory"):
        main(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--image-quality-evaluator",
                "clip-local",
            ]
        )

    assert not (output_dir / "clip_image_quality_summary.json").exists()


def test_cli_blip_validate_failure_is_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"

    def fail_validate_blip_model_dir(model_dir):
        raise RuntimeError("missing local BLIP directory")

    monkeypatch.setattr(
        evaluate_cli,
        "validate_blip_model_dir",
        fail_validate_blip_model_dir,
    )

    with pytest.raises(RuntimeError, match="missing local BLIP directory"):
        main(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--blip-itm-evaluator",
                "blip-itm-local",
            ]
        )

    assert not (output_dir / "blip_itm_summary.json").exists()


def test_cli_qwen_config_failure_is_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _attach_generated_image_to_minimal_run(run_dir)
    output_dir = tmp_path / "evaluation"

    def fail_load_qwen_config(path: str | Path, timeout_seconds: float = 120):
        raise RuntimeError("missing Qwen-VL config")

    monkeypatch.setattr(
        evaluate_cli,
        "load_qwen_vl_judge_config",
        fail_load_qwen_config,
    )

    with pytest.raises(RuntimeError, match="missing Qwen-VL config"):
        main(
            [
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--vlm-judge-evaluator",
                "qwen-vl-judge",
            ]
        )

    assert not (output_dir / "qwen_vl_judge_summary.json").exists()
