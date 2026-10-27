from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from mmkgc.evaluation.multi_image_quality_writer import (
    write_blip_itm_outputs,
    write_image_quality_model_comparison_summary,
    write_qwen_vl_judge_outputs,
)


def _entity_dir_name(entity_key: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in entity_key
    )
    slug = safe.strip("._") or "entity"
    digest = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def _qwen_result(entity_key: str, overall: float) -> dict:
    return {
        "entity": {
            "entity_key": entity_key,
            "label": f"Entity {entity_key}",
            "missing_mode": "both",
        },
        "image_count": 1,
        "model_name": "Qwen/Qwen2-VL-7B-Instruct",
        "base_url": "http://localhost:12345/v1",
        "averages": {
            "entity_match_score": overall + 5,
            "prompt_alignment_score": overall - 5,
            "kg_evidence_score": overall - 10,
            "artifact_score": overall + 10,
            "overall_judge_score": overall,
        },
        "best_image_candidate_id": f"{entity_key}-best",
        "worst_image_candidate_id": f"{entity_key}-worst",
        "image_scores": [
            {
                "entity_key": entity_key,
                "image_candidate_id": f"{entity_key}-best",
                "image_path": "generated_images/image_001.png",
                "provider": "qwen-vl-judge",
                "model_name": "Qwen/Qwen2-VL-7B-Instruct",
                "base_url": "http://localhost:12345/v1",
                "status": "judged",
                "fallback_used": False,
                "parse_failed": False,
                "entity_match_score": overall + 5,
                "prompt_alignment_score": overall - 5,
                "kg_evidence_score": overall - 10,
                "artifact_score": overall + 10,
                "overall_judge_score": overall,
                "rationale": "图像基本符合。",
            }
        ],
    }


def _blip_result(entity_key: str, overall: float) -> dict:
    return {
        "entity": {
            "entity_key": entity_key,
            "label": f"Entity {entity_key}",
            "missing_mode": "both",
        },
        "image_count": 1,
        "model_dir": "blip-itm-large-coco",
        "averages": {
            "blip_prompt_itm_score": overall - 11,
            "blip_text_itm_score": overall,
            "blip_evidence_itm_score": overall + 11,
            "blip_overall_itm_score": overall,
        },
        "best_image_candidate_id": f"{entity_key}-best",
        "worst_image_candidate_id": f"{entity_key}-worst",
        "image_scores": [
            {
                "entity_key": entity_key,
                "image_candidate_id": f"{entity_key}-best",
                "image_path": "generated_images/image_001.png",
                "provider": "blip-itm-local",
                "model_dir": "blip-itm-large-coco",
                "status": "scored",
                "fallback_used": False,
                "blip_prompt_itm_score": overall - 11,
                "blip_text_itm_score": overall,
                "blip_evidence_itm_score": overall + 11,
                "blip_overall_itm_score": overall,
            }
        ],
    }


def _clip_summary(evaluated_images: int = 2) -> dict:
    return {
        "run_dir": "run",
        "output_dir": "evaluation",
        "evaluator": "clip-local",
        "clip_model_name": "ViT-B/32",
        "clip_weight_path": "CLIP/ViT-B-32.pt",
        "evaluated_entities": 2,
        "evaluated_images": evaluated_images,
        "averages": {
            "clip_prompt_similarity": 41.0,
            "clip_text_similarity": 42.0,
            "clip_evidence_similarity": 43.0,
            "clip_overall_image_quality_score": 44.0,
        },
        "output_files": {"summary_json": "clip_image_quality_summary.json"},
    }


def _bertscore_summary() -> dict:
    return {
        "run_dir": "run",
        "output_dir": "evaluation",
        "evaluator": "bertscore-local",
        "model_type": "roberta-large",
        "evaluated_entities": 2,
        "evaluated_text_candidates": 4,
        "averages": {
            "bertscore_precision": 0.81,
            "bertscore_recall": 0.82,
            "bertscore_f1": 0.83,
            "bertscore_overall_text_score": 0.83,
        },
        "output_files": {"summary_json": "bertscore_text_quality_summary.json"},
    }


def _gemini_summary(evaluated_images: int = 2) -> dict:
    return {
        "run_dir": "run",
        "output_dir": "evaluation",
        "evaluator": "gemini-api",
        "model_name": "gemini-embedding-2",
        "evaluated_entities": 2,
        "evaluated_text_candidates": 4,
        "evaluated_images": evaluated_images,
        "evaluated_pairs": 4,
        "averages": {
            "gemini_overall_text_embedding_score": 71.0,
            "gemini_overall_multimodal_score": 72.0,
            "gemini_overall_pair_embedding_score": 73.0,
        },
        "output_files": {"summary_json": "gemini_embedding_summary.json"},
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _call_writer(writer_name: str, output_dir: Path, overwrite: bool = True) -> None:
    if writer_name == "blip":
        write_blip_itm_outputs(
            output_dir,
            output_dir.parent / "run",
            [_blip_result("Q1", 22.0)],
            overwrite=overwrite,
        )
        return
    if writer_name == "qwen":
        write_qwen_vl_judge_outputs(
            output_dir,
            output_dir.parent / "run",
            [_qwen_result("Q1", 75.0)],
            overwrite=overwrite,
        )
        return
    if writer_name == "comparison":
        write_image_quality_model_comparison_summary(
            output_dir,
            clip_summary=_clip_summary(evaluated_images=1),
            blip_summary={
                "evaluated_images": 1,
                "averages": {"blip_overall_itm_score": 22.0},
            },
            qwen_vl_summary={
                "evaluated_images": 1,
                "averages": {"overall_judge_score": 75.0},
            },
            overwrite=overwrite,
        )
        return
    raise AssertionError(f"unknown writer: {writer_name}")


def test_write_blip_itm_outputs_writes_summary_csv_and_entity_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    summary = write_blip_itm_outputs(
        output_dir,
        run_dir,
        [_blip_result("Q1", 22.0), _blip_result("Q/Unsafe", 44.0)],
    )

    assert summary["run_dir"] == str(run_dir)
    assert summary["output_dir"] == str(output_dir)
    assert summary["evaluator"] == "blip-itm-local"
    assert summary["model_dir"] == "blip-itm-large-coco"
    assert summary["evaluated_entities"] == 2
    assert summary["evaluated_images"] == 2
    assert summary["failed_images"] == 0
    assert summary["fallback_images"] == 0
    assert summary["averages"]["blip_overall_itm_score"] == pytest.approx(33.0)
    assert summary["best_entities"][0]["entity_key"] == "Q/Unsafe"
    assert summary["worst_entities"][0]["entity_key"] == "Q1"
    assert (
        summary["output_files"]["entity_scores_csv"]
        == "blip_itm_entity_scores.csv"
    )

    summary_path = output_dir / "blip_itm_summary.json"
    csv_path = output_dir / "blip_itm_entity_scores.csv"
    entity_path = (
        output_dir
        / "entities"
        / _entity_dir_name("Q/Unsafe")
        / "blip_itm.json"
    )
    assert json.loads(summary_path.read_text(encoding="utf-8"))["evaluator"] == (
        "blip-itm-local"
    )
    assert json.loads(entity_path.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "Q/Unsafe"

    rows_by_key = {row["entity_key"]: row for row in _read_csv(csv_path)}
    assert set(rows_by_key) == {"Q1", "Q/Unsafe"}
    assert rows_by_key["Q1"]["label"] == "Entity Q1"
    assert rows_by_key["Q1"]["missing_mode"] == "both"
    assert rows_by_key["Q1"]["image_count"] == "1"
    assert rows_by_key["Q1"]["blip_overall_itm_score"] == "22.0"
    assert rows_by_key["Q1"]["best_image_candidate_id"] == "Q1-best"
    assert rows_by_key["Q1"]["worst_image_candidate_id"] == "Q1-worst"


def test_write_qwen_vl_judge_outputs_writes_summary_csv_and_entity_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    summary = write_qwen_vl_judge_outputs(
        output_dir,
        run_dir,
        [_qwen_result("Q1", 75.0), _qwen_result("Q/Unsafe", 85.0)],
    )

    assert summary["run_dir"] == str(run_dir)
    assert summary["output_dir"] == str(output_dir)
    assert summary["evaluator"] == "qwen-vl-judge"
    assert summary["model_name"] == "Qwen/Qwen2-VL-7B-Instruct"
    assert summary["base_url"] == "http://localhost:12345/v1"
    assert summary["evaluated_entities"] == 2
    assert summary["evaluated_images"] == 2
    assert summary["failed_images"] == 0
    assert summary["fallback_images"] == 0
    assert summary["parse_failed_images"] == 0
    assert summary["averages"]["overall_judge_score"] == pytest.approx(80.0)
    assert summary["best_entities"][0]["entity_key"] == "Q/Unsafe"
    assert summary["worst_entities"][0]["entity_key"] == "Q1"
    assert (
        summary["output_files"]["entity_scores_csv"]
        == "qwen_vl_judge_entity_scores.csv"
    )

    summary_path = output_dir / "qwen_vl_judge_summary.json"
    csv_path = output_dir / "qwen_vl_judge_entity_scores.csv"
    entity_path = (
        output_dir
        / "entities"
        / _entity_dir_name("Q/Unsafe")
        / "qwen_vl_judge.json"
    )
    assert json.loads(summary_path.read_text(encoding="utf-8"))["evaluator"] == (
        "qwen-vl-judge"
    )
    assert json.loads(entity_path.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "Q/Unsafe"

    rows_by_key = {row["entity_key"]: row for row in _read_csv(csv_path)}
    assert set(rows_by_key) == {"Q1", "Q/Unsafe"}
    assert rows_by_key["Q1"]["label"] == "Entity Q1"
    assert rows_by_key["Q1"]["missing_mode"] == "both"
    assert rows_by_key["Q1"]["image_count"] == "1"
    assert rows_by_key["Q1"]["overall_judge_score"] == "75.0"
    assert rows_by_key["Q1"]["best_image_candidate_id"] == "Q1-best"
    assert rows_by_key["Q1"]["worst_image_candidate_id"] == "Q1-worst"


def test_write_image_quality_model_comparison_summary_records_all_three_models(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    run_dir = tmp_path / "run"
    blip_summary = write_blip_itm_outputs(
        output_dir,
        run_dir,
        [_blip_result("Q1", 22.0), _blip_result("Q2", 22.0)],
    )
    qwen_summary = write_qwen_vl_judge_outputs(
        output_dir,
        run_dir,
        [_qwen_result("Q1", 75.0), _qwen_result("Q2", 75.0)],
        overwrite=True,
    )

    summary = write_image_quality_model_comparison_summary(
        output_dir,
        run_dir,
        clip_summary=_clip_summary(),
        blip_summary=blip_summary,
        qwen_vl_summary=qwen_summary,
    )

    assert summary["run_dir"] == str(run_dir)
    assert summary["output_dir"] == str(output_dir)
    assert summary["evaluated_images"] == 2
    assert summary["clip"]["summary_json"] == "clip_image_quality_summary.json"
    assert summary["clip"]["averages"]["clip_overall_image_quality_score"] == 44.0
    assert summary["clip"]["clip_overall_image_quality_score"] == 44.0
    assert summary["blip_itm"]["summary_json"] == "blip_itm_summary.json"
    assert summary["blip_itm"]["averages"]["blip_overall_itm_score"] == 22.0
    assert summary["blip_itm"]["blip_overall_itm_score"] == 22.0
    assert summary["qwen_vl_judge"]["summary_json"] == "qwen_vl_judge_summary.json"
    assert summary["qwen_vl_judge"]["averages"]["overall_judge_score"] == 75.0
    assert summary["qwen_vl_judge"]["overall_judge_score"] == 75.0
    assert "different signals" in summary["notes"][0]
    assert (
        output_dir / "image_quality_model_comparison_summary.json"
    ).is_file()


def test_write_image_quality_model_comparison_summary_records_stage11_models(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    run_dir = tmp_path / "run"

    summary = write_image_quality_model_comparison_summary(
        output_dir,
        run_dir,
        clip_summary=_clip_summary(),
        bertscore_summary=_bertscore_summary(),
        gemini_summary=_gemini_summary(),
    )

    assert summary["clip"]["clip_overall_image_quality_score"] == 44.0
    assert summary["blip_itm"] is None
    assert summary["qwen_vl_judge"] is None
    assert summary["bertscore"]["summary_json"] == "bertscore_text_quality_summary.json"
    assert summary["bertscore"]["model_type"] == "roberta-large"
    assert summary["bertscore"]["evaluated_text_candidates"] == 4
    assert summary["bertscore"]["bertscore_overall_text_score"] == 0.83
    assert summary["gemini_embedding"]["summary_json"] == "gemini_embedding_summary.json"
    assert summary["gemini_embedding"]["model_name"] == "gemini-embedding-2"
    assert summary["gemini_embedding"]["evaluated_images"] == 2
    assert summary["gemini_embedding"]["evaluated_pairs"] == 4
    assert summary["gemini_embedding"]["gemini_overall_multimodal_score"] == 72.0
    assert "clip_overall_image_quality_score" in summary["clip"]["averages"]


def test_write_image_quality_model_comparison_summary_supports_plan_signature(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"

    summary = write_image_quality_model_comparison_summary(
        output_dir,
        clip_summary=_clip_summary(evaluated_images=1),
        blip_summary={
            "run_dir": "run-from-blip",
            "evaluated_images": 1,
            "averages": {"blip_overall_itm_score": 22.0},
        },
        qwen_vl_summary={
            "run_dir": "run-from-qwen",
            "evaluated_images": 1,
            "averages": {"overall_judge_score": 75.0},
        },
    )

    assert summary["run_dir"] == "run"
    assert summary["evaluated_images"] == 1
    assert (
        output_dir / "image_quality_model_comparison_summary.json"
    ).is_file()


def test_write_image_quality_model_comparison_summary_rejects_image_count_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="evaluated_images"):
        write_image_quality_model_comparison_summary(
            tmp_path / "evaluation",
            tmp_path / "run",
            clip_summary=_clip_summary(evaluated_images=2),
            blip_summary={
                "evaluated_images": 3,
                "averages": {"blip_overall_itm_score": 22.0},
            },
            qwen_vl_summary={
                "evaluated_images": 2,
                "averages": {"overall_judge_score": 75.0},
            },
        )


def test_overwrite_true_replaces_only_multi_model_writer_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    write_blip_itm_outputs(output_dir, run_dir, [_blip_result("Q1", 22.0)])
    write_qwen_vl_judge_outputs(output_dir, run_dir, [_qwen_result("Q1", 75.0)])
    write_image_quality_model_comparison_summary(
        output_dir,
        run_dir,
        clip_summary=_clip_summary(evaluated_images=1),
        blip_summary={"evaluated_images": 1, "averages": {"blip_overall_itm_score": 22.0}},
        qwen_vl_summary={"evaluated_images": 1, "averages": {"overall_judge_score": 75.0}},
    )
    entity_dir = output_dir / "entities" / _entity_dir_name("Q1")
    root_notes = output_dir / "notes.txt"
    offline_evaluation = entity_dir / "evaluation.json"
    clip_quality = entity_dir / "clip_image_quality.json"
    root_notes.write_text("keep root notes", encoding="utf-8")
    offline_evaluation.write_text('{"kind":"offline"}', encoding="utf-8")
    clip_quality.write_text('{"kind":"clip"}', encoding="utf-8")

    write_blip_itm_outputs(
        output_dir,
        run_dir,
        [_blip_result("Q2", 44.0)],
        overwrite=True,
    )
    write_qwen_vl_judge_outputs(
        output_dir,
        run_dir,
        [_qwen_result("Q2", 85.0)],
        overwrite=True,
    )
    write_image_quality_model_comparison_summary(
        output_dir,
        run_dir,
        clip_summary=_clip_summary(evaluated_images=1),
        blip_summary={"evaluated_images": 1, "averages": {"blip_overall_itm_score": 44.0}},
        qwen_vl_summary={"evaluated_images": 1, "averages": {"overall_judge_score": 85.0}},
        overwrite=True,
    )

    assert root_notes.read_text(encoding="utf-8") == "keep root notes"
    assert json.loads(offline_evaluation.read_text(encoding="utf-8"))["kind"] == (
        "offline"
    )
    assert json.loads(clip_quality.read_text(encoding="utf-8"))["kind"] == "clip"
    assert not (entity_dir / "blip_itm.json").exists()
    assert not (entity_dir / "qwen_vl_judge.json").exists()
    assert (
        output_dir
        / "entities"
        / _entity_dir_name("Q2")
        / "blip_itm.json"
    ).is_file()
    assert (
        output_dir
        / "entities"
        / _entity_dir_name("Q2")
        / "qwen_vl_judge.json"
    ).is_file()


def test_overwrite_true_preserves_empty_unrelated_entity_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    unrelated_entity_dir = output_dir / "entities" / "unrelated_empty"
    unrelated_entity_dir.mkdir(parents=True)

    write_blip_itm_outputs(
        output_dir,
        tmp_path / "run",
        [_blip_result("Q1", 22.0)],
        overwrite=True,
    )

    assert unrelated_entity_dir.is_dir()
    assert (
        output_dir
        / "entities"
        / _entity_dir_name("Q1")
        / "blip_itm.json"
    ).is_file()


def test_writer_refuses_existing_owned_outputs_without_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    run_dir = tmp_path / "run"
    write_qwen_vl_judge_outputs(output_dir, run_dir, [_qwen_result("Q1", 75.0)])

    with pytest.raises(FileExistsError, match="qwen_vl_judge_summary.json"):
        write_qwen_vl_judge_outputs(output_dir, run_dir, [_qwen_result("Q2", 85.0)])


def test_writer_refuses_symlinked_entities_dir_without_deleting_target(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    outside_entities = tmp_path / "outside_entities"
    outside_entity_dir = outside_entities / "external"
    outside_entity_dir.mkdir(parents=True)
    outside_qwen = outside_entity_dir / "qwen_vl_judge.json"
    outside_qwen.write_text('{"entity":{"entity_key":"outside"}}', encoding="utf-8")
    (output_dir / "entities").symlink_to(
        outside_entities,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        write_qwen_vl_judge_outputs(
            output_dir,
            tmp_path / "run",
            [_qwen_result("Q1", 75.0)],
            overwrite=True,
        )

    assert json.loads(outside_qwen.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert not (output_dir / "qwen_vl_judge_summary.json").exists()


def test_writer_refuses_symlinked_entity_subdir_without_deleting_target(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    entities_dir = output_dir / "entities"
    entities_dir.mkdir(parents=True)
    outside_entity_dir = tmp_path / "outside_entity"
    outside_entity_dir.mkdir()
    outside_blip = outside_entity_dir / "blip_itm.json"
    outside_blip.write_text('{"entity":{"entity_key":"outside"}}', encoding="utf-8")
    (entities_dir / "external").symlink_to(
        outside_entity_dir,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        write_blip_itm_outputs(
            output_dir,
            tmp_path / "run",
            [_blip_result("Q1", 22.0)],
            overwrite=True,
        )

    assert json.loads(outside_blip.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert not (output_dir / "blip_itm_summary.json").exists()


@pytest.mark.parametrize("writer_name", ["blip", "qwen", "comparison"])
def test_writer_refuses_symlinked_output_dir_without_touching_target(
    tmp_path: Path,
    writer_name: str,
) -> None:
    target_dir = tmp_path / "target_evaluation"
    target_dir.mkdir()
    target_notes = target_dir / "notes.txt"
    target_notes.write_text("keep target", encoding="utf-8")
    output_dir = tmp_path / "evaluation_link"
    output_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        _call_writer(writer_name, output_dir, overwrite=True)

    assert target_notes.read_text(encoding="utf-8") == "keep target"
    assert sorted(path.name for path in target_dir.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize("writer_name", ["blip", "qwen", "comparison"])
def test_writer_refuses_file_output_dir_without_unlinking(
    tmp_path: Path,
    writer_name: str,
) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="output dir"):
        _call_writer(writer_name, output_dir, overwrite=True)

    assert output_dir.read_text(encoding="utf-8") == "not a directory"


def test_non_finite_scores_are_filtered_from_summary_csv_and_json(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    result = _blip_result("Q1", 22.0)
    result["averages"]["blip_prompt_itm_score"] = math.inf
    result["averages"]["blip_text_itm_score"] = -math.inf
    result["averages"]["blip_overall_itm_score"] = math.nan
    result["image_scores"][0]["blip_overall_itm_score"] = math.inf

    summary = write_blip_itm_outputs(
        output_dir,
        tmp_path / "run",
        [result],
    )

    assert summary["averages"]["blip_prompt_itm_score"] is None
    assert summary["averages"]["blip_text_itm_score"] is None
    assert summary["averages"]["blip_overall_itm_score"] is None
    rows = _read_csv(output_dir / "blip_itm_entity_scores.csv")
    assert rows[0]["blip_prompt_itm_score"] == ""
    assert rows[0]["blip_text_itm_score"] == ""
    assert rows[0]["blip_overall_itm_score"] == ""
    summary_text = (output_dir / "blip_itm_summary.json").read_text(
        encoding="utf-8"
    )
    entity_text = (
        output_dir
        / "entities"
        / _entity_dir_name("Q1")
        / "blip_itm.json"
    ).read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    assert "NaN" not in entity_text
    assert "Infinity" not in entity_text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parse_failed", True),
        ("status", "parse_failed"),
    ],
)
def test_qwen_summary_counts_parse_failed_images(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    result = _qwen_result("Q1", 75.0)
    result["image_scores"][0][field] = value

    summary = write_qwen_vl_judge_outputs(
        tmp_path / "evaluation",
        tmp_path / "run",
        [result],
    )

    assert summary["parse_failed_images"] == 1
