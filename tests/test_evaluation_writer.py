from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mmkgc.evaluation.writer import EvaluationWriter, write_evaluation_outputs


def _entity_evaluation(
    entity_key: str,
    *,
    label: str | None = None,
    overall_offline_evaluation_score: float = 0.5,
    calibrated_overall_offline_evaluation_score: float | None = None,
    text_score: float = 0.4,
    calibrated_text_score: float | None = None,
    image_score: float = 0.5,
    calibrated_image_score: float | None = None,
    pair_score: float = 0.6,
    calibrated_pair_score: float | None = None,
    case_tags: list[str] | None = None,
    calibration_tags: list[str] | None = None,
    include_calibration: bool = True,
    fallback_count: int = 0,
) -> dict:
    tags = case_tags or []
    calibrated_text = (
        calibrated_text_score if calibrated_text_score is not None else text_score
    )
    calibrated_image = calibrated_image_score if calibrated_image_score is not None else image_score
    calibrated_pair = calibrated_pair_score if calibrated_pair_score is not None else pair_score
    calibrated_overall = (
        calibrated_overall_offline_evaluation_score
        if calibrated_overall_offline_evaluation_score is not None
        else overall_offline_evaluation_score
    )
    evaluation = {
        "entity": {
            "entity_key": entity_key,
            "label": label or f"Entity {entity_key}",
            "missing_mode": "image",
        },
        "text_evaluation": {
            "candidate_count": 2,
            "best_candidate_id": f"{entity_key}-text",
            "best_offline_evaluation_score": text_score,
            "top5_avg_offline_evaluation_score": text_score,
        },
        "image_evaluation": {
            "candidate_count": 3,
            "best_candidate_id": f"{entity_key}-image",
            "best_offline_evaluation_score": image_score,
            "top5_avg_offline_evaluation_score": image_score,
        },
        "pair_evaluation": {
            "candidate_count": 1,
            "best_pair_id": f"{entity_key}-pair",
            "best_offline_evaluation_score": pair_score,
            "top5_avg_offline_evaluation_score": pair_score,
        },
        "fallback_summary": {
            "fallback_candidate_count": fallback_count,
            "has_fallback": "fallback" in tags,
        },
        "overall_offline_evaluation_score": overall_offline_evaluation_score,
        "case_tags": tags,
        "warnings": [],
    }
    if include_calibration:
        evaluation.update(
            {
                "calibrated_text_best_offline_evaluation_score": calibrated_text,
                "calibrated_image_best_offline_evaluation_score": calibrated_image,
                "calibrated_pair_best_offline_evaluation_score": calibrated_pair,
                "calibrated_overall_offline_evaluation_score": calibrated_overall,
                "calibration_summary": {
                    "calibration_tags": calibration_tags or [],
                },
            }
        )
    return evaluation


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_writer_refuses_existing_output_dir_by_default(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        EvaluationWriter(output_dir)


def test_overwrite_removes_stale_entity_outputs(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = run_dir / "evaluation"

    write_evaluation_outputs(
        output_dir,
        [_entity_evaluation("Q1"), _entity_evaluation("Q2")],
        run_dir,
    )
    unknown_file = output_dir / "caller_notes.txt"
    unknown_file.write_text("keep me", encoding="utf-8")

    summary = write_evaluation_outputs(
        output_dir,
        [_entity_evaluation("Q1")],
        run_dir,
        overwrite=True,
    )

    assert summary["evaluated_entities"] == 1
    entity_keys = {
        json.loads(path.read_text(encoding="utf-8"))["entity"]["entity_key"]
        for path in (output_dir / "entities").glob("*/evaluation.json")
    }
    assert entity_keys == {"Q1"}
    assert unknown_file.read_text(encoding="utf-8") == "keep me"


def test_overwrite_preserves_owned_filename_directories(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    conflicting_directory = output_dir / "summary.csv"
    conflicting_directory.mkdir()
    (conflicting_directory / "caller_notes.txt").write_text("keep me", encoding="utf-8")

    EvaluationWriter(output_dir, overwrite=True)

    assert conflicting_directory.is_dir()
    assert (
        (conflicting_directory / "caller_notes.txt").read_text(encoding="utf-8")
        == "keep me"
    )


def test_overwrite_rejects_output_dir_equal_to_run_dir_without_deleting_inputs(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    final_json = run_dir / "entities" / "Q1" / "final.json"
    final_json.parent.mkdir(parents=True)
    final_json.write_text('{"entity":{"entity_key":"Q1"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir must not be the run_dir"):
        write_evaluation_outputs(
            run_dir,
            [_entity_evaluation("Q1")],
            run_dir,
            overwrite=True,
        )

    assert final_json.read_text(encoding="utf-8") == '{"entity":{"entity_key":"Q1"}}'


def test_writer_outputs_json_csv_and_entity_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "run" / "evaluation"
    evaluations = [
        _entity_evaluation("Q1", overall_offline_evaluation_score=0.8, text_score=0.7),
        _entity_evaluation(
            "Q/Unsafe Key",
            label="Unsafe",
            overall_offline_evaluation_score=0.2,
            case_tags=["low_score", "fallback"],
            fallback_count=1,
        ),
    ]

    summary = write_evaluation_outputs(output_dir, evaluations, run_dir)

    assert summary["evaluated_entities"] == 2
    assert summary["failed_entities"] == 0
    assert summary["averages"]["overall_offline_evaluation_score"] == pytest.approx(0.5)
    assert summary["averages"]["calibrated_overall_offline_evaluation_score"] == pytest.approx(
        0.5
    )
    assert summary["counts"]["fallback_entities"] == 1
    assert summary["counts"]["low_score_entities"] == 1
    assert (output_dir / "evaluation_summary.json").is_file()
    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "entity_scores.csv").is_file()
    assert (output_dir / "best_cases.jsonl").is_file()
    assert (output_dir / "worst_cases.jsonl").is_file()
    assert (output_dir / "fallback_cases.jsonl").is_file()
    entity_json_paths = sorted((output_dir / "entities").glob("*/evaluation.json"))
    assert len(entity_json_paths) == 2
    entity_keys = {
        json.loads(path.read_text(encoding="utf-8"))["entity"]["entity_key"]
        for path in entity_json_paths
    }
    assert entity_keys == {"Q1", "Q/Unsafe Key"}

    persisted_summary = json.loads(
        (output_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert persisted_summary["output_files"]["entity_scores_csv"] == "entity_scores.csv"

    with (output_dir / "entity_scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {
        "text_best_offline_evaluation_score",
        "image_best_offline_evaluation_score",
        "pair_best_offline_evaluation_score",
        "overall_offline_evaluation_score",
        "calibrated_text_best_offline_evaluation_score",
        "calibrated_image_best_offline_evaluation_score",
        "calibrated_pair_best_offline_evaluation_score",
        "calibrated_overall_offline_evaluation_score",
        "calibration_tags",
    } <= set(rows[0])
    assert "text_best_score" not in rows[0]
    assert rows[0]["entity_key"] == "Q1"
    assert rows[1]["entity_key"] == "Q/Unsafe Key"

    persisted_entities = [
        json.loads(path.read_text(encoding="utf-8")) for path in entity_json_paths
    ]
    for persisted_entity in persisted_entities:
        assert "overall_offline_evaluation_score" in persisted_entity
        assert "overall" + "_score" not in persisted_entity
        assert "best_offline_evaluation_score" in persisted_entity["text_evaluation"]
        assert "top5_avg_offline_evaluation_score" in persisted_entity["text_evaluation"]
        assert "best" + "_score" not in persisted_entity["text_evaluation"]


def test_writer_persists_failed_entity_count_and_keys(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"

    summary = write_evaluation_outputs(
        output_dir,
        [_entity_evaluation("Q1")],
        tmp_path / "run",
        failed_entity_keys=["Q_bad"],
    )

    assert summary["failed_entities"] == 1
    assert summary["failed_entity_keys"] == ["Q_bad"]
    persisted_summary = json.loads(
        (output_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert persisted_summary["failed_entities"] == 1
    assert persisted_summary["failed_entity_keys"] == ["Q_bad"]

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}

    assert rows["failed_entities"] == "1"
    assert rows["failed_entity_keys"] == '["Q_bad"]'


def test_entity_directories_do_not_collide_after_sanitization(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation("Q/Unsafe Key", overall_offline_evaluation_score=0.1),
        _entity_evaluation("Q_Unsafe_Key", overall_offline_evaluation_score=0.9),
    ]

    write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    entity_json_paths = sorted((output_dir / "entities").glob("*/evaluation.json"))
    assert len(entity_json_paths) == 2
    persisted = {
        json.loads(path.read_text(encoding="utf-8"))["entity"]["entity_key"]: path
        for path in entity_json_paths
    }
    assert set(persisted) == {"Q/Unsafe Key", "Q_Unsafe_Key"}
    assert persisted["Q/Unsafe Key"].parent != persisted["Q_Unsafe_Key"].parent


def test_case_exports_respect_limit(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation(f"Q{index}", overall_offline_evaluation_score=index / 10)
        for index in range(5)
    ]

    write_evaluation_outputs(output_dir, evaluations, tmp_path / "run", case_limit=2)

    best_cases = _read_jsonl(output_dir / "best_cases.jsonl")
    worst_cases = _read_jsonl(output_dir / "worst_cases.jsonl")
    assert [case["entity_key"] for case in best_cases] == ["Q4", "Q3"]
    assert [case["entity_key"] for case in worst_cases] == ["Q0", "Q1"]
    assert {
        "overall_offline_evaluation_score",
        "text_best_offline_evaluation_score",
        "image_best_offline_evaluation_score",
        "pair_best_offline_evaluation_score",
        "calibrated_overall_offline_evaluation_score",
        "calibrated_text_best_offline_evaluation_score",
        "calibrated_image_best_offline_evaluation_score",
        "calibrated_pair_best_offline_evaluation_score",
        "calibration_tags",
    } <= set(best_cases[0])
    assert {
        "overall" + "_score",
        "text_best" + "_score",
        "image_best" + "_score",
        "pair_best" + "_score",
    }.isdisjoint(best_cases[0])


def test_case_exports_sort_by_calibrated_overall(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation(
            "Q_raw_high",
            overall_offline_evaluation_score=0.9,
            calibrated_overall_offline_evaluation_score=0.4,
        ),
        _entity_evaluation(
            "Q_calibrated_high",
            overall_offline_evaluation_score=0.7,
            calibrated_overall_offline_evaluation_score=0.8,
        ),
    ]

    write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    first_best = json.loads(
        (output_dir / "best_cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    first_worst = json.loads(
        (output_dir / "worst_cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_best["entity_key"] == "Q_calibrated_high"
    assert first_worst["entity_key"] == "Q_raw_high"


def test_calibrated_scores_and_tags_are_exported(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation(
            "Q1",
            calibrated_text_score=0.1,
            calibrated_image_score=0.2,
            calibrated_pair_score=0.3,
            calibrated_overall_offline_evaluation_score=0.4,
            calibration_tags=["fallback_penalized", "template_prompt_penalized"],
        ),
        _entity_evaluation(
            "Q2",
            calibrated_text_score=0.5,
            calibrated_image_score=0.6,
            calibrated_pair_score=0.7,
            calibrated_overall_offline_evaluation_score=0.8,
            calibration_tags=["generic_entity_type", "pair_single_side_penalized"],
        ),
    ]

    summary = write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    assert summary["averages"]["calibrated_text_best_offline_evaluation_score"] == 0.3
    assert summary["averages"]["calibrated_image_best_offline_evaluation_score"] == 0.4
    assert summary["averages"]["calibrated_pair_best_offline_evaluation_score"] == 0.5
    assert summary["averages"]["calibrated_overall_offline_evaluation_score"] == 0.6
    assert summary["counts"]["fallback_penalized_entities"] == 1
    assert summary["counts"]["generic_entity_type_entities"] == 1
    assert summary["counts"]["pair_single_side_penalized_entities"] == 1
    assert summary["counts"]["template_prompt_penalized_entities"] == 1

    with (output_dir / "entity_scores.csv").open(encoding="utf-8", newline="") as handle:
        score_rows = list(csv.DictReader(handle))

    assert score_rows[0]["calibrated_text_best_offline_evaluation_score"] == "0.1"
    assert score_rows[0]["calibrated_image_best_offline_evaluation_score"] == "0.2"
    assert score_rows[0]["calibrated_pair_best_offline_evaluation_score"] == "0.3"
    assert score_rows[0]["calibrated_overall_offline_evaluation_score"] == "0.4"
    assert score_rows[0]["calibration_tags"] == (
        "fallback_penalized|template_prompt_penalized"
    )

    best_case = _read_jsonl(output_dir / "best_cases.jsonl")[0]
    assert best_case["calibrated_overall_offline_evaluation_score"] == 0.8
    assert best_case["calibration_tags"] == [
        "generic_entity_type",
        "pair_single_side_penalized",
    ]

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = {row["metric"]: row["value"] for row in csv.DictReader(handle)}

    assert summary_rows["avg_calibrated_text_best_offline_evaluation_score"] == "0.3"
    assert summary_rows["avg_calibrated_image_best_offline_evaluation_score"] == "0.4"
    assert summary_rows["avg_calibrated_pair_best_offline_evaluation_score"] == "0.5"
    assert summary_rows["avg_calibrated_overall_offline_evaluation_score"] == "0.6"
    assert summary_rows["fallback_penalized_entities"] == "1"
    assert summary_rows["generic_entity_type_entities"] == "1"
    assert summary_rows["pair_single_side_penalized_entities"] == "1"
    assert summary_rows["template_prompt_penalized_entities"] == "1"


def test_legacy_evaluations_fallback_to_raw_scores_for_calibrated_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation(
            "Q_raw_high",
            overall_offline_evaluation_score=0.9,
            text_score=0.7,
            image_score=0.8,
            pair_score=0.6,
            include_calibration=False,
        ),
        _entity_evaluation(
            "Q_raw_low",
            overall_offline_evaluation_score=0.3,
            text_score=0.2,
            image_score=0.4,
            pair_score=0.3,
            include_calibration=False,
        ),
    ]

    summary = write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    assert summary["averages"]["calibrated_text_best_offline_evaluation_score"] == 0.45
    assert summary["averages"]["calibrated_image_best_offline_evaluation_score"] == 0.6
    assert summary["averages"]["calibrated_pair_best_offline_evaluation_score"] == 0.45
    assert summary["averages"]["calibrated_overall_offline_evaluation_score"] == 0.6

    best_case = _read_jsonl(output_dir / "best_cases.jsonl")[0]
    worst_case = _read_jsonl(output_dir / "worst_cases.jsonl")[0]
    assert best_case["entity_key"] == "Q_raw_high"
    assert worst_case["entity_key"] == "Q_raw_low"
    assert best_case["calibrated_overall_offline_evaluation_score"] == 0.9
    assert best_case["calibrated_text_best_offline_evaluation_score"] == 0.7
    assert best_case["calibrated_image_best_offline_evaluation_score"] == 0.8
    assert best_case["calibrated_pair_best_offline_evaluation_score"] == 0.6
    assert best_case["calibration_tags"] == []

    with (output_dir / "entity_scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["calibrated_text_best_offline_evaluation_score"] == "0.7"
    assert rows[0]["calibrated_image_best_offline_evaluation_score"] == "0.8"
    assert rows[0]["calibrated_pair_best_offline_evaluation_score"] == "0.6"
    assert rows[0]["calibrated_overall_offline_evaluation_score"] == "0.9"
    assert rows[0]["calibration_tags"] == ""


def test_case_exports_sort_by_raw_overall_when_calibrated_overall_is_none(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation(
            "Q_raw_high",
            overall_offline_evaluation_score=0.9,
        ),
        _entity_evaluation(
            "Q_raw_low",
            overall_offline_evaluation_score=0.3,
        ),
    ]
    for evaluation in evaluations:
        evaluation["calibrated_overall_offline_evaluation_score"] = None

    write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    first_best = _read_jsonl(output_dir / "best_cases.jsonl")[0]
    first_worst = _read_jsonl(output_dir / "worst_cases.jsonl")[0]
    assert first_best["entity_key"] == "Q_raw_high"
    assert first_worst["entity_key"] == "Q_raw_low"


def test_fallback_cases_only_include_fallback_tagged_entities(tmp_path) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation("Q1", case_tags=["fallback"], fallback_count=1),
        _entity_evaluation("Q2", case_tags=[]),
        _entity_evaluation("Q3", case_tags=["low_score"]),
    ]

    write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    fallback_cases = _read_jsonl(output_dir / "fallback_cases.jsonl")
    assert [case["entity_key"] for case in fallback_cases] == ["Q1"]
