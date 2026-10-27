from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mmkgc.evaluation.text_quality_writer import (
    ENTITY_FILENAME,
    ENTITY_SCORES_CSV_FILENAME,
    SUMMARY_FILENAME,
    write_bertscore_text_quality_outputs,
)


def _quality(entity_key: str, overall: float) -> dict:
    return {
        "entity": {
            "entity_key": entity_key,
            "label": f"Entity {entity_key}",
            "missing_mode": "both",
        },
        "model_type": "roberta-large",
        "text_candidate_count": 3,
        "scored_text_candidate_count": 2,
        "averages": {
            "bertscore_precision": overall + 0.01,
            "bertscore_recall": overall + 0.02,
            "bertscore_f1": overall,
            "bertscore_overall_text_score": overall,
        },
        "best_text_candidate_id": f"{entity_key}-best",
        "worst_text_candidate_id": f"{entity_key}-worst",
        "text_scores": [
            {
                "text_candidate_id": f"{entity_key}-best",
                "bertscore_overall_text_score": overall,
            },
            {
                "text_candidate_id": f"{entity_key}-unscored",
                "bertscore_overall_text_score": None,
            },
        ],
    }


def _entity_quality_paths(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "entities").glob(f"*/{ENTITY_FILENAME}"))


def _entity_dir_name_for_test(entity_key: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in entity_key
    )
    slug = safe.strip("._") or "entity"
    digest = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def test_writes_summary_csv_and_entity_quality_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    summary = write_bertscore_text_quality_outputs(
        output_dir,
        run_dir,
        [_quality("Q1", 0.6), _quality("Q/Unsafe", 0.8)],
    )

    assert summary["evaluator"] == "bertscore-local"
    assert summary["model_type"] == "roberta-large"
    assert summary["evaluated_entities"] == 2
    assert summary["evaluated_text_candidates"] == 4
    assert summary["averages"]["bertscore_overall_text_score"] == pytest.approx(0.7)
    assert summary["best_entities"][0]["entity_key"] == "Q/Unsafe"
    assert summary["worst_entities"][0]["entity_key"] == "Q1"
    assert (output_dir / SUMMARY_FILENAME).is_file()
    assert (output_dir / ENTITY_SCORES_CSV_FILENAME).is_file()

    persisted_summary = json.loads(
        (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted_summary["output_files"]["summary_json"] == SUMMARY_FILENAME
    assert (
        persisted_summary["output_files"]["entity_scores_csv"]
        == ENTITY_SCORES_CSV_FILENAME
    )

    with (output_dir / ENTITY_SCORES_CSV_FILENAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    rows_by_key = {row["entity_key"]: row for row in rows}
    assert set(rows_by_key) == {"Q1", "Q/Unsafe"}
    assert rows_by_key["Q/Unsafe"]["bertscore_overall_text_score"] == "0.8"
    assert len(_entity_quality_paths(output_dir)) == 2


def test_overwrite_only_replaces_owned_bertscore_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    clip_summary = output_dir / "clip_image_quality_summary.json"
    clip_summary.write_text('{"keep":"clip"}', encoding="utf-8")
    gemini_summary = output_dir / "gemini_embedding_summary.json"
    gemini_summary.write_text('{"keep":"gemini"}', encoding="utf-8")

    write_bertscore_text_quality_outputs(output_dir, run_dir, [_quality("Q1", 0.6)])
    write_bertscore_text_quality_outputs(
        output_dir,
        run_dir,
        [_quality("Q2", 0.8)],
        overwrite=True,
    )

    assert clip_summary.read_text(encoding="utf-8") == '{"keep":"clip"}'
    assert gemini_summary.read_text(encoding="utf-8") == '{"keep":"gemini"}'
    entity_paths = _entity_quality_paths(output_dir)
    assert len(entity_paths) == 1
    persisted_entity = json.loads(entity_paths[0].read_text(encoding="utf-8"))
    assert persisted_entity["entity"]["entity_key"] == "Q2"


def test_overwrite_preserves_other_entity_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    write_bertscore_text_quality_outputs(output_dir, run_dir, [_quality("Q1", 0.6)])
    entity_dir = next((output_dir / "entities").iterdir())
    clip_quality = entity_dir / "clip_image_quality.json"
    notes = entity_dir / "notes.txt"
    clip_quality.write_text('{"kind":"clip"}', encoding="utf-8")
    notes.write_text("keep notes", encoding="utf-8")

    write_bertscore_text_quality_outputs(
        output_dir,
        run_dir,
        [_quality("Q2", 0.8)],
        overwrite=True,
    )

    assert not (entity_dir / ENTITY_FILENAME).exists()
    assert json.loads(clip_quality.read_text(encoding="utf-8"))["kind"] == "clip"
    assert notes.read_text(encoding="utf-8") == "keep notes"
    entity_paths = _entity_quality_paths(output_dir)
    assert len(entity_paths) == 1
    persisted_entity = json.loads(entity_paths[0].read_text(encoding="utf-8"))
    assert persisted_entity["entity"]["entity_key"] == "Q2"


def test_refuses_to_overwrite_existing_owned_outputs_by_default(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    write_bertscore_text_quality_outputs(output_dir, run_dir, [_quality("Q1", 0.6)])

    with pytest.raises(FileExistsError, match=SUMMARY_FILENAME):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 0.8)],
        )


def test_refuses_stale_entity_quality_output_by_default(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    stale_entity_dir = output_dir / "entities" / "stale"
    stale_entity_dir.mkdir(parents=True)
    stale_quality = stale_entity_dir / ENTITY_FILENAME
    stale_quality.write_text('{"entity":{"entity_key":"stale"}}', encoding="utf-8")

    with pytest.raises(FileExistsError, match=ENTITY_FILENAME):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 0.8)],
        )

    assert json.loads(stale_quality.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "stale"


def test_refuses_symlinked_entities_dir_without_deleting_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    summary = output_dir / SUMMARY_FILENAME
    entity_scores_csv = output_dir / ENTITY_SCORES_CSV_FILENAME
    summary.write_text('{"existing":true}', encoding="utf-8")
    entity_scores_csv.write_text("entity_key\nQ1\n", encoding="utf-8")
    outside_entities = tmp_path / "outside_entities"
    outside_entity_dir = outside_entities / "external"
    outside_entity_dir.mkdir(parents=True)
    outside_quality = outside_entity_dir / ENTITY_FILENAME
    outside_quality.write_text(
        '{"entity":{"entity_key":"outside"}}',
        encoding="utf-8",
    )
    (output_dir / "entities").symlink_to(outside_entities, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 0.8)],
            overwrite=True,
        )

    assert json.loads(outside_quality.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert summary.read_text(encoding="utf-8") == '{"existing":true}'
    assert entity_scores_csv.read_text(encoding="utf-8") == "entity_key\nQ1\n"


def test_refuses_symlinked_output_dir_without_writing_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    outside_output = tmp_path / "outside_output"
    outside_output.mkdir()
    output_dir = tmp_path / "evaluation"
    output_dir.symlink_to(outside_output, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked output dir"):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 0.8)],
        )

    assert not (outside_output / SUMMARY_FILENAME).exists()
    assert not (outside_output / ENTITY_SCORES_CSV_FILENAME).exists()


def test_refuses_non_directory_output_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    output_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="non-directory output path"):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 0.8)],
        )

    assert output_dir.read_text(encoding="utf-8") == "not a directory"


def test_refuses_to_write_into_symlinked_entity_subdir_by_default(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    entities_dir = output_dir / "entities"
    entities_dir.mkdir(parents=True)
    outside_entity_dir = tmp_path / "outside_entity"
    outside_entity_dir.mkdir()
    outside_quality = outside_entity_dir / ENTITY_FILENAME
    (entities_dir / _entity_dir_name_for_test("Q2")).symlink_to(
        outside_entity_dir,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 0.8)],
        )

    assert not outside_quality.exists()
    assert not (output_dir / SUMMARY_FILENAME).exists()
    assert not (output_dir / ENTITY_SCORES_CSV_FILENAME).exists()


def test_refuses_inconsistent_model_type(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    second = _quality("Q2", 0.8)
    second["model_type"] = "bert-base-uncased"

    with pytest.raises(ValueError, match="model_type"):
        write_bertscore_text_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q1", 0.6), second],
        )


def test_non_finite_scores_are_not_written_to_json_or_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    result = _quality("Q1", 0.6)
    result["averages"]["bertscore_overall_text_score"] = float("nan")
    result["text_scores"][0]["bertscore_overall_text_score"] = float("inf")

    summary = write_bertscore_text_quality_outputs(output_dir, run_dir, [result])

    assert summary["averages"]["bertscore_overall_text_score"] is None
    persisted_summary_text = (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert "NaN" not in persisted_summary_text
    assert "Infinity" not in persisted_summary_text
    persisted_entity_text = _entity_quality_paths(output_dir)[0].read_text(
        encoding="utf-8"
    )
    assert "NaN" not in persisted_entity_text
    assert "Infinity" not in persisted_entity_text

    with (output_dir / ENTITY_SCORES_CSV_FILENAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["bertscore_overall_text_score"] == ""
