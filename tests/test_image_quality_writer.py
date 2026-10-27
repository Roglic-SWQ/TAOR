from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mmkgc.evaluation.image_quality_writer import (
    ENTITY_FILENAME,
    ENTITY_SCORES_CSV_FILENAME,
    SUMMARY_FILENAME,
    write_clip_image_quality_outputs,
)


def _quality(entity_key: str, overall: float) -> dict:
    return {
        "entity": {
            "entity_key": entity_key,
            "label": f"Entity {entity_key}",
            "missing_mode": "both",
        },
        "image_count": 2,
        "clip_model_name": "ViT-B/32",
        "clip_weight_path": "CLIP/ViT-B-32.pt",
        "averages": {
            "clip_prompt_similarity": overall + 1,
            "clip_text_similarity": overall + 2,
            "clip_evidence_similarity": overall + 3,
            "clip_overall_image_quality_score": overall,
        },
        "best_image_candidate_id": f"{entity_key}-best",
        "worst_image_candidate_id": f"{entity_key}-worst",
        "image_scores": [
            {
                "image_candidate_id": f"{entity_key}-best",
                "clip_overall_image_quality_score": overall,
            }
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


def test_writes_summary_csv_and_entity_quality_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"

    summary = write_clip_image_quality_outputs(
        output_dir,
        run_dir,
        [_quality("Q1", 10.0), _quality("Q/Unsafe", 30.0)],
    )

    assert summary["evaluated_entities"] == 2
    assert summary["evaluated_images"] == 4
    assert summary["clip_model_name"] == "ViT-B/32"
    assert summary["clip_weight_path"] == "CLIP/ViT-B-32.pt"
    assert summary["averages"]["clip_overall_image_quality_score"] == pytest.approx(
        20.0
    )
    assert summary["best_entities"][0]["entity_key"] == "Q/Unsafe"
    assert summary["worst_entities"][0]["entity_key"] == "Q1"
    assert (output_dir / SUMMARY_FILENAME).is_file()
    assert (output_dir / ENTITY_SCORES_CSV_FILENAME).is_file()

    persisted_summary = json.loads(
        (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted_summary["clip_model_name"] == "ViT-B/32"
    assert persisted_summary["clip_weight_path"] == "CLIP/ViT-B-32.pt"
    assert (
        persisted_summary["output_files"]["entity_scores_csv"]
        == "clip_image_quality_entity_scores.csv"
    )

    with (output_dir / ENTITY_SCORES_CSV_FILENAME).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    rows_by_key = {row["entity_key"]: row for row in rows}
    assert set(rows_by_key) == {"Q1", "Q/Unsafe"}
    assert rows_by_key["Q/Unsafe"]["clip_overall_image_quality_score"] == "30.0"
    assert len(_entity_quality_paths(output_dir)) == 2


def test_overwrite_only_replaces_owned_clip_outputs(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"
    output_dir.mkdir()
    caller_notes = output_dir / "caller_notes.txt"
    caller_notes.write_text("keep me", encoding="utf-8")

    write_clip_image_quality_outputs(output_dir, run_dir, [_quality("Q1", 10.0)])
    write_clip_image_quality_outputs(
        output_dir,
        run_dir,
        [_quality("Q2", 30.0)],
        overwrite=True,
    )

    assert caller_notes.read_text(encoding="utf-8") == "keep me"
    entity_paths = _entity_quality_paths(output_dir)
    assert len(entity_paths) == 1
    persisted_entity = json.loads(entity_paths[0].read_text(encoding="utf-8"))
    assert persisted_entity["entity"]["entity_key"] == "Q2"


def test_overwrite_preserves_other_entity_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"

    write_clip_image_quality_outputs(output_dir, run_dir, [_quality("Q1", 10.0)])
    entity_dir = next((output_dir / "entities").iterdir())
    offline_evaluation = entity_dir / "evaluation.json"
    notes = entity_dir / "notes.txt"
    offline_evaluation.write_text(
        '{"entity":{"entity_key":"Q1"},"kind":"offline"}',
        encoding="utf-8",
    )
    notes.write_text("keep entity notes", encoding="utf-8")

    write_clip_image_quality_outputs(
        output_dir,
        run_dir,
        [_quality("Q2", 30.0)],
        overwrite=True,
    )

    assert not (entity_dir / ENTITY_FILENAME).exists()
    assert (
        json.loads(offline_evaluation.read_text(encoding="utf-8"))["kind"]
        == "offline"
    )
    assert notes.read_text(encoding="utf-8") == "keep entity notes"
    entity_paths = _entity_quality_paths(output_dir)
    assert len(entity_paths) == 1
    persisted_entity = json.loads(entity_paths[0].read_text(encoding="utf-8"))
    assert persisted_entity["entity"]["entity_key"] == "Q2"


def test_overwrite_refuses_symlinked_entities_dir_without_deleting_target(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"
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
    (output_dir / "entities").symlink_to(
        outside_entities,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        write_clip_image_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 30.0)],
            overwrite=True,
        )

    assert json.loads(outside_quality.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert summary.read_text(encoding="utf-8") == '{"existing":true}'
    assert entity_scores_csv.read_text(encoding="utf-8") == "entity_key\nQ1\n"


def test_overwrite_refuses_symlinked_entity_subdir_without_deleting_target(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"
    entities_dir = output_dir / "entities"
    entities_dir.mkdir(parents=True)
    summary = output_dir / SUMMARY_FILENAME
    entity_scores_csv = output_dir / ENTITY_SCORES_CSV_FILENAME
    summary.write_text('{"existing":true}', encoding="utf-8")
    entity_scores_csv.write_text("entity_key\nQ1\n", encoding="utf-8")
    outside_entity_dir = tmp_path / "outside_entity"
    outside_entity_dir.mkdir()
    outside_quality = outside_entity_dir / ENTITY_FILENAME
    outside_quality.write_text(
        '{"entity":{"entity_key":"outside"}}',
        encoding="utf-8",
    )
    (entities_dir / "some_entity").symlink_to(
        outside_entity_dir,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        write_clip_image_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q2", 30.0)],
            overwrite=True,
        )

    assert json.loads(outside_quality.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert summary.read_text(encoding="utf-8") == '{"existing":true}'
    assert entity_scores_csv.read_text(encoding="utf-8") == "entity_key\nQ1\n"


def test_refuses_to_overwrite_existing_owned_outputs_by_default(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"

    write_clip_image_quality_outputs(output_dir, run_dir, [_quality("Q1", 10.0)])

    with pytest.raises(FileExistsError, match=SUMMARY_FILENAME):
        write_clip_image_quality_outputs(output_dir, run_dir, [_quality("Q2", 30.0)])


def test_refuses_stale_entity_quality_output_by_default(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"
    stale_entity_dir = output_dir / "entities" / "stale"
    stale_entity_dir.mkdir(parents=True)
    stale_quality = stale_entity_dir / ENTITY_FILENAME
    stale_quality.write_text(
        '{"entity":{"entity_key":"stale"}}',
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match=ENTITY_FILENAME):
        write_clip_image_quality_outputs(output_dir, run_dir, [_quality("Q2", 30.0)])

    assert json.loads(stale_quality.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "stale"


def test_refuses_to_write_into_symlinked_entity_subdir_by_default(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"
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
        write_clip_image_quality_outputs(output_dir, run_dir, [_quality("Q2", 30.0)])

    assert not outside_quality.exists()
    assert not (output_dir / SUMMARY_FILENAME).exists()
    assert not (output_dir / ENTITY_SCORES_CSV_FILENAME).exists()


def test_refuses_inconsistent_clip_metadata(tmp_path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "quality"
    second = _quality("Q2", 30.0)
    second["clip_weight_path"] = "CLIP/other.pt"

    with pytest.raises(ValueError, match="clip_weight_path"):
        write_clip_image_quality_outputs(
            output_dir,
            run_dir,
            [_quality("Q1", 10.0), second],
        )
