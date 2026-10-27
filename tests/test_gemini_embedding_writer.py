from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mmkgc.evaluation.gemini_embedding_writer import (
    ENTITY_FILENAME,
    ENTITY_SCORES_CSV_FILENAME,
    IMAGE_SCORES_CSV_FILENAME,
    PAIR_SCORES_CSV_FILENAME,
    SUMMARY_FILENAME,
    write_gemini_embedding_outputs,
)


def _embedding_result(entity_key: str, overall: float) -> dict:
    return {
        "entity": {
            "entity_key": entity_key,
            "label": f"Entity {entity_key}",
            "missing_mode": "both",
        },
        "provider": "gemini-api",
        "gemini_model_name": "gemini-embedding-2",
        "text_count": 2,
        "image_count": 1,
        "pair_count": 1,
        "averages": {
            "gemini_text_ground_truth_similarity": overall + 1,
            "gemini_text_evidence_similarity": overall + 2,
            "gemini_overall_text_embedding_score": overall + 3,
            "gemini_prompt_similarity": overall + 4,
            "gemini_text_similarity": overall + 5,
            "gemini_evidence_similarity": overall + 6,
            "gemini_overall_multimodal_score": overall,
            "gemini_pair_image_text_similarity": overall + 7,
            "gemini_pair_image_evidence_similarity": overall + 8,
            "gemini_pair_text_evidence_similarity": overall + 9,
            "gemini_overall_pair_embedding_score": overall + 10,
        },
        "best_text_candidate_id": f"{entity_key}-text-best",
        "best_image_candidate_id": f"{entity_key}-image-best",
        "best_pair_id": f"{entity_key}-pair-best",
        "text_scores": [
            {
                "entity_key": entity_key,
                "text_candidate_id": f"{entity_key}-text-1",
                "status": "scored",
                "gemini_text_ground_truth_similarity": overall + 1,
                "gemini_text_evidence_similarity": overall + 2,
                "gemini_overall_text_embedding_score": overall + 3,
            },
            {
                "entity_key": entity_key,
                "text_candidate_id": f"{entity_key}-text-2",
                "status": "scored",
                "gemini_text_ground_truth_similarity": overall + 2,
                "gemini_text_evidence_similarity": overall + 3,
                "gemini_overall_text_embedding_score": overall + 4,
            },
        ],
        "image_scores": [
            {
                "entity_key": entity_key,
                "image_candidate_id": f"{entity_key}-image-1",
                "image_path": "generated_images/image_001.png",
                "provider": "gemini-api",
                "status": "scored",
                "gemini_prompt_similarity": overall + 4,
                "gemini_text_similarity": overall + 5,
                "gemini_evidence_similarity": overall + 6,
                "gemini_overall_multimodal_score": overall,
            }
        ],
        "pair_scores": [
            {
                "entity_key": entity_key,
                "pair_id": f"{entity_key}-pair-1",
                "text_candidate_id": f"{entity_key}-text-1",
                "image_candidate_id": f"{entity_key}-image-1",
                "provider": "gemini-api",
                "status": "scored",
                "gemini_pair_image_text_similarity": overall + 7,
                "gemini_pair_image_evidence_similarity": overall + 8,
                "gemini_pair_text_evidence_similarity": overall + 9,
                "gemini_overall_pair_embedding_score": overall + 10,
            }
        ],
    }


def _entity_dir_name(entity_key: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in entity_key
    )
    slug = safe.strip("._") or "entity"
    digest = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _entity_paths(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "entities").glob(f"*/{ENTITY_FILENAME}"))


def test_writes_summary_csvs_and_entity_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    summary = write_gemini_embedding_outputs(
        output_dir,
        run_dir,
        [_embedding_result("Q1", 70.0), _embedding_result("Q/Unsafe", 90.0)],
    )

    assert summary["run_dir"] == str(run_dir)
    assert summary["output_dir"] == str(output_dir)
    assert summary["evaluator"] == "gemini-api"
    assert summary["model_name"] == "gemini-embedding-2"
    assert summary["evaluated_entities"] == 2
    assert summary["evaluated_text_candidates"] == 4
    assert summary["evaluated_images"] == 2
    assert summary["evaluated_pairs"] == 2
    assert summary["averages"]["gemini_overall_multimodal_score"] == pytest.approx(
        80.0
    )
    assert summary["best_entities"][0]["entity_key"] == "Q/Unsafe"
    assert summary["worst_entities"][0]["entity_key"] == "Q1"
    assert summary["output_files"] == {
        "summary_json": SUMMARY_FILENAME,
        "entity_scores_csv": ENTITY_SCORES_CSV_FILENAME,
        "image_scores_csv": IMAGE_SCORES_CSV_FILENAME,
        "pair_scores_csv": PAIR_SCORES_CSV_FILENAME,
        "entities_dir": "entities/",
    }

    assert (output_dir / SUMMARY_FILENAME).is_file()
    assert (output_dir / ENTITY_SCORES_CSV_FILENAME).is_file()
    assert (output_dir / IMAGE_SCORES_CSV_FILENAME).is_file()
    assert (output_dir / PAIR_SCORES_CSV_FILENAME).is_file()

    persisted_summary = json.loads(
        (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert persisted_summary["evaluator"] == "gemini-api"

    entity_path = (
        output_dir
        / "entities"
        / _entity_dir_name("Q/Unsafe")
        / ENTITY_FILENAME
    )
    assert json.loads(entity_path.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "Q/Unsafe"

    entity_rows = {
        row["entity_key"]: row
        for row in _read_csv(output_dir / ENTITY_SCORES_CSV_FILENAME)
    }
    assert set(entity_rows) == {"Q1", "Q/Unsafe"}
    assert entity_rows["Q1"]["gemini_overall_multimodal_score"] == "70.0"
    assert entity_rows["Q1"]["best_image_candidate_id"] == "Q1-image-best"

    image_rows = _read_csv(output_dir / IMAGE_SCORES_CSV_FILENAME)
    assert len(image_rows) == 2
    assert image_rows[0]["image_candidate_id"] == "Q1-image-1"
    assert image_rows[0]["image_path"] == "generated_images/image_001.png"

    pair_rows = _read_csv(output_dir / PAIR_SCORES_CSV_FILENAME)
    assert len(pair_rows) == 2
    assert pair_rows[0]["pair_id"] == "Q1-pair-1"
    assert pair_rows[0]["text_candidate_id"] == "Q1-text-1"


def test_overwrite_only_replaces_gemini_owned_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    clip_summary = output_dir / "clip_image_quality_summary.json"
    clip_summary.write_text('{"keep":"clip"}', encoding="utf-8")
    caller_notes = output_dir / "caller_notes.txt"
    caller_notes.write_text("keep me", encoding="utf-8")

    write_gemini_embedding_outputs(
        output_dir,
        run_dir,
        [_embedding_result("Q1", 70.0)],
    )
    first_entity_dir = next((output_dir / "entities").iterdir())
    clip_entity = first_entity_dir / "clip_image_quality.json"
    clip_entity.write_text('{"keep":"clip-entity"}', encoding="utf-8")

    write_gemini_embedding_outputs(
        output_dir,
        run_dir,
        [_embedding_result("Q2", 90.0)],
        overwrite=True,
    )

    assert clip_summary.read_text(encoding="utf-8") == '{"keep":"clip"}'
    assert caller_notes.read_text(encoding="utf-8") == "keep me"
    assert clip_entity.read_text(encoding="utf-8") == '{"keep":"clip-entity"}'
    assert not (first_entity_dir / ENTITY_FILENAME).exists()
    entity_paths = _entity_paths(output_dir)
    assert len(entity_paths) == 1
    assert json.loads(entity_paths[0].read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "Q2"


def test_refuses_existing_owned_outputs_without_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"

    write_gemini_embedding_outputs(
        output_dir,
        run_dir,
        [_embedding_result("Q1", 70.0)],
    )

    with pytest.raises(FileExistsError, match=SUMMARY_FILENAME):
        write_gemini_embedding_outputs(
            output_dir,
            run_dir,
            [_embedding_result("Q2", 90.0)],
        )


def test_refuses_stale_entity_gemini_output_without_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    stale_dir = output_dir / "entities" / "stale"
    stale_dir.mkdir(parents=True)
    stale_output = stale_dir / ENTITY_FILENAME
    stale_output.write_text('{"entity":{"entity_key":"stale"}}', encoding="utf-8")

    with pytest.raises(FileExistsError, match=ENTITY_FILENAME):
        write_gemini_embedding_outputs(
            output_dir,
            run_dir,
            [_embedding_result("Q2", 90.0)],
        )

    assert json.loads(stale_output.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "stale"


def test_overwrite_refuses_symlinked_entities_dir_without_deleting_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / SUMMARY_FILENAME).write_text('{"existing":true}', encoding="utf-8")
    outside_entities = tmp_path / "outside_entities"
    outside_entity_dir = outside_entities / "external"
    outside_entity_dir.mkdir(parents=True)
    outside_gemini = outside_entity_dir / ENTITY_FILENAME
    outside_gemini.write_text('{"entity":{"entity_key":"outside"}}', encoding="utf-8")
    (output_dir / "entities").symlink_to(outside_entities, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_gemini_embedding_outputs(
            output_dir,
            run_dir,
            [_embedding_result("Q2", 90.0)],
            overwrite=True,
        )

    assert json.loads(outside_gemini.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8") == (
        '{"existing":true}'
    )


def test_overwrite_refuses_symlinked_entity_subdir_without_deleting_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    entities_dir = output_dir / "entities"
    entities_dir.mkdir(parents=True)
    (output_dir / SUMMARY_FILENAME).write_text('{"existing":true}', encoding="utf-8")
    outside_entity_dir = tmp_path / "outside_entity"
    outside_entity_dir.mkdir()
    outside_gemini = outside_entity_dir / ENTITY_FILENAME
    outside_gemini.write_text('{"entity":{"entity_key":"outside"}}', encoding="utf-8")
    (entities_dir / "some_entity").symlink_to(
        outside_entity_dir,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        write_gemini_embedding_outputs(
            output_dir,
            run_dir,
            [_embedding_result("Q2", 90.0)],
            overwrite=True,
        )

    assert json.loads(outside_gemini.read_text(encoding="utf-8"))["entity"][
        "entity_key"
    ] == "outside"
    assert (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8") == (
        '{"existing":true}'
    )


def test_normal_write_refuses_symlinked_output_and_entity_targets(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    outside_output = tmp_path / "outside_output"
    outside_output.mkdir()
    symlinked_output = tmp_path / "evaluation_symlink"
    symlinked_output.symlink_to(outside_output, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked output dir"):
        write_gemini_embedding_outputs(
            symlinked_output,
            run_dir,
            [_embedding_result("Q2", 90.0)],
        )

    assert not (outside_output / SUMMARY_FILENAME).exists()

    output_dir = tmp_path / "evaluation"
    entities_dir = output_dir / "entities"
    outside_entities = tmp_path / "outside_entities_normal"
    outside_entities.mkdir()
    output_dir.mkdir()
    entities_dir.symlink_to(outside_entities, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinked entities dir"):
        write_gemini_embedding_outputs(
            output_dir,
            run_dir,
            [_embedding_result("Q2", 90.0)],
        )

    entities_dir.unlink()
    entities_dir.mkdir()
    outside_entity_dir = tmp_path / "outside_entity_normal"
    outside_entity_dir.mkdir()
    (entities_dir / _entity_dir_name("Q2")).symlink_to(
        outside_entity_dir,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlinked entity dir"):
        write_gemini_embedding_outputs(
            output_dir,
            run_dir,
            [_embedding_result("Q2", 90.0)],
        )

    assert not (outside_entity_dir / ENTITY_FILENAME).exists()


def test_non_finite_scores_are_not_written_to_json_or_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    result = _embedding_result("Q1", 70.0)
    result["averages"]["gemini_overall_multimodal_score"] = float("nan")
    result["image_scores"][0]["gemini_overall_multimodal_score"] = float("inf")
    result["pair_scores"][0]["gemini_overall_pair_embedding_score"] = -float("inf")

    summary = write_gemini_embedding_outputs(output_dir, run_dir, [result])

    assert summary["averages"]["gemini_overall_multimodal_score"] is None
    assert summary["best_entities"] == []
    persisted_summary_text = (output_dir / SUMMARY_FILENAME).read_text(
        encoding="utf-8"
    )
    persisted_entity_text = _entity_paths(output_dir)[0].read_text(encoding="utf-8")
    assert "NaN" not in persisted_summary_text
    assert "Infinity" not in persisted_summary_text
    assert "NaN" not in persisted_entity_text
    assert "Infinity" not in persisted_entity_text

    image_row = _read_csv(output_dir / IMAGE_SCORES_CSV_FILENAME)[0]
    pair_row = _read_csv(output_dir / PAIR_SCORES_CSV_FILENAME)[0]
    assert image_row["gemini_overall_multimodal_score"] == ""
    assert pair_row["gemini_overall_pair_embedding_score"] == ""


def test_sensitive_fields_are_redacted_from_persisted_entity_json(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    result = _embedding_result("Q1", 70.0)
    result["api_key"] = "secret-value"
    result["headers"] = {"Authorization": "Bearer secret-value"}
    result["raw_request"] = {"body": "secret request"}
    result["raw_response"] = {"body": "secret response"}

    write_gemini_embedding_outputs(output_dir, run_dir, [result])

    persisted_entity_text = _entity_paths(output_dir)[0].read_text(encoding="utf-8")
    assert "secret-value" not in persisted_entity_text
    assert "secret request" not in persisted_entity_text
    assert "secret response" not in persisted_entity_text
    persisted_entity = json.loads(persisted_entity_text)
    assert persisted_entity["api_key"] == "[redacted]"
    assert persisted_entity["headers"] == "[redacted]"
    assert persisted_entity["raw_request"] == "[redacted]"
    assert persisted_entity["raw_response"] == "[redacted]"
