import json
from pathlib import Path

import pytest

from mmkgc.evaluation.loader import (
    EntityArtifactLoadResult,
    RunEntityArtifacts,
    iter_entity_artifact_results,
    iter_entity_artifacts,
    validate_run_dir,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_entity(entity_dir: Path, entity_key: str = "Q1") -> None:
    entity_dir.mkdir(parents=True)
    _write_json(entity_dir / "1_knowledge_modeling.json", {"stage": "knowledge"})
    _write_json(entity_dir / "1_knowledge_modeling.json", {"stage": "reasoning"})
    _write_json(entity_dir / "4_kg_aware_candidate_consistency_ranking.json", {"stage": "ranking"})
    _write_json(entity_dir / "final.json", {"entity": {"entity_key": entity_key}})


def test_validate_run_dir_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_run_dir(tmp_path / "missing")


def test_validate_run_dir_rejects_missing_required_files(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "entities").mkdir(parents=True)

    with pytest.raises(ValueError, match="results\\.jsonl"):
        validate_run_dir(run_dir)


def test_iter_entity_artifacts_reads_expected_stage_files(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "entities").mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    _write_minimal_entity(run_dir / "entities" / "Q1")

    artifacts = list(iter_entity_artifacts(run_dir))

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert isinstance(artifact, RunEntityArtifacts)
    assert artifact.entity_dir == run_dir / "entities" / "Q1"
    assert artifact.entity_key == "Q1"
    assert artifact.knowledge == {"stage": "knowledge"}
    assert artifact.reasoning == {"stage": "reasoning"}
    assert artifact.ranking == {"stage": "ranking"}
    assert artifact.final == {"entity": {"entity_key": "Q1"}}


def test_iter_entity_artifacts_skips_non_directories(tmp_path):
    run_dir = tmp_path / "run"
    entities_dir = run_dir / "entities"
    entities_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    (entities_dir / "not-an-entity.txt").write_text("ignore me", encoding="utf-8")
    _write_minimal_entity(entities_dir / "Q1")

    artifacts = list(iter_entity_artifacts(run_dir))

    assert [artifact.entity_key for artifact in artifacts] == ["Q1"]


def test_iter_entity_artifact_results_reports_load_failure_with_final_key(tmp_path):
    run_dir = tmp_path / "run"
    entities_dir = run_dir / "entities"
    entities_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    _write_minimal_entity(entities_dir / "Q1", entity_key="Q_from_final")
    (entities_dir / "Q1" / "4_kg_aware_candidate_consistency_ranking.json").unlink()

    results = list(iter_entity_artifact_results(run_dir))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, EntityArtifactLoadResult)
    assert result.entity_key == "Q_from_final"
    assert result.artifacts is None
    assert isinstance(result.error, FileNotFoundError)


def test_iter_entity_artifact_results_falls_back_to_directory_name_when_final_is_bad(
    tmp_path,
):
    run_dir = tmp_path / "run"
    entities_dir = run_dir / "entities"
    entities_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
    _write_minimal_entity(entities_dir / "Q_bad_final")
    (entities_dir / "Q_bad_final" / "final.json").write_text("{bad json", encoding="utf-8")

    results = list(iter_entity_artifact_results(run_dir))

    assert results[0].entity_key == "Q_bad_final"
    assert results[0].artifacts is None
    assert isinstance(results[0].error, json.JSONDecodeError)
