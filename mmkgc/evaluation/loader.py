"""Load existing MMKGC run artifacts for offline evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class RunEntityArtifacts:
    entity_dir: Path
    entity_key: str
    knowledge: dict[str, Any]
    reasoning: dict[str, Any]
    ranking: dict[str, Any]
    final: dict[str, Any]


@dataclass(frozen=True)
class EntityArtifactLoadResult:
    entity_dir: Path
    entity_key: str
    artifacts: RunEntityArtifacts | None
    error: Exception | None


def validate_run_dir(run_dir: str | Path) -> Path:
    """Return a valid run directory path or raise with the missing requirements."""
    path = Path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"run_dir is not a directory: {path}")

    missing = []
    if not (path / "results.jsonl").is_file():
        missing.append("results.jsonl")
    if not (path / "entities").is_dir():
        missing.append("entities/")
    if missing:
        raise ValueError(f"run_dir is missing required files: {', '.join(missing)}")

    return path


def iter_entity_artifacts(run_dir: str | Path) -> Iterator[RunEntityArtifacts]:
    """Yield stage artifacts for each entity directory in a validated run."""
    path = validate_run_dir(run_dir)
    entities_dir = path / "entities"

    for entity_dir in sorted(entities_dir.iterdir()):
        if not entity_dir.is_dir():
            continue

        yield _load_entity_artifacts(entity_dir)


def iter_entity_artifact_results(run_dir: str | Path) -> Iterator[EntityArtifactLoadResult]:
    """Yield per-entity artifact load results without aborting on bad entities."""
    path = validate_run_dir(run_dir)
    entities_dir = path / "entities"

    for entity_dir in sorted(entities_dir.iterdir()):
        if not entity_dir.is_dir():
            continue

        try:
            artifacts = _load_entity_artifacts(entity_dir)
        except Exception as exc:
            yield EntityArtifactLoadResult(
                entity_dir=entity_dir,
                entity_key=_load_failure_entity_key(entity_dir),
                artifacts=None,
                error=exc,
            )
        else:
            yield EntityArtifactLoadResult(
                entity_dir=entity_dir,
                entity_key=artifacts.entity_key,
                artifacts=artifacts,
                error=None,
            )


def _load_entity_artifacts(entity_dir: Path) -> RunEntityArtifacts:
    final = _read_json_object(entity_dir / "final.json")
    knowledge = _read_json_object(entity_dir / "1_knowledge_modeling.json")
    knowledge_data = knowledge.get("data")
    knowledge_payload = knowledge_data if isinstance(knowledge_data, dict) else knowledge
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key=_entity_key(entity_dir, final),
        knowledge=knowledge,
        reasoning=knowledge_payload.get("constraint_compilation", {}),
        ranking=_read_json_object(entity_dir / "4_kg_aware_candidate_consistency_ranking.json"),
        final=final,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _entity_key(entity_dir: Path, final: dict[str, Any]) -> str:
    entity = final.get("entity")
    if isinstance(entity, dict):
        entity_key = entity.get("entity_key")
        if entity_key:
            return str(entity_key)
    return entity_dir.name


def _load_failure_entity_key(entity_dir: Path) -> str:
    try:
        final = _read_json_object(entity_dir / "final.json")
    except Exception:
        return entity_dir.name
    return _entity_key(entity_dir, final)
