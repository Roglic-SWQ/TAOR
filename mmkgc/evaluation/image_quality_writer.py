"""Write local CLIP image quality summaries, CSVs, and entity files."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mmkgc.utils import write_json


SUMMARY_FILENAME = "clip_image_quality_summary.json"
ENTITY_SCORES_CSV_FILENAME = "clip_image_quality_entity_scores.csv"
ENTITY_FILENAME = "clip_image_quality.json"
CLIP_SCORE_FIELDS = (
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
)

ENTITY_SCORE_FIELDS = [
    "entity_key",
    "label",
    "missing_mode",
    "image_count",
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
    "best_image_candidate_id",
    "worst_image_candidate_id",
]


def write_clip_image_quality_outputs(
    output_dir: str | Path,
    run_dir: str | Path,
    entity_quality_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write CLIP image quality artifacts and return the summary."""
    output_path = Path(output_dir)
    entities_dir = output_path / "entities"
    results = [_json_ready(result) for result in entity_quality_results]
    summary = _build_summary(
        run_dir=run_dir,
        output_dir=output_path,
        entity_quality_results=results,
    )

    if overwrite:
        _clear_owned_artifacts(output_path)
    else:
        _raise_if_owned_outputs_exist(output_path, results)

    output_path.mkdir(parents=True, exist_ok=True)
    entities_dir.mkdir(parents=True, exist_ok=True)
    _raise_if_unsafe_entity_write_targets(entities_dir, results)

    write_json(output_path / SUMMARY_FILENAME, summary)
    _write_entity_scores_csv(output_path / ENTITY_SCORES_CSV_FILENAME, results)
    _write_entity_files(entities_dir, results)

    return summary


def _build_summary(
    *,
    run_dir: str | Path,
    output_dir: Path,
    entity_quality_results: list[dict[str, Any]],
) -> dict[str, Any]:
    clip_metadata = _clip_metadata(entity_quality_results)
    return {
        "run_dir": str(Path(run_dir)),
        "output_dir": str(output_dir),
        "evaluator": "clip-local",
        "clip_model_name": clip_metadata["clip_model_name"],
        "clip_weight_path": clip_metadata["clip_weight_path"],
        "evaluated_entities": len(entity_quality_results),
        "evaluated_images": sum(
            _image_count(result) for result in entity_quality_results
        ),
        "averages": {
            field: _average(
                _numeric_score(_dict_value(result.get("averages")).get(field))
                for result in entity_quality_results
            )
            for field in CLIP_SCORE_FIELDS
        },
        "best_entities": _rank_entities(entity_quality_results, reverse=True),
        "worst_entities": _rank_entities(entity_quality_results, reverse=False),
        "output_files": {
            "summary_json": SUMMARY_FILENAME,
            "entity_scores_csv": ENTITY_SCORES_CSV_FILENAME,
            "entities_dir": "entities/",
        },
    }


def _clip_metadata(
    entity_quality_results: list[dict[str, Any]],
) -> dict[str, str | None]:
    return {
        "clip_model_name": _consistent_metadata_value(
            entity_quality_results,
            "clip_model_name",
        ),
        "clip_weight_path": _consistent_metadata_value(
            entity_quality_results,
            "clip_weight_path",
        ),
    }


def _consistent_metadata_value(
    entity_quality_results: list[dict[str, Any]],
    field: str,
) -> str | None:
    selected: str | None = None
    for result in entity_quality_results:
        value = _metadata_text(result.get(field))
        if value is None:
            continue
        if selected is None:
            selected = value
            continue
        if selected != value:
            raise ValueError(
                f"inconsistent {field} across CLIP image quality results: "
                f"{selected!r} != {value!r}"
            )
    return selected


def _write_entity_scores_csv(
    path: Path,
    entity_quality_results: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENTITY_SCORE_FIELDS)
        writer.writeheader()
        for result in entity_quality_results:
            writer.writerow(_entity_score_row(result))


def _write_entity_files(
    entities_dir: Path,
    entity_quality_results: list[dict[str, Any]],
) -> None:
    for result in entity_quality_results:
        entity_key = _entity_key(result)
        write_json(
            entities_dir / _entity_dir_name(entity_key) / ENTITY_FILENAME,
            result,
        )


def _raise_if_owned_outputs_exist(
    output_dir: Path,
    entity_quality_results: list[dict[str, Any]],
) -> None:
    for path in _owned_output_paths(output_dir, entity_quality_results):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"CLIP image quality output already exists: {path}. "
                "Use overwrite=True to replace CLIP image quality artifacts."
            )


def _clear_owned_artifacts(output_dir: Path) -> None:
    entities_dir = output_dir / "entities"
    entity_dirs = _safe_entity_dirs(entities_dir, "overwrite")

    for filename in (SUMMARY_FILENAME, ENTITY_SCORES_CSV_FILENAME):
        path = output_dir / filename
        if path.is_symlink() or path.is_file():
            path.unlink()

    for entity_dir in entity_dirs:
        path = entity_dir / ENTITY_FILENAME
        if path.is_symlink() or path.is_file():
            path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass


def _owned_output_paths(
    output_dir: Path,
    entity_quality_results: list[dict[str, Any]],
) -> list[Path]:
    entities_dir = output_dir / "entities"
    paths = [
        output_dir / SUMMARY_FILENAME,
        output_dir / ENTITY_SCORES_CSV_FILENAME,
    ]
    paths.extend(
        entities_dir / _entity_dir_name(_entity_key(result)) / ENTITY_FILENAME
        for result in entity_quality_results
    )
    paths.extend(_existing_entity_quality_paths(output_dir))
    return paths


def _existing_entity_quality_paths(output_dir: Path) -> list[Path]:
    entities_dir = output_dir / "entities"
    return [
        entity_dir / ENTITY_FILENAME
        for entity_dir in _safe_entity_dirs(entities_dir, "inspect")
    ]


def _raise_if_unsafe_entity_write_targets(
    entities_dir: Path,
    entity_quality_results: list[dict[str, Any]],
) -> None:
    for result in entity_quality_results:
        entity_dir = entities_dir / _entity_dir_name(_entity_key(result))
        if entity_dir.is_symlink():
            raise ValueError(f"refusing to write symlinked entity dir: {entity_dir}")


def _safe_entity_dirs(entities_dir: Path, action: str) -> list[Path]:
    if entities_dir.is_symlink():
        raise ValueError(
            f"refusing to {action} symlinked entities dir: {entities_dir}"
        )
    if not entities_dir.is_dir():
        return []

    entity_dirs: list[Path] = []
    for child in sorted(entities_dir.iterdir()):
        if child.is_symlink():
            raise ValueError(f"refusing to {action} symlinked entity dir: {child}")
        if child.is_dir():
            entity_dirs.append(child)
    return entity_dirs


def _entity_score_row(result: dict[str, Any]) -> dict[str, Any]:
    entity = _dict_value(result.get("entity"))
    averages = _dict_value(result.get("averages"))
    row = {
        "entity_key": _entity_key(result),
        "label": entity.get("label", ""),
        "missing_mode": entity.get("missing_mode", ""),
        "image_count": _image_count(result),
        "best_image_candidate_id": result.get("best_image_candidate_id"),
        "worst_image_candidate_id": result.get("worst_image_candidate_id"),
    }
    for field in CLIP_SCORE_FIELDS:
        row[field] = _numeric_score(averages.get(field))
    return row


def _rank_entities(
    entity_quality_results: list[dict[str, Any]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for result in entity_quality_results:
        averages = _dict_value(result.get("averages"))
        score = _numeric_score(averages.get("clip_overall_image_quality_score"))
        if score is None:
            continue
        ranked.append((_entity_rank_record(result, score), score))

    ranked.sort(key=lambda item: item[1], reverse=reverse)
    return [record for record, _score in ranked]


def _entity_rank_record(result: dict[str, Any], score: float) -> dict[str, Any]:
    entity = _dict_value(result.get("entity"))
    return {
        "entity_key": _entity_key(result),
        "label": entity.get("label", ""),
        "missing_mode": entity.get("missing_mode", ""),
        "image_count": _image_count(result),
        "clip_overall_image_quality_score": score,
        "best_image_candidate_id": result.get("best_image_candidate_id"),
        "worst_image_candidate_id": result.get("worst_image_candidate_id"),
    }


def _average(values: Any) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _image_count(result: dict[str, Any]) -> int:
    try:
        return max(int(result.get("image_count") or 0), 0)
    except (TypeError, ValueError):
        return 0


def _entity_key(result: dict[str, Any]) -> str:
    entity = _dict_value(result.get("entity"))
    return str(entity.get("entity_key") or entity.get("label") or "entity")


def _entity_dir_name(entity_key: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in entity_key
    )
    slug = safe.strip("._") or "entity"
    digest = hashlib.sha256(entity_key.encode("utf-8")).hexdigest()[:8]
    return f"{slug}__{digest}"


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _numeric_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
