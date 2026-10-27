"""Write Gemini embedding summaries, CSVs, and entity files."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mmkgc.evaluation.gemini_embedding import (
    GEMINI_EMBEDDING_PROVIDER,
    GEMINI_IMAGE_SCORE_FIELDS,
    GEMINI_PAIR_SCORE_FIELDS,
    GEMINI_TEXT_SCORE_FIELDS,
)
from mmkgc.utils import write_json


SUMMARY_FILENAME = "gemini_embedding_summary.json"
ENTITY_SCORES_CSV_FILENAME = "gemini_embedding_entity_scores.csv"
IMAGE_SCORES_CSV_FILENAME = "gemini_embedding_image_scores.csv"
PAIR_SCORES_CSV_FILENAME = "gemini_embedding_pair_scores.csv"
ENTITY_FILENAME = "gemini_embedding.json"

OVERALL_TEXT_FIELD = "gemini_overall_text_embedding_score"
OVERALL_IMAGE_FIELD = "gemini_overall_multimodal_score"
OVERALL_PAIR_FIELD = "gemini_overall_pair_embedding_score"

ENTITY_SCORE_FIELDS = [
    "entity_key",
    "label",
    "missing_mode",
    "text_count",
    "image_count",
    "pair_count",
    *GEMINI_TEXT_SCORE_FIELDS,
    *GEMINI_IMAGE_SCORE_FIELDS,
    *GEMINI_PAIR_SCORE_FIELDS,
    "best_text_candidate_id",
    "best_image_candidate_id",
    "best_pair_id",
]

IMAGE_SCORE_FIELDS = [
    "entity_key",
    "image_candidate_id",
    "image_path",
    "provider",
    "status",
    *GEMINI_IMAGE_SCORE_FIELDS,
]

PAIR_SCORE_FIELDS = [
    "entity_key",
    "pair_id",
    "text_candidate_id",
    "image_candidate_id",
    "provider",
    "status",
    *GEMINI_PAIR_SCORE_FIELDS,
]

SUMMARY_AVERAGE_FIELDS = (
    OVERALL_TEXT_FIELD,
    OVERALL_IMAGE_FIELD,
    OVERALL_PAIR_FIELD,
)

SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "headers",
    "raw_response",
    "raw_request",
}


def write_gemini_embedding_outputs(
    output_dir: str | Path,
    run_dir: str | Path,
    entity_embedding_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write Gemini embedding artifacts and return the summary."""
    output_path = Path(output_dir)
    _raise_if_unsafe_output_dir(output_path, "write")
    entities_dir = output_path / "entities"
    results = [_json_ready(result) for result in entity_embedding_results]
    summary = _build_summary(
        run_dir=run_dir,
        output_dir=output_path,
        entity_embedding_results=results,
    )

    if overwrite:
        _clear_owned_artifacts(output_path)
    else:
        _raise_if_owned_outputs_exist(output_path, results)

    output_path.mkdir(parents=True, exist_ok=True)
    _ensure_entities_dir(entities_dir)
    _raise_if_unsafe_entity_write_targets(entities_dir, results)

    write_json(output_path / SUMMARY_FILENAME, summary)
    _write_entity_scores_csv(output_path / ENTITY_SCORES_CSV_FILENAME, results)
    _write_image_scores_csv(output_path / IMAGE_SCORES_CSV_FILENAME, results)
    _write_pair_scores_csv(output_path / PAIR_SCORES_CSV_FILENAME, results)
    _write_entity_files(entities_dir, results)
    return summary


def _build_summary(
    *,
    run_dir: str | Path,
    output_dir: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "run_dir": str(Path(run_dir)),
        "output_dir": str(output_dir),
        "evaluator": GEMINI_EMBEDDING_PROVIDER,
        "model_name": _consistent_metadata_value(
            entity_embedding_results,
            "gemini_model_name",
        ),
        "evaluated_entities": len(entity_embedding_results),
        "evaluated_text_candidates": sum(
            _text_count(result) for result in entity_embedding_results
        ),
        "evaluated_images": sum(
            _image_count(result) for result in entity_embedding_results
        ),
        "evaluated_pairs": sum(
            _pair_count(result) for result in entity_embedding_results
        ),
        "averages": {
            field: _average(
                _numeric_score(_dict_value(result.get("averages")).get(field))
                for result in entity_embedding_results
            )
            for field in SUMMARY_AVERAGE_FIELDS
        },
        "best_entities": _rank_entities(entity_embedding_results, reverse=True),
        "worst_entities": _rank_entities(entity_embedding_results, reverse=False),
        "output_files": {
            "summary_json": SUMMARY_FILENAME,
            "entity_scores_csv": ENTITY_SCORES_CSV_FILENAME,
            "image_scores_csv": IMAGE_SCORES_CSV_FILENAME,
            "pair_scores_csv": PAIR_SCORES_CSV_FILENAME,
            "entities_dir": "entities/",
        },
    }


def _write_entity_scores_csv(
    path: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENTITY_SCORE_FIELDS)
        writer.writeheader()
        for result in entity_embedding_results:
            writer.writerow(_entity_score_row(result))


def _write_image_scores_csv(
    path: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_SCORE_FIELDS)
        writer.writeheader()
        for result in entity_embedding_results:
            for score in _image_scores(result):
                writer.writerow(_selected_row(score, IMAGE_SCORE_FIELDS))


def _write_pair_scores_csv(
    path: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_SCORE_FIELDS)
        writer.writeheader()
        for result in entity_embedding_results:
            for score in _pair_scores(result):
                writer.writerow(_selected_row(score, PAIR_SCORE_FIELDS))


def _write_entity_files(
    entities_dir: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> None:
    for result in entity_embedding_results:
        entity_key = _entity_key(result)
        write_json(
            entities_dir / _entity_dir_name(entity_key) / ENTITY_FILENAME,
            result,
        )


def _entity_score_row(result: dict[str, Any]) -> dict[str, Any]:
    entity = _dict_value(result.get("entity"))
    averages = _dict_value(result.get("averages"))
    row = {
        "entity_key": _entity_key(result),
        "label": entity.get("label", ""),
        "missing_mode": entity.get("missing_mode", ""),
        "text_count": _text_count(result),
        "image_count": _image_count(result),
        "pair_count": _pair_count(result),
        "best_text_candidate_id": result.get("best_text_candidate_id"),
        "best_image_candidate_id": result.get("best_image_candidate_id"),
        "best_pair_id": result.get("best_pair_id"),
    }
    for field in (
        *GEMINI_TEXT_SCORE_FIELDS,
        *GEMINI_IMAGE_SCORE_FIELDS,
        *GEMINI_PAIR_SCORE_FIELDS,
    ):
        row[field] = _numeric_score(averages.get(field))
    return row


def _selected_row(score: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    row = {}
    for field in fieldnames:
        value = score.get(field)
        row[field] = _numeric_score(value) if field.startswith("gemini_") else value
    return row


def _raise_if_owned_outputs_exist(
    output_dir: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> None:
    for path in _owned_output_paths(output_dir, entity_embedding_results):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Gemini embedding output already exists: {path}. "
                "Use overwrite=True to replace Gemini embedding artifacts."
            )


def _clear_owned_artifacts(output_dir: Path) -> None:
    _raise_if_unsafe_output_dir(output_dir, "overwrite")
    entities_dir = output_dir / "entities"
    entity_dirs = _safe_entity_dirs(entities_dir, "overwrite")

    for filename in (
        SUMMARY_FILENAME,
        ENTITY_SCORES_CSV_FILENAME,
        IMAGE_SCORES_CSV_FILENAME,
        PAIR_SCORES_CSV_FILENAME,
    ):
        _unlink_owned_file(output_dir / filename)

    for entity_dir in entity_dirs:
        path = entity_dir / ENTITY_FILENAME
        if _unlink_owned_file(path):
            try:
                path.parent.rmdir()
            except OSError:
                pass


def _owned_output_paths(
    output_dir: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> list[Path]:
    entities_dir = output_dir / "entities"
    paths = [
        output_dir / SUMMARY_FILENAME,
        output_dir / ENTITY_SCORES_CSV_FILENAME,
        output_dir / IMAGE_SCORES_CSV_FILENAME,
        output_dir / PAIR_SCORES_CSV_FILENAME,
    ]
    paths.extend(
        entities_dir / _entity_dir_name(_entity_key(result)) / ENTITY_FILENAME
        for result in entity_embedding_results
    )
    paths.extend(_existing_entity_output_paths(output_dir))
    return paths


def _existing_entity_output_paths(output_dir: Path) -> list[Path]:
    entities_dir = output_dir / "entities"
    return [
        entity_dir / ENTITY_FILENAME
        for entity_dir in _safe_entity_dirs(entities_dir, "inspect")
    ]


def _unlink_owned_file(path: Path) -> bool:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    if path.exists():
        raise FileExistsError(f"owned output path is not a file: {path}")
    return False


def _raise_if_unsafe_output_dir(output_dir: Path, action: str) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"refusing to {action} symlinked output dir: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"refusing to {action} non-directory output dir: {output_dir}")


def _ensure_entities_dir(entities_dir: Path) -> None:
    if entities_dir.is_symlink():
        raise ValueError(f"refusing to write symlinked entities dir: {entities_dir}")
    entities_dir.mkdir(parents=True, exist_ok=True)


def _raise_if_unsafe_entity_write_targets(
    entities_dir: Path,
    entity_embedding_results: list[dict[str, Any]],
) -> None:
    if entities_dir.is_symlink():
        raise ValueError(f"refusing to write symlinked entities dir: {entities_dir}")
    for result in entity_embedding_results:
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


def _rank_entities(
    entity_embedding_results: list[dict[str, Any]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for result in entity_embedding_results:
        averages = _dict_value(result.get("averages"))
        score = _numeric_score(averages.get(OVERALL_IMAGE_FIELD))
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
        "text_count": _text_count(result),
        "image_count": _image_count(result),
        "pair_count": _pair_count(result),
        OVERALL_IMAGE_FIELD: score,
        "best_text_candidate_id": result.get("best_text_candidate_id"),
        "best_image_candidate_id": result.get("best_image_candidate_id"),
        "best_pair_id": result.get("best_pair_id"),
    }


def _consistent_metadata_value(
    entity_embedding_results: list[dict[str, Any]],
    field: str,
) -> str | None:
    selected: str | None = None
    for result in entity_embedding_results:
        value = _metadata_text(result.get(field))
        if value is None:
            continue
        if selected is None:
            selected = value
            continue
        if selected != value:
            raise ValueError(
                f"inconsistent {field} across Gemini embedding results: "
                f"{selected!r} != {value!r}"
            )
    return selected


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


def _text_count(result: dict[str, Any]) -> int:
    explicit = _optional_int(result.get("text_count"))
    if explicit is not None:
        return max(explicit, 0)
    return len(_text_scores(result))


def _image_count(result: dict[str, Any]) -> int:
    explicit = _optional_int(result.get("image_count"))
    if explicit is not None:
        return max(explicit, 0)
    return len(_image_scores(result))


def _pair_count(result: dict[str, Any]) -> int:
    explicit = _optional_int(result.get("pair_count"))
    if explicit is not None:
        return max(explicit, 0)
    return len(_pair_scores(result))


def _text_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    return _score_list(result.get("text_scores"))


def _image_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    return _score_list(result.get("image_scores"))


def _pair_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    return _score_list(result.get("pair_scores"))


def _score_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _average(values: Any) -> float | None:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _numeric_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_ready(value: Any, key: str | None = None) -> Any:
    if key is not None and key.lower() in SENSITIVE_KEYS:
        return "[redacted]"
    if is_dataclass(value):
        return _json_ready(asdict(value), key=key)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(item_key): _json_ready(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
