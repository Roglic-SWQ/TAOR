"""Write BLIP/ITM, Qwen-VL judge, and comparison image quality outputs."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mmkgc.utils import write_json


BLIP_SUMMARY_FILENAME = "blip_itm_summary.json"
BLIP_ENTITY_SCORES_CSV_FILENAME = "blip_itm_entity_scores.csv"
BLIP_ENTITY_FILENAME = "blip_itm.json"
QWEN_SUMMARY_FILENAME = "qwen_vl_judge_summary.json"
QWEN_ENTITY_SCORES_CSV_FILENAME = "qwen_vl_judge_entity_scores.csv"
QWEN_ENTITY_FILENAME = "qwen_vl_judge.json"
COMPARISON_SUMMARY_FILENAME = "image_quality_model_comparison_summary.json"

BLIP_SCORE_FIELDS = (
    "blip_prompt_itm_score",
    "blip_text_itm_score",
    "blip_evidence_itm_score",
    "blip_overall_itm_score",
)
QWEN_SCORE_FIELDS = (
    "entity_match_score",
    "prompt_alignment_score",
    "kg_evidence_score",
    "artifact_score",
    "overall_judge_score",
)
CLIP_SCORE_FIELDS = (
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
)
BERTSCORE_SCORE_FIELDS = (
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
    "bertscore_overall_text_score",
)
GEMINI_SCORE_FIELDS = (
    "gemini_overall_text_embedding_score",
    "gemini_overall_multimodal_score",
    "gemini_overall_pair_embedding_score",
)

BLIP_ENTITY_SCORE_FIELDS = [
    "entity_key",
    "label",
    "missing_mode",
    "image_count",
    "blip_prompt_itm_score",
    "blip_text_itm_score",
    "blip_evidence_itm_score",
    "blip_overall_itm_score",
    "best_image_candidate_id",
    "worst_image_candidate_id",
]
QWEN_ENTITY_SCORE_FIELDS = [
    "entity_key",
    "label",
    "missing_mode",
    "image_count",
    "entity_match_score",
    "prompt_alignment_score",
    "kg_evidence_score",
    "artifact_score",
    "overall_judge_score",
    "best_image_candidate_id",
    "worst_image_candidate_id",
]


def write_blip_itm_outputs(
    output_dir: str | Path,
    run_dir: str | Path,
    entity_itm_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write BLIP/ITM summary, CSV, and per-entity JSON artifacts."""
    output_path = Path(output_dir)
    _raise_if_unsafe_output_dir(output_path, "write")
    entities_dir = output_path / "entities"
    results = [_json_ready(result) for result in entity_itm_results]
    summary = _build_provider_summary(
        run_dir=run_dir,
        output_dir=output_path,
        entity_results=results,
        evaluator="blip-itm-local",
        score_fields=BLIP_SCORE_FIELDS,
        overall_field="blip_overall_itm_score",
        metadata_fields=("model_dir",),
        summary_filename=BLIP_SUMMARY_FILENAME,
        entity_scores_csv_filename=BLIP_ENTITY_SCORES_CSV_FILENAME,
    )

    if overwrite:
        _clear_owned_artifacts(
            output_path,
            summary_filename=BLIP_SUMMARY_FILENAME,
            entity_scores_csv_filename=BLIP_ENTITY_SCORES_CSV_FILENAME,
            entity_filename=BLIP_ENTITY_FILENAME,
        )
    else:
        _raise_if_owned_outputs_exist(
            output_path,
            results,
            summary_filename=BLIP_SUMMARY_FILENAME,
            entity_scores_csv_filename=BLIP_ENTITY_SCORES_CSV_FILENAME,
            entity_filename=BLIP_ENTITY_FILENAME,
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _ensure_entities_dir(entities_dir)
    _raise_if_unsafe_entity_write_targets(entities_dir, results)

    write_json(output_path / BLIP_SUMMARY_FILENAME, summary)
    _write_entity_scores_csv(
        output_path / BLIP_ENTITY_SCORES_CSV_FILENAME,
        results,
        BLIP_ENTITY_SCORE_FIELDS,
        BLIP_SCORE_FIELDS,
    )
    _write_entity_files(entities_dir, results, BLIP_ENTITY_FILENAME)
    return summary


def write_qwen_vl_judge_outputs(
    output_dir: str | Path,
    run_dir: str | Path,
    entity_judge_results: list[dict[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write Qwen-VL judge summary, CSV, and per-entity JSON artifacts."""
    output_path = Path(output_dir)
    _raise_if_unsafe_output_dir(output_path, "write")
    entities_dir = output_path / "entities"
    results = [_json_ready(result) for result in entity_judge_results]
    summary = _build_provider_summary(
        run_dir=run_dir,
        output_dir=output_path,
        entity_results=results,
        evaluator="qwen-vl-judge",
        score_fields=QWEN_SCORE_FIELDS,
        overall_field="overall_judge_score",
        metadata_fields=("model_name", "base_url"),
        summary_filename=QWEN_SUMMARY_FILENAME,
        entity_scores_csv_filename=QWEN_ENTITY_SCORES_CSV_FILENAME,
    )
    summary["parse_failed_images"] = _count_parse_failed_images(results)

    if overwrite:
        _clear_owned_artifacts(
            output_path,
            summary_filename=QWEN_SUMMARY_FILENAME,
            entity_scores_csv_filename=QWEN_ENTITY_SCORES_CSV_FILENAME,
            entity_filename=QWEN_ENTITY_FILENAME,
        )
    else:
        _raise_if_owned_outputs_exist(
            output_path,
            results,
            summary_filename=QWEN_SUMMARY_FILENAME,
            entity_scores_csv_filename=QWEN_ENTITY_SCORES_CSV_FILENAME,
            entity_filename=QWEN_ENTITY_FILENAME,
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _ensure_entities_dir(entities_dir)
    _raise_if_unsafe_entity_write_targets(entities_dir, results)

    write_json(output_path / QWEN_SUMMARY_FILENAME, summary)
    _write_entity_scores_csv(
        output_path / QWEN_ENTITY_SCORES_CSV_FILENAME,
        results,
        QWEN_ENTITY_SCORE_FIELDS,
        QWEN_SCORE_FIELDS,
    )
    _write_entity_files(entities_dir, results, QWEN_ENTITY_FILENAME)
    return summary


def write_image_quality_model_comparison_summary(
    output_dir: str | Path,
    run_dir: str | Path | None = None,
    *,
    clip_summary: dict[str, Any] | None = None,
    blip_summary: dict[str, Any] | None = None,
    qwen_vl_summary: dict[str, Any] | None = None,
    bertscore_summary: dict[str, Any] | None = None,
    gemini_summary: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one comparison summary from available image quality summaries."""
    output_path = Path(output_dir)
    _raise_if_unsafe_output_dir(output_path, "write")
    summaries = {
        "clip": _json_ready(clip_summary) if clip_summary is not None else None,
        "blip_itm": _json_ready(blip_summary) if blip_summary is not None else None,
        "qwen_vl_judge": (
            _json_ready(qwen_vl_summary) if qwen_vl_summary is not None else None
        ),
        "bertscore": (
            _json_ready(bertscore_summary) if bertscore_summary is not None else None
        ),
        "gemini_embedding": (
            _json_ready(gemini_summary) if gemini_summary is not None else None
        ),
    }
    evaluated_images = _consistent_evaluated_images(summaries)
    comparison = {
        "run_dir": _comparison_run_dir(run_dir, summaries),
        "output_dir": str(output_path),
        "evaluated_images": evaluated_images,
        "clip": _comparison_model_record(
            summaries["clip"],
            summary_filename="clip_image_quality_summary.json",
            score_fields=CLIP_SCORE_FIELDS,
            metadata_fields=("clip_model_name", "clip_weight_path"),
        ),
        "blip_itm": _comparison_model_record(
            summaries["blip_itm"],
            summary_filename=BLIP_SUMMARY_FILENAME,
            score_fields=BLIP_SCORE_FIELDS,
            metadata_fields=("model_dir",),
        ),
        "qwen_vl_judge": _comparison_model_record(
            summaries["qwen_vl_judge"],
            summary_filename=QWEN_SUMMARY_FILENAME,
            score_fields=QWEN_SCORE_FIELDS,
            metadata_fields=("model_name", "base_url"),
        ),
        "bertscore": _comparison_model_record(
            summaries["bertscore"],
            summary_filename="bertscore_text_quality_summary.json",
            score_fields=BERTSCORE_SCORE_FIELDS,
            metadata_fields=("model_type",),
            count_fields=("evaluated_text_candidates",),
        ),
        "gemini_embedding": _comparison_model_record(
            summaries["gemini_embedding"],
            summary_filename="gemini_embedding_summary.json",
            score_fields=GEMINI_SCORE_FIELDS,
            metadata_fields=("model_name",),
            count_fields=("evaluated_text_candidates", "evaluated_pairs"),
        ),
        "notes": [
            "CLIP, BLIP/ITM, Qwen-VL, Gemini, and BERTScore measure different "
            "signals; do not treat them as the same metric. CLIP/BLIP/Qwen-VL/"
            "Gemini/BERTScore "
            "衡量不同信号，不应直接当作同一指标。",
        ],
        "output_files": {
            "summary_json": COMPARISON_SUMMARY_FILENAME,
        },
    }

    comparison_path = output_path / COMPARISON_SUMMARY_FILENAME
    if overwrite:
        _unlink_owned_file(comparison_path)
    elif comparison_path.exists() or comparison_path.is_symlink():
        raise FileExistsError(
            f"image quality model comparison output already exists: "
            f"{comparison_path}. Use overwrite=True to replace it."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    write_json(comparison_path, comparison)
    return comparison


def _build_provider_summary(
    *,
    run_dir: str | Path,
    output_dir: Path,
    entity_results: list[dict[str, Any]],
    evaluator: str,
    score_fields: tuple[str, ...],
    overall_field: str,
    metadata_fields: tuple[str, ...],
    summary_filename: str,
    entity_scores_csv_filename: str,
) -> dict[str, Any]:
    summary = {
        "run_dir": str(Path(run_dir)),
        "output_dir": str(output_dir),
        "evaluator": evaluator,
        "evaluated_entities": len(entity_results),
        "evaluated_images": sum(_image_count(result) for result in entity_results),
        "failed_images": _count_failed_images(entity_results),
        "fallback_images": _count_image_flags(entity_results, "fallback_used"),
        "averages": {
            field: _average(
                _numeric_score(_dict_value(result.get("averages")).get(field))
                for result in entity_results
            )
            for field in score_fields
        },
        "best_entities": _rank_entities(
            entity_results,
            overall_field=overall_field,
            reverse=True,
        ),
        "worst_entities": _rank_entities(
            entity_results,
            overall_field=overall_field,
            reverse=False,
        ),
        "output_files": {
            "summary_json": summary_filename,
            "entity_scores_csv": entity_scores_csv_filename,
            "entities_dir": "entities/",
        },
    }
    for metadata_field in metadata_fields:
        summary[metadata_field] = _consistent_metadata_value(
            entity_results,
            metadata_field,
            evaluator=evaluator,
        )
    return summary


def _write_entity_scores_csv(
    path: Path,
    entity_results: list[dict[str, Any]],
    fieldnames: list[str],
    score_fields: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in entity_results:
            writer.writerow(_entity_score_row(result, score_fields))


def _write_entity_files(
    entities_dir: Path,
    entity_results: list[dict[str, Any]],
    entity_filename: str,
) -> None:
    for result in entity_results:
        entity_key = _entity_key(result)
        write_json(
            entities_dir / _entity_dir_name(entity_key) / entity_filename,
            result,
        )


def _raise_if_owned_outputs_exist(
    output_dir: Path,
    entity_results: list[dict[str, Any]],
    *,
    summary_filename: str,
    entity_scores_csv_filename: str,
    entity_filename: str,
) -> None:
    for path in _owned_output_paths(
        output_dir,
        entity_results,
        summary_filename=summary_filename,
        entity_scores_csv_filename=entity_scores_csv_filename,
        entity_filename=entity_filename,
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"image quality output already exists: {path}. "
                "Use overwrite=True to replace owned image quality artifacts."
            )


def _clear_owned_artifacts(
    output_dir: Path,
    *,
    summary_filename: str,
    entity_scores_csv_filename: str,
    entity_filename: str,
) -> None:
    _raise_if_unsafe_output_dir(output_dir, "overwrite")
    entities_dir = output_dir / "entities"
    entity_dirs = _safe_entity_dirs(entities_dir, "overwrite")

    for filename in (summary_filename, entity_scores_csv_filename):
        _unlink_owned_file(output_dir / filename)

    for entity_dir in entity_dirs:
        path = entity_dir / entity_filename
        if _unlink_owned_file(path):
            try:
                path.parent.rmdir()
            except OSError:
                pass


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


def _owned_output_paths(
    output_dir: Path,
    entity_results: list[dict[str, Any]],
    *,
    summary_filename: str,
    entity_scores_csv_filename: str,
    entity_filename: str,
) -> list[Path]:
    entities_dir = output_dir / "entities"
    paths = [
        output_dir / summary_filename,
        output_dir / entity_scores_csv_filename,
    ]
    paths.extend(
        entities_dir / _entity_dir_name(_entity_key(result)) / entity_filename
        for result in entity_results
    )
    paths.extend(_existing_entity_output_paths(output_dir, entity_filename))
    return paths


def _existing_entity_output_paths(output_dir: Path, entity_filename: str) -> list[Path]:
    entities_dir = output_dir / "entities"
    return [
        entity_dir / entity_filename
        for entity_dir in _safe_entity_dirs(entities_dir, "inspect")
    ]


def _ensure_entities_dir(entities_dir: Path) -> None:
    if entities_dir.is_symlink():
        raise ValueError(f"refusing to write symlinked entities dir: {entities_dir}")
    entities_dir.mkdir(parents=True, exist_ok=True)


def _raise_if_unsafe_entity_write_targets(
    entities_dir: Path,
    entity_results: list[dict[str, Any]],
) -> None:
    if entities_dir.is_symlink():
        raise ValueError(f"refusing to write symlinked entities dir: {entities_dir}")
    for result in entity_results:
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


def _entity_score_row(
    result: dict[str, Any],
    score_fields: tuple[str, ...],
) -> dict[str, Any]:
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
    for field in score_fields:
        row[field] = _numeric_score(averages.get(field))
    return row


def _rank_entities(
    entity_results: list[dict[str, Any]],
    *,
    overall_field: str,
    reverse: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for result in entity_results:
        averages = _dict_value(result.get("averages"))
        score = _numeric_score(averages.get(overall_field))
        if score is None:
            continue
        ranked.append((_entity_rank_record(result, overall_field, score), score))

    ranked.sort(key=lambda item: item[1], reverse=reverse)
    return [record for record, _score in ranked]


def _entity_rank_record(
    result: dict[str, Any],
    overall_field: str,
    score: float,
) -> dict[str, Any]:
    entity = _dict_value(result.get("entity"))
    return {
        "entity_key": _entity_key(result),
        "label": entity.get("label", ""),
        "missing_mode": entity.get("missing_mode", ""),
        "image_count": _image_count(result),
        overall_field: score,
        "best_image_candidate_id": result.get("best_image_candidate_id"),
        "worst_image_candidate_id": result.get("worst_image_candidate_id"),
    }


def _comparison_model_record(
    summary: dict[str, Any] | None,
    *,
    summary_filename: str,
    score_fields: tuple[str, ...],
    metadata_fields: tuple[str, ...],
    count_fields: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if summary is None:
        return None

    averages = _dict_value(summary.get("averages"))
    selected_averages = {
        field: _numeric_score(averages.get(field)) for field in score_fields
    }
    output_files = _dict_value(summary.get("output_files"))
    record: dict[str, Any] = {
        "summary_json": output_files.get("summary_json") or summary_filename,
        "evaluator": summary.get("evaluator"),
        "evaluated_entities": _optional_int(summary.get("evaluated_entities")),
        "evaluated_images": _optional_int(summary.get("evaluated_images")),
        "averages": selected_averages,
    }
    for count_field in count_fields:
        record[count_field] = _optional_int(summary.get(count_field))
    for field, value in selected_averages.items():
        record[field] = value
    for metadata_field in metadata_fields:
        record[metadata_field] = summary.get(metadata_field)
    return record


def _comparison_run_dir(
    run_dir: str | Path | None,
    summaries: dict[str, dict[str, Any] | None],
) -> str | None:
    if run_dir is not None:
        return str(Path(run_dir))

    for summary in summaries.values():
        if summary is None:
            continue
        value = summary.get("run_dir")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _consistent_evaluated_images(
    summaries: dict[str, dict[str, Any] | None],
) -> int:
    selected: tuple[str, int] | None = None
    for name, summary in summaries.items():
        if summary is None or "evaluated_images" not in summary:
            continue
        value = _optional_int(summary.get("evaluated_images"))
        if value is None:
            continue
        if selected is None:
            selected = (name, value)
            continue
        selected_name, selected_value = selected
        if value != selected_value:
            raise ValueError(
                "inconsistent evaluated_images across image quality summaries: "
                f"{selected_name}={selected_value} != {name}={value}"
            )
    return selected[1] if selected is not None else 0


def _consistent_metadata_value(
    entity_results: list[dict[str, Any]],
    field: str,
    *,
    evaluator: str,
) -> str | None:
    selected: str | None = None
    for result in entity_results:
        value = _metadata_text(result.get(field))
        if value is None:
            continue
        if selected is None:
            selected = value
            continue
        if selected != value:
            raise ValueError(
                f"inconsistent {field} across {evaluator} results: "
                f"{selected!r} != {value!r}"
            )
    return selected


def _count_failed_images(entity_results: list[dict[str, Any]]) -> int:
    failed_statuses = {"failed", "error", "parse_failed"}
    count = 0
    for result in entity_results:
        for image_score in _image_scores(result):
            status = str(image_score.get("status") or "").strip().lower()
            if status in failed_statuses:
                count += 1
    return count


def _count_image_flags(entity_results: list[dict[str, Any]], field: str) -> int:
    return sum(
        1
        for result in entity_results
        for image_score in _image_scores(result)
        if image_score.get(field) is True
    )


def _count_parse_failed_images(entity_results: list[dict[str, Any]]) -> int:
    count = 0
    for result in entity_results:
        for image_score in _image_scores(result):
            status = str(image_score.get("status") or "").strip().lower()
            if image_score.get("parse_failed") is True or status == "parse_failed":
                count += 1
    return count


def _image_scores(result: dict[str, Any]) -> list[dict[str, Any]]:
    scores = result.get("image_scores")
    if not isinstance(scores, list):
        return []
    return [score for score in scores if isinstance(score, dict)]


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


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
