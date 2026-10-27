"""Lightweight case export helpers for offline evaluation outputs."""

from __future__ import annotations

from typing import Any


def best_cases(entity_evaluations: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Return highest overall-score case records."""
    return _case_records(
        sorted(entity_evaluations, key=_overall_score, reverse=True),
        limit,
    )


def worst_cases(entity_evaluations: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Return lowest overall-score case records."""
    return _case_records(
        sorted(entity_evaluations, key=_overall_score),
        limit,
    )


def fallback_cases(entity_evaluations: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Return fallback-tagged case records."""
    tagged = [
        evaluation
        for evaluation in entity_evaluations
        if "fallback" in _string_list(evaluation.get("case_tags"))
    ]
    return _case_records(tagged, limit)


def case_record(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Build the compact JSONL record used for human case review."""
    entity = _dict_value(evaluation.get("entity"))
    text_evaluation = _dict_value(evaluation.get("text_evaluation"))
    image_evaluation = _dict_value(evaluation.get("image_evaluation"))
    pair_evaluation = _dict_value(evaluation.get("pair_evaluation"))
    calibration_summary = _dict_value(evaluation.get("calibration_summary"))

    return {
        "entity_key": _entity_key(evaluation),
        "label": entity.get("label"),
        "overall_offline_evaluation_score": _score(
            evaluation.get("overall_offline_evaluation_score")
        ),
        "text_best_offline_evaluation_score": _score(
            text_evaluation.get("best_offline_evaluation_score")
        ),
        "image_best_offline_evaluation_score": _score(
            image_evaluation.get("best_offline_evaluation_score")
        ),
        "pair_best_offline_evaluation_score": _score(
            pair_evaluation.get("best_offline_evaluation_score")
        ),
        "calibrated_overall_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_overall_offline_evaluation_score"),
            evaluation.get("overall_offline_evaluation_score"),
        ),
        "calibrated_text_best_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_text_best_offline_evaluation_score"),
            text_evaluation.get("best_offline_evaluation_score"),
        ),
        "calibrated_image_best_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_image_best_offline_evaluation_score"),
            image_evaluation.get("best_offline_evaluation_score"),
        ),
        "calibrated_pair_best_offline_evaluation_score": _score_with_fallback(
            evaluation.get("calibrated_pair_best_offline_evaluation_score"),
            pair_evaluation.get("best_offline_evaluation_score"),
        ),
        "best_text_candidate_id": text_evaluation.get("best_candidate_id"),
        "best_image_candidate_id": image_evaluation.get("best_candidate_id"),
        "best_pair_id": pair_evaluation.get("best_pair_id"),
        "case_tags": _string_list(evaluation.get("case_tags")),
        "calibration_tags": _string_list(calibration_summary.get("calibration_tags")),
    }


def _case_records(
    entity_evaluations: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return [case_record(evaluation) for evaluation in entity_evaluations[:limit]]


def _entity_key(evaluation: dict[str, Any]) -> str:
    entity = _dict_value(evaluation.get("entity"))
    return str(entity.get("entity_key") or entity.get("label") or "entity")


def _overall_score(evaluation: dict[str, Any]) -> float:
    return _score_with_fallback(
        evaluation.get("calibrated_overall_offline_evaluation_score"),
        evaluation.get("overall_offline_evaluation_score"),
    )


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _score(value: Any) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _score_with_fallback(value: Any, fallback: Any) -> float:
    return _score(fallback if value is None else value)
