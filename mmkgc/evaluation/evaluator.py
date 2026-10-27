"""Entity-level orchestration for deterministic offline evaluation."""

from __future__ import annotations

from typing import Any, Callable

from mmkgc.evaluation.calibration import calibrate_entity_scores
from mmkgc.evaluation.loader import RunEntityArtifacts
from mmkgc.evaluation.metrics import (
    average_top,
    is_fallback_candidate,
    score_image_candidate,
    score_pair,
    score_text_candidate,
)


LOW_SCORE_THRESHOLD = 0.25


def evaluate_entity(artifacts: RunEntityArtifacts) -> dict[str, Any]:
    """Evaluate one entity's text, image, and pair ranking artifacts."""
    entity = _dict_value(artifacts.final.get("entity")).copy()
    entity.setdefault("entity_key", artifacts.entity_key)

    entity_label = _string_or_none(entity.get("label"))
    ground_truth_text = _string_or_none(entity.get("description")) or ""
    entity_type = _entity_type(artifacts)
    missing_mode = _missing_mode(artifacts)
    entity["entity_type"] = entity_type
    entity["missing_mode"] = missing_mode

    warnings: list[str] = []
    case_tags: set[str] = set()

    text_candidates = _list_value(
        _dict_value(artifacts.ranking.get("text_ranking")).get("ranked_candidates")
    )
    image_candidates = _list_value(
        _dict_value(artifacts.ranking.get("image_ranking")).get("ranked_candidates")
    )
    pair_candidates = _list_value(
        _dict_value(artifacts.ranking.get("pair_ranking")).get("ranked_pairs")
    )

    text_evaluation = _evaluate_candidates(
        candidates=text_candidates,
        scorer=lambda candidate: score_text_candidate(
            candidate,
            ground_truth_text,
            entity_label,
            entity_type,
        ),
        id_key_name="best_candidate_id",
        summary_key="ranked_candidates",
        missing_tag="missing_text_candidates",
        missing_warning="no text candidates available for entity evaluation",
        case_tags=case_tags,
        warnings=warnings,
    )
    image_evaluation = _evaluate_candidates(
        candidates=image_candidates,
        scorer=lambda candidate: score_image_candidate(candidate, entity_label, entity_type),
        id_key_name="best_candidate_id",
        summary_key="ranked_candidates",
        missing_tag="missing_image_candidates",
        missing_warning="no image candidates available for entity evaluation",
        case_tags=case_tags,
        warnings=warnings,
    )
    pair_evaluation = _evaluate_candidates(
        candidates=pair_candidates,
        scorer=score_pair,
        id_key_name="best_pair_id",
        summary_key="ranked_pairs",
        missing_tag="missing_pair_candidates",
        missing_warning="no pair candidates available for entity evaluation",
        case_tags=case_tags,
        warnings=warnings,
        pair_ids=True,
        report_missing=False,
    )

    fallback_summary = _fallback_summary(text_candidates, image_candidates, pair_candidates)
    if fallback_summary["has_fallback"]:
        case_tags.add("fallback")

    available_best_scores = [
        evaluation["best_offline_evaluation_score"]
        for evaluation in (text_evaluation, image_evaluation, pair_evaluation)
        if evaluation["candidate_count"] > 0
    ]
    overall_offline_evaluation_score = _round_score(
        sum(available_best_scores) / len(available_best_scores)
        if available_best_scores
        else 0.0
    )
    if overall_offline_evaluation_score < LOW_SCORE_THRESHOLD:
        case_tags.add("low_score")

    calibration = calibrate_entity_scores(
        entity=entity,
        text_evaluation=text_evaluation,
        image_evaluation=image_evaluation,
        pair_evaluation=pair_evaluation,
        fallback_summary=fallback_summary,
    )
    case_tags.update(calibration["calibration_summary"]["calibration_tags"])

    return {
        "entity": entity,
        "text_evaluation": text_evaluation,
        "image_evaluation": image_evaluation,
        "pair_evaluation": pair_evaluation,
        "fallback_summary": fallback_summary,
        "overall_offline_evaluation_score": overall_offline_evaluation_score,
        "calibrated_text_best_offline_evaluation_score": calibration[
            "calibrated_text_best_offline_evaluation_score"
        ],
        "calibrated_image_best_offline_evaluation_score": calibration[
            "calibrated_image_best_offline_evaluation_score"
        ],
        "calibrated_pair_best_offline_evaluation_score": calibration[
            "calibrated_pair_best_offline_evaluation_score"
        ],
        "calibrated_overall_offline_evaluation_score": calibration[
            "calibrated_overall_offline_evaluation_score"
        ],
        "calibration_summary": calibration["calibration_summary"],
        "case_tags": sorted(case_tags),
        "warnings": warnings,
    }


def _evaluate_candidates(
    *,
    candidates: list[dict[str, Any]],
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
    id_key_name: str,
    summary_key: str,
    missing_tag: str,
    missing_warning: str,
    case_tags: set[str],
    warnings: list[str],
    pair_ids: bool = False,
    report_missing: bool = True,
) -> dict[str, Any]:
    scored = [_candidate_summary(candidate, scorer(candidate), pair_ids) for candidate in candidates]
    scored.sort(key=lambda item: item["offline_evaluation_score"], reverse=True)

    if not scored:
        if report_missing:
            case_tags.add(missing_tag)
            warnings.append(missing_warning)
        return {
            "candidate_count": 0,
            id_key_name: None,
            "best_offline_evaluation_score": 0.0,
            "top5_avg_offline_evaluation_score": 0.0,
            summary_key: [],
        }

    best = scored[0]
    return {
        "candidate_count": len(scored),
        id_key_name: best["pair_id"] if pair_ids else best["candidate_id"],
        "best_offline_evaluation_score": best["offline_evaluation_score"],
        "top5_avg_offline_evaluation_score": average_top(
            scored,
            "offline_evaluation_score",
            limit=5,
        ),
        summary_key: scored,
    }


def _candidate_summary(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    pair_ids: bool,
) -> dict[str, Any]:
    identifier_key = "pair_id" if pair_ids else "candidate_id"
    summary = {
        identifier_key: _candidate_id(candidate, pair_ids),
        "offline_evaluation_score": metrics["offline_evaluation_score"],
    }
    for key, value in metrics.items():
        if key != "offline_evaluation_score":
            summary[key] = value
    for source_key in (
        "primary_relation",
        "relation_role",
        "metadata",
        "expected_relation_coverage_detail",
        "generated_content",
        "prompt_or_instruction",
        "positive_prompt",
        "text_candidate",
        "image_candidate",
    ):
        if source_key in candidate:
            summary[source_key] = candidate[source_key]
    if not pair_ids:
        summary["is_fallback"] = is_fallback_candidate(candidate)
    return summary


def _fallback_summary(
    text_candidates: list[dict[str, Any]],
    image_candidates: list[dict[str, Any]],
    pair_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    text_fallback_count = sum(1 for candidate in text_candidates if is_fallback_candidate(candidate))
    image_fallback_count = sum(1 for candidate in image_candidates if is_fallback_candidate(candidate))
    fallback_pair_count = sum(
        1
        for pair in pair_candidates
        if is_fallback_candidate(pair)
        or is_fallback_candidate(_dict_value(pair.get("text_candidate")))
        or is_fallback_candidate(_dict_value(pair.get("image_candidate")))
    )
    fallback_candidate_count = text_fallback_count + image_fallback_count
    return {
        "fallback_candidate_count": fallback_candidate_count,
        "has_fallback": fallback_candidate_count > 0 or fallback_pair_count > 0,
        "text_fallback_candidate_count": text_fallback_count,
        "image_fallback_candidate_count": image_fallback_count,
        "fallback_pair_count": fallback_pair_count,
    }


def _missing_mode(artifacts: RunEntityArtifacts) -> str:
    final_mode = artifacts.final.get("missing_mode")
    if final_mode:
        return str(final_mode)

    consistency_ranking = _dict_value(
        artifacts.ranking.get("kg_aware_candidate_consistency_ranking")
    )
    target_modality = consistency_ranking.get("target_modality")
    return str(target_modality) if target_modality else "unknown"


def _entity_type(artifacts: RunEntityArtifacts) -> str:
    entity_profile = _dict_value(artifacts.knowledge.get("entity_profile"))
    entity_type = entity_profile.get("entity_type")
    return str(entity_type) if entity_type else "entity"


def _candidate_id(candidate: dict[str, Any], pair_ids: bool) -> str | None:
    keys = ("pair_id", "id") if pair_ids else ("candidate_id", "id")
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return str(value)
    return None


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _round_score(value: float) -> float:
    return round(float(value), 6)
