"""Deterministic offline metrics for MMKGC evaluation."""

from __future__ import annotations

from typing import Any, Iterable

from mmkgc.utils import clamp01, token_cosine, token_overlap


def candidate_text(candidate: dict[str, Any]) -> str:
    """Return the textual surface available for a candidate, in ranking order."""
    content = candidate.get("generated_content")
    if isinstance(content, dict) and content.get("value"):
        return str(content["value"])

    value = candidate.get("prompt_or_instruction")
    if value:
        return str(value)

    return ""


def is_fallback_candidate(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate was produced from a fallback/global profile path."""
    metadata = candidate.get("metadata")
    coverage_detail = candidate.get("expected_relation_coverage_detail")
    return any(
        (
            candidate.get("primary_relation") == "global_profile",
            isinstance(metadata, dict) and metadata.get("fallback") is True,
            candidate.get("relation_role") == "fallback",
            isinstance(coverage_detail, dict)
            and coverage_detail.get("primary") == "global_profile",
        )
    )


def score_text_candidate(
    candidate: dict[str, Any],
    ground_truth_text: str | None,
    entity_label: str | None,
    entity_type: str | None,
) -> dict[str, Any]:
    """Score a text candidate without model calls or network access."""
    text = candidate_text(candidate)
    stage4_score = _candidate_stage4_score(candidate)
    relation_coverage_score = _metric_value(candidate, "relation_coverage_score")
    must_term_score = _metric_value(candidate, "must_term_score")
    ground_truth_available = bool(str(ground_truth_text or "").strip())
    identity_coverage_score = token_overlap(
        text,
        [term for term in (entity_label, entity_type) if term],
    )

    if ground_truth_available:
        ground_truth_similarity_score = token_cosine(text, str(ground_truth_text))
        offline_score = (
            (0.6 * ground_truth_similarity_score)
            + (0.2 * identity_coverage_score)
            + (0.2 * stage4_score)
        )
    else:
        ground_truth_similarity_score = 0.0
        offline_score = (
            (0.5 * stage4_score)
            + (0.3 * relation_coverage_score)
            + (0.2 * must_term_score)
        )

    offline_score = _score(offline_score)
    return {
        "offline_evaluation_score": offline_score,
        "stage4_ranking_score": _score(stage4_score),
        "ground_truth_similarity_score": _score(ground_truth_similarity_score),
        "identity_coverage_score": _score(identity_coverage_score),
        "ground_truth_available": ground_truth_available,
        "relation_coverage_score": _score(relation_coverage_score),
        "must_term_score": _score(must_term_score),
    }


def score_image_candidate(
    candidate: dict[str, Any],
    entity_label: str | None,
    entity_type: str | None,
) -> dict[str, Any]:
    """Score an image prompt candidate using only lexical coverage signals."""
    text = _image_prompt_text(candidate)
    stage4_score = _candidate_stage4_score(candidate)
    relation_coverage = token_overlap(text, _string_list(candidate.get("expected_relation_coverage")))
    constraint_coverage = token_overlap(text, _constraint_terms(candidate.get("used_constraints")))
    entity_type_coverage = token_overlap(
        text,
        [term for term in (entity_label, entity_type) if term],
    )
    forbidden_awareness_score = _forbidden_awareness(
        str(candidate.get("negative_prompt") or ""),
        _candidate_forbidden_terms(candidate),
    )
    offline_score = (
        (0.35 * relation_coverage)
        + (0.25 * constraint_coverage)
        + (0.20 * entity_type_coverage)
        + (0.20 * stage4_score)
    )

    return {
        "offline_evaluation_score": _score(offline_score),
        "stage4_ranking_score": _score(stage4_score),
        "entity_type_coverage": _score(entity_type_coverage),
        "relation_coverage": _score(relation_coverage),
        "constraint_coverage": _score(constraint_coverage),
        "forbidden_awareness_score": _score(forbidden_awareness_score),
    }


def score_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Score a text-image pair from existing Stage 4 and lexical pair signals."""
    text_candidate = pair.get("text_candidate")
    image_candidate = pair.get("image_candidate")
    stage4_text_score = (
        _candidate_stage4_score(text_candidate) if isinstance(text_candidate, dict) else 0.0
    )
    stage4_image_score = (
        _candidate_stage4_score(image_candidate) if isinstance(image_candidate, dict) else 0.0
    )
    stage4_pair_score = _candidate_stage4_score(pair)
    pair_balance_score = clamp01(1.0 - abs(stage4_text_score - stage4_image_score))
    pair_similarity_score = _metric_value(pair, "pair_similarity_score")
    kg_anchor_score = _metric_value(pair, "kg_anchor_score")
    offline_score = (
        (0.4 * stage4_pair_score)
        + (0.3 * pair_balance_score)
        + (0.2 * pair_similarity_score)
        + (0.1 * kg_anchor_score)
    )

    return {
        "offline_evaluation_score": _score(offline_score),
        "stage4_pair_score": _score(stage4_pair_score),
        "pair_balance_score": _score(pair_balance_score),
        "pair_similarity_score": _score(pair_similarity_score),
        "kg_anchor_score": _score(kg_anchor_score),
        "stage4_text_score": _score(stage4_text_score),
        "stage4_image_score": _score(stage4_image_score),
    }


def average_top(items: list[dict[str, Any]], key: str, limit: int = 5) -> float:
    """Average the highest values for a numeric key."""
    if not items or limit <= 0:
        return 0.0

    values = sorted((_metric_value(item, key) for item in items), reverse=True)[:limit]
    if not values:
        return 0.0
    return _score(sum(values) / len(values))


def _candidate_stage4_score(candidate: dict[str, Any]) -> float:
    if "score" in candidate:
        return _metric_value(candidate, "score")
    return _metric_value(candidate, "final_score")


def _image_prompt_text(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("positive_prompt", "prompt_or_instruction"):
        value = candidate.get(key)
        if value:
            parts.append(str(value))

    content = candidate.get("generated_content")
    if isinstance(content, dict) and content.get("value"):
        parts.append(str(content["value"]))

    return " ".join(parts)


def _metric_value(item: dict[str, Any], key: str) -> float:
    try:
        return clamp01(float(item.get(key, 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return [str(value)] if value else []
    return [str(item) for item in value if item]


def _constraint_terms(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        value = [value] if value else []

    terms = []
    for item in value:
        if isinstance(item, dict):
            term = item.get("constraint")
        else:
            term = item
        if term:
            terms.append(str(term))
    return terms


def _candidate_forbidden_terms(candidate: dict[str, Any]) -> list[str]:
    for key in ("applied_forbidden_content", "forbidden_constraints", "forbidden_terms"):
        terms = _constraint_terms(candidate.get(key))
        if terms:
            return terms
    return []


def _forbidden_awareness(negative_prompt: str, forbidden_constraints: Any) -> float:
    forbidden_terms = _constraint_terms(forbidden_constraints)
    if not forbidden_terms:
        return 1.0
    return token_overlap(negative_prompt, forbidden_terms)


def _score(value: float) -> float:
    return round(clamp01(float(value)), 6)
