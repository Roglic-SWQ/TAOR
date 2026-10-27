"""Calibration helpers for deterministic offline MMKGC evaluation scores."""

from __future__ import annotations

from typing import Any

from mmkgc.evaluation.metrics import is_fallback_candidate
from mmkgc.utils import clamp01


DIRECT_IDENTITY_RELATIONS = {"genre", "performer", "occupation"}
TYPE_SUPPORT_RELATIONS = {"instance of"}
NEIGHBOR_INDIRECT_RELATIONS = {
    "cast member",
    "original broadcaster",
    "member of sports team",
}
RELATION_SUPPORT_FACTORS = {
    "direct_identity": 1.0,
    "type_support": 0.95,
    "neighbor_indirect": 0.90,
    "fallback": 0.75,
    "unknown": 0.95,
}
RELATION_WEAKNESS_PRIORITY = {
    "fallback": 0,
    "neighbor_indirect": 1,
    "unknown": 2,
    "type_support": 3,
    "direct_identity": 4,
}


def relation_support_strength(candidate: dict[str, Any]) -> str:
    """Classify how directly a candidate relation supports entity identity."""
    if candidate.get("is_fallback") is True or is_fallback_candidate(candidate):
        return "fallback"

    relation = str(candidate.get("primary_relation") or "").strip().lower()
    if relation in DIRECT_IDENTITY_RELATIONS:
        return "direct_identity"
    if relation in TYPE_SUPPORT_RELATIONS:
        return "type_support"
    if relation in NEIGHBOR_INDIRECT_RELATIONS:
        return "neighbor_indirect"
    return "unknown"


def calibrate_entity_scores(
    *,
    entity: dict[str, Any],
    text_evaluation: dict[str, Any],
    image_evaluation: dict[str, Any],
    pair_evaluation: dict[str, Any],
    fallback_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return calibrated best scores and diagnostics for one evaluated entity."""
    text = _score(text_evaluation.get("best_offline_evaluation_score", 0.0))
    image = _score(image_evaluation.get("best_offline_evaluation_score", 0.0))
    pair = _score(pair_evaluation.get("best_offline_evaluation_score", 0.0))
    tags: set[str] = set()
    penalties = {
        "fallback_penalty": 0.0,
        "entity_type_generic_penalty": 0.0,
        "template_prompt_penalty": 0.0,
        "single_side_pair_penalty": 0.0,
        "relation_support_penalty": 0.0,
    }

    best_text = _first_dict(text_evaluation.get("ranked_candidates"))
    best_image = _first_dict(image_evaluation.get("ranked_candidates"))
    best_pair = _first_dict(pair_evaluation.get("ranked_pairs"))
    text_strength = relation_support_strength(best_text)
    image_strength = relation_support_strength(best_image)
    pair_strength = _pair_relation_support_strength(best_pair, text_strength, image_strength)

    has_fallback = bool(fallback_summary.get("has_fallback"))
    if has_fallback:
        previous = (text, image, pair)
        text = _score(text * 0.75)
        image = _score(image * 0.75)
        pair = _score(pair * 0.70)
        penalties["fallback_penalty"] += _loss(previous, (text, image, pair))
        tags.add("fallback_penalized")

    entity_type = str(entity.get("entity_type") or "")
    is_generic_entity = entity_type.lower() == "entity"
    if is_generic_entity:
        previous = (text, image)
        text = _score(text * 0.85)
        image = _score(image * 0.90)
        penalties["entity_type_generic_penalty"] += _loss(previous, (text, image))
        tags.add("generic_entity_type")

    if _has_generic_entity_text(best_text):
        previous = text
        text = _score(text * 0.80)
        penalties["entity_type_generic_penalty"] += previous - text
        tags.add("generic_entity_text")

    weakest_relation_strength = _weakest_strength(text_strength, image_strength)
    relation_factor = RELATION_SUPPORT_FACTORS[weakest_relation_strength]
    if relation_factor < 1.0:
        previous = (text, image)
        text = _score(text * relation_factor)
        image = _score(image * relation_factor)
        penalties["relation_support_penalty"] += _loss(previous, (text, image))
        tags.add(f"relation_support_{weakest_relation_strength}")

    template_prompt_factor = _image_template_prompt_factor(
        best_image,
        entity_type=entity_type,
        image_strength=image_strength,
    )
    if template_prompt_factor < 1.0:
        previous = image
        image = _score(image * template_prompt_factor)
        penalties["template_prompt_penalty"] += previous - image
        tags.add("template_prompt_penalized")

    previous_pair = pair
    if text < 0.40:
        pair = min(pair, 0.65)
    elif text < 0.50:
        pair = min(pair, 0.75)
    if image < 0.60:
        pair = min(pair, 0.80)
    pair = _score(pair)
    if pair < previous_pair:
        penalties["single_side_pair_penalty"] += previous_pair - pair
        tags.add("pair_single_side_penalized")

    available_scores = [
        score
        for evaluation, score in (
            (text_evaluation, text),
            (image_evaluation, image),
            (pair_evaluation, pair),
        )
        if _has_candidates(evaluation)
    ]
    overall = _score(
        sum(available_scores) / len(available_scores)
        if available_scores
        else 0.0
    )
    if has_fallback:
        overall = _score(overall * 0.90)

    return {
        "calibrated_text_best_offline_evaluation_score": text,
        "calibrated_image_best_offline_evaluation_score": image,
        "calibrated_pair_best_offline_evaluation_score": pair,
        "calibrated_overall_offline_evaluation_score": overall,
        "calibration_summary": {
            "fallback_penalty": _score(penalties["fallback_penalty"]),
            "entity_type_generic_penalty": _score(penalties["entity_type_generic_penalty"]),
            "template_prompt_penalty": _score(penalties["template_prompt_penalty"]),
            "single_side_pair_penalty": _score(penalties["single_side_pair_penalty"]),
            "relation_support_penalty": _score(penalties["relation_support_penalty"]),
            "relation_support_strength": {
                "text": text_strength,
                "image": image_strength,
                "pair": pair_strength,
            },
            "calibration_tags": sorted(tags),
        },
    }


def _pair_relation_support_strength(
    pair: dict[str, Any],
    text_strength: str,
    image_strength: str,
) -> str:
    text_candidate = pair.get("text_candidate")
    image_candidate = pair.get("image_candidate")
    nested_text = relation_support_strength(text_candidate) if isinstance(text_candidate, dict) else "unknown"
    nested_image = relation_support_strength(image_candidate) if isinstance(image_candidate, dict) else "unknown"
    if nested_text == "unknown" and nested_image == "unknown":
        return _weakest_strength(text_strength, image_strength)
    return _weakest_strength(nested_text, nested_image)


def _weakest_strength(left: str, right: str) -> str:
    if RELATION_WEAKNESS_PRIORITY[left] <= RELATION_WEAKNESS_PRIORITY[right]:
        return left
    return right


def _image_template_prompt_factor(
    candidate: dict[str, Any],
    *,
    entity_type: str,
    image_strength: str,
) -> float:
    prompt = _image_prompt_text(candidate)
    prompt_lower = prompt.lower()
    fallback_prompt = image_strength == "fallback"
    generic_entity_prompt = entity_type.lower() == "entity" and any(
        marker in prompt_lower for marker in ("identity", "profile context", "entity")
    )
    template_markers = ("identity", "profile context", "grounded details")
    marker_count = sum(1 for marker in template_markers if marker in prompt_lower)
    non_identity_template_prompt = image_strength != "direct_identity" and marker_count >= 2
    if fallback_prompt or generic_entity_prompt or non_identity_template_prompt:
        return 0.85
    return 1.0


def _has_generic_entity_text(candidate: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for value in (
            _generated_content_value(candidate),
            candidate.get("prompt_or_instruction"),
        )
        if value
    ).lower()
    return "as a entity" in text or "entity type entity" in text


def _generated_content_value(candidate: dict[str, Any]) -> str:
    content = candidate.get("generated_content")
    if isinstance(content, dict) and content.get("value"):
        return str(content["value"])
    return ""


def _image_prompt_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key))
        for key in ("positive_prompt", "prompt_or_instruction")
        if candidate.get(key)
    )


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _has_candidates(evaluation: dict[str, Any]) -> bool:
    try:
        return int(evaluation.get("candidate_count", 0)) > 0
    except (TypeError, ValueError):
        return False


def _loss(before: tuple[float, ...], after: tuple[float, ...]) -> float:
    return sum(before_value - after_value for before_value, after_value in zip(before, after))


def _score(value: Any) -> float:
    try:
        return round(clamp01(float(value)), 6)
    except (TypeError, ValueError):
        return 0.0
