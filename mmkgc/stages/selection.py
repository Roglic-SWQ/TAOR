"""Stage 2: fine-grained relation selection."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

STAGE2_EMBEDDING_SCORE_METHOD = "embedding_component_formula_v2"
STAGE2_LLM_SEMANTIC_ADJUSTMENT_VERSION = "llm_semantic_adjustment_v1"
_SCORE_COMPONENT_KEYS = (
    "modality_fit",
    "constraint_alignment",
    "evidence_strength",
    "generation_utility",
    "risk_penalty",
)
_LLM_ADJUSTABLE_COMPONENT_KEYS = (
    "modality_fit",
    "constraint_alignment",
    "generation_utility",
)
_LLM_APPLIED_STATUSES = {"applied", "generated", "success", "adjusted", "adjustment_needed"}

_MODALITY_FIT_ANCHOR_WEIGHTS = {
    "entity": 0.45,
    "relation": 0.55,
}
_GENERATION_UTILITY_ANCHOR_WEIGHTS = {
    "entity": 0.40,
    "relation": 0.60,
}

_IMAGE_CREDIT_RELATIONS = {
    "actor",
    "cast member",
    "composer",
    "creator",
    "director",
    "lyricist",
    "performer",
    "producer",
    "screenwriter",
}
_IMAGE_VISUAL_RELATIONS = {
    "architectural style",
    "color",
    "colour",
    "depicts",
    "has part",
    "located in",
    "location",
    "material used",
    "movement",
    "part of",
    "uses",
}

def _target_modalities(missing_mode: str) -> list[str]:
    if missing_mode in {"text", "image"}:
        return [missing_mode]
    raise ValueError("missing_mode must be one of: text, image")


def _source_fact_refs(fact: dict[str, Any]) -> list[str]:
    fact_id = fact.get("fact_id")
    return [str(fact_id)] if fact_id else []


def _supported_fact_refs(reasoning: dict[str, Any], fact: dict[str, Any]) -> list[str]:
    refs = set(_source_fact_refs(fact))
    relation = str(fact.get("relation") or "")
    matched: list[str] = []
    for supported in reasoning.get("supported_facts", []):
        if not isinstance(supported, dict):
            continue
        supported_id = supported.get("fact_id")
        source_refs = {str(ref) for ref in supported.get("source_fact_refs", [])}
        source_relations = {str(ref) for ref in supported.get("source_relations", [])}
        if refs & source_refs or (relation and relation in source_relations):
            if supported_id:
                matched.append(str(supported_id))
    return matched or list(refs)


def _constraint_buckets(reasoning: dict[str, Any], modality: str) -> dict[str, list[dict[str, Any]]]:
    constraints = reasoning.get("modality_constraints", {}).get(modality, {})
    return {
        "must_include": list(constraints.get("must_include", [])),
        "should_include": list(constraints.get("should_include", [])),
        "context_only": list(constraints.get("context_only", [])),
    }


def _constraints_for_relation(
    reasoning: dict[str, Any],
    modalities: list[str],
    relation: str,
) -> dict[str, list[dict[str, Any]]]:
    plan = {"must_include": [], "should_include": [], "context_only": []}
    for modality in modalities:
        buckets = _constraint_buckets(reasoning, modality)
        for bucket, items in buckets.items():
            for item in items:
                source_relations = item.get("source_relations") or []
                if relation in source_relations or not source_relations:
                    if not any(existing.get("constraint") == item.get("constraint") for existing in plan[bucket]):
                        plan[bucket].append(item)
    return plan


def _source_relations(item: dict[str, Any]) -> set[str]:
    values = item.get("source_relations") or []
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value).strip()}


def _relation_constraints(
    plan: dict[str, list[dict[str, Any]]],
    bucket: str,
    relation: str,
) -> list[dict[str, Any]]:
    return [
        item
        for item in plan.get(bucket, [])
        if relation in _source_relations(item)
    ]


def _risk_is_high(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) >= 4.0
    normalized = str(value or "").strip().lower().replace("-", "_")
    return normalized in {"high", "very_high", "severe", "unsafe", "forbidden"}


def _hard_exclusion_reasons(
    fact: dict[str, Any],
    generation_plan: dict[str, list[dict[str, Any]]],
    relation: str,
) -> list[str]:
    reasons = []
    if fact.get("allowed_use") == "excluded":
        reasons.append("credibility_policy_excluded")
    if not fact.get("evidence_triples"):
        reasons.append("no_evidence_triples")
    if _risk_is_high(fact.get("risk")):
        reasons.append("high_risk_relation_fact")
    return reasons


def _context_only_reasons(
    fact: dict[str, Any],
    generation_plan: dict[str, list[dict[str, Any]]],
    relation: str,
) -> list[str]:
    allowed_use = str(fact.get("allowed_use") or "").strip().lower()
    if allowed_use == "context_only":
        return ["credibility_policy_context_only"]
    generation_use = str(fact.get("generation_use") or "").strip().lower()
    if generation_use in {"context", "context_only", "use_as_context"}:
        return ["generation_use_context_only"]
    relation_context = _relation_constraints(generation_plan, "context_only", relation)
    relation_positive = _relation_constraints(generation_plan, "must_include", relation) + _relation_constraints(
        generation_plan,
        "should_include",
        relation,
    )
    if relation_context and not relation_positive:
        return ["relation_specific_context_only_constraint"]
    return []


def _score_fact(
    fact: dict[str, Any],
    target_modalities: list[str],
    generation_plan: dict[str, list[dict[str, Any]]],
    embedding_scorer: Any,
    entity_context: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    relation = str(fact.get("relation") or "")
    fact_text = _embedding_fact_text(fact)
    modality_fit = _embedding_modality_fit(
        fact,
        target_modalities,
        embedding_scorer,
        entity_context=entity_context,
    )
    constraint_alignment = _embedding_constraint_alignment(
        fact_text,
        generation_plan,
        embedding_scorer,
    )
    utility = _embedding_generation_utility(
        fact,
        target_modalities,
        embedding_scorer,
        entity_context=entity_context,
    )

    evidence_strength = _evidence_strength(fact)
    risk_penalty = _risk_penalty(fact, generation_plan, relation, target_modalities)
    fine_score = (
        0.30 * modality_fit
        + 0.25 * constraint_alignment
        + 0.20 * evidence_strength
        + 0.20 * utility
        - 0.25 * risk_penalty
    )
    return round(min(1.0, max(0.0, fine_score)), 6), {
        "modality_fit": round(modality_fit, 6),
        "constraint_alignment": round(constraint_alignment, 6),
        "evidence_strength": round(evidence_strength, 6),
        "generation_utility": round(utility, 6),
        "risk_penalty": round(risk_penalty, 6),
    }


def _score_from_components(components: dict[str, float]) -> float:
    fine_score = (
        0.30 * components["modality_fit"]
        + 0.25 * components["constraint_alignment"]
        + 0.20 * components["evidence_strength"]
        + 0.20 * components["generation_utility"]
        - 0.25 * components["risk_penalty"]
    )
    return round(min(1.0, max(0.0, fine_score)), 6)


def _bounded_adjustment(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return round(min(0.15, max(-0.15, numeric)), 6)


def _bounded_component(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return round(min(1.0, max(0.0, numeric)), 6)


def _apply_llm_assessment(
    assessor: Any,
    *,
    relation: str,
    fact: dict[str, Any],
    target_modalities: list[str],
    generation_plan: dict[str, list[dict[str, Any]]],
    baseline_fine_score: float,
    score_components: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, Any], dict[str, Any] | None]:
    """Apply bounded assessor deltas while retaining local evidence and formula ownership."""
    raw = assessor.assess(
        relation=relation,
        fact=str(fact.get("fact") or ""),
        target_modalities=list(target_modalities),
        generation_plan=deepcopy(generation_plan),
        score_components=dict(score_components),
    )
    raw = raw if isinstance(raw, dict) else {}
    raw_adjustments = raw.get("component_adjustments")
    assessment_status = str(raw.get("status") or "unavailable").strip().lower()
    adjustments_are_valid = (
        isinstance(raw_adjustments, dict)
        and _is_finite_number(raw.get("risk_penalty_adjustment"))
        and all(
            key in _LLM_ADJUSTABLE_COMPONENT_KEYS
            and _is_finite_number(value)
            for key, value in raw_adjustments.items()
        )
    )
    assessment_is_applicable = assessment_status in _LLM_APPLIED_STATUSES and adjustments_are_valid
    if assessment_is_applicable:
        adjustments = {
            key: _bounded_adjustment(raw_adjustments.get(key))
            for key in _LLM_ADJUSTABLE_COMPONENT_KEYS
            if key in raw_adjustments
        }
        risk_adjustment = _bounded_adjustment(raw.get("risk_penalty_adjustment"))
    else:
        adjustments = {}
        risk_adjustment = 0.0
    adjusted_components = {
        key: _bounded_component(
            score_components[key]
            + (risk_adjustment if key == "risk_penalty" else adjustments.get(key, 0.0))
        )
        for key in _SCORE_COMPONENT_KEYS
    }
    assessment: dict[str, Any] = {
        "status": assessment_status,
        "component_adjustments": adjustments,
        "risk_penalty_adjustment": risk_adjustment,
        "confidence": _bounded_component(raw.get("confidence")),
        "reason": str(raw.get("reason") or ""),
    }
    recommended_role = raw.get("recommended_role")
    if recommended_role is not None:
        assessment["recommended_role"] = str(recommended_role)
    model_call_ref = raw.get("model_call_ref")
    if model_call_ref is not None:
        assessment["model_call_ref"] = str(model_call_ref)
    model_call = raw.get("model_call")
    model_call = dict(model_call) if isinstance(model_call, dict) else None
    if not assessment_is_applicable:
        return baseline_fine_score, score_components, assessment, model_call
    return _score_from_components(adjusted_components), adjusted_components, assessment, model_call


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _embedding_fact_text(fact: dict[str, Any]) -> str:
    values = [
        str(fact.get("fact") or ""),
        str(fact.get("relation") or ""),
        *[str(value) for value in fact.get("neighbor_entities") or []],
    ]
    return " ".join(value for value in values if value.strip())


def _embedding_similarity(embedding_scorer: Any, left: str, right: str) -> float:
    score = float(embedding_scorer.similarity(left, right))
    if not math.isfinite(score):
        raise ValueError("embedding similarity must be finite")
    return min(1.0, max(0.0, score))


def _embedding_average(embedding_scorer: Any, left: str, anchors: list[str]) -> float:
    if not anchors:
        return 0.0
    return sum(_embedding_similarity(embedding_scorer, left, anchor) for anchor in anchors) / len(anchors)


def _ordered_unique_strings(values: Any) -> list[str]:
    if isinstance(values, (str, int, float)):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text.lower() in {"unknown", "none", "null"}:
            continue
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _stage2_entity_anchor_context(knowledge: dict[str, Any]) -> dict[str, Any]:
    """Collect Stage 1 entity/profile signals used by dynamic Stage 2 anchors."""
    stage1_v3 = knowledge.get("stage1_v3")
    stage1_v3 = stage1_v3 if isinstance(stage1_v3, dict) else {}
    v3_profile = stage1_v3.get("entity_profile")
    v3_profile = v3_profile if isinstance(v3_profile, dict) else {}
    legacy_profile = knowledge.get("entity_profile")
    legacy_profile = legacy_profile if isinstance(legacy_profile, dict) else {}

    profile = {
        "primary_type": v3_profile.get("primary_type")
        or legacy_profile.get("entity_type")
        or legacy_profile.get("type"),
        "secondary_types": v3_profile.get("secondary_types")
        or legacy_profile.get("secondary_types", []),
        "semantic_domains": v3_profile.get("semantic_domains")
        or legacy_profile.get("semantic_domain", []),
        "core_identity": v3_profile.get("core_identity")
        or legacy_profile.get("core_identity"),
    }
    visual_profile = stage1_v3.get("text_visual_profile")
    if not isinstance(visual_profile, dict):
        visual_profile = knowledge.get("text_visual_profile")
    visual_profile = visual_profile if isinstance(visual_profile, dict) else {}
    return {
        **profile,
        "visual_scene_cues": visual_profile.get("visual_scene_cues", []),
        "context_cues": visual_profile.get("context_cues", []),
    }


def _entity_modality_fit_anchors(
    entity_context: dict[str, Any] | None,
    modality: str,
) -> list[str]:
    if not entity_context:
        return []
    primary_type = str(entity_context.get("primary_type") or "").strip()
    secondary_types = _ordered_unique_strings(entity_context.get("secondary_types"))
    domains = _ordered_unique_strings(entity_context.get("semantic_domains"))
    identity = str(entity_context.get("core_identity") or "").strip()
    scene_cues = _ordered_unique_strings(entity_context.get("visual_scene_cues"))
    context_cues = _ordered_unique_strings(entity_context.get("context_cues"))
    types = _ordered_unique_strings([primary_type, *secondary_types])
    anchors: list[str] = []

    if modality == "image":
        anchors.extend(f"visible representation of a {value}" for value in types)
        if identity:
            anchors.append(f"visual appearance or scene representing {identity}")
        anchors.extend(f"{value} visual scene, subject, or environment" for value in domains)
        anchors.extend(f"visual scene cue: {value}" for value in scene_cues)
        anchors.extend(f"visual context: {value}" for value in context_cues)
    elif modality == "text":
        anchors.extend(f"grounded factual description of a {value}" for value in types)
        if identity:
            anchors.append(f"factual explanation of {identity}")
        anchors.extend(f"{value} background and explanatory information" for value in domains)
        anchors.extend(f"background information about {value}" for value in context_cues)
    return _ordered_unique_strings(anchors)


def _relation_modality_fit_anchors(fact: dict[str, Any], modality: str) -> list[str]:
    relation = str(fact.get("relation") or "").strip()
    if not relation:
        return []
    objects = _ordered_unique_strings(
        [fact.get("object"), *_ordered_unique_strings(fact.get("neighbor_entities"))]
    )
    relation_fact = relation
    if objects:
        relation_fact = f"{relation}: {', '.join(objects[:4])}"
    if modality == "image":
        return [
            f"visible content grounded in {relation_fact}",
            f"visual representation of {relation_fact}",
        ]
    if modality == "text":
        return [
            f"grounded factual statement about {relation_fact}",
            f"factual description of {relation_fact}",
        ]
    return []


def _embedding_modality_fit_anchor_groups(
    fact: dict[str, Any],
    target_modalities: list[str],
    entity_context: dict[str, Any] | None,
) -> dict[str, list[str]]:
    entity: list[str] = []
    relation: list[str] = []
    for modality in target_modalities:
        entity.extend(_entity_modality_fit_anchors(entity_context, modality))
        relation.extend(_relation_modality_fit_anchors(fact, modality))
    return {
        "entity": _ordered_unique_strings(entity),
        "relation": _ordered_unique_strings(relation),
    }


def _embedding_modality_fit(
    fact: dict[str, Any],
    target_modalities: list[str],
    embedding_scorer: Any,
    entity_context: dict[str, Any] | None = None,
) -> float:
    fact_text = _embedding_fact_text(fact)
    groups = _embedding_modality_fit_anchor_groups(fact, target_modalities, entity_context)
    available_groups = {
        name: anchors
        for name, anchors in groups.items()
        if anchors
    }
    if not available_groups:
        return 0.0
    total_weight = sum(_MODALITY_FIT_ANCHOR_WEIGHTS[name] for name in available_groups)
    return sum(
        _MODALITY_FIT_ANCHOR_WEIGHTS[name]
        * _embedding_average(embedding_scorer, fact_text, anchors)
        for name, anchors in available_groups.items()
    ) / total_weight


def _embedding_constraint_alignment(
    fact_text: str,
    generation_plan: dict[str, list[dict[str, Any]]],
    embedding_scorer: Any,
) -> float:
    constraints = [
        str(item.get("constraint") or "")
        for bucket in ("must_include", "should_include", "context_only")
        for item in generation_plan.get(bucket, [])
        if isinstance(item, dict) and str(item.get("constraint") or "").strip()
    ]
    if not constraints:
        return 0.2
    return _embedding_average(embedding_scorer, fact_text, constraints)


def _entity_generation_utility_anchors(
    entity_context: dict[str, Any] | None,
    modality: str,
) -> list[str]:
    if not entity_context:
        return []
    primary_type = str(entity_context.get("primary_type") or "").strip()
    secondary_types = _ordered_unique_strings(entity_context.get("secondary_types"))
    domains = _ordered_unique_strings(entity_context.get("semantic_domains"))
    identity = str(entity_context.get("core_identity") or "").strip()
    scene_cues = _ordered_unique_strings(entity_context.get("visual_scene_cues"))
    context_cues = _ordered_unique_strings(entity_context.get("context_cues"))
    types = _ordered_unique_strings([primary_type, *secondary_types])
    anchors: list[str] = []

    if modality == "image":
        anchors.extend(f"visual generation of a {value}" for value in types)
        if identity:
            anchors.append(f"specific visual subject or scene representing {identity}")
        anchors.extend(f"{value} visual content suitable for generation" for value in domains)
        anchors.extend(f"visual generation detail: {value}" for value in scene_cues)
        anchors.extend(f"visual context for generation: {value}" for value in context_cues)
    elif modality == "text":
        anchors.extend(f"factual generation about a {value}" for value in types)
        if identity:
            anchors.append(f"specific factual subject describing {identity}")
        anchors.extend(f"{value} information suitable for factual generation" for value in domains)
        anchors.extend(f"text generation detail: {value}" for value in context_cues)
    return _ordered_unique_strings(anchors)


def _relation_generation_utility_anchors(
    fact: dict[str, Any],
    modality: str,
) -> list[str]:
    relation = _normalize_relation(str(fact.get("relation") or ""))
    if not relation:
        return []
    objects = _ordered_unique_strings(
        [fact.get("object"), *_ordered_unique_strings(fact.get("neighbor_entities"))]
    )
    relation_fact = relation
    if objects:
        relation_fact = f"{relation}: {', '.join(objects[:4])}"

    if modality == "image":
        anchors = [
            f"generation-ready visual content grounded in {relation_fact}",
            f"specific visual subject, attribute, or scene for {relation_fact}",
        ]
        if relation in _IMAGE_CREDIT_RELATIONS:
            anchors.append(f"visual context associated with {relation_fact}")
        elif relation in _IMAGE_VISUAL_RELATIONS:
            anchors.append(f"visible attribute or scene detail described by {relation_fact}")
        else:
            anchors.append(f"supporting visual context for {relation_fact}")
        return anchors
    if modality == "text":
        anchors = [
            f"generation-ready factual statement about {relation_fact}",
            f"specific descriptive detail suitable for explaining {relation_fact}",
        ]
        if relation in _IMAGE_CREDIT_RELATIONS:
            anchors.append(f"factual credit or affiliation information about {relation_fact}")
        elif relation in _IMAGE_VISUAL_RELATIONS:
            anchors.append(f"factual descriptive attribute expressed by {relation_fact}")
        else:
            anchors.append(f"grounded explanatory context for {relation_fact}")
        return anchors
    return []


def _embedding_generation_utility_anchor_groups(
    fact: dict[str, Any],
    target_modalities: list[str],
    entity_context: dict[str, Any] | None,
) -> dict[str, list[str]]:
    entity: list[str] = []
    relation: list[str] = []
    for modality in target_modalities:
        entity.extend(_entity_generation_utility_anchors(entity_context, modality))
        relation.extend(_relation_generation_utility_anchors(fact, modality))
    return {
        "entity": _ordered_unique_strings(entity),
        "relation": _ordered_unique_strings(relation),
    }


def _embedding_generation_utility(
    fact: dict[str, Any],
    target_modalities: list[str],
    embedding_scorer: Any,
    entity_context: dict[str, Any] | None = None,
) -> float:
    fact_text = _embedding_fact_text(fact)
    groups = _embedding_generation_utility_anchor_groups(
        fact,
        target_modalities,
        entity_context,
    )
    available_groups = {
        name: anchors
        for name, anchors in groups.items()
        if anchors
    }
    if not available_groups:
        return 0.0
    total_weight = sum(
        _GENERATION_UTILITY_ANCHOR_WEIGHTS[name]
        for name in available_groups
    )
    return sum(
        _GENERATION_UTILITY_ANCHOR_WEIGHTS[name]
        * _embedding_average(embedding_scorer, fact_text, anchors)
        for name, anchors in available_groups.items()
    ) / total_weight


def _evidence_strength(fact: dict[str, Any]) -> float:
    credibility = fact.get("credibility_score")
    if isinstance(credibility, (int, float)) and not isinstance(credibility, bool):
        return min(1.0, max(0.0, float(credibility)))
    evidence_items = (
        list(fact.get("evidence_triples") or [])
        + list(fact.get("evidence_refs") or [])
        + list(fact.get("stage0_fact_ids") or [])
    )
    evidence_strength = min(1.0, len(evidence_items) / 3)
    support = str(fact.get("support_level") or "").strip().lower()
    if support == "strong":
        evidence_strength = max(evidence_strength, 0.9)
    elif support == "moderate":
        evidence_strength = max(evidence_strength, 0.65)
    elif support == "weak":
        evidence_strength = max(evidence_strength, 0.35)
    if fact.get("fact_id"):
        evidence_strength = min(1.0, evidence_strength + 0.05)
    return min(1.0, max(0.0, evidence_strength))


def _risk_penalty(
    fact: dict[str, Any],
    generation_plan: dict[str, list[dict[str, Any]]],
    relation: str,
    target_modalities: list[str],
) -> float:
    risk = str(fact.get("risk") or "").strip().lower().replace("-", "_")
    penalty = 0.0
    if risk == "medium":
        penalty += 0.25
    elif risk in {"high", "very_high", "severe", "unsafe", "forbidden"}:
        penalty += 0.55
    elif risk and risk != "low":
        penalty += 0.35

    if "image" in target_modalities and _normalize_relation(relation) in _IMAGE_CREDIT_RELATIONS:
        penalty += 0.1
    return min(1.0, max(0.0, penalty))


def _normalize_relation(relation: str) -> str:
    return " ".join(str(relation or "").strip().lower().replace("_", " ").split())


def _role(index: int, item: dict[str, Any]) -> str:
    if item["fine_score"] < 0.35:
        return "excluded"
    if index < 3:
        return "primary"
    if item["fine_score"] >= 0.5:
        return "secondary"
    return "context"


def _assign_quota_values(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    candidate_budget: int,
) -> dict[str, Any]:
    remaining = max(0, candidate_budget)
    primary_quotas: dict[str, int] = {}
    secondary_quotas: dict[str, int] = {}
    for item in primary:
        if remaining <= 0:
            break
        item["_quota_value"] = int(item.get("_quota_value", 0)) + 1
        primary_quotas[item["relation"]] = int(item["_quota_value"])
        remaining -= 1

    secondary_by_score = sorted(secondary, key=lambda item: (-item["fine_score"], item["relation"].lower()))
    for item in secondary_by_score[:1]:
        if remaining <= 0:
            break
        item["_quota_value"] = int(item.get("_quota_value", 0)) + 1
        secondary_quotas[item["relation"]] = int(item["_quota_value"])
        remaining -= 1

    if primary:
        primary_by_score = sorted(primary, key=lambda item: (-item["fine_score"], item["relation"].lower()))
        index = 0
        while remaining > 0:
            item = primary_by_score[index % len(primary_by_score)]
            item["_quota_value"] = int(item.get("_quota_value", 0)) + 1
            primary_quotas[item["relation"]] = int(item["_quota_value"])
            remaining -= 1
            index += 1
    elif secondary:
        index = 0
        while remaining > 0:
            item = secondary_by_score[index % len(secondary_by_score)]
            item["_quota_value"] = int(item.get("_quota_value", 0)) + 1
            secondary_quotas[item["relation"]] = int(item["_quota_value"])
            remaining -= 1
            index += 1

    for item in primary:
        quota = int(item.get("_quota_value", 0))
        if quota > 0:
            primary_quotas[item["relation"]] = quota
    for item in secondary:
        quota = int(item.get("_quota_value", 0))
        if quota > 0:
            secondary_quotas[item["relation"]] = quota

    assigned_total = sum(int(item.get("_quota_value", 0)) for item in primary + secondary)
    return {
        "candidate_budget": candidate_budget,
        "primary_quota": primary_quotas,
        "secondary_quota": secondary_quotas,
        "relation_quotas": {**primary_quotas, **secondary_quotas},
        "assigned_total": assigned_total,
        "fallback_quota": max(0, candidate_budget - assigned_total),
        "secondary_fill_order": [item["relation"] for item in secondary],
        "global_profile_fallback": assigned_total < candidate_budget,
    }


def _assign_candidate_quotas(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    candidate_budget: int,
    target_modalities: list[str],
) -> dict[str, Any]:
    for item in primary + secondary:
        item["candidate_quota"] = 0
        item["candidate_quota_by_modality"] = {}

    modality_quota: dict[str, dict[str, Any]] = {}
    combined_primary: dict[str, int] = {}
    combined_secondary: dict[str, int] = {}
    assigned_total = 0
    fallback_total = 0

    modalities = list(target_modalities) or ["text"]
    for modality in modalities:
        modality_primary = [
            item
            for item in primary
            if modality in item.get("target_modality", [])
        ]
        modality_secondary = [
            item
            for item in secondary
            if modality in item.get("target_modality", [])
        ]
        quota_items = [{"_source": item, "_quota_value": 0, **item} for item in modality_primary + modality_secondary]
        quota_primary = quota_items[: len(modality_primary)]
        quota_secondary = quota_items[len(modality_primary):]
        summary = _assign_quota_values(quota_primary, quota_secondary, candidate_budget)
        modality_quota[modality] = summary
        assigned_total += int(summary["assigned_total"])
        fallback_total += int(summary["fallback_quota"])

        for quota_item in quota_primary + quota_secondary:
            quota = int(quota_item.get("_quota_value", 0))
            if quota <= 0:
                continue
            source = quota_item["_source"]
            by_modality = source["candidate_quota_by_modality"]
            by_modality[modality] = quota
            source["candidate_quota"] += quota
            target = combined_primary if source in modality_primary else combined_secondary
            target[source["relation"]] = target.get(source["relation"], 0) + quota

    return {
        "candidate_budget": candidate_budget,
        "quota_mode": "per_modality" if len(modalities) > 1 else "single_modality",
        "target_modalities": list(modalities),
        "primary_quota": combined_primary,
        "secondary_quota": combined_secondary,
        "relation_quotas": {**combined_primary, **combined_secondary},
        "assigned_total": assigned_total,
        "fallback_quota": fallback_total,
        "modality_quota": modality_quota,
        "global_profile_fallback": fallback_total > 0,
    }


def fine_grained_relation_selection(
    missing_mode: str,
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    candidate_budget: int = 5,
    *,
    embedding_scorer: Any,
    llm_assessor: Any | None = None,
) -> dict[str, Any]:
    """Score Stage 1 facts and assign Stage 2 generation roles."""
    target_modalities = _target_modalities(missing_mode)
    entity_anchor_context = _stage2_entity_anchor_context(knowledge)
    scored = []
    model_calls: list[dict[str, Any]] = []
    credibility_by_fact_id = {
        str(item.get("fact_id")): item
        for item in reasoning.get("fact_credibility_assessments", [])
        if isinstance(item, dict) and item.get("fact_id")
    }
    for source_fact in knowledge.get("relation_facts", []):
        if not isinstance(source_fact, dict):
            continue
        fact = dict(source_fact)
        assessment = credibility_by_fact_id.get(str(fact.get("fact_id") or ""))
        if assessment:
            for key in (
                "credibility_score",
                "credibility_label",
                "allowed_use",
                "risk",
                "reason",
            ):
                if key in assessment:
                    target_key = "credibility_reason" if key == "reason" else key
                    fact[target_key] = assessment[key]
        relation = str(fact.get("relation") or "related to")
        relation_modalities = list(target_modalities)
        generation_plan = _constraints_for_relation(reasoning, relation_modalities, relation)
        fine_score, breakdown = _score_fact(
            fact,
            relation_modalities,
            generation_plan,
            embedding_scorer=embedding_scorer,
            entity_context=entity_anchor_context,
        )
        if llm_assessor is None:
            llm_assessment = None
            model_call = None
        else:
            fine_score, breakdown, llm_assessment, model_call = _apply_llm_assessment(
                llm_assessor,
                relation=relation,
                fact=fact,
                target_modalities=relation_modalities,
                generation_plan=generation_plan,
                baseline_fine_score=fine_score,
                score_components=breakdown,
            )
        if model_call is not None:
            model_calls.append(model_call)
        hard_exclusion_reasons = _hard_exclusion_reasons(fact, generation_plan, relation)
        context_only_reasons = (
            []
            if hard_exclusion_reasons
            else _context_only_reasons(fact, generation_plan, relation)
        )
        item = {
                "relation": relation,
                "relation_role": "unassigned",
                "fallback": False,
                "score_method": (
                    f"{STAGE2_EMBEDDING_SCORE_METHOD}+"
                    f"{STAGE2_LLM_SEMANTIC_ADJUSTMENT_VERSION}"
                ),
                "fine_score": fine_score,
                "score_breakdown": breakdown,
                "target_modality": relation_modalities,
                "role_decision": "",
                "relation_constraint_refs_by_bucket": _constraint_refs_by_bucket(
                    generation_plan,
                    relation,
                )[1],
                "global_constraint_refs_by_bucket": _constraint_refs_by_bucket(
                    generation_plan,
                    relation,
                )[0],
                "candidate_quota": 0,
                "candidate_quota_by_modality": {},
                "evidence_triples": list(fact.get("evidence_triples", [])),
                "source_fact": fact,
                "source_fact_refs": _source_fact_refs(fact),
                "supported_fact_refs": _supported_fact_refs(reasoning, fact),
                "stage2_hard_exclusion_reasons": hard_exclusion_reasons,
                "stage2_context_only_reasons": context_only_reasons,
        }
        if llm_assessment is not None:
            item["llm_assessment"] = llm_assessment
            if llm_assessment.get("model_call_ref"):
                item["model_call_ref"] = llm_assessment["model_call_ref"]
        scored.append(item)

    scored.sort(key=lambda item: (-item["fine_score"], item["relation"].lower()))
    buckets = {"primary": [], "secondary": [], "context": [], "excluded": []}
    eligible_index = 0
    for item in scored:
        if item["stage2_hard_exclusion_reasons"]:
            role = "excluded"
            item["stage2_hard_excluded"] = True
            item["role_decision"] = (
                "Excluded before score ranking: "
                + ", ".join(item["stage2_hard_exclusion_reasons"])
                + f" (fine_score={item['fine_score']:.3f})."
            )
        elif item["stage2_context_only_reasons"]:
            role = "context"
            item["stage2_context_only"] = True
            item["role_decision"] = (
                "Kept as context before score ranking: "
                + ", ".join(item["stage2_context_only_reasons"])
                + f" (fine_score={item['fine_score']:.3f})."
            )
        elif item["source_fact"].get("allowed_use") == "supporting_only":
            role = "secondary" if item["fine_score"] >= 0.5 else "context"
            item["stage2_supporting_only"] = True
            item["role_decision"] = (
                f"Credibility policy capped this relation at {role} "
                f"(fine_score={item['fine_score']:.3f})."
            )
        else:
            role = _role(eligible_index, item)
            item["role_decision"] = f"Selected as {role} with fine_score={item['fine_score']:.3f}."
            eligible_index += 1
        item["relation_role"] = role
        buckets[role].append(item)
    plan_index = 1
    for role in ("primary", "secondary", "context", "excluded"):
        for item in buckets[role]:
            item["plan_id"] = f"rp:{plan_index:04d}"
            plan_index += 1

    result = {
        "fine_grained_relation_selection": {
            "target_modality": list(target_modalities),
            "selection_summary": (
                f"Selected {len(buckets['primary'])} primary, {len(buckets['secondary'])} secondary, "
                f"and {len(buckets['context'])} context relations from {len(scored)} facts."
            ),
            "primary_relations": buckets["primary"],
            "secondary_relations": buckets["secondary"],
            "context_relations": buckets["context"],
            "excluded_relations": buckets["excluded"],
            "quota_summary": _assign_candidate_quotas(
                buckets["primary"],
                buckets["secondary"],
                candidate_budget,
                target_modalities,
            ),
        },
    }
    result["model_calls"] = model_calls
    return result


def _constraint_refs_by_bucket(
    generation_plan: dict[str, list[dict[str, Any]]],
    relation: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return global and relation-specific constraint references for Stage2 output."""
    buckets = ("must_include", "should_include", "context_only")
    global_refs = {bucket: [] for bucket in buckets}
    relation_refs = {bucket: [] for bucket in buckets}
    for bucket in buckets:
        for index, item in enumerate(generation_plan.get(bucket, []), start=1):
            if not isinstance(item, dict):
                continue
            ref = item.get("constraint_id") or item.get("constraint_ref") or item.get("id")
            ref = str(ref) if ref else f"constraint:{bucket}:{index:04d}"
            source_relations = item.get("source_relations")
            target = (
                relation_refs
                if isinstance(source_relations, list)
                and relation in {str(value) for value in source_relations}
                else global_refs
            )
            if ref not in target[bucket]:
                target[bucket].append(ref)
    return global_refs, relation_refs
