"""Fact credibility scoring and deterministic generation-use policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CREDIBILITY_SCHEMA_VERSION = "fact_credibility_v1"

DIRECT_GENERATION_THRESHOLD = 0.80
SUPPORTING_ONLY_THRESHOLD = 0.60
CONTEXT_ONLY_THRESHOLD = 0.40
QWEN_SCORE_DELTA_MIN = -0.20
QWEN_SCORE_DELTA_MAX = 0.20

ALLOWED_USES = {
    "direct_generation",
    "supporting_only",
    "context_only",
    "excluded",
}

HIGH_RISK_LABELS = {"high", "very_high", "severe", "unsafe", "forbidden"}
CONTEXT_GENERATION_USES = {"context", "context_only", "use_as_context"}
FORBIDDEN_GENERATION_USES = {"forbidden", "exclude", "excluded", "do_not_use"}


def clamp_credibility_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return round(min(1.0, max(0.0, score)), 6)


def clamp_score_delta(value: Any) -> float:
    """Limit the semantic planner correction applied to the evidence prior."""
    try:
        delta = float(value)
    except (TypeError, ValueError):
        delta = 0.0
    return round(min(QWEN_SCORE_DELTA_MAX, max(QWEN_SCORE_DELTA_MIN, delta)), 6)


def credibility_label(score: float) -> str:
    if score >= DIRECT_GENERATION_THRESHOLD:
        return "high"
    if score >= SUPPORTING_ONLY_THRESHOLD:
        return "medium"
    if score >= CONTEXT_ONLY_THRESHOLD:
        return "low"
    return "very_low"


def evidence_credibility(
    *,
    evidence_triples: list[Any],
    evidence_refs: list[Any],
    risk: Any,
) -> tuple[float, dict[str, Any]]:
    """Score evidence reliability without using relation relevance or utility."""
    triples = _unique_non_empty(evidence_triples)
    refs = _unique_non_empty(evidence_refs)
    if triples:
        score = 0.76 + 0.04 * min(3, len(triples))
        if refs:
            score += 0.04
        evidence_type = "direct_kg"
    elif refs:
        score = 0.62
        evidence_type = "referenced_kg"
    else:
        score = 0.20
        evidence_type = "unsupported"

    normalized_risk = _normalized_label(risk)
    if normalized_risk == "medium":
        score -= 0.10
    elif normalized_risk in HIGH_RISK_LABELS:
        score -= 0.35

    return clamp_credibility_score(score), {
        "evidence_type": evidence_type,
        "direct_evidence_count": len(triples),
        "evidence_ref_count": len(refs),
    }


def allowed_use_for_fact(fact: Mapping[str, Any], score: float) -> str:
    risk = _normalized_label(fact.get("risk"))
    generation_use = _normalized_label(fact.get("generation_use"))
    policy_generated_use = fact.get("generation_use_policy_source") == "credibility_policy"
    if risk in HIGH_RISK_LABELS or (
        not policy_generated_use and generation_use in FORBIDDEN_GENERATION_USES
    ):
        return "excluded"
    if not policy_generated_use and generation_use in CONTEXT_GENERATION_USES:
        return "context_only"

    has_direct_evidence = bool(
        _unique_non_empty(fact.get("evidence_triples", []))
        or _unique_non_empty(fact.get("stage0_fact_ids", []))
        or _unique_non_empty(fact.get("evidence_refs", []))
    )
    if score >= DIRECT_GENERATION_THRESHOLD and has_direct_evidence:
        return "direct_generation"
    if score >= SUPPORTING_ONLY_THRESHOLD:
        return "supporting_only"
    if score >= CONTEXT_ONLY_THRESHOLD:
        return "context_only"
    return "excluded"


def ensure_fact_credibility(fact: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fact with normalized credibility fields and deterministic use."""
    result = dict(fact)
    evidence_refs = _list_value(result.get("stage0_fact_ids")) + _list_value(
        result.get("evidence_refs")
    )
    if result.get("evidence_prior_score") is None:
        score, basis = evidence_credibility(
            evidence_triples=_list_value(result.get("evidence_triples")),
            evidence_refs=evidence_refs,
            risk=result.get("risk"),
        )
        # Preserve compatibility for manually constructed legacy facts while
        # keeping new Stage 1 facts independent from coarse relevance.
        support = _normalized_label(result.get("support_level"))
        if support == "strong" and result.get("evidence_triples"):
            score = max(score, 0.84)
        elif support == "moderate" and result.get("evidence_triples"):
            score = max(score, 0.68)
        result["evidence_prior_score"] = round(score, 6)
    else:
        score = clamp_credibility_score(result.get("evidence_prior_score"))
        _, basis = evidence_credibility(
            evidence_triples=_list_value(result.get("evidence_triples")),
            evidence_refs=evidence_refs,
            risk=result.get("risk"),
        )

    if result.get("credibility_score") is None:
        result["credibility_score"] = score
    else:
        result["credibility_score"] = clamp_credibility_score(
            result.get("credibility_score")
        )
    result.setdefault("qwen_score_delta", 0.0)

    result["credibility_schema_version"] = CREDIBILITY_SCHEMA_VERSION
    final_score = clamp_credibility_score(result["credibility_score"])
    result["credibility_score"] = round(final_score, 6)
    result["credibility_label"] = credibility_label(final_score)
    result["credibility_basis"] = basis
    result["allowed_use"] = allowed_use_for_fact(result, final_score)
    result["support_level"] = {
        "high": "strong",
        "medium": "moderate",
    }.get(result["credibility_label"], "weak")
    result["generation_use"] = {
        "direct_generation": "use_as_grounded_constraint",
        "supporting_only": "use_as_supporting_context",
        "context_only": "context_only",
        "excluded": "forbidden",
    }[result["allowed_use"]]
    result["generation_use_policy_source"] = "credibility_policy"
    result.setdefault(
        "credibility_reason",
        _default_reason(result["allowed_use"], basis["evidence_type"]),
    )
    return result


def apply_fact_assessment(
    fact: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a bounded Qwen correction to the deterministic evidence prior."""
    result = dict(fact)
    base_score = result.get("evidence_prior_score")
    if base_score is None:
        base_score = result.get("credibility_score")
    if base_score is None:
        base_score, _ = evidence_credibility(
            evidence_triples=_list_value(result.get("evidence_triples")),
            evidence_refs=_list_value(result.get("stage0_fact_ids"))
            + _list_value(result.get("evidence_refs")),
            risk=result.get("risk"),
        )
    base_score = clamp_credibility_score(base_score)

    if assessment.get("score_delta") is not None:
        score_delta = clamp_score_delta(assessment.get("score_delta"))
        result["qwen_score_delta"] = score_delta
        result["credibility_score"] = clamp_credibility_score(base_score + score_delta)
    else:
        # Accept old saved responses during migration, but new prompts use deltas.
        absolute_score = assessment.get("credibility_score")
        score_delta = clamp_score_delta(
            clamp_credibility_score(absolute_score) - base_score
            if absolute_score is not None
            else 0.0
        )
        result["qwen_score_delta"] = score_delta
        result["credibility_score"] = clamp_credibility_score(base_score + score_delta)
    result["evidence_prior_score"] = base_score
    if assessment.get("risk") is not None:
        result["risk"] = str(assessment["risk"])
    reason = str(assessment.get("reason") or "").strip()
    if reason:
        result["credibility_reason"] = reason
    result["credibility_source"] = "semantic_planner"
    return ensure_fact_credibility(result)


def _default_reason(allowed_use: str, evidence_type: str) -> str:
    return (
        f"Deterministic credibility policy classified {evidence_type} evidence "
        f"as {allowed_use}."
    )


def _normalized_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _unique_non_empty(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []
