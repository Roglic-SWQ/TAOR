"""Compile model relation plans into candidate contexts and ranking references."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SELECTION_SOURCE = "stage2_fact_plan"


def compile_plan_context(
    knowledge: dict[str, Any], selection: dict[str, Any], missing_mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate decisions without re-evaluating credibility or changing any quota."""
    result = deepcopy(selection)
    payload = result["fine_grained_relation_selection"]
    plans = {
        item["fact_id"]: item
        for role in ("primary", "secondary", "context", "excluded")
        for item in payload[f"{role}_relations"]
    }
    profile = knowledge["stage1_v3"]["entity_profile"]
    identity = []
    if profile.get("evidence_refs"):
        identity.append({
            "constraint_id": f"constraint:{missing_mode}:identity",
            "constraint": profile["core_identity"], "constraint_type": "identity",
            "source": "entity_profile", "source_relations": [], "core": True,
            "evidence_triples": [], "evidence_refs": list(profile["evidence_refs"]),
            "severity": "hard", "gate_action": "hard_required",
            "violation_code": "missing_core_must_include", "selection_source": SELECTION_SOURCE,
        })

    def constraint(item: dict[str, Any], *, required: bool = False) -> dict[str, Any]:
        fact = item["source_fact"]
        role = item["relation_role"]
        return {
            "constraint_id": f"constraint:{missing_mode}:{fact['fact_id']}",
            "constraint": fact["claim"],
            "constraint_type": "context" if role == "context" else "grounded_fact",
            "source": "relation_plan", "source_relations": [fact["relation"]],
            "source_fact_refs": [fact["fact_id"]],
            "evidence_triples": list(fact["evidence_triples"]),
            "evidence_refs": list(item["selected_evidence_refs"]),
            "reason": item["role_decision"], "core": False, "severity": "soft",
            "gate_action": "context_only" if role == "context" else "soft_required" if required else "prefer",
            "violation_code": "context_only_evidence" if role == "context" else (
                "missing_non_core_must_include" if required else "missing_should_include"
            ),
            "selection_source": SELECTION_SOURCE,
        }

    context = [constraint(item) for item in payload["context_relations"]]
    supported = []
    for role in ("primary", "secondary"):
        for item in payload[f"{role}_relations"]:
            source = deepcopy(item["source_fact"])
            supported.append({
                **source, "source_fact_refs": [source["fact_id"]],
                "source_relations": [source["relation"]], "usable_for": [missing_mode],
                "selection_source": SELECTION_SOURCE, "selected_role": role,
                "relation_role": role, "source_plan_id": item["plan_id"],
            })
            item["generation_context"] = {
                "must_include": [*deepcopy(identity), constraint(item, required=True)],
                "should_include": [constraint(plans[ref]) for ref in item["supporting_fact_ids"]],
                "context_only": deepcopy(context),
                "candidate_quota": item["candidate_quota"],
                "candidate_quota_by_modality": deepcopy(item["candidate_quota_by_modality"]),
            }
    return result, {
        "schema_version": "plan_generation_context_v1",
        "supported_facts": supported, "safe_inferences": [],
        "uncertain_facts": deepcopy(knowledge.get("uncertainties", [])),
        "modality_constraints": {missing_mode: {
            "must_include": identity,
            "should_include": [constraint(item) for role in ("primary", "secondary")
                               for item in payload[f"{role}_relations"]],
            "context_only": context,
        }},
        "constraint_policy": {"ground_truth_used": False},
    }


def ranking_knowledge(knowledge: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Keep unused/context facts in Stage 1 artifacts but outside positive ranking references."""
    return {**knowledge, "relation_facts": deepcopy(context["supported_facts"])}
