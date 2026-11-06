"""Compile model relation plans into candidate contexts and ranking references."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mmkgc.observation import compile_verification_targets
from mmkgc.stages.candidate_planning import PLAN_SCHEMAS, SELECTION_SOURCE as CANDIDATE_SOURCE
from mmkgc.stages.triples import fact_text
from mmkgc.stages.visual_evidence import generation_eligible
from mmkgc.plan_schema import FACT_SELECTION_SCHEMA

SELECTION_SOURCE = "stage2_fact_plan"


def compile_plan_context(
    knowledge: dict[str, Any], selection: dict[str, Any], missing_mode: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate decisions without re-evaluating credibility or changing any quota."""
    schema = selection["fine_grained_relation_selection"].get("schema_version")
    if schema == FACT_SELECTION_SCHEMA:
        selection = deepcopy(selection)
        payload = selection["fine_grained_relation_selection"]
        payload["candidates"] = ([{"plan_id": "fs:shared", "fact_ids": list(payload["selected_fact_ids"])}]
                                 if payload["status"] == "ready" else [])
        return _compile_candidate_context(knowledge, selection, missing_mode)
    if schema in PLAN_SCHEMAS:
        return _compile_candidate_context(knowledge, selection, missing_mode)
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
            "constraint": fact_text(fact),
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
            item["verification_targets"] = compile_verification_targets(
                item["generation_context"], knowledge, missing_mode,
            )
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


def _compile_candidate_context(knowledge, selection, modality):
    result = deepcopy(selection)
    payload = result["fine_grained_relation_selection"]
    facts = {fact["fact_id"]: fact for fact in knowledge["relation_facts"]}
    profile = knowledge["stage1_v3"]["entity_profile"]
    identity = []
    if profile.get("evidence_refs"):
        identity = [{
            "constraint_id": f"constraint:{modality}:identity", "constraint": profile["core_identity"],
            "constraint_type": "identity", "evidence_refs": list(profile["evidence_refs"]),
            "source_fact_refs": [], "source_relations": [], "evidence_triples": [],
            "selection_source": CANDIDATE_SOURCE,
        }]
    selected = {}

    def requirement(fact):
        return {
            "constraint_id": f"constraint:{modality}:{fact['fact_id']}",
            "constraint": fact_text(fact), "constraint_type": "grounded_fact",
            "source_fact_refs": [fact["fact_id"]], "source_relations": [fact["relation"]],
            "evidence_refs": list(fact["evidence_refs"]),
            "evidence_triples": list(fact.get("evidence_triples", [])),
            "selection_source": CANDIDATE_SOURCE,
        }

    for candidate in payload["candidates"]:
        candidate["plan_schema_version"] = payload["schema_version"]
        sources = [deepcopy(facts[ref]) for ref in candidate["fact_ids"]]
        if any(not generation_eligible(fact) for fact in sources):
            raise ValueError("uncertain or incidental image observations cannot become generation requirements")
        for fact in sources:
            selected[fact["fact_id"]] = {
                **fact, "fact": fact_text(fact),
                "source_fact_refs": [fact["fact_id"]], "source_relations": [fact["relation"]],
                "usable_for": [modality], "selection_source": CANDIDATE_SOURCE,
            }
        candidate["source_facts"] = sources
        candidate["relation"] = "; ".join(dict.fromkeys(fact["relation"] for fact in sources))
        candidate["source_fact_refs"] = list(candidate["fact_ids"])
        candidate["evidence_triples"] = list(dict.fromkeys(
            triple for fact in sources for triple in fact.get("evidence_triples", [])))
        image_views = payload["schema_version"] == FACT_SELECTION_SCHEMA and modality == "image"
        from mmkgc.visual_description import enabled, TASK
        visual_description = modality == 'text' and enabled(knowledge)
        requirements = [requirement(fact) for fact in sources]
        candidate["generation_context"] = {
            "must_include": [*deepcopy(identity), *([] if image_views or visual_description else requirements)],
            "should_include": requirements if image_views or visual_description else [], "context_only": [],
            **({'description_task': TASK} if visual_description else {}),
        }
        candidate["verification_targets"] = compile_verification_targets(
            candidate["generation_context"], knowledge, modality)
    return result, {
        "schema_version": payload["schema_version"], "supported_facts": list(selected.values()), "safe_inferences": [],
        "uncertain_facts": deepcopy(knowledge.get("uncertainties", [])),
        "modality_constraints": {modality: {"must_include": identity,
            "should_include": [requirement(fact) for fact in selected.values()], "context_only": []}},
        "constraint_policy": {"ground_truth_used": False},
    }


def ranking_knowledge(knowledge: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Keep unused/context facts in Stage 1 artifacts but outside positive ranking references."""
    facts = deepcopy(context['supported_facts'])
    from mmkgc.visual_description import enabled
    if enabled(knowledge):
        # Pixel-review prose and old captions are audit observations, not extra
        # selected assertions. Do not let the Judge use them to excuse a scene
        # inventory that the writing task deliberately excluded.
        for fact in facts:
            if 'visual_source_checks' in fact:
                fact['visual_source_checks'] = [{key: deepcopy(value) for key,value in check.items()
                    if key not in {'observed','reason'}} for check in fact['visual_source_checks']]
        catalog = deepcopy(knowledge.get('evidence_catalog', []))
        for source in catalog:
            if source.get('modality') == 'image':
                source.pop('content', None)
                if source.get('entity_match'):
                    source['entity_match']={key:value for key,value in source['entity_match'].items()
                                            if key in {'status','identity_verified'}}
        return {**knowledge,'relation_facts':facts,'evidence_catalog':catalog}
    return {**knowledge, 'relation_facts': facts}
