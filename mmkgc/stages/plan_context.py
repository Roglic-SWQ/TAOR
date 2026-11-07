"""Bind selected evidence to candidates without coverage obligations."""
from copy import deepcopy
from typing import Any

from mmkgc.observation import compile_verification_targets
from mmkgc.plan_schema import FACT_SELECTION_SCHEMA
from mmkgc.stages.candidate_planning import PLAN_SCHEMAS, SELECTION_SOURCE
from mmkgc.stages.triples import fact_text
from mmkgc.stages.visual_evidence import generation_eligible

CONTEXT_VERSION = "selected_evidence_context_v2"


def compile_plan_context(knowledge, selection, missing_mode):
    result = deepcopy(selection)
    payload = result["fine_grained_relation_selection"]
    schema = payload.get("schema_version")
    shared = schema == FACT_SELECTION_SCHEMA
    simple = shared or schema in PLAN_SCHEMAS
    if shared:
        payload["candidates"] = ([{"plan_id": "fs:shared", "fact_ids": list(payload["selected_fact_ids"])}]
                                 if payload["status"] == "ready" else [])
    candidates = (payload["candidates"] if simple else
                  [item for role in ("primary", "secondary") for item in payload[role + "_relations"]])
    facts = {fact["fact_id"]: fact for fact in knowledge["relation_facts"]}
    profile = knowledge["stage1_v3"]["entity_profile"]
    identity = []
    if profile.get("evidence_refs"):
        identity = [{"constraint_id": f"constraint:{missing_mode}:identity",
            "constraint": profile["core_identity"], "constraint_type": "identity",
            "evidence_refs": list(profile["evidence_refs"]), "source_fact_refs": [],
            "selection_source": SELECTION_SOURCE}]

    def check(fact):
        return {"constraint_id": f"constraint:{missing_mode}:{fact['fact_id']}",
            "constraint": fact_text(fact), "constraint_type": "grounded_fact",
            "source_fact_refs": [fact["fact_id"]], "source_relations": [fact["relation"]],
            "evidence_refs": list(fact["evidence_refs"]),
            "evidence_triples": list(fact.get("evidence_triples", [])), "selection_source": SELECTION_SOURCE}

    selected = {}
    for item in candidates:
        if simple:
            ids = item["fact_ids"]
        else:
            by_id = {entry["fact_id"]: entry for role in ("primary", "secondary", "context", "excluded")
                     for entry in payload[role + "_relations"]}
            ids = [item["fact_id"], *item.get("supporting_fact_ids", [])]
            facts.update({ref: by_id[ref]["source_fact"] for ref in ids})
        sources = [deepcopy(facts[ref]) for ref in ids]
        if any(not generation_eligible(fact) for fact in sources):
            raise ValueError("uncertain or incidental image observations cannot become selected evidence")
        for fact in sources:
            selected[fact["fact_id"]] = {**fact, "fact": fact_text(fact),
                "source_fact_refs": [fact["fact_id"]], "source_relations": [fact["relation"]],
                "usable_for": [missing_mode], "selection_source": SELECTION_SOURCE}
        item.update(plan_schema_version=schema, source_facts=sources,
            relation="; ".join(dict.fromkeys(fact["relation"] for fact in sources)),
            source_fact_refs=list(ids), evidence_triples=list(dict.fromkeys(
                triple for fact in sources for triple in fact.get("evidence_triples", []))))
        from mmkgc.visual_description import enabled, TASK
        item["generation_context"] = {"schema_version": CONTEXT_VERSION,
            "checks": [*deepcopy(identity), *[check(fact) for fact in sources]], "context_only": [],
            **({key: deepcopy(item[key]) for key in ('candidate_quota', 'candidate_quota_by_modality')
                if key in item} if not simple else {}),
            **({"description_task": TASK} if missing_mode == 'text' and enabled(knowledge) else {})}
        item["verification_targets"] = compile_verification_targets(item["generation_context"], knowledge, missing_mode)
    return result, {"schema_version": CONTEXT_VERSION, "supported_facts": list(selected.values()),
        "safe_inferences": [], "uncertain_facts": deepcopy(knowledge.get("uncertainties", [])),
        "modality_constraints": {missing_mode: {"checks": identity + [check(fact) for fact in selected.values()],
                                               "context_only": []}},
        "constraint_policy": {"ground_truth_used": False}}


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
