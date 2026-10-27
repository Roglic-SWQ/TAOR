"""Stage 1 constraint compilation for missing-modality generation."""

from __future__ import annotations

from typing import Any

from mmkgc.stages.credibility import ensure_fact_credibility

CORE_MISSING_CODE = "missing_core_must_include"
NON_CORE_MISSING_CODE = "missing_non_core_must_include"
SHOULD_MISSING_CODE = "missing_should_include"

def _constraint_text(fact: dict[str, Any]) -> str:
    neighbors = ", ".join(map(str, fact.get("neighbor_entities", [])))
    return f"{fact.get('relation')}: {neighbors}" if neighbors else str(fact.get("relation"))


def _constraint_object(
    constraint: str,
    constraint_type: str,
    source: str,
    source_relations: list[str],
    evidence_triples: list[str],
    confidence: str,
    reason: str,
    *,
    severity: str,
    gate_action: str,
    violation_code: str,
    core: bool = False,
    evidence_policy: str = "direct_kg",
) -> dict[str, Any]:
    return {
        "constraint": constraint,
        "constraint_type": constraint_type,
        "source": source,
        "source_relations": source_relations,
        "evidence_triples": evidence_triples,
        "confidence": confidence,
        "reason": reason,
        "severity": severity,
        "gate_action": gate_action,
        "violation_code": violation_code,
        "core": core,
        "evidence_policy": evidence_policy,
    }


def _empty_constraints() -> dict[str, list[dict[str, Any]]]:
    return {"must_include": [], "should_include": [], "context_only": []}


def _assign_constraint_ids(modality_constraints: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    for modality, buckets in modality_constraints.items():
        if not isinstance(buckets, dict):
            continue
        for bucket, items in buckets.items():
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items, start=1):
                if isinstance(item, dict) and not item.get("constraint_id"):
                    item["constraint_id"] = f"constraint:{modality}:{bucket}:{index:04d}"


def _assign_global_constraint_ids(global_constraints: dict[str, list[dict[str, Any]]]) -> None:
    for bucket, items in global_constraints.items():
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, start=1):
            if isinstance(item, dict) and not item.get("constraint_id"):
                item["constraint_id"] = f"constraint:global:{bucket}:{index:04d}"


def _append_unique(bucket: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if not any(existing.get("constraint") == item["constraint"] for existing in bucket):
        bucket.append(item)


def _categorical_confidence(supported_count: int, entity_confidence: float) -> str:
    if supported_count >= 3 and entity_confidence >= 0.7:
        return "high"
    if supported_count >= 1 or entity_confidence >= 0.45:
        return "medium"
    return "low"


def _confidence_for_fact(fact: dict[str, Any]) -> str:
    label = str(fact.get("credibility_label") or "").strip().lower()
    if label in {"high", "medium", "low"}:
        return label
    return "high" if fact.get("support_level") == "strong" else "medium"


def _normalized_entity_profile(knowledge: dict[str, Any]) -> dict[str, Any]:
    """Return a Stage1-ready profile, preferring the cleaned Stage1 v3 profile."""
    profile = dict(knowledge.get("entity_profile", {}))
    stage1_v3 = knowledge.get("stage1_v3")
    if not isinstance(stage1_v3, dict):
        return profile
    v3_profile = stage1_v3.get("entity_profile")
    if not isinstance(v3_profile, dict):
        return profile
    primary_type = v3_profile.get("primary_type")
    if primary_type:
        profile["type"] = primary_type
        profile["entity_type"] = primary_type
    if v3_profile.get("core_identity"):
        profile["core_identity"] = v3_profile["core_identity"]
    if v3_profile.get("confidence") is not None:
        profile["confidence"] = v3_profile["confidence"]
    if v3_profile.get("type_confidence"):
        profile["type_confidence"] = v3_profile["type_confidence"]
    if isinstance(v3_profile.get("evidence_refs"), list):
        profile["type_evidence"] = list(v3_profile["evidence_refs"])
    if isinstance(v3_profile.get("secondary_types"), list):
        profile["secondary_types"] = list(v3_profile["secondary_types"])
    if isinstance(v3_profile.get("semantic_domains"), list):
        profile["semantic_domain"] = list(v3_profile["semantic_domains"])
    uncertain = stage1_v3.get("uncertain_aspects")
    if isinstance(uncertain, list):
        profile["uncertain_aspects"] = list(uncertain)
        profile["uncertainty"] = [
            str(item.get("reason") or item.get("aspect"))
            for item in uncertain
            if isinstance(item, dict) and (item.get("reason") or item.get("aspect"))
        ]
    return profile


def _modality_usage_for_fact(fact: dict[str, Any], target_modality: str) -> dict[str, str]:
    allowed_use = str(fact.get("allowed_use") or "supporting_only")
    if allowed_use == "direct_generation":
        bucket = "must_include"
    elif allowed_use == "supporting_only":
        bucket = "should_include"
    else:
        bucket = "context_only"
    return {target_modality: bucket}


def _fact_type(fact: dict[str, Any]) -> str:
    relation = str(fact.get("relation") or "").lower()
    if relation == "instance of":
        return "entity_type"
    if fact.get("fact_type"):
        return str(fact["fact_type"])
    return "relation"


def _supported_fact(fact: dict[str, Any], target_modality: str) -> dict[str, Any]:
    fact = ensure_fact_credibility(fact)
    source_relations = list(fact.get("source_relations") or [fact.get("relation")])
    evidence = list(fact.get("evidence_triples", []))
    usage = _modality_usage_for_fact(fact, target_modality)
    return {
        "fact_id": fact.get("fact_id"),
        "fact": fact.get("fact"),
        "fact_type": _fact_type(fact),
        "source": "relation_facts",
        "source_relations": source_relations,
        "source_fact_refs": [str(fact["fact_id"])] if fact.get("fact_id") else [],
        "evidence_refs": list(fact.get("stage0_fact_ids", [])),
        "evidence_triples": evidence,
        "usable_for": [
            modality
            for modality, bucket in usage.items()
            if bucket in {"must_include", "should_include", "context_only"}
        ],
        "confidence": _confidence_for_fact(fact),
        "credibility_score": fact.get("credibility_score"),
        "credibility_label": fact.get("credibility_label"),
        "allowed_use": fact.get("allowed_use"),
        "support_level": fact.get("support_level"),
        "reason": fact.get("credibility_reason")
        or "Directly supported by selected KG relation evidence.",
    }


def _uncertain_relation_fact(fact: dict[str, Any]) -> dict[str, Any]:
    action = "context_only" if fact.get("allowed_use") == "context_only" else "exclude"
    return {
        "fact_id": fact.get("fact_id"),
        "uncertainty": fact.get("fact"),
        "source": "relation_facts",
        "source_relations": list(fact.get("source_relations") or [fact.get("relation")]),
        "source_fact_refs": [str(fact["fact_id"])] if fact.get("fact_id") else [],
        "evidence_refs": list(fact.get("stage0_fact_ids", [])),
        "evidence_triples": list(fact.get("evidence_triples", [])),
        "credibility_score": fact.get("credibility_score"),
        "credibility_label": fact.get("credibility_label"),
        "allowed_use": fact.get("allowed_use"),
        "recommended_action": action,
        "severity": "soft",
        "gate_action": action,
        "violation_code": "unsupported_high_risk_fact",
        "reason": fact.get("credibility_reason")
        or "The fact does not meet the threshold for generation use.",
    }


def _profile_supported_fact(profile: dict[str, Any], target_modality: str) -> dict[str, Any] | None:
    entity_type = profile.get("type")
    if not entity_type or entity_type == "entity":
        return None
    evidence = list(profile.get("type_evidence", []))
    confidence = "high" if profile.get("confidence", 0) >= 0.7 else "medium"
    return {
        "fact_id": "sf:entity_type",
        "fact": f"{profile.get('core_identity', 'The target entity')} has entity type: {entity_type}",
        "fact_type": "entity_type",
        "source": "entity_profile",
        "source_relations": ["instance of"] if evidence else [],
        "source_fact_refs": [],
        "evidence_refs": evidence,
        "evidence_triples": evidence,
        "usable_for": [target_modality],
        "confidence": confidence,
        "support_level": "strong" if confidence == "high" else "moderate",
        "reason": "The entity type is directly supported by the graph-derived profile.",
    }


def _safe_inference(profile: dict[str, Any], entity_type: str, target_modality: str) -> dict[str, Any]:
    identity = profile.get("core_identity", "the entity")
    return {
        "inference": f"Describe {identity} as a {entity_type}.",
        "inference_type": "generation_hint",
        "based_on_facts": [f"entity type: {entity_type}"],
        "source_relations": [],
        "evidence_refs": list(profile.get("type_evidence", [])),
        "evidence_triples": list(profile.get("type_evidence", [])),
        "usable_for": [target_modality],
        "confidence": "medium",
        "risk": "Do not add exact visual or biographical details beyond the evidence.",
        "reason": "Entity type is safe as a coarse generation hint.",
    }


def _uncertain_facts(items: list[Any]) -> list[dict[str, Any]]:
    """Preserve uncertainty as context without compiling a negative list."""
    result = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(
            {
                "uncertainty": text,
                "source": "knowledge_uncertainty",
                "recommended_action": "context_only",
                "severity": "soft",
                "gate_action": "context_only",
                "violation_code": "unsupported_high_risk_fact",
                "note": "Use only as uncertainty context; do not promote to a supported fact.",
            }
        )
    return result


def _text_profile_constraints(knowledge: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stage1_v3 = knowledge.get("stage1_v3")
    profile = (
        stage1_v3.get("text_visual_profile", {})
        if isinstance(stage1_v3, dict)
        else knowledge.get("text_visual_profile", {})
    )
    if not isinstance(profile, dict) or not profile:
        return {"should_include": [], "context_only": []}
    should = []
    context = []
    for cue in profile.get("visual_scene_cues", [])[:4]:
        text = str(cue).strip()
        if not text:
            continue
        should.append(
            _constraint_object(
                constraint=f"text visual cue: {text}",
                constraint_type="grounded_fact",
                source="text_visual_profile",
                source_relations=[],
                evidence_triples=[],
                confidence="medium",
                reason="Derived from the existing text modality as a safe visual generation cue.",
                severity="soft",
                gate_action="prefer",
                violation_code=SHOULD_MISSING_CODE,
                core=False,
                evidence_policy="entity_text",
            )
        )
    for cue in profile.get("context_cues", [])[:6]:
        text = str(cue).strip()
        if not text:
            continue
        context.append(
            _constraint_object(
                constraint=f"text context cue: {text}",
                constraint_type="context",
                source="text_visual_profile",
                source_relations=[],
                evidence_triples=[],
                confidence="medium",
                reason="Derived from the existing text modality as context, not a hard visual requirement.",
                severity="soft",
                gate_action="context_only",
                violation_code="context_only_evidence",
                core=False,
                evidence_policy="entity_text",
            )
        )
    return {"should_include": should, "context_only": context}


def build_knowledge_from_v3_artifacts(
    stage0: dict[str, Any],
    stage1: dict[str, Any],
    *,
    entity_label: str | None = None,
) -> dict[str, Any]:
    """Adapt compact v3 Stage 0/1 artifacts to the Stage 1 function input."""
    label = entity_label or _infer_entity_label(stage0) or "target entity"
    relation_refs = _relation_refs_by_relation(stage1)
    relation_facts = [
        _relation_fact_from_stage0_relation(
            index,
            relation,
            stage0,
            label,
            relation_refs.get(str(relation.get("relation") or "")),
        )
        for index, relation in enumerate(_selected_stage0_relations(stage0), start=1)
    ]
    profile = dict(stage1.get("entity_profile", {}))
    primary_type = str(profile.get("primary_type") or profile.get("entity_type") or "entity")
    uncertainties = [
        str(item.get("reason") or item.get("aspect"))
        for item in stage1.get("uncertain_aspects", [])
        if isinstance(item, dict) and (item.get("reason") or item.get("aspect"))
    ]
    text_visual_profile = dict(stage1.get("text_visual_profile", {}))
    return {
        "entity_profile": {
            "type": primary_type,
            "entity_type": primary_type,
            "secondary_types": list(profile.get("secondary_types", [])),
            "semantic_domain": list(profile.get("semantic_domains", [])),
            "core_identity": profile.get("core_identity") or f"a {primary_type}",
            "confidence": profile.get("confidence", 0),
            "type_confidence": profile.get("type_confidence"),
            "type_evidence": list(profile.get("evidence_refs", [])),
            "uncertainty": uncertainties,
        },
        "stage1_v3": stage1,
        "text_visual_profile": text_visual_profile,
        "relation_facts": relation_facts,
        "uncertainties": uncertainties,
    }


def _selected_stage0_relations(stage0: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        relation
        for relation in stage0.get("relations", [])
        if isinstance(relation, dict) and relation.get("decision") == "selected"
    ]


def _stage0_facts_by_id(stage0: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(fact.get("fact_id")): fact
        for fact in stage0.get("facts", [])
        if isinstance(fact, dict) and fact.get("fact_id")
    }


def _relation_refs_by_relation(stage1: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs = {}
    for ref in stage1.get("relation_fact_refs", []):
        if isinstance(ref, dict) and ref.get("relation"):
            refs[str(ref["relation"])] = ref
    return refs


def _relation_fact_from_stage0_relation(
    index: int,
    relation: dict[str, Any],
    stage0: dict[str, Any],
    entity_label: str,
    relation_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    fact_lookup = _stage0_facts_by_id(stage0)
    fact_ids = [str(item) for item in relation.get("fact_ids", [])]
    facts = [fact_lookup[fact_id] for fact_id in fact_ids if fact_id in fact_lookup]
    direction = str(relation.get("direction") or "out")
    predicate = str(relation.get("relation") or "")
    neighbors = list(relation.get("neighbor_entities", []))
    first_fact = facts[0] if facts else {}
    if direction == "in":
        subject = str(first_fact.get("subject") or (neighbors[0] if neighbors else "unknown"))
        obj = entity_label
        fact_text = f"{subject} has {predicate} to {obj}"
    else:
        subject = entity_label
        obj = str(first_fact.get("object") or (neighbors[0] if neighbors else "unknown"))
        fact_text = f"{subject} has {predicate}: {', '.join(map(str, neighbors))}"
    fact = {
        "fact_id": f"rf:{index:04d}",
        "relation": predicate,
        "source_relations": [predicate] if predicate else [],
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "fact": fact_text,
        "fact_type": "graph_relation",
        "evidence_triples": [str(fact.get("triple")) for fact in facts if fact.get("triple")],
        "coarse_relevance_score": float(relation.get("score") or 0),
        "support_level": str(
            (relation_ref or {}).get("support_level")
            or ("strong" if float(relation.get("score") or 0) >= 4 else "moderate")
        ),
        "direction": direction,
        "neighbor_entities": neighbors,
        "generation_use": "use_as_grounded_constraint",
        "risk": (relation_ref or {}).get("risk") or ("low" if facts else "medium"),
        "credibility_score": (relation_ref or {}).get("credibility_score"),
        "credibility_label": (relation_ref or {}).get("credibility_label"),
        "allowed_use": (relation_ref or {}).get("allowed_use"),
        "stage0_relation_id": relation.get("relation_id"),
        "stage0_fact_ids": fact_ids,
    }
    return ensure_fact_credibility(fact)


def _infer_entity_label(stage0: dict[str, Any]) -> str | None:
    votes: dict[str, int] = {}
    fact_lookup = _stage0_facts_by_id(stage0)
    for relation in _selected_stage0_relations(stage0):
        direction = str(relation.get("direction") or "out")
        for fact_id in relation.get("fact_ids", []):
            fact = fact_lookup.get(str(fact_id))
            if not fact:
                continue
            value = fact.get("object") if direction == "in" else fact.get("subject")
            if value:
                votes[str(value)] = votes.get(str(value), 0) + 1
    if not votes:
        return None
    return max(votes.items(), key=lambda item: item[1])[0]


def constraint_compilation(missing_mode: str, knowledge: dict[str, Any]) -> dict[str, Any]:
    """Translate modeled knowledge into conservative modality constraints."""
    relation_facts = [
        ensure_fact_credibility(fact)
        for fact in knowledge.get("relation_facts", [])
        if isinstance(fact, dict)
    ]
    if missing_mode not in {"text", "image"}:
        raise ValueError("missing_mode must be one of: text, image")
    modality_constraints = {missing_mode: _empty_constraints()}
    profile = _normalized_entity_profile(knowledge)
    entity_type = profile.get("type")
    supported_facts = []
    if profile_fact := _profile_supported_fact(profile, missing_mode):
        supported_facts.append(profile_fact)
    supported_facts.extend(
        _supported_fact(fact, missing_mode)
        for fact in relation_facts
        if fact.get("allowed_use") in {"direct_generation", "supporting_only"}
    )
    if entity_type and entity_type != "entity":
        modality_constraints[missing_mode]["must_include"].append(
                _constraint_object(
                    constraint=f"entity type: {entity_type}",
                    constraint_type="identity",
                    source="entity_profile",
                    source_relations=[],
                    evidence_triples=list(profile.get("type_evidence", [])),
                    confidence="high" if profile.get("confidence", 0) >= 0.7 else "medium",
                    reason="Entity type anchors generation to the supported graph profile.",
                    severity="hard",
                    gate_action="hard_required",
                    violation_code=CORE_MISSING_CODE,
                    core=True,
                    evidence_policy="entity_profile",
                )
            )

    for fact in relation_facts:
        item = _constraint_text(fact)
        modality = missing_mode
        if fact.get("allowed_use") == "excluded":
            continue
        if fact.get("allowed_use") == "direct_generation":
            bucket = "must_include"
            constraint_type = "grounded_fact"
            severity = "soft"
            gate_action = "soft_required"
            violation_code = NON_CORE_MISSING_CODE
        elif fact.get("allowed_use") == "supporting_only":
            bucket = "should_include"
            constraint_type = "grounded_fact"
            severity = "soft"
            gate_action = "prefer"
            violation_code = SHOULD_MISSING_CODE
        else:
            bucket = "context_only"
            constraint_type = "context"
            severity = "soft"
            gate_action = "context_only"
            violation_code = "context_only_evidence"
        _append_unique(
            modality_constraints[modality][bucket],
            _constraint_object(
                constraint=item,
                constraint_type=constraint_type,
                source="relation_fact",
                source_relations=list(fact.get("source_relations") or [fact.get("relation")]),
                evidence_triples=list(fact.get("evidence_triples", [])),
                confidence=_confidence_for_fact(fact),
                reason=f"Derived from {fact.get('direction', 'out')} graph evidence.",
                severity=severity,
                gate_action=gate_action,
                violation_code=violation_code,
                core=False,
            ),
        )

    text_profile_constraints = _text_profile_constraints(knowledge)
    if missing_mode == "image":
        for item in text_profile_constraints["should_include"]:
            _append_unique(modality_constraints["image"]["should_include"], item)
        for item in text_profile_constraints["context_only"]:
            _append_unique(modality_constraints["image"]["context_only"], item)

    global_constraints: dict[str, list[dict[str, Any]]] = {}

    safe_inferences = [
        _safe_inference(profile, str(entity_type), missing_mode)
    ] if entity_type and entity_type != "entity" else []
    uncertain = _uncertain_facts(
        list(knowledge.get("uncertainties", [])) + list(profile.get("uncertainty", []))
    )
    uncertain.extend(
        _uncertain_relation_fact(fact)
        for fact in relation_facts
        if fact.get("allowed_use") in {"context_only", "excluded"}
    )
    target_modes = [missing_mode]
    reasoning_confidence = _categorical_confidence(len(supported_facts), float(profile.get("confidence", 0)))
    _assign_constraint_ids(modality_constraints)
    _assign_global_constraint_ids(global_constraints)

    return {
        "reasoning_summary": (
            f"Built graph-constrained guidance for {', '.join(target_modes)} "
            f"from {len(supported_facts)} supported relation facts."
        ),
        "supported_facts": supported_facts,
        "fact_credibility_assessments": [
            {
                "fact_id": fact.get("fact_id"),
                "credibility_score": fact.get("credibility_score"),
                "credibility_label": fact.get("credibility_label"),
                "allowed_use": fact.get("allowed_use"),
                "risk": fact.get("risk"),
                "reason": fact.get("credibility_reason"),
            }
            for fact in relation_facts
        ],
        "safe_inferences": safe_inferences,
        "uncertain_facts": uncertain,
        "global_constraints": global_constraints,
        "modality_constraints": modality_constraints,
        "constraint_policy": {
            "hard_violation_codes": [CORE_MISSING_CODE],
            "soft_violation_codes": [
                NON_CORE_MISSING_CODE,
                SHOULD_MISSING_CODE,
                "context_only_evidence",
                "unsupported_high_risk_fact",
            ],
            "ground_truth_used": False,
        },
        "reasoning_confidence": reasoning_confidence,
    }
