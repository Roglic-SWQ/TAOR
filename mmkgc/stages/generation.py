"""Stage 3: relation alternation generation."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

from mmkgc.adapters import GeneratedImageArtifact, ImageGenerator, ImagePromptGenerator, TextGenerator
from mmkgc.generation_prompts import PROMPT_VERSION, QWEN_PROMPT_VERSION


STAGE_OWNED_METADATA_KEYS = {
    "status",
    "fallback",
    "entity_type",
    "relation_role",
}

# These failures describe input, scorer, or candidate-transport problems. They
# cannot be repaired by changing the generated text or image prompt.
_NON_STAGE3_VIOLATION_CODES = frozenset(
    {
        "candidate_id_missing",
        "generator_failed",
        "judge_service_error",
        "malformed_candidate",
        "missing_image_artifact",
        "missing_image_input",
        "schema_invalid",
    }
)

_STAGE3_REPAIR_ACTIONS = {
    "unsupported_claim": "remove claims that are not supported by graph evidence",
    "unsupported_relation": "remove relations that are not supported by graph evidence",
    "unsupported_high_risk_fact": "remove unsupported high-risk facts",
    "forbidden_content": "remove forbidden or excluded content",
    "hard_constraint_violation": "remove content that violates hard constraints",
    "missing_core_evidence": "include the required core evidence",
    "missing_must_include": "include all required relations and facts",
    "missing_should_include": "increase coverage of recommended relations and facts",
    "core_relation_missing": "include the missing core relation",
    "low_relation_coverage": "increase coverage of the required relations",
    "weak_evidence_support": "use stronger graph-supported evidence",
    "context_only_evidence": "do not treat context-only evidence as a core fact",
    "generic_entity_type_only": "include entity-specific supported evidence",
    "entity_type_conflict": "keep the generated content consistent with the entity type",
    "low_semantic_alignment": "make the generated content more consistent with the selected relation",
    "low_visual_alignment": "make the image prompt explicitly express the required visual cues",
    "target_modality_mismatch": "generate the requested target modality",
    "empty_generated_content": "return a complete non-empty candidate",
}

_STAGE3_PRESERVE = (
    "entity identity",
    "supported core attributes",
    "selected relation roles",
)


def _target_modalities(missing_mode: str) -> list[str]:
    if missing_mode in {"text", "image"}:
        return [missing_mode]
    raise ValueError("missing_mode must be one of: text, image")


def _selection_payload(selection: dict[str, Any]) -> dict[str, Any]:
    return selection.get("fine_grained_relation_selection", selection)


def _constraints_for_modality(reasoning: dict[str, Any], modality: str) -> dict[str, list[dict[str, Any]]]:
    constraints = reasoning.get("modality_constraints", {}).get(modality, {})
    return {
        "must_include": list(constraints.get("must_include", [])),
        "should_include": list(constraints.get("should_include", [])),
        "context_only": list(constraints.get("context_only", [])),
    }


def _constraint_lookup(reasoning: dict[str, Any], modality: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    constraints = reasoning.get("modality_constraints", {}).get(modality, {})
    if not isinstance(constraints, Mapping):
        return lookup
    for bucket in ("must_include", "should_include", "context_only"):
        items = constraints.get(bucket, [])
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                continue
            ref = item.get("constraint_id") or item.get("constraint_ref") or item.get("id")
            ref = str(ref) if ref else f"constraint:{modality}:{bucket}:{index:04d}"
            lookup[ref] = dict(item)
    return lookup


def _generation_context_for_item(
    item: dict[str, Any],
    reasoning: dict[str, Any],
    modality: str,
) -> dict[str, Any]:
    """Resolve Stage2 constraint references into the Stage3 prompt context."""
    existing = item.get("generation_context")
    if isinstance(existing, Mapping):
        return dict(existing)
    relation_refs = item.get("relation_constraint_refs_by_bucket", {})
    global_refs = item.get("global_constraint_refs_by_bucket", {})
    lookup = _constraint_lookup(reasoning, modality)
    context: dict[str, Any] = {
        "must_include": [],
        "should_include": [],
        "context_only": [],
        "candidate_quota": int(item.get("candidate_quota", 0) or 0),
        "candidate_quota_by_modality": dict(item.get("candidate_quota_by_modality", {})),
    }
    for bucket in ("must_include", "should_include", "context_only"):
        refs = []
        for source in (global_refs, relation_refs):
            values = source.get(bucket, []) if isinstance(source, Mapping) else []
            if isinstance(values, list):
                refs.extend(str(value) for value in values)
        for ref in refs:
            constraint = lookup.get(ref)
            if constraint and not any(
                existing_item.get("constraint") == constraint.get("constraint")
                for existing_item in context[bucket]
            ):
                context[bucket].append(constraint)
    return context


def _unique_constraints(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    for item in items:
        if not any(existing.get("constraint") == item.get("constraint") for existing in unique):
            unique.append(item)
    return unique


def _fallback_relation(
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    modality: str,
    candidate_quota: int = 1,
) -> dict[str, Any]:
    constraints = _constraints_for_modality(reasoning, modality)
    constraints["candidate_quota"] = candidate_quota
    profile = knowledge.get("entity_profile", {})
    return {
        "relation": "global_profile",
        "relation_role": "fallback",
        "fallback": True,
        "fine_score": 0.0,
        "score_breakdown": {},
        "target_modality": [modality],
        "role_decision": "Fallback profile candidate used because relation quota did not fill the budget.",
        "generation_context": constraints,
        "evidence_triples": list(profile.get("type_evidence", [])),
        "source_fact": {
            "relation": "global_profile",
            "fact": f"{profile.get('core_identity', 'Entity')} has entity type {profile.get('entity_type', 'entity')}",
            "evidence_triples": list(profile.get("type_evidence", [])),
            "neighbor_entities": [],
        },
    }


def _relation_sequence(
    selection: dict[str, Any],
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    modality: str,
    candidate_budget: int,
) -> list[dict[str, Any]]:
    payload = _selection_payload(selection)
    if payload.get("schema_version") == "model_relation_plan_v1" and payload.get("status") == "insufficient_evidence":
        return []
    relations = list(payload.get("primary_relations", [])) + list(payload.get("secondary_relations", []))
    relations = [
        relation
        for relation in relations
        if modality in relation.get("target_modality", [])
    ]
    if "candidate_anchors" in payload:
        anchors = payload["candidate_anchors"]
        by_fact = {item["fact_id"]: item for item in relations}
        if not isinstance(anchors, list) or len(anchors) != candidate_budget or any(
            not isinstance(ref, str) or ref not in by_fact for ref in anchors
        ):
            raise ValueError("Stage 2 candidate anchors must exactly fill candidate_budget using selected facts")
        return [by_fact[ref] for ref in anchors]
    sequence = []
    for relation in relations:
        context = _generation_context_for_item(relation, reasoning, modality)
        by_modality = context.get("candidate_quota_by_modality", {})
        if isinstance(by_modality, Mapping) and modality in by_modality:
            quota = int(by_modality.get(modality) or 0)
        else:
            quota = int(context.get("candidate_quota", 0) or 0)
        sequence.extend([relation] * max(0, quota))

    if len(sequence) < candidate_budget:
        if payload.get("schema_version") == "model_relation_plan_v1":
            raise ValueError("Stage 2 model quota does not fill candidate_budget")
        fallback = _fallback_relation(knowledge, reasoning, modality, candidate_budget - len(sequence))
        sequence.extend([fallback] * (candidate_budget - len(sequence)))

    return sequence[:candidate_budget]


def _facts_for_relation(
    item: dict[str, Any],
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    modality: str,
) -> list[str]:
    fact = item.get("source_fact", {})
    facts = []
    if fact.get("fact"):
        facts.append(str(fact["fact"]))
    context = _generation_context_for_item(item, reasoning, modality)
    for constraint in list(context.get("must_include", [])) + list(context.get("should_include", [])):
        text = constraint.get("constraint")
        if text and text not in facts:
            facts.append(str(text))
    if not facts:
        profile = knowledge.get("entity_profile", {})
        facts.append(f"entity type: {profile.get('entity_type', 'entity')}")
    return facts


def _constraint_strings(items: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("constraint"))
        for item in items
        if isinstance(item, Mapping) and item.get("constraint")
    ]


def _constraint_refs(items: list[dict[str, Any]]) -> list[str]:
    refs = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            continue
        ref = item.get("constraint_id") or item.get("constraint_ref") or item.get("id")
        refs.append(str(ref) if ref else f"constraint:stage3:{index:04d}")
    return refs


def _ordered_strings(values: Any) -> list[str]:
    if isinstance(values, (str, int, float)):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = []
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split())
        if text and text.lower() not in {"unknown", "none", "null"} and text not in result:
            result.append(text)
    return result


def _generation_entity_context(knowledge: dict[str, Any]) -> dict[str, Any]:
    stage1_v3 = knowledge.get("stage1_v3")
    stage1_v3 = stage1_v3 if isinstance(stage1_v3, Mapping) else {}
    v3_profile = stage1_v3.get("entity_profile")
    v3_profile = v3_profile if isinstance(v3_profile, Mapping) else {}
    legacy_profile = knowledge.get("entity_profile")
    legacy_profile = legacy_profile if isinstance(legacy_profile, Mapping) else {}
    visual_profile = stage1_v3.get("text_visual_profile")
    if not isinstance(visual_profile, Mapping):
        visual_profile = knowledge.get("text_visual_profile")
    visual_profile = visual_profile if isinstance(visual_profile, Mapping) else {}
    return {
        "primary_type": v3_profile.get("primary_type")
        or legacy_profile.get("entity_type")
        or legacy_profile.get("type")
        or "entity",
        "secondary_types": v3_profile.get("secondary_types")
        or legacy_profile.get("secondary_types", []),
        "semantic_domains": v3_profile.get("semantic_domains")
        or legacy_profile.get("semantic_domain", []),
        "core_identity": v3_profile.get("core_identity")
        or legacy_profile.get("core_identity"),
        "visual_scene_cues": visual_profile.get("visual_scene_cues", []),
        "context_cues": visual_profile.get("context_cues", []),
    }


def _angle_fragment(value: Any, *, max_words: int = 10) -> str:
    words = " ".join(str(value or "").split()).split()
    return " ".join(words[:max_words])


def _generation_angle_specs(
    modality: str,
    item: dict[str, Any],
    knowledge: dict[str, Any],
) -> list[dict[str, str]]:
    context = _generation_entity_context(knowledge)
    primary_type = _angle_fragment(context.get("primary_type")) or "entity"
    identity = _angle_fragment(context.get("core_identity"), max_words=14) or primary_type
    domains = _ordered_strings(context.get("semantic_domains"))
    secondary_types = _ordered_strings(context.get("secondary_types"))
    scene_cues = _ordered_strings(context.get("visual_scene_cues"))
    context_cues = _ordered_strings(context.get("context_cues"))
    relation = _angle_fragment(item.get("relation")) or "global profile"
    source_fact = item.get("source_fact")
    source_fact = source_fact if isinstance(source_fact, Mapping) else {}
    neighbors = _ordered_strings(
        [source_fact.get("object"), *_ordered_strings(source_fact.get("neighbor_entities"))]
    )

    relation_target = ", ".join(neighbors[:2])
    relation_focus = relation if not relation_target else f"{relation} with {relation_target}"
    domain_focus = ", ".join(domains[:2] or secondary_types[:2]) or f"supported {primary_type} context"
    if modality == "image":
        modality_cues = scene_cues[:2] or context_cues[:2]
        modality_focus = ", ".join(modality_cues) or f"inspectable {primary_type} scene"
        attribute_axis = f"{primary_type} visual attributes in {domain_focus}"
        bridge_axis = f"image representation through {modality_focus}"
    else:
        modality_cues = context_cues[:2]
        modality_focus = ", ".join(modality_cues) or f"concise {primary_type} explanation"
        attribute_axis = f"{primary_type} attributes and {domain_focus}"
        bridge_axis = f"text explanation through {modality_focus}"

    return [
        {
            "angle": "identity anchor",
            "diversity_axis": f"core identity of {identity} as a {primary_type}",
        },
        {
            "angle": "evidence-grounded description",
            "diversity_axis": f"graph-supported facts about the {primary_type}",
        },
        {
            "angle": f"relation focus: {relation_focus}",
            "diversity_axis": f"relation-specific evidence for {relation_focus}",
        },
        {
            "angle": f"{primary_type} attribute and context",
            "diversity_axis": attribute_axis,
        },
        {
            "angle": f"{modality} semantic bridge for {primary_type}",
            "diversity_axis": bridge_axis,
        },
    ]


def _generation_brief(
    modality: str,
    item: dict[str, Any],
    angle: str,
    facts: list[str],
    reasoning: dict[str, Any],
    diversity_axis: str | None = None,
) -> dict[str, Any]:
    relation = str(item.get("relation", "global_profile"))
    context = _generation_context_for_item(item, reasoning, modality)
    must_include = _constraint_strings(list(context.get("must_include", [])))
    should_include = _constraint_strings(list(context.get("should_include", [])))
    context_only = _constraint_strings(list(context.get("context_only", [])))
    if not must_include:
        must_include = facts[:2]
    diversity_axis = diversity_axis or relation
    return {
        "generation_intent": (
            f"Create a {modality} candidate centered on {relation} using the {angle} angle."
        ),
        "semantic_focus": relation,
        "diversity_axis": diversity_axis,
        "evidence_policy": "grounded_graph_only",
        "style_guidance": (
            "natural concise description"
            if modality == "text"
            else "clear inspectable composition, concrete but not over-specific"
        ),
        "must_include": must_include,
        "should_include": should_include,
        "context_only": context_only,
        "evidence_facts": list(facts),
    }


def _model_generation_context(
    knowledge: dict[str, Any], item: dict[str, Any], reasoning: dict[str, Any],
    modality: str, angle: str, diversity_axis: str,
    regeneration_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep candidate facts, requirements, identity and visual evidence distinct."""
    context = _generation_context_for_item(item, reasoning, modality)
    by_id = {fact.get("fact_id"): fact for fact in knowledge.get("relation_facts", []) if fact.get("fact_id")}
    source_facts = []
    if item.get("fact_id"):
        for fact_id in dict.fromkeys([item["fact_id"], *item.get("supporting_fact_ids", [])]):
            fact = item.get("source_fact") if fact_id == item["fact_id"] else by_id.get(fact_id)
            if not isinstance(fact, Mapping):
                raise ValueError(f"generation source fact missing: {fact_id}")
            source_facts.append({key: deepcopy(fact[key]) for key in (
                "fact_id", "claim", "subject", "relation", "object", "direction", "evidence_refs",
            ) if key in fact})
    else:
        source_facts = [{"claim": fact} for fact in _facts_for_relation(item, knowledge, reasoning, modality)]

    def requirements(bucket: str) -> list[dict[str, Any]]:
        return [{key: deepcopy(value[key]) for key in (
            "constraint_id", "constraint", "constraint_type", "source_fact_refs", "evidence_refs",
        ) if key in value} for value in context.get(bucket, []) if isinstance(value, Mapping)]

    profile = knowledge.get("stage1_v3", {}).get("entity_profile", knowledge.get("entity_profile", {}))
    identity = {key: deepcopy(profile[key]) for key in (
        "primary_type", "entity_type", "core_identity", "evidence_refs",
    ) if key in profile}
    selected_refs = {ref for fact in source_facts for ref in fact.get("evidence_refs", [])}
    identity_refs = set(identity.get("evidence_refs", []))
    context_refs = {ref for value in requirements("context_only") for ref in value.get("evidence_refs", [])}
    catalog = {value["evidence_id"]: value for value in knowledge.get("evidence_catalog", [])}
    evidence = []
    for ref, value in catalog.items():
        if ref in selected_refs | identity_refs | context_refs:
            evidence.append({
                **{key: deepcopy(value[key]) for key in ("evidence_id", "source", "content") if key in value},
                "usage": "candidate_fact" if ref in selected_refs else "identity_only" if ref in identity_refs else "context_only",
            })
    visual = knowledge.get("stage1_v3", {}).get("text_visual_profile", knowledge.get("text_visual_profile", {}))
    if modality == "image":
        identity["non_visual_facts"] = deepcopy(visual.get("non_visual_facts", []))
    attributes = knowledge.get("stage1_v3", {}).get("key_attributes", [])
    visual_details = [
        {"detail": value["value"], "evidence_refs": [value["source_ref"]]}
        for value in attributes if isinstance(value, Mapping)
        and value.get("type") in {"visual", "visual_scene"} and value.get("value")
        and value.get("source_ref") in selected_refs and value["source_ref"] in catalog
    ]
    feedback = regeneration_feedback or {}
    return {
        "task": {"target_modality": modality, "generation_angle": angle, "diversity_axis": diversity_axis},
        "source_facts": source_facts, "source_evidence": evidence,
        "must_include": requirements("must_include"), "should_include": requirements("should_include"),
        "context_only": requirements("context_only"), "identity_context": identity,
        "supported_visual_details": visual_details,
        "candidate_visual_cues": deepcopy(visual.get("visual_scene_cues", [])),
        "regeneration_feedback": {key: deepcopy(feedback[key]) for key in (
            "repair_actions", "preserve", "hard_violations", "soft_violations",
        ) if key in feedback},
    }


def _brief_instruction(brief: dict[str, Any]) -> str:
    parts = [
        str(brief.get("generation_intent", "")),
        f"Evidence policy: {brief.get('evidence_policy', 'grounded_graph_only')}.",
        f"Composition focus: {brief.get('diversity_axis', '')}.",
    ]
    must = brief.get("must_include") or []
    should = brief.get("should_include") or []
    context = brief.get("context_only") or []
    if must:
        parts.append("Must include: " + "; ".join(map(str, must)))
    if should:
        parts.append("Should include: " + "; ".join(map(str, should)))
    if context:
        parts.append("Context only: " + "; ".join(map(str, context)))
    return " ".join(part for part in parts if part.strip())


def _semantic_prompt_fragment(value: Any, *, max_words: int = 6) -> str:
    """Keep the visual meaning of one constraint while removing control metadata."""
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    prefixes = (
        "entity type:",
        "text visual cue:",
        "visual cue:",
        "text context cue:",
        "context cue:",
    )
    lowered = text.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    words = text.split()
    return " ".join(words[:max_words]).strip(" ,;:")


def _positive_prompt_suffix(brief: dict[str, Any]) -> str:
    """Compile only high-value visual semantics for the SDXL prompt.

    Policy labels, context-only cues, and full evidence sentences remain in the
    structured generation brief and are deliberately excluded from the SDXL
    positive prompt. This keeps the prompt below SDXL CLIP's short context while
    preserving the entity anchor, relation object, and visual scene cues.
    """
    fragments: list[str] = []
    seen: set[str] = set()

    for field_name, limit in (("must_include", 2), ("should_include", 2)):
        values = brief.get(field_name) or []
        if not isinstance(values, (list, tuple)):
            continue
        for value in values[:limit]:
            fragment = _semantic_prompt_fragment(value)
            normalized = fragment.casefold()
            if fragment and normalized not in seen:
                fragments.append(fragment)
                seen.add(normalized)

    if not fragments:
        values = brief.get("evidence_facts") or []
        if isinstance(values, (list, tuple)):
            for value in values[:1]:
                fragment = _semantic_prompt_fragment(value)
                if fragment:
                    fragments.append(fragment)

    suffix = ", ".join(fragments)
    if len(suffix) <= 180:
        return suffix
    return suffix[:180].rsplit(" ", 1)[0].rstrip(" ,;:")


def _constraints_used(item: dict[str, Any], reasoning: dict[str, Any], modality: str) -> list[dict[str, Any]]:
    context = _generation_context_for_item(item, reasoning, modality)
    return _unique_constraints(
        list(context.get("must_include", []))
        + list(context.get("should_include", []))
        + list(context.get("context_only", []))
    )


def _candidate_base(
    candidate_id: str,
    modality: str,
    item: dict[str, Any],
    angle: str,
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
) -> dict[str, Any]:
    used_constraints = _constraints_used(item, reasoning, modality)
    relation = item.get("relation", "global_profile")
    coverage = [] if relation == "global_profile" else [str(relation)]
    source_fact = item.get("source_fact", {})
    supporting_relations = []
    for constraint in _generation_context_for_item(item, reasoning, modality).get("should_include", []):
        supporting_relations.extend(constraint.get("source_relations", []))
    supporting_relations = list(dict.fromkeys(supporting_relations))
    if item.get("fact_id"):
        coverage = list(dict.fromkeys([*coverage, *supporting_relations]))
    evidence_refs = []
    if isinstance(source_fact, Mapping):
        refs = source_fact.get("evidence_refs") or source_fact.get("stage0_fact_ids")
        if isinstance(refs, list):
            evidence_refs = [str(ref) for ref in refs]
    used_evidence = list(item.get("evidence_triples", []))
    if item.get("fact_id"):
        evidence_refs = list(dict.fromkeys(
            ref for constraint in used_constraints for ref in constraint.get("evidence_refs", [])
        ))
        used_evidence = list(dict.fromkeys(
            triple for constraint in used_constraints for triple in constraint.get("evidence_triples", [])
        ))
    return {
        "candidate_id": candidate_id,
        "target_modality": modality,
        "primary_relation": relation,
        "source_plan_id": item.get("plan_id"),
        "generation_angle": angle,
        "supporting_relations": supporting_relations if item.get("fact_id") else [relation],
        "used_evidence_triples": used_evidence,
        "source_fact_refs": list(item.get("source_fact_refs", [])) + list(item.get("supporting_fact_ids", [])),
        "evidence_refs": evidence_refs,
        "used_constraints": used_constraints,
        "constraint_refs": _constraint_refs(used_constraints),
        "prompt_or_instruction": "",
        "positive_prompt": "",
        "generated_content": {},
        "expected_relation_coverage": coverage,
        "expected_relation_coverage_detail": {
            "primary": relation,
            "supporting": supporting_relations if item.get("fact_id") else ([] if relation == "global_profile" else [relation]),
        },
        "metadata": {
            "relation_role": item.get("relation_role"),
            "fallback": bool(item.get("fallback")),
            "entity_type": knowledge.get("entity_profile", {}).get("entity_type", "entity"),
        },
    }


def _provider_candidate_metadata(text_generator: TextGenerator) -> dict[str, Any]:
    metadata_provider = getattr(text_generator, "last_candidate_metadata", None)
    if not callable(metadata_provider):
        return {}
    try:
        metadata = metadata_provider()
    except Exception:
        return {}
    if not isinstance(metadata, Mapping):
        return {}
    return {
        str(key): value
        for key, value in metadata.items()
        if str(key) not in STAGE_OWNED_METADATA_KEYS
    }


def _provider_model_call(
    text_generator: TextGenerator,
    entity_key: str,
    candidate_id: str,
) -> dict[str, Any] | None:
    call_provider = getattr(text_generator, "consume_last_call", None)
    if not callable(call_provider):
        return None
    try:
        call = call_provider(entity_key, candidate_id)
    except Exception:
        return None
    if not isinstance(call, Mapping):
        return None
    return dict(call)


def _generate_modality_candidates(
    modality: str,
    entity_key: str,
    record_label: str,
    entity_type: str,
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    selection: dict[str, Any],
    text_generator: TextGenerator,
    image_prompt_generator: ImagePromptGenerator,
    image_generator: ImageGenerator | None,
    candidate_budget: int,
    regeneration_feedback: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[GeneratedImageArtifact]]:
    sequence = _relation_sequence(selection, knowledge, reasoning, modality, candidate_budget)
    candidates = []
    model_calls = []
    image_artifacts = []
    safe_variant = bool(regeneration_feedback)
    generator = text_generator if modality == "text" else image_prompt_generator
    original_qwen_prompts = getattr(generator, "prompt_version", None) == QWEN_PROMPT_VERSION
    workers = getattr(generator, "max_concurrency", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("Stage 3 max_concurrency must be a positive integer")

    def prepare(entry):
        index, relation_item = entry
        candidate_calls = []
        angle_specs = _generation_angle_specs(modality, relation_item, knowledge)
        angle_spec = angle_specs[index % len(angle_specs)]
        base_angle = angle_spec["angle"]
        angle = f"safe retry {base_angle}" if safe_variant else base_angle
        diversity_axis = angle_spec["diversity_axis"]
        if safe_variant:
            diversity_axis = f"safer regeneration of {diversity_axis}"
        facts = _facts_for_relation(relation_item, knowledge, reasoning, modality)
        if original_qwen_prompts and regeneration_feedback:
            facts = _feedback_facts(regeneration_feedback) + facts
        brief = _generation_brief(
            modality,
            relation_item,
            angle,
            facts,
            reasoning,
            diversity_axis=diversity_axis,
        )
        if original_qwen_prompts:
            for key in ("must_include", "should_include", "context_only", "evidence_facts"):
                brief[key] = brief[key][:4]
        base = _candidate_base(
            (
                f"{modality}_safe_{index + 1:03d}"
                if safe_variant
                else f"{modality}_{index + 1:03d}"
            ),
            modality,
            relation_item,
            angle,
            knowledge,
            reasoning,
        )
        generation_context = _generation_context_for_item(relation_item, reasoning, modality)
        base["generation_context"] = generation_context
        base["generation_brief"] = brief
        if regeneration_feedback:
            if not base["expected_relation_coverage"]:
                base["expected_relation_coverage"] = ["global_profile"]
                base["expected_relation_coverage_detail"] = {
                    "primary": "global_profile",
                    "supporting": ["global_profile"],
                }
            base["metadata"].update(_regeneration_candidate_metadata(regeneration_feedback))
        relation = str(relation_item.get("relation", "global_profile"))
        structured_generate = getattr(generator, "generate_with_context", None)
        if not getattr(generator, "uses_structured_prompts", True):
            structured_generate = None
        if getattr(generator, "prompt_version", None):
            base["metadata"]["prompt_version"] = generator.prompt_version
        model_context = _model_generation_context(
            knowledge, relation_item, reasoning, modality, angle, diversity_axis, regeneration_feedback,
        ) if callable(structured_generate) else {}
        legacy_facts = (
            _feedback_facts(regeneration_feedback) + facts
            if regeneration_feedback and not original_qwen_prompts else facts
        )
        if modality == "text":
            if callable(structured_generate):
                generated = structured_generate(record_label, entity_type, relation, angle, facts, context=model_context)
                base["metadata"]["prompt_version"] = PROMPT_VERSION
            else:
                generated = text_generator.generate(
                    record_label, entity_type, relation, angle, [_brief_instruction(brief), *legacy_facts]
                )
            base["prompt_or_instruction"] = (
                f"Generate grounded text for {record_label} using {relation} evidence. "
                f"{_brief_instruction(brief)}"
            )
            base["generated_content"] = {"type": "text", "value": generated}
            base["metadata"]["status"] = "generated"
            base["metadata"].update(_provider_candidate_metadata(text_generator))
            call = _provider_model_call(text_generator, entity_key, base["candidate_id"])
            if call:
                candidate_calls.append(call)
        else:
            if callable(structured_generate):
                prompt = structured_generate(record_label, entity_type, relation, angle, facts, context=model_context)
                base["metadata"]["prompt_version"] = PROMPT_VERSION
            else:
                prompt = image_prompt_generator.generate(
                    record_label, entity_type, relation, angle, [_brief_instruction(brief), *legacy_facts]
                )
            instruction = _brief_instruction(brief)
            base["prompt_or_instruction"] = " ".join(
                part
                for part in (str(prompt.get("prompt_or_instruction", "")), instruction)
                if part.strip()
            )
            positive = str(prompt.get("positive_prompt", "")).strip()
            brief_suffix = "" if callable(structured_generate) else _positive_prompt_suffix(brief)
            if brief_suffix:
                positive = ", ".join(part for part in (positive, brief_suffix) if part)
            base["positive_prompt"] = positive
            base["generated_content"] = {"type": "image_prompt", "value": base["positive_prompt"]}
            prompt_call = _provider_model_call(
                image_prompt_generator,
                entity_key,
                base["candidate_id"],
            )
            if prompt_call:
                candidate_calls.append(prompt_call)
            base["metadata"]["status"] = "planned"
        return base, candidate_calls

    def prepare_with_audit(entry):
        try:
            return prepare(entry)
        except Exception as exc:
            index, _relation_item = entry
            candidate_id = f"{modality}_{'safe_' if safe_variant else ''}{index + 1:03d}"
            call = _provider_model_call(generator, entity_key, candidate_id)
            if call is not None:
                call.update(stage="stage3_generation", status="failed", modality=modality)
                setattr(exc, "model_call", call)
            setattr(exc, "candidate_id", candidate_id)
            raise

    if workers > 1 and len(sequence) > 1:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(sequence)), thread_name_prefix="mmkgc-stage3",
        ) as executor:
            prepared = list(executor.map(prepare_with_audit, enumerate(sequence)))
    else:
        prepared = [prepare_with_audit(entry) for entry in enumerate(sequence)]

    # Merge in candidate order and keep the shared local GPU pipeline on the caller thread.
    for base, candidate_calls in prepared:
        model_calls.extend(candidate_calls)
        if modality == "image":
            if image_generator is not None:
                artifact = image_generator.generate(
                    entity_key,
                    base["candidate_id"],
                    base["positive_prompt"],
                    "",
                )
                model_calls.append(artifact.model_call)
                if artifact.image_bytes or artifact.metadata is not None:
                    image_artifacts.append(artifact)
                if artifact.image_bytes:
                    artifact_metadata = artifact.metadata
                    base["generated_content"] = {
                        "type": "image",
                        "value": base["positive_prompt"],
                        "image_path": artifact.relative_image_path,
                        "metadata_path": artifact.relative_metadata_path,
                        "prompt": base["positive_prompt"],
                        "width": artifact_metadata.get("width"),
                        "height": artifact_metadata.get("height"),
                        "steps": artifact_metadata.get("steps"),
                        "seed": artifact_metadata.get("seed"),
                    }
                    base["metadata"].update(
                        {
                            "status": "generated",
                            "generator": artifact_metadata.get("generator"),
                            "model_name": artifact_metadata.get("model_name"),
                            "model_path": artifact_metadata.get("model_path"),
                            "fallback_used": False,
                        }
                    )
                else:
                    artifact_metadata = artifact.metadata
                    base["metadata"].update(
                        {
                            "status": "failed",
                            "generator": artifact_metadata.get("generator"),
                            "fallback_used": False,
                            "error_type": artifact_metadata.get("error_type"),
                            "error_message": artifact_metadata.get("error_message"),
                        }
                    )
        candidates.append(base)
    return candidates, model_calls, image_artifacts


def relation_alternation_generation(
    record: Any,
    missing_mode: str,
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    selection: dict[str, Any],
    text_generator: TextGenerator,
    image_prompt_generator: ImagePromptGenerator,
    image_generator: ImageGenerator | None = None,
    candidate_budget: int = 5,
) -> dict[str, Any]:
    """Generate deterministic Stage 3 text and image prompt candidates."""
    entity_type = str(knowledge.get("entity_profile", {}).get("entity_type") or "entity")
    entity_key = str(getattr(record, "entity_key", None) or record.label)
    model_calls = []
    image_artifacts = []
    output = {
        "target_modalities": _target_modalities(missing_mode),
        "text_candidates": [],
        "image_candidates": [],
        "candidate_budget": candidate_budget,
    }
    if missing_mode == "text":
        output["text_candidates"], text_model_calls, _text_image_artifacts = _generate_modality_candidates(
            "text",
            entity_key,
            record.label,
            entity_type,
            knowledge,
            reasoning,
            selection,
            text_generator,
            image_prompt_generator,
            image_generator,
            candidate_budget,
        )
        model_calls.extend(text_model_calls)
    if missing_mode == "image":
        output["image_candidates"], image_model_calls, image_artifacts = _generate_modality_candidates(
            "image",
            entity_key,
            record.label,
            entity_type,
            knowledge,
            reasoning,
            selection,
            text_generator,
            image_prompt_generator,
            image_generator,
            candidate_budget,
        )
        model_calls.extend(image_model_calls)
    return {
        "relation_alternation_generation": output,
        "model_calls": model_calls,
        "generated_image_artifacts": image_artifacts,
    }


def safer_relation_alternation_generation(
    record: Any,
    missing_mode: str,
    knowledge: dict[str, Any],
    reasoning: dict[str, Any],
    selection: dict[str, Any],
    text_generator: TextGenerator,
    image_prompt_generator: ImagePromptGenerator,
    image_generator: ImageGenerator | None = None,
    candidate_budget: int = 5,
    *,
    failed_ranking: Mapping[str, Any],
    attempt: int = 1,
    stage3_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Regenerate Stage 3 candidates using Stage 4 generation feedback."""
    feedback = dict(
        stage3_feedback
        if isinstance(stage3_feedback, Mapping)
        else build_stage3_feedback(failed_ranking, attempt=attempt)
    )
    entity_type = str(knowledge.get("entity_profile", {}).get("entity_type") or "entity")
    entity_key = str(getattr(record, "entity_key", None) or record.label)
    model_calls = []
    image_artifacts = []
    output = {
        "target_modalities": _target_modalities(missing_mode),
        "text_candidates": [],
        "image_candidates": [],
        "candidate_budget": candidate_budget,
        "stage3_regeneration": feedback,
    }
    if missing_mode == "text":
        output["text_candidates"], text_model_calls, _text_image_artifacts = _generate_modality_candidates(
            "text",
            entity_key,
            record.label,
            entity_type,
            knowledge,
            reasoning,
            selection,
            text_generator,
            image_prompt_generator,
            image_generator,
            candidate_budget,
            regeneration_feedback=feedback,
        )
        model_calls.extend(text_model_calls)
    if missing_mode == "image":
        output["image_candidates"], image_model_calls, image_artifacts = _generate_modality_candidates(
            "image",
            entity_key,
            record.label,
            entity_type,
            knowledge,
            reasoning,
            selection,
            text_generator,
            image_prompt_generator,
            image_generator,
            candidate_budget,
            regeneration_feedback=feedback,
        )
        model_calls.extend(image_model_calls)
    return {
        "relation_alternation_generation": output,
        "model_calls": model_calls,
        "generated_image_artifacts": image_artifacts,
    }


def build_stage3_feedback(
    failed_ranking: Mapping[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    candidate_feedback: list[dict[str, Any]] = []
    hard: list[str] = []
    soft: list[str] = []
    for section_name, candidate_key, modality in (
        ("text_ranking", "rejected_candidates", "text"),
        ("image_ranking", "rejected_candidates", "image"),
        ("pair_ranking", "ranked_pairs", "pair"),
    ):
        section = failed_ranking.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        items = section.get(candidate_key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            gate = item.get("hard_gate", {})
            if not isinstance(gate, Mapping):
                continue
            item_hard = _stage3_repairable_codes(gate.get("hard_violations", []))
            item_soft = _stage3_repairable_codes(gate.get("soft_violations", []))
            if not item_hard and not item_soft:
                continue
            hard.extend(item_hard)
            soft.extend(item_soft)
            codes = _unique_strings([*item_hard, *item_soft])
            candidate_feedback.append(
                {
                    "candidate_id": str(
                        item.get("candidate_id")
                        or item.get("pair_id")
                        or "unknown"
                    ),
                    "modality": modality,
                    "hard_violations": item_hard,
                    "soft_violations": item_soft,
                    "evidence": _string_values(gate.get("evidence", [])),
                    "reason": str(
                        item.get("ranking_reason")
                        or gate.get("reason")
                        or "Stage 4 flagged the candidate for revision"
                    ),
                    "repair_actions": _repair_actions(codes),
                    "preserve": list(_STAGE3_PRESERVE),
                }
            )

    hard = _unique_strings(hard)
    soft = _unique_strings(soft)
    codes = _unique_strings([*hard, *soft])
    return {
        "attempt": max(1, int(attempt)),
        "target_modality": _feedback_modality(candidate_feedback),
        "hard_violations": hard,
        "soft_violations": soft,
        "candidate_feedback": candidate_feedback,
        "repair_actions": _repair_actions(codes),
        "preserve": list(_STAGE3_PRESERVE),
    }


def _stage3_repairable_codes(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    return sorted(
        _unique_strings(
            [
                str(code)
                for code in values
                if str(code).strip() and str(code) not in _NON_STAGE3_VIOLATION_CODES
            ]
        )
    )


def _string_values(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    return sorted(_unique_strings([str(value) for value in values if str(value).strip()]))


def _repair_actions(codes: list[str]) -> list[str]:
    return _unique_strings(
        [
            _STAGE3_REPAIR_ACTIONS.get(
                code,
                f"correct the Stage 4 flagged issue: {code}",
            )
            for code in codes
        ]
    )


def _feedback_modality(candidate_feedback: list[dict[str, Any]]) -> str:
    modalities = _unique_strings(
        [str(item.get("modality")) for item in candidate_feedback if item.get("modality")]
    )
    return modalities[0] if len(modalities) == 1 else "mixed"


def _feedback_facts(feedback: Mapping[str, Any]) -> list[str]:
    facts = [
        "Regenerate safer candidates using only graph-supported facts.",
        "Preserve entity identity, supported core attributes, and selected relation roles.",
    ]
    actions = feedback.get("repair_actions", [])
    if isinstance(actions, (list, tuple)) and actions:
        facts.append("Apply these Stage 3 repair actions: " + "; ".join(str(action) for action in actions))
    return facts


def _regeneration_candidate_metadata(feedback: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage3_regeneration": True,
        "regeneration_attempt": int(feedback.get("attempt", 1) or 1),
        "regeneration_hard_violations": list(feedback.get("hard_violations", [])),
        "regeneration_soft_violations": list(feedback.get("soft_violations", [])),
    }


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
