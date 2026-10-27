"""Stage 1 model-built knowledge with immutable Stage 0 evidence bindings."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mmkgc.data import EntityRecord

KNOWLEDGE_SCHEMA = "qwen_grounded_knowledge_v1"


def entity_context(knowledge: dict[str, Any]) -> dict[str, Any]:
    v3 = knowledge.get("stage1_v3", {})
    return {
        "entity_profile": deepcopy(v3.get("entity_profile", {})),
        "text_visual_profile": deepcopy(v3.get("text_visual_profile", {})),
        "uncertain_aspects": deepcopy(v3.get("uncertain_aspects", [])),
    }


def public_modality_context(record: EntityRecord, missing_mode: str) -> dict[str, Any]:
    if missing_mode not in {"text", "image"}:
        raise ValueError("missing_mode must be one of: text, image")
    # Exclude the target modality even for manually constructed records.
    if missing_mode == "image":
        return {"text": record.existing_modalities.get("text")}
    return {
        "has_image": bool(record.existing_modalities.get("image")),
        "image_summary": record.existing_modalities.get("image_summary"),
    }


def build_knowledge_input(
    record: EntityRecord, missing_mode: str, coarse: dict[str, Any]
) -> dict[str, Any]:
    """Serialize selected evidence without type inference or credibility scoring."""
    stage0 = coarse.get("stage0_v3", {})
    relations = {
        (item["relation"], item.get("direction", "out")): item
        for item in stage0.get("relations", [])
    }
    source_facts = {item["fact_id"]: item for item in stage0.get("facts", [])}
    paths = {item["path_id"]: item for item in stage0.get("paths", [])}
    catalog: dict[str, dict[str, Any]] = {}
    candidates = []
    for group_index, group in enumerate(coarse.get("selected_relations", []), 1):
        relation = group["relation"]
        direction = group.get("direction") or (group.get("directions") or ["out"])[0]
        source_group = relations.get((relation, direction), {})
        group_id = source_group.get("relation_id") or f"relation:{group_index:04d}"
        neighbors = list(group.get("neighbor_entities", []))
        triples = list(group.get("evidence_triples", []))
        fact_ids = list(source_group.get("fact_ids", []))
        atomic_sources = [source_facts[ref] for ref in fact_ids if ref in source_facts]
        if not atomic_sources:
            atomic_sources = [
                {
                    "subject": neighbor if direction == "in" else record.label,
                    "object": record.label if direction == "in" else neighbor,
                }
                for neighbor in (neighbors or ["unknown"])
            ]
        for index, source in enumerate(atomic_sources):
            refs: list[str] = []
            source_id = source.get("fact_id")
            triple = source.get("triple") or (triples[index] if index < len(triples) else None)
            if triple:
                ref = source_id or f"evidence:kg:{len(candidates) + 1:04d}"
                catalog[ref] = {"evidence_id": ref, "source": "stage0_kg", "content": triple}
                refs.append(ref)
            for path_id in source_group.get("path_ids", []):
                if path_id in paths:
                    catalog[path_id] = {
                        "evidence_id": path_id, "source": "stage0_path", "content": paths[path_id]
                    }
                    refs.append(path_id)
            candidates.append({
                "fact_id": f"rf:{len(candidates) + 1:04d}",
                "relation_group_id": group_id,
                "subject": source["subject"], "relation": relation,
                "object": source["object"], "direction": direction,
                "neighbor_entities": [source["subject"] if direction == "in" else source["object"]],
                "evidence_refs": refs,
                "stage0_fact_ids": [source_id] if source_id else [],
            })
    existing = public_modality_context(record, missing_mode)
    context_refs = []
    for key in ("text", "image_summary"):
        if existing.get(key):
            ref = f"existing:{key}"
            catalog[ref] = {"evidence_id": ref, "source": "existing_modality", "content": existing[key]}
            context_refs.append(ref)
    for candidate in candidates:
        candidate["evidence_refs"].extend(context_refs)
    return {
        "schema": KNOWLEDGE_SCHEMA,
        "entity": {"entity_key": record.entity_key, "label": record.label},
        "target_modality": missing_mode,
        "existing_modalities": existing,
        "fact_candidates": candidates,
        "evidence_catalog": list(catalog.values()),
    }


def _text_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def apply_knowledge_response(inputs: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Bind model-authored claims to source facts without model-authored evidence."""
    profile = response.get("entity_profile")
    visual = response.get("text_visual_profile")
    if not isinstance(profile, dict) or not isinstance(visual, dict):
        raise ValueError("Stage 1 requires entity_profile and text_visual_profile objects")
    profile = {key: deepcopy(profile[key]) for key in (
        "primary_type", "secondary_types", "semantic_domains", "core_identity", "evidence_refs"
    ) if key in profile}
    for key in ("primary_type", "core_identity"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"entity_profile.{key} must be non-empty text")
    for key in ("secondary_types", "semantic_domains", "evidence_refs"):
        profile[key] = _text_list(profile.get(key, []), f"entity_profile.{key}")
    catalog = {item["evidence_id"]: item for item in inputs["evidence_catalog"]}
    if not set(profile["evidence_refs"]).issubset(catalog):
        raise ValueError("entity_profile contains unknown evidence references")
    visual = {key: _text_list(visual.get(key, []), f"text_visual_profile.{key}") for key in (
        "visual_scene_cues", "context_cues", "non_visual_facts"
    )}
    claims = response.get("relation_facts")
    if not isinstance(claims, list):
        raise ValueError("relation_facts must be a list")
    by_id = {item["fact_id"]: item for item in inputs["fact_candidates"]}
    modeled: dict[str, dict[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("each relation fact must be an object")
        fact_id = claim.get("fact_id")
        if fact_id not in by_id or fact_id in modeled:
            raise ValueError("Stage 1 returned an unknown or duplicate fact_id")
        text = claim.get("claim")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each fact requires a non-empty claim")
        fact = deepcopy(by_id[fact_id])
        bundle = [deepcopy(catalog[ref]) for ref in fact["evidence_refs"]]
        modeled[fact_id] = {
            **fact, "claim": text.strip(), "fact": text.strip(),
            "fact_type": "graph_relation", "evidence_bundle": bundle,
            "evidence_triples": [item["content"] for item in bundle if item["source"] == "stage0_kg"],
            "evidence_match_score": None, "scoring_status": "pending",
            "evidence_basis": "source_match" if len(bundle) == 1 and bundle[0]["source"] == "stage0_kg" else "provided_context",
        }
    if set(modeled) != set(by_id):
        raise ValueError("Stage 1 must model every provided fact_id exactly once")
    attributes = response.get("key_attributes", [])
    uncertainties = response.get("uncertain_aspects", [])
    if not isinstance(attributes, list) or not isinstance(uncertainties, list):
        raise ValueError("key_attributes and uncertain_aspects must be lists")
    for attribute in attributes:
        if not isinstance(attribute, dict) or attribute.get("source_ref") not in catalog:
            raise ValueError("key_attributes require an existing source_ref")
    facts = [modeled[ref] for ref in by_id]
    v3 = {
        "schema_version": KNOWLEDGE_SCHEMA, "entity_profile": profile,
        "text_visual_profile": visual, "key_attributes": deepcopy(attributes),
        "uncertain_aspects": deepcopy(uncertainties),
        "modality_state": {"targets": [inputs["target_modality"]], "existing": inputs["existing_modalities"]},
    }
    return {
        "schema_version": KNOWLEDGE_SCHEMA, "stage1_v3": v3,
        "entity_profile": {
            "entity_type": profile["primary_type"], "type": profile["primary_type"],
            "secondary_types": profile["secondary_types"], "semantic_domain": profile["semantic_domains"],
            "core_identity": profile["core_identity"], "type_evidence": profile["evidence_refs"],
            "uncertain_aspects": deepcopy(uncertainties),
        },
        "text_visual_profile": visual, "relation_facts": facts,
        "evidence_catalog": list(catalog.values()), "existing_modalities": inputs["existing_modalities"],
        "uncertainties": deepcopy(uncertainties),
    }


def knowledge_modeling(
    record: EntityRecord, missing_mode: str, coarse: dict[str, Any], *, planner: Any
) -> dict[str, Any]:
    """Require Qwen to build knowledge; there is no rule-only modeling path."""
    return planner.build_knowledge(record, missing_mode, coarse)
