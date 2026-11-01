"""Small JSON grammars for the installed local Qwen service."""
from __future__ import annotations

import json
from typing import Any


def _payload_from_prompt(prompt: str) -> dict[str, Any] | None:
    marker = "INPUT_JSON:\n"
    index = prompt.find(marker)
    text = prompt[index + len(marker):] if index >= 0 else prompt
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _find_evidence_ids(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        if isinstance(value.get("evidence_catalog"), list):
            return [item["evidence_id"] for item in value["evidence_catalog"]
                    if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)]
        for child in value.values():
            found = _find_evidence_ids(child)
            if found is not None:
                return found
    return None


def semantic_output_token_cap(prompt: str) -> int | None:
    """Limit only targeted field/claim patches; full replies use model config."""
    payload = _payload_from_prompt(prompt)
    if isinstance(payload, dict) and payload.get("repair_kind") == "stage1_targeted_repair_v1":
        return min(2048, max(768, 256 + 128 * len(payload["facts_to_replace"])
                            + 192 * len(payload["fields_to_replace"])))
    return None


def local_relation_json_schema() -> dict[str, Any]:
    """Small structural repair grammar; role/evidence/cardinality stay locally checked."""
    text = {"type": "string"}
    strings = {"type": "array", "items": text}
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    return obj({"plan": obj({
        "status": {"type": "string", "enum": ["ready", "insufficient_evidence"]},
        "candidate_anchors": strings,
        "relation_plans": {"type": "array", "items": obj({
            "fact_id": text, "role": {"type": "string", "enum": ["primary", "secondary", "context", "excluded"]},
            "evidence_refs": strings, "supporting_fact_ids": strings,
            "reason": {"type": "string", "minLength": 1, "maxLength": 160},
        })},
    })})


def local_semantic_json_schema(prompt: str) -> dict[str, Any]:
    """Constrain syntax/types and preserve the targeted-repair evidence contract."""
    payload = _payload_from_prompt(prompt)
    if isinstance(payload, dict) and payload.get("repair_kind") == "stage1_targeted_repair_v1":
        return payload["response_schema"]
    if not all(f'"{key}"' in prompt for key in ('entity_profile', 'text_visual_profile', 'relation_facts')):
        return {"type": "object"}
    text = {"type": "string"}
    strings = {"type": "array", "items": text}
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    evidence_ids = _find_evidence_ids(payload)
    evidence_ref = {"type": "string", **({"enum": evidence_ids} if evidence_ids else {})}
    refs = {"type": "array", "items": evidence_ref}
    attributes = {"type": "array", "items": obj({
        "value": text, "type": text, "source_ref": evidence_ref})}
    if evidence_ids == []:
        refs["maxItems"] = 0
        attributes["maxItems"] = 0
    return obj({
        "relation_facts": {"type": "array", "items": obj({
            "fact_id": text, "claim": {"type": "string", "minLength": 1}})},
        "entity_profile": obj({"primary_type": text, "secondary_types": strings,
            "semantic_domains": strings, "core_identity": text, "evidence_refs": refs}),
        "text_visual_profile": obj({"visual_scene_cues": strings, "context_cues": strings,
                                    "non_visual_facts": strings}),
        "key_attributes": attributes,
        "uncertain_aspects": {"type": "array", "items": obj({"aspect": text, "reason": text})},
    })
