"""Small JSON grammars for the installed local Qwen service."""
from typing import Any


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
    """Constrain syntax/types; factual decisions and evidence remain model validated."""
    if not all(f'"{key}"' in prompt for key in ('entity_profile', 'text_visual_profile', 'relation_facts')):
        return {"type": "object"}
    text = {"type": "string"}
    strings = {"type": "array", "items": text}
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    return obj({
        "relation_facts": {"type": "array", "items": obj({"fact_id": text, "claim": text})},
        "entity_profile": obj({"primary_type": text, "secondary_types": strings,
            "semantic_domains": strings, "core_identity": text, "evidence_refs": strings}),
        "text_visual_profile": obj({"visual_scene_cues": strings, "context_cues": strings,
                                    "non_visual_facts": strings}),
        "key_attributes": {"type": "array", "items": obj({"value": text, "type": text, "source_ref": text})},
        "uncertain_aspects": {"type": "array", "items": obj({"aspect": text, "reason": text})},
    })
