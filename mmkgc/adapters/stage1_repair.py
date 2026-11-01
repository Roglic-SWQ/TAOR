"""Model-authored Stage 1 patches with immutable successful fields."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from mmkgc.adapters.json_contract import local_semantic_json_schema
from mmkgc.stages.modeling import KnowledgeResponseValidationError

TASK_KIND = "stage1_targeted_repair_v1"


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


@dataclass(frozen=True)
class KnowledgeRepairTask:
    fields: tuple[str, ...]
    facts: tuple[str, ...]
    schema: dict[str, Any]
    prompt: str


def targeted_knowledge_repair(inputs, response, errors) -> KnowledgeRepairTask | None:
    """Request only unambiguous invalid fields and missing/empty claims."""
    if not isinstance(response, dict) or not errors:
        return None
    expected = {f["fact_id"]: f for f in inputs["fact_candidates"]}
    claims = response.get("relation_facts")
    if not isinstance(claims, list):
        return None
    ids = [c.get("fact_id") if isinstance(c, dict) else None for c in claims]
    if any(not isinstance(fid, str) or fid not in expected for fid in ids) or len(set(ids)) != len(ids):
        return None
    fields, facts = set(), set()
    for error in errors:
        code, path = error["code"], error["path"]
        if code == "missing_fact_ids":
            facts.update(error["missing_fact_ids"])
        elif code == "empty_claim" and error.get("fact_id") in expected:
            facts.add(error["fact_id"])
        elif code in {"invalid_profile", "invalid_visual_profile", "empty_profile_text",
                      "invalid_string_list", "unknown_profile_evidence", "invalid_attributes",
                      "invalid_uncertainties"}:
            fields.add(path)
        elif code == "unknown_attribute_evidence":
            index = int(path.split("/")[2])
            attribute = response["key_attributes"][index]
            fields.add(path if isinstance(attribute, dict) else path.rsplit("/", 1)[0])
        else:
            return None
    fields = tuple(sorted(p for p in fields if not any(p.startswith(parent + "/") for parent in fields)))
    facts = tuple(fid for fid in expected if fid in facts)
    full_schema = local_semantic_json_schema(
        '"entity_profile" "text_visual_profile" "relation_facts"\nINPUT_JSON:\n' + json.dumps(inputs))
    field_schemas = {}
    for path in fields:
        node = full_schema
        for part in path.strip("/").split("/"):
            if node.get("type") == "array" and part.isdigit():
                node = node["items"]
            else:
                node = node.get("properties", {}).get(part)
            if not isinstance(node, dict):
                return None
        field_schemas[path] = deepcopy(node)
    schema = _object({"fields": _object(field_schemas), "facts": _object({
        fid: {"type": "string", "minLength": 1, "maxLength": 800} for fid in facts})})
    # The source input contains public modalities and immutable Stage 0 evidence.
    # No invalid/repetitive model response is included in the request.
    payload = {"repair_kind": TASK_KIND, "source_input": inputs,
               "validation_errors": errors, "fields_to_replace": list(fields),
               "facts_to_replace": [expected[fid] for fid in facts],
               "response_schema": schema,
               "response_template": {"fields": {p: None for p in fields}, "facts": {fid: "" for fid in facts}}}
    prompt = (
        "Repair ONLY the requested Stage 1 fields and claims using source_input as evidence. "
        "Return exactly the fields/facts objects in response_template. Field keys are JSON pointers; "
        "fill their values using response_schema. Fact keys are fixed source IDs; fill one concise, "
        "nonempty claim for each requested fact, preserving its subject/relation/object/direction. "
        "Choose evidence references ONLY from source_input.evidence_catalog. Other successful fields "
        "and claims are frozen; do not return them. Do not invent facts or delete valid attributes. "
        "Return JSON only, without Markdown.\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)
    )
    return KnowledgeRepairTask(fields, facts, schema, prompt)


def merge_knowledge_repair(base, task: KnowledgeRepairTask, raw: str):
    def fail(code, message):
        raise KnowledgeResponseValidationError([{"code": code, "path": "/targeted_repair", "message": message}])

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("duplicate_repair_key", "Each patch key must occur exactly once")
            result[key] = value
        return result

    try:
        patch = json.loads(raw, object_pairs_hook=unique)
    except (json.JSONDecodeError, TypeError) as exc:
        fail("invalid_repair_json", str(exc))
    if not isinstance(patch, dict) or set(patch) != {"fields", "facts"}:
        fail("invalid_repair_shape", "Return exactly fields and facts objects")
    for name, keys in (("fields", task.fields), ("facts", task.facts)):
        if not isinstance(patch[name], dict) or set(patch[name]) != set(keys):
            fail("repair_key_mismatch", f"{name} must contain exactly {list(keys)}")
    result = deepcopy(base)
    for path, value in patch["fields"].items():
        parts = path.strip("/").split("/")
        target = result
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        key = int(parts[-1]) if isinstance(target, list) else parts[-1]
        target[key] = deepcopy(value)
    claims = {item["fact_id"]: item for item in result["relation_facts"]}
    for fid, claim in patch["facts"].items():
        if not isinstance(claim, str) or not claim.strip():
            fail("invalid_repair_claim", "Every repaired claim must be nonempty text")
        if fid in claims:
            claims[fid]["claim"] = claim
        else:
            result["relation_facts"].append({"fact_id": fid, "claim": claim})
    # The caller performs the unchanged complete source/evidence validation.
    return result
