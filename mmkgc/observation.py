"""Candidate-local factual observations and evidence-bound repair operations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any


OBSERVATION_VERSION = "fact_observation_v1"
REPAIR_CONFIDENCE = 0.8
CHECK_STATUSES = ("supported", "contradicted", "missing", "unverifiable", "uncertain")
VERIFICATION_MODES = ("direct", "appearance_only", "unverifiable")


def compile_verification_targets(context: Mapping[str, Any], knowledge: Mapping[str, Any], modality: str) -> list[dict[str, Any]]:
    """Compile only this candidate's requirements; never infer new graph facts."""
    facts = {fact["fact_id"]: fact for fact in knowledge.get("relation_facts", [])}
    non_visual = knowledge.get("text_visual_profile", {}).get("non_visual_facts", [])
    targets = []
    for bucket in ("must_include", "should_include"):
        for constraint in context.get(bucket, []):
            claim = constraint.get("constraint")
            if not claim:
                continue
            refs = list(constraint.get("source_fact_refs", []))
            fact = facts.get(refs[0], {}) if refs else {}
            targets.append({
                "check_id": constraint["constraint_id"],
                "fact_id": refs[0] if refs else None,
                "claim": claim, "requirement": bucket,
                "subject": fact.get("subject", ""), "relation": fact.get("relation", ""),
                "object": fact.get("object", ""), "direction": fact.get("direction", ""),
                "verification_mode": ("direct" if modality == "text" else
                                      "unverifiable" if claim in non_visual else "assess"),
                "evidence_refs": list(constraint.get("evidence_refs", [])),
            })
    return validate_verification_targets(targets)


def validate_verification_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("verification_targets must be an array")
    result, seen = [], set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("verification target must be an object")
        check_id = _text(item.get("check_id"), "check_id")
        if check_id in seen:
            raise ValueError("duplicate verification check_id")
        seen.add(check_id)
        fact_id = item.get("fact_id")
        if fact_id is not None:
            fact_id = _text(fact_id, "fact_id")
        requirement = item.get("requirement")
        mode = item.get("verification_mode")
        if requirement not in {"must_include", "should_include"}:
            raise ValueError("invalid verification requirement")
        if mode not in {*VERIFICATION_MODES, "assess"}:
            raise ValueError("invalid verification mode")
        target = {
            "check_id": check_id, "fact_id": fact_id,
            "claim": _text(item.get("claim"), "claim"), "requirement": requirement,
            "verification_mode": mode, "evidence_refs": _strings(item.get("evidence_refs"), "evidence_refs"),
        }
        for key in ("subject", "relation", "object", "direction"):
            target[key] = _text(item.get(key, ""), key, allow_empty=True)
        result.append(target)
    return result


def observation_response_schema() -> dict[str, Any]:
    return {
        "fact_checks": [{
            "check_id": "one supplied check_id; return each exactly once",
            "status": "|".join(CHECK_STATUSES),
            "verification_mode": "|".join(VERIFICATION_MODES),
            "observed": "actual candidate content, not the expected fact",
            "evidence": ["text: exact span or absence explanation; pixel: visible observation"],
            "confidence": "number from 0 to 1", "reason": "concise explanation",
        }],
        "unsupported_claims": [{
            "claim": "specific candidate assertion unsupported by ALL supplied evidence",
            "evidence": ["text: actual span; pixel: actual visible detail"],
            "confidence": "number from 0 to 1", "reason": "why evidence does not support it",
        }],
    }


def bind_observation(raw: Any, targets: Any, modality: str, candidate_id: str) -> dict[str, Any]:
    """Validate model observations and attach trusted requirements and provenance."""
    targets = validate_verification_targets(targets)
    by_id = {item["check_id"]: item for item in targets}
    if not isinstance(raw, Mapping) or set(raw) != {"fact_checks", "unsupported_claims"}:
        raise ValueError("observation requires fact_checks and unsupported_claims")
    if not isinstance(raw["fact_checks"], list) or not isinstance(raw["unsupported_claims"], list):
        raise ValueError("observation entries must be arrays")
    checks, seen = [], set()
    for item in raw["fact_checks"]:
        fields = {"check_id", "status", "verification_mode", "observed", "evidence", "confidence", "reason"}
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError("invalid fact check fields")
        check_id = item["check_id"]
        if not isinstance(check_id, str) or check_id not in by_id or check_id in seen:
            raise ValueError("unknown or duplicate observation check_id")
        seen.add(check_id)
        target = by_id[check_id]
        status, mode = item["status"], item["verification_mode"]
        if status not in CHECK_STATUSES or mode not in VERIFICATION_MODES:
            raise ValueError("invalid observation status or verification mode")
        if target["verification_mode"] == "unverifiable" and mode != "unverifiable":
            raise ValueError("unverifiable target cannot be verified from candidate content")
        if (mode == "unverifiable" and status != "unverifiable") or (
            status == "unverifiable" and mode == "direct"
        ) or (mode == "appearance_only" and status not in {"unverifiable", "uncertain"}):
            raise ValueError("appearance or invisible evidence cannot verify a full fact")
        evidence = _strings(item["evidence"], "evidence")
        if status in {"supported", "contradicted", "missing"}:
            prefix = "pixel:" if modality == "image" else "text:"
            if not any(text.startswith(prefix) and text[len(prefix):].strip() for text in evidence):
                raise ValueError(f"fact check requires {prefix} content evidence")
        checks.append({
            **deepcopy(target), "expected": target["claim"], "status": status,
            "verification_mode": mode, "observed": _text(item["observed"], "observed", allow_empty=True),
            "evidence": evidence, "confidence": _confidence(item["confidence"]),
            "reason": _text(item["reason"], "reason"),
        })
    if seen != set(by_id):
        raise ValueError("observation must check every supplied check_id exactly once")
    unsupported = []
    for index, item in enumerate(raw["unsupported_claims"], 1):
        if not isinstance(item, Mapping) or set(item) != {"claim", "evidence", "confidence", "reason"}:
            raise ValueError("invalid unsupported claim fields")
        evidence = _strings(item["evidence"], "evidence")
        prefix = "pixel:" if modality == "image" else "text:"
        if not any(text.startswith(prefix) and text[len(prefix):].strip() for text in evidence):
            raise ValueError("unsupported claim requires actual candidate evidence")
        unsupported.append({
            "claim_id": f"claim:{index:04d}", "claim": _text(item["claim"], "claim"),
            "evidence": evidence, "confidence": _confidence(item["confidence"]),
            "reason": _text(item["reason"], "reason"),
        })
    observation = {
        "schema_version": OBSERVATION_VERSION, "candidate_id": candidate_id, "modality": modality,
        "fact_checks": checks, "unsupported_claims": unsupported,
        "violated_facts": [item for item in checks if item["status"] == "contradicted"],
        "missing_facts": [item for item in checks if item["status"] == "missing"],
        "unverifiable_facts": [item for item in checks if item["status"] == "unverifiable"],
    }
    observation["repair_needed"] = bool(repair_operations(observation))
    return observation


def repair_operations(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only actionable observations with supplied evidence become edits."""
    operations = []
    for item in observation.get("fact_checks", []):
        if item["confidence"] < REPAIR_CONFIDENCE or item["verification_mode"] != "direct" or not item["evidence_refs"]:
            continue
        status = item["status"]
        if status != "contradicted" and not (status == "missing" and item["requirement"] == "must_include"):
            continue
        operations.append({
            "action": "replace" if status == "contradicted" else "add",
            "check_id": item["check_id"], "fact_id": item["fact_id"],
            "expected": item["expected"], "observed": item["observed"],
            "evidence_refs": list(item["evidence_refs"]), "evidence": list(item["evidence"]),
        })
    for item in observation.get("unsupported_claims", []):
        if item["confidence"] >= REPAIR_CONFIDENCE:
            operations.append({"action": "remove", "claim_id": item["claim_id"],
                               "observed": item["claim"], "evidence": list(item["evidence"])})
    return operations


def observation_priority(candidate: Mapping[str, Any]) -> tuple[int, int, int]:
    """Prefer accepted candidates and fewer actionable defects before quality scores."""
    operations = repair_operations(candidate.get("observation") or {})
    return (
        int(candidate.get("hard_gate", {}).get("decision") != "pass"),
        sum(item["action"] in {"replace", "remove"} for item in operations),
        sum(item["action"] == "add" for item in operations),
    )


def repair_is_eligible(original: Mapping[str, Any], repaired: Mapping[str, Any]) -> bool:
    """A repair must resolve an observed defect without sacrificing verified facts."""
    if repaired.get("hard_gate", {}).get("decision") != "pass":
        return False
    before = original.get("observation") or {}
    operations = repair_operations(before)
    if not operations:
        return True
    after = repaired.get("observation") or {}
    checked = {item["check_id"]: item for item in after.get("fact_checks", [])}

    def verified(item):
        return item.get("status") == "supported" and item.get("verification_mode") == "direct" and item.get("confidence", 0) >= REPAIR_CONFIDENCE

    for item in before.get("fact_checks", []):
        if verified(item) and not verified(checked.get(item["check_id"], {})):
            return False
    old_conflicts = {item["check_id"] for item in operations if item["action"] == "replace"}
    old_unsupported = {item["observed"] for item in operations if item["action"] == "remove"}
    for item in repair_operations(after):
        if item["action"] == "replace" and item["check_id"] not in old_conflicts:
            return False
        if item["action"] == "remove" and item["observed"] not in old_unsupported:
            return False
    remaining_claims = {item["claim"] for item in after.get("unsupported_claims", [])}
    return any(
        (item["observed"] not in remaining_claims if item["action"] == "remove" else
         verified(checked.get(item["check_id"], {})))
        for item in operations
    ) if after else False


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'non-empty ' if not allow_empty else ''}string")
    return value.strip()


def _strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return [_text(item, name) for item in value]


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be a finite number from 0 to 1")
    return float(value)
