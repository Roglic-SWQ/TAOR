"""Candidate-local factual observations and evidence-bound repair operations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any


OBSERVATION_VERSION = "fact_observation_v1"
REPAIR_CONFIDENCE = 0.8
REFLECTION_POLICY = "targeted_reflection_v2"
CHECK_STATUSES = ("supported", "contradicted", "missing", "unverifiable", "uncertain")
VERIFICATION_MODES = ("direct", "appearance_only", "unverifiable")


def validate_repair_targets(value: Any) -> list[dict[str, Any]]:
    """Carry original rejected assertions into a repaired candidate's review."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("repair_targets must be an array")
    result, seen = [], set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"issue_id", "claim", "evidence_refs"}:
            raise ValueError("repair target requires issue_id, claim, evidence_refs")
        issue_id = _text(item["issue_id"], "issue_id")
        if issue_id in seen:
            raise ValueError("duplicate repair issue_id")
        seen.add(issue_id)
        result.append({"issue_id": issue_id, "claim": _text(item["claim"], "claim"),
                       "evidence_refs": _strings(item["evidence_refs"], "evidence_refs")})
    return result


def removal_targets(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return validate_repair_targets([{"issue_id": item["claim_id"], "claim": item["observed"],
        "evidence_refs": item.get("evidence_refs") or [entry[len("context:"):].strip() for entry in item["evidence"]
                          if entry.startswith("context:")]} for item in operations if item["action"] == "remove"])


def bind_repair_checks(raw: Any, targets: Any, modality: str) -> list[dict[str, Any]]:
    targets = validate_repair_targets(targets)
    if not isinstance(raw, list):
        raise ValueError("repair_checks must be an array")
    by_id = {item["issue_id"]: item for item in targets}
    result, seen = [], set()
    prefix = "pixel:" if modality == "image" else "text:"
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"issue_id", "status", "observed", "evidence", "confidence", "reason"}:
            raise ValueError("invalid repair check fields")
        issue_id = item["issue_id"]
        if not isinstance(issue_id, str) or issue_id not in by_id or issue_id in seen:
            raise ValueError("unknown or duplicate repair issue_id")
        seen.add(issue_id)
        if item["status"] not in {"resolved", "unresolved", "uncertain"}:
            raise ValueError("invalid repair check status")
        evidence = _strings(item["evidence"], "evidence")
        if not any(text.startswith(prefix) and text[len(prefix):].strip() for text in evidence):
            raise ValueError("repair check requires actual candidate evidence")
        result.append({**deepcopy(by_id[issue_id]), "status": item["status"],
            "observed": _text(item["observed"], "observed"), "evidence": evidence,
            "confidence": _confidence(item["confidence"]), "reason": _text(item["reason"], "reason")})
    if seen != set(by_id):
        raise ValueError("repair checks must cover every supplied issue_id exactly once")
    return result


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


def bind_observation(raw: Any, targets: Any, modality: str, candidate_id: str, *, repair_targets=None) -> dict[str, Any]:
    """Validate model observations and attach trusted requirements and provenance."""
    targets = validate_verification_targets(targets)
    by_id = {item["check_id"]: item for item in targets}
    repair_targets = validate_repair_targets(repair_targets or [])
    fields = {"fact_checks", "unsupported_claims"} | ({"repair_checks"} if repair_targets else set())
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("observation requires exactly " + ", ".join(sorted(fields)))
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
    if repair_targets:
        observation["repair_checks"] = bind_repair_checks(raw["repair_checks"], repair_targets, modality)
    observation["repair_needed"] = bool(repair_operations(observation))
    return observation


def repair_operations(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only actionable observations with supplied evidence become edits."""
    operations = []
    for item in observation.get("fact_checks", []):
        if item["confidence"] < REPAIR_CONFIDENCE or item["verification_mode"] != "direct" or not item["evidence_refs"]:
            continue
        status = item["status"]
        if status != "contradicted" and not (status == "missing" and item["requirement"] == "must_include"
                                            and observation.get("modality") == "text"):
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
    reported = {item["observed"].casefold() for item in operations if item["action"] == "remove"}
    for item in observation.get("repair_checks", []):
        if item["status"] == "unresolved" and item["confidence"] >= REPAIR_CONFIDENCE and item["claim"].casefold() not in reported:
            operations.append({"action": "remove", "claim_id": item["issue_id"], "observed": item["claim"],
                               "evidence": list(item["evidence"]), "evidence_refs": list(item["evidence_refs"])})
            reported.add(item["claim"].casefold())
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
        return False
    after = repaired.get("observation") or {}
    checked = {item["check_id"]: item for item in after.get("fact_checks", [])}
    if set(checked) != {item["check_id"] for item in before.get("fact_checks", [])}:
        return False
    for item in before.get("fact_checks", []):
        current = checked[item["check_id"]]
        if any(item.get(key) != current.get(key) for key in
               ("expected", "fact_id", "subject", "relation", "object", "direction", "requirement")):
            return False
        if set(item.get("evidence_refs", [])) != set(current.get("evidence_refs", [])):
            return False

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
    # A vanished warning is not proof that its original assertion was removed.
    completion = repair_outcomes(original, repaired)
    if any(item["status"] == "uncertain" for item in completion):
        return False
    before_scores = original.get("judge_rubrics", {}).get("rubric_scores", {})
    after_scores = repaired.get("judge_rubrics", {}).get("rubric_scores", {})
    for key in ("textual_usability", "visual_usability"):
        old, new = before_scores.get(key, {}).get("score"), after_scores.get(key, {}).get("score")
        if old is not None and new is not None and old >= 2 and new <= 1:
            return False
    return any(item["status"] == "resolved" for item in completion) if after else False


def repair_outcomes(original: Mapping[str, Any], repaired: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve the same original issues; uncertainty never counts as success."""
    after = repaired.get("observation") or {}
    facts = {item["check_id"]: item for item in after.get("fact_checks", [])}
    removals = {item["issue_id"]: item for item in after.get("repair_checks", [])}
    outcomes = []
    for operation in repair_operations(original.get("observation") or {}):
        remove = operation["action"] == "remove"
        issue_id = operation["claim_id"] if remove else operation["check_id"]
        check = (removals if remove else facts).get(issue_id, {})
        state = "uncertain"
        if check.get("confidence", 0) >= REPAIR_CONFIDENCE:
            if remove and check.get("claim") == operation["observed"]:
                state = check.get("status", "uncertain")
            elif not remove and check.get("verification_mode") == "direct":
                state = ("resolved" if check.get("status") == "supported" else
                         "unresolved" if check.get("status") in {"contradicted", "missing"} else "uncertain")
        outcomes.append({"issue_id": issue_id, "action": operation["action"], "status": state})
    return outcomes


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
