"""Four-dimensional judge contract and deterministic score adaptation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


RUBRIC_VERSION = "entity_modality_rubrics_v1"
AGGREGATION_VERSION = "semantic_mean_with_evidence_caps_v1"
DIMENSIONS = {
    "text": ("entity_semantics", "verified_fact_consistency", "semantic_coverage", "textual_usability"),
    "image": ("entity_semantics", "verified_fact_consistency", "visual_semantic_coverage", "visual_usability"),
}
RESPONSE_FIELDS = {"scores", "observation", "unverified_additions", "repair_suggestions"}


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def response_schema(modality: str, observation_schema: dict[str, Any]) -> dict[str, Any]:
    text = {"type": "string", "minLength": 1}
    strings = {"type": "array", "items": text}
    # LM Format Enforcer cannot parse a mixed integer/null enum. Keep the same
    # legal values via separate branches, rather than allowing arbitrary scores.
    dimension = _object({"score": {"anyOf": [{"type": "integer", "enum": [0, 1, 2, 3, 4]}, {"type": "null"}]},
                         "reason": text, "evidence": strings})
    dimensions = {key: deepcopy(dimension) for key in DIMENSIONS[modality]}
    for key, value in dimensions.items():
        if key != "verified_fact_consistency":
            value["properties"]["evidence"]["minItems"] = 1
    # Prefix and null-dependent evidence rules remain in the strict parser.
    # Keep the wire schema simple for local structured-decoding backends.
    return _object({
        "scores": _object(dimensions),
        "observation": deepcopy(observation_schema),
        "unverified_additions": {"type": "array", "items": _object({
            "claim": text, "evidence": strings, "reason": text})},
        "repair_suggestions": {"type": "array", "items": _object({
            "issue": text, "evidence_refs": strings, "suggestion": text})},
    })


def instructions(modality: str) -> str:
    path = Path(__file__).with_name("prompts") / f"{modality}_judge_rubrics_v1.txt"
    return path.read_text(encoding="utf-8")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _strings(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return [_text(item, field) for item in value]


def _candidate_evidence(value, modality, field):
    values = _strings(value, field)
    prefix = "pixel:" if modality == "image" else "text:"
    if not any(item.startswith(prefix) and item[len(prefix):].strip() for item in values):
        raise ValueError(f"{field} needs actual candidate evidence beginning {prefix}")
    return values


def parse_payload(payload: Any, modality: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != RESPONSE_FIELDS:
        raise ValueError("rubric judgment requires exactly scores, observation, unverified_additions, repair_suggestions")
    if not isinstance(payload["scores"], Mapping) or set(payload["scores"]) != set(DIMENSIONS[modality]):
        raise ValueError("rubric scores must contain exactly " + ", ".join(DIMENSIONS[modality]))
    scores = {}
    for name in DIMENSIONS[modality]:
        item = payload["scores"][name]
        if not isinstance(item, Mapping) or set(item) != {"score", "reason", "evidence"}:
            raise ValueError(f"{name} requires score, reason, evidence")
        value = item["score"]
        if value is not None and (type(value) is not int or not 0 <= value <= 4):
            raise ValueError(f"{name}.score must be an integer 0-4 or null")
        if modality == "text" and name != "verified_fact_consistency" and value is None:
            raise ValueError(f"{name} must be scored for a text candidate")
        evidence = (_candidate_evidence(item["evidence"], modality, name + ".evidence") if value is not None
                    else _strings(item["evidence"], name + ".evidence"))
        scores[name] = {"score": value, "reason": _text(item["reason"], name + ".reason"), "evidence": evidence}
    if any(scores[name]["score"] is None for name in DIMENSIONS[modality] if name != "verified_fact_consistency"):
        raise ValueError("image visual assessment unavailable; do not replace null visual dimensions with invented scores")
    observation = payload["observation"]
    if (not isinstance(observation, Mapping) or set(observation) not in (
            {"fact_checks", "unsupported_claims"}, {"fact_checks", "unsupported_claims", "repair_checks"})
            or not isinstance(observation["fact_checks"], list) or not isinstance(observation["unsupported_claims"], list)):
        raise ValueError("rubric observation requires fact_checks and unsupported_claims arrays")
    unverified = []
    if not isinstance(payload["unverified_additions"], list):
        raise ValueError("unverified_additions must be an array")
    for item in payload["unverified_additions"]:
        if not isinstance(item, Mapping) or set(item) != {"claim", "evidence", "reason"}:
            raise ValueError("unverified addition requires claim, evidence, reason")
        unverified.append({"claim": _text(item["claim"], "claim"),
            "evidence": _candidate_evidence(item["evidence"], modality, "unverified addition evidence"),
            "reason": _text(item["reason"], "reason")})
    suggestions = []
    if not isinstance(payload["repair_suggestions"], list):
        raise ValueError("repair_suggestions must be an array")
    for item in payload["repair_suggestions"]:
        if not isinstance(item, Mapping) or set(item) != {"issue", "evidence_refs", "suggestion"}:
            raise ValueError("repair suggestion requires issue, evidence_refs, suggestion")
        suggestions.append({"issue": _text(item["issue"], "issue"),
            "evidence_refs": _strings(item["evidence_refs"], "evidence_refs"),
            "suggestion": _text(item["suggestion"], "suggestion")})
    values = [scores[name]["score"] for name in DIMENSIONS[modality] if name != "verified_fact_consistency"]
    semantic_score = sum(values) / (4 * len(values))
    caps = {"entity_semantics": scores["entity_semantics"]["score"] / 4}
    fact_score = scores["verified_fact_consistency"]["score"]
    if fact_score is not None:
        caps["verified_fact_consistency"] = fact_score / 4
    normalized = min(semantic_score, *caps.values())
    return {"rubric_version": RUBRIC_VERSION, "rubric_scores": scores,
        "rubric_aggregation": {"version": AGGREGATION_VERSION, "semantic_mean": semantic_score,
            "caps": caps, "normalized_score": normalized, "calibration_status": "uncalibrated_baseline",
            "fact_assessment": "unavailable" if fact_score is None else "available"},
        "unverified_additions": unverified, "repair_suggestions": suggestions}


def validate_bound_observation(details, observation, targets):
    """Keep uncertain additions outside the unchanged executable repair contract."""
    direct = [item for item in observation["fact_checks"]
              if item["verification_mode"] == "direct" and item["status"] in {"supported", "contradicted"}]
    removals = observation["unsupported_claims"]
    for check in observation.get("repair_checks", []):
        if check["status"] == "resolved" and any(item["claim"].casefold() == check["claim"].casefold() for item in removals):
            raise ValueError("a resolved repair issue must not remain a confirmed conflicting claim")
    known_refs = {ref for item in targets for ref in item["evidence_refs"]}
    unresolved = {item["claim"].casefold() for item in details["unverified_additions"]}
    for item in removals:
        source_refs = {value[len("context:"):].strip() for value in item["evidence"] if value.startswith("context:")}
        if not source_refs & known_refs:
            raise ValueError("a confirmed remove claim needs context:<supplied evidence ID>, not KG silence")
        if item["claim"].casefold() in unresolved:
            raise ValueError("an unverified addition must not also enter unsupported_claims removal")
    for suggestion in details["repair_suggestions"]:
        if not set(suggestion["evidence_refs"]) <= known_refs:
            raise ValueError("repair suggestion references evidence outside the supplied checks")
    fact_score = details["rubric_scores"]["verified_fact_consistency"]["score"]
    unresolved_repairs = any(item["status"] == "unresolved" and item.get('issue_kind', 'fact') == 'fact'
                             for item in observation.get("repair_checks", []))
    assessable = bool(direct or removals or unresolved_repairs)
    conflict = any(item["status"] == "contradicted" for item in direct) or bool(removals) or unresolved_repairs
    if (fact_score is None) == assessable:
        raise ValueError("fact score must be null exactly when no assertions are verifiable from these checks")
    if fact_score == 4 and conflict:
        raise ValueError("fact score 4 contradicts a confirmed factual conflict")
    if fact_score is not None and fact_score < 4 and not conflict:
        raise ValueError("missing, uncertain and unverified content must not be scored as verified factual errors")
