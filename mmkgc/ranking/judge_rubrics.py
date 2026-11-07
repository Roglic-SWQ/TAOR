"""Four-dimensional judge contract and deterministic score adaptation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from mmkgc.observation import repair_operations


RUBRIC_VERSION = "entity_modality_rubrics_v1"
AGGREGATION_VERSION = "semantic_mean_with_evidence_caps_v1"
DIMENSIONS = {
    "text": ("entity_semantics", "verified_fact_consistency", "semantic_coverage", "textual_usability"),
    "image": ("entity_semantics", "verified_fact_consistency", "visual_semantic_coverage", "visual_usability"),
}
REQUIRED_RESPONSE_FIELDS = frozenset({'scores'})


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def response_schema(modality: str, observation_schema: dict[str, Any]) -> dict[str, Any]:
    text = {"type": "string"}
    strings = {"type": "array", "items": text}
    # LM Format Enforcer cannot parse a mixed integer/null enum. Keep the same
    # legal values via separate branches, rather than allowing arbitrary scores.
    dimension = _object({"score": {"anyOf": [{"type": "integer", "enum": [0, 1, 2, 3, 4]}, {"type": "null"}]},
                         "reason": text, "evidence": strings})
    dimension['required'] = ['score']
    dimensions = {key: deepcopy(dimension) for key in DIMENSIONS[modality]}
    schema = _object({
        "scores": _object(dimensions),
        "observation": deepcopy(observation_schema),
        "unverified_additions": {"type": "array", "items": _object({
            "claim": text, "evidence": strings, "reason": text})},
        "repair_suggestions": {"type": "array", "items": _object({
            "issue": text, "evidence_refs": strings, "suggestion": text})},
    })
    schema['required'] = sorted(REQUIRED_RESPONSE_FIELDS)
    return schema


def instructions(modality: str) -> str:
    path = Path(__file__).with_name("prompts") / f"{modality}_judge_rubrics_v1.txt"
    return path.read_text(encoding="utf-8")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _candidate_evidence(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def parse_payload(payload: Any, modality: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get('scores'), Mapping):
        raise ValueError("rubric judgment requires four dimension scores")
    if not set(DIMENSIONS[modality]) <= set(payload['scores']):
        raise ValueError("rubric scores must include " + ", ".join(DIMENSIONS[modality]))
    scores, diagnostics = {}, []
    for name in DIMENSIONS[modality]:
        item = payload["scores"][name]
        if not isinstance(item, Mapping) or 'score' not in item:
            raise ValueError(f"{name} requires a score")
        value = item["score"]
        if value is not None and (type(value) is not int or not 0 <= value <= 4):
            raise ValueError(f"{name}.score must be an integer 0-4 or null")
        if modality == "text" and name != "verified_fact_consistency" and value is None:
            raise ValueError(f"{name} must be scored for a text candidate")
        evidence = _candidate_evidence(item.get('evidence'))
        if value is not None and not evidence:
            diagnostics.append({'code': 'score_without_citation', 'dimension': name})
        reason = item.get('reason', '')
        scores[name] = {"score": value, "reason": reason if isinstance(reason, str) else '', "evidence": evidence}
    if any(scores[name]["score"] is None for name in DIMENSIONS[modality] if name != "verified_fact_consistency"):
        raise ValueError("image visual assessment unavailable; do not replace null visual dimensions with invented scores")
    observation = payload.get('observation') or {}
    if not isinstance(observation, Mapping):
        raise ValueError('rubric observation must be an object')
    unverified = []
    if not isinstance(payload.get("unverified_additions", []), list):
        raise ValueError("unverified_additions must be an array")
    for item in payload.get("unverified_additions", []):
        if not isinstance(item, Mapping) or not isinstance(item.get('claim'), str) or not item['claim'].strip():
            diagnostics.append({'code': 'unreadable_addition'})
            continue
        unverified.append({"claim": _text(item["claim"], "claim"),
            "evidence": _candidate_evidence(item.get('evidence')),
            "reason": item.get('reason', '')})
    suggestions = []
    if not isinstance(payload.get("repair_suggestions", []), list):
        raise ValueError("repair_suggestions must be an array")
    for item in payload.get("repair_suggestions", []):
        if not isinstance(item, Mapping) or not all(isinstance(item.get(key), str) and item[key].strip() for key in ('issue', 'suggestion')):
            diagnostics.append({'code': 'unreadable_repair_suggestion'})
            continue
        suggestions.append({"issue": _text(item["issue"], "issue"),
            "evidence_refs": _candidate_evidence(item.get('evidence_refs')),
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
        "unverified_additions": unverified, "repair_suggestions": suggestions, 'rubric_diagnostics': diagnostics}


def validate_bound_observation(details, observation, targets):
    """Diagnose inconsistent assessments without discarding model scores."""
    details = dict(details)
    diagnostics = list(details.get('rubric_diagnostics', []))
    details['rubric_diagnostics'] = diagnostics
    diagnostics.extend(observation.get('diagnostics', []))
    direct = [item for item in observation["fact_checks"]
              if item["verification_mode"] == "direct" and item["status"] in {"supported", "contradicted"}]
    removals = observation["unsupported_claims"]
    for check in observation.get("repair_checks", []):
        if check["status"] == "resolved" and any(item["claim"].casefold() == check["claim"].casefold() for item in removals):
            diagnostics.append({'code': 'resolved_issue_still_reported', 'issue_id': check['issue_id']})
            check['repair_supported'] = False
    known_refs = {ref for item in targets for ref in item["evidence_refs"]}
    unresolved = {item["claim"].casefold() for item in details["unverified_additions"]}
    for item in removals:
        source_refs = set(item.get('evidence_refs', [])) | {
            value[len("context:"):].strip() for value in item["evidence"] if value.startswith("context:")}
        if not source_refs or not source_refs <= known_refs:
            diagnostics.append({'code': 'claim_without_source_citation', 'claim_id': item['claim_id']})
            item['repair_supported'] = False
        if item["claim"].casefold() in unresolved:
            diagnostics.append({'code': 'conflict_also_unverified', 'claim_id': item['claim_id']})
            item['repair_supported'] = False
    for suggestion in details["repair_suggestions"]:
        if not set(suggestion["evidence_refs"]) <= known_refs:
            diagnostics.append({'code': 'suggestion_has_unknown_source_refs'})
    fact_score = details["rubric_scores"]["verified_fact_consistency"]["score"]
    unresolved_repairs = any(item["status"] == "unresolved" and item.get('issue_kind', 'fact') == 'fact'
                             for item in observation.get("repair_checks", []))
    assessable = bool(direct or removals or unresolved_repairs)
    conflict = any(item["status"] == "contradicted" for item in direct) or bool(removals) or unresolved_repairs
    if (fact_score is None) == assessable:
        diagnostics.append({'code': 'fact_score_observation_disagreement'})
    if fact_score == 4 and conflict:
        diagnostics.append({'code': 'high_fact_score_with_conflict'})
    if fact_score is not None and fact_score < 4 and not conflict:
        diagnostics.append({'code': 'low_fact_score_without_confirmed_conflict'})
    observation['repair_needed'] = bool(repair_operations(observation))
    return details
