"""Candidate-local factual observations and evidence-bound repair operations."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any


OBSERVATION_VERSION = "fact_observation_v2"
REPAIR_CONFIDENCE = 0.8
REFLECTION_POLICY = "targeted_reflection_v4"
CHECK_STATUSES = ("supported", "contradicted", "missing", "unverifiable", "uncertain")
VERIFICATION_MODES = ("direct", "appearance_only", "unverifiable")
SEMANTIC_ISSUE_ID = 'semantic:entity'


def _plain_metadata(value):
    """Copy JSON metadata even after ranking has frozen its nested mappings."""
    if isinstance(value, Mapping):
        return {key: _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    return value


def _evidence(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [v for v in value if isinstance(v, str) and v.strip()] if isinstance(value, (list, tuple)) else []


def _routing_confidence(value):
    """Missing or unusable confidence cannot authorize an automatic edit."""
    try:
        return _confidence(value)
    except ValueError:
        return 0.0


def _has_candidate_evidence(evidence):
    # Prefixes are optional. Explicit source/prompt metadata is still not a
    # citation of candidate content and cannot authorize an automatic edit.
    return any(not value.lstrip().lower().startswith(('context:', 'metadata:')) for value in evidence)


def _metadata_text(item, key):
    value = item.get(key, '')
    return value if isinstance(value, str) else ''


def validate_repair_targets(value: Any) -> list[dict[str, Any]]:
    """Carry original rejected assertions into a repaired candidate's review."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("repair_targets must be an array")
    result, seen = [], set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) not in (
                {"issue_id", "claim", "evidence_refs"}, {"issue_id", "claim", "evidence_refs", "issue_kind"}):
            raise ValueError("repair target requires issue_id, claim, evidence_refs")
        issue_id = _text(item["issue_id"], "issue_id")
        if issue_id in seen:
            raise ValueError("duplicate repair issue_id")
        seen.add(issue_id)
        result.append({"issue_id": issue_id, "claim": _text(item["claim"], "claim"),
                       "evidence_refs": _strings(item["evidence_refs"], "evidence_refs")})
        kind = item.get('issue_kind', 'fact')
        if kind not in {'fact', 'semantic'}:
            raise ValueError('Unknown repair issue kind')
        if kind == 'semantic':
            result[-1]['issue_kind'] = kind
    return result


def removal_targets(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return validate_repair_targets([{"issue_id": item.get("claim_id", item.get("check_id")), "claim": item["observed"],
        **({'issue_kind': 'semantic'} if item['action'] == 'reframe' else {}),
        "evidence_refs": item.get("evidence_refs") or [entry[len("context:"):].strip() for entry in item["evidence"]
                          if entry.startswith("context:")]} for item in operations if item["action"] in {"remove", "reframe"}])


def bind_repair_checks(raw: Any, targets: Any, modality: str) -> list[dict[str, Any]]:
    targets = validate_repair_targets(targets)
    if not isinstance(raw, list):
        raise ValueError('repair_checks must be an array')
    by_id = {item['issue_id']: item for item in targets}
    result, seen = [], set()
    for item in raw:
        if not isinstance(item, Mapping) or not {'issue_id', 'status'} <= set(item):
            raise ValueError('invalid repair check fields')
        issue_id = item['issue_id']
        if not isinstance(issue_id, str) or issue_id not in by_id or issue_id in seen:
            raise ValueError('unknown or duplicate repair issue_id')
        seen.add(issue_id)
        if item['status'] not in {'resolved', 'unresolved', 'uncertain'}:
            raise ValueError('invalid repair check status')
        evidence = _evidence(item.get('evidence'))
        observed = _metadata_text(item, 'observed')
        result.append({**deepcopy(by_id[issue_id]), 'status': item['status'],
            'observed': observed, 'evidence': evidence, 'repair_supported': _has_candidate_evidence(evidence),
            'confidence': _routing_confidence(item.get('confidence')), 'reason': _metadata_text(item, 'reason')})
    return result


def compile_verification_targets(context: Mapping[str, Any], knowledge: Mapping[str, Any], modality: str) -> list[dict[str, Any]]:
    """Bind candidate-local evidence without assigning coverage obligations."""
    facts = {fact["fact_id"]: fact for fact in knowledge.get("relation_facts", [])}
    non_visual = knowledge.get("text_visual_profile", {}).get("non_visual_facts", [])
    targets = []
    catalog = {item['evidence_id']: item for item in knowledge.get('evidence_catalog', [])}
    for bucket in ("checks",):
        for constraint in context.get(bucket, []):
            claim = constraint.get("constraint")
            if not claim:
                continue
            refs = list(constraint.get("source_fact_refs", []))
            fact = facts.get(refs[0], {}) if refs else {}
            targets.append({
                "check_id": constraint["constraint_id"],
                "fact_id": refs[0] if refs else None,
                "claim": claim,
                "subject": fact.get("subject", ""), "relation": fact.get("relation", ""),
                "object": fact.get("object", ""), "direction": fact.get("direction", ""),
                "verification_mode": ("direct" if modality == "text" else
                                      "unverifiable" if claim in non_visual else "assess"),
                "evidence_refs": list(constraint.get("evidence_refs", [])),
                "source_modalities": sorted(set(fact.get('source_modalities', [])) | {
                    catalog[ref]['modality'] for ref in constraint.get('evidence_refs', [])
                    if ref in catalog and catalog[ref].get('modality')}),
                "visual_observations": _plain_metadata(fact.get('visual_observations', [])),
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
        mode = item.get("verification_mode")
        if 'requirement' in item:
            raise ValueError('Legacy coverage requirements must be converted at the saved-input boundary')
        if mode not in {*VERIFICATION_MODES, "assess"}:
            raise ValueError("invalid verification mode")
        target = {
            "check_id": check_id, "fact_id": fact_id,
            "claim": _text(item.get("claim"), "claim"),
            "verification_mode": mode, "evidence_refs": _strings(item.get("evidence_refs"), "evidence_refs"),
        }
        for key in ("subject", "relation", "object", "direction"):
            target[key] = _text(item.get(key, ""), key, allow_empty=True)
        target['source_modalities'] = _strings(item.get('source_modalities', []), 'source_modalities')
        visual = item.get('visual_observations', [])
        if not isinstance(visual, (list, tuple)) or any(not isinstance(v, Mapping) for v in visual):
            raise ValueError('visual_observations must contain source observation objects')
        target['visual_observations'] = _plain_metadata(visual)
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
    """Bind trusted source IDs; incomplete citations remain assessment diagnostics."""
    targets = validate_verification_targets(targets)
    by_id = {item['check_id']: item for item in targets}
    repair_targets = validate_repair_targets(repair_targets or [])
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise ValueError('observation must be an object')
    if not isinstance(raw.get('fact_checks', []), list) or not isinstance(raw.get('unsupported_claims', []), list):
        raise ValueError('observation entries must be arrays')
    checks, seen, diagnostics = [], set(), []
    for item in raw.get('fact_checks', []):
        if not isinstance(item, Mapping) or not {'check_id', 'status', 'verification_mode'} <= set(item):
            raise ValueError('invalid fact check fields')
        check_id = item['check_id']
        if not isinstance(check_id, str) or check_id not in by_id or check_id in seen:
            raise ValueError('unknown or duplicate observation check_id')
        seen.add(check_id)
        target = by_id[check_id]
        status, mode = item['status'], item['verification_mode']
        if status not in CHECK_STATUSES or mode not in VERIFICATION_MODES:
            raise ValueError('invalid observation status or verification mode')
        evidence = _evidence(item.get('evidence'))
        observed = _metadata_text(item, 'observed')
        compatible = not ((target['verification_mode'] == 'unverifiable' and mode != 'unverifiable')
            or (mode == 'unverifiable' and status != 'unverifiable')
            or (status == 'unverifiable' and mode == 'direct')
            or (mode == 'appearance_only' and status not in {'uncertain', 'unverifiable'}))
        if not compatible:
            diagnostics.append({'code': 'observation_mode_disagreement', 'check_id': check_id})
        if not evidence:
            diagnostics.append({'code': 'check_without_citation', 'check_id': check_id})
        checks.append({**deepcopy(target), 'expected': target['claim'], 'status': status,
            'verification_mode': mode, 'observed': observed, 'evidence': evidence,
            'repair_supported': compatible and _has_candidate_evidence(evidence),
            'confidence': _routing_confidence(item.get('confidence')), 'reason': _metadata_text(item, 'reason')})
    if seen != set(by_id):
        diagnostics.append({'code': 'unassessed_checks', 'check_ids': sorted(set(by_id) - seen)})
    unsupported = []
    for index, item in enumerate(raw.get('unsupported_claims', []), 1):
        if not isinstance(item, Mapping) or not _metadata_text(item, 'claim').strip():
            diagnostics.append({'code': 'unreadable_conflict'})
            continue
        evidence = _evidence(item.get('evidence'))
        unsupported.append({'claim_id': f'claim:{index:04d}', 'claim': item['claim'],
            'evidence': evidence, 'evidence_refs': _evidence(item.get('evidence_refs')),
            'repair_supported': _has_candidate_evidence(evidence), 'confidence': _routing_confidence(item.get('confidence')),
            'reason': _metadata_text(item, 'reason')})
    observation = {'schema_version': OBSERVATION_VERSION, 'candidate_id': candidate_id, 'modality': modality,
        'fact_checks': checks, 'unsupported_claims': unsupported,
        'violated_facts': [c for c in checks if c['status'] == 'contradicted'],
        'missing_facts': [c for c in checks if c['status'] == 'missing'],
        'unverifiable_facts': [c for c in checks if c['status'] == 'unverifiable'], 'diagnostics': diagnostics}
    if repair_targets:
        observation['repair_checks'] = bind_repair_checks(raw.get('repair_checks', []), repair_targets, modality)
        absent = {t['issue_id'] for t in repair_targets} - {c['issue_id'] for c in observation['repair_checks']}
        if absent:
            diagnostics.append({'code': 'unassessed_repair_issues', 'issue_ids': sorted(absent)})
    observation['repair_needed'] = bool(repair_operations(observation))
    return observation


def repair_operations(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only actionable observations with supplied evidence become edits."""
    operations = []
    for item in observation.get("fact_checks", []):
        if item.get("repair_supported") is False:
            continue
        if item["confidence"] < REPAIR_CONFIDENCE or item["verification_mode"] != "direct" or not item["evidence_refs"]:
            continue
        status = item["status"]
        if status != "contradicted":
            continue
        operations.append({
            "action": "replace" if status == "contradicted" else "add",
            "check_id": item["check_id"], "fact_id": item["fact_id"],
            "expected": item["expected"], "observed": item["observed"],
            "evidence_refs": list(item["evidence_refs"]), "evidence": list(item["evidence"]),
        })
    for item in observation.get("unsupported_claims", []):
        if item.get("repair_supported") is False:
            continue
        if item["confidence"] >= REPAIR_CONFIDENCE:
            operations.append({"action": "remove", "claim_id": item["claim_id"],
                               "observed": item["claim"], "evidence": list(item["evidence"]),
                               **({'evidence_refs': list(item['evidence_refs'])} if item.get('evidence_refs') else {})})
    reported = {item["observed"].casefold() for item in operations if item["action"] == "remove"}
    for item in observation.get("repair_checks", []):
        if item.get("repair_supported") is False:
            continue
        if item.get('issue_kind', 'fact') == 'semantic':
            continue
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


def candidate_repair_operations(candidate):
    """Route a grounded semantic rejection without relabeling it a factual error."""
    operations = repair_operations(candidate.get('observation') or {})
    gate = candidate.get('hard_gate', {})
    rubric = candidate.get('judge_rubrics', {}).get('rubric_scores', {}).get('entity_semantics', {})
    refs = sorted({ref for item in candidate.get('verification_targets', []) for ref in item.get('evidence_refs', [])})
    evidence = list(rubric.get('evidence') or gate.get('evidence') or [])
    if gate.get('decision') == 'reject' and rubric.get('score') == 0 and refs and evidence:
        issue = rubric.get('reason') or gate.get('reason') or '; '.join(evidence)
        operations.append({'action': 'reframe', 'check_id': SEMANTIC_ISSUE_ID,
            'expected': 'Address this entity-semantic issue using the supplied sources: ' + issue,
            'observed': issue,
            'evidence': evidence, 'evidence_refs': refs})
    return operations


def repair_is_eligible(original: Mapping[str, Any], repaired: Mapping[str, Any]) -> bool:
    """A repair must resolve an observed defect without sacrificing verified facts."""
    if repaired.get("hard_gate", {}).get("decision") != "pass":
        return False
    before = original.get("observation") or {}
    operations = candidate_repair_operations(original)
    if not operations:
        return False
    after = repaired.get("observation") or {}
    checked = {item["check_id"]: item for item in after.get("fact_checks", [])}
    if set(checked) != {item["check_id"] for item in before.get("fact_checks", [])}:
        return False
    for item in before.get("fact_checks", []):
        current = checked[item["check_id"]]
        if any(item.get(key) != current.get(key) for key in
               ("expected", "fact_id", "subject", "relation", "object", "direction", "source_modalities", "visual_observations")):
            return False
        if set(item.get("evidence_refs", [])) != set(current.get("evidence_refs", [])):
            return False

    def verified(item):
        return item.get("repair_supported") is not False and item.get("status") == "supported" and item.get("verification_mode") == "direct" and item.get("confidence", 0) >= REPAIR_CONFIDENCE

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
    for operation in candidate_repair_operations(original):
        remove = operation["action"] == "remove"
        explicit_issue = operation['action'] in {'remove', 'reframe'}
        issue_id = operation["claim_id"] if remove else operation["check_id"]
        check = (removals if explicit_issue else facts).get(issue_id, {})
        state = "uncertain"
        if check.get("repair_supported") is not False and check.get("confidence", 0) >= REPAIR_CONFIDENCE:
            if explicit_issue and check.get("claim") == operation["observed"]:
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
