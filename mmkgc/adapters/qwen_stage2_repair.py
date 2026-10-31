"""Focused model repairs which preserve all other Stage 2 decisions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from mmkgc.stages.relation_planning import RelationPlanValidationError

REPAIR_VERSION = 'qwen_stage2_targeted_repair_v2'
TASK_MARKER = 'x-mmkgc-qwen-repair'
FACT_FIELDS = {'role', 'evidence_refs', 'supporting_fact_ids', 'reason'}


@dataclass(frozen=True)
class RepairTask:
    kind: str
    keys: tuple[str, ...]
    prompt: str
    response_schema: dict[str, Any] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        # Small, direct-key schemas avoid the slow global per-fact anyOf branches.
        return {**deepcopy(self.response_schema or {}), TASK_MARKER: self.kind}


def targeted_repair_tasks(source_prompt: str, response: Any, errors: list[dict[str, Any]],
                          candidate_budget: int) -> list[RepairTask]:
    """Use focused repair only for isolated missing facts, slot errors or support links."""
    if not errors or not isinstance(response, dict) or not isinstance(response.get('plan'), dict):
        return []
    plan = response['plan']
    plans = plan.get('relation_plans')
    if not isinstance(plans, list) or any(not isinstance(p, dict) for p in plans):
        return []
    status = plan.get('status')
    if not isinstance(status, str) or status not in {'ready', 'insufficient_evidence'}:
        return []
    source = json.loads(source_prompt)
    facts = {f['fact_id']: f for f in source['relation_facts']}
    ids = [p.get('fact_id') for p in plans]
    if any(not isinstance(fid, str) or fid not in facts for fid in ids) or len(ids) != len(set(ids)):
        return []
    roles = {p['fact_id']: p.get('role') for p in plans}
    if any(not isinstance(role, str) or role not in {'primary', 'secondary', 'context', 'excluded'} for role in roles.values()):
        return []
    codes = {e['code'] for e in errors}
    context = {k: deepcopy(v) for k, v in source.items() if k != 'relation_facts'}
    tasks = []

    def object_schema(properties):
        return {'type': 'object', 'additionalProperties': False, 'properties': properties,
                'required': list(properties)}

    def id_array(values):
        return {'type': 'array', 'maxItems': len(values),
                'items': {'type': 'string', **({'enum': values} if values else {})}}

    def task(kind, keys, details, instructions, schema):
        payload = {'original_context': context, 'facts_to_review': [facts[k] for k in keys if k in facts],
                   **details}
        tasks.append(RepairTask(kind, tuple(keys), instructions + '\nINPUT_JSON:\n'
                                + json.dumps(payload, ensure_ascii=False, separators=(',', ':')), schema))

    slot_codes = {'candidate_slot_count', 'unknown_anchor', 'invalid_anchor_role', 'unanchored_primary', 'invalid_anchors'}
    if 'missing_facts' in codes and codes <= {'missing_facts', *slot_codes} and isinstance(plan.get('candidate_anchors'), list):
        missing = [fid for fid in facts if fid not in roles]
        # Four fact decisions per request keep review cardinality unrelated to the image budget.
        for start in range(0, len(missing), 4):
            batch = missing[start:start + 4]
            properties = {}
            for fid in batch:
                allowed_roles = (['context', 'excluded'] if status == 'insufficient_evidence'
                                 else ['primary', 'secondary', 'context', 'excluded'] if fid in plan['candidate_anchors']
                                 else ['secondary', 'context', 'excluded'])
                support_ids = [key for key in facts if key != fid and (key in missing or roles.get(key) == 'secondary')]
                properties[fid] = object_schema({
                    'role': {'type': 'string', 'enum': allowed_roles},
                    'evidence_refs': id_array(facts[fid]['evidence_refs']),
                    'supporting_fact_ids': id_array(support_ids if status == 'ready' else []),
                    'reason': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                })
            task('missing_facts', batch, {
                'status': status, 'fixed_candidate_anchors': plan['candidate_anchors'],
                'fixed_roles': roles, 'other_missing_fact_ids': [fid for fid in missing if fid not in batch],
                'response_template': {'facts': {fid: dict(role='', evidence_refs=[], supporting_fact_ids=[], reason='') for fid in batch}},
            }, 'You are completing ONLY omitted fact reviews, NOT allocating images. '
               'Return ONLY {"facts":{fact_id:{"role":...,"evidence_refs":[...],'
               '"supporting_fact_ids":[...],"reason":...}}}. '
               'Include EXACTLY the fact_id keys shown in response_template, even if they receive ZERO images. '
               'Do not return plan, status or candidate_anchors. Do not repeat already reviewed facts. '
               'Choose each role and a short nonblank evidence-grounded reason. '
               'An ID absent from fixed_candidate_anchors CANNOT be primary; choose secondary, context or excluded. '
               'For insufficient_evidence choose only context/excluded. '
               'Each evidence_refs list must use ONLY that same input fact\'s own evidence_refs. '
               'Support links may reference only OTHER secondary facts; context/excluded use []. '
               'Use [] when no support is needed. Preserve all fixed decisions. Fill every template field.',
               object_schema({'facts': object_schema(properties)}))
    elif codes <= {'invalid_support_target', 'context_support_links'}:
        bad_ids = list(dict.fromkeys(plans[int(e['path'].split('/')[3])]['fact_id'] for e in errors))
        for start in range(0, len(bad_ids), 4):
            batch = bad_ids[start:start + 4]
            allowed = {fid: [k for k, role in roles.items() if role == 'secondary' and k != fid]
                       if roles[fid] in {'primary', 'secondary'} else [] for fid in batch}
            task('support_links', batch, {
                'fixed_roles': roles,
                'allowed_support_targets': allowed,
                'response_template': {'supports': {fid: [] for fid in batch}},
            }, 'Repair ONLY support links. Return ONLY {"supports":{fact_id:[support_fact_id,...]}} '
               'with EXACTLY the response_template keys. For each key choose targets ONLY from its '
               'allowed_support_targets, or [] if no support is needed. Never reference a primary fact or itself. '
               'Do not return role decisions, plan or image slots; those decisions are fixed.',
               object_schema({'supports': object_schema({fid: id_array(allowed[fid]) for fid in batch})}))
    else:
        slot_only = all(e['code'] in slot_codes or (e['code'] == 'missing_field'
                        and e['path'] == '/plan/candidate_anchors') for e in errors)
        primary = [fid for fid, role in roles.items() if role == 'primary']
        eligible = [fid for fid, role in roles.items() if role in {'primary', 'secondary'}]
        if not slot_only or set(ids) != set(facts) or not primary or len(primary) > candidate_budget:
            return []
        keys = [f'slot_{i:03d}' for i in range(1, candidate_budget + 1)]
        task('candidate_slots', keys, {
            'relation_facts': source['relation_facts'], 'fixed_roles': roles,
            'eligible_fact_ids': eligible, 'required_primary_fact_ids': primary,
            'response_template': {'slots': {k: '' for k in keys}},
        }, 'Allocate candidate image slots ONLY; all fact reviews and roles are already fixed. '
           'Return ONLY {"slots":{"slot_001":"fact_id",...}} with EVERY response_template key. '
           'Each value must be a STRING from eligible_fact_ids; evidence IDs such as fact: or path: are forbidden. '
           'Use the actual eligible IDs, not the literal string fact_id. Fill ALL numbered slots. '
           'Repeating an eligible ID in several slots is allowed and required when there are fewer eligible IDs than slots. '
           'Every required_primary_fact_id MUST occur in at least one slot. '
           'Choose the allocation from the source evidence. Do not return roles, relation_plans, quotas or plan.',
           object_schema({'slots': object_schema({key: {'type': 'string', 'enum': eligible} for key in keys})}))
    return tasks


def merge_targeted_response(base: dict[str, Any], task: RepairTask, text: str) -> dict[str, Any]:
    """Merge only explicit model choices; reject missing/extra/duplicate keys."""
    def fail(code, message):
        raise RelationPlanValidationError([{'code': code, 'path': '/targeted_repair', 'message': message}])

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail('duplicate_repair_key', f'Duplicate key {key}; return each requested key exactly once')
            result[key] = value
        return result

    try:
        parsed = json.loads(text, object_pairs_hook=unique_pairs)
    except json.JSONDecodeError as exc:
        fail('invalid_repair_json', str(exc))
    field = {'missing_facts': 'facts', 'support_links': 'supports', 'candidate_slots': 'slots'}[task.kind]
    if not isinstance(parsed, dict) or set(parsed) != {field} or not isinstance(parsed[field], dict):
        fail('invalid_repair_shape', f'Return only an object with a {field} object')
    values = parsed[field]
    if set(values) != set(task.keys):
        fail('repair_key_mismatch', f'Expected exactly {list(task.keys)}; received {list(values)}')
    result = deepcopy(base)
    plan = result['plan']
    if task.kind == 'missing_facts':
        existing = {p['fact_id'] for p in plan['relation_plans']}
        if existing.intersection(task.keys):
            fail('repair_overwrites_fact', 'A missing-fact repair cannot overwrite a reviewed fact')
        for fid in task.keys:
            value = values[fid]
            if not isinstance(value, dict) or set(value) != FACT_FIELDS:
                fail('invalid_fact_repair_fields', f'{fid} must contain exactly {sorted(FACT_FIELDS)}')
            plan['relation_plans'].append({'fact_id': fid, **deepcopy(value)})
    elif task.kind == 'support_links':
        for item in plan['relation_plans']:
            if item['fact_id'] in values:
                item['supporting_fact_ids'] = deepcopy(values[item['fact_id']])
    else:
        plan['candidate_anchors'] = [deepcopy(values[key]) for key in task.keys]
    # Cross-field and evidence validation happens on the entire merged response before acceptance.
    return result
