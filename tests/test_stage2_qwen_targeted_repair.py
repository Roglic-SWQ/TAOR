"""Regression coverage for omitted facts, evidence/anchor confusion and frozen links."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import httpx

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.qwen_stage2_repair import (
    TASK_MARKER, RepairTask, merge_targeted_response, targeted_repair_tasks,
)
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.stages.relation_planning import RelationPlanValidationError, selection_prompt


def example(count=12):
    knowledge = {'relation_facts': [
        {'fact_id': f'rf:{i:04d}', 'relation_group_id': f'rg:{i:04d}',
         'subject': 'City', 'relation': 'located in', 'object': f'Area {i}',
         'direction': 'out', 'claim': f'City is in Area {i}.',
         'evidence_refs': [f'fact:{i:04d}'], 'evidence_match_score': .7,
         'evidence_triples': [f'(City, located in, Area {i})']}
        for i in range(1, count + 1)]}
    plans = [{'fact_id': f['fact_id'], 'role': 'primary' if i < min(5, count) else 'secondary',
              'evidence_refs': f['evidence_refs'], 'supporting_fact_ids': [],
              'reason': 'Supported by the supplied location evidence.'}
             for i, f in enumerate(knowledge['relation_facts'])]
    anchors = [plans[i % min(5, count)]['fact_id'] for i in range(5)]
    return knowledge, {'plan': {'status': 'ready', 'candidate_anchors': anchors, 'relation_plans': plans}}


def planner():
    return OpenAICompatibleRelationPlanner(QwenConfig('Qwen/Qwen2-VL-7B-Instruct',
                                                      'http://127.0.0.1:12345/v1', 'EMPTY'),
                                          retry_backoff_seconds=0)


def test_missing_seven_facts_are_model_reviewed_in_two_batches_without_new_slots(monkeypatch):
    knowledge, full = example()
    original = deepcopy(knowledge)
    partial = deepcopy(full)
    partial['plan']['relation_plans'] = partial['plan']['relation_plans'][:5]
    client = planner()
    captured = []

    def complete(messages, schema):
        if TASK_MARKER not in schema:
            return json.dumps(partial)
        captured.append((messages, schema))
        data = json.loads(messages[-1]['content'].split('INPUT_JSON:\n')[1])
        return json.dumps({'facts': {f['fact_id']: {
            'role': 'secondary', 'evidence_refs': f['evidence_refs'], 'supporting_fact_ids': [],
            'reason': 'Adds supplied geographic context without consuming image slots.'}
            for f in data['facts_to_review']}})

    monkeypatch.setattr(client, '_complete', complete)
    result = client.plan(SimpleNamespace(label='City'), 'image', knowledge, 5)
    selection = result['fine_grained_relation_selection']
    assert knowledge == original
    assert selection['candidate_anchors'] == full['plan']['candidate_anchors']
    assert selection['quota_summary']['assigned_total'] == 5
    assert len(selection['secondary_relations']) == 7
    assert [len(json.loads(m[-1]['content'].split('INPUT_JSON:\n')[1])['facts_to_review']) for m, _ in captured] == [4, 3]
    assert client.calls[-1]['contract_repair_retries'] == 1
    assert client.calls[-1]['validation_attempts'][-1]['response_source'] == 'model_patch_merge'
    merged = json.loads(client.calls[-1]['response_text'])
    assert merged['plan']['relation_plans'][:5] == partial['plan']['relation_plans']


@pytest.mark.parametrize('bad_slots', [None, ['rf:0001', 'rf:0002', 'rf:0003'],
                                      ['fact:0001', 'fact:0002', 'path:0001', 'path:0002', 'fact:0003']])
def test_slot_repair_requires_five_explicit_model_choices_and_preserves_reviews(monkeypatch, bad_slots):
    knowledge, valid = example(3)
    invalid = deepcopy(valid)
    if bad_slots is None:
        invalid['plan'].pop('candidate_anchors')
    else:
        invalid['plan']['candidate_anchors'] = bad_slots
    client = planner()
    requests = []

    def complete(messages, schema):
        if TASK_MARKER not in schema:
            return json.dumps(invalid)
        requests.append(schema[TASK_MARKER])
        return json.dumps({'slots': dict(zip([f'slot_{i:03d}' for i in range(1, 6)],
                                            ['rf:0003', 'rf:0001', 'rf:0002', 'rf:0002', 'rf:0001']))})

    monkeypatch.setattr(client, '_complete', complete)
    result = client.plan(SimpleNamespace(label='City'), 'image', knowledge, 5)
    assert requests == ['candidate_slots']
    assert result['fine_grained_relation_selection']['candidate_anchors'] == ['rf:0003', 'rf:0001', 'rf:0002', 'rf:0002', 'rf:0001']
    assert json.loads(client.calls[-1]['response_text'])['plan']['relation_plans'] == valid['plan']['relation_plans']


def test_support_only_repair_preserves_all_nine_fact_reviews(monkeypatch):
    knowledge, valid = example(9)
    valid['plan']['relation_plans'][1]['role'] = 'secondary'
    invalid = deepcopy(valid)
    invalid['plan']['relation_plans'][1]['supporting_fact_ids'] = ['rf:0001']
    client = planner()

    def complete(messages, schema):
        if TASK_MARKER not in schema:
            return json.dumps(invalid)
        assert schema[TASK_MARKER] == 'support_links'
        return json.dumps({'supports': {'rf:0002': ['rf:0006']}})

    monkeypatch.setattr(client, '_complete', complete)
    client.plan(SimpleNamespace(label='City'), 'image', knowledge, 5)
    merged = json.loads(client.calls[-1]['response_text'])
    expected = deepcopy(invalid)
    expected['plan']['relation_plans'][1]['supporting_fact_ids'] = ['rf:0006']
    assert merged == expected


def test_missing_facts_then_context_support_repair_keeps_all_reviews(monkeypatch):
    knowledge, full = example(6)
    partial = deepcopy(full)
    partial['plan']['relation_plans'].pop()
    client = planner()
    kinds = []

    def complete(messages, schema):
        if TASK_MARKER not in schema:
            return json.dumps(partial)
        kind = schema[TASK_MARKER]
        kinds.append(kind)
        if kind == 'missing_facts':
            return json.dumps({'facts': {'rf:0006': {'role': 'context', 'evidence_refs': ['fact:0006'],
                                'supporting_fact_ids': ['rf:0001'], 'reason': 'Background context from source.'}}})
        assert kind == 'support_links'
        support_schema = schema['properties']['supports']['properties']['rf:0006']
        assert support_schema['maxItems'] == 0
        return '{"supports":{"rf:0006":[]}}'

    monkeypatch.setattr(client, '_complete', complete)
    client.plan(SimpleNamespace(label='City'), 'image', knowledge, 5)
    assert kinds == ['missing_facts', 'support_links']
    merged = json.loads(client.calls[-1]['response_text'])['plan']
    assert len(merged['relation_plans']) == 6
    assert merged['relation_plans'][:5] == partial['plan']['relation_plans']
    assert merged['relation_plans'][-1]['role'] == 'context'
    assert merged['relation_plans'][-1]['supporting_fact_ids'] == []


@pytest.mark.parametrize('response', [
    '{"facts":{}}', '{"facts":{"rf:0006":{},"unknown":{}}}',
    '{"facts":{"rf:0006":{},"rf:0006":{}}}',
    '{"facts":{"rf:0006":{"role":"secondary"}}}', '{"plan":{}}', '{broken',
])
def test_malformed_patch_is_rejected_without_changing_base(response):
    _, full = example(6)
    base = deepcopy(full)
    base['plan']['relation_plans'].pop()
    original = deepcopy(base)
    task = RepairTask('missing_facts', ('rf:0006',), 'Review the missing fact')
    with pytest.raises(RelationPlanValidationError):
        merge_targeted_response(base, task, response)
    assert base == original


def test_wrong_evidence_in_model_patch_is_revalidated_and_never_accepted(monkeypatch):
    knowledge, full = example(6)
    partial = deepcopy(full)
    partial['plan']['relation_plans'].pop()
    client = planner()

    def complete(messages, schema):
        if TASK_MARKER not in schema:
            return json.dumps(partial)
        return json.dumps({'facts': {'rf:0006': {'role': 'secondary', 'evidence_refs': ['fact:0001'],
                           'supporting_fact_ids': [], 'reason': 'Wrong citation must be rejected.'}}})

    monkeypatch.setattr(client, '_complete', complete)
    with pytest.raises(RelationPlanValidationError):
        client.plan(SimpleNamespace(label='City'), 'image', knowledge, 5)
    assert client.calls[-1]['status'] == 'failed'
    assert any(e['code'] == 'foreign_evidence_refs' for a in client.calls[-1]['validation_attempts'] for e in a['errors'])


def test_partial_patch_never_becomes_accepted_and_retries_are_bounded(monkeypatch):
    knowledge, full = example(6)
    partial = deepcopy(full)
    partial['plan']['relation_plans'].pop()
    client = planner()
    requests = []

    def complete(messages, schema):
        requests.append(schema.get(TASK_MARKER))
        return json.dumps(partial) if TASK_MARKER not in schema else '{"facts":{}}'

    monkeypatch.setattr(client, '_complete', complete)
    with pytest.raises(RelationPlanValidationError):
        client.plan(SimpleNamespace(label='City'), 'image', knowledge, 5)
    assert requests == [None, 'missing_facts', 'missing_facts']
    assert client.calls[-1]['status'] == 'failed'
    assert client.calls[-1]['contract_repair_stopped_reason'] == 'repeated_invalid_response'


def test_mixed_errors_keep_full_replanning_instead_of_freezing_invalid_decisions():
    knowledge, full = example(6)
    partial = deepcopy(full)
    partial['plan']['relation_plans'].pop()
    prompt = selection_prompt('City', 'image', knowledge, 5)[1]
    errors = [{'code': 'missing_facts'}, {'code': 'foreign_evidence_refs'}]
    assert targeted_repair_tasks(prompt, partial, errors, 5) == []


def test_small_missing_fact_schema_disallows_unanchored_primary_and_foreign_evidence():
    knowledge, full = example(6)
    partial = deepcopy(full)
    partial['plan']['relation_plans'].pop()
    prompt = selection_prompt('City', 'image', knowledge, 5)[1]
    tasks = targeted_repair_tasks(prompt, partial, [{'code': 'missing_facts'}], 5)
    schema = tasks[0].schema
    schema.pop(TASK_MARKER)
    validator = pytest.importorskip('jsonschema').Draft202012Validator(schema)
    good = {'facts': {'rf:0006': {'role': 'secondary', 'evidence_refs': ['fact:0006'],
                                'supporting_fact_ids': [], 'reason': 'Supported geographic context.'}}}
    validator.validate(good)
    for field, value in [('role', 'primary'), ('evidence_refs', ['fact:0001']), ('reason', '')]:
        bad = deepcopy(good)
        bad['facts']['rf:0006'][field] = value
        assert list(validator.iter_errors(bad))
    assert 'anyOf' not in json.dumps(schema)
    assert 'pattern' not in json.dumps(schema)


def test_only_targeted_qwen_transport_uses_tiny_guided_schema(monkeypatch):
    knowledge, invalid = example(3)
    invalid['plan']['candidate_anchors'] = ['rf:0001']
    prompt = selection_prompt('City', 'image', knowledge, 5)[1]
    tasks = targeted_repair_tasks(prompt, invalid, [{'code': 'candidate_slot_count'}], 5)
    bodies = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json, headers):
            request = httpx.Request('POST', url)
            if url.endswith('/tokenize'):
                return httpx.Response(200, request=request, json={'count': 1000, 'max_model_len': 12288})
            bodies.append(json)
            return httpx.Response(200, request=request, json={
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"slots":{}}'}}]})

    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    client = planner()
    messages = [{'role': 'user', 'content': tasks[0].prompt}]
    client._complete(messages, tasks[0].schema)
    body = bodies[0]
    assert body['messages'] == messages
    assert body['guided_decoding_backend'] == 'lm-format-enforcer'
    assert TASK_MARKER not in body['guided_json']
    assert set(body['guided_json']['properties']['slots']['properties']) == {f'slot_{i:03d}' for i in range(1, 6)}
    assert 'response_format' not in body
