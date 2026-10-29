"""Stage 2 model routing and vLLM compatibility preserve the planning contract."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.relation_planner import (
    GPT4oMiniRelationPlanner, OpenAICompatibleRelationPlanner, _qwen_planning_messages, QWEN_STAGE2_SYSTEM,
)
from mmkgc.stages.relation_planning import candidate_slot_response_schema


@pytest.mark.parametrize('local', [True, False])
def test_stage2_transport_selects_backend_without_altering_contract(monkeypatch, local):
    captured = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json, headers):
            if url.endswith('/tokenize'):
                return httpx.Response(200, request=httpx.Request('POST', url), json={
                    'count': 10600, 'max_model_len': 12288, 'tokens': []})
            captured.append((url, json))
            return httpx.Response(200, request=httpx.Request('POST', url), json={
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{"plan":{}}'}}]})

    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    model = 'Qwen/Qwen2-VL-7B-Instruct' if local else 'gpt-4o-mini'
    base = 'http://127.0.0.1:12345/v1' if local else 'https://gpt.example.test/v1'
    planner = OpenAICompatibleRelationPlanner(QwenConfig(model, base, 'test-key', max_token=3072))
    schema = candidate_slot_response_schema({'relation_facts': []}, 5)
    schema['$defs']['reason'] = {'type': 'string', 'minLength': 1, 'pattern': r'\S'}
    original = deepcopy(schema)
    assert planner._complete([{'role': 'user', 'content': 'Plan from evidence.'}], schema) == '{"plan":{}}'
    url, payload = captured[0]
    assert url == base + '/chat/completions'
    assert payload['model'] == model
    assert original == schema
    assert payload['max_tokens'] == (1688 if local else 4096)
    if local:
        assert 'response_format' not in payload
        assert 'Return ONLY this object after filling it:' in payload['messages'][-1]['content']
        assert 'guided_decoding_backend' not in payload
        assert payload['repetition_penalty'] == 1.05
    else:
        assert payload['response_format']['json_schema']['schema'] == original
        assert payload['response_format']['json_schema']['strict'] is True
        assert len(payload['messages']) == 1
        assert 'guided_decoding_backend' not in payload
        assert 'repetition_penalty' not in payload


def test_local_qwen_applies_same_contract_and_reports_actual_model(monkeypatch):
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/Qwen2-VL-7B-Instruct',
                                                       'http://localhost:12345/v1', 'EMPTY'))
    source = {'relation_facts': []}
    reply = {'plan': {'status': 'insufficient_evidence', 'relation_plans': [], 'candidate_anchors': []}}
    monkeypatch.setattr(planner, '_complete', lambda *args: json.dumps(reply))
    result = planner.plan(SimpleNamespace(label='Unknown'), 'image', source, 5)
    assert result['fine_grained_relation_selection']['quota_summary']['assigned_total'] == 0
    assert planner.calls[-1]['provider'] == 'qwen-local-global-relation-planner'
    assert planner.calls[-1]['prompt_version'] == 'qwen7b_stage2_candidate_slots_v3'
    assert planner.calls[-1]['status'] == 'applied'
    assert GPT4oMiniRelationPlanner is OpenAICompatibleRelationPlanner


@pytest.mark.parametrize('problem', ['empty_reason', 'foreign_evidence_refs'])
def test_local_qwen_keeps_contract_checks_outside_decoder(monkeypatch, problem):
    planner = OpenAICompatibleRelationPlanner(QwenConfig('Qwen/Qwen2-VL-7B-Instruct',
                                                       'http://localhost:12345/v1', 'EMPTY'))
    source = {'relation_facts': [{'fact_id': 'rf:0001', 'relation_group_id': 'rg:0001',
        'subject': 'Museum', 'relation': 'located in', 'object': 'City', 'direction': 'out',
        'claim': 'Museum is in City.', 'evidence_refs': ['fact:0001'],
        'evidence_triples': ['(Museum, located in, City)'], 'evidence_match_score': 0.8}]}
    original = deepcopy(source)
    valid = {'plan': {'status': 'ready', 'candidate_anchors': ['rf:0001'] * 5,
                     'relation_plans': [{'fact_id': 'rf:0001', 'role': 'primary',
                        'evidence_refs': ['fact:0001'], 'supporting_fact_ids': [],
                        'reason': 'Supported by the supplied location evidence.'}]}}
    invalid = deepcopy(valid)
    if problem == 'empty_reason':
        invalid['plan']['relation_plans'][0]['reason'] = ' \n\t'
    else:
        invalid['plan']['relation_plans'][0]['evidence_refs'] = ['fact:9999']
    responses = iter([invalid, valid])
    monkeypatch.setattr(planner, '_complete', lambda *args: json.dumps(next(responses)))
    result = planner.plan(SimpleNamespace(label='Museum'), 'image', source, 5)
    assert result['fine_grained_relation_selection']['quota_summary']['assigned_total'] == 5
    assert source == original
    assert planner.calls[-1]['contract_repair_retries'] == 1
    errors = planner.calls[-1]['validation_attempts'][0]['errors']
    assert any(error['code'] == problem for error in errors)


def test_local_contract_prompt_lists_every_fact_and_budget_without_mutating_input():
    source = {'relation_facts': [
        {'fact_id': f'rf:{i:04d}', 'evidence_refs': [f'fact:{i:04d}']}
        for i in range(1, 13)]}
    schema = candidate_slot_response_schema(source, 5)
    original = deepcopy(schema)
    messages = [{'role': 'system', 'content': 'Use source evidence.'}, {'role': 'user', 'content': 'Original source'}]
    sent = _qwen_planning_messages(messages, schema)
    assert schema == original
    assert sent[0]['content'] == QWEN_STAGE2_SYSTEM and sent[1] == messages[1]
    assert messages[0]['content'] == 'Use source evidence.' and len(messages) == 2
    contract = json.loads(sent[-1]['content'].split('Return ONLY this object after filling it:\n')[1])
    expected_ids = [f['fact_id'] for f in source['relation_facts']]
    assert set(contract) == {'plan'}
    assert list(contract['plan']) == ['status', 'candidate_anchors', 'relation_plans']
    assert [p['fact_id'] for p in contract['plan']['relation_plans']] == expected_ids
    assert contract['plan']['candidate_anchors'] == [''] * 5
    assert '"ready":5,"insufficient_evidence":0' in sent[-1]['content']
