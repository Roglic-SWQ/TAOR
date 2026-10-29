from copy import deepcopy
from types import SimpleNamespace
import asyncio
import json
import time
from urllib.error import HTTPError

import pytest

from mmkgc.adapters.qwen import QwenConfig, QwenLocalImagePromptGenerator
from mmkgc.adapters.relation_planner import GPT4oMiniRelationPlanner
from mmkgc.stages.relation_planning import RelationPlanValidationError


def case():
    source = {'relation_facts': [
        {'fact_id': f'rf:{i:04d}', 'relation_group_id': f'rg:{i:04d}',
         'subject': 'Place', 'relation': 'located in', 'object': f'Area {i}',
         'direction': 'out', 'claim': f'Place is in Area {i}', 'evidence_refs': [f'fact:{i:04d}'],
         'evidence_triples': [f'(Place, located in, Area {i})'], 'evidence_match_score': .8}
        for i in range(1, 4)]}
    value = {'plan': {'status': 'ready', 'candidate_anchors': ['rf:0001'] * 5,
        'relation_plans': [{'fact_id': f['fact_id'], 'role': 'primary' if i == 0 else 'secondary',
            'evidence_refs': f['evidence_refs'], 'supporting_fact_ids': [], 'reason': 'Supported by supplied evidence.'}
            for i, f in enumerate(source['relation_facts'])]}}
    return source, value


def planner(**kwargs):
    return GPT4oMiniRelationPlanner(QwenConfig('gpt-4o-mini', 'http://localhost/v1', 'private-key'),
                                   retry_backoff_seconds=0, **kwargs)


@pytest.mark.parametrize('failure', [TimeoutError('timeout'), ConnectionResetError('reset'),
    HTTPError('http://localhost', 429, 'rate', {}, None), HTTPError('http://localhost', 503, 'busy', {}, None)])
def test_network_retry_recovers_without_consuming_contract_budget(monkeypatch, failure):
    source, valid = case()
    p = planner()
    count = 0

    def complete(*args):
        nonlocal count
        count += 1
        if count == 1:
            raise failure
        return json.dumps(valid)

    monkeypatch.setattr(p, '_complete', complete)
    p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    call = p.calls[-1]
    assert call['transport_retries'] == 1
    assert call.get('contract_repair_retries', 0) == 0
    assert [a['status'] for a in call['network_attempts']] == ['failed', 'received']
    assert len(call['validation_attempts']) == 1
    assert len({a['local_request_id'] for a in call['network_attempts']}) == 2


@pytest.mark.parametrize('status', [400, 401, 403, 422])
def test_permanent_http_errors_are_not_retried(monkeypatch, status):
    source, _ = case()
    p = planner()
    monkeypatch.setattr(p, '_complete', lambda *args: (_ for _ in ()).throw(HTTPError('http://localhost', status, 'bad', {}, None)))
    with pytest.raises(HTTPError):
        p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert len(p.calls[-1]['network_attempts']) == 1


def test_network_retries_are_bounded_and_exhaustion_keeps_diagnostics(monkeypatch):
    source, _ = case()
    p = planner()
    monkeypatch.setattr(p, '_complete', lambda *args: (_ for _ in ()).throw(ConnectionResetError('reset')))
    with pytest.raises(ConnectionResetError) as caught:
        p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert len(p.calls[-1]['network_attempts']) == 2
    assert caught.value.model_call is p.calls[-1]
    assert p.calls[-1]['validation_attempts'] == []


def test_stage_budget_stops_further_requests(monkeypatch):
    source, _ = case()
    p = planner(stage_timeout_seconds=.01)

    def complete(*args):
        time.sleep(.02)
        raise TimeoutError('timed out')

    monkeypatch.setattr(p, '_complete', complete)
    with pytest.raises(TimeoutError, match='total time budget'):
        p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert len(p.calls[-1]['network_attempts']) == 1


def test_http_total_timeout_cancels_request(monkeypatch):
    source, _ = case()
    cancelled = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

    monkeypatch.setattr('httpx.AsyncClient', Client)
    p = planner(timeout_seconds=.02, max_transport_retries=0)
    with pytest.raises(TimeoutError):
        p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert cancelled == [True]
    assert p.calls[-1]['network_attempts'][0]['status'] == 'failed'


def test_support_only_repair_uses_dynamic_schema_without_invalid_assistant_text(monkeypatch):
    source, valid = case()
    invalid = deepcopy(valid)
    invalid['plan']['relation_plans'][1]['supporting_fact_ids'] = ['rf:0001']
    p = planner()
    requests = []
    before = deepcopy((source, invalid))

    def complete(messages, schema):
        requests.append(deepcopy((messages, schema)))
        if len(requests) == 1:
            return json.dumps(invalid)
        assert [m['role'] for m in messages] == ['system', 'user', 'user']
        assert json.dumps(invalid) not in json.dumps(messages)
        feedback = json.loads(messages[-1]['content'].split('VALIDATION_ERRORS_JSON:\n')[1])
        assert feedback['allowed_supporting_fact_ids']['rf:0002'] == ['rf:0003']
        validator = pytest.importorskip('jsonschema').Draft202012Validator(schema)
        assert list(validator.iter_errors(invalid))
        validator.validate(valid)
        return json.dumps(valid)

    monkeypatch.setattr(p, '_complete', complete)
    p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert p.calls[-1]['contract_repair_retries'] == 1
    assert p.calls[-1]['transport_retries'] == 0
    assert (source, invalid) == before


def test_support_repair_cannot_silently_change_roles_or_slot_order(monkeypatch):
    source, valid = case()
    invalid = deepcopy(valid)
    invalid['plan']['relation_plans'][1]['supporting_fact_ids'] = ['rf:0001']
    changed = deepcopy(valid)
    changed['plan']['candidate_anchors'][0] = 'rf:0002'
    outputs = iter([invalid, changed, valid])
    p = planner()
    monkeypatch.setattr(p, '_complete', lambda *args: json.dumps(next(outputs)))
    p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert p.calls[-1]['validation_attempts'][1]['errors'][0]['code'] == 'repair_changed_anchors'
    assert p.calls[-1]['contract_repair_retries'] == 2


def test_transport_and_business_retries_have_separate_audit_counters(monkeypatch):
    source, valid = case()
    invalid = deepcopy(valid)
    invalid['plan']['relation_plans'][1]['supporting_fact_ids'] = ['rf:0001']
    values = iter([ConnectionResetError('reset'), invalid, TimeoutError('timeout'), valid])
    p = planner()

    def complete(*args):
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)

    monkeypatch.setattr(p, '_complete', complete)
    p.plan(SimpleNamespace(label='Place'), 'image', source, 5)
    assert p.calls[-1]['transport_retries'] == 2
    assert p.calls[-1]['contract_repair_retries'] == 1
    assert [e['contract_attempt'] for e in p.calls[-1]['network_attempts']] == [0, 0, 1, 1]


@pytest.mark.parametrize('field,value', [('max_transport_retries', -1), ('max_transport_retries', True),
    ('stage_timeout_seconds', 0), ('stage_timeout_seconds', float('inf')), ('retry_backoff_seconds', -1)])
def test_invalid_transport_limits_rejected(field, value):
    kwargs = {field: value}
    if field == 'retry_backoff_seconds':
        with pytest.raises(ValueError):
            GPT4oMiniRelationPlanner(QwenConfig('gpt-4o-mini', 'http://localhost', ''), **kwargs)
    else:
        with pytest.raises(ValueError):
            planner(**kwargs)


def qwen_generator():
    return QwenLocalImagePromptGenerator(QwenConfig('Qwen/Qwen2-VL-7B-Instruct', 'http://localhost:12345/v1', 'EMPTY'))


def reply(content, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}


@pytest.mark.parametrize('bad,finish', [('{"positive_prompt":"broken', 'length'),
    (json.dumps({'positive_prompt': 'old movie film scene ' * 5}), 'stop')])
def test_local_qwen_repair_uses_bounded_json_and_no_invalid_response(monkeypatch, bad, finish):
    good = json.dumps({'positive_prompt': 'A grounded black and white city scene', 'used_text_cues': [], 'omitted_risky_details': []})
    g = qwen_generator()
    requests = []

    def complete(prompt, **kwargs):
        requests.append((prompt, kwargs))
        if len(requests) == 1:
            return reply(bad, finish), 1
        assert bad not in prompt
        assert kwargs['structured_repair']
        assert 'SOURCE_JSON:' in prompt
        return reply(good), 1

    monkeypatch.setattr(g, '_request_json', complete)
    result = g.generate('Example Film', 'film', 'director', 'identity', ['A supported fact'])
    assert result['positive_prompt'] == json.loads(good)['positive_prompt']
    call = g.consume_last_call('Q1', 'image_003')
    assert call['metadata']['contract_repair_retries'] == 1
    assert call['image_prompt_repair_version'] == 'qwen_image_prompt_source_rebuild_v1'


def test_local_qwen_repair_budget_is_bounded_without_local_json_salvage(monkeypatch):
    g = qwen_generator()
    requests = []

    def complete(prompt, **kwargs):
        requests.append(prompt)
        return reply('{"positive_prompt":"broken', 'length'), 1

    monkeypatch.setattr(g, '_request_json', complete)
    with pytest.raises(json.JSONDecodeError) as caught:
        g.generate('Film', 'film', 'director', 'identity', ['Supported fact'])
    assert len(requests) == 3
    assert caught.value.model_call['metadata']['contract_repair_retries'] == 2
    assert len(caught.value.model_call['validation_attempts']) == 3


def test_json_guidance_is_sent_only_on_local_qwen_repair(monkeypatch):
    g = qwen_generator()
    bodies = []
    responses = iter([reply('{"positive_prompt":"broken', 'length'), reply(json.dumps({
        'positive_prompt': 'An evidence grounded city scene', 'used_text_cues': [], 'omitted_risky_details': []}))])

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(next(responses)).encode()

    def urlopen(req, **kwargs):
        bodies.append(json.loads(req.data))
        return Response()

    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', urlopen)
    g.generate('City', 'city', 'located in', 'identity', ['A supplied fact'])
    assert 'guided_json' not in bodies[0] and 'repetition_penalty' not in bodies[0]
    assert bodies[1]['guided_json']['properties']['positive_prompt']['maxLength'] == 400
    assert bodies[1]['repetition_penalty'] == 1.1
    assert bodies[1]['guided_decoding_backend'] == 'lm-format-enforcer'
    assert bodies[0]['max_tokens'] == bodies[1]['max_tokens'] == g.config.image_prompt_max_token
