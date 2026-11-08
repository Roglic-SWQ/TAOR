"""Replay the observed intermittent gateway 400 without loosening ordinary 4xx."""
from io import BytesIO
import json
import time
from urllib.error import HTTPError

import httpx
import pytest

from mmkgc.adapters.http_retry import retryable_http_error
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.ranking.factory import _OpenAICompatibleCompletionClient
from mmkgc.ranking.scorers.gpt4omini_judge import GPT4OMiniRankingJudge


GATEWAY_ERROR = {'error': {'code': 'invalid_request_format', 'type': 'invalid_request_error',
    'message': 'The request format is invalid or unsupported by the upstream provider.'}}


@pytest.mark.parametrize('host,code,body,expected', [
    ('api.invalid', 400, GATEWAY_ERROR, True),
    ('localhost', 400, GATEWAY_ERROR, False),
    ('api.invalid', 400, {'error': {'code': 'invalid_request_format'}}, False),
    ('api.invalid', 400, {'error': {'code': 'invalid_json'}}, False),
    ('api.invalid', 401, GATEWAY_ERROR, False),
    ('api.invalid', 422, GATEWAY_ERROR, False),
    ('api.invalid', 503, {}, True),
])
def test_only_observed_remote_gateway_error_is_added(host, code, body, expected):
    raw = json.dumps(body).encode()
    exc = HTTPError('https://' + host + '/v1', code, 'bad', {}, BytesIO(raw))
    assert retryable_http_error(exc) is expected
    assert retryable_http_error(exc) is expected
    assert exc.read() == raw


def mock_httpx(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: real(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False))


@pytest.mark.parametrize('always_fails', [False, True])
def test_stage2_replay_is_bounded_and_preserves_provider_error(monkeypatch, always_fails):
    sent = []
    async def handler(req):
        sent.append(json.loads(req.content))
        if len(sent) == 1 or always_fails:
            return httpx.Response(400, json=GATEWAY_ERROR)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                   'message': {'content': '{"ok":true}'}}]})
    mock_httpx(monkeypatch, handler)
    planner = OpenAICompatibleRelationPlanner(QwenConfig('gpt-test', 'https://api.invalid/v1', 'key'),
        retry_backoff_seconds=0)
    call = {'network_attempts': [], 'transport_retries': 0}
    execute = lambda: planner._complete_with_retries(
        [{'role': 'user', 'content': 'Return JSON.'}], {}, call, 0, time.perf_counter()+5)
    if always_fails:
        with pytest.raises(HTTPError) as caught:
            execute()
        assert json.loads(caught.value.read()) == GATEWAY_ERROR
    else:
        assert json.loads(execute()) == {'ok': True}
    assert len(sent) == 2 and sent[0] == sent[1]
    assert call['transport_retries'] == 1
    assert call['network_attempts'][0]['provider_error']['code'] == 'invalid_request_format'


def test_stage1_retries_same_request_once(monkeypatch):
    from test_triple_modeling import example
    from test_stage1_source_bound import reply
    response = reply()
    response['entity_profile']['evidence_refs'] = ['existing:text']
    for fact in response['extracted_triples']:
        fact['evidence_refs'] = ['existing:text']
    sent = []
    def send(req, timeout):
        sent.append(req.data)
        if len(sent) == 1:
            raise HTTPError(req.full_url, 400, 'bad', {}, BytesIO(json.dumps(GATEWAY_ERROR).encode()))
        return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(response)}}]}).encode())
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', send)
    monkeypatch.setattr('mmkgc.adapters.qwen.time.sleep', lambda _: None)
    record, coarse, _, _ = example()
    planner = QwenLocalSemanticStagePlanner(QwenConfig('gpt-test', 'https://api.invalid/v1', 'key'))
    planner.build_knowledge(record, 'image', coarse)
    assert len(sent) == 2 and sent[0] == sent[1]
    assert [x['status'] for x in planner.calls[-1]['network_attempts']] == ['failed', 'received']
    assert planner.calls[-1]['contract_repair_retries'] == 0


def test_judge_worker_http_keeps_gateway_body_for_one_retry(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    sent = []
    async def handler(req):
        sent.append(req.content)
        if len(sent) == 1:
            return httpx.Response(400, json=GATEWAY_ERROR)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                                    'message': {'content': 'test-ok'}}]})
    mock_httpx(monkeypatch, handler)
    judge = GPT4OMiniRankingJudge(_OpenAICompatibleCompletionClient(
        QwenConfig('gpt-test', 'https://api.invalid/v1', 'key'), timeout_seconds=1),
        max_retries=1, retry_backoff_seconds=0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(judge._complete_with_retries, 'Test', image_input=None,
                             kind='image', candidate_id='image_001').result(timeout=3)
    assert result == 'test-ok' and len(sent) == 2 and sent[0] == sent[1]
