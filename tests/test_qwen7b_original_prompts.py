from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import threading

import pytest

from mmkgc.adapters.qwen import (
    QwenConfig, QwenLocalImagePromptGenerator, QwenLocalSemanticStagePlanner,
    QwenLocalTextGenerator, load_qwen_config,
)
from mmkgc.data import EntityRecord
from mmkgc.generation_prompts import PROMPT_VERSION, QWEN_PROMPT_VERSION
from mmkgc.stages.generation import relation_alternation_generation, safer_relation_alternation_generation
from mmkgc.stages.modeling import apply_knowledge_response, build_knowledge_input
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.relation_planning import apply_candidate_slot_plans, apply_relation_plans


ROOT = Path(__file__).resolve().parents[1]
# Request digests were recorded by executing the original backup, including its
# own semantic planner, generation scheduler and HTTP adapters, without HTTP.
REFERENCE = json.loads((Path(__file__).parent / 'fixtures/qwen7b_original_prompts.json').read_text())


def _hash_request(body):
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _capture(monkeypatch, responses):
    requests = []

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': self.content}}]}).encode()

    def request(req, **kwargs):
        body = json.loads(req.data)
        assert body['model'] == 'Qwen/Qwen2-VL-7B-Instruct'
        assert 'PRIVATE_TARGET' not in json.dumps(body)
        requests.append(body)
        return Response(responses[len(requests) - 1])

    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', request)
    return requests


@pytest.mark.parametrize('case', REFERENCE['semantic_cases'], ids=lambda case: case['id'])
def test_semantic_requests_match_original_backup_including_repair(monkeypatch, case):
    target = EntityRecord(**deepcopy(REFERENCE['records'][case['mode']]))
    coarse = deepcopy(REFERENCE['coarse'])
    config_name = 'qwen7b-stage1.yaml' if case['stage'] == 'stage1_knowledge' else 'qwen7b-mainline.yaml'
    planner = QwenLocalSemanticStagePlanner(load_qwen_config(ROOT / 'configs' / config_name))
    requests = _capture(monkeypatch, case['responses'])
    if case['stage'] == 'stage0_coarse':
        result = planner.refine_coarse(target, case['mode'], coarse)
    else:
        result = planner.build_knowledge(target, case['mode'], coarse)
    assert [_hash_request(body) for body in requests] == case['expected_request_sha256']
    assert result['semantic_refinement']['prompt_version'] == QWEN_PROMPT_VERSION
    assert planner.calls[-1]['prompt_version'] == QWEN_PROMPT_VERSION
    payload = json.loads(requests[0]['messages'][1]['content'].split('INPUT_JSON:\n', 1)[1])
    assert 'prompt_version' not in payload


def _generation_inputs(mode):
    target = EntityRecord(**deepcopy(REFERENCE['records'][mode]))
    setup = deepcopy(REFERENCE['generation_setups'][mode])
    inputs = build_knowledge_input(target, mode, deepcopy(REFERENCE['coarse']))
    knowledge = apply_knowledge_response(inputs, setup['knowledge_response'])
    selection, reasoning = compile_plan_context(
        knowledge, apply_relation_plans(knowledge, setup['relation_plan_response'], mode, 1), mode,
    )
    return target, knowledge, reasoning, selection


@pytest.mark.parametrize('case', REFERENCE['generation_cases'], ids=lambda case: case['id'])
def test_generation_requests_and_final_sdxl_prompt_match_original_backup(monkeypatch, case):
    target, knowledge, reasoning, selection = _generation_inputs(case['mode'])
    before = deepcopy((knowledge, reasoning, selection))
    config = load_qwen_config(ROOT / 'configs/qwen7b-mainline.yaml')
    requests = _capture(monkeypatch, case['responses'])
    args = (target, case['mode'], knowledge, reasoning, selection,
            QwenLocalTextGenerator(config), QwenLocalImagePromptGenerator(config))
    if case['retry']:
        result = safer_relation_alternation_generation(
            *args, candidate_budget=1, failed_ranking={}, stage3_feedback=deepcopy(case['feedback']),
        )
    else:
        result = relation_alternation_generation(*args, candidate_budget=1)
    candidate = result['relation_alternation_generation'][f"{case['mode']}_candidates"][0]
    assert [_hash_request(body) for body in requests] == case['expected_request_sha256']
    assert candidate['generation_brief'] == case['expected_generation_brief']
    assert candidate['generated_content'] == case['expected_generated_content']
    assert candidate['metadata']['prompt_version'] == QWEN_PROMPT_VERSION
    assert result['model_calls'][0]['prompt_version'] == QWEN_PROMPT_VERSION
    assert (knowledge, reasoning, selection) == before


@pytest.mark.parametrize('mode', ['text', 'image'])
def test_original_prompt_profile_keeps_configurable_stage3_concurrency(monkeypatch, mode):
    target, knowledge, reasoning, selection = _generation_inputs(mode)
    config = replace(load_qwen_config(ROOT / 'configs/qwen7b-mainline.yaml'), stage3_max_concurrency=5)
    plans = deepcopy(REFERENCE['generation_setups'][mode]['relation_plan_response']['relation_plans'])
    for plan in plans:
        del plan['candidate_quota']
    first, second = [fact['fact_id'] for fact in knowledge['relation_facts'][:2]]
    anchors = [second, first, second, first, first]
    selection, reasoning = compile_plan_context(knowledge, apply_candidate_slot_plans(knowledge, {
        'plan': {'status': 'ready', 'relation_plans': plans, 'candidate_anchors': anchors},
    }, mode, 5), mode)
    barrier = threading.Barrier(5)
    lock = threading.Lock()
    active = peak = 0

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': self.content}}]}).encode()

    def request(req, **kwargs):
        nonlocal active, peak
        body = json.loads(req.data)
        prompt = body['messages'][1]['content']
        angle = (re.search(r'Use the (.*?) angle', prompt).group(1) if mode == 'text'
                 else json.loads(prompt.split('Input JSON:\n', 1)[1])['generation_angle'])
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        content = 'Text ' + angle if mode == 'text' else json.dumps({'positive_prompt': 'Image ' + angle})
        return Response(content)

    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen', request)
    result = relation_alternation_generation(
        target, mode, knowledge, reasoning, selection,
        QwenLocalTextGenerator(config), QwenLocalImagePromptGenerator(config), candidate_budget=5,
    )
    candidates = result['relation_alternation_generation'][f'{mode}_candidates']
    by_id = {fact['fact_id']: fact for fact in knowledge['relation_facts']}
    assert [candidate['primary_relation'] for candidate in candidates] == [by_id[ref]['relation'] for ref in anchors]
    assert peak == 5
    assert [candidate['candidate_id'] for candidate in candidates] == [f'{mode}_{index:03d}' for index in range(1, 6)]
    for candidate, call in zip(candidates, result['model_calls'], strict=True):
        assert call['candidate_id'] == candidate['candidate_id']
        assert candidate['generation_angle'] in call['response_text']
        assert call['prompt_version'] == QWEN_PROMPT_VERSION


@pytest.mark.parametrize('value', ['', 'unknown', None, [], True])
def test_invalid_prompt_profile_fails_before_request(value):
    with pytest.raises(ValueError, match='prompt_profile'):
        QwenConfig('Qwen/Qwen2-VL-7B-Instruct', 'http://localhost/v1', 'EMPTY', prompt_profile=value)


def test_gpt_profile_retains_v2_prompts(monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://gpt.example.test/v1')
    config = load_qwen_config(ROOT / 'configs/gpt4omini-mainline.yaml')
    assert config.prompt_profile == PROMPT_VERSION
    assert QwenLocalTextGenerator(config).uses_structured_prompts
    assert QwenLocalImagePromptGenerator(config).uses_structured_prompts
