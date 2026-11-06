"""Final entity writing sees owned original pixels together with selected facts."""
import base64
import hashlib
import json

import pytest

from test_image_knowledge import fixture, transport
from test_entity_visual_features import assessment, feature, response, pixel_reply, planner, run
from mmkgc.adapters.qwen import QwenConfig, QwenLocalTextGenerator
from mmkgc.generation_prompts import TEXT_VISION_PROMPT_VERSION
from mmkgc.ranking.image_input import ImageInputError
from mmkgc.stages.candidate_planning import apply_fact_selection
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.generation import relation_alternation_generation, _model_generation_context


def prepared(tmp_path, monkeypatch):
    target, path = fixture(tmp_path)
    requests = transport(monkeypatch, [json.dumps(assessment()), 'RAW_CAPTION_NOT_FOR_FINAL_WRITER',
        json.dumps(response([feature()])), pixel_reply(), 'The associated image shows a blue subject.'])
    knowledge = run(planner(tmp_path), target)
    selection, reasoning = compile_plan_context(knowledge, apply_fact_selection(knowledge,
        {'status':'ready','selected_fact_ids':[knowledge['relation_facts'][0]['fact_id']]}, 'text', 8), 'text')
    return target, path, knowledge, selection, reasoning, requests


def generator(root, enabled=True):
    return QwenLocalTextGenerator(QwenConfig('gpt-6-luna','https://vision.example.test/v1',
        'private-test-key',text_use_source_images=enabled), dataset_root=root)


def test_entity_name_relations_and_exact_original_pixels_reach_final_writer(tmp_path, monkeypatch):
    target, path, knowledge, selection, reasoning, requests = prepared(tmp_path, monkeypatch)
    client = generator(tmp_path)
    result = relation_alternation_generation(target,'text',knowledge,reasoning,selection,client,None,None,
                                            candidate_budget=1)
    parts = requests[-1]['messages'][-1]['content']
    assert [p['type'] for p in parts] == ['text','text','image_url']
    payload = json.loads(parts[0]['text'].split('INPUT_JSON:\n')[1])
    assert payload['entity']['label'] == 'Scene'
    assert payload['source_facts'][0]['relation'] == 'visual feature'
    assert payload['source_facts'][0]['object'] == 'blue'
    assert 'PRIVATE_' not in parts[0]['text'] and 'RAW_CAPTION_NOT_FOR_FINAL_WRITER' not in parts[0]['text']
    assert base64.b64decode(parts[-1]['image_url']['url'].split(',',1)[1]) == path.read_bytes()
    call = result['model_calls'][-1]
    assert call['prompt_version'] == TEXT_VISION_PROMPT_VERSION
    assert call['image_input_metadata'][0]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert call['image_input_metadata'][0]['admission']['identity_verified'] is False
    assert 'data:image' not in json.dumps(call)


def test_facts_only_profile_keeps_the_previous_text_transport(tmp_path, monkeypatch):
    target, _, knowledge, selection, reasoning, requests = prepared(tmp_path, monkeypatch)
    result = relation_alternation_generation(target,'text',knowledge,reasoning,selection,
        generator(tmp_path,False),None,None,candidate_budget=1)
    assert isinstance(requests[-1]['messages'][-1]['content'], str)
    assert 'image_input_metadata' not in result['model_calls'][-1]


def test_rejected_or_unselected_images_are_not_attached(tmp_path, monkeypatch):
    _, _, knowledge, selection, reasoning, _ = prepared(tmp_path, monkeypatch)
    item = selection['fine_grained_relation_selection']['candidates'][0]
    knowledge['image_modeling']['entity_match']['status'] = 'unrelated'
    context = _model_generation_context(knowledge,item,reasoning,'text','','',include_source_images=True)
    assert 'source_image_inputs' not in context
    knowledge['image_modeling']['entity_match']['status'] = 'compatible'
    knowledge['image_modeling']['original_image_ref'] = 'not_selected_original'
    context = _model_generation_context(knowledge,item,reasoning,'text','','',include_source_images=True)
    assert 'source_image_inputs' not in context
    context = _model_generation_context(knowledge,item,reasoning,'image','','',include_source_images=True)
    assert 'source_image_inputs' not in context


@pytest.mark.parametrize('problem', ['changed_pixels','outside_root','missing_root','duplicate'])
def test_invalid_source_binding_fails_before_api_and_preserves_failure(tmp_path, monkeypatch, problem):
    _, path, knowledge, selection, reasoning, requests = prepared(tmp_path, monkeypatch)
    item = selection['fine_grained_relation_selection']['candidates'][0]
    context = _model_generation_context(knowledge,item,reasoning,'text','','',include_source_images=True)
    client = generator(None if problem == 'missing_root' else tmp_path)
    if problem == 'changed_pixels': path.write_bytes(path.read_bytes()+b'changed')
    if problem == 'outside_root':
        outside=tmp_path.parent/(tmp_path.name+'-outside.png');outside.write_bytes(path.read_bytes())
        context['source_image_inputs'][0]['asset']=str(outside)
    if problem == 'duplicate': context['source_image_inputs'] *= 2
    with pytest.raises(ImageInputError):
        client.generate_with_context('Scene','entity','visual feature','',[],context=context)
    assert len(requests) == 4
    call = client.consume_last_call('TEST_SCENE','text_001')
    assert call['status'] == 'failed' and call['metadata']['error_type'] == 'ImageInputError'


def test_only_supported_boolean_config_values_enable_image_transport():
    for value in ('false', 1, None):
        with pytest.raises(ValueError):
            QwenConfig('model','http://localhost/v1','key',text_use_source_images=value)
