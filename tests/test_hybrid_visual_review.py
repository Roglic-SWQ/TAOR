"""Local extraction and independent image review must stay on distinct model routes."""
import json

import pytest

from test_image_knowledge import fixture, transport, text_and_image
from test_entity_visual_features import assessment, pixel_reply, run
from test_compact_visual_contract import feature_reply
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.adapters.image_feature_check import check_prompt
from mmkgc.adapters.image_knowledge import IMAGE_REF, DESCRIPTION_REF
from mmkgc.io.writer import RunWriter
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT


def client(root):
    return QwenLocalSemanticStagePlanner(
        QwenConfig('Qwen/Qwen2-VL-7B-Instruct','http://localhost/v1','EMPTY'), dataset_root=root,
        visual_verifier_config=QwenConfig('independent-test','https://review.example.test/v1',
                                         'review-test-key',api_response_format='json_object'))


def capture(monkeypatch, replies):
    requests=transport(monkeypatch,replies)
    from mmkgc.adapters.qwen import request
    original=request.urlopen
    urls=[]
    def send(req,**kwargs):
        urls.append(req.full_url)
        return original(req,**kwargs)
    monkeypatch.setattr(request,'urlopen',send)
    return requests,urls


def test_hybrid_only_admission_and_reobservation_use_api(tmp_path,monkeypatch):
    target,_=fixture(tmp_path)
    requests,urls=capture(monkeypatch,[json.dumps(assessment()),'A blue circle.',
        json.dumps(feature_reply()),pixel_reply()])
    planner=client(tmp_path);result=run(planner,target)
    assert urls==['https://review.example.test/v1/chat/completions',
                  'http://localhost/v1/chat/completions','http://localhost/v1/chat/completions',
                  'https://review.example.test/v1/chat/completions']
    assert [row['model'] for row in requests]==['independent-test','Qwen/Qwen2-VL-7B-Instruct',
                                               'Qwen/Qwen2-VL-7B-Instruct','independent-test']
    assert all(requests[i]['response_format']=={'type':'json_object'} for i in (0,3))
    assert 'features' in requests[2]['guided_json']['properties']
    assert len({text_and_image(request)[1] for request in requests})==1
    policy=result['stage1_v3']['visual_fact_policy']['pixel_recheck_reviewer']
    assert policy=={'model_name':'independent-test','endpoint_host':'review.example.test',
                    'separate_from_extractor':True,'performed':True}
    assert [c['model_name'] for c in planner.calls]==[r['model'] for r in requests]
    assert 'review-test-key' not in json.dumps(planner.calls)
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_api_rejection_never_calls_local_model(tmp_path,monkeypatch):
    target,_=fixture(tmp_path)
    requests,urls=capture(monkeypatch,[json.dumps(assessment('unrelated'))])
    result=run(client(tmp_path),target)
    assert len(requests)==1 and urls[0].startswith('https://review.example.test/')
    assert result['relation_facts']==[]
    assert result['stage1_v3']['visual_fact_policy']['pixel_recheck_reviewer']['performed'] is False


def test_failed_review_keeps_local_success_and_all_calls_without_releasing_unchecked_facts(tmp_path,monkeypatch):
    target,_=fixture(tmp_path)
    capture(monkeypatch,[json.dumps(assessment()),'A blue circle.',json.dumps(feature_reply()),'{}','{}'])
    planner=client(tmp_path)
    with pytest.raises(ValueError) as error:
        run(planner,target)
    assert len(error.value.model_calls)==4
    assert len(error.value.model_calls[-1]['network_attempts'])==2
    assert len(error.value.model_calls[-1]['contract_failures'])==2
    assert error.value.partial_knowledge['relation_facts'][0]['object']=='blue'
    writer=RunWriter(tmp_path,'failed',{})
    writer.record_failure(target,error.value);writer.write_summary()
    failure=json.loads((tmp_path/'failed/failures.json').read_text())[0]
    assert failure['partial_knowledge']['source_validation_status']=='not_completed'
    assert len(failure['model_calls'])==4
    assert not list((tmp_path/'failed').rglob('3_relation_alternation_generation.json'))
    assert 'review-test-key' not in json.dumps(failure)


def test_pixel_review_payload_cites_only_originals():
    prompt=check_prompt([{'observation_id':'mf:1:visual:0','relation':'coat','object':'brown',
                         'evidence_refs':[IMAGE_REF,DESCRIPTION_REF]}],{IMAGE_REF})
    payload=json.loads(prompt.split('PROPOSED_PROPERTIES:\n')[1])
    assert payload[0]['evidence_refs']==[IMAGE_REF]
    assert DESCRIPTION_REF not in prompt
