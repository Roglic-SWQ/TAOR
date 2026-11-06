"""The local visual transport bounds verbosity without changing source facts."""
import json

import jsonschema
import pytest

from test_image_knowledge import fixture, transport, text_and_image
from test_entity_visual_features import assessment, planner, run, pixel_reply
from mmkgc.adapters.image_knowledge import IMAGE_REF, DESCRIPTION_REF, with_image_assessment
from mmkgc.adapters import visual_feature_contract as contract
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response, KnowledgeResponseValidationError
from mmkgc.stages.semantic import Stage1CompletionError, KNOWLEDGE_REQUEST_CONTEXT


def feature_reply(refs=None):
    return {'features': [{'attribute':'color', 'value':'blue', 'kind':'entity_feature',
        'basis':'direct', 'uncertainty':'', 'evidence_refs':refs or [IMAGE_REF, DESCRIPTION_REF]}],
        'uncertain_aspects': []}


def source_inputs(tmp_path):
    target,_=fixture(tmp_path)
    return with_image_assessment(build_knowledge_input(target,'text',coarse_relation_filtering(target,'text')),
        assessment(), {'evidence_id':IMAGE_REF,'asset':'source.png','sha256':'0'*64}, 'describe:1')


def test_compact_properties_keep_canonical_source_contract(tmp_path):
    inputs=source_inputs(tmp_path)
    decoded=contract.decode(json.dumps(feature_reply()),inputs)
    result=apply_knowledge_response(inputs,decoded)
    fact=result['relation_facts'][0]
    assert (fact['subject'],fact['relation'],fact['object'])==('Scene','color','blue')
    assert fact['visual_observations'][0]['claim_scope']=='depicted_instance'
    assert set(fact['evidence_refs'])=={IMAGE_REF,DESCRIPTION_REF}
    assert 'Preserve the source' not in contract.prompt(inputs)
    assert contract.response_schema(inputs)['properties']['features']['maxItems']==8


def test_compact_schema_rejects_overflow_instead_of_truncating(tmp_path):
    inputs=source_inputs(tmp_path)
    value=feature_reply();value['features']*=9
    with pytest.raises(jsonschema.ValidationError):
        contract.decode(json.dumps(value),inputs)
    with pytest.raises(KnowledgeResponseValidationError):
        apply_knowledge_response(inputs,contract.decode(json.dumps(feature_reply([DESCRIPTION_REF])),inputs))


def test_compact_repair_keeps_small_contract_and_original_pixels(tmp_path,monkeypatch):
    target,_=fixture(tmp_path)
    requests=transport(monkeypatch,[json.dumps(assessment()),'A blue circle.','{}',
        json.dumps(feature_reply()),pixel_reply()])
    client=planner(tmp_path,'Qwen/Qwen2-VL-7B-Instruct')
    run(client,target)
    assert len(requests)==5
    assert requests[2]['guided_json']==requests[3]['guided_json']
    assert 'features' in requests[3]['guided_json']['properties']
    assert text_and_image(requests[2])[1]==text_and_image(requests[3])[1]
    assert client.calls[2]['contract_repair_retries']==1


@pytest.mark.parametrize('reply,code',[('', 'empty_response'),(('{}','length'),'output_token_limit')])
def test_compact_incomplete_output_is_not_json_repair(tmp_path,monkeypatch,reply,code):
    target,_=fixture(tmp_path)
    requests=transport(monkeypatch,[json.dumps(assessment()),'A blue circle.',reply])
    client=planner(tmp_path,'Qwen/Qwen2-VL-7B-Instruct')
    with pytest.raises(Stage1CompletionError) as error:
        run(client,target)
    assert error.value.code==code and len(requests)==3
    assert client.calls[-1]['completion_failure']==code
    assert client.calls[-1]['network_attempts'][0]['max_tokens']==4096
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None
