"""Regression coverage for observed Stage 1 evidence and Stage 2 slot failures."""
from copy import deepcopy
import json
from types import SimpleNamespace

import jsonschema
import pytest

from mmkgc.adapters.json_contract import local_semantic_json_schema, semantic_output_token_cap
from mmkgc.adapters.qwen import QwenConfig, QwenLocalSemanticStagePlanner
from mmkgc.adapters.stage1_repair import targeted_knowledge_repair, merge_knowledge_repair
from mmkgc.adapters.qwen_stage2_repair import TASK_MARKER, targeted_repair_tasks, merge_targeted_response
from mmkgc.stages.modeling import KnowledgeResponseValidationError, knowledge_response_errors
from mmkgc.stages.semantic import StrictJsonStagePlanner, KNOWLEDGE_REQUEST_CONTEXT
from mmkgc.stages.relation_planning import RelationPlanValidationError, selection_prompt
from test_stage1_contract_repair import fixture, invalid
from test_stage2_qwen_targeted_repair import example, planner


def test_stage1_patches_only_bad_references_and_missing_empty_claims():
    record, coarse, inputs, valid = fixture()
    bad = deepcopy(valid)
    bad['entity_profile']['evidence_refs'] = ['invented']
    bad['key_attributes'] = [{'value': 'Example Player', 'type': 'identity', 'source_ref': 'invented'}]
    bad['relation_facts'][1]['claim'] = ''
    bad['relation_facts'].pop()
    original = deepcopy(bad)
    prompts = []
    ref = inputs['evidence_catalog'][0]['evidence_id']
    patch = {'fields': {'/entity_profile/evidence_refs': [ref], '/key_attributes/0/source_ref': ref},
             'facts': {'rf:0002': valid['relation_facts'][1]['claim'], 'rf:0004': valid['relation_facts'][3]['claim']}}
    def complete(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return json.dumps(bad)
        schema = local_semantic_json_schema(prompt)
        jsonschema.validate(patch, schema)
        assert semantic_output_token_cap(prompt) < 2560
        return json.dumps(patch)
    client = StrictJsonStagePlanner(complete, targeted_knowledge_repairs=True)
    result = client.build_knowledge(record, 'image', coarse)
    assert len(prompts) == 2
    assert bad == original
    merged = json.loads(client.calls[-1]['response_text'])
    assert merged['relation_facts'][0] == bad['relation_facts'][0]
    assert merged['relation_facts'][2] == bad['relation_facts'][2]
    assert merged['entity_profile']['core_identity'] == bad['entity_profile']['core_identity']
    assert merged['key_attributes'][0]['value'] == bad['key_attributes'][0]['value']
    assert not knowledge_response_errors(inputs, merged)
    assert len(result['relation_facts']) == 4
    assert result['relation_facts'][0]['subject'] == inputs['fact_candidates'][0]['subject']
    from mmkgc.pipeline import _semantic_model_calls
    exported = _semantic_model_calls(client,start_index=0)[-1]
    assert exported['knowledge_repair_version']=='stage1_targeted_source_repair_v4'
    assert exported['stage_timeout_seconds']==180
    assert exported['validation_attempts']==client.calls[-1]['validation_attempts']
    assert exported['targeted_repairs'][0]['fact_ids']==['rf:0002','rf:0004']


def test_stage1_schema_rejects_empty_claim_and_unknown_evidence():
    _, _, inputs, valid = fixture()
    # Include all visual fields required on the wire.
    valid['text_visual_profile'] = {k: [] for k in ['visual_scene_cues','context_cues','non_visual_facts']}
    schema = local_semantic_json_schema('"entity_profile" "text_visual_profile" "relation_facts"\nINPUT_JSON:\n' + json.dumps(inputs))
    jsonschema.validate(valid, schema)
    for field in ['claim','evidence','source_ref']:
        bad = deepcopy(valid)
        if field == 'claim': bad['relation_facts'][0]['claim'] = ''
        elif field == 'evidence': bad['entity_profile']['evidence_refs'] = ['made-up']
        else: bad['key_attributes'] = [{'value':'X','type':'identity','source_ref':'made-up'}]
        with pytest.raises(jsonschema.ValidationError): jsonschema.validate(bad, schema)


@pytest.mark.parametrize('raw', [
    '{"fields":{},"facts":{}}',
    '{"fields":{"/entity_profile/evidence_refs":[],"/entity_profile/evidence_refs":[]},"facts":{}}',
    '{"fields":{"/entity_profile/evidence_refs":[],"/entity_profile/core_identity":"forged"},"facts":{}}',
])
def test_stage1_patch_rejects_missing_duplicate_extra_keys_without_mutation(raw):
    _, _, inputs, valid = fixture()
    bad = invalid(valid,'evidence')
    task = targeted_knowledge_repair(inputs,bad,knowledge_response_errors(inputs,bad))
    before = deepcopy(bad)
    with pytest.raises(KnowledgeResponseValidationError): merge_knowledge_repair(bad,task,raw)
    assert bad == before


def test_stage1_wrong_patch_is_revalidated_instead_of_accepted():
    record, coarse, _, valid = fixture()
    replies = iter([json.dumps(invalid(valid,'evidence')),
                    '{"fields":{"/entity_profile/evidence_refs":["still-invented"]},"facts":{}}'])
    client = StrictJsonStagePlanner(lambda _:next(replies),targeted_knowledge_repairs=True,max_knowledge_repairs=1)
    with pytest.raises(KnowledgeResponseValidationError): client.build_knowledge(record,'image',coarse)
    assert client.calls[-1]['validation_errors'][0]['code'] == 'unknown_profile_evidence'


def test_stage1_total_deadline_applies_to_initial_response_and_prevents_repair(monkeypatch):
    record, coarse, _, valid = fixture()
    clock = [0.0]
    monkeypatch.setattr('mmkgc.stages.semantic.time.perf_counter',lambda:clock[0])
    calls = []
    def complete(prompt):
        calls.append(prompt); clock[0] += 5
        return json.dumps(invalid(valid,'missing'))
    client = StrictJsonStagePlanner(complete,knowledge_stage_timeout_seconds=4)
    with pytest.raises(TimeoutError,match='total time budget'): client.build_knowledge(record,'image',coarse)
    assert len(calls) == 1
    assert client.calls[-1]['contract_repair_stopped_reason'] == 'stage_time_budget_exhausted'
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def test_stage1_network_retry_cannot_restart_exhausted_stage_budget(monkeypatch):
    record, coarse, _, _ = fixture()
    clock = [0.0]; sent = []
    monkeypatch.setattr('mmkgc.stages.semantic.time.perf_counter',lambda:clock[0])
    def send(req,timeout):
        sent.append(timeout); clock[0] += timeout
        raise TimeoutError('test timeout')
    monkeypatch.setattr('mmkgc.adapters.qwen.request.urlopen',send)
    client = QwenLocalSemanticStagePlanner(QwenConfig('Qwen/test','http://localhost/v1','EMPTY'),
                                         timeout_seconds=180,knowledge_stage_timeout_seconds=5)
    with pytest.raises(TimeoutError):client.build_knowledge(record,'image',coarse)
    assert sent == [5]
    assert len(client.calls[-1]['network_attempts']) == 1
    assert KNOWLEDGE_REQUEST_CONTEXT.get() is None


def joint_case():
    knowledge, full = example(4)
    full['plan']['candidate_anchors'] = ['rf:0001','rf:0002','rf:0003']
    patch = {'primary_fact_ids':['rf:0001','rf:0002','rf:0003'], 'secondary_fact_ids':['rf:0004'],
             'reasons':{f'rf:{i:04d}':'Model selected this role from the supplied evidence.' for i in range(1,5)},
             'slots': {f'slot_{i:03d}':f'rf:{i:04d}' for i in range(1,4)}}
    return knowledge,full,patch


def test_four_primary_facts_with_three_slots_use_joint_model_patch(monkeypatch):
    knowledge, full, patch = joint_case()
    before = deepcopy(full); client = planner(); kinds=[]
    def complete(messages,schema):
        if TASK_MARKER not in schema:return json.dumps(full)
        kinds.append(schema[TASK_MARKER]); jsonschema.validate(patch,{k:v for k,v in schema.items() if k!=TASK_MARKER})
        return json.dumps(patch)
    monkeypatch.setattr(client,'_complete',complete)
    result = client.plan(SimpleNamespace(label='City'),'image',knowledge,3)
    assert kinds == ['roles_and_slots']
    assert full == before
    merged = json.loads(client.calls[-1]['response_text'])['plan']
    for original,new in zip(full['plan']['relation_plans'],merged['relation_plans']):
        assert new['fact_id'] == original['fact_id']
        assert new['evidence_refs'] == original['evidence_refs']
        assert new['supporting_fact_ids'] == original['supporting_fact_ids']
    assert merged['relation_plans'][3]['role']=='secondary'
    assert result['fine_grained_relation_selection']['quota_summary']['assigned_total']==3


def test_joint_patch_that_leaves_four_primary_facts_is_never_auto_fixed(monkeypatch):
    knowledge,full,patch=joint_case()
    patch['primary_fact_ids'].append('rf:0004')
    patch['secondary_fact_ids']=[]
    client=planner()
    monkeypatch.setattr(client,'_complete',lambda messages,schema:json.dumps(patch if TASK_MARKER in schema else full))
    with pytest.raises(RelationPlanValidationError):client.plan(SimpleNamespace(label='City'),'image',knowledge,3)
    assert any(e['code']=='invalid_role_partition' for e in client.calls[-1]['validation_errors'])


def test_joint_patch_cannot_change_evidence_or_omit_a_role():
    knowledge,full,patch=joint_case()
    _,source=selection_prompt('City','image',knowledge,3)
    errors=[{'code':'unanchored_primary','path':'/plan/relation_plans/3/role','message':'unanchored'}]
    task=targeted_repair_tasks(source,full,errors,3)[0]
    before=deepcopy(full)
    patch['evidence_refs']=['forged']
    with pytest.raises(RelationPlanValidationError):merge_targeted_response(full,task,json.dumps(patch))
    assert full==before
