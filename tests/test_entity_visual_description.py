from copy import deepcopy
import json
import pytest

from mmkgc.visual_description import TASK
from mmkgc.adapters.qwen import QwenConfig,load_qwen_config
from mmkgc.adapters.visual_feature_contract import response_schema
from mmkgc.adapters.image_knowledge import image_assessment_prompt
from mmkgc.stages.candidate_planning import planning_facts,fact_selection_prompt,apply_fact_selection
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.generation_prompts import generation_payload,build_generation_prompt
from mmkgc.observation import repair_operations
from scripts.run_wn18rr_completion_batch import publication_ready
from test_candidate_fact_plan import source


def visual_source():
    value=source();value['generation_task']=TASK
    value['stage1_v3']['entity_profile']={'primary_type':'entity','core_identity':'Museum','evidence_refs':[]}
    value['stage1_v3']['generation_task']=TASK
    fact=value['relation_facts'][1]
    fact.update(source_modalities=['image'],generation_eligible=True,
        visual_observations=[{'basis':'direct','scope':'target','feature_kind':'entity_feature',
            'claim_scope':'depicted_instance','uncertainty':'','uncertainty_scope':'none',
            'evidence_refs':fact['evidence_refs']}])
    return value


def test_selection_has_only_visual_facts_and_does_not_mutate_complete_graph():
    value=visual_source();before=deepcopy(value)
    system,prompt=fact_selection_prompt('Museum','text',value,4)
    payload=json.loads(prompt)
    assert payload['allowed_fact_ids']==['rf:2']
    assert len(planning_facts(value))==1
    assert 'architect' not in prompt and 'Jane' not in prompt
    assert value==before and len(value['relation_facts'])==4
    with pytest.raises(ValueError):
        apply_fact_selection(value,{'status':'ready','selected_fact_ids':['rf:4']},'text',4)


def test_checked_visual_features_are_optional_coverage_and_do_not_force_relations():
    value=visual_source()
    selection=apply_fact_selection(value,{'status':'ready','selected_fact_ids':['rf:2']},'text',4)
    compiled,reasoning=compile_plan_context(value,selection,'text')
    candidate=compiled['fine_grained_relation_selection']['candidates'][0]
    assert 'must_include' not in candidate['generation_context']
    assert 'requirement' not in candidate['verification_targets'][0]
    observation={'modality':'text','fact_checks':[{'status':'missing','confidence':0.99,'verification_mode':'direct','evidence_refs':['evidence:2']}]}
    assert repair_operations(observation)==[]
    context={**candidate['generation_context'],'source_facts':candidate['source_facts']}
    payload=generation_payload('Museum','entity','facade material','none',[],context)
    prompt=build_generation_prompt('text',payload)
    assert json.loads(prompt.split('INPUT_JSON:\n')[1])['prompt_version']==TASK
    assert 'primary_relation' not in payload['task']
    assert 'Cover every distinct supported selected fact' not in prompt


def test_name_only_admission_omits_graph_assertions():
    inputs={'generation_task':TASK,'entity':{'label':'Museum'},
            'fact_candidates':[{'subject':'Museum','relation':'architect','object':'Jane','evidence_refs':['kg:1']}]}
    before=deepcopy(inputs)
    prompt=image_assessment_prompt(inputs,{'evidence_id':'original:1'})
    target=json.loads(prompt.split('TARGET_CONTEXT:\n')[1])
    assert target['source_assertions']==[] and 'Jane' not in prompt
    assert inputs==before


def test_feature_ceiling_is_joint_not_per_image_inventory():
    inputs={'generation_task':TASK,'evidence_catalog':[
        {'evidence_id':f'image:{i}','source':'existing_modality','representation':'original_image'} for i in range(4)]}
    assert response_schema(inputs)['properties']['features']['maxItems']==6
    assert 'minItems' not in response_schema(inputs)['properties']['features']


def test_visual_export_requires_supported_content_but_not_every_optional_feature():
    candidate={'gate': {'decision':'pass'}, 'description_task':TASK,'observation':{'fact_checks':[
        {'status':'supported'},
        {'status':'missing'}]}}
    assert publication_ready(candidate)
    candidate['gate']['decision']='reject'
    assert not publication_ready(candidate)
    candidate['observation']={'fact_checks':[]}
    assert not publication_ready(candidate)


def test_visual_task_is_explicit_opt_in():
    assert not QwenConfig('test','http://localhost','EMPTY').entity_visual_description
    assert load_qwen_config('configs/qwen7b-visual-description.yaml').entity_visual_description
    assert not load_qwen_config('configs/qwen7b-visual-description.yaml').text_use_source_images
    with pytest.raises(ValueError):
        QwenConfig('test','http://localhost','EMPTY',entity_visual_description='yes')


def test_description_task_survives_ranking_sanitization_and_activates_its_judge():
    from types import SimpleNamespace
    from mmkgc.ranking.context import _sanitize_candidates
    from mmkgc.ranking.scorers.gpt4omini_judge import _build_rubric_prompt
    original={'candidate_id':'text_001','target_modality':'text','description_task':TASK,
              'generated_content':{'type':'text','value':'The pictured object is elongated.'}}
    candidate=_sanitize_candidates([original],'text')[0]
    assert candidate['description_task']==TASK
    context=SimpleNamespace(entity_key='Q0',entity_label='Object',to_prompt_payload=lambda:{})
    prompt=_build_rubric_prompt(context,candidate,'text')
    assert 'ENTITY-FOCUSED VISUAL DESCRIPTION' in prompt
    assert 'occupation or religious role' in prompt


def test_description_task_survives_grounded_stage_writer_and_resume(tmp_path):
    from mmkgc.io.writer import RunWriter
    from mmkgc.io.resume import load_saved_run
    from mmkgc.pipeline import MMKGCPipeline,PreparedEntity
    from mmkgc.plan_schema import FACT_SELECTION_SCHEMA
    from mmkgc.stages.sources import public_sources
    from test_candidate_fact_plan import target
    from test_shared_fact_selection import Text
    value=visual_source();record=target()
    value['schema_version']=value['stage1_v3']['schema_version']='grounded_triples_v2'
    value['entity']={'entity_key':record.entity_key,'label':record.label}
    value['source_inputs']=public_sources(record,'text')[0]
    choice={'status':'ready','selected_fact_ids':['rf:2']}
    selection,reasoning=compile_plan_context(value,apply_fact_selection(value,choice,'text',4),'text')
    call={'stage':'stage2_selection','status':'applied','response_schema_version':FACT_SELECTION_SCHEMA,
          'max_selected_facts':4,'response_text':json.dumps(choice)}
    prepared=PreparedEntity(record,'text',{},value,selection,reasoning,[call])
    pipeline=MMKGCPipeline(candidate_budget=1,text_generator=Text(),skip_stage4=True)
    writer=RunWriter(tmp_path,'saved',{});writer.write_entity(pipeline.finish(prepared))
    entity=writer.entities_dir/record.entity_key
    stage=json.loads((entity/'1_knowledge_modeling.json').read_text())
    assert stage['data']['generation_task']==TASK
    restored,generation=load_saved_run(entity,record,1,missing_mode='text')
    assert restored.knowledge['generation_task']==TASK
    assert all('requirement' not in t for t in generation['relation_alternation_generation']['text_candidates'][0]['verification_targets'])


def test_visual_judge_reference_does_not_reintroduce_unselected_caption_details():
    from mmkgc.stages.plan_context import ranking_knowledge
    value=visual_source()
    value['evidence_catalog'][1].update(modality='image',content='A facade, parked cars and a blue sky.',
        entity_match={'status':'compatible','identity_verified':False,'reason':'extra unselected prose'})
    fact=deepcopy(value['relation_facts'][1])
    fact['visual_source_checks']=[{'status':'supported','observed':'Red facade and cars.',
        'reason':'extra source details','evidence_refs':['evidence:2']}]
    before=deepcopy(value)
    result=ranking_knowledge(value,{'supported_facts':[fact]})
    assert value==before
    assert 'content' not in result['evidence_catalog'][1]
    assert 'observed' not in result['relation_facts'][0]['visual_source_checks'][0]
    assert result['evidence_catalog'][1]['entity_match']=={'status':'compatible','identity_verified':False}
