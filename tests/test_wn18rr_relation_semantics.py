"""Native edge direction must survive model prompts without leaking unselected paths."""
import json
from copy import deepcopy

from test_wn18rr_data import fixture
from mmkgc.data import WN18RRDataStore
from mmkgc.adapters.image_knowledge import with_image_description, image_assessment_prompt, IMAGE_REF
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response
from mmkgc.stages.triples import model_input_view
from mmkgc.stages.candidate_planning import apply_fact_selection, fact_selection_prompt
from mmkgc.stages.plan_context import compile_plan_context, ranking_knowledge
from mmkgc.stages.generation import _model_generation_context
from mmkgc.generation_prompts import generation_payload
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.scorers.gpt4omini_judge import build_text_judge_prompt


def prepared(tmp_path):
    store=WN18RRDataStore(fixture(tmp_path))
    record=store.build_record(0,'text')
    inputs=build_knowledge_input(record,'text',coarse_relation_filtering(record,'text',catalogue_only=True))
    inputs=with_image_description(inputs,'Unverified source picture',
        {'evidence_id':IMAGE_REF,'asset':'a.JPEG','sha256':'0'*64},'description:1')
    knowledge=apply_knowledge_response(inputs,{
        'entity_profile':{'primary_type':'entity','core_identity':record.label,'evidence_refs':[]},
        'extracted_triples':[],'uncertain_aspects':[]})
    return record,inputs,knowledge


def test_relation_meaning_preserves_original_incoming_edge(tmp_path):
    record,inputs,knowledge=prepared(tmp_path)
    fact=next(f for f in knowledge['relation_facts'] if f['relation']=='hypernym')
    assert (fact['subject'],fact['relation'],fact['object'],fact['direction'])==(
        'neighbor','hypernym','target lemma','in')
    assert fact['relation_meaning']==record.relation_semantics['hypernym']
    assert 'broader category' in fact['relation_meaning']
    assert 'has part' in record.relation_semantics
    _,raw=fact_selection_prompt(record.label,'text',knowledge,8)
    assert fact['relation_meaning'] in raw
    # Graph semantics aid admission, but never supply answers to pixel extraction.
    gate=image_assessment_prompt(inputs,{'evidence_id':IMAGE_REF})
    assert fact['relation_meaning'] in gate
    assert 'relation_semantics' not in model_input_view(inputs)
    assert fact['relation_meaning'] not in json.dumps(model_input_view(inputs))
    assert 'PRIVATE' not in gate and 'HIDDEN' not in gate


def test_unselected_path_content_stays_in_audit_not_generation(tmp_path):
    record,inputs,knowledge=prepared(tmp_path)
    fact=next(f for f in knowledge['relation_facts'] if f['relation']=='hypernym')
    knowledge['evidence_catalog'].append({'evidence_id':'path:unselected', 'source':'stage0_path',
                                         'content':{'terminal_entities':['UNSELECTED_PATH_DETAIL']}})
    fact['evidence_refs'].append('path:unselected')
    selection,reasoning=compile_plan_context(knowledge,apply_fact_selection(knowledge,
        {'status':'ready','selected_fact_ids':[fact['fact_id']]},'text',8),'text')
    context=_model_generation_context(knowledge,selection['fine_grained_relation_selection']['candidates'][0],
                                      reasoning,'text','','')
    payload=generation_payload(record.label,'entity','hypernym','',[],context)
    assert payload['source_facts'][0]['relation_meaning']==fact['relation_meaning']
    assert 'UNSELECTED_PATH_DETAIL' not in json.dumps(payload)
    assert 'UNSELECTED_PATH_DETAIL' in json.dumps(knowledge)
    assert 'path:unselected' in {item['evidence_id'] for item in payload['source_evidence']}
    ranked=build_ranking_context(record.entity_key,record.label,'text',inputs['existing_modalities'],
        ranking_knowledge(knowledge,reasoning),reasoning,
        {'relation_alternation_generation':{'text_candidates':[]}}, {})
    assert ranked.to_prompt_payload()['supported_facts'][0]['relation_meaning']==fact['relation_meaning']


def test_each_judge_check_receives_its_edge_meaning_without_rewriting_archived_targets(tmp_path):
    record,inputs,knowledge=prepared(tmp_path)
    fact=next(f for f in knowledge['relation_facts'] if f['relation']=='hypernym')
    selection,reasoning=compile_plan_context(knowledge,apply_fact_selection(knowledge,
        {'status':'ready','selected_fact_ids':[fact['fact_id']]},'text',8),'text')
    targets=deepcopy(selection['fine_grained_relation_selection']['candidates'][0]['verification_targets'])
    candidate={'candidate_id':'text_001','target_modality':'text','verification_targets':targets,
               'generated_content':{'type':'text','value':'A neighbor is a kind of the target.'}}
    before=deepcopy(candidate)
    context=build_ranking_context(record.entity_key,record.label,'text',inputs['existing_modalities'],
        ranking_knowledge(knowledge,reasoning),reasoning,
        {'relation_alternation_generation':{'text_candidates':[candidate]}},{})
    prompt=build_text_judge_prompt(context,candidate,use_rubrics=True)
    payload=json.loads(prompt.split('INPUT_JSON:\n')[1])
    assert payload['CHECK_TARGETS'][0]['relation_meaning']==fact['relation_meaning']
    assert payload['CHECK_TARGETS'][0]['check_id']==targets[0]['check_id']
    assert payload['CHECK_TARGETS'][0]['subject']=='neighbor'
    assert payload['CHECK_TARGETS'][0]['object']=='target lemma'
    assert candidate==before and 'relation_meaning' not in targets[0]
