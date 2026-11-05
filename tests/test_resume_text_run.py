from dataclasses import replace
import json
import pytest
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.plan_schema import FACT_SELECTION_SCHEMA
from test_shared_fact_selection import selected, reply, Text
from test_candidate_fact_plan import target


def saved(tmp_path):
    knowledge,selection,reasoning=selected('text')
    call={'stage':'stage2_selection','status':'applied','response_schema_version':FACT_SELECTION_SCHEMA,
          'max_selected_facts':8,'response_text':json.dumps(reply())}
    pipeline=MMKGCPipeline(candidate_budget=1,text_generator=Text(),skip_stage4=True)
    result=pipeline.finish(PreparedEntity(target(),'text',{},knowledge,selection,reasoning,[call]))
    writer=RunWriter(tmp_path,'source',{})
    writer.write_entity(result)
    return next(writer.entities_dir.iterdir()),result


def test_saved_text_is_restored_without_regeneration_or_changing_checks(tmp_path):
    root,result=saved(tmp_path)
    prepared,generation=load_saved_run(root,target(),1,missing_mode='text')
    original=result['relation_alternation_generation']['relation_alternation_generation']['text_candidates'][0]
    restored=generation['relation_alternation_generation']['text_candidates'][0]
    assert prepared.missing_mode=='text'
    assert restored['generated_content']==original['generated_content']
    assert restored['verification_targets']==original['verification_targets']
    assert restored['source_fact_refs']==original['source_fact_refs']
    assert generation['generated_image_artifacts']==[]


@pytest.mark.parametrize('change',['checks','budget','mode','source'])
def test_saved_text_rejects_changed_references_or_inputs(tmp_path,change):
    root,_=saved(tmp_path)
    record=target();budget=1;mode='text'
    if change=='budget':budget=2
    elif change=='mode':mode='image'
    elif change=='source':record=replace(record,existing_modalities={'text':'changed','image':None})
    else:
        path=root/'3_relation_alternation_generation.json'
        d=json.loads(path.read_text());d['data']['text_candidates'][0]['verification_targets']=[]
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError):load_saved_run(root,record,budget,missing_mode=mode)
