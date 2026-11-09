from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_wn18rr_parallel_completion import dataset
from test_stage1_source_bound import reply
from test_triple_modeling import example
from test_sdxl_adapter import FakePipeline
from mmkgc.adapters.source_text_triples import parse_source_response, source_text_task, source_extraction_report
from mmkgc.adapters.sdxl import SDXLConfig, SDXLImageGenerator
from mmkgc.retry_context import retry_scope, current_retry, RetrySeededGenerator
from mmkgc.generation_prompts import build_image_focus_prompt, build_image_description_prompt, build_generation_prompt, generation_payload
from scripts import run_wn18rr_completion_batch as batch
from scripts import run_wn18rr_parallel_completion as parallel
from scripts import run_wn18rr_retry_completion as retry


def test_escape_container_preserves_json_string_escapes_and_rejects_fragments():
    value=reply();value['uncertain_aspects']=[{'aspect':'literal path','reason':r'C:\new\tree'}]
    raw=json.dumps(value,indent=2).replace('\n',r'\n')
    parsed,removed=parse_source_response('<escape>'+raw+'</escape>')
    assert removed and parsed==value
    for bad in ['<escape>'+raw[:-4]+'</escape>','<escape>'+raw+' '+raw+'</escape>']:
        with pytest.raises(ValueError):parse_source_response(bad)


@pytest.mark.parametrize('annotation',[{'subject':'background','relation':'hops','object':None},'unspecified exclusion'])
def test_incomplete_exclusion_stays_audit_only_without_discarding_valid_facts(annotation):
    _,_,inputs,_=example();task=source_text_task(inputs)
    response=reply();response['excluded_triples']=[annotation]
    before=deepcopy(response);canonical=task.decode(response)
    assert len(canonical['extracted_triples'])==2 and response==before
    report=source_extraction_report(task,response,inputs)
    assert report['excluded_triples'][0]['raw_annotation']==annotation
    assert report['excluded_triples'][0]['reason'] is None


def test_retry_context_is_owned_by_entity_and_does_not_leak_between_workers():
    def worker(key):
        with retry_scope({'entity_key':key}):
            assert current_retry()['entity_key']==key
        assert current_retry() is None
    with ThreadPoolExecutor(max_workers=128) as pool:list(pool.map(worker,range(256)))


def test_retry_guidance_survives_focus_description_and_compression():
    payload={'entity':{'label':'sense','type':'entity'},'source_facts':[], 'source_evidence':[],
             'retry_diagnostics':{'judge_feedback':[{'ranking_reason':'wrong meaning'}],
                                  'rendering_strategy':'clear action illustration'}}
    payload=generation_payload('sense','entity','means','core',[],payload)
    for prompt in [build_image_focus_prompt(payload),build_image_description_prompt(payload),
                   build_generation_prompt('image',{**payload,'generation_step':'compress'})]:
        assert 'wrong meaning' in prompt and 'diagnostics, not new evidence' in prompt


def test_round_and_entity_seeds_change_without_loading_more_sdxl_weights(tmp_path):
    pipe=FakePipeline();loads=[]
    def factory(config):loads.append(config);return pipe
    generator=SDXLImageGenerator(SDXLConfig(model_dir=tmp_path,width=32,height=32,device='cpu'),factory)
    def draw(entity,round_number,namespace=1):
        return RetrySeededGenerator(generator,round_number,namespace).generate(entity,'image_001','subject','')
    a=draw('one',1);again=draw('one',1);b=draw('one',2);c=draw('two',1);d=draw('one',1,2)
    assert a.metadata['seed']==again.metadata['seed']
    assert len({x.metadata['seed'] for x in [a,b,c,d]})==4
    assert all(x.candidate_id=='image_001' for x in [a,b,c,d])
    assert len(loads)==1


def baseline(dataset):
    source=parallel.prepare(dataset/'source',workers=1)
    rows=batch.read(source/'manifest.json')
    path=source/'kept.png';path.write_bytes(b'original export bytes')
    rows[0]={**rows[0],'status':'exported','export':{'path':str(path),'sha256':batch.digest(path),'candidate_id':'image_001'}}
    rows[1]={**rows[1],'status':'no_usable_candidate','completion_reason':'no_accepted_export'}
    batch.save(source/'summary.json',{'rows':rows,'counts':{'exported':1,'no_usable_candidate':1}})
    batch.save(source/'workers/00/sdxl-attempts.json',[{'entity_key':'00000003','candidate_id':f'image_{i}'} for i in range(3)])
    return source,path


def test_outer_retry_skips_successes_carries_budget_and_stops_on_first_export(dataset,monkeypatch):
    source,kept=baseline(dataset);before=kept.read_bytes()
    root=dataset/'retry';retry.prepare(root,source,workers=1,max_rounds=3)
    seen=[]
    def execute(round_root,*,shared_local_models):
        rows=batch.read(round_root/'manifest.json');seen.append((round_root.name,[x['entity_key'] for x in rows],id(shared_local_models)))
        assert [x['entity_key'] for x in rows]==['00000003']
        row={**rows[0],'status':'no_usable_candidate'}
        if round_root.name=='02':
            path=round_root/'fresh.png';path.write_bytes(b'new accepted export')
            row.update(status='exported',export={'path':str(path),'candidate_id':'image_001','sha256':batch.digest(path)})
        batch.save(round_root/'summary.json',{'rows':[row],'counts':{row['status']:1}})
        batch.save(round_root/'workers/00/sdxl-attempts.json',[{'entity_key':'00000003','candidate_id':f'image_{i}'} for i in range(3)])
        batch.save(round_root/'finished.json',{'counts':{row['status']:1}})
    monkeypatch.setattr(parallel,'run',execute)
    result=retry.run(root)
    assert result['counts']=={'exported':2} and result['new_exports']==1 and result['remaining']==0
    assert len(seen)==2 and len({x[2] for x in seen})==1
    assert kept.read_bytes()==before
    assert batch.read(root/'cumulative-draws.json')['00000003']==9
    assert batch.read(root/'rounds/02/workers/00/runtime-settings.json')['model_config']=='configs/gpt6-luna-retry-image.yaml'
    retry.run(root)
    assert len(seen)==2


def test_no_usable_candidates_retry_until_configured_budget_then_remain_explicit(dataset,monkeypatch):
    source,_=baseline(dataset);root=dataset/'retry';retry.prepare(root,source,workers=1,max_rounds=2)
    calls=[]
    def execute(round_root,**kwargs):
        calls.append(round_root.name);row=batch.read(round_root/'manifest.json')[0]
        batch.save(round_root/'summary.json',{'rows':[{**row,'status':'no_usable_candidate'}],'counts':{'no_usable_candidate':1}})
        batch.save(round_root/'finished.json',{})
    monkeypatch.setattr(parallel,'run',execute)
    result=retry.run(root)
    assert calls==['01','02'] and result['remaining']==1
    assert batch.read(root/'phase.json')['phase']=='retry_rounds_exhausted'
    retry.run(root);assert calls==['01','02']
