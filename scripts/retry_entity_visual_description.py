"""Retry entity-relevant visual descriptions on frozen WN18RR originals.

Admission and blind captions are reused, fingerprinted inputs. Feature extraction,
pixel checks, semantic selection, writing and Stage 4 are rerun; no target text.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import time

from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner, QwenLocalTextGenerator, load_qwen_config
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.adapters.image_knowledge import image_modeling_entries, load_source_images, with_image_assessment, image_assessment_schema
from mmkgc.adapters.image_batches import image_batch_input, merge_image_batches
from mmkgc.adapters.media_knowledge import SourceMedia
from mmkgc.data import build_dataset_store
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response
from mmkgc.stages.plan_context import compile_plan_context
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT
from mmkgc.visual_description import TASK
from scripts.run_wn18rr_api_preflight import api_ranker
from scripts.run_wn18rr_completion_batch import read,save,digest,ROOT,DATASET

SOURCE = ROOT/'outputs/wn18rr-completion-20-20261107-r2'
KEYS = ('01838038','02855793','04151581','04362821','10338231','10257221')


def freeze(root,reuse_extraction=None):
    root.mkdir(parents=True, exist_ok=False)
    old = read(SOURCE/'manifest.json')
    rows=[]
    for key in KEYS:
        row=next(r for r in old if r['entity_key']==key)
        state=read(SOURCE/'state'/f'text-{key}.json')
        assert row['missing_mode']=='text' and state['status']=='exported'
        rows.append({**row,'source_entity_dir':state['entity_dir'],
                     'source_preparation':str(SOURCE/f'attempts/text-{key}/001/prepared.json')})
        if reuse_extraction is not None:
            previous=next(r for r in read(reuse_extraction/'manifest.json') if r['entity_key']==key)
            assert previous['images']==row['images']
            rows[-1]['reused_api_knowledge']=str((reuse_extraction/'api'/key/'checked-knowledge.json').resolve())
            rows[-1]['reused_api_calls']=str((reuse_extraction/'api'/key/'extractor-calls.json').resolve())
    save(root/'manifest.json',rows)
    files={str(p):digest(p) for row in rows for p in [Path(row['source_preparation']),
        Path(row['source_entity_dir'])/'3_relation_alternation_generation.json']}
    for row in rows:
        files.update(read(SOURCE/'state'/f"text-{row['entity_key']}.json")['artifact_sha256'])
    files.update(read(SOURCE/'binding.json')['files'])
    for row in rows:
        for key in ('reused_api_knowledge','reused_api_calls'):
            if row.get(key):files[row[key]]=digest(row[key])
    files.update({i['path']:i['sha256'] for row in rows for i in row['images']})
    files[str(DATASET/'train.tsv')]=digest(DATASET/'train.tsv')
    sources={str(p.relative_to(ROOT)):digest(p) for folder in ('mmkgc','scripts','configs')
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.sh'}}
    save(root/'binding.json',{'files':files,'sources':sources,'manifest_sha256':digest(root/'manifest.json'),
        'reused_admission_and_blind_captions':True,'hidden_target_text_used':False})
    print('Frozen six entity-description controls.',flush=True)


def validate(root):
    binding=read(root/'binding.json')
    assert digest(root/'manifest.json')==binding['manifest_sha256']
    assert all(digest(p)==sha for p,sha in binding['files'].items())
    assert all(digest(ROOT/p)==sha for p,sha in binding['sources'].items())


def reextract(record, previous, extractor):
    inputs=build_knowledge_input(record,'text',previous.coarse)
    inputs['generation_task']=TASK
    entries=image_modeling_entries(previous.knowledge['image_modeling'])
    originals=load_source_images(record,DATASET)
    assert {m['evidence_id']:m['sha256'] for _,m in originals}=={
        e['original_image_ref']:e['original_image']['sha256'] for e in entries}
    for entry in entries:
        assessment={k:entry['entity_match'][k] for k in image_assessment_schema()['properties']}
        inputs=with_image_assessment(inputs,assessment,entry['original_image'],entry['entity_match']['model_call_ref'],
            description=entry['description'],description_call_ref=entry['description_model_call_ref'])
    admitted={e['original_image_ref'] for e in entries if e['entity_match']['status']=='compatible'}
    media=[SourceMedia('image',metadata,image=image) for image,metadata in originals if metadata['evidence_id'] in admitted]
    deadline=time.perf_counter()+180
    subset=image_batch_input(inputs,admitted)
    if not media:
        partial=apply_knowledge_response(subset,{'entity_profile':{'primary_type':'entity','core_identity':record.label,
            'evidence_refs':[]},'extracted_triples':[],'uncertain_aspects':[]})
    else:
        context={'deadline':deadline,'media_inputs':media,
            'media_input_metadata':[{'modality':'image',**s.metadata} for s in media],
            'image_description_model_call_refs':{e['original_image_ref']:e['description_model_call_ref']
                for e in entries if e['original_image_ref'] in admitted}}
        token=KNOWLEDGE_REQUEST_CONTEXT.set(context)
        try:
            local=('qwen' in extractor.config.model_name.lower())
            partial=(extractor._compact_visual_knowledge(subset,context,deadline) if local else
                     extractor._refine('stage1_knowledge',record,'text',subset,
                         lambda response:apply_knowledge_response(subset,response)))
        finally:
            KNOWLEDGE_REQUEST_CONTEXT.reset(token)
    knowledge=merge_image_batches(inputs,[partial])
    return knowledge,media,deadline


def case(row,root,phase,shared_pipeline=None):
    key=row['entity_key'];folder=root/phase/key;folder.mkdir(parents=True,exist_ok=False)
    store=build_dataset_store('wn18rr',DATASET,graph_split='train')
    record=store.build_record(row['entity_id'],'text')
    assert not record.description and not record.existing_modalities.get('text')
    previous,_=load_saved_run(row['source_entity_dir'],record,1,missing_mode='text')
    api=load_qwen_config(ROOT/'configs/gpt6-luna-visual-description.yaml')
    config=(api if phase=='api' else load_qwen_config(ROOT/'configs/qwen7b-visual-description.yaml'))
    extractor=QwenLocalSemanticStagePlanner(config,timeout_seconds=120,knowledge_stage_timeout_seconds=180,
        dataset_root=DATASET,visual_verifier_config=api)
    selector=OpenAICompatibleRelationPlanner(api,max_selected_facts=4,timeout_seconds=120)
    writer=RunWriter(folder,'run',{'description_task':TASK,'generation_model':config.model_name,
        'selection_model':api.model_name,'stage4_model':api.model_name,'graph_split':'train',
        'graph_relations_in_description_input':False,'reused_admission_and_blind_captions':True,
        'ground_truth_used':False,'local_neural_scoring':phase!='api',
        'feature_source':'frozen_api_semantic_features' if phase=='writer' else 'rerun_extraction'})
    summary={'entity_key':key,'label':record.label,'phase':phase,'status':'running'}
    try:
        if phase=='writer':
            frozen=read(root/'api'/key/'prepared.json')
            knowledge=frozen['knowledge']
            selection,reasoning=frozen['selection'],frozen['reasoning']
            source_calls=[*read(root/'api'/key/'extractor-calls.json'),*read(root/'api'/key/'selector-calls.json')]
            reused_calls=[{**c,'reused_source_run':str(root/'api'/key)} for c in source_calls]
            summary.update(feature_extraction_model='gpt-6-luna',reused_api_extraction=True)
        elif phase=='api' and row.get('reused_api_knowledge'):
            knowledge=read(row['reused_api_knowledge'])
            assert knowledge['generation_task']==TASK and knowledge['entity']['entity_key']==key
            extractor.calls.extend([{**c,'reused_extraction_source':row['reused_api_knowledge']}
                                    for c in read(row['reused_api_calls'])])
            summary['reused_api_extraction']=True
        else:
            knowledge,media,deadline=reextract(record,previous,extractor)
            save(folder/'proposed-knowledge.json',knowledge)
            knowledge=extractor._verify_visual_features(knowledge,media,key,deadline)
        save(folder/'checked-knowledge.json',knowledge)
        summary['proposed_features']=sum(bool(f.get('visual_observations')) for f in knowledge['relation_facts'])
        if phase!='writer':
            selection=selector.plan(record,'text',knowledge,1)
            selection,reasoning=compile_plan_context(knowledge,selection,'text')
            reused_calls=[*extractor.calls,*selector.calls]
        prepared=PreparedEntity(record,'text',previous.coarse,knowledge,selection,reasoning,reused_calls)
        save(folder/'prepared.json',{'knowledge':knowledge,'selection':selection,'reasoning':reasoning})
        chosen=selection['fine_grained_relation_selection']
        summary.update(selected_features=[{'fact_id':f['fact_id'],'relation':f['relation'],'object':f['object'],
                'visual_observations':f['visual_observations']} for f in reasoning['supported_facts']])
        if chosen['status']=='insufficient_evidence':
            summary.update(status='insufficient_entity_visual_evidence',texts=[],
                new_model_calls=sum(not(c.get('reused_extraction_source') or c.get('reused_source_run')) for c in reused_calls))
        else:
            if shared_pipeline is None:
                ranker,_=api_ranker(api,120)
                pipeline=MMKGCPipeline(candidate_budget=1,
                    text_generator=QwenLocalTextGenerator(config,timeout_seconds=120,dataset_root=DATASET),
                    ranking_engine=ranker,dataset_root=DATASET)
            else:
                pipeline=shared_pipeline
            result=pipeline.finish(prepared)
            writer.write_entity(result)
            save(folder/'raw-ranking.json',result['kg_aware_candidate_consistency_ranking'])
            texts=result['relation_alternation_generation']['relation_alternation_generation']['text_candidates']
            summary.update(status=result['final']['ranking_status'],final=result['final'],
                texts=[{'candidate_id':c['candidate_id'],'text':c['generated_content']['value']} for c in texts],
                entity_dir=str(writer.entities_dir/key))
    except Exception as exc:
        writer.record_failure(record,exc)
        summary.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        save(folder/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),
            'partial_knowledge':getattr(exc,'partial_knowledge',None)})
    finally:
        writer.write_summary()
        save(folder/'extractor-calls.json',extractor.calls);save(folder/'selector-calls.json',selector.calls)
        save(folder/'summary.json',summary)
    print(json.dumps({k:summary.get(k) for k in ('entity_key','label','status','proposed_features')}) ,flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--phase',choices=('api','native','writer'),default='api')
    parser.add_argument('--reuse-api-extraction',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.prepare:
        freeze(root,args.reuse_api_extraction);return
    validate(root);(root/args.phase).mkdir(exist_ok=False)
    rows=read(root/'manifest.json');results=[]
    if args.phase=='api':
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(case,row,root,args.phase) for row in rows]
            for future in as_completed(futures):
                results.append(future.result());save(root/f'{args.phase}-summary.json',results)
    else:
        from mmkgc.defaults import build_default_mainline_runtime
        runtime=build_default_mainline_runtime(missing_mode='text',candidate_budget=1,
            model_config=ROOT/'configs/qwen7b-visual-description.yaml',dataset_root=DATASET,
            stage1_model_config=ROOT/'configs/qwen7b-visual-description.yaml',
            stage2_model_config=ROOT/'configs/gpt6-luna-visual-description.yaml',
            stage1_visual_verifier_config=ROOT/'configs/gpt6-luna-visual-description.yaml',
            stage1_reranker_model_dir='/medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B',
            model_call_timeout_seconds=120,stage4_config=ROOT/'configs/stage4-gpt6-luna-gpu0.yaml')
        for row in rows:
            results.append(case(row,root,args.phase,runtime.pipeline));save(root/f'{args.phase}-summary.json',results)
    validate(root)


if __name__=='__main__':
    main()
