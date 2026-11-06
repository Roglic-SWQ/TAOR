"""Freeze the existing ten-entity cohort, then compare versioned text writers.

API-only diagnostic: no GPU, local reranker, neural scoring or new images.
All originals and failures are retained. Both writers use the same frozen facts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner, QwenLocalTextGenerator, load_qwen_config
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.data import build_dataset_store
from mmkgc.io.writer import RunWriter, _safe_output
from mmkgc.pipeline import MMKGCPipeline, PreparedEntity
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.plan_context import compile_plan_context
from scripts.run_wn18rr_api_preflight import api_ranker


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_output(value), ensure_ascii=False, indent=2))


def freeze_preparation(record, config, dataset, folder, binding):
    folder.mkdir(parents=True, exist_ok=False)
    planner = QwenLocalSemanticStagePlanner(config, timeout_seconds=120,
        knowledge_stage_timeout_seconds=180, dataset_root=dataset, visual_verifier_config=config)
    selector = OpenAICompatibleRelationPlanner(config, timeout_seconds=120, max_selected_facts=64)
    try:
        coarse = coarse_relation_filtering(record, 'text', catalogue_only=True)
        knowledge = planner.build_knowledge(record, 'text', coarse)
        selection = selector.plan(record, 'text', knowledge, 1)
        selection, reasoning = compile_plan_context(knowledge, selection, 'text')
        assert all(digest(i['path']) == i['sha256'] for i in binding['images'])
        frozen = {'entity_key':record.entity_key, 'label':record.label, 'source_binding':binding,
                  'coarse':coarse, 'knowledge':knowledge, 'selection':selection, 'reasoning':reasoning,
                  'model_calls':[*planner.calls, *selector.calls]}
        write(folder/'prepared.json', frozen)
        selected = selection['fine_grained_relation_selection']
        return {'status':'ready', 'entity_key':record.entity_key, 'label':record.label,
                'selection_status':selected['status'], 'selected_fact_count':len(selected.get('selected_fact_ids', []))}
    except Exception as exc:
        failure = {'status':'failed','entity_key':record.entity_key,'label':record.label,
                   'error_type':type(exc).__name__, 'error':str(exc),
                   'model_calls':[*planner.calls, *selector.calls]}
        if getattr(exc, 'partial_knowledge', None):
            failure['partial_knowledge'] = {'source_validation_status':'not_completed',
                                            'data':exc.partial_knowledge}
        write(folder/'failure.json', failure)
        return {k:failure[k] for k in ('status','entity_key','label','error_type')}


def frozen_prompt_generator(config, dataset, builder):
    class BaselineWriter(QwenLocalTextGenerator):
        def _build_prompt(self, label, entity_type, relation, angle, facts, generation_context=None):
            return builder['build_generation_prompt']('text', builder['generation_payload'](
                label, entity_type, relation, angle, facts, generation_context))
    return BaselineWriter(config, timeout_seconds=120, dataset_root=dataset)


def compare_case(index, record, config, dataset, root, builder):
    frozen_path = root/'frozen'/record.entity_key/'prepared.json'
    if not frozen_path.exists():
        return {'entity_key':record.entity_key,'label':record.label,'status':'upstream_failed','arms':{}}
    frozen_sha = digest(frozen_path)
    f = json.loads(frozen_path.read_text())
    assert f['entity_key'] == record.entity_key
    assert digest(dataset/'train.tsv') == f['source_binding']['train_sha256']
    assert all(digest(i['path']) == i['sha256'] for i in f['source_binding']['images'])
    prepared = PreparedEntity(record,'text',f['coarse'],f['knowledge'],f['selection'],f['reasoning'],f['model_calls'])
    out = {'entity_key':record.entity_key, 'label':record.label, 'frozen_preparation_sha256':frozen_sha,
           'status':'completed', 'arms':{}}
    plans = prepared.selection['fine_grained_relation_selection']['candidates']
    if not plans:
        out['status'] = 'insufficient_evidence'
        write(root/'cases'/f'{record.entity_key}.json', out)
        return out
    panel = plans[0]['verification_targets']
    out['panel_sha256'] = hashlib.sha256(json.dumps(panel,sort_keys=True).encode()).hexdigest()
    # Alternate order in the already-frozen manifest, without selecting winners.
    arms = ['baseline_v1','new_v2'] if index % 2 == 0 else ['new_v2','baseline_v1']
    for arm in arms:
        writer = RunWriter(root/'runs',f'{arm}-{record.entity_key}', {
            'model_name':config.model_name,'local_gpu_inference':False,'local_neural_scoring':False,
            'local_reranker':False,'independent_judge':False,'panel_sha256':out['panel_sha256'],
            'frozen_preparation_sha256':frozen_sha,'arm':arm})
        ranker, _ = api_ranker(config, 120)
        generator = (frozen_prompt_generator(config,dataset,builder) if arm == 'baseline_v1'
                     else QwenLocalTextGenerator(config,timeout_seconds=120,dataset_root=dataset))
        pipeline = MMKGCPipeline(candidate_budget=1,text_generator=generator,
                                ranking_engine=ranker,dataset_root=dataset)
        try:
            result = pipeline.finish(deepcopy(prepared))
            writer.write_entity(result)
            ranked = result['kg_aware_candidate_consistency_ranking']
            generation = result['relation_alternation_generation']['relation_alternation_generation']
            status = 'completed' if result['final']['ranking_status'] == 'completed' else 'evaluation_incomplete'
            if status != 'completed':
                out['status'] = 'partial_failure'
            out['arms'][arm] = {'status':status,'final':result['final'],
                'texts':[{k:c[k] for k in ('candidate_id','generated_content')} for c in generation['text_candidates']],
                'reflect':{k:v for k,v in (ranked.get('stage3_regeneration') or {}).items()
                           if k in ('attempted','attempts','status','final_status')},
                'observations':[{'candidate_id':c['candidate_id'],'observation':c.get('observation'),
                    'rubrics':c.get('judge_rubrics'),'repair_eligible':c.get('repair_eligible')}
                    for c in ranked.get('text_ranking',{}).get('ranked_candidates',[])]}
        except Exception as exc:
            writer.record_failure(record, exc)
            out['arms'][arm] = {'status':'failed','error_type':type(exc).__name__}
            out['status'] = 'partial_failure'
        finally:
            writer.write_summary()
            write(root/'cases'/f'{record.entity_key}.json', out)
            print(json.dumps({'phase':'compare','entity_key':record.entity_key,'arm':arm,
                              'status':out['arms'][arm]['status']}),flush=True)
    assert digest(frozen_path) == frozen_sha
    return out


def reselect_case(record, config, dataset, root, *, refine_properties=False, reuse_properties=False):
    """Diagnose a selection change separately; never rewrite the shared A/B panel."""
    source = root/'frozen'/record.entity_key/'prepared.json'
    frozen_sha = digest(source)
    f = json.loads(source.read_text())
    assert digest(dataset/'train.tsv') == f['source_binding']['train_sha256']
    assert all(digest(i['path']) == i['sha256'] for i in f['source_binding']['images'])
    folder = root/('property-selection-controls' if reuse_properties else
                  'property-controls' if refine_properties else 'selection-controls')/record.entity_key
    folder.mkdir(parents=True,exist_ok=False)
    selector = OpenAICompatibleRelationPlanner(config,timeout_seconds=120,max_selected_facts=64)
    writer = RunWriter(folder,'new-selection-new-writer',{'source_frozen_preparation_sha256':frozen_sha,
        'model_name':config.model_name,'local_gpu_inference':False,'local_neural_scoring':False,
        'note':'New selection changes the reference panel; do not compare internal scores as a writing-only ablation.'})
    summary = {'entity_key':record.entity_key,'label':record.label,
               'source_frozen_preparation_sha256':frozen_sha,
               'old_selected_fact_ids':f['selection']['fine_grained_relation_selection']['selected_fact_ids']}
    planner = None
    try:
        knowledge = deepcopy(f['knowledge'])
        revised_calls = []
        if reuse_properties:
            property_source = root/'property-controls'/record.entity_key
            path = property_source/'checked-knowledge.json'
            knowledge = json.loads(path.read_text())
            revised_calls = json.loads((property_source/'property-model-calls.json').read_text())
            summary['checked_property_source_sha256'] = digest(path)
        if refine_properties:
            from mmkgc.adapters.image_knowledge import image_modeling_entries, load_source_images
            from mmkgc.adapters.image_knowledge import with_image_assessment, image_assessment_schema
            from mmkgc.adapters.media_knowledge import SourceMedia
            from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response
            from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT
            entries = image_modeling_entries(knowledge['image_modeling'])
            if not entries or any(e['entity_match']['status'] != 'compatible' for e in entries):
                raise ValueError('Frozen-caption property control requires admitted originals')
            images = load_source_images(record,dataset)
            assert {m['evidence_id']:m['sha256'] for _,m in images} == {
                e['original_image_ref']:e['original_image']['sha256'] for e in entries}
            inputs = build_knowledge_input(record,'text',f['coarse'])
            for entry in entries:
                assessment = {k:entry['entity_match'][k] for k in image_assessment_schema()['properties']}
                inputs = with_image_assessment(inputs,assessment,entry['original_image'],
                    entry['entity_match']['model_call_ref'],description=entry['description'],
                    description_call_ref=entry['description_model_call_ref'])
            planner = QwenLocalSemanticStagePlanner(config,timeout_seconds=120,dataset_root=dataset)
            deadline = time.perf_counter()+180
            media = [SourceMedia('image',metadata,image=image) for image,metadata in images]
            context = {'deadline':deadline,'media_inputs':media,
                       'media_input_metadata':[{'modality':'image',**m} for _,m in images],
                       'image_description_model_call_refs':{
                           e['original_image_ref']:e['description_model_call_ref'] for e in entries}}
            token = KNOWLEDGE_REQUEST_CONTEXT.set(context)
            try:
                knowledge = planner._refine('stage1_knowledge',record,'text',inputs,
                    lambda response:apply_knowledge_response(inputs,response))
            finally:
                KNOWLEDGE_REQUEST_CONTEXT.reset(token)
            write(folder/'proposed-knowledge.json',knowledge)
            for entry in entries:
                for reason in entry['entity_match']['uncertainties']:
                    knowledge['uncertainties'].append({'aspect':entry['original_image_ref'],'reason':reason})
            knowledge = planner._verify_visual_features(knowledge,media,record.entity_key,deadline)
            revised_calls = planner.calls
            write(folder/'checked-knowledge.json',knowledge)
            summary['control'] = 'same_originals_admission_and_blind_captions_atomic_properties'
        selection = selector.plan(record,'text',knowledge,1)
        selection, reasoning = compile_plan_context(knowledge,selection,'text')
        replaced = {'stage2_selection'} | ({'stage1_knowledge','stage1_image_feature_check'}
                                          if refine_properties or reuse_properties else set())
        calls = [c for c in f['model_calls'] if c.get('stage') not in replaced] + revised_calls + selector.calls
        prepared = PreparedEntity(record,'text',f['coarse'],knowledge,selection,reasoning,calls)
        write(folder/'prepared.json',{'entity_key':record.entity_key,'knowledge':knowledge,
                                     'selection':selection,'reasoning':reasoning,'model_calls':calls})
        ranker, _ = api_ranker(config,120)
        pipeline = MMKGCPipeline(candidate_budget=1,
            text_generator=QwenLocalTextGenerator(config,timeout_seconds=120,dataset_root=dataset),
            ranking_engine=ranker,dataset_root=dataset)
        result = pipeline.finish(prepared)
        writer.write_entity(result)
        summary.update(status=result['final']['ranking_status'],final=result['final'],
            new_selected_fact_ids=selection['fine_grained_relation_selection']['selected_fact_ids'],
            selected_facts=selection['fine_grained_relation_selection']['candidates'][0]['source_facts'],
            candidates=result['relation_alternation_generation']['relation_alternation_generation']['text_candidates'])
    except Exception as exc:
        writer.record_failure(record,exc)
        summary.update(status='failed',error_type=type(exc).__name__)
    finally:
        writer.write_summary()
        summary['original_preparation_unchanged'] = digest(source) == frozen_sha
        write(folder/'summary.json',summary)
        write(folder/'selection-model-calls.json',selector.calls)
        if planner is not None:
            write(folder/'property-model-calls.json',planner.calls)
    return summary


def rejudge_case(record, config, dataset, root, bad_text_source=None, probe_name=None):
    from mmkgc.io.resume import load_saved_run
    from mmkgc.ranking.context import build_ranking_context
    from mmkgc.stages.plan_context import ranking_knowledge
    source = (root/'runs'/f'baseline_v1-{record.entity_key}'/'entities'/record.entity_key
              if bad_text_source else root/'selection-controls'/record.entity_key/
              'new-selection-new-writer'/'entities'/record.entity_key)
    fingerprint = {str(p):digest(p) for p in source.iterdir() if p.is_file()}
    prepared, generation = load_saved_run(source,record,1,missing_mode='text')
    payload = generation['relation_alternation_generation']
    payload['text_candidates'] = [c for c in payload['text_candidates'] if c['candidate_id'] == 'text_001']
    payload.pop('stage3_regeneration',None)
    bad_sha = None
    if bad_text_source:
        bad_sha = digest(bad_text_source)
        old = json.loads(bad_text_source.read_text())['data']
        old = old.get('relation_alternation_generation',old)
        text = next(c['generated_content']['value'] for c in old['text_candidates'] if c['candidate_id']=='text_001')
        payload['text_candidates'][0]['generated_content']['value'] = text
    control_name = 'negative-paraphrase-controls' if bad_text_source else 'paraphrase-controls'
    if probe_name:
        control_name += '-'+probe_name
    folder = root/control_name/record.entity_key
    folder.mkdir(parents=True,exist_ok=False)
    context = build_ranking_context(record.entity_key,record.label,'text',record.existing_modalities,
        ranking_knowledge(prepared.knowledge,prepared.reasoning),prepared.reasoning,generation,{})
    ranker, _ = api_ranker(config,120)
    result = ranker.rank(context)
    write(folder/'ranking.json',result)
    summary = {'status':result['ranking_status'],'entity_key':record.entity_key,
        'mode':'frozen_original_text_rejudgment_no_generation','source_files':fingerprint,
        'bad_text_source':str(bad_text_source) if bad_text_source else None,'bad_text_source_sha256':bad_sha,
        'source_files_unchanged':all(digest(p)==sha for p,sha in fingerprint.items()),
        'observations':[c.get('observation') for c in result.get('text_ranking',{}).get('ranked_candidates',[])]}
    write(folder/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','compare','reselect','reextract','reselect_properties','rejudge'),required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--frozen-inputs',type=Path,
                        default=Path('outputs/wn18rr-real-test-20261106/ten-inputs.json'))
    parser.add_argument('--dataset-dir',type=Path,default=Path('datasets/wn18rr-import-20261106/WN18RR'))
    parser.add_argument('--model-config',type=Path,default=Path('configs/gpt6-luna-text-vision.yaml'))
    parser.add_argument('--baseline-commit',default='c43faa8')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--entity-key',action='append')
    parser.add_argument('--bad-text-source',type=Path)
    parser.add_argument('--probe-name')
    args = parser.parse_args()
    if args.probe_name and not all(c.isalnum() or c in '-_' for c in args.probe_name):
        parser.error('--probe-name must contain only letters, numbers, hyphens or underscores')
    if args.phase == 'prepare':
        args.output_root.mkdir(parents=True,exist_ok=False)
        cohort = json.loads(args.frozen_inputs.read_text())
        assert cohort['graph_split'] == 'train' and cohort['selected_before_new_model_outputs']
        write(args.output_root/'cohort.json',cohort)
    else:
        cohort = json.loads((args.output_root/'cohort.json').read_text())
    phase_name = 'negative-rejudge' if args.phase == 'rejudge' and args.bad_text_source else args.phase
    if args.probe_name:
        phase_name += '-'+args.probe_name
    phase_file = args.output_root/f'{phase_name}-summary.json'
    if phase_file.exists():
        raise FileExistsError('This phase already ran; use a new experiment directory')
    config = load_qwen_config(args.model_config)
    assert config.model_name == 'gpt-6-luna' and config.text_use_source_images
    store = build_dataset_store('wn18rr',args.dataset_dir,graph_split='train')
    records = [store.build_record(store.entity_uri_to_id[e['entity_key']],'text') for e in cohort['entities']]
    for record, binding in zip(records,cohort['entities']):
        assert record.label == binding['label']
        assert set(map(str,record.image_paths)) == {i['path'] for i in binding['images']}
        assert all(digest(i['path']) == i['sha256'] for i in binding['images'])
    builder = {}
    if args.phase == 'compare':
        source = subprocess.check_output(['/usr/bin/git','show',args.baseline_commit+':mmkgc/generation_prompts.py'])
        exec(compile(source,'frozen-baseline-generation-prompts.py','exec'),builder)
        write(args.output_root/'baseline-source.json',{'commit':args.baseline_commit,
              'sha256':hashlib.sha256(source).hexdigest()})
    sources = [*Path('mmkgc').rglob('*.py'),Path(__file__),args.model_config]
    write(args.output_root/f'{phase_name}-execution.json',{'source_sha256':{str(p):digest(p) for p in sources},
          'gpu_allocation':False,'workers':args.workers,'local_reranker':False,'local_neural_scoring':False})
    summary = {'phase':args.phase,'entities':{},'diagnostic_cohort_not_representative':True}
    write(phase_file,summary)
    if args.phase in ('reselect','reextract','reselect_properties','rejudge'):
        if not args.entity_key or not set(args.entity_key) <= {r.entity_key for r in records}:
            parser.error('--entity-key must identify an entity from the frozen cohort')
        for record in records:
            if record.entity_key not in args.entity_key:
                continue
            value = (rejudge_case(record,config,args.dataset_dir,args.output_root,args.bad_text_source,args.probe_name)
                     if args.phase == 'rejudge' else reselect_case(record,config,args.dataset_dir,args.output_root,
                                                                  refine_properties=args.phase == 'reextract',
                                                                  reuse_properties=args.phase == 'reselect_properties'))
            summary['entities'][record.entity_key] = value
            write(phase_file,summary)
            print(json.dumps({'phase':args.phase,'entity_key':record.entity_key,'status':value['status']}),flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for index,(record,binding) in enumerate(zip(records,cohort['entities'])):
            if args.phase == 'prepare':
                binding = {**binding,'train_sha256':digest(args.dataset_dir/'train.tsv')}
                future = pool.submit(freeze_preparation,record,config,args.dataset_dir,
                                     args.output_root/'frozen'/record.entity_key,binding)
            else:
                future = pool.submit(compare_case,index,record,config,args.dataset_dir,args.output_root,builder)
            futures[future] = record
        for future in as_completed(futures):
            record = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = {'entity_key':record.entity_key,'label':record.label,'status':'failed',
                         'error_type':type(exc).__name__}
            summary['entities'][record.entity_key] = value
            write(phase_file,summary)
            print(json.dumps({'phase':args.phase,'entity_key':record.entity_key,
                              'label':record.label,'status':value['status']}),flush=True)


if __name__ == '__main__':
    main()
