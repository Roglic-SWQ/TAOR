"""Compare API generation and repair on one frozen real WN18RR fact panel.

No upstream re-extraction, local neural scoring, GPU allocation or new pixels.
The repair arm starts from the actual saved failed native text, not a synthetic
bad candidate. Each arm retains its own calls, observations and failures.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from mmkgc.adapters.qwen import QwenLocalTextGenerator, QwenLocalImagePromptGenerator, load_qwen_config
from mmkgc.data import build_dataset_store
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter, _safe_output
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.stages.generation import _model_generation_context
from mmkgc.stages.triples import fact_text
from scripts.run_wn18rr_api_preflight import api_ranker


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_generation(saved):
    """Retain the actual initial candidate; never reuse its previous repair."""
    saved = deepcopy(saved)
    payload = saved['relation_alternation_generation']
    payload['text_candidates'] = [c for c in payload['text_candidates']
                                  if c['candidate_id'] == 'text_001']
    if len(payload['text_candidates']) != 1:
        raise ValueError('Expected exactly one saved initial native text')
    payload.pop('stage3_regeneration', None)
    saved['model_calls'] = [c for c in saved['model_calls'] if c.get('candidate_id') == 'text_001']
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-entity', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--task', choices=('text', 'image_prompt'), default='text')
    parser.add_argument('--dataset-dir', type=Path,
                        default=Path('datasets/wn18rr-import-20261106/WN18RR'))
    parser.add_argument('--model-config', type=Path, default=Path('configs/gpt6-luna-mainline.yaml'))
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    sources = {str(p.resolve()): fingerprint(p) for p in args.source_entity.iterdir() if p.is_file()}
    config = load_qwen_config(args.model_config)
    if config.model_name != 'gpt-6-luna':
        raise ValueError('This experiment requires the explicitly requested gpt-6-luna model')
    store = build_dataset_store('wn18rr', args.dataset_dir, graph_split='train')
    modality = 'image' if args.task == 'image_prompt' else 'text'
    record = store.build_record(store.entity_uri_to_id[args.source_entity.name], modality)
    prepared, saved = load_saved_run(args.source_entity, record, 1, missing_mode=modality)
    def save(name, value):
        (args.output_root/name).write_text(json.dumps(_safe_output(value), ensure_ascii=False, indent=2))
    source_paths = [*Path('mmkgc').rglob('*.py'), Path(__file__), args.model_config]
    save('execution.json', {'git_head':subprocess.check_output(['/usr/bin/git','rev-parse','HEAD'],text=True).strip(),
                           'source_sha256':{str(p):fingerprint(p) for p in source_paths}})
    plan = prepared.selection['fine_grained_relation_selection']['candidates'][0]
    panel = plan['verification_targets']
    panel_sha = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
    if args.task == 'image_prompt':
        generator = QwenLocalImagePromptGenerator(config, timeout_seconds=args.timeout)
        summary = {'task':args.task, 'model_name':config.model_name, 'panel_sha256':panel_sha,
                   'source_files':sources, 'new_pixels_generated':False,
                   'image_max_tokens':config.image_prompt_max_token}
        try:
            context = _model_generation_context(prepared.knowledge, plan, prepared.reasoning, 'image', '', '')
            summary.update(status='completed', prompt=generator.generate_with_context(
                record.label, prepared.knowledge['entity_profile']['entity_type'], plan['relation'], '',
                [fact_text(f) for f in plan['source_facts']], context=context))
        except Exception as exc:
            summary.update(status='failed', error_type=type(exc).__name__)
        finally:
            summary['model_call'] = generator.consume_last_call(record.entity_key, 'prompt_preflight_001')
            summary['source_files_unchanged'] = all(fingerprint(Path(p)) == sha for p,sha in sources.items())
            save('result.json', summary)
            print(json.dumps({k:summary[k] for k in ('task','status','model_name')}), flush=True)
        if summary['status'] != 'completed' or not summary['source_files_unchanged']:
            raise SystemExit(1)
        return
    initial = initial_generation(saved)
    save('frozen-panel.json', {'checks':panel, 'sha256':panel_sha, 'source_files':sources,
                             'initial_text':initial['relation_alternation_generation']['text_candidates'][0]})
    summary = {'model_name':config.model_name, 'scope':'fixed_panel_api_generation_and_repair',
               'local_gpu_inference':False, 'new_pixels_generated':False,
               'local_neural_scoring':False, 'independent_judge':False,
               'judge_model_name':config.model_name, 'panel_sha256':panel_sha, 'arms':{}}
    try:
        for arm in ('repair_native_failure', 'fresh_api_generation'):
            writer = RunWriter(args.output_root, arm, {**summary, 'arm':arm,
                               'source_entity':str(args.source_entity), 'max_tokens':config.max_token})
            ranker, _ = api_ranker(config, args.timeout)
            generator = QwenLocalTextGenerator(config, timeout_seconds=args.timeout)
            pipeline = MMKGCPipeline(candidate_budget=1, text_generator=generator,
                                     ranking_engine=ranker, dataset_root=args.dataset_dir)
            try:
                result = pipeline.finish(deepcopy(prepared),
                    generation_result=deepcopy(initial) if arm == 'repair_native_failure' else None)
                writer.write_entity(result)
                ranking = result['kg_aware_candidate_consistency_ranking']
                summary['arms'][arm] = {
                    'status':'completed', 'final':result['final'],
                    'reflect':{k:v for k,v in (ranking.get('stage3_regeneration') or {}).items()
                               if k in ('attempted','attempts','status','initial_status','final_status')},
                    'candidates':[{'candidate_id':c['candidate_id'],
                        'text':c['generated_content']['value'],
                        'checks':[{k:check[k] for k in ('check_id','status','verification_mode')}
                                  for check in c.get('observation',{}).get('fact_checks',[])],
                        'repair_eligible':c.get('repair_eligible'),
                        'repair_outcomes':c.get('repair_outcomes'),
                        'repair_skip_reason':c.get('repair_skip_reason'),
                        'unverified_additions':c.get('judge_rubrics',{}).get('unverified_additions',[])}
                        for c in ranking.get('text_ranking',{}).get('ranked_candidates',[])]}
            except Exception as exc:
                writer.record_failure(record, exc)
                summary['arms'][arm] = {'status':'failed','error_type':type(exc).__name__}
            finally:
                writer.write_summary()
                save('summary.json', summary)
                print(json.dumps({'arm':arm,'status':summary['arms'][arm]['status']}), flush=True)
    finally:
        summary['source_files_unchanged'] = all(fingerprint(Path(p)) == sha for p,sha in sources.items())
        save('summary.json', summary)
        if not summary['source_files_unchanged']:
            raise RuntimeError('Original experiment artifacts changed')
    if any(a['status'] != 'completed' for a in summary['arms'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
