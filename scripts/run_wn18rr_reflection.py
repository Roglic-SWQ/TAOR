"""Complete Stage 4 and one native Reflect pass on frozen, real WN18RR outputs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path

from mmkgc.data import build_dataset_store
from mmkgc.defaults import build_default_mainline_runtime
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter, _safe_output


def fingerprints(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stage4-config', default='configs/stage4-qwen7b-local.yaml')
    parser.add_argument('--entity-key', default='02436645')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    before = fingerprints(args.source_root)
    execution_files = [args.stage4_config, 'configs/qwen7b-judge.yaml',
        'mmkgc/ranking/judge_rubrics.py', 'mmkgc/ranking/scorers/gpt4omini_judge.py', 'mmkgc/ranking/prompts/text_judge_rubrics_v1.txt',
        'mmkgc/ranking/prompts/image_judge_rubrics_v1.txt', 'mmkgc/io/resume.py']
    store = build_dataset_store('wn18rr', args.dataset_dir, graph_split='train')
    summary = {'judge_model': 'Qwen/Qwen2-VL-7B-Instruct', 'judge_independence': False,
               'source_files_sha256': before,
               'execution_files_sha256': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in execution_files},
               'modes': {}}
    for mode, source_name in [('text', 'text-qwen7b'), ('image', 'image-qwen7b-sdxl')]:
        record = store.build_record(store.entity_uri_to_id[args.entity_key], mode)
        source = args.source_root / source_name / 'entities' / args.entity_key
        prepared, generation = load_saved_run(source, record, 1, missing_mode=mode)
        for call in prepared.model_calls + generation['model_calls']:
            call['reused_source_run'] = str(source)
        config = {'dataset_type': 'wn18rr', 'dataset_dir': str(args.dataset_dir), 'graph_split': 'train',
                  'missing_mode': mode, 'source_run': str(source), 'stage4_config': args.stage4_config,
                  'judge_model': summary['judge_model'], 'stage0_3_reused': True, 'candidate_budget': 1}
        writer = RunWriter(args.output_root, mode, config)
        try:
            runtime = build_default_mainline_runtime(
                missing_mode=mode, candidate_budget=1,
                stage1_reranker_model_dir='/medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B',
                stage1_reranker_device='cuda:0', dataset_root=args.dataset_dir,
                model_call_timeout_seconds=120, stage4_config=args.stage4_config,
                sdxl_device='cuda:0', sdxl_width=1024, sdxl_height=1024, sdxl_steps=20,
            )
            result = runtime.pipeline.finish(prepared, generation_result=generation)
            writer.write_entity(result)
            ranking = result['kg_aware_candidate_consistency_ranking']
            summary['modes'][mode] = {
                'final': result['final'],
                'reflection': ranking.get('stage3_regeneration', {'attempted': False}),
                'ranking_model_call_count': len(result.get('ranking_model_calls', [])),
                'candidate_ids': [c['candidate_id'] for c in
                    result['relation_alternation_generation']['relation_alternation_generation'][mode+'_candidates']],
            }
            print(mode, json.dumps(_safe_output(summary['modes'][mode]), ensure_ascii=False), flush=True)
            del runtime, result
        except Exception as exc:
            writer.record_failure(record, exc)
            summary['modes'][mode] = {'error_type': type(exc).__name__, 'error': str(exc)}
            print(mode, type(exc).__name__, str(exc), flush=True)
        writer.write_summary()
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        (args.output_root/'closure-summary.json').write_text(json.dumps(_safe_output(summary), ensure_ascii=False, indent=2))
    if before != fingerprints(args.source_root):
        raise RuntimeError('Original source run was changed')
    failures = [mode for mode, result in summary['modes'].items() if 'error' in result or result.get('final', {}).get('ranking_status') == 'evaluation_failed']
    if failures:
        raise SystemExit('Failed modes: '+', '.join(failures))


if __name__ == '__main__':
    main()
