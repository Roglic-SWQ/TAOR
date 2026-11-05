"""Rejudge the actual old bad text against the repaired upstream basis, then Reflect once."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from scripts.run_wn18rr_api_preflight import (
    ROOT, resolve_configuration, load_proxy_exports, pace_api_requests, api_ranker,
)
from mmkgc.adapters.qwen import QwenLocalTextGenerator
from mmkgc.data import build_dataset_store
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter, _safe_output
from mmkgc.pipeline import MMKGCPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--proxy-from-bashrc', type=Path)
    parser.add_argument('--basis-entity-dir', type=Path, required=True)
    parser.add_argument('--old-entity-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT/'datasets/wn18rr-import-20261106/WN18RR')
    parser.add_argument('--entity-key', default='02436645')
    parser.add_argument('--api-response-format', choices=('json_schema','json_object'), default='json_object')
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    proxies = load_proxy_exports(args.proxy_from_bashrc) if args.proxy_from_bashrc else {}
    config, status = resolve_configuration(ROOT/'configs/gpt4omini-mainline.yaml', os.environ.get('OPENAI_MODEL'))
    if config is None:
        print(json.dumps(status))
        raise SystemExit(2)
    config = replace(config, api_response_format=args.api_response_format)
    args.output_root.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        text = json.dumps(_safe_output(value), ensure_ascii=False, indent=2)
        text = text.replace(config.api_key, '<redacted>')
        for proxy in proxies.values():
            text = text.replace(proxy, '<redacted-proxy-url>')
        (args.output_root/name).write_text(text)
    store = build_dataset_store('wn18rr', args.dataset_dir, graph_split='train')
    record = store.build_record(store.entity_uri_to_id[args.entity_key], 'text')
    prepared, generation = load_saved_run(args.basis_entity_dir, record, 1, missing_mode='text')
    old_file = args.old_entity_dir/'3_relation_alternation_generation.json'
    original_sha = hashlib.sha256(old_file.read_bytes()).hexdigest()
    old_candidate = json.loads(old_file.read_text())['data']['text_candidates'][0]
    original_text = old_candidate['generated_content']['value']
    # New references are intentionally from the repaired upstream basis. Only
    # candidate content is frozen from the historical run; neither run is edited.
    candidate = generation['relation_alternation_generation']['text_candidates'][0]
    candidate['generated_content']['value'] = original_text
    candidate['metadata'] = {'status': 'generated', 'reused_frozen_candidate': True,
        'original_source': str(old_file), 'source_sha256': original_sha}
    generation['model_calls'] = [{'stage': 'stage3_frozen_candidate_reuse', 'status': 'reused',
        'source': str(old_file), 'sha256': original_sha}]
    prepared = replace(prepared, model_calls=[{**call, 'reused_source_run': str(args.basis_entity_dir)}
                                               for call in prepared.model_calls])
    ranker, _ = api_ranker(config, 120)
    pipeline = MMKGCPipeline(candidate_budget=1, text_generator=QwenLocalTextGenerator(config,120),
                            ranking_engine=ranker, dataset_root=args.dataset_dir)
    writer = RunWriter(args.output_root, 'reflection', {'scope': 'real_old_text_new_upstream_basis',
        'model': config.model_name, 'api_host': status['api_host'], 'local_gpu_inference': False,
        'source_candidate': str(old_file), 'source_basis': str(args.basis_entity_dir)})
    restore = pace_api_requests(status['api_host'], os.environ.get('OPENAI_MIN_REQUEST_INTERVAL_MS','0'))
    summary = {'status': 'running', 'original_candidate_text': original_text,
        'original_candidate_file_sha256': original_sha, 'basis_source': str(args.basis_entity_dir),
        'note': 'Old content is evaluated against new source-isolated references; not a same-prompt generation benchmark.'}
    try:
        result = pipeline.finish(prepared, generation_result=generation)
        writer.write_entity(result)
        writer.write_summary()
        ranking = result['kg_aware_candidate_consistency_ranking']
        feedback = ranking.get('stage3_regeneration')
        final_id = result['final']['best_text_candidate_id']
        candidates = result['relation_alternation_generation']['relation_alternation_generation']['text_candidates']
        summary.update(status=result['final']['ranking_status'], reflection=feedback,
            final_candidate_id=final_id, candidates=[{'candidate_id':item['candidate_id'],
                'text':item['generated_content']['value']} for item in candidates])
        assert original_sha == hashlib.sha256(old_file.read_bytes()).hexdigest(), 'Historical candidate was changed'
        assert summary['status']=='completed', 'API observation or reevaluation failed'
        assert feedback and feedback.get('attempted'), 'No actual Reflect pass was triggered'
        assert final_id != candidate['candidate_id'], 'The repair was not selected over the old candidate'
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        restore()
        save('summary.json', summary)
    print(json.dumps({'status':summary['status'],'output_root':str(args.output_root)}),flush=True)
    if summary['status']=='failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
