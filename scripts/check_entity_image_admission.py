"""Real API controls: genuine violin, wrong instrument, unrelated animal, mixed images.

Only the derived in-memory records have altered image mappings. Dataset files,
target text masking and train graph remain unchanged; no GPU or new pixels.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from scripts.run_wn18rr_api_preflight import (ROOT, resolve_configuration, load_proxy_exports,
                                            pace_api_requests)
from mmkgc.adapters.image_knowledge import image_modeling_entries
from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner
from mmkgc.data import build_dataset_store
from mmkgc.io.writer import _safe_output
from mmkgc.stages.coarse import coarse_relation_filtering
from mmkgc.stages.candidate_planning import planning_facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--proxy-from-bashrc', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, default=ROOT/'configs/gpt4omini-mainline.yaml')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--controls', nargs='+', choices=['correct_violin', 'wrong_instrument',
        'unrelated_subject', 'mixed_sources'], help='Run selected controls; default all four')
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    proxies = load_proxy_exports(args.proxy_from_bashrc)
    config, status = resolve_configuration(args.model_config, os.environ.get('OPENAI_MODEL'))
    if config is None:
        print(json.dumps(status)); raise SystemExit(2)
    config = replace(config, api_response_format='json_object')
    args.output_root.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        text = json.dumps(_safe_output(value), ensure_ascii=False, indent=2)
        for secret in [config.api_key, *proxies.values(),
                       *(urlsplit(proxy).password for proxy in proxies.values())]:
            if secret:
                text = text.replace(secret, '<redacted>')
        (args.output_root/name).write_text(text)
    dataset = ROOT/'datasets/wn18rr-import-20261106/WN18RR'
    store = build_dataset_store('wn18rr', dataset, graph_split='train')
    keys = {'violin': '04536866', 'guitar': '03467517', 'animal': '02436645'}
    records = {name: store.build_record(store.entity_uri_to_id[key], 'text') for name, key in keys.items()}
    images = {name: store.image_paths(store.entity_uri_to_id[key])[0] for name, key in keys.items()}
    manifest = {name: {'owner_entity': keys[name], 'path': path,
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name, path in images.items()}
    save('manifest.json', {'sources': manifest, 'graph_split': 'train', 'target_text_masked': True,
        'local_gpu_inference': False, 'model': config.model_name,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'execution_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('mmkgc', 'scripts') for p in sorted((ROOT/folder).rglob('*.py'))}})
    controls = [('correct_violin', ['violin']), ('wrong_instrument', ['guitar']),
                ('unrelated_subject', ['animal']), ('mixed_sources', ['animal', 'violin'])]
    restore = pace_api_requests(status['api_host'], os.environ.get('OPENAI_MIN_REQUEST_INTERVAL_MS', '0'))
    results = []
    try:
        for name, owners in controls:
            if args.controls and name not in args.controls:
                continue
            paths = [images[owner] for owner in owners]
            record = replace(records['violin'], existing_modalities={
                **records['violin'].existing_modalities, 'image': paths[0]}, image_paths=paths)
            planner = QwenLocalSemanticStagePlanner(config, dataset_root=dataset,
                timeout_seconds=120, knowledge_stage_timeout_seconds=240)
            try:
                knowledge = planner.build_knowledge(record, 'text',
                    coarse_relation_filtering(record, 'text', catalogue_only=True))
                save(name+'-knowledge.json', knowledge)
                entries = image_modeling_entries(knowledge['image_modeling'])
                decisions = [entry['entity_match']['status'] for entry in entries]
                wrong_refs = {entry['original_image_ref'] for entry, owner in zip(entries, owners)
                              if owner != 'violin'}
                leaked = [f['fact_id'] for f in planning_facts(knowledge)
                          if wrong_refs.intersection(f['evidence_refs'])]
                visual_features = [f['object'] for f in planning_facts(knowledge) if f.get('visual_observations')]
                passed = (not leaked and all(decision != 'compatible' for decision, owner in zip(decisions, owners)
                                             if owner != 'violin')
                          and all(decision == 'compatible' for decision, owner in zip(decisions, owners)
                                  if owner == 'violin') and ('violin' not in owners or bool(visual_features)))
                result = {'control': name, 'image_owners': owners, 'decisions': decisions,
                    'wrong_source_eligible_fact_ids': leaked, 'passed': passed,
                    'eligible_visual_features': visual_features}
            except Exception as exc:
                result = {'control': name, 'passed': False, 'error_type': type(exc).__name__, 'error': str(exc)}
            finally:
                save(name+'-calls.json', planner.calls)
            results.append(result)
            save('summary.json', {'results': results, 'completed': len(results) == len(args.controls or controls)})
            print(json.dumps({key: value for key, value in result.items() if key != 'error'}, ensure_ascii=False), flush=True)
    finally:
        restore()
        assert all(hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
                   for item in manifest.values()), 'source image bytes changed'
    if not all(result['passed'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
