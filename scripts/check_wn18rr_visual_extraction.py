"""Real Stage 1 control: keep the previously failing caption AND original image fixed."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import os
from urllib.parse import urlsplit
from dotenv import load_dotenv

from mmkgc.adapters.image_knowledge import load_source_images, with_image_description
from mmkgc.adapters.media_knowledge import SourceMedia
from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner, load_qwen_config
from mmkgc.data import build_dataset_store
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import _safe_output, _stage_artifact
from mmkgc.stages.modeling import build_knowledge_input, apply_knowledge_response
from mmkgc.stages.semantic import KNOWLEDGE_REQUEST_CONTEXT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model-config', default='configs/qwen7b-stage1.yaml')
    parser.add_argument('--model')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--proxy-from-bashrc', type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset = Path('datasets/wn18rr-import-20261106/WN18RR')
    original = Path('outputs/wn18rr-onboarding-20261106/attempt-r3/runs/text-qwen7b/entities/02436645')
    store = build_dataset_store('wn18rr', dataset, graph_split='train')
    record = store.build_record(store.entity_uri_to_id['02436645'], 'text')
    prepared, _ = load_saved_run(original, record, 1, missing_mode='text')
    saved = prepared.knowledge['image_modeling']
    images = load_source_images(record, dataset)
    assert len(images) == 1
    image, metadata = images[0]
    assert metadata['sha256'] == saved['original_image']['sha256']
    inputs = with_image_description(build_knowledge_input(record, 'text', prepared.coarse),
        saved['description'], metadata, saved['description_model_call_ref'])
    proxies = {}
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    if args.proxy_from_bashrc:
        from scripts.run_wn18rr_api_preflight import load_proxy_exports
        proxies = load_proxy_exports(args.proxy_from_bashrc)
    config = load_qwen_config(args.model_config)
    if args.model:
        config = replace(config, model_name=args.model)
    from scripts.run_wn18rr_api_preflight import pace_api_requests
    restore = pace_api_requests(urlsplit(config.base_url).hostname, os.environ.get('OPENAI_MIN_REQUEST_INTERVAL_MS','0'))
    def dump(value):
        text = json.dumps(_safe_output(value), indent=2)
        if config.api_key:
            text = text.replace(config.api_key, '<redacted>')
        for proxy in proxies.values():
            text = text.replace(proxy, '<redacted-proxy-url>')
        return text
    planner = QwenLocalSemanticStagePlanner(config,
        dataset_root=dataset, timeout_seconds=120, knowledge_stage_timeout_seconds=180)
    # Bind the SAME frozen image to every extraction/contract-repair request.
    # This is an extraction-only control, not a new image description or a whole-loop quality claim.
    token = KNOWLEDGE_REQUEST_CONTEXT.set({'deadline': time.perf_counter() + 180,
        'media_inputs': [SourceMedia('image', metadata, image=image)],
        'image_input_metadata': metadata,
        'image_description_model_call_ref': saved['description_model_call_ref']})
    try:
        result = planner._refine('stage1_knowledge', record, 'text', inputs,
                                lambda response: apply_knowledge_response(inputs, response))
    finally:
        KNOWLEDGE_REQUEST_CONTEXT.reset(token)
        restore()
        (args.output_root/'model-calls.json').write_text(dump(planner.calls))
    (args.output_root/'knowledge.json').write_text(json.dumps(_stage_artifact('knowledge_modeling', result), indent=2))
    summary = {'control': 'same_original_image_and_original_caption_new_extraction_policy', 'model': config.model_name,
        'image_sha256': metadata['sha256'], 'caption': saved['description'],
        'source_stage1_sha256': hashlib.sha256((original/'1_knowledge_modeling.json').read_bytes()).hexdigest(),
        'facts': [{key: fact[key] for key in ('fact_id', 'subject', 'relation', 'object',
                  'source_weight', 'generation_eligible', 'visual_observations')} for fact in result['relation_facts']],
        'uncertainties': result['uncertainties']}
    (args.output_root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
