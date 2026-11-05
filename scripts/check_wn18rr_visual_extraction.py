"""Real Stage 1 control: keep the previously failing caption AND original image fixed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

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
    planner = QwenLocalSemanticStagePlanner(load_qwen_config('configs/qwen7b-stage1.yaml'),
        dataset_root=dataset, knowledge_stage_timeout_seconds=180)
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
        (args.output_root/'model-calls.json').write_text(json.dumps(_safe_output(planner.calls), indent=2))
    (args.output_root/'knowledge.json').write_text(json.dumps(_stage_artifact('knowledge_modeling', result), indent=2))
    summary = {'control': 'same_original_image_and_original_caption_new_extraction_policy',
        'image_sha256': metadata['sha256'], 'caption': saved['description'],
        'source_stage1_sha256': hashlib.sha256((original/'1_knowledge_modeling.json').read_bytes()).hexdigest(),
        'facts': [{key: fact[key] for key in ('fact_id', 'subject', 'relation', 'object',
                  'source_weight', 'generation_eligible', 'visual_observations')} for fact in result['relation_facts']],
        'uncertainties': result['uncertainties']}
    (args.output_root/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
