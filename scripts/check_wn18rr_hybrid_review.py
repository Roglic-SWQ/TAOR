"""No-GPU preflight: independently review the same real native proposed features."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from mmkgc.adapters.image_knowledge import load_source_images, image_modeling_entries
from mmkgc.adapters.media_knowledge import SourceMedia
from mmkgc.adapters.qwen import QwenLocalSemanticStagePlanner, load_qwen_config
from mmkgc.data import build_dataset_store
from mmkgc.io.writer import _safe_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-knowledge', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(args.source_knowledge.read_bytes()).hexdigest()
    knowledge = json.loads(args.source_knowledge.read_text())
    dataset = Path('datasets/wn18rr-import-20261106/WN18RR')
    store = build_dataset_store('wn18rr', dataset, graph_split='train')
    key = knowledge['entity']['entity_key']
    record = store.build_record(store.entity_uri_to_id[key], 'text')
    sources = [SourceMedia('image', metadata, image=image)
               for image, metadata in load_source_images(record, dataset)]
    recorded = {entry['original_image_ref']: entry['original_image']['sha256']
                for entry in image_modeling_entries(knowledge['image_modeling'])}
    assert recorded == {s.metadata['evidence_id']: s.metadata['sha256'] for s in sources}
    planner = QwenLocalSemanticStagePlanner(load_qwen_config('configs/qwen7b-stage1.yaml'),
        timeout_seconds=120, dataset_root=dataset,
        visual_verifier_config=load_qwen_config('configs/independent-visual-verifier.yaml'))
    def save(name, data):
        (args.output_root/name).write_text(json.dumps(_safe_output(data), ensure_ascii=False, indent=2))
    try:
        checked = planner._verify_visual_features(knowledge, sources, key, time.perf_counter()+180)
        save('checked-knowledge.json', checked)
        summary = {'status':'completed', 'local_gpu_inference':False, 'source_knowledge_sha256':before,
            'reviewer':checked['stage1_v3']['visual_fact_policy']['pixel_recheck_reviewer'],
            'facts':[{'object':f['object'],'eligible':f['generation_eligible'],
                      'checks':f.get('visual_source_checks',[])} for f in checked['relation_facts']]}
        save('summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        save('model-calls.json', planner.calls)
        assert hashlib.sha256(args.source_knowledge.read_bytes()).hexdigest() == before


if __name__ == '__main__':
    main()
