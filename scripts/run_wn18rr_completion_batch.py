"""Frozen WN18RR completion batches; immutable attempts and reusable Stage 0–3."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import re
import time

from mmkgc.data import build_dataset_store
from mmkgc.io.resume import load_saved_run
from mmkgc.io.writer import RunWriter, _safe_output
from mmkgc.stages.generation import relation_alternation_generation, _fill_image_seed_variants

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'datasets/wn18rr-import-20261106/WN18RR'
# Deliberate diagnostic coverage, fixed before generation; not representative sampling.
PILOT = {
    'image': ['01140794', '02036982', '12310153', '04559275', '14980579',
              '10314952', '10372373', '05941423', '13991823', '13448778'],
    'text': ['01838038', '02638596', '02855793', '04151581', '04362821',
             '10557854', '10338231', '10030277', '09779124', '10257221'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(_safe_output(value), ensure_ascii=False, indent=2))
    temp.replace(path)


def freeze(root, runtime_settings=None):
    root.mkdir(parents=True, exist_ok=False)
    store = build_dataset_store('wn18rr', DATASET, graph_split='train')
    test = {row[i] for row in (line.split('\t') for line in
            (DATASET/'test.tsv').read_text().splitlines()) for i in (0, 2)}
    alignment = read(DATASET/'image_alignment/entity2images.json')
    missing_images = sorted(key for key in test if not alignment.get(key))
    missing_texts = sorted(key for key in test if not store.entity_texts.get(key))
    rows = []
    for mode, keys in PILOT.items():
        for key in keys:
            assert key in test
            entity_id = store.entity_uri_to_id[key]
            images = store.image_paths(entity_id)
            assert (not images if mode == 'image' else 1 <= len(images) <= 4)
            rows.append({'entity_key': key, 'entity_id': entity_id,
                'label': store.entity_info[key]['label'], 'missing_mode': mode,
                'scope': 'real_missing' if mode == 'image' else 'simulated_missing',
                'train_degree': len(store._adjacency.get(entity_id, [])),
                'images': [{'path': path, 'sha256': digest(path)} for path in images]})
    bindings = {str(path): digest(path) for path in
        [DATASET/name for name in ('entities.txt', 'relations.txt', 'train.tsv',
         'test.tsv', 'entity2text.txt', 'wordnet-mlj12-definitions.txt',
         'image_alignment/entity2images.json')]}
    if runtime_settings is not None:
        allowed = {'model_config', 'stage1_model_config', 'stage2_model_config',
                   'max_selected_facts', 'knowledge_stage_timeout_seconds', 'model_call_timeout_seconds'}
        if not isinstance(runtime_settings, dict) or set(runtime_settings) - allowed:
            raise ValueError('Unsupported pilot runtime setting')
        save(root/'runtime-settings.json', runtime_settings)
        bindings[str(root/'runtime-settings.json')] = digest(root/'runtime-settings.json')
    sources = {str(path.relative_to(ROOT)): digest(path) for folder in ('mmkgc', 'scripts', 'configs')
               for path in (ROOT/folder).rglob('*') if path.suffix in {'.py', '.yaml', '.sh', '.txt'}}
    save(root/'scope.json', {'test_entities': len(test), 'real_missing_image': missing_images,
         'real_missing_text': missing_texts, 'pilot_is_diagnostic': True,
         'test_relations_are_not_generation_inputs': True})
    save(root/'manifest.json', rows)
    save(root/'binding.json', {'dataset_root': str(DATASET), 'files': bindings,
                             'sources': sources, 'manifest_sha256': digest(root/'manifest.json')})
    save(root/'summary.json', {'status': 'frozen', 'expected_entities': len(rows)})
    print(json.dumps({'frozen_entities': len(rows), 'real_missing_images': len(missing_images),
                      'real_missing_texts': len(missing_texts), 'root': str(root)}, ensure_ascii=False))


def validate(root, *, check_sources=True):
    binding = read(root/'binding.json')
    if digest(root/'manifest.json') != binding['manifest_sha256']:
        raise ValueError('Frozen entity manifest changed')
    for path, expected in binding['files'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen dataset input changed: {Path(path).name}')
    if check_sources:
        for path, expected in binding['sources'].items():
            if digest(ROOT/path) != expected:
                raise ValueError(f'Frozen implementation changed: {path}')
    rows = read(root/'manifest.json')
    for row in rows:
        for image in row['images']:
            if digest(image['path']) != image['sha256']:
                raise ValueError('Frozen original image changed')
    return rows


class BudgetedImages:
    """Persist attempts before drawing; retries cannot reset an entity's budget."""
    def __init__(self, delegate, ledger):
        self.delegate, self.ledger = delegate, Path(ledger)

    def generate(self, entity_key, candidate_id, positive_prompt, negative_prompt):
        rows = read(self.ledger) if self.ledger.exists() else []
        previous = [r for r in rows if r['entity_key'] == entity_key]
        if len(previous) >= 3 or ('_safe_' in candidate_id and any('_safe_' in r['candidate_id'] for r in previous)):
            raise RuntimeError('Per-entity SDXL budget exhausted; originals retained')
        rows.append({'entity_key': entity_key, 'candidate_id': candidate_id, 'time': time.time()})
        save(self.ledger, rows)
        return self.delegate.generate(entity_key, candidate_id, positive_prompt, negative_prompt)


def add_seed_variant(generation, record, image_generator):
    """One accepted prompt, two seeds; retain the first image if the second fails."""
    payload = generation['relation_alternation_generation']
    payload['candidate_budget'] = 2
    candidates = payload['image_candidates']
    if not candidates:
        return generation
    planned = _fill_image_seed_variants(deepcopy(candidates), ['image_001', 'image_002'])
    variant = planned[1]
    variant['metadata']['seed_variant_reason'] = 'frozen_same_prompt_seed_control'
    try:
        artifact = image_generator.generate(record.entity_key, variant['candidate_id'],
                                            variant['positive_prompt'], '')
        for key in ('image_candidate_kind', 'prompt_source_candidate_id', 'seed_variant_index', 'seed_variant_reason'):
            artifact.metadata[key] = artifact.model_call[key] = variant['metadata'][key]
        variant['generated_content'] = {'type': 'image', 'value': variant['positive_prompt'],
            'image_path': artifact.relative_image_path, 'metadata_path': artifact.relative_metadata_path,
            'seed': artifact.metadata['seed']}
        variant['metadata'].update(status='generated', seed=artifact.metadata['seed'])
        candidates.append(variant)
        generation['generated_image_artifacts'].append(artifact)
        generation['model_calls'].append(artifact.model_call)
    except Exception as exc:
        generation['model_calls'].append({'stage': 'stage3_image_generation', 'candidate_id': 'image_002',
            'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
    return generation


def persist_initial(pipeline, prepared, generation, root, name):
    previous = pipeline.skip_stage4
    pipeline.skip_stage4 = True
    try:
        result = pipeline.finish(prepared, generation_result=generation)
        writer = RunWriter(root, name, {'stage4_deferred': True, 'ground_truth_used': False})
        writer.write_entity(result)
        writer.write_summary()
        return writer.entities_dir/prepared.record.entity_key
    finally:
        pipeline.skip_stage4 = previous


def publication_ready(candidate):
    """Use the persisted Stage4 decision; do not invent a second coverage gate."""
    return (candidate.get('gate', {}).get('decision') == 'pass'
            and candidate.get('repair_eligible') is not False)


def export_best(entity_dir, mode, destination):
    """Export only a persisted Stage 4 pass; leave uncertain results identifiable."""
    final = read(entity_dir/'final.json')
    if final.get('ranking_status') != 'completed':
        return None
    ranking = read(entity_dir/'4_kg_aware_candidate_consistency_ranking.json')['data'][mode+'_ranking']
    best = final.get('best_'+mode+'_candidate_id')
    if not best:
        return None
    if not re.fullmatch(mode + r'_(?:safe_)?\d+', best) or ranking.get('best_candidate_id') != best:
        raise ValueError('Invalid persisted best candidate')
    candidates = [row for row in ranking.get('ranked_candidates', []) if row['candidate_id'] == best]
    if len(candidates) != 1 or candidates[0].get('gate', {}).get('decision') != 'pass':
        raise ValueError('Best candidate did not pass its persisted gate')
    candidate = candidates[0]
    if not publication_ready(candidate):
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite a completed export')
    if mode == 'image':
        from PIL import Image
        source = entity_dir/'generated_images'/f'{best}.png'
        with Image.open(source) as opened:
            opened.load()
            if opened.format != 'PNG' or opened.size != (1024, 1024):
                raise ValueError('SDXL output is not a 1024x1024 PNG')
        destination.write_bytes(source.read_bytes())
    else:
        payload = read(entity_dir/'3_relation_alternation_generation.json')['data']
        texts = [c for c in payload['text_candidates'] if c['candidate_id'] == best]
        if len(texts) != 1 or not texts[0]['generated_content']['value'].strip():
            raise ValueError('Missing generated text')
        destination.write_text(texts[0]['generated_content']['value'])
    return {'candidate_id': best, 'path': str(destination), 'sha256': digest(destination),
            'stage4_candidate': candidate, 'identity_verified': False}


def terminal(row, root):
    if row.get('status') not in {'exported', 'needs_review', 'no_usable_candidate'}:
        return False
    entity = Path(row['entity_dir'])
    if not entity.is_relative_to(root) or not (entity/'final.json').is_file():
        raise ValueError('Terminal checkpoint is missing its artifacts')
    for path, expected in row['artifact_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Completed artifact changed')
    if row.get('export') and digest(row['export']['path']) != row['export']['sha256']:
        raise ValueError('Completed export changed')
    return True


def summarize(root, manifest):
    rows = []
    for item in manifest:
        path = root/'state'/f"{item['missing_mode']}-{item['entity_key']}.json"
        rows.append(read(path) if path.exists() else {**item, 'status': 'pending'})
    settings = read(root/'runtime-settings.json') if (root/'runtime-settings.json').exists() else {}
    from mmkgc.adapters.qwen import load_qwen_config
    model = load_qwen_config(ROOT/settings.get('model_config', 'configs/qwen7b-text-vision.yaml')).model_name
    save(root/'summary.json', {'expected_entities': len(rows), 'counts': dict(Counter(r['status'] for r in rows)),
        'ground_truth_used': False, 'generation_model': model,
        'image_model': 'SDXL', 'review_model': 'gpt-6-luna', 'rows': rows})
    (root/'completion_manifest.jsonl').write_text(''.join(json.dumps(_safe_output(r), ensure_ascii=False)+'\n' for r in rows))


def review(root):
    """Apply the current export rule to immutable historical results, offline.

    Generation still requires matching frozen source code. Review deliberately
    uses today's rule and verifies historical inputs/artifacts, not regeneration.
    """
    manifest = validate(root, check_sources=False)
    destination = root/'reviewed'
    destination.mkdir(exist_ok=False)
    rows, accepted = [], []
    for item in manifest:
        mode, key = item['missing_mode'], item['entity_key']
        original = read(root/'state'/f'{mode}-{key}.json')
        row = {'entity_key': key, 'label': item['label'], 'missing_mode': mode,
               'scope': item['scope'], 'original_batch_status': original['status'],
               'source_entity_dir': original.get('entity_dir')}
        if original['status'] == 'failed':
            row.update(status='failed', error_type=original.get('error_type'), error=original.get('error'))
        else:
            if not terminal(original, root):
                raise ValueError('Review requires a terminal batch')
            entity = Path(original['entity_dir'])
            ranking = read(entity/'4_kg_aware_candidate_consistency_ranking.json')['data'][mode+'_ranking']
            best = ranking.get('best_candidate_id')
            candidate = next((c for c in ranking.get('ranked_candidates', []) if c['candidate_id'] == best), {})
            manual_path = root/'verification/manual-review'/f'{key}.json'
            manual = read(manual_path) if manual_path.exists() else {}
            if manual and original.get('export') and manual.get('export_sha256') != original['export']['sha256']:
                raise ValueError('Manual review refers to a different candidate export')
            eligible = publication_ready(candidate)
            row.update(status='needs_review', structured_checks_ready=eligible,
                       manual_review=manual.get('decision', 'pending'), candidate_id=best)
            if eligible and manual.get('decision') in {'panel_compatible', 'concept_compatible'}:
                suffix = '.png' if mode == 'image' else '.txt'
                exported = export_best(entity, mode, destination/'exports'/mode/(key+suffix))
                if exported:
                    row.update(status='accepted', export=exported)
                    accepted.append(row)
        rows.append(row)
    summary = {'revalidation_only': True, 'new_model_calls': 0, 'ground_truth_used': False,
        'review_source_sha256': digest(Path(__file__)),
        'historical_execution': read(root/'verification/execution-commit.json'),
        'counts': dict(Counter(r['status'] for r in rows)),
        'by_modality': {mode: dict(Counter(r['status'] for r in rows if r['missing_mode'] == mode))
                        for mode in ('text', 'image')},
        'manual_review_is_independent_human_annotation': False,
        'batch_expansion_allowed': False, 'reason': 'Pilot quality acceptance failed.'}
    save(destination/'summary.json', summary)
    for name, values in (('completion_manifest.jsonl', rows), ('accepted_completions.jsonl', accepted)):
        (destination/name).write_text(''.join(json.dumps(_safe_output(r), ensure_ascii=False)+'\n' for r in values))
    print(json.dumps(summary, ensure_ascii=False))


def selection_control(root, failure_path):
    """Replay one actual failed selection-repair request against local 7B."""
    from mmkgc.adapters.qwen import load_qwen_config
    from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
    from mmkgc.stages.candidate_planning import fact_selection_schema, apply_fact_selection
    from mmkgc.stages.candidate_planning import parse_candidate_response
    failure = read(failure_path)
    original = next(c for c in failure['calls'] if c.get('stage') == 'stage2_selection')
    payload = json.loads(original['prompt'])
    knowledge = {'relation_facts': payload['triples'], 'evidence_catalog': payload['evidence_catalog']}
    limit = original['max_selected_facts']
    schema = fact_selection_schema(knowledge, limit)
    messages = [{'role': 'system', 'content': original['system_prompt']},
                {'role': 'user', 'content': original['prompt']},
                {'role': 'user', 'content': original['selection_repairs'][-1]['prompt']}]
    planner = OpenAICompatibleRelationPlanner(load_qwen_config(ROOT/'configs/qwen7b-stage2.yaml'))
    call = {'network_attempts': [], 'transport_retries': 0, 'stage_timeout_seconds': 120}
    try:
        raw = planner._complete_with_retries(messages, schema, call, 1, time.perf_counter()+120)
        parsed = parse_candidate_response(raw)
        result = apply_fact_selection(knowledge, parsed, payload['target_modality'], limit)
        save(root/'selection-control-result.json', {'status': 'applied', 'model_name': planner.config.model_name,
             'source_failure': str(failure_path), 'source_sha256': digest(failure_path),
             'same_frozen_repair_messages': True, 'response_text': raw, 'selection': result,
             'network_attempts': call['network_attempts'], 'semantic_quality_not_established': True})
    except Exception as exc:
        save(root/'selection-control-failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
             'source_sha256': digest(failure_path), 'network_attempts': call['network_attempts']})
        raise


def attempt_budget_used(row):
    """Administrative stops retain execution history but are not model failures."""
    return max(0, int(row.get('attempt', 0)) - int(row.get('interrupted_attempts', 0)))


def run(root, retry_failed=False, *, shared_local_models=None, max_attempts=None, entity_admission=None):
    from mmkgc.defaults import build_default_mainline_runtime
    rows = validate(root)
    settings = read(root/'runtime-settings.json') if (root/'runtime-settings.json').exists() else {}
    retry_policy = read(root/'retry-policy.json') if (root/'retry-policy.json').exists() else {}
    retry_contexts = read(root/'retry-contexts.json') if (root/'retry-contexts.json').exists() else {}
    from mmkgc.retry_context import retry_scope, RetrySeededGenerator
    store = (shared_local_models.dataset_store('wn18rr', DATASET, graph_split='train')
             if shared_local_models is not None else
             build_dataset_store('wn18rr', DATASET, graph_split='train'))
    for mode in ('image', 'text'):
        pending = []
        for item in (r for r in rows if r['missing_mode'] == mode):
            state = root/'state'/f"{mode}-{item['entity_key']}.json"
            previous = read(state) if state.exists() else {}
            if terminal(previous, root) or (previous.get('status') == 'failed' and not retry_failed):
                continue
            if (max_attempts is not None and previous.get('status') == 'failed'
                    and attempt_budget_used(previous) >= max_attempts):
                continue
            pending.append((item, state, previous))
        if not pending:
            continue
        runtime = build_default_mainline_runtime(missing_mode=mode, candidate_budget=2 if mode == 'image' else 1,
            **{'model_config': ROOT/'configs/qwen7b-text-vision.yaml', 'model_call_timeout_seconds': 120, **settings},
            shared_local_models=shared_local_models,
            dataset_root=DATASET,
            stage1_reranker_model_dir='/medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B',
            stage1_reranker_device='cuda:0',
            stage1_visual_verifier_config=ROOT/'configs/gpt6-luna-mainline.yaml',
            stage4_config=ROOT/'configs/stage4-gpt6-luna-gpu0.yaml',
            sdxl_model_dir='/medisco/linjiashi/models/SDXL/stable-diffusion-xl-base-1.0', sdxl_device='cuda:0')
        pipeline = runtime.pipeline
        if mode == 'image':
            if retry_policy:
                pipeline.image_generator = RetrySeededGenerator(pipeline.image_generator, retry_policy['round'],
                                                               retry_policy.get('seed_namespace', 0))
            pipeline.image_generator = BudgetedImages(pipeline.image_generator, root/'sdxl-attempts.json')
        for item, state, previous in pending:
            with (entity_admission if entity_admission is not None else nullcontext()), retry_scope(retry_contexts.get(item['entity_key'])):
                record = store.build_record(item['entity_id'], mode)
                if mode == 'text':
                    assert not record.description and not record.existing_modalities.get('text')
                else:
                    assert not record.image_paths and not record.existing_modalities.get('image')
                number = int(previous.get('attempt', 0)) + 1
                attempt = root/'attempts'/f"{mode}-{record.entity_key}"/f'{number:03d}'
                attempt.mkdir(parents=True, exist_ok=False)
                row = {**item, 'attempt': number, 'status': 'running', 'started_at': time.time(),
                       'interrupted_attempts': int(previous.get('interrupted_attempts', 0)),
                       'initial_entity_dir': previous.get('initial_entity_dir')}
                save(state, row)
                summarize(root, rows)
                print(json.dumps({'event': 'entity_start', 'mode': mode, 'key': record.entity_key}), flush=True)
                try:
                    if row['initial_entity_dir']:
                        prepared, generation = load_saved_run(row['initial_entity_dir'], record,
                            2 if mode == 'image' else 1, missing_mode=mode)
                    else:
                        prepared = pipeline.prepare(record, mode)
                        save(attempt/'prepared.json', {'coarse': prepared.coarse, 'knowledge': prepared.knowledge,
                             'selection': prepared.selection, 'reasoning': prepared.reasoning, 'model_calls': prepared.model_calls})
                        generation = relation_alternation_generation(record, mode, prepared.knowledge,
                            prepared.reasoning, prepared.selection, pipeline.text_generator,
                            pipeline.image_prompt_generator, pipeline.image_generator, candidate_budget=1)
                        if mode == 'image':
                            # Persist the first actual PNG before attempting another draw.
                            generation['relation_alternation_generation']['candidate_budget'] = 2
                            first = persist_initial(pipeline, prepared, generation, attempt, 'first')
                            row['initial_entity_dir'] = str(first)
                            save(state, row)
                            generation = add_seed_variant(generation, record, pipeline.image_generator)
                        initial = persist_initial(pipeline, prepared, generation, attempt, 'initial')
                        row['initial_entity_dir'] = str(initial)
                        save(state, row)
                    result = pipeline.finish(prepared, generation_result=generation)
                    writer = RunWriter(attempt, 'evaluated', {'missing_mode': mode, 'scope': item['scope'],
                        'generation_model': runtime.model_name, 'ground_truth_used': False,
                        'initial_entity_dir': row['initial_entity_dir'], 'candidate_budget': 2 if mode == 'image' else 1})
                    writer.write_entity(result)
                    writer.write_summary()
                    entity = writer.entities_dir/record.entity_key
                    row['entity_dir'] = str(entity)
                    row['ranking_status'] = result['final']['ranking_status']
                    suffix = '.png' if mode == 'image' else '.txt'
                    row['export'] = export_best(entity, mode, root/'exports'/mode/(record.entity_key+suffix))
                    payload = result['relation_alternation_generation']['relation_alternation_generation']
                    row['generated_candidate_count'] = len(payload[mode+'_candidates'])
                    row['status'] = 'exported' if row['export'] else 'no_usable_candidate'
                    if row['ranking_status'] == 'evaluation_failed':
                        row['status'] = 'failed'
                        row['error_type'] = 'EvaluationFailure'
                        row['evaluation_failures'] = result['kg_aware_candidate_consistency_ranking'].get('evaluation_failures', [])
                        row['error'] = 'Evaluation or repair did not finish; retained candidates are recoverable.'
                    elif not row['export']:
                        row['completion_reason'] = ('no_accepted_export' if row['generated_candidate_count']
                                                    else 'no_candidate_generated')
                    row['artifact_sha256'] = {str(p): digest(p) for p in entity.rglob('*') if p.is_file()}
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                    save(attempt/'failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
                        'partial_knowledge': getattr(exc, 'partial_knowledge', None),
                        'model_call': getattr(exc, 'model_call', None),
                        'calls': [call for component in (pipeline.semantic_planner, pipeline.knowledge_planner,
                                 pipeline.relation_planner, pipeline.evidence_reranker)
                                 for call in getattr(component, 'calls', [])]})
                row['seconds'] = round(time.time()-row['started_at'], 3)
                save(state, row)
                summarize(root, rows)
                print(json.dumps({'event': 'entity_end', 'mode': mode, 'key': record.entity_key,
                                  'status': row['status'], 'seconds': row['seconds']}), flush=True)
                for component in (pipeline.semantic_planner, pipeline.knowledge_planner,
                                  pipeline.relation_planner, pipeline.evidence_reranker):
                    if isinstance(getattr(component, 'calls', None), list):
                        component.calls.clear()
        del pipeline, runtime
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    validate(root)
    summarize(root, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--review', action='store_true')
    parser.add_argument('--selection-control', type=Path)
    parser.add_argument('--runtime-config', type=Path, help='Freeze pilot model settings during --prepare')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.prepare:
        if args.runtime_config:
            import yaml
            freeze(root, yaml.safe_load(args.runtime_config.read_text()))
        else:
            freeze(root)
    elif args.review:
        review(root)
    elif args.selection_control:
        selection_control(root, args.selection_control)
    else:
        with (root/'.worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root, args.retry_failed)


if __name__ == '__main__':
    main()
