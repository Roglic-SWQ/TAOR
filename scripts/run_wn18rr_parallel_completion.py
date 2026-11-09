"""Run disjoint WN18RR checkpoint shards with one shared local model set."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import random
import shutil
import threading
import time

from mmkgc.data import build_dataset_store
from mmkgc.shared_models import SharedLocalModels
from scripts import run_wn18rr_completion_batch as batch

MAX_WORKERS = 256


class EntityAdmission:
    """Resize active entities between checkpoints without restarting models."""

    def __init__(self, root, maximum):
        self.root, self.maximum = Path(root), maximum
        self._condition = threading.Condition()
        self._active = self._peak = 0
        self._target = maximum
        self._checked = 0.0

    def _limit(self):
        now = time.monotonic()
        if now - self._checked >= 1:
            self._checked = now
            path = self.root / 'entity-concurrency.json'
            try:
                target = batch.read(path)['active_entities'] if path.exists() else self.maximum
                if type(target) is not int or not 1 <= target <= self.maximum:
                    raise ValueError('active_entities outside prepared worker capacity')
                if target != self._target:
                    self._target = target
                    with (self.root / 'concurrency-events.jsonl').open('a') as output:
                        output.write(json.dumps({'time': time.time(), 'active_entities_target': target}) + '\n')
            except (ValueError, KeyError, TypeError, OSError) as exc:
                # A malformed manual edit retains the last valid setting.
                with (self.root / 'concurrency-events.jsonl').open('a') as output:
                    output.write(json.dumps({'time': time.time(), 'error_type': type(exc).__name__,
                                             'retained_target': self._target}) + '\n')
        return self._target

    def __enter__(self):
        with self._condition:
            while self._active >= self._limit():
                self._condition.wait(timeout=1)
            self._active += 1
            self._peak = max(self._peak, self._active)
        return self

    def __exit__(self, *args):
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {'target': self._limit(), 'active': self._active, 'peak_active': self._peak}


def validate_workers(workers):
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'workers must be in [1, {MAX_WORKERS}]')


def source_binding():
    files = {str(batch.DATASET / name): batch.digest(batch.DATASET / name) for name in
             ('entities.txt', 'relations.txt', 'train.tsv', 'test.tsv', 'entity2text.txt',
              'wordnet-mlj12-definitions.txt', 'image_alignment/entity2images.json')}
    sources = {str(path.relative_to(batch.ROOT)): batch.digest(path)
               for folder in ('mmkgc', 'scripts', 'configs')
               for path in (batch.ROOT / folder).rglob('*')
               if path.suffix in {'.py', '.yaml', '.sh', '.txt'}}
    requirements = batch.ROOT / 'requirements-qwen35-batch.txt'
    if requirements.is_file():
        files[str(requirements)] = batch.digest(requirements)
    return {'dataset_root': str(batch.DATASET), 'files': files, 'sources': sources}


def prepare(root, *, workers=10, keys=None, limit=None, seed=20261108, reuse=(), regenerate_keys=()):
    validate_workers(workers)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    store = build_dataset_store('wn18rr', batch.DATASET, graph_split='train')
    test = {row[index] for row in (line.split('\t') for line in
            (batch.DATASET / 'test.tsv').read_text().splitlines()) for index in (0, 2)}
    alignment = batch.read(batch.DATASET / 'image_alignment/entity2images.json')
    missing = sorted(key for key in test if not alignment.get(key))
    chosen = list(keys) if keys is not None else missing
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(missing):
        raise ValueError('Scope must contain unique, truly missing test entities')
    if limit is not None:
        chosen = random.Random(seed).sample(chosen, min(limit, len(chosen)))
    rows = []
    for key in chosen:
        entity_id = store.entity_uri_to_id[key]
        assert not store.image_paths(entity_id)
        rows.append({'entity_key': key, 'entity_id': entity_id,
                     'label': store.entity_info[key]['label'], 'missing_mode': 'image',
                     'scope': 'real_missing', 'train_degree': len(store._adjacency.get(entity_id, [])),
                     'images': []})
    settings = {'model_config': 'configs/qwen35-9b-hybrid.yaml',
                'stage1_model_config': 'configs/qwen35-9b-hybrid.yaml',
                'stage2_model_config': 'configs/gpt6-luna-visual-description.yaml',
                'max_selected_facts': 4, 'knowledge_stage_timeout_seconds': 600,
                'model_call_timeout_seconds': 120}
    binding = source_binding()
    wanted = {row['entity_key'] for row in rows}
    regenerate = set(regenerate_keys)
    if not regenerate <= wanted:
        raise ValueError('Regeneration keys must belong to the chosen missing-image scope')
    reused = {}
    for previous_root in (Path(path).resolve() for path in reuse):
        batch.validate(previous_root, check_sources=False)
        previous_binding = batch.read(previous_root / 'binding.json')
        for path, expected in binding['files'].items():
            if Path(path).is_relative_to(batch.DATASET) and previous_binding['files'].get(path) != expected:
                raise ValueError('Reuse dataset differs from current dataset')
        for path in sorted((previous_root / 'state').glob('image-*.json')):
            previous = batch.read(path)
            key = previous['entity_key']
            if key not in wanted or key in regenerate or key in reused or previous.get('status') != 'exported':
                continue
            batch.terminal(previous, previous_root)
            source = Path(previous['entity_dir'])
            destination = root / 'reused' / key
            shutil.copytree(source, destination)
            export = batch.export_best(destination, 'image', root / 'exports/image' / (key + '.png'))
            if export is None:
                raise ValueError('Previously exported candidate no longer satisfies publication checks')
            reused[key] = {**next(row for row in rows if row['entity_key'] == key),
                           'status': 'exported', 'reused': True, 'export': export,
                           'entity_dir': str(destination), 'source_entity_dir': str(source)}
    pending = [row for row in rows if row['entity_key'] not in reused]
    shards = [pending[index::workers] for index in range(workers)]
    for index, shard_rows in enumerate(shards):
        shard = root / 'workers' / f'{index:02d}'
        batch.save(shard / 'runtime-settings.json', settings)
        batch.save(shard / 'manifest.json', shard_rows)
        batch.save(shard / 'binding.json', {**binding, 'files': {
            **binding['files'], str(shard / 'runtime-settings.json'): batch.digest(shard / 'runtime-settings.json')},
            'manifest_sha256': batch.digest(shard / 'manifest.json')})
        batch.save(shard / 'summary.json', {'expected_entities': len(shard_rows),
                   'counts': {'pending': len(shard_rows)}, 'rows': [{**row, 'status': 'pending'} for row in shard_rows]})
    batch.save(root / 'manifest.json', rows)
    batch.save(root / 'reused.json', list(reused.values()))
    batch.save(root / 'orchestration.json', {'workers': workers, 'max_attempts': 2,
               'model': 'Qwen/Qwen3.5-9B', 'backend': 'vllm', 'gpu_model_copies': 1,
               'source_split': 'train', 'scope_uses_test_head_and_tail': True,
               'missing_image_population': len(missing), 'entities': len(rows),
               'reused_entities': len(reused), 'pending_entities': len(pending)})
    if regenerate:
        config = batch.read(root / 'orchestration.json')
        batch.save(root / 'orchestration.json', {**config, 'regenerate_keys': sorted(regenerate)})
    batch.save(root / 'binding.json', {**binding, 'manifest_sha256': batch.digest(root / 'manifest.json'),
               'files': {**binding['files'], str(root / 'orchestration.json'): batch.digest(root / 'orchestration.json'),
                         str(root / 'reused.json'): batch.digest(root / 'reused.json'),
                         **{str(path): batch.digest(path) for folder in ('reused', 'exports')
                            for path in (root / folder).rglob('*') if path.is_file()}}})
    aggregate(root)
    return root


def repartition(root, *, workers):
    """Increase concurrency offline; completed and interrupted work stays in place.

    Only entities without checkpoints move. Existing attempt directories, image
    ledgers, failure counters and saved Stage 0–3 inputs remain with their owner.
    Archive old manifests/bindings before freezing the new scheduling layout.
    """
    validate_workers(workers)
    root = Path(root).resolve()
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = batch.validate(root, check_sources=False)
        config = batch.read(root / 'orchestration.json')
        old_shards = sorted((root / 'workers').iterdir(), key=lambda path: int(path.name))
        if workers < len(old_shards):
            raise ValueError('Repartition cannot reduce the prepared worker capacity')
        if (root / 'finished.json').exists():
            raise ValueError('Completed runs do not need repartitioning')
        binding = source_binding()
        # Scheduling changes may resume saved work; semantic changes require a
        # separate run rather than silently rebinding the old implementation.
        allowed = {'scripts/run_wn18rr_parallel_completion.py',
                   'scripts/run_qwen35_batch_server.sh', 'scripts/check_qwen_batch_service.py',
                   'mmkgc/shared_models.py', 'scripts/run_wn18rr_completion_batch.py',
                   'scripts/run_with_cluster_api.py', 'scripts/run_wn18rr_api_preflight.py'}
        previous = batch.read(root / 'binding.json')
        changed = {path for path in previous['sources'].keys() | binding['sources'].keys()
                   if previous['sources'].get(path) != binding['sources'].get(path)}
        if changed - allowed:
            raise ValueError('Cannot repartition across semantic implementation changes')
        shards = [root / 'workers' / f'{index:02d}' for index in range(workers)]
        assigned = [[] for _ in shards]
        loads = [0 for _ in shards]
        pending = []
        seen = set()
        archive = root / 'repartition-history' / str(time.time_ns())
        archive.mkdir(parents=True, exist_ok=False)
        for name in ['manifest.json', 'binding.json', 'orchestration.json', 'summary.json']:
            shutil.copy2(root / name, archive / name)
        settings = batch.read(old_shards[0] / 'runtime-settings.json')
        for index, shard in enumerate(old_shards):
            manifest = batch.validate(shard, check_sources=False)
            if batch.read(shard / 'runtime-settings.json') != settings:
                raise ValueError('Worker settings differ; cannot redistribute pending work')
            backup = archive / 'workers' / shard.name
            backup.mkdir(parents=True)
            for name in ['manifest.json', 'binding.json', 'runtime-settings.json', 'sdxl-attempts.json']:
                if (shard / name).exists():
                    shutil.copy2(shard / name, backup / name)
            for row in manifest:
                key = row['entity_key']
                if key in seen:
                    raise ValueError('Entity occurs in more than one shard')
                seen.add(key)
                state = shard / 'state' / f"{row['missing_mode']}-{key}.json"
                if state.exists():
                    saved = batch.read(state)
                    done = batch.terminal(saved, shard)
                    assigned[index].append(row)
                    if not done:
                        loads[index] += 1
                else:
                    pending.append(row)
        reused = {row['entity_key'] for row in batch.read(root / 'reused.json')}
        if seen & reused or seen | reused != {row['entity_key'] for row in rows}:
            raise ValueError('Existing shard scope differs from the global manifest')
        for row in pending:
            index = min(range(workers), key=lambda i: (loads[i], i))
            assigned[index].append(row)
            loads[index] += 1
        # All checks above are read-only apart from the audit archive. The job
        # remains stopped until this function finishes and validates the layout.
        for shard, manifest in zip(shards, assigned):
            batch.save(shard / 'runtime-settings.json', settings)
            batch.save(shard / 'manifest.json', manifest)
            batch.save(shard / 'binding.json', {**binding, 'files': {
                **binding['files'], str(shard / 'runtime-settings.json'): batch.digest(shard / 'runtime-settings.json')},
                'manifest_sha256': batch.digest(shard / 'manifest.json')})
            batch.summarize(shard, manifest)
        batch.save(root / 'orchestration.json', {**config, 'workers': workers,
                   'resumed_from_workers': len(old_shards)})
        batch.save(root / 'binding.json', {**previous, 'sources': binding['sources'],
                   'files': {**previous['files'],
                       str(root / 'orchestration.json'): batch.digest(root / 'orchestration.json')}})
        batch.validate(root)
        for shard in shards:
            batch.validate(shard)
        summary = aggregate(root)
        batch.save(archive / 'completed.json', {'workers_before': len(old_shards),
                   'workers_after': workers, 'moved_pending_entities': len(pending),
                   'counts': summary['counts'], 'changed_sources': sorted(changed)})
        return summary


def aggregate(root):
    root = Path(root)
    reused = batch.read(root / 'reused.json')
    rows = list(reused)
    recovered_summaries = []
    for shard in sorted((root / 'workers').iterdir()):
        try:
            worker_rows = batch.read(shard / 'summary.json')['rows']
        except (FileNotFoundError, json.JSONDecodeError):
            # A shared-filesystem reader can briefly miss an atomically replaced
            # summary. Checkpoints remain authoritative; never skip that shard
            # or abort otherwise healthy workers because progress reporting failed.
            worker_rows = []
            for item in batch.read(shard / 'manifest.json'):
                state = shard / 'state' / f"{item['missing_mode']}-{item['entity_key']}.json"
                try:
                    worker_rows.append(batch.read(state))
                except FileNotFoundError:
                    worker_rows.append({**item, 'status': 'pending'})
            recovered_summaries.append(shard.name)
        rows.extend(worker_rows)
    expected = batch.read(root / 'manifest.json')
    keys = [row['entity_key'] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != {row['entity_key'] for row in expected}:
        raise ValueError('Worker progress differs from the frozen global entity scope')
    # Keep the global manifest compact; each worker retains complete Stage 0–4
    # data and every candidate. All export paths remain directly usable.
    compact = []
    for row in rows:
        entry = {key: row.get(key) for key in ('entity_key', 'label', 'missing_mode', 'status',
                 'seconds', 'entity_dir', 'completion_reason', 'error_type', 'error', 'reused')}
        if row.get('export'):
            entry['export'] = {key: row['export'].get(key) for key in ('path', 'sha256', 'candidate_id')}
        compact.append(entry)
    summary = {'expected_entities': len(expected), 'updated_at_unix': time.time(),
               'counts': dict(Counter(row['status'] for row in rows)), 'reused_entities': len(reused),
               'workers': batch.read(root / 'orchestration.json')['workers'],
               'ground_truth_used': False, 'generation_model': 'Qwen/Qwen3.5-9B',
               'image_model': 'SDXL', 'review_model': 'gpt-6-luna', 'rows': compact}
    if recovered_summaries:
        summary['reconstructed_worker_summaries'] = recovered_summaries
    batch.save(root / 'summary.json', summary)
    (root / 'completion_manifest.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in compact))
    return summary


def run(root):
    root = Path(root).resolve()
    batch.validate(root)
    config = batch.read(root / 'orchestration.json')
    validate_workers(config['workers'])
    shared = SharedLocalModels()
    admission = EntityAdmission(root, config['workers'])

    def worker(shard):
        with (shard / '.worker.lock').open('a') as shard_lock:
            fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recovered = False
            for path in (shard / 'state').glob('*.json'):
                previous = batch.read(path)
                if previous.get('status') == 'running':
                    previous.update(status='failed', error_type='InterruptedAttempt',
                                    interrupted_attempts=int(previous.get('interrupted_attempts', 0)) + 1,
                                    error='Previous worker ended before completing this checkpoint')
                    batch.save(path, previous)
                    recovered = True
                elif (previous.get('status') == 'failed' and previous.get('error_type') == 'InterruptedAttempt'
                      and not previous.get('interrupted_attempts')):
                    previous['interrupted_attempts'] = 1
                    batch.save(path, previous)
                    recovered = True
            if recovered:
                batch.summarize(shard, batch.read(shard / 'manifest.json'))
            batch.run(shard, shared_local_models=shared, max_attempts=config['max_attempts'],
                      entity_admission=admission)
            # One additional entity attempt for engineering failures. Accepted
            # and no-usable-candidate checkpoints are terminal; budgets persist.
            batch.run(shard, retry_failed=True, shared_local_models=shared,
                      max_attempts=config['max_attempts'], entity_admission=admission)

    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=config['workers']) as executor:
            futures = {executor.submit(worker, path): str(path) for path in sorted((root / 'workers').iterdir())}
            pending = set(futures)
            while pending:
                completed, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                summary = aggregate(root)
                admission_status = admission.snapshot()
                batch.save(root / 'admission-status.json', admission_status)
                print(json.dumps({'event': 'parallel_progress', 'root': str(root), 'counts': summary['counts'],
                                  'entity_admission': admission_status,
                                  'elapsed_seconds': round(time.monotonic()-started, 1)}), flush=True)
                for future in completed:
                    # Infrastructure/source-binding errors end the run instead
                    # of silently declaring all queued entities processed.
                    future.result()
        batch.validate(root)
        summary = aggregate(root)
        batch.save(root / 'finished.json', {'counts': summary['counts'], 'elapsed_seconds': time.monotonic()-started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--keys', nargs='+')
    parser.add_argument('--reuse', type=Path, action='append', default=[])
    parser.add_argument('--regenerate-keys', nargs='+', default=[])
    parser.add_argument('--repartition', action='store_true',
                        help='Increase workers on a stopped run without repeating completed entities')
    args = parser.parse_args()
    if args.prepare and args.repartition:
        parser.error('--prepare and --repartition are mutually exclusive')
    if args.repartition:
        summary = repartition(args.root, workers=args.workers)
        print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}), flush=True)
    elif args.prepare:
        prepare(args.root, workers=args.workers, keys=args.keys, limit=args.limit, reuse=args.reuse,
                regenerate_keys=args.regenerate_keys)
        print(json.dumps({'prepared': str(args.root)}), flush=True)
    else:
        run(args.root)


if __name__ == '__main__':
    main()
