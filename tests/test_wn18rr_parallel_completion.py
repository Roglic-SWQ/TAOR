"""Full-scope partitioning and worker isolation without loading GPU weights."""
from types import SimpleNamespace

import pytest

from scripts import run_wn18rr_parallel_completion as parallel
from scripts import run_wn18rr_completion_batch as batch


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    data = repo / 'data'
    data.mkdir(parents=True)
    for name in ['entities.txt', 'relations.txt', 'train.tsv', 'entity2text.txt',
                 'wordnet-mlj12-definitions.txt']:
        (data / name).write_text('frozen original data\n')
    (data / 'test.tsv').write_text('00000001\tr\t00000002\n00000003\tr\t00000001\n')
    batch.save(data / 'image_alignment/entity2images.json',
               {'00000001': [], '00000002': ['original.jpg'], '00000003': []})
    for folder in ['mmkgc', 'scripts', 'configs']:
        (repo / folder).mkdir()
        (repo / folder / 'source.py').write_text('immutable = True\n')
    (repo / 'configs/qwen35-9b-hybrid.yaml').write_text(
        'lmm:\n  model_name: test-qwen\n  base_url: http://localhost/v1\n  api_key: EMPTY\n')
    store = SimpleNamespace(entity_uri_to_id={'00000001': 1, '00000002': 2, '00000003': 3},
                            entity_info={key: {'label': key} for key in ['00000001', '00000002', '00000003']},
                            _adjacency={}, image_paths=lambda entity_id: [] if entity_id != 2 else ['original.jpg'])
    monkeypatch.setattr(parallel, 'build_dataset_store', lambda *args, **kwargs: store)
    monkeypatch.setattr(batch, 'DATASET', data)
    monkeypatch.setattr(batch, 'ROOT', repo)
    return tmp_path


def test_scope_includes_both_query_directions_without_duplicate_processing(dataset):
    root = parallel.prepare(dataset / 'run', workers=2)
    assert [row['entity_key'] for row in batch.read(root / 'manifest.json')] == ['00000001', '00000003']
    shards = [batch.validate(path) for path in sorted((root / 'workers').iterdir())]
    assert sorted(row['entity_key'] for shard in shards for row in shard) == ['00000001', '00000003']
    assert len({row['entity_key'] for shard in shards for row in shard}) == 2
    assert batch.read(root / 'summary.json')['counts'] == {'pending': 2}
    (batch.DATASET / 'train.tsv').write_text('changed\n')
    with pytest.raises(ValueError, match='Frozen dataset input changed'):
        batch.validate(root)


def test_nonmissing_or_duplicate_scope_is_rejected(dataset):
    with pytest.raises(ValueError, match='unique, truly missing'):
        parallel.prepare(dataset / 'has-original', keys=['00000002'])
    with pytest.raises(ValueError, match='unique, truly missing'):
        parallel.prepare(dataset / 'duplicates', keys=['00000001', '00000001'])


def test_reuse_accepts_relative_paths_and_revalidates_persisted_results(dataset, monkeypatch):
    from PIL import Image
    old = parallel.prepare(dataset / 'old', workers=1)
    shard = old / 'workers/00'
    row = batch.read(shard / 'manifest.json')[0]
    entity = shard / 'attempts/image-00000001/001/evaluated/entities/00000001'
    image = entity / 'generated_images/image_001.png'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (1024, 1024), 'blue').save(image)
    candidate = {'candidate_id': 'image_001', 'gate': {'decision': 'pass'}}
    batch.save(entity / 'final.json', {'ranking_status': 'completed', 'best_image_candidate_id': 'image_001'})
    batch.save(entity / '4_kg_aware_candidate_consistency_ranking.json', {'data': {'image_ranking': {
        'best_candidate_id': 'image_001', 'ranked_candidates': [candidate]}}})
    exported = batch.export_best(entity, 'image', shard / 'exports/image/00000001.png')
    batch.save(shard / 'state/image-00000001.json', {**row, 'status': 'exported',
        'entity_dir': str(entity), 'export': exported,
        'artifact_sha256': {str(path): batch.digest(path) for path in entity.rglob('*') if path.is_file()}})
    monkeypatch.chdir(dataset)
    current = parallel.prepare(dataset / 'new', workers=2, reuse=['old/workers/00'])
    assert batch.read(current / 'orchestration.json')['reused_entities'] == 1
    pending = [row for path in (current / 'workers').iterdir() for row in batch.read(path / 'manifest.json')]
    assert [row['entity_key'] for row in pending] == ['00000003']
    assert batch.digest(current / 'exports/image/00000001.png') == batch.digest(image)
    assert batch.read(current / 'summary.json')['counts'] == {'exported': 1, 'pending': 1}
    regenerated = parallel.prepare(dataset / 'regenerated', workers=2, reuse=['old/workers/00'],
                                   regenerate_keys=['00000001'])
    assert batch.read(regenerated / 'summary.json')['counts'] == {'pending': 2}
    assert batch.digest(shard / 'exports/image/00000001.png') == batch.digest(image)


def test_worker_failures_do_not_lose_successes_and_retry_is_bounded(dataset, monkeypatch):
    root = parallel.prepare(dataset / 'run', workers=2)
    invocations = []

    def worker(shard, retry_failed=False, *, shared_local_models=None, max_attempts=None, entity_admission=None):
        invocations.append((str(shard), retry_failed, max_attempts, id(shared_local_models)))
        rows = batch.read(shard / 'manifest.json')
        status = 'exported' if rows[0]['entity_key'] == '00000001' else 'failed'
        batch.save(shard / 'summary.json', {'rows': [{**row, 'status': status} for row in rows]})

    monkeypatch.setattr(batch, 'run', worker)
    parallel.run(root)
    assert batch.read(root / 'finished.json')['counts'] == {'exported': 1, 'failed': 1}
    assert len(invocations) == 4
    assert sum(retry for _, retry, _, _ in invocations) == 2
    assert all(limit == 2 for _, _, limit, _ in invocations)
    assert len({identity for _, _, _, identity in invocations}) == 1


def test_repartition_retains_checkpoints_image_budgets_and_attempts(dataset):
    root = parallel.prepare(dataset / 'run', workers=1)
    shard = root / 'workers/00'
    first, second = batch.read(shard / 'manifest.json')
    entity = shard / 'attempts/image-00000001/001/evaluated/entities/00000001'
    batch.save(entity / 'final.json', {'ranking_status': 'completed'})
    saved = {**first, 'status': 'no_usable_candidate', 'entity_dir': str(entity),
             'attempt': 1, 'artifact_sha256': {str(entity / 'final.json'): batch.digest(entity / 'final.json')}}
    batch.save(shard / 'state/image-00000001.json', saved)
    ledger = [{'entity_key': first['entity_key'], 'candidate_id': 'image_001'}]
    batch.save(shard / 'sdxl-attempts.json', ledger)
    attempt = shard / 'attempts/image-00000003/001'
    batch.save(attempt / 'prepared.json', {'retained': True})
    active = {**second, 'status': 'running', 'attempt': 1,
              'initial_entity_dir': str(attempt / 'initial/entities/00000003')}
    batch.save(shard / 'state/image-00000003.json', active)
    before = {str(p): batch.digest(p) for p in shard.rglob('*') if p.is_file()
              and ('/attempts/' in str(p) or '/state/' in str(p) or p.name == 'sdxl-attempts.json')}
    summary = parallel.repartition(root, workers=24)
    assert summary['counts'] == {'no_usable_candidate': 1, 'running': 1}
    assert summary['workers'] == 24
    assert all(batch.digest(path) == expected for path, expected in before.items())
    assert batch.read(shard / 'sdxl-attempts.json') == ledger
    assert batch.read(shard / 'state/image-00000003.json') == active
    assert sum(len(batch.validate(p)) for p in (root / 'workers').iterdir()) == 2
    assert batch.read(shard / 'manifest.json') == [first, second]
    assert len(list((root / 'repartition-history').glob('*/completed.json'))) == 1


def test_repartition_moves_only_pending_and_rejects_running_or_semantic_changes(dataset):
    import fcntl
    root = parallel.prepare(dataset / 'run', workers=1)
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            parallel.repartition(root, workers=2)
    summary = parallel.repartition(root, workers=2)
    assert summary['counts'] == {'pending': 2}
    shards = sorted((root / 'workers').iterdir())
    assert [len(batch.validate(p)) for p in shards] == [1, 1]
    assert batch.validate(root) == batch.read(root / 'manifest.json')
    (batch.ROOT / 'mmkgc/source.py').write_text('semantic_change = True\n')
    with pytest.raises(ValueError, match='semantic implementation'):
        parallel.repartition(root, workers=3)


@pytest.mark.parametrize('workers', [0, 257, True, 2.0])
def test_invalid_worker_counts_fail_before_creating_run(dataset, workers):
    with pytest.raises(ValueError, match='workers must'):
        parallel.prepare(dataset / 'invalid', workers=workers)
    assert not (dataset / 'invalid').exists()


def test_admission_resizes_without_interrupting_active_entities(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    batch.save(tmp_path / 'entity-concurrency.json', {'active_entities': 1})
    gate = parallel.EntityAdmission(tmp_path, 256)
    entered = threading.Event()
    with gate:
        with ThreadPoolExecutor(max_workers=1) as pool:
            def second():
                with gate:
                    entered.set()
                    return gate.snapshot()['active']
            future = pool.submit(second)
            assert not entered.wait(.05)
            batch.save(tmp_path / 'entity-concurrency.json', {'active_entities': 128})
            gate._checked = 0
            assert future.result(timeout=3) == 2
    assert gate.snapshot()['active'] == 0
    assert gate.snapshot()['target'] == 128
    batch.save(tmp_path / 'entity-concurrency.json', {'active_entities': 256})
    gate._checked = 0
    assert gate.snapshot()['target'] == 256
    batch.save(tmp_path / 'entity-concurrency.json', {'active_entities': 999})
    gate._checked = 0
    assert gate.snapshot()['target'] == 256
    with pytest.raises(ValueError):
        with gate:
            raise ValueError('entity failed')
    assert gate.snapshot()['active'] == 0


def test_admission_actually_runs_128_then_256_without_restarting_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    batch.save(tmp_path / 'entity-concurrency.json', {'active_entities': 128})
    gate = parallel.EntityAdmission(tmp_path, 256)
    release = threading.Event()
    def worker(_):
        with gate:
            assert release.wait(timeout=10)
    def wait_active(target):
        deadline = time.monotonic() + 5
        while gate.snapshot()['active'] != target:
            assert time.monotonic() < deadline
            time.sleep(.01)
    with ThreadPoolExecutor(max_workers=256) as pool:
        futures = [pool.submit(worker, i) for i in range(256)]
        try:
            wait_active(128)
            batch.save(tmp_path / 'entity-concurrency.json', {'active_entities': 256})
            gate._checked = 0
            wait_active(256)
            assert gate.snapshot()['peak_active'] == 256
        finally:
            release.set()
        for future in futures:
            future.result(timeout=3)
    assert gate.snapshot()['active'] == 0


@pytest.mark.parametrize('concurrency,serial', [(0, None), (257, None), (128, 0), (128, 129)])
def test_invalid_probe_load_fails_before_any_requests(tmp_path, concurrency, serial):
    from scripts.check_qwen_batch_service import check
    with pytest.raises(ValueError):
        check(tmp_path / 'probe', concurrency=concurrency, serial_requests=serial)
    assert not (tmp_path / 'probe').exists()


def test_refreeze_preserves_owner_above_99_and_execution_history(dataset):
    root = parallel.prepare(dataset / 'run', workers=128)
    origin = root / 'workers/01'
    owner = root / 'workers/100'
    row = batch.read(origin / 'manifest.json')[0]
    for shard, rows in [(origin, []), (owner, [row])]:
        batch.save(shard / 'manifest.json', rows)
        binding = batch.read(shard / 'binding.json')
        batch.save(shard / 'binding.json', {**binding, 'manifest_sha256': batch.digest(shard / 'manifest.json')})
    state = {**row, 'status': 'running', 'attempt': 2, 'initial_entity_dir': str(owner / 'saved')}
    batch.save(owner / 'state/image-00000003.json', state)
    parallel.repartition(root, workers=128)
    assert batch.read(owner / 'manifest.json') == [row]
    assert batch.read(owner / 'state/image-00000003.json') == state
    assert batch.read(origin / 'manifest.json') == []


def test_admin_interruption_preserves_cache_without_exhausting_failure_budget(dataset, monkeypatch):
    root = parallel.prepare(dataset / 'run', workers=1)
    shard = root / 'workers/00'
    row = batch.read(shard / 'manifest.json')[0]
    state = shard / 'state/image-00000001.json'
    batch.save(state, {**row, 'status': 'running', 'attempt': 2, 'initial_entity_dir': 'unchanged-cache'})
    seen = []
    def execute(path, **kwargs):
        saved = batch.read(state)
        seen.append(saved)
        assert saved['initial_entity_dir'] == 'unchanged-cache'
        assert saved['interrupted_attempts'] == 1
        assert batch.attempt_budget_used(saved) == 1
    monkeypatch.setattr(batch, 'run', execute)
    parallel.run(root)
    assert len(seen) == 2
    assert batch.attempt_budget_used({'attempt': 2}) == 2
