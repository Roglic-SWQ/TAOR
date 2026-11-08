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

    def worker(shard, retry_failed=False, *, shared_local_models=None, max_attempts=None):
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
