"""Concurrent entities must not duplicate GPU models or mix their call records."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import time

import pytest

from mmkgc.shared_models import LockedLocalModel, SharedLocalModels


def test_many_workers_share_one_dataset_index_without_duplicate_loads(monkeypatch, tmp_path):
    from mmkgc import data
    built = []
    store = object()
    def build(*args, **kwargs):
        built.append((args, kwargs))
        return store
    monkeypatch.setattr(data, 'build_dataset_store', build)
    shared = SharedLocalModels()
    with ThreadPoolExecutor(max_workers=128) as pool:
        results = list(pool.map(lambda _: shared.dataset_store('wn18rr', tmp_path, graph_split='train'), range(256)))
    assert len(built) == 1
    assert all(result is store for result in results)

def test_concurrent_reranking_keeps_each_entities_audit_even_on_failure():
    barrier = threading.Barrier(10)

    class Reranker:
        def __init__(self):
            self.calls, self.active, self.peak = [], 0, 0

        def score(self, key):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                time.sleep(0.002)
                if key == 3:
                    raise ValueError('one entity failed')
                return key
            finally:
                self.calls.append({'entity_key': key})
                self.active -= 1

    delegate = Reranker()
    shared = LockedLocalModel(delegate, threading.RLock())

    def worker(key):
        barrier.wait(timeout=3)
        try:
            shared.score(key)
        except ValueError:
            assert key == 3
        result = list(shared.calls)
        shared.calls.clear()
        return key, result

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, range(10)))
    assert delegate.peak == 1
    assert not delegate.calls
    assert all(calls == [{'entity_key': key}] for key, calls in results)


def test_sdxl_is_constructed_once_and_serialized_across_entity_workers(monkeypatch):
    from mmkgc.adapters import sdxl
    constructed = []

    class Generator:
        def __init__(self, config):
            constructed.append(config)
            self.active = self.peak = 0

        def generate(self, key):
            self.active += 1
            self.peak = max(self.peak, self.active)
            time.sleep(0.002)
            self.active -= 1
            return key

    monkeypatch.setattr(sdxl, 'SDXLImageGenerator', Generator)
    pool = SharedLocalModels()
    with ThreadPoolExecutor(max_workers=10) as executor:
        generators = list(executor.map(lambda _: pool.image_generator('frozen-config'), range(10)))
        results = list(executor.map(generators[0].generate, range(10)))
    assert constructed == ['frozen-config']
    assert len({id(generator) for generator in generators}) == 1
    assert results == list(range(10))
    assert generators[0]._delegate.peak == 1


def test_rank_engines_share_local_weights_but_have_separate_control_state(monkeypatch):
    from mmkgc.ranking import factory
    from mmkgc.ranking.profiles import GPT4OMiniJudgeConfig, LocalScorerConfig, Stage4Profile
    disabled = LocalScorerConfig(False, False, None, None)
    profile = Stage4Profile('shared', disabled, disabled, disabled, disabled,
                           GPT4OMiniJudgeConfig(True, 'unused.yaml', 120, max_concurrency=10))
    judge, local = object(), object()
    built = []

    def build(*args):
        built.append(args)
        return SimpleNamespace(profile=profile, scorers={'gpt4omini_judge': judge, 'clip': local})

    monkeypatch.setattr(factory, 'build_model_fusion_engine', build)
    pool = SharedLocalModels()
    engines = [pool.ranking_engine('root', 'profile') for _ in range(10)]
    assert len(built) == 1
    assert len({id(engine) for engine in engines}) == 10
    assert len({id(engine.scorers['clip']) for engine in engines}) == 1
    assert all(engine.scorers['gpt4omini_judge'] is judge for engine in engines)
    assert all(engine.profile.gpt4omini_judge.max_concurrency == 1 for engine in engines)
    engines[0].judge_wait_callback = lambda: None
    assert all(engine.judge_wait_callback is None for engine in engines[1:])
