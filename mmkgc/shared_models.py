"""Share local model weights across entity workers, keeping audit calls owned."""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
from pathlib import Path
import threading


class LockedLocalModel:
    """Serialize local GPU calls and retain each caller's own reranker audit."""

    def __init__(self, model, lock):
        self._delegate = model
        self._lock = lock
        self._local = threading.local()

    @property
    def calls(self):
        if not hasattr(self._local, 'calls'):
            self._local.calls = []
        return self._local.calls

    def __getattr__(self, name):
        attribute = getattr(self._delegate, name)
        if not callable(attribute):
            return attribute

        @wraps(attribute)
        def invoke(*args, **kwargs):
            with self._lock:
                history = getattr(self._delegate, 'calls', None)
                start = len(history) if isinstance(history, list) else 0
                try:
                    return attribute(*args, **kwargs)
                finally:
                    if isinstance(history, list):
                        self.calls.extend(history[start:])
                        del history[start:]
        return invoke


class SharedLocalModels:
    """One local model set; remote clients and pipeline audit stay per worker."""

    def __init__(self):
        self._initialization = threading.Lock()
        self._gpu = threading.RLock()
        self._models = {}
        self._datasets = {}

    def dataset_store(self, name, root, *, graph_split):
        """Share read-only source indexes; build_record returns owned records."""
        from mmkgc.data import build_dataset_store
        key = (name, str(Path(root).resolve()), graph_split)
        with self._initialization:
            if key not in self._datasets:
                self._datasets[key] = build_dataset_store(name, root, graph_split=graph_split)
            return self._datasets[key]

    def reranker(self, model_dir, **kwargs):
        from mmkgc.adapters.evidence_reranker import Qwen3EvidenceReranker
        with self._initialization:
            if 'reranker' not in self._models:
                self._models['reranker'] = LockedLocalModel(
                    Qwen3EvidenceReranker(model_dir, **kwargs), self._gpu)
            return self._models['reranker']

    def image_generator(self, config):
        from mmkgc.adapters.sdxl import SDXLImageGenerator
        with self._initialization:
            if 'sdxl' not in self._models:
                self._models['sdxl'] = LockedLocalModel(SDXLImageGenerator(config), self._gpu)
            return self._models['sdxl']

    def ranking_engine(self, project_root, stage4_config):
        from mmkgc.ranking.factory import build_model_fusion_engine
        from mmkgc.ranking.engine import ModelFusionRankingEngine
        with self._initialization:
            if 'ranking' not in self._models:
                template = build_model_fusion_engine(project_root, stage4_config)
                scorers = {name: scorer if name == 'gpt4omini_judge' else
                           LockedLocalModel(scorer, self._gpu)
                           for name, scorer in template.scorers.items()}
                # Entity workers provide the remote concurrency. Keeping each
                # engine serial prevents workers*candidates remote fan-out.
                profile = replace(template.profile, gpt4omini_judge=replace(
                    template.profile.gpt4omini_judge, max_concurrency=1))
                self._models['ranking'] = (profile, scorers)
            profile, scorers = self._models['ranking']
            return ModelFusionRankingEngine(profile=profile, scorers=scorers)
