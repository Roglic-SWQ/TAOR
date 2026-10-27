"""Sentence-BERT semantic scorer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mmkgc.ranking.contracts import ScoreResult
from mmkgc.utils import clamp01

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    class SentenceTransformer:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("sentence-transformers is required for SentenceBERTScorer")


class SentenceBERTScorer:
    """Scores text candidates with a local Sentence-BERT model."""

    model_name = "sentence-bert"

    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model = SentenceTransformer(str(model_dir), device=device)

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        if not candidates:
            return []

        embeddings = self.model.encode(
            [reference, *candidates],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2:
            raise ValueError("SentenceBERTScorer expected 2D embeddings")
        expected_rows = len(candidates) + 1
        if array.shape[0] != expected_rows:
            raise ValueError(
                "SentenceBERTScorer embedding row count must match "
                f"reference plus candidates: expected {expected_rows}, got {array.shape[0]}"
            )
        if array.shape[1] == 0:
            raise ValueError("SentenceBERTScorer embedding vector dimension must be non-zero")
        if not np.isfinite(array).all():
            raise ValueError("SentenceBERTScorer expected finite embeddings")

        reference_embedding = array[0]
        candidate_embeddings = array[1:]
        results: list[ScoreResult] = []
        for index, candidate_embedding in enumerate(candidate_embeddings):
            cosine = float(np.dot(reference_embedding, candidate_embedding))
            normalized = clamp01((cosine + 1.0) / 2.0)
            results.append(
                ScoreResult(
                    raw_score=cosine,
                    normalized_score=normalized,
                    status="scored",
                    model_name=self.model_name,
                    model_version=str(self.model_dir),
                    details={"candidate_index": index},
                )
            )
        return results
