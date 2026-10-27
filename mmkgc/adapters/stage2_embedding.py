"""Local Sentence-BERT similarity adapter for Stage 2 relation selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SentenceBertStage2EmbeddingScorer:
    """Cache normalized sentence embeddings and expose bounded similarity scores."""

    def __init__(self, model_dir: str | Path, *, rescale_cosine: bool = True) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ValueError(
                "Stage 2 embedding scoring requires numpy. Please install numpy."
            ) from exc
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError(
                "Stage 2 embedding scoring requires sentence-transformers. "
                "Please install sentence-transformers."
            ) from exc

        self._np = np
        self.model_dir = str(model_dir)
        self.rescale_cosine = bool(rescale_cosine)
        self._embedding_cache: dict[str, Any] = {}
        try:
            self._model = SentenceTransformer(self.model_dir)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Stage 2 Sentence-BERT model from {self.model_dir}: {exc}"
            ) from exc

    def similarity(self, left: str, right: str) -> float:
        """Return normalized cosine similarity on the closed interval [0, 1]."""
        cosine = float(self._np.dot(self._embedding(left), self._embedding(right)))
        cosine = min(1.0, max(-1.0, cosine))
        if self.rescale_cosine:
            return (cosine + 1.0) / 2.0
        return max(0.0, cosine)

    def _embedding(self, text: str) -> Any:
        normalized_text = str(text)
        cached = self._embedding_cache.get(normalized_text)
        if cached is not None:
            return cached
        try:
            embedding = self._np.asarray(
                self._model.encode(
                    normalized_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=float,
            ).reshape(-1)
        except Exception as exc:
            raise RuntimeError("failed to encode text with Stage 2 Sentence-BERT model") from exc
        norm = float(self._np.linalg.norm(embedding))
        if embedding.size == 0 or norm == 0.0:
            raise ValueError("Stage 2 Sentence-BERT model returned an empty or zero embedding")
        normalized = embedding / norm
        self._embedding_cache[normalized_text] = normalized
        return normalized


def build_stage2_embedding_scorer(
    kind: str,
    model_dir: str | Path | None,
    *,
    rescale_cosine: bool = True,
) -> SentenceBertStage2EmbeddingScorer | None:
    """Build the requested Stage 2 scorer without affecting the rule default."""
    normalized_kind = str(kind).strip().lower().replace("_", "-")
    if normalized_kind == "rule-weighted":
        return None
    if normalized_kind != "embedding-component":
        raise ValueError(
            "stage2 scorer must be one of: rule-weighted, embedding-component"
        )
    if not str(model_dir or "").strip():
        raise ValueError("stage2 embedding model directory is required for embedding-component")
    return SentenceBertStage2EmbeddingScorer(
        model_dir,
        rescale_cosine=rescale_cosine,
    )
