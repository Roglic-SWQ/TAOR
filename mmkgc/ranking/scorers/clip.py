"""CLIP visual scorer adapter for Stage 4 ranking."""

from __future__ import annotations

import math
from typing import Any

from mmkgc.ranking.contracts import ScoreResult
from mmkgc.ranking.image_input import ImageInput
from mmkgc.utils import clamp01


class CLIPRankingScorer:
    """Scores generated images through CLIP evaluator PIL hooks."""

    model_name = "clip-ranking"

    def __init__(self, evaluator: Any, *, model_version: str = "") -> None:
        self.evaluator = evaluator
        self.model_version = model_version

    def score_image_quality_batch(
        self,
        image: ImageInput,
        texts: list[str],
    ) -> list[ScoreResult]:
        return self._score_pil_texts(image, texts)

    def score_image_text_batch(
        self,
        image: ImageInput,
        texts: list[str],
    ) -> list[ScoreResult]:
        return self._score_pil_texts(image, texts)

    def image_embeddings(self, image: ImageInput) -> list[float]:
        pil_image = image.open_pil()
        try:
            embedding = self.evaluator.encode_pil_image(pil_image)
        finally:
            pil_image.close()
        values = [float(value) for value in embedding]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("CLIP image embeddings must be finite")
        return values

    def _score_pil_texts(self, image: ImageInput, texts: list[str]) -> list[ScoreResult]:
        if not texts:
            return []
        pil_image = image.open_pil()
        try:
            raw_scores = self.evaluator.score_pil_texts(pil_image, texts)
        finally:
            pil_image.close()
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(texts):
            raise ValueError(
                f"CLIP evaluator returned {len(scores)} scores for {len(texts)} texts"
            )
        return [
            ScoreResult(
                raw_score=score,
                normalized_score=clamp01(score / 100.0),
                status="scored",
                model_name=self.model_name,
                model_version=self.model_version,
                details={"candidate_index": index},
            )
            for index, score in enumerate(scores)
        ]
