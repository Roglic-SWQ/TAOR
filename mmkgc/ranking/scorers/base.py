"""Shared scorer protocols and test doubles."""

from __future__ import annotations

from typing import Protocol

from mmkgc.ranking.contracts import ScoreResult


class TextSemanticScorer(Protocol):
    """Scores candidate text against a reference text."""

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        ...


class FixedTextScorer:
    """Deterministic semantic scorer useful for tests and fallbacks."""

    def __init__(
        self,
        *,
        normalized_score: float = 0.0,
        raw_score: float | None = None,
        status: str = "scored",
        model_name: str = "fixed-text",
        model_version: str = "",
    ) -> None:
        self.normalized_score = normalized_score
        self.raw_score = raw_score
        self.status = status
        self.model_name = model_name
        self.model_version = model_version

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        del reference
        return [
            ScoreResult(
                raw_score=self.raw_score,
                normalized_score=self.normalized_score,
                status=self.status,  # type: ignore[arg-type]
                model_name=self.model_name,
                model_version=self.model_version,
                details={"candidate_index": index},
            )
            for index, _candidate in enumerate(candidates)
        ]
