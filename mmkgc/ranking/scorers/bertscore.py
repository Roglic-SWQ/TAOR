"""BERTScore auxiliary scorer adapter for Stage 4 ranking."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mmkgc.evaluation.text_quality import BERTScoreLocalScorer
from mmkgc.ranking.contracts import ScoreResult
from mmkgc.utils import clamp01


class BERTScoreAuxScorer:
    """Scores text candidates against a KG reference without affecting ranking weights."""

    model_name = "bertscore"

    def __init__(
        self,
        model_type: str | Path = "roberta-large",
        device: str | None = None,
        scorer: Any | None = None,
    ) -> None:
        self.model_type = str(model_type)
        self.device = device
        self.scorer = scorer or BERTScoreLocalScorer(
            model_type=self.model_type,
            device=device,
        )

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        if not candidates:
            return []
        if not reference.strip():
            return [
                ScoreResult(
                    raw_score=None,
                    normalized_score=None,
                    status="skipped",
                    model_name=self.model_name,
                    model_version=self.model_type,
                    details={"candidate_index": index, "reason": "empty_reference"},
                )
                for index, _candidate in enumerate(candidates)
            ]

        rows = self.scorer.score_pairs(candidates, [reference] * len(candidates))
        if len(rows) != len(candidates):
            raise ValueError(
                "BERTScoreAuxScorer row count must match candidates: "
                f"expected {len(candidates)}, got {len(rows)}"
            )

        results: list[ScoreResult] = []
        for index, row in enumerate(rows):
            precision = _required_score(row, "precision", index)
            recall = _required_score(row, "recall", index)
            f1 = _required_score(row, "f1", index)
            results.append(
                ScoreResult(
                    raw_score=f1,
                    normalized_score=clamp01(f1),
                    status="scored",
                    model_name=self.model_name,
                    model_version=self.model_type,
                    details={
                        "candidate_index": index,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "reference_kind": "kg_reference",
                    },
                )
            )
        return results


def _required_score(row: dict[str, Any], field: str, index: int) -> float:
    if field not in row:
        raise ValueError(f"BERTScore row {index} is missing {field}")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"BERTScore row {index} has non-finite {field}")
    return round(value, 6)
