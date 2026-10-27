"""BERTScore-based text quality evaluation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import os
from typing import Any, Protocol

from mmkgc.evaluation.image_quality_targets import build_entity_evidence_summary
from mmkgc.evaluation.loader import RunEntityArtifacts


class BERTScoreTextQualityError(RuntimeError):
    """Raised when BERTScore text quality evaluation cannot run safely."""


class BERTScoreScorer(Protocol):
    def score_pairs(
        self,
        candidates: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        """Return one BERTScore row for each candidate-reference pair."""


@dataclass
class FixedBERTScoreScorer:
    scores: list[Any]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def score_pairs(
        self,
        candidates: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        self.calls.append({"candidates": list(candidates), "references": list(references)})
        if len(candidates) != len(references):
            raise BERTScoreTextQualityError("candidates and references length mismatch")
        rows = self.scores
        if rows and isinstance(rows[0], dict):
            selected = rows
        else:
            selected = [
                {"precision": row[0], "recall": row[1], "f1": row[2]}
                for row in rows
            ]
        if len(selected) != len(candidates):
            raise BERTScoreTextQualityError("fake scorer row count mismatch")
        return [
            {
                "precision": round(float(row["precision"]), 6),
                "recall": round(float(row["recall"]), 6),
                "f1": round(float(row["f1"]), 6),
            }
            for row in selected
        ]


@dataclass(frozen=True)
class BERTScoreLocalScorer:
    model_type: str = "roberta-large"
    device: str | None = None
    _scorer: Any | None = field(default=None, init=False, repr=False, compare=False)

    def score_pairs(
        self,
        candidates: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        if len(candidates) != len(references):
            raise BERTScoreTextQualityError("candidates and references length mismatch")
        if not candidates:
            return []
        try:
            with _offline_model_env():
                precision, recall, f1 = self._cached_scorer().score(
                    candidates,
                    references,
                    verbose=False,
                )
            return [
                {
                    "precision": _round_score(_tensor_item(p)),
                    "recall": _round_score(_tensor_item(r)),
                    "f1": _round_score(_tensor_item(f)),
                }
                for p, r, f in zip(precision, recall, f1)
            ]
        except Exception as exc:  # pragma: no cover - exercised with dependency fakes.
            raise BERTScoreTextQualityError(
                f"BERTScore scoring failed for model_type={self.model_type!r}: {exc}"
            ) from exc

    def _cached_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer

        from bert_score import BERTScorer

        scorer = BERTScorer(
            model_type=self.model_type,
            num_layers=_recommended_num_layers(self.model_type),
            device=self.device,
        )
        object.__setattr__(self, "_scorer", scorer)
        return scorer


def _recommended_num_layers(model_type: str) -> int | None:
    if not os.path.isdir(model_type):
        return None

    from bert_score.utils import model2layers

    canonical_model_name = os.path.basename(os.path.normpath(model_type))
    if canonical_model_name not in model2layers:
        raise ValueError(
            "no official recommended BERTScore layer for local model "
            f"basename {canonical_model_name!r}"
        )
    return model2layers[canonical_model_name]


@contextmanager
def _offline_model_env() -> Any:
    """Force Hugging Face-backed BERTScore loading to use local cache only."""
    names = ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def evaluate_entity_text_quality(
    artifacts: RunEntityArtifacts,
    scorer: BERTScoreScorer,
    model_type: str,
) -> dict[str, Any]:
    """Evaluate ranked text candidates against ground truth or KG evidence."""
    entity = _entity(artifacts)
    reference_text, reference_kind = _reference(artifacts)
    text_candidates = _text_candidates(artifacts)

    score_rows: list[dict[str, Any]] = []
    scorable_indexes: list[int] = []
    scorable_candidates: list[str] = []
    scorable_references: list[str] = []

    for index, candidate in enumerate(text_candidates):
        candidate_text = _candidate_text(candidate)
        score_rows.append(_base_score_row(candidate, index, candidate_text))
        if candidate_text and reference_text:
            scorable_indexes.append(index)
            scorable_candidates.append(candidate_text)
            scorable_references.append(reference_text)

    if scorable_candidates:
        bertscore_rows = scorer.score_pairs(scorable_candidates, scorable_references)
        if len(bertscore_rows) != len(scorable_indexes):
            raise BERTScoreTextQualityError("scorer row count mismatch")
        for index, bertscore_row in zip(scorable_indexes, bertscore_rows):
            candidate_id = str(score_rows[index]["text_candidate_id"])
            precision = _required_score(
                bertscore_row,
                "precision",
                candidate_id=candidate_id,
            )
            recall = _required_score(
                bertscore_row,
                "recall",
                candidate_id=candidate_id,
            )
            f1 = _required_score(
                bertscore_row,
                "f1",
                candidate_id=candidate_id,
            )
            score_rows[index].update(
                {
                    "bertscore_precision": precision,
                    "bertscore_recall": recall,
                    "bertscore_f1": f1,
                    "bertscore_reference_kind": reference_kind,
                    "bertscore_overall_text_score": f1,
                }
            )

    scored_rows = [
        row for row in score_rows if row["bertscore_overall_text_score"] is not None
    ]
    averages = {
        "bertscore_precision": _average(row["bertscore_precision"] for row in scored_rows),
        "bertscore_recall": _average(row["bertscore_recall"] for row in scored_rows),
        "bertscore_f1": _average(row["bertscore_f1"] for row in scored_rows),
        "bertscore_overall_text_score": _average(
            row["bertscore_overall_text_score"] for row in scored_rows
        ),
    }
    best_row = _best_row(scored_rows)
    worst_row = _worst_row(scored_rows)

    return {
        "entity": entity,
        "model_type": model_type,
        "reference_kind": reference_kind if reference_text else "none",
        "text_candidate_count": len(text_candidates),
        "scored_text_candidate_count": len(scored_rows),
        "text_scores": score_rows,
        "averages": averages,
        "best_text_candidate_id": _row_candidate_id(best_row),
        "worst_text_candidate_id": _row_candidate_id(worst_row),
    }


def _entity(artifacts: RunEntityArtifacts) -> dict[str, Any]:
    entity = _as_dict(artifacts.final.get("entity")).copy()
    entity.setdefault("entity_key", artifacts.entity_key)
    return entity


def _reference(artifacts: RunEntityArtifacts) -> tuple[str, str]:
    final_entity = _as_dict(artifacts.final.get("entity"))
    ground_truth = _clean_text(final_entity.get("description"))
    if ground_truth:
        return ground_truth, "ground_truth"
    evidence_summary = _clean_text(build_entity_evidence_summary(artifacts))
    if evidence_summary:
        return evidence_summary, "evidence_summary"
    return "", "none"


def _text_candidates(artifacts: RunEntityArtifacts) -> list[dict[str, Any]]:
    text_ranking = _as_dict(artifacts.ranking.get("text_ranking"))
    candidates = text_ranking.get("ranked_candidates")
    if not isinstance(candidates, list):
        return []
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def _base_score_row(
    candidate: dict[str, Any],
    index: int,
    candidate_text: str,
) -> dict[str, Any]:
    return {
        "text_candidate_id": _candidate_id(candidate, index),
        "candidate_text": candidate_text,
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None,
        "bertscore_reference_kind": "unscored",
        "bertscore_overall_text_score": None,
    }


def _candidate_id(candidate: dict[str, Any], index: int) -> str:
    return _first_text(
        candidate.get("candidate_id"),
        candidate.get("text_candidate_id"),
        str(index),
    )


def _candidate_text(candidate: dict[str, Any]) -> str:
    generated_content = _as_dict(candidate.get("generated_content"))
    return _first_text(
        generated_content.get("value"),
        candidate.get("text"),
        candidate.get("value"),
        candidate.get("prompt_or_instruction"),
    )


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: row["bertscore_overall_text_score"])


def _worst_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: row["bertscore_overall_text_score"])


def _row_candidate_id(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return str(row["text_candidate_id"])


def _average(values: Any) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return _round_score(sum(numeric_values) / len(numeric_values))


def _round_score(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise BERTScoreTextQualityError("score must be finite")
    return round(number, 6)


def _required_score(
    row: dict[str, Any],
    field: str,
    *,
    candidate_id: str,
) -> float:
    if field not in row:
        raise BERTScoreTextQualityError(
            f"BERTScore row for {candidate_id} is missing {field}"
        )
    try:
        return _round_score(row[field])
    except (TypeError, ValueError) as exc:
        raise BERTScoreTextQualityError(
            f"BERTScore row for {candidate_id} has invalid {field}"
        ) from exc
    except BERTScoreTextQualityError as exc:
        raise BERTScoreTextQualityError(
            f"BERTScore row for {candidate_id} has non-finite {field}"
        ) from exc


def _tensor_item(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
