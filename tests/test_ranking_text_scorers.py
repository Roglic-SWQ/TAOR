from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmkgc.ranking.contracts import ScoreResult
from mmkgc.ranking.scorers import (
    BERTScoreAuxScorer,
    FixedTextScorer,
    SentenceBERTScorer,
)




class FakeSentenceTransformer:
    def __init__(self, embeddings=None) -> None:
        self.encode_calls: list[dict[str, object]] = []
        self.embeddings = embeddings

    def encode(self, texts, **kwargs):
        self.encode_calls.append({"texts": list(texts), "kwargs": kwargs})
        if self.embeddings is not None:
            return np.array(self.embeddings, dtype=float)
        vectors = []
        for text in texts:
            if "Example Film" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "film" in text:
                vectors.append([0.75, 0.25, 0.0])
            else:
                vectors.append([-1.0, 0.0, 0.0])
        return np.array(vectors, dtype=float)


class FakeBERTScoreBackend:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.calls = []

    def score_pairs(self, candidates, references):
        self.calls.append({"candidates": list(candidates), "references": list(references)})
        return list(self.rows)


def test_bertscore_aux_scorer_scores_against_repeated_reference() -> None:
    backend = FakeBERTScoreBackend(
        [
            {"precision": 0.91, "recall": 0.92, "f1": 0.93},
            {"precision": 0.81, "recall": 0.82, "f1": 0.83},
        ]
    )
    scorer = BERTScoreAuxScorer(
        model_type=Path("models/bertscore/roberta-large"),
        device="cpu",
        scorer=backend,
    )

    results = scorer.score_candidates("Example Film is a film.", ["A film.", "A city."])

    assert backend.calls == [
        {
            "candidates": ["A film.", "A city."],
            "references": [
                "Example Film is a film.",
                "Example Film is a film.",
            ],
        }
    ]
    assert results[0] == ScoreResult(
        raw_score=0.93,
        normalized_score=0.93,
        status="scored",
        model_name="bertscore",
        model_version="models/bertscore/roberta-large",
        details={
            "candidate_index": 0,
            "precision": 0.91,
            "recall": 0.92,
            "f1": 0.93,
            "reference_kind": "kg_reference",
        },
    )
    assert results[1].normalized_score == 0.83


def test_bertscore_aux_scorer_skips_empty_reference() -> None:
    backend = FakeBERTScoreBackend([])
    scorer = BERTScoreAuxScorer(scorer=backend)

    results = scorer.score_candidates("", ["A film."])

    assert backend.calls == []
    assert results == [
        ScoreResult(
            raw_score=None,
            normalized_score=None,
            status="skipped",
            model_name="bertscore",
            model_version="roberta-large",
            details={"candidate_index": 0, "reason": "empty_reference"},
        )
    ]


def test_bertscore_aux_scorer_rejects_wrong_row_count() -> None:
    scorer = BERTScoreAuxScorer(scorer=FakeBERTScoreBackend([]))

    with pytest.raises(ValueError, match="row count"):
        scorer.score_candidates("reference", ["candidate"])


def test_sentence_bert_batches_reference_and_candidates(monkeypatch) -> None:
    fake_model = FakeSentenceTransformer()
    monkeypatch.setattr(
        "mmkgc.ranking.scorers.sentence_bert.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )

    scorer = SentenceBERTScorer(
        model_dir=Path("Sentence-BERT/all-mpnet-base-v2"),
        device="cpu",
    )
    results = scorer.score_candidates(
        "Example Film is a film.",
        ["A film.", "A mountain."],
    )

    assert len(fake_model.encode_calls) == 1
    call = fake_model.encode_calls[0]
    assert call["texts"] == [
        "Example Film is a film.",
        "A film.",
        "A mountain.",
    ]
    assert call["kwargs"]["normalize_embeddings"] is True
    assert call["kwargs"]["convert_to_numpy"] is True
    assert all(isinstance(row, ScoreResult) for row in results)
    assert all(0.0 <= row.normalized_score <= 1.0 for row in results)


def test_sentence_bert_handles_empty_candidate_list(monkeypatch) -> None:
    fake_model = FakeSentenceTransformer()
    monkeypatch.setattr(
        "mmkgc.ranking.scorers.sentence_bert.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )

    scorer = SentenceBERTScorer(model_dir=Path("Sentence-BERT/all-mpnet-base-v2"))

    assert scorer.score_candidates("Example Film is a film.", []) == []
    assert fake_model.encode_calls == []


def test_sentence_bert_rejects_too_few_embedding_rows(monkeypatch) -> None:
    fake_model = FakeSentenceTransformer(embeddings=[[1.0, 0.0, 0.0]])
    monkeypatch.setattr(
        "mmkgc.ranking.scorers.sentence_bert.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )
    scorer = SentenceBERTScorer(model_dir=Path("Sentence-BERT/all-mpnet-base-v2"))

    with pytest.raises(ValueError, match="embedding row count"):
        scorer.score_candidates("Example Film is a film.", ["A film."])


def test_sentence_bert_rejects_non_2d_embeddings(monkeypatch) -> None:
    fake_model = FakeSentenceTransformer(embeddings=[1.0, 0.0, 0.0])
    monkeypatch.setattr(
        "mmkgc.ranking.scorers.sentence_bert.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )
    scorer = SentenceBERTScorer(model_dir=Path("Sentence-BERT/all-mpnet-base-v2"))

    with pytest.raises(ValueError, match="2D"):
        scorer.score_candidates("Example Film is a film.", ["A film."])


def test_sentence_bert_rejects_non_finite_embeddings(monkeypatch) -> None:
    fake_model = FakeSentenceTransformer(
        embeddings=[
            [1.0, 0.0, 0.0],
            [float("nan"), 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(
        "mmkgc.ranking.scorers.sentence_bert.SentenceTransformer",
        lambda *args, **kwargs: fake_model,
    )
    scorer = SentenceBERTScorer(model_dir=Path("Sentence-BERT/all-mpnet-base-v2"))

    with pytest.raises(ValueError, match="finite embeddings"):
        scorer.score_candidates("Example Film is a film.", ["A film."])


def test_fixed_text_scorer_returns_configured_score_results() -> None:
    scorer = FixedTextScorer(normalized_score=0.25, raw_score=-0.5, model_name="fixed-test")

    results = scorer.score_candidates("reference", ["first", "second"])

    assert results == [
        ScoreResult(
            raw_score=-0.5,
            normalized_score=0.25,
            status="scored",
            model_name="fixed-test",
            details={"candidate_index": 0},
        ),
        ScoreResult(
            raw_score=-0.5,
            normalized_score=0.25,
            status="scored",
            model_name="fixed-test",
            details={"candidate_index": 1},
        ),
    ]
