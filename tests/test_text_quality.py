from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mmkgc.evaluation.loader import RunEntityArtifacts
from mmkgc.evaluation.text_quality import (
    BERTScoreLocalScorer,
    BERTScoreTextQualityError,
    FixedBERTScoreScorer,
    evaluate_entity_text_quality,
)


def _artifacts(
    entity_dir: Path,
    *,
    description: str = "Example Film is a drama film from Exampleland.",
    text_candidates: list[dict[str, object]] | None = None,
    knowledge: dict[str, object] | None = None,
    reasoning: dict[str, object] | None = None,
) -> RunEntityArtifacts:
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key="Q1",
        knowledge=knowledge
        or {
            "entity_profile": {
                "entity_key": "Q1",
                "entity_label": "Example Film",
                "entity_type": "film",
            },
            "relation_facts": [
                {"relation_label": "director", "neighbor_label": "Jane Example"}
            ],
        },
        reasoning=reasoning
        or {
            "modality_constraints": {
                "image": {"must_include": ["film poster"]},
            }
        },
        ranking={
            "text_ranking": {
                "ranked_candidates": text_candidates
                if text_candidates is not None
                else [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "A drama film directed by Jane Example."
                        },
                    },
                    {
                        "candidate_id": "text_002",
                        "generated_content": {
                            "value": "An unrelated candidate description."
                        },
                    },
                ]
            }
        },
        final={
            "entity": {
                "entity_key": "Q1",
                "label": "Example Film",
                "description": description,
            },
            "missing_mode": "both",
        },
    )


def test_evaluate_entity_text_quality_uses_ground_truth_description(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path / "Q1")
    scorer = FixedBERTScoreScorer([(0.91, 0.81, 0.86), (0.31, 0.21, 0.26)])

    result = evaluate_entity_text_quality(
        artifacts,
        scorer=scorer,
        model_type="roberta-large",
    )

    assert scorer.calls == [
        {
            "candidates": [
                "A drama film directed by Jane Example.",
                "An unrelated candidate description.",
            ],
            "references": [
                "Example Film is a drama film from Exampleland.",
                "Example Film is a drama film from Exampleland.",
            ],
        }
    ]
    assert result["entity"]["entity_key"] == "Q1"
    assert result["model_type"] == "roberta-large"
    assert result["text_candidate_count"] == 2
    assert result["scored_text_candidate_count"] == 2
    assert result["best_text_candidate_id"] == "text_001"
    assert result["worst_text_candidate_id"] == "text_002"
    assert result["averages"]["bertscore_precision"] == pytest.approx(0.61)
    assert result["averages"]["bertscore_recall"] == pytest.approx(0.51)
    assert result["averages"]["bertscore_f1"] == pytest.approx(0.56)
    assert result["averages"]["bertscore_overall_text_score"] == pytest.approx(0.56)
    first = result["text_scores"][0]
    assert first["text_candidate_id"] == "text_001"
    assert first["bertscore_precision"] == 0.91
    assert first["bertscore_recall"] == 0.81
    assert first["bertscore_f1"] == 0.86
    assert first["bertscore_reference_kind"] == "ground_truth"
    assert first["bertscore_overall_text_score"] == 0.86


def test_evaluate_entity_text_quality_falls_back_to_evidence_summary(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path / "Q1", description="")
    scorer = FixedBERTScoreScorer(
        [
            {"precision": 0.4, "recall": 0.5, "f1": 0.45},
            {"precision": 0.2, "recall": 0.3, "f1": 0.25},
        ]
    )

    result = evaluate_entity_text_quality(
        artifacts,
        scorer=scorer,
        model_type="roberta-large",
    )

    assert len(scorer.calls[0]["candidates"]) == 2
    assert "Example Film" in scorer.calls[0]["references"][0]
    assert "Jane Example" in scorer.calls[0]["references"][0]
    assert result["text_scores"][0]["bertscore_reference_kind"] == "evidence_summary"
    assert result["text_scores"][0]["bertscore_f1"] == 0.45


def test_evaluate_entity_text_quality_marks_candidates_unscored_without_reference(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path / "Q1",
        description="",
        knowledge={"entity_profile": {}, "relation_facts": []},
        reasoning={"modality_constraints": {}},
    )
    artifacts.final["entity"]["label"] = ""
    artifacts.final["entity"]["entity_key"] = ""
    scorer = FixedBERTScoreScorer([])

    result = evaluate_entity_text_quality(
        artifacts,
        scorer=scorer,
        model_type="roberta-large",
    )

    assert scorer.calls == []
    assert result["text_candidate_count"] == 2
    assert result["scored_text_candidate_count"] == 0
    assert result["best_text_candidate_id"] is None
    assert result["worst_text_candidate_id"] is None
    assert result["averages"]["bertscore_f1"] is None
    assert result["text_scores"][0]["bertscore_reference_kind"] == "unscored"
    assert result["text_scores"][0]["bertscore_overall_text_score"] is None


def test_evaluate_entity_text_quality_skips_blank_candidates(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path / "Q1",
        text_candidates=[
            {"candidate_id": "text_blank", "generated_content": {}},
            {"candidate_id": "text_real", "text": "A useful text candidate."},
        ],
    )
    scorer = FixedBERTScoreScorer([(0.7, 0.8, 0.75)])

    result = evaluate_entity_text_quality(
        artifacts,
        scorer=scorer,
        model_type="roberta-large",
    )

    assert scorer.calls[0]["candidates"] == ["A useful text candidate."]
    assert result["scored_text_candidate_count"] == 1
    assert result["text_scores"][0]["text_candidate_id"] == "text_blank"
    assert result["text_scores"][0]["bertscore_reference_kind"] == "unscored"
    assert result["text_scores"][1]["text_candidate_id"] == "text_real"
    assert result["best_text_candidate_id"] == "text_real"
    assert result["worst_text_candidate_id"] == "text_real"


def test_fixed_bertscore_scorer_validates_input_lengths() -> None:
    scorer = FixedBERTScoreScorer([(0.1, 0.2, 0.3)])

    with pytest.raises(BERTScoreTextQualityError, match="length mismatch"):
        scorer.score_pairs(["candidate"], ["reference", "extra"])


def test_fixed_bertscore_scorer_validates_row_count() -> None:
    scorer = FixedBERTScoreScorer([(0.1, 0.2, 0.3)])

    with pytest.raises(BERTScoreTextQualityError, match="row count mismatch"):
        scorer.score_pairs(["one", "two"], ["ref one", "ref two"])


def test_bert_score_local_scorer_uses_installed_bert_score_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bert_score

    observed_kwargs: dict[str, object] = {}
    score_calls: list[dict[str, object]] = []

    class _FakeTensorValue:
        def item(self) -> float:
            return 0.5

    class FakeBERTScorer:
        def __init__(self, **kwargs: object) -> None:
            observed_kwargs.update(kwargs)

        def score(
            self,
            candidates: list[str],
            references: list[str],
            **kwargs: object,
        ) -> tuple[list[_FakeTensorValue], ...]:
            score_calls.append(
                {
                    "candidates": candidates,
                    "references": references,
                    "kwargs": kwargs,
                }
            )
            return ([_FakeTensorValue()], [_FakeTensorValue()], [_FakeTensorValue()])

    monkeypatch.setattr(bert_score, "BERTScorer", FakeBERTScorer)

    result = BERTScoreLocalScorer(device="cpu").score_pairs(
        ["candidate"],
        ["reference"],
    )

    assert result == [{"precision": 0.5, "recall": 0.5, "f1": 0.5}]
    assert observed_kwargs["num_layers"] is None
    assert score_calls == [
        {
            "candidates": ["candidate"],
            "references": ["reference"],
            "kwargs": {"verbose": False},
        }
    ]


def test_bert_score_local_scorer_reuses_bert_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bert_score

    init_count = 0

    class _FakeTensorValue:
        def item(self) -> float:
            return 0.5

    class FakeBERTScorer:
        def __init__(self, **kwargs: object) -> None:
            nonlocal init_count
            init_count += 1

        def score(
            self,
            candidates: list[str],
            references: list[str],
            **kwargs: object,
        ) -> tuple[list[_FakeTensorValue], ...]:
            return ([_FakeTensorValue()], [_FakeTensorValue()], [_FakeTensorValue()])

    monkeypatch.setattr(bert_score, "BERTScorer", FakeBERTScorer)
    scorer = BERTScoreLocalScorer(device="cpu")

    scorer.score_pairs(["candidate one"], ["reference one"])
    scorer.score_pairs(["candidate two"], ["reference two"])

    assert init_count == 1


def test_bert_score_local_scorer_passes_local_model_num_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bert_score

    model_dir = tmp_path / "roberta-large"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "roberta", "num_hidden_layers": 24}',
        encoding="utf-8",
    )
    observed_kwargs: dict[str, object] = {}

    class _FakeTensorValue:
        def item(self) -> float:
            return 0.5

    class FakeBERTScorer:
        def __init__(self, **kwargs: object) -> None:
            observed_kwargs.update(kwargs)

        def score(
            self,
            candidates: list[str],
            references: list[str],
            **kwargs: object,
        ) -> tuple[list[_FakeTensorValue], ...]:
            return ([_FakeTensorValue()], [_FakeTensorValue()], [_FakeTensorValue()])

    monkeypatch.setattr(bert_score, "BERTScorer", FakeBERTScorer)

    BERTScoreLocalScorer(model_type=str(model_dir), device="cpu").score_pairs(
        ["candidate"],
        ["reference"],
    )

    assert observed_kwargs["num_layers"] == 17


def test_bert_score_local_scorer_rejects_unknown_local_model_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bert_score

    model_dir = tmp_path / "unknown-local-model"
    model_dir.mkdir()

    with pytest.raises(
        BERTScoreTextQualityError,
        match="no official recommended BERTScore layer",
    ):
        BERTScoreLocalScorer(model_type=str(model_dir), device="cpu").score_pairs(
            ["candidate"],
            ["reference"],
        )


def test_bert_score_local_scorer_converts_dependency_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "bert_score":
            raise ImportError("missing bert_score")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    scorer = BERTScoreLocalScorer(model_type="roberta-large", device="cpu")

    with pytest.raises(BERTScoreTextQualityError, match="BERTScore scoring failed"):
        scorer.score_pairs(["candidate"], ["reference"])


def test_bert_score_local_scorer_forces_offline_model_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__
    observed_env: dict[str, str | None] = {}

    class _FakeTensorValue:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class FakeBERTScorer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def score(
            self,
            candidates: list[str],
            references: list[str],
            **kwargs: object,
        ) -> tuple[list[_FakeTensorValue], ...]:
            import os

            observed_env["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
            observed_env["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
            return (
                [_FakeTensorValue(0.7)],
                [_FakeTensorValue(0.8)],
                [_FakeTensorValue(0.75)],
            )

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "bert_score":
            return SimpleNamespace(BERTScorer=FakeBERTScorer)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    scorer = BERTScoreLocalScorer(model_type="roberta-large", device="cpu")

    result = scorer.score_pairs(["candidate"], ["reference"])

    assert result == [{"precision": 0.7, "recall": 0.8, "f1": 0.75}]
    assert observed_env == {"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}


class _MalformedScorer:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def score_pairs(
        self,
        candidates: list[str],
        references: list[str],
    ) -> list[dict[str, object]]:
        return self.rows


def test_evaluate_entity_text_quality_rejects_missing_scorer_fields(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path / "Q1",
        text_candidates=[
            {
                "candidate_id": "text_001",
                "generated_content": {"value": "A drama film."},
            }
        ],
    )
    scorer = _MalformedScorer([{"precision": 0.7, "recall": 0.8}])

    with pytest.raises(BERTScoreTextQualityError, match="missing f1"):
        evaluate_entity_text_quality(
            artifacts,
            scorer=scorer,
            model_type="roberta-large",
        )


def test_evaluate_entity_text_quality_rejects_non_finite_scorer_values(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path / "Q1",
        text_candidates=[
            {
                "candidate_id": "text_001",
                "generated_content": {"value": "A drama film."},
            }
        ],
    )
    scorer = _MalformedScorer(
        [{"precision": 0.7, "recall": 0.8, "f1": float("nan")}]
    )

    with pytest.raises(BERTScoreTextQualityError, match="non-finite f1"):
        evaluate_entity_text_quality(
            artifacts,
            scorer=scorer,
            model_type="roberta-large",
        )
