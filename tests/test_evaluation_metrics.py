from __future__ import annotations

import pytest

from mmkgc.evaluation.loader import RunEntityArtifacts
from mmkgc.evaluation.metrics import (
    average_top,
    candidate_text,
    is_fallback_candidate,
    score_image_candidate,
    score_pair,
    score_text_candidate,
)
from mmkgc.evaluation.evaluator import evaluate_entity
from mmkgc.utils import token_cosine, token_overlap


def test_score_text_candidate_uses_ground_truth_when_available() -> None:
    candidate = {
        "generated_content": {"value": "Ada Lovelace mathematician writer"},
        "final_score": 0.5,
    }

    result = score_text_candidate(
        candidate,
        "Ada Lovelace was a mathematician",
        "Ada Lovelace",
        "mathematician",
    )

    expected = (
        0.6 * token_cosine(candidate_text(candidate), "Ada Lovelace was a mathematician")
        + 0.2 * token_overlap(candidate_text(candidate), ["Ada Lovelace", "mathematician"])
        + 0.2 * 0.5
    )
    assert result["ground_truth_available"] is True
    assert result["ground_truth_similarity_score"] > 0
    assert result["offline_evaluation_score"] == pytest.approx(round(expected, 6))
    assert "score" not in result
    assert result["stage4_ranking_score"] == pytest.approx(0.5)


def test_score_text_candidate_falls_back_without_ground_truth() -> None:
    candidate = {
        "prompt_or_instruction": "Fallback profile text",
        "final_score": 0.8,
        "relation_coverage_score": 0.5,
        "must_term_score": 0.25,
    }

    result = score_text_candidate(candidate, "", "Example", "film")

    assert result["ground_truth_available"] is False
    assert result["offline_evaluation_score"] == pytest.approx(0.6)
    assert "score" not in result


def test_score_image_candidate_counts_relation_and_constraints() -> None:
    candidate = {
        "positive_prompt": "Portrait of Ada Lovelace",
        "prompt_or_instruction": "mathematician",
        "generated_content": {"value": "with analytical engine"},
        "expected_relation_coverage": ["analytical engine"],
        "used_constraints": [{"constraint": "portrait"}],
        "score": 0.5,
    }

    result = score_image_candidate(candidate, "Ada Lovelace", "mathematician")

    assert result["relation_coverage"] == 1.0
    assert result["constraint_coverage"] == 1.0
    assert result["entity_type_coverage"] == 1.0
    assert result["offline_evaluation_score"] == pytest.approx(0.9)
    assert "score" not in result
    assert result["stage4_ranking_score"] == pytest.approx(0.5)


def test_score_image_candidate_scores_forbidden_terms_from_negative_prompt() -> None:
    candidate = {
        "positive_prompt": "Ada Lovelace portrait",
        "negative_prompt": "avoid logos and watermark",
        "applied_forbidden_content": ["logos", "watermark", "weapons"],
        "score": 0.5,
    }

    result = score_image_candidate(candidate, "Ada Lovelace", "person")

    assert result["forbidden_awareness_score"] == pytest.approx(
        token_overlap("avoid logos and watermark", ["logos", "watermark", "weapons"])
    )


def test_score_pair_uses_balance_and_stage_scores() -> None:
    pair = {
        "score": 0.7,
        "text_candidate": {"score": 0.8, "generated_content": {"value": "Ada mathematician"}},
        "image_candidate": {"final_score": 0.6, "positive_prompt": "Ada portrait"},
        "pair_similarity_score": 0.25,
        "kg_anchor_score": 0.5,
    }

    result = score_pair(pair)

    assert result["pair_balance_score"] == pytest.approx(0.8)
    assert result["stage4_text_score"] == pytest.approx(0.8)
    assert result["stage4_image_score"] == pytest.approx(0.6)
    assert "text_" + "stage4_score" not in result
    assert "image_" + "stage4_score" not in result
    assert result["offline_evaluation_score"] == pytest.approx(0.62)
    assert "score" not in result
    assert result["stage4_pair_score"] == pytest.approx(0.7)


def test_is_fallback_candidate_detects_global_profile_and_metadata() -> None:
    assert is_fallback_candidate({"primary_relation": "global_profile"})
    assert is_fallback_candidate({"metadata": {"fallback": True}})
    assert is_fallback_candidate({"relation_role": "fallback"})
    assert is_fallback_candidate(
        {"expected_relation_coverage_detail": {"primary": "global_profile"}}
    )
    assert not is_fallback_candidate({"primary_relation": "occupation"})


def test_candidate_text_prefers_generated_content_value() -> None:
    assert (
        candidate_text(
            {
                "generated_content": {"value": "generated text"},
                "positive_prompt": "positive prompt",
                "prompt_or_instruction": "instruction",
            }
        )
        == "generated text"
    )


def test_candidate_text_falls_back_to_prompt_or_instruction() -> None:
    assert (
        candidate_text(
            {
                "generated_content": {"value": ""},
                "positive_prompt": "positive prompt",
                "prompt_or_instruction": "instruction",
            }
        )
        == "instruction"
    )


def test_average_top_returns_zero_for_empty_items() -> None:
    assert average_top([], "score") == 0.0


def test_average_top_sorts_by_key_and_respects_limit() -> None:
    items = [
        {"score": 0.2, "final_score": 0.95},
        {"score": 0.8, "final_score": 0.1},
        {"score": 0.4, "final_score": 0.9},
        {"score": 0.6, "final_score": 0.2},
    ]

    assert average_top(items, "score", limit=2) == pytest.approx(0.7)
    assert average_top(items, "final_score", limit=2) == pytest.approx(0.925)


def test_non_numeric_stage4_scores_fall_back_to_zero() -> None:
    text_result = score_text_candidate(
        {
            "generated_content": {"value": "Ada Lovelace mathematician"},
            "score": "not numeric",
        },
        "",
        "Ada Lovelace",
        "mathematician",
    )
    image_result = score_image_candidate(
        {
            "positive_prompt": "Ada Lovelace mathematician",
            "expected_relation_coverage": [],
            "used_constraints": [],
            "final_score": object(),
        },
        "Ada Lovelace",
        "mathematician",
    )
    pair_result = score_pair(
        {
            "score": "not numeric",
            "text_candidate": {"score": object()},
            "image_candidate": {"final_score": "not numeric"},
            "pair_similarity_score": "not numeric",
            "kg_anchor_score": object(),
        }
    )

    assert text_result["stage4_ranking_score"] == 0.0
    assert image_result["stage4_ranking_score"] == 0.0
    assert pair_result["stage4_pair_score"] == 0.0
    assert pair_result["stage4_text_score"] == 0.0
    assert pair_result["stage4_image_score"] == 0.0
    assert pair_result["offline_evaluation_score"] == pytest.approx(0.3)


def test_evaluate_entity_returns_three_layer_scores(tmp_path) -> None:
    artifacts = RunEntityArtifacts(
        entity_dir=tmp_path / "Q1",
        entity_key="Q1",
        knowledge={"entity_profile": {"entity_type": "person"}},
        reasoning={},
        ranking={
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text-1",
                        "generated_content": {"value": "Ada Lovelace mathematician"},
                        "final_score": 0.8,
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image-1",
                        "positive_prompt": "Ada Lovelace person portrait",
                        "expected_relation_coverage": ["portrait"],
                        "used_constraints": [{"constraint": "portrait"}],
                        "final_score": 0.7,
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "pair_id": "pair-1",
                        "score": 0.6,
                        "text_candidate": {"score": 0.8},
                        "image_candidate": {"score": 0.7},
                        "pair_similarity_score": 0.5,
                        "kg_anchor_score": 0.4,
                    }
                ]
            },
        },
        final={
            "missing_mode": "image",
            "entity": {
                "entity_key": "Q1",
                "label": "Ada Lovelace",
                "description": "Ada Lovelace was a mathematician.",
            },
        },
    )

    result = evaluate_entity(artifacts)

    assert result["text_evaluation"]["best_offline_evaluation_score"] > 0
    assert result["image_evaluation"]["best_offline_evaluation_score"] > 0
    assert result["pair_evaluation"]["best_offline_evaluation_score"] > 0
    assert "best" + "_score" not in result["text_evaluation"]
    assert "top5_avg" + "_score" not in result["text_evaluation"]
    assert result["text_evaluation"]["best_candidate_id"] == "text-1"
    assert result["image_evaluation"]["best_candidate_id"] == "image-1"
    assert result["pair_evaluation"]["best_pair_id"] == "pair-1"
    assert "overall_offline_evaluation_score" in result
    assert "calibrated_overall_offline_evaluation_score" in result
    assert "calibration_summary" in result
    assert result["overall_offline_evaluation_score"] > 0
    assert "overall" + "_score" not in result


def test_evaluate_entity_adds_calibrated_scores_for_fallback(tmp_path) -> None:
    artifacts = RunEntityArtifacts(
        entity_dir=tmp_path / "Q-fallback",
        entity_key="Q-fallback",
        knowledge={"entity_profile": {"entity_type": "entity"}},
        reasoning={},
        ranking={
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text-fallback",
                        "generated_content": {
                            "value": "As a entity type entity, Fallback Entity has profile context."
                        },
                        "metadata": {"fallback": True},
                        "primary_relation": "global_profile",
                        "final_score": 0.95,
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image-fallback",
                        "positive_prompt": (
                            "Fallback Entity, entity, identity, profile context"
                        ),
                        "metadata": {"fallback": True},
                        "primary_relation": "global_profile",
                        "final_score": 0.96,
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "pair_id": "pair-fallback",
                        "score": 0.97,
                        "text_candidate": {
                            "candidate_id": "pair-text-fallback",
                            "generated_content": {
                                "value": "Fallback Entity generic profile context."
                            },
                            "metadata": {"fallback": True},
                            "primary_relation": "global_profile",
                            "score": 0.95,
                        },
                        "image_candidate": {
                            "candidate_id": "pair-image-fallback",
                            "positive_prompt": (
                                "Fallback Entity, entity, identity, profile context"
                            ),
                            "metadata": {"fallback": True},
                            "primary_relation": "global_profile",
                            "score": 0.96,
                        },
                        "pair_similarity_score": 0.9,
                        "kg_anchor_score": 0.9,
                    }
                ]
            },
        },
        final={
            "missing_mode": "image",
            "entity": {
                "entity_key": "Q-fallback",
                "label": "Fallback Entity",
                "description": "Fallback Entity is an entity profile used for evaluation.",
            },
        },
    )

    result = evaluate_entity(artifacts)

    for key in (
        "calibrated_text_best_offline_evaluation_score",
        "calibrated_image_best_offline_evaluation_score",
        "calibrated_pair_best_offline_evaluation_score",
        "calibrated_overall_offline_evaluation_score",
        "calibration_summary",
    ):
        assert key in result
    assert (
        result["overall_offline_evaluation_score"]
        > result["calibrated_overall_offline_evaluation_score"]
    )
    assert result["calibration_summary"]["fallback_penalty"] > 0
    assert "fallback_penalized" in result["calibration_summary"]["calibration_tags"]
    assert "fallback_penalized" in result["case_tags"]
    assert "text_candidate" in result["pair_evaluation"]["ranked_pairs"][0]
    assert "image_candidate" in result["pair_evaluation"]["ranked_pairs"][0]


def test_evaluate_entity_handles_empty_candidates(tmp_path) -> None:
    artifacts = RunEntityArtifacts(
        entity_dir=tmp_path / "Q2",
        entity_key="Q2",
        knowledge={},
        reasoning={},
        ranking={
            "text_ranking": {"ranked_candidates": []},
            "image_ranking": {"ranked_candidates": []},
            "pair_ranking": {"ranked_pairs": []},
        },
        final={"entity": {"entity_key": "Q2", "label": "Empty Entity"}},
    )

    result = evaluate_entity(artifacts)

    assert result["text_evaluation"]["best_offline_evaluation_score"] == 0.0
    assert result["image_evaluation"]["best_offline_evaluation_score"] == 0.0
    assert result["pair_evaluation"]["best_offline_evaluation_score"] == 0.0
    assert result["overall_offline_evaluation_score"] == 0.0
    assert "missing_text_candidates" in result["case_tags"]
    assert "missing_image_candidates" in result["case_tags"]
    assert "missing_pair_candidates" not in result["case_tags"]
    assert result["warnings"]


def test_evaluate_entity_tags_fallback(tmp_path) -> None:
    artifacts = RunEntityArtifacts(
        entity_dir=tmp_path / "Q3",
        entity_key="Q3",
        knowledge={"entity_profile": {"entity_type": "place"}},
        reasoning={},
        ranking={
            "text_ranking": {"ranked_candidates": []},
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image-fallback",
                        "positive_prompt": "global profile image",
                        "metadata": {"fallback": True},
                    }
                ]
            },
            "pair_ranking": {"ranked_pairs": []},
        },
        final={"entity": {"entity_key": "Q3", "label": "Fallback Entity"}},
    )

    result = evaluate_entity(artifacts)

    assert "fallback" in result["case_tags"]
    assert result["fallback_summary"]["has_fallback"] is True
    assert result["fallback_summary"]["fallback_candidate_count"] == 1
