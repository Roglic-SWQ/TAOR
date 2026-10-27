from __future__ import annotations

from mmkgc.evaluation.calibration import (
    calibrate_entity_scores,
    relation_support_strength,
)


def test_relation_support_strength_classifies_known_relations() -> None:
    assert relation_support_strength({"primary_relation": "genre"}) == "direct_identity"
    assert relation_support_strength({"primary_relation": "performer"}) == "direct_identity"
    assert relation_support_strength({"primary_relation": "occupation"}) == "direct_identity"
    assert relation_support_strength({"primary_relation": "instance of"}) == "type_support"
    assert relation_support_strength({"primary_relation": "cast member"}) == "neighbor_indirect"
    assert relation_support_strength({"primary_relation": "original broadcaster"}) == "neighbor_indirect"
    assert relation_support_strength({"primary_relation": "member of sports team"}) == "neighbor_indirect"
    assert relation_support_strength({"primary_relation": "global_profile"}) == "fallback"
    assert relation_support_strength({"metadata": {"fallback": True}}) == "fallback"
    assert relation_support_strength({"is_fallback": True}) == "fallback"
    assert relation_support_strength({"primary_relation": "unknown relation"}) == "unknown"


def test_calibrate_entity_scores_penalizes_fallback_and_generic_entity_type() -> None:
    result = calibrate_entity_scores(
        entity={"entity_key": "Q1", "label": "Fallback", "entity_type": "entity"},
        text_evaluation={
            "best_offline_evaluation_score": 0.36,
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "offline_evaluation_score": 0.36,
                    "is_fallback": True,
                    "generated_content": {"value": "Fallback is described as a entity."},
                    "primary_relation": "global_profile",
                }
            ],
        },
        image_evaluation={
            "best_offline_evaluation_score": 0.96,
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "offline_evaluation_score": 0.96,
                    "is_fallback": True,
                    "positive_prompt": "Fallback, entity, identity, profile context",
                    "primary_relation": "global_profile",
                }
            ],
        },
        pair_evaluation={
            "best_offline_evaluation_score": 0.90,
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "offline_evaluation_score": 0.90,
                    "text_candidate": {"metadata": {"fallback": True}},
                    "image_candidate": {"metadata": {"fallback": True}},
                }
            ],
        },
        fallback_summary={"has_fallback": True},
    )

    assert result["calibrated_text_best_offline_evaluation_score"] < 0.36
    assert result["calibrated_image_best_offline_evaluation_score"] < 0.96
    assert result["calibrated_pair_best_offline_evaluation_score"] <= 0.65
    assert result["calibrated_overall_offline_evaluation_score"] < 0.70
    assert result["calibration_summary"]["fallback_penalty"] > 0
    assert result["calibration_summary"]["entity_type_generic_penalty"] > 0
    assert "fallback_penalized" in result["calibration_summary"]["calibration_tags"]
    assert "generic_entity_type" in result["calibration_summary"]["calibration_tags"]


def test_calibrate_entity_scores_caps_pair_when_text_is_weak() -> None:
    result = calibrate_entity_scores(
        entity={"entity_key": "Q2", "label": "Weak Text", "entity_type": "person"},
        text_evaluation={
            "best_offline_evaluation_score": 0.34,
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "offline_evaluation_score": 0.34,
                    "generated_content": {"value": "Weak Text appears in one film."},
                    "primary_relation": "cast member",
                }
            ],
        },
        image_evaluation={
            "best_offline_evaluation_score": 0.72,
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "offline_evaluation_score": 0.72,
                    "positive_prompt": "Weak Text, person, cast member context",
                    "primary_relation": "cast member",
                }
            ],
        },
        pair_evaluation={
            "best_offline_evaluation_score": 0.89,
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "offline_evaluation_score": 0.89,
                    "text_candidate": {"primary_relation": "cast member"},
                    "image_candidate": {"primary_relation": "cast member"},
                }
            ],
        },
        fallback_summary={"has_fallback": False},
    )

    assert result["calibrated_pair_best_offline_evaluation_score"] <= 0.65
    assert result["calibration_summary"]["single_side_pair_penalty"] > 0
    assert "pair_single_side_penalized" in result["calibration_summary"]["calibration_tags"]


def test_calibrate_entity_scores_applies_template_prompt_penalty_once() -> None:
    result = calibrate_entity_scores(
        entity={"entity_key": "Q3", "label": "Template", "entity_type": "entity"},
        text_evaluation={
            "best_offline_evaluation_score": 1.0,
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "offline_evaluation_score": 1.0,
                    "generated_content": {"value": "Template belongs to a genre."},
                    "primary_relation": "genre",
                }
            ],
        },
        image_evaluation={
            "best_offline_evaluation_score": 1.0,
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "offline_evaluation_score": 1.0,
                    "positive_prompt": "Template, entity, identity, profile context, grounded details",
                    "primary_relation": "global_profile",
                }
            ],
        },
        pair_evaluation={
            "best_offline_evaluation_score": 0.5,
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "offline_evaluation_score": 0.5,
                    "text_candidate": {"primary_relation": "genre"},
                    "image_candidate": {"primary_relation": "global_profile"},
                }
            ],
        },
        fallback_summary={"has_fallback": False},
    )

    assert result["calibrated_image_best_offline_evaluation_score"] == 0.57375
    assert result["calibration_summary"]["template_prompt_penalty"] == 0.10125
    assert "template_prompt_penalized" in result["calibration_summary"]["calibration_tags"]


def test_calibrate_entity_scores_averages_only_available_modalities() -> None:
    result = calibrate_entity_scores(
        entity={"entity_key": "Q4", "label": "Partial", "entity_type": "person"},
        text_evaluation={
            "candidate_count": 1,
            "best_offline_evaluation_score": 0.6,
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "offline_evaluation_score": 0.6,
                    "generated_content": {"value": "Partial is a performer."},
                    "primary_relation": "performer",
                }
            ],
        },
        image_evaluation={
            "candidate_count": 1,
            "best_offline_evaluation_score": 0.8,
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "offline_evaluation_score": 0.8,
                    "positive_prompt": "Partial person performer portrait",
                    "primary_relation": "performer",
                }
            ],
        },
        pair_evaluation={
            "candidate_count": 0,
            "best_offline_evaluation_score": 0.0,
            "ranked_pairs": [],
        },
        fallback_summary={"has_fallback": False},
    )

    assert result["calibrated_pair_best_offline_evaluation_score"] == 0.0
    assert result["calibrated_overall_offline_evaluation_score"] == 0.7
