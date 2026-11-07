from mmkgc.pipeline import _needs_stage4_regeneration
from mmkgc.stages.generation import build_stage3_feedback


def _ranking_with_items(items):
    return {
        "ranking_status": "no_valid_candidate",
        "text_ranking": {
            "rejected_candidates": items,
        },
        "image_ranking": {
            "rejected_candidates": [],
        },
        "pair_ranking": {
            "ranked_pairs": [],
        },
    }


def test_stage3_feedback_keeps_generation_problems_and_builds_repair_actions():
    ranking = _ranking_with_items(
        [
            {
                "candidate_id": "text_001",
                "ranking_reason": "claim is not supported",
                "hard_gate": {
                    "hard_violations": ["unsupported_claim"],
                    "soft_violations": ["low_relation_coverage"],
                    "evidence": ["missing graph evidence"],
                },
            },
            {
                "candidate_id": "text_002",
                "hard_gate": {
                    "hard_violations": ["judge_service_error"],
                    "soft_violations": [],
                },
            },
        ]
    )

    feedback = build_stage3_feedback(ranking, attempt=1)

    assert feedback["target_modality"] == "text"
    assert feedback["hard_violations"] == ["unsupported_claim"]
    assert feedback["soft_violations"] == []
    assert "judge_service_error" not in feedback["hard_violations"]
    assert feedback["candidate_feedback"][0]["candidate_id"] == "text_001"
    assert "address the reported claim conflict using supplied source evidence" in feedback["repair_actions"]
    assert not any('coverage' in action for action in feedback['repair_actions'])
    assert feedback["preserve"] == [
        "entity identity",
        "supported core attributes",
        "selected relation roles",
    ]


def test_stage4_system_failures_do_not_trigger_stage3_regeneration():
    ranking = _ranking_with_items(
        [
            {
                "candidate_id": "text_001",
                "hard_gate": {
                    "hard_violations": ["judge_service_error", "schema_invalid"],
                    "soft_violations": [],
                },
            }
        ]
    )

    feedback = build_stage3_feedback(ranking, attempt=1)

    assert feedback["candidate_feedback"] == []
    assert feedback["hard_violations"] == []
    assert feedback["soft_violations"] == []
    assert _needs_stage4_regeneration(ranking, feedback) is False


def test_coverage_only_legacy_feedback_does_not_force_regeneration():
    ranking = _ranking_with_items([{'candidate_id': 'text_001', 'hard_gate': {
        'hard_violations': ['missing_required_content'],
        'soft_violations': ['low_relation_coverage']}}])
    feedback = build_stage3_feedback(ranking, attempt=1)
    assert feedback['candidate_feedback'] == []
    assert not _needs_stage4_regeneration(ranking, feedback)
