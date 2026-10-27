from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmkgc.ranking.contracts import GateResult
from mmkgc.ranking.hard_gate import (
    deterministic_candidate_gate,
    gate_from_judge,
)


def test_deterministic_text_gate_rejects_empty_candidate() -> None:
    result = deterministic_candidate_gate(
        {
            "candidate_id": "text_001",
            "target_modality": "text",
            "generated_content": {"type": "text", "value": ""},
        },
        modality="text",
        image_input=None,
    )

    assert result.decision == "reject"
    assert "empty_generated_content" in result.hard_violations


def test_deterministic_gate_rejects_wrong_modality_and_missing_id() -> None:
    result = deterministic_candidate_gate(
        {
            "target_modality": "image",
            "generated_content": {"type": "text", "value": "caption"},
        },
        modality="text",
    )

    assert result.decision == "reject"
    assert result.hard_violations == (
        "candidate_id_missing",
        "target_modality_mismatch",
    )


def test_deterministic_image_gate_requires_generated_image_source() -> None:
    result = deterministic_candidate_gate(
        {
            "candidate_id": "image_001",
            "target_modality": "image",
            "generated_content": {"type": "image"},
        },
        modality="image",
    )

    assert result.decision == "reject"
    assert "missing_image_artifact" in result.hard_violations


def test_deterministic_image_gate_rejects_prompt_only_value_as_image_source() -> None:
    result = deterministic_candidate_gate(
        {
            "candidate_id": "image_001",
            "target_modality": "image",
            "generated_content": {"type": "image", "value": "prompt-only description"},
        },
        modality="image",
    )

    assert result.decision == "reject"
    assert "missing_image_artifact" in result.hard_violations


def test_deterministic_candidate_gate_rejects_both_mode_contract() -> None:
    with pytest.raises(ValueError, match="individual text/image candidates"):
        deterministic_candidate_gate(
            {
                "candidate_id": "pair_001",
                "target_modality": "both",
                "generated_content": {"type": "both", "value": "pair"},
            },
            modality="both",
        )


def test_deterministic_gate_rejects_malformed_generated_content() -> None:
    result = deterministic_candidate_gate(
        {"candidate_id": "text_001", "generated_content": "not-a-dict"},
        modality="text",
    )

    assert result == GateResult(
        "reject",
        "deterministic",
        hard_violations=("schema_invalid",),
    )


def test_gate_from_judge_requires_allowed_codes_confidence_and_evidence() -> None:
    judge = SimpleNamespace(
        reject=True,
        hard_violation_codes=["entity_type_conflict"],
        soft_violation_codes=["weak_evidence_support"],
        confidence=0.82,
        evidence=["the subject type conflicts"],
        reason="conflict",
    )

    assert gate_from_judge(judge, ("entity_type_conflict",)) == GateResult(
        "reject",
        "gpt4omini_judge",
        hard_violations=("entity_type_conflict",),
        soft_violations=("weak_evidence_support",),
        judge_confidence=0.82,
        evidence=("the subject type conflicts",),
        reason="conflict",
    )


def test_gate_from_judge_accepts_task4_style_judge_payload() -> None:
    judge = SimpleNamespace(
        hard_reject=True,
        hard_rejection_codes=["entity_type_conflict"],
        soft_violations=["weak_evidence_support"],
        confidence=0.93,
        evidence=["conflicting entity type"],
        reason="conflict",
    )

    result = gate_from_judge(judge, ("entity_type_conflict",))

    assert result.decision == "reject"
    assert result.hard_violations == ("entity_type_conflict",)
    assert result.soft_violations == ("weak_evidence_support",)
    assert result.judge_confidence == 0.93


def test_gate_from_judge_accepts_task4_gate_result_directly() -> None:
    judge = GateResult(
        "reject",
        "gpt4omini-test",
        hard_violations=("entity_type_conflict",),
        soft_violations=("weak_evidence_support",),
        judge_confidence=0.91,
        evidence=("conflicting entity type",),
        reason="conflict",
    )

    assert gate_from_judge(judge, ("entity_type_conflict",)) == judge


def test_gate_from_judge_softens_hard_codes_when_reject_conditions_are_incomplete() -> None:
    judge = SimpleNamespace(
        reject=True,
        hard_violation_codes=["entity_type_conflict"],
        soft_violation_codes=[],
        confidence=0.79,
        evidence=["weak signal"],
        reason="below threshold",
    )

    result = gate_from_judge(judge, ("entity_type_conflict",))

    assert result.decision == "pass"
    assert result.hard_violations == ()
    assert result.soft_violations == ("entity_type_conflict",)


def test_gate_from_judge_rejects_unknown_hard_codes() -> None:
    judge = SimpleNamespace(
        reject=True,
        hard_violation_codes=["mystery_code"],
        soft_violation_codes=[],
        confidence=0.95,
        evidence=["evidence"],
        reason="bad code",
    )

    with pytest.raises(ValueError, match="unknown hard violation code"):
        gate_from_judge(judge, ("entity_type_conflict",))
