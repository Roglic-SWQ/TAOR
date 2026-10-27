from __future__ import annotations

import inspect
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import get_args

import pytest

from mmkgc.ranking.context import (
    RankingContext,
    build_kg_reference,
    build_ranking_context,
    build_visual_constraint_text,
    candidate_generated_text,
)
from mmkgc.ranking.contracts import (
    FrozenDict,
    GateResult,
    RankingError,
    RankingStatus,
    RequiredScorerError,
    ScoreResult,
    freeze_value,
)
from mmkgc.data import EntityRecord
from mmkgc.adapters import RuleImagePromptGenerator, RuleTextGenerator
from mmkgc.stages import (
    coarse_relation_filtering,
    fine_grained_relation_selection,
    constraint_compilation,
    knowledge_modeling,
    relation_alternation_generation,
)


GROUND_TRUTH_SENTINEL = "ZZZ_SECRET_GROUND_TRUTH"
INSTRUCTION_SENTINEL = "ZZZ_INSTRUCTION_ONLY"


class _CustomStringable:
    def __str__(self) -> str:
        return "custom-string-value"


class _UncopyableOpaque:
    def __deepcopy__(self, memo):
        raise RuntimeError("copy disabled")


class _SelfCopyingOpaque:
    def __deepcopy__(self, memo):
        return self


class _AliasedDeepcopyOpaque:
    def __init__(self) -> None:
        self.buffer = bytearray(b"shared")

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        copied.buffer = self.buffer
        return copied


class _StringSubclass(str):
    pass


class _ComparisonOverridingString(str):
    def __eq__(self, other):
        return True

    __hash__ = str.__hash__


class _ForgedCategoricalString(str):
    def __new__(cls, value: str, matching_value: str):
        result = super().__new__(cls, value)
        result.matching_value = matching_value
        return result

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(self.matching_value)


class _HostileString(str):
    """A scalar subclass that must never control normalization methods."""

    def strip(self, *_args, **_kwargs):
        raise AssertionError("subclass strip must not run")

    def casefold(self):
        raise AssertionError("subclass casefold must not run")


class _HostileCategoricalString(_HostileString):
    def __new__(cls, value: str, matching_value: str):
        result = super().__new__(cls, value)
        result.matching_value = matching_value
        return result

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(self.matching_value)


class _BytesSubclass(bytes):
    pass


class _PathSubclass(type(Path("."))):
    pass


class _ComparisonOverridingInt(int):
    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True


class _ComparisonOverridingFloat(float):
    def __lt__(self, other):
        return False

    def __le__(self, other):
        return True


def _context_inputs() -> dict[str, object]:
    text_candidate = {
        "candidate_id": "text_001",
        "target_modality": "text",
        "primary_relation": "directed by",
        "supporting_relations": ["country"],
        "relation_role": "primary",
        "generation_angle": "identity",
        "used_evidence_triples": ["Q1 P57 Q2"],
        "prompt_or_instruction": f"Use hidden instruction {INSTRUCTION_SENTINEL}",
        "positive_prompt": f"Positive {INSTRUCTION_SENTINEL}",
        "expected_relation_coverage": ["directed by"],
        "expected_relation_coverage_detail": {
            "primary": "directed by",
            "supporting": ["country"],
        },
        "rationale": f"rationale {INSTRUCTION_SENTINEL}",
        "system_prompt": INSTRUCTION_SENTINEL,
        "generated_content": {
            "type": "text",
            "value": "Example Film is a film.",
            "system_instruction": INSTRUCTION_SENTINEL,
        },
        "metadata": {
            "relation_role": "primary",
            "status": "generated",
            "instruction": INSTRUCTION_SENTINEL,
            "error_message": INSTRUCTION_SENTINEL,
        },
    }
    image_candidate = {
        "candidate_id": "image_001",
        "target_modality": "image",
        "primary_relation": "depicts",
        "supporting_relations": ["country"],
        "generation_angle": "relation emphasis",
        "prompt_or_instruction": f"Draw {INSTRUCTION_SENTINEL}",
        "positive_prompt": f"Positive {INSTRUCTION_SENTINEL}",
        "generated_content": {
            "type": "image",
            "value": f"Positive image prompt {INSTRUCTION_SENTINEL}",
            "image_artifact_id": "artifact-image-001",
            "image_path": "/private/generated.png",
            "system_instruction": INSTRUCTION_SENTINEL,
        },
        "metadata": {
            "relation_role": "primary",
            "status": "generated",
            "instruction": INSTRUCTION_SENTINEL,
        },
    }
    secondary_candidate = {
        "candidate_id": "text_002",
        "target_modality": "text",
        "primary_relation": "country",
        "generated_content": {"type": "text", "value": "A country relation."},
        "metadata": {"relation_role": "secondary", "status": "generated"},
    }
    return {
        "entity_key": "Q1",
        "entity_label": "Example Film",
        "missing_mode": "both",
        "existing_modalities": {
            "text": "Existing public text.",
            "image": "/private/existing.png",
        },
        "knowledge": {
            "entity_profile": {
                "core_identity": "Example Film",
                "entity_type": "film",
                "secondary_types": ["creative work"],
                "type_confidence": "high",
                "semantic_domain": ["cinema", "arts"],
                "key_attributes": [
                    {
                        "attribute": "festival",
                        "attribute_type": "contextual",
                        "source": "KG",
                        "evidence": "(Example Film, genre, Drama)",
                    }
                ],
                "conflicting_evidence": [
                    {
                        "conflict_type": "type_description_mismatch",
                        "preferred": "kg_instance_of",
                    }
                ],
                "uncertain_aspects": [{"aspect": "visual_style", "reason": "No image evidence."}],
                "visual_style": "documentary",
                "description": GROUND_TRUTH_SENTINEL,
                "groundTruth": GROUND_TRUTH_SENTINEL,
                "system_prompt": INSTRUCTION_SENTINEL,
                "source_file": Path("/private/profile.json"),
            },
            "relation_facts": [
                {
                    "relation": "directed by",
                    "fact": "Primary supported fact",
                    "support_level": "moderate",
                    "modality_relevance": ["text", "image"],
                    "instruction": INSTRUCTION_SENTINEL,
                    "system_message": INSTRUCTION_SENTINEL,
                    "location": Path("/private/fact.json"),
                },
                {
                    "relation": "country",
                    "fact": "Secondary strong fact",
                    "support_level": "strong",
                    "modality_relevance": ["text", "image"],
                },
                {
                    "relation": "depicts",
                    "fact": "primary relation visual clue: lighthouse",
                    "support_level": "strong",
                    "modality_relevance": ["image"],
                },
            ],
            "ground_truth_modalities": {"text": GROUND_TRUTH_SENTINEL},
        },
        "reasoning": {
            "supported_facts": [
                {
                    "fact": "Primary supported fact",
                    "support_level": "moderate",
                    "source_relations": ["directed by"],
                    "reference_answer": GROUND_TRUTH_SENTINEL,
                    "system_prompt": INSTRUCTION_SENTINEL,
                },
                {
                    "fact": "Secondary strong fact",
                    "confidence": "high",
                    "source_relations": ["country"],
                    "support_level": "strong",
                },
                {
                    "fact": "primary relation visual clue: lighthouse",
                    "source_relations": ["depicts"],
                    "support_level": "strong",
                },
                {
                    "fact": "Secondary weak fact",
                    "relation_role": "secondary",
                    "confidence": "medium",
                },
                {
                    "fact": "Uncertain primary fact",
                    "relation_role": "primary",
                    "support_level": "uncertain",
                },
                {
                    "fact": "Context-only fact",
                    "relation_role": "context_only",
                    "support_level": "strong",
                },
                {
                    "fact": "Excluded fact",
                    "relation_role": "excluded",
                    "support_level": "strong",
                },
            ],
            "safe_inferences": [
                "A conservative safe inference.",
                {"instruction": INSTRUCTION_SENTINEL},
            ],
            "uncertain_facts": [GROUND_TRUTH_SENTINEL],
            "modality_constraints": {
                "text": {
                    "must_include": [
                        {"constraint": "entity type: film", "confidence": "high"},
                        {
                            "constraint": "Text must include",
                            "confidence": "high",
                            "system_instruction": INSTRUCTION_SENTINEL,
                            "source_file": Path("/private/constraint.json"),
                        },
                    ],
                    "should_include": [
                        {
                            "constraint": "Text should include",
                            "safe": True,
                            "confidence": "medium",
                        },
                        {
                            "constraint": "Unsafe text should",
                            "safe": False,
                            "confidence": "high",
                        },
                    ],
                    "context_only": [{"constraint": "CONTEXT_ONLY_MARKER"}],
                    "avoid": [{"constraint": "AVOID_MARKER"}],
                },
                "image": {
                    "must_include": [
                        {"constraint": "Image must include", "confidence": "high"}
                    ],
                    "should_include": [
                        {
                            "constraint": "Image should include",
                            "safe": True,
                            "confidence": "medium",
                        },
                        {
                            "constraint": "Unsafe image should",
                            "safe": False,
                            "confidence": "high",
                        },
                    ],
                    "context_only": [{"constraint": "IMAGE_CONTEXT_ONLY_MARKER"}],
                    "avoid": [{"constraint": "IMAGE_AVOID_MARKER"}],
                },
            },
            "forbidden_content": [
                "FORBIDDEN_POSITIVE_MARKER",
                {"instruction": INSTRUCTION_SENTINEL},
            ],
            "ground_truth_modalities": {"image": GROUND_TRUTH_SENTINEL},
        },
        "generation": {
            "relation_alternation_generation": {
                "text_candidates": [text_candidate, secondary_candidate],
                "image_candidates": [image_candidate],
            },
            "model_calls": [{"response": GROUND_TRUTH_SENTINEL}],
        },
        "generated_image_inputs": {
            "image_001": {"bytes": b"raw-image", "path": "/private/generated.png"}
        },
        "existing_image_input": {"path": "/private/existing.png"},
    }


def _build_context() -> RankingContext:
    return build_ranking_context(**_context_inputs())


def test_contracts_define_stable_statuses_and_frozen_results():
    assert get_args(RankingStatus) == (
        "completed",
        "degraded",
        "no_valid_candidate",
        "failed",
    )
    assert issubclass(RequiredScorerError, RankingError)

    score = ScoreResult(
        raw_score=0.7,
        normalized_score=0.8,
        status="scored",
        model_name="example",
    )
    assert score.model_version == ""
    assert score.latency_ms == 0
    assert score.details == {}
    with pytest.raises(AttributeError):
        score.status = "failed"  # type: ignore[misc]

    assert GateResult("pass", "deterministic").hard_violations == ()


def test_score_result_defensively_freezes_nested_details():
    source = {"metrics": {"labels": ["safe"]}}
    score = ScoreResult(0.5, 0.5, "scored", "example", details=source)

    source["metrics"]["labels"].append("source mutation")
    assert score.details["metrics"]["labels"] == ("safe",)
    with pytest.raises(TypeError):
        score.details["new"] = "mutation"  # type: ignore[index]
    with pytest.raises(TypeError):
        score.details["metrics"]["new"] = "nested mutation"  # type: ignore[index]
    assert json.loads(json.dumps(asdict(score)))["details"] == {
        "metrics": {"labels": ["safe"]}
    }


def test_score_result_rejects_non_json_details_with_explicit_paths():
    pre_frozen = freeze_value({"nested": {"values": [1, "safe"]}})
    assert ScoreResult(0.5, 0.5, "scored", "example", details={"valid": pre_frozen}).details[
        "valid"
    ] is not pre_frozen
    for key, bad in (
        (7, "non-string key"),
        ("path", Path("/private/detail.json")),
        ("bytes", b"not-json"),
        ("object", _CustomStringable()),
        ("set", {"unordered", "values"}),
        ("nan", float("nan")),
        ("infinity", float("inf")),
    ):
        with pytest.raises((TypeError, ValueError), match=r"\$"):
            ScoreResult(0.5, 0.5, "scored", "example", details={key: bad})


@pytest.mark.parametrize("field", ["raw_score", "normalized_score"])
@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), Path("/tmp/score"), b"score", _CustomStringable()],
    ids=["nan", "positive-inf", "negative-inf", "path", "bytes", "object"],
)
def test_score_result_rejects_nonfinite_or_opaque_scores(field, bad):
    values = {
        "raw_score": 0.4,
        "normalized_score": 0.5,
        "status": "scored",
        "model_name": "example",
    }
    values[field] = bad

    with pytest.raises((TypeError, ValueError), match=field):
        ScoreResult(**values)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_score_result_rejects_out_of_range_normalized_score(bad):
    with pytest.raises(ValueError, match="normalized_score"):
        ScoreResult(0.4, bad, "scored", "example")


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1,
        1.5,
        True,
        Path("/tmp/latency"),
        b"latency",
        _CustomStringable(),
    ],
    ids=[
        "nan",
        "positive-inf",
        "negative-inf",
        "negative",
        "fractional",
        "bool",
        "path",
        "bytes",
        "object",
    ],
)
def test_score_result_rejects_invalid_latency_ms(bad):
    with pytest.raises((TypeError, ValueError), match="latency_ms"):
        ScoreResult(0.4, 0.5, "scored", "example", latency_ms=bad)


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        1.01,
        Path("/tmp/confidence"),
        b"confidence",
        _CustomStringable(),
    ],
    ids=[
        "nan",
        "positive-inf",
        "negative-inf",
        "below-zero",
        "above-one",
        "path",
        "bytes",
        "object",
    ],
)
def test_gate_result_rejects_invalid_judge_confidence(bad):
    with pytest.raises((TypeError, ValueError), match="judge_confidence"):
        GateResult("pass", "judge", judge_confidence=bad)


@pytest.mark.parametrize(
    ("contract", "field"),
    [
        (lambda bad: GateResult("pass", "judge", hard_violations=(bad,)), "hard_violations"),
        (lambda bad: GateResult("pass", "judge", soft_violations=(bad,)), "soft_violations"),
        (lambda bad: GateResult("pass", "judge", evidence=(bad,)), "evidence"),
    ],
)
@pytest.mark.parametrize(
    "bad",
    [Path("/tmp/evidence"), b"evidence", _CustomStringable(), float("nan"), float("inf")],
    ids=["path", "bytes", "object", "nan", "infinity"],
)
def test_contract_string_sequences_reject_non_string_values(contract, field, bad):
    with pytest.raises(TypeError, match=field):
        contract(bad)


def test_contract_payloads_are_strict_json_serializable():
    contracts = (
        ScoreResult(0.4, None, "scored", "example", latency_ms=3, details={"ok": True}),
        GateResult(
            "reject",
            "judge",
            hard_violations=("unsafe",),
            soft_violations=("weak",),
            judge_confidence=0.8,
            evidence=("fact",),
            reason="unsafe output",
        ),
    )

    for contract in contracts:
        json.dumps(asdict(contract), allow_nan=False)


def test_freeze_value_rejects_unordered_sets_and_rebuilds_frozen_mappings():
    frozen = freeze_value({"nested": {"value": 1}})
    rebuilt = freeze_value(frozen)

    assert isinstance(frozen, FrozenDict)
    assert rebuilt == frozen
    assert rebuilt is not frozen
    assert rebuilt["nested"] is not frozen["nested"]
    with pytest.raises(TypeError, match="set"):
        freeze_value({"unordered": {"b", "a"}})


def test_freeze_value_normalizes_exact_scalar_subclasses_without_aliasing():
    source = _StringSubclass("safe")
    frozen = freeze_value({"value": source})

    assert frozen["value"] == "safe"
    assert type(frozen["value"]) is str
    assert frozen["value"] is not source


def test_context_factory_has_an_explicit_leak_resistant_signature():
    parameters = inspect.signature(build_ranking_context).parameters
    assert "record" not in parameters
    assert "entity_record" not in parameters
    assert "ground_truth_modalities" not in parameters
    assert "description" not in parameters
    assert not any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())

    inputs = _context_inputs()
    with pytest.raises(TypeError):
        build_ranking_context(**inputs, ground_truth_modalities={"text": GROUND_TRUTH_SENTINEL})
    with pytest.raises(TypeError):
        build_ranking_context(**inputs, description=GROUND_TRUTH_SENTINEL)


@pytest.mark.parametrize("field", ["entity_key", "entity_label", "missing_mode"])
@pytest.mark.parametrize(
    "bad_value",
    [Path("/private/key"), b"bytes-key", _CustomStringable()],
    ids=["path", "bytes", "custom-str"],
)
def test_context_key_fields_reject_non_string_values_in_factory_and_direct_init(
    field, bad_value
):
    inputs = _context_inputs()
    inputs[field] = bad_value
    with pytest.raises(TypeError):
        build_ranking_context(**inputs)

    context = _build_context()
    with pytest.raises(TypeError):
        replace(context, **{field: bad_value})


def test_missing_mode_rejects_unknown_string_in_factory_and_direct_init():
    inputs = _context_inputs()
    inputs["missing_mode"] = "future_mode"
    with pytest.raises(ValueError):
        build_ranking_context(**inputs)

    with pytest.raises(ValueError):
        replace(_build_context(), missing_mode="future_mode")


def test_context_factory_extracts_only_allowlisted_stage_data():
    inputs = _context_inputs()
    context = build_ranking_context(**inputs)

    assert context.text_candidates[0]["candidate_id"] == "text_001"
    assert context.image_candidates[0]["candidate_id"] == "image_001"
    assert context.text_candidates[0]["expected_relation_coverage"] == ("directed by",)
    assert list(context.text_candidates[0]) == [
        "candidate_id",
        "target_modality",
        "primary_relation",
        "supporting_relations",
        "relation_role",
        "generation_angle",
        "used_evidence_triples",
        "expected_relation_coverage",
        "expected_relation_coverage_detail",
        "generated_content",
        "metadata",
    ]
    assert context.image_candidates[0]["generated_content"] == {
        "type": "image",
        "image_artifact_id": "artifact-image-001",
    }
    assert context.generated_image_inputs == inputs["generated_image_inputs"]
    assert context.existing_image_input == inputs["existing_image_input"]
    assert context.ground_truth_used is False
    assert context.context_schema_version == "ranking_context_v1"
    assert GROUND_TRUTH_SENTINEL not in repr(context)
    assert INSTRUCTION_SENTINEL not in repr(context)
    assert "PosixPath" not in repr(context.entity_profile)
    assert "PosixPath" not in repr(context.relation_facts)
    assert INSTRUCTION_SENTINEL not in repr(asdict(context))
    assert context.entity_profile["secondary_types"] == ("creative work",)
    assert context.entity_profile["type_confidence"] == "high"
    assert context.entity_profile["semantic_domain"] == ("cinema", "arts")
    assert context.entity_profile["key_attributes"][0]["attribute"] == "festival"
    assert context.entity_profile["conflicting_evidence"][0]["preferred"] == "kg_instance_of"
    assert context.entity_profile["uncertain_aspects"][0]["aspect"] == "visual_style"

    field_by_name = {item.name: item for item in fields(RankingContext)}
    assert field_by_name["ground_truth_used"].init is False
    assert field_by_name["context_schema_version"].init is False
    with pytest.raises(TypeError):
        RankingContext(
            entity_key="Q1",
            entity_label="Example Film",
            missing_mode="text",
            existing_modalities={},
            entity_profile={},
            relation_facts=(),
            supported_facts=(),
            safe_inferences=(),
            modality_constraints={},
            forbidden_content=(),
            text_candidates=(),
            image_candidates=(),
            generated_image_inputs={},
            ground_truth_used=True,
        )


def test_context_is_defensively_copied_and_nested_data_is_immutable():
    inputs = _context_inputs()
    context = build_ranking_context(**inputs)

    inputs["knowledge"]["entity_profile"]["core_identity"] = "source mutation"
    inputs["reasoning"]["supported_facts"][0]["fact"] = "source mutation"
    inputs["generation"]["relation_alternation_generation"]["text_candidates"][0][
        "generated_content"
    ]["value"] = "source mutation"
    inputs["generated_image_inputs"]["image_001"]["path"] = "/source/mutation.png"
    inputs["existing_image_input"]["path"] = "/source/existing-mutation.png"

    assert context.entity_profile["core_identity"] == "Example Film"
    assert context.supported_facts[0]["fact"] == "Primary supported fact"


def test_transport_mappings_rebuild_pre_frozen_sources_without_aliasing():
    inputs = _context_inputs()
    generated_inputs = inputs["generated_image_inputs"]
    existing_input = inputs["existing_image_input"]

    context = build_ranking_context(**inputs)

    assert context.generated_image_inputs == generated_inputs
    assert context.generated_image_inputs is not generated_inputs
    assert context.generated_image_inputs["image_001"] is not generated_inputs["image_001"]
    assert context.existing_image_input == existing_input
    assert context.existing_image_input is not existing_input
    assert candidate_generated_text(context.text_candidates[0]) == "Example Film is a film."
    assert context.generated_image_inputs["image_001"]["path"] == "/private/generated.png"
    assert context.existing_image_input["path"] == "/private/existing.png"
    with pytest.raises(TypeError):
        context.entity_profile["core_identity"] = "nested mutation"  # type: ignore[index]
    with pytest.raises(TypeError):
        context.generated_image_inputs["image_001"]["path"] = "nested mutation"

    payload = context.to_prompt_payload()
    payload["entity_profile"]["core_identity"] = "payload mutation"
    payload["supported_facts"][0]["fact"] = "payload mutation"
    assert context.entity_profile["core_identity"] == "Example Film"
    assert context.supported_facts[0]["fact"] == "Primary supported fact"


@pytest.mark.parametrize("opaque", [_UncopyableOpaque(), _SelfCopyingOpaque(), _AliasedDeepcopyOpaque()])
def test_transport_rejects_opaque_values_that_cannot_prove_isolation(opaque):
    inputs = _context_inputs()
    inputs["generated_image_inputs"] = {"image_001": {"opaque": opaque}}

    with pytest.raises(TypeError, match="opaque transport"):
        build_ranking_context(**inputs)


@pytest.mark.parametrize(
    "scalar",
    [
        _StringSubclass("safe"),
        _BytesSubclass(b"safe"),
        _ComparisonOverridingInt(7),
        _ComparisonOverridingFloat(0.7),
        _PathSubclass("/tmp/safe"),
    ],
)
def test_transport_rejects_scalar_subclasses(scalar):
    inputs = _context_inputs()
    inputs["generated_image_inputs"] = {"image_001": {"label": scalar}}

    with pytest.raises(TypeError, match="opaque transport"):
        build_ranking_context(**inputs)


def test_prompt_payload_serializes_existing_text_but_not_image_inputs_or_prompts():
    payload = _build_context().to_prompt_payload()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["existing_modalities"] == {"text": "Existing public text."}
    assert payload["has_existing_image"] is True
    assert payload["ground_truth_used"] is False
    assert payload["context_schema_version"] == "ranking_context_v1"
    assert payload["metadata"] == {
        "ground_truth_used": False,
        "context_schema_version": "ranking_context_v1",
    }
    for forbidden in (
        GROUND_TRUTH_SENTINEL,
        INSTRUCTION_SENTINEL,
        "prompt_or_instruction",
        "/private/existing.png",
        "/private/generated.png",
        "raw-image",
        "reference_answer",
        "system_message",
        "system_prompt",
        "system_instruction",
        "source_file",
    ):
        assert forbidden not in serialized


def test_image_candidate_cannot_smuggle_prompt_value_by_claiming_text_content_type():
    inputs = _context_inputs()
    image_candidate = inputs["generation"]["relation_alternation_generation"][
        "image_candidates"
    ][0]
    image_candidate["generated_content"] = {
        "type": "text",
        "value": INSTRUCTION_SENTINEL,
        "image_artifact_id": "artifact-image-001",
    }

    context = build_ranking_context(**inputs)

    assert "value" not in context.image_candidates[0]["generated_content"]
    assert INSTRUCTION_SENTINEL not in repr(context.image_candidates)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (
            {
                "prompt_or_instruction": f"Use {INSTRUCTION_SENTINEL}",
                "positive_prompt": INSTRUCTION_SENTINEL,
                "expected_relation_coverage": [INSTRUCTION_SENTINEL],
                "rationale": INSTRUCTION_SENTINEL,
                "generated_content": {
                    "type": "text",
                    "value": "  Example Film is a film.  ",
                },
            },
            "Example Film is a film.",
        ),
        ({"generated_content": {"value": "   "}}, ""),
        ({"generated_content": {"value": 123}}, ""),
        ({"generated_content": None}, ""),
        ({}, ""),
    ],
)
def test_candidate_generated_text_reads_only_nonempty_string_value(candidate, expected):
    result = candidate_generated_text(candidate)

    assert result == expected
    assert INSTRUCTION_SENTINEL not in result


def test_kg_reference_uses_only_supported_positive_facts_and_safe_constraints():
    reference = build_kg_reference(_build_context(), "text")

    assert reference == " ".join(
        [
            "Example Film",
            "entity type: film",
            "Primary supported fact",
            "Secondary strong fact",
            "Text must include",
            "Text should include",
        ]
    )
    assert reference.count("Primary supported fact") == 1
    for forbidden in (
        "Secondary weak fact",
        "Uncertain primary fact",
        "Context-only fact",
        "Excluded fact",
        "Unsafe text should",
        "CONTEXT_ONLY_MARKER",
        "AVOID_MARKER",
        "FORBIDDEN_POSITIVE_MARKER",
        GROUND_TRUTH_SENTINEL,
        INSTRUCTION_SENTINEL,
    ):
        assert forbidden not in reference


def test_kg_reference_fails_closed_for_missing_or_unknown_relation_roles():
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"] = [
        {
            "relation": "instance of",
            "fact": "Stage 1 strong fact without role",
            "support_level": "strong",
            "evidence_triples": ["Q1 P31 Q11424"],
        },
        {
            "relation": "tertiary relation",
            "fact": "Tertiary fact",
            "relation_role": "tertiary",
            "support_level": "strong",
        },
    ]
    inputs["reasoning"]["supported_facts"] = [
        {
            "fact": "Stage 2 fact without role",
            "source_relations": ["instance of"],
            "evidence": ["Q1 P31 Q11424"],
            "support_level": "strong",
        },
        {
            "fact": "Fallback fact",
            "relation_role": "fallback",
            "support_level": "strong",
        },
        {
            "fact": "Unknown-role fact",
            "relation_role": "future_role",
            "support_level": "strong",
        },
    ]
    inputs["reasoning"]["modality_constraints"]["text"] = {
        "must_include": [],
        "should_include": [],
        "context_only": [],
        "avoid": [],
    }

    reference = build_kg_reference(build_ranking_context(**inputs), "text")

    assert reference == "Example Film entity type: film"
    assert "Stage 1 strong fact without role" not in reference
    assert "Stage 2 fact without role" not in reference
    assert "Tertiary fact" not in reference
    assert "Fallback fact" not in reference
    assert "Unknown-role fact" not in reference


def test_factory_joins_real_stage_schemas_with_stage4_role_provenance():
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"] = [
        {
            "relation": "directed by",
            "fact": "Text-only director fact",
            "support_level": "strong",
            "modality_relevance": ["text"],
        },
        {
            "relation": "depicts",
            "fact": "Image-only lighthouse fact",
            "support_level": "strong",
            "modality_relevance": ["image"],
        },
        {
            "relation": "country",
            "fact": "Strong secondary country fact",
            "support_level": "strong",
            "modality_relevance": ["text", "image"],
        },
    ]
    inputs["reasoning"]["supported_facts"] = [
        {
            "fact": "Text-only director fact",
            "source_relations": ["directed by"],
            "evidence": ["Q1 P57 Q2"],
            "support_level": "strong",
        },
        {
            "fact": "Image-only lighthouse fact",
            "source_relations": ["depicts"],
            "evidence": ["Q1 P180 Q3"],
            "support_level": "strong",
        },
        {
            "fact": "Strong secondary country fact",
            "source_relations": ["country"],
            "evidence": ["Q1 P495 Q4"],
            "support_level": "strong",
        },
    ]
    stage_four = inputs["generation"]["relation_alternation_generation"]
    stage_four["text_candidates"].append(
        {
            "candidate_id": "text_002",
            "target_modality": "text",
            "primary_relation": "country",
            "generated_content": {"type": "text", "value": "Safe text."},
            "metadata": {"relation_role": "secondary", "status": "generated"},
        }
    )

    context = build_ranking_context(**inputs)
    by_fact = {item["fact"]: item for item in context.supported_facts}
    text_reference = build_kg_reference(context, "text")
    image_reference = build_kg_reference(context, "image")

    assert by_fact["Text-only director fact"]["relation_role"] == "primary"
    assert by_fact["Image-only lighthouse fact"]["modality_relevance"] == ("image",)
    assert by_fact["Strong secondary country fact"]["relation_role"] == "secondary"
    assert "Text-only director fact" in text_reference
    assert "Strong secondary country fact" in text_reference
    assert "Image-only lighthouse fact" not in text_reference
    assert "Image-only lighthouse fact" in image_reference
    assert "Strong secondary country fact" in image_reference
    assert "Text-only director fact" not in image_reference


def test_factory_rejects_incomplete_source_and_modality_provenance():
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"] = [
        {
            "relation": "directed by",
            "fact": "Invalid source declaration fact",
            "support_level": "strong",
            "modality_relevance": ["text"],
        },
        {
            "relation": "country",
            "fact": "Unknown modality fact",
            "support_level": "strong",
            "modality_relevance": ["text", "future"],
        },
    ]
    inputs["reasoning"]["supported_facts"] = [
        {
            "fact": "Invalid source declaration fact",
            "source_relations": ["directed by", 7],
            "support_level": "strong",
        },
        {
            "fact": "Unknown modality fact",
            "source_relations": ["country"],
            "support_level": "strong",
        },
    ]

    reference = build_kg_reference(build_ranking_context(**inputs), "text")

    assert "Invalid source declaration fact" not in reference
    assert "Unknown modality fact" not in reference


def test_direct_context_rejects_top_level_fact_role_spoofing():
    context = replace(
        _build_context(),
        relation_facts=(
            {
                "relation": "spoofed relation",
                "fact": "Spoofed fact",
                "support_level": "strong",
                "modality_relevance": ["text"],
            },
        ),
        supported_facts=(
            {
                "fact": "Spoofed fact",
                "source_relations": ["spoofed relation"],
                "relation_role": "primary",
                "support_level": "strong",
                "modality_relevance": ["text"],
            },
        ),
        text_candidates=(
            {
                "candidate_id": "text_spoof",
                "primary_relation": "spoofed relation",
                "relation_role": "primary",
                "generated_content": {"type": "text", "value": "Spoof."},
                "metadata": {"status": "generated"},
            },
        ),
        image_candidates=(),
    )

    assert "Spoofed fact" not in build_kg_reference(context, "text")


def test_entity_type_and_visual_profile_values_obey_forbidden_content():
    inputs = _context_inputs()
    inputs["reasoning"]["forbidden_content"] = ["film", "cinema", "documentary"]
    context = build_ranking_context(**inputs)

    reference = build_kg_reference(context, "text")
    visual_text = build_visual_constraint_text(context, context.image_candidates[0])

    assert "entity type: film" not in reference
    assert "entity type: film" not in visual_text
    assert "cinema" not in visual_text
    assert "documentary" not in visual_text


@pytest.mark.parametrize("entity_type", ["uncertain", "unknown", "future"])
def test_non_positive_entity_types_are_excluded_from_positive_references(entity_type):
    inputs = _context_inputs()
    inputs["knowledge"]["entity_profile"]["entity_type"] = entity_type
    inputs["reasoning"]["modality_constraints"]["text"]["must_include"][0][
        "constraint"
    ] = f"entity type: {entity_type}"
    inputs["reasoning"]["modality_constraints"]["image"]["must_include"].insert(
        0, {"constraint": f"entity type: {entity_type}", "confidence": "high"}
    )
    context = build_ranking_context(**inputs)

    assert f"entity type: {entity_type}" not in build_kg_reference(context, "text")
    assert f"entity type: {entity_type}" not in build_visual_constraint_text(
        context, context.image_candidates[0]
    )


def test_schema_version_has_one_authoritative_value():
    context = _build_context()
    payload = context.to_prompt_payload()

    assert payload["context_schema_version"] == context.context_schema_version
    assert payload["metadata"]["context_schema_version"] == context.context_schema_version


def test_visual_constraint_text_uses_primary_fact_and_positive_image_constraints_only():
    context = _build_context()
    candidate = context.image_candidates[0]

    visual_text = build_visual_constraint_text(context, candidate)

    assert visual_text == " ".join(
        [
            "entity type: film",
            "Image must include",
            "primary relation visual clue: lighthouse",
                "Image should include",
                "cinema",
                "arts",
                "documentary",
            ]
        )
    for forbidden in (
        "Unsafe image should",
        "IMAGE_CONTEXT_ONLY_MARKER",
        "IMAGE_AVOID_MARKER",
        "FORBIDDEN_POSITIVE_MARKER",
        GROUND_TRUTH_SENTINEL,
        INSTRUCTION_SENTINEL,
    ):
        assert forbidden not in visual_text


def test_visual_primary_clue_requires_matching_stage2_supported_fact():
    inputs = _context_inputs()
    inputs["reasoning"]["supported_facts"] = []
    context = build_ranking_context(**inputs)

    assert "primary relation visual clue: lighthouse" not in build_visual_constraint_text(
        context, context.image_candidates[0]
    )


def test_kg_reference_never_falls_back_to_relation_facts_when_stage2_is_empty():
    inputs = _context_inputs()
    inputs["reasoning"]["supported_facts"] = []
    context = build_ranking_context(**inputs)

    assert build_kg_reference(context, "text") == "Example Film entity type: film Text must include Text should include"


@pytest.mark.parametrize("bad", [Path("/tmp/fact"), b"fact", _CustomStringable(), ""])
def test_invalid_provenance_sequence_fails_closed_atomically(bad):
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"] = [{
        "relation": "directed by",
        "fact": "must not leak",
        "support_level": "strong",
        "modality_relevance": ["text", bad],
    }]
    inputs["reasoning"]["supported_facts"] = [{
        "fact": "must not leak",
        "source_relations": ["directed by", bad],
        "support_level": "strong",
    }]
    context = build_ranking_context(**inputs)
    assert context.relation_facts == ()
    assert context.supported_facts == ()
    assert "must not leak" not in build_kg_reference(context, "text")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("supported", "yes"),
        ("excluded", "no"),
        ("context_only", 0),
        ("status", "future"),
        ("safety", "unsafe"),
    ],
)
def test_factory_semantic_invalidity_cannot_be_laundered_by_double_sanitization(field, value):
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"][0][field] = value
    inputs["reasoning"]["supported_facts"] = [{
        "fact": "Primary supported fact",
        "source_relations": ["directed by"],
        "support_level": "strong",
    }]
    context = build_ranking_context(**inputs)

    assert "Primary supported fact" not in build_kg_reference(context, "text")
    assert "_invalid_semantics" not in repr(context)
    assert "_invalid_semantics" not in json.dumps(context.to_prompt_payload())


@pytest.mark.parametrize(
    ("field", "value"),
    [("safe", "yes"), ("status", "future"), ("safety", "unsafe")],
)
def test_factory_constraint_semantic_invalidity_is_dropped_without_markers(field, value):
    inputs = _context_inputs()
    inputs["reasoning"]["modality_constraints"]["text"]["must_include"] = [{
        "constraint": "Malformed constraint must not leak",
        "confidence": "high",
        field: value,
    }]
    context = build_ranking_context(**inputs)

    assert "Malformed constraint must not leak" not in build_kg_reference(context, "text")
    assert "_invalid_semantics" not in json.dumps(context.to_prompt_payload())


def test_provenance_sequences_are_atomic_for_facts_constraints_and_candidates():
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"][0]["evidence_triples"] = ["Q1 P57 Q2", ""]
    inputs["reasoning"]["supported_facts"] = [{
        "fact": "Primary supported fact",
        "source_relations": ["directed by"],
        "evidence": ["Q1 P57 Q2", ""],
        "support_level": "strong",
    }]
    inputs["reasoning"]["modality_constraints"]["text"]["must_include"] = [{
        "constraint": "Invalid provenance constraint",
        "confidence": "high",
        "source_relations": ["directed by", ""],
    }]
    inputs["generation"]["relation_alternation_generation"]["text_candidates"][0][
        "used_evidence_triples"
    ] = ["Q1 P57 Q2", ""]
    inputs["generation"]["relation_alternation_generation"]["text_candidates"][1][
        "expected_relation_coverage_detail"
    ] = {"primary": "country", "supporting": ["directed by", ""]}

    context = build_ranking_context(**inputs)

    assert "Primary supported fact" not in build_kg_reference(context, "text")
    assert "Invalid provenance constraint" not in build_kg_reference(context, "text")
    assert all(candidate["candidate_id"] != "text_001" for candidate in context.text_candidates)
    assert all(candidate["candidate_id"] != "text_002" for candidate in context.text_candidates)


@pytest.mark.parametrize("missing_mode", ["text", "image"])
def test_missing_mode_removes_opposite_modality_context_and_transport(missing_mode):
    inputs = _context_inputs()
    inputs["missing_mode"] = missing_mode
    context = build_ranking_context(**inputs)
    opposite = "image" if missing_mode == "text" else "text"

    assert not getattr(context, f"{opposite}_candidates")
    assert opposite not in context.modality_constraints
    if missing_mode == "text":
        assert context.generated_image_inputs == {}
        assert context.existing_image_input == inputs["existing_image_input"]
        payload = context.to_prompt_payload()
        assert payload["has_existing_image"] is True
        assert "/private/existing.png" not in json.dumps(payload)
        with pytest.raises(ValueError, match="missing_mode"):
            build_kg_reference(context, "image")
        with pytest.raises(ValueError, match="missing_mode"):
            build_visual_constraint_text(context, {"target_modality": "image"})
    else:
        assert context.existing_modalities["text"] is None
        with pytest.raises(ValueError, match="missing_mode"):
            build_kg_reference(context, "text")


@pytest.mark.parametrize("bad", [Path("/tmp/path"), b"bytes", _CustomStringable(), {"x"}])
def test_freeze_value_rejects_non_json_values_with_a_path(bad):
    with pytest.raises((TypeError, ValueError), match=r"\$"):
        freeze_value({"nested": [bad]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_score_details_reject_nonfinite_values_with_a_path(bad):
    with pytest.raises(ValueError, match=r"\$"):
        ScoreResult(0.1, 0.1, "scored", "model", details={"metric": bad})


@pytest.mark.parametrize("field", ["knowledge", "reasoning", "generation", "existing_modalities"])
def test_factory_rejects_non_mapping_stage_inputs(field):
    inputs = _context_inputs()
    inputs[field] = Path("/tmp/not-a-mapping")
    with pytest.raises(TypeError, match=field):
        build_ranking_context(**inputs)


@pytest.mark.parametrize("field", ["entity_key", "entity_label"])
def test_context_keys_must_be_nonempty_strings(field):
    inputs = _context_inputs()
    inputs[field] = "   "
    with pytest.raises(ValueError, match=field):
        build_ranking_context(**inputs)


def test_real_stage1_to_stage4_outputs_build_context_smoke():
    record = EntityRecord(
        entity_id=1,
        entity_key="Q-smoke",
        label="Smoke Entity",
        description="A grounded entity.",
        neighbors=[
            {
                "entity_key": "Q-type",
                "label": "film",
                "description": "a work",
                "relation": "instance of",
                "direction": "out",
            },
            {
                "entity_key": "Q-country",
                "label": "courtroom portrait",
                "description": "visual evidence",
                "relation": "depicts",
                "direction": "out",
            },
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "secret", "image": "secret.png"},
    )
    missing_mode = "both"
    coarse = coarse_relation_filtering(record, missing_mode)
    knowledge = knowledge_modeling(record, missing_mode, coarse)
    reasoning = constraint_compilation(missing_mode, knowledge)
    selection = fine_grained_relation_selection(missing_mode, knowledge, reasoning, candidate_budget=1)
    generation = relation_alternation_generation(
        record,
        missing_mode,
        knowledge,
        reasoning,
        selection,
        RuleTextGenerator(),
        RuleImagePromptGenerator(),
        candidate_budget=1,
    )
    context = build_ranking_context(
        entity_key=record.entity_key,
        entity_label=record.label,
        missing_mode=missing_mode,
        existing_modalities=record.existing_modalities,
        knowledge=knowledge,
        reasoning=reasoning,
        generation=generation,
        generated_image_inputs={},
    )
    payload = context.to_prompt_payload()
    json.dumps(payload, allow_nan=False)
    assert context.text_candidates and context.image_candidates


def test_ranking_context_preserves_stage2_hard_soft_constraint_metadata():
    inputs = _context_inputs()
    constraint = inputs["reasoning"]["modality_constraints"]["text"]["must_include"][0]
    constraint.update(
        {
            "severity": "hard",
            "gate_action": "hard_required",
            "violation_code": "missing_core_must_include",
            "core": True,
        }
    )

    context = build_ranking_context(**inputs)
    item = context.modality_constraints["text"]["must_include"][0]

    assert item["severity"] == "hard"
    assert item["gate_action"] == "hard_required"
    assert item["violation_code"] == "missing_core_must_include"
    assert item["core"] is True


def test_invalid_fact_and_constraint_rejections_survive_replace_and_factory_reentry():
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"] = [{
        "relation": "depicts",
        "fact": "LAUNDERED_VISUAL_FACT",
        "support_level": "strong",
        "modality_relevance": ["image"],
        "status": "future",
    }]
    inputs["reasoning"]["supported_facts"] = [{
        "fact": "LAUNDERED_VISUAL_FACT",
        "source_relations": ["depicts"],
        "support_level": "strong",
    }]
    inputs["reasoning"]["modality_constraints"]["text"]["must_include"] = [{
        "constraint": "LAUNDERED_TEXT_CONSTRAINT",
        "constraint_type": "unknown_type",
        "confidence": "high",
    }]
    inputs["reasoning"]["modality_constraints"]["image"]["must_include"] = [{
        "constraint": "LAUNDERED_IMAGE_CONSTRAINT",
        "constraint_type": "unknown_type",
        "confidence": "high",
    }]
    first = build_ranking_context(**inputs)
    replaced = replace(first)
    reentered = build_ranking_context(
        entity_key=replaced.entity_key,
        entity_label=replaced.entity_label,
        missing_mode=replaced.missing_mode,
        existing_modalities=replaced.existing_modalities,
        knowledge={
            "entity_profile": replaced.entity_profile,
            "relation_facts": replaced.relation_facts,
        },
        reasoning={
            "supported_facts": replaced.supported_facts,
            "safe_inferences": replaced.safe_inferences,
            "modality_constraints": replaced.modality_constraints,
            "forbidden_content": replaced.forbidden_content,
        },
        generation={
            "relation_alternation_generation": {
                "text_candidates": replaced.text_candidates,
                "image_candidates": replaced.image_candidates,
            }
        },
        generated_image_inputs=replaced.generated_image_inputs,
        existing_image_input=replaced.existing_image_input,
    )

    for context in (first, replaced, reentered):
        assert "LAUNDERED_VISUAL_FACT" not in build_kg_reference(context, "image")
        assert "LAUNDERED_TEXT_CONSTRAINT" not in build_kg_reference(context, "text")
        assert "LAUNDERED_IMAGE_CONSTRAINT" not in build_visual_constraint_text(
            context, context.image_candidates[0]
        )
        assert "_invalid_" not in json.dumps(context.to_prompt_payload())


def test_contracts_normalize_scalar_subclasses_before_validation_and_storage():
    score = ScoreResult(
        _ComparisonOverridingFloat(0.4),
        _ComparisonOverridingFloat(0.5),
        _StringSubclass("scored"),
        _StringSubclass("model"),
        _StringSubclass("version"),
        _ComparisonOverridingInt(7),
    )
    gate = GateResult(
        _StringSubclass("pass"),
        _StringSubclass("judge"),
        hard_violations=(_StringSubclass("hard"),),
        soft_violations=(_StringSubclass("soft"),),
        judge_confidence=_ComparisonOverridingFloat(0.8),
        evidence=(_StringSubclass("evidence"),),
        reason=_StringSubclass("safe"),
    )

    assert type(score.raw_score) is float
    assert type(score.normalized_score) is float
    assert type(score.status) is str
    assert type(score.model_name) is str
    assert type(score.model_version) is str
    assert type(score.latency_ms) is int
    assert type(gate.decision) is str
    assert type(gate.source) is str
    assert type(gate.hard_violations[0]) is str
    assert type(gate.soft_violations[0]) is str
    assert type(gate.judge_confidence) is float
    assert type(gate.evidence[0]) is str
    assert type(gate.reason) is str


def test_contracts_reject_comparison_overriding_scalar_subclasses_by_value():
    with pytest.raises(ValueError, match="status"):
        ScoreResult(0.4, 0.5, _ForgedCategoricalString("forged", "scored"), "model")
    with pytest.raises(ValueError, match="latency_ms"):
        ScoreResult(0.4, 0.5, "scored", "model", latency_ms=_ComparisonOverridingInt(-1))
    with pytest.raises(ValueError, match="decision"):
        GateResult(_ForgedCategoricalString("forged", "pass"), "judge")
    with pytest.raises(ValueError, match="judge_confidence"):
        GateResult("pass", "judge", judge_confidence=_ComparisonOverridingFloat(1.5))


def test_context_keys_and_missing_mode_normalize_or_reject_string_subclasses():
    inputs = _context_inputs()
    inputs["entity_key"] = _StringSubclass("Q-normalized")
    inputs["entity_label"] = _StringSubclass("Normalized Entity")
    inputs["missing_mode"] = _StringSubclass("both")
    context = build_ranking_context(**inputs)

    assert type(context.entity_key) is str
    assert type(context.entity_label) is str
    assert type(context.missing_mode) is str
    assert type(replace(context).entity_key) is str

    for value in (
        _ForgedCategoricalString("future", "both"),
        _ComparisonOverridingString(""),
    ):
        with pytest.raises(ValueError, match="missing_mode"):
            replace(context, missing_mode=value)


def test_constraint_categories_are_allowlisted_and_fail_closed():
    inputs = _context_inputs()
    inputs["reasoning"]["modality_constraints"] = {
        "text": {
            "must_include": [
                {"constraint": "identity allowed", "constraint_type": "identity", "confidence": "high"},
                {"constraint": "grounded allowed", "constraint_type": "grounded_fact", "confidence": "high"},
                {"constraint": "unknown type", "constraint_type": "future_type", "confidence": "high"},
                {"constraint": "unknown status", "status": "future", "confidence": "high"},
                {"constraint": "unknown safety", "safety": "future", "confidence": "high"},
                {"constraint": "unknown confidence", "confidence": "future"},
                {"constraint": "unknown role", "role": "future", "confidence": "high"},
            ],
            "should_include": [],
            "context_only": [
                {"constraint": "context bucket", "constraint_type": "context", "confidence": "high"}
            ],
            "avoid": [
                {"constraint": "avoid bucket", "constraint_type": "avoid_unsupported_detail", "confidence": "high"}
            ],
        }
    }
    context = build_ranking_context(**inputs)
    reference = build_kg_reference(context, "text")

    assert "identity allowed" in reference
    assert "grounded allowed" in reference
    for rejected in (
        "unknown type",
        "unknown status",
        "unknown safety",
        "unknown confidence",
        "unknown role",
        "context bucket",
        "avoid bucket",
    ):
        assert rejected not in reference


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", _HostileString("supported")),
        ("safety", _HostileString("safe")),
        ("constraint_type", _HostileString("identity")),
        ("role", _HostileString("primary")),
    ],
)
def test_hostile_fact_and_constraint_labels_are_normalized_without_subclass_methods(
    field, value
):
    inputs = _context_inputs()
    inputs["knowledge"]["relation_facts"][0][field] = value
    inputs["reasoning"]["supported_facts"][0][field] = value
    inputs["reasoning"]["modality_constraints"]["text"]["must_include"].append(
        {"constraint": "hostile allowed", "confidence": "high", field: value}
    )

    context = build_ranking_context(**inputs)

    assert "hostile allowed" in build_kg_reference(context, "text")


def test_hostile_public_strings_are_normalized_before_string_methods():
    inputs = _context_inputs()
    inputs["knowledge"]["entity_profile"]["entity_type"] = _HostileString("film")
    inputs["knowledge"]["relation_facts"][0]["fact"] = _HostileString("Primary supported fact")
    inputs["generation"]["relation_alternation_generation"]["text_candidates"][0][
        "candidate_id"
    ] = _HostileString("text_001")
    inputs["generation"]["relation_alternation_generation"]["image_candidates"][0][
        "generated_content"
    ]["image_artifact_id"] = _HostileString("artifact-image-001")
    inputs["reasoning"]["modality_constraints"]["text"]["must_include"].append(
        _HostileString("bare hostile constraint")
    )

    context = build_ranking_context(**inputs)

    assert candidate_generated_text(
        {"generated_content": {"value": _HostileString(" generated hostile text ")}}
    ) == "generated hostile text"
    assert context.modality_constraints["text"]["must_include"][-1]["constraint"] == (
        "bare hostile constraint"
    )
    assert type(context.entity_profile["entity_type"]) is str
    assert type(context.relation_facts[0]["fact"]) is str
    assert type(context.text_candidates[0]["candidate_id"]) is str
    assert type(context.image_candidates[0]["generated_content"]["image_artifact_id"]) is str


def test_public_modality_and_visual_candidate_checks_use_exact_strings():
    context = _build_context()

    with pytest.raises(ValueError, match="modality"):
        build_kg_reference(context, _HostileCategoricalString("future", "text"))
    with pytest.raises(ValueError, match="candidate"):
        build_visual_constraint_text(
            context,
            {
                "target_modality": _HostileCategoricalString("future", "image"),
                "primary_relation": _HostileString("depicts"),
            },
        )

    candidate = {
        "target_modality": "image",
        "primary_relation": _HostileString("depicts"),
    }
    assert "primary relation visual clue: lighthouse" in build_visual_constraint_text(
        context, candidate
    )
