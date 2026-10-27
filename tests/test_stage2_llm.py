"""Behavioral coverage for the optional Stage 3 LLM semantic assessor."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import error

import pytest

import mmkgc.stages.selection as selection_module
from mmkgc.adapters.qwen import QwenConfig
from mmkgc.adapters.stage2_llm import QwenLocalStage2LLMAssessor
from mmkgc.data import EntityRecord
from mmkgc.io import RunWriter
from mmkgc.stages import fine_grained_relation_selection as _main_fine_grained_relation_selection



def _knowledge() -> dict:
    return {
        "relation_facts": [
            {
                "fact_id": "fact:director",
                "relation": "director",
                "fact": "Example Film has director: Jane Example.",
                "support_level": "strong",
                "modality_relevance": ["text"],
                "generation_use": "use_as_grounded_constraint",
                "risk": "low",
                "evidence_triples": ["(Example Film, director, Jane Example)"],
            }
        ]
    }


def _reasoning() -> dict:
    return {
        "supported_facts": [{"fact_id": "supported:director", "source_fact_refs": ["fact:director"]}],
        "modality_constraints": {
            "text": {
                "must_include": [
                    {"constraint": "director: Jane Example", "source_relations": ["director"]}
                ],
                "should_include": [],
                "context_only": [],
                "avoid": [],
            }
        },
    }


class _AdjustingAssessor:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def assess(self, **kwargs):
        self.inputs.append(kwargs)
        return {
            "status": "applied",
            "recommended_role": "secondary",
            "component_adjustments": {"generation_utility": 9.0},
            "risk_penalty_adjustment": -9.0,
            "confidence": 0.8,
            "reason": "The fact is especially useful for a textual description.",
            "model_call_ref": "stage2_llm:fact:director",
            "evidence_triples": [],
            "prompt": "secret prompt that belongs only in JSONL",
            "response_text": "secret response that belongs only in JSONL",
            "model_call": {
                "stage": "stage2_llm_assessment",
                "provider": "fake-stage2-assessor",
                "prompt": "secret prompt that belongs only in JSONL",
                "response_text": "secret response that belongs only in JSONL",
                "latency_ms": 1,
                "status": "generated",
            },
        }


class _FixedEmbeddingScorer:
    def similarity(self, _left, _right):
        return 0.7


class _FailedAdjustingAssessor:
    def assess(self, **_kwargs):
        return {
            "status": "failed",
            "component_adjustments": {"generation_utility": 0.15},
            "risk_penalty_adjustment": -0.15,
            "confidence": 0.9,
            "reason": "The remote assessment failed.",
        }


class _StatusAssessor:
    def __init__(self, status: str) -> None:
        self.status = status

    def assess(self, **_kwargs):
        return {
            "status": self.status,
            "component_adjustments": {"generation_utility": 0.1},
            "risk_penalty_adjustment": 0.0,
            "confidence": 0.9,
            "reason": "Compatibility status test.",
        }


class _EvidenceAdjustingAssessor:
    def assess(self, **_kwargs):
        return {
            "status": "applied",
            "component_adjustments": {"evidence_strength": 0.1},
            "risk_penalty_adjustment": 0.0,
            "confidence": 0.9,
            "reason": "The assessor must not own evidence strength.",
        }


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _relation(selection: dict) -> dict:
    payload = selection["fine_grained_relation_selection"]
    return next(
        item
        for key in ("primary_relations", "secondary_relations", "context_relations", "excluded_relations")
        for item in payload[key]
        if item["relation"] == "director"
    )


def fine_grained_relation_selection(*args, **kwargs):
    kwargs.setdefault("embedding_scorer", _FixedEmbeddingScorer())
    return _main_fine_grained_relation_selection(*args, **kwargs)


def test_stage2_default_has_no_assessment_or_model_calls():
    selection = fine_grained_relation_selection("text", _knowledge(), _reasoning())

    assert selection["model_calls"] == []
    assert "llm_assessment" not in _relation(selection)


def test_stage2_assessor_clamps_adjustments_and_recomputes_local_score():
    baseline = fine_grained_relation_selection("text", _knowledge(), _reasoning())
    assessor = _AdjustingAssessor()
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=assessor
    )

    baseline_relation = _relation(baseline)
    assessed_relation = _relation(selection)
    assessment = assessed_relation["llm_assessment"]

    assert assessor.inputs[0]["relation"] == "director"
    assert assessor.inputs[0]["fact"] == "Example Film has director: Jane Example."
    assert assessor.inputs[0]["target_modalities"] == ["text"]
    assert assessor.inputs[0]["generation_plan"]["must_include"]
    assert assessor.inputs[0]["score_components"] == baseline_relation["score_breakdown"]
    assert assessment["component_adjustments"] == {"generation_utility": 0.15}
    assert assessment["risk_penalty_adjustment"] == -0.15
    assert assessed_relation["score_breakdown"]["generation_utility"] == 0.85
    assert assessed_relation["score_breakdown"]["risk_penalty"] == 0.0
    assert assessed_relation["fine_score"] == pytest.approx(0.745)
    assert assessed_relation["source_fact"]["evidence_triples"] == [
        "(Example Film, director, Jane Example)"
    ]


def test_stage2_failed_assessment_retains_local_score_and_zeroes_adjustments():
    baseline = fine_grained_relation_selection("text", _knowledge(), _reasoning())
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_FailedAdjustingAssessor()
    )

    baseline_relation = _relation(baseline)
    assessed_relation = _relation(selection)
    assert assessed_relation["fine_score"] == baseline_relation["fine_score"]
    assert assessed_relation["score_breakdown"] == baseline_relation["score_breakdown"]
    assert assessed_relation["llm_assessment"]["status"] == "failed"
    assert assessed_relation["llm_assessment"]["component_adjustments"] == {}
    assert assessed_relation["llm_assessment"]["risk_penalty_adjustment"] == 0.0


def test_stage2_failed_assessment_preserves_unrounded_baseline_score(monkeypatch):
    baseline_score = 0.193564
    rounded_components = {
        "modality_fit": 0.64521,
        "constraint_alignment": 0.0,
        "evidence_strength": 0.0,
        "generation_utility": 0.0,
        "risk_penalty": 0.0,
    }
    monkeypatch.setattr(
        selection_module,
        "_score_fact",
        lambda *_args, **_kwargs: (baseline_score, dict(rounded_components)),
    )

    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_FailedAdjustingAssessor()
    )
    relation = _relation(selection)

    assert selection_module._score_from_components(rounded_components) == 0.193563
    assert relation["fine_score"] == baseline_score
    assert relation["score_breakdown"] == rounded_components


def test_stage2_adjusted_assessment_applies_bounded_delta():
    baseline = fine_grained_relation_selection("text", _knowledge(), _reasoning())
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_StatusAssessor("adjusted")
    )

    baseline_relation = _relation(baseline)
    assessed_relation = _relation(selection)
    assert assessed_relation["llm_assessment"]["status"] == "adjusted"
    assert assessed_relation["llm_assessment"]["component_adjustments"] == {
        "generation_utility": 0.1
    }
    assert assessed_relation["score_breakdown"]["generation_utility"] == pytest.approx(
        baseline_relation["score_breakdown"]["generation_utility"] + 0.1
    )
    assert assessed_relation["fine_score"] > baseline_relation["fine_score"]


def test_stage2_accept_assessment_retains_baseline_score_without_delta():
    baseline = fine_grained_relation_selection("text", _knowledge(), _reasoning())
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_StatusAssessor("accept")
    )

    baseline_relation = _relation(baseline)
    assessed_relation = _relation(selection)
    assert assessed_relation["llm_assessment"]["status"] == "accept"
    assert assessed_relation["llm_assessment"]["component_adjustments"] == {}
    assert assessed_relation["llm_assessment"]["risk_penalty_adjustment"] == 0.0
    assert assessed_relation["fine_score"] == baseline_relation["fine_score"]
    assert assessed_relation["score_breakdown"] == baseline_relation["score_breakdown"]


def test_stage2_assessment_does_not_adjust_evidence_strength():
    baseline = fine_grained_relation_selection("text", _knowledge(), _reasoning())
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_EvidenceAdjustingAssessor()
    )

    baseline_relation = _relation(baseline)
    assessed_relation = _relation(selection)
    assert assessed_relation["llm_assessment"]["component_adjustments"] == {}
    assert assessed_relation["fine_score"] == baseline_relation["fine_score"]
    assert assessed_relation["score_breakdown"] == baseline_relation["score_breakdown"]


def test_stage2_llm_adjusted_score_method_is_explicit_in_selection_and_artifact(tmp_path):
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_AdjustingAssessor()
    )
    embedding_selection = fine_grained_relation_selection(
        "text",
        _knowledge(),
        _reasoning(),
        embedding_scorer=_FixedEmbeddingScorer(),
        llm_assessor=_AdjustingAssessor(),
    )
    writer = RunWriter(tmp_path / "outputs", "score_method", {})

    writer.write_entity(
        {
            "entity": {"entity_key": "example_film"},
            "fine_grained_relation_selection": selection,
            "model_calls": selection["model_calls"],
        }
    )

    stage_data = json.loads(
        (writer.run_dir / "entities" / "example_film" / "2_fine_grained_relation_selection.json").read_text(
            encoding="utf-8"
        )
    )["data"]["relation_plans"][0]
    assert _relation(selection)["score_method"] == "embedding_component_formula_v2+llm_semantic_adjustment_v1"
    assert _relation(embedding_selection)["score_method"] == (
        "embedding_component_formula_v2+llm_semantic_adjustment_v1"
    )
    assert stage_data["score"]["method"] == "embedding_component_formula_v2+llm_semantic_adjustment_v1"


def test_stage2_writer_keeps_only_structured_assessment_and_call_reference(tmp_path):
    record = EntityRecord(
        entity_id=1,
        entity_key="example_film",
        label="Example Film",
        description="",
        neighbors=[],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": None, "image": None},
    )
    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_AdjustingAssessor()
    )
    relation = _relation(selection)
    relation["llm_assessment"]["recommended_role"] = "unsafe-role"
    relation["llm_assessment"]["reason"] = "prompt response api_key=secret-stage2-token " + "x" * 300
    result = {
        "entity": {"entity_key": record.entity_key},
        "fine_grained_relation_selection": selection,
        "model_calls": selection["model_calls"],
    }
    writer = RunWriter(tmp_path / "outputs", "stage2_llm", {"stage2_llm_assessor": "qwen-local"})

    writer.write_entity(result)

    entity_dir = writer.run_dir / "entities" / record.entity_key
    stage_json = (entity_dir / "2_fine_grained_relation_selection.json").read_text(encoding="utf-8")
    stage_data = json.loads(stage_json)["data"]["relation_plans"][0]
    model_call_jsonl = (entity_dir / "model_calls.jsonl").read_text(encoding="utf-8")

    assert stage_data["model_call_ref"] == "stage2_llm:fact:director"
    assert stage_data["llm_assessment"]["component_adjustments"] == {"generation_utility": 0.15}
    assert "secret prompt" not in stage_json
    assert "secret response" not in stage_json
    assert "secret prompt" in model_call_jsonl
    assert "secret response" in model_call_jsonl
    assert "recommended_role" not in stage_data["llm_assessment"]
    assert len(stage_data["llm_assessment"]["reason"]) <= 240
    assert "secret-stage2-token" not in stage_data["llm_assessment"]["reason"]
    assert "prompt" not in stage_data["llm_assessment"]["reason"].lower()
    assert "response" not in stage_data["llm_assessment"]["reason"].lower()


def test_qwen_stage2_assessor_success_and_malformed_response_redact_persisted_values(tmp_path, monkeypatch):
    secret = "secret-stage2-token"
    independent_password = "independent-password"
    sensitive_url = f"https://user:{independent_password}@host/path"
    assessor = QwenLocalStage2LLMAssessor(
        QwenConfig(
            model_name="Qwen",
            base_url=f"http://user:{secret}@localhost/v1?api_key={secret}",
            api_key=secret,
        )
    )
    success = {
        "choices": [{"message": {"content": json.dumps({
            "status": "applied", "component_adjustments": {"generation_utility": 0.1},
            "risk_penalty_adjustment": 0.0, "confidence": 0.9,
            "reason": f"use {secret} via {sensitive_url}",
        })}}],
        "api_key": secret,
    }
    monkeypatch.setattr(
        "mmkgc.adapters.stage2_llm.request.urlopen",
        lambda *_args, **_kwargs: _Response(success),
    )

    outcome = assessor.assess(
        relation="director", fact="Example Film has director: Jane Example.",
        target_modalities=["text"], generation_plan={"must_include": []},
        score_components={"modality_fit": 1.0, "constraint_alignment": 1.0, "evidence_strength": 0.9, "generation_utility": 0.8, "risk_penalty": 0.0},
    )

    call_text = json.dumps(outcome["model_call"])
    assert outcome["status"] == "applied"
    assert outcome["model_call"]["base_url"] == "http://localhost/v1?api_key=%5BREDACTED%5D"
    assert secret not in call_text
    assert independent_password not in call_text
    assert independent_password not in outcome["model_call"]["response_text"]
    assert independent_password not in json.dumps(outcome["model_call"]["response"])

    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_AdjustingAssessor()
    )
    _relation(selection)["llm_assessment"]["reason"] = outcome["reason"]
    writer = RunWriter(tmp_path / "outputs", "url_reason", {})
    writer.write_entity(
        {
            "entity": {"entity_key": "example_film"},
            "fine_grained_relation_selection": selection,
            "model_calls": [outcome["model_call"]],
        }
    )
    stage_data = json.loads(
        (writer.run_dir / "entities" / "example_film" / "2_fine_grained_relation_selection.json").read_text(
            encoding="utf-8"
        )
    )["data"]["relation_plans"][0]
    assert independent_password not in stage_data["llm_assessment"]["reason"]

    malformed = {"choices": [{"message": {"content": f"not-json {secret}"}}]}
    monkeypatch.setattr(
        "mmkgc.adapters.stage2_llm.request.urlopen",
        lambda *_args, **_kwargs: _Response(malformed),
    )
    with pytest.raises(RuntimeError) as exc_info:
        assessor.assess(
            relation="director", fact="fact", target_modalities=["text"], generation_plan={},
            score_components={"modality_fit": 1.0, "constraint_alignment": 1.0, "evidence_strength": 0.9, "generation_utility": 0.8, "risk_penalty": 0.0},
        )
    assert secret not in str(exc_info.value)


def test_qwen_stage2_assessor_network_failure_redacts_error(monkeypatch):
    secret = "secret-stage2-token"
    independent_password = "independent-password"
    assessor = QwenLocalStage2LLMAssessor(
        QwenConfig(model_name="Qwen", base_url="http://localhost/v1", api_key=secret)
    )
    monkeypatch.setattr(
        "mmkgc.adapters.stage2_llm.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            error.URLError(f"endpoint token={secret} https://user:{independent_password}@host/path")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        assessor.assess(
            relation="director", fact="fact", target_modalities=["text"], generation_plan={},
            score_components={"modality_fit": 1.0, "constraint_alignment": 1.0, "evidence_strength": 0.9, "generation_utility": 0.8, "risk_penalty": 0.0},
        )

    assert secret not in str(exc_info.value)
    assert independent_password not in str(exc_info.value)


@pytest.mark.parametrize(
    "response_text",
    [
        '{"status":"ok","component_adjustments":{}}',
        '```json\n{"status":"ok","component_adjustments":{}}\n```',
        '```\n{"status":"ok","component_adjustments":{}}\n```',
    ],
)
def test_qwen_stage2_assessor_parses_bare_and_fenced_json_objects(response_text):
    assert QwenLocalStage2LLMAssessor._parse_assessment(response_text) == {
        "status": "ok",
        "component_adjustments": {},
    }


@pytest.mark.parametrize(
    "response_text",
    [
        "[]",
        "```json\n[]\n```",
    ],
)
def test_qwen_stage2_assessor_rejects_non_object_json(response_text):
    with pytest.raises(ValueError, match="JSON object"):
        QwenLocalStage2LLMAssessor._parse_assessment(response_text)


def test_qwen_stage2_assessor_prompt_restricts_statuses_and_adjustment_components():
    prompt = QwenLocalStage2LLMAssessor._build_prompt(
        relation="director",
        fact="Example Film has director: Jane Example.",
        target_modalities=["text"],
        generation_plan={"must_include": []},
        score_components={
            "modality_fit": 1.0,
            "constraint_alignment": 1.0,
            "evidence_strength": 0.9,
            "generation_utility": 0.8,
            "risk_penalty": 0.0,
        },
    )

    assert "applied means apply the deltas" in prompt
    assert "ok means no adjustment is needed" in prompt
    assert "failed means the assessment is unavailable" in prompt
    assert "modality_fit, constraint_alignment, generation_utility" in prompt
    assert "Do not include evidence_strength or risk_penalty" in prompt
    assert "risk_penalty_adjustment" in prompt


def test_qwen_assessment_and_writer_outputs_redact_string_credentials(tmp_path, monkeypatch):
    api_key = "known-api-key"
    other_secret = "other-secret"
    config_password = "config-password"
    config_token = "config-token"
    assessor = QwenLocalStage2LLMAssessor(
        QwenConfig(model_name="Qwen", base_url="http://localhost/v1", api_key=api_key)
    )
    response = {
        "choices": [{"message": {"content": json.dumps({
            "status": "applied",
            "component_adjustments": {},
            "risk_penalty_adjustment": 0.0,
            "confidence": 0.9,
            "reason": (
                f'echo {api_key}; {{"api_key":"{other_secret}"}}; '
                f'https://user:{other_secret}@host/path?token={other_secret}; '
                '{"authorization":"Bearer keep-me"}; '
                'https://host/v1?access_token=keep-me&client_secret=also-keep; '
                '{"client_secret":"also-keep"}'
            ),
        })}}],
    }
    monkeypatch.setattr(
        "mmkgc.adapters.stage2_llm.request.urlopen",
        lambda *_args, **_kwargs: _Response(response),
    )

    outcome = assessor.assess(
        relation="director", fact="fact", target_modalities=["text"], generation_plan={},
        score_components={"modality_fit": 1.0, "constraint_alignment": 1.0, "evidence_strength": 0.9, "generation_utility": 0.8, "risk_penalty": 0.0},
    )
    for secret in (api_key, other_secret):
        assert secret not in json.dumps(outcome)

    selection = fine_grained_relation_selection(
        "text", _knowledge(), _reasoning(), llm_assessor=_AdjustingAssessor()
    )
    _relation(selection)["llm_assessment"] = {
        key: value
        for key, value in outcome.items()
        if key != "model_call"
    }
    writer = RunWriter(
        tmp_path / "outputs",
        "string_credentials",
        {
            "base_url": (
                f"https://user:{config_password}@host/path?token={config_token}&mode=local; "
                'https://host/v1?access_token=keep-me&client_secret=also-keep'
            ),
            "note": (
                f'{{"authorization":"{other_secret}"}}; '
                '{"authorization":"Bearer keep-me"}; '
                '{"client_secret":"also-keep"}'
            ),
        },
    )
    writer.write_entity(
        {
            "entity": {"entity_key": "example_film"},
            "fine_grained_relation_selection": selection,
            "model_calls": [outcome["model_call"]],
        }
    )

    entity_dir = writer.run_dir / "entities" / "example_film"
    artifact = (entity_dir / "2_fine_grained_relation_selection.json").read_text(encoding="utf-8")
    model_calls = (entity_dir / "model_calls.jsonl").read_text(encoding="utf-8")
    config = (writer.run_dir / "config.json").read_text(encoding="utf-8")
    assessment_text = json.dumps(outcome)
    for secret in (
        api_key,
        other_secret,
        config_password,
        config_token,
        "keep-me",
        "also-keep",
    ):
        assert secret not in assessment_text
        assert secret not in artifact
        assert secret not in model_calls
        assert secret not in config
