from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mmkgc.ranking import ModelFusionRankingEngine
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.contracts import GateResult, RequiredScorerError, ScoreResult
from mmkgc.ranking.image_input import ImageInput
from mmkgc.ranking.profiles import (
    LocalScorerConfig,
    GPT4OMiniJudgeConfig,
    Stage4Profile,
)
from mmkgc.ranking.scorers import GPT4OMiniJudgeParseError


INSTRUCTION_SENTINEL = "ZZZ_INSTRUCTION_ONLY"
RATIONALE_SENTINEL = "ZZZ_RATIONALE_ONLY"
GROUND_TRUTH_SENTINEL = "ZZZ_GROUND_TRUTH_ONLY"


class FakeSentenceBERTScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        self.calls.append({"reference": reference, "candidates": list(candidates)})
        return [
            ScoreResult(
                raw_score=self.scores[candidate],
                normalized_score=self.scores[candidate],
                status="scored",
                model_name="fake-sentence-bert",
            )
            for candidate in candidates
        ]


class FakeBERTScoreScorer:
    def __init__(self, scores: dict[str, tuple[float, float, float]]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        self.calls.append({"reference": reference, "candidates": list(candidates)})
        results = []
        for index, candidate in enumerate(candidates):
            precision, recall, f1 = self.scores[candidate]
            results.append(
                ScoreResult(
                    raw_score=f1,
                    normalized_score=f1,
                    status="scored",
                    model_name="fake-bertscore",
                    model_version="fake-roberta-large",
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


class FakeVisualScorer:
    def __init__(self, name: str, scores: dict[str, float]) -> None:
        self.model_name = name
        self.scores = scores
        self.quality_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

    def score_image_quality_batch(
        self,
        image: Any,
        texts: list[str],
    ) -> list[ScoreResult]:
        assert isinstance(image, ImageInput)
        with image.open_pil() as pil_image:
            assert pil_image.size == (1, 1)
        self.quality_calls.append({"image": image, "texts": list(texts)})
        return [
            ScoreResult(
                raw_score=self.scores[image.candidate_id] * 100.0,
                normalized_score=self.scores[image.candidate_id],
                status="scored",
                model_name=self.model_name,
                details={"safe_metric": f"{self.model_name}-quality"},
            )
            for _text in texts
        ]

    def score_image_text_batch(
        self,
        image: Any,
        texts: list[str],
    ) -> list[ScoreResult]:
        assert isinstance(image, ImageInput)
        with image.open_pil() as pil_image:
            assert pil_image.size == (1, 1)
        self.text_calls.append({"image": image, "texts": list(texts)})
        return [
            ScoreResult(
                raw_score=self.scores[image.candidate_id] * 100.0,
                normalized_score=self.scores[image.candidate_id],
                status="scored",
                model_name=self.model_name,
                details={"safe_metric": f"{self.model_name}-text"},
            )
            for _text in texts
        ]


class BadCountTextScorer:
    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        del reference, candidates
        return []


class StatusTextScorer:
    def __init__(self, status: str) -> None:
        self.status = status

    def score_candidates(
        self,
        reference: str,
        candidates: list[str],
    ) -> list[ScoreResult]:
        del reference
        return [
            ScoreResult(
                raw_score=None,
                normalized_score=None,
                status=self.status,  # type: ignore[arg-type]
                model_name="bad-status-text",
            )
            for _candidate in candidates
        ]


class FakeGPT4OMiniJudge:
    model_name = "fake-gpt4omini-judge"

    def __init__(
        self,
        *,
        text_scores: dict[str, float],
        image_scores: dict[str, float],
        pair_scores: dict[tuple[str, str], float],
        rejected_text_ids: set[str] | None = None,
        rejected_image_ids: set[str] | None = None,
        malformed_text_ids: set[str] | None = None,
        malformed_image_ids: set[str] | None = None,
        malformed_pair_ids: set[tuple[str, str]] | None = None,
    ) -> None:
        self.text_scores = text_scores
        self.image_scores = image_scores
        self.pair_scores = pair_scores
        self.rejected_text_ids = rejected_text_ids or set()
        self.rejected_image_ids = rejected_image_ids or set()
        self.malformed_text_ids = malformed_text_ids or set()
        self.malformed_image_ids = malformed_image_ids or set()
        self.malformed_pair_ids = malformed_pair_ids or set()
        self.calls: list[dict[str, Any]] = []

    def judge_text(self, context: Any, candidate: dict[str, Any]) -> Any:
        self.calls.append(
            {
                "kind": "text",
                "payload": context.to_prompt_payload(),
                "candidate": candidate,
            }
        )
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in self.malformed_text_ids:
            raise GPT4OMiniJudgeParseError(
                "malformed JSON judge response",
                details={"contract_repair_failed": True},
            )
        rejected = candidate_id in self.rejected_text_ids
        return _judge_result(
            self.text_scores[candidate_id],
            rejected=rejected,
            hard_code="unsupported_claim",
            reason=f"text judge {candidate_id}",
        )

    def extract_text_graph(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "triples": [
                {
                    "subject": "Example Entity",
                    "predicate": "genre",
                    "object": "film",
                    "polarity": "positive",
                    "confidence": 1.0,
                    "evidence": str(candidate["generated_content"]["value"]),
                }
            ]
        }

    def judge_image(
        self,
        context: Any,
        candidate: dict[str, Any],
        image_input: ImageInput | None = None,
    ) -> Any:
        self.calls.append(
            {
                "kind": "image",
                "payload": context.to_prompt_payload(),
                "candidate": candidate,
                "image_input": image_input,
            }
        )
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in self.malformed_image_ids:
            raise GPT4OMiniJudgeParseError(
                "malformed JSON judge response",
                details={"contract_repair_failed": True},
            )
        rejected = candidate_id in self.rejected_image_ids
        return _judge_result(
            self.image_scores[candidate_id],
            rejected=rejected,
            hard_code="identity_mismatch",
            reason=f"image judge {candidate_id}",
        )

    def extract_visual_graph(
        self,
        candidate: dict[str, Any],
        *,
        image_input: ImageInput,
    ) -> dict[str, Any]:
        del candidate, image_input
        return {
            "nodes": [
                {
                    "id": "entity",
                    "label": "film",
                    "attributes": ["poster", "cinematic", "cinematic poster"],
                    "confidence": 1.0,
                    "evidence": "visible film poster",
                }
            ],
            "edges": [],
            "ocr": [],
        }

    def judge_pair(
        self,
        context: Any,
        text_candidate: dict[str, Any],
        image_candidate: dict[str, Any],
        image_input: ImageInput | None = None,
    ) -> Any:
        self.calls.append(
            {
                "kind": "pair",
                "payload": context.to_prompt_payload(),
                "text_candidate": text_candidate,
                "image_candidate": image_candidate,
                "image_input": image_input,
            }
        )
        key = (str(text_candidate["candidate_id"]), str(image_candidate["candidate_id"]))
        if key in self.malformed_pair_ids:
            raise GPT4OMiniJudgeParseError(
                "malformed JSON judge response",
                details={"contract_repair_failed": True},
            )
        return _judge_result(
            self.pair_scores.get(key, 0.5),
            rejected=False,
            hard_code="pair_mismatch",
            reason=f"pair judge {key[0]} {key[1]}",
        )


class FailingGPT4OMiniJudge:
    model_name = "failing-qwen-judge"

    def judge_text(self, context: Any, candidate: dict[str, Any]) -> Any:
        raise RuntimeError("judge service unavailable")

    def extract_text_graph(self, candidate: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("judge service unavailable")

    def judge_image(
        self,
        context: Any,
        candidate: dict[str, Any],
        image_input: ImageInput | None = None,
    ) -> Any:
        del image_input
        raise RuntimeError("judge service unavailable")

    def extract_visual_graph(
        self,
        candidate: dict[str, Any],
        *,
        image_input: ImageInput,
    ) -> dict[str, Any]:
        del candidate, image_input
        raise RuntimeError("judge service unavailable")

    def judge_pair(
        self,
        context: Any,
        text_candidate: dict[str, Any],
        image_candidate: dict[str, Any],
        image_input: ImageInput | None = None,
    ) -> Any:
        del image_input
        raise RuntimeError("judge service unavailable")


def _judge_result(
    score: float,
    *,
    rejected: bool,
    hard_code: str,
    reason: str,
) -> Any:
    class Result:
        pass

    result = Result()
    result.score = ScoreResult(
        raw_score=score * 100.0,
        normalized_score=score,
        status="scored",
        model_name="fake-gpt4omini-judge",
        details={"safe_metric": "judge"},
    )
    result.gate = GateResult(
        "reject" if rejected else "pass",
        "fake-gpt4omini-judge",
        hard_violations=(hard_code,) if rejected else (),
        soft_violations=("missing_should_include",) if not rejected else (),
        judge_confidence=0.9,
        evidence=("visible evidence",) if rejected else (),
        reason=reason,
    )
    return result


def _profile(
    *,
    sentence_bert_required: bool = True,
    bertscore_enabled: bool = False,
    bertscore_required: bool = False,
) -> Stage4Profile:
    required_local = LocalScorerConfig(
        required=True,
        enabled=True,
        model_dir=Path("models/fake"),
        device="cpu",
    )
    sentence_bert = LocalScorerConfig(
        required=sentence_bert_required,
        enabled=True,
        model_dir=Path("models/fake"),
        device="cpu",
    )
    return Stage4Profile(
        profile_name="test-profile",
        sentence_bert=sentence_bert,
        bertscore=LocalScorerConfig(
            required=bertscore_required,
            enabled=bertscore_enabled,
            model_dir=Path("models/bertscore/roberta-large"),
            device="cpu",
        ),
        clip=required_local,
        blip_itm=required_local,
        gpt4omini_judge=GPT4OMiniJudgeConfig(
            required=True,
            model_config=Path("configs/fake-gpt4omini.yaml"),
            timeout_seconds=30.0,
        ),
    )


def _context() -> Any:
    text_candidates = [
        _text_candidate("text_001", "Legacy first text."),
        _text_candidate("text_002", "Good grounded text."),
        _text_candidate("text_003", "Best grounded text."),
        _text_candidate("text_004", "Rejected text."),
    ]
    image_candidates = [
        _image_candidate("image_001"),
        _image_candidate("image_002"),
        _image_candidate("image_003"),
        _image_candidate("image_004"),
        _image_candidate("image_005"),
    ]
    image_inputs = {
        candidate["candidate_id"]: _image_transport(candidate["candidate_id"])
        for candidate in image_candidates
    }
    generation = {
        "relation_alternation_generation": {
            "text_candidates": text_candidates,
            "image_candidates": image_candidates,
        },
        "model_calls": [{"response": GROUND_TRUTH_SENTINEL}],
    }
    return build_ranking_context(
        entity_key="Q1",
        entity_label="Example Entity",
        missing_mode="both",
        existing_modalities={"text": "Existing public text", "image": None},
        knowledge={
            "entity_profile": {
                "core_identity": "Example Entity",
                "entity_type": "film",
                "confidence": "high",
                "visual_style": "poster art",
            },
            "relation_facts": [
                {
                    "relation": "genre",
                    "subject": "Example Entity",
                    "predicate": "genre",
                    "object": "film",
                    "fact": "Example Entity is a film.",
                    "visual_clue": "cinematic poster",
                    "support_level": "high",
                    "modality_relevance": ["text", "image"],
                    "relation_role": "primary",
                }
            ],
        },
        reasoning={
            "supported_facts": [
                {
                    "fact": "Example Entity is a film.",
                    "source_relations": ["genre"],
                    "support_level": "high",
                    "modality_relevance": ["text", "image"],
                    "relation_role": "primary",
                }
            ],
            "modality_constraints": {
                "text": {
                    "must_include": [{"constraint": "film", "confidence": "high"}],
                    "should_include": [{"constraint": "grounded", "confidence": "high"}],
                },
                "image": {
                    "must_include": [{"constraint": "poster", "confidence": "high"}],
                    "should_include": [
                        {"constraint": "cinematic", "confidence": "high"}
                    ],
                },
            },
            "forbidden_content": [
                {"instruction": INSTRUCTION_SENTINEL},
                GROUND_TRUTH_SENTINEL,
            ],
        },
        generation=generation,
        generated_image_inputs=image_inputs,
    )


def _text_only_context() -> Any:
    return build_ranking_context(
        entity_key="Q1",
        entity_label="Example Entity",
        missing_mode="text",
        existing_modalities={"text": None, "image": None},
        knowledge={
            "entity_profile": {
                "core_identity": "Example Entity",
                "entity_type": "film",
                "confidence": "high",
            },
            "relation_facts": [],
        },
        reasoning={
            "supported_facts": [],
            "modality_constraints": {
                "text": {
                    "must_include": [{"constraint": "film", "confidence": "high"}],
                }
            },
        },
        generation={
            "relation_alternation_generation": {
                "text_candidates": [_text_candidate("text_001", "Example film.")]
            }
        },
        generated_image_inputs={},
    )


def _both_context(
    *,
    text_candidates: list[dict[str, Any]] | None = None,
    image_candidates: list[dict[str, Any]] | None = None,
    image_inputs: dict[str, Any] | None = None,
    missing_mode: str = "both",
) -> Any:
    text_candidates = (
        [_text_candidate("text_001", "Example film.")]
        if text_candidates is None
        else text_candidates
    )
    image_candidates = (
        [_image_candidate("image_001")]
        if image_candidates is None
        else image_candidates
    )
    if image_inputs is None:
        image_inputs = {
            candidate["candidate_id"]: _image_transport(candidate["candidate_id"])
            for candidate in image_candidates
        }
    return build_ranking_context(
        entity_key="Q1",
        entity_label="Example Entity",
        missing_mode=missing_mode,
        existing_modalities={"text": None, "image": None},
        knowledge={
            "entity_profile": {
                "core_identity": "Example Entity",
                "entity_type": "film",
                "confidence": "high",
                "visual_style": "poster art",
            },
            "relation_facts": [],
        },
        reasoning={
            "supported_facts": [],
            "modality_constraints": {
                "text": {
                    "must_include": [{"constraint": "film", "confidence": "high"}],
                },
                "image": {
                    "must_include": [{"constraint": "poster", "confidence": "high"}],
                },
            },
        },
        generation={
            "relation_alternation_generation": {
                "text_candidates": text_candidates,
                "image_candidates": image_candidates,
            }
        },
        generated_image_inputs=image_inputs,
    )


def _image_input(candidate_id: str) -> ImageInput:
    return ImageInput.from_bytes(
        entity_key="Q1",
        candidate_id=candidate_id,
        image_bytes=_png_bytes(),
    )


def _image_transport(candidate_id: str) -> dict[str, Any]:
    del candidate_id
    return {"image_bytes": _png_bytes()}


def _png_bytes() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _text_candidate(candidate_id: str, value: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "target_modality": "text",
        "primary_relation": "genre",
        "expected_relation_coverage": ["genre"],
        "prompt_or_instruction": f"Do not leak {INSTRUCTION_SENTINEL}",
        "rationale": f"private {RATIONALE_SENTINEL}",
        "generated_content": {"type": "text", "value": value},
    }


def _image_candidate(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "target_modality": "image",
        "primary_relation": "genre",
        "expected_relation_coverage": ["genre"],
        "positive_prompt": f"Do not leak {INSTRUCTION_SENTINEL}",
        "rationale": f"private {RATIONALE_SENTINEL}",
        "generated_content": {"type": "image", "image_artifact_id": candidate_id},
    }


def test_engine_selects_model_fusion_top1_and_keeps_sentinel_out() -> None:
    sentence_bert = FakeSentenceBERTScorer(
        {
            "Legacy first text.": 0.2,
            "Good grounded text.": 0.7,
            "Best grounded text.": 0.99,
        }
    )
    clip = FakeVisualScorer(
        "clip",
        {
            "image_001": 0.3,
            "image_002": 0.65,
            "image_003": 0.67,
            "image_004": 0.99,
            "image_005": 0.2,
        },
    )
    blip = FakeVisualScorer(
        "blip",
        {
            "image_001": 0.3,
            "image_002": 0.65,
            "image_003": 0.67,
            "image_004": 0.99,
            "image_005": 0.2,
        },
    )
    judge = FakeGPT4OMiniJudge(
        text_scores={"text_001": 0.2, "text_002": 0.7, "text_003": 1.0, "text_004": 0.1},
        image_scores={
            "image_001": 0.3,
            "image_002": 0.65,
            "image_003": 0.67,
            "image_004": 1.0,
            "image_005": 0.2,
        },
        pair_scores={("text_003", "image_004"): 1.0},
        rejected_text_ids={"text_004"},
        rejected_image_ids={"image_005"},
    )
    engine = ModelFusionRankingEngine(
        profile=_profile(),
        scorers={
            "sentence_bert": sentence_bert,
            "clip": clip,
            "blip_itm": blip,
            "gpt4omini_judge": judge,
        },
    )

    result = engine.rank(_context())

    assert result["ranking_status"] == "completed"
    assert result["ranker"] == "model_fusion"
    assert result["score_version"] == "stage4_image_fusion_v4_no_visual_graph"
    assert result["profile_name"] == "test-profile"
    assert "calibration_profile" not in result
    assert result["required_scorers"] == [
        "blip_itm",
        "clip",
        "gpt4omini_judge",
        "sentence_bert",
    ]
    assert result["optional_scorers"] == []
    assert result["failed_optional_scorers"] == []
    assert result["candidate_score_weights"] == {
        "text": {
            "sentence_bert": 0.20,
            "structured_kg_alignment": 0.45,
            "judge": 0.35,
        },
        "image": {
            "clip": 0.20,
            "blip_itm": 0.30,
            "judge": 0.50,
        },
    }
    assert result["text_ranking"]["best_candidate_id"] == "text_003"
    assert result["image_ranking"]["best_candidate_id"] == "image_004"
    assert result["pair_ranking"] == {
        "ranked_pairs": [],
        "top_5_pairs": [],
        "best_pair_id": None,
        "best_pair": None,
    }
    assert (
        result["kg_aware_candidate_consistency_ranking"]["ground_truth_used"] is False
    )

    text_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in result["text_ranking"]["ranked_candidates"]
    }
    image_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in result["image_ranking"]["ranked_candidates"]
    }
    assert "legacy_score" not in text_by_id["text_001"]
    assert "legacy_score" not in image_by_id["image_001"]
    assert text_by_id["text_003"]["model_scores"]["sentence_bert"] == 0.99
    assert text_by_id["text_003"]["candidate_score"] == pytest.approx(
        text_by_id["text_003"]["model_scores"]["candidate_score"]
    )
    assert text_by_id["text_003"]["candidate_score"] == pytest.approx(
        0.20 * text_by_id["text_003"]["model_scores"]["sentence_bert"]
        + 0.45
        * text_by_id["text_003"]["model_scores"]["structured_kg_alignment"]
        + 0.35 * text_by_id["text_003"]["model_scores"]["gpt4omini_judge"]
    )
    assert image_by_id["image_004"]["candidate_score"] == pytest.approx(
        0.20 * image_by_id["image_004"]["model_scores"]["clip"]
        + 0.30 * image_by_id["image_004"]["model_scores"]["blip_itm"]
        + 0.50 * image_by_id["image_004"]["model_scores"]["gpt4omini_judge"]
    )
    assert "quality" not in text_by_id["text_003"]["model_scores"]
    assert "kg_consistency" not in text_by_id["text_003"]
    assert text_by_id["text_003"]["hard_gate"]["decision"] == "pass"
    assert text_by_id["text_003"]["soft_violations"] == ["missing_should_include"]
    assert "penalty" not in text_by_id["text_003"]
    assert text_by_id["text_003"]["fused_score"] == text_by_id["text_003"]["final_score"]
    assert text_by_id["text_003"]["score"] == text_by_id["text_003"]["final_score"]
    assert text_by_id["text_003"]["ranking_reason"]
    assert text_by_id["text_004"]["hard_gate"]["decision"] == "reject"

    judge_calls = [
        call
        for call in result["ranking_model_calls"]
        if call["scorer"] == "gpt4omini_judge"
    ]
    assert judge_calls
    assert {call["kind"] for call in judge_calls} <= {"text", "image"}
    assert all("raw_score" in call for call in result["ranking_model_calls"])
    assert all("normalized_score" in call for call in result["ranking_model_calls"])
    assert all("details" in call for call in result["ranking_model_calls"])
    assert all("gate" in call for call in judge_calls)
    assert judge_calls[0]["gate"]["decision"] in {"pass", "reject"}
    assert clip.text_calls == []
    assert blip.text_calls == []

    serialized_calls = repr(
        {
            "sentence_bert": sentence_bert.calls,
            "clip_quality": clip.quality_calls,
            "clip_text": clip.text_calls,
            "blip_quality": blip.quality_calls,
            "blip_text": blip.text_calls,
            "judge": judge.calls,
            "model_calls": result["ranking_model_calls"],
        }
    )
    for sentinel in (
        INSTRUCTION_SENTINEL,
        RATIONALE_SENTINEL,
        GROUND_TRUTH_SENTINEL,
        "prompt_or_instruction",
        "positive_prompt",
        "rationale",
    ):
        assert sentinel not in serialized_calls


def test_engine_passes_image_input_to_gpt4omini_image_judge_only() -> None:
    judge = FakeGPT4OMiniJudge(
        text_scores={"text_001": 0.8, "text_002": 0.7, "text_003": 0.6},
        image_scores={
            "image_001": 0.8,
            "image_002": 0.7,
            "image_003": 0.6,
            "image_004": 0.5,
            "image_005": 0.4,
        },
        pair_scores={},
    )
    engine = _engine(gpt4omini_judge=judge)

    engine.rank(_both_context())

    image_calls = [call for call in judge.calls if call["kind"] == "image"]
    pair_calls = [call for call in judge.calls if call["kind"] == "pair"]

    assert image_calls
    assert pair_calls == []
    for call in image_calls:
        image_input = call["image_input"]
        assert isinstance(image_input, ImageInput)
        assert image_input.candidate_id == call["candidate"]["candidate_id"]
        assert image_input.as_data_url().startswith("data:image/png;base64,")
def test_missing_generated_image_input_rejects_without_visual_scorer_call() -> None:
    clip = FakeVisualScorer("clip", {"image_001": 0.9})
    engine = _engine(
        clip=clip,
        blip_itm=FakeVisualScorer("blip", {"image_001": 0.9}),
        gpt4omini_judge=FakeGPT4OMiniJudge(
            text_scores={"text_001": 0.8},
            image_scores={"image_001": 0.8},
            pair_scores={},
        ),
        sentence_bert=FakeSentenceBERTScorer({"Example film.": 0.8}),
    )

    result = engine.rank(_both_context(image_inputs={}))

    assert result["image_ranking"]["best_candidate_id"] is None
    assert result["image_ranking"]["ranked_candidates"][0]["hard_gate"] == {
        "decision": "reject",
        "source": "deterministic",
        "hard_violations": ["missing_image_input"],
        "soft_violations": [],
        "judge_confidence": 0.0,
        "evidence": [],
        "reason": "generated ImageInput is required for image scoring",
    }
    assert clip.quality_calls == []
    assert result["ranking_status"] == "completed"



def test_visual_scores_use_raw_zero_to_one_normalization() -> None:
    context = _both_context(
        missing_mode="image",
        text_candidates=[],
        image_candidates=[_image_candidate("image_001")],
        image_inputs={"image_001": _image_transport("image_001")},
    )
    result = _engine(
        clip=FakeVisualScorer("clip", {"image_001": 0.9}),
        blip_itm=FakeVisualScorer("blip", {"image_001": 0.6}),
        gpt4omini_judge=FakeGPT4OMiniJudge(
            text_scores={},
            image_scores={"image_001": 0.1},
            pair_scores={},
        ),
    ).rank(context)

    assert result["score_version"] == "stage4_image_fusion_v4_no_visual_graph"
    assert "calibration_profile" not in result
    scores = result["image_ranking"]["ranked_candidates"][0]["model_scores"]
    assert scores["clip_raw_score"] == 90.0
    assert scores["clip_normalized_score"] == 0.9
    assert scores["clip"] == pytest.approx(0.9)
    assert scores["blip_itm"] == pytest.approx(0.6)
    assert scores["gpt4omini_judge"] == pytest.approx(0.1)


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_required_scorer_failed_or_skipped_raises(status: str) -> None:
    engine = _engine(sentence_bert=StatusTextScorer(status))

    with pytest.raises(RequiredScorerError, match="sentence_bert"):
        engine.rank(_text_only_context())


def test_required_scorer_wrong_count_raises() -> None:
    engine = _engine(sentence_bert=BadCountTextScorer())

    with pytest.raises(RequiredScorerError, match="sentence_bert"):
        engine.rank(_text_only_context())


def test_optional_text_scorer_failure_degrades_and_uses_fallback_scores() -> None:
    engine = _engine(
        profile=_profile(sentence_bert_required=False),
        sentence_bert=StatusTextScorer("failed"),
    )

    result = engine.rank(_text_only_context())

    assert result["ranking_status"] == "degraded"
    assert result["required_scorers"] == ["gpt4omini_judge"]
    assert result["optional_scorers"] == ["sentence_bert"]
    assert result["failed_optional_scorers"] == ["sentence_bert"]
    assert result["text_ranking"]["best_candidate_id"] == "text_001"
    ranked = result["text_ranking"]["ranked_candidates"][0]
    assert "sentence_bert" not in ranked["model_scores"]
    assert ranked["score"] == ranked["final_score"]

    sentence_calls = [
        call
        for call in result["ranking_model_calls"]
        if call["scorer"] == "sentence_bert"
    ]
    assert sentence_calls == [
        {
            "scorer": "sentence_bert",
            "kind": "text",
            "candidate_id": "text_001",
            "status": "optional_failed",
            "raw_score": None,
            "normalized_score": None,
            "model_name": "bad-status-text",
            "model_version": "",
            "latency_ms": 0,
            "details": {},
        }
    ]
    assert GROUND_TRUTH_SENTINEL not in repr(sentence_calls)


def test_optional_bertscore_records_auxiliary_metric_without_changing_model_scores() -> None:
    bertscore = FakeBERTScoreScorer({"Example film.": (0.91, 0.92, 0.93)})
    engine = _engine(
        profile=_profile(bertscore_enabled=True),
        bertscore=bertscore,
    )

    result = engine.rank(_text_only_context())

    assert result["ranking_status"] == "completed"
    assert result["optional_scorers"] == ["bertscore"]
    ranked = result["text_ranking"]["ranked_candidates"][0]
    assert "bertscore" not in ranked["model_scores"]
    assert ranked["auxiliary_metrics"]["bertscore"] == {
        "precision": 0.91,
        "recall": 0.92,
        "f1": 0.93,
        "normalized_score": 0.93,
        "reference_kind": "kg_reference",
        "model_name": "fake-bertscore",
        "model_version": "fake-roberta-large",
    }
    bertscore_calls = [
        call
        for call in result["ranking_model_calls"]
        if call["scorer"] == "bertscore"
    ]
    assert bertscore_calls == [
        {
            "scorer": "bertscore",
            "kind": "text",
            "candidate_id": "text_001",
            "status": "scored",
            "raw_score": 0.93,
            "normalized_score": 0.93,
            "model_name": "fake-bertscore",
            "model_version": "fake-roberta-large",
            "latency_ms": 0,
            "details": {
                "candidate_index": 0,
                "precision": 0.91,
                "recall": 0.92,
                "f1": 0.93,
                "reference_kind": "kg_reference",
            },
        }
    ]
    assert bertscore.calls[0]["candidates"] == ["Example film."]


def test_optional_bertscore_failure_degrades_without_blocking_ranking() -> None:
    engine = _engine(
        profile=_profile(bertscore_enabled=True),
        bertscore=StatusTextScorer("failed"),
    )

    result = engine.rank(_text_only_context())

    assert result["ranking_status"] == "degraded"
    assert result["optional_scorers"] == ["bertscore"]
    assert result["failed_optional_scorers"] == ["bertscore"]
    ranked = result["text_ranking"]["ranked_candidates"][0]
    assert "bertscore" not in ranked["model_scores"]
    assert "bertscore" not in ranked["auxiliary_metrics"]


def test_all_rejected_candidates_do_not_publish_best_candidate() -> None:
    engine = _engine(
        gpt4omini_judge=FakeGPT4OMiniJudge(
            text_scores={"text_001": 0.1},
            image_scores={"image_001": 0.1},
            pair_scores={},
            rejected_text_ids={"text_001"},
            rejected_image_ids={"image_001"},
        )
    )

    result = engine.rank(_both_context())

    assert result["ranking_status"] == "no_valid_candidate"
    assert result["text_ranking"]["best_candidate_id"] is None
    assert result["image_ranking"]["best_candidate_id"] is None


def test_both_mode_ranks_surviving_modality_without_pair_requirement() -> None:
    engine = _engine()

    result = engine.rank(_both_context(image_candidates=[], image_inputs={}))

    assert result["ranking_status"] == "completed"
    assert result["text_ranking"]["best_candidate_id"] == "text_001"
    assert result["image_ranking"]["best_candidate_id"] is None
    assert result["pair_ranking"]["ranked_pairs"] == []


def test_required_gpt4omini_candidate_parse_error_rejects_candidate_and_continues() -> None:
    image_candidates = [_image_candidate("image_001"), _image_candidate("image_002")]
    engine = _engine(
        clip=FakeVisualScorer("clip", {"image_001": 0.8, "image_002": 0.7}),
        blip_itm=FakeVisualScorer("blip", {"image_001": 0.8, "image_002": 0.7}),
        gpt4omini_judge=FakeGPT4OMiniJudge(
            text_scores={"text_001": 0.8},
            image_scores={"image_001": 0.8, "image_002": 0.7},
            pair_scores={("text_001", "image_002"): 0.8},
            malformed_image_ids={"image_001"},
        ),
    )

    result = engine.rank(_both_context(image_candidates=image_candidates))

    assert result["ranking_status"] == "completed"
    assert result["image_ranking"]["best_candidate_id"] == "image_002"
    failed_calls = [
        call
        for call in result["ranking_model_calls"]
        if call["scorer"] == "gpt4omini_judge" and call["status"] == "failed"
    ]
    assert failed_calls == [
        {
            "scorer": "gpt4omini_judge",
            "kind": "image",
            "candidate_id": "image_001",
            "status": "failed",
            "raw_score": None,
            "normalized_score": None,
            "model_name": "fake-gpt4omini-judge",
            "model_version": "",
            "latency_ms": 0,
            "details": {
                "error_type": "GPT4OMiniJudgeParseError",
                "contract_repair_failed": True,
            },
        }
    ]


def test_required_gpt4omini_candidate_service_error_rejects_candidate_and_continues() -> None:
    engine = _engine(gpt4omini_judge=FailingGPT4OMiniJudge())

    result = engine.rank(_text_only_context())

    assert result["ranking_status"] == "no_valid_candidate"
    assert result["text_ranking"]["best_candidate_id"] is None
    rejected = result["text_ranking"]["rejected_candidates"]
    assert rejected[0]["hard_gate"]["hard_violations"] == ["judge_service_error"]
    failed_calls = [
        call
        for call in result["ranking_model_calls"]
        if call["scorer"] == "gpt4omini_judge" and call["status"] == "failed"
    ]
    assert failed_calls == [
        {
            "scorer": "gpt4omini_judge",
            "kind": "text",
            "candidate_id": "text_001",
            "status": "failed",
            "raw_score": None,
            "normalized_score": None,
            "model_name": "failing-qwen-judge",
            "model_version": "",
            "latency_ms": 0,
            "details": {
                "error_type": "RuntimeError",
                "error": "judge service unavailable",
            },
        }
    ]


def _engine(
    *,
    profile: Stage4Profile | None = None,
    **overrides: Any,
) -> ModelFusionRankingEngine:
    scorers = {
        "sentence_bert": FakeSentenceBERTScorer({"Example film.": 0.8}),
        "clip": FakeVisualScorer("clip", {"image_001": 0.8}),
        "blip_itm": FakeVisualScorer("blip", {"image_001": 0.8}),
        "gpt4omini_judge": FakeGPT4OMiniJudge(
            text_scores={"text_001": 0.8},
            image_scores={"image_001": 0.8},
            pair_scores={("text_001", "image_001"): 0.8},
        ),
    }
    scorers.update(overrides)
    return ModelFusionRankingEngine(
        profile=profile or _profile(),
        scorers=scorers,
    )
