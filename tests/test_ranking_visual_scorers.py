from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from mmkgc.ranking.context import (
    build_ranking_context,
    build_visual_constraint_text,
)
from mmkgc.ranking.contracts import ScoreResult
from mmkgc.ranking.image_input import ImageInput
from mmkgc.ranking.scorers import BLIPITMRankingScorer, CLIPRankingScorer


GROUND_TRUTH_SENTINEL = "ZZZ_SECRET_GROUND_TRUTH"
INSTRUCTION_SENTINEL = "ZZZ_SECRET_INSTRUCTION"


class FakeVisualEvaluator:
    def __init__(self, scores: list[float], embedding: list[float] | None = None) -> None:
        self.scores = scores
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.pil_calls: list[dict[str, object]] = []
        self.path_calls: list[dict[str, object]] = []
        self.embedding_calls: list[dict[str, object]] = []

    def score_pil_texts(self, image: object, texts: list[str]) -> list[float]:
        self.pil_calls.append({"mode": image.mode, "size": image.size, "texts": list(texts)})
        return list(self.scores)

    def encode_pil_image(self, image: object) -> list[float]:
        self.embedding_calls.append({"mode": image.mode, "size": image.size})
        return list(self.embedding)

    def score_image_texts(self, image_path: object, texts: list[str]) -> list[float]:
        self.path_calls.append({"image_path": image_path, "texts": list(texts)})
        raise AssertionError("visual ranking scorers must use PIL hooks")


class TrackingImage:
    mode = "RGB"
    size = (4, 4)

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_clip_and_blip_use_pil_hooks_without_prompt_leakage() -> None:
    context = _build_context()
    image_input = _image_input()
    prompt = build_visual_constraint_text(context, context.image_candidates[0])

    assert GROUND_TRUTH_SENTINEL not in prompt
    assert INSTRUCTION_SENTINEL not in prompt
    assert "prompt_or_instruction" not in prompt

    clip_evaluator = FakeVisualEvaluator([120.0, 50.0])
    blip_evaluator = FakeVisualEvaluator([-5.0, 75.0])

    clip_results = CLIPRankingScorer(clip_evaluator).score_image_quality_batch(
        image_input,
        [prompt, "entity type: film"],
    )
    blip_results = BLIPITMRankingScorer(blip_evaluator).score_image_text_batch(
        image_input,
        [prompt, "entity type: film"],
    )

    assert clip_evaluator.path_calls == []
    assert blip_evaluator.path_calls == []
    assert clip_evaluator.pil_calls == [
        {"mode": "RGB", "size": (4, 4), "texts": [prompt, "entity type: film"]}
    ]
    assert blip_evaluator.pil_calls == [
        {"mode": "RGB", "size": (4, 4), "texts": [prompt, "entity type: film"]}
    ]
    assert clip_results == [
        ScoreResult(
            raw_score=120.0,
            normalized_score=1.0,
            status="scored",
            model_name="clip-ranking",
            details={"candidate_index": 0},
        ),
        ScoreResult(
            raw_score=50.0,
            normalized_score=0.5,
            status="scored",
            model_name="clip-ranking",
            details={"candidate_index": 1},
        ),
    ]
    assert blip_results == [
        ScoreResult(
            raw_score=-5.0,
            normalized_score=0.0,
            status="scored",
            model_name="blip-itm-ranking",
            details={"candidate_index": 0},
        ),
        ScoreResult(
            raw_score=75.0,
            normalized_score=0.75,
            status="scored",
            model_name="blip-itm-ranking",
            details={"candidate_index": 1},
        ),
    ]


def test_visual_scorers_batch_empty_texts_without_opening_image() -> None:
    image_input = _image_input()
    evaluator = FakeVisualEvaluator([])

    assert CLIPRankingScorer(evaluator).score_image_text_batch(image_input, []) == []
    assert BLIPITMRankingScorer(evaluator).score_image_quality_batch(image_input, []) == []
    assert evaluator.pil_calls == []


def test_clip_image_embeddings_use_in_memory_image_input() -> None:
    image_input = _image_input()
    evaluator = FakeVisualEvaluator([], embedding=[0.25, 0.5, 0.75])

    assert CLIPRankingScorer(evaluator).image_embeddings(image_input) == [0.25, 0.5, 0.75]
    assert evaluator.embedding_calls == [{"mode": "RGB", "size": (4, 4)}]


def test_clip_and_blip_close_pil_images_after_scoring(monkeypatch) -> None:
    clip_image = TrackingImage()
    blip_image = TrackingImage()
    images = [clip_image, blip_image]

    def fake_open_pil(self: ImageInput) -> TrackingImage:
        return images.pop(0)

    monkeypatch.setattr(ImageInput, "open_pil", fake_open_pil)
    image_input = _image_input()

    CLIPRankingScorer(FakeVisualEvaluator([25.0])).score_image_text_batch(
        image_input,
        ["first"],
    )
    BLIPITMRankingScorer(FakeVisualEvaluator([50.0])).score_image_quality_batch(
        image_input,
        ["second"],
    )

    assert clip_image.closed is True
    assert blip_image.closed is True


def test_clip_closes_pil_image_after_embedding(monkeypatch) -> None:
    tracking_image = TrackingImage()

    monkeypatch.setattr(ImageInput, "open_pil", lambda self: tracking_image)

    assert CLIPRankingScorer(FakeVisualEvaluator([])).image_embeddings(_image_input()) == [
        0.1,
        0.2,
        0.3,
    ]
    assert tracking_image.closed is True


def test_visual_scorers_reject_wrong_score_count() -> None:
    image_input = _image_input()
    evaluator = FakeVisualEvaluator([12.0])

    with pytest.raises(ValueError, match="returned 1 scores for 2 texts"):
        CLIPRankingScorer(evaluator).score_image_quality_batch(
            image_input,
            ["first", "second"],
        )


@pytest.mark.parametrize(
    ("scorer_factory", "method_name"),
    [
        (CLIPRankingScorer, "score_image_quality_batch"),
        (BLIPITMRankingScorer, "score_image_text_batch"),
    ],
)
@pytest.mark.parametrize("raw_score", [float("nan"), float("inf"), float("-inf")])
def test_visual_scorers_reject_non_finite_raw_scores(
    scorer_factory,
    method_name: str,
    raw_score: float,
) -> None:
    scorer = scorer_factory(FakeVisualEvaluator([raw_score]))
    method = getattr(scorer, method_name)

    with pytest.raises(ValueError, match="raw_score must be a finite numeric value"):
        method(_image_input(), ["text"])


@pytest.mark.parametrize("embedding", [[float("nan")], [float("inf")], [float("-inf")]])
def test_clip_image_embeddings_reject_non_finite_values(embedding: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        CLIPRankingScorer(FakeVisualEvaluator([], embedding=embedding)).image_embeddings(
            _image_input()
        )


def _image_input() -> ImageInput:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    return ImageInput.from_bytes(
        entity_key="Q42",
        candidate_id="image_001",
        image_bytes=buffer.getvalue(),
    )


def _build_context():
    return build_ranking_context(
        entity_key="Q42",
        entity_label="Example Film",
        missing_mode="both",
        existing_modalities={"text": None, "image": None},
        knowledge={
            "entity_profile": {
                "entity_key": "Q42",
                "entity_label": "Example Film",
                "entity_type": "film",
                "confidence": "high",
            },
            "relation_facts": [],
        },
        reasoning={
            "supported_facts": [],
            "safe_inferences": [],
            "forbidden_content": [GROUND_TRUTH_SENTINEL],
            "modality_constraints": {
                "image": {
                    "must_include": [
                        {
                            "constraint": "cinematic poster",
                            "confidence": "high",
                            "safe": True,
                        }
                    ]
                }
            },
        },
        generation={
            "relation_alternation_generation": {
                "text_candidates": [],
                "image_candidates": [
                    {
                        "candidate_id": "image_001",
                        "target_modality": "image",
                        "prompt_or_instruction": INSTRUCTION_SENTINEL,
                        "ground_truth": GROUND_TRUTH_SENTINEL,
                        "generated_content": {
                            "type": "image",
                            "image_path": "generated_images/image_001.png",
                        },
                    }
                ],
                "model_calls": [{"response": GROUND_TRUTH_SENTINEL}],
            }
        },
        generated_image_inputs={},
    )
