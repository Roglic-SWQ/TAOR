from __future__ import annotations

import pytest

from mmkgc.ranking.fusion import (
    fuse_image_candidate_score,
    fuse_text_candidate_score,
    normalize_cosine,
    normalize_judge_1_to_5,
    normalize_visual_quality,
)

def test_normalization_formulas_are_stable() -> None:
    assert normalize_cosine(-1.0) == 0.0
    assert normalize_cosine(0.0) == 0.5
    assert normalize_cosine(1.0) == 1.0
    assert normalize_judge_1_to_5(3.0) == 0.5
    assert normalize_visual_quality(0.0) == 0.0
    assert normalize_visual_quality(60.0) == 0.6
    assert normalize_visual_quality(100.0) == 1.0
    assert normalize_visual_quality(120.0) == 1.0



def test_flat_candidate_score_formulas_are_exact_and_clamped() -> None:
    assert fuse_text_candidate_score(
        sentence_bert=0.8,
        structured_kg_alignment=0.7,
        judge=0.9,
    ) == pytest.approx(0.79)
    assert fuse_image_candidate_score(
        clip=0.8,
        blip=0.7,
        judge=0.9,
    ) == pytest.approx(0.82)
    assert fuse_text_candidate_score(
        sentence_bert=1.0,
        structured_kg_alignment=1.0,
        judge=1.0,
    ) == 1.0



@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fusion_rejects_non_finite_inputs(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_cosine(value)

    with pytest.raises(ValueError, match="finite"):
        normalize_visual_quality(value)

    with pytest.raises(ValueError, match="finite"):
        fuse_text_candidate_score(
            sentence_bert=value,
            structured_kg_alignment=1.0,
            judge=1.0,
        )
