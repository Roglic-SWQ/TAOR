from __future__ import annotations

import pytest

from mmkgc.ranking.graph_alignment import (
    structured_kg_alignment,
    visual_graph_alignment,
)


def test_structured_kg_alignment_matches_triples_one_to_one() -> None:
    target = [
        {
            "subject": "Example Film",
            "predicate": "director",
            "object": "Jane Example",
            "polarity": "positive",
        }
    ]
    candidate = [
        {
            "subject": "Example Film",
            "predicate": "director",
            "object": "Jane Example",
            "polarity": "positive",
            "confidence": 1.0,
        }
    ]

    result = structured_kg_alignment(target, candidate)

    assert result["score"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert len(result["matches"]) == 1


def test_structured_kg_alignment_preserves_direction_and_polarity() -> None:
    target = [
        {
            "subject": "Film",
            "predicate": "director",
            "object": "Person",
            "polarity": "positive",
        }
    ]
    reversed_candidate = [
        {
            "subject": "Person",
            "predicate": "director",
            "object": "Film",
            "polarity": "positive",
            "confidence": 1.0,
        }
    ]
    negative_candidate = [
        {
            "subject": "Film",
            "predicate": "director",
            "object": "Person",
            "polarity": "negative",
            "confidence": 1.0,
        }
    ]

    assert structured_kg_alignment(target, reversed_candidate)["score"] == pytest.approx(0.4)
    assert structured_kg_alignment(target, negative_candidate)["score"] == 0.0


def test_visual_graph_alignment_renormalizes_only_present_target_components() -> None:
    target = {
        "nodes": [{"id": "scene", "label": "courtroom", "weight": 1.0}],
        "attributes": [{"text": "black robe"}],
        "edges": [],
        "ocr": [],
    }
    observed = {
        "nodes": [
            {
                "id": "scene_1",
                "label": "courtroom",
                "attributes": ["black robe"],
                "confidence": 1.0,
            }
        ],
        "edges": [],
        "ocr": [],
    }

    result = visual_graph_alignment(target, observed)

    assert result["score"] == pytest.approx(0.931186)
    assert result["component_weights"] == {
        "nodes": pytest.approx(0.625),
        "attributes": pytest.approx(0.375),
    }
    assert result["components"]["edges"]["target_count"] == 0
    assert result["components"]["ocr"]["target_count"] == 0


def test_visual_graph_alignment_normalizes_visual_wording() -> None:
    target = {
        "nodes": [{"id": "scene", "label": "place"}],
        "attributes": [
            {"text": "location establishing view"},
            {"text": "environmental context"},
        ],
        "edges": [],
        "ocr": [],
    }
    observed = {
        "nodes": [
            {
                "id": "scene_1",
                "label": "tropical landscape",
                "attributes": ["lush greenery"],
                "confidence": 1.0,
            }
        ],
        "edges": [],
        "ocr": [],
    }

    result = visual_graph_alignment(target, observed)

    assert result["components"]["nodes"]["f1"] > 0.0
    assert result["components"]["attributes"]["f1"] > 0.0
    assert result["score"] > 0.0
