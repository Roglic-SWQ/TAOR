from __future__ import annotations

from pathlib import Path

import pytest

from mmkgc.evaluation.gemini_embedding import (
    FixedGeminiEmbeddingClient,
    GeminiEmbeddingError,
    cosine_score,
    evaluate_entity_gemini_embedding,
    load_gemini_embedding_config,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(64, 128, 192)).save(path)


def _artifacts(entity_dir: Path) -> RunEntityArtifacts:
    _write_png(entity_dir / "generated_images" / "image_001.png")
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key="Q42",
        knowledge={
            "entity_profile": {
                "entity_label": "Ada Lovelace",
                "entity_type": "person",
            },
            "relation_facts": [
                {
                    "relation_label": "occupation",
                    "neighbor_label": "mathematician",
                }
            ],
        },
        reasoning={
            "modality_constraints": {
                "image": {
                    "must_include": ["Victorian portrait"],
                }
            }
        },
        ranking={
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Ada Lovelace was a mathematician."
                        },
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image_001",
                        "positive_prompt": "portrait of Ada Lovelace",
                        "generated_content": {
                            "type": "image",
                            "value": "fallback prompt",
                            "image_path": "generated_images/image_001.png",
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "pair_id": "pair_001",
                        "text_candidate_id": "text_001",
                        "image_candidate": {"candidate_id": "image_001"},
                    }
                ]
            },
        },
        final={
            "entity": {
                "entity_key": "Q42",
                "label": "Ada Lovelace",
                "description": "Ada Lovelace was an English mathematician.",
            },
            "missing_mode": "both",
        },
    )


def test_load_gemini_embedding_config_rejects_path_outside_project_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    outside = tmp_path / "gemini-embedding.yaml"
    outside.write_text(
        "gemini:\n"
        "  api_key_env: GEMINI_API_KEY\n"
        "  model_name: gemini-embedding-2\n",
        encoding="utf-8",
    )

    with pytest.raises(GeminiEmbeddingError, match="must be under project root"):
        load_gemini_embedding_config(outside, project_root=project_root)


def test_load_gemini_embedding_config_requires_api_key_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    project_root = tmp_path / "MMKGC"
    config_path = project_root / "configs" / "gemini-embedding.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "gemini:\n"
        "  api_key_env: GEMINI_API_KEY\n"
        "  model_name: gemini-embedding-2\n",
        encoding="utf-8",
    )

    with pytest.raises(GeminiEmbeddingError, match="environment variable"):
        load_gemini_embedding_config(config_path, project_root=project_root)


def test_load_gemini_embedding_config_reads_safe_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    project_root = tmp_path / "MMKGC"
    config_path = project_root / "configs" / "gemini-embedding.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "gemini:\n"
        "  api_key_env: GEMINI_API_KEY\n"
        "  base_url: https://example.test/v1beta\n"
        "  model_name: gemini-embedding-2\n"
        "  output_dimensionality: 128\n"
        "  timeout_seconds: 30\n"
        "  max_retries: 3\n",
        encoding="utf-8",
    )

    config = load_gemini_embedding_config(config_path, project_root=project_root)

    assert config.api_key == "secret-value"
    assert config.api_key_env == "GEMINI_API_KEY"
    assert config.model_name == "gemini-embedding-2"
    assert config.output_dimensionality == 128
    assert config.timeout_seconds == 30
    assert config.max_retries == 3


def test_load_gemini_embedding_config_wraps_malformed_numeric_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    project_root = tmp_path / "MMKGC"
    config_path = project_root / "configs" / "gemini-embedding.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "gemini:\n"
        "  api_key_env: GEMINI_API_KEY\n"
        "  model_name: gemini-embedding-2\n"
        "  output_dimensionality: not-an-int\n",
        encoding="utf-8",
    )

    with pytest.raises(GeminiEmbeddingError, match="output_dimensionality"):
        load_gemini_embedding_config(config_path, project_root=project_root)


def test_fixed_gemini_embedding_client_returns_deterministic_vectors(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    _write_png(image_path)
    client = FixedGeminiEmbeddingClient(
        {
            "text:hello": [1.0, 2.0],
            f"image:{image_path.resolve()}": [3.0, 4.0],
        }
    )

    assert client.embed_text("hello") == [1.0, 2.0]
    assert client.embed_image(image_path) == [3.0, 4.0]


def test_cosine_score_maps_identical_to_100_and_orthogonal_to_0() -> None:
    assert cosine_score([1.0, 0.0], [1.0, 0.0]) == 100.0
    assert cosine_score([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_evaluate_entity_gemini_embedding_outputs_text_image_and_pair_scores(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path / "Q42")
    image_path = artifacts.entity_dir / "generated_images" / "image_001.png"
    client = FixedGeminiEmbeddingClient(
        {
            "text:Ada Lovelace was a mathematician.": [1.0, 0.0],
            "text:Ada Lovelace was an English mathematician.": [1.0, 0.0],
            "text:portrait of Ada Lovelace": [0.0, 1.0],
            "text:Entity: Ada Lovelace. Entity key: Q42. Type: person. "
            "Evidence: occupation: mathematician. Visual requirements: "
            "Victorian portrait": [1.0, 1.0],
            f"image:{image_path.resolve()}": [0.0, 1.0],
        }
    )

    result = evaluate_entity_gemini_embedding(
        artifacts,
        client=client,
        model_name="gemini-embedding-2",
    )

    assert result["entity"]["entity_key"] == "Q42"
    assert result["gemini_model_name"] == "gemini-embedding-2"
    assert result["text_count"] == 1
    assert result["image_count"] == 1
    assert result["pair_count"] == 1

    text_score = result["text_scores"][0]
    assert text_score["entity_key"] == "Q42"
    assert text_score["text_candidate_id"] == "text_001"
    assert text_score["gemini_text_ground_truth_similarity"] == 100.0
    assert text_score["gemini_text_evidence_similarity"] == pytest.approx(70.710678)
    assert text_score["gemini_overall_text_embedding_score"] == pytest.approx(
        85.355339
    )

    image_score = result["image_scores"][0]
    assert image_score["image_candidate_id"] == "image_001"
    assert image_score["gemini_prompt_similarity"] == 100.0
    assert image_score["gemini_text_similarity"] == 0.0
    assert image_score["gemini_evidence_similarity"] == pytest.approx(70.710678)
    assert image_score["gemini_overall_multimodal_score"] == pytest.approx(56.903559)

    pair_score = result["pair_scores"][0]
    assert pair_score["pair_id"] == "pair_001"
    assert pair_score["gemini_pair_image_text_similarity"] == 0.0
    assert pair_score["gemini_pair_image_evidence_similarity"] == pytest.approx(
        70.710678
    )
    assert pair_score["gemini_pair_text_evidence_similarity"] == pytest.approx(
        70.710678
    )
    assert pair_score["gemini_overall_pair_embedding_score"] == pytest.approx(
        47.140452
    )
