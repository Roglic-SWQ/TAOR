from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from mmkgc.evaluation.image_quality import (
    ClipImageQualityError,
    ClipLocalConfig,
    ClipLocalImageQualityEvaluator,
    FixedImageTextSimilarityScorer,
    build_entity_evidence_summary,
    clip_weight_filename,
    evaluate_entity_image_quality,
    validate_clip_model_dir,
    validate_clip_weight_path,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


class FakeClipModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> None:
        self.eval_calls += 1


class FakeClipModule:
    def __init__(self) -> None:
        self.load_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.model = FakeClipModel()
        self.preprocess = object()

    def load(self, *args: object, **kwargs: object) -> tuple[FakeClipModel, object]:
        self.load_calls.append((args, kwargs))
        return self.model, self.preprocess


def test_clip_local_evaluator_score_image_texts_delegates_to_pil_hook(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)
    (model_dir / "ViT-B-32.pt").write_bytes(b"fake")
    image_path = tmp_path / "image.png"
    _write_png(image_path)

    class FakeEvaluator(ClipLocalImageQualityEvaluator):
        def score_pil_texts(self, image: object, texts: list[str]) -> list[float]:
            self.pil_call = {"mode": image.mode, "size": image.size, "texts": texts}
            return [12.5, 25.0]

    evaluator = FakeEvaluator(
        ClipLocalConfig(model_dir="CLIP", project_root=project_root)
    )

    assert evaluator.score_image_texts(image_path, ["prompt", "caption"]) == [
        12.5,
        25.0,
    ]
    assert evaluator.pil_call == {
        "mode": "RGB",
        "size": (4, 4),
        "texts": ["prompt", "caption"],
    }


def test_clip_local_evaluator_encode_pil_image_uses_injected_model_without_loading(
    tmp_path: Path,
) -> None:
    import torch

    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)
    (model_dir / "ViT-B-32.pt").write_bytes(b"fake")
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    class FakePreprocess:
        def __call__(self, pil_image: object) -> object:
            self.image_mode = pil_image.mode
            return torch.tensor([1.0, 2.0])

    class FakeEncoded:
        def float(self) -> torch.Tensor:
            return torch.tensor([[3.0, 4.0]])

    class FakeModel(FakeClipModel):
        def __init__(self) -> None:
            super().__init__()
            self.image_inputs: list[object] = []

        def encode_image(self, image_input: object) -> FakeEncoded:
            self.image_inputs.append(image_input)
            return FakeEncoded()

    fake_clip = FakeClipModule()
    fake_model = FakeModel()
    fake_preprocess = FakePreprocess()
    evaluator = ClipLocalImageQualityEvaluator(
        ClipLocalConfig(model_dir="CLIP", device="cpu", project_root=project_root),
        clip_module=fake_clip,
        model=fake_model,
        preprocess=fake_preprocess,
    )

    assert evaluator.encode_pil_image(image) == [0.6, 0.8]
    assert fake_preprocess.image_mode == "RGB"
    assert fake_clip.load_calls == []


def test_clip_weight_filename_maps_model_names() -> None:
    assert clip_weight_filename("ViT-B/32") == "ViT-B-32.pt"
    assert clip_weight_filename("ViT-L/14") == "ViT-L-14.pt"


def test_clip_weight_filename_rejects_unknown_model() -> None:
    with pytest.raises(ClipImageQualityError, match="unsupported clip_model_name"):
        clip_weight_filename("RN50")


def test_validate_clip_model_dir_accepts_project_local_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)

    resolved = validate_clip_model_dir("CLIP", project_root=project_root)

    assert resolved == model_dir.resolve()


def test_validate_clip_model_dir_rejects_external_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    external = tmp_path / "outside" / "CLIP"
    external.mkdir(parents=True)

    with pytest.raises(ClipImageQualityError, match="clip_model_dir must be under"):
        validate_clip_model_dir(external, project_root=project_root)


def test_validate_clip_model_dir_rejects_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = project_root / "CLIP"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ClipImageQualityError, match="clip_model_dir must be under"):
        validate_clip_model_dir("CLIP", project_root=project_root)


def test_validate_clip_weight_path_requires_expected_file(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)

    with pytest.raises(ClipImageQualityError, match="missing CLIP weight"):
        validate_clip_weight_path(model_dir, "ViT-B/32")

    weight = model_dir / "ViT-B-32.pt"
    weight.write_bytes(b"fake")

    assert validate_clip_weight_path(model_dir, "ViT-B/32") == weight.resolve()


def test_validate_clip_weight_path_rejects_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_weight = outside / "ViT-B-32.pt"
    outside_weight.write_bytes(b"fake")
    link = model_dir / "ViT-B-32.pt"
    link.symlink_to(outside_weight)

    with pytest.raises(ClipImageQualityError, match="CLIP weight must stay under"):
        validate_clip_weight_path(model_dir, "ViT-B/32")


def test_clip_local_evaluator_accepts_injected_loaded_clip_without_loading(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)
    (model_dir / "ViT-B-32.pt").write_bytes(b"fake")
    fake_clip = FakeClipModule()
    model = FakeClipModel()
    preprocess = object()

    evaluator = ClipLocalImageQualityEvaluator(
        ClipLocalConfig(model_dir="CLIP", project_root=project_root),
        clip_module=fake_clip,
        model=model,
        preprocess=preprocess,
    )

    assert fake_clip.load_calls == []
    assert evaluator._load_clip() == (model, preprocess)
    assert fake_clip.load_calls == []


def test_clip_local_evaluator_loads_from_validated_local_weight_path(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "CLIP"
    model_dir.mkdir(parents=True)
    weight = model_dir / "ViT-B-32.pt"
    weight.write_bytes(b"fake")
    fake_clip = FakeClipModule()

    evaluator = ClipLocalImageQualityEvaluator(
        ClipLocalConfig(model_dir="CLIP", project_root=project_root),
        clip_module=fake_clip,
    )

    assert evaluator._load_clip() == (fake_clip.model, fake_clip.preprocess)
    assert len(fake_clip.load_calls) == 1
    args, kwargs = fake_clip.load_calls[0]
    assert args[0] == str(weight.resolve())
    assert args[0] != "ViT-B/32"
    assert str(args[0]).endswith("CLIP/ViT-B-32.pt")
    assert kwargs == {"device": "cuda", "jit": False}
    assert fake_clip.model.eval_calls == 1


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)


def _artifacts_with_generated_image(entity_dir: Path) -> RunEntityArtifacts:
    _write_png(entity_dir / "generated_images" / "image_001.png")
    knowledge = {
        "entity_profile": {
            "entity_type": "film",
            "entity_label": "Example Film",
        }
    }
    reasoning = {
        "modality_constraints": {
            "must_include": ["film poster"],
            "forbidden": ["watermark"],
        }
    }
    ranking = {
        "text_ranking": {
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "generated_content": {
                        "value": "Example Film is a drama film set in Exampleland."
                    },
                    "offline_evaluation_score": 0.8,
                }
            ]
        },
        "image_ranking": {
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "positive_prompt": "a cinematic poster for Example Film",
                    "generated_content": {
                        "type": "image",
                        "value": "a cinematic poster for Example Film",
                        "image_path": "generated_images/image_001.png",
                    },
                }
            ]
        },
        "pair_ranking": {
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "text_candidate": {
                        "candidate_id": "text_001",
                        "generated_content": {
                            "value": "Example Film is a drama film set in Exampleland."
                        },
                    },
                    "image_candidate": {"candidate_id": "image_001"},
                    "offline_evaluation_score": 0.9,
                }
            ]
        },
    }
    final = {
        "entity": {
            "entity_key": "Q1",
            "label": "Example Film",
            "description": "ground truth must not be copied into evidence summary",
        },
        "missing_mode": "both",
    }
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key="Q1",
        knowledge=knowledge,
        reasoning=reasoning,
        ranking=ranking,
        final=final,
    )


def test_build_entity_evidence_summary_uses_entity_and_kg_signals(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")

    summary = build_entity_evidence_summary(artifacts)

    assert "Example Film" in summary
    assert "film" in summary
    assert "film poster" in summary
    assert "ground truth must not be copied" not in summary


def test_build_entity_evidence_summary_uses_real_relation_and_image_constraints(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    artifacts.knowledge["relation_facts"] = [
        {
            "relation_label": "cast member",
            "neighbor_label": "Linda Darnell",
        }
    ]
    artifacts.reasoning["modality_constraints"] = {
        "image": {
            "must_include": ["public figure portrait"],
            "should_include": ["period clothing"],
            "context_only": ["Hollywood film"],
        }
    }

    summary = build_entity_evidence_summary(artifacts)

    assert "cast member" in summary
    assert "Linda Darnell" in summary
    assert "public figure portrait" in summary
    assert "period clothing" in summary
    assert "Hollywood film" in summary
    assert "ground truth must not be copied" not in summary


def test_build_entity_evidence_summary_flattens_dict_image_constraints(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    artifacts.reasoning["modality_constraints"] = {
        "image": {
            "must_include": [
                {
                    "constraint_type": "relation_evidence",
                    "relation": "director",
                    "object_label": "John Ford",
                    "source": "kg",
                    "evidence_triples": [
                        {
                            "relation": "director",
                            "tail_label": "John Ford",
                            "source": "kg",
                        }
                    ],
                }
            ]
        }
    }

    summary = build_entity_evidence_summary(artifacts)

    assert "director" in summary
    assert "John Ford" in summary
    assert "constraint_type" not in summary
    assert "evidence_triples" not in summary
    assert "{" not in summary
    assert "}" not in summary


def test_evaluate_entity_image_quality_scores_three_targets(tmp_path: Path) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    scorer = FixedImageTextSimilarityScorer([10.0, 20.0, 30.0])

    result = evaluate_entity_image_quality(
        artifacts,
        scorer=scorer,
        clip_model_name="ViT-B/32",
        clip_weight_path=Path("CLIP/ViT-B-32.pt"),
    )

    assert result["entity"]["entity_key"] == "Q1"
    assert result["image_count"] == 1
    image_score = result["image_scores"][0]
    assert image_score["image_candidate_id"] == "image_001"
    assert image_score["clip_prompt_similarity"] == 10.0
    assert image_score["clip_text_similarity"] == 20.0
    assert image_score["clip_evidence_similarity"] == 30.0
    assert image_score["clip_overall_image_quality_score"] == 20.0
    assert result["averages"]["clip_overall_image_quality_score"] == 20.0
    assert scorer.calls[0]["texts"][0] == "a cinematic poster for Example Film"


def test_evaluate_entity_image_quality_rejects_missing_png(tmp_path: Path) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    (artifacts.entity_dir / "generated_images" / "image_001.png").unlink()

    with pytest.raises(ClipImageQualityError, match="image file does not exist"):
        evaluate_entity_image_quality(
            artifacts,
            scorer=FixedImageTextSimilarityScorer([1.0, 2.0, 3.0]),
            clip_model_name="ViT-B/32",
            clip_weight_path=Path("CLIP/ViT-B-32.pt"),
        )


def test_evaluate_entity_image_quality_rejects_traversal_image_path(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    _write_png(tmp_path / "outside.png")
    image_candidate = artifacts.ranking["image_ranking"]["ranked_candidates"][0]
    image_candidate["generated_content"]["image_path"] = "../outside.png"

    with pytest.raises(ClipImageQualityError, match="image path must stay under"):
        evaluate_entity_image_quality(
            artifacts,
            scorer=FixedImageTextSimilarityScorer([1.0, 2.0, 3.0]),
            clip_model_name="ViT-B/32",
            clip_weight_path=Path("CLIP/ViT-B-32.pt"),
        )


def test_evaluate_entity_image_quality_rejects_symlink_image_escape(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    outside = tmp_path / "outside.png"
    _write_png(outside)
    link = artifacts.entity_dir / "generated_images" / "escape.png"
    link.symlink_to(outside)
    image_candidate = artifacts.ranking["image_ranking"]["ranked_candidates"][0]
    image_candidate["generated_content"]["image_path"] = "generated_images/escape.png"

    with pytest.raises(ClipImageQualityError, match="image path must stay under"):
        evaluate_entity_image_quality(
            artifacts,
            scorer=FixedImageTextSimilarityScorer([1.0, 2.0, 3.0]),
            clip_model_name="ViT-B/32",
            clip_weight_path=Path("CLIP/ViT-B-32.pt"),
        )


def test_evaluate_entity_image_quality_maps_missing_target_score_to_none(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    artifacts.ranking["text_ranking"]["ranked_candidates"] = []
    artifacts.ranking["pair_ranking"]["ranked_pairs"] = []
    scorer = FixedImageTextSimilarityScorer([10.0, 30.0])

    result = evaluate_entity_image_quality(
        artifacts,
        scorer=scorer,
        clip_model_name="ViT-B/32",
        clip_weight_path=Path("CLIP/ViT-B-32.pt"),
    )

    image_score = result["image_scores"][0]
    assert image_score["clip_prompt_similarity"] == 10.0
    assert image_score["clip_text_similarity"] is None
    assert image_score["clip_evidence_similarity"] == 30.0
    assert image_score["clip_overall_image_quality_score"] == 20.0
    assert result["averages"]["clip_text_similarity"] is None
    assert result["averages"]["clip_overall_image_quality_score"] == 20.0
    assert scorer.calls[0]["texts"] == [
        "a cinematic poster for Example Film",
        build_entity_evidence_summary(artifacts),
    ]


def test_evaluate_entity_image_quality_aggregates_two_images(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    _write_png(artifacts.entity_dir / "generated_images" / "image_002.png")
    artifacts.ranking["image_ranking"]["ranked_candidates"].append(
        {
            "candidate_id": "image_002",
            "positive_prompt": "a second cinematic poster for Example Film",
            "generated_content": {
                "type": "image",
                "value": "a second cinematic poster for Example Film",
                "image_path": "generated_images/image_002.png",
            },
        }
    )
    artifacts.ranking["pair_ranking"]["ranked_pairs"].append(
        {
            "pair_id": "pair_002",
            "text_candidate": {
                "candidate_id": "text_001",
                "generated_content": {
                    "value": "Example Film is a drama film set in Exampleland."
                },
            },
            "image_candidate": {"candidate_id": "image_002"},
            "offline_evaluation_score": 0.7,
        }
    )
    scorer = FixedImageTextSimilarityScorer(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )

    result = evaluate_entity_image_quality(
        artifacts,
        scorer=scorer,
        clip_model_name="ViT-B/32",
        clip_weight_path=Path("CLIP/ViT-B-32.pt"),
    )

    assert result["image_count"] == 2
    assert result["averages"]["clip_prompt_similarity"] == 25.0
    assert result["averages"]["clip_text_similarity"] == 35.0
    assert result["averages"]["clip_evidence_similarity"] == 45.0
    assert result["averages"]["clip_overall_image_quality_score"] == 35.0
    assert result["best_image_candidate_id"] == "image_002"
    assert result["worst_image_candidate_id"] == "image_001"
    assert result["image_scores"][0]["clip_overall_image_quality_score"] == 20.0
    assert result["image_scores"][1]["clip_overall_image_quality_score"] == 50.0


def test_evaluate_entity_image_quality_rejects_entity_without_text_targets(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts_with_generated_image(tmp_path / "Q1")
    artifacts.ranking["image_ranking"]["ranked_candidates"][0].pop("positive_prompt")
    artifacts.ranking["image_ranking"]["ranked_candidates"][0]["generated_content"] = {
        "type": "image",
        "image_path": "generated_images/image_001.png",
    }
    artifacts.ranking["text_ranking"]["ranked_candidates"] = []
    artifacts.ranking["pair_ranking"]["ranked_pairs"] = []
    artifacts.knowledge["entity_profile"] = {}
    artifacts.reasoning["modality_constraints"] = {}
    artifacts.final["entity"] = {"entity_key": "Q1"}

    with pytest.raises(ClipImageQualityError, match="no text targets available"):
        evaluate_entity_image_quality(
            artifacts,
            scorer=FixedImageTextSimilarityScorer([]),
            clip_model_name="ViT-B/32",
            clip_weight_path=Path("CLIP/ViT-B-32.pt"),
        )
