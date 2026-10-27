from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import mmkgc.evaluation.blip_itm as blip_itm
from mmkgc.evaluation.blip_itm import (
    BLIPITMError,
    BlipITMConfig,
    BlipITMLocalScorer,
    FixedBLIPITMScorer,
    evaluate_entity_blip_itm,
    validate_blip_model_dir,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


REQUIRED_BLIP_FILES = (
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)


def _write_valid_blip_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_BLIP_FILES:
        file_path = path / filename
        if filename == "config.json":
            file_path.write_text(
                json.dumps({"architectures": ["BlipForImageTextRetrieval"]}),
                encoding="utf-8",
            )
        elif filename == "pytorch_model.bin":
            file_path.write_bytes(b"fake")
        else:
            file_path.write_text("{}", encoding="utf-8")


def _minimal_artifacts(entity_dir: Path) -> RunEntityArtifacts:
    _write_png(entity_dir / "generated_images" / "image_001.png")
    return RunEntityArtifacts(
        entity_dir=entity_dir,
        entity_key="Q1",
        knowledge={
            "entity_profile": {
                "entity_key": "Q1",
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
                "image": {"must_include": ["portrait", "mathematician"]}
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
                        "positive_prompt": (
                            "portrait of Ada Lovelace as a mathematician"
                        ),
                        "generated_content": {
                            "image_path": "generated_images/image_001.png"
                        },
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "text_candidate_id": "text_001",
                        "image_candidate_id": "image_001",
                    }
                ]
            },
        },
        final={
            "entity": {"entity_key": "Q1", "label": "Ada Lovelace"},
            "missing_mode": "both",
        },
    )


def test_validate_blip_model_dir_accepts_project_local_model(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)

    resolved = validate_blip_model_dir("blip-itm-large-coco", project_root=project_root)

    assert resolved == model_dir.resolve()


def test_validate_blip_model_dir_rejects_external_path(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    external = tmp_path / "outside" / "blip-itm-large-coco"
    _write_valid_blip_model_dir(external)

    with pytest.raises(BLIPITMError, match="blip_model_dir must be under"):
        validate_blip_model_dir(external, project_root=project_root)


def test_validate_blip_model_dir_rejects_symlink_escape(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    project_root.mkdir()
    external = tmp_path / "outside" / "blip-itm-large-coco"
    _write_valid_blip_model_dir(external)
    link = project_root / "blip-itm-large-coco"
    link.symlink_to(external, target_is_directory=True)

    with pytest.raises(BLIPITMError, match="blip_model_dir must be under"):
        validate_blip_model_dir("blip-itm-large-coco", project_root=project_root)


def test_validate_blip_model_dir_requires_expected_files(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    (model_dir / "vocab.txt").unlink()

    with pytest.raises(BLIPITMError, match="missing BLIP model file"):
        validate_blip_model_dir("blip-itm-large-coco", project_root=project_root)


def test_validate_blip_model_dir_requires_itm_architecture(tmp_path: Path) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["BlipForConditionalGeneration"]}),
        encoding="utf-8",
    )

    with pytest.raises(BLIPITMError, match="BlipForImageTextRetrieval"):
        validate_blip_model_dir("blip-itm-large-coco", project_root=project_root)


def test_fixed_blip_itm_scorer_returns_one_score_per_text(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_png(image_path)
    scorer = FixedBLIPITMScorer([[10, 20, 30]])

    scores = scorer.score_image_texts(image_path, ["prompt", "text", "evidence"])

    assert scores == [10.0, 20.0, 30.0]
    assert scorer.calls == [
        {"image_path": image_path, "texts": ["prompt", "text", "evidence"]}
    ]


def test_blip_itm_local_scorer_score_image_texts_delegates_to_pil_hook(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    image_path = tmp_path / "image.png"
    _write_png(image_path)

    class FakeScorer(BlipITMLocalScorer):
        def score_pil_texts(self, image: object, texts: list[str]) -> list[float]:
            self.pil_call = {"mode": image.mode, "size": image.size, "texts": texts}
            return [44.0]

    scorer = FakeScorer(
        BlipITMConfig(model_dir=model_dir, device="cpu", project_root=tmp_path)
    )

    assert scorer.score_image_texts(image_path, ["text target"]) == [44.0]
    assert scorer.pil_call == {
        "mode": "RGB",
        "size": (4, 4),
        "texts": ["text target"],
    }


def test_blip_itm_local_scorer_score_pil_texts_uses_injected_model_without_loading(
    tmp_path: Path,
) -> None:
    import torch

    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__({"input_ids": "fake-input-ids"})
            self.to_calls: list[str] = []

        def to(self, device: str) -> "FakeInputs":
            self.to_calls.append(device)
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(
            self,
            *,
            images: object,
            text: str,
            return_tensors: str,
            padding: bool,
        ) -> FakeInputs:
            inputs = FakeInputs()
            self.calls.append(
                {
                    "images": images,
                    "text": text,
                    "return_tensors": return_tensors,
                    "padding": padding,
                    "inputs": inputs,
                }
            )
            return inputs

    class FakeModel:
        def __init__(self) -> None:
            self.to_calls: list[str] = []
            self.eval_calls = 0
            self.call_kwargs: list[dict[str, object]] = []

        def to(self, device: str) -> "FakeModel":
            self.to_calls.append(device)
            return self

        def eval(self) -> None:
            self.eval_calls += 1

        def __call__(self, **inputs: object) -> object:
            self.call_kwargs.append(inputs)
            return type("FakeOutputs", (), {"itm_score": torch.tensor([[0.0, 2.0]])})()

    processor = FakeProcessor()
    model = FakeModel()
    scorer = BlipITMLocalScorer(
        BlipITMConfig(model_dir=model_dir, device="cpu", project_root=tmp_path),
        processor=processor,
        model=model,
    )

    scores = scorer.score_pil_texts(image, ["text target"])

    expected = torch.softmax(torch.tensor([[0.0, 2.0]]), dim=-1)[0, 1].item() * 100
    assert scores == [round(expected, 6)]
    assert processor.calls[0]["images"].mode == "RGB"
    assert processor.calls[0]["text"] == "text target"
    assert processor.calls[0]["return_tensors"] == "pt"
    assert processor.calls[0]["padding"] is True
    assert processor.calls[0]["inputs"].to_calls == ["cpu"]
    assert model.call_kwargs == [{"input_ids": "fake-input-ids"}]


def test_blip_itm_local_scorer_loads_local_model_and_scores_itm_softmax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    image_path = tmp_path / "image.png"
    _write_png(image_path)

    class FakeInputs(dict):
        def __init__(self) -> None:
            super().__init__({"input_ids": "fake-input-ids"})
            self.to_calls: list[str] = []

        def to(self, device: str) -> "FakeInputs":
            self.to_calls.append(device)
            return self

    class FakeProcessor:
        from_pretrained_calls: list[dict[str, object]] = []
        call_inputs: list[FakeInputs] = []

        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool) -> "FakeProcessor":
            cls.from_pretrained_calls.append(
                {"path": path, "local_files_only": local_files_only}
            )
            return cls()

        def __call__(
            self,
            *,
            images: object,
            text: str,
            return_tensors: str,
            padding: bool,
        ) -> FakeInputs:
            inputs = FakeInputs()
            self.call_inputs.append(inputs)
            self.last_call = {
                "images": images,
                "text": text,
                "return_tensors": return_tensors,
                "padding": padding,
            }
            return inputs

    class FakeModel:
        from_pretrained_calls: list[dict[str, object]] = []
        instances: list["FakeModel"] = []

        def __init__(self) -> None:
            self.to_calls: list[str] = []
            self.eval_calls = 0
            self.call_kwargs: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool) -> "FakeModel":
            cls.from_pretrained_calls.append(
                {"path": path, "local_files_only": local_files_only}
            )
            model = cls()
            cls.instances.append(model)
            return model

        def to(self, device: str) -> "FakeModel":
            self.to_calls.append(device)
            return self

        def eval(self) -> None:
            self.eval_calls += 1

        def __call__(self, **inputs: object) -> object:
            self.call_kwargs.append(inputs)
            return type("FakeOutputs", (), {"itm_score": torch.tensor([[0.0, 2.0]])})()

    monkeypatch.setattr(blip_itm, "BlipProcessor", FakeProcessor)
    monkeypatch.setattr(blip_itm, "BlipForImageTextRetrieval", FakeModel)

    scorer = BlipITMLocalScorer(
        BlipITMConfig(model_dir=model_dir, device="cpu", project_root=tmp_path)
    )

    scores = scorer.score_image_texts(image_path, ["text target"])

    expected = torch.softmax(torch.tensor([[0.0, 2.0]]), dim=-1)[0, 1].item() * 100
    assert scores == [round(expected, 6)]
    assert FakeProcessor.from_pretrained_calls == [
        {"path": str(model_dir.resolve()), "local_files_only": True}
    ]
    assert FakeModel.from_pretrained_calls == [
        {"path": str(model_dir.resolve()), "local_files_only": True}
    ]
    model = FakeModel.instances[0]
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert model.call_kwargs == [{"input_ids": "fake-input-ids"}]
    processor = scorer._processor
    assert processor.last_call["text"] == "text target"
    assert processor.last_call["return_tensors"] == "pt"
    assert processor.last_call["padding"] is True
    assert processor.call_inputs[0].to_calls == ["cpu"]


def test_blip_itm_local_scorer_rejects_outputs_without_itm_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    image_path = tmp_path / "image.png"
    _write_png(image_path)

    class FakeInputs(dict):
        def to(self, device: str) -> "FakeInputs":
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool) -> "FakeProcessor":
            return cls()

        def __call__(self, **kwargs: object) -> FakeInputs:
            return FakeInputs()

    class FakeModelWithoutITMScore:
        @classmethod
        def from_pretrained(
            cls,
            path: str,
            *,
            local_files_only: bool,
        ) -> "FakeModelWithoutITMScore":
            return cls()

        def to(self, device: str) -> "FakeModelWithoutITMScore":
            return self

        def eval(self) -> None:
            pass

        def __call__(self, **inputs: object) -> object:
            return object()

    monkeypatch.setattr(blip_itm, "BlipProcessor", FakeProcessor)
    monkeypatch.setattr(
        blip_itm,
        "BlipForImageTextRetrieval",
        FakeModelWithoutITMScore,
    )
    scorer = BlipITMLocalScorer(
        BlipITMConfig(model_dir=model_dir, device="cpu", project_root=tmp_path)
    )

    with pytest.raises(BLIPITMError, match="does not contain itm_score"):
        scorer.score_image_texts(image_path, ["text target"])


def test_blip_itm_local_scorer_rejects_non_finite_itm_probability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    project_root = tmp_path / "MMKGC"
    model_dir = project_root / "blip-itm-large-coco"
    _write_valid_blip_model_dir(model_dir)
    image_path = tmp_path / "image.png"
    _write_png(image_path)

    class FakeInputs(dict):
        def to(self, device: str) -> "FakeInputs":
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path: str, *, local_files_only: bool) -> "FakeProcessor":
            return cls()

        def __call__(self, **kwargs: object) -> FakeInputs:
            return FakeInputs()

    class FakeModelWithNaNScore:
        @classmethod
        def from_pretrained(
            cls,
            path: str,
            *,
            local_files_only: bool,
        ) -> "FakeModelWithNaNScore":
            return cls()

        def to(self, device: str) -> "FakeModelWithNaNScore":
            return self

        def eval(self) -> None:
            pass

        def __call__(self, **inputs: object) -> object:
            return type(
                "FakeOutputs",
                (),
                {"itm_score": torch.tensor([[float("nan"), 0.0]])},
            )()

    monkeypatch.setattr(blip_itm, "BlipProcessor", FakeProcessor)
    monkeypatch.setattr(blip_itm, "BlipForImageTextRetrieval", FakeModelWithNaNScore)
    scorer = BlipITMLocalScorer(
        BlipITMConfig(model_dir=model_dir, device="cpu", project_root=tmp_path)
    )

    with pytest.raises(BLIPITMError, match="BLIP/ITM probability"):
        scorer.score_image_texts(image_path, ["text target"])


@pytest.mark.parametrize(
    "probability",
    [float("nan"), float("inf"), float("-inf")],
)
def test_probability_to_score_rejects_non_finite_values(probability: float) -> None:
    with pytest.raises(BLIPITMError, match="BLIP/ITM probability"):
        blip_itm._probability_to_score(probability)


def test_evaluate_entity_blip_itm_scores_three_targets_and_averages(
    tmp_path: Path,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q1")
    scorer = FixedBLIPITMScorer([[11, 22, 33]])

    result = evaluate_entity_blip_itm(
        artifacts,
        scorer,
        Path("blip-itm-large-coco"),
    )

    assert result["entity"] == {
        "entity_key": "Q1",
        "label": "Ada Lovelace",
        "missing_mode": "both",
    }
    assert result["image_count"] == 1
    assert result["model_dir"] == "blip-itm-large-coco"
    assert result["averages"]["blip_prompt_itm_score"] == 11.0
    assert result["averages"]["blip_text_itm_score"] == 22.0
    assert result["averages"]["blip_evidence_itm_score"] == 33.0
    assert result["averages"]["blip_overall_itm_score"] == 22.0
    assert result["best_image_candidate_id"] == "image_001"
    assert result["worst_image_candidate_id"] == "image_001"

    image_score = result["image_scores"][0]
    assert image_score["entity_key"] == "Q1"
    assert image_score["image_candidate_id"] == "image_001"
    assert image_score["image_path"] == "generated_images/image_001.png"
    assert image_score["provider"] == "blip-itm-local"
    assert image_score["model_dir"] == "blip-itm-large-coco"
    assert image_score["status"] == "scored"
    assert image_score["fallback_used"] is False
    assert image_score["blip_prompt_itm_score"] == 11.0
    assert image_score["blip_text_itm_score"] == 22.0
    assert image_score["blip_evidence_itm_score"] == 33.0
    assert image_score["blip_overall_itm_score"] == 22.0
    assert image_score["text_targets"]["prompt_text"] == (
        "portrait of Ada Lovelace as a mathematician"
    )
    assert image_score["text_targets"]["candidate_text"] == (
        "Ada Lovelace was a mathematician."
    )
    assert "occupation: mathematician" in image_score["text_targets"][
        "evidence_summary_text"
    ]
    assert scorer.calls[0]["texts"] == [
        "portrait of Ada Lovelace as a mathematician",
        "Ada Lovelace was a mathematician.",
        image_score["text_targets"]["evidence_summary_text"],
    ]


def test_evaluate_entity_blip_itm_wraps_scorer_errors(tmp_path: Path) -> None:
    class BrokenScorer:
        def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
            raise ValueError("boom")

    artifacts = _minimal_artifacts(tmp_path / "Q1")

    with pytest.raises(BLIPITMError, match="BLIP/ITM scorer failed"):
        evaluate_entity_blip_itm(
            artifacts,
            BrokenScorer(),
            Path("blip-itm-large-coco"),
        )


@pytest.mark.parametrize(
    "bad_score",
    [-1, 101, float("nan"), float("inf")],
)
def test_evaluate_entity_blip_itm_rejects_non_finite_or_out_of_range_scores(
    tmp_path: Path,
    bad_score: float,
) -> None:
    artifacts = _minimal_artifacts(tmp_path / "Q1")

    with pytest.raises(BLIPITMError, match="blip_prompt_itm_score.*image_001"):
        evaluate_entity_blip_itm(
            artifacts,
            FixedBLIPITMScorer([[bad_score, 22, 33]]),
            Path("blip-itm-large-coco"),
        )
