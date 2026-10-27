import pytest
from PIL import Image

from mmkgc.adapters.base import RuleImagePromptGenerator
from mmkgc.adapters.sdxl import SDXLConfig, SDXLImageGenerator, _load_diffusers_pipeline


class FakeImageResult:
    def __init__(self, image):
        self.images = [image]


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeImageResult(
            Image.new("RGB", (kwargs["width"], kwargs["height"]), "white")
        )


class FailingPipeline:
    def __call__(self, **kwargs):
        raise RuntimeError("forced candidate failure")


def test_sdxl_generator_uses_prompt_parameters_and_returns_png(tmp_path):
    fake_pipeline = FakePipeline()
    config = SDXLConfig(
        model_dir=tmp_path,
        width=64,
        height=64,
        steps=3,
        seed=123,
        guidance_scale=7.5,
        device="cpu",
    )
    generator = SDXLImageGenerator(config, pipeline_factory=lambda _: fake_pipeline)

    artifact = generator.generate(
        entity_key="Q1",
        candidate_id="image_001",
        positive_prompt="Ada Lovelace portrait, mathematician",
        negative_prompt="unsupported detail",
    )

    assert artifact.entity_key == "Q1"
    assert artifact.candidate_id == "image_001"
    assert artifact.relative_image_path == "generated_images/image_001.png"
    assert artifact.relative_metadata_path == "generated_images/image_001.json"
    assert artifact.metadata["modality"] == "image"
    assert artifact.metadata["status"] == "generated"
    assert artifact.metadata["generator"] == "sdxl-local"
    assert artifact.metadata["model_name"] == "stable-diffusion-xl-base-1.0"
    assert artifact.metadata["model_path"] == str(tmp_path)
    assert artifact.metadata["positive_prompt"] == "Ada Lovelace portrait, mathematician"
    assert artifact.metadata["negative_prompt"] == "unsupported detail"
    assert artifact.metadata["width"] == 64
    assert artifact.metadata["height"] == 64
    assert artifact.metadata["steps"] == 3
    assert artifact.metadata["seed"] == 124
    assert artifact.metadata["guidance_scale"] == 7.5
    assert artifact.metadata["fallback_used"] is False
    assert isinstance(artifact.metadata["elapsed_ms"], int)
    assert artifact.model_call["modality"] == "image"
    assert artifact.model_call["status"] == "generated"
    assert artifact.model_call["entity_key"] == "Q1"
    assert artifact.model_call["candidate_id"] == "image_001"
    assert fake_pipeline.calls[0]["prompt"] == "Ada Lovelace portrait, mathematician"
    assert fake_pipeline.calls[0]["negative_prompt"] == "unsupported detail"
    assert fake_pipeline.calls[0]["num_inference_steps"] == 3
    assert fake_pipeline.calls[0]["guidance_scale"] == 7.5
    assert artifact.image_bytes.startswith(b"\x89PNG")


def test_sdxl_generator_uses_stable_candidate_specific_seeds(tmp_path):
    fake_pipeline = FakePipeline()
    config = SDXLConfig(
        model_dir=tmp_path,
        width=32,
        height=32,
        seed=20260708,
        device="cpu",
    )
    generator = SDXLImageGenerator(config, pipeline_factory=lambda _: fake_pipeline)

    first = generator.generate(
        entity_key="Q1",
        candidate_id="image_001",
        positive_prompt="first visual candidate",
        negative_prompt="unsupported detail",
    )
    second = generator.generate(
        entity_key="Q1",
        candidate_id="image_002",
        positive_prompt="second visual candidate",
        negative_prompt="unsupported detail",
    )

    assert first.metadata["seed"] == 20260709
    assert second.metadata["seed"] == 20260710
    assert first.model_call["seed"] == 20260709
    assert second.model_call["seed"] == 20260710
    if "generator" in fake_pipeline.calls[0]:
        assert fake_pipeline.calls[0]["generator"].initial_seed() == 20260709
        assert fake_pipeline.calls[1]["generator"].initial_seed() == 20260710


def test_sdxl_candidate_generation_exception_fails_mainline(tmp_path):
    config = SDXLConfig(model_dir=tmp_path, width=32, height=32, device="cpu")
    generator = SDXLImageGenerator(config, pipeline_factory=lambda _: FailingPipeline())

    with pytest.raises(
        RuntimeError,
        match="SDXL generation failed for candidate image_002: RuntimeError: forced candidate failure",
    ):
        generator.generate(
            entity_key="Q2",
            candidate_id="image_002",
            positive_prompt="supported prompt",
            negative_prompt="avoid unsupported",
        )


def test_sdxl_default_factory_raises_when_model_dir_is_missing(tmp_path):
    missing_model_dir = tmp_path / "missing-sdxl"

    with pytest.raises(FileNotFoundError, match=str(missing_model_dir)):
        _load_diffusers_pipeline(SDXLConfig(model_dir=missing_model_dir))


def test_rule_image_prompt_includes_entity_label_and_fact_details():
    prompt = RuleImagePromptGenerator().generate(
        record_label="Lee Tergesen",
        entity_type="actor",
        relation="cast member",
        angle="evidence-grounded description",
        facts=["cast member: The Collection", "text visual cue: cinematic portrait"],
    )

    positive = prompt["positive_prompt"]
    assert "actor" in positive.lower()
    assert "cast member context" in positive
    assert "evidence-grounded scene" in positive
    assert "Lee Tergesen" in positive
    assert "text visual cue evidence" in positive
    assert "The Collection" in positive


def test_rule_image_prompt_uses_relation_scene_when_entity_type_is_generic():
    prompt = RuleImagePromptGenerator().generate(
        record_label="Kanye West",
        entity_type="entity",
        relation="performer",
        angle="identity anchor",
        facts=["performer: Good Life"],
    )

    positive = prompt["positive_prompt"]
    assert "stage performance scene" in positive
    assert "contextual visual scene" not in positive
    assert "performer evidence" in positive
    assert "Kanye West" in positive
    assert "Good Life" in positive
