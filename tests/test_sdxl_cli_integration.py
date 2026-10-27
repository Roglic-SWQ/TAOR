import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

import mmkgc.run as run_module
import mmkgc.defaults as defaults_module
from mmkgc.adapters import GeneratedImageArtifact
from mmkgc.data import EntityRecord
from mmkgc.io import RunWriter
from mmkgc.pipeline import MMKGCPipeline


def _load_validate_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_sdxl_image_run.py"
    spec = importlib.util.spec_from_file_location("validate_sdxl_image_run", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_module = _load_validate_module()
validate_run = validate_module.validate_run
validate_main = validate_module.main


class RecordingImageGenerator:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
    ) -> GeneratedImageArtifact:
        self.calls.append(
            {
                "entity_key": entity_key,
                "candidate_id": candidate_id,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
            }
        )
        generated = candidate_id == "image_001"
        metadata = {
            "generator": "sdxl-local",
            "model_name": "stable-diffusion-xl-base-1.0",
            "model_path": "SDXL/stable-diffusion-xl-base-1.0",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "seed": 20260708,
            "status": "generated" if generated else "failed_fallback",
            "fallback_used": not generated,
        }
        if not generated:
            metadata["error_type"] = "RuntimeError"
            metadata["error_message"] = "fake image generation failed"
        return GeneratedImageArtifact(
            entity_key=entity_key,
            candidate_id=candidate_id,
            relative_image_path=f"generated_images/{candidate_id}.png",
            relative_metadata_path=f"generated_images/{candidate_id}.json",
            image_bytes=b"fake-png-bytes" if generated else b"",
            metadata=metadata,
            model_call={
                "entity_key": entity_key,
                "candidate_id": candidate_id,
                "modality": "image",
                "status": metadata["status"],
                "generator": metadata["generator"],
                "fallback_used": metadata["fallback_used"],
            },
        )


def _record() -> EntityRecord:
    return EntityRecord(
        entity_id=1,
        entity_key="Q1",
        label="Example Film",
        description="Example Film is a drama film.",
        neighbors=[
            {
                "entity_id": 2,
                "entity_key": "Q2",
                "label": "film",
                "description": "a motion picture work",
                "relation_id": 10,
                "relation": "instance of",
                "direction": "out",
            }
        ],
        existing_modalities={"text": None, "image": None},
        ground_truth_modalities={"text": "Example Film is a drama film.", "image": "image.jpg"},
    )


def _stage4(result):
    return result["relation_alternation_generation"]["relation_alternation_generation"]


def test_cli_help_includes_sdxl_options():
    help_text = run_module.build_parser().format_help()

    assert "--image-generator" not in help_text
    assert "--sdxl-model-dir" in help_text
    assert "--sdxl-width" in help_text
    assert "--sdxl-height" in help_text
    assert "--sdxl-steps" in help_text
    assert "--sdxl-seed" in help_text
    assert "--sdxl-guidance-scale" in help_text
    assert "--sdxl-device" in help_text




def test_validate_sdxl_model_dir_rejects_missing_dir_or_missing_model_index(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    sdxl_root = project_root / "SDXL"
    sdxl_root.mkdir(parents=True)
    base_model_dir = sdxl_root / "stable-diffusion-xl-base-1.0"
    monkeypatch.setattr(defaults_module, "MMKGC_ROOT", project_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_ROOT", sdxl_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_BASE_MODEL_DIR", base_model_dir, raising=False)

    missing_dir = sdxl_root / "missing-sdxl"
    file_path = sdxl_root / "not-a-dir"
    file_path.write_text("not a directory\n", encoding="utf-8")
    model_dir = base_model_dir
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="sdxl_model_dir does not exist"):
        defaults_module.validate_sdxl_model_dir(missing_dir)
    with pytest.raises(NotADirectoryError, match="sdxl_model_dir is not a directory"):
        defaults_module.validate_sdxl_model_dir(file_path)
    with pytest.raises(FileNotFoundError, match="sdxl_model_dir is missing model_index.json"):
        defaults_module.validate_sdxl_model_dir(model_dir)


def test_validate_sdxl_model_dir_rejects_existing_external_model_dir(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    sdxl_root = project_root / "SDXL"
    base_model_dir = sdxl_root / "stable-diffusion-xl-base-1.0"
    external_model_dir = tmp_path / "external-sdxl"
    external_model_dir.mkdir(parents=True)
    (external_model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(defaults_module, "MMKGC_ROOT", project_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_ROOT", sdxl_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_BASE_MODEL_DIR", base_model_dir, raising=False)

    with pytest.raises(ValueError, match="sdxl_model_dir must be under"):
        defaults_module.validate_sdxl_model_dir(external_model_dir)


def test_validate_sdxl_model_dir_rejects_existing_refiner_model_dir(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    sdxl_root = project_root / "SDXL"
    base_model_dir = sdxl_root / "stable-diffusion-xl-base-1.0"
    refiner_model_dir = sdxl_root / "stable-diffusion-xl-refiner-1.0"
    refiner_model_dir.mkdir(parents=True)
    (refiner_model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(defaults_module, "MMKGC_ROOT", project_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_ROOT", sdxl_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_BASE_MODEL_DIR", base_model_dir, raising=False)

    with pytest.raises(
        ValueError,
        match=r"only SDXL/stable-diffusion-xl-base-1\.0 is supported",
    ):
        defaults_module.validate_sdxl_model_dir("SDXL/stable-diffusion-xl-refiner-1.0")


def test_validate_sdxl_model_dir_accepts_relative_path_under_sdxl_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    sdxl_root = project_root / "SDXL"
    model_dir = sdxl_root / "stable-diffusion-xl-base-1.0"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(defaults_module, "MMKGC_ROOT", project_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_ROOT", sdxl_root, raising=False)
    monkeypatch.setattr(defaults_module, "SDXL_BASE_MODEL_DIR", model_dir, raising=False)

    assert defaults_module.validate_sdxl_model_dir("SDXL/stable-diffusion-xl-base-1.0") == model_dir.resolve()






def _write_fake_sdxl_run(
    run_dir: Path,
    *,
    entity_count: int = 1,
    images_per_entity: int = 1,
    image_size: tuple[int, int] = (32, 32),
    write_image_calls: bool = True,
) -> None:
    (run_dir / "entities").mkdir(parents=True)
    (run_dir / "metrics_summary.json").write_text(
        json.dumps({"processed_entities": entity_count, "failed_entities": 0}),
        encoding="utf-8",
    )
    for entity_index in range(1, entity_count + 1):
        entity_dir = run_dir / "entities" / f"Q{entity_index}"
        images_dir = entity_dir / "generated_images"
        images_dir.mkdir(parents=True)
        image_call_rows = []
        for image_index in range(1, images_per_entity + 1):
            candidate_id = f"image_{image_index:03d}"
            image_path = images_dir / f"{candidate_id}.png"
            Image.new("RGB", image_size, "white").save(image_path)
            (images_dir / f"{candidate_id}.json").write_text(
                json.dumps(
                    {
                        "candidate_id": candidate_id,
                        "width": image_size[0],
                        "height": image_size[1],
                        "status": "generated",
                    }
                ),
                encoding="utf-8",
            )
            image_call_rows.append(
                json.dumps({"candidate_id": candidate_id, "modality": "image", "status": "generated"})
            )
        if write_image_calls:
            (entity_dir / "image_model_calls.jsonl").write_text("\n".join(image_call_rows) + "\n", encoding="utf-8")


def test_validate_sdxl_image_run_accepts_expected_artifacts(tmp_path):
    run_dir = tmp_path / "outputs" / "sdxl"
    _write_fake_sdxl_run(run_dir, entity_count=2, images_per_entity=2, image_size=(32, 32))

    summary = validate_run(
        run_dir,
        expected_entities=2,
        expected_images_per_entity=2,
        expected_width=32,
        expected_height=32,
    )

    assert summary == {"ok": True, "entities_checked": 2, "images_checked": 4}


def test_validate_sdxl_image_run_rejects_missing_image_model_calls(tmp_path):
    run_dir = tmp_path / "outputs" / "sdxl"
    _write_fake_sdxl_run(run_dir, entity_count=1, images_per_entity=1, write_image_calls=False)

    with pytest.raises(AssertionError, match="image_model_calls.jsonl"):
        validate_run(
            run_dir,
            expected_entities=1,
            expected_images_per_entity=1,
            expected_width=32,
            expected_height=32,
        )


def test_validate_sdxl_image_run_rejects_non_object_metrics_summary(tmp_path):
    run_dir = tmp_path / "outputs" / "sdxl"
    _write_fake_sdxl_run(run_dir, entity_count=1, images_per_entity=1)
    (run_dir / "metrics_summary.json").write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(AssertionError, match="metrics_summary\\.json must contain a JSON object"):
        validate_run(
            run_dir,
            expected_entities=1,
            expected_images_per_entity=1,
            expected_width=32,
            expected_height=32,
        )


def test_validate_sdxl_image_run_rejects_non_object_image_model_call_row(tmp_path):
    run_dir = tmp_path / "outputs" / "sdxl"
    _write_fake_sdxl_run(run_dir, entity_count=1, images_per_entity=1)
    calls_path = run_dir / "entities" / "Q1" / "image_model_calls.jsonl"
    calls_path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError, match=r"image_model_calls\.jsonl:1 must contain a JSON object"):
        validate_run(
            run_dir,
            expected_entities=1,
            expected_images_per_entity=1,
            expected_width=32,
            expected_height=32,
        )


def test_validate_sdxl_image_run_main_returns_nonzero_for_validation_failure(tmp_path, capsys):
    run_dir = tmp_path / "outputs" / "sdxl"
    _write_fake_sdxl_run(run_dir, entity_count=1, images_per_entity=1)
    (run_dir / "metrics_summary.json").write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    exit_code = validate_main(
        [
            "--run-dir",
            str(run_dir),
            "--expected-entities",
            "1",
            "--expected-images-per-entity",
            "1",
            "--expected-width",
            "32",
            "--expected-height",
            "32",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.startswith("SDXL image run validation failed:")
    assert "metrics_summary.json must contain a JSON object" in captured.err
