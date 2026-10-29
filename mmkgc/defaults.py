"""Python API helpers for building MMKG-aligned pipeline defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from mmkgc.adapters import (
    QwenLocalImagePromptGenerator,
    QwenLocalSemanticStagePlanner,
    QwenLocalTextGenerator,
    SDXLConfig,
    SDXLImageGenerator,
    load_qwen_config,
)
from mmkgc.adapters.evidence_reranker import Qwen3EvidenceReranker
from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
from mmkgc.pipeline import MMKGCPipeline


MMKGC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = MMKGC_ROOT / "datasets/MKG-W"
SDXL_ROOT = MMKGC_ROOT / "SDXL"
SDXL_BASE_MODEL_DIR = SDXL_ROOT / "stable-diffusion-xl-base-1.0"

MissingMode = Literal["text", "image"]


def load_runtime_environment(project_root: str | Path | None = None) -> None:
    """Read project credentials without overriding explicit process settings."""
    load_dotenv(Path(project_root or MMKGC_ROOT) / ".env", override=False)


@dataclass(frozen=True)
class PipelineRuntime:
    """Built pipeline plus resolved runtime metadata matching CLI config fields."""

    pipeline: MMKGCPipeline
    image_generator_mode: str
    model_name: str | None
    base_url: str | None
    api_key_configured: bool


def build_default_mainline_pipeline(**kwargs) -> MMKGCPipeline:
    """Build the Python API equivalent of the CLI default MMKG mainline."""

    return build_default_mainline_runtime(**kwargs).pipeline


def build_default_mainline_runtime(
    *,
    missing_mode: MissingMode = "text",
    candidate_budget: int = 5,
    stage1_reranker_model_dir: str | Path = "Qwen3-Reranker-4B",
    stage1_reranker_device: str = "cuda:0",
    stage1_reranker_max_length: int = 8192,
    stage1_model_config: str | Path = "configs/qwen7b-stage1.yaml",
    stage2_model_config: str | Path = "configs/qwen7b-stage2.yaml",
    model_config: str | Path = "configs/qwen7b-mainline.yaml",
    model_call_timeout_seconds: float = 60.0,
    stage4_config: str | Path | None = "configs/stage4-model-fusion.yaml",
    dataset_root: str | Path | None = DEFAULT_DATASET_DIR,
    sdxl_model_dir: str | Path = "SDXL/stable-diffusion-xl-base-1.0",
    sdxl_width: int = 1024,
    sdxl_height: int = 1024,
    sdxl_steps: int = 20,
    sdxl_seed: int = 20260708,
    sdxl_guidance_scale: float = 7.5,
    sdxl_device: str = "cuda",
) -> PipelineRuntime:
    """Build the single supported MMKGC mainline."""

    load_runtime_environment()
    qwen_config = load_qwen_config(model_config)
    model_name = qwen_config.model_name
    base_url = qwen_config.base_url
    api_key_configured = bool(qwen_config.api_key)
    text_generator = QwenLocalTextGenerator(
        qwen_config,
        timeout_seconds=model_call_timeout_seconds,
    )
    image_prompt_generator = (
        QwenLocalImagePromptGenerator(
            qwen_config,
            timeout_seconds=model_call_timeout_seconds,
        )
        if missing_mode == "image"
        else None
    )
    semantic_planner = QwenLocalSemanticStagePlanner(
        qwen_config,
        timeout_seconds=model_call_timeout_seconds,
    )
    stage1_config_path = Path(stage1_model_config)
    stage2_config_path = Path(stage2_model_config)
    if not stage1_config_path.is_absolute() and not stage1_config_path.exists():
        stage1_config_path = MMKGC_ROOT / stage1_config_path
    if not stage2_config_path.is_absolute() and not stage2_config_path.exists():
        stage2_config_path = MMKGC_ROOT / stage2_config_path
    knowledge_planner = QwenLocalSemanticStagePlanner(
        load_qwen_config(stage1_config_path),
        timeout_seconds=model_call_timeout_seconds,
    )
    relation_planner = OpenAICompatibleRelationPlanner(
        load_qwen_config(stage2_config_path), timeout_seconds=model_call_timeout_seconds
    )
    reranker_path = Path(stage1_reranker_model_dir)
    if not reranker_path.is_absolute():
        reranker_path = MMKGC_ROOT / reranker_path
    evidence_reranker = Qwen3EvidenceReranker(
        reranker_path, device=stage1_reranker_device, max_length=stage1_reranker_max_length
    )

    image_generator = None
    if missing_mode == "image":
        validated_model_dir = validate_sdxl_model_dir(sdxl_model_dir)
        image_generator = SDXLImageGenerator(
            SDXLConfig(
                model_dir=validated_model_dir,
                width=sdxl_width,
                height=sdxl_height,
                steps=sdxl_steps,
                seed=sdxl_seed,
                guidance_scale=sdxl_guidance_scale,
                device=sdxl_device,
            )
        )

    pipeline = MMKGCPipeline(
        candidate_budget=candidate_budget,
        text_generator=text_generator,
        image_prompt_generator=image_prompt_generator,
        image_generator=image_generator,
        semantic_planner=semantic_planner,
        knowledge_planner=knowledge_planner,
        evidence_reranker=evidence_reranker,
        relation_planner=relation_planner,
        stage4_config=stage4_config,
        dataset_root=resolve_dataset_root(dataset_root),
    )
    return PipelineRuntime(
        pipeline=pipeline,
        image_generator_mode=(
            "sdxl-local" if missing_mode == "image" else "not-required"
        ),
        model_name=model_name,
        base_url=base_url,
        api_key_configured=api_key_configured,
    )


def resolve_dataset_root(dataset_root: str | Path | None) -> Path | None:
    if dataset_root is None:
        return None
    path = Path(dataset_root)
    if not path.is_absolute():
        path = MMKGC_ROOT / path
    return path.resolve()


def validate_sdxl_model_dir(model_dir: str | Path) -> Path:
    input_path = Path(model_dir)
    candidate = input_path if input_path.is_absolute() else MMKGC_ROOT / input_path
    path = candidate.resolve(strict=False)
    sdxl_root = SDXL_ROOT.resolve(strict=False)
    allowed_model_dir = SDXL_BASE_MODEL_DIR.resolve(strict=False)
    try:
        path.relative_to(sdxl_root)
    except ValueError as exc:
        raise ValueError(f"sdxl_model_dir must be under {sdxl_root}: {path}") from exc
    if not path.exists():
        raise FileNotFoundError(f"sdxl_model_dir does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"sdxl_model_dir is not a directory: {path}")
    model_index = path / "model_index.json"
    if not model_index.is_file():
        raise FileNotFoundError(f"sdxl_model_dir is missing model_index.json: {model_index}")
    if path != allowed_model_dir:
        raise ValueError(
            "sdxl_model_dir: only SDXL/stable-diffusion-xl-base-1.0 "
            f"is supported in this stage; got: {path}"
        )
    return path


__all__ = [
    "PipelineRuntime",
    "build_default_mainline_pipeline",
    "build_default_mainline_runtime",
    "load_runtime_environment",
    "resolve_dataset_root",
    "validate_sdxl_model_dir",
]
