"""Local SDXL image generation adapter."""

from __future__ import annotations

import time
import hashlib
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from .base import GeneratedImageArtifact


_GENERATOR = "sdxl-local"
_MODEL_NAME = "stable-diffusion-xl-base-1.0"
_MAX_TORCH_SEED = 2**63 - 1


@dataclass(frozen=True)
class SDXLConfig:
    """Configuration for local Stable Diffusion XL image generation."""

    model_dir: str | Path
    width: int = 1024
    height: int = 1024
    steps: int = 20
    seed: int = 20260708
    guidance_scale: float = 7.5
    device: str = "cuda"


def _load_diffusers_pipeline(config: SDXLConfig):
    """Load the local SDXL pipeline, raising setup errors directly."""

    model_dir = Path(config.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"SDXL model directory does not exist: {model_dir}")

    import torch
    from diffusers import StableDiffusionXLPipeline

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    return pipeline.to(config.device)


def _candidate_seed(base_seed: int, candidate_id: str) -> int:
    """Derive a stable per-candidate seed from the configured run seed."""

    match = re.search(r"(\d+)$", str(candidate_id or ""))
    if match:
        offset = int(match.group(1))
    else:
        digest = hashlib.sha256(str(candidate_id).encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16)
    return (int(base_seed) + offset) % _MAX_TORCH_SEED


def _attach_generator(kwargs: dict[str, Any], config: SDXLConfig, seed: int) -> dict[str, Any]:
    """Attach a deterministic torch generator when torch is available."""

    try:
        import torch

        kwargs["generator"] = torch.Generator(device=config.device).manual_seed(seed)
    except Exception:
        pass
    return kwargs


def _png_bytes(image) -> bytes:
    """Encode a PIL Image as PNG bytes."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class SDXLImageGenerator:
    """Generate PNG artifacts with a lazily loaded local SDXL pipeline."""

    def __init__(
        self,
        config: SDXLConfig,
        pipeline_factory: Callable[[SDXLConfig], Any] | None = None,
    ):
        self.config = config
        self._pipeline_factory = pipeline_factory or _load_diffusers_pipeline
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(self.config)
        return self._pipeline

    def _relative_image_path(self, candidate_id: str) -> str:
        return f"generated_images/{candidate_id}.png"

    def _relative_metadata_path(self, candidate_id: str) -> str:
        return f"generated_images/{candidate_id}.json"

    def _base_record(
        self,
        *,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
        status: str,
        elapsed_ms: int,
        fallback_used: bool,
        seed: int,
    ) -> dict[str, Any]:
        return {
            "entity_key": entity_key,
            "candidate_id": candidate_id,
            "modality": "image",
            "status": status,
            "generator": _GENERATOR,
            "model_name": _MODEL_NAME,
            "model_path": str(self.config.model_dir),
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": self.config.width,
            "height": self.config.height,
            "steps": self.config.steps,
            "guidance_scale": self.config.guidance_scale,
            "elapsed_ms": elapsed_ms,
            "fallback_used": fallback_used,
        }

    def _artifact(
        self,
        *,
        entity_key: str,
        candidate_id: str,
        image_bytes: bytes,
        metadata: dict[str, Any],
        model_call: dict[str, Any],
    ) -> GeneratedImageArtifact:
        return GeneratedImageArtifact(
            entity_key=entity_key,
            candidate_id=candidate_id,
            relative_image_path=self._relative_image_path(candidate_id),
            relative_metadata_path=self._relative_metadata_path(candidate_id),
            image_bytes=image_bytes,
            metadata=metadata,
            model_call=model_call,
        )

    def generate(
        self,
        entity_key: str,
        candidate_id: str,
        positive_prompt: str,
        negative_prompt: str,
    ) -> GeneratedImageArtifact:
        """Generate a PNG artifact or fail the fixed image-generation mainline."""

        pipeline = self._get_pipeline()
        start_time = time.perf_counter()
        seed = _candidate_seed(self.config.seed, candidate_id)
        kwargs: dict[str, Any] = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": self.config.width,
            "height": self.config.height,
            "num_inference_steps": self.config.steps,
            "guidance_scale": self.config.guidance_scale,
        }
        _attach_generator(kwargs, self.config, seed)

        try:
            result = pipeline(**kwargs)
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            metadata = self._base_record(
                entity_key=entity_key,
                candidate_id=candidate_id,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                status="generated",
                elapsed_ms=elapsed_ms,
                fallback_used=False,
                seed=seed,
            )
            model_call = dict(metadata)
            return self._artifact(
                entity_key=entity_key,
                candidate_id=candidate_id,
                image_bytes=_png_bytes(result.images[0]),
                metadata=metadata,
                model_call=model_call,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SDXL generation failed for candidate {candidate_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
