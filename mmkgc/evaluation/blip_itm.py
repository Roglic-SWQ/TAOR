from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mmkgc.evaluation.image_quality_targets import (
    ImageQualityTargetError,
    collect_entity_image_quality_targets,
    target_entity_key,
    target_entity_label,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


MMKGC_ROOT = Path(__file__).resolve().parents[2]
BLIP_ITM_PROVIDER = "blip-itm-local"
REQUIRED_BLIP_MODEL_FILES = (
    "config.json",
    "pytorch_model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
REQUIRED_BLIP_ARCHITECTURE = "BlipForImageTextRetrieval"
BLIP_ITM_SCORE_FIELDS = (
    "blip_prompt_itm_score",
    "blip_text_itm_score",
    "blip_evidence_itm_score",
    "blip_overall_itm_score",
)
BlipProcessor: Any | None = None
BlipForImageTextRetrieval: Any | None = None


class BLIPITMError(ImageQualityTargetError):
    """Raised when local BLIP/ITM evaluation cannot run safely."""


class BLIPITMScorer(Protocol):
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        """Return one 0-100 ITM score per text target."""


@dataclass
class BlipITMConfig:
    model_dir: Path = Path("blip-itm-large-coco")
    device: str = "cuda"
    project_root: Path = MMKGC_ROOT


@dataclass
class FixedBLIPITMScorer:
    scores: list[Any]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        call_index = len(self.calls)
        self.calls.append({"image_path": Path(image_path), "texts": list(texts)})
        scores = self._scores_for_call(call_index)
        if len(scores) != len(texts):
            raise BLIPITMError(
                f"fake scorer expected {len(texts)} texts, got {len(scores)}"
            )
        return [round(float(score), 6) for score in scores]

    def _scores_for_call(self, call_index: int) -> list[Any]:
        if self.scores and isinstance(self.scores[0], (list, tuple)):
            if call_index >= len(self.scores):
                raise BLIPITMError(
                    f"fake scorer has no score row for call {call_index + 1}"
                )
            return list(self.scores[call_index])
        return list(self.scores)


class BlipITMLocalScorer:
    def __init__(
        self,
        config: BlipITMConfig | None = None,
        *,
        processor: Any | None = None,
        model: Any | None = None,
        processor_cls: Any | None = None,
        model_cls: Any | None = None,
    ) -> None:
        self.config = config or BlipITMConfig()
        self.model_dir = validate_blip_model_dir(
            self.config.model_dir,
            project_root=self.config.project_root,
        )
        self._processor: Any | None = processor
        self._model: Any | None = model
        self._processor_cls: Any | None = processor_cls
        self._model_cls: Any | None = model_cls

        if self._model is not None:
            self._model.to(self.config.device)
            self._model.eval()

    def _load_blip(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        try:
            if self._processor_cls is None or self._model_cls is None:
                processor_cls, model_cls = _load_blip_classes()
                self._processor_cls = processor_cls
                self._model_cls = model_cls

            self._processor = self._processor_cls.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
            self._model = self._model_cls.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
            self._model.to(self.config.device)
            self._model.eval()
        except Exception as exc:
            raise BLIPITMError(
                f"failed to load local BLIP/ITM model from {self.model_dir}: {exc}"
            ) from exc

        return self._processor, self._model

    def score_pil_texts(self, image: Any, texts: list[str]) -> list[float]:
        if not texts:
            return []

        processor, model = self._load_blip()

        try:
            import torch
        except Exception as exc:
            raise BLIPITMError(
                "torch is required for BLIP/ITM scoring"
            ) from exc

        scores: list[float] = []
        rgb_image = image.convert("RGB")
        for text in texts:
            inputs = processor(
                images=rgb_image,
                text=text,
                return_tensors="pt",
                padding=True,
            )
            inputs = _move_inputs_to_device(inputs, self.config.device)

            with torch.no_grad():
                outputs = model(**inputs)

            itm_score = getattr(outputs, "itm_score", None)
            if itm_score is None:
                raise BLIPITMError("BLIP output does not contain itm_score")

            if getattr(itm_score, "shape", None) is None or itm_score.shape[-1] < 2:
                raise BLIPITMError(
                    "BLIP itm_score must contain at least two class logits"
                )

            probabilities = torch.softmax(itm_score, dim=-1)
            probability = probabilities[..., 1].reshape(-1)[0].item()
            scores.append(_probability_to_score(probability))

        return scores

    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        if not texts:
            return []

        path = Path(image_path)
        if not path.is_file():
            raise BLIPITMError(f"image file does not exist: {path}")

        try:
            from PIL import Image
        except Exception as exc:
            raise BLIPITMError(
                "Pillow is required for BLIP/ITM scoring"
            ) from exc

        with Image.open(path) as image:
            return self.score_pil_texts(image.convert("RGB"), texts)


def validate_blip_model_dir(
    model_dir: str | Path,
    *,
    project_root: str | Path = MMKGC_ROOT,
) -> Path:
    """Return a safe project-local BLIP/ITM model directory."""
    root = Path(project_root).resolve()
    candidate = Path(model_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BLIPITMError(
            f"blip_model_dir must be under project root {root}: {resolved}"
        ) from exc

    if not resolved.exists():
        raise BLIPITMError(f"blip_model_dir does not exist: {resolved}")
    if not resolved.is_dir():
        raise BLIPITMError(f"blip_model_dir is not a directory: {resolved}")

    for filename in REQUIRED_BLIP_MODEL_FILES:
        _validate_required_model_file(resolved, filename)

    _validate_blip_architecture(resolved / "config.json")
    return resolved


def evaluate_entity_blip_itm(
    artifacts: RunEntityArtifacts,
    scorer: BLIPITMScorer,
    model_dir: str | Path,
) -> dict[str, Any]:
    """Score all generated image candidates for one entity."""
    targets = collect_entity_image_quality_targets(
        artifacts,
        error_cls=BLIPITMError,
    )
    model_dir_text = str(model_dir)
    image_scores: list[dict[str, Any]] = []

    for target in targets:
        target_pairs = [
            ("blip_prompt_itm_score", target.prompt_text),
            ("blip_text_itm_score", target.candidate_text),
            ("blip_evidence_itm_score", target.evidence_summary_text),
        ]
        available_targets = [(field, text) for field, text in target_pairs if text]
        if not available_targets:
            raise BLIPITMError(
                "no text targets available for image candidate "
                f"{target.image_candidate_id}"
            )

        texts = [text for _, text in available_targets]
        try:
            scores = scorer.score_image_texts(target.image_path, texts)
        except BLIPITMError:
            raise
        except Exception as exc:
            raise BLIPITMError(
                "BLIP/ITM scorer failed for image candidate "
                f"{target.image_candidate_id}: {exc}"
            ) from exc

        if len(scores) != len(texts):
            raise BLIPITMError(
                f"scorer returned {len(scores)} scores for {len(texts)} text targets"
            )

        score_by_field: dict[str, float | None] = {
            "blip_prompt_itm_score": None,
            "blip_text_itm_score": None,
            "blip_evidence_itm_score": None,
        }
        for (field, _text), score in zip(available_targets, scores):
            score_by_field[field] = _round_score(
                score,
                field=field,
                image_candidate_id=target.image_candidate_id,
            )

        available_scores = [
            score for score in score_by_field.values() if score is not None
        ]
        overall = round(sum(available_scores) / len(available_scores), 6)

        image_scores.append(
            {
                "entity_key": target.entity_key,
                "image_candidate_id": target.image_candidate_id,
                "image_path": target.relative_image_path,
                "provider": BLIP_ITM_PROVIDER,
                "model_dir": model_dir_text,
                "status": "scored",
                "fallback_used": False,
                "blip_prompt_itm_score": score_by_field["blip_prompt_itm_score"],
                "blip_text_itm_score": score_by_field["blip_text_itm_score"],
                "blip_evidence_itm_score": score_by_field[
                    "blip_evidence_itm_score"
                ],
                "blip_overall_itm_score": overall,
                "text_targets": {
                    "prompt_text": target.prompt_text,
                    "candidate_text": target.candidate_text,
                    "evidence_summary_text": target.evidence_summary_text,
                },
            }
        )

    averages = {
        field: _average_score([score[field] for score in image_scores])
        for field in BLIP_ITM_SCORE_FIELDS
    }

    return {
        "entity": {
            "entity_key": target_entity_key(artifacts),
            "label": target_entity_label(artifacts),
            "missing_mode": artifacts.final.get("missing_mode"),
        },
        "image_count": len(image_scores),
        "model_dir": model_dir_text,
        "averages": averages,
        "best_image_candidate_id": _extreme_image_candidate_id(
            image_scores,
            reverse=True,
        ),
        "worst_image_candidate_id": _extreme_image_candidate_id(
            image_scores,
            reverse=False,
        ),
        "image_scores": image_scores,
    }


def _validate_required_model_file(model_dir: Path, filename: str) -> Path:
    file_path = model_dir / filename
    resolved_file_path = file_path.resolve()
    try:
        resolved_file_path.relative_to(model_dir)
    except ValueError as exc:
        raise BLIPITMError(
            f"BLIP model file must stay under blip_model_dir {model_dir}: "
            f"{resolved_file_path}"
        ) from exc

    if not resolved_file_path.exists():
        raise BLIPITMError(f"missing BLIP model file: {file_path}")
    if not resolved_file_path.is_file():
        raise BLIPITMError(f"BLIP model path is not a file: {file_path}")
    return resolved_file_path


def _load_blip_classes() -> tuple[Any, Any]:
    global BlipForImageTextRetrieval, BlipProcessor

    if BlipProcessor is None or BlipForImageTextRetrieval is None:
        from transformers import (
            BlipForImageTextRetrieval as TransformersBlipForImageTextRetrieval,
            BlipProcessor as TransformersBlipProcessor,
        )

        BlipProcessor = TransformersBlipProcessor
        BlipForImageTextRetrieval = TransformersBlipForImageTextRetrieval

    return BlipProcessor, BlipForImageTextRetrieval


def _validate_blip_architecture(config_path: Path) -> None:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BLIPITMError(f"failed to read BLIP config.json: {exc}") from exc

    if not isinstance(config, dict):
        raise BLIPITMError("BLIP config.json must contain a JSON object")

    architectures = config.get("architectures")
    if isinstance(architectures, str):
        architecture_values = [architectures]
    elif isinstance(architectures, list):
        architecture_values = [str(value) for value in architectures]
    else:
        architecture_values = []

    if REQUIRED_BLIP_ARCHITECTURE not in architecture_values:
        raise BLIPITMError(
            "BLIP config.json architectures must include "
            f"{REQUIRED_BLIP_ARCHITECTURE}"
        )


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _probability_to_score(probability: Any) -> float:
    try:
        value = float(probability)
    except (TypeError, ValueError) as exc:
        raise BLIPITMError(
            f"invalid BLIP/ITM probability: {probability!r}"
        ) from exc

    if not math.isfinite(value):
        raise BLIPITMError(
            f"invalid BLIP/ITM probability: probability must be finite, "
            f"got {probability!r}"
        )

    value = max(0.0, min(1.0, value))
    return round(value * 100.0, 6)


def _round_score(
    score: Any,
    *,
    field: str,
    image_candidate_id: str,
) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise BLIPITMError(
            f"invalid BLIP/ITM score for {field} on image candidate "
            f"{image_candidate_id}: {score!r}"
        ) from exc

    if not math.isfinite(value):
        raise BLIPITMError(
            f"invalid BLIP/ITM score for {field} on image candidate "
            f"{image_candidate_id}: score must be finite, got {score!r}"
        )
    if not 0.0 <= value <= 100.0:
        raise BLIPITMError(
            f"invalid BLIP/ITM score for {field} on image candidate "
            f"{image_candidate_id}: score must be between 0 and 100, got {score!r}"
        )
    return round(value, 6)


def _average_score(values: list[Any]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _extreme_image_candidate_id(
    image_scores: list[dict[str, Any]],
    *,
    reverse: bool,
) -> str | None:
    if not image_scores:
        return None
    selected = sorted(
        image_scores,
        key=lambda score: score["blip_overall_itm_score"],
        reverse=reverse,
    )[0]
    return selected["image_candidate_id"]
