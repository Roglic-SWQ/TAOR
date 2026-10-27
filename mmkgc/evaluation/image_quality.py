from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mmkgc.evaluation.image_quality_targets import (
    ImageQualityTargetError,
    build_entity_evidence_summary,
    collect_entity_image_quality_targets,
    target_entity_key,
    target_entity_label,
)
from mmkgc.evaluation.loader import RunEntityArtifacts


MMKGC_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_CLIP_MODELS = {
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-L/14": "ViT-L-14.pt",
}
CLIP_SCORE_FIELDS = (
    "clip_prompt_similarity",
    "clip_text_similarity",
    "clip_evidence_similarity",
    "clip_overall_image_quality_score",
)


class ClipImageQualityError(ImageQualityTargetError):
    """Raised when CLIP image quality evaluation cannot run safely."""


class ImageTextSimilarityScorer(Protocol):
    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        """Return one non-negative percentage score per text target."""


@dataclass
class FixedImageTextSimilarityScorer:
    scores: list[Any]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        call_index = len(self.calls)
        self.calls.append({"image_path": image_path, "texts": texts})
        scores = self._scores_for_call(call_index)
        if len(scores) != len(texts):
            raise ClipImageQualityError(
                f"fake scorer expected {len(scores)} texts, got {len(texts)}"
            )
        return [round(float(score), 6) for score in scores]

    def _scores_for_call(self, call_index: int) -> list[Any]:
        if self.scores and isinstance(self.scores[0], list | tuple):
            if call_index >= len(self.scores):
                raise ClipImageQualityError(
                    f"fake scorer has no score row for call {call_index + 1}"
                )
            return list(self.scores[call_index])
        return self.scores


def clip_weight_filename(model_name: str) -> str:
    try:
        return SUPPORTED_CLIP_MODELS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_CLIP_MODELS))
        raise ClipImageQualityError(
            f"unsupported clip_model_name {model_name!r}; supported: {supported}"
        ) from exc


def validate_clip_model_dir(
    model_dir: str | Path,
    *,
    project_root: str | Path = MMKGC_ROOT,
) -> Path:
    root = Path(project_root).resolve()
    candidate = Path(model_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ClipImageQualityError(
            f"clip_model_dir must be under project root {root}: {resolved}"
        ) from exc

    if not resolved.exists():
        raise ClipImageQualityError(f"clip_model_dir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ClipImageQualityError(f"clip_model_dir is not a directory: {resolved}")

    return resolved


def validate_clip_weight_path(model_dir: str | Path, model_name: str) -> Path:
    filename = clip_weight_filename(model_name)
    resolved_model_dir = Path(model_dir).resolve()
    weight_path = resolved_model_dir / filename

    if not weight_path.exists():
        raise ClipImageQualityError(f"missing CLIP weight: {weight_path}")
    if not weight_path.is_file():
        raise ClipImageQualityError(f"CLIP weight is not a file: {weight_path}")

    resolved_weight_path = weight_path.resolve()
    try:
        resolved_weight_path.relative_to(resolved_model_dir)
    except ValueError as exc:
        raise ClipImageQualityError(
            f"CLIP weight must stay under clip_model_dir {resolved_model_dir}: "
            f"{resolved_weight_path}"
        ) from exc

    return resolved_weight_path


@dataclass
class ClipLocalConfig:
    model_name: str = "ViT-B/32"
    model_dir: Path = Path("CLIP")
    device: str = "cuda"
    project_root: Path = MMKGC_ROOT


class ClipLocalImageQualityEvaluator:
    def __init__(
        self,
        config: ClipLocalConfig | None = None,
        *,
        clip_module: Any | None = None,
        model: Any | None = None,
        preprocess: Any | None = None,
    ) -> None:
        self.config = config or ClipLocalConfig()
        self.model_dir = validate_clip_model_dir(
            self.config.model_dir,
            project_root=self.config.project_root,
        )
        self.weight_path = validate_clip_weight_path(
            self.model_dir,
            self.config.model_name,
        )
        self._clip_module: Any | None = clip_module
        self._model: Any | None = model
        self._preprocess: Any | None = preprocess

    def _load_clip(self) -> tuple[Any, Any]:
        if self._model is None or self._preprocess is None:
            if self._clip_module is None:
                import clip

                self._clip_module = clip
            self._model, self._preprocess = self._clip_module.load(
                str(self.weight_path),
                device=self.config.device,
                jit=False,
            )
            self._model.eval()

        return self._model, self._preprocess

    def encode_pil_image(self, image: Any) -> list[float]:
        model, preprocess = self._load_clip()

        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise ClipImageQualityError(
                "torch is required for CLIP image encoding"
            ) from exc

        image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(
            self.config.device
        )
        with torch.no_grad():
            image_features = F.normalize(model.encode_image(image_input).float(), dim=-1)

        return [round(float(value), 6) for value in image_features[0].detach().cpu().tolist()]

    def score_pil_texts(self, image: Any, texts: list[str]) -> list[float]:
        if not texts:
            return []

        model, preprocess = self._load_clip()
        if self._clip_module is None:
            raise ClipImageQualityError("CLIP module is not loaded")

        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise ClipImageQualityError(
                "torch is required for CLIP scoring"
            ) from exc

        image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(
            self.config.device
        )
        text_input = self._clip_module.tokenize(texts, truncate=True).to(
            self.config.device
        )

        with torch.no_grad():
            image_features = F.normalize(model.encode_image(image_input).float(), dim=-1)
            text_features = F.normalize(model.encode_text(text_input).float(), dim=-1)
            similarities = image_features @ text_features.T

        values = similarities[0].detach().cpu().tolist()
        return [round(max(0.0, float(value)) * 100.0, 6) for value in values]

    def score_image_texts(self, image_path: Path, texts: list[str]) -> list[float]:
        if not texts:
            return []

        path = Path(image_path)
        if not path.is_file():
            raise ClipImageQualityError(f"image file does not exist: {path}")

        try:
            from PIL import Image
        except Exception as exc:
            raise ClipImageQualityError(
                "Pillow is required for CLIP image scoring"
            ) from exc

        with Image.open(path) as image:
            return self.score_pil_texts(image.convert("RGB"), texts)


def evaluate_entity_image_quality(
    artifacts: RunEntityArtifacts,
    *,
    scorer: ImageTextSimilarityScorer,
    clip_model_name: str,
    clip_weight_path: Path,
) -> dict[str, Any]:
    targets = collect_entity_image_quality_targets(
        artifacts,
        error_cls=ClipImageQualityError,
    )
    image_scores: list[dict[str, Any]] = []

    for target in targets:
        target_pairs = [
            ("clip_prompt_similarity", target.prompt_text),
            ("clip_text_similarity", target.candidate_text),
            ("clip_evidence_similarity", target.evidence_summary_text),
        ]
        available_targets = [(field, text) for field, text in target_pairs if text]
        if not available_targets:
            raise ClipImageQualityError(
                "no text targets available for image candidate "
                f"{target.image_candidate_id}"
            )

        texts = [text for _, text in available_targets]
        scores = scorer.score_image_texts(target.image_path, texts)
        if len(scores) != len(texts):
            raise ClipImageQualityError(
                f"scorer returned {len(scores)} scores for {len(texts)} text targets"
            )

        score_by_field: dict[str, float | None] = {
            "clip_prompt_similarity": None,
            "clip_text_similarity": None,
            "clip_evidence_similarity": None,
        }
        for (field, _text), score in zip(available_targets, scores):
            score_by_field[field] = round(float(score), 6)

        available_scores = [
            score for score in score_by_field.values() if score is not None
        ]
        overall = round(sum(available_scores) / len(available_scores), 6)

        image_scores.append(
            {
                "entity_key": target.entity_key,
                "image_candidate_id": target.image_candidate_id,
                "image_path": target.relative_image_path,
                "clip_model_name": clip_model_name,
                "clip_weight_path": str(clip_weight_path),
                "clip_prompt_similarity": score_by_field["clip_prompt_similarity"],
                "clip_text_similarity": score_by_field["clip_text_similarity"],
                "clip_evidence_similarity": score_by_field[
                    "clip_evidence_similarity"
                ],
                "clip_overall_image_quality_score": overall,
                "text_targets": {
                    "prompt_text": target.prompt_text,
                    "candidate_text": target.candidate_text,
                    "evidence_summary_text": target.evidence_summary_text,
                },
            }
        )

    averages = {
        field: _average_score([score[field] for score in image_scores])
        for field in CLIP_SCORE_FIELDS
    }
    best = _extreme_image_candidate_id(image_scores, reverse=True)
    worst = _extreme_image_candidate_id(image_scores, reverse=False)

    return {
        "entity": {
            "entity_key": target_entity_key(artifacts),
            "label": target_entity_label(artifacts),
            "missing_mode": artifacts.final.get("missing_mode"),
        },
        "image_count": len(image_scores),
        "clip_model_name": clip_model_name,
        "clip_weight_path": str(clip_weight_path),
        "averages": averages,
        "best_image_candidate_id": best,
        "worst_image_candidate_id": worst,
        "image_scores": image_scores,
    }


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
        key=lambda score: score["clip_overall_image_quality_score"],
        reverse=reverse,
    )[0]
    return selected["image_candidate_id"]
