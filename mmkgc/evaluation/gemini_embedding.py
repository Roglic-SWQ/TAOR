from __future__ import annotations

import base64
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

from mmkgc.evaluation.image_quality_targets import (
    ImageQualityTargetError,
    build_entity_evidence_summary,
    collect_entity_image_quality_targets,
    target_entity_key,
    target_entity_label,
)
from mmkgc.evaluation.loader import RunEntityArtifacts

try:
    import yaml
except Exception:  # pragma: no cover - explicit fallback if PyYAML is unavailable.
    yaml = None


MMKGC_ROOT = Path(__file__).resolve().parents[2]
GEMINI_EMBEDDING_PROVIDER = "gemini-api"
GEMINI_TEXT_SCORE_FIELDS = (
    "gemini_text_ground_truth_similarity",
    "gemini_text_evidence_similarity",
    "gemini_overall_text_embedding_score",
)
GEMINI_IMAGE_SCORE_FIELDS = (
    "gemini_prompt_similarity",
    "gemini_text_similarity",
    "gemini_evidence_similarity",
    "gemini_overall_multimodal_score",
)
GEMINI_PAIR_SCORE_FIELDS = (
    "gemini_pair_image_text_similarity",
    "gemini_pair_image_evidence_similarity",
    "gemini_pair_text_evidence_similarity",
    "gemini_overall_pair_embedding_score",
)


class GeminiEmbeddingError(ImageQualityTargetError):
    """Raised when Gemini embedding configuration or evaluation fails."""


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    api_key_env: str
    api_key: str = field(repr=False)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    model_name: str = "gemini-embedding-2"
    output_dimensionality: int = 768
    timeout_seconds: int = 120
    max_retries: int = 2


class GeminiEmbeddingProtocol(Protocol):
    def embed_text(self, text: str) -> list[float]:
        """Return an embedding vector for text."""

    def embed_image(self, image_path: str | Path) -> list[float]:
        """Return an embedding vector for an image path."""


@dataclass
class FixedGeminiEmbeddingClient:
    embeddings_by_key: dict[str, list[float]]

    def __post_init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def embed_text(self, text: str) -> list[float]:
        key = f"text:{text}"
        self.calls.append({"kind": "text", "key": key})
        return self._embedding_for_key(key)

    def embed_image(self, image_path: str | Path) -> list[float]:
        key = f"image:{Path(image_path).resolve()}"
        self.calls.append({"kind": "image", "key": key})
        return self._embedding_for_key(key)

    def _embedding_for_key(self, key: str) -> list[float]:
        try:
            return [float(value) for value in self.embeddings_by_key[key]]
        except KeyError as exc:
            raise GeminiEmbeddingError(f"fake Gemini embedding missing key: {key}") from exc


class GeminiEmbeddingClient:
    """Minimal Gemini REST embedding client boundary."""

    def __init__(self, config: GeminiEmbeddingConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    def embed_text(self, text: str) -> list[float]:
        payload = {
            "content": {
                "parts": [{"text": f"task: sentence similarity | query: {text}"}],
            },
            "output_dimensionality": self.config.output_dimensionality,
        }
        return self._post_embedding(payload)

    def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        if not path.is_file():
            raise GeminiEmbeddingError(f"image file does not exist: {path}")
        payload = {
            "content": {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": _image_mime_type(path),
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                        }
                    }
                ]
            },
            "output_dimensionality": self.config.output_dimensionality,
        }
        return self._post_embedding(payload)

    def _post_embedding(self, payload: dict[str, Any]) -> list[float]:
        url = f"{self.base_url}/models/{self.config.model_name}:embedContent"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key,
        }
        last_exc: Exception | None = None
        attempts = max(0, self.config.max_retries) + 1

        for attempt in range(attempts):
            req = request.Request(url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    response_text = response.read().decode("utf-8")
                return _extract_embedding_values(response_text)
            except error.HTTPError as exc:
                last_exc = exc
                if not _should_retry_http(exc.code, attempt, attempts):
                    break
            except error.URLError as exc:
                last_exc = exc
            except GeminiEmbeddingError:
                raise
            except Exception as exc:
                last_exc = exc

            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4) * 0.25)

        message = _redact_sensitive(str(last_exc), secrets=(self.config.api_key,))
        raise GeminiEmbeddingError(f"Gemini embedding request failed: {message}")


def load_gemini_embedding_config(
    path: str | Path,
    *,
    project_root: str | Path = MMKGC_ROOT,
) -> GeminiEmbeddingConfig:
    if yaml is None:
        raise GeminiEmbeddingError(
            "PyYAML is required to load Gemini embedding config. Please install PyYAML."
        )

    root = Path(project_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    resolved = config_path.resolve()

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GeminiEmbeddingError(
            f"Gemini embedding config must be under project root {root}: {resolved}"
        ) from exc

    if not resolved.is_file():
        raise GeminiEmbeddingError(f"Gemini embedding config does not exist: {resolved}")

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        raise GeminiEmbeddingError(
            f"failed to read Gemini embedding config: {exc}"
        ) from exc

    return _coerce_config(data or {}, resolved)


def cosine_score(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise GeminiEmbeddingError(
            f"embedding dimensions differ: {len(vec_a)} != {len(vec_b)}"
        )
    if not vec_a:
        raise GeminiEmbeddingError("embedding vector must not be empty")

    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        raise GeminiEmbeddingError("embedding vector norm must be non-zero")

    similarity = dot / (norm_a * norm_b)
    return round(max(0.0, min(1.0, similarity)) * 100.0, 6)


def evaluate_entity_gemini_embedding(
    artifacts: RunEntityArtifacts,
    client: GeminiEmbeddingProtocol,
    model_name: str,
) -> dict[str, Any]:
    targets = collect_entity_image_quality_targets(
        artifacts,
        error_cls=GeminiEmbeddingError,
    )
    evidence_text = build_entity_evidence_summary(artifacts).strip()
    ground_truth_text = _ground_truth_text(artifacts)
    text_candidates = _ranked_text_candidates_by_id(artifacts)
    text_scores = _score_text_candidates(
        text_candidates,
        entity_key=target_entity_key(artifacts),
        ground_truth_text=ground_truth_text,
        evidence_text=evidence_text,
        client=client,
    )
    text_score_by_id = {score["text_candidate_id"]: score for score in text_scores}
    image_scores = _score_image_targets(
        targets,
        client=client,
    )
    target_by_image_id = {target.image_candidate_id: target for target in targets}
    pair_scores = _score_pair_candidates(
        artifacts,
        target_by_image_id=target_by_image_id,
        text_candidates=text_candidates,
        client=client,
        evidence_text=evidence_text,
    )

    return {
        "entity": {
            "entity_key": target_entity_key(artifacts),
            "label": target_entity_label(artifacts),
            "missing_mode": artifacts.final.get("missing_mode"),
        },
        "provider": GEMINI_EMBEDDING_PROVIDER,
        "gemini_model_name": model_name,
        "text_count": len(text_scores),
        "image_count": len(image_scores),
        "pair_count": len(pair_scores),
        "averages": {
            **{
                field: _average_score([score[field] for score in text_scores])
                for field in GEMINI_TEXT_SCORE_FIELDS
            },
            **{
                field: _average_score([score[field] for score in image_scores])
                for field in GEMINI_IMAGE_SCORE_FIELDS
            },
            **{
                field: _average_score([score[field] for score in pair_scores])
                for field in GEMINI_PAIR_SCORE_FIELDS
            },
        },
        "best_text_candidate_id": _extreme_id(
            text_scores,
            id_field="text_candidate_id",
            score_field="gemini_overall_text_embedding_score",
            reverse=True,
        ),
        "best_image_candidate_id": _extreme_id(
            image_scores,
            id_field="image_candidate_id",
            score_field="gemini_overall_multimodal_score",
            reverse=True,
        ),
        "best_pair_id": _extreme_id(
            pair_scores,
            id_field="pair_id",
            score_field="gemini_overall_pair_embedding_score",
            reverse=True,
        ),
        "text_scores": text_scores,
        "image_scores": image_scores,
        "pair_scores": pair_scores,
        "text_score_by_id": text_score_by_id,
    }


def _score_text_candidates(
    text_candidates: dict[str, dict[str, Any]],
    *,
    entity_key: str,
    ground_truth_text: str,
    evidence_text: str,
    client: GeminiEmbeddingProtocol,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for text_candidate_id, candidate in text_candidates.items():
        candidate_text = _text_from_candidate(candidate)
        if not candidate_text:
            continue

        candidate_embedding = _embed_text_cached(client, candidate_text)
        ground_truth_score = _similarity_to_text(
            client,
            candidate_embedding,
            ground_truth_text,
        )
        evidence_score = _similarity_to_text(
            client,
            candidate_embedding,
            evidence_text,
        )
        overall = _average_score([ground_truth_score, evidence_score])
        scores.append(
            {
                "entity_key": entity_key,
                "text_candidate_id": text_candidate_id,
                "status": "scored",
                "gemini_text_ground_truth_similarity": ground_truth_score,
                "gemini_text_evidence_similarity": evidence_score,
                "gemini_overall_text_embedding_score": overall,
                "text_targets": {
                    "candidate_text": candidate_text,
                    "ground_truth_text": ground_truth_text,
                    "evidence_summary_text": evidence_text,
                },
            }
        )
    return scores


def _score_image_targets(
    targets: list[Any],
    *,
    client: GeminiEmbeddingProtocol,
) -> list[dict[str, Any]]:
    image_scores: list[dict[str, Any]] = []
    for target in targets:
        image_embedding = _embed_image_cached(client, target.image_path)
        prompt_score = _similarity_to_text(client, image_embedding, target.prompt_text)
        text_score = _similarity_to_text(client, image_embedding, target.candidate_text)
        evidence_score = _similarity_to_text(
            client,
            image_embedding,
            target.evidence_summary_text,
        )
        overall = _average_score([prompt_score, text_score, evidence_score])
        image_scores.append(
            {
                "entity_key": target.entity_key,
                "image_candidate_id": target.image_candidate_id,
                "image_path": target.relative_image_path,
                "provider": GEMINI_EMBEDDING_PROVIDER,
                "status": "scored",
                "gemini_prompt_similarity": prompt_score,
                "gemini_text_similarity": text_score,
                "gemini_evidence_similarity": evidence_score,
                "gemini_overall_multimodal_score": overall,
                "text_targets": {
                    "prompt_text": target.prompt_text,
                    "candidate_text": target.candidate_text,
                    "evidence_summary_text": target.evidence_summary_text,
                },
            }
        )
    return image_scores


def _score_pair_candidates(
    artifacts: RunEntityArtifacts,
    *,
    target_by_image_id: dict[str, Any],
    text_candidates: dict[str, dict[str, Any]],
    client: GeminiEmbeddingProtocol,
    evidence_text: str,
) -> list[dict[str, Any]]:
    pair_ranking = _as_dict(artifacts.ranking.get("pair_ranking"))
    pair_scores: list[dict[str, Any]] = []
    for index, pair in enumerate(_as_list(pair_ranking.get("ranked_pairs"))):
        if not isinstance(pair, dict):
            continue
        pair_id = _first_text(pair.get("pair_id"), f"pair_{index:03d}")
        image_candidate_id = _pair_image_candidate_id(pair)
        target = target_by_image_id.get(image_candidate_id)
        if target is None:
            continue
        text_candidate_id = _pair_text_candidate_id(pair)
        candidate_text = _text_from_candidate(_as_dict(pair.get("text_candidate")))
        if not candidate_text and text_candidate_id:
            candidate_text = _text_from_candidate(text_candidates.get(text_candidate_id, {}))
        if not candidate_text:
            candidate_text = target.candidate_text

        image_embedding = _embed_image_cached(client, target.image_path)
        text_embedding = (
            _embed_text_cached(client, candidate_text) if candidate_text else None
        )
        evidence_embedding = (
            _embed_text_cached(client, evidence_text) if evidence_text else None
        )
        image_text = (
            cosine_score(image_embedding, text_embedding)
            if text_embedding is not None
            else None
        )
        image_evidence = (
            cosine_score(image_embedding, evidence_embedding)
            if evidence_embedding is not None
            else None
        )
        text_evidence = (
            cosine_score(text_embedding, evidence_embedding)
            if text_embedding is not None and evidence_embedding is not None
            else None
        )
        overall = _average_score([image_text, image_evidence, text_evidence])
        pair_scores.append(
            {
                "entity_key": target.entity_key,
                "pair_id": pair_id,
                "text_candidate_id": text_candidate_id,
                "image_candidate_id": image_candidate_id,
                "provider": GEMINI_EMBEDDING_PROVIDER,
                "status": "scored",
                "gemini_pair_image_text_similarity": image_text,
                "gemini_pair_image_evidence_similarity": image_evidence,
                "gemini_pair_text_evidence_similarity": text_evidence,
                "gemini_overall_pair_embedding_score": overall,
            }
        )
    return pair_scores


def _embed_text_cached(
    client: GeminiEmbeddingProtocol,
    text: str,
) -> list[float]:
    cache = _client_cache(client)
    key = ("text", text)
    if key not in cache:
        cache[key] = client.embed_text(text)
    return cache[key]


def _embed_image_cached(
    client: GeminiEmbeddingProtocol,
    image_path: Path,
) -> list[float]:
    resolved = Path(image_path).resolve()
    cache = _client_cache(client)
    key = ("image", str(resolved))
    if key not in cache:
        cache[key] = client.embed_image(resolved)
    return cache[key]


def _client_cache(client: GeminiEmbeddingProtocol) -> dict[tuple[str, str], list[float]]:
    cache = getattr(client, "_mmkgc_embedding_cache", None)
    if cache is None:
        cache = {}
        setattr(client, "_mmkgc_embedding_cache", cache)
    return cache


def _similarity_to_text(
    client: GeminiEmbeddingProtocol,
    embedding: list[float],
    text: str,
) -> float | None:
    if not text:
        return None
    return cosine_score(embedding, _embed_text_cached(client, text))


def _coerce_config(data: Any, source_path: Path) -> GeminiEmbeddingConfig:
    if not isinstance(data, dict):
        raise GeminiEmbeddingError(f"{source_path}: config content must be a mapping")
    gemini = data.get("gemini")
    if not isinstance(gemini, dict):
        raise GeminiEmbeddingError(f"{source_path}: missing gemini config section")

    api_key_env = _required_text(gemini, "api_key_env", source_path)
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise GeminiEmbeddingError(
            f"{source_path}: environment variable {api_key_env} is required"
        )

    return GeminiEmbeddingConfig(
        api_key_env=api_key_env,
        api_key=api_key,
        base_url=_optional_text(
            gemini,
            "base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/"),
        model_name=_optional_text(gemini, "model_name", "gemini-embedding-2"),
        output_dimensionality=_positive_int(
            gemini,
            "output_dimensionality",
            768,
            source_path,
        ),
        timeout_seconds=_positive_int(
            gemini,
            "timeout_seconds",
            120,
            source_path,
        ),
        max_retries=_non_negative_int(gemini, "max_retries", 2, source_path),
    )


def _extract_embedding_values(response_text: str) -> list[float]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise GeminiEmbeddingError(
            f"Gemini embedding response was invalid JSON: {exc.msg}"
        ) from exc
    embedding = _as_dict(payload.get("embedding"))
    values = embedding.get("values")
    if not isinstance(values, list) or not values:
        raise GeminiEmbeddingError("Gemini embedding response missing embedding.values")
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise GeminiEmbeddingError(
            "Gemini embedding response contained non-numeric values"
        ) from exc


def _image_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    raise GeminiEmbeddingError(
        f"unsupported image extension {path.suffix!r}; supported: png, jpg, jpeg"
    )


def _should_retry_http(code: int, attempt: int, attempts: int) -> bool:
    return attempt + 1 < attempts and code in {408, 429, 500, 502, 503, 504}


def _required_text(data: dict[str, Any], key: str, source_path: Path) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise GeminiEmbeddingError(f"{source_path}: missing required field gemini.{key}")
    return value


def _optional_text(data: dict[str, Any], key: str, default: str) -> str:
    value = str(data.get(key, default)).strip()
    return value or default


def _positive_int(
    data: dict[str, Any],
    key: str,
    default: int,
    source_path: Path,
) -> int:
    value = _int_config_value(data, key, default, source_path)
    if value <= 0:
        raise GeminiEmbeddingError(f"{source_path}: gemini.{key} must be positive")
    return value


def _non_negative_int(
    data: dict[str, Any],
    key: str,
    default: int,
    source_path: Path,
) -> int:
    value = _int_config_value(data, key, default, source_path)
    if value < 0:
        raise GeminiEmbeddingError(f"{source_path}: gemini.{key} must be non-negative")
    return value


def _int_config_value(
    data: dict[str, Any],
    key: str,
    default: int,
    source_path: Path,
) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise GeminiEmbeddingError(
            f"{source_path}: gemini.{key} must be an integer"
        ) from exc


def _ground_truth_text(artifacts: RunEntityArtifacts) -> str:
    entity = _as_dict(artifacts.final.get("entity"))
    return _first_text(entity.get("description"), entity.get("ground_truth_description"))


def _ranked_text_candidates_by_id(
    artifacts: RunEntityArtifacts,
) -> dict[str, dict[str, Any]]:
    text_ranking = _as_dict(artifacts.ranking.get("text_ranking"))
    candidates: dict[str, dict[str, Any]] = {}
    for index, text_candidate in enumerate(
        _as_list(text_ranking.get("ranked_candidates")),
    ):
        if not isinstance(text_candidate, dict):
            continue
        candidate_id = _first_text(text_candidate.get("candidate_id"), str(index))
        candidates[candidate_id] = text_candidate
    return candidates


def _text_from_candidate(candidate: dict[str, Any]) -> str:
    generated_content = _as_dict(candidate.get("generated_content"))
    return _first_text(
        generated_content.get("value"),
        candidate.get("text"),
        candidate.get("value"),
    )


def _pair_image_candidate_id(pair: dict[str, Any]) -> str:
    image_candidate = _as_dict(pair.get("image_candidate"))
    return _first_text(
        image_candidate.get("candidate_id"),
        image_candidate.get("image_candidate_id"),
        pair.get("image_candidate_id"),
    )


def _pair_text_candidate_id(pair: dict[str, Any]) -> str:
    text_candidate = _as_dict(pair.get("text_candidate"))
    return _first_text(
        pair.get("text_candidate_id"),
        text_candidate.get("candidate_id"),
    )


def _average_score(values: list[Any]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round(sum(numeric_values) / len(numeric_values), 6)


def _extreme_id(
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    score_field: str,
    reverse: bool,
) -> str | None:
    scored_rows = [row for row in rows if row.get(score_field) is not None]
    if not scored_rows:
        return None
    selected = sorted(
        scored_rows,
        key=lambda row: row[score_field],
        reverse=reverse,
    )[0]
    return selected[id_field]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _redact_sensitive(value: Any, *, secrets: tuple[str, ...]) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text
