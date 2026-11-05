"""Factory for lazily building Stage 4 model-fusion ranking engines."""

from __future__ import annotations

import base64
import asyncio
from io import BytesIO
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib import error, request
from urllib.parse import urlsplit

from mmkgc.ranking.profiles import (
    LocalScorerConfig,
    Stage4Profile,
    load_stage4_profile,
    preflight_stage4_profile,
)

if TYPE_CHECKING:
    from mmkgc.ranking.engine import ModelFusionRankingEngine


def build_model_fusion_engine(
    project_root: str | Path,
    stage4_config: str | Path,
) -> "ModelFusionRankingEngine":
    """Build a model-fusion engine only after profile validation and preflight."""
    root = Path(project_root).resolve()
    config_path = _resolve_under_root(stage4_config, root)
    profile = load_stage4_profile(config_path, project_root=root)
    failures = preflight_stage4_profile(profile, project_root=root)
    if failures:
        raise RuntimeError(
            "Stage 4 model-fusion preflight failed: " + "; ".join(failures)
        )
    from mmkgc.ranking.engine import ModelFusionRankingEngine

    return ModelFusionRankingEngine(
        profile=profile,
        scorers=_build_scorers(profile, project_root=root),
    )


def _build_scorers(
    profile: Stage4Profile,
    *,
    project_root: Path,
) -> dict[str, Any]:
    from mmkgc.adapters.qwen import load_qwen_config
    from mmkgc.evaluation.blip_itm import BlipITMConfig, BlipITMLocalScorer
    from mmkgc.evaluation.image_quality import ClipLocalConfig, ClipLocalImageQualityEvaluator
    from mmkgc.ranking.scorers.bertscore import BERTScoreAuxScorer
    from mmkgc.ranking.scorers.blip_itm import BLIPITMRankingScorer
    from mmkgc.ranking.scorers.clip import CLIPRankingScorer
    from mmkgc.ranking.scorers.gpt4omini_judge import GPT4OMiniRankingJudge
    from mmkgc.ranking.scorers.sentence_bert import SentenceBERTScorer

    scorers: dict[str, Any] = {}
    if profile.sentence_bert.enabled:
        scorers["sentence_bert"] = SentenceBERTScorer(
            _required_model_dir(profile.sentence_bert, "sentence_bert"),
            device=profile.sentence_bert.device,
        )
    if profile.bertscore.enabled:
        scorers["bertscore"] = BERTScoreAuxScorer(
            model_type=profile.bertscore.model_dir or "roberta-large",
            device=profile.bertscore.device,
        )
    if profile.clip.enabled:
        clip_evaluator = ClipLocalImageQualityEvaluator(
            ClipLocalConfig(
                model_dir=_required_model_dir(profile.clip, "clip"),
                device=profile.clip.device or "cuda",
                project_root=project_root,
            )
        )
        scorers["clip"] = CLIPRankingScorer(
            clip_evaluator,
            model_version=str(profile.clip.model_dir or ""),
        )
    if profile.blip_itm.enabled:
        blip_evaluator = BlipITMLocalScorer(
            BlipITMConfig(
                model_dir=_required_model_dir(profile.blip_itm, "blip_itm"),
                device=profile.blip_itm.device or "cuda",
                project_root=project_root,
            )
        )
        scorers["blip_itm"] = BLIPITMRankingScorer(
            blip_evaluator,
            model_version=str(profile.blip_itm.model_dir or ""),
        )
    judge_config = load_qwen_config(profile.gpt4omini_judge.model_config)
    scorers["gpt4omini_judge"] = GPT4OMiniRankingJudge(
        _OpenAICompatibleCompletionClient(
            judge_config,
            timeout_seconds=profile.gpt4omini_judge.timeout_seconds,
        ),
        model_name=judge_config.model_name,
        max_retries=profile.gpt4omini_judge.max_retries,
        retry_backoff_seconds=profile.gpt4omini_judge.retry_backoff_seconds,
        use_rubrics=True,
    )
    return scorers


def _required_model_dir(config: LocalScorerConfig, name: str) -> Path:
    if config.model_dir is None:
        raise RuntimeError(f"{name}.model_dir is required")
    return config.model_dir


def _resolve_under_root(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"stage4_config must be under project root {root}: {resolved}") from exc
    return resolved


def _compact_judge_image_url(image_data_url: str) -> str:
    """Reduce large RGB PNG uploads without resizing the image or its artifact."""
    prefix = "data:image/png;base64,"
    if not image_data_url.startswith(prefix):
        return image_data_url
    payload = base64.b64decode(image_data_url[len(prefix):], validate=True)
    if len(payload) <= 512 * 1024:
        return image_data_url
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        # Preserve transparency and other modes exactly; SDXL artifacts are RGB.
        if image.mode != "RGB":
            return image_data_url
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    compact = buffer.getvalue()
    if len(compact) >= len(payload):
        return image_data_url
    return "data:image/jpeg;base64," + base64.b64encode(compact).decode("ascii")


class CompletionResponseError(RuntimeError):
    """Provider completion failure, distinct from parsing a complete judgment."""
    def __init__(self, code, choice, payload):
        super().__init__("ranking judge completion failed: " + code)
        self.code = code
        message = choice.get("message")
        message = message if isinstance(message, dict) else {}
        self.details = {"completion_failure": code, "finish_reason": choice.get("finish_reason"),
                        "response_text": message.get("content"),
                        "usage": payload.get("usage", {})}


class _OpenAICompatibleCompletionClient:
    """Minimal GPT-4o mini judge client that never exposes API keys in outputs."""

    def __init__(self, config: Any, *, timeout_seconds: float) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max(int(self.config.max_token), 1024)

    def complete(self, prompt: str, *, response_schema: dict[str, Any] | None = None) -> str:
        return self._chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON API for ranking. Return only one "
                        "valid JSON object that matches the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ], response_schema=response_schema,
        )

    def complete_multimodal(self, prompt: str, image_data_url: str, *, response_schema: dict[str, Any] | None = None) -> str:
        image_data_url = _compact_judge_image_url(image_data_url)
        return self._chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON API for visual ranking. Return only "
                        "one valid JSON object that matches the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                },
            ], response_schema=response_schema,
        )

    def complete_structured(self, prompt: str, response_schema: dict[str, Any]) -> str:
        return self.complete(prompt, response_schema=response_schema)

    def complete_multimodal_structured(self, prompt: str, image_data_url: str, response_schema: dict[str, Any]) -> str:
        return self.complete_multimodal(prompt, image_data_url, response_schema=response_schema)

    def _chat_completion(self, messages: list[dict[str, Any]], *, response_schema: dict[str, Any] | None = None) -> str:
        endpoint = f"{str(self.config.base_url).rstrip('/')}/chat/completions"
        payload = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.max_tokens,
        }
        if response_schema is not None:
            if ("qwen" in self.config.model_name.lower()
                    and urlsplit(self.config.base_url).hostname in {"127.0.0.1", "localhost", "::1"}):
                from mmkgc.adapters.json_contract import decoder_safe_json_schema
                response_schema = decoder_safe_json_schema(response_schema)
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "stage4_judgment", "strict": True, "schema": response_schema,
            }}
            if getattr(self.config, 'api_response_format', 'json_schema') == 'json_object':
                payload['response_format'] = {'type': 'json_object'}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            payload = self._open_json_with_timeout(req)
        except error.HTTPError as exc:
            if exc.code in {408, 429, 500, 502, 503, 504}:
                raise error.HTTPError(
                    endpoint, exc.code, "ranking judge service temporarily unavailable",
                    exc.headers, None,
                ) from exc
            raise RuntimeError(f"GPT-4o mini ranking judge HTTP error: {exc.code}") from exc
        if not isinstance(payload, dict):
            raise CompletionResponseError("invalid_provider_response", {}, {})
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise CompletionResponseError("invalid_provider_response", {}, payload)
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise CompletionResponseError("output_token_limit", choice, payload)
        message = choice.get("message")
        if not isinstance(message, dict):
            raise CompletionResponseError("invalid_provider_response", {}, payload)
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            raise CompletionResponseError("provider_refusal", choice, payload)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CompletionResponseError("empty_response", choice, payload)
        return content

    def _open_json_with_timeout(self, req: request.Request) -> dict[str, Any]:
        # Use the same cancellable request deadline on Windows, POSIX, and
        # worker threads, including when the server keeps trickling response data.
        return asyncio.run(self._open_json_async(req))

    async def _open_json_async(self, req: request.Request) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("GPT requests require httpx; install requirements.txt") from exc

        async def fetch():
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds, follow_redirects=True,
                ) as client:
                    response = await client.request(
                        req.get_method(), req.full_url, content=req.data,
                        headers=dict(req.header_items()),
                    )
                    if response.status_code >= 400:
                        raise error.HTTPError(
                            req.full_url, response.status_code, response.reason_phrase,
                            response.headers, None,
                        )
                    return response.json()
            except httpx.TimeoutException as exc:
                raise TimeoutError("ranking judge request timed out") from exc
            except httpx.RequestError as exc:
                raise error.URLError(f"ranking judge transport error: {type(exc).__name__}") from exc

        try:
            return await asyncio.wait_for(fetch(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("ranking judge request timed out") from exc
