"""Factory for lazily building Stage 4 model-fusion ranking engines."""

from __future__ import annotations

import base64
import asyncio
from io import BytesIO
import json
import signal
import threading
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib import error, request

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


class _OpenAICompatibleCompletionClient:
    """Minimal GPT-4o mini judge client that never exposes API keys in outputs."""

    def __init__(self, config: Any, *, timeout_seconds: float) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max(int(self.config.max_token), 1024)

    def complete(self, prompt: str) -> str:
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
            ]
        )

    def complete_multimodal(self, prompt: str, image_data_url: str) -> str:
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
            ]
        )

    def _chat_completion(self, messages: list[dict[str, Any]]) -> str:
        endpoint = f"{str(self.config.base_url).rstrip('/')}/chat/completions"
        body = json.dumps(
            {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
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
        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return str(content)

    def _open_json_with_timeout(self, req: request.Request) -> dict[str, Any]:
        if threading.current_thread() is not threading.main_thread():
            # SIGALRM is unavailable in worker threads. Async cancellation keeps
            # a total request deadline, including a response that trickles data.
            return asyncio.run(self._open_json_async(req))
        previous_timer: tuple[float, float] | None = None
        previous_handler: Any | None = None
        use_hard_timeout = threading.current_thread() is threading.main_thread()

        def _raise_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError("ranking judge request timed out")

        if use_hard_timeout:
            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_timer = signal.setitimer(
                signal.ITIMER_REAL,
                float(self.timeout_seconds),
            )
            signal.signal(signal.SIGALRM, _raise_timeout)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        finally:
            if use_hard_timeout:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    previous_timer[0] if previous_timer is not None else 0.0,
                    previous_timer[1] if previous_timer is not None else 0.0,
                )
                if previous_handler is not None:
                    signal.signal(signal.SIGALRM, previous_handler)

    async def _open_json_async(self, req: request.Request) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Concurrent GPT requests require httpx; install requirements.txt") from exc

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
