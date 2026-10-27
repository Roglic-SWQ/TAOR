from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError

import httpx
from PIL import Image
import pytest

from mmkgc.adapters.qwen import QwenConfig
from mmkgc.ranking.context import build_ranking_context
from mmkgc.ranking.contracts import GateResult, ScoreResult
from mmkgc.ranking.engine import ModelFusionRankingEngine
from mmkgc.ranking.factory import _OpenAICompatibleCompletionClient
from mmkgc.ranking.profiles import GPT4OMiniJudgeConfig, LocalScorerConfig, Stage4Profile
from mmkgc.ranking.scorers.gpt4omini_judge import GPT4OMiniJudgeParseError, GPT4OMiniRankingJudge


def context():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
    candidates = [dict(
        candidate_id=f"image_{i:03d}", target_modality="image", primary_relation="instance of",
        expected_relation_coverage=["instance of"], status="generated",
        generated_content={"type": "image", "image_artifact_id": f"image_{i:03d}"},
    ) for i in range(1, 6)]
    return build_ranking_context(
        "Q1", "Example", "image", {}, {"entity_profile": {"entity_type": "building"}},
        {}, {"relation_alternation_generation": {"image_candidates": candidates}},
        {candidate["candidate_id"]: {"image_bytes": buffer.getvalue()} for candidate in candidates},
    )


def profile(workers):
    disabled = LocalScorerConfig(False, False, None, None)
    enabled = LocalScorerConfig(True, True, Path("models/test"), "cpu")
    return Stage4Profile(
        "concurrency-test", disabled, disabled, enabled, enabled,
        GPT4OMiniJudgeConfig(True, Path("configs/test.yaml"), 1, max_concurrency=workers),
    )


class LocalVisualScorer:
    def score_image_quality_batch(self, image, texts):
        assert threading.current_thread() is threading.main_thread()
        assert image is not None
        return [ScoreResult(80, 0.8, "scored", "local-test") for _ in texts]


class ObservedJudge:
    model_name = "test-judge"

    def __init__(self, workers, failures=False):
        self.barrier = threading.Barrier(workers)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.finished = []
        self.workers = workers
        self.failures = failures

    def judge_image(self, context, candidate, image_input):
        index = int(candidate["candidate_id"].split("_")[-1])
        with self.lock:
            self.active += 1
            self.max_active = max(self.active, self.max_active)
        try:
            if index <= self.workers:
                self.barrier.wait(timeout=2)
            time.sleep(0.05 if index == 1 else 0.015)
            assert image_input.candidate_id == candidate["candidate_id"]
            if self.failures and index == 2:
                raise GPT4OMiniJudgeParseError("invalid test response")
            if self.failures and index == 3:
                raise TimeoutError("test request timeout")
            return SimpleNamespace(
                score=ScoreResult(70 + index, (70 + index) / 100, "scored", self.model_name),
                gate=GateResult("pass", self.model_name, reason="Test image checked."),
            )
        finally:
            with self.lock:
                self.active -= 1
                self.finished.append(candidate["candidate_id"])


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_bounded_judge_calls_keep_all_candidates_and_deterministic_audit_order(workers):
    judge = ObservedJudge(workers)
    engine = ModelFusionRankingEngine(profile=profile(workers), scorers={
        "gpt4omini_judge": judge, "clip": LocalVisualScorer(), "blip_itm": LocalVisualScorer(),
    })
    result = engine.rank(context())
    calls = [call for call in result["ranking_model_calls"] if call["scorer"] == "gpt4omini_judge"]
    assert judge.max_active == workers
    assert len(judge.finished) == 5
    assert [call["candidate_id"] for call in calls] == [f"image_{i:03d}" for i in range(1, 6)]
    assert all(call["latency_ms"] >= 10 for call in calls)
    assert len(result["image_ranking"]["ranked_candidates"]) == 5
    assert result["image_ranking"]["best_candidate_id"] == "image_005"
    if workers > 1:
        assert judge.finished[0] != "image_001"


def test_parallel_parse_and_service_failures_stay_with_their_candidates():
    judge = ObservedJudge(3, failures=True)
    engine = ModelFusionRankingEngine(profile=profile(3), scorers={
        "gpt4omini_judge": judge, "clip": LocalVisualScorer(), "blip_itm": LocalVisualScorer(),
    })
    result = engine.rank(context())
    rejected = {item["candidate_id"]: item for item in result["image_ranking"]["rejected_candidates"]}
    assert set(rejected) == {"image_002", "image_003"}
    assert rejected["image_002"]["hard_gate"]["hard_violations"] == ["malformed_candidate"]
    assert rejected["image_003"]["hard_gate"]["hard_violations"] == ["judge_service_error"]
    ranked = result["image_ranking"]["ranked_candidates"]
    assert len(ranked) == 5
    assert sum(item["hard_gate"]["decision"] == "pass" for item in ranked) == 3
    assert judge.max_active == 3


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_python_profile_also_requires_positive_integer_concurrency(bad):
    with pytest.raises(ValueError, match="max_concurrency"):
        replace(profile(1).gpt4omini_judge, max_concurrency=bad)


def mock_async_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(handler), trust_env=False,
    ))


def test_worker_http_preserves_model_headers_and_payload(monkeypatch):
    captured = []

    async def handle(req):
        captured.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "test-ok"}}]})

    mock_async_transport(monkeypatch, handle)
    client = _OpenAICompatibleCompletionClient(QwenConfig(
        "gpt-4o-mini", "http://test.invalid/v1", "private-test-key",
    ), timeout_seconds=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.complete, "Test prompt").result(timeout=2)
    assert result == "test-ok"
    assert captured[0].headers["authorization"] == "Bearer private-test-key"
    body = json.loads(captured[0].content)
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"][1]["content"] == "Test prompt"


def test_worker_total_deadline_cancels_a_slow_drip_response(monkeypatch):
    closed = []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for value in b'{"choices":[]}':
                await asyncio.sleep(0.025)
                yield bytes([value])

        async def aclose(self):
            closed.append(True)

    async def handle(req):
        return httpx.Response(200, stream=SlowBody())

    mock_async_transport(monkeypatch, handle)
    client = _OpenAICompatibleCompletionClient(QwenConfig(
        "gpt-4o-mini", "http://test.invalid/v1", "private-test-key",
    ), timeout_seconds=0.07)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(TimeoutError, match="request timed out"):
            executor.submit(client.complete, "Test prompt").result(timeout=2)
    assert closed


@pytest.mark.parametrize("code,retries", [(429, True), (503, True), (401, False)])
def test_worker_http_status_controls_bounded_retries(monkeypatch, code, retries):
    attempts = []
    delays = []

    async def handle(req):
        attempts.append(req)
        if len(attempts) == 1:
            return httpx.Response(code, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "test-ok"}}]})

    mock_async_transport(monkeypatch, handle)
    monkeypatch.setattr("mmkgc.ranking.scorers.gpt4omini_judge.time.sleep", delays.append)
    client = _OpenAICompatibleCompletionClient(QwenConfig(
        "gpt-4o-mini", "http://test.invalid/v1", "private-test-key",
    ), timeout_seconds=1)
    judge = GPT4OMiniRankingJudge(client, max_retries=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(judge._complete_with_retries, "Test", image_input=None, kind="image", candidate_id="image_001")
        if retries:
            assert future.result(timeout=2) == "test-ok"
        else:
            with pytest.raises(RuntimeError, match="401"):
                future.result(timeout=2)
    assert len(attempts) == (2 if retries else 1)
    assert delays == ([2.0] if retries else [])


def test_repeated_rate_limit_is_not_retried_without_a_bound(monkeypatch):
    calls = []

    class Client:
        def complete(self, prompt):
            calls.append(prompt)
            raise HTTPError("http://test.invalid", 429, "rate limit", {}, None)

    judge = GPT4OMiniRankingJudge(Client(), max_retries=1)
    with pytest.raises(HTTPError) as failure:
        judge._complete_with_retries("Test", image_input=None, kind="image", candidate_id="image_001")
    assert len(calls) == 2
    assert failure.value.gpt4omini_judge_diagnostic["attempt"] == 2
