"""Cross-entity overlap must preserve gates, logs, checkpoints and GPU ownership."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from mmkgc.io import RunWriter
from mmkgc.pipeline import MMKGCPipeline
from mmkgc.ranking.engine import ModelFusionRankingEngine


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = fixture_module("test_missing_image_completion")
judges = fixture_module("test_judge_concurrency")
grounded = fixture_module("test_grounded_model_mainline")
batch = helpers.batch


@pytest.mark.parametrize("workers", [1, 3])
def test_real_engine_prepares_on_caller_while_remote_requests_are_pending(workers):
    started, prepared = threading.Event(), threading.Event()
    order = []

    class Judge:
        def judge_image(self, context, candidate, image_input):
            assert threading.current_thread() is not threading.main_thread()
            started.set()
            assert prepared.wait(3), "preparation must happen before GPT returns"
            return SimpleNamespace(score=judges.ScoreResult(80, 0.8, "scored", "test"),
                                   gate=judges.GateResult("pass", "test", reason="Grounded."))

    class Local(judges.LocalVisualScorer):
        def score_image_quality_batch(self, image, texts):
            assert prepared.is_set()
            order.append("local")
            return super().score_image_quality_batch(image, texts)

    engine = ModelFusionRankingEngine(profile=judges.profile(workers), scorers={
        "gpt4omini_judge": Judge(), "clip": Local(), "blip_itm": Local()})

    def prepare_next():
        assert threading.current_thread() is threading.main_thread()
        assert started.wait(3)
        order.append("prepare")
        prepared.set()

    engine.judge_wait_callback = prepare_next
    result = engine.rank(judges.context())
    assert order[0] == "prepare"
    assert len(result["image_ranking"]["ranked_candidates"]) == 5
    calls = [call for call in result["ranking_model_calls"] if call["scorer"] == "gpt4omini_judge"]
    assert [call["candidate_id"] for call in calls] == [f"image_{i:03d}" for i in range(1, 6)]


class SplitPipeline:
    def __init__(self, job, fail_prepare=None, fail_finish=None, interrupt=None):
        self.job = job
        self.fail_prepare, self.fail_finish, self.interrupt = fail_prepare, fail_finish, interrupt
        self.ranking_engine = SimpleNamespace(judge_wait_callback=None)
        self.semantic_planner = SimpleNamespace(calls=[])
        self.events = []
        self.remote = threading.Event()
        self.release = threading.Event()

    def prepare(self, entity, mode):
        assert threading.current_thread() is threading.main_thread()
        assert len(self.job.active) <= 2
        key = entity.entity_key
        self.events.append(("prepare", key))
        if key == "Q2":
            assert self.remote.is_set() and not self.release.is_set()
        if key == self.interrupt:
            raise KeyboardInterrupt()
        if key == self.fail_prepare:
            raise ValueError("Invalid Stage 2 response")
        self.semantic_planner.calls.append({"stage": "stage1", "response_text": key})
        return SimpleNamespace(entity=entity, calls=list(self.semantic_planner.calls))

    def finish(self, prepared):
        assert threading.current_thread() is threading.main_thread()
        key = prepared.entity.entity_key
        self.events.append(("finish", key))
        self.remote.clear()
        self.release.clear()

        def judge():
            self.remote.set()
            assert self.release.wait(3)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(judge)
            assert self.remote.wait(3)
            try:
                self.ranking_engine.judge_wait_callback()
                # A feedback pass cannot prefetch a third entity while B waits.
                self.ranking_engine.judge_wait_callback()
            finally:
                self.release.set()
            future.result(timeout=3)
        if key == self.fail_finish:
            raise ValueError("Stage 3 failed")
        payload = helpers.result(prepared.entity, passed=key != "Q4")
        payload["model_calls"] = prepared.calls
        return payload


def setup_job(tmp_path, count=4):
    job = helpers.make_job(tmp_path, count)
    job.config["entity_pipeline"] = True
    batch.save(job.root / "job_config.json", job.config)
    job.state["status"] = "running"
    job.clock_started = time.perf_counter()
    store = SimpleNamespace(build_record=lambda entity_id, mode: helpers.record(entity_id))
    writer = RunWriter(job.root / "attempts", "attempt_0001", {})
    return job, store, writer


@pytest.mark.parametrize("failure", [None, "prepare", "finish"])
def test_pipeline_keeps_prefetch_bounded_and_failure_and_logs_owned(tmp_path, monkeypatch, failure):
    job, store, writer = setup_job(tmp_path)
    monkeypatch.setattr(batch, "qwen_health", lambda endpoint: None)
    pipeline = SplitPipeline(job, fail_prepare="Q2" if failure == "prepare" else None,
                             fail_finish="Q2" if failure == "finish" else None)
    batch.run_entities(job, pipeline, store, writer)
    assert job.rows["Q1"]["status"] == "completed"
    assert job.rows["Q2"]["status"] == ("failed" if failure else "completed")
    assert job.rows["Q3"]["status"] == "completed"
    assert job.rows["Q4"]["status"] == "no_valid_image"
    assert job.state["peak_inflight_entities"] == 2
    assert not job.state["active_entities"]
    assert pipeline.ranking_engine.judge_wait_callback is None
    assert job.state["timed_attempts_completed"] == 4
    assert job.state["active_wall_seconds"] > 0
    assert pipeline.events.index(("prepare", "Q2")) < pipeline.events.index(("finish", "Q3"))
    for key in ("Q1", "Q3", "Q4"):
        path = job.root / "entities" / key
        calls = [batch.json.loads(line) for line in (path / "model_calls.jsonl").read_text().splitlines()]
        assert [call["response_text"] for call in calls] == [key]
        assert (path / "generated_images/image_001.png").exists()
        assert not (path / "original_images").exists()
    assert not (job.root / "completed_images/Q4/image.png").exists()
    assert all((job.root / "checkpoints" / f"Q{i}.json").exists() for i in range(1, 5))
    assert not pipeline.semantic_planner.calls


def test_interruption_marks_current_and_prefetched_entities_for_resume(tmp_path, monkeypatch):
    job, store, writer = setup_job(tmp_path, 3)
    pipeline = SplitPipeline(job, interrupt="Q2")
    with pytest.raises(KeyboardInterrupt):
        batch.run_entities(job, pipeline, store, writer)
    assert job.rows["Q1"]["status"] == job.rows["Q2"]["status"] == "interrupted"
    assert job.rows["Q3"]["status"] == "pending"
    assert not job.active
    assert pipeline.ranking_engine.judge_wait_callback is None
    resumed = batch.Job(job.root)
    resumed.state["status"] = "running"
    writer2 = RunWriter(job.root / "attempts", "attempt_0002", {})
    batch.run_entities(resumed, SplitPipeline(resumed), store, writer2)
    assert all(row["status"] == "completed" for row in resumed.rows.values())
    assert resumed.rows["Q1"]["attempt"] == resumed.rows["Q2"]["attempt"] == 2
    assert resumed.rows["Q3"]["attempt"] == 1


def test_prefetch_service_failure_preserves_current_result_and_stops_batch(tmp_path, monkeypatch):
    job, store, writer = setup_job(tmp_path, 3)

    def unhealthy(endpoint):
        raise RuntimeError("Qwen is unavailable")

    monkeypatch.setattr(batch, "qwen_health", unhealthy)
    pipeline = SplitPipeline(job, fail_prepare="Q2")
    with pytest.raises(RuntimeError, match="Qwen is unavailable"):
        batch.run_entities(job, pipeline, store, writer)
    assert job.rows["Q1"]["status"] == "completed"
    assert batch.resumable_complete(job.rows["Q1"], job.root)
    assert job.rows["Q2"]["status"] == "failed"
    assert job.rows["Q3"]["status"] == "pending"
    assert pipeline.ranking_engine.judge_wait_callback is None
    assert not job.active


def test_pipeline_resume_skips_valid_exports(tmp_path):
    job, store, writer = setup_job(tmp_path, 3)
    batch.run_entities(job, SplitPipeline(job), store, writer)
    before = {key: row["image_sha256"] for key, row in job.rows.items()}
    resumed = batch.Job(job.root)
    pipeline = SplitPipeline(resumed)
    writer2 = RunWriter(job.root / "attempts", "attempt_0002", {})
    batch.run_entities(resumed, pipeline, store, writer2)
    assert pipeline.events == []
    assert {key: row["image_sha256"] for key, row in resumed.rows.items()} == before


def test_nested_stage_timing_uses_each_entity_context(tmp_path):
    job, _, _ = setup_job(tmp_path, 2)
    row1, row2 = job.rows["Q1"], job.rows["Q2"]
    observer = batch.StageTimer(job)
    owner = SimpleNamespace(prepare=lambda: 42)
    observer.observe(owner, "prepare", "stage1_knowledge")

    def rank():
        with observer.entity(row2):
            assert owner.prepare() == 42

    owner.rank = rank
    observer.observe(owner, "rank", "stage4")
    try:
        with observer.entity(row1):
            owner.rank()
        assert set(row1["stages"]) == {"stage4"}
        assert set(row2["stages"]) == {"stage1_knowledge"}
        assert batch._CURRENT_ENTITY.get() is None
    finally:
        observer.stack.close()


def test_split_real_pipeline_matches_serial_and_preserves_owned_preparation_logs():
    class Builder:
        def __init__(self):
            self.calls = []

        def refine_coarse(self, target, mode, source):
            self.calls.append({"stage": "stage0", "response_text": target.entity_key})
            return grounded.coarse()

        def build_knowledge(self, target, mode, source):
            inputs = grounded.build_knowledge_input(target, mode, source)
            return grounded.apply_knowledge_response(inputs, grounded.model_response(inputs))

    class Planner:
        def __init__(self):
            self.calls = []

        def plan(self, target, mode, value, budget):
            self.calls.append({"stage": "stage2", "response_text": target.entity_key})
            return grounded.apply_relation_plans(value, grounded.decisions(value), mode, budget)

    class Generator:
        def generate(self, *args):
            return "The museum is located in Example City."

    class Ranking:
        def rank(self, context):
            return {"ranking_status": "valid", "text_ranking": {"ranked_candidates": list(context.text_candidates)}}

    def pipeline():
        return MMKGCPipeline(semantic_planner=Builder(), evidence_reranker=grounded.Scorer(),
                             relation_planner=Planner(), text_generator=Generator(), ranking_engine=Ranking())

    serial, split = pipeline(), pipeline()
    target = grounded.record()
    expected = serial.run(target, "text")
    first = split.prepare(target, "text")
    batch.clear_call_history(split)
    split.prepare(replace(target, entity_key="Q999999"), "text")
    batch.clear_call_history(split)
    actual = split.finish(first)
    assert actual == expected
    semantic = [call["response_text"] for call in actual["model_calls"] if call.get("response_text")]
    assert semantic == [target.entity_key, target.entity_key]
