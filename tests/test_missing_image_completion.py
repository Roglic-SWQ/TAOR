"""Batch completion must preserve old images, gates, and resumable results."""
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from mmkgc.data import EntityRecord
from mmkgc.io import RunWriter


spec = importlib.util.spec_from_file_location("missing_image_completion", Path(__file__).parents[1] / "scripts/complete_mkgw_missing_images.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def png():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "green").save(stream, format="PNG")
    return stream.getvalue()


def record(entity_id=1):
    return EntityRecord(entity_id, f"Q{entity_id}", f"Entity {entity_id}", "Evidence text", [],
                        {"text": "Evidence text", "image": "OLD.jpg", "image_summary": "Old synthetic summary"},
                        {"text": "Evidence text", "image": "OLD.jpg"})


def result(entity, passed=True, dishonest_best=False):
    best = "image_001" if passed or dishonest_best else None
    candidate = {"candidate_id": "image_001", "target_modality": "image", "final_score": 0.9,
                 "hard_gate": {"decision": "pass" if passed else "reject", "hard_reasons": []}}
    return {"entity": asdict(entity), "generated_image_artifacts": [{"candidate_id": "image_001",
                "relative_image_path": "generated_images/image_001.png", "image_bytes": png()}],
            "kg_aware_candidate_consistency_ranking": {"image_ranking": {
                "best_candidate_id": best, "ranked_candidates": [candidate],
                "top_5_candidates": [candidate] if passed else [], "rejected_candidates": [] if passed else [candidate]}},
            "final": {"entity_key": entity.entity_key, "best_image_candidate_id": best,
                      "ranking_status": "completed" if passed else "rejected", "counts": {"image": 1}}}


def make_job(tmp_path, count=3):
    root = tmp_path / "job"
    root.mkdir()
    batch.save(root / "job_config.json", {"project_root": str(Path(__file__).parents[1]), "qwen_endpoint": "http://test/v1",
                                         "runtime_kwargs": {"candidate_budget": 3}})
    batch.save(root / "entity_manifest.json", [{"entity_key": f"Q{i}", "entity_id": i, "label": f"Entity {i}", "train_degree": i} for i in range(1, count + 1)])
    return batch.Job(root)


def test_old_synthetic_image_is_not_input_or_ground_truth():
    original = record()
    fresh = batch.fresh_missing_record(original)
    assert fresh.existing_modalities == {"text": "Evidence text", "image": None, "image_summary": None}
    assert fresh.ground_truth_modalities == {"text": "Evidence text", "image": None}
    assert fresh.modality_completion_policy["image"]["status"] == "real_missing"
    assert fresh.modality_completion_policy["image"]["can_evaluate_against_ground_truth"] is False
    assert original.ground_truth_modalities["image"] == "OLD.jpg"


@pytest.mark.parametrize("passed,dishonest,expected", [(True, False, "completed"), (False, False, "no_valid_image"), (False, True, "export_failed")])
def test_export_checks_persisted_gate_and_never_falls_back(tmp_path, passed, dishonest, expected):
    writer = RunWriter(tmp_path, "artifacts", {})
    writer.write_entity(result(record(), passed, dishonest))
    entity = writer.entities_dir / "Q1"
    target = tmp_path / "new_images/Q1/image.png"
    if expected == "export_failed":
        with pytest.raises(ValueError, match="did not pass"):
            batch.export_best(entity, target)
    else:
        exported = batch.export_best(entity, target)
        assert bool(exported) == passed
        assert target.exists() == passed
        if passed:
            assert exported["image_sha256"] == batch.sha256(target)
        else:
            # Regression: compact writer may list rejected images as top5.
            ranking = batch.load(entity / "4_kg_aware_candidate_consistency_ranking.json")["data"]["image_ranking"]
            assert ranking["ranked_candidate_ids"] == ["image_001"]
    assert (entity / "generated_images/image_001.png").is_file()


def test_batch_continues_after_failure_and_resumes_without_regenerating_success(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    old = tmp_path / "OLD.jpg"
    old.write_bytes(b"old synthetic file")
    health = []
    monkeypatch.setattr(batch, "qwen_health", lambda url: health.append(url))

    class Pipeline:
        def __init__(self):
            self.processed = []
            self.semantic_planner = SimpleNamespace(calls=[])
            self.fail = True

        def run(self, entity, mode):
            assert mode == "image" and entity.ground_truth_modalities["image"] is None
            self.processed.append(entity.entity_key)
            self.semantic_planner.calls.append({"response": "large raw response"})
            if entity.entity_key == "Q2" and self.fail:
                raise ValueError("Invalid model response")
            return result(entity, passed=entity.entity_key != "Q3")

    pipeline = Pipeline()
    store = SimpleNamespace(build_record=lambda entity_id, mode: record(entity_id))
    writer = RunWriter(job.root / "attempts", "attempt_0001", {})
    batch.run_entities(job, pipeline, store, writer)
    assert [job.rows[f"Q{i}"]["status"] for i in range(1, 4)] == ["completed", "failed", "no_valid_image"]
    assert old.read_bytes() == b"old synthetic file"
    assert len(health) == 1
    assert not pipeline.semantic_planner.calls
    assert len(list(job.root.glob("checkpoints/*.json"))) == 3
    assert (job.root / "entity_timing.csv").read_text(encoding="utf-8-sig").count("\n") == 4
    assert (writer.entities_dir / "Q3/final.json").is_file()
    assert not (job.root / "completed_images/Q3/image.png").exists()
    assert len(batch.load(job.root / "failures.json")) == 2
    assert job.state["candidate_budget"] == 3
    assert job.rows["Q1"]["candidate_budget"] == 3
    assert batch.load(job.root / "completed_images/Q1/metadata.json")["candidate_budget"] == 3

    resumed = batch.Job(job.root)
    writer2 = RunWriter(job.root / "attempts", "attempt_0002", {})
    pipeline.fail = False
    pipeline.processed.clear()
    batch.run_entities(resumed, pipeline, store, writer2)
    assert pipeline.processed == []  # Failures remain auditable unless explicitly retried.
    batch.run_entities(resumed, pipeline, store, writer2, retry_failed=True)
    assert pipeline.processed == ["Q2", "Q3"]
    assert resumed.rows["Q1"]["artifact_dir"] == str(job.root / "entities/Q1")
    assert resumed.rows["Q1"]["attempt_artifact_dir"].endswith("attempt_0001/entities/Q1")
    assert (job.root / "entities/Q1").resolve() == writer.entities_dir / "Q1"
    assert batch.load(job.root / "entities/Q2/final.json")["best_image_candidate_id"] == "image_001"
    assert resumed.rows["Q2"]["status"] == "completed"
    assert resumed.rows["Q3"]["status"] == "no_valid_image"

    # A corrupt success is no longer skipped on resume.
    Path(resumed.rows["Q1"]["image_path"]).write_bytes(b"corrupt")
    assert not batch.resumable_complete(resumed.rows["Q1"], job.root)
    writer3 = RunWriter(job.root / "attempts", "attempt_0003", {})
    pipeline.processed.clear()
    batch.run_entities(resumed, pipeline, store, writer3)
    assert pipeline.processed == ["Q1"]
    assert (writer.entities_dir / "Q1/final.json").is_file()
    assert (job.root / "entities/Q1").resolve() == writer3.entities_dir / "Q1"
    finals = [batch.load(job.root / "entities" / f"Q{i}/final.json") for i in (1, 2, 3)]
    indexed = [batch.json.loads(line) for line in (job.root / "results.jsonl").read_text().splitlines()]
    assert indexed == finals
    assert batch.load(job.root / "metrics_summary.json")["completed_image_entities"] == 2
    assert batch.load(job.root / "metrics_summary.json")["processed_entities"] == 3


@pytest.mark.parametrize("status,alive,decision", [
    ("running", True, "wait"), ("completed", True, "wait"),
    ("completed", False, "ready"), ("completed_with_failures", False, "ready"),
    ("interrupted", False, "blocked"), ("running", False, "blocked"), ("failed", True, "blocked"),
])
def test_queue_requires_normal_completion_and_process_exit(status, alive, decision):
    assert batch.prerequisite_decision({"status": status}, alive) == decision


def test_waiter_does_not_release_gpu_before_test_process_exits(tmp_path, monkeypatch):
    job = make_job(tmp_path, 1)
    prerequisite = tmp_path / "test50"
    prerequisite.mkdir()
    progress = prerequisite / "timing_progress.json"
    batch.save(progress, {"status": "running", "completed_entities": 49})
    job.config["prerequisite"] = {"run_dir": str(prerequisite), "pid": 123, "process_identity": "same-process"}
    identity = ["same-process"]
    sleeps = []
    monkeypatch.setattr(batch, "process_identity", lambda pid: identity[0])

    def advance(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            batch.save(progress, {"status": "completed_with_failures", "completed_entities": 49})
        else:
            identity[0] = None
    monkeypatch.setattr(batch.time, "sleep", advance)
    assert batch.wait_for_test(job)
    assert sleeps == [20, 20]
    assert "waiting_for" not in job.state


def test_stage_measurements_accumulate_and_restore_even_on_error(tmp_path):
    job = make_job(tmp_path, 1)
    row = job.rows["Q1"]
    owner = SimpleNamespace(operation=lambda: 42)
    original = owner.operation
    timer = batch.StageTimer(job, row)
    timer.observe(owner, "operation", "stage4")
    assert owner.operation() == 42
    assert owner.operation() == 42
    timer.stack.close()
    assert owner.operation is original
    assert "stage4" in row["stages"]


def test_changed_queued_input_stops_before_model_initialization(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("version1")
    config = {"input_sha256": {str(source): batch.sha256(source)}}
    batch.verify_inputs(config)
    source.write_text("version2")
    with pytest.raises(RuntimeError, match="Queued input changed"):
        batch.verify_inputs(config)


def test_resume_rejects_success_outside_own_output(tmp_path):
    image = tmp_path / "other_job/image.png"
    image.parent.mkdir()
    image.write_bytes(png())
    assert not batch.resumable_complete({"status": "completed", "image_path": str(image), "image_sha256": batch.sha256(image)}, tmp_path / "this_job")


@pytest.mark.parametrize("budget", [3, 5])
def test_prepared_batch_propagates_budget_to_runtime_and_reports(tmp_path, monkeypatch, budget):
    project = tmp_path / "project"
    dataset = project / "datasets/fixture"
    dataset.mkdir(parents=True)
    (dataset / "entity2id.txt").write_text("http://www.wikidata.org/entity/Q1 0\nhttp://www.wikidata.org/entity/Q2 1\n")
    (dataset / "relation2id.txt").write_text("http://www.wikidata.org/entity/P31 0\n")
    triple = "http://www.wikidata.org/entity/Q1 http://www.wikidata.org/entity/P31 http://www.wikidata.org/entity/Q2 .\n"
    for name in ("train.txt", "test.txt"):
        (dataset / name).write_text(triple)
    (dataset / "valid.txt").write_text("")
    batch.save(dataset / "entity_info.json", {})
    batch.save(dataset / "test_missing_image_qids.json", ["Q1", "Q2"])
    configs = project / "configs"
    configs.mkdir()
    for name in ("qwen7b-mainline.yaml", "qwen7b-stage1.yaml", "qwen7b-stage2.yaml", "openai-gpt4o-mini-judge.yaml"):
        (configs / name).write_text("lmm:\n  model_name: Qwen/Qwen2-VL-7B-Instruct\n  api_key: EMPTY\n")
    (configs / "stage4-model-fusion-gpu1-visible.yaml").write_text("gpt4omini_judge:\n  model_config: configs/openai-gpt4o-mini-judge.yaml\n")
    manifest = configs / "manifest.json"
    batch.save(manifest, [{"entity_key": "Q1", "entity_id": 0}, {"entity_key": "Q2", "entity_id": 1}])
    monkeypatch.setattr(batch, "PROJECT", project)
    monkeypatch.setattr(batch.sys, "path", list(batch.sys.path))
    args = SimpleNamespace(dataset_dir=str(dataset), entity_manifest=str(manifest), expected_entities=2,
                           candidate_budget=budget, job_dir=str(project / "outputs/new_job"), wait_for_run=None)
    root = batch.prepare(args)
    config = batch.load(root / "job_config.json")
    assert config["runtime_kwargs"]["candidate_budget"] == budget
    assert config["candidate_budget"] == budget
    assert batch.load(root / "progress.json")["candidate_budget"] == budget
    assert f"{budget} 张" in (root / "timing_summary.zh.md").read_text()
    batch.verify_inputs(config)


def test_prepare_rejects_invalid_budget_before_creating_output():
    with pytest.raises(ValueError, match="candidate_budget"):
        batch.prepare(SimpleNamespace(candidate_budget=0))


def test_reference_layout_preserves_stages_three_candidates_and_logs_without_originals(tmp_path):
    from copy import deepcopy
    from mmkgc.io.writer import STAGE_FILES

    job = make_job(tmp_path, 1)
    batch.initialize_stage_layout(job)
    writer = RunWriter(job.root / "attempts", "attempt_0001", {})
    payload = result(batch.fresh_missing_record(record()))
    ranking = payload["kg_aware_candidate_consistency_ranking"]["image_ranking"]
    template = ranking["ranked_candidates"][0]
    ranking["ranked_candidates"] = []
    payload["generated_image_artifacts"] = []
    payload["image_model_calls"] = []
    for index in range(1, 4):
        candidate_id = f"image_{index:03d}"
        ranking["ranked_candidates"].append({**deepcopy(template), "candidate_id": candidate_id})
        payload["generated_image_artifacts"].append({"candidate_id": candidate_id,
            "relative_image_path": f"generated_images/{candidate_id}.png", "image_bytes": png(),
            "relative_metadata_path": f"generated_images/{candidate_id}.json", "metadata": {"candidate_id": candidate_id, "generator": "test"}})
        payload["image_model_calls"].append({"candidate_id": candidate_id, "modality": "image", "status": "generated"})
    ranking["top_5_candidates"] = list(ranking["ranked_candidates"])
    payload["final"]["counts"]["image"] = 3
    payload["model_calls"] = [{"stage": "stage0_coarse", "status": "applied"}]
    payload["ranking_model_calls"] = [{"candidate_id": "image_001", "status": "scored", "scorer": "clip", "latency_ms": 1}]
    pipeline = SimpleNamespace(run=lambda record, mode: payload)
    store = SimpleNamespace(build_record=lambda entity_id, mode: record(entity_id))
    batch.run_entities(job, pipeline, store, writer)
    entity = job.root / "entities/Q1"
    assert entity.is_dir()
    for _, filename in STAGE_FILES:
        assert set(batch.load(entity / filename)) == {"stage", "summary", "data"}
    for filename in ("final.json", "model_calls.jsonl", "image_model_calls.jsonl", "ranking_model_calls.jsonl", "entity_description.json", "entity_description.txt"):
        assert (entity / filename).is_file()
    assert len(list((entity / "generated_images").glob("*.png"))) == 3
    assert len(list((entity / "generated_images").glob("*.json"))) == 3
    assert not list(job.root.rglob("original_images"))
    assert not list(job.root.rglob("original_images_manifest.json"))
    assert batch.load(job.root / "config.json")["storage_policy"]["copy_original_images"] is False
    assert batch.load(job.root / "metrics_summary.json")["avg_image_candidates"] == 3
    assert batch.load(job.root / "completed_images/Q1/metadata.json")["ground_truth_used"] is False


def test_failed_entity_has_discoverable_failure_reason_in_same_layout(tmp_path, monkeypatch):
    job = make_job(tmp_path, 1)
    writer = RunWriter(job.root / "attempts", "attempt_0001", {})
    monkeypatch.setattr(batch, "qwen_health", lambda endpoint: None)
    def fail(record, mode):
        raise ValueError("Invalid Stage 2 response")
    batch.run_entities(job, SimpleNamespace(run=fail), SimpleNamespace(build_record=lambda entity_id, mode: record(entity_id)), writer)
    failure = batch.load(job.root / "entities/Q1/failure.json")
    assert failure["status"] == "failed"
    assert failure["error_message"] == "Invalid Stage 2 response"
    assert len(batch.load(job.root / "failures.json")) == 1
    assert (job.root / "results.jsonl").read_text() == ""
