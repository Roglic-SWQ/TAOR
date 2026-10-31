"""Queue a fresh MKG-W image completion run and resume it from entity checkpoints.

The original missing-image list defines scope even when old synthetic images exist.
Only a best candidate that passed Stage 4 is exported; source data is read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import csv
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from urllib import request
from unittest.mock import patch
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "failed", "no_valid_image", "export_failed"}
STAGES = ("stage0", "stage1_knowledge", "stage1_reranker", "stage2", "stage3", "stage3_regeneration", "stage4")
_CURRENT_ENTITY = ContextVar("mmkgc_timed_entity", default=None)
STORAGE_POLICY = {
    "layout": "entity_stage_files_v1",
    "entity_directory": "entities/<QID>",
    "save_stage_results": True,
    "save_generated_candidates": True,
    "save_model_calls": True,
    "copy_original_images": False,
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def save(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_identity(pid):
    """A PID plus its Linux start tick avoids waiting on a reused PID."""
    if not pid:
        return None
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def prerequisite_decision(progress, alive):
    if progress.get("status") in {"completed", "completed_with_failures"}:
        return "wait" if alive else "ready"
    if progress.get("status") in {"interrupted", "failed"}:
        return "blocked"
    return "wait" if alive else "blocked"


def fresh_missing_record(record):
    from mmkgc.data.contracts import build_modality_completion_policy

    existing = {**record.existing_modalities, "image": None, "image_summary": None}
    truth = {**record.ground_truth_modalities, "image": None}
    return replace(record, existing_modalities=existing, ground_truth_modalities=truth,
                   modality_completion_policy=build_modality_completion_policy("image", existing, truth))


def export_best(entity_dir, destination):
    """Read persisted output, independently check the gate, and export atomically.

    Do not use top_5_candidate_ids: older compact writers populate it even when
    all candidates were rejected. No fallback to the first generated image.
    """
    from PIL import Image

    entity_dir, destination = Path(entity_dir), Path(destination)
    final = load(entity_dir / "final.json")
    ranking = load(entity_dir / "4_kg_aware_candidate_consistency_ranking.json")["data"]["image_ranking"]
    candidate_id = final.get("best_image_candidate_id")
    if not candidate_id:
        return None
    if not re.fullmatch(r"image_(?:safe_)?[0-9]+", candidate_id):
        raise ValueError("Unsafe best image candidate ID")
    if ranking.get("best_candidate_id") != candidate_id:
        raise ValueError("Final and Stage 4 best candidate IDs disagree")
    candidates = [item for item in ranking.get("ranked_candidates", []) if item.get("candidate_id") == candidate_id]
    if len(candidates) != 1 or candidates[0].get("gate", {}).get("decision") != "pass":
        raise ValueError("Best candidate did not pass Stage 4")
    image = entity_dir / "generated_images" / (candidate_id + ".png")
    with Image.open(image) as opened:
        if opened.format != "PNG":
            raise ValueError("Generated image must be a PNG")
        opened.verify()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(image, temp)
    temp.replace(destination)
    return {"best_candidate_id": candidate_id, "image_path": str(destination.resolve()),
            "image_sha256": sha256(destination), "source_image": str(image.resolve())}


def resumable_complete(row, job_dir):
    if row.get("status") != "completed":
        return False
    image = Path(row.get("image_path", "/nonexistent"))
    try:
        image.resolve().relative_to(Path(job_dir).resolve() / "completed_images")
        return image.is_file() and sha256(image) == row.get("image_sha256")
    except ValueError:
        return False


def initialize_stage_layout(job):
    """Expose the same entity/stage entry points as the reference 50-entity run."""
    from mmkgc.io.writer import _safe_output

    (job.root / "entities").mkdir(exist_ok=True)
    save(job.root / "config.json", _safe_output({
        **job.config.get("runtime_kwargs", {}),
        "dataset_dir": job.config.get("dataset_dir"),
        "graph_split": "train", "expected_entities": len(job.manifest),
        "entity_pipeline": job.config.get("entity_pipeline", False),
        "max_inflight_entities": 2 if job.config.get("entity_pipeline", False) else 1,
        "storage_policy": STORAGE_POLICY,
    }))


def publish_entity_directory(job, row, entity_dir):
    """Point entities/QID to its current attempt without duplicating PNGs.

    Prior attempts stay intact when an interrupted or failed entity is retried.
    The link only targets generated artifacts within this same job directory.
    """
    key = row["entity_key"]
    if not re.fullmatch(r"Q[0-9]+", key):
        raise ValueError("Invalid entity key for output directory")
    source = Path(entity_dir).resolve()
    source.relative_to(job.root)
    if not source.is_dir():
        raise FileNotFoundError(source)
    parent = job.root / "entities"
    parent.mkdir(exist_ok=True)
    parent.resolve().relative_to(job.root)
    destination = parent / key
    if destination.is_symlink():
        destination.resolve().relative_to(job.root)
    elif destination.exists():
        raise FileExistsError(f"Refusing to replace an existing entity directory: {destination}")
    temporary = parent / ("." + key + ".tmp")
    if temporary.is_symlink():
        temporary.unlink()
    elif temporary.exists():
        raise FileExistsError(temporary)
    temporary.symlink_to(os.path.relpath(source, parent), target_is_directory=True)
    temporary.replace(destination)
    row.update(artifact_dir=str(destination), attempt_artifact_dir=str(source))


class Job:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = load(self.root / "job_config.json")
        self.manifest = load(self.root / "entity_manifest.json")
        self.rows = {}
        for item in self.manifest:
            path = self.root / "checkpoints" / (item["entity_key"] + ".json")
            self.rows[item["entity_key"]] = load(path) if path.exists() else {**item, "status": "pending", "stages": {}}
        self.state = load(self.root / "progress.json") if (self.root / "progress.json").exists() else {}
        self.state.update(expected_entities=len(self.manifest), run_dir=str(self.root),
                          candidate_budget=self.config.get("runtime_kwargs", {}).get("candidate_budget", 5))
        self.lock = threading.RLock()
        self.active = {}
        self.clock_started = None
        self.clock_base = self.state.get("active_wall_seconds", 0)

    def event(self, kind, **details):
        event = {"at": now(), "event": kind, **details}
        with self.lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)

    def checkpoint(self, row):
        with self.lock:
            self.rows[row["entity_key"]] = row
            save(self.root / "checkpoints" / (row["entity_key"] + ".json"), row)

    def flush(self, reports=False):
        with self.lock:
            counts = Counter(row["status"] for row in self.rows.values())
            self.state.update(updated_at=now(), counts=dict(counts),
                              completed_entities=counts["completed"], failed_entities=counts["failed"],
                              no_valid_image_entities=counts["no_valid_image"], export_failed_entities=counts["export_failed"],
                              attempted_entities=sum(counts[key] for key in TERMINAL))
            if self.clock_started is not None:
                self.state["active_wall_seconds"] = round(self.clock_base + time.perf_counter() - self.clock_started, 3)
                attempts = self.state.get("timed_attempts_completed", 0)
                self.state["throughput_seconds_per_attempt"] = round(self.state["active_wall_seconds"] / attempts, 3) if attempts else None
            if self.config.get("entity_pipeline", False):
                active = [{"entity_key": row["entity_key"], "label": row.get("label"),
                           "stage": row.get("current_stage")} for row in self.active.values()]
                self.state.update(entity_pipeline=True, max_inflight_entities=2, active_entities=active)
                self.state["peak_inflight_entities"] = max(self.state.get("peak_inflight_entities", 0), len(active))
                if self.state.get("status") == "running":
                    live = [row for row in active if row["stage"] != "waiting_for_stage3"]
                    current = (live or active or [{}])[-1]
                    self.state.update(current_entity=current.get("entity_key"), current_stage=current.get("stage", "between_entities"))
            save(self.root / "progress.json", self.state)
            if not reports:
                return
            ordered = [self.rows[item["entity_key"]] for item in self.manifest]
            fields = ["entity_key", "label", "status", "candidate_budget", "seconds", "prepare_seconds", "queue_wait_seconds", "prefetch_seconds", *STAGES, "best_candidate_id", "image_path",
                      "artifact_dir", "failed_stage", "error_type", "error_message", "train_degree"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in ordered:
                flat = {key: row.get(key, "") for key in fields}
                flat.update({key: row.get("stages", {}).get(key, "") for key in STAGES})
                writer.writerow(flat)
            atomic_text(self.root / "entity_timing.csv", "\ufeff" + stream.getvalue())
            atomic_text(self.root / "completion_manifest.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
            save(self.root / "failures.json", [row for row in ordered if row["status"] in TERMINAL - {"completed"}])
            finals = [row["pipeline_final"] for row in ordered if isinstance(row.get("pipeline_final"), dict)]
            atomic_text(self.root / "results.jsonl", "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in finals))
            save(self.root / "metrics_summary.json", {
                "processed_entities": len(finals), "failed_entities": counts["failed"],
                "avg_text_candidates": round(statistics.mean(value.get("counts", {}).get("text", 0) for value in finals), 6) if finals else 0.0,
                "avg_image_candidates": round(statistics.mean(value.get("counts", {}).get("image", 0) for value in finals), 6) if finals else 0.0,
                "completed_image_entities": counts["completed"], "no_valid_image_entities": counts["no_valid_image"],
                "export_failed_entities": counts["export_failed"],
                "stage_failure_counts": dict(Counter(row.get("failed_stage", "pipeline") for row in ordered if row["status"] == "failed")),
            })
            successful = [row["seconds"] for row in ordered if row["status"] == "completed" and "seconds" in row]
            mean = round(statistics.mean(successful), 3) if successful else None
            summary = [f"状态：{self.state.get('status')}；目标实体 {len(ordered)}。",
                       f"每实体候选图片预算：{self.state['candidate_budget']} 张；通过 Stage 4 后导出最佳图。",
                       f"有效图片已导出 {counts['completed']}；流程失败 {counts['failed']}；无入选图片 {counts['no_valid_image']}；导出失败 {counts['export_failed']}。",
                       f"当前：{self.state.get('current_entity')} / {self.state.get('current_stage')}。",
                       f"成功实体平均全流程耗时：{mean} 秒（含重试；首实体含本地模型加载）。",
                       "Stage 1 分别记录知识建模和 Reranker；Stage 3 重生成、Stage 4 多轮评分分别累计。",
                       "原始图片保持原样；旧合成图不作为本轮输入或真实参考图。仅使用 train.txt 的图谱邻居。",
                       "五阶段结果、候选图和调用日志在 entities/<QID>/；各次尝试原件在 attempts/，通过实体入口查看无需复制图片。",
                       "不创建 original_images/ 或 original_images_manifest.json；历史生成图保留在源数据目录中。",
                       "每实体检查点在 checkpoints/；逐实体耗时 entity_timing.csv；最佳新图 completed_images/<QID>/image.png。",
                       f"更新时间：{self.state['updated_at']}（UTC）。"]
            if self.config.get("entity_pipeline", False):
                summary.insert(5, "实体流水线：GPT 评分请求等待期间，主线程提前执行下一实体的 Stage 0–2；最多两个在途实体，本地 GPU 模型顺序执行。")
                summary.insert(6, "逐实体耗时可相互重叠，Stage 4 墙钟耗时包含预取下一实体；不能相加当作整批耗时。queue_wait_seconds 和 prefetch_seconds 分别记录排队与预取耗时。")
                summary.insert(7, f"实际运行墙钟：{self.state.get('active_wall_seconds', 0)} 秒；已计时处理次数：{self.state.get('timed_attempts_completed', 0)}；吞吐：{self.state.get('throughput_seconds_per_attempt')} 秒/处理次数（含失败，排除断点间停机）。")
            if self.state.get("waiting_for"):
                summary.insert(1, "等待 50 实体测试结束：" + self.state["waiting_for"])
            if self.state.get("error_message"):
                summary.append("任务信息：" + self.state["error_message"])
            atomic_text(self.root / "timing_summary.zh.md", "\n\n".join(summary) + "\n")
            recent = [row for row in ordered if row["status"] != "pending"][-20:]
            cells = "".join("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in
                (row['entity_key'], row.get('label', ''), row['status'], row.get('seconds', ''), row.get('image_path', ''))) + "</tr>" for row in recent)
            page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>4580 实体补全</title><body style="font:16px system-ui;margin:24px">'
            page += "".join("<p>" + html.escape(line) + "</p>" for line in summary)
            page += '<p><a href="entity_timing.csv">全部实体耗时 CSV</a> · <a href="completion_manifest.jsonl">完整结果索引</a></p><table border="1" cellpadding="6"><tr><th>实体</th><th>名称</th><th>状态</th><th>秒</th><th>新图</th></tr>' + cells + '</table></body></html>'
            atomic_text(self.root / "timing_report.html", page)


class StageTimer:
    def __init__(self, job, row=None):
        self.job, self.row = job, row
        self.stack = ExitStack()

    @contextmanager
    def entity(self, row):
        token = _CURRENT_ENTITY.set(row)
        try:
            yield
        finally:
            _CURRENT_ENTITY.reset(token)

    def observe(self, owner, method, stage):
        original = getattr(owner, method)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            row = self.row if self.row is not None else _CURRENT_ENTITY.get()
            if row is None:
                return original(*args, **kwargs)
            with self.job.lock:
                if self.row is None:
                    previous = row.get("current_stage")
                    row["current_stage"] = stage
                else:
                    previous = self.job.state.get("current_stage")
                    self.job.state["current_stage"] = stage
                self.job.flush()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            except BaseException:
                row["failed_stage"] = stage
                raise
            finally:
                with self.job.lock:
                    row["stages"][stage] = round(row["stages"].get(stage, 0) + time.perf_counter() - started, 3)
                    if self.row is None:
                        row["current_stage"] = previous
                    else:
                        self.job.state["current_stage"] = previous
        self.stack.enter_context(patch.object(owner, method, measured))

    def __enter__(self):
        from mmkgc import pipeline as module
        from mmkgc.adapters.relation_planner import OpenAICompatibleRelationPlanner
        from mmkgc.ranking.engine import ModelFusionRankingEngine
        from mmkgc.stages.semantic import StrictJsonStagePlanner

        for owner, method, stage in (
            (StrictJsonStagePlanner, "refine_coarse", "stage0"),
            (StrictJsonStagePlanner, "build_knowledge", "stage1_knowledge"),
            (module, "score_knowledge_evidence", "stage1_reranker"),
            (OpenAICompatibleRelationPlanner, "plan", "stage2"),
            (module, "relation_alternation_generation", "stage3"),
            (module, "safer_relation_alternation_generation", "stage3_regeneration"),
            (ModelFusionRankingEngine, "rank", "stage4"),
        ):
            self.observe(owner, method, stage)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def clear_call_history(pipeline):
    # prepare() owns a copy of each entity's logs; retaining raw responses for
    # 4580 entities in planner.calls would cause unbounded RAM growth.
    for name in ("semantic_planner", "knowledge_planner", "evidence_reranker", "relation_planner"):
        calls = getattr(getattr(pipeline, name, None), "calls", None)
        if isinstance(calls, list):
            calls.clear()


def begin_entity(job, item, index):
    previous = job.rows[item["entity_key"]]
    row = {**item, "status": "running", "stages": {}, "started_at": now(),
           "candidate_budget": job.state["candidate_budget"], "attempt": int(previous.get("attempt", 0)) + 1,
           "current_stage": "preparing"}
    job.state.update(current_entity=row["entity_key"], current_stage="entity")
    if job.config.get("entity_pipeline", False):
        job.active[row["entity_key"]] = row
    job.checkpoint(row)
    job.flush()
    job.event("ENTITY_START", entity_key=row["entity_key"], index=index)
    return {"row": row, "item": item, "started": time.perf_counter(), "phase": "pipeline"}


def persist_entity_result(job, writer, entry, result):
    from mmkgc.io.writer import _safe_output

    row, record = entry["row"], entry["record"]
    key = row["entity_key"]
    entry["phase"] = "persist"
    writer.write_entity(result)
    entity_dir = writer.entities_dir / key
    row["pipeline_final"] = _safe_output(result.get("final", {}))
    save(entity_dir / "entity_description.json", {"entity_key": key, "entity_id": record.entity_id,
         "label": record.label, "description": record.description})
    atomic_text(entity_dir / "entity_description.txt", record.label + "\n\n" + record.description + "\n")
    publish_entity_directory(job, row, entity_dir)
    ranking = result.get("kg_aware_candidate_consistency_ranking", {})
    if ranking.get("ranking_status") == "evaluation_failed":
        row.update(status="failed", failed_stage="stage4", error_type="RankingEvaluationError",
                   error_message="Required image judgment could not be validated; generated candidates preserved")
        save(entity_dir / "failure.json", {key: row.get(key) for key in
             ("entity_key", "status", "failed_stage", "error_type", "error_message")})
        return
    entry["phase"] = "export"
    exported = export_best(entity_dir, job.root / "completed_images" / key / "image.png")
    row.update(status="completed" if exported else "no_valid_image")
    if exported:
        row.update(exported)
        save(job.root / "completed_images" / key / "metadata.json", {**exported, "entity_key": key,
             "label": entry["item"].get("label"), "artifact_dir": row["artifact_dir"],
             "candidate_budget": row["candidate_budget"], "ground_truth_used": False})


def record_entity_failure(job, writer, entry, exc):
    from mmkgc.io.writer import _safe_output

    row = entry["row"]
    key = row["entity_key"]
    call = getattr(exc, "model_call", {})
    if not isinstance(call, dict):
        call = {}
    row.update(status="export_failed" if entry["phase"] == "export" else "failed",
               error_type=type(exc).__name__, error_message=_safe_output(str(exc)),
               failed_stage=call.get("stage") or row.get("failed_stage") or entry["phase"])
    if call or getattr(exc, "diagnostic_events", None):
        save(job.root / "failure_details" / f"{key}_attempt{row['attempt']}.json",
             _safe_output({"model_call": call, "diagnostic_events": getattr(exc, "diagnostic_events", [])}))
    entity_dir = writer.entities_dir / key
    entity_dir.mkdir(parents=True, exist_ok=True)
    save(entity_dir / "failure.json", {name: row.get(name) for name in
         ("entity_key", "label", "candidate_budget", "status", "failed_stage", "error_type", "error_message", "attempt")})
    publish_entity_directory(job, row, entity_dir)
    writer.record_failure(entity=entry["item"], error=exc, stage=row["failed_stage"])


def finish_entity(job, writer, entry):
    if entry.get("finalized"):
        return
    entry["finalized"] = True
    row = entry["row"]
    row.update(seconds=round(time.perf_counter() - entry["started"], 3), finished_at=now())
    if job.clock_started is not None and row["status"] in TERMINAL:
        job.state["timed_attempts_completed"] = job.state.get("timed_attempts_completed", 0) + 1
    job.active.pop(row["entity_key"], None)
    job.checkpoint(row)
    job.state.update(current_entity=None, current_stage="between_entities")
    job.flush(reports=True)
    writer.write_summary()
    job.event("ENTITY_DONE", entity_key=row["entity_key"], status=row["status"], seconds=row["seconds"])


def pending_entities(job, retry_failed):
    for index, item in enumerate(job.manifest, 1):
        previous = job.rows[item["entity_key"]]
        if resumable_complete(previous, job.root):
            continue
        if previous["status"] in TERMINAL - {"completed"} and not retry_failed:
            continue
        yield index, item


def run_entities(job, pipeline, store, writer, retry_failed=False):
    if job.config.get("entity_pipeline", False):
        return run_entities_pipelined(job, pipeline, store, writer, retry_failed)
    for index, item in pending_entities(job, retry_failed):
        entry = begin_entity(job, item, index)
        try:
            record = fresh_missing_record(store.build_record(item["entity_id"], "image"))
            if record.entity_key != item["entity_key"]:
                raise ValueError("Manifest entity ID and key disagree")
            entry["record"] = record
            with StageTimer(job, entry["row"]):
                result = pipeline.run(record, "image")
            persist_entity_result(job, writer, entry, result)
            result = None
        except Exception as exc:
            record_entity_failure(job, writer, entry, exc)
        except BaseException:
            entry["row"]["status"] = "interrupted"
            raise
        finally:
            finish_entity(job, writer, entry)
            clear_call_history(pipeline)
        if entry["row"]["status"] == "failed":
            qwen_health(job.config["qwen_endpoint"])


def run_entities_pipelined(job, pipeline, store, writer, retry_failed=False):
    """Prepare at most one next entity while only remote GPT calls run on workers.

    All planner, Reranker, SDXL, CLIP, BLIP, writer and checkpoint calls stay on
    this caller thread. Prepared model-call records are owned by their entity.
    """
    if not all(callable(getattr(pipeline, name, None)) for name in ("prepare", "finish")):
        raise TypeError("Entity pipelining requires separate prepare and finish methods")
    source = iter(pending_entities(job, retry_failed))
    pending = None
    current = None
    fatal_error = None
    exhausted = False
    engine = pipeline.ranking_engine
    original_callback = engine.judge_wait_callback
    job.state.update(entity_pipeline=True, max_inflight_entities=2)

    with StageTimer(job) as observer:
        def prepare_next():
            nonlocal fatal_error, exhausted
            try:
                index, item = next(source)
            except StopIteration:
                exhausted = True
                return None
            entry = begin_entity(job, item, index)
            try:
                record = fresh_missing_record(store.build_record(item["entity_id"], "image"))
                if record.entity_key != item["entity_key"]:
                    raise ValueError("Manifest entity ID and key disagree")
                entry["record"] = record
                with observer.entity(entry["row"]):
                    entry["prepared"] = pipeline.prepare(record, "image")
                entry["ready"] = time.perf_counter()
                entry["row"].update(prepare_seconds=round(entry["ready"] - entry["started"], 3),
                                    prepared_at=now(), current_stage="waiting_for_stage3")
                job.checkpoint(entry["row"])
                job.flush()
                return entry
            except Exception as exc:
                record_entity_failure(job, writer, entry, exc)
                finish_entity(job, writer, entry)
                try:
                    qwen_health(job.config["qwen_endpoint"])
                except Exception as health_error:
                    fatal_error = health_error
            except BaseException:
                entry["row"]["status"] = "interrupted"
                finish_entity(job, writer, entry)
                raise
            finally:
                clear_call_history(pipeline)
            return None

        def prefetch():
            nonlocal pending
            if pending is not None or fatal_error is not None or current is None:
                return
            started = time.perf_counter()
            pending = prepare_next()
            row = current["row"]
            row["prefetch_seconds"] = round(row.get("prefetch_seconds", 0) + time.perf_counter() - started, 3)
            job.event("ENTITY_PREFETCH", entity_key=row["entity_key"],
                      next_entity=pending["row"]["entity_key"] if pending else None,
                      seconds=row["prefetch_seconds"])

        engine.judge_wait_callback = prefetch
        try:
            while True:
                if fatal_error is not None:
                    raise fatal_error
                if pending is not None:
                    current, pending = pending, None
                else:
                    current = prepare_next()
                    if current is None:
                        if fatal_error is not None:
                            raise fatal_error
                        if exhausted:
                            break
                        continue
                row = current["row"]
                row.update(queue_wait_seconds=round(time.perf_counter() - current["ready"], 3),
                           current_stage="generating_and_ranking")
                try:
                    with observer.entity(row):
                        result = pipeline.finish(current["prepared"])
                    persist_entity_result(job, writer, current, result)
                    result = None
                except Exception as exc:
                    record_entity_failure(job, writer, current, exc)
                except BaseException:
                    row["status"] = "interrupted"
                    raise
                finally:
                    finish_entity(job, writer, current)
                    if current["row"]["status"] == "failed":
                        try:
                            qwen_health(job.config["qwen_endpoint"])
                        except Exception as exc:
                            fatal_error = exc
                    current = None
        finally:
            engine.judge_wait_callback = original_callback
            if pending is not None and not pending.get("finalized"):
                pending["row"]["status"] = "interrupted"
                finish_entity(job, writer, pending)


def verify_inputs(config):
    for name, expected in config["input_sha256"].items():
        if sha256(name) != expected:
            raise RuntimeError(f"Queued input changed: {name}")


def qwen_health(endpoint):
    body = {"model": "Qwen/Qwen2-VL-7B-Instruct", "messages": [{"role": "user", "content": "Reply OK."}],
            "temperature": 0, "max_tokens": 4}
    req = request.Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
    if not result.get("choices") or not result["choices"][0].get("message", {}).get("content"):
        raise RuntimeError("Qwen health request returned no content")


def wait_for_test(job):
    prerequisite = job.config.get("prerequisite")
    if not prerequisite:
        return True
    job.state.update(status="queued", waiting_for=prerequisite["run_dir"], current_stage="waiting_for_test50")
    job.flush(reports=True)
    last_progress = None
    while True:
        progress = load(Path(prerequisite["run_dir"]) / "timing_progress.json")
        identity = process_identity(prerequisite["pid"])
        alive = identity is not None and identity == prerequisite["process_identity"]
        decision = prerequisite_decision(progress, alive)
        view = {key: progress.get(key) for key in ("status", "completed_entities", "failed_entities", "current_entity", "current_stage")}
        job.state["prerequisite_progress"] = view
        job.flush()
        if view != last_progress:
            job.event("PREREQUISITE_PROGRESS", **view)
            last_progress = view
        if decision == "ready":
            job.state.pop("waiting_for", None)
            job.event("PREREQUISITE_FINISHED", run_dir=prerequisite["run_dir"])
            return True
        if decision == "blocked":
            job.state.update(status="blocked", error_message="50-entity prerequisite stopped without a normal completion; full batch remains unstarted.")
            job.flush(reports=True)
            return False
        time.sleep(20)


def worker(root, retry_failed=False):
    job = Job(root)
    with (job.root / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another worker owns this job", file=sys.stderr)
            return 2
        job.state.update(pid=os.getpid(), worker_started_at=now())
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(20):
                try:
                    job.flush()
                except Exception as exc:
                    print(f"Progress update failed: {type(exc).__name__}", flush=True)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if not wait_for_test(job):
                return 2
            verify_inputs(job.config)
            qwen_health(job.config["qwen_endpoint"])
            # All GPU/model imports and initialization happen AFTER test50 exits.
            project = Path(job.config["project_root"])
            sys.path.insert(0, str(project))
            os.chdir(project)
            from mmkgc.data import MKGWDataStore
            from mmkgc.defaults import build_default_mainline_runtime
            from mmkgc.io import RunWriter
            initialize_stage_layout(job)
            job.clock_started = time.perf_counter()
            job.state.update(status="initializing", current_stage="initialization", pipeline_started_at=now())
            job.state.pop("error_message", None)
            job.flush(reports=True)
            store = MKGWDataStore(job.config["dataset_dir"], graph_split="train")
            runtime = build_default_mainline_runtime(**job.config["runtime_kwargs"])
            attempts = job.root / "attempts"
            attempts.mkdir(exist_ok=True)
            number = max([int(path.name.split("_")[-1]) for path in attempts.glob("attempt_*") if path.is_dir()] or [0]) + 1
            writer = RunWriter(attempts, f"attempt_{number:04d}", {"job": str(job.root), **job.config["runtime_kwargs"]})
            job.state.update(status="running", current_stage="between_entities", attempt_dir=str(writer.run_dir))
            job.flush(reports=True)
            job.event("FULL_BATCH_START", expected_entities=len(job.manifest), attempt=number)
            run_entities(job, runtime.pipeline, store, writer, retry_failed)
            job.state.update(status="completed" if all(row["status"] == "completed" for row in job.rows.values()) else "completed_with_failures", finished_at=now())
            return 0
        except BaseException as exc:
            job.state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                             error_type=type(exc).__name__, error_message=str(exc))
            job.event("WORKER_STOPPED", error_type=type(exc).__name__)
            if not isinstance(exc, KeyboardInterrupt):
                import traceback
                traceback.print_exc()
            return 1
        finally:
            stop.set()
            thread.join(timeout=1)
            job.flush(reports=True)


def prepare(args):
    import yaml
    sys.path.insert(0, str(PROJECT))
    from mmkgc.data import MKGWDataStore

    if type(args.candidate_budget) is not int or args.candidate_budget <= 0:
        raise ValueError("candidate_budget must be a positive integer")
    dataset, manifest = Path(args.dataset_dir).resolve(), Path(args.entity_manifest).resolve()
    rows = load(manifest)
    original = load(dataset / "test_missing_image_qids.json")
    keys = [row["entity_key"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(original) or len(keys) != args.expected_entities:
        raise ValueError("Manifest must match the complete original missing-image list without duplicates")
    if any(not re.fullmatch(r"Q[0-9]+", key) for key in keys):
        raise ValueError("Invalid entity key in manifest")
    store = MKGWDataStore(dataset, graph_split="train")
    test_keys = {part.rsplit("/", 1)[-1] for line in (dataset / "test.txt").read_text().splitlines()
                 for part in line.split()[:3:2]}
    if not set(keys) <= test_keys:
        raise ValueError("Target entities must all occur in test.txt")
    for row in rows:
        if store.entity_key(row["entity_id"]) != row["entity_key"]:
            raise ValueError("Entity mapping and manifest disagree")
    root = Path(args.job_dir).resolve()
    root.relative_to((PROJECT / "outputs").resolve())
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "config_snapshots"
    snapshots.mkdir()
    save(root / "entity_manifest.json", rows)
    shutil.copyfile(dataset / "test_missing_image_qids.json", snapshots / "original_test_missing_image_qids.json")
    for name in ("qwen7b-mainline.yaml", "qwen7b-stage1.yaml", "qwen7b-stage2.yaml", "openai-gpt4o-mini-judge.yaml"):
        shutil.copyfile(PROJECT / "configs" / name, snapshots / name)
    stage4 = yaml.safe_load((PROJECT / "configs/stage4-model-fusion-gpu1-visible.yaml").read_text())
    stage4["gpt4omini_judge"]["model_config"] = str(snapshots / "openai-gpt4o-mini-judge.yaml")
    (snapshots / "stage4-model-fusion-gpu1-visible.yaml").write_text(yaml.safe_dump(stage4, sort_keys=False))
    runner = snapshots / "completion_runner.py"
    shutil.copyfile(Path(__file__), runner)
    kwargs = dict(missing_mode="image", candidate_budget=args.candidate_budget, stage1_reranker_model_dir=str(PROJECT / "Qwen3-Reranker-4B"),
                  stage1_reranker_device="cuda:0", stage1_reranker_max_length=8192,
                  stage1_model_config=str(snapshots / "qwen7b-stage1.yaml"), stage2_model_config=str(snapshots / "qwen7b-stage2.yaml"),
                  model_config=str(snapshots / "qwen7b-mainline.yaml"), model_call_timeout_seconds=120,
                  stage4_config=str(snapshots / "stage4-model-fusion-gpu1-visible.yaml"), dataset_root=str(dataset),
                  sdxl_model_dir=str(PROJECT / "SDXL/stable-diffusion-xl-base-1.0"), sdxl_width=1024, sdxl_height=1024,
                  sdxl_steps=20, sdxl_seed=20261012, sdxl_guidance_scale=7.5, sdxl_device="cuda:0")
    prerequisite = None
    if args.wait_for_run:
        progress = load(Path(args.wait_for_run) / "timing_progress.json")
        prerequisite = {"run_dir": str(Path(args.wait_for_run).resolve()), "pid": progress.get("pid"),
                        "process_identity": process_identity(progress.get("pid"))}
    sources = sorted((PROJECT / "mmkgc").rglob("*.py"))
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, source.relative_to(PROJECT))
    inputs = sources + [root / "entity_manifest.json"] + list(snapshots.iterdir()) + [dataset / name for name in
        ("entity2id.txt", "relation2id.txt", "train.txt", "valid.txt", "test.txt", "entity_info.json", "entity_texts.tsv", "test_missing_image_qids.json", "relation_info.json") if (dataset / name).exists()]
    config = {"created_at": now(), "project_root": str(PROJECT), "dataset_dir": str(dataset), "prerequisite": prerequisite,
              "scope": "regenerate_original_missing_image_list", "expected_entities": len(rows),
              "candidate_budget": args.candidate_budget,
              "entity_pipeline": not getattr(args, "serial_entities", False),
              "storage_policy": STORAGE_POLICY,
              "source_images_untouched": True, "old_synthetic_images_used_as_ground_truth": False,
              "graph_split": "train", "zero_train_degree_entities": sum(row.get("train_degree") == 0 for row in rows),
              "runtime_kwargs": kwargs, "qwen_endpoint": "http://127.0.0.1:12345/v1",
              "input_sha256": {str(path): sha256(path) for path in inputs}}
    save(root / "job_config.json", config)
    job = Job(root)
    initialize_stage_layout(job)
    job.state.update(status="prepared", current_entity=None, current_stage="prepared")
    job.flush(reports=True)
    return root


def launch(root, retry_failed=False):
    job = Job(root)
    with (job.root / "worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This job already has a running worker") from None
    log = Path(job.config["project_root"]) / "outputs" / "_logs" / (job.root.name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("PIPELINE_GPU", "1"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    command = [sys.executable, "-u", str(job.root / "config_snapshots/completion_runner.py"), "--worker", str(job.root)]
    if retry_failed:
        command.append("--retry-failed")
    with log.open("a") as stream:
        process = subprocess.Popen(command,
                                   cwd=job.config["project_root"], env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"pid": process.pid, "run_dir": str(job.root), "log": str(log), "queued_after": job.config["prerequisite"],
               "progress": str(job.root / "progress.json"), "report": str(job.root / "timing_report.html")}
    save(job.root / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(PROJECT / "datasets/MKG-W bak"))
    parser.add_argument("--entity-manifest", default=str(PROJECT / "configs/mkgw-test-original-missing-images-4580.json"))
    parser.add_argument("--expected-entities", type=int, default=4580)
    parser.add_argument("--candidate-budget", type=int, default=3, help="Candidate images per entity (default: 3)")
    parser.add_argument("--serial-entities", action="store_true", help="For a new job, disable next-entity preparation during GPT waits")
    parser.add_argument("--job-dir")
    parser.add_argument("--wait-for-run")
    parser.add_argument("--launch", action="store_true", help="Start a detached worker for a new job, or resume with --worker")
    parser.add_argument("--worker", type=Path, help="Run/resume a prepared job in the foreground")
    parser.add_argument("--retry-failed", action="store_true", help="With --worker, also retry terminal failures and rejected entities")
    args = parser.parse_args()
    if args.worker:
        if args.launch:
            launch(args.worker, retry_failed=args.retry_failed)
            return 0
        return worker(args.worker, args.retry_failed)
    if not args.job_dir:
        parser.error("--job-dir is required when preparing a new job")
    root = prepare(args)
    if args.launch:
        launch(root)
    else:
        print(json.dumps({"prepared_job": str(root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
