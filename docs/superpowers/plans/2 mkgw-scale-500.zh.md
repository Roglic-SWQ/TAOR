# MKG-W Scale 500 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the already validated MMKGC offline pipeline at 500 MKG-W entities, evaluate it with calibrated metrics, and produce a Chinese scale-validation report.

**Architecture:** This plan does not change generation or evaluation code. It runs the existing CLI with the confirmed 500-entity configuration, then runs the existing evaluation CLI, verifies raw/calibrated outputs, and writes a report from actual generated artifacts. The existing 100-entity acceptance run remains untouched.

**Tech Stack:** Python 3.11, existing `mmkgc.run`, existing `mmkgc.evaluate`, pytest, compileall, csv/json/pathlib from the Python standard library.

---

## Controller Rules

The controller must enforce the user-confirmed workflow:

1. Main Codex session acts as controller.
2. Each plan task gets one implementer subagent.
3. Each implementer must receive explicit write scope, validation command, and `no commit` instruction.
4. After each task, dispatch two reviewers:
   - spec reviewer: checks task output against the spec and user-confirmed decisions.
   - quality reviewer: checks execution quality, path risk, artifact risk, report quality, and scope boundaries.
5. Reviewer issues must go back to implementer for fixes.
6. Mark a task complete only after both reviewers approve.
7. Dispatch a final reviewer after all tasks.
8. Run final complete verification and report evidence.
9. Do not revert unrelated or user changes.
10. Do not automatically commit.

## Confirmed Configuration

```text
dataset_dir: ../KGC/datasets/MKG-W
limit: 500
missing_mode: both
candidate_budget: 5
output_root: outputs
run_name: scale_500_both_20260707
run_dir: outputs/scale_500_both_20260707
evaluation_dir: outputs/scale_500_both_20260707/evaluation
validation_report: docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

## Files And Artifacts

Create or regenerate:

```text
MMKGC/outputs/scale_500_both_20260707/
MMKGC/outputs/scale_500_both_20260707/evaluation/
MMKGC/docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

Read but do not modify:

```text
MMKGC/docs/superpowers/specs/2026-07-07-mkgw-scale-500-design.zh.md
MMKGC/outputs/acceptance_pytest_20260706_0002/
```

Forbidden:

```text
MMKGC/mmkgc/run.py
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/stages/
MMKGC/mmkgc/evaluation/
MMKGC/tests/
MMKG/
KGC/
```

## Task 1: Preflight Verification And Output Guard

**Goal:** Confirm the codebase is currently passing tests and that the 500 run output path is not already occupied.

**Files:**

- Read: `MMKGC/docs/superpowers/specs/2026-07-07-mkgw-scale-500-design.zh.md`
- Read: `MMKGC/outputs/acceptance_pytest_20260706_0002/evaluation/evaluation_summary.json`
- Do not create or modify project files.

**Write scope:** none.

**Commit:** forbidden.

- [ ] **Step 1: Confirm current directory**

Run:

```bash
pwd
```

Expected:

```text
/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
```

- [ ] **Step 2: Verify output directory is absent**

Run:

```bash
test ! -e outputs/scale_500_both_20260707
```

Expected:

```text
exit code 0
```

If this command exits nonzero, stop the task and report `BLOCKED`. Do not delete or overwrite the directory.

- [ ] **Step 3: Run full tests**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
```

Expected:

```text
PASS
```

The current expected count after calibration work is `94 passed`.

- [ ] **Step 4: Run compileall**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
```

Expected:

```text
exit code 0
```

- [ ] **Step 5: Report preflight evidence**

Report these exact fields to the controller:

```text
Status
output_dir_absent
pytest_result
compileall_exit_code
files_changed
```

`files_changed` must be `none`.

## Task 2: Run 500 Entity Generation

**Goal:** Run the existing offline generator for 500 MKG-W entities in `both` mode with `candidate_budget=5`.

**Files:**

- Create: `MMKGC/outputs/scale_500_both_20260707/`
- Do not modify code, tests, README, docs, the 100-entity acceptance run, or Stage 0-5 implementation files.

**Write scope:**

```text
MMKGC/outputs/scale_500_both_20260707/
```

**Commit:** forbidden.

- [ ] **Step 1: Re-check output directory is absent**

Run:

```bash
test ! -e outputs/scale_500_both_20260707
```

Expected:

```text
exit code 0
```

If nonzero, stop and report `BLOCKED`; do not delete or overwrite.

- [ ] **Step 2: Run 500 entity generation**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir ../KGC/datasets/MKG-W \
  --missing-mode both \
  --limit 500 \
  --candidate-budget 5 \
  --output-root outputs \
  --run-name scale_500_both_20260707
```

Expected:

```text
process exits with code 0
```

- [ ] **Step 3: Inspect run summary**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import json
from pathlib import Path
root = Path("outputs/scale_500_both_20260707")
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "metrics_summary.json").read_text(encoding="utf-8"))
print("dataset_dir", config.get("dataset_dir"))
print("missing_mode", config.get("missing_mode"))
print("limit", config.get("limit"))
print("candidate_budget", config.get("candidate_budget"))
print("processed_entities", metrics.get("processed_entities"))
print("failed_entities", metrics.get("failed_entities"))
print("avg_text_candidates", metrics.get("avg_text_candidates"))
print("avg_image_candidates", metrics.get("avg_image_candidates"))
print("avg_pair_candidates", metrics.get("avg_pair_candidates"))
print("stage_failure_counts", metrics.get("stage_failure_counts"))
'
```

Expected final-success values:

```text
dataset_dir ../KGC/datasets/MKG-W
missing_mode both
limit 500
candidate_budget 5
processed_entities 500
failed_entities 0
avg_text_candidates 5.0
avg_image_candidates 5.0
avg_pair_candidates 9.0
stage_failure_counts {}
```

If `failed_entities` is greater than `0`, still report the values to the controller. The controller may allow Task 3 to run for diagnostic reporting, but the stage cannot be marked complete until failures are fixed and the run is regenerated with `failed_entities = 0`.

- [ ] **Step 4: Confirm expected run files exist**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
from pathlib import Path
root = Path("outputs/scale_500_both_20260707")
paths = [
    root / "config.json",
    root / "results.jsonl",
    root / "metrics_summary.json",
    root / "entities",
]
for path in paths:
    print(path, path.exists())
'
```

Expected:

```text
outputs/scale_500_both_20260707/config.json True
outputs/scale_500_both_20260707/results.jsonl True
outputs/scale_500_both_20260707/metrics_summary.json True
outputs/scale_500_both_20260707/entities True
```

- [ ] **Step 5: Report generation evidence**

Report these exact fields to the controller:

```text
Status
run_command_exit_code
processed_entities
failed_entities
stage_failure_counts
avg_text_candidates
avg_image_candidates
avg_pair_candidates
created_run_dir
concerns
```

## Task 3: Run Evaluation And Scale Acceptance Checks

**Goal:** Evaluate the 500 run using the calibrated evaluation layer and collect machine-readable acceptance evidence.

**Files:**

- Create or overwrite: `MMKGC/outputs/scale_500_both_20260707/evaluation/`
- Read: `MMKGC/outputs/scale_500_both_20260707/metrics_summary.json`
- Do not modify code, tests, README, docs, or the 100-entity acceptance run.

**Write scope:**

```text
MMKGC/outputs/scale_500_both_20260707/evaluation/
```

**Commit:** forbidden.

- [ ] **Step 1: Run evaluation**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/scale_500_both_20260707 \
  --overwrite
```

Expected final-success output:

```json
{"evaluation_dir": "outputs/scale_500_both_20260707/evaluation", "evaluated_entities": 500, "failed_entities": 0, "failed_entity_keys": []}
```

- [ ] **Step 2: Check output files and calibrated fields**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import csv
from pathlib import Path
root = Path("outputs/scale_500_both_20260707/evaluation")
required_files = [
    "evaluation_summary.json",
    "summary.csv",
    "entity_scores.csv",
    "best_cases.jsonl",
    "worst_cases.jsonl",
    "fallback_cases.jsonl",
]
for filename in required_files:
    path = root / filename
    print(filename, path.exists())
rows = list(csv.DictReader(open(root / "entity_scores.csv", encoding="utf-8")))
required_columns = {
    "overall_offline_evaluation_score",
    "calibrated_overall_offline_evaluation_score",
    "text_best_offline_evaluation_score",
    "calibrated_text_best_offline_evaluation_score",
    "image_best_offline_evaluation_score",
    "calibrated_image_best_offline_evaluation_score",
    "pair_best_offline_evaluation_score",
    "calibrated_pair_best_offline_evaluation_score",
    "calibration_tags",
}
print("rows", len(rows))
print("missing_columns", sorted(required_columns.difference(rows[0].keys() if rows else [])))
'
```

Expected:

```text
evaluation_summary.json True
summary.csv True
entity_scores.csv True
best_cases.jsonl True
worst_cases.jsonl True
fallback_cases.jsonl True
rows 500
missing_columns []
```

- [ ] **Step 3: Run scale acceptance check**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import csv, json, pathlib
root = pathlib.Path("outputs/scale_500_both_20260707")
eval_root = root / "evaluation"
metrics = json.loads((root / "metrics_summary.json").read_text(encoding="utf-8"))
summary = json.loads((eval_root / "evaluation_summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open(eval_root / "entity_scores.csv", encoding="utf-8")))
fallback = [r for r in rows if "fallback" in r["case_tags"] or "fallback_penalized" in r.get("calibration_tags", "")]
print("processed_entities", metrics.get("processed_entities"))
print("run_failed_entities", metrics.get("failed_entities"))
print("stage_failure_counts", metrics.get("stage_failure_counts"))
print("evaluated_entities", summary.get("evaluated_entities"))
print("evaluation_failed_entities", summary.get("failed_entities"))
print("evaluation_failed_entity_keys", summary.get("failed_entity_keys"))
print("rows", len(rows))
print("raw_overall_avg", summary["averages"].get("overall_offline_evaluation_score"))
print("calibrated_overall_avg", summary["averages"].get("calibrated_overall_offline_evaluation_score"))
print("fallback_count", len(fallback))
if fallback:
    raw_avg = sum(float(r["overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
    cal_avg = sum(float(r["calibrated_overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
    print("fallback_raw_avg", round(raw_avg, 6))
    print("fallback_calibrated_avg", round(cal_avg, 6))
    print("fallback_drop", round(raw_avg - cal_avg, 6))
    print("fallback_all_above_070", all(float(r["calibrated_overall_offline_evaluation_score"]) > 0.70 for r in fallback))
else:
    print("fallback_raw_avg", "NA")
    print("fallback_calibrated_avg", "NA")
    print("fallback_drop", "NA")
    print("fallback_all_above_070", "NA")
print("generic_entity_type_entities", summary["counts"].get("generic_entity_type_entities"))
print("pair_single_side_penalized_entities", summary["counts"].get("pair_single_side_penalized_entities"))
print("template_prompt_penalized_entities", summary["counts"].get("template_prompt_penalized_entities"))
print("best_cases", summary["case_exports"].get("best_cases"))
print("worst_cases", summary["case_exports"].get("worst_cases"))
print("fallback_cases", summary["case_exports"].get("fallback_cases"))
'
```

Expected final-success values:

```text
processed_entities 500
run_failed_entities 0
stage_failure_counts {}
evaluated_entities 500
evaluation_failed_entities 0
evaluation_failed_entity_keys []
rows 500
best_cases 20
worst_cases 20
```

If `fallback_count` is greater than `0`, expected:

```text
fallback_calibrated_avg < fallback_raw_avg
fallback_drop > 0
```

If `fallback_count` is `0`, the `NA` fallback values are acceptable and must be explained in the report.

- [ ] **Step 4: Report evaluation evidence**

Report these exact fields to the controller:

```text
Status
evaluate_exit_code
evaluated_entities
evaluation_failed_entities
rows
missing_columns
raw_overall_avg
calibrated_overall_avg
fallback_count
fallback_raw_avg
fallback_calibrated_avg
fallback_drop
best_cases
worst_cases
fallback_cases
concerns
```

## Task 4: Write Scale 500 Validation Report

**Goal:** Create a Chinese validation report from actual 500-run and evaluation artifacts.

**Files:**

- Create: `MMKGC/docs/evaluation/2026-07-07-scale-500-validation.zh.md`
- Read: `MMKGC/outputs/scale_500_both_20260707/config.json`
- Read: `MMKGC/outputs/scale_500_both_20260707/metrics_summary.json`
- Read: `MMKGC/outputs/scale_500_both_20260707/evaluation/evaluation_summary.json`
- Read: `MMKGC/outputs/scale_500_both_20260707/evaluation/entity_scores.csv`

**Write scope:**

```text
MMKGC/docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

**Commit:** forbidden.

- [ ] **Step 1: Confirm source artifacts exist**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
from pathlib import Path
paths = [
    Path("outputs/scale_500_both_20260707/config.json"),
    Path("outputs/scale_500_both_20260707/metrics_summary.json"),
    Path("outputs/scale_500_both_20260707/evaluation/evaluation_summary.json"),
    Path("outputs/scale_500_both_20260707/evaluation/entity_scores.csv"),
]
for path in paths:
    print(path, path.exists())
'
```

Expected:

```text
outputs/scale_500_both_20260707/config.json True
outputs/scale_500_both_20260707/metrics_summary.json True
outputs/scale_500_both_20260707/evaluation/evaluation_summary.json True
outputs/scale_500_both_20260707/evaluation/entity_scores.csv True
```

- [ ] **Step 2: Generate report content from artifacts**

Use this command to write the report from actual artifact values:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import csv, json
from pathlib import Path

run_root = Path("outputs/scale_500_both_20260707")
eval_root = run_root / "evaluation"
report_path = Path("docs/evaluation/2026-07-07-scale-500-validation.zh.md")

config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
metrics = json.loads((run_root / "metrics_summary.json").read_text(encoding="utf-8"))
summary = json.loads((eval_root / "evaluation_summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open(eval_root / "entity_scores.csv", encoding="utf-8")))
fallback = [r for r in rows if "fallback" in r["case_tags"] or "fallback_penalized" in r.get("calibration_tags", "")]

fallback_lines = []
if fallback:
    fallback_raw_avg = sum(float(r["overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
    fallback_cal_avg = sum(float(r["calibrated_overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
    fallback_drop = fallback_raw_avg - fallback_cal_avg
    fallback_all_above = all(float(r["calibrated_overall_offline_evaluation_score"]) > 0.70 for r in fallback)
    fallback_lines.extend([
        f"- fallback raw overall 平均值: {fallback_raw_avg:.6f}",
        f"- fallback calibrated overall 平均值: {fallback_cal_avg:.6f}",
        f"- fallback 平均降幅: {fallback_drop:.6f}",
        f"- fallback calibrated 是否全部高于 0.70: {fallback_all_above}",
    ])
    fallback_conclusion = "满足：fallback calibrated overall 低于 fallback raw overall。" if fallback_drop > 0 else "不满足：fallback calibrated overall 未低于 fallback raw overall。"
else:
    fallback_lines.extend([
        "- fallback raw overall 平均值: NA",
        "- fallback calibrated overall 平均值: NA",
        "- fallback 平均降幅: NA",
        "- fallback calibrated 是否全部高于 0.70: NA",
    ])
    fallback_conclusion = "本次 500 样本没有 entity-level fallback 样本，fallback 均值对比不适用。"

run_failed = int(metrics.get("failed_entities") or 0)
eval_failed = int(summary.get("failed_entities") or 0)
processed = int(metrics.get("processed_entities") or 0)
evaluated = int(summary.get("evaluated_entities") or 0)
stage_failures = metrics.get("stage_failure_counts") or {}
complete = (
    processed == 500
    and run_failed == 0
    and evaluated == 500
    and eval_failed == 0
    and len(rows) == 500
    and summary["case_exports"].get("best_cases", 0) > 0
    and summary["case_exports"].get("worst_cases", 0) > 0
    and (not fallback or fallback_drop > 0)
)

report = f"""# MMKGC 500 实体扩大运行验证报告

## 验证对象

- Run path: `outputs/scale_500_both_20260707`
- Evaluation path: `outputs/scale_500_both_20260707/evaluation`
- Dataset: `{config.get("dataset_dir")}`
- Missing mode: `{config.get("missing_mode")}`
- Limit: `{config.get("limit")}`
- Candidate budget: `{config.get("candidate_budget")}`

## Run 稳定性

- processed_entities: {processed}
- failed_entities: {run_failed}
- stage_failure_counts: `{stage_failures}`
- avg_text_candidates: {metrics.get("avg_text_candidates")}
- avg_image_candidates: {metrics.get("avg_image_candidates")}
- avg_pair_candidates: {metrics.get("avg_pair_candidates")}

## Evaluation 稳定性

- evaluated_entities: {evaluated}
- failed_entities: {eval_failed}
- failed_entity_keys: `{summary.get("failed_entity_keys")}`
- entity_scores.csv rows: {len(rows)}

## Raw / Calibrated 分布

- raw overall 平均值: {summary["averages"].get("overall_offline_evaluation_score")}
- calibrated overall 平均值: {summary["averages"].get("calibrated_overall_offline_evaluation_score")}
- generic_entity_type_entities: {summary["counts"].get("generic_entity_type_entities")}
- pair_single_side_penalized_entities: {summary["counts"].get("pair_single_side_penalized_entities")}
- template_prompt_penalized_entities: {summary["counts"].get("template_prompt_penalized_entities")}

## Fallback 校准对比

- fallback_count: {len(fallback)}
{chr(10).join(fallback_lines)}

{fallback_conclusion}

## Case 导出

- best_cases: {summary["case_exports"].get("best_cases")}
- worst_cases: {summary["case_exports"].get("worst_cases")}
- fallback_cases: {summary["case_exports"].get("fallback_cases")}

## 结论

- 是否达到 processed_entities = 500: {processed == 500}
- run failed_entities 是否为 0: {run_failed == 0}
- evaluation evaluated_entities 是否为 500: {evaluated == 500}
- evaluation failed_entities 是否为 0: {eval_failed == 0}
- entity_scores.csv 是否为 500 行: {len(rows) == 500}
- case 导出是否正常: {summary["case_exports"].get("best_cases", 0) > 0 and summary["case_exports"].get("worst_cases", 0) > 0}
- 本阶段是否满足完成定义: {complete}

## 验证命令

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 500 --candidate-budget 5 --output-root outputs --run-name scale_500_both_20260707
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/scale_500_both_20260707 --overwrite
```
"""

report_path.write_text(report, encoding="utf-8")
print(report_path)
'
```

Expected:

```text
docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

- [ ] **Step 3: Scan report for unresolved content**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
from pathlib import Path
text = Path("docs/evaluation/2026-07-07-scale-500-validation.zh.md").read_text(encoding="utf-8")
markers = ["TO" + "DO", "T" + "BD", "<" + "填入" + ">", "." * 3, "place" + "holder"]
matches = [marker for marker in markers if marker in text]
print("unresolved_markers", matches)
raise SystemExit(1 if matches else 0)
'
```

Expected:

```text
unresolved_markers []
exit code 0
```

- [ ] **Step 4: Review report excerpt**

Run:

```bash
sed -n '1,220p' docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

Expected:

```text
the report contains concrete values from the 500 run and no unresolved template text
```

- [ ] **Step 5: Report documentation evidence**

Report these exact fields to the controller:

```text
Status
report_path
report_created
unresolved_content_scan_exit_code
completion_definition_in_report
concerns
```

## Task 5: Final Verification

**Goal:** Run full final verification after the 500 report exists.

**Files:**

- Read: all generated 500 run/evaluation/report artifacts.
- Do not create or modify files unless the final verification command regenerates evaluation with the approved `--overwrite` option.

**Write scope:**

```text
MMKGC/outputs/scale_500_both_20260707/evaluation/
```

Only the evaluation CLI may write here during final verification. Do not modify code, tests, README, spec, plan, or report in this task.

**Commit:** forbidden.

- [ ] **Step 1: Run full tests**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
```

Expected:

```text
PASS
```

- [ ] **Step 2: Run compileall**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: Re-run evaluation**

Run:

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/scale_500_both_20260707 \
  --overwrite
```

Expected:

```json
{"evaluation_dir": "outputs/scale_500_both_20260707/evaluation", "evaluated_entities": 500, "failed_entities": 0, "failed_entity_keys": []}
```

- [ ] **Step 4: Run final field check**

Run:

```bash
rg -n "calibrated_overall_offline_evaluation_score|calibration_summary|fallback_penalty|entity_type_generic_penalty|single_side_pair_penalty" \
  outputs/scale_500_both_20260707/evaluation docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

Expected:

```text
matches in generated evaluation outputs
```

- [ ] **Step 5: Run final acceptance check**

Run the same acceptance-check command from Task 3 Step 3.

Expected final-success values:

```text
processed_entities 500
run_failed_entities 0
evaluated_entities 500
evaluation_failed_entities 0
rows 500
best_cases 20
worst_cases 20
```

- [ ] **Step 6: Report final verification evidence**

Report these exact fields to the controller:

```text
Status
pytest_result
compileall_exit_code
evaluate_result
field_check_exit_code
processed_entities
run_failed_entities
evaluated_entities
evaluation_failed_entities
rows
fallback_count
fallback_drop
best_cases
worst_cases
fallback_cases
concerns
```

## Final Reviewer Requirements

The final reviewer must check:

- The 500 run uses `both`, `limit=500`, `candidate_budget=5`, and run name `scale_500_both_20260707`.
- The 100-entity acceptance run is not overwritten.
- No Stage 0-5 implementation files were changed.
- No evaluation calibration formulas were changed.
- Run artifacts exist and contain 500 processed entities.
- Evaluation artifacts exist and contain 500 evaluated entities.
- Raw and calibrated fields are present in `entity_scores.csv`, case JSONL files, summary JSON, and per-entity evaluation JSON.
- `best_cases.jsonl` and `worst_cases.jsonl` are non-empty.
- If fallback samples exist, fallback calibrated overall is lower than fallback raw overall.
- `docs/evaluation/2026-07-07-scale-500-validation.zh.md` contains concrete values and a clear conclusion.
- Full verification commands passed.
- No commit was made.
