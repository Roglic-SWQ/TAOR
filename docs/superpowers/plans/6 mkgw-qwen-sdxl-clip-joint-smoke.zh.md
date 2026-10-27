# MKG-W Qwen + SDXL + CLIP Joint Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and validate a 5-entity joint real-model smoke where Qwen generates text candidates, SDXL generates real PNGs, and CLIP evaluates the PNGs.

**Architecture:** This stage does not add application code. It orchestrates existing Qwen, SDXL, offline evaluation, and CLIP evaluation entry points, then records strict validation evidence in a Chinese smoke report. The controller must keep model weights, generated outputs, pycache, and user-renamed docs out of accidental commits.

**Tech Stack:** Python 3.11, existing `mmkgc.run`, existing `mmkgc.evaluate`, vLLM OpenAI-compatible Qwen service, local SDXL Base 1.0, local OpenAI CLIP ViT-B-32, pytest/compileall for regression checks.

---

## Controller Rules For This Plan

- 主 Codex 会话作为 controller。
- 每个 task 派一个 implementer subagent。
- 每个 implementer 必须收到本 task 的写入范围、验证命令、禁止 commit 指令。
- 每个任务完成后派两个 reviewer：
  - spec reviewer：检查是否符合 `docs/superpowers/specs/6 mkgw-qwen-sdxl-clip-joint-smoke-design.zh.md` 和用户确认方案。
  - quality reviewer：检查命令质量、报告质量、路径/权重/outputs/pycache/提交风险。
- reviewer 发现问题时，必须回给 implementer 修，不跳过。
- 两个 reviewer 都 approve 后才标记 task 完成。
- 最后派 final reviewer 审完整变更。
- 最后运行完整验证命令并汇报证据。
- 工作区已有用户重命名文档的未提交改动，不 revert 任何非本任务改动。
- 不自动 commit。

## Existing Dirty Workspace Notes

执行前必须保留以下既有状态：

```text
旧日期命名 plan/spec 文件已被用户整理为编号命名文件
Qwen/
SDXL/
CLIP/
outputs/
```

不要恢复旧文档名，不要提交模型权重或 generated outputs。

## Files And Responsibilities

- No intended code changes.
- No intended test changes.
- Create `docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md`
  - Owns the final stage 7 smoke evidence.
  - Must contain real numbers copied from validation outputs, not placeholders.
- Modify `docs/project-progress-summary.zh.md`
  - Add stage 7 summary only after the run succeeds or explicitly record failure if it fails.
- Modify `README.md` and `README.zh.md` only if the controller decides the joint smoke command should be documented in the README now.
  - If edited, state that this is a small joint smoke, not a scale experiment.
- Generated run directory:
  - `outputs/qwen_sdxl_clip_joint_smoke_20260709`
  - Must remain ignored by git.

## Task 1: Preflight And Qwen Service Health Check

**Files:**
- No writes expected.

**Implementer scope:**
- Verify local prerequisites and Qwen service readiness.
- Do not start long-running services from Codex unless the controller explicitly approves escalated execution.
- Do not commit.

- [ ] **Step 1: Confirm local files exist**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test -d datasets/MKG-W
test -f configs/mkgw-knowledge-extraction.yaml
test -d Qwen/Qwen2-VL-7B-Instruct
test -d SDXL/stable-diffusion-xl-base-1.0
test -f CLIP/ViT-B-32.pt
```

Expected:

```text
exit code 0
```

- [ ] **Step 2: Ask user to start Qwen/vLLM on GPU 0 if it is not already running**

User-facing command:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

CUDA_VISIBLE_DEVICES=0 vllm serve /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC/Qwen/Qwen2-VL-7B-Instruct \
  --dtype auto \
  --api-key token-abc123 \
  --max_model_len 8192 \
  --port 12345 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.9 \
  --disable-log-stats \
  --trust-remote-code \
  --limit-mm-per-prompt image=4
```

Expected:

```text
vLLM listens on http://localhost:12345/v1
GPU 0 memory usage rises in nvidia-smi
```

- [ ] **Step 3: Run Qwen health check**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
QWEN_HEALTH_TIMEOUT_SECONDS=30 /home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py
```

Expected JSON:

```json
{"ok": true, "base_url": "http://localhost:12345/v1"}
```

If the normal sandbox cannot reach localhost, controller may rerun the same command with escalated execution. If health check fails, stop this task and report the exact JSON error.

- [ ] **Step 4: Check GPU split**

Run:

```bash
nvidia-smi
```

Expected:

```text
GPU 0 has Qwen/vLLM memory usage
GPU 1 has enough free memory for SDXL
```

If run from Codex normal sandbox, `nvidia-smi` may fail because GPU devices are not exposed. In that case, run through approved escalated execution or ask the user to paste `nvidia-smi`.

## Task 2: Run Joint Qwen + SDXL Smoke

**Files:**
- Generated output only: `outputs/qwen_sdxl_clip_joint_smoke_20260709/`
- No code or docs writes.

**Implementer scope:**
- Run the existing pipeline with Qwen text generation and SDXL image generation.
- Strictly reject fallback as success.
- Do not commit.

- [ ] **Step 1: Run the joint smoke command**

Run from an environment that can access GPU 1:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260708 \
  --sdxl-guidance-scale 7.5 \
  --sdxl-device cuda:1 \
  --output-root outputs \
  --run-name qwen_sdxl_clip_joint_smoke_20260709
```

Expected:

```text
exit code 0
outputs/qwen_sdxl_clip_joint_smoke_20260709 exists
```

If the run directory already exists, do not delete it without explicit controller approval. Use a new run name only if the controller updates the spec/report expectation.

- [ ] **Step 2: Validate run metrics**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -c "import json; from pathlib import Path; run=Path('outputs/qwen_sdxl_clip_joint_smoke_20260709'); d=json.load(open(run/'metrics_summary.json', encoding='utf-8')); assert d['processed_entities'] == 5, d; assert d['failed_entities'] == 0, d; print(json.dumps({'processed_entities': d['processed_entities'], 'failed_entities': d['failed_entities']}, ensure_ascii=False))"
```

Expected:

```json
{"processed_entities": 5, "failed_entities": 0}
```

- [ ] **Step 3: Validate Qwen calls are all real generated calls**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -c "import json; from pathlib import Path; run=Path('outputs/qwen_sdxl_clip_joint_smoke_20260709'); rows=[]; [rows.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()) for p in sorted(run.glob('entities/*/model_calls.jsonl'))]; text_rows=[r for r in rows if r.get('provider') == 'qwen-local-openai-compatible']; image_rows=[r for r in rows if r.get('modality') == 'image']; assert len(rows) == 50, len(rows); assert len(text_rows) == 25, len(text_rows); assert len(image_rows) == 25, len(image_rows); assert all(r.get('status') == 'generated' for r in text_rows), text_rows[0]; assert all(not r.get('fallback_used') for r in text_rows), text_rows[0]; assert all(r.get('metadata', {}).get('model_status') == 'generated' for r in text_rows), text_rows[0]; print(json.dumps({'model_calls_total': len(rows), 'qwen_text_calls': len(text_rows), 'image_calls_in_model_calls': len(image_rows), 'qwen_generated': sum(r.get('status') == 'generated' for r in text_rows), 'qwen_fallback_used': sum(bool(r.get('fallback_used')) for r in text_rows)}, ensure_ascii=False))"
```

Expected:

```json
{"model_calls_total": 50, "qwen_text_calls": 25, "image_calls_in_model_calls": 25, "qwen_generated": 25, "qwen_fallback_used": 0}
```

- [ ] **Step 4: Validate SDXL image artifacts**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py \
  --run-dir outputs/qwen_sdxl_clip_joint_smoke_20260709 \
  --expected-entities 5 \
  --expected-images-per-entity 5 \
  --expected-width 1024 \
  --expected-height 1024
```

Expected:

```json
{"ok": true, "entities_checked": 5, "images_checked": 25}
```

- [ ] **Step 5: Validate SDXL calls are all generated and not fallback**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -c "import json; from pathlib import Path; run=Path('outputs/qwen_sdxl_clip_joint_smoke_20260709'); rows=[]; [rows.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()) for p in sorted(run.glob('entities/*/image_model_calls.jsonl'))]; image_rows=[r for r in rows if r.get('modality') == 'image']; assert len(image_rows) == 25, len(image_rows); assert all(r.get('status') == 'generated' for r in image_rows), image_rows[0]; assert all(not r.get('fallback_used') for r in image_rows), image_rows[0]; assert all(r.get('width') == 1024 and r.get('height') == 1024 for r in image_rows), image_rows[0]; print(json.dumps({'image_model_calls': len(image_rows), 'generated': sum(r.get('status') == 'generated' for r in image_rows), 'fallback_used': sum(bool(r.get('fallback_used')) for r in image_rows)}, ensure_ascii=False))"
```

Expected:

```json
{"image_model_calls": 25, "generated": 25, "fallback_used": 0}
```

## Task 3: Run Offline Evaluation And CLIP Evaluation

**Files:**
- Generated evaluation output under `outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/`
- No code writes.

**Implementer scope:**
- Run existing evaluator with CLIP enabled.
- Validate offline and CLIP output counts.
- Do not commit.

- [ ] **Step 1: Run evaluation with CLIP image quality**

Run from an environment that can access GPU 1:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_sdxl_clip_joint_smoke_20260709 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda:1 \
  --overwrite
```

Expected stdout JSON:

```json
{"evaluated_entities": 5, "failed_entities": 0, "image_quality_evaluator": "clip-local", "clip_evaluated_entities": 5, "clip_evaluated_images": 25}
```

If Codex normal sandbox cannot see CUDA, controller can run this command with approved escalated execution. CPU fallback is allowed only for debugging; final stage 7 success should prefer `cuda:1` because the user selected dual-GPU execution.

- [ ] **Step 2: Validate evaluation summary and CLIP summary**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -c "import json; from pathlib import Path; run=Path('outputs/qwen_sdxl_clip_joint_smoke_20260709'); e=json.load(open(run/'evaluation/evaluation_summary.json', encoding='utf-8')); c=json.load(open(run/'evaluation/clip_image_quality_summary.json', encoding='utf-8')); assert e['evaluated_entities'] == 5, e; assert e['failed_entities'] == 0, e; assert c['evaluated_entities'] == 5, c; assert c['evaluated_images'] == 25, c; print(json.dumps({'evaluation': {'evaluated_entities': e['evaluated_entities'], 'failed_entities': e['failed_entities']}, 'clip': {'evaluated_entities': c['evaluated_entities'], 'evaluated_images': c['evaluated_images'], 'averages': c['averages']}}, ensure_ascii=False))"
```

Expected:

```json
{"evaluation": {"evaluated_entities": 5, "failed_entities": 0}, "clip": {"evaluated_entities": 5, "evaluated_images": 25, "averages": {"clip_prompt_similarity": 0.0, "clip_text_similarity": 0.0, "clip_evidence_similarity": 0.0, "clip_overall_image_quality_score": 0.0}}}
```

The `0.0` values above are schema examples only. The actual command output must contain the real numeric averages, and those actual values must be recorded in the final report.

- [ ] **Step 3: Validate CLIP CSV row count**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
wc -l outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/clip_image_quality_entity_scores.csv
```

Expected:

```text
6 outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/clip_image_quality_entity_scores.csv
```

## Task 4: Write Joint Smoke Report And Progress Update

**Files:**
- Create: `docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md`
- Modify: `docs/project-progress-summary.zh.md`
- Modify: `README.md` only if documenting the joint smoke command is useful.
- Modify: `README.zh.md` only if documenting the joint smoke command is useful.

**Implementer scope:**
- Write documentation using actual outputs from Tasks 1-3.
- Do not invent numbers.
- Do not commit.

- [ ] **Step 1: Create the validation report with real values**

Create `docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md` with this structure. Every metric line must use the actual value copied from Tasks 1-3 before saving; do not save schema examples or prose instructions as metric values.

```markdown
# Qwen + SDXL + CLIP 联合真实模型 Smoke 验收报告

日期：2026-07-09
工作目录：`/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC`

## 范围

本报告验收阶段 7：同一次 MKG-W 5 实体 run 中，使用本地 Qwen2-VL-7B 生成 text candidates，使用本地 SDXL Base 1.0 生成真实 PNG，并使用本地 OpenAI CLIP ViT-B-32 评价 PNG 的跨模态相似度。

## 资源分工

- GPU 0：Qwen/vLLM 服务。
- GPU 1：SDXL 生成与 CLIP 评价。

## Qwen 服务

记录 Qwen 启动命令和 health check JSON。

## Joint Run 命令

记录实际 `python -m mmkgc.run` 命令。

## Run 结果

- `processed_entities = 5`
- `failed_entities = 0`
- `model_calls_total = 50`
- `qwen_text_calls = 25`
- `qwen_generated = 25`
- `qwen_fallback_used = 0`
- `image_model_calls = 25`
- `sdxl_generated = 25`
- `sdxl_fallback_used = 0`
- `png_files = 25`
- `image_size = 1024x1024`

## Evaluation 与 CLIP 结果

- `evaluation.evaluated_entities = 5`
- `evaluation.failed_entities = 0`
- `clip.evaluated_entities = 5`
- `clip.evaluated_images = 25`
- `clip_prompt_similarity = 从 Task 3 Step 2 输出复制真实数值`
- `clip_text_similarity = 从 Task 3 Step 2 输出复制真实数值`
- `clip_evidence_similarity = 从 Task 3 Step 2 输出复制真实数值`
- `clip_overall_image_quality_score = 从 Task 3 Step 2 输出复制真实数值`

## 结论

如果所有严格成功标准都满足，写明阶段 7 joint smoke 通过。如果任一标准不满足，写明失败项和下一步排查建议。

## 限制

- 本阶段只覆盖 5 实体、25 text candidates、25 PNG。
- CLIP 是跨模态相似度，不是 VLM judge。
- 本阶段不代表 50/100/500 实体真实模型实验的质量分布。
```

Expected:

```text
No bracketed placeholder remains.
All numbers match validation command outputs.
```

- [ ] **Step 2: Update project progress summary**

Modify `docs/project-progress-summary.zh.md`:

- Add a new section `阶段 7：Qwen + SDXL + CLIP 联合真实模型 Smoke`.
- Include actual run name `outputs/qwen_sdxl_clip_joint_smoke_20260709`.
- Include actual strict validation counts.
- State whether the stage passed or failed.
- Keep the limitations clear: this is not large-scale evidence and not VLM judging.

Expected:

```text
The progress summary no longer implies that Qwen, SDXL, and CLIP have only been tested separately if the joint run passed.
```

- [ ] **Step 3: Optionally update README**

If controller decides the joint smoke should be user-facing now, update `README.md` and `README.zh.md` with a short command section. The section must state:

- Start Qwen/vLLM separately on GPU 0.
- Run `mmkgc.run` with Qwen and SDXL.
- Run `mmkgc.evaluate` with CLIP.
- This is a 5-entity smoke, not a scale experiment.

If README is not updated, record the reason in the final response.

- [ ] **Step 4: Validate report formatting and placeholders**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
rg -n "Qwen \\+ SDXL \\+ CLIP|qwen_sdxl_clip_joint_smoke_20260709|clip_overall_image_quality_score|fallback" docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md docs/project-progress-summary.zh.md
rg -n "待补|示例值|替换为真实数值|从 Task" docs/evaluation/2026-07-09-qwen-sdxl-clip-joint-smoke-validation.zh.md
```

Expected:

```text
First command finds the stage 7 evidence.
Second command has no output.
```

## Task 5: Final Verification And Review

**Files:**
- No intended writes.

**Implementer scope:**
- Run final verification.
- Dispatch final reviewer after verification.
- Do not commit.

- [ ] **Step 1: Run regression tests without writing pycache**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
```

Expected:

```text
all tests pass
```

- [ ] **Step 2: Compile without writing repo pycache**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
PYTHONPYCACHEPREFIX=/tmp/mmkgc-joint-smoke-compileall-pycache /home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: Confirm generated outputs and weights are ignored**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
git check-ignore -v Qwen/Qwen2-VL-7B-Instruct SDXL/stable-diffusion-xl-base-1.0 CLIP/ViT-B-32.pt outputs/qwen_sdxl_clip_joint_smoke_20260709
git status --short
```

Expected:

```text
Qwen, SDXL, CLIP, and outputs paths are ignored or otherwise not listed as untracked model/output files.
git status may still show user stage-doc renames and current task docs.
No pycache files are modified.
```

- [ ] **Step 4: Dispatch final reviewer**

Final reviewer checklist:

- Spec requirements in `docs/superpowers/specs/6 mkgw-qwen-sdxl-clip-joint-smoke-design.zh.md` are satisfied.
- Qwen health check passed.
- Joint run produced 5 entities and 0 failed entities.
- Qwen generated 25 text candidates with 0 fallback.
- SDXL generated 25 PNG files with 0 fallback.
- CLIP evaluated 25 images.
- offline evaluation completed with 0 failed entities.
- Report contains actual values and no placeholders.
- No code was changed unless explicitly approved.
- No generated outputs, model weights, local configs, or pycache files are staged/untracked for commit.
- No commit was made.

- [ ] **Step 5: Final response to user**

Final response must include:

- Whether the joint smoke passed or failed.
- Commands run.
- Qwen, SDXL, offline evaluation, and CLIP counts.
- Report path.
- Any remaining limitation.
- Confirmation that no commit was made.
