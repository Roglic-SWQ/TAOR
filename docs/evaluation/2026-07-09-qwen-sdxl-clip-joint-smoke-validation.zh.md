# Qwen + SDXL + CLIP 联合真实模型 Smoke 验收报告

日期：2026-07-09

工作目录：`/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC`

Run 目录：`outputs/qwen_sdxl_clip_joint_smoke_20260709`

## 范围

本次验收验证 Stage 7：在同一次 MKG-W 5 实体 smoke run 中，串联本地 Qwen2-VL-7B 文本候选生成、本地 SDXL Base 1.0 真实 PNG 生成，以及本地 OpenAI CLIP ViT-B-32 真实 PNG 跨模态相似度评价。

本阶段不是 500 实体扩大实验，也不是最终模型质量结论；它只判断联合真实模型链路是否已经具备继续扩大样本的工程基础。

## 资源分工

- GPU 0：用户手动启动的 Qwen/vLLM OpenAI-compatible 服务。
- GPU 1：SDXL 生成和 CLIP 评价。
- Qwen/vLLM 生命周期不由 `mmkgc.run` 管理；controller 只通过健康检查确认服务可用。

## Qwen 服务

Qwen 服务由用户手动部署为本地 vLLM OpenAI-compatible 服务。controller 使用本地配置中的模型 ID、base URL 和 API key 访问服务，不在本报告中记录或编造完整启动命令。

健康检查通过，实际 JSON：

```json
{"base_url":"http://localhost:12345/v1","latency_ms":129,"model_name":"Qwen/Qwen2-VL-7B-Instruct","ok":true,"response_preview":"OK."}
```

说明：Task 1 曾发现本机 gitignored 运行配置里模型 ID 与 vLLM model id 不匹配，并已将本地配置修正为 `Qwen/Qwen2-VL-7B-Instruct`。该配置文件为本机运行配置，不作为本次待提交文档变更记录。

## Joint Run 命令

```bash
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

stdout 摘要：

```json
{"run_dir": "outputs/qwen_sdxl_clip_joint_smoke_20260709", "processed_entities": 5, "failed_entities": 0}
```

## Run Metrics

`metrics_summary.json` 验证结果：

```text
processed_entities = 5
failed_entities = 0
```

## Qwen 调用统计

`model_calls.jsonl` 统计：

```text
model_calls_total = 50
qwen_text_calls = 25
image_calls_in_model_calls = 25
qwen_generated = 25
qwen_fallback_used = 0
metadata.model_status = generated for all Qwen text rows
```

注意：`model_calls.jsonl` 总行数为 50 是预期设计，因为 writer 保留了 text 与 image model calls；这不是 Qwen 调用了 50 次。过滤 `provider = qwen-local-openai-compatible` 后，Qwen text 调用为 25 行，且全部真实生成、未使用 fallback。

## SDXL 产物统计

SDXL artifact validator 结果：

```json
{"ok": true, "entities_checked": 5, "images_checked": 25}
```

`image_model_calls.jsonl` 统计：

```text
rows = 25
generated = 25
fallback_used = 0
width = 1024
height = 1024
```

每个实体 5 张 PNG，共 25 张真实图片，并包含同名 JSON sidecar metadata。

## Offline Evaluation

评价命令：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_sdxl_clip_joint_smoke_20260709 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda:1 \
  --overwrite
```

stdout：

```json
{"evaluation_dir": "outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation", "evaluated_entities": 5, "failed_entities": 0, "failed_entity_keys": [], "image_quality_evaluator": "clip-local", "clip_evaluated_entities": 5, "clip_evaluated_images": 25}
```

`evaluation/evaluation_summary.json`：

```text
evaluated_entities = 5
failed_entities = 0
```

## CLIP Image Quality

`evaluation/clip_image_quality_summary.json`：

```text
evaluated_entities = 5
evaluated_images = 25
clip_prompt_similarity = 26.94383
clip_text_similarity = 27.602151
clip_evidence_similarity = 25.091251
clip_overall_image_quality_score = 26.545744
```

`clip_image_quality_entity_scores.csv` 验证：

```text
wc -l = 6
CSV data rows = 5
image_count = 5 for every entity row
```

CLIP 分数是图片与 prompt、实体文本和 KG evidence 的跨模态相似度信号，不是 BLIP/VLM judge，也不替代人工视觉验收或开放式视觉问答。

## 输出文件

`outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/` 包含：

- `best_cases.jsonl`
- `clip_image_quality_entity_scores.csv`
- `clip_image_quality_summary.json`
- `entity_scores.csv`
- `evaluation_summary.json`
- `fallback_cases.jsonl`
- `summary.csv`
- `worst_cases.jsonl`

## 结论

Stage 7 联合真实模型 smoke 验收通过。

本次 5 实体运行同时满足：

- Qwen health check 通过。
- Joint run `processed_entities = 5`、`failed_entities = 0`。
- Qwen text 调用 25 次全部 `generated`，`fallback_used = 0`。
- SDXL 生成 25 张 `1024x1024` PNG，`fallback_used = 0`。
- Offline evaluation 评价 5 个实体，失败 0。
- CLIP 评价 5 个实体、25 张图片，`clip_overall_image_quality_score = 26.545744`。

因此，“Qwen text generation + SDXL image generation + CLIP image quality evaluation” 已经在同一次真实模型 smoke 中串通，可以作为后续 50、100 或 500 实体真实模型实验的前置验收证据。

## 限制与下一步

- 本次规模为 5 个实体、25 张图片，不代表大规模质量分布。
- 本次未接 BLIP、VLM judge、ImageReward、HPS、aesthetic predictor 或人工视觉验收。
- 本次未接 SDXL Refiner、Stable Diffusion 3.5 或 Flux。
- 下一步建议先补充开放式视觉质量判断，再扩大到 100 或 500 实体联合真实模型运行。
