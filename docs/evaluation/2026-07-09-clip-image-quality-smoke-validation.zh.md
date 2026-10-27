# CLIP 真实图像质量评价 Smoke 验收报告

日期：2026-07-09

## 范围

本报告验收本地 OpenAI CLIP ViT-B-32 图像质量评价链路，目标 run 为 `outputs/sdxl_image_smoke_20260708`。该 run 已由 SDXL Base 1.0 生成真实 PNG，本次评价读取这些图片，并计算图片与 prompt、实体文本、KG evidence 的跨模态相似度。

本次只验收 CLIP similarity 评价链路，不接 BLIP、VLM judge 或人工视觉验收。CLIP 分数是额外的 run 后图像质量产物，不替换 `calibrated_image_best_offline_evaluation_score` 或既有 offline evaluation。

## 验收命令

工作目录：

```text
/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
```

先按计划尝试 CUDA 命令，但当前 Codex 进程不可见 CUDA，失败信息为：

```text
RuntimeError: No CUDA GPUs are available
```

随后改用 CPU 成功完成 smoke：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cpu \
  --overwrite
```

CPU 与 CUDA 使用同一套评价逻辑，只改变运行设备和速度。

## 结果

命令 stdout：

```json
{"evaluation_dir": "outputs/sdxl_image_smoke_20260708/evaluation", "evaluated_entities": 5, "failed_entities": 0, "failed_entity_keys": [], "image_quality_evaluator": "clip-local", "clip_evaluated_entities": 5, "clip_evaluated_images": 25}
```

`clip_image_quality_summary.json` 摘录：

```json
{
  "evaluated_entities": 5,
  "evaluated_images": 25,
  "averages": {
    "clip_prompt_similarity": 26.945619,
    "clip_text_similarity": 28.143429,
    "clip_evidence_similarity": 25.113275,
    "clip_overall_image_quality_score": 26.734108
  },
  "clip_model_name": "ViT-B/32",
  "clip_weight_path": "/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC/CLIP/ViT-B-32.pt"
}
```

CSV 行数复核：

```text
6 outputs/sdxl_image_smoke_20260708/evaluation/clip_image_quality_entity_scores.csv
```

其中 1 行 header，5 行实体分数。

本次 smoke 的关键结论：

- `evaluated_entities = 5`
- `failed_entities = 0`
- `clip_evaluated_entities = 5`
- `clip_evaluated_images = 25`
- `clip_overall_image_quality_score = 26.734108`
- CLIP 权重路径为 `/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC/CLIP/ViT-B-32.pt`

输出文件：

- `outputs/sdxl_image_smoke_20260708/evaluation/clip_image_quality_summary.json`
- `outputs/sdxl_image_smoke_20260708/evaluation/clip_image_quality_entity_scores.csv`
- 真实实体例子：`outputs/sdxl_image_smoke_20260708/evaluation/entities/Q1356749__113e5a3d/clip_image_quality.json`

## 限制

- 当前 Codex 进程不可见 CUDA，所以真实 smoke 用 CPU 完成；这不改变评价逻辑。
- CLIP 计算跨模态相似度，不是 VLM judge，也不做开放式视觉推理。
- CLIP 分数不能替代人工视觉验收、BLIP/VLM judge 或既有 offline evaluation。
- 本次只覆盖 5 个实体、25 张图片，不代表 500 实体或全量 MKG-W 的质量分布。
