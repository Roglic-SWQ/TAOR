# MKG-W SDXL 图像生成烟测验收报告

## 结论

`outputs/sdxl_image_smoke_20260708` 已完成本地 SDXL Base 1.0 图像生成工程烟测。

本次 run 处理 5 个 MKG-W 实体，每个实体生成 5 张图片，共 25 张 PNG。图片工程验收通过，现有 offline evaluation 也能继续消费该 run。

本报告只证明真实图像生成链路、落盘产物和既有 evaluation 链路可运行；不证明生成图片与实体语义高度一致，也不替代后续 CLIP、BLIP 或 VLM judge 评价。

## 运行配置

- 输出目录：`outputs/sdxl_image_smoke_20260708`
- 数据集目录：`datasets/MKG-W`
- missing mode：`both`
- 实体数量：`5`
- candidate budget：`5`
- text generator：`rule`
- image generator：`sdxl-local`
- SDXL 模型路径：`SDXL/stable-diffusion-xl-base-1.0`
- 图片尺寸：`1024x1024`
- steps：`20`
- seed：`20260708`
- guidance scale：`7.5`
- device：`cuda`

`outputs/sdxl_image_smoke_20260708/config.json` 记录了上述参数，且 `api_key_configured=false`。

## 运行命令

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator rule \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260708 \
  --output-root outputs \
  --run-name sdxl_image_smoke_20260708
```

命令输出：

```json
{
  "run_dir": "outputs/sdxl_image_smoke_20260708",
  "processed_entities": 5,
  "failed_entities": 0
}
```

## 产物复核

`outputs/sdxl_image_smoke_20260708/metrics_summary.json`：

```json
{
  "processed_entities": 5,
  "failed_entities": 0,
  "avg_text_candidates": 5.0,
  "avg_image_candidates": 5.0,
  "avg_pair_candidates": 9.0,
  "fallback_counts": {},
  "stage_failure_counts": {}
}
```

实体目录：

```text
outputs/sdxl_image_smoke_20260708/entities/Q1356749/
outputs/sdxl_image_smoke_20260708/entities/Q1373204/
outputs/sdxl_image_smoke_20260708/entities/Q234900/
outputs/sdxl_image_smoke_20260708/entities/Q370252/
outputs/sdxl_image_smoke_20260708/entities/Q3857815/
```

每个实体目录包含：

- `generated_images/image_001.png` 到 `generated_images/image_005.png`
- `generated_images/image_001.json` 到 `generated_images/image_005.json`
- `image_model_calls.jsonl`
- 通用六阶段 JSON 产物与 `final.json`

复核计数：

- PNG 文件数：`25`
- `image_model_calls.jsonl` 文件数：`5`
- image model call 总行数：`25`

## 图片工程验收

验收命令：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python scripts/validate_sdxl_image_run.py \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --expected-entities 5 \
  --expected-images-per-entity 5 \
  --expected-width 1024 \
  --expected-height 1024
```

验收输出：

```json
{
  "ok": true,
  "entities_checked": 5,
  "images_checked": 25
}
```

该验收检查了 run 目录、`metrics_summary.json`、实体目录数量、每实体 PNG 数量、PNG 可打开性、图片尺寸、同名 metadata sidecar 和 `image_model_calls.jsonl`。

## Offline Evaluation

evaluation 命令：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/sdxl_image_smoke_20260708 \
  --output-dir outputs/sdxl_image_smoke_20260708/evaluation \
  --overwrite
```

命令输出：

```json
{
  "evaluation_dir": "outputs/sdxl_image_smoke_20260708/evaluation",
  "evaluated_entities": 5,
  "failed_entities": 0,
  "failed_entity_keys": []
}
```

`outputs/sdxl_image_smoke_20260708/evaluation/evaluation_summary.json` 中：

- `evaluated_entities`: `5`
- `failed_entities`: `0`
- `overall_offline_evaluation_score`: `0.75719`
- `calibrated_overall_offline_evaluation_score`: `0.561852`

这些分数仍来自当前离线规则评价与校准逻辑。它们证明 evaluation 链路能继续运行，但不代表真实图像视觉质量或图文语义一致性已经被模型化评价。

## 限制

- 本阶段只使用 SDXL Base 1.0，没有接入 Refiner。
- 本阶段使用 rule text generator，没有同时运行 Qwen。
- 本阶段没有接入 CLIP、BLIP、VLM judge 或人工多 agent 图像质量评价。
- 当前 image offline score 仍主要评价 image prompt 文本和 Stage 5 ranking 信号，不直接理解 PNG 内容。
- 本结果是 5 实体、25 图片的小样本工程烟测，不代表 500 实体或全量 MKG-W 的稳定性与质量。
