# Qwen-VL Judge + BLIP/ITM 图像评价 Smoke 验收报告

更新时间：2026-07-11

## 输入 Run

本阶段在 Stage 7 已生成的联合真实模型 run 上追加图像评价模型，不重新生成 text 或 image。

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709
```

输入规模：

```text
entities = 5
generated PNG images = 25
existing CLIP image quality summary = present
```

## 运行命令

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_sdxl_clip_joint_smoke_20260709 \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda:1 \
  --blip-itm-evaluator blip-itm-local \
  --blip-model-dir blip-itm-large-coco \
  --blip-device cuda:1 \
  --vlm-judge-evaluator qwen-vl-judge \
  --vlm-judge-model-config configs/mkgw-knowledge-extraction.yaml \
  --vlm-judge-timeout-seconds 120 \
  --overwrite
```

运行前曾在普通 sandbox 中遇到 `localhost:12345` 权限限制；最终在 sandbox 外确认 Qwen/vLLM 服务可达后执行成功。

## CLI 输出

```json
{
  "evaluation_dir": "outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation",
  "evaluated_entities": 5,
  "failed_entities": 0,
  "failed_entity_keys": [],
  "image_quality_evaluator": "clip-local",
  "clip_evaluated_entities": 5,
  "clip_evaluated_images": 25,
  "blip_itm_evaluator": "blip-itm-local",
  "blip_evaluated_entities": 5,
  "blip_evaluated_images": 25,
  "vlm_judge_evaluator": "qwen-vl-judge",
  "vlm_judged_entities": 5,
  "vlm_judged_images": 25,
  "image_quality_comparison_written": true,
  "image_quality_comparison_evaluated_images": 25
}
```

## Qwen-VL Judge 结果

输出文件：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/qwen_vl_judge_summary.json
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/qwen_vl_judge_entity_scores.csv
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/entities/*/qwen_vl_judge.json
```

真实统计：

```text
evaluated_entities = 5
evaluated_images = 25
failed_images = 0
fallback_images = 0
parse_failed_images = 0
```

真实均值：

```text
entity_match_score = 43.2
prompt_alignment_score = 43.2
kg_evidence_score = 43.2
artifact_score = 43.6
overall_judge_score = 43.2
```

Qwen-VL Judge 最高实体为 `Linda Darnell`，实体均值 `overall_judge_score = 100.0`；最低实体为 `My Darling Clementine`，实体均值 `overall_judge_score = 0.0`。这些分数来自 VLM judge 的开放式判断，不等于人工金标准。

## BLIP/ITM 结果

输出文件：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/blip_itm_summary.json
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/blip_itm_entity_scores.csv
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/entities/*/blip_itm.json
```

真实统计：

```text
evaluated_entities = 5
evaluated_images = 25
failed_images = 0
fallback_images = 0
```

真实均值：

```text
blip_prompt_itm_score = 20.946176
blip_text_itm_score = 37.375633
blip_evidence_itm_score = 12.165737
blip_overall_itm_score = 23.495848
```

BLIP/ITM 最高实体为 `Linda Darnell`，实体均值 `blip_overall_itm_score = 46.972073`；最低实体为 `Ernesto Gastaldi`，实体均值 `blip_overall_itm_score = 4.831722`。

## CLIP / BLIP / Qwen-VL 对比

输出文件：

```text
outputs/qwen_sdxl_clip_joint_smoke_20260709/evaluation/image_quality_model_comparison_summary.json
```

真实对比：

```text
evaluated_images = 25

clip_overall_image_quality_score = 26.545744
blip_overall_itm_score = 23.495848
overall_judge_score = 43.2
```

三者衡量的是不同信号：

- CLIP：图像与 prompt / generated text / KG evidence 的跨模态相似度。
- BLIP/ITM：图像与 prompt / generated text / KG evidence 的图文匹配概率信号。
- Qwen-VL Judge：开放式视觉判断，按 entity match、prompt alignment、KG evidence、artifact 等维度输出解释性分数。

因此三者不能直接当作同一种指标比较，也不替代现有 calibrated offline evaluation。

## 严格验收

严格验收命令确认：

```text
qwen_vl.evaluated_entities = 5
qwen_vl.evaluated_images = 25
qwen_vl.failed_images = 0
qwen_vl.fallback_images = 0
qwen_vl.parse_failed_images = 0

blip_itm.evaluated_entities = 5
blip_itm.evaluated_images = 25
blip_itm.failed_images = 0
blip_itm.fallback_images = 0

comparison.evaluated_images = 25
qwen_vl_judge_entity_scores.csv rows = 5
blip_itm_entity_scores.csv rows = 5
```

验收脚本输出：

```json
{
  "qwen_vl": {
    "entities": 5,
    "images": 25,
    "overall": 43.2
  },
  "blip_itm": {
    "entities": 5,
    "images": 25,
    "overall": 23.495848
  },
  "comparison_images": 25
}
```

同时检查 `qwen_vl_judge_summary.json` 和 `image_quality_model_comparison_summary.json` 未包含本地 API token 明文。

## 结论

Stage 8 严格通过：

```text
Qwen-VL Judge = 25/25 success
BLIP/ITM = 25/25 success
fallback = 0
parse failure = 0
```

本阶段已经在 Stage 7 的真实 PNG 上补充了 CLIP 之外的两类图像评价信号，为下一步扩大到 100 或 500 实体的联合真实模型实验提供了更强的图像验收依据。

## 限制

- Qwen-VL Judge 与 Qwen text generation 使用同类 Qwen2-VL 模型，可能存在模型自我偏向。
- Qwen-VL Judge 仍不是人工金标准，后续需要人工抽样复核 best/worst cases。
- BLIP/ITM 和 CLIP 都是图文匹配信号，不等于 KG 关系在图像中真实可见。
- 当前结果仍是 5 实体、25 图 smoke，不代表 500 实体规模的真实模型质量分布。
