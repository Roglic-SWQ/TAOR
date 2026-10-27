# Stage 4 Qwen + SDXL 生成质量 smoke 验收（2026-07-22）

本次验收目标是验证当前 Stage 4 结构化 `generation_brief` 补丁后，真实 Qwen 文本候选与 SDXL 图像候选能否稳定生成，并用现有自动评价器给出质量信号。

## 输入 run

```text
outputs/stage4_qwen_sdxl_quality_5_20260722_gpu
```

运行配置：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --stage02-planner qwen-local \
  --stage3-planner qwen-local \
  --text-generator qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --image-generator sdxl-local \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 \
  --sdxl-height 1024 \
  --sdxl-steps 20 \
  --sdxl-seed 20260722 \
  --sdxl-guidance-scale 7.5 \
  --sdxl-device cuda:0 \
  --stage5-ranker legacy \
  --output-root outputs \
  --run-name stage4_qwen_sdxl_quality_5_20260722_gpu
```

运行结果：

```text
processed_entities = 5
failed_entities = 0
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
stage_failure_counts = {}
```

## Stage 4 结构与真实生成覆盖

从 5 个实体 artifact 统计：

```text
text_candidates = 25
image_candidates = 25
pair_candidates = 45
generation_brief_text = 25/25
generation_brief_image = 25/25
image_artifact_paths = 25/25
existing_image_files = 25/25
semantic_model_calls = 20
semantic_fallback_calls = 0
text_model_calls = 25
text_fallback_calls = 0
image_model_calls = 25
image_fallback_calls = 0
```

结论：当前 Stage 4 的结构化 brief 已完整进入 text/image 候选；Qwen 文本生成和 SDXL 图像生成均为真实模型调用，没有候选级 fallback。

## 自动质量评价

图像质量评价输出目录：

```text
outputs/stage4_qwen_sdxl_quality_5_20260722_gpu/evaluation
```

评价命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/stage4_qwen_sdxl_quality_5_20260722_gpu \
  --overwrite \
  --image-quality-evaluator clip-local \
  --clip-model-name ViT-B/32 \
  --clip-model-dir CLIP \
  --clip-device cuda:0 \
  --blip-itm-evaluator blip-itm-local \
  --blip-model-dir blip-itm-large-coco \
  --blip-device cuda:0 \
  --vlm-judge-evaluator qwen-vl-judge \
  --vlm-judge-model-config configs/mkgw-knowledge-extraction.yaml \
  --vlm-judge-timeout-seconds 120
```

评价结果：

```text
evaluated_entities = 5
failed_entities = 0
clip_evaluated_images = 25
blip_evaluated_images = 25
vlm_judged_images = 25
```

Offline evaluation 均值：

```text
text_best_offline_evaluation_score = 0.383545
image_best_offline_evaluation_score = 0.962953
pair_best_offline_evaluation_score = 0.799429
overall_offline_evaluation_score = 0.715309
calibrated_overall_offline_evaluation_score = 0.430354
```

CLIP 图像质量均值：

```text
clip_prompt_similarity = 25.924172
clip_text_similarity = 25.711200
clip_evidence_similarity = 24.564433
clip_overall_image_quality_score = 25.399935
```

BLIP/ITM 图文匹配均值：

```text
blip_prompt_itm_score = 25.854118
blip_text_itm_score = 50.968882
blip_evidence_itm_score = 24.032926
blip_overall_itm_score = 33.618642
```

Qwen-VL Judge 均值：

```text
entity_match_score = 52.0
prompt_alignment_score = 52.0
kg_evidence_score = 52.0
artifact_score = 52.0
overall_judge_score = 52.0
parse_failed_images = 0
```

文本 BERTScore 输出目录：

```text
outputs/stage4_qwen_sdxl_quality_5_20260722_gpu/evaluation_bertscore_stage4_quality_20260722
```

BERTScore 命令使用本地模型目录，避免 HuggingFace 下载：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/stage4_qwen_sdxl_quality_5_20260722_gpu \
  --output-dir outputs/stage4_qwen_sdxl_quality_5_20260722_gpu/evaluation_bertscore_stage4_quality_20260722 \
  --overwrite \
  --text-quality-evaluator bertscore-local \
  --bertscore-model-type models/bertscore/roberta-large \
  --bertscore-device cuda:0
```

BERTScore 结果：

```text
bertscore_evaluated_entities = 5
bertscore_evaluated_text_candidates = 25
bertscore_precision = 0.826518
bertscore_recall = 0.854138
bertscore_f1 = 0.839549
bertscore_overall_text_score = 0.839549
```

## 质量判断

正向结论：

- Stage 4 当前结构补丁在真实 run 中生效，`generation_brief` 覆盖 text/image 全部 50 个候选。
- Qwen text generation、SDXL image generation 均完成真实模型调用，候选级 fallback 为 0。
- 25 张 PNG 均存在，CLIP、BLIP/ITM、Qwen-VL Judge 均成功评价，Qwen-VL parse failure 为 0。
- BERTScore 本地模型评价完成，文本候选语义分数稳定在 0.84 左右。

限制与风险：

- Offline evaluation 仍标出 `fallback_entities = 3`、`fallback_penalized_entities = 3`、`template_prompt_penalized_entities = 5`。这不是 Qwen/SDXL 候选生成 fallback，而是评价侧认为部分实体 relation support 弱或 profile 过泛。
- CLIP overall 约 25.40、BLIP overall 约 33.62，说明图像和 prompt/evidence 的自动相似度仍偏低；尤其 `Francis Eugene Worley`、`The Violent Professionals`、`Ernesto Gastaldi` 等实体的图像语义对齐仍需人工抽样看图。
- Qwen-VL Judge 均值 52，但既往 500 图校准已发现 Qwen-VL 容易极端高估或低估，因此这里只能作为辅助信号。
- 首次 BERTScore 直接用 `roberta-large` 会尝试 HuggingFace 下载并卡住；本报告的 BERTScore 结果使用本地 `models/bertscore/roberta-large`。

## 结论

Stage 4 “结构补上但真实生成质量未验收”的偏离已经完成 5 实体真实模型 smoke 验收。工程链路通过，候选结构和真实模型调用稳定；但图像质量指标仍不支持直接认为质量已经达标。下一步应基于该 run 的 best/worst 图像做人工抽样复核，重点看 generic entity type、relation support fallback 和 template prompt penalized 样本。
