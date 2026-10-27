# MKG-W BERTScore + Gemini Embedding 50 实体评分测试设计文档

## 目标

本阶段目标是在现有 MKG-W 真实模型链路上新增两个并列评分信号，并先用 50 个实体做 smoke 测试：

- BERTScore：用于 text candidate 的语义质量评价。
- Gemini Embedding / Gemini 多模态 embedding：作为 text、image、pair 的新增语义对齐评价信号。

本阶段不替换现有 CLIP、BLIP/ITM、Qwen-VL Judge、Stage 5 ranking 或 calibrated offline evaluation。Gemini 和 BERTScore 先作为独立 evaluator 输出，与现有指标并排比较；只有 50 实体 smoke 和后续人工/自动校准结果可信后，才考虑进入最终融合权重。

## 已确认方案

用户已确认：

- Gemini 不替换任何已有模型。
- BERTScore 用于候选文本评分。
- Gemini Embedding 与现有 CLIP、BLIP/ITM、Qwen-VL Judge 一起作为并列信号。
- 第一版在 50 个实体上测试。
- 不自动 commit。

## 非目标

本阶段不做以下事情：

- 不删除、替换或降级 CLIP、BLIP/ITM、Qwen-VL Judge。
- 不把 Gemini 或 BERTScore 直接写入生产 accept/reject 决策。
- 不直接全量运行 MKG-W。
- 不修改 Qwen text generation prompt。
- 不修改 SDXL image generation prompt。
- 不接 SDXL Refiner、SD3.5、Flux 或其它图像生成模型。
- 不提交 API key、模型权重、datasets、outputs 或本机配置。
- 不自动 commit。

## 输入与规模

第一版测试规模固定为：

```text
missing_mode = both
limit = 50
text_generator = qwen-local
image_generator = sdxl-local
image candidates per entity = 5
expected generated PNG count = 250
```

如果已有合适的 50 实体 Qwen + SDXL run，可以复用；否则新建 run：

```text
outputs/qwen_sdxl_gemini_bertscore_50_<date>
```

评价阶段读取该 run 的六阶段输出、真实 PNG、text candidates、image candidates、pair ranking，并写入同一 run 的 `evaluation/` 目录。

## 模型与 provider

### BERTScore

新增 evaluator：

```text
--text-quality-evaluator bertscore-local
```

第一版使用本地 Python `bertscore` 包。默认模型建议：

```text
roberta-large
```

如果本地环境没有模型缓存，实施时应 fail-fast 提示缺少依赖或模型；不在测试流程里静默联网下载。

输出字段：

```text
bertscore_precision
bertscore_recall
bertscore_f1
bertscore_reference_kind
bertscore_overall_text_score
```

参考文本优先级：

1. MKG-W ground-truth description。
2. 如果 ground truth 为空，使用 KG evidence summary。
3. 如果 KG evidence summary 为空，记录该候选不可评分，不参与 BERTScore 平均。

### Gemini Embedding

新增 evaluator：

```text
--gemini-embedding-evaluator gemini-api
```

第一版模型：

```text
gemini-embedding-2
```

理由：Google 官方 Gemini API 文档说明 `gemini-embedding-2` 支持 text、image、video、audio、PDF 等输入，并映射到统一 embedding 空间，适合本项目的 text-image-KG 跨模态比较；`gemini-embedding-001` 是 text-only，且与 `gemini-embedding-2` embedding 空间不兼容。

配置来源：

```text
configs/gemini-embedding.yaml
```

该文件 gitignored，只保存本机 API key 或环境变量名。仓库内只提交脱敏模板：

```text
configs/gemini-embedding.example.yaml
```

建议字段：

```yaml
gemini:
  api_key_env: GEMINI_API_KEY
  base_url: https://generativelanguage.googleapis.com/v1beta
  model_name: gemini-embedding-2
  output_dimensionality: 768
  timeout_seconds: 120
  max_retries: 2
```

## Gemini 评分位置

### Text 侧

对每个 text candidate 计算：

```text
gemini_text_ground_truth_similarity
gemini_text_evidence_similarity
gemini_overall_text_embedding_score
```

第一版不替换 `ground_truth_similarity_score` 或 BERTScore，只并排输出。

### Image 侧

对每张生成 PNG 计算：

```text
gemini_prompt_similarity
gemini_text_similarity
gemini_evidence_similarity
gemini_overall_multimodal_score
```

其中：

- image embedding 来自 PNG。
- prompt embedding 来自 image prompt。
- text embedding 来自 paired/generated text candidate。
- evidence embedding 来自 entity KG evidence summary。

第一版与 CLIP、BLIP/ITM、Qwen-VL Judge 并排输出，不替换任何已有图像评价结果。

### Pair 侧

对 text-image pair 计算：

```text
gemini_pair_image_text_similarity
gemini_pair_image_evidence_similarity
gemini_pair_text_evidence_similarity
gemini_overall_pair_embedding_score
```

pair 分数只用于 50 实体分析报告；不写回 Stage 5 ranking，也不改变现有 best pair。

## 输出

新增输出位于：

```text
<run_dir>/evaluation/
  bertscore_text_quality_summary.json
  bertscore_text_quality_entity_scores.csv
  gemini_embedding_summary.json
  gemini_embedding_entity_scores.csv
  gemini_embedding_image_scores.csv
  gemini_embedding_pair_scores.csv
  model_signal_comparison_summary.json
```

每个实体目录新增：

```text
<run_dir>/evaluation/entities/<entity_key>/
  bertscore_text_quality.json
  gemini_embedding.json
```

如果同时启用 CLIP、BLIP/ITM、Qwen-VL Judge，则 `model_signal_comparison_summary.json` 应记录：

```text
clip_overall_image_quality_score
blip_overall_itm_score
qwen_vl_overall_judge_score
gemini_overall_multimodal_score
bertscore_overall_text_score
```

## 50 实体成功标准

工程成功标准：

- `mmkgc.run` 处理 50 个实体，`failed_entities = 0`。
- 预期生成 250 张 PNG。
- offline evaluation 完成 50 个实体。
- BERTScore 至少成功评价所有存在可用 reference 的 text candidates。
- Gemini text embedding 至少成功评价所有存在可用 text/evidence target 的 text candidates。
- Gemini image embedding 成功评价 250 张 PNG。
- 输出 summary、CSV、entity JSON 均存在且行数一致。
- 不泄露 Gemini API key。

质量分析成功标准：

- 报告 BERTScore 与现有 text offline score 的相关性。
- 报告 Gemini image score 与 CLIP、BLIP、Qwen-VL 的均值、分布和 disagreement 样本。
- 标出 Gemini 与现有视觉模型严重分歧的 best/worst case。
- 明确说明 Gemini/BERTScore 是否已经适合进入后续融合权重；第一版默认结论应是“仅作为候选校准信号”。

## 失败策略

- BERTScore 依赖缺失：fail-fast，提示安装依赖或准备本地模型缓存。
- Gemini API key 缺失：fail-fast，提示设置 `GEMINI_API_KEY` 或配置文件。
- Gemini 单个请求失败：按 `max_retries` 重试；仍失败则记录该 item failed，并使 50 实体验证不通过。
- PNG 缺失或不可读：fail-fast。
- API key、Authorization header、完整 raw response 中的敏感字段不得写入输出。

## 后续扩展

50 实体 smoke 通过后，再考虑：

- 在 100/500 实体 run 上复验 Gemini 与 BERTScore。
- 将 BERTScore 加入 text candidate 的实验性融合分。
- 将 Gemini multimodal 加入 Stage 10 visual risk gate 的校准扫描。
- 基于人工复核结果重新估计 CLIP、BLIP、Qwen-VL、Gemini 的融合权重。

## 参考

- Google Gemini API Embeddings 文档：`gemini-embedding-2` 支持多模态统一 embedding；`gemini-embedding-001` 为 text-only，且两者 embedding 空间不兼容。
