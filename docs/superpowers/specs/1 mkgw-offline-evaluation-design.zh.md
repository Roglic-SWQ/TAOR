# MKG-W 离线实验评估器设计文档

## 目标

在 `MMKGC/` 下新增一个独立的离线评估器，用于评估已有 MMKGC run 输出，不改动现有六阶段生成 pipeline。

第一版评估器目标是建立稳定评价闭环：

- 读取已有 run 目录，例如 `MMKGC/outputs/acceptance_pytest_20260706_0002`。
- 覆盖 text candidates、image prompt candidates、text-image pair ranking 三层结果。
- 完全离线运行，不新增模型依赖，不联网。
- 输出机器可读 JSON 和人工分析友好的 CSV。
- 默认把评估结果写入 `<run_dir>/evaluation/`。
- 支持 `--output-dir` 把评估结果写入指定目录。
- 支持任意实体数量的 run，本阶段验收使用已有 100 条 run。
- 导出轻量误差分析样本，包括高分、低分、fallback 样本。

## 非目标

第一版不做以下事情：

- 不接入 Sentence-BERT、CLIP、BLIP、LLM/VLM judge 或其它模型评价器。
- 不生成真实图片。
- 不修改 `mmkgc.run` 主流程。
- 不新增 `--evaluate` 到主 pipeline。
- 不做跨 run 对比 UI。
- 不做复杂人工标注工作流。
- 不自动 commit。

## 用户确认方案

brainstorming 阶段确认如下：

- 下一阶段先做“实验评估版”。
- 评估入口采用独立评估器。
- 评估对象覆盖 text、image prompt、pair 三层。
- 第一版指标完全离线、无新增模型依赖。
- 输出 JSON + CSV。
- 默认输出到原 run 目录，也允许 `--output-dir` 指定其它位置。
- 代码支持任意 run，本阶段验收用已有 100 条 run。
- 误差分析采用轻量版，只导出 top/bottom/fallback 样本。

## CLI 设计

新增命令：

```bash
python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002
```

默认输出：

```text
outputs/<run_name>/evaluation/
```

支持显式输出目录：

```bash
python -m mmkgc.evaluate \
  --run-dir outputs/acceptance_pytest_20260706_0002 \
  --output-dir outputs/evaluations/acceptance_eval
```

建议参数：

- `--run-dir PATH`：必填，指向已有 run 目录。
- `--output-dir PATH`：可选，默认是 `<run_dir>/evaluation`。
- `--case-limit N`：可选，默认 `20`，控制 best/worst/fallback 样本导出数量。
- `--overwrite`：可选，允许覆盖已存在的 evaluation 目录。默认不覆盖。

CLI 行为：

- `run_dir` 不存在时失败并返回非 0。
- `run_dir` 缺少 `entities/` 或 `results.jsonl` 时失败并返回非 0。
- 默认输出目录已存在时失败，避免误覆盖。
- 指定 `--overwrite` 时允许覆盖评估目录中的评估器产物。

## 输入文件

评估器读取已有 run 目录：

```text
outputs/<run_name>/
  config.json
  results.jsonl
  metrics_summary.json
  entities/
    Q.../
      0_coarse_relation_filtering.json
      1_knowledge_modeling.json
      2_graph_constrained_reasoning.json
      3_fine_grained_relation_selection.json
      4_relation_alternation_generation.json
      5_kg_aware_pair_ranking.json
      final.json
```

第一版必需读取：

- `entities/<entity>/1_knowledge_modeling.json`
- `entities/<entity>/2_graph_constrained_reasoning.json`
- `entities/<entity>/5_kg_aware_pair_ranking.json`
- `entities/<entity>/final.json`

其中：

- ground truth text 从 `final.json.entity.description` 读取。
- entity label 从 `final.json.entity.label` 读取。
- entity key 从 `final.json.entity.entity_key` 读取。
- text candidates 从 `5_kg_aware_pair_ranking.json.text_ranking.ranked_candidates` 读取。
- image prompt candidates 从 `5_kg_aware_pair_ranking.json.image_ranking.ranked_candidates` 读取。
- pair candidates 从 `5_kg_aware_pair_ranking.json.pair_ranking.ranked_pairs` 读取。
- fallback 信息从候选的 `metadata.fallback`、`primary_relation == "global_profile"`、`relation_role == "fallback"` 等字段识别。

## 输出结构

默认输出目录：

```text
outputs/<run_name>/evaluation/
  evaluation_summary.json
  summary.csv
  entity_scores.csv
  best_cases.jsonl
  worst_cases.jsonl
  fallback_cases.jsonl
  entities/
    Q.../
      evaluation.json
```

### `evaluation_summary.json`

run 级别完整摘要，包含：

- `run_dir`
- `evaluated_entities`
- `failed_entities`
- `averages`
- `counts`
- `fallback_counts`
- `case_exports`
- `output_files`

### `summary.csv`

run 级别扁平指标，便于放入表格或实验记录。

建议字段：

```text
metric,value
evaluated_entities,100
failed_entities,0
avg_text_best_score,...
avg_image_best_score,...
avg_pair_best_score,...
fallback_entities,...
```

### `entity_scores.csv`

每个实体一行，便于排序和人工筛选。

建议字段：

```text
entity_key,label,missing_mode,text_best_score,image_best_score,pair_best_score,
text_candidate_count,image_candidate_count,pair_candidate_count,
best_text_candidate_id,best_image_candidate_id,best_pair_id,
fallback_candidate_count,has_ground_truth_text
```

### `entities/<entity>/evaluation.json`

每个实体完整评估结果，包含：

- entity metadata
- text evaluation
- image prompt evaluation
- pair evaluation
- fallback summary
- case tags
- warnings

### case exports

轻量误差分析导出：

- `best_cases.jsonl`：按综合分最高导出前 `case_limit` 个实体。
- `worst_cases.jsonl`：按综合分最低导出前 `case_limit` 个实体。
- `fallback_cases.jsonl`：导出包含 fallback candidate 的前 `case_limit` 个实体。

每行至少包含：

- `entity_key`
- `label`
- `overall_score`
- `text_best_score`
- `image_best_score`
- `pair_best_score`
- `best_text_candidate_id`
- `best_image_candidate_id`
- `best_pair_id`
- `case_tags`

## 指标设计

### Text candidate evaluation

目标：评估生成文本与 ground truth description 的词面相似度和候选排序质量。

候选文本读取规则：

- 优先读取 `candidate.generated_content.value`。
- 为空时退回 `candidate.prompt_or_instruction`。

ground truth 读取规则：

- 读取 `final.json.entity.description`。
- 如果 ground truth text 为空，则 text ground-truth 指标记为不可评估，但仍统计 KG coverage 和候选数量。

指标：

- `text_ground_truth_token_cosine`
- `text_ground_truth_token_overlap`
- `text_kg_similarity_score`，来自 stage 5 candidate score。
- `text_must_term_score`，来自 stage 5 candidate score。
- `text_relation_coverage_score`，来自 stage 5 candidate score。
- `text_best_score`
- `text_top5_avg_score`
- `text_candidate_count`

第一版综合 text 分数：

```text
text_score = 0.6 * token_cosine(candidate_text, ground_truth_text)
           + 0.2 * token_overlap(candidate_text, [entity_label, entity_type])
           + 0.2 * stage5_final_score
```

如果 ground truth text 缺失：

```text
text_score = 0.5 * stage5_final_score
           + 0.3 * relation_coverage_score
           + 0.2 * must_term_score
```

### Image prompt evaluation

目标：在不生成真实图片、不接 CLIP/BLIP 的情况下，评估 image prompt 是否覆盖 KG 约束。

候选文本拼接规则：

```text
positive_prompt + " " + prompt_or_instruction + " " + generated_content.value
```

指标：

- `image_entity_type_coverage`
- `image_relation_coverage`
- `image_constraint_coverage`
- `image_forbidden_awareness`
- `image_stage5_score`
- `image_best_score`
- `image_top5_avg_score`
- `image_candidate_count`

第一版综合 image prompt 分数：

```text
image_score = 0.35 * relation_coverage
            + 0.25 * constraint_coverage
            + 0.20 * entity_type_coverage
            + 0.20 * stage5_final_score
```

`negative_prompt` 覆盖 forbidden content 时记录 `image_forbidden_awareness`，但不把它作为强惩罚指标；它主要用于人工分析。

### Pair evaluation

目标：评估 text-image pair 的一致性和排序摘要。

读取：

- `pair_ranking.ranked_pairs`
- `pair_ranking.top_5_pairs`
- `pair_ranking.best_pair`

指标：

- `pair_best_score`
- `pair_top5_avg_score`
- `pair_candidate_count`
- `pair_text_score`
- `pair_image_score`
- `pair_similarity_score`
- `pair_kg_anchor_score`
- `pair_balance_score`

`pair_balance_score` 定义：

```text
pair_balance_score = 1 - abs(text_score - image_score)
```

第一版综合 pair 分数：

```text
pair_score = 0.4 * stage5_pair_score
           + 0.3 * pair_balance_score
           + 0.2 * pair_similarity_score
           + 0.1 * kg_anchor_score
```

### Overall score

实体综合分数：

```text
overall_score = average(non_null(text_best_score, image_best_score, pair_best_score))
```

如果某个模态没有候选，则该模态不参与平均，并在 `warnings` 中记录。

## 错误处理

评估器应尽量做到 run 级别稳定：

- run 目录级别缺失必要结构时直接失败。
- 单个实体缺失某个 stage 文件时，该实体记为 failed entity，但其它实体继续评估。
- 单个实体候选为空时，不抛异常，输出 0 分和 warning。
- JSON 解析失败时记录该实体失败原因。
- 输出目录默认不覆盖，除非传入 `--overwrite`。

## 测试策略

新增测试应覆盖：

- run_dir 验证失败：不存在、不是目录、缺少 `entities`、缺少 `results.jsonl`。
- 从 fixture run 中读取 entity stage JSON。
- text evaluation 在有 ground truth 时计算分数。
- text evaluation 在 ground truth 缺失时使用 fallback 公式。
- image prompt evaluation 计算 relation/constraint/entity_type coverage。
- pair evaluation 计算 pair balance 和 top5 summary。
- writer 写出 summary JSON、CSV、per-entity JSON、case exports。
- CLI 对已有 output_dir 默认拒绝覆盖。
- CLI 使用 `--overwrite` 后可重新写评估产物。
- 使用当前 100 条 acceptance run 可完成评估。

## 成功标准

本阶段完成时必须满足：

- `python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002` 成功运行。
- 默认输出目录为 `outputs/acceptance_pytest_20260706_0002/evaluation/`。
- 输出 `evaluation_summary.json`、`summary.csv`、`entity_scores.csv`。
- 每个成功评估实体输出 `entities/<entity>/evaluation.json`。
- 输出 `best_cases.jsonl`、`worst_cases.jsonl`、`fallback_cases.jsonl`。
- `evaluation_summary.json.evaluated_entities` 对 100 条 acceptance run 应为 100。
- `pytest` 全量通过。
- 不修改 `KGC/` 数据或代码。
- 不自动 commit。

## 后续扩展

本阶段结束后，可以在后续阶段扩展：

- cross-run comparison。
- Sentence-BERT text semantic metric。
- CLIP/BLIP image-text consistency。
- LLM/VLM judge。
- pipeline `--evaluate` 可选集成。
- 大规模 run 的评估缓存和增量评估。
