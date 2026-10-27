# Stage 0-3 Qwen semantic 500 实体验收

日期：2026-07-23

本次验收把 Stage 0-3 Qwen LMM 语义主链路扩大到 500 实体，比较 deterministic rule baseline 与 `--stage02-planner qwen-local --stage3-planner qwen-local` 的差异。为隔离前段质量，本次两组均使用 rule 文本/图像候选和 `--stage5-ranker legacy`，不引入 SDXL 或 Stage 5 model-fusion 变量。

## 运行口径

Rule baseline：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 500 \
  --candidate-budget 5 \
  --stage02-planner rule \
  --stage3-planner rule \
  --image-generator rule \
  --stage5-ranker legacy \
  --output-root outputs \
  --run-name stage03_rule_500_20260723
```

Qwen semantic 由于本地 vLLM 服务在长跑中多次中断，采用 shard/repair 方式补齐；最终只汇总满足严格口径的实体：

- 每个实体必须有 4 条 `qwen-local-stage03-semantic-planner` 调用。
- 覆盖 `stage0_coarse`、`stage1_knowledge`、`stage2_reasoning`、`stage3_selection`。
- 4 条调用状态全部为 `applied`。

clean 汇总目录：

```text
outputs/stage03_qwen_500_20260723_combined_real_applied
```

对照命令：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/compare_stage02_semantic.py \
  --baseline-run-dir outputs/stage03_rule_500_20260723 \
  --semantic-run-dir outputs/stage03_qwen_500_20260723_combined_real_applied \
  --output docs/evaluation/2026-07-23-stage03-qwen-500-comparison.json \
  --csv-output docs/evaluation/2026-07-23-stage03-qwen-500-comparison.csv \
  --overwrite
```

## 结果

- Rule baseline：`processed_entities = 500`，`failed_entities = 0`。
- Qwen semantic clean combined：`processed_entities = 500`，`failed_entities = 0`。
- Stage 0/1/2/3 semantic calls：`2000/2000 applied`。
- 严格缺失实体：`0`。
- Prompt 泄漏边界：semantic prompt 不读取 ground truth modalities、缺失 description、人工 reviewer 分数或 evaluation 结果。

对照汇总：

```json
{
  "avg_best_pair_score_delta": -0.002588,
  "positive_best_pair_score_delta": 89,
  "negative_best_pair_score_delta": 131,
  "zero_best_pair_score_delta": 280,
  "best_pair_changed": 31,
  "best_text_changed": 46,
  "best_image_changed": 11,
  "stage0_relation_set_changed": 61,
  "stage0_relation_order_changed": 61,
  "stage0_top_relation_changed": 55,
  "stage1_entity_type_changed": 22,
  "stage1_relation_facts_changed": 80,
  "stage2_supported_facts_changed": 102,
  "stage4_text_primary_changed": 87,
  "stage4_image_primary_changed": 80
}
```

## 典型变化

正向 delta 最大的样本：

- `Q3806` Tübingen：`+0.101238`
- `Q7903322` Uttar Dakshin：`+0.088606`
- `Q2466764` Laser Beam/Kasuka na Kaori：`+0.088096`
- `Q48` Asia：`+0.081072`
- `Q185051` Christopher Walken：`+0.070979`

负向 delta 最大的样本：

- `Q137535` county of Iran：`-0.082444`
- `Q40831` comedy：`-0.074186`
- `Q142292` Traffic：`-0.072482`
- `Q1867` Taipei：`-0.068470`
- `Q171736` Robert Duvall：`-0.064404`

## 过程中修复

500 shard 补跑中，`Q6096` Snoop Dogg 稳定触发 Stage 1 Qwen 输出退化：模型返回 `allowlisted_semantic_fields` 并重复 `uncertain_aspects`，导致 `SemanticStageParseError` 和 deterministic fallback。已将 Stage 1 prompt 改为更具体的业务字段约束，明确禁止 `allowlisted_semantic_fields`、meta 字段和 schema placeholder。修复后 `Q6096` 的 Stage 0/1/2/3 均为 `applied`。

## 结论

Stage 0-3 Qwen LMM 主链路的工程验收已通过 500 实体规模：严格口径下 `2000/2000` semantic calls applied，Stage 3 先前 100 验收中的 context/fallback 问题已经补齐到 500 clean run。

质量侧信号偏混合：Qwen 语义链路确实改变了前段关系与事实建模，`stage0_relation_set_changed = 61`、`stage2_supported_facts_changed = 102`，说明不是空跑；但在 legacy Stage 5 分数下，平均 best-pair delta 为 `-0.002588`，正向 89、负向 131、零变化 280。这个结果更适合作为“前段真实 LMM 接入和稳定性验收通过”，不能单独证明最终生成质量显著提升。

后续质量判断仍应落到 Stage 5 model-fusion/Qwen-VL gate 校准和人工复核闭环上；Stage 0-3 当前主要剩余边界是受 deterministic evidence boundary 约束，不能引入图谱候选之外的新关系或更深层全图 path search。
