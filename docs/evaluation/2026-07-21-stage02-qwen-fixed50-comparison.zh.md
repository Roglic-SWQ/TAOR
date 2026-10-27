# Stage 0-2 Qwen semantic fixed50 对照报告

日期：2026-07-21（2026-07-22 以 Stage0 sanitized prompt 重新验收）

## 范围

本次补充 Stage 0-2 LMM 语义主链路的 fixed50 质量对照，复用 Stage 5 fixed50 manifest 中的 50 个实体，比较 deterministic Stage 0-2 与 `--stage02-planner qwen-local` 的差异。

## Run

Deterministic baseline：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --entity-manifest docs/evaluation/2026-07-17-stage5-model-fusion-fixed-50-manifest.json \
  --limit 50 \
  --candidate-budget 5 \
  --output-root outputs \
  --run-name stage02_rule_fixed50_20260721
```

Qwen semantic sanitized：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --entity-manifest docs/evaluation/2026-07-17-stage5-model-fusion-fixed-50-manifest.json \
  --limit 50 \
  --candidate-budget 5 \
  --stage02-planner qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --output-root outputs \
  --run-name stage02_qwen_fixed50_stage0_sanitized_20260722
```

对照脚本：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/compare_stage02_semantic.py \
  --baseline-run-dir outputs/stage02_rule_fixed50_20260721 \
  --semantic-run-dir outputs/stage02_qwen_fixed50_stage0_sanitized_20260722 \
  --sample-manifest docs/evaluation/2026-07-17-stage5-model-fusion-fixed-50-manifest.json \
  --output docs/evaluation/2026-07-21-stage02-qwen-fixed50-comparison.json \
  --csv-output docs/evaluation/2026-07-21-stage02-qwen-fixed50-comparison.csv \
  --overwrite
```

## 结果

两组 run 均完成：

```text
processed_entities = 50
failed_entities = 0
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
fallback_counts = {"image": 95, "text": 10}
stage_failure_counts = {}
```

Semantic call 状态：

```text
stage0_coarse     applied = 50 / 50
stage1_knowledge  applied = 50 / 50
stage2_reasoning  applied = 50 / 50
total             applied = 150 / 150, fallback = 0 / 150
contract_repair   stage0 = 0, stage1 = 0, stage2 = 0
local_salvage     stage0 = 0, stage1 = 0, stage2 = 0
```

Prompt 泄漏检查：

```text
checked_semantic_calls = 150
semantic_prompt_leaks = []
```

检查覆盖 `ground_truth_modalities`、测试密钥/令牌片段、Bearer 字样，以及 record 级 target description。未发现进入 semantic prompt。

## 对照摘要

```text
stage0_relation_set_changed = 4 / 50
stage0_relation_order_changed = 4 / 50
stage0_top_relation_changed = 3 / 50
stage1_entity_type_changed = 0 / 50
stage1_relation_facts_changed = 4 / 50
stage2_supported_facts_changed = 5 / 50
stage4_text_primary_changed = 4 / 50
stage4_image_primary_changed = 4 / 50
best_pair_changed = 1 / 50
best_text_changed = 1 / 50
best_image_changed = 1 / 50
avg_best_pair_score_delta = +0.014313
positive / zero / negative best-pair deltas = 25 / 24 / 1
```

主要变化样本：

```text
Q1393576   Gerald Fink                    Stage0/1/2 changed, delta +0.042530
Q16225071  Anton Shenfeld                 Stage0/1/2 changed, delta +0.065551
Q95030     Warren Beatty                  Stage1/2 changed, best pair changed, delta +0.070370
Q10904448  The Man Who Could Not Shudder  Stage0/1/2 changed, delta +0.052294
Q84        London                         Stage2 facts changed, delta +0.000000
```

最大正向 delta：

```text
Q5294      DVD                 +0.090936
Q95030     Warren Beatty       +0.070370
Q16225071  Anton Shenfeld      +0.065551
Q159846    Paramount Pictures  +0.060854
Q1321      Spanish             +0.056087
```

最大负向 delta：

```text
Q42574     James Cameron       -0.004643
```

## 实现补充

- `mmkgc.run` 新增 `--entity-manifest`，可按 manifest 中 `entity_key` 顺序选择实体，并在 `config.json` 中记录 `entity_manifest` 与 `selected_entity_count`。
- `scripts/compare_stage02_semantic.py` 新增 deterministic vs semantic Stage 0-2 对照输出，产物为 JSON 与 CSV。
- Stage0 rule 输出新增 `cleaned_relation_groups` / `compressed_evidence`，对应 MMKG 设计中的 rule-based cleaning and compression；Qwen Stage0 默认从这些候选中生成最终 `selected_relations`。
- Semantic prompt payload 对自环/同 label 邻居清空 `description`，避免目标实体 description 通过 self-neighbor 绕过 prompt 边界；sanitized fixed50 泄漏复查为 `0/150`。
- Stage1/Stage2 semantic prompt 进一步 compact，避免高证据实体输出过长 JSON 后被截断。
- Stage 0-2 semantic planner 增加本地 contract salvage：严格 JSON parse 失败时先回收截断响应中已经完整的 allowlisted 字段；salvage 不足时再走一次模型 contract repair。

## 结论

Stage 0-2 fixed50 对照已补齐。当前结论是：Qwen semantic Stage 0-2 在 fixed50 上工程链路稳定，prompt 泄漏边界通过；sanitized run 中 semantic fallback 为 `0/150`，本地 salvage/repair 为 `0/150`，Stage5 最终 best pair 变化 `1/50`，best-pair 均值 delta 为 `+0.014313`。

因此这项偏离可以从“未接入/未对照”降级为“已接入且已 fixed50 对照，工程 fallback 已清零”。剩余关键问题从 contract 稳定性转为质量判断：Qwen semantic 的净增益仍偏小，需要人工抽样判断 Stage0 relation set/order、Stage1 relation facts 与 Stage2 supported facts 的变化是否确实更优。
