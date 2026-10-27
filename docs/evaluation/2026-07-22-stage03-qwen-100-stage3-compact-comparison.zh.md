# Stage 3 Qwen prompt compact 100 实体验收

日期：2026-07-22

本次修复针对上一轮 Stage 0-3 Qwen 100 验收中暴露的 Stage 3 fallback 问题：`45/100` Stage 3 relation-selection 请求因 prompt 超过 Qwen/vLLM `8192` context length 返回 `HTTPError: 400 Bad Request`。

## 修改

Stage 3 prompt 改为 compact 输入：

- `deterministic_output` 只保留 relation plan 决策所需字段：`relation`、`relation_role`、`fine_score`、`score_breakdown`、`target_modality`、`candidate_quota`、`candidate_quota_by_modality`。
- `prior_stage_outputs.reasoning` 只保留 compact `supported_facts`、`safe_inferences`、`uncertain_facts`、`reasoning_confidence` 和 `reasoning_summary`。
- `prior_stage_outputs.knowledge` 复用 compact knowledge payload。
- Stage 3 的 `record.neighbors` 只保留前 6 个邻居的 `relation`、`direction`、`label`，不再带长 description。
- Stage 0-2 prompt payload 不变。

对上一轮 100 baseline artifacts 重建 Stage 3 prompt 后，Qwen tokenizer 计数最大值为 `2928` tokens，低于 `8192` context length；加上 `max_tokens=1024` 后仍有充足余量。

## 运行命令

Qwen semantic compact rerun：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 100 \
  --candidate-budget 5 \
  --stage02-planner qwen-local \
  --stage3-planner qwen-local \
  --stage5-ranker legacy \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --model-call-timeout-seconds 120 \
  --output-root outputs \
  --run-name stage03_qwen_100_stage3_compact_20260722
```

对照：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/compare_stage02_semantic.py \
  --baseline-run-dir outputs/stage03_rule_100_20260722 \
  --semantic-run-dir outputs/stage03_qwen_100_stage3_compact_20260722 \
  --output docs/evaluation/2026-07-22-stage03-qwen-100-stage3-compact-comparison.json \
  --csv-output docs/evaluation/2026-07-22-stage03-qwen-100-stage3-compact-comparison.csv \
  --overwrite
```

## 结果

- Rule baseline：`processed_entities = 100`，`failed_entities = 0`。
- Qwen semantic compact：`processed_entities = 100`，`failed_entities = 0`。
- Stage 0/1/2 semantic calls：`300/300 applied`。
- Stage 3 semantic calls：`100/100 applied`。
- Stage 3 fallback：从上一轮 `45/100` 降为 `0/100`。

对照汇总：

```json
{
  "avg_best_pair_score_delta": 0.011724,
  "positive_best_pair_score_delta": 37,
  "negative_best_pair_score_delta": 10,
  "zero_best_pair_score_delta": 53,
  "best_pair_changed": 10,
  "best_text_changed": 4,
  "best_image_changed": 2,
  "stage0_relation_set_changed": 11,
  "stage0_relation_order_changed": 13,
  "stage0_top_relation_changed": 9,
  "stage1_relation_facts_changed": 11,
  "stage2_supported_facts_changed": 15,
  "stage4_text_primary_changed": 15,
  "stage4_image_primary_changed": 12
}
```

## 结论

Stage 3 的工程稳定性问题已修复：100 实体中 Stage 0/1/2/3 共 `400/400` semantic calls applied，没有 fallback。与 rule baseline 相比，best-pair 平均分仍为小幅正向，`+0.011724`，与上一轮 `+0.011728` 基本一致，说明 prompt compact 主要修复 contract/context 问题，没有显著改变整体 scoring 分布。

剩余问题转为质量判断：虽然 Qwen Stage 0-3 主链路工程上已通过 100 实体验收，但 `53/100` 样本 best-pair delta 为 0，净增益仍偏小。下一步应对正负 delta 样本做人工抽样，判断 relation selection 的语义调整是否真正优于 deterministic baseline，再决定是否扩大到 500。

