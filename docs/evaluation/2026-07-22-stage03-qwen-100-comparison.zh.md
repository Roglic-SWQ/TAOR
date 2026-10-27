# Stage 0-3 Qwen semantic 100 实体验收

日期：2026-07-22

本次验收把默认 Qwen LMM 语义主链路扩大到 100 实体，比较 deterministic rule baseline 与 `--stage02-planner qwen-local --stage3-planner qwen-local` 的差异。为避免 SDXL 和 Stage 5 model-fusion 引入额外变量，本次两组都使用 `--stage5-ranker legacy`，图像候选保持 rule prompt。

## 运行命令

Rule baseline：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 100 \
  --candidate-budget 5 \
  --stage02-planner rule \
  --stage3-planner rule \
  --stage5-ranker legacy \
  --output-root outputs \
  --run-name stage03_rule_100_20260722
```

Qwen semantic：

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
  --run-name stage03_qwen_100_20260722
```

对照：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/compare_stage02_semantic.py \
  --baseline-run-dir outputs/stage03_rule_100_20260722 \
  --semantic-run-dir outputs/stage03_qwen_100_20260722 \
  --output docs/evaluation/2026-07-22-stage03-qwen-100-comparison.json \
  --csv-output docs/evaluation/2026-07-22-stage03-qwen-100-comparison.csv \
  --overwrite
```

## 结果

- Rule baseline：`processed_entities = 100`，`failed_entities = 0`。
- Qwen semantic：`processed_entities = 100`，`failed_entities = 0`。
- Stage 0/1/2 semantic calls：`300/300 applied`。
- Stage 3 semantic calls：`55/100 applied`，`45/100 fallback`；fallback 错误均为 `HTTPError: 400 Bad Request`。vLLM 日志显示部分 Stage 3 prompt 超过 `8192` context length，例如请求 `10632` 或 `12561` tokens。
- Prompt 泄漏复查：未发现 `ground_truth_modalities`、测试密钥片段或目标 description sentinel 进入 semantic prompt。

对照汇总：

```json
{
  "avg_best_pair_score_delta": 0.011728,
  "positive_best_pair_score_delta": 38,
  "negative_best_pair_score_delta": 7,
  "zero_best_pair_score_delta": 55,
  "best_pair_changed": 9,
  "best_text_changed": 4,
  "best_image_changed": 1,
  "stage0_relation_set_changed": 11,
  "stage0_relation_order_changed": 13,
  "stage0_top_relation_changed": 9,
  "stage1_relation_facts_changed": 11,
  "stage2_supported_facts_changed": 15,
  "stage4_text_primary_changed": 12,
  "stage4_image_primary_changed": 11
}
```

## 典型变化

正向 delta 最大的样本：

- `Q389284` Aftermath Entertainment：`+0.103118`
- `Q214665` Nino Rota：`+0.099710`
- `Q48` Asia：`+0.096884`
- `Q639095` Avtomobilist Yekaterinburg：`+0.078759`
- `Q180338` Christopher Lee：`+0.078117`

负向 delta 最大的样本：

- `Q6060` 50 Cent：`-0.029502`
- `Q142292` Traffic：`-0.019815`
- `Q559819` Toby Keith：`-0.019672`
- `Q15935` Kanye West：`-0.017224`
- `Q2831` Michael Jackson：`-0.015836`

## 结论

Stage 0-2 在 100 实体上工程稳定性已经通过：无实体失败，`300/300` semantic calls applied，泄漏边界未发现异常。质量侧有轻微正向信号，legacy best-pair 平均分提升 `+0.011728`，但 `55/100` 样本最终 best pair 没变化，说明 Qwen Stage 0-2 的改动仍偏保守。

Stage 3 是本次 100 验收暴露出的主要问题：`45/100` fallback，根因主要是 Stage 3 prompt 在高约束实体上超出 Qwen/vLLM `8192` context length。由于 fallback 后仍使用 deterministic Stage 3，pipeline 可以完成，但这还不能算 Stage 0-3 Qwen 主链路完全质量验收通过。

建议下一步优先修 Stage 3 contract 稳定性：压缩 Stage 3 输入、限制 relation_plans 输出数量、增强本地 salvage，或把 Stage 3 response schema 改得更窄。修完后重新跑同一 100 样本对照，再决定是否扩大到 500。
