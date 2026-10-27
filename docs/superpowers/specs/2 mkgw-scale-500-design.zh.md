# MKG-W 500 实体扩大运行设计文档

## 目标

在已完成的 100 实体 MVP 和 evaluation 校准基础上，进行第一轮正式扩大运行，验证当前离线六阶段流程在 500 个 MKG-W 实体上的稳定性、输出完整性和校准评估分布。

本阶段的核心目标不是提升生成策略，也不是接入真实模型，而是回答：

- 当前 `both` 模式在 500 实体规模下是否能稳定完成。
- 生成输出和 evaluation 输出是否保持完整。
- raw / calibrated evaluation 指标是否能继续用于规模化质量判断。
- fallback、generic entity、pair 单侧偏弱、template image prompt 等校准标签在更大样本下是否有可解释分布。

## 背景

上一阶段已完成：

- 100 实体 `both` 模式 MVP 运行。
- 离线 evaluation。
- evaluation calibration。
- 100 实体验收报告。

当前 100 实体验收基线：

```text
run_dir: MMKGC/outputs/acceptance_pytest_20260706_0002
processed_entities: 100
run failed_entities: 0
evaluation evaluated_entities: 100
evaluation failed_entities: 0
fallback_raw_avg: 0.758801
fallback_calibrated_avg: 0.361703
fallback_drop: 0.397098
film_weak_in_best_top10: 2
```

本阶段沿用该基线的生成配置，只扩大实体数量。

## 用户确认方案

brainstorming 阶段已确认：

- 方案选择：直接 500 正式扩大运行。
- 规模：`500` 实体。
- 缺失模态：`missing_mode=both`。
- 候选数量：`candidate_budget=5`。
- 输出位置：`MMKGC/outputs/scale_500_both_20260707`。
- 成功标准：稳定性 + 校准分布。
- 失败处理：如果出现少量失败，可以先继续 evaluation 和报告；但最终验收仍要求修复到 `0 failed`，不能把 `failed_entities > 0` 的结果标记为阶段完成。

## 非目标

本阶段不做以下事情：

- 不修改 Stage 0-5 生成逻辑。
- 不修改 relation filtering / modeling / reasoning / relation selection / candidate generation / ranking 规则。
- 不修改 evaluation calibration 公式。
- 不接入真实 LMM、SDXL、CLIP、BLIP、Sentence-BERT、Gemini、embedding、retrieval 或联网服务。
- 不进行三 agent 人工式抽样评审。
- 不扩大到全量 MKG-W。
- 不覆盖已验收的 100 实体 run。
- 不自动 commit。

## 运行配置

从 `MMKGC/` 目录执行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir ../KGC/datasets/MKG-W \
  --missing-mode both \
  --limit 500 \
  --candidate-budget 5 \
  --output-root outputs \
  --run-name scale_500_both_20260707
```

期望 run 输出：

```text
MMKGC/outputs/scale_500_both_20260707/
  config.json
  results.jsonl
  metrics_summary.json
  entities/
```

如果 run 目录已存在，本阶段不得直接覆盖。应停止并报告，由用户决定是否换 run name 或清理旧输出。

## Evaluation 配置

run 完成后执行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/scale_500_both_20260707 \
  --overwrite
```

期望 evaluation 输出：

```text
MMKGC/outputs/scale_500_both_20260707/evaluation/
  evaluation_summary.json
  summary.csv
  entity_scores.csv
  best_cases.jsonl
  worst_cases.jsonl
  fallback_cases.jsonl
  entities/
```

## 验收指标

### Run 稳定性

`metrics_summary.json` 必须满足最终验收：

```text
processed_entities = 500
failed_entities = 0
stage_failure_counts = {}
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
```

如果 `failed_entities > 0`：

- 可以继续运行 evaluation 和报告，保留失败实体信息。
- 报告必须列出失败数量、失败实体键和失败阶段。
- 本阶段不得标记完成。
- 后续必须进入修复 / 重跑分支，直到最终 `failed_entities = 0`。

### Evaluation 稳定性

`evaluation/evaluation_summary.json` 必须满足最终验收：

```text
evaluated_entities = 500
failed_entities = 0
failed_entity_keys = []
```

### 输出完整性

必须确认以下文件存在：

```text
outputs/scale_500_both_20260707/config.json
outputs/scale_500_both_20260707/results.jsonl
outputs/scale_500_both_20260707/metrics_summary.json
outputs/scale_500_both_20260707/evaluation/evaluation_summary.json
outputs/scale_500_both_20260707/evaluation/summary.csv
outputs/scale_500_both_20260707/evaluation/entity_scores.csv
outputs/scale_500_both_20260707/evaluation/best_cases.jsonl
outputs/scale_500_both_20260707/evaluation/worst_cases.jsonl
outputs/scale_500_both_20260707/evaluation/fallback_cases.jsonl
```

`entity_scores.csv` 必须有 500 行实体记录，并包含 raw 与 calibrated 字段：

```text
overall_offline_evaluation_score
calibrated_overall_offline_evaluation_score
text_best_offline_evaluation_score
calibrated_text_best_offline_evaluation_score
image_best_offline_evaluation_score
calibrated_image_best_offline_evaluation_score
pair_best_offline_evaluation_score
calibrated_pair_best_offline_evaluation_score
calibration_tags
```

### 校准分布

必须计算并报告：

```text
rows
fallback_count
fallback_raw_avg
fallback_calibrated_avg
fallback_drop
fallback_all_above_070
calibrated_overall_avg
raw_overall_avg
generic_entity_type_entities
pair_single_side_penalized_entities
template_prompt_penalized_entities
```

最低验收要求：

```text
rows = 500
fallback_count >= 0
fallback_calibrated_avg < fallback_raw_avg    # 如果 fallback_count > 0
fallback_drop > 0                             # 如果 fallback_count > 0
best_cases.jsonl 行数 > 0
worst_cases.jsonl 行数 > 0
```

如果 `fallback_count = 0`，报告中应明确说明本次 500 样本没有 entity-level fallback 样本，因此 fallback raw/calibrated 均值不适用。

## 报告交付物

新增中文报告：

```text
MMKGC/docs/evaluation/2026-07-07-scale-500-validation.zh.md
```

报告必须包含：

- run 配置。
- run 稳定性结果。
- evaluation 稳定性结果。
- 输出完整性检查。
- raw / calibrated 分布摘要。
- fallback raw vs calibrated 对比。
- best / worst / fallback case 导出数量。
- 是否满足本阶段验收标准。
- 如果出现失败，必须列出失败实体和后续修复建议。

报告应写入真实数值和结论，不得保留任何待办、待定、待填写或省略号式内容。

## 验证命令

执行前基础验证：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
```

运行 500 实体：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir ../KGC/datasets/MKG-W \
  --missing-mode both \
  --limit 500 \
  --candidate-budget 5 \
  --output-root outputs \
  --run-name scale_500_both_20260707
```

运行 evaluation：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/scale_500_both_20260707 \
  --overwrite
```

规模验收检查：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import csv, json, pathlib
root = pathlib.Path("outputs/scale_500_both_20260707")
eval_root = root / "evaluation"
metrics = json.loads((root / "metrics_summary.json").read_text(encoding="utf-8"))
summary = json.loads((eval_root / "evaluation_summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader(open(eval_root / "entity_scores.csv", encoding="utf-8")))
fallback = [r for r in rows if "fallback" in r["case_tags"] or "fallback_penalized" in r.get("calibration_tags", "")]
print("processed_entities", metrics.get("processed_entities"))
print("run_failed_entities", metrics.get("failed_entities"))
print("stage_failure_counts", metrics.get("stage_failure_counts"))
print("evaluated_entities", summary.get("evaluated_entities"))
print("evaluation_failed_entities", summary.get("failed_entities"))
print("rows", len(rows))
print("raw_overall_avg", summary["averages"].get("overall_offline_evaluation_score"))
print("calibrated_overall_avg", summary["averages"].get("calibrated_overall_offline_evaluation_score"))
print("fallback_count", len(fallback))
if fallback:
    raw_avg = sum(float(r["overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
    cal_avg = sum(float(r["calibrated_overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
    print("fallback_raw_avg", round(raw_avg, 6))
    print("fallback_calibrated_avg", round(cal_avg, 6))
    print("fallback_drop", round(raw_avg - cal_avg, 6))
    print("fallback_all_above_070", all(float(r["calibrated_overall_offline_evaluation_score"]) > 0.70 for r in fallback))
else:
    print("fallback_raw_avg", "NA")
    print("fallback_calibrated_avg", "NA")
    print("fallback_drop", "NA")
    print("fallback_all_above_070", "NA")
print("generic_entity_type_entities", summary["counts"].get("generic_entity_type_entities"))
print("pair_single_side_penalized_entities", summary["counts"].get("pair_single_side_penalized_entities"))
print("template_prompt_penalized_entities", summary["counts"].get("template_prompt_penalized_entities"))
print("best_cases", summary["case_exports"].get("best_cases"))
print("worst_cases", summary["case_exports"].get("worst_cases"))
print("fallback_cases", summary["case_exports"].get("fallback_cases"))
'
```

## 执行流程约束

本阶段执行时必须继续遵守用户确认过的多 agent 流程：

1. 主 Codex 会话作为 controller。
2. 每个 plan task 派一个 implementer subagent。
3. 每个 implementer 必须有明确写入范围、验证命令、禁止 commit。
4. 每个任务完成后派两个 reviewer：
   - spec reviewer：检查是否符合 spec 和用户确认方案。
   - quality reviewer：检查代码质量、测试质量、职责边界、路径/模板/打包风险。
5. reviewer 发现问题时，不要跳过，回给 implementer 修。
6. 两个 reviewer 都 approve 后才标记任务完成。
7. 最后派 final reviewer 审完整变更。
8. 最后运行完整验证命令并汇报证据。
9. 工作区可能有用户未提交改动，不要 revert 任何非本任务改动。
10. 不要自动 commit。

## 完成定义

本阶段只有在以下条件全部满足时才算完成：

- 500 run 完成。
- 最终 run `failed_entities = 0`。
- evaluation `evaluated_entities = 500` 且 `failed_entities = 0`。
- evaluation 输出 raw / calibrated 字段齐全。
- best / worst / fallback cases 正常导出。
- 如果存在 fallback 样本，fallback calibrated overall 明显低于 fallback raw overall。
- 规模验证报告已写入 `docs/evaluation/2026-07-07-scale-500-validation.zh.md`。
- 完整验证命令通过。
- final reviewer approve。
- 没有自动 commit。
