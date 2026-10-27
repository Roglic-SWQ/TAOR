# MMKGC Scale-500 验证报告（2026-07-07）

## run 配置
- run_dir: `outputs/scale_500_both_20260707`
- evaluation_dir: `outputs/scale_500_both_20260707/evaluation`
- dataset_dir: `../KGC/datasets/MKG-W`
- missing_mode: `both`
- limit: `500`
- candidate_budget: `5`
- stage 统计输入: `outputs/scale_500_both_20260707/config.json`, `outputs/scale_500_both_20260707/metrics_summary.json`

## run 稳定性结果
- processed: `500`
- run failed: `0`
- avg_text_candidates: `5.0`
- avg_image_candidates: `5.0`
- avg_pair_candidates: `9.0`
- stage_failure_counts: 无

## evaluation 稳定性结果
- evaluated: `500`
- eval failed: `0`
- failed_entities: `0`
- failed_entity_keys: `[]`
- rows: `500`
- low_score_entities: `0`
- fallback_penalized_entities: `40`
- generic_entity_type_entities: `353`
- pair_single_side_penalized_entities: `405`
- template_prompt_penalized_entities: `496`

## 输出完整性检查
- results.jsonl: `500` 行（与 `limit=500` 一致）
- entity_scores.csv: `500` 条实体评分记录（文件行数 501）
- best_cases.jsonl: `20` 条
- worst_cases.jsonl: `20` 条
- fallback_cases.jsonl: `20` 条
- summary.csv: `18` 条汇总指标
- required run/eval 文件存在：
  - `outputs/scale_500_both_20260707/config.json`
  - `outputs/scale_500_both_20260707/metrics_summary.json`
  - `outputs/scale_500_both_20260707/results.jsonl`
  - `outputs/scale_500_both_20260707/evaluation/evaluation_summary.json`
  - `outputs/scale_500_both_20260707/evaluation/entity_scores.csv`
  - `outputs/scale_500_both_20260707/evaluation/summary.csv`
  - `outputs/scale_500_both_20260707/evaluation/best_cases.jsonl`
  - `outputs/scale_500_both_20260707/evaluation/worst_cases.jsonl`
  - `outputs/scale_500_both_20260707/evaluation/fallback_cases.jsonl`

## raw / calibrated 分布摘要
- Raw text_best_offline_evaluation_score: min `0.279081`, 中位数 `0.461422`, max `0.736435`, 平均 `0.458807`
- Raw image_best_offline_evaluation_score: min `0.679186`, 中位数 `0.975267`, max `0.994938`, 平均 `0.929764`
- Raw pair_best_offline_evaluation_score: min `0.707886`, 中位数 `0.892121`, max `0.974041`, 平均 `0.884526`
- Raw overall_offline_evaluation_score: min `0.564387`, 中位数 `0.767381`, max `0.900844`, 平均 `0.757699`
- Calibrated text_best_offline_evaluation_score: min `0.137764`, 中位数 `0.300199`, max `0.699613`, 平均 `0.334327`
- Calibrated image_best_offline_evaluation_score: min `0.414385`, 中位数 `0.714044`, max `0.937046`, 平均 `0.676363`
- Calibrated pair_best_offline_evaluation_score: min `0.610768`, 中位数 `0.650000`, max `0.973337`, 平均 `0.694379`
- Calibrated overall_offline_evaluation_score: min `0.348875`, 中位数 `0.551256`, max `0.849359`, 平均 `0.565168`

## fallback raw vs calibrated 对比
- rows: `500`
- fallback_count: `40`
- fallback_raw_avg: `0.752454525`
- fallback_calibrated_avg: `0.358755575`
- fallback_drop: `0.39369895`
- fallback_all_above_070: `false`（`0`）
- 结论：fallback 样本整体确有显著下降，下降幅度明显，且 `fallback_all_above_070 = false`，即不是全部超过 0.70 门槛。

## best / worst / fallback case 导出数量
- best_cases: `20`
- worst_cases: `20`
- fallback_cases: `20`

## 是否满足本阶段验收标准
- 规模扩展目标（500）已完成，且无 run/eval 失败。
- raw 维持较高总体稳定水平（`raw_overall_avg = 0.75769925`）。
- calibrated 降噪后整体分布下降且可区分 fallback 偏差（`calibrated_overall_avg = 0.565167504`）。
- fallback_penalized 样本下降充分（`fallback_drop = 0.39369895 >= 0.12`）且 `fallback_all_above_070 = false`。
- 因无失败实体，结论：`failed_entities=0`，无需失败修复分支。

### 验收结论
本阶段为 500 scale validation，未接入真实模型、未改评估公式。基于真实 run/eval artifact，任务验收通过。
