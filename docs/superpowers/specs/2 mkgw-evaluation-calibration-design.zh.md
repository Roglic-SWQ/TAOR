# MKG-W Evaluation 校准设计文档

## 目标

在 `MMKGC/` 现有离线 evaluation 基础上增加一层可解释的校准分数，用于缓解三 Agent 人工验收中发现的稳定偏差：

- fallback 样本 overall 偏高。
- `entity_type=entity` 泛化样本没有被显式惩罚。
- text 弱时 pair 分数仍然偏高。
- image prompt 只靠 label、模板词和关系词即可获得过高分数。
- KG 关系证据没有区分核心关系和弱邻居关系。

本轮只校准 evaluation 层，不修改六阶段生成流程。校准后仍保留现有 raw offline 分数，同时新增 calibrated 分数和 penalty 诊断字段，方便比较校准前后变化。

## 非目标

本轮不做以下事情：

- 不修改 Stage 0-5。
- 不修改 text candidate 生成策略。
- 不修改 image prompt 生成策略。
- 不修改 Stage 5 ranking 逻辑。
- 不接入真实模型、embedding、CLIP、BLIP、LLM judge 或联网服务。
- 不扩大到 500 实体。
- 不自动 commit。

## 用户确认方案

brainstorming 阶段已确认：

- 范围选择：只改 evaluation 层。
- 分数策略：保留现有 raw 分数，新增 calibrated 分数。
- case 导出排序：按 `calibrated_overall_offline_evaluation_score` 排序。
- 校准强度：中等校准。
- 成功标准：测试通过，并且 100 实体重新评估后 fallback 样本 calibrated overall 明显下降，原 best 后半段 film 弱样本不再全部排在 best 前列。

## 输入依据

本轮设计基于以下已生成文档：

- `MMKGC/docs/evaluation/2026-07-07-acceptance-evaluation-review.zh.md`
- `MMKGC/docs/evaluation/2026-07-07-manual-acceptance-samples.zh.md`
- `MMKGC/docs/evaluation/2026-07-07-three-agent-acceptance-review.zh.md`

三 Agent 共识优先级：

| 优先级 | 问题类型 | 说明 |
| --- | --- | --- |
| P0 | fallback应扣分 | fallback 样本 overall 过高，是最明确校准点。 |
| P0 | entity_type过泛 | `entity_type=entity` 应成为显式风险信号。 |
| P0 | pair虚高 | text 弱时 pair 不应维持 0.85+。 |
| P1 | image虚高 | 模板化 prompt 过容易拿高分。 |
| P1 | 关系证据不足 | 需要区分核心关系和弱邻居关系。 |
| P1 | text弱 | 当前 text 指标能暴露问题，但还需要更好利用实体类型和真实描述。 |

## 架构设计

新增 `mmkgc.evaluation.calibration` 模块，职责是：

- 接收 evaluator 已经算出的 raw text/image/pair evaluation。
- 根据 entity metadata、fallback summary、候选字段和关系名推断校准 penalty。
- 输出 calibrated 分数和 `calibration_summary`。

为了让校准层能工作，`evaluator.py` 的 candidate summary 必须保留轻量校准源字段：

```text
primary_relation
relation_role
metadata
expected_relation_coverage_detail
generated_content
prompt_or_instruction
positive_prompt
text_candidate
image_candidate
```

这些字段只用于 evaluation 输出和校准诊断，不回写 Stage 0-5 原始产物。

不把校准逻辑塞进 `metrics.py` 的原始评分函数中，原因是：

- `metrics.py` 保持 raw offline metric 的稳定含义。
- 校准逻辑可以明确依赖 entity-level 上下文，例如 fallback、entity type、text/image 单侧分数。
- 后续可同时比较 raw 与 calibrated，而不会丢失历史分数。

## 输出字段

保留现有 raw 字段：

```text
text_best_offline_evaluation_score
image_best_offline_evaluation_score
pair_best_offline_evaluation_score
overall_offline_evaluation_score
```

新增 entity-level calibrated 字段：

```text
calibrated_text_best_offline_evaluation_score
calibrated_image_best_offline_evaluation_score
calibrated_pair_best_offline_evaluation_score
calibrated_overall_offline_evaluation_score
```

新增 `calibration_summary`：

```json
{
  "fallback_penalty": 0.0,
  "entity_type_generic_penalty": 0.0,
  "template_prompt_penalty": 0.0,
  "single_side_pair_penalty": 0.0,
  "relation_support_penalty": 0.0,
  "relation_support_strength": {
    "text": "direct_identity",
    "image": "direct_identity",
    "pair": "direct_identity"
  },
  "calibration_tags": []
}
```

penalty 字段表示扣掉的分数差值，范围 `[0, 1]`。例如 raw overall 为 `0.78`、calibrated overall 为 `0.60`，则相关 penalty 之和可解释这 `0.18` 的下降来源。

## 校准规则

### Fallback penalty

如果 entity 或候选包含 fallback：

- 增加 `fallback_penalty`。
- 增加 `fallback_penalized` tag。
- 降低 calibrated text/image/pair/overall。
- fallback 样本不应仅靠 image/pair 高分保持在 `0.70+`。

建议中等校准规则：

```text
if fallback_summary.has_fallback:
    calibrated_text = raw_text * 0.75
    calibrated_image = raw_image * 0.75
    calibrated_pair = raw_pair * 0.70
    calibrated_overall = average(calibrated_text, calibrated_image, calibrated_pair) * 0.90
```

### Entity type generic penalty

如果 `entity.entity_type == "entity"`：

- 增加 `entity_type_generic_penalty`。
- 增加 `generic_entity_type` tag。
- text 和 image 分数降权。

如果最佳文本候选包含明显泛化描述，例如：

```text
is described as a entity
has entity type entity
```

则 text 降权更明显。

建议中等校准规则：

```text
if entity_type == "entity":
    calibrated_text *= 0.85
    calibrated_image *= 0.90

if best_text contains generic entity template:
    calibrated_text *= 0.80
```

### Pair single-side penalty

pair 不能掩盖单侧质量问题。如果 calibrated text 或 calibrated image 明显低，calibrated pair 必须被压低。

建议中等校准规则：

```text
calibrated_pair = raw_pair

if calibrated_text < 0.40:
    calibrated_pair = min(calibrated_pair, 0.65)
elif calibrated_text < 0.50:
    calibrated_pair = min(calibrated_pair, 0.75)

if calibrated_image < 0.60:
    calibrated_pair = min(calibrated_pair, 0.80)
```

如果 pair 被 cap，增加 `pair_single_side_penalized` tag。

### Image template penalty

如果 image prompt 主要由 label、`entity`、`identity`、`profile context` 等模板词组成，而缺少具体类型或真实描述关键词，则降低 image calibrated 分数。

建议启发式：

- `entity_type == "entity"` 且 prompt 包含 `identity` 或 `profile context`：触发。
- prompt 中没有实体真实描述关键词，且 relation support 不是 `direct_identity`：触发。
- fallback image prompt：触发。

建议中等校准规则：

```text
if template_prompt_detected:
    calibrated_image *= 0.85

if fallback_template_prompt_detected:
    calibrated_image *= 0.75
```

### Relation support strength

只在 evaluation 层根据候选字段和关系名推断，不修改 Stage 3/5 输出。

支持类别：

```text
direct_identity
type_support
neighbor_indirect
fallback
unknown
```

建议推断规则：

- fallback candidate：`fallback`
- `primary_relation == "global_profile"`：`fallback`
- `relation_role == "fallback"`：`fallback`
- relation in `genre`, `performer`, `occupation`：`direct_identity`
- relation in `instance of`：`type_support`
- relation in `cast member`, `original broadcaster`, `member of sports team`：`neighbor_indirect`
- 无法识别：`unknown`

pair 的 relation strength 优先从 pair 内部的 `text_candidate` / `image_candidate` 推断；如果 pair summary 中没有保留嵌套候选，则使用当前 best text 和 best image 的较弱 relation strength 作为 pair strength。

建议 penalty：

```text
direct_identity: 0.00
type_support: 0.05
neighbor_indirect: 0.10
fallback: 0.20
unknown: 0.05
```

penalty 以乘法或差值方式体现在 calibrated text/image 中，但 raw 字段保持不变。

## Case 导出

`best_cases.jsonl` 和 `worst_cases.jsonl` 改为按：

```text
calibrated_overall_offline_evaluation_score
```

排序。

case record 同时包含 raw 和 calibrated：

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

`fallback_cases.jsonl` 仍筛选 fallback 样本，但记录 calibrated 分数和 penalty 诊断。

## CSV 和 Summary

`entity_scores.csv` 增加：

```text
calibrated_text_best_offline_evaluation_score
calibrated_image_best_offline_evaluation_score
calibrated_pair_best_offline_evaluation_score
calibrated_overall_offline_evaluation_score
calibration_tags
```

`evaluation_summary.json` 增加 calibrated averages：

```text
calibrated_text_best_offline_evaluation_score
calibrated_image_best_offline_evaluation_score
calibrated_pair_best_offline_evaluation_score
calibrated_overall_offline_evaluation_score
```

并增加 counts：

```text
fallback_penalized_entities
generic_entity_type_entities
pair_single_side_penalized_entities
template_prompt_penalized_entities
neighbor_indirect_entities
```

## 成功标准

实现完成后必须满足：

1. 全量测试通过。
2. `python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002 --overwrite` 成功。
3. 100 个实体仍全部评估成功，失败实体为 0。
4. raw 分数字段仍存在。
5. calibrated 分数字段存在于 entity JSON、`entity_scores.csv`、case JSONL、summary JSON。
6. `best_cases.jsonl` / `worst_cases.jsonl` 按 calibrated overall 排序。
7. 5 个 fallback 样本的 calibrated overall 平均值比 raw overall 平均值至少下降 `0.12`，且 fallback 样本 calibrated overall 不再全部高于 `0.70`。
8. 原 best 后半段 film 弱样本不再全部出现在 calibrated best top 10。
9. 不修改 Stage 0-5，不接模型，不联网，不新增第三方依赖。
10. 不自动 commit。

## 验证命令

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002 --overwrite
```

额外验收检查应读取：

```text
outputs/acceptance_pytest_20260706_0002/evaluation/evaluation_summary.json
outputs/acceptance_pytest_20260706_0002/evaluation/entity_scores.csv
outputs/acceptance_pytest_20260706_0002/evaluation/best_cases.jsonl
outputs/acceptance_pytest_20260706_0002/evaluation/fallback_cases.jsonl
```

确认 raw/calibrated 字段共存，并确认 fallback 与 film 弱样本的排序变化符合成功标准。
