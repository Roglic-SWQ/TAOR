# MMKGC 离线评价标准验收报告

## 验收对象

- 运行目录：`outputs/acceptance_pytest_20260706_0002`
- 评估目录：`outputs/acceptance_pytest_20260706_0002/evaluation`
- 样本规模：100 个实体
- 失败实体：0 个
- 本轮目标：判断当前离线 evaluation 是否能帮助我们区分生成结果的好坏，并发现扩大规模前需要校准的评价偏差。

## 总体分布

| 指标 | 最小值 | 中位数 | 最大值 | 平均值 |
| --- | ---: | ---: | ---: | ---: |
| text_best_offline_evaluation_score | 0.327104 | 0.460883 | 0.675256 | 0.462437 |
| image_best_offline_evaluation_score | 0.689324 | 0.976339 | 0.994938 | 0.927464 |
| pair_best_offline_evaluation_score | 0.719658 | 0.886944 | 0.967228 | 0.885805 |
| overall_offline_evaluation_score | 0.638978 | 0.770133 | 0.869880 | 0.758569 |

导出的 case 数量：

- `best_cases.jsonl`：20 条
- `worst_cases.jsonl`：20 条
- `fallback_cases.jsonl`：5 条

## 初步结论

当前评价标准可以用于下一步人工验收和小规模扩大运行，但还不适合直接作为最终论文级评价指标。

它已经能暴露一个关键问题：文本候选质量偏弱，尤其是很多实体仍被描述成通用的 `entity`，导致真实描述匹配不足。与此同时，图像候选和 pair 候选分数整体偏高，容易把 fallback 或弱文本样本的 overall 分数抬高。

因此，本轮结论是：

- 可以继续扩大到 500 实体做稳定性验证。
- 扩大前建议先记录当前偏差，后续增加更严格的 type-specific 和 fallback penalty 校准。
- 暂时不要把当前 overall 分数当成最终模型效果，只把它当成离线规则版的筛查指标。

## 代表样本

### 高分样本

| entity | label | overall | text | image | pair | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q336177 | EVE Online | 0.869880 | 0.664297 | 0.988008 | 0.957336 | genre 关系能支撑文本和图像 prompt，text/image/pair 三者一致性较好。 |
| Q7499123 | Shit Hits The Fan | 0.869366 | 0.675256 | 0.990402 | 0.942441 | performer 关系直接覆盖真实描述中的歌手信息，属于规则信号较充分的样本。 |
| Q3498834 | Westside Story | 0.863344 | 0.673311 | 0.987313 | 0.929409 | 关系证据和候选 prompt 之间较一致，排序结果符合直觉。 |

高分样本的共同点：

- 最佳文本候选通常包含与真实描述相关的强关系，例如 `genre`、`performer`。
- 图像 prompt 明确包含实体 label 和关系上下文。
- pair 分数高，多数是因为 text/image 两侧的 Stage 5 分数比较均衡。

### 低分样本

| entity | label | overall | text | image | pair | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q3401588 | Prem Chopra | 0.638978 | 0.346578 | 0.701292 | 0.869065 | 真实描述是 “Indian actor”，但文本候选主要使用 cast member 反向证据，没有充分表达身份类型。 |
| Q132952 | Adam Sandler | 0.639582 | 0.344990 | 0.695414 | 0.878342 | 真实描述是 “American actor and comedian”，候选只抓到某部作品的 cast member 关系，身份覆盖不足。 |
| Q234900 | Linda Darnell | 0.653218 | 0.359024 | 0.711604 | 0.889027 | 真实描述是 “American actress”，候选更多是作品关联，缺少人物职业/国籍等核心描述。 |

低分样本的共同点：

- text 分数低是合理的，因为候选文本没有很好覆盖真实描述中的职业、国籍、类型等核心身份词。
- image 和 pair 仍然偏高，这说明当前图像/配对评价更偏向“有 label、有关系词、有约束”而不是“语义上足够具体”。
- low-score case 没有被触发，因为 overall 被 image 和 pair 拉高；当前 low-score 阈值和 overall 公式需要后续校准。

### Fallback 样本

| entity | label | overall | text | image | pair | 观察 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Q193559 | Valve Corporation | 0.741702 | 0.364737 | 0.966954 | 0.893416 | fallback 文本只说 entity type 是 entity，信息很弱，但 image/pair 分数仍高。 |
| Q3577037 | The West Wing | 0.748475 | 0.368000 | 0.969828 | 0.907597 | fallback 暴露出 entity type 建模过粗，真实描述中的 political drama/NBC 信息没有被表达。 |
| Q3713428 | Don't Fall in Love with Everyone You See | 0.766441 | 0.377104 | 0.977995 | 0.944223 | fallback case 的 overall 仍不低，说明 fallback penalty 需要更显式。 |

Fallback 样本的共同点：

- `case_tags` 能正确标出 `fallback`，这一点是有用的。
- 但 fallback 样本的 overall 并没有明显降低，主要因为 image 和 pair 分数过高。
- 扩大规模时必须单独统计 fallback，而不能只看 overall 平均分。

## 当前评价标准的可用性

### 可以保留的部分

- `text_best_offline_evaluation_score` 能较好暴露文本候选是否覆盖真实描述。
- `fallback_cases.jsonl` 能帮助定位关系证据不足或 entity type 过粗的样本。
- `best_cases.jsonl` / `worst_cases.jsonl` 能形成可人工抽查的 case pool。
- `stage5_text_score` / `stage5_image_score` 和 `*_offline_evaluation_score` 已经分开命名，便于后续分析。

### 需要校准的部分

- 图像分数整体过高，说明 relation/constraint/entity label 的词面覆盖过容易得分。
- pair 分数整体过高，尤其当 text 和 image 的 Stage 5 分数都高时，pair 会掩盖单侧质量问题。
- fallback 样本应有显式 penalty，否则 fallback 文本弱但 overall 仍可能偏高。
- `entity_type` 过多退化为通用 `entity`，会影响 text 和 image 的真实语义质量。
- 当前 low-score 阈值没有实际区分作用，100 个样本中 low-score entity 为 0。

## 建议的验收标准

在扩大到 500 实体前，建议暂时采用以下人工验收标准：

1. Best case 中，至少 70% 应该能从候选文本或图像 prompt 中看出与真实描述相关的关系证据。
2. Worst case 中，至少 70% 应该能解释出明确问题，例如身份词缺失、关系过窄、fallback、或候选过泛。
3. Fallback case 必须单独统计，不能只混入 overall 平均分。
4. 如果 fallback case 的 overall 仍普遍高于 0.70，下一轮需要加入 fallback penalty 或单独的 fallback-adjusted score。
5. 如果 image 平均分持续高于 0.90，需要收紧图像评价公式，降低 label/模板词的贡献，提高具体关系和类型覆盖要求。

## 下一步建议

推荐下一轮执行：

1. 先生成一个人工验收表，抽取 best/worst/fallback 各 10 条，增加人工列：`合理`、`不合理`、`问题类型`、`备注`。
2. 基于人工验收结果决定是否调整公式：
   - fallback penalty
   - image score 降权或增加具体性约束
   - pair score 增加单侧低分惩罚
   - entity type 退化为 `entity` 的专项标记
3. 再扩大到 500 实体运行，并比较 100 实体与 500 实体的分数分布和 fallback 比例。

本轮不建议直接全量运行，也不建议立即接真实模型。更稳的顺序是：先完成评价标准人工验收，再扩大规则版规模，最后接真实模型。
