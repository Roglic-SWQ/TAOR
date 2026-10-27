# MKG-W 多模态补全后续路线图

## 核心思路

当前 `MMKGC` 已经完成离线 MVP：把 `MMKG/` 中的六阶段设计落地为一个可运行、可测试、可追溯的 MKG-W 双模态补全流水线。

后续不建议直接跳到真实模型生成，而是按下面顺序推进：

```text
MVP 六阶段可跑通
-> 先建立评价标准
-> 再扩大运行规模
-> 最后接入真实生成模型
```

这个顺序的目的，是让每一步都有明确反馈闭环。先知道如何评价，再扩大实验规模，最后接真实模型，才能判断复杂模型是否真的带来提升。

## 当前基础：离线 MVP 六阶段

当前 MVP 已经具备以下能力：

- 读取 MKG-W 数据集。
- 支持 `missing_mode=text`、`missing_mode=image`、`missing_mode=both`。
- 默认处理 `both` 模式，同时生成 text candidates 和 image prompt candidates。
- 每个实体保存六阶段 JSON 产物。
- 输出 `results.jsonl`、`metrics_summary.json` 和 per-entity `final.json`。
- 不依赖外部模型服务，便于稳定复现实验。
- 为后续 LMM、SDXL、CLIP、BLIP、Sentence-BERT、Gemini 等模型预留 adapter 边界。

六阶段流程是：

```text
MKG-W entity
-> EntityRecord
-> 0 Coarse Relation Filtering
-> 1 Knowledge Modeling
-> 2 Graph-Constrained Reasoning
-> 3 Fine-grained Relation Selection
-> 4 Relation Alternation Generation
-> 5 KG-aware Candidate Pair Ranking
-> FinalResult
```

这个阶段的意义不是追求最终生成质量，而是先把项目主干跑通，让每个阶段都有稳定输入、输出和可检查产物。

## 第一阶段：先建立评价标准

第一阶段建议做“实验评估版”。

目标是回答：

```text
当前 pipeline 生成的 text、image prompt、text-image pair 到底好不好？
```

建议新增能力：

- 基于 MKG-W ground truth text 的文本相似度评价。
- 对 text candidates 统计 Top-1、Top-5、平均分、覆盖率等指标。
- 对 image prompt candidates 设计结构化评价字段，例如实体类型覆盖、关系覆盖、约束覆盖、风险词或 unsupported detail 统计。
- 对 text-image pair ranking 统计 best pair、Top-5 pair、pair consistency 分布。
- 输出可分析的 CSV / JSON summary。
- 支持按实体类型、关系类型、missing mode、fallback 类型做分组统计。
- 增加人工误差分析用的样本导出，例如最优、最差、fallback、多关系冲突样本。

这一阶段的成功标准：

- 能对一次运行结果生成稳定评估报告。
- 能比较不同 run 的指标差异。
- 能指出当前规则 pipeline 的主要错误来源。
- 后续接入真实模型时，可以用同一套评价脚本判断是否提升。

为什么先做这个：

如果没有评价标准，接入真实模型后只能主观看样例，很难判断模型是否真的提升了整体效果。评价标准先建立起来，后续每次改 prompt、换模型、扩大数据规模，都有可比较的数字和样本依据。

## 第二阶段：再扩大运行规模

第二阶段建议做“全量运行与工程化版”。

目标是回答：

```text
当前 pipeline 能不能稳定处理更大规模的 MKG-W 实体？
```

建议新增能力：

- 支持全量或大规模实体运行。
- 支持断点续跑，避免中断后从头开始。
- 支持跳过已完成实体。
- 增加更清晰的日志、进度和失败记录。
- 增加 run manifest，记录数据集路径、参数、代码版本信息、开始时间、结束时间。
- 支持多实验目录管理，例如按 run name、missing mode、candidate budget、模型版本组织结果。
- 增加失败恢复策略，例如单实体失败不影响整体运行。
- 增加结果校验命令，检查 jsonl 行数、实体目录数、阶段文件完整性和 metrics 是否一致。

这一阶段的成功标准：

- 可以稳定处理远大于 100 条的实体集合。
- 中断后可以从已有结果继续。
- 每次实验目录都可追溯、可复查、可比较。
- 大规模输出可以被第一阶段的评价脚本直接消费。

为什么放在评价之后：

大规模运行会产生大量结果。如果没有评价脚本，结果越多越难分析。先有评价标准，再扩大规模，才能让全量运行真正服务于实验结论。

## 第三阶段：最后接入真实模型

第三阶段建议做“真实生成版”。

目标是回答：

```text
真实模型是否比离线规则 MVP 生成得更好？
```

建议接入方向：

- 使用 LLM / LMM 生成更自然的 text description candidates。
- 使用 SDXL / Flux 等图像生成模型，从 image prompt candidates 生成真实图片。
- 使用 CLIP / BLIP 做图文一致性评价。
- 使用 Sentence-BERT / embedding 模型做文本语义相似度评价。
- 使用 LLM / VLM judge 做可解释质量判断。
- 保留每次模型调用的 prompt、参数、输出、失败信息和耗时。
- 支持规则候选与模型候选并排比较。

这一阶段的成功标准：

- 真实模型输出能接入现有六阶段流程，而不是另起一套不可比较流程。
- 真实模型结果能被第一阶段评价标准自动评估。
- 大规模运行能力可以支撑模型实验的批量执行。
- 能回答“真实模型在哪些类型实体或关系上提升明显，在哪些地方没有提升或更差”。

为什么最后做这个：

真实模型会引入环境、显卡、模型权重、API、速度、成本和随机性问题。如果在评价和工程化之前接入，会很快变成“能生成一些样例，但难以复现和比较”。把它放在第三阶段，可以让模型接入成为可度量的增强，而不是不可控的展示。

## 推荐推进顺序

推荐顺序是：

```text
1. 实验评估版
2. 全量运行与工程化版
3. 真实生成版
```

对应关系：

| 阶段 | 要解决的问题 | 主要产物 |
| --- | --- | --- |
| MVP 六阶段 | 能不能把流程跑通 | 六阶段 JSON、final result、rule candidates |
| 实验评估版 | 结果好不好 | metrics、CSV/JSON summary、误差分析样本 |
| 全量工程化版 | 能不能稳定大规模跑 | run manifest、断点续跑、完整性校验、大规模结果 |
| 真实生成版 | 模型是否真的提升 | LLM text、generated images、model-based ranking、对比实验 |

## 下一步建议

下一步优先写“实验评估版”的 spec 和 implementation plan。

建议它的第一版只做离线可验证指标，不接外部模型：

- text candidate vs ground truth text 的规则相似度。
- image prompt 的 KG 约束覆盖指标。
- pair ranking 的 Top-1 / Top-5 汇总。
- run-level summary。
- entity-level evaluation JSON。
- 可导出的 CSV 表格。

这样可以继续保持低风险、可测试、可复现，同时为后续全量运行和真实模型接入打好评价基础。
