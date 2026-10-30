# 按事实观察与候选定向修复

2026-10-30 实现，适用于文本和图片补全。沿用五阶段主链路，默认启用在
Stage 2 编译候选上下文时生成的事实检查清单。

## 检查范围

每个 Stage 3 候选携带 `verification_targets`，来自该候选自己的
`must_include` 和 `should_include`，不包含其他候选的必需事实。
每项保留 `check_id`、`fact_id`、claim、主语、关系、宾语、方向和
`evidence_refs`。身份约束使用独立的 `check_id`，`fact_id` 为 null。

文本目标默认直接检查。图片目标交由视觉 Judge 判断完整事实能否从像素验证；
如果 Stage 1 的 `non_visual_facts` 已逐字列出该 claim，则固定为不可验证。
这里不依靠关系名称的硬编码来推断外观。

## Observe 输出

Stage 4 正常评分请求增加 `observation`，不增加一轮独立的观察请求。
每个 `check_id` 必须恰好返回一次；未知、重复或漏写 ID 均触发一次有界契约修正。
图片契约修正仍携带原始评估要求及图片，避免在没有画面的情况下补写视觉判断。

`fact_checks` 的状态：

| status | 含义 |
|---|---|
| `supported` | 候选表达了目标，并有内容证据 |
| `contradicted` | 候选内容与提供的事实明确冲突 |
| `missing` | 目标能够直接验证、画面或文字足以判断，但未表达 |
| `unverifiable` | 当前模态无法验证完整事实 |
| `uncertain` | 内容模糊、遮挡、裁切或其他原因导致证据不足 |

`verification_mode` 为 `direct`、`appearance_only` 或 `unverifiable`。
仅有外观线索时不能把完整 KG 事实判为 supported、contradicted 或 missing。
例如，画面中存在河流不能证明它是指定的某条河，建筑外观不能证明设计者身份。

每项还包含 `expected`、`observed`、`evidence`、`confidence`、`reason`。
程序从原检查清单补入事实及来源，Judge 不能改写事实或引用另一候选的要求。
文本证据以 `text:` 开头，图片证据以 `pixel:` 开头；生成 prompt 或 KG 要求本身
不能充当已观察到的内容证据。当前证据检查校验字段及来源标记，不等于自动证明
模型的语义判断正确。

`unsupported_claims` 单独记录没有任何已提供证据支持的新增断言。它表示证据不足，
不表示 KG 未记录的事必然为假。普通取景、光照等渲染选择不属于新增事实。

程序另提供 `violated_facts`、`missing_facts`、`unverifiable_facts` 和
`repair_needed`，方便直接查看诊断结果。

## Reflect 和候选选择

每个实体最多执行一轮修复，针对当前最佳候选。如果没有合格候选，则从存在可修复
观察条目的候选中选一个。不同候选的反馈不混用。

当前使用 `mmkgc.observation.REPAIR_CONFIDENCE = 0.8` 作为初始动作阈值；
这是工程默认值，尚未经过真实样本校准。

| 观察结果 | 动作 |
|---|---|
| 高置信、可直接验证的明确冲突 | `replace`，使用原证据支持的正确内容 |
| 高置信、可直接验证的必需事实遗漏 | `add` |
| 高置信的无依据断言 | `remove` |
| 已支持的内容 | `preserve` |
| 推荐事实遗漏、无法验证或不确定 | 记录诊断，不单独触发修复 |

replace/add 还必须有原计划的证据引用。修复请求包含该候选的原文本或原 prompt、
完整 observation、具体 operations 和允许使用的事实/证据。千问原始提示词配置
和 GPT 对照配置均接收这些字段；初始生成的原始千问提示词保持原有格式。
图片修复输出新的最终 positive prompt，再由 SDXL 出图。

修复结果再经 Stage 4 观察。程序保留原候选和修复候选的记录，并在同一候选的
两个版本之间先比较事实问题，再比较质量。修复必须实际解决至少一个原问题，
保留此前已经确认的事实，并且不能引入新的高置信冲突或无依据断言。
把原遗漏改判为模糊、无法验证，不视为解决问题。
之后再让胜出的版本与其他候选按原有质量评分排序。

修复生成或评估失败时保留原排序，记录异常。原本无合格候选的实体仍保持
`no_valid_candidate`，不会因为记录了修复尝试而变成成功。

## 查看结果

`3_relation_alternation_generation.json` 的 `data` 保留候选检查清单、原候选、
修复候选及修复反馈。新候选使用 `text_safe_001` / `image_safe_001` 等 ID，
`source_candidate_id` 指向原候选。

`4_kg_aware_candidate_consistency_ranking.json` 的 `data`：

- `text_ranking` / `image_ranking` 的 `ranked_candidates[].observation`：逐事实检查。
- `stage3_regeneration.feedback.candidate_feedback`：原内容及具体修复操作。
- `stage3_regeneration`：次数、修复前后状态和候选 ID；异常时记录失败原因。
- `observation_version`：`fact_observation_v1`。

`ranking_model_calls.jsonl` 同样保存 Judge 返回的绑定后 observation。
GPT Judge 配置的输出预算由 1024 调整为 4096 token，以容纳逐事实检查。

## 离线验证

`tests/test_fact_observation.py` 覆盖候选隔离、方向和事实引用、图片不可见关系、
必需与推荐遗漏、低置信判断、响应完整性、真实 Judge/排序器接口、千问修复提示词、
修复退化以及文本和图片的流水线/输出保存。

这些测试使用模拟模型响应，不代表真实 GPT、千问或 SDXL 的检测与修复质量；
需要另行用实际实体校准置信度阈值和检查视觉误判。

2026-10-30 本地验证：173 项事实观察、生成反馈、千问提示词、并发及流水线相关
测试通过；另外 20 项现有 Judge 解析测试通过。未运行真实模型生成。
