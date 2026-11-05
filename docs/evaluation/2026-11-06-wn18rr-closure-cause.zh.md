# WN18RR 闭环失效：真实轨迹与程序因果审计

## 结论与证据范围

本例中闭合的是“候选是否复述 Stage 2 精选事实”，没有闭合“这些精选事实是否被原始来源支持”。错误在生成之前进入参照，Stage 4 将它们判为 supported，Reflect 的操作列表为空；即使人为让现有修复路由启动，它仍以原错误参照作为修复目标。

本次研究读取原有真实 Qwen／SDXL／Stage 4 产物，重建实际 Judge 输入，并做离线程序反事实检查。没有新 GPU／API 调用，没有改生产逻辑或原始运行产物。反事实中的标签修改是人为控制输入，**不是新的模型判断，更不是模型修复成功的证据**。

原始数据：`outputs/wn18rr-onboarding-20261106/attempt-r3/runs/`。实际 Stage 4：`outputs/wn18rr-stage4-20261106/attempt-r4/closure/`。机器审计：`outputs/wn18rr-closure-cause-20261106/causal-audit.json`；重建的真实文本 Judge payload：同目录 `text-judge-payload.json`。

## 1. 最早的确定性失真：或与猜测变成两条确定事实

| 位置 | 实际内容 |
| --- | --- |
| 图片描述 | “The animal **appears to be** a type of rodent **or** marsupial.” |
| Stage 1 抽取 | `(water deer, hypernym, rodent)` 与 `(water deer, hypernym, marsupial)` |
| 不确定性 | `uncertain_aspects=[]` |
| 合法 train 图谱 | `(water deer, hypernym, mouse deer)`，路径中另有 ruminant／tragulidae |
| Stage 2 | 三条全部选入，状态 ready |
| Stage 3 | “a type of rodent … also classified as a marsupial” |

这一步即使不调用任何外部生物学知识，也已发生可确认的逻辑错误：将“看起来像 A 或 B”升级成“A 且 B”。图片描述与原图是同一来源链，两个 citation ID 不构成独立证实。

`mmkgc/stages/triples.py::triple_response_errors` 校验字段、目标绑定、来源 ID 及原图引用，但不验证来源是否蕴含断言、不保留“或／猜测”的逻辑，也不校验跨来源语义冲突。当前每个抽取项只有 subject／relation／object／evidence_refs；顶层 uncertain_aspects 由同一模型填写，没有与具体事实绑定的强制不确定性。`apply_triple_response` 随后将图谱与模型抽取都标成 source_triple，保留 source_modalities，却没有区分“可观察属性”与“未经证实的分类推断”。

因此“有来源引用”在这一环节被实际当成了足以进入事实目录的条件，引用有效不等于内容获证实。

## 2. Rerank 与 Stage 2 没有拦住；不能靠相关性阈值补救

实际相关性分数：rodent 0.989672，mouse deer 0.979668，marsupial 0.651355。错误 rodent 排在第一，证明这些分数不能解释为真实性。

但本例只有三条事实，默认 top-k=64，全部都会暴露给 Stage 2。**尚不能归因于排序顺序导致错误被纳入**；没有顺序消融。真正的后续事实是：Stage 2 虽被提示选择 coherent set，仍返回全部三个 ID；`apply_fact_selection` 校验合法 ID、预算、引用与状态，没有语义冲突检查。

应继续保持 reranker 是相关性粗筛。简单提高阈值既不能保证去掉 rodent，还可能先去掉分数稍低的合法图谱事实。

## 3. Stage 4 看见的是已“确定化”的参照，不是完整来源链

重建 `build_text_judge_prompt` 的结果确认：

- payload 的 supported_facts 包含以上三条确定三元组，全部作为 must_include 检查目标。
- 原始图片描述中的 appears／or 没有保留在 EVIDENCE_BUNDLE 中。
- 文本 Judge payload 中有 `has_existing_image=true`，但没有原图字节；`judge_text` 调用文本 completion，没有传 image_input。其他本地视觉评分器看图，并不等于这次语义 Judge 看到了原图。
- `RankingContext.to_prompt_payload` 没有传完整 evidence_catalog／原始描述内容；事实保留引用 ID，却不能靠 ID 重新读取来源。`candidate_ranking_context` 又将参照限定在精选事实中。

因此 Judge 没有完整条件重做一次“原图／带保留语气的描述 → 分类事实”的审计。它实际给出 supported 的理由，主要是“候选文本确实说了 rodent／marsupial”。这是候选出现性与参照一致性，不能独立证明断言为真。提示词虽然要求处理来源冲突，实际输入已经损失了关键保留语气，模型也没有报告剩余事实之间的语义冲突。

同一个 Qwen 用于描述、抽取、生成和评审，存在相关错误的风险；**没有独立 Judge 对照，不能据此确认自评偏差的因果量，也不能保证换大模型就会修好**。

## 4. Reflect 已接上，但没有收到可执行的错误信号

实际 text 的四个 fact_checks 均为 supported，confidence=1.0，unsupported_claims=[]。程序产生 `repair_operations=[]`，反馈 repair_needed=false，`_needs_stage4_regeneration` 因而返回 false。内容再生成与修复复评确实没有执行。

实际还有不一致的弱信号：Sentence-BERT=0.909149，structured_kg_alignment=0.228541，Judge=0.75（事实维度仍为 4/4）。但这些数值不会自动成为事实修复指令。现有控制器只执行有来源、有具体对象的 contradicted／必要文本 missing／明确冲突的额外断言。

这项限制本意是避免把相似度低直接当作事实错误；不能靠“总分低就重写”来获得可信纠错。可将评分分歧用作一次定向复核的触发信号，但它本身不提供正确修复内容。

离线反事实结果：

| 人为控制输入，仅用于程序诊断 | 结果 |
| --- | --- |
| 置信度阈值改为 0、0.5、0.8、1.0，保留真实 supported 标签 | 均为 0 个操作，均不触发 |
| 候选总分改为 0，保留真实 observation | 仍不触发事实修复 |
| 仅将 rodent 检查改为 contradicted | 路由立即触发，证明连线存在 |

所以本例的直接阻断点是错误未被报告，不是 REPAIR_CONFIDENCE=0.8 太高，也不是缺少无条件重试次数。

## 5. 即使触发，当前修复的目标也无法纠正坏参照

将 rodent 检查人工标为 contradicted 后，实际生成的操作是：

```json
{"action":"replace", "expected":"(water deer, hypernym, rodent)"}
```

`_fact_stage3_feedback` 的任务是“按 supported fact 修正候选”，不会修订 Stage 1／2。`repair_is_eligible` 还要求检查对象、expected、来源保持不变，并保留之前被高置信度判为 supported 的事实。

另一组独立的合成程序控制确认：普通遗漏可补回并通过 eligibility；若同时将原先被当成正确的 rodent 检查降为 unverifiable，修复候选会被拒绝。这是防止通过删事实逃避验收的保护机制；保护本身应保留，但前提“原参照已核验”在本例不成立。

当前缺少“参照有误／来源冲突 → 上游重新核验或隔离 → 新事实版本 → 新生成尝试”的处理路径。增加 Stage 3 Reflect 次数，可能只会反复生成符合错误参照的文本。

## 6. 图片还有可核验范围与证据绑定问题

图片 habitat 完整断言包含 west Africa，largest chevrotain 包含相对最大及物种分类。单幅无定位信息、无比较对象的画面不能独立证明这些完整限定；Judge 却将它们标为 supported/direct。

`compile_verification_targets` 依赖 text_visual_profile.non_visual_facts 的精确匹配标记不可视觉验证事实；统一三元组构建当前给出 text_visual_profile={}，这些检查退化为 assess，由模型自行决定验证方式。反事实将 habitat 完整断言显式设为 unverifiable 后，`bind_observation` 会拒绝原来的 supported/direct 响应，说明保护逻辑存在，但未得到合适的输入范围标记。

此外，绑定器目前主要检查 text:／pixel: 前缀及非空字符串。实际 `text:0`、`pixel:1` 等索引式表述可以通过格式检查；它们不是程序核对过的文本跨度或实际像素证据。补齐 JSON／证据前缀只解决执行合同，不证明观察正确。即使强化引用绑定，仍需处理坏参照与事实可核验性的根问题。

## 建议修复顺序与验收

1. **先保留不确定性和来源语义。** 不把视觉猜测、二选一描述直接升级为确定分类事实。将猜测保留在可追溯的不确定记录，或先以原始合法来源复核；不能仅凭 evidence_refs 存在认定断言获证实。先用本例的 or→and 转换做回归，再覆盖其他不确定／否定／时间限定，避免动物分类特例列表。
2. **增加有界的参照复核路径。** 将“候选偏离已核验参照”与“参照本身冲突／缺乏支持”分开。前者继续现有 Stage 3 定向修复；后者先隔离并复核争议来源，保留旧事实／旧结果及版本，再决定是否建立新事实集。不能原地改 expected 来制造改善，也不宜每次重跑整条流水线。
3. **让评审访问争议项的合法原始证据，明确核验范围。** 文本问题需要时附已有原图；图片将可见属性与地理／比较／精确身份限定分开评价。评分分歧可以触发复核，不能直接代替矛盾证据。保留不可核验状态和安全的修复接受条件。

后续真实控制实验应分别比较“当前压缩参照 vs 带原始来源和不确定性的审计任务”、同证据下不同 Judge，以及参照修订后再生成。固定候选／来源与预算，不喂隐藏目标答案，人工逐项复核。它们尚未在本次执行，不能把建议写成已经改善。

复跑本次只读审计：

```bash
PYTHONPATH=. python scripts/analyze_wn18rr_closure.py
```

需要本机保存的原始产物及重建 payload；脚本明确标记所有合成反事实，并在结束时验证原始 run 文件哈希未变化。生产流水线本次未作修改。
