# MMKGC 当前阶段设计

本文按当前代码实现整理运行时五个逻辑阶段。历史文件名和部分输出键仍保留旧的 Stage 编号，属于兼容性命名，不代表仍有独立的模型阶段。

## 单一主链路约束

当前运行入口只保留一条链路：

```text
Stage 0  关系粗筛与证据边界建立
Stage 1  知识建模 + Qwen 可信度修正 + 约束编译
Stage 2  Sentence-BERT + Qwen assessor + Qwen global planner 关系选择
Stage 3  Qwen 文本生成 / Qwen 图像提示词 + SDXL 图像生成
Stage 4  model-fusion 单模态候选排序
```

CLI 和 Python 默认构建器不再提供 `rule`、`legacy`、`none` 等模式选择参数。Qwen、Embedding、SDXL 或 Stage 4 必需评分器失败时，本次实体处理直接失败并记录错误，不会静默切换到规则生成、规则评分或 legacy ranker。

仍保留的确定性操作不是备用链路：Stage 0 的关系清洗与聚合负责建立证据边界；Stage 1 的可信度策略负责把 Qwen 修正编译成约束；Stage 4 hard gate 负责拒绝无效候选；Stage 3 的一次 safer regeneration 属于既定的“反馈→修正”闭环。

```text
EntityRecord
 → Stage 0 关系粗筛
 → Stage 1 知识建模、可信度修正与约束编译
 → Stage 2 LLM + Embedding 关系选择
 → Stage 3 关系条件化候选生成
 → Stage 4 KG-aware 多模型融合排序
```

## Stage 0：Coarse Relation Filtering

实现：`mmkgc/stages/coarse.py`

目标是高召回压缩 KG 邻域，而不是最终选择关系。代码按 `(relation, direction)` 聚合一跳邻居，保存频次、邻居实体、最多三条证据三元组和路径上下文。包含 `id`、`url`、`wikimedia`、`archive` 等 token 的低价值元数据关系会被清除。

其余关系按照频次、高价值关系类型、文本/视觉相关性、已有模态 token overlap 等信号计算 `coarse_score`。规则基线保留 Top-8 且 `coarse_score >= 3`，同时输出最多 24 条 `high_recall_relation_candidates` 供 Qwen 语义粗筛。输出包括 `cleaned_relation_groups`、`selected_relations`、`filtered_triples`、`compressed_evidence` 和 `discarded_relations`。

## Stage 1：Knowledge Modeling + Credibility Assessment

实现：`mmkgc/stages/modeling.py`

目标是把关系组转换成可追踪的实体知识模型。首先推断实体类型、次级类型、核心身份和语义领域，再为每个关系建立 `relation_fact`。事实中保留 subject/object、relation、direction、neighbor_entities、evidence_triples、risk 以及 Stage 0 的 fact/path 引用。当前一次运行只生成一个目标模态，因此不再为每条关系额外判断 `modality_relevance`。

Stage 1 明确分离三类分数：`coarse_relevance_score` 只继承 Stage 0 的关系召回相关性；`evidence_prior_score` 表示确定性证据先验；最终 `credibility_score` 由证据先验和 Qwen 的有界修正量相加得到；`generation_utility` 留到关系选择阶段根据目标模态计算。每条事实还包含由最终可信度策略生成的 `credibility_label` 和 `allowed_use`，兼容字段 `support_level`、`generation_use` 由这套策略派生，不再由 `coarse_score` 决定。

`stage1_v3` 是本阶段的权威知识模型。Qwen 一次返回 `entity_profile_patch`、`text_visual_profile_patch`、`key_attributes`、`uncertain_aspects` 以及针对已有 `fact_id` 的 `fact_assessments`。事实评定只允许返回 `score_delta`、`risk` 和 `reason`；关系事实、证据引用、模态状态和策略引用不可由 Qwen 改写。代码将 `score_delta` 限制在 `[-0.20, 0.20]`，计算最终可信度，再编译 `allowed_use` 和目标模态生成约束。

同时记录 `modality_observations`、已有文本/图像、图像摘要和 completion policy。原 `missing_modality_clues` 与 `missing_modality_clue_refs` 未被后续阶段消费，现已删除；后续阶段直接根据当前 `missing_mode`、Stage 1 的 `allowed_use` 和目标模态约束选择事实。Ground truth availability 可以被记录，但后续 prompt 和 ranking context 会清洗答案字段。

## Stage 1 内部：Credibility Policy + Constraint Compilation

实现：`mmkgc/stages/reasoning.py`

Stage 1 内部完成“生成可信度评定”和图约束编译。Qwen 只能针对 Stage 1 已存在的 `fact_id` 返回有界 `score_delta`、`risk` 和 `reason`；不能新增事实、改写事实内容、直接指定 `allowed_use`，也不能直接生成约束。未知或重复 `fact_id`、非法修正量和非法风险等级会使该实体处理失败。

代码根据最终可信度和风险统一计算 `allowed_use`：

```text
score >= 0.80 且有直接证据 → direct_generation
score >= 0.60              → supporting_only
score >= 0.40              → context_only
其余或 high risk           → excluded
```

随后再映射为约束：`direct_generation` 进入相关模态的 `must_include`，`supporting_only` 进入 `should_include`，`context_only` 进入 `context_only`，`excluded` 不产生正向生成约束。事实同时分为 `supported_facts`、`safe_inferences` 和 `uncertain_facts`，text/image 分别维护：

```text
must_include / should_include / context_only
```

实体类型仍作为身份锚点进入 `must_include`。不适合当前模态的可信事实只进入 `context_only`，不确定或排除信息仅保留在 uncertainty/context 中。Stage 2 会继续遵守 `allowed_use`：`supporting_only` 不能提升为 primary，`excluded` 必须排除。

Stage 1 还定义约束代码：硬约束包括 `missing_core_must_include`；软约束包括 `missing_non_core_must_include`、`missing_should_include`、`context_only_evidence` 和 `unsupported_high_risk_fact`。这些代码贯穿 Stage 2–4。

## Stage 2：LLM + Embedding Fine-Grained Relation Selection

实现：`mmkgc/stages/selection.py`

这是当前版本的固定混合筛选阶段：Sentence-BERT embedding component、Qwen 单关系 assessor 和 Qwen global semantic planner 都是主链路必需组件。Stage2 不再提供规则评分、关闭 assessor 或其他 mode 分支。

### Embedding 评分

Sentence-BERT 将 relation fact、目标模态语义锚点和 Stage 1 约束编码，得到：

```text
modality_fit          关系对目标模态的适配性
constraint_alignment  关系与 must/should/context 的匹配度
generation_utility    关系作为生成依据的价值
```

其中 `modality_fit` 不再使用通用模态锚点，而是根据 Stage 1 的知识模型动态构造两组正向锚点：

```text
实体动态锚点       45%   entity_profile / text_visual_profile 中的实体类型、领域和场景
关系动态锚点       55%   当前关系及其邻居实体对应的可视或可描述内容
```

计算方式为：

```text
modality_fit = 0.45×entity_similarity
             + 0.55×relation_similarity
```

如果实体信息缺失，只使用关系动态锚点；如果关系信息缺失，则该关系无法形成有效的关系锚点。当前版本只使用正向锚点，不加入 negative anchor；因此不会因为暂未考虑的消极语义额外扣分。

`generation_utility` 也使用动态锚点，并与 `modality_fit` 使用独立的锚点集合。它只按照实体画像和当前关系两组计算：

```text
实体动态锚点       40%
关系动态锚点       60%
```

实体锚点来自 `primary_type`、`secondary_types`、`semantic_domains`、`core_identity`、`visual_scene_cues` 和 `context_cues`；关系锚点来自当前 relation、object/neighbor entities，并根据 image/text 以及创作者关系、视觉属性关系等关系类型生成不同的表达模板。这样不同实体和关系会得到不同的生成效用语义空间。

`evidence_strength` 和 `risk_penalty` 仍由规则计算。最终分数为：

```text
fine_score = 0.30×modality_fit
           + 0.25×constraint_alignment
           + 0.20×evidence_strength
           + 0.20×generation_utility
           - 0.25×risk_penalty
```

### LLM assessor

Qwen assessor 在 embedding baseline 上做单条关系的语义校正，主要调整前三个语义分量，不能随意修改证据强度和风险惩罚，也不能创造 Stage 0/1 中不存在的关系。每次调用都会保存 assessment 和 model call 引用。

### 全局 planner、排除与角色

Qwen global planner 从整体角度输出 `relation`、`role`、`target_modality`、`candidate_quota` 和 `reason`，但不能覆盖 hard exclusion；调用或契约校验失败时终止该实体处理，不再采用本地结果继续运行。

以下情况直接 `excluded`：无 evidence triples、risk 为 high/severe 或 `generation_use` 不允许直接使用。其余按分数和顺序分配：

```text
fine_score < 0.35   → excluded
通过排除后的前 3 个 → primary
其余且 fine_score ≥ 0.5 → secondary
其余                 → context
```

### Quota

每个 primary 先分一个候选，剩余配额优先给最高分 secondary，再轮流增加 primary。关系不足时使用 global profile fallback。默认预算为 5；每次运行只为当前目标模态分配候选。

## Stage 3：Relation Alternation Generation

实现：`mmkgc/stages/generation.py`

Stage 3 读取 Stage 2 的关系角色和 quota，按不同 generation angle 生成受控候选，而不是无约束随机采样。候选绑定 `primary_relation`、`supporting_relations`、`generation_angle`、证据三元组、使用的约束和 `generation_brief`。

生成角度采用“2 个稳定基线 + 3 个动态角度”：identity anchor 和 evidence-grounded description 保持稳定；relation focus 根据当前关系及邻居实体变化；attribute/context 根据实体类型和语义领域变化；semantic bridge 根据目标模态、实体类型及 text visual cues 变化。动态信息来自 Stage 1 的 `primary_type`、`secondary_types`、`semantic_domains`、`core_identity`、`visual_scene_cues`、`context_cues`，以及当前 relation fact。`generation_brief.diversity_axis` 与动态角度同步变化。

文本固定由 Qwen 生成；图像固定由 Qwen 生成 prompt，再由 SDXL 生成真实 PNG artifact。任一必需生成步骤失败时不会切换到规则生成器。若 Stage 4 返回 `no_valid_candidate`，会将诊断信息加入新一轮 evidence guidance，执行一次 safer regeneration；这是主链路中的反馈修正步骤，不是备用生成链。

## Stage 4：KG-Aware Candidate Consistency Ranking

实现：`mmkgc/ranking/engine.py`、`graph_alignment.py`、`hard_gate.py`、`fusion.py`、`context.py`

只使用 `model-fusion`。运行入口已移除 legacy lexical ranker 分支；首先构造经过 allowlist 清洗的 ranking context，防止 ground truth 进入排序。

### Hard gate

候选 ID 缺失、模态不匹配、生成器失败、schema 无效、文本为空或图像没有真实 artifact 时直接拒绝。Judge 只有在明确 reject、给出 hard code、置信度足够且有证据时才升级为 hard reject，否则降级为 soft violation。

### 单模态候选评分

Text 和 Image 分别输出一个 `candidate_score`，不再把两种模态组合成 Pair。所有评分组件先归一化到 `[0, 1]` 后直接加权，不再使用 `violation_penalty`：

```text
text_candidate_score = 0.20×sentence_bert
                     + 0.45×structured_kg_alignment
                     + 0.35×judge

image_candidate_score = 0.20×normalized_clip
                      + 0.30×normalized_blip_itm
                      + 0.50×visual_judge
```

Text 先由模型在不读取目标 KG 的情况下从生成文本抽取带 `polarity`、`confidence` 和证据片段的三元组，再与 Stage 1 支持的原始 KG 三元组按 subject/predicate/object=`0.30/0.40/0.30` 做一对一软匹配，以图匹配 Precision/Recall/F1 作为 `structured_kg_alignment`。当前不使用 NLI。

当前图片排序暂不启用 `visual_graph_alignment`。图片只使用 CLIP、BLIP 和 Visual Judge，图谱投影与候选视觉图抽取代码保留在隔离模块中，待目标子图和匹配规则稳定后再恢复。

CLIP、BLIP 和 Visual Judge 的 `0–100` 原始分数直接除以 100 并截断到 `[0, 1]`，不再使用人工复审得到的偏差校准。`soft_violations` 仅用于诊断和重生成反馈，不参与候选分数。Stage 4 输出通过 `candidate_score_weights` 记录当前实际权重，`score_version=stage4_image_fusion_v4_no_visual_graph`。

Stage 4 只评估当前目标模态的候选，暂不处理 Text 和 Image 同时缺失及 pair 组合评分。输出仍保留统一结果结构，但未生成的模态候选为空；只有 `no_valid_candidate` 才会回到 Stage 3 重生成一次。

## 阶段职责总结

| 阶段 | 核心问题 | 主要产物 |
|---|---|---|
| Stage 0 | 哪些邻域关系可能有用？ | 高召回关系候选 |
| Stage 1 | 这些关系说明实体是什么，可信度如何？ | 实体画像、关系事实与生成约束 |
| Stage 2 | 哪些关系真正驱动生成？ | LLM/Embedding 关系计划与 quota |
| Stage 3 | 如何产生多样且受控的候选？ | text/image candidates |
| Stage 4 | 哪个 Text/Image 候选最可靠？ | final candidate ranking result |

总结：先用 KG 证据建立知识和约束，再用 Embedding 与 LLM 选择关系，按关系生成候选，最后通过 hard gate 和多模型信号完成 Text/Image 候选评分与排序。
