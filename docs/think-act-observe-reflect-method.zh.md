# 知识图谱约束的 Think–Act–Observe–Reflect 多模态补全方法

## 摘要式概述

本文提出一种知识图谱约束的 Think–Act–Observe–Reflect 多模态补全框架，用于在已有实体模态和局部知识图谱证据不完整的情况下生成缺失的文本或图像模态。框架首先对局部关系进行高召回筛选，并构建带有事实可信度和证据引用的实体知识模型；随后通过关系重排序确定生成重点，将关系计划编译为受约束的 prompt；生成结果再由 KG-aware 的观察模块进行事实一致性、关系一致性和内容覆盖评估；最后根据违反项和遗漏项构造有限的反思反馈，用于安全重生成。

该设计将知识图谱从静态输入转化为贯穿规划、生成和评估的可追溯约束。

## 1. 问题定义

给定目标实体 `e` 的已有模态 `x_e`、局部知识图谱 `G_e`，以及需要补全的缺失模态 `m`，目标是利用已有模态和知识图谱中的结构化事实与关系证据，生成缺失模态的候选 `y`，使其与实体身份、已有模态语义和图谱事实保持一致。

本文重点处理 image completion，同时保留 text completion 作为统一接口。

其中：

```text
G_e = (V_e, E_e)
y_m* = argmax_y Consistency(y | x_e, G_e)
      subject to Valid(y, m)
```

- `V_e` 和 `E_e` 分别表示目标实体局部邻域中的实体和关系边。
- `Valid(y, m)` 表示候选满足目标模态、格式和生成器约束。
- `Consistency(y, G_e)` 表示候选与知识图谱证据、实体画像和生成约束的一致程度。

## 2. Think–Act–Observe–Reflect 总体框架

现有 MMKGC 五阶段主链路可以在更高层抽象为四个认知环节。Think 对应关系粗筛、知识建模、可信度评估和关系重排序；Act 对应受关系计划控制的 prompt 生成和图像生成；Observe 对应 KG-aware 候选一致性排序；Reflect 对应根据观察结果进行的一次受约束 safer regeneration。

该抽象不改变五个阶段的工程职责，而是为方法提供统一的闭环解释。

| 闭环环节 | 对应阶段 | 核心问题 | 主要输出 |
|---|---|---|---|
| Think | Stage 0–2 | 哪些证据可靠，哪些关系值得驱动生成？ | 知识模型、事实可信度、关系计划 |
| Act | Stage 3 | 如何把关系计划转化为生成行为？ | prompt、文本候选或图像候选 |
| Observe | Stage 4 | 候选是否符合实体、事实和关系约束？ | 质量分数、违反项、覆盖信号 |
| Reflect | 反馈到 Stage 3，必要时反馈到 Stage 2 | 如何修复违反和遗漏？ | 结构化修正反馈、重新生成结果 |

整体过程可以表示为：

```text
K = Think(G_e, x_e)
P = Act(K, m)
y = Generate(P)
O = Observe(y, K)
P' = Reflect(P, O, K)
```

## 3. Think：知识建模与关系重排序

Think 阶段不要求生成模型自由地产生事实，而是首先建立一个可追溯、可分级使用的知识模型。其输入为局部图谱证据和已有模态，输出为实体画像、关系事实、可信度标签、模态约束和关系生成计划。

### 3.1 局部关系筛选

对于目标实体，从一跳邻域及有限路径上下文中获得候选关系集合：

```text
R_e = {r_1, r_2, ..., r_n}
```

系统首先按关系和方向聚合邻居，保留频次、邻居实体、证据三元组和路径上下文，并过滤 ID、URL、归档和其他低生成价值的元数据关系。

该步骤的目标是高召回地压缩知识邻域，而不是在此处完成最终关系选择。

### 3.2 知识建模与事实可信度

每个候选关系被转换为带证据引用的事实对象：

```text
f_i = (e, r_i, o_i, d_i, E_i, c_i)
```

其中：

- `r_i`：关系；
- `o_i`：邻居实体；
- `d_i`：关系方向；
- `E_i`：支持事实的证据集合；
- `c_i`：最终可信度。

可信度由确定性证据先验和有限的语义修正共同决定：

```text
c_i = clip(c_i^prior + Δ_i, 0, 1)
Δ_i ∈ [-δ, δ]
```

语义模型只能针对已有 `fact_id` 返回有限的修正量、风险等级和理由，不能新增事实、修改证据或直接指定生成权限。

根据最终可信度和风险，事实被编译为以下四类使用策略：

| 事实类别 | 生成使用方式 | 含义 |
|---|---|---|
| `direct_generation` | `must_include` | 有直接证据支持，可作为核心生成内容 |
| `supporting_only` | `should_include` | 可以辅助生成，但不应替代实体身份 |
| `context_only` | `context_only` | 只提供背景，不作为确定事实表达 |
| `excluded` | 不进入正向约束 | 风险过高或证据不足，禁止正向生成 |

### 3.3 关系重排序与生成计划

为了使有限的候选预算集中于最有价值的关系，系统综合模态适配性、约束匹配、证据强度、生成效用和风险惩罚进行关系评分：

```text
S(r_i) = λ_1 M_i + λ_2 A_i + λ_3 E_i + λ_4 U_i - λ_5 Q_i
```

其中：

- `M_i`：关系与目标模态的适配性；
- `A_i`：关系与生成约束的匹配度；
- `E_i`：证据强度；
- `U_i`：关系作为生成依据的效用；
- `Q_i`：风险惩罚。

关系按分数和硬排除条件划分为 `primary`、`secondary` 和 `context` 三种角色，并进一步分配候选配额。

示例关系计划：

```json
{
  "primary_relations": ["instance of", "creator"],
  "secondary_relations": ["genre"],
  "context_relations": ["country"],
  "must_include": ["entity type", "creator"],
  "should_include": ["genre"]
}
```

## 4. Act：关系条件化 Prompt 与图像生成

Act 阶段将 Think 阶段产生的关系计划、实体画像和事实约束编译为可执行的生成指令。与直接将原始 KG 拼接到 prompt 不同，本文只将允许使用的事实和目标模态相关的关系传递给生成器，同时明确限制模型不得补充未被证据支持的精确细节。

### 4.1 Prompt 规划

对于目标模态 `m`，prompt 由实体身份、主要关系、支持事实、视觉或文本场景线索以及不确定性边界共同组成：

```text
P_m = PromptPlanner(K, R_primary, R_secondary, C_m)
```

Prompt 主要包含：

- **实体身份**：实体类型、核心身份和语义领域。
- **关系重点**：当前候选对应的 primary relation 及其支持关系。
- **证据约束**：`must_include`、`should_include` 和 `context_only` 的分层内容。
- **安全边界**：禁止把不确定事实转换为确定的视觉或文本细节。

### 4.2 多候选图像生成

为了避免单一 prompt 对生成结果的偶然性，系统围绕不同生成角度产生多个候选：

- `identity anchor`
- `evidence-grounded description`
- `relation focus`
- `attribute/context`
- `semantic bridge`

每个候选都绑定其 primary relation、supporting relations、证据三元组和使用的约束。

```text
I_j = G(P_j, z_j),    j = 1, ..., B
```

其中 `B` 为候选预算，`P_j` 为关系条件化 prompt，`z_j` 表示采样随机性或生成条件。

对于图像补全，Qwen prompt planner 负责生成图像 prompt，SDXL 负责生成实际 PNG artifact；对于文本补全，则由文本生成器直接生成候选文本。

## 5. Observe：生成结果的知识一致性观察

Observe 阶段不重新改变知识模型，而是检查候选是否符合 Think 阶段确定的证据边界和关系计划。它将生成质量、语义对齐、结构化知识对齐和 Judge 诊断组合为候选观察结果。

```text
O(y, K) = (q(y), a(y, K), g(y, K), V(y, K), M(y, K))
```

其中：

- `q`：生成质量；
- `a`：语义和模态对齐；
- `g`：与知识图谱事实的结构化对齐；
- `V`：违反项；
- `M`：遗漏项。

### 5.1 事实与关系违反

观察模块重点识别以下类型的问题：

- 实体身份不一致，例如实体类型或核心身份发生偏移。
- 候选引入了原始 KG 和允许事实中不存在的关系或属性。
- 候选内容与 KG 支持的关系方向、邻居实体或事实发生冲突。
- 候选使用了 `excluded` 或 high-risk 信息。
- 图像或文本没有体现当前候选的主要关系。

### 5.2 关系覆盖与内容遗漏

每个候选保留 `expected_relation_coverage`，用于记录该候选理论上应表达的关系。观察模块将其与候选实际表达出的事实或视觉语义进行比较，形成覆盖信号。

对于文本候选，可以通过结构化三元组抽取与 KG 三元组对齐计算 Precision、Recall 和 F1：

```text
Precision = |T_pred ∩ T_KG| / |T_pred|
Recall = |T_pred ∩ T_KG| / |T_KG|
F1 = 2 · Precision · Recall / (Precision + Recall)
```

对于图像候选，则使用图文相似度、视觉判别器和 KG-aware Judge 的联合信号评估关系表达。

在论文表述中，观察模块不应只报告一个总分，还应将低 Recall 解释为潜在遗漏，将低 Precision 解释为潜在无依据内容。

当前工程实现已经保留候选关系覆盖、结构化 KG 对齐、hard/soft violation 和 Judge evidence 等诊断字段，这些字段共同构成 Observe 的输出。

## 6. Reflect：约束保持的反馈与安全重生成

Reflect 阶段根据 Observe 的结果生成修正反馈，但不重新发明事实，也不允许模型绕过知识模型。其作用是将观察结果转换为下一次生成可执行的限制。

```text
Δ = Reflect(O, K, P)
P' = UpdatePrompt(P, Δ)
```

| 观察问题 | 反馈动作 | 是否适合重生成 |
|---|---|---|
| unsupported claim / fact | 删除不受支持的内容 | 是 |
| relation or identity conflict | 保持实体身份并修正关系表达 | 是；严重时重新选择关系 |
| missing core evidence | 补充已有 KG 支持的核心事实 | 是 |
| low relation coverage | 增强 selected relation 的表达 | 是 |
| generator/schema/service failure | 记录系统失败，不把它伪装成内容问题 | 否；需由系统处理 |

反思反馈可以使用如下结构化表示：

```json
{
  "repair_scope": "stage3",
  "required_fact_refs": ["fact:creator"],
  "remove_fact_refs": [],
  "remove_relations": ["unsupported location"],
  "missing_relations": ["creator"],
  "repair_actions": [
    "include graph-supported creator evidence",
    "remove unsupported location claims"
  ]
}
```

本文采用有限反思策略。若候选没有有效结果，或观察结果包含可由生成器修复的严重违反项，系统构造 safer regeneration feedback 并重新生成一次候选；否则直接保留当前排序结果。

这种设计保证反思过程可审计、成本可控，也避免生成系统陷入无界循环。

## 7. 统一推理算法

```text
Algorithm 1: KG-Grounded Think–Act–Observe–Reflect Completion

Input: entity e, local KG G_e, existing modality x_e, target modality m

1:  C <- CoarseRelationFiltering(G_e, m)
2:  K <- KnowledgeModeling(C, x_e)
3:  K <- CredibilityAssessmentAndConstraintCompilation(K)
4:  R <- RelationReranking(K, m)
5:  P <- PromptPlanning(K, R, m)
6:  Y <- GenerateCandidates(P)
7:  O <- ObserveCandidates(Y, K, R)
8:  if O contains repairable violations or unacceptable omissions then
9:      Δ <- Reflect(O, K, R)
10:     P' <- UpdatePrompt(P, Δ)
11:     Y' <- GenerateCandidates(P')
12:     O' <- ObserveCandidates(Y', K, R)
13:     return BestValidCandidate(Y ∪ Y')
14: else
15:     return BestValidCandidate(Y)
16: end if
```

## 8. 与 MMKGC 当前实现的对应关系

该方法抽象与当前代码的对应关系如下。论文中应明确说明：Think–Act–Observe–Reflect 是对现有五阶段主链路的统一方法解释，而不是声称已经实现了无限多轮自主反思。

| 方法概念 | 当前实现 | 说明 |
|---|---|---|
| Think | Stage 0–2 | coarse filtering、knowledge modeling、credibility/constraint compilation、relation selection |
| Act | Stage 3 | 关系条件化候选生成；图像路径为 Qwen prompt planner + SDXL |
| Observe | Stage 4 | hard gate、文本结构化 KG 对齐、CLIP/BLIP、Visual Judge、KG-aware Judge |
| Reflect | Stage 3 feedback | 当前为一次 safer regeneration；反馈来自 Stage 4 的可修复 violations |

## 9. 方法边界与论文表述建议

- 将该方法描述为知识图谱约束的有限闭环，而不是通用自主智能体。
- 强调关系计划是 Think 与 Act 之间的中间表示，使知识证据可追踪到 prompt 和候选。
- 将 hard violation、soft violation、结构化对齐和 Judge evidence 统一作为 Observe 信号。
- 明确 Reflect 的修正范围：删除不支持内容、补充已有证据、强化已选关系，不创造新事实。
- 如果视觉关系抽取尚未完全稳定，应将图像侧关系遗漏描述为联合观察信号，而不要声称有完备视觉场景图推理。

## 10. 论文方法章节结构建议

1. `3.1 Problem Definition`
2. `3.2 Overview of the Think–Act–Observe–Reflect Framework`
3. `3.3 Think: Knowledge Modeling and Relation Reranking`
4. `3.4 Act: Relation-conditioned Prompt Planning and Image Generation`
5. `3.5 Observe: KG-aware Multimodal Consistency Evaluation`
6. `3.6 Reflect: Constraint-preserving Regeneration`
7. `3.7 Candidate Selection and Inference Procedure`

## 11. 可概括的论文贡献

1. 提出一个以知识图谱为证据边界的 Think–Act–Observe–Reflect 多模态补全框架。
2. 将关系重排序结果显式编译为 prompt 生成约束，降低无依据内容和关系偏移。
3. 通过生成后观察与有限反思反馈，统一处理事实违反、关系不一致和内容遗漏。

## 说明

本文档是根据当前 MMKGC 五阶段主链路整理的方法章节草稿。其中 Observe 和 Reflect 的表述保留了当前实现边界；若后续新增独立的关系覆盖检测或多轮反思控制器，可在第 5、6 节中继续扩展。
