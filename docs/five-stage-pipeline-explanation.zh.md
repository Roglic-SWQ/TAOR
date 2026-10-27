# MMKGC 当前阶段流程说明

更新时间：2026-09-21

本文说明 `MMKGC` 项目中五个逻辑阶段的 multimodal missing KG completion pipeline 实际实现方式。历史输出文件仍可能使用六阶段命名，这是为了兼容已有结果结构。

## 1. 总体入口

主流程由 `mmkgc.pipeline.MMKGCPipeline.run()` 串联：

```text
Stage 0: coarse_relation_filtering
Stage 1: knowledge_modeling + credibility_assessment + constraint_compilation
Stage 2: fine_grained_relation_selection
Stage 3: relation_alternation_generation
Stage 4: kg_aware_candidate_consistency_ranking
```

代码入口：

- CLI：`mmkgc/run.py`
- Pipeline：`mmkgc/pipeline.py`
- 阶段实现：`mmkgc/stages/*.py`
- 默认主链路构造：`mmkgc/defaults.py`
- Stage 4 model-fusion：`mmkgc/ranking/*.py`

当前只保留固定主链路：

- Stage 0-1 固定由确定性证据边界和本地 Qwen 严格 JSON semantic refinement 共同执行；Stage 1 的一次 Qwen 调用同时完成知识画像 patch 和事实可信度修正。
- Stage 2 固定使用 Sentence-BERT、Qwen assessor 和 Qwen global planner，不再提供规则评分或关闭 assessor 的分支。
- Stage 3 文本固定使用 Qwen；图像固定使用 Qwen prompt planner 和 SDXL。
- Stage 4 只使用 model-fusion ranker。

规则生成、rule-only planner、关闭 assessor、prompt-only image 和 legacy ranker 等替代执行链已从 CLI、默认构建器和 Pipeline 分支中移除。必需模型调用失败时直接记录失败并停止当前实体处理，不再静默切换链路。

CLI 主要参数：

```bash
python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode text \
  --candidate-budget 5
```

其中 `--missing-mode` 只支持 `text` 或 `image`，一次运行只生成一种模态。暂不处理两种模态同时缺失，也不生成 text-image pairs。

## 2. 数据输入

MKG-W 数据由 `mmkgc/data/mkgw.py` 读取：

- `entity2id.txt`、`relation2id.txt`：实体和关系映射。
- `train.txt`、`valid.txt`、`test.txt`：KG triples。
- `entity_info.json`、`relation_info.json`：实体 label、description、relation label、图片摘要等元信息。
- `images/`：实体已有图片。

默认 `--graph-split train` 只用 train triples 构建邻域，避免 valid/test 泄漏。每个样本被包装成 `EntityRecord`，包含：

- `entity_key`、`label`、`description`
- 一跳邻居 `neighbors`
- 每个一跳邻居的少量二跳邻居 `neighbors`
- `existing_modalities`
- `ground_truth_modalities`
- `modality_completion_policy`

当目标为 `text` 或 `image` 时，只将目标模态从 `existing_modalities` 中置空，另一侧模态保留为已有上下文。

## 3. Stage 0: Coarse Relation Filtering

设计文档：`MMKG/0 Coarse_Relation_Filtering.md`

实现文件：`mmkgc/stages/coarse.py`

### 目标

Stage 0 做高召回关系粗筛。它不是最终精排，而是在 Knowledge Modeling 前把邻域三元组清噪、压缩、分组，减少 prompt 噪声和长度。

### 实际处理

实现先按 `(relation, direction)` 聚合目标实体的一跳邻居：

```text
(Target, relation, Neighbor)
或
(Neighbor, relation, Target)
```

每个 relation group 会记录：

- `relation`
- `direction`
- `frequency`
- `neighbor_entities`
- 前 3 条 `evidence_triples`
- `coarse_score`
- `target_modality_priority`
- `relation_semantic_summary`
- `path_context`
- `neighbor_modality_priority`

规则清噪会过滤低价值关系，例如包含 `id`、`identifier`、`url`、`website`、`wikimedia`、`commons`、`catalog`、`archive`、`freebase`、`imdb`、`viaf` 等 token 的关系。

`coarse_score` 主要由以下信号组成：

- relation group 频次，最多贡献 3 分。
- 高价值关系加分，例如 `instance of`、`subclass of`、`genre`、`occupation`、`country`、`creator`、`author`、`director`、`depicts` 等。
- 视觉相关词加分，例如 `color`、`material`、`shape`、`depicts`、`flag`、`logo`、`cover`、`located`。
- 文本相关词加分，例如 `instance`、`genre`、`occupation`、`author`、`creator`、`director`、`country`、`subject`。
- 与已有 text 或 image summary 的 token cosine overlap。

规则 baseline 会保留：

```text
Top 8 relation groups
且 coarse_score >= 3
```

同时输出 `high_recall_relation_candidates`，最多保留前 24 个候选，供 Qwen semantic planner 继续粗筛。

### Qwen semantic refinement

固定主链路会调用 `mmkgc/stages/semantic.py` 中的 `StrictJsonStagePlanner.refine_coarse()`，让 Qwen 在规则清洗后的候选中选择更语义化的 `selected_relations`。

约束很严格：

- 只能使用输入 JSON。
- 不能引入 deterministic output 或 prior stage 中不存在的新关系。
- 不能使用 ground truth modalities。
- 只能返回指定 JSON shape。
- 解析失败会尝试本地 salvage 或 repair prompt。
- 仍失败则记录失败的 `model_calls` 并终止当前实体处理，不返回 deterministic 输出继续运行。

### 输出

主要输出：

- `cleaned_relation_groups`
- `high_recall_relation_candidates`
- `compressed_evidence`
- `selected_relations`
- `discarded_relations`
- `filtered_triples`

其中 `selected_relations` 会进入 Stage 1。

## 4. Stage 1: Knowledge Modeling

设计文档：`MMKG/1 knowledge_modeling_stage_design.md`

实现文件：`mmkgc/stages/modeling.py`、`mmkgc/stages/semantic.py`

### 目标

Stage 1 把 Stage 0 筛出的关系和已有模态整理成结构化实体知识模型。它不生成缺失模态，也不做最终候选排序。

它回答的问题是：

```text
这些筛选后的关系和已有模态共同说明目标实体是什么、有哪些事实、哪些信息可用于补全？
```

### 实际处理

实现会构建以下核心结构：

- `stage1_v3`，本阶段唯一权威语义模型
- `entity_profile`
- `modality_observations`
- `relation_facts`
- `uncertainties`

实体类型推断优先使用 Stage 0 中 `instance of` 且方向为 `out` 的关系。如果没有明确类型，则从已有 text 或 image summary 的前几个 token 做 fallback。`entity_profile` 中会记录：

- `entity_type` / `type`
- `secondary_types`
- `confidence`
- `type_confidence`
- `core_identity`
- `semantic_domain`
- `key_attributes`
- `type_evidence`
- `conflicting_evidence`
- `uncertain_aspects`

`relation_facts` 会把每条 selected relation 转为可追溯 fact：

- 关系方向为 `out` 时：`Target has relation: neighbors`
- 关系方向为 `in` 时：`Neighbor has relation to Target`
- 保留 `evidence_triples`
- `relation` 作为唯一关系字段；`predicate` 和单元素 `source_relations` 不在 Stage 1 新事实中重复保存
- 将 Stage 0 的 `coarse_score` 单独保存为 `coarse_relevance_score`
- 根据证据三元组、Stage 0 fact 引用和风险计算 `evidence_prior_score`
- Qwen 返回有界的 `score_delta`，最终分数为 `evidence_prior_score + score_delta`
- 由最终可信度策略派生 `credibility_label`、`allowed_use`、兼容字段 `support_level` 和 `generation_use`
- 不再推断关系级 `modality_relevance`；当前目标模态由运行参数 `missing_mode` 唯一确定
- `generation_utility` 不在本阶段计算，留给 Stage 3 根据目标模态评估

`modality_observations` 记录已有模态状态，包括 text、image path、image summary、ground truth 是否可用、completion policy。代码中会显式保留 ground truth availability，但后续 semantic planner prompt 会 scrub ground truth keys，避免生成链路泄漏答案。

### Qwen semantic refinement

固定主链路通过一次 `refine_knowledge()` 调用让 Qwen 同时返回针对 `stage1_v3` 的语义 patch 和事实可信度修正，包括 `entity_profile_patch`、`text_visual_profile_patch`、`key_attributes`、`uncertain_aspects` 以及 `fact_assessments`。

它仍受同样边界约束：不能新增或改写 `relation_facts`，不能修改 evidence refs、modality state 和 policy refs，不能使用 ground truth。对每个已有 `fact_id`，Qwen 只能返回 `score_delta`、`risk` 和 `reason`，其中 `score_delta` 限制在 `[-0.20, 0.20]`；代码负责计算最终 `credibility_score`、`allowed_use` 和生成约束。Patch 合并后，代码从 `stage1_v3` 同步派生外层兼容 `entity_profile` 和 `text_visual_profile`，避免两套画像不一致。

### 输出

Stage 1 输出给关系选择阶段使用，核心是：

- 目标实体的稳定身份和类型。
- 可追溯 KG fact。
- 不确定项和禁止自由推断的风险。

原 `missing_modality_clues` 和 `stage1_v3.missing_modality_clue_refs` 没有被后续阶段使用，现已删除。事实选择直接使用当前 `missing_mode` 和 Stage 1 生成的最终可信度/约束结果，不再依赖关系级模态相关性判断。

## 5. Stage 1 内部子步骤：Constraint Compilation

设计文档：`MMKG/2 constraint_compilation_stage_design.md`

实现文件：`mmkgc/stages/reasoning.py`

### 目标

当前事实可信度评定和约束编译已经合并到 Stage 1。这里保留原 `constraint_compilation` 输出结构，用于记录 Stage 1 生成的确定性约束；它不再是独立的 Qwen 阶段，也不会再次调用 Qwen 评价事实。

它回答：

```text
哪些内容必须包含？
哪些内容建议包含？
哪些只能作为背景？
哪些必须避免？
```

### 实际处理

Stage 1 的约束编译输出三类事实：

- `supported_facts`：有直接证据支持，可以用于生成。
- `safe_inferences`：由实体类型等强证据弱推出，只能作为安全提示。
- `uncertain_facts`：证据不足，只作为不确定性上下文。

同时按目标模态生成约束：

```text
modality_constraints.text.must_include
modality_constraints.text.should_include
modality_constraints.text.context_only
modality_constraints.image.must_include
modality_constraints.image.should_include
modality_constraints.image.context_only
```

可信度使用规则：

- `credibility_score >= 0.80` 且存在直接证据：`direct_generation`。
- `credibility_score >= 0.60`：`supporting_only`。
- `credibility_score >= 0.40`：`context_only`。
- 更低分或 `risk=high`：`excluded`。

约束编译规则：

- `entity_profile.type` 非 `entity` 时，text/image 都加入核心 `must_include: entity type`。
- `direct_generation` 映射到相关模态的 `must_include`。
- `supporting_only` 映射到相关模态的 `should_include`。
- `context_only` 映射到 `context_only`，`excluded` 不产生正向约束。
- 与当前模态不直接相关的事实进入 `context_only`。
约束编译还会输出 `constraint_policy`，其中核心 hard violation code 包括：

- `missing_core_must_include`

soft violation codes 包括：

- `missing_non_core_must_include`
- `missing_should_include`
- `context_only_evidence`
- `unsupported_high_risk_fact`

这些 codes 会被 Stage 4 hard gate、诊断输出和重生成反馈使用，不再产生候选扣分。

### 输出

Stage 1 的主要输出：

- `constraint_schema_version=graph_constraints_v2`
- `reasoning_summary`
- `fact_credibility_assessments`
- `supported_facts`
- `safe_inferences`
- `uncertain_facts`
- `modality_constraints`
- `constraint_policy`
- `reasoning_confidence`

这些结果会被 Stage 2、Stage 3、Stage 4 使用。

## 6. Stage 2: Modality-Aware Fine-Grained Relation Selection

设计文档：`MMKG/3 modality_aware_fine_grained_relation_selection_stage_design.md`

实现文件：`mmkgc/stages/selection.py`

### 目标

Stage 2 是高精度关系选择。Stage 0 负责别漏掉可能有用的关系，Stage 2 则在 Stage 1 的安全边界内决定哪些关系真正驱动生成。

它回答：

```text
哪些关系做 primary？
哪些做 secondary？
哪些只能做 context？
哪些必须 excluded？
每个关系分到多少候选 quota？
```

### 实际评分

对每个 `relation_fact`，实现计算：

```text
fine_score =
0.30 * modality_fit
+ 0.25 * constraint_alignment
+ 0.20 * evidence_strength
+ 0.20 * generation_utility
- 0.25 * risk_penalty
```

各项含义：

- `modality_fit`：关系是否适合当前目标模态。
- `constraint_alignment`：是否命中 `must_include`、`should_include`、`context_only`。
- `evidence_strength`：证据三元组数量和 `support_level`。
- `generation_utility`：当前事实是否容易转化为目标模态中的具体生成内容。它只使用 Stage1 实体画像动态锚点和当前关系/对象动态锚点，通过 Sentence-BERT 相似度计算，权重为 `40%/60%`。
- `risk_penalty`：低/中/高风险扣分。

硬排除条件：

- 没有 `evidence_triples`。
- `risk` 为 high 或 severe。
- `generation_use` 明确表示不允许直接使用。

context only 条件：

- `generation_use` 明确为 context only。
- 关系只出现在 `context_only`，且没有对应正向约束。

### 角色分配

排序后：

- 前 3 个高分 eligible relation 通常为 `primary`。
- 分数足够但不在前 3 的可为 `secondary`。
- 较弱但可用的为 `context`。
- 有硬排除原因的为 `excluded`。

每个 item 会记录：

- `relation_role`
- `fine_score`
- `score_breakdown`
- `target_modality`
- `role_decision`
- `generation_plan`
- `stage2_hard_exclusion_reasons`
- `stage2_context_only_reasons`

### 候选配额

`_assign_candidate_quotas()` 会按目标模态分配 quota。

单模态时：

- 先给每个 primary 至少 1 个候选。
- 最强 secondary 可补 1 个。
- 剩余 quota 轮流分给 primary。
- 若 primary/secondary 不足，则产生 fallback quota。

单模态模式下，目标模态按 `candidate_budget` 分配，因此默认会生成：

```text
5 text candidates
5 image candidates
```

### Qwen semantic refinement

固定主链路通过 `refine_selection()` 让 Qwen 输出 relation plans，包括：

- relation
- role
- target_modality
- candidate_quota
- reason

代码会把 Qwen 的计划应用回 deterministic selection，并重新计算/修补 quota，仍不能引入不存在的关系。

### 输出

Stage 3 输出：

- `primary_relations`
- `secondary_relations`
- `context_relations`
- `excluded_relations`
- `quota_summary`

这些内容决定 Stage 3 的候选生成顺序和每个关系生成几个候选。

## 7. Stage 3: Relation Alternation Generation

设计文档：`MMKG/4 relation_alternation_generation_stage_design.md`

实现文件：`mmkgc/stages/generation.py`

### 目标

Stage 3 把 Stage 2 的 relation plan 转成多样但受控的 text/image 候选。

它的核心不是“随机生成 N 个”，而是：

```text
relation-conditioned candidate generation
```

每个候选都绑定：

- `primary_relation`
- `generation_angle`
- `supporting_relations`
- `used_evidence_triples`
- `used_constraints`
- `expected_relation_coverage`
- `generation_brief`

这样 Stage 4 才能检查候选是否覆盖它声称使用的关系和约束。

### 候选生成顺序

`_relation_sequence()` 会读取 Stage 3 的 `primary_relations` 和 `secondary_relations`，按 `candidate_quota_by_modality` 展开为候选序列。

如果 quota 不足，会加入 `global_profile` fallback relation，保证每个目标模态都生成 `candidate_budget` 个候选。

generation angles 采用两个稳定基线和三个实体感知动态角度：

```text
identity anchor
evidence-grounded description
relation focus: {relation + neighbor entities}
{entity_type} attribute and context
{modality} semantic bridge for {entity_type}
```

其中动态角度读取 Stage 1 的实体类型、核心身份、语义领域、视觉场景线索和上下文线索。电影、人物、建筑等实体会得到不同的 `generation_angle` 和 `generation_brief.diversity_axis`，但仍保留前两个稳定角度以维持候选之间的可比较性。

### generation_brief

每个候选会先构建 `generation_brief`：

- `generation_intent`
- `semantic_focus`
- `diversity_axis`
- `evidence_policy=grounded_graph_only`
- `style_guidance`
- `must_include`
- `should_include`
- `context_only`
- `evidence_facts`

文本候选会把 brief 拼进 instruction。图像候选会把 brief 拼进 positive prompt。

### 文本生成

当目标模态包含 text 时，固定使用 `QwenLocalTextGenerator`。

每个文本候选输出：

```json
{
  "generated_content": {
    "type": "text",
    "value": "..."
  },
  "metadata": {
    "status": "generated"
  }
}
```

Qwen 调用信息会通过 `model_calls` 记录，包括 prompt、response 和状态。调用失败不会产生规则候选。

### 图像生成

当目标模态包含 image 时：

- 固定由 `QwenLocalImagePromptGenerator` 生成 image prompt。
- 固定由 `SDXLImageGenerator` 生成真实 PNG artifact。
- prompt-only 图像候选不属于支持的运行链路，因为 Stage 4 图像评分需要真实 image artifact。

生成成功时，候选的 `generated_content` 会变为：

```json
{
  "type": "image",
  "value": "...positive prompt...",
  "image_path": "...",
  "metadata_path": "...",
  "prompt": "...",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "seed": 20260708
}
```

生成失败时，metadata 会标记 `failed`，不会生成规则候选替代它。

### Stage 4 反馈再生成

如果 Stage 4 model-fusion 返回：

- `no_valid_candidate`

Pipeline 会调用 `safer_relation_alternation_generation()` 做一次安全再生成。

再生成会把 Stage 4 hard/soft violations 加入新的 evidence facts 中，候选 id 改为：

```text
text_safe_001
image_safe_001
```

然后重新进入 Stage 4 排序。

### 输出

Stage 4 输出：

- `relation_alternation_generation`
- `text_candidates`
- `image_candidates`
- `model_calls`
- `generated_image_artifacts`

## 8. Stage 4: KG-Aware Candidate Consistency Ranking

设计文档：`MMKG/5 kg_aware_candidate_consistency_ranking_stage_design.md`

实现文件：

- model-fusion ranker：`mmkgc/ranking/engine.py`
- hard gate：`mmkgc/ranking/hard_gate.py`
- fusion formulas：`mmkgc/ranking/fusion.py`
- ranking context：`mmkgc/ranking/context.py`
- ranker factory/config：`mmkgc/ranking/factory.py`、`mmkgc/ranking/profiles.py`、`configs/stage4-model-fusion.yaml`

### 目标

Stage 4 分别对 Text 和 Image 候选进行可信度评定。当前实现不再拆分“生成质量”和“KG consistency”两个中间总分，而是直接融合模型与规则组件得到一个 `candidate_score`。

### Ranking context

Pipeline 会用 `build_ranking_context()` 构造 leak-resistant `RankingContext`：

- `entity_profile`
- `relation_facts`
- `supported_facts`
- `safe_inferences`
- `modality_constraints`
- `text_candidates`
- `image_candidates`
- `generated_image_inputs`
- `existing_image_input`

`context.py` 中对字段做 allowlist 和类型清洗，避免 ground truth 或非预期字段进入排序上下文。

### Model-fusion ranker

默认主链路是 `ModelFusionRankingEngine`。

默认配置 `configs/stage4-model-fusion.yaml`：

- `sentence_bert`：required，用 `Sentence-BERT/all-mpnet-base-v2`。
- `clip`：required，用 `CLIP`。
- `blip_itm`：required，用 `blip-itm-large-coco`。
- `gpt4omini_judge`：required，用 `configs/mkgw-knowledge-extraction.yaml`。
- `bertscore`：enabled 但 optional。

### Step 1: Hard gate

每个单模态候选先过 `deterministic_candidate_gate()`：

硬拒绝包括：

- `candidate_id` 缺失。
- `target_modality` 不匹配。
- generator 失败。
- `generated_content` schema 无效。
- 文本为空。
- 图像缺少真实 image artifact。

通过 deterministic gate 的候选再交给 GPT-4o-mini judge。Judge 可返回 hard/soft violations。只有当 judge 明确要求 reject、有 hard code、置信度不低于 0.80、且有 evidence 时，才转成 hard reject；否则 hard code 降级为 soft violations。

### Step 2: 单模态评分

Text candidates：

- `gpt4omini_judge`：事实性、约束满足、流畅性等 judge score。
- `sentence_bert`：generated text 与 KG reference 的语义一致性。
- `bertscore`：optional auxiliary。
- `structured_kg_alignment`：模型盲抽取候选三元组后，与 Stage 1 支持的原始 KG 三元组做一对一结构化匹配。

Text 最终候选分：

```text
text_candidate_score =
0.20 * sentence_bert
+ 0.45 * structured_kg_alignment
+ 0.35 * judge
```

结构化匹配使用 subject/predicate/object=`0.30/0.40/0.30` 的软相似度，并由匹配 Precision、Recall 和 F1 得到最终组件分数。候选抽取保留 `polarity`，当前不调用 NLI。

Image candidates：

- `gpt4omini_judge`：视觉约束和 KG 一致性 judge。
- `clip`：image 与 visual constraint text 的匹配。
- `blip_itm`：image-text matching。
- `visual_graph_alignment`：VLM 不读取 KG，先从图片像素盲抽取 nodes、attributes、edges、OCR，再与可视觉化 KG 子图进行本地匹配。
- 视觉模型的 `0–100` 原始分数直接归一化到 `[0, 1]`，再按固定候选权重融合。

Image 最终候选分：

```text
image_candidate_score =
0.10 * normalized_clip
+ 0.15 * normalized_blip_itm
+ 0.50 * visual_graph_alignment
+ 0.25 * visual_judge
```

视觉图内部默认使用 nodes/attributes/edges/OCR=`0.35/0.20/0.30/0.15`；没有目标的组件不参与评分，其余权重重新归一化。soft violations 继续输出，但不再扣分。

### Step 3: Pair 暂停

当前不处理 Text 和 Image 同时缺失时的 Pair 组合评分。每次运行只生成和评估 `missing_mode` 指定的一侧候选，`pair_ranking` 固定为空。

### Ranking status

Stage 4 可能返回：

- `completed`：正常完成。
- `degraded`：optional scorer 失败，但 required scorer 正常。
- `no_valid_candidate`：text/image 候选都没有 survivor。

`no_valid_candidate` 会触发 Stage 3 safer regeneration。

### 输出

Model-fusion 输出：

- `ranking_status`
- `ranker=model_fusion`
- `score_version=stage4_structured_graph_v3`
- `profile_name`
- `required_scorers`
- `optional_scorers`
- `failed_optional_scorers`
- `candidate_score_weights`
- `ranking_model_calls`
- `text_ranking`
- `image_ranking`
- `pair_ranking`

Pipeline 最终会再整理 `final` 字段，包含：

- `best_text_candidate_id`
- `best_image_candidate_id`
- `best_pair_id`
- `best_pair`
- `top_5_pairs`
- 候选计数
- `ranking_status`
- `stage4_ranker`
- `ground_truth_used`

## 9. 五个逻辑阶段之间的数据依赖

可以把整条链路理解为：

```text
EntityRecord
  |
  v
Stage 0
  关系清洗、聚合、粗筛
  输出 selected_relations / filtered_triples
  |
  v
Stage 1
  结构化实体知识、可信度修正与约束
  输出 entity_profile / relation_facts / modality_constraints
  |
  v
Stage 2
  关系精选与 quota
  输出 primary/secondary/context/excluded relations
  |
  v
Stage 3
  按关系和约束生成 text/image candidates
  输出 candidates / model_calls / image artifacts
  |
  v
Stage 4
  hard gate + 多模型评分 + KG-aware fusion
  输出 top candidates / final result
```

每个阶段都保留 provenance。越往后，越少“发现新知识”，越多“约束、生成、验证”。

## 10. 当前实现与设计文档的对应关系

外层 `MMKG/` 的设计文档描述了理想阶段目标；`MMKGC/mmkgc/` 中的当前实现基本按这个设计落地，但做了几个工程化调整：

- Stage 0-2 的 deterministic baseline 是 Qwen refinement 的证据边界，不是失败后的备用链路。
- Qwen refinement 只允许修正已有结构，不允许越过 deterministic evidence boundary。
- Stage 3 将设计中的 `candidate_specs` 和 `generated_candidates` 合并为一个 candidate object，用 `metadata.status` 标记状态。
- Stage 3 图像候选必须包含真实 image artifact，不能退化为 prompt-only 候选。
- Stage 4 只运行 model-fusion，不再提供 legacy ranker 分支。
- Stage 4 的 CLIP、BLIP/ITM 和 Visual Judge 原始 `0-100` 分数直接归一化到 `[0, 1]` 后参与融合，不使用人工拟合偏差。
- 当 Stage 4 没有有效候选或有效 pair 时，Pipeline 会自动触发一次 Stage 3 safer regeneration。

## 11. 一句话总结

这 5 个逻辑阶段的核心思想是：先从 KG 邻域中高召回找关系，再在同一次 Stage 1 语义处理中完成知识建模、可信度修正和约束编译，接着选出最适合当前缺失模态的关系，按关系生成多样候选，最后用 KG-aware hard gate 和多模型融合排序选出最可靠的补全结果。
