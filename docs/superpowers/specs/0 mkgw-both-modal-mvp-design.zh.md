# MKG-W 双模态补全 MVP 设计文档

## 目标

在 `MMKGC/` 下构建一个独立的 Relation-guided Knowledge Bridger pipeline，把 `MMKG/` 中的六阶段设计落地到 MKG-W 数据集上，用于多模态知识图谱缺失模态补全。

第一版是可验证 MVP：

- 支持 `missing_mode=text`、`missing_mode=image`、`missing_mode=both`。
- 默认验收模式是 `both`。
- 默认在 MKG-W 上处理 100 个实体。
- 不依赖外部模型服务，不需要网络。
- 输出 text candidates、image prompt candidates 和 text-image pair ranking。
- 每个阶段都保存可追溯 JSON 产物。
- 为后续接入 LMM、SDXL、CLIP、BLIP、Sentence-BERT、Gemini embedding、LLM/VLM judge 预留 adapter 边界。

## 设计来源

实现对应 `MMKG/` 下的六个阶段计划：

- `MMKG/0 Coarse_Relation_Filtering.md`
- `MMKG/1 knowledge_modeling_stage_design.md`
- `MMKG/2 graph_constrained_reasoning_stage_design.md`
- `MMKG/3 modality_aware_fine_grained_relation_selection_stage_design.md`
- `MMKG/4 relation_alternation_generation_stage_design.md`
- `MMKG/5 kg_aware_candidate_consistency_ranking_stage_design.md`

实现使用已有 MKG-W 数据作为只读输入：

- `KGC/datasets/MKG-W/entity2id.txt`
- `KGC/datasets/MKG-W/relation2id.txt`
- `KGC/datasets/MKG-W/train.txt`
- `KGC/datasets/MKG-W/valid.txt`
- `KGC/datasets/MKG-W/test.txt`
- `KGC/datasets/MKG-W/entity_info.json`
- `KGC/datasets/MKG-W/relation_info.json`
- `KGC/datasets/MKG-W/images/`

## 工程边界

`MMKGC/` 是独立 Python 包，不修改 `KGC/` 的代码或数据。

建议目录结构：

```text
MMKGC/
  mmkgc/
    adapters/
    data/
    io/
    ranking/
    stages/
    run.py
  configs/
  docs/
  outputs/
  tests/
```

`MMKGC` 可以读取 MKG-W 文件格式，但数据加载、六阶段 pipeline、输出格式、测试都由 `MMKGC` 自己维护，避免和现有 `KGC` 项目耦合。

## 任务模式

命令行支持三种模式：

```text
missing_mode=text
missing_mode=image
missing_mode=both
```

### text 模式

只补文本模态。

输入可包含：

- KG 结构
- entity label
- existing image path

输出：

- text candidates
- text ranking

### image 模式

只补图像模态。

第一版不生成真实图像，只生成 image prompt candidates。

输入可包含：

- KG 结构
- entity label
- existing text / description

输出：

- image positive prompt
- image negative prompt
- image prompt ranking

### both 模式

同时补 text 和 image。

这是第一版默认验收模式。

生成输入只使用：

- KG 结构
- entity key
- entity label
- relation labels
- neighbor labels
- neighbor descriptions

`description` 和真实 `image_path` 只保存在 `ground_truth_modalities` 中，用于后续评估或人工检查，不进入生成上下文。

输出：

- 5 个 text candidates
- 5 个 image prompt candidates
- Top-3 text x Top-3 image 组成 9 个 pair candidates
- Top-1 pair
- Top-5 pair ranking breakdown

## 核心数据模型

每个实体会转换成一个 `EntityRecord`。

示例：

```json
{
  "entity_id": 0,
  "entity_key": "Q...",
  "label": "...",
  "description": "...",
  "neighbors": [],
  "existing_modalities": {
    "text": null,
    "image": null
  },
  "ground_truth_modalities": {
    "text": "...",
    "image": "..."
  }
}
```

在 `both` 模式下，`existing_modalities.text` 和 `existing_modalities.image` 固定为 `null`，确保“双模态都缺失”的实验定义干净。

邻居三元组保留方向：

```json
{
  "relation_id": 0,
  "relation": "instance of",
  "neighbor_id": 1,
  "neighbor_key": "Q...",
  "neighbor_label": "film",
  "neighbor_description": "type of creative work",
  "direction": "out"
}
```

## 六阶段 Pipeline

整体流程：

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

### Stage 0: Coarse Relation Filtering

目标：在 Knowledge Modeling 前做高召回关系粗筛，降低噪声和 prompt/context 长度。

输入：

- target entity metadata
- missing mode
- 一跳 directed KG neighbors
- available modality summary，如果当前模式允许使用已有模态

职责：

- 按 relation 和 direction 聚合 triples。
- 删除 ID、URL、website、category、metadata 等低价值关系。
- 每个 relation group 最多保留 3 条 evidence triples。
- 使用 deterministic heuristic 计算 `coarse_score`。
- 默认保留 Top-8 relations，同时保留 `coarse_score >= 3` 的关系。

输出：

```json
{
  "selected_relations": [],
  "discarded_relations": [],
  "filtered_triples": []
}
```

### Stage 1: Knowledge Modeling

目标：把 Stage 0 的 selected relations 和 filtered triples 转换成结构化实体知识模型。

职责：

- 构建 `entity_profile`。
- 构建 `modality_observations`。
- 把 selected relation groups 转换成可追溯 `relation_facts`。
- 生成 `missing_modality_clues`。
- 把证据不足或高风险内容写入 `uncertainties`。

输出：

```json
{
  "entity_profile": {},
  "modality_observations": {},
  "relation_facts": [],
  "missing_modality_clues": [],
  "uncertainties": []
}
```

### Stage 2: Graph-Constrained Reasoning

目标：把 Stage 1 的知识转换成生成约束。

职责：

- 区分 `supported_facts`、`safe_inferences`、`uncertain_facts`。
- 为 text 和 image 分别构建 `modality_constraints`。
- 构建 hard / soft `forbidden_content`。

输出：

```json
{
  "reasoning_summary": "",
  "supported_facts": [],
  "safe_inferences": [],
  "uncertain_facts": [],
  "modality_constraints": {
    "image": {
      "must_include": [],
      "should_include": [],
      "context_only": [],
      "avoid": []
    },
    "text": {
      "must_include": [],
      "should_include": [],
      "context_only": [],
      "avoid": []
    }
  },
  "forbidden_content": [],
  "reasoning_confidence": "high|medium|low"
}
```

### Stage 3: Fine-Grained Relation Selection

目标：在 Stage 2 的安全边界内，选择真正驱动生成的 primary relations。

职责：

- 使用 `MMKG/3` 中的固定公式评分。
- 将 relations 分为 `primary`、`secondary`、`context`、`excluded`。
- 为 primary relations 分配 candidate quota。
- 最多保留 3 个 primary relations。

第一版固定候选预算：

```text
text candidates = 5
image prompt candidates = 5
```

### Stage 4: Relation Alternation Generation

目标：把 fine-grained relation plan 转换成候选模态。

职责：

- 生成 5 个 text candidates。
- 生成 5 个 image prompt candidates。
- 每个 candidate 保存 primary relation、generation angle、evidence triples、constraints、forbidden content。
- relation quota 不足时使用 `global_profile` fallback。

第一版不生成真实图片。Image candidate 保存：

- `positive_prompt`
- `negative_prompt`
- `generated_content.type = "image_prompt"`
- `generated_content.value = positive_prompt`

Text candidate 保存：

- `prompt_or_instruction`
- `generated_content.type = "text"`
- `generated_content.value = generated text`

### Stage 5: KG-Aware Candidate Consistency Ranking

单模态模式：

- 对候选进行 deterministic local scoring。
- 输出 Top-1 和 Top-5 breakdown。

`both` 模式：

- 先 rank 5 个 text candidates。
- 再 rank 5 个 image prompt candidates。
- 取 Top-3 text x Top-3 image 形成 9 个 pairs。
- 对 9 个 pairs 排序。
- 输出 Top-1 pair 和 Top-5 pair breakdown。

第一版使用本地 fallback 指标：

- token cosine 作为文本语义相似度。
- token overlap 作为 relation coverage。
- rule-based constraint satisfaction。
- rule-based forbidden violation penalty。
- text candidate 与 image positive prompt 的 token similarity 作为 pair consistency。

## Adapter 接口

第一版默认 adapter：

```text
TextGenerator: rule/template text generation
ImagePromptGenerator: rule/template positive and negative prompts
ImageGenerator: no-op
TextEmbedder: token vectors
ImageTextScorer: token overlap
KGJudge: rule-based hard gate and penalties
```

后续可替换为：

```text
TextGenerator -> Qwen / OpenAI-compatible LMM
ImagePromptGenerator -> LMM prompt refiner
ImageGenerator -> SDXL
TextEmbedder -> Sentence-BERT or Gemini Embedding
ImageTextScorer -> CLIP, BLIP, or Gemini multimodal embedding
KGJudge -> LLM/VLM judge
```

主 pipeline 只依赖 adapter 接口，不直接依赖外部模型。

## 输出结构

每次运行创建一个 run 目录：

```text
MMKGC/outputs/
  mkgw_both_YYYYMMDD_HHMMSS/
    config.json
    results.jsonl
    metrics_summary.json
    entities/
      Qxxx/
        0_coarse_relation_filtering.json
        1_knowledge_modeling.json
        2_graph_constrained_reasoning.json
        3_fine_grained_relation_selection.json
        4_relation_alternation_generation.json
        5_kg_aware_pair_ranking.json
        final.json
```

`results.jsonl` 每行保存一个实体的 compact final result，方便批量统计。

每个实体目录保存完整阶段产物，方便 debug 和人工检查。

## 验收标准

MVP 满足以下条件即通过：

1. 在 `MMKGC/` 下运行：

   ```bash
   python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 100
   ```

2. 运行不需要网络或外部模型服务。
3. 成功处理 100 个实体时，`results.jsonl` 有 100 行。
4. 每个实体都有 6 个阶段 JSON 和 `final.json`。
5. 每个 `both` 实体有 5 个 text candidates。
6. 每个 `both` 实体有 5 个 image prompt candidates。
7. 每个 `both` 实体有 9 个 text-image pair candidates。
8. 每个 `both` 实体有 Top-1 pair 和 Top-5 pair ranking。
9. `metrics_summary.json` 记录 processed count、failed count、平均候选数、fallback counts、stage failure counts。
10. Pytest 覆盖数据加载、阶段 schema、候选数量、pair ranking、输出文件结构。

## MVP 不包含的内容

第一版不做：

- 调用 Qwen 或任何 LMM。
- 调用 SDXL 或生成真实图片文件。
- 计算 CLIP、BLIP、Sentence-BERT、Gemini、BERTScore 等真实模型指标。
- 全量数据集质量评估。
- 修改 `KGC/`。

## 自检

- 范围清晰：MVP 只做离线六阶段闭环。
- 输入定义清晰：`both` 模式不使用 description 和真实 image 作为生成输入。
- 输出定义清晰：同时保存总 jsonl 和每实体阶段文件。
- 扩展路径清晰：外部模型能力通过 adapter 接入。
