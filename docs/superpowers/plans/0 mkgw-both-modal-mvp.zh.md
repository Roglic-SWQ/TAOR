# MKG-W 双模态补全 MVP 实施计划

> **给 agentic workers：** 执行本计划时必须使用 `superpowers:subagent-driven-development`，推荐每个任务派发一个独立 subagent，任务之间由主 agent review。步骤使用 checkbox 语法便于跟踪。

**目标：** 在 `MMKGC/` 下实现一个离线、可测试、adapter-ready 的六阶段 MKG-W 双模态补全 MVP。

**架构：** 新建独立 Python 包 `mmkgc`，包含数据加载、六阶段实现、adapter、ranking、输出写入和 CLI。默认实现不调用外部模型，但代码结构为后续 Qwen、SDXL、CLIP、BLIP、Sentence-BERT、Gemini、LLM/VLM judge 预留替换点。

**技术栈：** Python 3.11 standard library、dataclasses、argparse、json、pathlib、pytest。

---

## 文件结构

需要创建这些文件：

```text
MMKGC/
  mmkgc/
    __init__.py
    utils.py
    data/
      __init__.py
      mkgw.py
    adapters/
      __init__.py
      base.py
    stages/
      __init__.py
      coarse.py
      modeling.py
      reasoning.py
      selection.py
      generation.py
    ranking/
      __init__.py
      scoring.py
    io/
      __init__.py
      writer.py
    pipeline.py
    run.py
  configs/
    default.json
  tests/
    test_data.py
    test_stages.py
    test_pipeline_outputs.py
  README.md
```

不修改 `MMKGC/` 之外的文件。

英文执行计划中已经给出每个文件的完整代码草案；本中文计划用于阅读、review 和确认任务拆分。

---

## Task 1: 包骨架和基础工具函数

**目标：** 建立 `mmkgc` 包，并实现 deterministic MVP 需要的基础工具函数。

**创建文件：**

- `MMKGC/mmkgc/__init__.py`
- `MMKGC/mmkgc/utils.py`
- `MMKGC/tests/test_stages.py`

**实现内容：**

- `tokenize(text)`
- `clamp01(value)`
- `token_cosine(left, right)`
- `token_overlap(text, required_terms)`
- `slugify(value)`
- `write_json(path, data)`

**测试内容：**

- token overlap 可以识别完整词组覆盖。
- token cosine 对相同 token 集合返回 1.0。
- 完全不相关文本返回 0.0。
- `slugify("country of citizenship")` 返回 `country_of_citizenship`。
- `clamp01` 能把小于 0 和大于 1 的值截断到 `[0, 1]`。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests/test_stages.py -v
```

**期望结果：**

```text
PASS
```

---

## Task 2: MKG-W 数据加载器

**目标：** 读取 MKG-W 文件，构建 `EntityRecord`，并正确处理 `missing_mode=text|image|both` 的输入策略。

**创建文件：**

- `MMKGC/mmkgc/data/__init__.py`
- `MMKGC/mmkgc/data/mkgw.py`
- `MMKGC/tests/test_data.py`

**核心类型：**

```python
@dataclass(frozen=True)
class EntityRecord:
    entity_id: int
    entity_key: str
    label: str
    description: str
    neighbors: List[Dict[str, Any]]
    existing_modalities: Dict[str, Optional[str]]
    ground_truth_modalities: Dict[str, Optional[str]]
```

**实现内容：**

- 读取 `entity2id.txt` 和 `relation2id.txt`。
- 读取 `train.txt`、`valid.txt`、`test.txt`。
- 读取 `entity_info.json` 和 `relation_info.json`。
- 构建 directed adjacency：

  ```text
  head -> (relation, tail, out)
  tail -> (relation, head, in)
  ```

- 根据 entity label 或 Q-ID 查找图片路径。
- `both` 模式下，`existing_modalities = {"text": None, "image": None}`。
- `text` 模式下，existing image 可进入上下文。
- `image` 模式下，existing text/description 可进入上下文。
- ground truth text/image 始终保留在 `ground_truth_modalities`。

**测试内容：**

- fixture 数据能被正确读取。
- `both` 模式不会把 description 和 image 放入 generation context。
- neighbor relation label 和 direction 正确。
- `select_records(..., limit=2)` 返回 2 个实体。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests/test_data.py -v
```

**期望结果：**

```text
PASS
```

---

## Task 3: Stage 0-2 实现

**目标：** 实现 Coarse Relation Filtering、Knowledge Modeling、Graph-Constrained Reasoning 三个阶段。

**创建文件：**

- `MMKGC/mmkgc/stages/__init__.py`
- `MMKGC/mmkgc/stages/coarse.py`
- `MMKGC/mmkgc/stages/modeling.py`
- `MMKGC/mmkgc/stages/reasoning.py`

**修改文件：**

- `MMKGC/tests/test_stages.py`

### Stage 0: Coarse Relation Filtering

实现内容：

- 按 relation 聚合 neighbor triples。
- 删除低价值关系：

  ```text
  id, identifier, url, website, wikimedia, commons,
  category, catalog, archive, freebase, imdb, viaf
  ```

- 每个 relation 最多保留 3 条 evidence triples。
- 使用规则分数生成 `coarse_score`。
- 默认保留 Top-8 和 `coarse_score >= 3` 的关系。

输出字段：

```json
{
  "selected_relations": [],
  "discarded_relations": [],
  "filtered_triples": []
}
```

### Stage 1: Knowledge Modeling

实现内容：

- 从 `instance of` 推断 `entity_type`。
- 构建 `entity_profile`。
- 构建 `modality_observations`。
- 把 selected relations 转成 `relation_facts`。
- 为 text 和 image 生成 `missing_modality_clues`。
- 记录 exact face、logo、dates、awards 等 unsupported details 的风险。

输出字段：

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

实现内容：

- 把 relation facts 转成 `supported_facts`。
- 构建 text/image 的 `must_include`、`should_include`、`context_only`、`avoid`。
- 添加默认 `forbidden_content`。
- 输出 `reasoning_confidence`。

输出字段：

```json
{
  "reasoning_summary": "",
  "supported_facts": [],
  "safe_inferences": [],
  "uncertain_facts": [],
  "modality_constraints": {},
  "forbidden_content": [],
  "reasoning_confidence": "medium"
}
```

**测试内容：**

- `IMDb ID` 被 Stage 0 丢弃。
- `instance of` 被保留。
- `filtered_triples` 包含 `(Example Film, instance of, film)`。
- Stage 1 能推断 `entity_type = film`。
- Stage 2 能生成 text/image must_include constraints 和 forbidden content。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests/test_stages.py -v
```

---

## Task 4: Adapter、Stage 3、Stage 4

**目标：** 建立 adapter 边界，实现细粒度关系选择和候选生成。

**创建文件：**

- `MMKGC/mmkgc/adapters/__init__.py`
- `MMKGC/mmkgc/adapters/base.py`
- `MMKGC/mmkgc/stages/selection.py`
- `MMKGC/mmkgc/stages/generation.py`

**修改文件：**

- `MMKGC/mmkgc/stages/__init__.py`
- `MMKGC/tests/test_stages.py`

### Adapter

第一版实现：

- `RuleTextGenerator`
- `RuleImagePromptGenerator`

接口：

```text
TextGenerator.generate(record_label, entity_type, relation, angle, facts)
ImagePromptGenerator.generate(record_label, entity_type, relation, angle, facts, avoid)
```

### Stage 3: Fine-Grained Relation Selection

实现内容：

- 使用固定权重计算 `fine_score`：

  ```text
  0.30 * modality_fit
  + 0.25 * constraint_alignment
  + 0.20 * evidence_strength
  + 0.20 * generation_utility
  - 0.25 * risk_penalty
  ```

- 将 relations 分为：

  ```text
  primary
  secondary
  context
  excluded
  ```

- primary 默认 quota 为 2。
- secondary 默认不生成候选；当 primary 不足时补足 quota。
- 总候选预算为 5。

### Stage 4: Relation Alternation Generation

实现内容：

- text angles：

  ```text
  identity-centered
  fact-centered
  relation-centered
  disambiguation-centered
  concise-summary
  ```

- image angles：

  ```text
  entity-type-centered
  role-centered
  scene-centered
  style-centered
  attribute-centered
  ```

- 生成 5 个 text candidates。
- 生成 5 个 image prompt candidates。
- 每个 candidate 保存：

  ```text
  candidate_id
  target_modality
  primary_relation
  generation_angle
  used_evidence_triples
  used_constraints
  applied_forbidden_content
  generated_content
  expected_relation_coverage
  metadata
  ```

**测试内容：**

- Stage 3 至少生成一个 primary relation。
- Stage 4 在 `both` 模式下生成 5 个 text candidates。
- Stage 4 在 `both` 模式下生成 5 个 image candidates。
- image candidate 包含 `positive_prompt`。
- text candidate 的 `generated_content.type = text`。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests/test_stages.py -v
```

---

## Task 5: Stage 5 Pair Ranking

**目标：** 实现 KG-aware candidate ranking，包括单候选排序和 text-image pair 排序。

**创建文件：**

- `MMKGC/mmkgc/ranking/__init__.py`
- `MMKGC/mmkgc/ranking/scoring.py`

**修改文件：**

- `MMKGC/tests/test_stages.py`

**实现内容：**

单候选评分：

- `semantic = token_cosine(candidate, KG_reference_text)`
- `relation_coverage = token_overlap(candidate, expected_relation_coverage)`
- `constraint_satisfaction = token_overlap(candidate, must_include)`
- `evidence_support = token_cosine(candidate, evidence_triples)`
- `kg_score` 按固定权重融合。
- `single_quality` 按语义、KG、一致性融合。

pair 评分：

- 对 5 个 text candidates 排序。
- 对 5 个 image prompt candidates 排序。
- 取 Top-3 text x Top-3 image，形成 9 个 pairs。
- pair 分数融合：

  ```text
  text_score
  image_score
  text-image token consistency
  shared primary relation
  entity type consistency
  ```

输出：

```json
{
  "text_ranking": {},
  "image_ranking": {},
  "pair_ranking": {
    "ranked_pairs": [],
    "top_5_pairs": [],
    "best_pair_id": "",
    "best_pair": {}
  }
}
```

**测试内容：**

- text ranking 有 5 个候选。
- image ranking 有 5 个候选。
- pair ranking 有 9 个 pair。
- Top-5 pair 长度为 5。
- `best_pair_id` 等于 Top-5 第一名的 `pair_id`。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests/test_stages.py -v
```

---

## Task 6: Pipeline、输出 Writer、CLI

**目标：** 串起六阶段 pipeline，写出 run 目录和每个实体的阶段产物，并提供命令行入口。

**创建文件：**

- `MMKGC/mmkgc/io/__init__.py`
- `MMKGC/mmkgc/io/writer.py`
- `MMKGC/mmkgc/pipeline.py`
- `MMKGC/mmkgc/run.py`
- `MMKGC/configs/default.json`
- `MMKGC/tests/test_pipeline_outputs.py`

### Pipeline

`MMKGCPipeline.run(record, missing_mode)` 按顺序执行：

```text
coarse_relation_filtering
knowledge_modeling
graph_constrained_reasoning
fine_grained_relation_selection
relation_alternation_generation
kg_aware_candidate_ranking
```

并生成：

```text
coarse_relation_filtering
knowledge_modeling
graph_constrained_reasoning
fine_grained_relation_selection
relation_alternation_generation
kg_aware_candidate_consistency_ranking
final
```

### Writer

`RunWriter` 负责：

- 创建 run 目录。
- 写 `config.json`。
- 写每个实体的 6 阶段 JSON。
- 写每个实体的 `final.json`。
- 追加写 `results.jsonl`。
- 写 `metrics_summary.json`。

输出目录：

```text
outputs/<run_name>/
  config.json
  results.jsonl
  metrics_summary.json
  entities/Q.../
```

### CLI

入口命令：

```bash
python -m mmkgc.run \
  --dataset-dir ../KGC/datasets/MKG-W \
  --missing-mode both \
  --limit 100 \
  --candidate-budget 5
```

参数：

- `--dataset-dir`
- `--missing-mode`
- `--limit`
- `--candidate-budget`
- `--output-root`
- `--run-name`

### 默认配置

`configs/default.json` 保存：

```json
{
  "dataset": {
    "name": "MKG-W",
    "dataset_dir": "../KGC/datasets/MKG-W"
  },
  "run": {
    "missing_mode": "both",
    "limit": 100,
    "candidate_budget": 5
  },
  "adapters": {
    "text_generator": "rule",
    "image_prompt_generator": "rule",
    "image_generator": "noop",
    "text_embedder": "token",
    "image_text_scorer": "token",
    "kg_judge": "rule"
  }
}
```

**测试内容：**

- pipeline 能对 fixture 实体跑完。
- writer 创建 run 目录。
- writer 写出 6 个阶段 JSON 和 `final.json`。
- `results.jsonl` 存在。
- `metrics_summary.json` 存在。
- `final.json` 中候选数量正确：

  ```text
  text_candidates = 5
  image_candidates = 5
  pair_candidates = 9
  ```

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests/test_pipeline_outputs.py -v
```

---

## Task 7: README 和完整验收

**目标：** 补充项目说明，运行测试和真实 MKG-W smoke / acceptance 命令。

**创建文件：**

- `MMKGC/README.md`

README 需要说明：

- MVP scope。
- 三种 missing mode。
- 默认 both 模式。
- 不调用外部模型。
- 运行命令。
- 输出结构。
- 测试命令。

### 完整测试

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m pytest tests -v
```

期望：

```text
全部测试通过
```

### Smoke test

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 3 --run-name smoke_test
```

期望输出：

```json
{
  "run_dir": "outputs/smoke_test",
  "processed_entities": 3,
  "failed_entities": 0
}
```

### Acceptance test

运行：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 100
```

验收：

- 命令退出码为 0。
- `results.jsonl` 有 100 行。
- `metrics_summary.json` 中 `processed_entities = 100`。
- 随机抽样的实体目录包含 6 个阶段 JSON 和 `final.json`。

---

## 执行顺序建议

使用 `superpowers:subagent-driven-development` 执行时，建议按任务顺序分派：

1. Task 1：包骨架和工具函数。
2. Task 2：MKG-W 数据加载。
3. Task 3：Stage 0-2。
4. Task 4：Adapter、Stage 3-4。
5. Task 5：Stage 5 ranking。
6. Task 6：Pipeline、writer、CLI。
7. Task 7：README、测试、smoke、acceptance。

每个任务完成后主 agent 做 review：

- 是否只改 `MMKGC/`。
- 是否有测试。
- 测试是否通过。
- 输出 schema 是否与 spec 对齐。
- 是否破坏前序任务。

---

## 自检

覆盖关系：

- 独立 `MMKGC` 工程：Task 1、2、6、7。
- MKG-W 数据读取：Task 2。
- 六阶段 pipeline：Task 3、4、5、6。
- text 和 image prompt candidates：Task 4。
- pair ranking：Task 5。
- 混合输出结构：Task 6。
- CLI 和 100 实体验收：Task 6、7。
- adapter-ready 边界：Task 4。

实现边界：

- 不调用外部模型。
- 不修改 `KGC/`。
- 不需要网络。
- 默认 `both` 模式只用 KG + label 作为生成输入。

验证边界：

- 单元测试覆盖数据、阶段、ranking、输出。
- smoke test 覆盖真实 MKG-W 的小规模运行。
- acceptance test 覆盖 100 实体运行。
