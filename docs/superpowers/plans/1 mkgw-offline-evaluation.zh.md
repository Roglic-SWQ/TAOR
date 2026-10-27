# MKG-W Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `MMKGC/` 下实现一个独立、离线、无新增模型依赖的 run 评估器，覆盖 text candidates、image prompt candidates 和 text-image pair ranking。

**Architecture:** 新增 `mmkgc.evaluation` 包负责读取 run 输出、计算离线指标、写出 JSON/CSV/case exports；新增 `mmkgc.evaluate` CLI 作为独立入口。现有 `mmkgc.run` 和六阶段 pipeline 不改动。

**Tech Stack:** Python 3.11 standard library、dataclasses、argparse、csv、json、pathlib、pytest、现有 `mmkgc.utils`。

---

## Controller 执行规则

执行本计划时必须遵守用户确认的流程：

1. 主 Codex 会话作为 controller。
2. 每个 plan task 派一个 implementer subagent。
3. 每个 implementer 必须有明确写入范围、验证命令、禁止 commit。
4. 每个任务完成后派两个 reviewer：
   - spec reviewer：检查是否符合 spec 和用户确认方案。
   - quality reviewer：检查代码质量、测试质量、职责边界、路径/模板/打包风险。
5. reviewer 发现问题时，不要跳过，回给 implementer 修。
6. 两个 reviewer 都 approve 后才标记任务完成。
7. 最后派 final reviewer 审完整变更。
8. 最后运行完整验证命令并汇报证据。
9. 工作区可能有用户未提交改动，不要 revert 任何非本任务改动。
10. 不要自动 commit。

## 文件结构

新增文件：

```text
MMKGC/
  mmkgc/
    evaluation/
      __init__.py
      cases.py
      evaluator.py
      loader.py
      metrics.py
      writer.py
    evaluate.py
  tests/
    test_evaluation_loader.py
    test_evaluation_metrics.py
    test_evaluation_writer.py
    test_evaluate_cli.py
```

修改文件：

```text
MMKGC/
  README.md
  README.zh.md
```

不修改：

```text
KGC/
MMKGC/mmkgc/run.py
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/stages/
```

## Task 1: Evaluation package skeleton and run loader

**目标：** 新增评估包骨架和 run/entity 读取逻辑，能安全读取已有 run 输出。

**写入范围：**

- Create: `MMKGC/mmkgc/evaluation/__init__.py`
- Create: `MMKGC/mmkgc/evaluation/loader.py`
- Create: `MMKGC/tests/test_evaluation_loader.py`

**禁止：**

- 不修改 `mmkgc.run`。
- 不修改已有输出目录。
- 不 commit。

**实现要求：**

- 定义 `RunEntityArtifacts` dataclass，字段：
  - `entity_dir: Path`
  - `entity_key: str`
  - `knowledge: dict[str, Any]`
  - `reasoning: dict[str, Any]`
  - `ranking: dict[str, Any]`
  - `final: dict[str, Any]`
- 定义 `validate_run_dir(run_dir: str | Path) -> Path`。
- 定义 `iter_entity_artifacts(run_dir: str | Path) -> Iterator[RunEntityArtifacts]`。
- run 级别必须存在：
  - `results.jsonl`
  - `entities/`
- entity 级别读取：
  - `1_knowledge_modeling.json`
  - `2_graph_constrained_reasoning.json`
  - `5_kg_aware_pair_ranking.json`
  - `final.json`

**测试步骤：**

- [ ] 写 `test_validate_run_dir_rejects_missing_path`
  - 构造不存在路径。
  - 期望 `FileNotFoundError`。

- [ ] 写 `test_validate_run_dir_rejects_missing_required_files`
  - 临时目录只创建 `entities/`。
  - 期望 `ValueError`，消息包含 `results.jsonl`。

- [ ] 写 `test_iter_entity_artifacts_reads_expected_stage_files`
  - 构造最小 run fixture：
    - `results.jsonl`
    - `entities/Q1/1_knowledge_modeling.json`
    - `entities/Q1/2_graph_constrained_reasoning.json`
    - `entities/Q1/5_kg_aware_pair_ranking.json`
    - `entities/Q1/final.json`
  - 期望读取 1 个 `RunEntityArtifacts`。
  - 期望 `entity_key == "Q1"`。

- [ ] 写 `test_iter_entity_artifacts_skips_non_directories`
  - 在 `entities/` 下创建普通文件。
  - 期望不会被当作实体。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_loader.py -v
```

**期望结果：**

```text
PASS
```

## Task 2: Offline metric functions

**目标：** 实现 text、image prompt、pair 三层离线指标函数。

**写入范围：**

- Create: `MMKGC/mmkgc/evaluation/metrics.py`
- Create: `MMKGC/tests/test_evaluation_metrics.py`

**禁止：**

- 不新增第三方依赖。
- 不调用模型。
- 不 commit。

**实现要求：**

- 使用现有 `mmkgc.utils.token_cosine`、`token_overlap`、`clamp01`。
- 定义：
  - `candidate_text(candidate: dict[str, Any]) -> str`
  - `is_fallback_candidate(candidate: dict[str, Any]) -> bool`
  - `score_text_candidate(candidate, ground_truth_text, entity_label, entity_type) -> dict[str, Any]`
  - `score_image_candidate(candidate, entity_label, entity_type) -> dict[str, Any]`
  - `score_pair(pair) -> dict[str, Any]`
  - `average_top(items, key, limit=5) -> float`
- text ground truth 存在时公式：

```text
0.6 * token_cosine(candidate_text, ground_truth_text)
+ 0.2 * token_overlap(candidate_text, [entity_label, entity_type])
+ 0.2 * stage5_final_score
```

- text ground truth 缺失时公式：

```text
0.5 * stage5_final_score
+ 0.3 * relation_coverage_score
+ 0.2 * must_term_score
```

- image prompt 公式：

```text
0.35 * relation_coverage
+ 0.25 * constraint_coverage
+ 0.20 * entity_type_coverage
+ 0.20 * stage5_final_score
```

- pair 公式：

```text
0.4 * stage5_pair_score
+ 0.3 * pair_balance_score
+ 0.2 * pair_similarity_score
+ 0.1 * kg_anchor_score
```

**测试步骤：**

- [ ] 写 `test_score_text_candidate_uses_ground_truth_when_available`
  - candidate text 和 ground truth 有重叠。
  - 期望 score 大于 0。
  - 期望 `ground_truth_available is True`。

- [ ] 写 `test_score_text_candidate_falls_back_without_ground_truth`
  - ground truth 为空。
  - candidate 带 `final_score`、`relation_coverage_score`、`must_term_score`。
  - 期望 score 按 fallback 公式计算。

- [ ] 写 `test_score_image_candidate_counts_relation_and_constraints`
  - candidate prompt 覆盖 `expected_relation_coverage` 和 `used_constraints`。
  - 期望 relation/constraint coverage 为 1。

- [ ] 写 `test_score_pair_uses_balance_and_stage_scores`
  - pair 中 text score 0.8、image score 0.6、pair score 0.7。
  - 期望 balance score 为 0.8。

- [ ] 写 `test_is_fallback_candidate_detects_global_profile_and_metadata`
  - 覆盖 `primary_relation == "global_profile"` 和 `metadata.fallback is True` 两种情况。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_metrics.py -v
```

**期望结果：**

```text
PASS
```

## Task 3: Entity evaluator orchestration

**目标：** 把 loader artifacts 和 metric functions 组合成 entity-level evaluation。

**写入范围：**

- Create: `MMKGC/mmkgc/evaluation/evaluator.py`
- Modify: `MMKGC/tests/test_evaluation_metrics.py`

**禁止：**

- 不写文件输出。
- 不新增 CLI。
- 不 commit。

**实现要求：**

- 定义 `evaluate_entity(artifacts: RunEntityArtifacts) -> dict[str, Any]`。
- 输出包含：
  - `entity`
  - `text_evaluation`
  - `image_evaluation`
  - `pair_evaluation`
  - `fallback_summary`
  - `overall_score`
  - `case_tags`
  - `warnings`
- text candidates 从 `ranking["text_ranking"]["ranked_candidates"]` 读取。
- image candidates 从 `ranking["image_ranking"]["ranked_candidates"]` 读取。
- pair candidates 从 `ranking["pair_ranking"]["ranked_pairs"]` 读取。
- entity metadata 从 `final["entity"]` 读取。
- entity type 从 `knowledge["entity_profile"]["entity_type"]` 读取，缺失则用 `"entity"`。
- 候选为空时不抛异常，相关 best score 为 0，加入 warning。
- `case_tags` 至少支持：
  - `fallback`
  - `low_score`
  - `missing_text_candidates`
  - `missing_image_candidates`
  - `missing_pair_candidates`

**测试步骤：**

- [ ] 写 `test_evaluate_entity_returns_three_layer_scores`
  - 使用最小 artifacts。
  - text/image/pair 都有候选。
  - 期望三层 best score 存在。

- [ ] 写 `test_evaluate_entity_handles_empty_candidates`
  - 三层候选为空。
  - 期望不抛异常。
  - 期望 warnings 包含 missing candidate 信息。

- [ ] 写 `test_evaluate_entity_tags_fallback`
  - image candidate 使用 fallback。
  - 期望 `case_tags` 包含 `fallback`。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_metrics.py -v
```

**期望结果：**

```text
PASS
```

## Task 4: Evaluation writer and case exports

**目标：** 写出 evaluation summary、CSV、per-entity JSON 和轻量 case exports。

**写入范围：**

- Create: `MMKGC/mmkgc/evaluation/writer.py`
- Create: `MMKGC/mmkgc/evaluation/cases.py`
- Create: `MMKGC/tests/test_evaluation_writer.py`

**禁止：**

- 默认不覆盖已有 output_dir。
- 不写入 run_dir 之外的路径，除非调用方显式传入 `output_dir`。
- 不 commit。

**实现要求：**

- 定义 `EvaluationWriter(output_dir, overwrite=False)`。
- 默认 `output_dir` 已存在时报 `FileExistsError`。
- `overwrite=True` 时允许重写评估器产物。
- 定义 `write_evaluation_outputs(output_dir, entity_evaluations, run_dir, overwrite=False, case_limit=20) -> dict[str, Any]`。
- 写出：
  - `evaluation_summary.json`
  - `summary.csv`
  - `entity_scores.csv`
  - `best_cases.jsonl`
  - `worst_cases.jsonl`
  - `fallback_cases.jsonl`
  - `entities/<entity_key>/evaluation.json`
- CSV 使用 Python standard library `csv`。
- summary 计算：
  - `evaluated_entities`
  - `failed_entities`
  - average text/image/pair/overall scores
  - fallback entity count
  - low score count

**测试步骤：**

- [ ] 写 `test_writer_refuses_existing_output_dir_by_default`
  - 创建 output_dir。
  - 期望 `FileExistsError`。

- [ ] 写 `test_writer_outputs_json_csv_and_entity_files`
  - 输入两个 entity evaluations。
  - 期望所有核心文件存在。
  - 期望 `entity_scores.csv` 有 header 和两行数据。

- [ ] 写 `test_case_exports_respect_limit`
  - 输入 5 个 entity evaluations，`case_limit=2`。
  - 期望 best/worst 最多 2 行。

- [ ] 写 `test_fallback_cases_only_include_fallback_tagged_entities`
  - 输入 fallback 和非 fallback entity。
  - 期望 `fallback_cases.jsonl` 只包含 fallback。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_writer.py -v
```

**期望结果：**

```text
PASS
```

## Task 5: Evaluation CLI

**目标：** 新增 `python -m mmkgc.evaluate` 独立入口。

**写入范围：**

- Create: `MMKGC/mmkgc/evaluate.py`
- Create: `MMKGC/tests/test_evaluate_cli.py`

**禁止：**

- 不修改 `MMKGC/mmkgc/run.py`。
- 不修改主 pipeline。
- 不 commit。

**实现要求：**

- CLI 参数：
  - `--run-dir` 必填。
  - `--output-dir` 可选。
  - `--case-limit` 默认 20。
  - `--overwrite` flag。
- 默认 `output_dir = run_dir / "evaluation"`。
- 主流程：
  1. validate run dir。
  2. iter entity artifacts。
  3. evaluate each entity。
  4. write outputs。
  5. print JSON summary：

```json
{
  "evaluation_dir": "...",
  "evaluated_entities": 100,
  "failed_entities": 0
}
```

- 单实体失败时记录 failed entity，继续处理其它实体。
- run 级结构错误时直接返回非 0。

**测试步骤：**

- [ ] 写 `test_cli_requires_run_dir`
  - 调用 `main([])`。
  - 期望 argparse SystemExit。

- [ ] 写 `test_cli_writes_default_evaluation_dir`
  - 构造最小 run fixture。
  - 调用 `main(["--run-dir", str(run_dir)])`。
  - 期望 `run_dir/evaluation/evaluation_summary.json` 存在。

- [ ] 写 `test_cli_rejects_existing_evaluation_dir_without_overwrite`
  - 先创建 `run_dir/evaluation`。
  - 调用 CLI。
  - 期望 `FileExistsError` 或非 0。

- [ ] 写 `test_cli_allows_custom_output_dir`
  - 指定 `--output-dir`。
  - 期望写到指定目录。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluate_cli.py -v
```

**期望结果：**

```text
PASS
```

## Task 6: Documentation updates

**目标：** 在中英文 README 中补充评估器使用方式。

**写入范围：**

- Modify: `MMKGC/README.md`
- Modify: `MMKGC/README.zh.md`

**禁止：**

- 不修改 roadmap/spec/plan 之外的旧文档内容语义。
- 不 commit。

**实现要求：**

英文 README 增加 `Evaluation` section，包含：

```bash
python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002
```

中文 README 增加 `评估` section，包含同样命令和输出目录说明。

说明：

- 评估器独立于主 pipeline。
- 默认输出 `<run_dir>/evaluation/`。
- 支持 `--output-dir`。
- 完全离线、无新增模型依赖。

**验证命令：**

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
grep -n "evaluate" README.md README.zh.md
```

**期望结果：**

```text
README.md:...
README.zh.md:...
```

## Task 7: Full verification and acceptance run

**目标：** 跑全量测试和 100 条 acceptance run 评估，收集最终证据。

**写入范围：**

- 不写代码。
- 可以写入默认 evaluation 输出目录：
  - `MMKGC/outputs/acceptance_pytest_20260706_0002/evaluation/`

**禁止：**

- 不删除用户输出。
- 不 commit。

**验证步骤：**

- [ ] 运行全量 pytest：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
```

期望：全部 PASS。

- [ ] 运行 compileall：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall -q mmkgc tests
```

期望：exit 0。

- [ ] 评估 100 条 acceptance run：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/acceptance_pytest_20260706_0002 \
  --overwrite
```

期望输出包含：

```json
{
  "evaluated_entities": 100,
  "failed_entities": 0
}
```

- [ ] 检查输出文件：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
test -f outputs/acceptance_pytest_20260706_0002/evaluation/evaluation_summary.json
test -f outputs/acceptance_pytest_20260706_0002/evaluation/summary.csv
test -f outputs/acceptance_pytest_20260706_0002/evaluation/entity_scores.csv
test -f outputs/acceptance_pytest_20260706_0002/evaluation/best_cases.jsonl
test -f outputs/acceptance_pytest_20260706_0002/evaluation/worst_cases.jsonl
test -f outputs/acceptance_pytest_20260706_0002/evaluation/fallback_cases.jsonl
```

期望：全部 exit 0。

- [ ] 检查 per-entity evaluation 数量：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
find outputs/acceptance_pytest_20260706_0002/evaluation/entities -name evaluation.json | wc -l
```

期望：`100`。

## Self-review checklist

计划执行前，controller 应确认：

- 每个 task 都有明确写入范围。
- 每个 task 都有验证命令。
- 每个 task 都写明禁止 commit。
- spec 中的独立 CLI、三层评估、JSON+CSV、默认输出目录、`--output-dir`、100 条验收、轻量 case export 都有对应 task。
- 没有要求修改 `mmkgc.run` 或主 pipeline。
- 没有新增模型依赖。
- 没有自动 commit 步骤。
