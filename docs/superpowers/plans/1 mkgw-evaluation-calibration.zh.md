# MKG-W Evaluation Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有离线 evaluation 上增加 calibrated 分数和 penalty 诊断字段，缓解 fallback、entity type 泛化、image/pair 虚高和弱关系证据导致的评分偏差。

**Architecture:** 新增 `mmkgc.evaluation.calibration` 作为独立校准层，保留 `metrics.py` 的 raw offline 分数含义；`evaluator.py` 在生成 entity evaluation 后调用校准层补充 calibrated 字段；`writer.py` 和 `cases.py` 同时输出 raw 与 calibrated，并按 calibrated overall 导出 best/worst case。现有 Stage 0-5、run pipeline、candidate generation 不改动。

**Tech Stack:** Python 3.11 standard library、现有 `mmkgc.utils`、pytest、argparse/csv/json/pathlib。

---

## Controller 执行规则

执行本计划时必须遵守用户确认流程：

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
MMKGC/mmkgc/evaluation/calibration.py
MMKGC/tests/test_evaluation_calibration.py
```

修改文件：

```text
MMKGC/mmkgc/evaluation/evaluator.py
MMKGC/mmkgc/evaluation/writer.py
MMKGC/mmkgc/evaluation/cases.py
MMKGC/tests/test_evaluation_metrics.py
MMKGC/tests/test_evaluation_writer.py
MMKGC/README.md
MMKGC/README.zh.md
```

通过 CLI 重新生成：

```text
MMKGC/outputs/acceptance_pytest_20260706_0002/evaluation/
```

禁止修改：

```text
MMKGC/mmkgc/run.py
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/stages/
MMKG/
KGC/
```

## Task 1: Calibration module and unit tests

**目标：** 新增独立校准模块，能根据 raw 分数、entity metadata、fallback、候选文本和关系名输出 calibrated 分数与 penalty 诊断。

**写入范围：**

- Create: `MMKGC/mmkgc/evaluation/calibration.py`
- Create: `MMKGC/tests/test_evaluation_calibration.py`

**禁止：**

- 不修改 evaluator/writer/cases。
- 不修改 Stage 0-5。
- 不新增第三方依赖。
- 不 commit。

- [ ] **Step 1: 写 relation strength 测试**

在 `MMKGC/tests/test_evaluation_calibration.py` 中创建测试：

```python
from __future__ import annotations

import pytest

from mmkgc.evaluation.calibration import (
    calibrate_entity_scores,
    relation_support_strength,
)


def test_relation_support_strength_classifies_known_relations() -> None:
    assert relation_support_strength({"primary_relation": "genre"}) == "direct_identity"
    assert relation_support_strength({"primary_relation": "performer"}) == "direct_identity"
    assert relation_support_strength({"primary_relation": "instance of"}) == "type_support"
    assert relation_support_strength({"primary_relation": "cast member"}) == "neighbor_indirect"
    assert relation_support_strength({"primary_relation": "original broadcaster"}) == "neighbor_indirect"
    assert relation_support_strength({"primary_relation": "global_profile"}) == "fallback"
    assert relation_support_strength({"metadata": {"fallback": True}}) == "fallback"
    assert relation_support_strength({"primary_relation": "unknown relation"}) == "unknown"
```

- [ ] **Step 2: 写 fallback penalty 测试**

追加测试：

```python
def test_calibrate_entity_scores_penalizes_fallback_and_generic_entity_type() -> None:
    result = calibrate_entity_scores(
        entity={"entity_key": "Q1", "label": "Fallback", "entity_type": "entity"},
        text_evaluation={
            "best_offline_evaluation_score": 0.36,
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "offline_evaluation_score": 0.36,
                    "is_fallback": True,
                    "generated_content": {"value": "Fallback is described as a entity."},
                    "primary_relation": "global_profile",
                }
            ],
        },
        image_evaluation={
            "best_offline_evaluation_score": 0.96,
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "offline_evaluation_score": 0.96,
                    "is_fallback": True,
                    "positive_prompt": "Fallback, entity, identity, profile context",
                    "primary_relation": "global_profile",
                }
            ],
        },
        pair_evaluation={
            "best_offline_evaluation_score": 0.90,
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "offline_evaluation_score": 0.90,
                    "text_candidate": {"metadata": {"fallback": True}},
                    "image_candidate": {"metadata": {"fallback": True}},
                }
            ],
        },
        fallback_summary={"has_fallback": True},
    )

    assert result["calibrated_text_best_offline_evaluation_score"] < 0.36
    assert result["calibrated_image_best_offline_evaluation_score"] < 0.96
    assert result["calibrated_pair_best_offline_evaluation_score"] <= 0.65
    assert result["calibrated_overall_offline_evaluation_score"] < 0.70
    assert result["calibration_summary"]["fallback_penalty"] > 0
    assert result["calibration_summary"]["entity_type_generic_penalty"] > 0
    assert "fallback_penalized" in result["calibration_summary"]["calibration_tags"]
    assert "generic_entity_type" in result["calibration_summary"]["calibration_tags"]
```

- [ ] **Step 3: 写 pair 单侧惩罚测试**

追加测试：

```python
def test_calibrate_entity_scores_caps_pair_when_text_is_weak() -> None:
    result = calibrate_entity_scores(
        entity={"entity_key": "Q2", "label": "Weak Text", "entity_type": "person"},
        text_evaluation={
            "best_offline_evaluation_score": 0.34,
            "ranked_candidates": [
                {
                    "candidate_id": "text_001",
                    "offline_evaluation_score": 0.34,
                    "generated_content": {"value": "Weak Text appears in one film."},
                    "primary_relation": "cast member",
                }
            ],
        },
        image_evaluation={
            "best_offline_evaluation_score": 0.72,
            "ranked_candidates": [
                {
                    "candidate_id": "image_001",
                    "offline_evaluation_score": 0.72,
                    "positive_prompt": "Weak Text, person, cast member context",
                    "primary_relation": "cast member",
                }
            ],
        },
        pair_evaluation={
            "best_offline_evaluation_score": 0.89,
            "ranked_pairs": [
                {
                    "pair_id": "pair_001",
                    "offline_evaluation_score": 0.89,
                    "text_candidate": {"primary_relation": "cast member"},
                    "image_candidate": {"primary_relation": "cast member"},
                }
            ],
        },
        fallback_summary={"has_fallback": False},
    )

    assert result["calibrated_pair_best_offline_evaluation_score"] <= 0.65
    assert result["calibration_summary"]["single_side_pair_penalty"] > 0
    assert "pair_single_side_penalized" in result["calibration_summary"]["calibration_tags"]
```

- [ ] **Step 4: 实现 `calibration.py`**

实现以下接口：

```python
from __future__ import annotations

from typing import Any

from mmkgc.evaluation.metrics import is_fallback_candidate
from mmkgc.utils import clamp01


DIRECT_IDENTITY_RELATIONS = {"genre", "performer", "occupation"}
TYPE_SUPPORT_RELATIONS = {"instance of"}
NEIGHBOR_INDIRECT_RELATIONS = {
    "cast member",
    "original broadcaster",
    "member of sports team",
}


def relation_support_strength(candidate: dict[str, Any]) -> str:
    if is_fallback_candidate(candidate):
        return "fallback"
    relation = str(candidate.get("primary_relation") or "").strip().lower()
    if relation in DIRECT_IDENTITY_RELATIONS:
        return "direct_identity"
    if relation in TYPE_SUPPORT_RELATIONS:
        return "type_support"
    if relation in NEIGHBOR_INDIRECT_RELATIONS:
        return "neighbor_indirect"
    return "unknown"


def calibrate_entity_scores(
    *,
    entity: dict[str, Any],
    text_evaluation: dict[str, Any],
    image_evaluation: dict[str, Any],
    pair_evaluation: dict[str, Any],
    fallback_summary: dict[str, Any],
) -> dict[str, Any]:
    text_raw = _score(text_evaluation.get("best_offline_evaluation_score"))
    image_raw = _score(image_evaluation.get("best_offline_evaluation_score"))
    pair_raw = _score(pair_evaluation.get("best_offline_evaluation_score"))

    best_text = _first_dict(text_evaluation.get("ranked_candidates"))
    best_image = _first_dict(image_evaluation.get("ranked_candidates"))
    best_pair = _first_dict(pair_evaluation.get("ranked_pairs"))

    text_strength = relation_support_strength(best_text)
    image_strength = relation_support_strength(best_image)
    pair_strength = _pair_relation_strength(best_pair, text_strength, image_strength)

    tags: list[str] = []
    text = text_raw
    image = image_raw
    pair = pair_raw

    fallback_penalty = 0.0
    entity_type_penalty = 0.0
    template_penalty = 0.0
    relation_penalty = 0.0
    single_side_penalty = 0.0

    if bool(fallback_summary.get("has_fallback")):
        before = text + image + pair
        text *= 0.75
        image *= 0.75
        pair *= 0.70
        fallback_penalty += max(0.0, before - (text + image + pair)) / 3
        tags.append("fallback_penalized")

    if _is_generic_entity_type(entity):
        before = text + image
        text *= 0.85
        image *= 0.90
        entity_type_penalty += max(0.0, before - (text + image)) / 2
        tags.append("generic_entity_type")

    if _generic_entity_text(best_text):
        before = text
        text *= 0.80
        entity_type_penalty += max(0.0, before - text)
        if "generic_entity_text" not in tags:
            tags.append("generic_entity_text")

    relation_factor = min(
        _relation_strength_factor(text_strength),
        _relation_strength_factor(image_strength),
    )
    if relation_factor < 1.0:
        before = text + image
        text *= relation_factor
        image *= relation_factor
        relation_penalty += max(0.0, before - (text + image)) / 2
        tags.append(f"relation_support_{_weakest_strength(text_strength, image_strength)}")

    if _template_prompt_detected(best_image, entity, image_strength):
        before = image
        image *= 0.85
        template_penalty += max(0.0, before - image)
        tags.append("template_prompt_penalized")

    before_pair = pair
    if text < 0.40:
        pair = min(pair, 0.65)
    elif text < 0.50:
        pair = min(pair, 0.75)
    if image < 0.60:
        pair = min(pair, 0.80)
    single_side_penalty = max(0.0, before_pair - pair)
    if single_side_penalty > 0:
        tags.append("pair_single_side_penalized")

    text = _score(text)
    image = _score(image)
    pair = _score(pair)
    overall_base = (text + image + pair) / 3 if any((text_raw, image_raw, pair_raw)) else 0.0
    overall_factor = 0.90 if bool(fallback_summary.get("has_fallback")) else 1.0
    overall = _score(overall_base * overall_factor)
    if overall_factor < 1.0:
        fallback_penalty += max(0.0, overall_base - overall)

    return {
        "calibrated_text_best_offline_evaluation_score": text,
        "calibrated_image_best_offline_evaluation_score": image,
        "calibrated_pair_best_offline_evaluation_score": pair,
        "calibrated_overall_offline_evaluation_score": overall,
        "calibration_summary": {
            "fallback_penalty": _score(fallback_penalty),
            "entity_type_generic_penalty": _score(entity_type_penalty),
            "template_prompt_penalty": _score(template_penalty),
            "single_side_pair_penalty": _score(single_side_penalty),
            "relation_support_penalty": _score(relation_penalty),
            "relation_support_strength": {
                "text": text_strength,
                "image": image_strength,
                "pair": pair_strength,
            },
            "calibration_tags": sorted(set(tags)),
        },
    }
```

Complete helper implementations in the same file:

```python
def _pair_relation_strength(
    pair: dict[str, Any],
    text_strength: str,
    image_strength: str,
) -> str:
    nested_text_strength = relation_support_strength(_dict_value(pair.get("text_candidate")))
    nested_image_strength = relation_support_strength(_dict_value(pair.get("image_candidate")))
    if nested_text_strength == "unknown" and nested_image_strength == "unknown":
        return _weakest_strength(text_strength, image_strength)
    strengths = [nested_text_strength, nested_image_strength]
    for strength in ("fallback", "neighbor_indirect", "unknown", "type_support", "direct_identity"):
        if strength in strengths:
            return strength
    return "unknown"


def _weakest_strength(*strengths: str) -> str:
    priority = {
        "fallback": 0,
        "neighbor_indirect": 1,
        "unknown": 2,
        "type_support": 3,
        "direct_identity": 4,
    }
    return min(strengths, key=lambda strength: priority.get(strength, 2))


def _relation_strength_factor(strength: str) -> float:
    return {
        "direct_identity": 1.0,
        "type_support": 0.95,
        "neighbor_indirect": 0.90,
        "fallback": 0.75,
        "unknown": 0.95,
    }.get(strength, 0.95)


def _template_prompt_detected(
    image_candidate: dict[str, Any],
    entity: dict[str, Any],
    relation_strength: str,
) -> bool:
    prompt = str(image_candidate.get("positive_prompt") or "").lower()
    if not prompt:
        return False
    if relation_strength == "fallback":
        return True
    if _is_generic_entity_type(entity) and (
        "identity" in prompt or "profile context" in prompt or " entity," in prompt
    ):
        return True
    generic_terms = ["identity", "profile context", "grounded details"]
    return relation_strength != "direct_identity" and sum(term in prompt for term in generic_terms) >= 2


def _generic_entity_text(text_candidate: dict[str, Any]) -> bool:
    text = ""
    content = text_candidate.get("generated_content")
    if isinstance(content, dict):
        text = str(content.get("value") or "")
    text = f"{text} {text_candidate.get('prompt_or_instruction') or ''}".lower()
    return "as a entity" in text or "entity type entity" in text


def _is_generic_entity_type(entity: dict[str, Any]) -> bool:
    return str(entity.get("entity_type") or "").strip().lower() == "entity"


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _score(value: Any) -> float:
    try:
        return round(clamp01(float(value)), 6)
    except (TypeError, ValueError):
        return 0.0
```

- [ ] **Step 5: 运行 Task 1 测试**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_calibration.py -v
```

Expected:

```text
3 passed
```

## Task 2: Integrate calibration into evaluator

**目标：** 在 entity-level evaluation 中补充 calibrated 字段和 `calibration_summary`，不删除 raw 字段。

**写入范围：**

- Modify: `MMKGC/mmkgc/evaluation/evaluator.py`
- Modify: `MMKGC/tests/test_evaluation_metrics.py`

**禁止：**

- 不修改 writer/cases。
- 不修改 Stage 0-5。
- 不 commit。

- [ ] **Step 1: 写 evaluator 校准测试**

在 `tests/test_evaluation_metrics.py` 中新增：

```python
def test_evaluate_entity_adds_calibrated_scores_for_fallback(tmp_path) -> None:
    artifacts = RunEntityArtifacts(
        entity_dir=tmp_path / "Q4",
        entity_key="Q4",
        knowledge={"entity_profile": {"entity_type": "entity"}},
        reasoning={},
        ranking={
            "text_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "text-fallback",
                        "generated_content": {
                            "value": "Fallback Entity is described as a entity; "
                            "Fallback Entity has entity type entity."
                        },
                        "metadata": {"fallback": True},
                        "primary_relation": "global_profile",
                        "final_score": 0.9,
                    }
                ]
            },
            "image_ranking": {
                "ranked_candidates": [
                    {
                        "candidate_id": "image-fallback",
                        "positive_prompt": "Fallback Entity, entity, identity, profile context",
                        "metadata": {"fallback": True},
                        "primary_relation": "global_profile",
                        "final_score": 0.95,
                    }
                ]
            },
            "pair_ranking": {
                "ranked_pairs": [
                    {
                        "pair_id": "pair-fallback",
                        "score": 0.9,
                        "text_candidate": {"metadata": {"fallback": True}},
                        "image_candidate": {"metadata": {"fallback": True}},
                    }
                ]
            },
        },
        final={
            "entity": {
                "entity_key": "Q4",
                "label": "Fallback Entity",
                "description": "specific company description",
            }
        },
    )

    result = evaluate_entity(artifacts)

    assert result["overall_offline_evaluation_score"] > result[
        "calibrated_overall_offline_evaluation_score"
    ]
    assert result["calibration_summary"]["fallback_penalty"] > 0
    assert "fallback_penalized" in result["calibration_summary"]["calibration_tags"]
```

- [ ] **Step 2: 修改 evaluator 调用校准层**

在 `MMKGC/mmkgc/evaluation/evaluator.py` 中：

```python
from mmkgc.evaluation.calibration import calibrate_entity_scores
```

先扩展 `_candidate_summary()`，让 text/image candidate summary 保留校准需要的轻量源字段：

```python
    for source_key in (
        "primary_relation",
        "relation_role",
        "metadata",
        "expected_relation_coverage_detail",
        "generated_content",
        "prompt_or_instruction",
        "positive_prompt",
        "text_candidate",
        "image_candidate",
    ):
        if source_key in candidate:
            summary[source_key] = candidate[source_key]
```

这些字段只进入 evaluation 输出，不修改 Stage 0-5 原始 JSON。
`text_candidate` 和 `image_candidate` 主要用于 pair summary；如果普通 text/image candidate 没有这些字段，不需要额外构造。

在计算 `overall_offline_evaluation_score` 后调用：

```python
    calibration = calibrate_entity_scores(
        entity=entity,
        text_evaluation=text_evaluation,
        image_evaluation=image_evaluation,
        pair_evaluation=pair_evaluation,
        fallback_summary=fallback_summary,
    )
```

返回 dict 中加入：

```python
        "calibrated_text_best_offline_evaluation_score": calibration[
            "calibrated_text_best_offline_evaluation_score"
        ],
        "calibrated_image_best_offline_evaluation_score": calibration[
            "calibrated_image_best_offline_evaluation_score"
        ],
        "calibrated_pair_best_offline_evaluation_score": calibration[
            "calibrated_pair_best_offline_evaluation_score"
        ],
        "calibrated_overall_offline_evaluation_score": calibration[
            "calibrated_overall_offline_evaluation_score"
        ],
        "calibration_summary": calibration["calibration_summary"],
```

并把 calibration tags 合并进 `case_tags`：

```python
    case_tags.update(calibration["calibration_summary"]["calibration_tags"])
```

`case_tags.update(...)` 必须发生在最终返回前，确保输出排序后包含新增 tags。

- [ ] **Step 3: 更新已有 evaluator 测试断言 raw 和 calibrated 共存**

在 `test_evaluate_entity_returns_three_layer_scores` 中增加：

```python
    assert "overall_offline_evaluation_score" in result
    assert "calibrated_overall_offline_evaluation_score" in result
    assert "calibration_summary" in result
```

- [ ] **Step 4: 运行 Task 2 测试**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_calibration.py tests/test_evaluation_metrics.py -v
```

Expected:

```text
PASS
```

## Task 3: Update writer, summary, CSV, and case sorting

**目标：** 输出 raw + calibrated 字段，并按 calibrated overall 导出 best/worst case。

**写入范围：**

- Modify: `MMKGC/mmkgc/evaluation/writer.py`
- Modify: `MMKGC/mmkgc/evaluation/cases.py`
- Modify: `MMKGC/tests/test_evaluation_writer.py`

**禁止：**

- 不修改 CLI 参数。
- 不修改 evaluation loader。
- 不 commit。

- [ ] **Step 1: 写 case sorting 测试**

在 `tests/test_evaluation_writer.py` 中新增或修改 case export 测试：

```python
def test_case_exports_sort_by_calibrated_overall(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    evaluations = [
        _entity_evaluation(
            "Q_raw_high",
            overall_offline_evaluation_score=0.9,
            calibrated_overall_offline_evaluation_score=0.4,
        ),
        _entity_evaluation(
            "Q_calibrated_high",
            overall_offline_evaluation_score=0.7,
            calibrated_overall_offline_evaluation_score=0.8,
        ),
    ]

    write_evaluation_outputs(output_dir, evaluations, tmp_path / "run")

    first_best = json.loads(
        (output_dir / "best_cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    first_worst = json.loads(
        (output_dir / "worst_cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_best["entity_key"] == "Q_calibrated_high"
    assert first_worst["entity_key"] == "Q_raw_high"
```

- [ ] **Step 2: 更新 `_entity_evaluation` test helper**

确保 helper 产生 raw 和 calibrated 字段：

```python
def _entity_evaluation(
    entity_key: str,
    *,
    overall_offline_evaluation_score: float = 0.5,
    calibrated_overall_offline_evaluation_score: float | None = None,
    text_score: float = 0.5,
    image_score: float = 0.5,
    pair_score: float = 0.5,
    case_tags: list[str] | None = None,
) -> dict:
    calibrated = (
        overall_offline_evaluation_score
        if calibrated_overall_offline_evaluation_score is None
        else calibrated_overall_offline_evaluation_score
    )
    return {
        "entity": {"entity_key": entity_key, "label": entity_key, "missing_mode": "both"},
        "text_evaluation": {
            "candidate_count": 1,
            "best_candidate_id": "text-1",
            "best_offline_evaluation_score": text_score,
            "top5_avg_offline_evaluation_score": text_score,
        },
        "image_evaluation": {
            "candidate_count": 1,
            "best_candidate_id": "image-1",
            "best_offline_evaluation_score": image_score,
            "top5_avg_offline_evaluation_score": image_score,
        },
        "pair_evaluation": {
            "candidate_count": 1,
            "best_pair_id": "pair-1",
            "best_offline_evaluation_score": pair_score,
            "top5_avg_offline_evaluation_score": pair_score,
        },
        "overall_offline_evaluation_score": overall_offline_evaluation_score,
        "calibrated_text_best_offline_evaluation_score": text_score,
        "calibrated_image_best_offline_evaluation_score": image_score,
        "calibrated_pair_best_offline_evaluation_score": pair_score,
        "calibrated_overall_offline_evaluation_score": calibrated,
        "calibration_summary": {
            "fallback_penalty": 0.0,
            "entity_type_generic_penalty": 0.0,
            "template_prompt_penalty": 0.0,
            "single_side_pair_penalty": 0.0,
            "relation_support_penalty": 0.0,
            "relation_support_strength": {
                "text": "direct_identity",
                "image": "direct_identity",
                "pair": "direct_identity",
            },
            "calibration_tags": [],
        },
        "fallback_summary": {"fallback_candidate_count": 0, "has_fallback": False},
        "case_tags": case_tags or [],
        "warnings": [],
    }
```

- [ ] **Step 3: 修改 `cases.py` 按 calibrated overall 排序**

在 `MMKGC/mmkgc/evaluation/cases.py` 中：

```python
def _overall_score(evaluation: dict[str, Any]) -> float:
    return _score(
        evaluation.get(
            "calibrated_overall_offline_evaluation_score",
            evaluation.get("overall_offline_evaluation_score"),
        )
    )
```

在 `case_record()` 中加入 calibrated 字段和 calibration tags：

```python
        "calibrated_overall_offline_evaluation_score": _score(
            evaluation.get("calibrated_overall_offline_evaluation_score")
        ),
        "calibrated_text_best_offline_evaluation_score": _score(
            evaluation.get("calibrated_text_best_offline_evaluation_score")
        ),
        "calibrated_image_best_offline_evaluation_score": _score(
            evaluation.get("calibrated_image_best_offline_evaluation_score")
        ),
        "calibrated_pair_best_offline_evaluation_score": _score(
            evaluation.get("calibrated_pair_best_offline_evaluation_score")
        ),
        "calibration_tags": _string_list(
            _dict_value(evaluation.get("calibration_summary")).get("calibration_tags")
        ),
```

- [ ] **Step 4: 修改 writer CSV/summary 字段**

在 `ENTITY_SCORE_FIELDS` 中追加：

```python
    "calibrated_text_best_offline_evaluation_score",
    "calibrated_image_best_offline_evaluation_score",
    "calibrated_pair_best_offline_evaluation_score",
    "calibrated_overall_offline_evaluation_score",
    "calibration_tags",
```

在 `_entity_score_row()` 中追加对应值：

```python
        "calibrated_text_best_offline_evaluation_score": _score(
            evaluation.get("calibrated_text_best_offline_evaluation_score")
        ),
        "calibrated_image_best_offline_evaluation_score": _score(
            evaluation.get("calibrated_image_best_offline_evaluation_score")
        ),
        "calibrated_pair_best_offline_evaluation_score": _score(
            evaluation.get("calibrated_pair_best_offline_evaluation_score")
        ),
        "calibrated_overall_offline_evaluation_score": _score(
            evaluation.get("calibrated_overall_offline_evaluation_score")
        ),
        "calibration_tags": "|".join(
            _string_list(
                _dict_value(evaluation.get("calibration_summary")).get("calibration_tags")
            )
        ),
```

在 `_build_summary()` 的 `averages` 中加入 calibrated averages：

```python
            "calibrated_text_best_offline_evaluation_score": _average(
                _score(evaluation.get("calibrated_text_best_offline_evaluation_score"))
                for evaluation in entity_evaluations
            ),
            "calibrated_image_best_offline_evaluation_score": _average(
                _score(evaluation.get("calibrated_image_best_offline_evaluation_score"))
                for evaluation in entity_evaluations
            ),
            "calibrated_pair_best_offline_evaluation_score": _average(
                _score(evaluation.get("calibrated_pair_best_offline_evaluation_score"))
                for evaluation in entity_evaluations
            ),
            "calibrated_overall_offline_evaluation_score": _average(
                _score(evaluation.get("calibrated_overall_offline_evaluation_score"))
                for evaluation in entity_evaluations
            ),
```

在 `counts` 中加入 calibration tag counts，使用 helper：

```python
            "fallback_penalized_entities": _calibration_tag_count(
                entity_evaluations, "fallback_penalized"
            ),
            "generic_entity_type_entities": _calibration_tag_count(
                entity_evaluations, "generic_entity_type"
            ),
            "pair_single_side_penalized_entities": _calibration_tag_count(
                entity_evaluations, "pair_single_side_penalized"
            ),
            "template_prompt_penalized_entities": _calibration_tag_count(
                entity_evaluations, "template_prompt_penalized"
            ),
```

Add helper:

```python
def _calibration_tag_count(entity_evaluations: list[dict[str, Any]], tag: str) -> int:
    return sum(
        1
        for evaluation in entity_evaluations
        if tag
        in _string_list(
            _dict_value(evaluation.get("calibration_summary")).get("calibration_tags")
        )
    )
```

- [ ] **Step 5: 运行 writer tests**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_evaluation_writer.py -v
```

Expected:

```text
PASS
```

## Task 4: Acceptance rerun and calibration validation report

**目标：** 重新生成 100 实体 evaluation 输出，并写出校准前后对比证据。

**写入范围：**

- Modify generated output: `MMKGC/outputs/acceptance_pytest_20260706_0002/evaluation/`
- Create: `MMKGC/docs/evaluation/2026-07-07-calibration-validation.zh.md`

**禁止：**

- 不修改代码文件。
- 不修改 Stage 0-5 输出。
- 不 commit。

- [ ] **Step 1: 运行完整测试**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
```

Expected:

```text
PASS
```

- [ ] **Step 2: 运行 compileall**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
```

Expected:

```text
exit code 0
```

- [ ] **Step 3: 重新生成 acceptance evaluation**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002 --overwrite
```

Expected stdout:

```json
{"evaluation_dir": "outputs/acceptance_pytest_20260706_0002/evaluation", "evaluated_entities": 100, "failed_entities": 0, "failed_entity_keys": []}
```

- [ ] **Step 4: 运行校准验收检查**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import csv, json, pathlib
root = pathlib.Path("outputs/acceptance_pytest_20260706_0002/evaluation")
rows = list(csv.DictReader(open(root / "entity_scores.csv", encoding="utf-8")))
fallback = [r for r in rows if "fallback" in r["case_tags"] or "fallback_penalized" in r.get("calibration_tags", "")]
raw_avg = sum(float(r["overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
cal_avg = sum(float(r["calibrated_overall_offline_evaluation_score"]) for r in fallback) / len(fallback)
best_keys = [json.loads(line)["entity_key"] for line in open(root / "best_cases.jsonl", encoding="utf-8")]
film_weak = {"Q2665674", "Q20814652", "Q17479333", "Q3857815", "Q3712040"}
print("rows", len(rows))
print("fallback_count", len(fallback))
print("fallback_raw_avg", round(raw_avg, 6))
print("fallback_calibrated_avg", round(cal_avg, 6))
print("fallback_drop", round(raw_avg - cal_avg, 6))
print("fallback_all_above_070", all(float(r["calibrated_overall_offline_evaluation_score"]) > 0.70 for r in fallback))
print("film_weak_in_best_top10", len(film_weak.intersection(best_keys[:10])))
'
```

Expected:

```text
rows 100
fallback_count 5
fallback_drop >= 0.12
fallback_all_above_070 False
film_weak_in_best_top10 <= 2
```

- [ ] **Step 5: 写校准验证报告**

Create `MMKGC/docs/evaluation/2026-07-07-calibration-validation.zh.md` with:

```markdown
# MMKGC Evaluation 校准验证报告

## 验证对象

- run: `outputs/acceptance_pytest_20260706_0002`
- evaluated_entities: 100
- failed_entities: 0

## 校准前后变化

这一节必须使用 Step 4 命令输出的真实数值，至少包含：

- fallback raw overall 平均值
- fallback calibrated overall 平均值
- fallback 平均降幅

## Best Top 10 变化

- 原三 Agent 指出的 weak film 样本集合：`Q2665674`, `Q20814652`, `Q17479333`, `Q3857815`, `Q3712040`
- calibrated best top 10 中出现数量：使用 Step 4 命令输出的真实数量

## 结论

- fallback calibrated overall 是否明显下降：是/否
- weak film 样本是否不再全部挤在 best 前列：是/否
- 是否满足本轮成功标准：是/否
```

不要保留尖括号占位符。写入真实数值。

## Task 5: Documentation and final verification

**目标：** 更新用户面文档说明 calibrated 分数含义，并做最终完整验证。

**写入范围：**

- Modify: `MMKGC/README.md`
- Modify: `MMKGC/README.zh.md`

**禁止：**

- 不新增执行入口。
- 不修改代码逻辑。
- 不 commit。

- [ ] **Step 1: 更新 README Evaluation 小节**

英文 README 增加说明：

```markdown
The evaluator preserves raw `*_offline_evaluation_score` fields and also writes
`calibrated_*_offline_evaluation_score` fields. Calibrated scores apply offline
penalties for fallback candidates, generic entity types, template-heavy image prompts,
weak relation support, and text/image single-side weaknesses in pair scoring.
Best and worst case exports are sorted by calibrated overall score.
```

中文 README 增加说明：

```markdown
评估器会同时保留原始 `*_offline_evaluation_score` 字段，并新增
`calibrated_*_offline_evaluation_score` 字段。校准分会对 fallback、通用
`entity` 类型、模板化 image prompt、弱关系证据，以及 pair 中 text/image 单侧过弱
等情况做离线惩罚。`best_cases.jsonl` 和 `worst_cases.jsonl` 按校准后的 overall 分排序。
```

- [ ] **Step 2: 运行完整验证**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002 --overwrite
```

Expected:

```text
pytest: PASS
compileall: exit code 0
evaluate: evaluated_entities 100, failed_entities 0
```

- [ ] **Step 3: 最终字段检查**

Run:

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
rg -n "calibrated_overall_offline_evaluation_score|calibration_summary|fallback_penalty|entity_type_generic_penalty|single_side_pair_penalty" \
  mmkgc tests README.md README.zh.md outputs/acceptance_pytest_20260706_0002/evaluation
```

Expected:

```text
matches in code, tests, docs, and generated evaluation outputs
```

## Final reviewer prompt requirements

最终 final reviewer 必须检查：

- 是否只改 evaluation 层和文档，没有改 Stage 0-5。
- raw 分数是否保留。
- calibrated 分数和 penalty 字段是否写入 entity JSON、CSV、case JSONL、summary。
- best/worst 是否按 calibrated overall 排序。
- fallback 样本 calibrated overall 是否明显下降。
- 原 best 后半段 weak film 样本是否不再全部挤在 best top 10。
- 测试是否覆盖 fallback、generic entity type、pair 单侧惩罚、case sorting。
- 是否无新增依赖、无联网、无 commit。
