# MMKGC Evaluation 校准验证报告

## 验证对象

- Run path: `outputs/acceptance_pytest_20260706_0002`
- Evaluation path: `outputs/acceptance_pytest_20260706_0002/evaluation`
- `evaluated_entities`: 100
- `failed_entities`: 0
- `failed_entity_keys`: `[]`

## 校准前后变化

Fallback 样本定义为 `case_tags` 包含 `fallback`，或 `calibration_tags` 包含 `fallback_penalized` 的实体。

- `rows`: 100
- `fallback_count`: 5
- Fallback raw overall average: 0.758801
- Fallback calibrated overall average: 0.361703
- Fallback drop: 0.397098
- `fallback_all_above_070`: `False`

结果显示，校准后 fallback 样本的平均 overall 分数从 0.758801 降至 0.361703，下降 0.397098，超过验收要求的 0.12。

## Best Top 10 变化

弱电影实体集合：

```text
Q2665674, Q20814652, Q17479333, Q3857815, Q3712040
```

- Calibrated best top 10 中弱电影实体数量: 2
- 验收阈值: `<= 2`

校准后的 best top 10 中只有 2 个弱电影实体，弱电影集合不再整体占据 best top list。

## 结论

本次校准验证满足验收标准：

- Fallback 样本分数出现明确下降，`fallback_drop = 0.397098 >= 0.12`。
- Fallback 样本并非全部高于 0.70，`fallback_all_above_070 = False`。
- 弱电影实体不再全部进入 best top list，`film_weak_in_best_top10 = 2`，满足 `<= 2`。
- Acceptance evaluation 重新生成成功，`evaluated_entities = 100`，`failed_entities = 0`。

## 验证命令 Evidence

### Pytest

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v
```

结果摘要：

```text
collected 94 items
94 passed in 0.33s
exit code 0
```

### Compileall

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
```

结果摘要：

```text
Listing 'mmkgc'...
Listing 'mmkgc/adapters'...
Listing 'mmkgc/data'...
Listing 'mmkgc/evaluation'...
Listing 'mmkgc/io'...
Listing 'mmkgc/ranking'...
Listing 'mmkgc/stages'...
Listing 'tests'...
Compiling 'tests/test_evaluation_calibration.py'...
Compiling 'tests/test_evaluation_metrics.py'...
Compiling 'tests/test_evaluation_writer.py'...
exit code 0
```

### Evaluate

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/acceptance_pytest_20260706_0002 --overwrite
```

输出：

```json
{"evaluation_dir": "outputs/acceptance_pytest_20260706_0002/evaluation", "evaluated_entities": 100, "failed_entities": 0, "failed_entity_keys": []}
```

### Calibration Acceptance Check

```bash
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

输出：

```text
rows 100
fallback_count 5
fallback_raw_avg 0.758801
fallback_calibrated_avg 0.361703
fallback_drop 0.397098
fallback_all_above_070 False
film_weak_in_best_top10 2
```
