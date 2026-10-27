# Qwen2-VL-7B 真实文本生成 5 实体 Smoke 补充验收报告

日期：2026-07-08
工作目录：`/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC`
验收结论：真实模型 text generation 链路 5 实体 smoke 验收通过，结论带小样本限制。

## 报告目标

本报告是在用户手动部署 `Qwen2-VL-7B` 并使本地 OpenAI-compatible 服务连通后，对真实模型 text generation 链路的补充验收。它对应输出目录 `outputs/qwen_text_real_7b_20260708`，不覆盖 2026-07-07 的 fallback-only 报告 `docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md`。

## 配置与服务

- 配置路径：`configs/mkgw-knowledge-extraction.yaml`
- 数据集路径：`datasets/MKG-W`
- 输出目录：`outputs/qwen_text_real_7b_20260708`
- Evaluation 目录：`outputs/qwen_text_real_7b_20260708/evaluation`
- 文本生成器：`qwen-local`
- 模型目录 / 模型名：`./Qwen/Qwen2-VL-7B-Instruct`
- 本地服务地址：`http://localhost:12345/v1`

`outputs/qwen_text_real_7b_20260708/config.json` 记录了 `model_config=configs/mkgw-knowledge-extraction.yaml`、`dataset_dir=datasets/MKG-W`、`model_name=./Qwen/Qwen2-VL-7B-Instruct`、`base_url=http://localhost:12345/v1`、`api_key_configured=true`。该 run 配置没有写出明文 `api_key`。

## 健康检查

controller 在真实模型 run 前的本地服务健康检查返回：

```json
{"ok": true, "latency_ms": 1690, "response_preview": "OK."}
```

该结果表明当时 `http://localhost:12345/v1` 已可达，后续 25 次文本生成调用也均记录为真实模型生成。

## Run 统计

`outputs/qwen_text_real_7b_20260708/metrics_summary.json` 复核结果：

- `processed_entities`: 5
- `failed_entities`: 0
- `avg_text_candidates`: 5.0
- `avg_image_candidates`: 5.0
- `avg_pair_candidates`: 9.0
- `fallback_counts`: `{}`
- `stage_failure_counts`: `{}`

实体输出目录共 5 个：`Q1356749`、`Q1373204`、`Q234900`、`Q370252`、`Q3857815`。

## 模型调用完整性

每个实体的 `model_calls.jsonl` 均为 5 行：

- `Q1356749`: 5
- `Q1373204`: 5
- `Q234900`: 5
- `Q370252`: 5
- `Q3857815`: 5

合计 25 行。跨全部 `model_calls.jsonl` 的状态统计为：

- `generated_calls`: 25
- `failed_fallback_calls`: 0
- `fallback_used=true`: 0
- `status_counts`: `{"generated": 25}`

因此本次是真实 Qwen2-VL-7B 生成 `25/25`，不是 fallback-only run。

## Evaluation 结果

`outputs/qwen_text_real_7b_20260708/evaluation/evaluation_summary.json` 复核结果：

- `evaluated_entities`: 5
- `failed_entities`: 0
- `failed_entity_keys`: `[]`
- `fallback_counts.entities`: 0
- `fallback_counts.candidates`: 0
- `case_exports.best_cases`: 5
- `case_exports.worst_cases`: 5
- `case_exports.fallback_cases`: 0

Evaluation 平均分：

- `text_best_offline_evaluation_score`: 0.394632
- `image_best_offline_evaluation_score`: 0.918222
- `pair_best_offline_evaluation_score`: 0.826642
- `overall_offline_evaluation_score`: 0.713165
- `calibrated_overall_offline_evaluation_score`: 0.569165

## 与 2026-07-07 fallback-only run 的关系

`outputs/qwen_text_smoke_20260707` 是服务未连通时的 5 实体 Qwen text smoke run。其模型调用复核结果为 `generated=0`、`failed_fallback=25`、`fallback_used=true` 为 25，因此 2026-07-07 报告只能证明 pipeline、fallback 记录和 evaluation 链路可运行，不能证明真实 Qwen 生成质量。

本报告对应 `outputs/qwen_text_real_7b_20260708`，是在服务连通后的真实 Qwen2-VL-7B run：25 次 text candidate 调用全部为 `generated`，没有进入 failed fallback。两份报告是前后互补关系，后一份补足真实模型接入验收，不替换前一份对 fallback-only 场景的记录。

## 与 500 规则 run 的关系

`outputs/scale_500_both_20260707` 是 500 实体规则生成的规模/评估基线，复核结果为 `processed_entities=500`、`failed_entities=0`、平均候选数 5.0 / 5.0 / 9.0，evaluation 中 `evaluated_entities=500`、`failed_entities=0`。

`outputs/qwen_text_real_7b_20260708` 是 5 实体小样本真实模型接入验收，只用于确认本地 Qwen2-VL-7B text generation 调用、产物记录和 evaluation 链路能在服务连通后跑通。它不能直接代表 500 规模质量，也不能替代 500 规则 run 的规模性结论。

## 限制

- 本次只跑 5 个实体，样本量很小。
- 本次只替换 text candidates 为 Qwen2-VL-7B 真实模型生成。
- image candidates 仍为规则 prompt 生成，不是视觉模型真实生成。
- pair candidates 基于当前 text/image candidates 组合与排序，仍受 image 规则候选影响。
- 尚未扩大到 500 实体或更大规模，因此不能据此判断大规模质量、吞吐、稳定性和成本。

## 结论

`outputs/qwen_text_real_7b_20260708` 证明：在用户手动部署 Qwen2-VL-7B 并使本地服务健康检查通过后，MMKGC 的 qwen-local text generation 链路可以完成 5 实体、25 次真实模型文本候选生成；产物完整，evaluation 成功，且 run 配置未写出明文 API key。当前结论限于小样本 text generation 接入验收。
