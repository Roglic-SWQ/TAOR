# Qwen 文本生成 5 实体 Smoke 验收报告

日期：2026-07-07
工作目录：`/mnt/data/LRJ/Knowledge-Bridger-main/MMKGC`
验收结论：`DONE_WITH_CONCERNS`

## 范围

- 配置路径：`configs/mkgw-knowledge-extraction.yaml`
- 数据集路径：`datasets/MKG-W`
- 输出目录：`outputs/qwen_text_smoke_20260707`
- Evaluation 目录：`outputs/qwen_text_smoke_20260707/evaluation`
- 文本生成器：`qwen-local`
- 模型名 / 目录：`./Qwen/Qwen2-VL-7B-Instruct`
- 本地服务地址：`http://localhost:12345/v1`

## 数据集自包含检查

`datasets/MKG-W` 可在本地直接加载，核心文件均存在：`entity2id.txt`、`relation2id.txt`、`train.txt`、`valid.txt`、`test.txt`、`entity_info.json`、`relation_info.json`。

本地子目录检查通过：`knowledge-extraction/`、`images/`、`entity-index/` 均存在。加载统计为 15,000 个实体、169 个关系、42,746 条三元组、14,774 个 knowledge-extraction JSON 文件、8,181 个图片文件。

本次 5 个实体均有本地描述和图谱邻居：`Q370252`、`Q234900`、`Q3857815`、`Q1356749`、`Q1373204`。其中 `Q1356749` 解析到本地图片路径，其余 4 个实体未解析到 ground-truth 图片路径；本次参数为 `--missing-mode both`，因此输入侧按文本和图片均缺失的 smoke 场景执行。

## 模型目录检查

`Qwen/Qwen2-VL-7B-Instruct` 目录存在。检查到 5 个 safetensors 分片：

- `model-00001-of-00005.safetensors`
- `model-00002-of-00005.safetensors`
- `model-00003-of-00005.safetensors`
- `model-00004-of-00005.safetensors`
- `model-00005-of-00005.safetensors`

同时存在 `config.json`、`configuration.json`、`model.safetensors.index.json`、`tokenizer.json`、`tokenizer_config.json`、`preprocessor_config.json`、`generation_config.json`。这只证明本地模型文件目录存在，不证明本地 Qwen 服务已经成功加载模型。

## 本地服务健康检查

健康检查命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py
```

结果为失败，返回 JSON：

```json
{"error":"[Errno 1] Operation not permitted","error_type":"connection_error","latency_ms":4,"ok":false}
```

该结果表明当前 sandbox 中 localhost 请求被拒绝。本次没有证明 `http://localhost:12345/v1` 的真实 Qwen 服务可达，也未证明真实 Qwen 生成质量。

## Run 结果

运行命令使用 `PYTHONDONTWRITEBYTECODE=1`，run name 为 `qwen_text_smoke_20260707`。命令返回：

```json
{"run_dir":"outputs/qwen_text_smoke_20260707","processed_entities":5,"failed_entities":0}
```

`metrics_summary.json` 检查结果：

- `processed_entities`: 5
- `failed_entities`: 0
- `avg_text_candidates`: 5.0
- `avg_image_candidates`: 5.0
- `avg_pair_candidates`: 9.0
- `fallback_counts`: `{}`
- `stage_failure_counts`: `{}`

`config.json` 检查结果：`dataset_dir=datasets/MKG-W`、`text_generator=qwen-local`、`model_name=./Qwen/Qwen2-VL-7B-Instruct`、`base_url=http://localhost:12345/v1`。运行配置没有写出明文 `api_key`，仅写出 `api_key_configured=true`。

实体目录数量为 5。每个实体的 `model_calls.jsonl` 行数均为 5，完整性计数为 `[5, 5, 5, 5, 5]`，合计 25 行。

模型调用统计：

- `generated_calls`: 0
- `failed_fallback_calls`: 25
- `status_counts`: `{"failed_fallback": 25}`
- `error_counts`: `{"URLError: [Errno 1] Operation not permitted": 25}`

因为 `generated_calls=0`，本次只验证了 fallback 链路，不能证明真实 Qwen 生成质量。

## Evaluation 结果

Evaluation 命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate --run-dir outputs/qwen_text_smoke_20260707 --overwrite
```

命令返回：

```json
{"evaluation_dir":"outputs/qwen_text_smoke_20260707/evaluation","evaluated_entities":5,"failed_entities":0,"failed_entity_keys":[]}
```

Evaluation 产物检查：

- `entity_scores.csv`: 5 行
- `best_cases.jsonl`: 5 行
- `worst_cases.jsonl`: 5 行
- `fallback_cases.jsonl`: 0 行
- `evaluation_summary.json` 中 `evaluated_entities=5`、`failed_entities=0`、`failed_entity_keys=[]`

Evaluation 平均分：

- `text_best_offline_evaluation_score`: 0.461230
- `image_best_offline_evaluation_score`: 0.918222
- `pair_best_offline_evaluation_score`: 0.892119
- `overall_offline_evaluation_score`: 0.757190
- `calibrated_overall_offline_evaluation_score`: 0.561852

注意：`fallback_cases.jsonl` 为 evaluator 自身的 case export 统计，本次为 0；这不改变 `model_calls.jsonl` 中 25 次 `failed_fallback` 的事实。

## 与 500 规则 Run 的关系

已有 500 规模规则 run 位于 `outputs/scale_500_both_20260707`，其配置为 `limit=500`、`candidate_budget=5`、`missing_mode=both`，指标为 `processed_entities=500`、`failed_entities=0`、平均候选数 5.0 / 5.0 / 9.0。

本次 `outputs/qwen_text_smoke_20260707` 是独立的 5 实体 Qwen text smoke run，使用同一 MKG-W 任务形态和候选预算验证 Qwen-local 接入路径、fallback 记录与 evaluation 链路。它没有覆盖或删除 500 规模 run，也不能替代 500 规模规则 run 的规模性结论。由于本次 `generated_calls=0`，它也不能作为真实 Qwen 文本生成质量的验收依据。

## 结论

5 实体 pipeline 与 evaluation 链路均跑通，产物完整，运行配置已避免写出明文 API key。主要限制是 localhost 健康检查在 sandbox 中被拒绝，25 次文本模型调用全部进入 `failed_fallback`，因此本次验收结论为结构链路通过、真实 Qwen 生成质量未证明。
