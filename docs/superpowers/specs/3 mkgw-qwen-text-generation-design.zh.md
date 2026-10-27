# MKG-W 本地 Qwen 文本生成接入设计文档

## 目标

在已完成离线 MVP、evaluation 校准和 500 实体扩大运行之后，进入真实模型接入的第一步：把本地部署的 `Qwen2-VL-7B-Instruct` 接入 MMKGC 的 Stage 4 文本候选生成。

本阶段只做小样本烟测版，目标是回答：

- 现有六阶段 pipeline 能否在不破坏输出结构的前提下调用本地 Qwen 服务生成 text candidates。
- Qwen 生成的 text candidates 能否继续被 Stage 5 ranking 和现有 evaluation 消费。
- 模型调用记录是否足够追溯 prompt、response、耗时和失败回退。
- MMKGC 是否能在运行时只依赖 `MMKGC/` 内部的配置和数据集路径。

## 已确认方案

brainstorming 阶段已确认：

- 接入模型：本地 `Qwen2-VL-7B-Instruct`。
- 调用方式：本机 OpenAI-compatible endpoint，不是外部云 API。
- 服务地址：`http://localhost:12345/v1`。
- 模型目录：`MMKGC/Qwen/Qwen2-VL-7B-Instruct`。
- 模型名 / 服务启动路径：`./Qwen/Qwen2-VL-7B-Instruct`。
- API key：本机服务 token，写入本地 gitignored 配置，不提交真实值。
- 模型配置文件：`MMKGC/configs/mkgw-knowledge-extraction.yaml`。
- 数据集复制到：`MMKGC/datasets/MKG-W`。
- MMKGC 默认数据集路径改为：`datasets/MKG-W`。
- 第一版只接入 `text_candidates`，不接真实图像生成。
- `image_candidates` 继续使用现有规则 image prompt generator。
- Qwen 生成语言：英文。
- Qwen 生成格式：一到两句实体描述，尽量贴近 MKG-W ground truth 描述风格。
- 模型调用失败策略：回退到现有规则文本生成，并在 metadata 和调用日志中记录失败。
- 第一版试跑配置：`missing_mode=both`、`limit=5`、`candidate_budget=5`。
- 模型调用日志：每个实体目录下保存 `model_calls.jsonl`。
- `model_calls.jsonl` 保存完整 prompt 和完整 response。

## 非目标

本阶段不做以下事情：

- 不接真实图像生成模型，不生成图片文件。
- 不接 CLIP、BLIP、Sentence-BERT、LLM judge 或其它真实评价模型。
- 不修改 Stage 0、Stage 1、Stage 2、Stage 3 的规则。
- 不重写 Stage 5 ranking，只让它继续消费候选文本。
- 不修改现有 evaluation 公式和 calibration 规则。
- 不直接跑 500 实体或全量 MKG-W。
- 不读取 `KGC/configs/` 下的配置文件。
- 运行时不依赖 `KGC/datasets/MKG-W`。
- 运行时不依赖 `MMKGC/` 外部的模型权重路径。
- 不删除或移动原始 `KGC/datasets/MKG-W`。
- 不把模型服务生命周期耦合进 `mmkgc.run`；pipeline 只调用已启动的本地 endpoint。
- 不自动 commit。

## 自包含目录结构

本阶段实施后，MMKGC 应具备以下自包含结构：

```text
MMKGC/
  configs/
    default.json
    mkgw-knowledge-extraction.yaml
  datasets/
    MKG-W/
      entity2id.txt
      relation2id.txt
      train.txt
      entity_info.json
      other MKG-W dataset files
  Qwen/
    Qwen2-VL-7B-Instruct/
      config.json
      model-00001-of-00005.safetensors
      model-00002-of-00005.safetensors
      model-00003-of-00005.safetensors
      model-00004-of-00005.safetensors
      model-00005-of-00005.safetensors
      tokenizer.json
      other Qwen model files
  mmkgc/
  scripts/
  tests/
  outputs/
```

`MMKGC/datasets/MKG-W` 通过一次性复制现有本地数据集得到。复制是实施阶段的 bootstrap 动作；复制完成后的 MMKGC 运行命令不得再默认指向 `../KGC/datasets/MKG-W`。

`MMKGC/Qwen/Qwen2-VL-7B-Instruct` 已由用户复制到 MMKGC 内，实施阶段只需校验目录和关键模型文件存在，不需要再从 MMKGC 外部读取模型权重。

## 配置文件

新增配置文件：

```text
MMKGC/configs/mkgw-knowledge-extraction.yaml
```

内容使用已确认的本地 Qwen 服务配置：

```yaml
lmm:
  model_name: ./Qwen/Qwen2-VL-7B-Instruct
  base_url: http://localhost:12345/v1
  api_key: <local-api-key>

dataset:
  name: mkgw
  data_dir: ./datasets/MKG-W
  entity_info_path: ./datasets/MKG-W/entity_info.json

inference:
  save_path: ./outputs/qwen_text_smoke
  max_token: 512
  temperature: 0.1
  num_workers: 1
```

说明：

- `base_url` 是本机服务地址，不是外部云服务。
- `model_name` 同时作为本地服务启动时的模型目录参数，必须指向 `MMKGC/` 内的 `./Qwen/Qwen2-VL-7B-Instruct`。
- `num_workers` 第一版固定为 `1`，避免多个并发请求给本地 7B 服务带来额外不确定性。
- `save_path` 在本阶段只作为配置记录，不替代 MMKGC 的 `--output-root` / `--run-name`。
- 实现不应新增远程 SDK 依赖。HTTP 调用优先使用 Python 标准库。
- YAML 解析不得成为新的脆弱依赖。实现可优先使用已安装的 YAML 库；如果环境没有该库，应支持当前简单嵌套 YAML 配置的最小解析能力，或在启动时给出清晰错误。

## 本地 Qwen 服务部署边界

本阶段需要在 MMKGC 内提供本地 Qwen 服务启动与健康检查辅助脚本，但 `mmkgc.run` 不自动启动或停止模型服务。

新增脚本：

```text
MMKGC/scripts/start_qwen_local_server.sh
MMKGC/scripts/check_qwen_local_server.py
```

启动脚本职责：

- 从 `MMKGC/` 目录启动。
- 使用 `configs/mkgw-knowledge-extraction.yaml` 读取 `model_name`、`base_url` 和 `api_key`。
- 校验 `./Qwen/Qwen2-VL-7B-Instruct` 存在。
- 通过 `vllm serve ./Qwen/Qwen2-VL-7B-Instruct --host 127.0.0.1 --port 12345` 启动 OpenAI-compatible 服务。
- 明确提示该脚本需要在能看到 GPU 的终端中运行。
- 不在后台隐式常驻；默认以前台方式运行，便于用户观察日志和手动停止。

健康检查脚本职责：

- 调用 `http://localhost:12345/v1/chat/completions`。
- 使用配置中的 `api_key` 和 `model_name`。
- 发送一个极小英文测试 prompt。
- 成功时输出 JSON，包含 `ok=true`、`model_name`、`base_url`、`latency_ms` 和响应摘要。
- 失败时输出 JSON，包含 `ok=false`、错误类型和错误信息。

如果当前 Codex 执行环境无法通过 `nvidia-smi` 看到 GPU，执行阶段不得假装已成功部署模型服务。它应写明环境限制，并要求用户在可见 GPU 的终端启动 `scripts/start_qwen_local_server.sh` 后，再运行健康检查和 Qwen 烟测。

## CLI 变更

`mmkgc.run` 需要保留当前规则模式，同时新增模型文本生成入口。

默认行为：

```bash
python -m mmkgc.run
```

仍应使用规则生成，且默认数据集路径改为：

```text
datasets/MKG-W
```

新增建议参数：

```text
--text-generator {rule,qwen-local}
--model-config configs/mkgw-knowledge-extraction.yaml
--model-call-timeout-seconds 60
```

示例烟测命令：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator qwen-local \
  --model-config configs/mkgw-knowledge-extraction.yaml \
  --output-root outputs \
  --run-name qwen_text_smoke_20260707
```

如果 `--text-generator` 不传或传 `rule`：

- 不读取模型配置。
- 不调用本地 Qwen 服务。
- 输出结构与现有规则 pipeline 保持兼容。

如果传 `qwen-local`：

- 读取 `MMKGC/configs/mkgw-knowledge-extraction.yaml`。
- 只替换 text generator。
- image prompt generator 仍使用 `RuleImagePromptGenerator`。

## Adapter 设计

新增本地 Qwen text generator adapter，遵循现有 `TextGenerator` 协议：

```text
TextGenerator.generate(record_label, entity_type, relation, angle, facts) -> str
```

为了记录模型调用，本阶段允许 adapter 额外维护调用日志，例如：

```text
QwenLocalTextGenerator.pop_call_logs() -> list[dict]
```

或由 pipeline 在每个实体运行结束后读取 adapter 的本实体日志。

每次文本候选生成需要构造英文 prompt，要求模型：

- 只根据提供的实体名、实体类型、关系、角度和 facts 生成。
- 输出一到两句英文实体描述。
- 不编造 facts 中没有的具体名称、日期、地点或身份。
- 不输出列表、Markdown、解释文本或多候选。
- 如果证据稀疏，给出保守描述。

## Candidate metadata

Qwen 生成成功时，text candidate 的 `generated_content.value` 写入模型输出文本，并在 `metadata` 中增加：

```json
{
  "generator": "qwen2_vl_7b",
  "model_name": "./Qwen/Qwen2-VL-7B-Instruct",
  "model_status": "generated",
  "fallback_used": false
}
```

Qwen 调用失败时，候选回退到 `RuleTextGenerator`，并在 `metadata` 中增加：

```json
{
  "generator": "qwen2_vl_7b",
  "model_name": "./Qwen/Qwen2-VL-7B-Instruct",
  "model_status": "failed_fallback",
  "fallback_used": true,
  "model_error": "<short error type and message>"
}
```

失败回退不应让整个 run 失败。只有数据集读取、输出目录冲突、配置缺失或非候选级严重错误才应阻止 run。

## 模型调用日志

每个实体目录新增：

```text
outputs/<run_name>/entities/<entity_key>/model_calls.jsonl
```

每行对应一个 text candidate 的模型调用，字段包括：

```json
{
  "entity_key": "Q12345",
  "candidate_id": "text_001",
  "provider": "qwen-local-openai-compatible",
  "model_name": "./Qwen/Qwen2-VL-7B-Instruct",
  "base_url": "http://localhost:12345/v1",
  "temperature": 0.1,
  "max_tokens": 512,
  "prompt": "<full prompt>",
  "response_text": "<full response text when available>",
  "latency_ms": 1234,
  "status": "generated",
  "error": null,
  "fallback_used": false
}
```

失败时：

```json
{
  "status": "failed_fallback",
  "error": "<short error type and message>",
  "response_text": "",
  "fallback_used": true
}
```

日志中可以保存完整 prompt 和完整 response，因为第一版只跑 `limit=5`，文件体积可控。

## Run 输出

Qwen 烟测 run 仍保持现有输出结构：

```text
outputs/qwen_text_smoke_20260707/
  config.json
  results.jsonl
  metrics_summary.json
  entities/
    <entity_key>/
      0_coarse_relation_filtering.json
      1_knowledge_modeling.json
      2_graph_constrained_reasoning.json
      3_fine_grained_relation_selection.json
      4_relation_alternation_generation.json
      5_kg_aware_pair_ranking.json
      final.json
      model_calls.jsonl
```

`config.json` 需要记录：

- `dataset_dir: datasets/MKG-W`
- `missing_mode: both`
- `limit: 5`
- `candidate_budget: 5`
- `text_generator: qwen-local`
- `model_config: configs/mkgw-knowledge-extraction.yaml`
- `model_name: ./Qwen/Qwen2-VL-7B-Instruct`
- `base_url: http://localhost:12345/v1`

`api_key` 不应写入 `config.json` 明文。可以记录是否配置了 key，例如：

```json
{
  "api_key_configured": true
}
```

## Evaluation

现有 evaluation 继续运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_text_smoke_20260707 \
  --overwrite
```

本阶段不修改 evaluation 公式。验收只要求 evaluation 能消费 Qwen run 并输出：

```text
evaluation/evaluation_summary.json
evaluation/summary.csv
evaluation/entity_scores.csv
evaluation/best_cases.jsonl
evaluation/worst_cases.jsonl
evaluation/fallback_cases.jsonl
evaluation/entities/
```

## 验收指标

### 基础验证

必须通过：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests
```

### 数据集自包含

必须满足：

```text
MMKGC/datasets/MKG-W/entity2id.txt exists
MMKGC/datasets/MKG-W/relation2id.txt exists
MMKGC/datasets/MKG-W/train.txt exists
MMKGC/datasets/MKG-W/entity_info.json exists
MMKGC/Qwen/Qwen2-VL-7B-Instruct/config.json exists
MMKGC/Qwen/Qwen2-VL-7B-Instruct/model.safetensors.index.json exists
```

默认运行配置必须使用：

```text
dataset_dir = datasets/MKG-W
```

### 规则模式回归

不启用 Qwen 时，规则 pipeline 必须仍可运行并通过现有测试。至少做一个小样本规则 run 或测试覆盖，证明 `--text-generator rule` 不读取模型配置、不调用本地服务。

### Qwen 烟测 run

在 Qwen 烟测 run 前，健康检查应先通过：

```text
scripts/check_qwen_local_server.py reports ok=true
```

如果本地 Qwen 服务可用，`qwen-local` run 最低验收：

```text
processed_entities = 5
failed_entities = 0
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
```

每个实体目录必须存在 `model_calls.jsonl`，并至少有 5 行 text candidate 调用记录。

每个 text candidate 必须有：

```text
metadata.generator
metadata.model_name
metadata.model_status
metadata.fallback_used
```

如果本地 Qwen 服务不可用，允许通过 fallback 完成 run，但验收报告必须明确：

```text
model_status = failed_fallback
fallback_used = true
```

该情况只能证明回退机制可用，不能证明真实模型生成质量。

### Evaluation 验收

Qwen 烟测 run 完成后必须运行 evaluation，最低验收：

```text
evaluated_entities = 5
failed_entities = 0
entity_scores.csv rows = 5
best_cases.jsonl rows > 0
worst_cases.jsonl rows > 0
```

## 报告交付物

新增中文报告：

```text
MMKGC/docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md
```

报告必须包含：

- 配置路径和数据集路径。
- 是否完成数据集复制。
- 模型目录是否位于 `MMKGC/Qwen/Qwen2-VL-7B-Instruct`。
- 本地 Qwen 服务是否已通过健康检查。
- Qwen run 的 processed / failed / candidate 统计。
- 真实模型生成成功数量和 fallback 数量。
- `model_calls.jsonl` 完整性检查。
- evaluation 结果摘要。
- 与 500 规则 run 的关系说明：本阶段是真实模型接入烟测，不直接与 500 规模结论混为一谈。
- 如果 Qwen 服务不可用，明确说明本次只验证了 fallback 链路，后续需要在服务可用时重跑。

报告不得保留占位符、待办项或省略号式内容。

## 实施边界

允许修改或新增：

```text
MMKGC/configs/default.json
MMKGC/configs/mkgw-knowledge-extraction.yaml
MMKGC/datasets/MKG-W/
MMKGC/scripts/start_qwen_local_server.sh
MMKGC/scripts/check_qwen_local_server.py
MMKGC/mmkgc/adapters/
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/run.py
MMKGC/mmkgc/io/writer.py
MMKGC/tests/
MMKGC/README.md
MMKGC/README.zh.md
MMKGC/docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md
```

禁止修改或覆盖：

```text
MMKGC/outputs/acceptance_pytest_20260706_0002/
MMKGC/outputs/scale_500_both_20260707/
KGC/configs/
KGC/datasets/MKG-W/
```

实施阶段可以只读 `KGC/datasets/MKG-W/` 作为一次性复制来源，但复制完成后 MMKGC 的默认运行与 Qwen 烟测不得依赖该外部路径。

禁止读取或引用 MMKGC 外部模型目录。模型权重必须使用：

```text
MMKGC/Qwen/Qwen2-VL-7B-Instruct/
```

## 风险与约束

- 本地 Qwen 服务调用可能受 sandbox 网络限制影响；如果在 sandbox 内访问 `localhost:12345` 失败，执行阶段应按工具权限规则请求必要授权后重试。
- 本地 Qwen 服务部署依赖 GPU 可见性。若当前 Codex shell 看不到 GPU，但用户的普通终端能看到 GPU，则模型服务应由用户在普通终端运行启动脚本，MMKGC 只负责健康检查和调用。
- 本地模型速度可能较慢，所以第一版固定 `limit=5`、`num_workers=1`。
- Qwen2-VL 是多模态模型，但本阶段只使用文本输入和文本输出。
- `api_key` 是本地服务 token，不应写入 run config 明文或报告正文中的敏感字段。配置文件本身按用户提供内容写入。
- 当前工作区不是可用 git 仓库，无法依赖 git diff 证明改动范围；实施阶段需要用路径清单和 reviewer 审查控制范围。

## 完成定义

本阶段只有在以下条件都满足时才算完成：

- spec 和 implementation plan 经用户确认。
- 使用 subagent-driven-development 执行。
- 每个 task 均有 implementer、spec reviewer、quality reviewer，并且 reviewer 问题全部修复。
- final reviewer approve。
- 数据集已复制到 `MMKGC/datasets/MKG-W`。
- 模型目录已确认位于 `MMKGC/Qwen/Qwen2-VL-7B-Instruct`。
- 默认数据集路径已改为 `datasets/MKG-W`。
- 已提供 Qwen 本地服务启动脚本和健康检查脚本。
- Qwen text adapter 已实现并有测试覆盖。
- 规则模式回归通过。
- Qwen `limit=5` 烟测 run 生成并通过 evaluation；如果模型服务不可用，则报告明确该 run 只验证 fallback 链路。
- 最终验证命令通过并汇报证据。
- 不自动 commit。
