# MKG-W Qwen 文本生成接入实施计划

> **给 agentic workers：** 必须使用 `superpowers:subagent-driven-development` 逐任务执行。本计划使用 checkbox（`- [ ]`）跟踪任务状态。

**目标：** 在 MMKGC 内建立一个自包含的 Qwen2-VL-7B-Instruct 文本生成烟测链路，让 Stage 4 的 text candidates 可以调用本地 Qwen OpenAI-compatible 服务，同时保留规则 baseline 和现有 evaluation 流程。

**架构：** 不改 Stage 0-5 的主流程契约，不改 Stage 5 ranking 和 offline evaluation 公式。新增 Qwen text adapter，挂在现有 `TextGenerator` 边界后面；模型调用日志由 `RunWriter` 写入每个实体目录的 `model_calls.jsonl`；模型服务启动不耦合进 `mmkgc.run`，而是通过脚本和健康检查显式管理。

**技术栈：** Python 3.11、标准库 HTTP (`urllib.request`)、已存在的 PyYAML、现有 MMKGC pipeline、vLLM 服务命令、pytest、compileall。

---

## Controller 规则

执行时必须遵守用户确认的流程：

1. 主 Codex 会话作为 controller。
2. 每个 plan task 派一个 implementer subagent。
3. 每个 implementer 必须有明确写入范围、验证命令、禁止 commit。
4. 每个任务完成后派两个 reviewer：
   - spec reviewer：检查是否符合 `docs/superpowers/specs/2026-07-07-mkgw-qwen-text-generation-design.zh.md` 和用户确认方案。
   - quality reviewer：检查代码质量、测试质量、职责边界、路径风险、服务风险、产物风险。
5. reviewer 发现问题时，必须回给 implementer 修复。
6. 两个 reviewer 都 approve 后才标记任务完成。
7. 所有任务完成后派 final reviewer。
8. final reviewer approve 后，controller 运行完整验证命令并汇报证据。
9. 工作区可能有用户未提交改动，不要 revert 任何非本任务改动。
10. 不要自动 commit。

## 已确认范围

```text
model_dir: Qwen/Qwen2-VL-7B-Instruct
base_url: http://localhost:12345/v1
model_config: configs/mkgw-knowledge-extraction.yaml
default_dataset_dir: datasets/MKG-W
text_generator_modes: rule, qwen-local
qwen_smoke_run_name: qwen_text_smoke_20260707
qwen_smoke_limit: 5
missing_mode: both
candidate_budget: 5
model_call_log: per entity model_calls.jsonl
image_generation: remains rule image prompt only
real image files: not generated
```

## 文件与所有权

新增：

```text
MMKGC/configs/mkgw-knowledge-extraction.yaml
MMKGC/scripts/start_qwen_local_server.sh
MMKGC/scripts/check_qwen_local_server.py
MMKGC/mmkgc/adapters/qwen.py
MMKGC/tests/test_qwen_adapter.py
MMKGC/tests/test_qwen_cli_integration.py
MMKGC/docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md
```

如不存在则复制创建：

```text
MMKGC/datasets/MKG-W/
```

修改：

```text
MMKGC/configs/default.json
MMKGC/mmkgc/adapters/__init__.py
MMKGC/mmkgc/adapters/base.py
MMKGC/mmkgc/stages/generation.py
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/run.py
MMKGC/mmkgc/io/writer.py
MMKGC/tests/test_pipeline_outputs.py
MMKGC/tests/test_stages.py
MMKGC/README.md
MMKGC/README.zh.md
```

只读：

```text
MMKGC/docs/superpowers/specs/2026-07-07-mkgw-qwen-text-generation-design.zh.md
MMKGC/Qwen/Qwen2-VL-7B-Instruct/
KGC/datasets/MKG-W/
MMKGC/outputs/acceptance_pytest_20260706_0002/
MMKGC/outputs/scale_500_both_20260707/
```

禁止修改或覆盖：

```text
KGC/configs/
KGC/datasets/MKG-W/
MMKGC/outputs/acceptance_pytest_20260706_0002/
MMKGC/outputs/scale_500_both_20260707/
```

## Task 1：自包含数据、配置与服务脚本

**目标：** 让 MMKGC 拥有自己的数据集默认路径、Qwen 配置文件、模型路径检查、服务启动脚本和健康检查脚本。

**写入范围：**

```text
MMKGC/configs/
MMKGC/scripts/
MMKGC/datasets/MKG-W/
```

**禁止 commit。**

- [ ] **Step 1：预检自包含路径**

从仓库根目录运行：

```bash
test -d MMKGC/Qwen/Qwen2-VL-7B-Instruct
test -f MMKGC/Qwen/Qwen2-VL-7B-Instruct/config.json
test -f MMKGC/Qwen/Qwen2-VL-7B-Instruct/model.safetensors.index.json
test -d KGC/datasets/MKG-W
test -f KGC/datasets/MKG-W/entity2id.txt
test -f KGC/datasets/MKG-W/relation2id.txt
test -f KGC/datasets/MKG-W/train.txt
test -f KGC/datasets/MKG-W/entity_info.json
```

期望：全部 exit code 为 `0`。任一失败则报告 `BLOCKED`。

- [ ] **Step 2：复制 MKG-W 到 MMKGC**

从仓库根目录运行：

```bash
test ! -e MMKGC/datasets/MKG-W
```

如果 exit code 为 `0`，执行复制：

```bash
mkdir -p MMKGC/datasets
cp -a KGC/datasets/MKG-W MMKGC/datasets/MKG-W
```

如果 `MMKGC/datasets/MKG-W` 已存在，不覆盖，改为验证：

```bash
test -f MMKGC/datasets/MKG-W/entity2id.txt
test -f MMKGC/datasets/MKG-W/relation2id.txt
test -f MMKGC/datasets/MKG-W/train.txt
test -f MMKGC/datasets/MKG-W/entity_info.json
```

不得删除或修改 `KGC/datasets/MKG-W`。

- [ ] **Step 3：新增 Qwen 配置**

创建 `MMKGC/configs/mkgw-knowledge-extraction.yaml`：

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

- [ ] **Step 4：更新默认数据集路径**

修改 `MMKGC/configs/default.json`，让 `dataset_dir` 为：

```json
"datasets/MKG-W"
```

其它默认字段保持不变。

- [ ] **Step 5：新增 Qwen 服务启动脚本**

创建 `MMKGC/scripts/start_qwen_local_server.sh`。脚本职责：

- 从 `MMKGC/` 目录启动。
- 读取 `configs/mkgw-knowledge-extraction.yaml`。
- 校验 `./Qwen/Qwen2-VL-7B-Instruct` 存在。
- 检查当前终端能看到 GPU。
- 前台运行 `vllm serve`，不隐式后台常驻。

核心命令必须等价于：

```bash
vllm serve ./Qwen/Qwen2-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port 12345 \
  --served-model-name ./Qwen/Qwen2-VL-7B-Instruct \
  --trust-remote-code
```

创建后运行：

```bash
chmod +x scripts/start_qwen_local_server.sh
```

- [ ] **Step 6：新增健康检查脚本**

创建 `MMKGC/scripts/check_qwen_local_server.py`。脚本职责：

- 读取 `configs/mkgw-knowledge-extraction.yaml`。
- 调用 `http://localhost:12345/v1/chat/completions`。
- 使用配置里的 `api_key` 和 `model_name`。
- 发送一个极小英文测试 prompt。
- 成功输出包含 `ok=true`、`model_name`、`base_url`、`latency_ms` 和 `response_preview`。
- 失败输出包含 `ok=false`、`error_type`、`error` 和 `latency_ms`。

必须使用 Python 标准库 HTTP，避免新增远程 SDK 依赖。

- [ ] **Step 7：验证 Task 1**

从 `MMKGC/` 运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py
```

如果服务已经启动，期望 exit code `0` 且 JSON 中 `ok=true`。如果服务未启动，允许 exit code 非 0，但必须报告服务未启动。

再运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall scripts
```

期望 exit code `0`。

报告字段：

```text
Status
dataset_copied_or_present
model_dir_present
config_created
scripts_created
health_check_status
compileall_exit_code
files_changed
```

## Task 2：Qwen Adapter 与单元测试

**目标：** 新增本地 Qwen text adapter，调用本地 OpenAI-compatible endpoint，失败时回退规则文本，并记录完整模型调用日志。

**写入范围：**

```text
MMKGC/mmkgc/adapters/
MMKGC/tests/test_qwen_adapter.py
```

**禁止 commit。**

- [ ] **Step 1：先写 adapter 测试**

创建 `MMKGC/tests/test_qwen_adapter.py`，覆盖：

- `load_qwen_config` 能读取 `model_name`、`base_url`、`api_key`、`max_token`、`temperature`。
- 成功调用时返回模型响应文本。
- 成功调用时 `consume_last_call(entity_key, candidate_id)` 返回 `status=generated`。
- 成功调用时 `last_candidate_metadata()` 返回 `model_status=generated`。
- HTTP 失败时回退 `RuleTextGenerator`。
- HTTP 失败时日志和 metadata 记录 `failed_fallback` 与错误信息。

测试中用 `monkeypatch` 替换 `urllib.request.urlopen`，不得真的访问本地服务。

- [ ] **Step 2：确认测试先失败**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_qwen_adapter.py -v -p no:cacheprovider
```

期望：失败，原因是 `mmkgc.adapters.qwen` 尚不存在。

- [ ] **Step 3：实现 Qwen adapter**

创建 `MMKGC/mmkgc/adapters/qwen.py`，包含：

```text
QwenConfig
load_qwen_config(path)
QwenLocalTextGenerator
```

`QwenLocalTextGenerator` 必须：

- 保持 `generate(record_label, entity_type, relation, angle, facts) -> str` 契约。
- 构造英文 prompt。
- 通过 `urllib.request` POST 到 `{base_url}/chat/completions`。
- 传入 `model`、`messages`、`max_tokens`、`temperature`。
- 成功时记录完整 prompt、完整 response、latency、status。
- 失败时调用 `RuleTextGenerator` 回退，并记录 `failed_fallback`。
- 提供 `consume_last_call(entity_key, candidate_id)`。
- 提供 `last_candidate_metadata()`。

- [ ] **Step 4：导出 adapter**

修改 `MMKGC/mmkgc/adapters/__init__.py`，导出：

```text
QwenConfig
QwenLocalTextGenerator
load_qwen_config
```

- [ ] **Step 5：补充 base 注释**

在 `MMKGC/mmkgc/adapters/base.py` 的协议附近增加短注释，说明模型生成器可以可选暴露：

```text
consume_last_call()
last_candidate_metadata()
```

不得改变 `TextGenerator.generate(record_label, entity_type, relation, angle, facts) -> str` 的必需契约。

- [ ] **Step 6：验证 adapter 测试**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_qwen_adapter.py -v -p no:cacheprovider
```

期望：PASS。

- [ ] **Step 7：编译检查**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc/adapters tests/test_qwen_adapter.py
```

期望：exit code `0`。

报告字段：

```text
Status
tests_result
compileall_exit_code
files_changed
concerns
```

## Task 3：Pipeline、CLI 与 per-entity model_calls

**目标：** 把 `qwen-local` 接入 `mmkgc.run`，把 Qwen metadata 写入 text candidates，并在每个实体目录写出 `model_calls.jsonl`。

**写入范围：**

```text
MMKGC/mmkgc/stages/generation.py
MMKGC/mmkgc/pipeline.py
MMKGC/mmkgc/run.py
MMKGC/mmkgc/io/writer.py
MMKGC/tests/
```

**禁止 commit。**

- [ ] **Step 1：先写集成测试**

创建 `MMKGC/tests/test_qwen_cli_integration.py`，测试：

- `MMKGCPipeline(candidate_budget=2, text_generator=RecordingTextGenerator())` 能收集 2 条 `model_calls`。
- text candidate metadata 中有 `generator=qwen2_vl_7b` 和 `model_status=generated`。
- `RunWriter.write_entity(result)` 会写实体目录下的 `model_calls.jsonl`。
- 规则模式不写 `model_calls.jsonl`。

测试中的 `RecordingTextGenerator` 必须是本地 fake generator，不访问真实服务。

- [ ] **Step 2：确认集成测试先失败**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_qwen_cli_integration.py -v -p no:cacheprovider
```

期望：失败，因为 pipeline 还不接受注入 text generator，writer 也还不写 model calls。

- [ ] **Step 3：修改 Stage 4**

修改 `MMKGC/mmkgc/stages/generation.py`：

- `_generate_modality_candidates` 增加 `entity_key` 参数。
- 返回 `(candidates, model_calls)`。
- text 生成后动态检查：

```python
metadata_provider = getattr(text_generator, "last_candidate_metadata", None)
call_provider = getattr(text_generator, "consume_last_call", None)
```

- 如果 `metadata_provider` 可调用，把返回值合并进 candidate metadata。
- 如果 `call_provider` 可调用，把调用日志追加到 `model_calls`。
- 规则 generator 不暴露这些方法时，行为保持不变。

- [ ] **Step 4：修改 Pipeline 注入点**

修改 `MMKGC/mmkgc/pipeline.py`：

- `MMKGCPipeline.__init__` 支持 `text_generator=None` 和 `image_prompt_generator=None`。
- 默认仍使用 `RuleTextGenerator()` 与 `RuleImagePromptGenerator()`。
- Stage 4 返回的 `model_calls` 放到 pipeline 顶层结果：

```python
"model_calls": model_calls
```

- 不把 `model_calls` 写进 `final`。

- [ ] **Step 5：修改 RunWriter**

修改 `MMKGC/mmkgc/io/writer.py`：

- 新增 `_write_jsonl(path, rows)` helper。
- 如果 result 中存在非空 `model_calls`，写入：

```text
entities/sanitized_entity_key/model_calls.jsonl
```

- 规则模式没有 model calls 时，不创建该文件。

- [ ] **Step 6：修改 CLI**

修改 `MMKGC/mmkgc/run.py`：

- 默认 dataset dir 改为 `datasets/MKG-W`。
- 新增参数：

```text
--text-generator {rule,qwen-local}
--model-config configs/mkgw-knowledge-extraction.yaml
--model-call-timeout-seconds 60
```

- `rule` 模式不读取 model config，不调用本地服务。
- `qwen-local` 模式读取 config，构造 `QwenLocalTextGenerator`。
- `config.json` 记录 `text_generator`、`model_config`、`model_name`、`base_url`、`api_key_configured`。
- `config.json` 不得写入明文 `api_key`。

- [ ] **Step 7：更新既有测试**

更新默认路径断言：

```text
../KGC/datasets/MKG-W -> datasets/MKG-W
```

新增规则模式测试，断言规则 pipeline 不写 `model_calls.jsonl`。

- [ ] **Step 8：跑聚焦测试**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests/test_qwen_adapter.py tests/test_qwen_cli_integration.py tests/test_pipeline_outputs.py -v -p no:cacheprovider
```

期望：PASS。

- [ ] **Step 9：跑完整测试**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
```

期望：PASS。当前基线是 `94 passed`，新增测试后通过数量应增加。

报告字段：

```text
Status
focused_tests_result
full_tests_result
files_changed
concerns
```

## Task 4：README 与操作说明

**目标：** 更新中英文 README，说明新的自包含数据集路径、本地 Qwen 服务启动、健康检查、烟测运行和 fallback 语义。

**写入范围：**

```text
MMKGC/README.md
MMKGC/README.zh.md
```

**禁止 commit。**

- [ ] **Step 1：更新英文 README**

把运行示例中的：

```bash
python -m mmkgc.run --dataset-dir ../KGC/datasets/MKG-W --missing-mode both --limit 100
```

改为：

```bash
python -m mmkgc.run --dataset-dir datasets/MKG-W --missing-mode both --limit 100
```

新增 `Local Qwen Text Generation` 小节，包含：

- GPU 可见终端启动：

```bash
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
scripts/start_qwen_local_server.sh
```

- 另一个终端健康检查：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py
```

- 5 实体烟测命令：

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

- 说明 Qwen 只替换 text candidates，image candidates 仍是规则 prompt。
- 说明失败时记录 `model_status=failed_fallback`。

- [ ] **Step 2：更新中文 README**

在 `MMKGC/README.zh.md` 中同步中文说明：

- 默认数据集路径：`datasets/MKG-W`。
- 本地 Qwen 服务启动命令。
- 健康检查命令。
- 5 实体烟测命令。
- fallback metadata 与 `model_calls.jsonl` 说明。

- [ ] **Step 3：扫描旧默认路径**

从 `MMKGC/` 运行：

```bash
rg -n "../KGC/datasets/MKG-W" README.md README.zh.md configs mmkgc tests
```

期望：active code/docs 中无旧默认路径。历史 spec/plan 可以保留旧 run 记录，不要改历史文档。

- [ ] **Step 4：检查 CLI help**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run --help
```

期望：help 中包含 `--text-generator`、`--model-config`。

报告字段：

```text
Status
readme_updated
stale_reference_scan
help_check
files_changed
concerns
```

## Task 5：Qwen 烟测、Evaluation 与中文验收报告

**目标：** 运行 5 实体 Qwen text smoke path，运行 evaluation，并基于真实产物写中文验收报告。

**写入范围：**

```text
MMKGC/outputs/qwen_text_smoke_20260707/
MMKGC/docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md
```

**禁止 commit。**

- [ ] **Step 1：保护输出路径**

从 `MMKGC/` 运行：

```bash
test ! -e outputs/qwen_text_smoke_20260707
```

期望：exit code `0`。如果非 0，报告 `BLOCKED`，不得删除或覆盖。

- [ ] **Step 2：运行健康检查**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python scripts/check_qwen_local_server.py
```

如果服务已启动，期望 exit code `0` 且 `ok=true`。如果服务未启动，记录 JSON 错误。controller 可以决定只验证 fallback 链路；报告必须明确这不能证明真实模型生成质量。

- [ ] **Step 3：运行 Qwen smoke pipeline**

运行：

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

期望：

```text
processed_entities = 5
failed_entities = 0
```

- [ ] **Step 4：检查 run 产物**

必须检查并报告：

```text
dataset_dir = datasets/MKG-W
text_generator = qwen-local
model_name = ./Qwen/Qwen2-VL-7B-Instruct
base_url = http://localhost:12345/v1
api_key_in_config = False
processed_entities = 5
failed_entities = 0
avg_text_candidates = 5.0
avg_image_candidates = 5.0
avg_pair_candidates = 9.0
entity_dirs = 5
model_call_counts = [5, 5, 5, 5, 5]
generated_calls = actual count
failed_fallback_calls = actual count
```

- [ ] **Step 5：运行 evaluation**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_text_smoke_20260707 \
  --overwrite
```

期望：

```text
evaluated_entities = 5
failed_entities = 0
failed_entity_keys = []
```

- [ ] **Step 6：检查 evaluation**

必须检查并报告：

```text
entity_scores.csv rows = 5
best_cases.jsonl rows > 0
worst_cases.jsonl rows > 0
fallback_cases.jsonl rows >= 0
```

- [ ] **Step 7：写中文验收报告**

创建：

```text
MMKGC/docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md
```

报告必须包含：

- 配置路径和数据集路径。
- 数据集自包含检查。
- 模型目录检查。
- 本地服务健康检查结果。
- Run processed / failed / candidate 统计。
- generated_calls / failed_fallback_calls。
- `model_calls.jsonl` 完整性。
- evaluation 结果。
- 与 500 规则 run 的关系说明。
- 如果 `generated_calls = 0`，明确本次只验证了 fallback 链路。

不得保留占位符或未完成段落。

- [ ] **Step 8：验证报告**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
from pathlib import Path
path = Path("docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md")
text = path.read_text(encoding="utf-8")
markers = ["TO" + "DO", "T" + "BD", "待" + "填写", "待" + "定", chr(60) + "." * 3 + chr(62), "." * 3]
found = [marker for marker in markers if marker in text]
print("markers", found)
raise SystemExit(1 if found else 0)
'
```

期望：`markers []`，exit code `0`。

报告字段：

```text
Status
health_check_status
run_status
model_call_counts
generated_calls
failed_fallback_calls
evaluation_status
report_path
files_changed
concerns
```

## Task 6：最终验证

**目标：** 重新运行完整验证，证明最终状态满足 spec。

**写入范围：** 无；如 Task 5 已生成 Qwen run，可以用 `--overwrite` 刷新 `outputs/qwen_text_smoke_20260707/evaluation/`。

**禁止 commit。**

- [ ] **Step 1：完整测试**

从 `MMKGC/` 运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m pytest tests -v -p no:cacheprovider
```

期望：PASS。

- [ ] **Step 2：compileall**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m compileall mmkgc tests scripts
```

期望：exit code `0`。

- [ ] **Step 3：规则模式自包含 smoke run**

先保护路径：

```bash
test ! -e outputs/rule_self_contained_smoke_20260707
```

如果路径不存在，运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --missing-mode both \
  --limit 5 \
  --candidate-budget 5 \
  --text-generator rule \
  --output-root outputs \
  --run-name rule_self_contained_smoke_20260707
```

期望：

```text
processed_entities = 5
failed_entities = 0
```

验证规则模式不写模型日志：

```bash
find outputs/rule_self_contained_smoke_20260707/entities -name model_calls.jsonl -print
```

期望：无输出。

- [ ] **Step 4：Qwen smoke artifact 最终检查**

如果 `outputs/qwen_text_smoke_20260707` 存在，重新运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -m mmkgc.evaluate \
  --run-dir outputs/qwen_text_smoke_20260707 \
  --overwrite
```

期望：exit code `0`、evaluated `5`、failed `0`。

- [ ] **Step 5：敏感字段检查**

运行：

```bash
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
import json
from pathlib import Path
for run_name in ["qwen_text_smoke_20260707", "rule_self_contained_smoke_20260707"]:
    path = Path("outputs") / run_name / "config.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        print(run_name, "api_key_in_config", "api_key" in config, "dataset_dir", config.get("dataset_dir"))
'
```

期望：

```text
api_key_in_config False
dataset_dir datasets/MKG-W
```

- [ ] **Step 6：最终报告检查**

运行：

```bash
test -f docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md
/home/lirunjin/miniconda3/envs/kb/bin/python -c '
from pathlib import Path
path = Path("docs/evaluation/2026-07-07-qwen-text-smoke-validation.zh.md")
text = path.read_text(encoding="utf-8")
markers = ["TO" + "DO", "T" + "BD", "待" + "填写", "待" + "定", chr(60) + "." * 3 + chr(62), "." * 3]
found = [marker for marker in markers if marker in text]
print("markers", found)
raise SystemExit(1 if found else 0)
'
```

期望：报告存在且无占位符。

报告字段：

```text
Status
pytest_result
compileall_exit_code
rule_smoke_result
qwen_smoke_result
evaluation_result
api_key_redaction_check
report_check
files_changed
concerns
```

## Final Reviewer 要求

所有 task 都通过双 reviewer 后，派 final reviewer 审完整变更。final reviewer 必须检查：

```text
1. 是否覆盖 spec。
2. 运行时是否不依赖 KGC/configs 或 KGC/datasets 默认路径。
3. 运行时是否不依赖 MMKGC 外部模型路径。
4. rule 模式是否仍可运行且不调用模型。
5. qwen-local 模式是否写入 metadata 和 per-entity model_calls.jsonl。
6. api_key 是否没有写入 run config 或报告。
7. README 指令是否准确。
8. 历史 100 和 500 输出是否未被修改或覆盖。
9. tests 和 compileall 是否通过。
10. 是否没有自动 commit。
```

final reviewer approve 后，controller 才运行最终验证命令并向用户汇报证据。
