# GPT-4o-mini 主流程与 Reranker 运行环境

当前 CLI、Python API 和 4 / 10 / 50 实体启动脚本的语言模型统一为 `gpt-4o-mini`。
Stage 0 / Stage 1 / Stage 3 采用 `gpt4omini_stage03_v2` 提示词，设计见 `docs/gpt4omini-prompts-v2.zh.md`。

| 阶段 | 当前模型与职责 |
| --- | --- |
| Stage 0 | GPT-4o-mini 做关系语义粗筛 |
| Stage 1 | GPT-4o-mini 构建知识模型；本地 Qwen3-Reranker-4B 做证据评分 |
| Stage 2 | GPT-4o-mini 选择关系角色并分配候选数量 |
| Stage 3 | GPT-4o-mini 生成文本或图像提示词；SDXL 生成图像 |
| Stage 4 | GPT-4o-mini 与 CLIP、BLIP 等评分器参与候选排序 |

Stage 0 / Stage 1 / Stage 3 使用 `configs/gpt4omini-mainline.yaml`，
保留 4096 token 的语言模型输出预算及 384 token 的图像提示词预算。
Stage 2 和 GPT 评分器使用 `configs/openai-gpt4o-mini-judge.yaml`。

Stage 2 使用 `gpt4omini_stage2_constraints_v2` 提示词及逐事实的严格 JSON Schema：
每个 `fact_id` 对应独立分支，其 `evidence_refs` 只能从该事实自己的证据 ID 中选择；
没有证据时只能返回空列表。所有角色（含 `context`、`excluded`）的 `reason`
都必须包含非空白字符。辅助事实通过 `supporting_fact_ids` 连接，不把辅助事实的
证据 ID 混入主事实的 `evidence_refs`。输出结构仍为 `model_relation_plan_v1`，
保留现有 `candidate_quota` 字段，Stage 3 无需迁移。

本地校验一次收集能够独立判断的全部错误，包含 JSON Pointer 路径、实际值、
合法证据的完整列表、错误引用所属事实、支持目标的当前角色，以及配额实际总和
与目标值。`supporting_fact_ids` 只能指向同一响应中角色为 `secondary` 的其他事实；
`ready` 计划必须有主事实并恰好填满预算，`insufficient_evidence` 计划保持零配额。
这些跨字段业务规则仍执行程序校验，不以 Schema 校验替代。

若 JSON 或业务契约不合法，修正对话保留最初的 system/user 输入，将上次输出
作为 assistant 消息，再用独立 user 消息提交全部结构化错误，要求返回完整计划。
默认最多修正两轮（首次请求之外）；无效结果在 JSON 解析后与之前结果相同，
即使只改了空白或对象键顺序，也会提前停止。`GPT4oMiniRelationPlanner` 的
`max_contract_repairs` 可配置该上限。网络错误不进入业务契约修正循环。
程序不自动改写角色、引用或配额。模型调用日志的 `validation_attempts` 保存每轮
响应和完整错误列表，`contract_repair_stopped_reason` 记录重复响应或次数耗尽。

视觉评分向 GPT 发送大于 512 KiB 的 RGB PNG 时，使用质量 90 的 JPEG 传输，
保持原始像素分辨率，以减小请求体；小图、已有 JPEG 和含透明度的图片保持原格式。
生成的 PNG 文件及 CLIP、BLIP 的本地图像输入不受传输编码影响。

Stage 4 的独立 GPT 评分请求支持有限并发。在使用的 `stage4` YAML 中设置：

```yaml
gpt4omini_judge:
  max_concurrency: 10
```

主配置和 `stage4-model-fusion-gpu1-visible.yaml` 已设为 10；旧配置未指定该字段时
默认串行（1）。并发仅用于同一实体的独立候选评分，本地模型在主线程执行。
实际并发取配置上限和当前待评分候选数的较小值；每实体 5 张候选图时，最多同时
执行 5 个评分请求。该设置不增加候选图数量，也不启用多个实体同时运行。
模型响应按原候选顺序合并，调用日志记录实际墙钟耗时。线程中的 HTTP 请求通过
异步总超时限制整个请求；超时、429 和临时服务错误按照 `max_retries` 有界重试，
数值型 `Retry-After` 最多等待 60 秒。并发所需 `httpx` 已列入运行环境依赖文件。

Stage 0 → Stage 1 → Stage 2 的数据依赖保持顺序。并发提高独立请求的吞吐，
Stage 2 的事实引用、角色和配额仍执行同样的业务校验。上游中转接口的实际并发
能力需要通过小批运行验证；本次修改没有恢复已由用户停止的 50 实体任务。

主流程在 `kb` 环境使用 Transformers 4.51.3 加载 Qwen3-Reranker-4B。

首次安装或重建环境，在项目根目录执行：

```bash
conda activate kb
python -m pip install -r requirements-stage1-reranker.txt
```

运行入口自动读取项目 `.env`，保留调用者已设置的环境变量。
`.env` 中的 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 用于 GPT 服务，
两个 GPT 配置都明确指定 `gpt-4o-mini`。

直接运行相应的 `scripts/run_mkgw_image_test_*.sh`，启动脚本通过 GPT 配置运行，
无需启动本地 Qwen7B 服务。`MODEL_CONFIG`、`STAGE1_MODEL_CONFIG`、`STAGE2_MODEL_CONFIG`
可以分别覆盖各阶段配置文件。

批量测试可向 `python -m mmkgc.run` 追加 `--continue-on-error`：单个实体失败时
写入 `failures.json` 和 `metrics_summary.json`，随后继续处理其余实体；只要有
失败，整个任务最终返回非零退出码。未指定时仍在首次失败后停止。

旧版 Qwen7B 对照环境保留在 `.venv-qwen7b`，供显式指定旧配置的实验使用。
现有 vLLM 0.6.4.post1 的 Qwen2-VL 实现依赖 Transformers 4.46.1，
可通过以下命令重建该对照环境：

```bash
conda activate kb
python -m venv --system-site-packages .venv-qwen7b
.venv-qwen7b/bin/python -m pip install -r requirements-qwen7b-service.txt
bash scripts/run_qwen7b.sh
```

`QWEN_PYTHON` 可覆盖旧版 Qwen 服务所用的 Python 路径。
