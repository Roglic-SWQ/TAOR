# 千问 7B、GPT-4o-mini 与 Reranker 运行环境

2026-10-27 实验后，CLI、Python API 和 4 / 10 / 50 实体启动脚本的 Stage 0 / 1 / 3
默认模型恢复为 `Qwen/Qwen2-VL-7B-Instruct`；Stage 2 和 Stage 4 的 GPT 评分保持
`gpt-4o-mini`。随后按用户要求，千问配置的提示词也恢复为修改前的版本，
使用 `inference.prompt_profile: qwen7b_original_v1` 标记。实际模型看调用日志的
`model_name`；原始模板说明见 `docs/qwen7b-original-prompts.zh.md`。
显式 GPT 对照配置仍采用 `gpt4omini_stage03_v2`，说明见 `docs/gpt4omini-prompts-v2.zh.md`。

| 阶段 | 当前模型与职责 |
| --- | --- |
| Stage 0 | Qwen2-VL-7B-Instruct 做关系语义粗筛 |
| Stage 1 | Qwen2-VL-7B-Instruct 构建知识模型；本地 Qwen3-Reranker-4B 做证据评分 |
| Stage 2 | GPT-4o-mini 选择关系角色并分配候选数量 |
| Stage 3 | Qwen2-VL-7B-Instruct 生成文本或图像提示词；SDXL 生成图像 |
| Stage 4 | GPT-4o-mini 与 CLIP、BLIP 等评分器参与候选排序 |

Stage 0 / Stage 3 使用 `configs/qwen7b-mainline.yaml`，Stage 1 知识建模使用
`configs/qwen7b-stage1.yaml`，两者默认连接 `http://127.0.0.1:12345/v1`。
主配置保留 4096 token 的语言模型输出预算及 384 token 的图像提示词预算。
Stage 2 和 GPT 评分器使用 `configs/openai-gpt4o-mini-judge.yaml`。

`qwen7b-mainline.yaml`、`qwen7b-stage1.yaml` 和环境变量版 `qwen-local.yaml`
均显式启用原始千问提示词。Stage 0 / 1 恢复原始 system、任务说明、输入组织和
格式修正提示；Stage 3 恢复原来的简短文本模板、图像 JSON 输入、generation brief
及正向提示词补充片段。角色、事实引用、输出结构和 SDXL 参数继续沿用当前管线。

Stage 3 的文本生成和图像提示词请求支持并发。在主模型配置中设置：

```yaml
inference:
  stage3_max_concurrency: 5
```

`qwen7b-mainline.yaml` 默认设为 1，恢复千问的串行请求调度；需要时可调整此字段。
保留的 `gpt4omini-mainline.yaml` 为 5；未指定该字段的配置默认串行（1）。
每个实体按配置准备最多 5 个候选的文本或图像提示词，再按原候选顺序汇总，
SDXL 在调用线程逐张生成，避免多个线程同时操作同一 GPU 模型。
初次生成和 Stage 4 反馈后的重新生成都采用这一调度。
调用缓存及元数据按线程隔离，候选 ID、提示词响应和日志保持对应；请求失败时
保留该候选的调用记录并抛出异常，不用其他候选结果代替。CLI 的 `config.json`
记录实际 `stage3_max_concurrency`。这项配置只作用于 Stage 3，实体和其他阶段
仍按原来的依赖顺序执行；SDXL 的图像尺寸、步数、种子和候选预算不变。

Stage 2 使用 `gpt4omini_stage2_candidate_slots_v3` 提示词及逐事实的严格 JSON Schema：
每个 `fact_id` 对应独立分支，其 `evidence_refs` 只能从该事实自己的证据 ID 中选择；
没有证据时只能返回空列表。所有角色（含 `context`、`excluded`）的 `reason`
都必须包含非空白字符。辅助事实通过 `supporting_fact_ids` 连接，不把辅助事实的
证据 ID 混入主事实的 `evidence_refs`。

模型返回 `plan` 对象，其中 `relation_plans` 保留所有事实及角色，但不再填写
`candidate_quota`。另用 `candidate_anchors` 指定每个候选锚定的事实 ID；正常
计划恰好包含 `candidate_budget` 个 ID，允许重复。程序仅统计这些 ID 的出现
次数得到配额。例如 `[A,A,A,B,B]` 得到 A 为 3、B 为 2，其余事实为 0。
辅助事实可以保留并支撑其他候选，配额为 0 不表示删除该事实。

Schema 通过 `plan` 下的两个分支区分状态：`ready` 要求槽位数恰好等于预算；
`insufficient_evidence` 要求空槽位、所有角色仅为 `context` / `excluded` 且没有
支持链接。本地继续检查每个事实恰好出现一次、每个主事实至少锚定一个候选，
以及槽位和支持链接的角色合法性。

内部输出仍为 `model_relation_plan_v1`，保留由模型槽位计数得到的 `candidate_quota`，
并增加 `allocation_method: model_candidate_slots_v2` 及原始 `candidate_anchors`。
Stage 3 按模型给出的槽位顺序生成，Stage 2 产物保存槽位和派生配额供审计。

本地校验一次收集能够独立判断的全部错误，包含 JSON Pointer 路径、实际值、
合法证据的完整列表、错误引用所属事实、支持目标的当前角色，以及槽位实际数量
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

运行默认 CLI、Python API 或 `scripts/run_mkgw_image_test_*.sh` 前，需要启动本地
Qwen7B 服务：`bash scripts/run_qwen7b.sh`。脚本默认将千问放在物理 GPU 0，
4 / 10 / 50 实体流水线通过 `CUDA_VISIBLE_DEVICES=1` 使用物理 GPU 1，避免将
常驻千问服务与 Reranker、SDXL 和本地评分模型放在同一张卡。
`MODEL_CONFIG`、`STAGE1_MODEL_CONFIG`、`STAGE2_MODEL_CONFIG` 可以分别覆盖阶段配置。
如服务地址不同，需要修改两个千问配置或分别指定替代配置。

保留 GPT 对照运行方式：

```bash
MODEL_CONFIG="$PWD/configs/gpt4omini-mainline.yaml" \
STAGE1_MODEL_CONFIG="$PWD/configs/gpt4omini-mainline.yaml" \
bash scripts/run_mkgw_image_test_4.sh
```

直接 CLI 的对应覆盖参数为 `--model-config` 和 `--stage1-model-config`。
已启动的 16 实体 GPT 测试使用输出目录中的配置快照，继续按该轮原配置执行；
上述默认值变更作用于后续新进程。

批量测试可向 `python -m mmkgc.run` 追加 `--continue-on-error`：单个实体失败时
写入 `failures.json` 和 `metrics_summary.json`，随后继续处理其余实体；只要有
失败，整个任务最终返回非零退出码。未指定时仍在首次失败后停止。

Qwen7B 服务环境保留在 `.venv-qwen7b`，与流水线的 `kb` 环境分离。
现有 vLLM 0.6.4.post1 的 Qwen2-VL 实现依赖 Transformers 4.46.1，
可通过以下命令重建服务环境：

```bash
conda activate kb
python -m venv --system-site-packages .venv-qwen7b
.venv-qwen7b/bin/python -m pip install -r requirements-qwen7b-service.txt
bash scripts/run_qwen7b.sh
```

`QWEN_PYTHON` 可覆盖 Qwen 服务所用的 Python 路径。
