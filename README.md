# TAOR / MMKGC

当前 MMKGC 五阶段多模态知识图谱补全代码。Stage 0、Stage 1 知识建模、Stage 2
关系精选及 Stage 3 文本／图像提示词生成默认采用本地 Qwen2-VL-7B-Instruct，
Stage 4 的 GPT 评分采用 GPT-4o-mini。文本和图片分别通过 `text` / `image`
模式执行，均支持 Think → Act → Observe → Reflect 的一次有限反馈闭环。

| 阶段 | 职责 |
| --- | --- |
| Stage 0 | Qwen2-VL-7B-Instruct 进行关系语义粗筛 |
| Stage 1 | Qwen2-VL-7B-Instruct 构建知识模型，Qwen3-Reranker-4B 匹配证据 |
| Stage 2 | Qwen2-VL-7B-Instruct 选择事实角色、证据引用和候选配额，程序编译候选专属事实检查清单 |
| Stage 3 | Qwen2-VL-7B-Instruct 生成文本或图像提示词，SDXL 生成候选图像；接收定向修复反馈 |
| Stage 4 | GPT-4o-mini 逐事实观察，结合 CLIP、BLIP 等评分器排序；修复后再次评估 |

Stage 2 按事实约束证据引用，保留每个事实的角色记录，并由模型单独选择固定数量
的候选锚点。程序统计锚点出现次数得到配额，辅助事实允许零配额。
一次反馈全部可检测的契约错误，最多请求模型修正两轮，对重复的无效响应提前停止。
千问遇到孤立的漏写事实、候选槽位或支持引用错误时，只请求模型补答对应字段，
小响应按实际事实或槽位键约束；保留其他合法决策，合并后再次执行完整校验。
当 primary 数超过图片预算时，千问显式将冲突事实划分为 primary / secondary，
同时重新选择槽位；程序验证划分完整且互斥，再合并角色和理由，其他事实与证据保持原样。

Stage 1 对漏写、重复或未知事实 ID 等契约错误同样提供具体修正反馈：列出完整
ID 集合及其原始三元组，最多额外修正两次。程序保留完整性校验，不补写模型 claim。
本地千问遇到可独立修正的错误时，只补答对应字段或以事实 ID 为键的 claim，
保留已合法的其他字段。损坏的 JSON 从原始输入重新构造修正请求，不回喂错误原文。
修正 Schema 要求 claim 非空，并限制画像和属性引用为原始证据目录中的 ID。
Stage 1 默认总时限 180 秒，初次请求、网络重试和业务修正共享；
Python API 可通过 `knowledge_stage_timeout_seconds` 调整。
字段补答单次输出按需限制为 768～2048 token。

Stage 2 每轮模型调用对超时、断连等瞬时故障默认重试一次，与业务修正分别计数，阶段总时限为 300 秒。
仅支持引用出错时，修正 Schema 按已确定角色限制合法支持 ID，修正请求不携带错误响应原文。
本地千问 Stage 3 遇到重复或截断时使用短请求、JSON 约束及有界重生成。

Stage 4 的 GPT 请求支持候选间并发，配置为 `max_concurrency: 10`。每实体
5 张候选图时实际最多并发 5 个评分请求。4／10／50 实体测试仍逐实体执行；
4580 实体补全脚本默认在 GPT 评分等待期间准备下一实体的 Stage 0–2，最多两个实体在途。
本地模型调用均在主线程顺序执行，避免同时占用 GPU。

Stage 3 支持通过 `inference.stage3_max_concurrency` 设置提示词请求并发。
千问主配置默认 3；保留的 GPT 对照配置为 5。SDXL 按候选顺序逐张生成图片。
千问配置通过 `inference.prompt_profile: qwen7b_original_v1` 恢复原来的 Stage 0 / 1 / 3
提示词，包括 system 指令、文本描述和 SDXL 提示词模板。GPT 对照配置继续使用
`gpt4omini_stage03_v2`。调用日志分别记录提示词版本和实际 `model_name`。
初次生成保留上述模板；Reflect 请求额外携带原内容和结构化修复上下文。

## Think → Act → Observe → Reflect

| 环节 | 当前实现 |
|---|---|
| Think | Stage 0–2：关系粗筛、知识建模、证据 rerank、事实角色和候选计划 |
| Act | Stage 3：生成文本，或先生成图像 prompt 再由 SDXL 出图 |
| Observe | Stage 4：对照当前候选应表达的事实逐项检查，并进行质量排序 |
| Reflect | 根据具体观察条目修改原文本或 prompt，反馈到 Stage 3，最多一轮 |

```mermaid
flowchart LR
    T["Think：知识与候选计划"] --> A["Act：文本或图像生成"]
    A --> O["Observe：逐事实检查与排序"]
    O --> F["Reflect：判断是否需要修复"]
    F -->|可修复问题，最多一轮| A
    F -->|无需修复或修复已完成| R["保留原结果，比较并选择最终候选"]
```

每个候选的 `verification_targets` 来自自己的 `must_include` 和 `should_include`，
通过 `check_id`、`fact_id` 和 `evidence_refs` 关联到原计划和证据。Observe 在同一次
GPT 评分请求中返回逐项 `fact_checks`：

| 状态 | 含义 |
|---|---|
| `supported` | 候选表达了该事实，并提供内容证据 |
| `contradicted` | 候选与提供的事实明确冲突 |
| `missing` | 当前模态能够检查、内容足以判断，但未表达该事实 |
| `unverifiable` | 当前模态无法验证完整事实，例如仅凭建筑外观判断设计者 |
| `uncertain` | 画面模糊、遮挡或文字证据不足，暂时无法判断 |

`unsupported_claims` 另行记录提供的证据无法支持的新增断言。文本证据使用
`text:` 标记，图片使用 `pixel:` 标记。图片 Judge 查看实际生成图，CLIP / BLIP
提供匹配分数；文本另有三元组抽取和 KG 对齐。图片侧目前依赖视觉 Judge 的逐项
判断，尚未启用视觉关系图对齐。

Reflect 当前采用 `REPAIR_CONFIDENCE = 0.8`。当前最佳候选有高置信的直接事实冲突、
必需事实遗漏或无依据断言时，即使它已通过原有筛选，也可触发修复；没有合格候选时，
从存在可修复条目的候选中选择一个。推荐事实遗漏、无法验证和不确定不单独触发修复。

修复操作分别为 `replace`、`add`、`remove`，并保留已经确认的内容。修复请求包含
该候选自己的原文本或原 prompt、完整 observation、具体操作和允许使用的事实证据；
`replace` / `add` 必须有原计划的证据引用。图片修复会生成新的 positive prompt，
再由 SDXL 出图，沿用 Stage 2 已确定的事实计划。

修复后再次 Observe，并保留两个版本的记录。修复版本必须实际解决至少一个原问题、
保留此前确认的事实且不引入新的高置信冲突或无依据断言，才允许参与选择；
先比较同一候选两个版本的事实问题，再比较质量分数。生成、评估失败或修复退化时保留原结果。
该阈值尚未经过真实样本校准，实际检测与修复质量仍需模型实验验证。

完整字段、约束和实现边界见 [事实观察与定向修复](docs/fact-observation-reflection.zh.md)。

## 环境与资源

完整的双环境安装、模型目录、GPU 分配、自检及启动步骤见
[环境配置 README](README_ENV.md)。

开发环境为 Python 3.11、Torch 2.5.1 / CUDA 12.1、Transformers 4.51.3、
Diffusers 0.31.0。依赖统一放在 `requirements.txt`，包含 CUDA 12.1 版 Torch、
vLLM、生成与评分组件及测试工具：

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写 `OPENAI_API_KEY` 和所使用的兼容接口 `OPENAI_BASE_URL`。
配置文件使用环境变量，不包含凭据。

模型权重、数据集、基线实验目录和生成结果不包含在代码仓库中。按配置准备本地资源：

```text
datasets/MKG-W/                         # entity2id.txt、relation2id.txt、train.txt 等
Qwen/Qwen2-VL-7B-Instruct/              # Stage 0 / 1 / 2 / 3 本地语言模型服务
Qwen3-Reranker-4B/                      # Stage 1 证据匹配
SDXL/stable-diffusion-xl-base-1.0/       # 图像生成
CLIP/                                  # 本地 CLIP 权重
blip-itm-large-coco/                    # 本地 BLIP 图文匹配模型
Sentence-BERT/all-mpnet-base-v2/         # 文本任务评分
models/bertscore/roberta-large/         # 可选 BERTScore
```

`configs/stage4-model-fusion.yaml` 配置评分器及设备；
`configs/qwen7b-mainline.yaml` 配置 Stage 0 / 3，`configs/qwen7b-stage1.yaml`
配置 Stage 1 知识建模，`configs/qwen7b-stage2.yaml` 配置 Stage 2；
三者默认连接本机 12345 端口的千问服务。
`configs/openai-gpt4o-mini-judge.yaml` 配置 Stage 4 GPT 评分器，输出预算为
4096 token，用于容纳逐事实 Observe 结果。
`configs/gpt4omini-mainline.yaml` 保留为显式指定的 GPT 对照配置。
需要对照时可传 `--stage2-model-config configs/openai-gpt4o-mini-judge.yaml`；
4 / 10 / 50 实体脚本也支持 `STAGE2_MODEL_CONFIG` 覆盖。
Stage 2 千问通过普通生成接口输出 JSON，提示词列出完整输出模板；
本地继续执行同样的事实完整性、证据、角色、候选槽位校验及有界修正。

## 运行

先在单独终端启动千问服务：

```bash
bash scripts/run_qwen7b.sh
```

原缺图名单的 4580 实体补全，在另一个终端执行：

```bash
bash scripts/run_mkgw_image_completion_4580.sh
```

默认后台运行，每实体 3 张初始候选图；触发 Reflect 时可额外生成一张修复图。
任务保留五阶段结果、原候选与修复候选、调用日志和耗时，不复制原图。
使用 `--resume outputs/<任务名>` 断点续跑，使用 `--prepare-only` 只准备任务。
详细说明见 `docs/mkgw-missing-image-output-layout.zh.md`。

服务使用独立的 `.venv-qwen7b` 环境，安装步骤见运行环境文档。
下面示例中 `CUDA_VISIBLE_DEVICES=1` 将流水线模型放在物理 GPU 1，
千问服务默认使用物理 GPU 0。设备和数据路径应与本地环境一致。

```bash
CUDA_VISIBLE_DEVICES=1 python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --graph-split train \
  --missing-mode image \
  --limit 4 \
  --candidate-budget 5 \
  --stage1-reranker-device cuda:0 \
  --sdxl-device cuda:0 \
  --stage4-config configs/stage4-model-fusion-gpu1-visible.yaml \
  --continue-on-error \
  --output-root outputs \
  --run-name image_smoke
```

运行结果写入 `outputs/<run-name>/`。指定 `--continue-on-error` 时记录单个实体失败后
继续处理其他实体；存在失败时进程最终返回非零退出码。各参数见 `python -m mmkgc.run --help`。

查看 Observe / Reflect 结果时，进入实体目录，读取阶段 JSON 的 `data`：

| 文件 / 字段 | 内容 |
|---|---|
| `3_relation_alternation_generation.json` 的候选 `verification_targets` | 该候选的事实检查清单 |
| `4_kg_aware_candidate_consistency_ranking.json` 的 `text_ranking` / `image_ranking` → `ranked_candidates[].observation` | 逐事实结果及证据；含 `violated_facts`、`missing_facts`、`unverifiable_facts` |
| 同一阶段的 `stage3_regeneration.feedback.candidate_feedback` | 发生修复时的原内容、具体操作及来源证据 |
| 同一阶段的 `stage3_regeneration` | 修复次数、前后状态和候选 ID；失败时包含异常原因 |
| `ranking_model_calls.jsonl` | Judge 调用记录及绑定事实后的 observation |

修复候选使用 `text_safe_001` / `image_safe_001` 等 ID，通过 `source_candidate_id`
关联原候选。`observation_version` 当前为 `fact_observation_v1`。

## 验证

```bash
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_fact_observation.py \
  tests/test_stage3_feedback.py \
  tests/test_grounded_model_mainline.py \
  tests/test_prompt_v2.py \
  tests/test_qwen7b_original_prompts.py \
  tests/test_stage3_concurrency.py \
  tests/test_judge_concurrency.py \
  tests/test_ranking_image_transport.py \
  tests/test_entity_pipeline.py \
  tests/test_single_mainline_contract.py

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_ranking_gpt4omini_judge.py -k gpt4omini_parser
```

2026-10-30，上述两组相关离线测试分别为 173 项和 20 项，合计 193 项通过，
覆盖逐事实观察、反馈隔离、必需遗漏、不可验证关系、修复退化、失败保留、千问提示词
及并发/流水线接口。测试使用模拟模型响应，未运行真实 GPT、千问或 SDXL 生成。
仓库还保留较早接口的测试，部分旧测试尚未适配当前主流程；该记录不代表全部测试通过。

Stage 2 千问加入定向修正后，7 个实体重新规划全部通过；先前版本为 4/7。
该验证未运行其余四阶段，也未重跑全部 50 个实体。
结果和限制见 [Stage 2 定向修正验证](docs/evaluation/2026-10-28-stage2-qwen-targeted-repair.zh.md)。

## 文档

- [从零配置与启动](README_ENV.md)
- [运行环境与并发设置](docs/stage1-stage2-runtime.zh.md)
- [事实观察与定向修复](docs/fact-observation-reflection.zh.md)
- [TAOR 方法章节草稿](docs/think-act-observe-reflect-method.zh.md)
- [批量补全输出与断点恢复](docs/mkgw-missing-image-output-layout.zh.md)
- [千问 7B 原始提示词](docs/qwen7b-original-prompts.zh.md)
- [GPT 对照提示词 v2](docs/gpt4omini-prompts-v2.zh.md)
- [Stage 2 契约与修正反馈](docs/evaluation/2026-10-27-stage2-contract-v2.zh.md)
- [Stage 1 事实 ID 完整性修正](docs/evaluation/2026-10-27-stage1-fact-id-repair.zh.md)
- [8 个失败实体的网络与生成恢复优化](docs/evaluation/2026-10-28-failed8-recovery.zh.md)
- [Stage 2 切换千问 7B](docs/evaluation/2026-10-28-stage2-qwen7b.zh.md)
- [千问 Stage 2 定向修正](docs/evaluation/2026-10-28-stage2-qwen-targeted-repair.zh.md)

`docs/evaluation/` 保留按日期记录的实验报告、结果和固定样本总名单。
其中包含旧版流程的历史实验，应按记录日期解读；当前运行方式以本 README 和代码为准。

本仓库保存当前工作代码快照；本地实验日志、运行产物和第三方模型资源单独管理。
