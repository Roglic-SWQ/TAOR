# TAOR / MMKGC

当前 MMKGC 五阶段多模态知识图谱补全代码。语言模型主流程采用 GPT-4o-mini，
本地模型完成证据匹配、图像生成及部分候选评分。

| 阶段 | 职责 |
| --- | --- |
| Stage 0 | GPT-4o-mini 进行关系语义粗筛 |
| Stage 1 | GPT-4o-mini 构建知识模型，Qwen3-Reranker-4B 匹配证据 |
| Stage 2 | GPT-4o-mini 选择事实角色、证据引用和候选配额 |
| Stage 3 | GPT-4o-mini 生成文本或图像提示词，SDXL 生成候选图像 |
| Stage 4 | GPT-4o-mini、CLIP、BLIP 等评分器进行候选排序 |

Stage 2 按事实约束证据引用，一次反馈全部可检测的契约错误，最多请求模型修正
两轮，并对重复的无效响应提前停止。程序保留模型的事实角色和候选分配决策。

Stage 4 的 GPT 请求支持候选间并发，配置为 `max_concurrency: 10`。每实体
5 张候选图时实际最多并发 5 个评分请求，实体之间仍按顺序处理。

## 环境与资源

开发环境为 Python 3.11、Torch 2.5.1 / CUDA 12.1、Transformers 4.51.3、
Diffusers 0.31.0。已有相应 Torch 环境后，可安装其余依赖：

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写 `OPENAI_API_KEY` 和所使用的兼容接口 `OPENAI_BASE_URL`。
配置文件使用环境变量，不包含凭据。

模型权重、数据集、基线实验目录和生成结果不包含在代码仓库中。按配置准备本地资源：

```text
datasets/MKG-W/                         # entity2id.txt、relation2id.txt、train.txt 等
Qwen3-Reranker-4B/                      # Stage 1 证据匹配
SDXL/stable-diffusion-xl-base-1.0/       # 图像生成
CLIP/                                  # 本地 CLIP 权重
blip-itm-large-coco/                    # 本地 BLIP 图文匹配模型
Sentence-BERT/all-mpnet-base-v2/         # 文本任务评分
models/bertscore/roberta-large/         # 可选 BERTScore
```

`configs/stage4-model-fusion.yaml` 配置评分器及设备；
`configs/gpt4omini-mainline.yaml`、`configs/openai-gpt4o-mini-judge.yaml`
分别配置生成模型和评分模型。

## 运行

下面的图像任务示例使用两张 GPU：证据匹配和候选排序使用 GPU 0，SDXL 使用 GPU 1。
设备和数据路径应与本地环境一致。

```bash
python -m mmkgc.run \
  --dataset-dir datasets/MKG-W \
  --graph-split train \
  --missing-mode image \
  --limit 4 \
  --candidate-budget 5 \
  --stage1-reranker-device cuda:0 \
  --sdxl-device cuda:1 \
  --stage4-config configs/stage4-model-fusion.yaml \
  --continue-on-error \
  --output-root outputs \
  --run-name image_smoke
```

运行结果写入 `outputs/<run-name>/`。指定 `--continue-on-error` 时记录单个实体失败后
继续处理其他实体；存在失败时进程最终返回非零退出码。各参数见 `python -m mmkgc.run --help`。

## 验证

```bash
python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_stage2_contract.py tests/test_grounded_model_mainline.py \
  tests/test_judge_concurrency.py tests/test_ranking_profiles.py \
  tests/test_ranking_image_transport.py tests/test_runtime_model_routing.py \
  tests/test_single_mainline_contract.py tests/test_prompt_v2.py \
  tests/test_stage03_semantic.py tests/test_qwen_adapter.py tests/test_stage3_feedback.py
```

上述 164 项相关离线测试在 2026-10-27 的本地环境通过。仓库还保留较早接口的测试及
历史设计文档，部分旧测试尚未适配当前五阶段主流程；该记录不代表整个测试集均通过。

## 文档

- [运行环境与并发设置](docs/stage1-stage2-runtime.zh.md)
- [Stage 0 / 1 / 3 提示词](docs/gpt4omini-prompts-v2.zh.md)
- [Stage 2 契约与修正反馈](docs/evaluation/2026-10-27-stage2-contract-v2.zh.md)
- [五阶段设计](docs/five-stage-pipeline-current-design.zh.md)

本仓库保存当前工作代码快照；本地实验日志、运行产物和第三方模型资源单独管理。
