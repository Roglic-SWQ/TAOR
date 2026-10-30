# MMKGC 从零配置与启动

适用环境：Ubuntu 22.04 / Linux x86_64、NVIDIA GPU、Python 3.11。
按本机现有部署整理，更新于 2026-10-30。本机为两张 RTX 4090，各约 24 GB 显存。

**依赖文件只保留一个：`requirements.txt`。** 按下面的顺序执行即可从安装基础工具
一直到运行单实体任务。示例中的项目路径请替换成你的实际路径。

## 1. 安装系统工具和显卡驱动

```bash
sudo apt update
sudo apt install -y git curl bzip2 build-essential ubuntu-drivers-common
```

先检查显卡驱动：

```bash
nvidia-smi
```

如果能正常显示显卡和驱动，继续下一步。如果还没有 NVIDIA 驱动，安装后重启：

```bash
sudo ubuntu-drivers install
sudo reboot
```

重启后再次确认 `nvidia-smi` 正常。本机驱动为 580.126.09；依赖文件安装的是
PyTorch 的 CUDA 12.1 构建，不需要先手动安装完整 CUDA Toolkit。

## 2. 安装 Conda，创建主环境

没有 Conda 时，先安装 Miniconda；已有 Conda 可直接创建环境。

```bash
curl --fail --location \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  --output /tmp/miniconda-install.sh
bash /tmp/miniconda-install.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

创建项目使用的 `kb` 环境：

```bash
conda create -n kb python=3.11.10 pip -y
conda activate kb
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
```

以后新开终端，先加载 Conda 再激活 `kb`。如果 Conda 安装在其他位置，相应修改
`conda.sh` 的路径。项目源码需要事先下载或复制到上面的目录。

## 3. 安装全部 Python 依赖

在 `kb` 环境、项目根目录中执行：

```bash
python -m pip install -r requirements.txt
```

这一份文件已经包含 Torch、torchvision、vLLM、xformers、Transformers、
Diffusers、文本/图片评分组件、HTTP/配置工具及测试依赖。CUDA 12.1 的下载源
也写在文件中，不需要再分别安装其他 requirements 文件。

主要固定版本为：

| 组件 | 版本 |
|---|---|
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| vLLM / xformers | 0.6.4.post1 / 0.0.28.post3 |
| 主环境 Transformers / Tokenizers | 4.51.3 / 0.21.4 |
| Diffusers / Accelerate | 0.31.0 / 1.1.1 |
| Sentence-Transformers / BERTScore | 5.6.0 / 0.3.13 |
| OpenAI CLIP | 1.0.1 |

## 4. 创建千问服务环境

主流水线的 Qwen3-Reranker 需要 Transformers 4.51.3；当前千问服务组合使用
4.46.1，因此仍需要一个轻量服务环境。它共享主环境的软件包，只覆盖
Transformers 和 Tokenizers，不再维护另一份依赖文件。

保持 `kb` 激活，在项目根目录执行：

```bash
python -m venv --system-site-packages .venv-qwen7b
.venv-qwen7b/bin/python -m pip install transformers==4.46.1 tokenizers==0.20.1
```

上面第二条命令使用的是服务环境的 Python。后续运行主流水线继续使用 `kb`，
启动脚本会自行选择 `.venv-qwen7b/bin/python`，不需要手动激活服务环境。

检查两个环境的版本：

```bash
python -c 'import torch, transformers, tokenizers; print(torch.__version__, transformers.__version__, tokenizers.__version__)'
.venv-qwen7b/bin/python -c 'import torch, transformers, tokenizers; print(torch.__version__, transformers.__version__, tokenizers.__version__)'
```

预期分别为 `2.5.1+cu121 / 4.51.3 / 0.21.4` 和
`2.5.1+cu121 / 4.46.1 / 0.20.1`。

## 5. 准备模型和数据

Python 依赖与模型权重是分开的。模型需要下载到以下目录，也可以从已有机器
复制完整目录。请保留配置、分词器、权重分片及索引文件。

| 模型 | 项目中的目录 |
|---|---|
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct/` |
| Qwen3-Reranker-4B | `Qwen3-Reranker-4B/` |
| SDXL Base 1.0 | `SDXL/stable-diffusion-xl-base-1.0/` |
| CLIP ViT-B/32 | `CLIP/ViT-B-32.pt` |
| BLIP ITM Large COCO | `blip-itm-large-coco/` |
| Sentence-BERT | `Sentence-BERT/all-mpnet-base-v2/` |
| BERTScore RoBERTa Large | `models/bertscore/roberta-large/`，可选辅助指标 |

没有模型时，可在 `kb` 环境、项目根目录运行下面的下载代码。需要网络连接，
模型总量较大；已经复制好模型时跳过此步骤。

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
import clip

models = [
    ("Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"),
    ("Qwen/Qwen3-Reranker-4B", "Qwen3-Reranker-4B"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL/stable-diffusion-xl-base-1.0"),
    ("Salesforce/blip-itm-large-coco", "blip-itm-large-coco"),
    ("sentence-transformers/all-mpnet-base-v2", "Sentence-BERT/all-mpnet-base-v2"),
    ("FacebookAI/roberta-large", "models/bertscore/roberta-large"),
]
for repo_id, local_dir in models:
    snapshot_download(repo_id=repo_id, local_dir=local_dir)

clip.load("ViT-B/32", device="cpu", download_root="CLIP")
PY
```

数据集另行准备到 `datasets/MKG-W/`：

```text
datasets/MKG-W/
├── entity2id.txt
├── relation2id.txt
├── train.txt
├── valid.txt
├── test.txt
├── entity_info.json
├── relation_info.json
└── images/
```

其中 `entity2id.txt`、`relation2id.txt`、`train.txt` 至少应为非空文件，实体描述、
关系名称和图片按任务准备。单实体 CLI 默认使用 `datasets/MKG-W`；10 / 50 实体
及 4580 补全脚本默认使用 `datasets/MKG-W bak`，含空格的路径需要加引号。

SDXL 当前入口固定校验项目内的 `SDXL/stable-diffusion-xl-base-1.0/`，
需要完整 Diffusers 目录和 `model_index.json`，不能只放一个单独权重文件。

## 6. 配置 GPT 接口

项目根目录下创建 `.env`；已有该文件时直接编辑其中的配置：

```bash
if [ ! -f .env ]; then
  cp .env.example .env
fi
```

填入实际可用的地址和密钥：

```dotenv
OPENAI_API_KEY=填写你的实际密钥
OPENAI_BASE_URL=https://api.openai.com/v1
```

项目启动会读取 `.env`，Shell 中同名变量优先。默认模型名为 `gpt-4o-mini`；
图片 Observe 会发送实际图片，因此所用接口需要支持图像输入。

本地千问地址写在这三个文件中，默认均为 `http://127.0.0.1:12345/v1`：

- `configs/qwen7b-mainline.yaml`：Stage 0 / 3。
- `configs/qwen7b-stage1.yaml`：Stage 1 知识建模。
- `configs/qwen7b-stage2.yaml`：Stage 2 规划。

修改千问主机、端口或服务名时，同步修改这三个文件。

## 7. 启动千问服务

终端 A 中执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate kb
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC
bash scripts/run_qwen7b.sh
```

默认使用物理 GPU 0、端口 12345、bfloat16、显存利用率 0.85，
上下文长度为 12288。保持终端运行，等待模型加载完成。

终端 B 中检查服务：

```bash
curl --fail --silent --show-error http://127.0.0.1:12345/v1/models
```

## 8. 自检并运行一个实体

终端 B 同样激活 `kb`，进入项目目录。当前双卡分配是 GPU 0 跑千问，
GPU 1 跑 reranker、SDXL 和本地评分模型。

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate kb
cd /mnt/data/LRJ/Knowledge-Bridger-main/MMKGC

CUDA_VISIBLE_DEVICES=1 python -m mmkgc.doctor \
  --dataset-dir 'datasets/MKG-W' \
  --missing-mode image \
  --stage1-reranker-device cuda:0 \
  --sdxl-device cuda:0 \
  --stage4-config configs/stage4-model-fusion-gpu1-visible.yaml
```

自检通过后运行一个图片补全实体：

```bash
CUDA_VISIBLE_DEVICES=1 python -m mmkgc.run \
  --dataset-dir 'datasets/MKG-W' \
  --graph-split train \
  --missing-mode image \
  --limit 1 \
  --candidate-budget 3 \
  --stage1-reranker-device cuda:0 \
  --sdxl-device cuda:0 \
  --stage4-config configs/stage4-model-fusion-gpu1-visible.yaml \
  --output-root outputs
```

这里 `CUDA_VISIBLE_DEVICES=1` 隔离出物理 GPU 1，因此进程内部使用 `cuda:0`。
应搭配上面的 `gpu1-visible` 配置，不要同时使用设备写成 `cuda:1` 的评分器配置。

结果写入 `outputs/<自动生成的任务名>/`。文本补全将 `--missing-mode image`
改为 `text` 即可；文本模式不执行 SDXL 出图。

## 9. 批量运行与常见问题

确认单实体任务正常后，再运行 4580 实体补全：

```bash
DATASET_DIR="$PWD/datasets/MKG-W bak" \
  bash scripts/run_mkgw_image_completion_4580.sh
```

该脚本依赖原缺图名单及配套数据，默认每实体 3 张初始候选图、后台运行。
恢复任务可使用：

```bash
bash scripts/run_mkgw_image_completion_4580.sh --resume 'outputs/实际任务目录'
```

已有任务使用保存的代码与配置快照。更多说明见
[批量补全输出说明](docs/mkgw-missing-image-output-layout.zh.md)。

| 问题 | 处理方式 |
|---|---|
| `conda` 找不到 | 先 `source` 实际 Conda 安装位置的 `etc/profile.d/conda.sh` |
| 下载包或模型失败 | 检查目标机器能否访问 PyPI、PyTorch 下载源和模型仓库 |
| Qwen3 不被识别 | 主环境应为 Transformers 4.51.3，重新安装 `requirements.txt` |
| 千问服务导入错误 | 确认使用 `.venv-qwen7b/bin/python`，服务中为 4.46.1 / 0.20.1 |
| `invalid device ordinal` | 隔离 GPU 1 后，流水线统一用逻辑 `cuda:0` 和 `gpu1-visible` 配置 |
| 12345 连接失败 | 检查服务是否加载完成，以及三个千问配置的地址 |
| GPT 401 / 403 或图片请求失败 | 检查 `.env`、模型权限及接口图像能力 |
| 找不到 CLIP 权重 | 应存在 `CLIP/ViT-B-32.pt` |
| SDXL 路径错误 | 检查固定目录及完整 `model_index.json` |

本次修改校验了依赖文件及文档命令，没有在空机器上实际下载安装全部模型和环境。
安装步骤参考官方资料：`https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/`、
`https://www.anaconda.com/docs/getting-started/miniconda/install`、
`https://pytorch.org/get-started/previous-versions/`、
`https://docs.vllm.ai/en/v0.6.4/getting_started/installation.html`。
