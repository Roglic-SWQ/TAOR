# WN18RR 原生接入与 Qwen2-VL-7B／SDXL 冒烟

实体名称、关系与原图共同参与最终文本写作的配置和实测，见 [原图联合写作](evaluation/2026-11-07-entity-context-image-writing.zh.md)。启用 `configs/gpt6-luna-text-vision.yaml`；只发送已准入且与所选事实关联的原图。

用户指定 gpt-6-luna 后的生成／反思测试、显式模型配置和单卡入口见 [gpt-6-luna 实测](evaluation/2026-11-06-wn18rr-gpt6-luna.zh.md)。

2026-11-06 后续混合评审入口与实测见 [独立审核报告](evaluation/2026-11-06-wn18rr-hybrid.zh.md)。该入口保留本地 7B 生成和真实 SDXL，显式使用独立 API 做准入、像素复核与 Stage 4 Judge；以下早期冒烟记录仍按当时的配置解读。

## 数据输入

主 CLI 与 doctor 均支持 `--dataset-type wn18rr`，不需要生成 MKG-W 中间文件。

必需文件：`entities.txt`、`relations.txt`、`entity2text.txt`、`train.tsv`。实体／关系列表为每行一个原始键；文本为 `实体键<TAB>文本`，图谱为 `head<TAB>relation<TAB>tail`。实体键保持八位数字及前导零，内部整数编号只按 entities.txt 的行顺序生成，不改变对外 entity_key。

- 默认只加载 train.tsv；`--graph-split train-valid` 显式加入 dev.tsv，`all` 再加入 test.tsv。错误三元组与未知键明确报错，不静默丢弃。
- 可选 `wordnet-mlj12-definitions.txt` 格式为 `offset<TAB>__lemma_POS_sense<TAB>definition`。仅使用独立的 lemma 作为公开名称，不把该文件的定义作为附加证据。没有该文件时名称回落为原始实体键，不从隐藏目标文本反推名称。
- 可选 `relation2text.txt` 提供关系名称；不改变原始主客体或入／出方向。
- `image_alignment/entity2images.json` 按原始键映射相对路径列表。一旦存在对齐表，以该表为准，空列表／缺少键保持缺图；不从已排除的原始目录重新绑定图片。声明路径逃逸或文件缺失明确失败。
- 没有对齐表的原生数据，可沿用 images/<原始键>/ 下的图片查找。
- 缺 text 时只提供已有图片与合法 train 邻域，源文本清空，真值仅留在 ground_truth；缺 image 时，全部原图从生成输入移除，保留文本。所有对齐的可用图片都遵循现有多图 Stage 1 协议。

## 当前集群路径

数据解压到新目录：`datasets/wn18rr-import-20261106/WN18RR`，原始压缩包 `/medisco/linjiashi/Code/WN18RR.rar` 保留。该数据有 40,943 个实体、11 种关系、86,835 条 train 三元组。

模型下载到共享目录：

- `/medisco/linjiashi/models/Qwen/Qwen2-VL-7B-Instruct`
- `/medisco/linjiashi/models/SDXL/stable-diffusion-xl-base-1.0`
- 原有 `/medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B`

Qwen、SDXL 通过 ModelScope 的 Qwen/Qwen2-VL-7B-Instruct、AI-ModelScope/stable-diffusion-xl-base-1.0 下载。每个下载文件记录 SHA256，核对仓库清单给出的校验值；完整来源及清单位于本地输出目录。SDXL 下载标准 Diffusers 权重，不使用其他模型的替代图或占位图。项目本地 SDXL 根目录链接到上述共享目录；该链接只在本地 Git exclude 中忽略，不提交权重或绝对路径链接。

隔离推理环境 `/medisco/linjiashi/venvs/taor-inference` 继承已有 worldmm_env 的 Torch 2.7.1，并单独安装 Transformers 4.51.3、Diffusers 0.31.0、Accelerate 1.1.1。不改动已有环境的包；继承环境中 worldmm 自身的版本要求与此覆盖不一致，因此该隔离环境仅用于 TAOR，不用于运行 WorldMM。Qwen 服务使用 `/medisco/linjiashi/venvs/taor-qwen7b`：它继承 skillrl 的 Torch 2.8.0，并独立安装 vLLM 0.11.0。vLLM 的 Qwen2-VL 在 A100 的 80 维视觉注意力头上回落到 upstream FlashAttention，但该分支传入 CPU 的 cu_seqlens，实际预热报 `cu_seqlens_q must be on CUDA`。`scripts/prepare_taor_qwen_vllm.py` 仅在这个 TAOR venv 内将该元数据移到 q.device，不修改共享包、模型权重或提示词。原／修复文件哈希记录在 qwen-vllm-patch.json。服务启用项目 RequestDeadlineMiddleware，保留新版 request_id 与超时取消协议。

重建该服务环境：

```bash
/medisco/linjiashi/miniforge3/envs/skillrl/bin/python -m venv --system-site-packages /medisco/linjiashi/venvs/taor-qwen7b
/medisco/linjiashi/venvs/taor-qwen7b/bin/python -m pip install --ignore-installed --no-deps vllm==0.11.0
python scripts/prepare_taor_qwen_vllm.py
```

以上仅对新建／专用环境执行；不要改装他人的已运行环境。

## 可复跑的单卡入口

原有集群操作说明见 `/medisco/linjiashi/WorldMM/ENVIRONMENT.md` 的 ClusterX 部分；本机旧 clusterx 脚本指向不存在的解释器，当前通过现存 Python 3.13 和原 ClusterX site-packages 运行：

```bash
export CLUSTERX_CFG_PATH=/medisco/linjiashi/clusterx/clusterx.yaml
export PYTHONPATH=/medisco/linjiashi/clusterx/.venv/lib/python3.13/site-packages
/medisco/linjiashi/worldmm_miniforge3/bin/python3 -m clusterx.cli --help
```

配置内凭据仅由 SDK 读取，不打印或复制到 Git。任务申请一节点、一 GPU，队列 `queue-p-reserved-medisco`，不继承当前无 GPU 的 CUDA 掩码；不重启／停止其他旧任务。镜像与 AFS 挂载取自用户已有成功作业的非凭据配置。

实际 GPU 任务入口为 `scripts/run_wn18rr_cluster_smoke.sh`，每次新建 attempt 目录，保留失败日志和已生成结果，已提交任务 ID 与资源参数保存在本地 `cluster-job.json`。脚本等待准备文件、启动本地真实 Qwen 服务，执行 doctor，然后分别跑文本和图片单实体任务；退出时仅关闭自己启动的 Qwen 服务，任务进程退出后释放申请资源。

本次冻结单实体 `02436645`（water deer），一张实际已对齐图片，用于工程冒烟，非质量对照采样；原始记录、train 哈希和源策略在 `frozen-smoke-inputs.json`。图片生成参数为 1024×1024、20 steps、guidance 7.5、base seed 20260708；候选使用适配器派生的实际种子并写入元数据。原生 Qwen 默认 token 预算保留。

运行入口的主要参数为：

```bash
python -m mmkgc.run \
  --dataset-type wn18rr \
  --dataset-dir datasets/wn18rr-import-20261106/WN18RR \
  --graph-split train \
  --entity-manifest outputs/wn18rr-onboarding-20261106/one-entity.json \
  --limit 1 --candidate-budget 1 \
  --missing-mode image --skip-stage4 \
  --stage1-reranker-model-dir /medisco/linjiashi/models/Qwen/Qwen3-Reranker-4B \
  --stage1-reranker-device cuda:0 --sdxl-device cuda:0 \
  --sdxl-model-dir SDXL/stable-diffusion-xl-base-1.0 \
  --sdxl-width 1024 --sdxl-height 1024 --sdxl-steps 20 \
  --output-root outputs/wn18rr-onboarding-20261106/runs \
  --run-name <新的运行名>
```

文本任务替换为 `--missing-mode text`。默认输出目录拒绝覆盖已有 run。

本次先使用新分支已有的 `--skip-stage4` 验证 Stage 0–3 与真实模型生成；Stage 4／Reflect 评审、质量对照及全量生成不属于本次冒烟的通过结论。没有 judge 凭据时不把缺失评分伪装成验收通过。

## 验证与运行结果

针对性测试入口：`tests/test_wn18rr_data.py`（20 项）；加上现有多图 Stage 1、共享事实选择和三元组建模，共 **103 passed**。另检查旧 doctor 测试时发现 `test_doctor_minimal_run_is_ok` 仍假设默认不探测模型、不启用评分，与本分支当前默认值不同；未改动其过时断言，也没有宣称全库通过。日志与实际运行状态写在 `outputs/wn18rr-onboarding-20261106/`。`attempt-*/gpu-smoke-completed.json` 只有在真实生成的 PNG 可以解码且尺寸为 1024×1024 后才写入；排队、doctor 通过、提示词生成本身均不代表真实出图成功。


启动阶段失败也保留：最初不带 `-ssh` 的历史镜像标签实际拉取失败；随后 skillrl 原环境触发上述视觉预热错误；旧 graphr1 vLLM 0.6.3 能启动，但不接受新分支 Stage 1 的 request_id 字段，首请求返回 400。后续使用独立环境与新版协议，不删除或替换这些失败记录。每次前一作业已终止／失败释放 GPU 后才提交下一次，始终最多申请一张卡。


## 本次真实实测结果（2026-11-06，上海时间）

成功作业：`taor-wn18rr-qwen7b-sdxl-1106-r3`，单张 NVIDIA A100-SXM4-80GB。

| 项目 | 实测结果 |
| --- | --- |
| Native WN18RR 接入与相关回归 | 103 passed |
| GPU doctor | 数据、Qwen／Stage 1／Stage 2 服务、reranker 文件、SDXL、CUDA 均通过；Stage 4 显式跳过 |
| 单实体补文本 | 1 完成、0 失败，1 条真实 Qwen 文本 |
| 单实体补图片 | 1 完成、0 失败，1 张真实 SDXL PNG；已实际打开检查像素 |
| Qwen chat 请求 | 成功尝试中 9 次 HTTP 200，0 次 400／500；此前启动失败另列保留 |
| 图片参数 | 1024×1024，20 steps，guidance 7.5，实际派生 seed 20260709 |
| SDXL 适配器记录的推理耗时 | 2,983 ms，不包含权重与 pipeline 加载耗时 |
| Stage 4／Reflect | 未执行，ranking_status=not_evaluated |

图片产物：`outputs/wn18rr-onboarding-20261106/attempt-r3/runs/image-qwen7b-sdxl/entities/02436645/generated_images/image_001.png`。同目录 JSON 包含实际提示词、种子、模型路径和无 fallback 标记。

文本产物：`outputs/wn18rr-onboarding-20261106/generated-text.txt`；机器汇总为 `smoke-summary.json`，每个 run 的完整阶段、调用和 metrics 均保留。原压缩包的完整 SHA256 为 `2bb6130b0da0e7ff62d09eab7c9b996f22f3f58ed766596011ec567f8027063b`。

**工程链路已跑通，内容质量未通过。** 实际文本同时将 water deer 写成 rodent 与 marsupial；来源是 Stage 1 从图片抽取的 mf:0001／mf:0002，Stage 2 将其与合法图谱 `(water deer, hypernym, mouse deer)` 一起保留，Stage 3 又复述了这些冲突分类。没有用隐藏真值纠正提示词或删掉失败观察。实际生成图是沼泽中的鹿状动物，能证明真实出图；精确物种身份和事实一致性尚未建立，不能把通用鹿图称为身份保真。下一步应先处理来源抽取与分类冲突，再开展质量验收和多实体比较；本次未启动全量。
