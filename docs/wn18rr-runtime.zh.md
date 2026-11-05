# WN18RR 原生接入与 Qwen2-VL-7B／SDXL 冒烟

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

隔离推理环境 `/medisco/linjiashi/venvs/taor-inference` 继承已有 worldmm_env 的 Torch 2.7.1，并单独安装 Transformers 4.51.3、Diffusers 0.31.0、Accelerate 1.1.1。不改动已有环境的包；继承环境中 worldmm 自身的版本要求与此覆盖不一致，因此该隔离环境仅用于 TAOR，不用于运行 WorldMM。Qwen 服务使用已有 skillrl 的 vLLM 0.11.0／Torch 2.8.0。

## 可复跑的单卡入口

原有集群操作说明见 `/medisco/linjiashi/WorldMM/ENVIRONMENT.md` 的 ClusterX 部分；本机旧 clusterx 脚本指向不存在的解释器，当前通过现存 Python 3.13 和原 ClusterX site-packages 运行：

```bash
export CLUSTERX_CFG_PATH=/medisco/linjiashi/clusterx/clusterx.yaml
export PYTHONPATH=/medisco/linjiashi/clusterx/.venv/lib/python3.13/site-packages
/medisco/linjiashi/worldmm_miniforge3/bin/python3 -m clusterx.cli --help
```

配置内凭据仅由 SDK 读取，不打印或复制到 Git。任务申请一节点、一 GPU，队列 `queue-p-reserved-medisco`，不继承当前无 GPU 的 CUDA 掩码；不重启／停止其他旧任务。镜像与 AFS 挂载取自用户已有成功作业的非凭据配置。

实际 GPU 任务入口为 `scripts/run_wn18rr_cluster_smoke.sh`，已提交任务 ID 与资源参数保存在本地 `cluster-job.json`。脚本等待准备文件、启动本地真实 Qwen 服务，执行 doctor，然后分别跑文本和图片单实体任务；退出时仅关闭自己启动的 Qwen 服务，任务进程退出后释放申请资源。

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

针对性测试入口：`tests/test_wn18rr_data.py`（20 项）；加上现有多图 Stage 1、共享事实选择和三元组建模，共 **103 passed**。另检查旧 doctor 测试时发现 `test_doctor_minimal_run_is_ok` 仍假设默认不探测模型、不启用评分，与本分支当前默认值不同；未改动其过时断言，也没有宣称全库通过。日志与实际运行状态写在 `outputs/wn18rr-onboarding-20261106/`。`gpu-smoke-completed.json` 只有在真实生成的 PNG 可以解码且尺寸为 1024×1024 后才写入；排队、doctor 通过、提示词生成本身均不代表真实出图成功。
