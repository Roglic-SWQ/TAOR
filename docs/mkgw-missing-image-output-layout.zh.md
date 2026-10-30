# MKG-W 原缺图名单：4580 实体的输出结构

原缺图名单为 `configs/mkgw-test-original-missing-images-4580.json`，共 4580 个实体。
`scripts/complete_mkgw_missing_images.py` 默认每实体生成 3 张候选图。
五阶段文件名及实体目录层级参考 `outputs/mkgw_image_test_50_20261014_fixed2`，
内容沿用当前五阶段代码。

## 启动与恢复

在 MMKGC 项目目录中，用第一个终端启动千问服务：

```bash
bash scripts/run_qwen7b.sh
```

服务就绪后，在另一个终端启动全量任务：

```bash
bash scripts/run_mkgw_image_completion_4580.sh
```

默认使用 `kb` 环境，后台运行，每实体 3 张候选图；千问位于 GPU 0，流水线位于 GPU 1。
主配置 `configs/qwen7b-mainline.yaml` 的 Stage 3 提示词请求并发为 3，SDXL 绘图按候选顺序执行。
新任务默认启用实体流水线：实体 A 的 Stage 4 GPT 评分请求在后台等待时，主线程准备
实体 B 的 Stage 0–2。最多两个实体在途，所有本地 GPU 模型仍顺序执行。
提前准备的实体独立保存日志和检查点，失败不会改变当前实体的评分结果。
终端会打印任务目录、后台 PID、日志和报告位置。所有任务均使用原缺图名单的 4580 个实体。
启动前校验千问实际响应；服务未就绪时保留已准备的任务，服务就绪后使用该目录恢复。

```bash
bash scripts/run_mkgw_image_completion_4580.sh --resume outputs/<任务名>
bash scripts/run_mkgw_image_completion_4580.sh --resume outputs/<任务名> --retry-failed
```

恢复时使用任务保存的配置与代码快照，保留已完成实体。
旧任务缺少 `entity_pipeline` 配置时继续串行执行；更新项目代码不会自动改变旧任务快照。
直接使用 Python 创建新任务时，可传 `--serial-entities` 关闭实体流水线。
`--retry-failed` 同时重试失败或没有合格图片的实体。
`--foreground` 在当前终端运行；`--prepare-only` 只准备或校验任务，不连接模型服务。
指定 `MMKGC_PYTHON=/path/to/python` 可跳过 conda 激活，其他选项见脚本的 `--help`。

## 文件保存

```text
<补全输出目录>/
  config.json
  results.jsonl
  metrics_summary.json
  failures.json
  entities/<QID>/
    0_coarse_relation_filtering.json
    1_knowledge_modeling.json
    2_fine_grained_relation_selection.json
    3_relation_alternation_generation.json
    4_kg_aware_candidate_consistency_ranking.json
    final.json
    entity_description.json
    entity_description.txt
    model_calls.jsonl
    image_model_calls.jsonl
    ranking_model_calls.jsonl
    generated_images/
      image_001.png
      image_001.json
      image_002.png
      image_002.json
      image_003.png
      image_003.json
  completed_images/<QID>/
    image.png
    metadata.json
  checkpoints/<QID>.json
  entity_timing.csv
  timing_report.html
  timing_summary.zh.md
  progress.json
  attempts/attempt_0001/entities/<QID>/
```

`entities/<QID>` 是指向该实体当前尝试产物的目录链接，图片只保存一份。
断点续跑时，已完成实体继续保留；重试实体的入口更新到新尝试，旧尝试保存在 `attempts/`。
没有有效最佳候选时不导出 `completed_images/<QID>/image.png`。
候选均被拒绝的实体仍保存五阶段结果与候选图。
流程中途失败的实体保存 `failure.json`，同时出现在根目录 `failures.json` 和检查点中。
调用日志在有相应调用时写入。

该名单中的历史 `kb_generated.jpg` 继续留在源数据目录，不作为真实图像答案，
不复制到新输出。不创建 `original_images/` 或 `original_images_manifest.json`。
`entity_description` 保存实体名称和可用文本，供查看生成依据。

逐实体总耗时、各阶段耗时、成功及失败状态、候选预算和断点检查点继续保留。
流水线模式下，逐实体耗时包括排队时间，Stage 4 墙钟耗时包含准备下一实体的时间，
实体耗时相互重叠，不能直接求和作为整批耗时。
`entity_timing.csv` 额外记录 `prepare_seconds`、`queue_wait_seconds` 和 `prefetch_seconds`；
`progress.json` 的 `active_entities` 展示两个实体各自的阶段。
实际速度使用 `active_wall_seconds / timed_attempts_completed` 比较整批吞吐，包含失败尝试，
排除断点间停机；从启用此计时方式后累计，不混入旧版本已完成的实体。
报告中的流程完成、图片通过评分和最终导出分别统计。
若触发反馈重生成，额外候选及对应日志也保留在该实体目录中。
