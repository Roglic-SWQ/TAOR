# Judge / Reflection 与 main 合并验证

本次将四维 Judge、定向 Reflection、图像多样性调整及相关兼容修复，与 main 的 `6eae2a90c21fa106d07a4c2273db715d82fa6b5a` 合并。main 的目标实体显式绑定、来源抽取恢复、partial 状态保存和共享事实选择恢复均保留。功能提交为 `a231544c6`，合并在独立分支 `judge-reflection-update-20261106` 完成。

## 合并后检查

在 Windows / Python 3.13 环境执行：

| 检查 | 结果 |
| --- | --- |
| Stage 3、Stage 4、目标绑定、事实选择恢复、generation-only 等 24 个测试文件 | 418 通过，2 跳过 |
| `test_stage1_source_bound.py`、`test_triple_modeling.py`、`test_stage03_semantic.py` | 44 通过 |
| `python -m compileall -q mmkgc` | 通过 |
| Git 差异空白检查和未解决冲突检查 | 通过 |

两项跳过分别是 Windows 无符号链接创建权限，以及批处理检查需要 POSIX `fcntl` 文件锁。后者改为只在对应测试内导入批处理辅助模块，避免该平台限制阻止其他可移植测试加载；没有修改生产文件锁逻辑。

测试进程通过隔离的 PYTHONPATH 使用 Starlette 1.10.0 和 lm-format-enforcer 0.10.12。后者符合仓库所用 vLLM 0.6.4.post1 的版本约束，用于实际运行新增的本地 JSON 约束解码器测试。这些是本地测试依赖，不修改项目运行依赖。

本次是有针对性的功能回归，并非全仓库测试全部通过。此前 `test_qwen7b_original_prompts.py` 的 10 项原始请求哈希快照失败已在旧基线复现；本次未更新这些历史快照，也未将其计入上述通过数。

## 真实模型验证边界

2026-11-06 在合并前重新运行了真实 API：8 项核心用例覆盖 prompt 修复路由、固定删除问题复核、颜色矛盾、修正图验收和裁剪误报控制；另通过实际 Stage 3 适配器完成 prompt 改写并复查四项可视事实。main 此次变更集中在 Stage 1/2，合并后进行了上述软件回归，没有重新执行这批 API 用例。

裁剪图仍被模型误判为 missing，但程序未产生补画任务，不能声称修好了模型的视觉判断。图像为受控对照图；未运行原生 Qwen/SDXL 端到端生成，也未验证下游 MMKGC 指标改善。协议、聚合规则及修复验收条件见 [Stage 4 设计说明](../stage4-judge-rubrics.zh.md)。

原始 API 响应、JUnit XML 和合并前备份位于本地忽略目录 `outputs/reviews/`，不随代码提交；API 密钥和环境文件不纳入版本控制。
