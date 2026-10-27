# Stage 3 并发优化与验证

## 当前行为

`configs/gpt4omini-mainline.yaml` 的 `inference.stage3_max_concurrency` 设为 5，
用于同一实体内独立的文本或图像提示词请求。未设置该字段的旧配置默认串行，
配置值必须是正整数，实际工作线程数不超过候选数。

执行顺序：准备候选上下文 → 并发请求 GPT → 按候选原顺序汇总 → SDXL 逐张出图。
初次生成和 Stage 4 反馈后的重新生成共用这一流程。图像生成仍在调用线程执行，
候选 ID、提示词、源证据、角度、配额和种子沿用原逻辑。

生成器的调用缓存和元数据按线程隔离，读取某个候选的日志时不会消费另一个
候选的记录。任何请求失败都会保留该候选 ID 与模型调用记录并向上抛出异常；
这一批提示词未成功完成时不启动本地出图。CLI `config.json` 新增
`stage3_max_concurrency`，用于确认实际运行参数。

修改文件：`mmkgc/adapters/qwen.py`、`mmkgc/stages/generation.py`、
`mmkgc/run.py`、`configs/gpt4omini-mainline.yaml`。

## 离线测试

新增 `tests/test_stage3_concurrency.py`，23 项测试覆盖：

- 文本与图像提示词，正常生成与安全重新生成，串行/并发 2/并发 5。
- 并发上限、候选顺序、元数据和模型日志一一对应、线程间日志隔离。
- 本地 GPU 始终在调用线程执行，所有提示词完成后才开始出图。
- 请求失败保留正确候选及审计信息，禁止用其他候选替代；配置的合法性与旧默认。

合计 187 项相关测试通过。JUnit：
`outputs/runtime_checks/20261027_stage3_concurrency_tests.xml`。

## 实际接口对照

使用之前 Q502245 / Andrei Makrov 保存的 5 个原始图像提示词请求，保持模型
`gpt-4o-mini`、输入提示词和 384 token 输出预算一致。

| 执行方式 | 成功候选 | 提示词请求总耗时 | 同时在途候选峰值 |
| --- | ---: | ---: | ---: |
| 串行 | 5 / 5 | 99.831 秒 | 1 |
| 并发 5 | 5 / 5 | 33.508 秒 | 5 |

本组提示词请求加速 **2.979 倍**，耗时减少 **66.44%**。
实测数据保存在 `outputs/runtime_checks/20261027_stage3_concurrency_benchmark.json`。

该测试只覆盖单实体、单组串行与并发对照，受接口负载、缓存和响应波动影响。
没有重新生成图片，也没有重新跑五阶段流程；不能将 2.979 倍直接解释为整个
Stage 3 或整条流水线的加速倍数。SDXL 耗时仍按原逻辑执行。

## 剩余失败实体

16 实体重测剩余的 5 个失败均发生在 Stage 2，与本次 Stage 3 优化分别处理。
逐实体原因见 `2026-10-27-remaining5-failures.zh.md`。
