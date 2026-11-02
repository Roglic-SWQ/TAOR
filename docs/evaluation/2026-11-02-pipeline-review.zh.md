# 2026-11-02 Pipeline 上传前审查

结论：本次改动可作为研发快照上传。检查了完整事实目录、跨模态来源绑定、相关性粗筛、
共享事实选择、多视角生成、共同评分参照和图片断点恢复；修正后未发现本次新增的阻断问题。
这不是全仓测试通过或真实模型效果通过的声明。

## 本轮修正

- 音频请求的 `input_audio.data` 原来包含 data URL 前缀，严格 base64 解码会失败。
  改为纯 base64；`audio_url` 继续使用 data URL。新增 WAV／MP3、两种传输形式的严格解码回归检查，
  并对两步媒体请求检查原始字节一致性。依据
  [OpenAI 官方音频输入示例](https://developers.openai.com/api/docs/guides/audio-chat-completions)。
- 并发图谱抽取测试的 mock 原来用候选 ID 作键，实际评分器按生成文本取值；已修正测试数据。
- 图片格式修复测试改为检查当前协议、原始事实保留和错误响应排除，移除旧提示词文案断言。
- README 改为准确描述共享事实、顺序协调、图片默认预算 3 次，以及历史提示词的兼容范围。

## 验证

Windows / Python 3.13 临时环境，模型请求使用 mock；媒体解码检查实际调用 ffmpeg／ffprobe。

专项选择：统一三元组、Stage 0、相关性粗筛、图片／视频／音频知识建模、共享事实选择，
以及上面修正的两个用例，共 **134 项通过**。

扩大扫描在相同环境、相同排除条件下对照上一提交 `1f92a9e` 的独立源码快照：

| 范围 | 通过 | 失败 | 未选择 |
| --- | ---: | ---: | ---: |
| 上一提交 | 799 | 179 | 12 |
| 本次最终源码 | 974 | 178 | 12 |

逐用例对照 JUnit 结果：**新增失败 0 项**；174 项新增或更名用例均通过。
已有 Python API 默认配置测试补全 mock 后通过。其余 178 项失败均能在上一提交复现，
包含旧 `both` 模式接口／历史提示词快照、缺少 torch 等模型依赖、Windows 路径与符号链接权限问题。
这些失败尚未全部修复。

扩大扫描命令：

```powershell
python -m pytest -q -p no:cacheprovider --tb=short `
  --ignore=tests/test_contract_output_recovery.py `
  --ignore=tests/test_entity_pipeline.py `
  --ignore=tests/test_missing_image_completion.py `
  --junitxml=<temporary-result.xml> -k "not launchers_route_stage_models"
```

三个忽略的模块会导入 Linux `fcntl`，本机首次收集测试时已复现导入失败；
12 项未选择用例属于 Bash 启动脚本路由。没有用通过数量掩盖这些未执行范围。
待上传 Python 文件语法解析、凭据模式扫描和 `git diff --check` 均通过。

## 实验与上传边界

本轮未调用真实 Qwen、GPT、reranker 或 SDXL，未重启批处理任务。
召回率、精选事实质量、生成质量、多视角互补性和下游收益仍需实际模型对照。
早先媒体小例及其已知音频感知错误单独记录在
[媒体两步验证](2026-11-02-media-two-step.zh.md)，不作为本轮修正后的接口实测。

上传范围为本次相关代码、测试、运行配置示例与文档；本地 `.claude/`、`WN18RR/`、
未纳入本轮实现的独立研究计划、凭据、模型权重和运行产物不进入本次提交。
