# WN18RR 批量推理与全量补图

范围是测试三元组头／尾实体的去重并集，包含两个预测方向的 query 实体。当前原始对齐表中有 4,493 个真实缺图实体，缺文本为 0；模拟缺文本案例不混入全量任务。测试边只用于确定实体范围，事实证据仍来自 train。原始数据保持只读。

## 服务与调度

`scripts/run_qwen35_batch_server.sh` 启动一套 Qwen3.5-9B 权重，使用 vLLM 连续批处理，最多 10 个运行中序列。固定服务依赖见 `requirements-qwen35-batch.txt`。vLLM 的 [0.19.1 模型支持列表](https://docs.vllm.ai/en/v0.19.1/models/supported_models/)列出 Qwen3.5 多模态架构；支持列表不代替本机 GPU 验证。

原来的 Transformers 服务每次持有全局生成锁，多个 HTTP 请求只能排队；新入口用推理引擎的序列调度，不能仅凭 HTTP 线程数宣称批量推理生效。`scripts/check_qwen_batch_service.py` 检查相同十请求的串行／并发吞吐、逐请求返回归属、finish_reason、token 计数，以及可选的真实图片输入；同时采样引擎 `num_requests_running`，验证实际模型调度存在重叠。

`scripts/run_wn18rr_parallel_completion.py` 把实体分到 10 个互不重叠的 checkpoint 分片，每片使用已有单实体流水线。各 worker 拥有独立的来源规划器、模型调用记录、实体状态、图片预算账本和输出目录。每个 worker 内只同时等待一个远程请求，十个实体合计提供最多十路源审核／Judge 并发；没有再乘以候选数展开请求。

Reranker、SDXL、SBERT、CLIP、BLIP 各只有一份本地权重，由 `SharedLocalModels` 保护本地调用。Reranker 的审计列表按线程保留，失败记录也归回原实体。每个 worker 使用独立的 Stage 4 引擎控制状态；共享的 Judge 客户端不保存候选历史。Qwen 服务与这些本地模型共用调度器分配的一张 GPU。

当前服务配置针对 A100 80GB：BF16、最长 24,576 token、十个序列、一次最多 8,192 个批处理 token，服务显存预算为整卡的 45%，为流水线本地模型保留空间。使用 eager 执行，图像编码器使用 Torch SDPA；全量补图的 Qwen 输入没有目标原图。服务仍接受原始图像，单独验证视觉输入；跳过最坏情况多图启动 profiling，实际视觉请求须符合预检和配置预算。

批量服务使用独立虚拟环境，关闭 `include-system-site-packages`，避免继承其他 Torch 版本编译的 FlashAttention 扩展。启动脚本优先使用该环境中的工具；CUDA 运行库可通过只包含匹配版本库的目录复用，不加入旧环境的整个 site-packages。

## 恢复、复用与错误

首轮之前冻结实体清单、源数据、配置和执行源码哈希。已完成结果只有在原始数据一致、原 checkpoint 哈希完整、真实 PNG 可解码且 Stage 4 已判通过时才复用；结果和原始阶段记录复制到新 run，保留来源路径，不覆盖旧任务。

每实体先运行同提示词两种子 SDXL，必要时最多一次 Reflection；三个实际出图预算持久化，失败调用同样消耗预算。成功结果直接导出；内容仍不适用的实体明确结束为 `no_usable_candidate`。工程失败最多再执行一轮实体尝试，累计两次；重启时已完成的结果复核后跳过，中断状态转为可恢复的失败，不留下永久 running。

远程网关在实测中对相同合法请求偶发返回特定 `400 invalid_request_format`（明确说明 upstream provider 格式问题）；Stage 1、Stage 2 和 Judge 在原有传输重试次数内允许重放一次，保留错误与重试记录。普通参数错误、认证错误和本地 Qwen 的 400 不因此重试。

源审核不再因“未原样重复第一条词典释义”抛错。改写或拆成多条事实可正常继续，仍保留原始来源和 `lexical_definition_not_verbatim` 诊断。原始定义本身也可能混有无关词义；提示词要求按实体名和完整来源识别相关义项并记录歧义，不能把多个无关定义强行合成实体身份。这个修改不要求 Stage 3 把所有来源事实都写进输出。

## 运行入口

确认批量服务和真实闭环验证后，在已分配 GPU 的作业中准备并启动任务：

```bash
python -m scripts.run_wn18rr_parallel_completion --prepare \
  --root outputs/<新的全量目录> --workers 10 \
  --reuse outputs/<已有批次>
python -m scripts.run_wn18rr_parallel_completion --root outputs/<新的全量目录>
```

`--keys` 可指定真实缺图实体做实卡冒烟；`--limit` 使用固定随机种子做小样本。没有 `--limit`／`--keys` 时范围就是全部 4,493 个真实缺图实体。多次 `--reuse` 可接入多个旧批次或旧分片；原始候选数据、评分和失败均保留。

顶层 `summary.json` 和 `completion_manifest.jsonl` 汇总全体实体；各 `workers/00`～`workers/09` 保存已有流水线格式的完整阶段输出与 checkpoint，`reused/` 保留复用来源。任务提交、服务预检和全量的实际状态记录在当次 outputs 中；只有调度器终止状态、最终清单与实际文件能证明执行完成。
