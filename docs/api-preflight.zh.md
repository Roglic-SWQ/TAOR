# API 优先预检

按用户要求，先使用 API 修复并验收逻辑，再申请 GPU 验证本地 Qwen／SDXL；API 配置就绪或 HTTP 200 不等于质量验收通过。预检不会自动申请 GPU。

本地配置由 `--env-file` 读取，不复制密钥到仓库。当前使用用户提供的 `/medisco/linjiashi/Code/.env.standardization.local`，包含 OPENAI_BASE_URL、OPENAI_API_KEY、OPENAI_MODEL 和 OPENAI_MIN_REQUEST_INTERVAL_MS。文件内容不打印；OPENAI_MODEL 作为默认模型名，可用 --model 覆盖。请求按配置间隔串行限速，传输包装不更改请求／响应数据。

```bash
cd /medisco/linjiashi/Code/TAOR
PYTHONPATH=. /medisco/linjiashi/venvs/taor-inference/bin/python \
  scripts/run_wn18rr_api_preflight.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --proxy-from-bashrc /root/.bashrc --api-response-format json_object --check-only

PYTHONPATH=. /medisco/linjiashi/venvs/taor-inference/bin/python \
  scripts/run_wn18rr_api_preflight.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --proxy-from-bashrc /root/.bashrc --api-response-format json_object \
  --output-root outputs/api-preflight-new-attempt
```

实际调用覆盖：原生 WN18RR 单实体的图像描述、来源隔离的视觉抽取、带来源权重的事实选择、文本生成、API Judge 和原生有界 Reflect；另检查缺图模式下的 API 图片提示词组件，以及保存的真实 SDXL 图片的视觉 Judge。新提示词和冻结图片是两个独立组件检查，不冒称二者构成一轮新出图。

新运行默认启用 `entity_visual_features_v2`：每图先匹配检查，compatible 图再独立进行不含目标名称和图谱的原图描述、回看原图抽取样本范围内的实体特征，并单独依据原图复核拟采用的属性。无关、冲突和无法判断的图保留审计而不进入生成；区分属性与身份的不确定性，不再使用模型自评 0.8 门槛或 context 半权。Stage 2 不读取完整机器图片描述。具体合同、旧策略对照和错配控制见 [实体图片与抽象特征](evaluation/2026-11-06-entity-image-features.zh.md)。复用旧 Stage 1 时会检查策略一致性及逐项原图复核完整性，不能用早期实验产物绕过新复核。

本预检不运行本地 Qwen3 reranker、SBERT、CLIP、BLIP 或 SDXL 推理，也不使用 GPU。API 预检画像只要求 API Judge，不加载本地神经模型。其融合分值不能与生产全评分器结果直接比较，主要检查合同、观察结果与路由。SDXL 图片复用自先前实际运行，来源路径明确落盘，不把它计为新的图片生成或本次图片 Reflect。

每次新建输出目录，分阶段保存结果、错误和源码 SHA256；后续步骤失败时，保留已完成文本结果。`contract_checks_completed_requires_quality_review` 表示结构调用完成，仍需逐项核对来源蕴含、相关性、主客体／限定语、不确定性和候选新增内容。API 质量检查通过后才考虑申请卡，不以降低阈值、改标签或换候选制造通过。

Stage 4 客户端现在在 JSON 解析前区分 output_token_limit、empty_response 和 provider_refusal，保留 finish_reason、usage 与响应文本。业务 schema、来源 ID、证据前缀、逐检查 reason 和检查数量保持严格；仅为本地 Qwen 解码提供去除 regex／uniqueItems 的语法视图，远程 json_schema 模式保留完整 wire schema；本服务显式使用 json_object 时，合同提示和 Python 验收仍保持。对 r6 真实错误回复的 CPU grammar 回放确认，原 pattern grammar 接受了缺必填字段对象的字节作为未完前缀，新的语法视图会拒绝；不把该离线检查当成新的模型纠错结果。

截至入口实现时，针对性回归 232 passed、3 skipped。真实 API 首次调用在开发容器被既有 closeai-proxy 以 Tunnel connection failed: 403 Forbidden 拦截；该容器直连同一端点也 ConnectTimeout。属于连接失败，尚未获得模型响应，不是凭据已判无效或模型质量失败。为排查网络位置差异，提交 0 GPU 的 CPU 作业，入口 `scripts/run_wn18rr_api_cpu_cluster.sh`。详细尝试记录与最终状态以 `outputs/wn18rr-api-preflight-20261106/` 为准。


用户补充代理后，已验证 `/root/.bashrc` 中的认证代理可达，并完成真实 API 调用。`--proxy-from-bashrc` 只读取 proxy exports，不执行文件里的其他命令；只在本测试进程配置环境。API 端点当前对 json_schema 请求返回 invalid_request_format，显式 json_object 模式已验证，默认模式未偷偷更改。可用 --reuse-text-knowledge 指向成功的 text-knowledge.json，源身份、原图和图谱验证通过后复用 Stage 1，避免重复生成掩盖下游失败。

当前真实结果：新文本流程完成；历史坏文本触发一次实际 Reflect 并通过复评；固定旧描述的不确定性得到保留；已有真实 SDXL 图片的地理／比较／精确身份不再被当成已核实。详细证据、触发原因和限制见 `docs/evaluation/2026-11-06-api-closure.zh.md`。针对性回归 两组相关回归分别为 238 passed、3 skipped 与 109 passed（合计 347 passed、3 skipped），API 产物在 `outputs/wn18rr-api-preflight-20261106/`。
