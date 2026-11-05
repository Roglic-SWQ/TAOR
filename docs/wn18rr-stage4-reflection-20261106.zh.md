# WN18RR 单实体 Stage 4／Reflect 实测

本次使用上一轮真实 Qwen／SDXL 的冻结输出，补齐 Stage 4 和原生条件 Reflect，检查原有错误能否被自动发现及修复。实体仍为 `02436645`（water deer），不重新抽取事实或更换候选来绕过错误。

## 运行设置

- 原始 Stage 0–3：`outputs/wn18rr-onboarding-20261106/attempt-r3/runs/`。
- 新产物：`outputs/wn18rr-stage4-20261106/attempt-*/closure/`，text／image 使用独立目录。
- 单张 A100 80GB，沿用本地 Qwen2-VL-7B 服务和 SDXL；Judge 也为 Qwen2-VL-7B，非独立 GPT judge。内部兼容键 gpt4omini_judge 不代表实际调用 GPT。
- Stage 4 必需评分器全部启用：Sentence-BERT all-mpnet-base-v2、OpenAI CLIP ViT-B/32、BLIP ITM large COCO、本地 Qwen 四维 Judge。辅助可选 BERTScore 未启用，不影响主线融合权重。
- 配置：`configs/stage4-qwen7b-local.yaml`、`configs/qwen7b-judge.yaml`。Judge 输出预算 4096，服务上下文 24576；生成预算不变。
- 新模型仍下载在 `/medisco/linjiashi/models`，项目内受路径约束的评分目录使用普通文件硬链接，保持 path-containment 校验，不弱化校验规则。CLIP 从 ModelScope 的 jinaai/clip-models 镜像取得，SHA256 与 OpenAI CLIP 包声明的原始 ViT-B/32 权重一致：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。
- 原始 run 全部文件 SHA256 在回放前后比较；恢复后的调用标记 reused_source_run，不作为本轮新增推理计数。图片／源文本仍按目标模态遮蔽，未把隐藏真值送给 Judge 或 Reflect。

## 工程变更与复跑

`mmkgc.io.resume.load_saved_run` 现在同时支持 text／image，原 load_saved_image_run 入口保留。文本恢复时校验来源、预算、检查目标、候选内容与目标模态，重建相同的事实上下文。专项测试同时检查旧图片恢复行为。

入口：

```bash
TAOR_STAGE4_ATTEMPT=<新的尝试名> bash scripts/run_wn18rr_stage4_cluster.sh
```

该脚本应在已分配的单卡作业中运行，保留调度器设备隔离。它启动真实 Qwen 服务，完整 doctor 后调用 `scripts/run_wn18rr_reflection.py`。每次尝试新建目录，拒绝覆盖。Python 回放工具调用原 `pipeline.finish` 执行 Stage 4、条件触发的一轮 Reflect 和固定问题复评，不通过手写反馈强制声称修复成功。

## 第一轮结果与合同修复

第一轮 `taor-wn18rr-stage4-reflect-1106` 的完整 GPU doctor 通过，三个本地评分模型均已实际加载。

- 文本 Judge 两次返回相同的不合合同响应：维度证据是裸引文，没有 `text:` 前缀，事实分 4 的证据数组还为空。程序拒绝该响应，文本 ranking_status=evaluation_failed，没有伪造分数或启动无依据的修复。
- 图片 Judge 返回有效结构，但把“西非”“最大鼷鹿”等不能由单幅画面独立证实的信息判成 supported，给出四维满分。图片分数高不代表该判断正确，也没有因此认定物种身份已验证。
- 两个目标均未触发修复。初次批处理进程退出只说明任务执行结束，不能当作文本评审成功。

最终保留的工程修复很小：非事实维度的 schema 要求非空证据；提示词明确 text:<实际引文>／pixel:<可见内容>。当事实分非 null 却没有证据时，严格解析先拒绝该响应，下一次有界合同修复才对该字段加 minItems=1，并从原始候选／来源重建请求，不复制错误 JSON。初次事实分 null 的合法空证据不受影响。没有在程序中补前缀、填证据或改分数；专项 **92 passed**。

中间两轮诊断也保留：

- r2 曾尝试前缀正则／可空事实维度的复杂 schema，本地解码器返回漏检查 ID 且尾部反复追加说明的长响应。文本三次有界尝试后仍失败；为避免继续浪费资源，停止该次自己提交的诊断任务，未完成的图片评审不算成功。该复杂 schema 已撤回。
- r3 使用简单 schema 后恢复了完整检查数组，但仍重复给事实分 4 配空证据，因此评审明确失败。最后采用上述仅在该具体错误后收紧的合同修复。

## 最终实测结果

作业 `taor-wn-s4-reflect-1106-r4`，单卡 A100 80GB。原始 Stage 0–3 文件和生成图片哈希在回放前后保持一致。

| 目标 | Stage 4 状态 | 四维分数 | 内容 Reflect | 原问题 |
| --- | --- | --- | --- | --- |
| text | completed | 实体语义 3／事实一致性 4／覆盖 3／可用性 3 | 未触发 | 分类冲突仍在 |
| image | completed | 4／4／4／4 | 未触发 | 精确物种身份仍未证实 |

**没有修复成功。** 文本 Judge 将 rodent、marsupial 与 mouse deer 的检查全部标为 supported，理由主要是候选确实重复了这些参照事实，没有发现上游事实集本身的分类冲突。事实一致性 4 因而不代表原错误已消失。图片 Judge 同样过度确认了无法由画面独立证实的地理和物种／相对大小信息。

程序依据 observation 判定没有可执行的矛盾／遗漏修复操作，所以两个模态都保留原候选，`attempted=false`，没有内容再生成、SDXL 重采样或修复候选复评。文本 Judge 的合同重试只是纠正评审格式，不能算作内容 Reflect。完整自动控制流已执行到修复决策，但本例没有进入实际内容修复分支；不能宣称验证了“生成→修复→复评”的成功闭环。

本例的故障位置是：上游图片推断产生错误分类 → 错误分类进入共享精选事实 → Stage 4 主要确认候选与这些参照的一致 → 没有修复触发。下一步需要先处理来源事实之间的冲突及图片推断的可核验边界，并检验更独立的 Judge；仅接上相同的自评模型不足以保证纠错。

证据：

- `outputs/wn18rr-stage4-20261106/result-audit.json`：最终状态、逐事实判断、分数及原内容不变的检查。
- `outputs/wn18rr-stage4-20261106/attempt-r4/closure/closure-summary.json`：配置／代码／原始来源指纹与修复决策。
- `attempt-r4/closure/{text,image}/entities/02436645/4_kg_aware_candidate_consistency_ranking.json`：完整 Stage 4 观察、分项分数及失败／重试记录。
- 同目录 ranking_model_calls.jsonl、原始 Stage 3 和 final.json；较早各 attempt 的真实失败记录仍保留。
