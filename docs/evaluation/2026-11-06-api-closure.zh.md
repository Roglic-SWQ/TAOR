# API 单实体闭环与不确定性回归

使用用户提供的本地 API 配置和 `/root/.bashrc` 中的认证代理，实际调用 `api.tensoris.ai` 的 `gpt-6-luna`，完成 WN18RR 实体 02436645 的上游、文本生成、Judge、一次真实 Reflect 与复评。API 阶段不使用 GPU，不运行本地神经评分器，不生成新图片；图片评审使用已有真实 SDXL PNG。

机器审计：`outputs/wn18rr-api-preflight-20261106/result-audit.json`。完整产物分开保留，原有 49 个工程冒烟／Stage 4 文件哈希未变化；差异和 API 产物也检查了已知凭据字面值，未发现泄漏。代理和 API 密钥未复制到仓库，原配置文件未修改。

## 真实测试结果

| 测试 | 结果与边界 |
| --- | --- |
| 新 API 描述／抽取／筛选／文本生成／评审 | 5 条候选选 3 条：mouse deer 图谱分类、外观、鼻部；文本 ranking_status=completed，正常好候选没有强行 Reflect |
| 原始坏文本＋修复后的上游事实 | 缺失两项外观信息触发一次原生 Reflect，新候选 text_safe_001 被选中并通过复评；原候选、检查 ID、事实和来源引用保留 |
| 固定旧原图＋旧错误描述 | 仍使用原文“appears … rodent or marsupial”与同一 JPEG；分类和动物园／保护区猜测写入 uncertain_aspects，没有升级为确定事实 |
| 图片提示词组件 | 实际 API 生成 SDXL 提示词，未调用本地生图，不把新提示词与冻结图片冒称为一轮新出图 |
| 同一已有真实 SDXL 图片的 API Judge | 精确身份 appearance_only/uncertain，地理／分类／相对最大均 unverifiable，事实评分 null，不再伪装成全事实已验证 |

新描述抽取中，外观权重 0.88、鼻部 0.83；围栏 0.485、地面 0.47，两条场景信息未选入生成事实。来源中的 rodent-like 被记录为“图片不能建立该分类”。固定旧描述控制则保留“rodent 或 marsupial 尚未建立”以及“围栏不等于已证实动物园／保护区”，可用毛色／耳朵等观察仍保留。没有按动物分类词表硬编码排除。

这些数值是策略权重，不是校准后的真实性概率。固定描述中的站立／地面关系仍可被模型划为 target，说明 scope 也依赖模型判断；目前只验证一个真实实体，不能外推为所有图片或本地 7B 同样可靠。

## 原始坏文本的实际 Reflect

原候选直接读取先前真实 Qwen 输出，文件 SHA256 为 `587215a2794373a176aabf742a33bd3d764fcda0585f43febadd1e38ae7e8883`，没有先重生成一个更容易修的样本。内容包括 rodent、marsupial 以及围栏／场景猜测。

本例是**同一历史候选对照新的来源隔离事实基础**，不是同提示词旧新版生成质量对照。新基础来自 API 上游，保留合法 train 图谱与同一已有原图，未使用隐藏目标文本。评审将未获核实的分类和场景列为 unverified_additions，没有据此伪造已核实的矛盾或 remove 操作；触发的是外观与鼻部两个 must_include 的 missing，执行两个 add。

生成器在收窄的事实范围内重写，最后选中的内容为：

> The water deer is a mouse deer. In the image, it appears in profile facing right, with a brown-gray rounded body, upright ears, a dark eye, slender legs, and a pointed pinkish nose.

未选入的 rodent／marsupial 和场景猜测不再出现，原候选仍保存在同一新运行中，修复前后验证目标完全相同。这证明一轮实际“观察→操作→生成→复评→选择”已执行；**不能据此声称 Judge 已独立证明所有未核实分类为假**，或把可核实事实维度的高分当成全文真实性证明。

产物：`outputs/wn18rr-api-preflight-20261106/old-text-reflection-r6/`。

## 图片评审

评审读取先前真实 SDXL 1024×1024 PNG，非占位图。结果：语义 2/4、覆盖 2/4、可用性 4/4、事实一致性 null，API rubric 归一分 0.5。它识别出可见鹿状动物与湿地，但没有把像素解释成已证明位于西非、属于精确物种或“同类最大”。这不等于图片生成已精确满足物种身份，也不是生产全评分器的可比融合分。

产物：`outputs/wn18rr-api-preflight-20261106/attempt-r5/frozen-image-judge.json`。新提示词独立保存在同目录的 `image-prompt-component.json`，本轮未据此生成新像素。

## API 兼容修复与回归

- 当前开发进程默认代理阻断端点；用户指定的 bashrc 代理可达。仅解析指定 proxy exports，不执行整个 shell 启动文件，账号口令不输出。
- 当前服务拒绝所提交的 json_schema 请求，上游错误为 invalid_request_format。显式选择已验证可用的 `--api-response-format json_object` 后完成流程，不做无条件错误降级。合同仍在提示词／程序中，字段、状态、来源 ID、引用唯一性、检查完整性与事实修复接受条件仍严格验证；默认 json_schema 模式保留。
- Stage 2 保存脱敏的提供方错误，避免只留下无原因的 HTTP 400。API 重试可用 `--reuse-text-knowledge`，先验证实体、源模态、原图 SHA256 与图谱，保留已成功的 Stage 1。
- Stage 4 在解析前区分 token 截断、空响应和提供方拒绝；本地 Qwen 仅使用解码兼容语法视图，远程 json_schema 模式保留完整 wire schema，json_object 模式依靠明确合同和程序验收。

最终针对性回归：两组相关回归分别为 238 passed、3 skipped 与 109 passed（合计 347 passed、3 skipped）。跳过的是推理环境缺少的可选解码依赖；实际服务环境的 CPU grammar 检查另已执行。此前扩展检查中的 64 项既有失败与基线集合相同，没有通过修改旧预期来掩盖失败。

## 复跑与剩余范围

通用入口和参数见 `docs/api-preflight.zh.md`。三个可审计测试目录：`attempt-r5`、`old-text-reflection-r6`、`fixed-old-caption-r7`。

本轮按 API 优先顺序完成单实体逻辑验证，未再次申请 GPU。此前自己的 GPU 诊断已结束，临时 0-GPU CPU 直连诊断也已结束。纯本地 Qwen2-VL-7B 的迁移稳定性、新提示词对应的 SDXL 新图，以及冻结 10～20 实体的控制实验仍待验证；未启动 4580 全量任务。

Stage 1 图像抽取也补上解析前的空响应／token 截断分类，失败直接带 completion_failure 退出，不再用 JSON 格式修复重试截断内容。已有文本分块恢复逻辑保留。额外 Stage 1 语义回归使用现有服务环境的纯 Python Starlette 补充临时测试路径，未改动共享包安装。测试日志与环境路径在 API 产物目录。
