# WN18RR 真实单卡测试：输出合同与模型能力边界

本轮按用户要求运行真实 WN18RR、本地 Qwen2-VL-7B、Qwen3-Reranker、SDXL 与 Stage 4。结论是：**真实生图成功，可靠的文本／图像纠错闭环未通过。简化合同能改善输出稳定性，不能保证事实判断可靠。**

## 数据与资源

实体保持 `02436645 / water deer`，原图 SHA256 为 `0dc11e1b26f68c44209be5b764a02441ad1fc89a5030a98a501d6fb4899cc456`。只用 train 图谱；缺文本时遮蔽目标文本，缺图片时移除原图输入。没有用隐藏定义纠正生成。模型路径、数据及评分权重沿用现有专用环境，未改装共享依赖。

两个作业 `taor-wn18rr-real-1106-r1`、`taor-wn18rr-real-1106-r2` 顺序申请单张 A100 80GB，前一作业结束后才提交下一作业。GPU doctor 通过。每次新建输出目录，没有覆盖失败记录或原始数据。

全部产物在 `outputs/wn18rr-real-test-20261106/`。`pilot-inputs.json` 保存 train 文件与原图指纹，`attempt-*/execution.json` 保存实际执行源码哈希。执行时基础提交为 eff6d30，新增测试入口及后续修复以运行源码哈希为准。

## 第一轮真实结果

| 环节 | 实际结果 |
| --- | --- |
| 本地图片抽取补文本 | Stage 1 截断失败，未生成新文本 |
| 本地缺图提示词与 SDXL | 新生成 1024×1024 PNG，20 steps，guidance 7.5，实际 seed=20260709，无 fallback |
| 图片完整 Stage 4 | CLIP、BLIP 与本地 Qwen Judge 实际执行，ranking_status=completed |
| 质量验收 | 未通过：本地 Judge 过度确认不可由像素证实的信息 |

文本抽取在 4096-token 原预算下输出 4096 tokens，finish_reason=length；响应反复生成 has body 与词片段，并照抄不确定性示例占位语。程序在 JSON 解析前归为 output_token_limit，合同修复次数为 0，未把截断伪装成 JSON 格式错误。失败及原始响应位于 `attempt-r1/runs/text-qwen7b/failures.json`。

新图片位于 `attempt-r1/runs/image-qwen7b-sdxl/entities/02436645/generated_images/image_001.png`，SHA256 为 `408b9cdbe74a9c25d7ac485b961dc86465ce723dbc51437fda7d0c69a4cae9b6`。实际打开检查，是湿地中的鹿状动物插图；不据此宣称精确物种正确。

## 同图、同参照的独立评审

固定上述新 SDXL 像素、候选、四个检查 ID 与期望断言，仅将评审模型换成现有 API 配置的 gpt-6-luna。没有重出图、改参照或回写原评分。

| 检查 | 本地 Qwen Judge | API Judge |
| --- | --- | --- |
| water deer 精确身份 | supported/direct，confidence=1 | unverifiable |
| 西非沼泽栖息地完整断言 | supported/direct，confidence=1 | uncertain/appearance_only：湿地可见，西非不可确认 |
| mouse deer 的 hypernym 关系 | supported/direct，confidence=1 | unverifiable |
| 最大 chevrotain | supported/direct，confidence=1 | unverifiable |

本地事实维度为 4；API 事实维度为 null。API 评审仅启用 Judge，没有重算本地神经模型，**不直接比较两个配置的融合总分**。逐项判断保存在 `independent-image-review/result.json`，原 run 文件哈希在评审前后相同。API 的谨慎判断仍不是精确身份真值认证。

这说明输入边界与合法输出格式并不能保证评审模型理解可核验性。同一模型的生成／自评不能据此被当作独立互证。API 将不可见断言保留为不可核验，也没有因此强行触发生图修复。

## 一次固定失败输入的合同对照

r2 固定 r1 原图、原始图片描述与原有 4096-token 预算。原始完整匹配调用没有随失败条目保存；控制仅恢复失败提示中已记录的 compatible、identity_verified=false 视图，缺失的完整匹配理由不伪造。程序推导的身份限制单独标注来源。

仅对本地纯图片抽取使用精简传输合同，让模型输出属性、值、种类、观察／推断依据、不确定性和来源；主体、样本范围、中性 profile 与完整图谱由程序按既有合同绑定。每张图最多八项是响应体积上限，没有最少项数，不是置信度阈值；不会删除图谱或截取超量响应制造通过。非图片与已有 API 路径保持原合同。

本地 7B 用 **303 tokens、finish_reason=stop** 输出六项属性，无合同修复，约 15.3 秒：棕色、slender 体形、大耳、hoofed 足部、long 口鼻、short 尾长。这些是模型提出的属性，**不是人工确认的正确答案**。

之后的原图复核两次给出完全相同的全 supported 判断，同时每项将同一原图引用重复两次。业务校验拒绝重复引用，r2 在复核阶段停止，没有进入新文本生成或 Stage 4；原提议及两次失败回复保留在 `attempt-r2/frozen-control/`。因此不能把“抽取不再截断”称为完整闭环成功。

另补等价的解码约束：来源引用数组的 maxItems 不超过合法原图数量。该限制本来已由引用唯一性隐含，但本地解码语法去除了后端不支持的 uniqueItems；显式上限可约束单图重复引用。它不改变业务验收集合，也不在程序中去重后冒充模型成功。CPU grammar 用 r2 原始坏响应验证，结果独立保存；未为此再申请 GPU 或改写 r2 结果。

十实体清单及原图指纹已预先冻结在 `ten-entities.json`、`ten-inputs.json`，包含难以仅靠外观核验的角色类实体。由于单实体质量验收未通过，**十实体、旧新版本、多种子／多视角对照均未启动**；本轮没有启动全量任务，也没有内容 Reflect 成功记录。

本轮保留的代码改动只处理本地传输的冗长输出和来源引用约束。事实辨认、自评过度确认及精确身份验证仍是模型／证据能力边界，未通过加规则、降低阈值或扩大预算宣称解决。

针对性回归为 266 passed、3 skipped，JUnit 位于 `verification/targeted-final.xml`；跳过的是推理环境中的可选本地解码依赖。实际服务环境另用 CPU 编译并回放解码合同：精简合同接受合法单项、拒绝单图九项；原图复核旧解码语法接受 r2 的重复引用回复，显式引用数量上限后的语法拒绝它。两轮 GPU 作业均已结束，不留占卡任务。
