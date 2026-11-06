# 图片匹配检查与实体特征抽取

按用户要求，将上游问题拆为两步：隔离无关、冲突或无法判断的图片；从可用图片提取与实体相关的抽象外观特征。新运行默认使用 `entity_visual_features_v2`，不再要求模型自报 confidence，也不使用 0.8 门槛或 context 半权。

## 当前流程

1. 每张原图先冻结字节与 SHA256。图片匹配检查只使用目标标识、已遮蔽目标模态的合法 train 图谱断言和原图。状态为 compatible、unrelated、conflicting 或 unresolved；后面三种不进入视觉特征抽取，保留原图、判断理由、不确定性和模型调用。conflicting 表示来源之间存在可见冲突，不自动裁定图谱或图片哪一方为真。
2. compatible 仅表示外观相容，程序固定 `identity_verified=false`。它不是物种或具名身份核验；同类别内的错配仍可能漏检。图谱断言也不因与图片相容而被认证。
3. 对通过检查的图片，另发独立的原图描述请求，只带原始像素，不带目标名称、图谱或匹配理由。之后回看同一原图及这份描述抽取特征。匹配检查生成的描述、理由、图谱内容都不作为抽取证据。这样避免把匹配检查中的图谱答案复制成“图片直接观察”。
4. 图片观察使用 basis、scope、feature_kind、claim_scope、uncertainty、uncertainty_scope。feature_kind 区分 entity_feature、depiction_detail、context；只有已通过匹配检查的 direct/target/entity_feature、且属性本身无未决问题的观察可进入生成。uncertainty_scope 区分 none、feature、identity：身份未证实不抹掉图中样本清楚可见的结构，但必须保留其归属限定；属性本身看不清或属推断时不作为确定事实采用。颜色、形态、构造、材质等以合适的抽象层次表达；朝向、姿态、光照、地面、围栏等画面细节不作为实体特征。不依赖词表识别特定物种或场景。
5. 程序对拟采用的实体特征追加一次有界原图复核，只传原始像素和待核属性，不带描述、图谱、目标名称或匹配理由。逐项返回 supported、uncertain 或 contradicted，以及实际可见内容、理由和原图引用。只有 supported 的视觉贡献进入生成；原抽取和所有复核结果都保留，已有独立图谱支持不因弱视觉观察丢失。此步骤仍是模型复核，不代表独立身份认证。
6. 所有视觉特征的 claim_scope 均为 depicted_instance，不能从一张图升级为整个类别的普遍属性。生成器和 Stage 4 均收到这一限定。未证实身份且无独立所选证据时，文本应描述与目标标签关联的图中样本，而非断言准确身份。profile 固定为中性 entity 和目标标签，避免通过身份摘要绕回未选断言。
7. 不确定、推断、画面细节和被隔离图片均保留用于审计。已有独立图谱／文本等来源沿用原策略。source_weight 在 v2 下只作来源准入的 0/1 标记，不是真实性概率。Stage 2 看抽象特征和原图复核记录，不再收到完整机器描述或匹配理由；生成器同样不接收未选择的完整描述。

总时间预算、原生模型 token 默认值不变；匹配合同最多修复一次。空响应和 token 截断在 JSON 解析前明确分类，不作为格式错误重试。无需新增 GPU 或替换评分模型。

旧版来源策略保留以解释历史产物，也可在 inference.stage1_image_policy 显式设置 `visual_observation_policy_v1` 做对照。原有图片回归固定旧策略，同时新增 v2 的完整入口、混合图片、语法修复、截断和生成边界回归。

## 实测记录与边界

产物均位于 `outputs/wn18rr-entity-visual-20261106/`，使用已配置代理与真实 API，未申请 GPU、未运行新 SDXL 像素生成。

`attempt-r1` 是初版 v2 控制：匹配检查与描述尚在同一个请求。02436645 原图输出相容判断，同时保留准确身份未验证的限定。抽取将体形、被毛、耳、腿、口鼻分开；不确定的牙齿样结构被隔离，背景不进入选中事实。最终文本聚焦图中样本的形态，Stage 4 completed；图片提示词组件与冻结真实 SDXL 图片 Judge 均完成。该次不能作为“匹配与感知已分离”的证据。

随后将通过检查后的原图描述独立出来，并加强生成和 Stage 4 对样本范围、身份限制的处理。后续错配控制和最终运行分别保存，不回写 r1。

`admission-controls-r2`：正确小提琴 compatible，吉他配小提琴 conflicting，动物配小提琴 unrelated，错误图未贡献可选事实。但混合图片暴露了过度过滤：模型把“准确身份未验证”放进每条清晰结构的 uncertainty，原策略将四条全部隔离。该混合结果**未达到好图仍应贡献特征的验收**；旧脚本只检查坏图不泄漏，曾错误返回 passed=true，其原始文件保留。现已加强验收，同时要求正确图通过且贡献非空特征，并显式区分属性与身份的不确定性。r1、r2 是本轮实现过程的实验合同，源码哈希随产物记录，不直接复用为最终合同的运行输入。

`admission-controls-r3`：正确小提琴保留弯曲边缘、音孔、琴弦三项；动物与小提琴混合时，动物 unrelated、小提琴 compatible，正确图仍贡献弯曲边缘、木质主体、边框、琴弦四项，准确身份未证实的逐项说明保留。未混入错误图片引用。

`attempt-r4` 暴露另一项质量缺陷：图片描述将动物称为 rodent，抽取把原图不能清楚支持的“长细尾巴”列为 direct 确定特征。Stage 2 最终只选图谱分类，文本为 “The water deer is a type of mouse deer.”；虽然 Stage 4 completed，却没有实现有用的图像特征补充，不能判为质量通过。原图复核与 Stage 2 描述隔离正是针对这个实际反例补入，未靠调整置信阈值解决。

`frozen-feature-recheck-r5` 固定 r4 的同一张原图、同一批已抽取属性，单独运行新增复核：棕色被毛、耳形、口鼻、可见眼睛 supported；“长细尾巴” uncertain，因为完整轮廓与长度被遮挡，程序保留原抽取但将其 generation_eligible 设为 false。原图与原 r4 知识文件哈希不变。这是同一失败输入的控制，不是更换输入后的成功样例。

`attempt-r6` 从原始 WN18RR 重新运行匹配检查、独立描述、特征抽取与原图复核。体形、毛色、耳形、口鼻获支持；腿部复数断言因部分遮挡而 uncertain，没有作为生成事实采用。Stage 2 选择 rf:0001、mf:0001、mf:0002、mf:0003、mf:0005，未选择腿部、姿态或背景。文本为：

> A water deer is identified here as a mouse deer. In the pictured sample associated with that label, the body is rounded and compact, with a brown-gray coat, prominent ears with pink inner surfaces, and a pointed snout with a pink nose area. The image does not establish the animal’s exact species.

文本 Stage 4 completed，ground_truth_used=false。这一次既使用清楚可见的实体相关特征，又保留了准确身份未证实的限定，没有把照片属性泛化为类别的普遍性质。图片提示词组件仍不等于新的 SDXL 图片生成。

r6 的图片提示词组件与冻结真实 SDXL 图片 Judge 均完成，后者为 scored。审计结果在 `result-audit.json`：三张控制源图及固定的失败知识文件哈希不变，当前原图复核记录完整，未发现凭据内容。该预检未运行本地 reranker／神经评分器，不与生产全模型融合分数作等价比较；同一 API 模型的多次请求也不构成独立模型互证。

针对性回归 260 passed、3 skipped，记录在 `verification/final-with-pixel-recheck.xml`。本地 Qwen 服务环境仅用 CPU 对匹配、特征、原图复核三份解码 schema 编译成功，结果在 `verification/with-recheck-grammar-result.json`；未加载本地模型权重推理，不冒称完成了本地 7B 迁移。历史 v1 回归显式固定旧策略，新 v2 另有完整入口及来源边界测试。

这些分类仍是模型判断，没有获得独立身份认证或统计校准。少量人为构造错配映射的通过不等于真实全量数据的错配检测率。冻结 SDXL 图片复评也不等于新提示词已生成正确新图；本地 Qwen2-VL-7B 与新 SDXL 仍须在 API 验证后单独迁移验证。

错配控制入口：

```bash
PYTHONPATH=. /medisco/linjiashi/venvs/taor-inference/bin/python \
  scripts/check_entity_image_admission.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --proxy-from-bashrc /root/.bashrc \
  --output-root outputs/wn18rr-entity-visual-20261106/new-control-attempt
```

控制固定真实小提琴、吉他、动物 JPEG，仅修改内存中的图片归属：正确小提琴、吉他错配给小提琴、动物错配给小提琴、动物与正确小提琴混合。源数据及已有运行目录不覆盖，每次输出新目录，保存图像哈希、调用及完整知识目录。
