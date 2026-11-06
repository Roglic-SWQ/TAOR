# 十实体文本诊断与上游事实选择优化

本轮继续优化实体名称、关系、图片共同参与的文本生成。主要结论是：**仅增加写作说明没有明显改善；把形态与展示细节分开，再做实体相关性选择，才得到更合适的内容。评审还需要就近读取关系释义，不能只匹配关系词。**

生产改动集中在现有抽取、选择、写作及评审输入，不新增生产推理阶段，不调整置信度阈值或权重。模型为现有 API 的 gpt-6-luna；没有申请 GPU 或重出 SDXL 图片。

## 冻结数据和对照范围

复用前轮在新结果产生前冻结的十实体，原清单为 `outputs/wn18rr-real-test-20261106/ten-inputs.json`。包括 water deer、lubber、archaist、sauropod dinosaur、burglar、nominee、student、land agent、epauliere、semifinalist，共 25 张原图。仍只加载 train，目标文本隐藏；原图与 train 指纹在前后相同。

输出根目录：`outputs/wn18rr-text-cohort-20261107/attempt-r1/`。

先用 API 保存每个实体的准入、盲描述、抽取、原图复核、事实选择与完整调用，再冻结到 `frozen/<entity_key>/prepared.json`。本轮省去本地 Qwen3-Reranker 与神经评分，是 **API 文本诊断，不是原生模型完整流水线或论文最终基准**。

第一组写作对照使用同一冻结 Stage 1／2、事实 panel、原图、gpt-6-luna、4096-token 上限与温度 0：

- `baseline_v1`：从本地提交 c43faa8 读取原写作代码，不切换当前分支，不修改原文件。
- `new_v2`：实体描述写作说明，名称和关键关系优先，合并相关特征，避免核验说明口吻；对应 `entity_semantics_text_v5` 和 `entity_semantics_text_with_images_v2`。

两组按已冻结清单交替执行顺序，均使用原有条件反思策略。完整初始文本和最终结果分别保留。

## 十实体实际结果

| 项目 | 实际结果 |
| --- | --- |
| 上游准备 | 10/10 完成，无实体被排除出报告 |
| 两组文本生成与评分 | 20/20 completed |
| 初始事实检查 | 共 40 项获评 supported；没有触发 Reflect |
| 图片准入 | 8 compatible、14 unresolved、2 conflicting、1 unrelated |
| 最终写作收到原图 | 两组均只有恐龙案例各一张，其余九例没有选中视觉事实 |

准入结论是模型判断，不是人工真值。compatible 仍不等于身份验证；unresolved 也不能报告成“错误图片”。

逐条查看两组文字，差别大多是冠词、连接词或同义表达。student、lubber 等只选了一条类别关系，文本很短。**没有证据证明本轮措辞变更带来明显质量优势，也不能把 20 次完成或 40 项支持称为真实身份／定义全部正确。** 生成和 Judge 同用 gpt-6-luna，评分不是独立互证。

这组清单多数是人物角色，且九例未向最终写作发送图片，因此不能据此称为十例联合看图写作的充分验证。它是完整保留失败可能性的固定诊断清单，不是代表性采样。

本轮可见调用记录为：准备阶段 50 次网络尝试记录、选择控制 4 次、属性重抽取／复核 2 次、文本生成 26 次、Judge 30 次响应记录。前述 20 次是主对照，其余文本来自单独控制及条件反思。上游与生成可见 usage 合计输入 378,022、输出 14,614 tokens，未包含未暴露的 Judge usage 或账单，不据此推算完整费用。去除缓存调用副本后的计数保存于 `visible-call-accounting.json`。

## 修正一：表示介质不能冒充实体特征

恐龙原图是一座蓝绿色的雕塑。旧抽取把“雕塑、长颈、尾部、四腿”打包成一条，还单独提供颜色、风化表面；原选择将这些内容一起纳入。它们可能确实可见，却不适合都用于描述所指的恐龙类别。

本轮在现有流程内做三项配合：

1. 抽取分别表达形态、描绘介质、颜色和表面状况，使后续可以独立取舍；不拆散不确定的备选解释。
2. 选择时区分目标实体与其模型／插画／展示物。展示物的涂色、材质、风化不自动成为被表示类别的属性；如果目标本来就是该实物，则仍可选择其属性。
3. 同时明确允许选择已准入描绘中的相关形态，但限定为该样本；身份未证实不能单独成为丢弃清楚可见属性的理由。不强制选任何图片事实，也不设置配额。

仅修改选择的第一轮，虽然去掉颜色、风化，却连形态一起漏掉；记录在 `selection-controls/01708998/`。同一原图、原准入与同一盲描述重新抽取并独立看像素复核后，相关形态与展示属性可以分开取舍，记录在 `property-controls/01708998/`。最后复用这份已复核属性进行选择，记录在 `property-selection-controls/01708998/`。

最终选择保留三项合法类别／成员关系及“长颈、尾部、四条支撑腿”的样本形态，没有选择蓝绿色和风化表面。最终实际文字中也未加入这两项展示细节，四项检查获评支持，无需 Reflect；必要的样本身份不确定性仍保留。没有据此断言雕塑还原了真实动物外观。

上述后续控制改变了抽取或所选事实，**不属于第一组同 panel 的写作对照，不能直接比较内部融合分数来归因收益**。原始冻结记录、失败尝试和已完成结果均未覆盖。

## 修正二：重复词汇关联只表达一次

burglar 与 heist 的词汇关联在原选择中正反两向同时保留，两版写作都重复解释这同一个联系。新选择允许保留其中一个能传达同一联系的断言；类别、部分等非对称关系仍保留方向，完整图谱目录不删除。

该例所选事实由三项变为两项。实际生成：

> Burglar is a kind of thief. The word “burglar” also has a derivational lexical relationship with “heist.”

原先第二次反向复述被去掉。控制记录在 `selection-controls/09880741/`。

## 修正三：评审理解关系语义和等价表达

控制中出现一次明确误报：原文将“suborder Sauropoda 包含 sauropod dinosaurs”表达成“sauropod dinosaur 是该 suborder 的成员”，Judge 却判为关系反转并要求改写。两种表达语义等价，不能把这次改写报告成真实事实纠错成功。

先补等价表达说明后，正确表达获支持，但历史 water deer 错误文本的真正类别反转仍被误评为支持。该失败保留在 `negative-paraphrase-controls/02436645/`，不能只展示成功的正例。

最终将现有 `relation_meaning` 直接放在每一条 `CHECK_TARGETS` 旁边，要求按明确释义判断，而不是匹配原始关系词。该标注只改变 Judge 的输入位置；原始 SRO、检查 ID、存档验证目标及严格响应合同不变。

用固定原文及同一参照重评，未重新生成文本：

| 控制 | 最终结果 |
| --- | --- |
| 正确的成员关系等价表达 | 三项 supported，原文与源文件不变 |
| 历史真正颠倒 hypernym 含义的文本 | contradicted，保留原错误文本不变 |

最终控制分别在 `paraphrase-controls-semantic-check/`、`negative-paraphrase-controls-semantic-check/`。这验证了两个具体正反例，仍不是所有关系表达都可靠的证明。

## 验证与复跑

相关专项最终 **389 passed、3 skipped**，JUnit 在 `verification/complete-semantic-final.xml`。跳过项为可选解码依赖。扩大检查另有十项旧 Qwen 请求哈希测试失败；在隔离的 c43faa8 源码副本上也是同样十项，记录于 `verification/final-legacy-comparison.json`，不声称全仓通过。

`scripts/check_wn18rr_text_cohort.py` 提供 prepare、compare、reselect、reextract、reselect_properties、rejudge 阶段，每阶段新目录／新记录，保留成功、失败和源指纹。调用方式：

```bash
PYTHONPATH=. /medisco/linjiashi/venvs/taor-inference/bin/python scripts/run_with_cluster_api.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --proxy-settings /medisco/linjiashi/.config/taor-runtime/proxy-hybrid-20261106.json \
  --module scripts.check_wn18rr_text_cohort -- \
  --phase prepare --output-root outputs/wn18rr-text-cohort-20261107/<新的实验目录>
```

准备完成后，使用同一目录运行 `--phase compare --baseline-commit c43faa8`。后续两个选择控制显式指定 `--entity-key 01708998 --entity-key 09880741`；属性控制仅指定恐龙案例，所有原始控制均独立保存。这里不是旧版／多种子／多视角 SDXL 基准，也未启动 4580 全量或下游 KGC 训练。
