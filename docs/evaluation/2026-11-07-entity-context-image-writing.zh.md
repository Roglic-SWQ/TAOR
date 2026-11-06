# 实体名称、关系与原图共同参与文本写作

本轮补上最终写作阶段的原图输入。之前图片只进入 Stage 1 的准入、描述、抽取和复核；Stage 3 文本生成实际只收实体名、已选关系与视觉事实。现在可在一次写作请求中同时提供这些语义上下文和对应的原始像素，不增加新的推理链或 GPU 作业。

## 实现边界

新增显式配置 `configs/gpt6-luna-text-vision.yaml`，开启 `text_use_source_images: true`。现有仅事实配置保持可复跑；主 CLI 的 `--model-config` 指向新配置即可启用，已知 dataset_root 由运行时传给文本适配器。Stage 1 的盲描述与原图复核保留，名称和关系用于最终写作的语义定位，不当作视觉真实性证明。

Stage 3 只为补文本绑定与已选事实关联、且准入为 compatible 的原图；不会发送被排除、未选中的图片，补图分支也不携带隐藏原图。读取限定在数据目录内，逐张核对建模时的 SHA256 和字节数，再冻结字节发送。文件改变、路径越界、重复引用和缺少数据根目录均在 API 前失败，保留失败调用。

图片传输 metadata、像素与公共提示字段分开。日志记录原图引用与指纹，不保存 base64；正文不带文件路径、机器原始描述或未选路径正文。新提示版本为 `entity_semantics_text_with_images_v1`。名称及关系帮助定位主体，原图帮助理解已选视觉属性；仍不补造背景、遮挡属性或类别普遍性质。必要身份限定保留一次，减少“所提供知识／核验程序”的说明口吻。

## 同源 API 测试

冻结来源为上一轮完整 Luna run 的 `text-gpt6-luna/entities/02436645`。只用 train，源目标文本仍隐藏。两组复用同一 Stage 1／2、六项验证目标、gpt-6-luna、4096-token 上限和温度 0，不重新抽取或选择。

- `facts_only`：最终生成不发送图片。
- `facts_and_images`：最终生成发送原图，SHA256 为 `0dc11e1b26f68c44209be5b764a02441ad1fc89a5030a98a501d6fb4899cc456`，与前轮同源。

产物位于 `outputs/wn18rr-text-vision-20261107/attempt-r1/`。两组均完成，六项检查获评 supported，均未触发 Reflect；实际生成响应模型均为 gpt-6-luna。原 run 文件指纹在前后相同。

带图版实际输出：

> Water deer is a kind of mouse deer. In the supplied image, the pictured animal associated with the water deer label has large rounded ears, a pointed snout, and a brown coat with darker-brown shading along the back and an underside lighter than the back; the image alone does not verify that it is a water deer.

逐句对照冻结事实，类别方向、耳形、口鼻、被毛颜色、背部与腹侧明暗均保留；没有增加围栏、地面、地点或未选的长尾、鼻部颜色。身份未证实的限定也保留了。该复核由助手完成，不是独立人工真值标注。

仅事实版仍使用“identified in the supplied knowledge”“associated with the water deer label”等较多说明性措辞。Judge 对其“与标签关联”的一句另列未核验提示，带图版未列；这可能含有评审波动，不能把它当成可靠的身份辨认改善。生成和 Judge 同为 gpt-6-luna，也不能视为独立互证。

本轮带图版还附加了自然实体写作说明，因此并非仅增加像素的单因素实验。结果证明联合输入已实际接通，并在这一个例子中保持事实范围、减少核验说明口吻；不证明普遍优于仅事实写作，更不证明源图精确身份正确。

## 复跑与验证

```bash
PYTHONPATH=. /medisco/linjiashi/venvs/taor-inference/bin/python scripts/run_with_cluster_api.py \
  --env-file /medisco/linjiashi/Code/.env.standardization.local \
  --proxy-settings /medisco/linjiashi/.config/taor-runtime/proxy-hybrid-20261106.json \
  --module scripts.check_wn18rr_gpt6_generation -- \
  --task text_compare --model-config configs/gpt6-luna-text-vision.yaml \
  --source-entity outputs/wn18rr-gpt6-luna-20261106/attempt-gpu-r1/runs/text-gpt6-luna/entities/02436645 \
  --output-root outputs/wn18rr-text-vision-20261107/<新的测试目录>
```

相关专项 **328 passed、3 skipped**，JUnit 在 `verification/final.xml`；跳过项为可选解码依赖。另一次检查包含了一个仍使用已移除 CLI 参数的旧测试，它在参数解析处失败；其测试和 parser 均未改动，单列记录在 `verification/initial.xml`，不据此宣称全库通过。新测试核对真实请求像素、名称／关系同传、图片来源绑定、隐藏目标隔离、隔离图片排除及旧配置的纯文本请求。

本轮没有申请 GPU、没有重新出图，没有启动多实体或全量任务。
