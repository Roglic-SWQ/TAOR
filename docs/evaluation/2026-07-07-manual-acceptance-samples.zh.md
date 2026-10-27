# MMKGC 人工验收样本表

## 用途

本表用于人工验收当前 100 实体离线 evaluation 的评价标准是否符合直觉。样本来自：

- `best_cases.jsonl` 前 10 条
- `worst_cases.jsonl` 前 10 条
- `fallback_cases.jsonl` 全部 5 条

注意：当前 fallback case 只有 5 条，因此本轮 fallback 不硬凑 10 条。

建议人工打标字段：

- `human_judgement`：`合理` / `不合理` / `待讨论`
- `issue_type`：`text弱` / `image虚高` / `pair虚高` / `fallback应扣分` / `entity_type过泛` / `关系证据不足` / `其他`
- `notes`：人工备注

详细可编辑表见同目录 CSV：

`2026-07-07-manual-acceptance-samples.csv`

## Best 样本

| entity | label | overall | text | image | pair | 真实描述 | 自动观察 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Q336177 | EVE Online | 0.869880 | 0.664297 | 0.988008 | 0.957336 | 2003 persistent-world massively multiplayer online video game | text 与真实描述关系较强；检查高分是否符合直觉 |
| Q7499123 | Shit Hits The Fan | 0.869366 | 0.675256 | 0.990402 | 0.942441 | single by Eminem, Dr. Dre, Obie Trice | text 与真实描述关系较强；检查高分是否符合直觉 |
| Q3498834 | Westside Story | 0.863344 | 0.673311 | 0.987313 | 0.929409 | single by 50 Cent and The Game | text 与真实描述关系较强；检查高分是否符合直觉 |
| Q2492348 | See You in My Nightmares | 0.853361 | 0.625632 | 0.990689 | 0.943762 | single by Lil Wayne and Kanye West | text 与真实描述关系较强；检查高分是否符合直觉 |
| Q2752852 | New Day | 0.835259 | 0.595686 | 0.987313 | 0.922777 | 2012 single by 50 Cent ft. Dr. Dre and Alicia Keys | 综合样本；检查 text/image/pair 是否一致 |
| Q2665674 | Bucky Larson: Born to Be a Star | 0.818730 | 0.509041 | 0.987602 | 0.959548 | 2011 film by Tom Brady | 综合样本；检查 text/image/pair 是否一致 |
| Q20814652 | Death of a Friend | 0.818557 | 0.508535 | 0.987599 | 0.959537 | 1959 film by Franco Rossi | 综合样本；检查 text/image/pair 是否一致 |
| Q17479333 | The Cheeky Devil | 0.815368 | 0.519358 | 0.982698 | 0.944047 | 1932 film by Carl Boese | 综合样本；检查 text/image/pair 是否一致 |
| Q3857815 | The Violent Professionals | 0.815368 | 0.519358 | 0.982698 | 0.944047 | 1973 film by Sergio Martino | 综合样本；检查 text/image/pair 是否一致 |
| Q3712040 | Folgore Division | 0.815006 | 0.526769 | 0.980792 | 0.937456 | 1954 film by Duilio Coletti | 综合样本；检查 text/image/pair 是否一致 |

## Worst 样本

| entity | label | overall | text | image | pair | 真实描述 | 自动观察 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Q3401588 | Prem Chopra | 0.638978 | 0.346578 | 0.701292 | 0.869065 | Indian actor | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q132952 | Adam Sandler | 0.639582 | 0.344990 | 0.695414 | 0.878342 | American actor and comedian | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q234900 | Linda Darnell | 0.653218 | 0.359024 | 0.711604 | 0.889027 | American actress (1923-1965) | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q312719 | Kabir Bedi | 0.660599 | 0.366032 | 0.720053 | 0.895711 | Indian actor (1946-) | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q464578 | Karisma Kapoor | 0.661162 | 0.365909 | 0.720755 | 0.896821 | Indian actress (born 1974) | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q3522075 | M. N. Nambiar | 0.663947 | 0.366513 | 0.722219 | 0.903110 | Indian actor (1919-2008) | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q844972 | The WB | 0.665490 | 0.368780 | 0.719893 | 0.907798 | American television network | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q60493 | Lee Tergesen | 0.665759 | 0.369708 | 0.723462 | 0.904108 | American actor | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q185051 | Christopher Walken | 0.667411 | 0.342289 | 0.927851 | 0.732094 | American actor (born 1943) | text 明显偏低；检查是否缺少实体类型/职业/核心描述 |
| Q639095 | Avtomobilist Yekaterinburg | 0.667839 | 0.416412 | 0.702083 | 0.885023 | Professional ice hockey club based in Yekaterinburg, Russia | 综合样本；检查 text/image/pair 是否一致 |

## Fallback 样本

| entity | label | overall | text | image | pair | 真实描述 | 自动观察 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Q193559 | Valve Corporation | 0.741702 | 0.364737 | 0.966954 | 0.893416 | American video game development and digital distribution company | fallback 样本；需要重点判断泛化模板是否应被扣分 |
| Q3577037 | The West Wing | 0.748475 | 0.368000 | 0.969828 | 0.907597 | American political drama series created by Aaron Sorkin for NBC (1999-2006) | fallback 样本；需要重点判断泛化模板是否应被扣分 |
| Q3713428 | Don't Fall in Love with Everyone You See | 0.766441 | 0.377104 | 0.977995 | 0.944223 | album by Okkervil River | fallback 样本；需要重点判断泛化模板是否应被扣分 |
| Q3714788 | Down the River of Golden Dreams | 0.788911 | 0.459835 | 0.975099 | 0.931799 | 1993 album by Okkervil River | fallback 样本；需要重点判断泛化模板是否应被扣分 |
| Q95492 | Hans-Otto Borgmann | 0.748475 | 0.368000 | 0.969828 | 0.907597 | German composer (1901-1977) | fallback 样本；需要重点判断泛化模板是否应被扣分 |

## 打标后如何使用

人工打标完成后，建议统计：

1. best 样本中 `合理` 的比例是否达到 70%。
2. worst 样本中是否至少 70% 能解释出明确问题。
3. fallback 样本中有多少应该被显式扣分。
4. `image虚高` 和 `pair虚高` 是否频繁出现。
5. 是否需要先校准公式，再扩大到 500 实体。
