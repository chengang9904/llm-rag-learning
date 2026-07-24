# corpus/

贯穿 v0–v6 所有版本、保持不变的共享教学语料：一个虚构的「青鸟（Qingniao）
消息推送服务」知识库。设计要求见 [`../docs/SPEC.md`](../docs/SPEC.md) 第四节，
8 类机制场景与文档/chunk 的对应关系如下：

| # | SPEC 场景 | 文档 | chunk | 服务的版本 |
|---|-----------|------|-------|-----------|
| 1 | 父子块两级结构 | `docs/01-架构总览.md` | `arch-parent`（822 字父块）+ `arch-01..04`（子块） | v5 父块还原 |
| 2 | FAQ Q/A 对 | `docs/02-常见问题FAQ.md` | `faq-01..05`（chunk_type=faq） | v5 FAQ 格式化 |
| 3 | 相邻重叠块 | `docs/03-集群部署指南.md` | `deploy-01..03`（01∩02=98 字，02∩03=123 字） | v5 重叠拼接 |
| 4 | 短块+邻居 | `docs/04-Python-SDK参考.md` | `sdk-01..04`（均 <350 字，pre/next 链完整） | v5 短块扩展 |
| 5 | 近重复内容对 | `docs/05-…-v1.md` / `06-…-v2.md` | `rel-v1-01` / `rel-v2-01`（不同 knowledge_id，语义高度重合） | v3 RRF 去重、v4 MMR |
| 6 | 仅关键词可命中 | `docs/07-错误码与配置项.md` | `err-01..03`（E-4703、QN_MAX_INFLIGHT、2.3.1 等精确 token） | v3 混合检索（关键词路） |
| 7 | 仅语义可命中 | `docs/08-流量控制与背压.md` | `bp-01..02`（工程措辞，与口语提问零词面重叠） | v3 混合检索（向量路） |
| 8 | 多轮对话历史 | `history_sample.json` | ——（对应 eval 的 q8） | v5 历史相关片段过滤 |

## chunks.json

预切好的 chunk 列表（共 24 条；切分/Embedding 生成流程不属于本课程范围）。
每条字段：

```
id, knowledge_id, chunk_type(text|parent_text|faq), content,
start_at, end_at, pre_chunk_id, next_chunk_id, parent_chunk_id, source
```

两条硬性不变量（`tests/test_v0_naive_rag.py` 有校验，改语料前先看测试）：

1. `content` 与源文档 `text[start_at:end_at]` **逐字符一致**——v5 重叠拼接
   依赖偏移量是真实的；
2. `pre/next/parent` 链接必须相互一致且指向存在的 chunk——v5 短块扩展与
   父块还原依赖链接完整。

检索时只索引叶子块（`chunk_type != "parent_text"`），父块由 v5 合并阶段还原。
