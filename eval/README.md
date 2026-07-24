# eval/

贯穿 v0–v6 所有版本、保持不变的固定评测问题集。设计要求见
[`../docs/SPEC.md`](../docs/SPEC.md) 第四节。

## queries.json

共 9 题，覆盖 corpus 的 8 类机制场景（关键词命中出两题）。每条字段：

- `id` / `scenario` / `query` — 题号、场景标签、用户问题
- `history`（仅 q8）— 本轮之前的多轮对话，与 `corpus/history_sample.json` 对应
- `expect_chunk_ids` — 判定检索通过的最小集合：**全部**出现在 top-k 里才算 PASS
- `note` — 这道题在考什么、预期哪个版本开始答对

判定只看检索命中（`expect ⊆ top-k`），不断言生成文本——生成文本不适合精确
断言（SPEC 第七节）。运行方式：

```bash
python v0_naive_rag.py --eval --mock    # 各版本同理：vN_xxx.py --eval --mock
```

## 各版本基线记录

| 题 | 场景 | v0(mock) | v1 | v2 | v3 | v4 | v5 | v6 |
|----|------|----------|----|----|----|----|----|----|
| q1 | 父块还原 | PASS（但只有子块片段） | 同 v0 | 同 v0 | | | | |
| q2 | FAQ | PASS | 同 v0 | 同 v0 | | | | |
| q3 | 重叠拼接 | MISS | 同 v0 | 同 v0 | | | | |
| q4 | 短块扩展 | PASS（侥幸双命中） | 同 v0 | 同 v0 | | | | |
| q5 | MMR 去冗余 | MISS（检回旧版漏掉新版） | 同 v0 | 同 v0 | | | | |
| q6 | 关键词命中 | PASS*（mock 即词面匹配） | 同 v0 | 同 v0 | | | | |
| q7 | 语义命中 | MISS（词面零重叠） | 同 v0 | 同 v0 | | | | |
| q8 | 历史依赖 | MISS（不读历史必跑偏） | 同 v0 | **PASS**（改写补全省略后 faq-03 进 top-3；但 faq-04 仍排更前，排序留给 v4） | | | | |
| q9 | 关键词命中 | PASS* | 同 v0 | 同 v0 | | | | |

v1 与 v0 逐题、逐命中列表完全一致（5/9）是**刻意的**：v1 只做结构重构
（洋葱式插件链），不加检索能力；这条不变量由
`tests/test_v1_pipeline_engine.py::test_v1_retrieval_identical_to_v0_on_all_eval_queries`
逐题守护。

逐题原因见 [`../docs/v0-朴素检索问答.md`](../docs/v0-朴素检索问答.md) 的评测记录；
后续版本实现后在此表补列。
