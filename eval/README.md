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
| q1 | 父块还原 | PASS（但只有子块片段） | 同 v0 | 同 v0 | PASS | PASS | PASS（还原 822 字完整父块——质变） | |
| q2 | FAQ | PASS | 同 v0 | 同 v0 | PASS | PASS | PASS（渲染成 Q/Answer 结构） | |
| q3 | 重叠拼接 | MISS | 同 v0 | 同 v0 | MISS（deploy-01/02 已进 top-2，03 在第 4——差一步） | MISS（阈值滤掉 02/03，只剩 01——需要的是拼接不是排序） | MISS（扩展拉回 02 即达 350 停止线，03 缺席——词面 mock 的诚实边界；拼接机制本身有「拼回=原文切片」单测证明） | |
| q4 | 短块扩展 | PASS（侥幸双命中） | 同 v0 | 同 v0 | PASS | **MISS（诚实回归：sdk-03 只答半个问题，模型分 0.158<0.2 被滤；v5 邻居扩展不依赖它被检索到）** | **PASS**（sdk-03 作为邻居拼回：sdk-02(+sdk-01,sdk-03)——回归治愈） | |
| q5 | MMR 去冗余 | MISS（检回旧版漏掉新版） | 同 v0 | 同 v0 | MISS（但 rel-v1/v2 首次同入候选池——v4 的问题真实产生了） | **PASS**（复合分数把 rel-v2 抬进 top-3） | PASS | |
| q6 | 关键词命中 | PASS*（mock 即词面匹配） | 同 v0 | 同 v0 | PASS（BM25 背书，星号摘除） | PASS | PASS（err-02 扩出前后邻居，错误码语境完整） | |
| q7 | 语义命中 | MISS（词面零重叠） | 同 v0 | 同 v0 | **PASS**（两路弱信号 RRF 叠加救回） | PASS（升至第 1） | PASS（bp-01(+bp-02)） | |
| q8 | 历史依赖 | MISS（不读历史必跑偏） | 同 v0 | **PASS**（改写补全省略后 faq-03 进 top-3；但 faq-04 仍排更前，排序留给 v4） | PASS | PASS（faq-03 终于压过 faq-04——排序债还清） | PASS（历史注入按 existingIDs 静默让路——机制间不打架） | |
| q9 | 关键词命中 | PASS* | 同 v0 | 同 v0 | PASS（BM25 背书） | PASS | PASS（err-03(+err-02)） | |

> v5 起判定升级为**覆盖命中**：期望块 id ∈ 结果自身 id ∪ `sub_chunk_ids`
> （合并账本，对照 WeKnora 的 `SubChunkID`）。内容进了 prompt 才算数——
> 按检索 id 判会漏记合并阶段的功劳。

v1 与 v0 逐题、逐命中列表完全一致（5/9）是**刻意的**：v1 只做结构重构
（洋葱式插件链），不加检索能力；这条不变量由
`tests/test_v1_pipeline_engine.py::test_v1_retrieval_identical_to_v0_on_all_eval_queries`
逐题守护。

逐题原因见 [`../docs/v0-朴素检索问答.md`](../docs/v0-朴素检索问答.md) 的评测记录；
后续版本实现后在此表补列。
