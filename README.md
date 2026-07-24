# llm-rag-learning — 从零搭建 WeKnora 级 RAG Chat Pipeline

**亲手实现一次生产级 RAG 问答链路。**

> 本项目是姊妹项目 [llm-wiki-learning](../llm-wiki-learning) 的另一半：那边覆盖
> WeKnora 的 Wiki 双范式（确定性流水线 + ReAct Agent），这里覆盖 WeKnora
> **问答时**的检索-处理-生成链路（RAG Chat Pipeline）。两者都以
> [WeKnora](../WeKnora) 的真实源码为对照，风格上借鉴
> [learn-claude-code](../learn-claude-code) 的渐进式单文件教学法。

---

## 为什么有这个项目

WeKnora 的 RAG Chat Pipeline 是一条「洋葱式中间件」插件链：查询理解、混合检索、
RRF 融合、精排、MMR 多样性选择、上下文合并、装饰器增强、流式生成……每一步都是
一个独立可测的算法，很多细节（RRF 的排名融合、精排后置于合并之前的顺序、
WikiBoost 的 `next()` 装饰器写法）在只读文档或走马观花读源码时很容易被忽略。

本项目把这条链路拆成 v0→v6 七个渐进版本，每个版本只引入一个核心概念，忠实照抄
WeKnora 生产代码里的具体参数（RRF k=60、MMR λ=0.7、复合分数权重 0.6/0.3/0.1
……），而不是发明一套「差不多」的简化算法——目标是学完之后，回头去读
`internal/application/service/chat_pipeline/` 的真实代码时能做到「似曾相识」。

## 你将学到什么

完成本教程后，你将理解：

- **洋葱式中间件插件链** — `next()` 如何实现前置+后置双向切面，注册顺序为何
  即链序
- **查询理解** — 为什么检索之前先花一次 LLM 调用做改写和意图分类
- **混合检索与 RRF 融合** — 为什么向量分数要归一化、关键词分数不需要，RRF 为何
  只比较排名不比较分数
- **精排与 MMR 多样性** — 复合分数解决什么问题，为什么相关性最高的结果反而
  应该被 MMR 换掉一部分
- **上下文合并** — 父块还原、重叠拼接、FAQ 格式化、短块扩展这四种「碎片重组」
  手术
- **装饰器增强与流式输出** — 如何在不改上游插件代码的前提下叠加行为，以及
  流式生成如何与同步的插件链解耦共存

## 学习路径

```
从这里开始
    |
    v
[v0: 朴素 RAG] ---------> "检索+拼接就是全部骨架"
    |                      ~70 行
    v
[v1: 中间件引擎] -------> "next() 就是整个中间件系统的秘密"
    |                      +Plugin/EventManager，~150 行
    v
[v2: 查询理解] ---------> "先花一次 LLM 调用换准确率"
    |                      +QueryUnderstand，~200 行
    v
[v3: 混合检索+RRF] -----> "RRF 只比较排名，不比较分数"
    |                      +BM25/归一化/RRF/查询扩展，~320 行
    v
[v4: 精排+MMR] ---------> "相关性最高常常等于互相冗余"
    |                      +复合分数/MMR，~420 行
    v
[v5: 合并四术] ---------> "检索给碎片，合并给上下文"
    |                      +父块/重叠/FAQ/短块扩展，~550 行
    v
[v6: 装饰器+流式] ------> "next() 先行的装饰器 + 流式解耦"
                           +WikiBoost模式/streaming，~620 行
```

**推荐学习方式：**
1. 先跑一遍 `eval/queries.json` 里的问题在 v0 上的表现，记住答得差的地方
2. 逐版本往后跑，每次重跑同一组问题，观察答案质量的变化
3. 每读完一个版本的代码，去 `docs/vN-*.md` 对照 WeKnora 真实源码的文件:行号
4. 对比相邻两个版本文件的 diff，确认自己能看出增量到底做了什么

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（可选——Mock 模式下多数版本不需要真实 Key）
cp .env.example .env

# 运行 v0：单条提问 / 跑完整评测集（--mock 无需任何 Key）
python v0_naive_rag.py --query "如何重置青鸟的 API 密钥？" --mock
python v0_naive_rag.py --eval --mock

# 单元测试（全部离线）
python -m pytest tests/
```

## 核心模式

七个版本共享同一个洋葱式中间件骨架（v1 起）：

```python
def build_handler(plugins):
    handler = lambda state: state          # 终止 no-op
    for plugin in reversed(plugins):
        handler = lambda state, p=plugin, nxt=handler: p.on_event(state, nxt)
    return handler                          # 调用它 = 触发整条链
```

插件只需要决定：在调用 `next()` 之前做什么（前置逻辑），在 `next()` 返回之后
做什么（后置逻辑，WikiBoost 用的就是这招）。其余一切——检索、融合、合并、
精排——都是往这条链上挂新插件。

## 版本对比表

| 版本 | 行数 | 新增机制 | 核心洞察 | 对照 WeKnora |
|------|------|----------|----------|--------------|
| v0 | 131 | 单路向量检索 | 检索+拼接就是最小骨架 | （刻意省略，作为基线） |
| v1 | ~150 | Plugin / EventManager / `next()` | 中间件链的秘密就是 `next()` | `chat_pipeline.go:11-68` |
| v2 | ~200 | 查询理解（LLM 改写+意图分类） | 检索前先花一次调用换准确率 | `query_understand.go:58-161` |
| v3 | ~320 | 混合检索 + 归一化 + RRF + 查询扩展 | RRF 只看排名不看分数 | `search.go`、`knowledgebase_search_fusion.go`、`normalizer.go`、`query_expansion.go` |
| v4 | ~420 | 复合分数精排 + MMR 多样性 | 精排在合并之前，为效率而设计 | `rerank.go` |
| v5 | ~550 | 父块还原/重叠拼接/FAQ/短块扩展 | 合并阶段把碎片拼回可读上下文 | `merge.go`、`merge_overlap.go`、`merge_faq.go`、`merge_expand.go`、`merge_history.go` |
| v6 | ~620 | 装饰器插件 + 流式输出 | `next()` 先行装饰器 + 流式解耦 | `wiki_boost.go`、`into_chat_message.go`、`chat_completion_stream.go` |

> v0 行数为实测值（核心链路约 70 行，其余为 --mock/--eval 辅助）；
> 未实现版本的行数为设计阶段的粗略估计。

## 算法参数速查表

这些数字直接照抄 WeKnora 生产代码，不是本课程发明的近似值：

| 参数 | 值 | 来源 |
|------|-----|------|
| RRF k | 60 | `retrieval_config.go:82-85` |
| RRF 向量权重 / 关键词权重 | 0.7 / 0.3 | `retrieval_config.go:88-103` |
| MMR λ | 0.7 | `rerank.go:223` |
| 精排复合分数权重 | 模型 0.6 / 基础 0.3 / 来源 0.1 | `rerank.go:439-460` |
| 精排阈值降级系数 / 下限 | ×0.7 / 0.3 | `rerank.go:148-179` |
| 短块扩展阈值 | <350 扩到 850（字符） | `merge_expand.go:16-17` |
| 历史片段 Jaccard 阈值 / 折扣 / 上限 | 0.15 / ×0.6 / 3 条 | `merge_history.go:17-85` |
| WikiBoost 因子 | ×1.3 | `wiki_boost.go:15` |

## 文件结构

```
llm-rag-learning/
├── v0_naive_rag.py            # 朴素检索问答
├── v1_pipeline_engine.py      # + 洋葱中间件引擎
├── v2_query_understanding.py  # + 查询理解
├── v3_hybrid_fusion.py        # + 混合检索/RRF/查询扩展
├── v4_rerank_mmr.py           # + 精排/MMR
├── v5_merge.py                # + 合并四术
├── v6_boost_stream.py         # + 装饰器/流式输出
├── corpus/                    # 贯穿所有版本的共享教学语料
├── eval/                      # 贯穿所有版本的固定评测问题集
├── docs/                      # 技术文档（SPEC.md + 各版本说明）
└── tests/                     # 单元测试与端到端测试
```

## 深入阅读

- [docs/SPEC.md](./docs/SPEC.md) — 本项目的权威开发规格（完整分阶段设计、
  WeKnora 源码映射、进度追踪）
- 各版本说明文档（`docs/vN-*.md`）随每个版本的实现逐步补充

## 与 llm-wiki-learning 的关系

两个项目是同系列姊妹项目，共享方法论（渐进版本、对照 WeKnora 真实源码、
Mock 优先、每版本独立可运行），但覆盖 WeKnora 的不同子系统，互不重叠：

| | llm-wiki-learning | llm-rag-learning（本项目） |
|---|---|---|
| 覆盖范围 | Wiki 生成（确定性流水线）+ Wiki 问答/修复（ReAct Agent） | 问答时的检索-处理-生成链路（RAG Chat Pipeline） |
| WeKnora 对照 | `internal/application/service/wiki_ingest*.go`、`internal/agent/` | `internal/application/service/chat_pipeline/` |
| 文件组织 | 分阶段目录 `s01_xxx/`（每阶段一个文件夹） | 扁平单文件 `vN_xxx.py`（learn-claude-code 风格） |
| 核心命题 | 「模型即函数」vs「模型即决策者」，各归其位 | 检索质量是一连串独立算法的叠加，不是一个黑盒 |

## 项目状态

当前进度：**v0 已完成**（2026-07-24）— 共享语料（青鸟消息推送服务知识库，
24 chunk）与 9 题固定评测集已就绪，v0 在 mock 模式下检索命中 5/9，答不好的
q3/q5/q7/q8 逐题分析见 [docs/v0-朴素检索问答.md](./docs/v0-朴素检索问答.md)。

- [x] v0_naive_rag（131 行，mock 评测 5/9，缺口即 v1–v6 的存在理由）
- [ ] v1_pipeline_engine
- [ ] v2_query_understanding
- [ ] v3_hybrid_fusion
- [ ] v4_rerank_mmr
- [ ] v5_merge
- [ ] v6_boost_stream

## License

MIT

---

**检索是一连串独立算法的叠加，不是一个黑盒。**
