# llm-rag-learning 开发提示词（原始规格，权威版本）

> 本文件是本项目的原始开发提示词，逐字保留设计决策，作为 v0–v6 全程的权威规格。
> 上下文压缩后，从这里恢复完整要求。执行进度见文末「进度追踪」。

---

你是一名资深检索增强生成（RAG）系统工程师、软件架构师和代码教学专家。

## 〇、背景事实（均已对照源码核实，勿凭空更改）

- **待分析的企业项目**：`C:\Desktop\Project\WeKnora`。LLM 知识库框架，本项目只学习其
  **RAG Chat Pipeline** 部分（问答时「检索 → 处理 → 生成」的链路），明确不涉及：
  Wiki 双范式生成（姊妹项目 `llm-wiki-learning` 已覆盖，见下）、文档摄取/切分
  （chunking）/ Embedding 生成流程（docreader 是另一条独立课程线）、向量数据库
  多引擎注册表、ReAct Agent、多租户/RBAC/asynq 并发治理。

- **上一轮课程差距分析的结论**（2026-07-23，7 个子系统并行源码深读）：
  `llm-wiki-learning` 完整复刻了 Wiki 双范式，但「RAG chat pipeline 全部算法、
  检索引擎注册表、docreader、embedding 生成」是**完全没有覆盖**的档位。本项目
  就是补上这块——具体来说是补「洋葱式中间件链+装饰器」「加权 RRF 融合」两个
  当时点名「价值高、依赖低」的知识点，外加合并/精排/MMR 等同等重要但当时未及
  展开的算法。`wiki_ingest_cite.go` 的 chunk 级引用分类+prefix caching 是 Wiki
  摄取专属机制，不属于 chat pipeline，本项目不覆盖，留给 `llm-wiki-learning`
  未来扩展。

- **WeKnora RAG Chat Pipeline 的真实架构**
  （`internal/application/service/chat_pipeline/`）：一个「洋葱式中间件」插件链。
  每个 Plugin 实现
  `OnEvent(ctx, eventType, chatManage, next func() *PluginError) *PluginError`
  （chat_pipeline.go:11-21）。`EventManager.buildHandler`（chat_pipeline.go:53-68）
  把同一 event 的插件切片**右折叠**成嵌套闭包——plugin[0] 包住 plugin[1] 包住
  ... 包住终止 no-op，调用 `next()` 才会继续链条，这与 Express/Koa 中间件同构。
  注册顺序（container.go:322-336）决定链序。WikiBoost 插件展示了「先 `next()`
  后处理」的装饰器写法（wiki_boost.go:42-97：先跑完排序，再对 wiki 类型分数
  ×1.3 重排）。

  一次正常问答的插件触发顺序（`session_knowledge_qa.go:658-795` 驱动）：
  `LOAD_HISTORY` → `QUERY_UNDERSTAND` → `CHUNK_SEARCH_PARALLEL`（内部并行跑
  chunk 检索 + 实体/图检索；chunk 检索内部在召回不足时会触发 query_expansion
  做一次本地启发式查询扩展并重新检索，这不是独立的 pipeline 事件，是
  search 阶段的内部行为）→ `CHUNK_RERANK`（Rerank → WikiBoost 装饰）→
  `WEB_FETCH`（可选）→ `CHUNK_MERGE`（去重/父块还原/重叠拼接/FAQ 格式化/
  历史相关片段/短块扩展）→ `FILTER_TOP_K` → `DATA_ANALYSIS`（可选）→
  `INTO_CHAT_MESSAGE` → `CHAT_COMPLETION_STREAM`。
  **精排在合并之前**——先从原始检索结果里选出最相关的子集，合并阶段才对着
  这个更小的子集做父块还原/拼接等较贵的操作，这是刻意的效率设计，课程要讲清楚。

  核心算法（学习项目必须忠实复刻其数学/逻辑，不用「差不多」的近似）：
  - 归一化：仅向量分数按引擎归一化到 `[0,1]`（Milvus 原始余弦 `(score+1)/2`，
    其余引擎 clamp01），关键词/BM25 分数不归一化，因为 RRF 只看排名不看分数
    （normalizer.go:16-18,110-158）。
  - RRF 融合：`score = vecWeight/(k+vecRank) + kwWeight/(k+kwRank)`，k=60，
    向量权重 0.7、关键词权重 0.3（knowledgebase_search_fusion.go:80-142，
    retrieval_config.go:80-103）；只有向量与关键词结果都存在时才融合，否则
    退化为按分数去重（knowledgebase_search_fusion.go:33-50）。
  - 查询扩展：去停用词关键词、引号短语、分隔符切分、去问句词（
    query_expansion.go:136-160），对每个变体并发重新检索（上限 16 并发，
    query_expansion.go:36-43）。
  - 精排：真实 cross-encoder 模型调用（rerank.go:345），复合分数
    `0.6*模型分 + 0.3*基础分 + 0.1*来源权重`（rerank.go:439-460），阈值不达标
    时降级重试一次（阈值 ×0.7，下限 0.3，rerank.go:148-179），最终用 MMR
    （`λ*相关性 - (1-λ)*冗余度`，λ=0.7）做多样性选择而非直接取分数前 N
    （rerank.go:223,463-543）。
  - 合并：父块还原（子块→完整 parent_text，merge.go:215-341）、重叠片段拼接
    （按 StartAt 排序后同 KnowledgeID+ChunkType 分组拼接，merge_overlap.go:15-78）、
    FAQ 专属格式化（merge_faq.go:11-97）、历史相关片段（Jaccard≥0.15 保留、
    分数打 6 折、最多 3 条，merge_history.go:17-85）、短块用邻居扩展（<350 字
    扩到 850 字，merge_expand.go:10-252）。

- **教学参考项目（学的是它的风格，不是代码）**：`C:\Desktop\Project\learn-claude-code`。
  v0–v4 五个自包含单文件脚本，每个版本只引入一个核心概念，配一篇「为什么」优先于
  「怎么做」的技术文档，附版本对比表（行数/工具/新增/关键洞察），根 README 有
  ASCII 学习路径图和 5 行核心模式代码块。核心哲学「模型是 80%，代码是 20%」在
  RAG pipeline 里要修正：多数阶段是确定性代码（排序、融合、合并——不需要 API
  Key 也该能完整跑通），只有查询理解、精排模型调用、最终生成三处才真正依赖
  LLM，每个版本和 README 都要清楚标注每一步是「代码算法」还是「模型调用」。

- **姊妹项目**：`C:\Desktop\Project\llm-wiki-learning`（同级目录，其
  `docs/SPEC.md` 是该项目的权威规格），已覆盖 Wiki 双范式（确定性流水线 +
  ReAct Agent）。本项目与它共享「渐进版本、对照 WeKnora 源码、Mock 优先」的
  方法论，但采用 learn-claude-code 的扁平单文件命名（`v0_xxx.py`）而不是
  llm-wiki-learning 的分阶段目录（`s01_xxx/`）——这是这次明确选定的风格，
  两个姊妹项目并列在根 README 互相引用，不合并。

- **学习项目位置**：`C:\Desktop\Project\llm-rag-learning`（与 WeKnora 同级）。

- **LLM/Embedding 接口**：OpenAI 兼容 API（`openai` SDK），`.env` 配置
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_ID` / `EMBEDDING_MODEL_ID`。
  向量检索用最小可行方案（numpy 余弦相似度），不引入外部向量数据库——学习者
  应该读懂检索本身，而不是学一个新的数据库 SDK。关键词检索用 `rank_bm25`。

- **安全红线**：语料全部是本项目自带的教学素材（`corpus/`），不读取 WeKnora
  或用户系统上的真实文件，不把真实密钥/隐私内容发给 LLM。

- **每个版本必须有一篇 `docs/vN-*.md`**，包含：这一版本新增了什么、核心洞察、
  代码走读、与上一版本的 diff 意味着什么、**对照 WeKnora 真实实现**（给出具体
  文件:行号）、这一版本又牺牲/简化了什么。

## 一、项目目标

在 `C:\Desktop\Project\llm-rag-learning` 创建一个 v0→v6 渐进式课程，让学习者从
「朴素检索拼 prompt」出发，亲手搭出 WeKnora 生产级 RAG Chat Pipeline 的每一个
关键机制，最终能回答：

1. 洋葱式中间件插件链如何用 `next()` 实现「前置+后置」双向切面，注册顺序为何
   即链序；
2. 为什么查询理解要放在检索之前、用一次 LLM 调用做改写和意图分类；
3. 混合检索（向量+关键词）为什么要先各自归一化、再用 RRF 按排名（而不是分数）
   融合，召回不足时又如何用零成本的启发式规则做查询扩展；
4. 精排为什么不能只看相关性分数——复合分数与 MMR 多样性选择解决什么问题，
   以及为什么精排要放在合并之前（先筛后拼，省掉对陪跑结果做昂贵合并的成本）；
5. 合并阶段的四种「手术」（父块还原、重叠拼接、FAQ 格式化、短块扩展）为什么
   检索选完之后还需要一整个阶段把碎片拼回可读上下文；
6. 装饰器式插件（WikiBoost 模式）如何在不修改上游插件代码的前提下叠加行为，
   以及流式生成如何与「插件链等待 next() 返回」的同步语义解耦共存。

范围边界（明确排除，避免与 llm-wiki-learning / 未来课程重叠）：
- 不涉及文档摄取、切分（chunking）、Embedding 生成流程——`corpus/` 里的语料
  直接就是切好的、标好元数据的 chunk，直接进课程；
- 不涉及向量数据库选型/多引擎注册表（`RETRIEVE_DRIVER`）——用一个内存实现
  代表「一个引擎」，重点是融合算法而不是存储引擎适配；
- 不涉及实体/图检索（`search_entity.go`，Neo4j 门控特性）、`WEB_FETCH`、
  `DATA_ANALYSIS`——这些是可选旁支功能，不是 RAG 的核心算法路径；
- 不涉及 Wiki 双范式、ReAct Agent（那是 `llm-wiki-learning` 的范围）；
- 不涉及多租户/RBAC/asynq 并发治理（工程基础设施，不是 RAG 算法本身）。

## 二、设计原则

延续 `llm-wiki-learning` 验证过的方法论，针对 RAG pipeline 的特点做调整：

### 1. 每个版本只引入一个核心概念
不在同一版本里同时讲架构和算法。v0 只证明「检索+拼接」的最小骨架；v1 只讲清楚
插件链机制本身（用最简单的两个插件+一个装饰器说明 `next()`），不引入检索算法
的复杂度；v2 起才在这个骨架上逐步叠加能力，且每次只加一层。

### 2. 同一份语料、同一组问题贯穿所有版本
`corpus/` 和 `eval/queries.json`（见四、七节）从 v0 到 v6 保持不变。学习者应该
能拿同一个问题跑 v0 和 v6，直接对比答案质量的差异——这是本课程最强的教学信号，
比任何文字解释都直观。

### 3. 代码路径 vs 模型路径要显式标注
每个版本的代码注释和 README 都要标出：这一步是纯代码算法（排序、融合、拼接——
确定性、可单元测试、不需要 API Key 也能跑）还是模型调用（查询理解、精排、
生成——需要真实/Mock LLM）。测试套件对纯代码路径要有不依赖网络的单元测试。

### 4. 忠实复刻算法参数，不用「差不多」的近似
RRF 的 k=60、权重 0.7/0.3，MMR 的 λ=0.7，复合分数的 0.6/0.3/0.1，短块阈值
350/850，Jaccard 阈值 0.15——这些数字是 WeKnora 生产调参的产物，直接照抄比
自己发明一套「差不多」的公式更有教学价值，README 要说明每个数字抄自哪个文件。

### 5. 每版本独立可运行，相邻版本允许大量重复
延续 learn-claude-code 的扁平单文件风格：`vN_xxx.py` 直接放仓库根目录，不拆成
包。学习者应该能直接 diff 两个版本文件看到增量做了什么。

### 6. Mock 优先
向量检索、融合、合并、MMR 部分不需要真实 API 也能完整跑通（用固定的假
embedding 或 hash 向量）；只有查询理解、精排模型调用、最终生成三处需要
真实/Mock LLM，提供 `--mock` 模式，无 Key 也能跑通整条链路做结构验证。

## 三、总体目录结构

```
llm-rag-learning/
├── README.md                          # 学习路径总览、快速开始、版本对比表
├── docs/
│   ├── SPEC.md                        # 本文件
│   ├── v0-朴素检索问答.md
│   ├── v1-洋葱中间件引擎.md
│   ├── v2-查询理解.md
│   ├── v3-混合检索与RRF融合.md
│   ├── v4-精排与MMR多样性.md
│   ├── v5-上下文合并四术.md
│   └── v6-装饰器增强与流式输出.md
├── corpus/                            # 共享教学语料（贯穿所有版本，见四节）
│   ├── docs/*.md                      # 原始知识库文档（人工设计，覆盖各机制）
│   └── chunks.json                    # 预切好的 chunk（含 parent/child、FAQ 等元数据）
├── eval/
│   └── queries.json                   # 贯穿所有版本的固定评测问题集（见四节）
├── v0_naive_rag.py
├── v1_pipeline_engine.py
├── v2_query_understanding.py
├── v3_hybrid_fusion.py
├── v4_rerank_mmr.py
├── v5_merge.py
├── v6_boost_stream.py
├── tests/
│   └── test_*.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## 四、语料与评测集设计要求

`corpus/docs/` 是人工设计的小型知识库（不是真实抓取的数据），必须覆盖以下机制，
每个机制至少对应一份文档或一组 chunk：

1. 一篇较长文档，切分后产生「子块（child chunk）」与「完整父块（parent_text）」
   两级结构，用于 v5 父块还原；
2. 一份 FAQ 文档（Q/A 对），用于 v5 的 FAQ 格式化；
3. 至少两个相邻/重叠的 chunk（同 KnowledgeID，StartAt 有重叠），用于 v5 重叠
   拼接；
4. 若干短块（<350 字），有明确的前驱/后继邻居，用于 v5 短块扩展；
5. 至少一对近重复内容（不同文档但语义高度重合），用于 v3 RRF 融合去重、v4
   MMR 冗余抑制的对比演示；
6. 至少一份文档包含只能被关键词精确命中、语义向量检索容易漏检的内容（如特定
   错误码、专有名词、版本号），用于证明「为什么需要混合检索」；
7. 至少一份用词与常见提问方式不同但语义相关的文档，用于证明「为什么单纯关键词
   检索也不够」（即向量检索存在的理由）；
8. 一小段多轮对话历史样例，用于 v5 历史相关片段过滤（Jaccard 相关性）。

`chunks.json` 直接是预处理好的 chunk 列表（跳过真实的切分/Embedding 生成过程，
这不是本课程范围），每个 chunk 至少含：
`id, knowledge_id, chunk_type(text|parent_text|faq|image), content, start_at,
end_at, pre_chunk_id, next_chunk_id, parent_chunk_id`。

`eval/queries.json` 是贯穿所有版本、固定不变的评测问题集，每条至少含
`{query, expect_chunk_ids, note}`，覆盖上面 8 类场景各至少一题（例如一题必须
靠关键词命中、一题必须靠向量命中、一题依赖历史、一题应该触发 MMR 去冗余）。

## 五、渐进式实现版本

**v0：朴素检索问答（Naive RAG）**
无框架、无插件链，一个线性脚本：embedding 语料 → embedding 查询 → 余弦 top-k
→ 直接拼进 prompt → 调 LLM。~60-80 行。
核心洞察：检索+拼接就是 RAG 的最小骨架，后续所有版本都是在这个骨架上做手术。
README 必须包含一段「这个版本会在 eval/queries.json 里哪些问题上答错/答不好」，
为后续版本的存在理由埋伏笔。
对照 WeKnora：不对应任何单一文件——这是刻意省略 WeKnora 全部工程化的基线。

**v1：洋葱式中间件插件引擎**
引入 Plugin 接口（`on_event(ctx, state, next) -> state`）与 EventManager，
右折叠构建调用链（对照 `chat_pipeline.go:53-68` 的 `buildHandler`）。用两个
极简插件复刻 v0 的逻辑（一个 Search 插件、一个 Generate 插件，不新增检索能力），
再加一个 LoggingBoost 装饰器插件：注册在 Search 之后、同一事件，先调用
`next()` 再对结果做一次简单后处理（如加一条 debug 字段）——直接复刻
`wiki_boost.go:42-97`「先 `next()` 后处理」的写法，为 v6 的完整版本埋伏笔。
核心洞察：`next()` 就是整个中间件系统的秘密；注册顺序即链序。
对照 WeKnora：`chat_pipeline.go:11-68`（Plugin 接口、buildHandler、Register）、
`wiki_boost.go:42-97`（装饰器写法参考）。

**v2：查询理解**
新增查询理解插件，注册在 Search 之前：一次 LLM 调用做查询改写 + 意图分类，
输出结构化 `{rewrite_query, intent}`（对照 `query_understand.go:58-161`），
下游 Search 插件改用 `rewrite_query` 而不是用户原始输入。
核心洞察：查询理解是「花一次 LLM 调用换取更准的检索输入」，而且必须发生在
检索之前——它是链条里第一个真正做事的插件。
对照 WeKnora：`query_understand.go:58-161`（结构化改写+意图分类）。

**v3：混合检索、RRF 融合与低召回扩展**
把 v0 的单路向量检索升级成向量+关键词（BM25）并行检索（`asyncio.gather`/
线程池，对照 Go 的 goroutine 并行，`search.go:99-118,317-461`）；每路结果先
各自归一化（向量走 clamp/线性映射，关键词不归一化——照抄 `normalizer.go:16-18`
的理由）；再用 RRF 融合（k=60，向量权重 0.7/关键词权重 0.3，照抄
`knowledgebase_search_fusion.go:80-124` 的公式），只有两路都有结果时才融合，
否则退化为单路去重（对照 `:33-50`）；当融合后结果数低于阈值时，触发查询扩展——
不再调用 LLM，而是本地启发式生成查询变体（去停用词关键词、引号短语、分隔符
切分，照抄 `query_expansion.go:136-160`），对每个变体并发重新检索。
核心洞察：RRF 不比较分数量纲，只比较排名，这是它能跨异构检索引擎融合的原因；
查询扩展是「零成本换召回率」，和 v2 查询理解「一次 LLM 调用换准确率」形成对照。
对照 WeKnora：`search.go:99-118,317-461`（并行检索）、
`knowledgebase_search_fusion.go:33-142`（融合调度与 RRF 公式）、
`normalizer.go:16-18,110-158`（按引擎归一化）、
`retrieval_config.go:80-103`（k=60、权重 0.7/0.3 默认值）、
`query_expansion.go:15-171`（启发式扩展规则与并发重检索）。

**v4：精排与 MMR 多样性选择**
新增精排插件，注册在 Search 之后、Merge 之前（对照真实链序：`CHUNK_RERANK`
先于 `CHUNK_MERGE`）：真实/Mock cross-encoder 打分（对照 `rerank.go:345`）、
复合分数 `0.6*模型分+0.3*基础分+0.1*来源权重`（照抄 `rerank.go:439-460`）、
阈值不达标时降级重试（阈值×0.7，下限 0.3，照抄 `rerank.go:148-179`）、最终用
MMR（`λ*相关性-(1-λ)*冗余度`，λ=0.7）做多样性选择替代直接取分数前 N（照抄
`rerank.go:463-543`）。用 corpus 里预埋的近重复内容（四节第 5 点）直接演示
「不用 MMR 会返回两条几乎一样的结果」。
核心洞察：相关性最高的 top-N 经常互相冗余，MMR 用一个参数在相关性和多样性之间
调节；精排放在合并之前是效率设计——先筛出小集合，合并阶段才不用对着一堆陪跑
结果做父块还原等较贵的操作。
对照 WeKnora：`rerank.go:148-179,223,345,439-460,463-543`。

**v5：上下文合并四术**
在精排之后插入合并阶段，实现四个独立可测的子步骤：父块还原（子块→完整
`parent_text`，照抄 `merge.go:215-341` 的判定逻辑）、重叠片段拼接（按
`StartAt` 排序、同 `KnowledgeID`+`ChunkType` 分组拼接，照抄
`merge_overlap.go:15-78`）、FAQ 专属格式化（照抄 `merge_faq.go:11-97` 的
Q/A 渲染）、短块邻居扩展（<350 字符扩展到 850 字符，照抄
`merge_expand.go:10-252`），外加历史相关片段过滤（Jaccard≥0.15、分数打 6 折、
最多 3 条，照抄 `merge_history.go:17-85`）；最后按 `filter_top_k.go` 的逻辑
截断到固定条数（排序后简单切片，是整条链路里最朴素的一步，不单独开版本）。
核心洞察：检索选完的是碎片，合并阶段才是把碎片重新变回「人能读的上下文」的
地方——这一步的工程量不比检索算法本身小。
对照 WeKnora：`merge.go:44-389`、`merge_overlap.go:15-78`、
`merge_faq.go:11-97`、`merge_expand.go:10-252`、`merge_history.go:17-85`、
`filter_top_k.go`（截断逻辑）。

**v6：装饰器增强与流式输出**
把 v1 里简化的 LoggingBoost 换成一个真正有意义的装饰器插件（对照
`wiki_boost.go:42-97`：先 `next()` 跑完精排排序，再对特定类型结果分数 ×1.3
重排），再把最终生成从一次性返回改成流式输出（对照
`chat_completion_stream.go` 的 fire-and-forget 语义：`OnEvent` 起一个
goroutine 消费流并往外发事件后立刻 `return next()`，不等流跑完），Python
版本用后台线程/`asyncio.Queue` 模拟同样的「插件不等生成完成」解耦写法；
同时实现 `into_chat_message.go:120-187` 的上下文模板渲染，把 v5 的合并结果
变成最终喂给 LLM 的 prompt。
核心洞察：`next()` 先行的装饰器模式不只是教学玩具，WikiBoost 是它在生产代码里
的真实用法；流式输出证明「插件链」和「流式生成」这两个正交的机制可以共存、
互不阻塞。
对照 WeKnora：`wiki_boost.go:37-97`、`into_chat_message.go:120-187`、
`chat_completion_stream.go:41-238`。

## 六、算法参数速查表（写入 README，供学习者对照）

| 参数 | 值 | 来源 |
|------|-----|------|
| RRF k | 60 | retrieval_config.go:82-85 |
| RRF 向量权重 / 关键词权重 | 0.7 / 0.3 | retrieval_config.go:88-103 |
| MMR λ | 0.7 | rerank.go:223 |
| 精排复合分数权重 | 模型0.6 / 基础0.3 / 来源0.1 | rerank.go:439-460 |
| 精排阈值降级系数 / 下限 | ×0.7 / 0.3 | rerank.go:148-179 |
| 短块扩展阈值 | <350 扩到 850（字符） | merge_expand.go:16-17 |
| 历史片段 Jaccard 阈值 / 折扣 / 上限 | 0.15 / ×0.6 / 3 条 | merge_history.go:17-85 |
| WikiBoost 因子 | ×1.3 | wiki_boost.go:15 |
| 默认 embedding_top_k / rerank_top_k | 30 / 30（YAML）或 50/10（租户默认） | config/config.yaml:9-38，retrieval_config.go |

## 七、测试要求

纯代码路径（归一化、RRF、合并四术、MMR）必须有不依赖网络/API Key 的单元测试，
用固定输入验证具体数值（如「给定这两路排名，RRF 分数应该等于 X」）。模型调用
路径（查询理解、精排、生成）提供 Mock LLM，覆盖：结构化输出解析、精排降级
重试路径、流式分块输出的正确拼接。每个版本至少一个端到端测试：跑
`eval/queries.json` 里的固定问题，断言检索到预期的 chunk id（不断言生成文本，
生成文本不适合做精确断言）。

## 八、README 要求

根 README（照抄 learn-claude-code 的结构）：为什么做这个项目、学习路径图
（v0→v6 ASCII 图，仿照 learn-claude-code 的箭头链）、快速开始、版本对比表
（行数/新增机制/核心洞察/对照 WeKnora 文件）、核心模式代码块（洋葱链的 5 行
伪代码，仿照 learn-claude-code 的 5 行 agent loop）、算法参数速查表（本 SPEC
六节）、与 `llm-wiki-learning` 的关系（同系列姊妹项目，覆盖 WeKnora 不同的
子系统）。
每版本 README（`docs/vN-*.md`）：核心洞察 / 完整代码或关键片段 / 这一版本
证明了什么 / 牺牲了什么（下一版本会补上什么）/ 对照 WeKnora 真实实现（具体
文件:行号）/「返回 README」链接。

## 九、执行方式

本次不实现任何 vN 代码，只做：

1. 创建 `C:\Desktop\Project\llm-rag-learning` 目录骨架（README.md、docs/SPEC.md、
   requirements.txt、.env.example、.gitignore、corpus/、eval/、tests/ 的占位）；
2. 根 README 写清楚 v0–v6 路线图、版本对比表（此时「新增机制/核心洞察/对照
   WeKnora」列按本 SPEC 五节填入，「行数」列留空待实现后补）；
3. git init（不 commit，留给用户确认后自己提交）；
4. 停止，不实现 v0，不生成 corpus 内容，等待下一次会话按本 SPEC 五节顺序实现，
   实现方式参照 `llm-wiki-learning` 的执行节奏：一次会话通常只推进一个版本，
   完成后运行验证、更新本文件末尾的进度追踪和根 README 的「当前进度」。

---

## 进度追踪（随开发更新）

- [x] v0_naive_rag（2026-07-24 完成。同批交付贯穿全程的语料与评测集：
  corpus/docs/ 8 份文档、chunks.json 24 条（含父子块/FAQ/重叠块/短块/近重复对/
  仅关键词/仅语义语料）、history_sample.json、eval/queries.json 9 题覆盖 8 类
  场景。mock 评测检索命中 5/9，q3/q5/q7/q8 如预期答不好，逐题记录在
  docs/v0-朴素检索问答.md）
- [x] v1_pipeline_engine（2026-07-24 完成。Plugin/EventManager/右折叠洋葱链 +
  Search/LoggingBoost/Generate 三插件，评测与 v0 逐题一致（5/9），重构不变量
  由测试逐题断言。行号全部对照当日源码核实；一个重要澄清写进了
  docs/v1-洋葱中间件引擎.md：WikiBoost 的「后置」地位来自「注册在后（内层）+
  next() 先行」的组合，它的 next() 打到的是终止 no-op，而非「调用 next() 去
  执行 Rerank」——Rerank(:323) 是外层，做完工作后在 rerank.go:265
  return next() 交棒）
- [ ] v2_query_understanding
- [ ] v3_hybrid_fusion
- [ ] v4_rerank_mmr
- [ ] v5_merge
- [ ] v6_boost_stream
