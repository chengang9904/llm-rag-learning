# -*- coding: utf-8 -*-
"""v3 —— 混合检索、RRF 融合与低召回查询扩展

把 v0-v2 的单路向量检索升级为「向量 + 关键词（BM25）」两路并行：
  两路并行检索（对照 search.go:97-120 的 goroutine）
    → 各自归一化：向量分按引擎归一到 [0,1]，BM25 分原样直通
      （对照 normalizer.go:15-18,111-158——RRF 只看排名，量纲无所谓）
    → 两路都有结果才 RRF 融合，否则退化为单路按分去重
      （对照 knowledgebase_search_fusion.go:33-50,84-142）
    → 融合后结果数低于阈值时，本地启发式生成查询变体并发重检索
      （对照 search.go:127-132 触发 + query_expansion.go 全文）

用法：
    python v3_hybrid_fusion.py --query "E-4703 是什么原因？" --mock
    python v3_hybrid_fusion.py --eval --mock
"""
import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = os.path.dirname(os.path.abspath(__file__))
MOCK_DIM = 256

# ── 算法参数（直接照抄 WeKnora，来源见注释） ──────────────────────────────
RRF_K = 60                   # retrieval_config.go:81-86
RRF_VECTOR_WEIGHT = 0.7      # retrieval_config.go:88-103
RRF_KEYWORD_WEIGHT = 0.3     # retrieval_config.go:88-103
EXPANSION_MAX_VARIANTS = 5   # query_expansion.go:163-165
EXPANSION_CONCURRENCY = 16   # query_expansion.go:36-43
EXPANSION_KW_DISCOUNT = 0.8  # query_expansion.go:29（扩展时关键词阈值打 8 折，向量阈值不变 :55-58）

# 课程语料只有 23 个叶子块，召回深度与阈值按比例缩小（生产值来自租户配置
# retrieval_config.go / config.yaml：embedding_top_k 默认 30 或 50）
EMBEDDING_TOP_K = 10         # 每路召回深度，也是低召回扩展的触发阈值（search.go:127）
VECTOR_THRESHOLD = 0.55      # 作用在归一化后的 [0,1] 向量分上
KEYWORD_THRESHOLD = 5.0      # 作用在原始 BM25 分上（本语料实测：真命中 6-19，噪声 2-5）


# ══════════════════ 与 v0-v2 相同的基础设施 ══════════════════

def load_chunks():
    """【代码算法】读入预切好的语料块。只索引叶子块，父块留给 v5 还原。"""
    with open(os.path.join(ROOT, "corpus", "chunks.json"), encoding="utf-8") as f:
        return [c for c in json.load(f) if c["chunk_type"] != "parent_text"]


def normalize(mat):
    """【代码算法】行向量归一化为单位长度，此后「点积」就是「余弦相似度」。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def mock_embed(texts):
    """【代码算法】Mock embedding：字符 2/3-gram 特征哈希，确定性、无需 API Key。"""
    mat = np.zeros((len(texts), MOCK_DIM))
    for i, text in enumerate(texts):
        grams = [text[j:j + n] for n in (2, 3) for j in range(len(text) - n + 1)]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            mat[i, h % MOCK_DIM] += 1.0 if (h >> 16) % 2 else -1.0
    return normalize(mat)


def real_embed(texts):
    """【模型调用】真实 embedding（OpenAI 兼容接口）。"""
    from openai import OpenAI
    resp = OpenAI().embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small"), input=texts)
    return normalize(np.array([d.embedding for d in resp.data]))


def build_prompt(query, chunks):
    """【代码算法】把检索结果原样拼进 prompt（用用户原话，改写只喂检索）。"""
    context = "\n\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
    return ("请只根据以下资料回答问题；资料不足以回答时，明确说不知道。\n\n"
            f"===== 资料 =====\n{context}\n\n===== 问题 =====\n{query}\n")


def generate(prompt, mock):
    """【模型调用】最终生成。--mock 返回占位回答。"""
    if mock:
        return "[mock 回答] 已跳过真实 LLM 调用；上面的 prompt 就是将喂给模型的最终内容。"
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("MODEL_ID", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


# ══════════════════ 与 v1/v2 相同的洋葱引擎 ══════════════════

@dataclass
class PluginError:
    description: str
    error_type: str
    err: Exception | None = None


ERR_SEARCH_NOTHING = PluginError("未检索到相关内容", "search_nothing")

QUERY_UNDERSTAND = "QUERY_UNDERSTAND"
CHUNK_SEARCH = "CHUNK_SEARCH"
CHAT_COMPLETION = "CHAT_COMPLETION"


@dataclass
class ChatState:
    """v3 新增 enable_query_expansion 开关（对照 ChatManage.EnableQueryExpansion）。
    search_result 从 v3 起保存**完整**的融合排序列表（不止 top-k）——真实系统
    也是到 FILTER_TOP_K 才截断，这个候选池是 v4 精排的输入。"""
    query: str
    k: int = 3
    mock: bool = True
    history: list = field(default_factory=list)
    enable_rewrite: bool = True
    enable_query_expansion: bool = True
    rewrite_query: str = ""
    intent: str = ""
    search_result: list = field(default_factory=list)
    answer: str = ""
    debug: dict = field(default_factory=dict)


class Plugin:
    def activation_events(self):
        raise NotImplementedError

    def on_event(self, event_type, state, next):
        raise NotImplementedError


class EventManager:
    """对照 chat_pipeline.go:24-78（详见 v1 文档）。"""

    def __init__(self):
        self.listeners = {}
        self.handlers = {}

    def register(self, plugin):
        for event_type in plugin.activation_events():
            self.listeners.setdefault(event_type, []).append(plugin)
            self.handlers[event_type] = self._build_handler(self.listeners[event_type])

    def _build_handler(self, plugins):
        def terminal(event_type, state):
            return None

        nxt = terminal
        for plugin in reversed(plugins):
            def wrapped(event_type, state, plugin=plugin, prev_next=nxt):
                return plugin.on_event(event_type, state,
                                       lambda: prev_next(event_type, state))
            nxt = wrapped
        return nxt

    def trigger(self, event_type, state):
        handler = self.handlers.get(event_type)
        return handler(event_type, state) if handler else None


# ══════════════════ 与 v2 相同的查询理解 ══════════════════

def needs_retrieval(intent):
    """对照 chat_manage.go:102-109。"""
    return intent in ("kb_search", "clarification", "summarize", "")


def format_conversation_history(history):
    """对照 query_understand.go:444-459。"""
    blocks, pending = [], None
    for msg in history:
        if msg["role"] == "user":
            pending = msg["content"]
        elif msg["role"] == "assistant" and pending is not None:
            blocks.append("------BEGIN------\n"
                          f"User question: {pending}\n"
                          f"Assistant answer: {msg['content']}\n"
                          "------END------\n")
            pending = None
    return "".join(blocks)


UNDERSTAND_SYSTEM = """你要对用户问题完成两件事，只输出一个 JSON 对象：

1. 改写问题：结合对话历史做指代消解与省略补全；保留原意和关键实体词；改写结果
   必须仍是一个问题，并包含可直接用于检索的关键词（不要输出「请在知识库中
   查找…」这类元指令）。无需改写时原样返回。
2. 意图分类，从中选一：kb_search（要查知识库，拿不准就选它）/ greeting（纯问候
   感谢告别）/ chitchat（闲聊，无需检索）/ follow_up（仅凭对话历史即可回答，
   无需新检索）。

输出格式（禁止 markdown、禁止解释）：
{"rewrite_query":"...","intent":"..."}

示例：
输入："你好" → {"rewrite_query":"你好","intent":"greeting"}
输入："它和传统搜索有什么区别"（历史在聊 RAG）
  → {"rewrite_query":"RAG 和传统搜索有什么区别","intent":"kb_search"}
输入："那超出之后呢？"（历史在聊免费版消息配额）
  → {"rewrite_query":"免费版消息配额超出之后会怎么样？","intent":"kb_search"}

## 对话历史
{{conversation}}"""

UNDERSTAND_USER = "## 用户问题\n{{query}}\n\n## JSON 输出\n"


def render_placeholders(template, values):
    """对照 types.RenderPromptPlaceholders：{{name}} 逐个替换（勿用 str.format）。"""
    for name, value in values.items():
        template = template.replace("{{" + name + "}}", value)
    return template


def build_understand_messages(query, history):
    conversation = format_conversation_history(history)
    return [
        {"role": "system",
         "content": render_placeholders(UNDERSTAND_SYSTEM, {"conversation": conversation})},
        {"role": "user", "content": render_placeholders(UNDERSTAND_USER, {"query": query})},
    ]


def parse_structured_output(raw):
    """对照 query_understand.go:321-407 的三层容错（详见 v2 文档）。"""
    content = raw.strip()
    if not content:
        return None
    parsed = _try_parse_json(content)
    if parsed is not None:
        return parsed
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    return _try_parse_json(content[start:end + 1])


def _try_parse_json(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    rewrite = ""
    for key in ("rewrite_query", "rewritten_query", "query", "question"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            rewrite = value.strip()
            break
    intent = obj.get("intent")
    return {"rewrite_query": rewrite,
            "intent": intent.strip() if isinstance(intent, str) else ""}


GREETING_WORDS = {"你好", "您好", "谢谢", "再见", "hi", "hello"}
CHITCHAT_WORDS = {"你是谁", "讲个笑话"}


def mock_understand_llm(messages):
    """确定性伪 LLM（详见 v2 文档）。"""
    system, user = messages[0]["content"], messages[-1]["content"]
    query = user.split("## 用户问题\n", 1)[1].split("\n\n## JSON 输出", 1)[0].strip()
    plain = query.strip("！!。.？?，, ")
    if plain in GREETING_WORDS:
        return json.dumps({"rewrite_query": query, "intent": "greeting"}, ensure_ascii=False)
    if plain in CHITCHAT_WORDS:
        return json.dumps({"rewrite_query": query, "intent": "chitchat"}, ensure_ascii=False)
    past = re.findall(r"User question: (.+)", system)
    if past and re.search(r"^(那|这|它|他|她)|呢[？?]?$", query):
        topic = past[-1].rstrip("？?")
        return json.dumps({"rewrite_query": f"{topic}，{query}", "intent": "kb_search"},
                          ensure_ascii=False)
    return json.dumps({"rewrite_query": query, "intent": "kb_search"}, ensure_ascii=False)


def real_understand_llm(messages):
    """【模型调用】temperature=0.3 / max_tokens=150 照抄 query_understand.go:121-122。"""
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("MODEL_ID", "gpt-4o-mini"), messages=messages,
        temperature=0.3, max_completion_tokens=150)
    return resp.choices[0].message.content


class PluginQueryUnderstand(Plugin):
    """对照 query_understand.go:58-161（详见 v2 文档）。"""

    def __init__(self, llm):
        self.llm = llm

    def activation_events(self):
        return [QUERY_UNDERSTAND]

    def on_event(self, event_type, state, next):
        state.rewrite_query = state.query
        if not state.enable_rewrite:
            return next()
        try:
            raw = self.llm(build_understand_messages(state.query, state.history))
        except Exception as exc:
            state.debug["understand_error"] = repr(exc)
            return next()
        parsed = parse_structured_output(raw)
        if parsed is None:
            if raw.strip():
                state.rewrite_query = raw.strip()
            return next()
        if parsed["rewrite_query"]:
            state.rewrite_query = parsed["rewrite_query"]
        state.intent = parsed["intent"]
        return next()


# ══════════════════ v3 新增：分词、归一化、两路检索 ══════════════════

def tokenize(text):
    """【代码算法】对照 query_expansion.go:232-259：汉字逐字成 token，
    ASCII 字母/数字连续段成 token，其余是分隔符。"E-4703" → ["E","4703"]。
    这是课程的极简中文分析器；生产引擎用的是 ES ik / PG 分词等真正的分析器。"""
    tokens, current = [], []
    for ch in text:
        if "一" <= ch <= "鿿":  # unicode.Is(unicode.Han, r)
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
        elif ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def clamp01(s):
    """【代码算法】对照 normalizer.go:163-174：NaN/Inf 也要安全落进 [0,1]，
    否则下游排序的严格弱序会被 NaN 破坏。"""
    if s != s:  # NaN
        return 0.0
    if s <= 0:
        return 0.0
    if s >= 1:
        return 1.0
    return s


def normalize_score(score, retriever_type, engine_type="memory"):
    """【代码算法】对照 normalizer.go:111-158：只归一化向量分。
    - 关键词（BM25）分原样直通（:117-121）——它的量纲无上界，硬压到 [0,1]
      会压塌长尾；反正 RRF 只看排名不看分数。
    - 本课程的内存引擎暴露的是原始余弦 ∈ [-1,1]，与 Milvus 同组：(s+1)/2
      再 clamp（:124-130）。其余生产引擎到手已是 [0,1]，只做 clamp（:131-153）。"""
    if retriever_type != "vector":
        return score
    if engine_type in ("memory", "milvus"):
        return clamp01((score + 1) / 2)
    return clamp01(score)


def vector_search(query_vec, chunk_mat, chunks, top_n, threshold):
    """【代码算法】向量一路：余弦 → 归一化 → 阈值过滤 → 取前 top_n。"""
    raw_scores = chunk_mat @ query_vec
    scored = []
    for i in np.argsort(-raw_scores, kind="stable")[:len(chunks)]:
        s = normalize_score(float(raw_scores[i]), "vector", "memory")
        if s > threshold:
            scored.append({**chunks[int(i)], "score": s, "retriever": "vector"})
        if len(scored) >= top_n:
            break
    return scored


def keyword_search(bm25, chunks, query, top_n, threshold):
    """【代码算法】关键词一路：BM25 原始分（不归一化），阈值过滤，取前 top_n。"""
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores, kind="stable")
    scored = []
    for i in order:
        s = float(scores[i])
        if s <= threshold:
            break  # 已按分数降序，后面只会更小
        scored.append({**chunks[int(i)], "score": s, "retriever": "keyword"})
        if len(scored) >= top_n:
            break
    return scored


# ══════════════════ v3 新增：RRF 融合与单路退化 ══════════════════

def deduplicate_by_score(results):
    """【代码算法】对照 knowledgebase_search_fusion.go:65-78：单路时按 chunk id
    去重、同 id 保留最高分，按分数降序返回（保留原始分数对 FAQ 场景很重要）。"""
    best = {}
    for r in results:
        if r["id"] not in best or r["score"] > best[r["id"]]["score"]:
            best[r["id"]] = r
    return sorted(best.values(), key=lambda r: -r["score"])


def fuse_with_rrf(vector_results, keyword_results,
                  k=RRF_K, vector_weight=RRF_VECTOR_WEIGHT, keyword_weight=RRF_KEYWORD_WEIGHT):
    """【代码算法】对照 knowledgebase_search_fusion.go:84-142：
    score = vecWeight/(k+vecRank) + kwWeight/(k+kwRank)，排名 1-indexed、
    同 id 取首次出现的名次；元数据优先取向量路的（分高者）；只在一路出现的
    chunk 只拿那一路的贡献。融合后按 RRF 分降序。"""
    vector_ranks = {}
    for i, r in enumerate(vector_results):
        vector_ranks.setdefault(r["id"], i + 1)  # :89-94 首次出现的名次
    keyword_ranks = {}
    for i, r in enumerate(keyword_results):
        keyword_ranks.setdefault(r["id"], i + 1)

    info = {}
    for r in vector_results:  # :103-108 向量路元数据优先、同 id 保留分高者
        if r["id"] not in info or r["score"] > info[r["id"]]["score"]:
            info[r["id"]] = r
    for r in keyword_results:  # :109-113 只补充向量路没有的
        info.setdefault(r["id"], r)

    fused = []
    for chunk_id, r in info.items():  # :116-127
        rrf = 0.0
        if chunk_id in vector_ranks:
            rrf += vector_weight / (k + vector_ranks[chunk_id])
        if chunk_id in keyword_ranks:
            rrf += keyword_weight / (k + keyword_ranks[chunk_id])
        fused.append({**r, "score": rrf, "retriever": "rrf"})
    return sorted(fused, key=lambda r: -r["score"])


def fuse_or_deduplicate(vector_results, keyword_results):
    """【代码算法】对照 knowledgebase_search_fusion.go:33-50：
    只有两路**都有**结果才融合；任何一路为空就退化为另一路的按分去重——
    对空列表做 RRF 毫无意义，还会把仅剩一路的原始分数抹掉。"""
    if not keyword_results:
        return deduplicate_by_score(vector_results)
    if not vector_results:
        return deduplicate_by_score(keyword_results)
    return fuse_with_rrf(vector_results, keyword_results)


# ══════════════════ v3 新增：低召回查询扩展 ══════════════════

# 对照 query_expansion.go:174-184（节选中文部分 + 常用英文虚词）
STOPWORDS = {"的", "是", "在", "了", "和", "与", "或",
             "a", "an", "the", "is", "are", "to", "of", "in", "for", "on",
             "what", "how", "why", "when", "where", "which", "who"}

# 对照 query_expansion.go:187 的疑问词前缀正则（逐字照抄）
QUESTION_WORDS = re.compile(
    r"^(什么是|什么|如何|怎么|怎样|为什么|为何|哪个|哪些|谁|何时|何地|请问|请告诉我|帮我|我想知道|我想了解)")


def _blen(s):
    """Go 的 len() 是字节长度——变体的长短判断必须按 UTF-8 字节数才忠实。"""
    return len(s.encode("utf-8"))


def expand_queries(rewrite_query, original_query):
    """【代码算法】对照 expandQueries（query_expansion.go:110-171）：
    零成本生成至多 5 个查询变体，全程无 LLM。四种手法：
    ① 去停用词后的关键词串（:137-140） ② 引号短语提取（:143-146）
    ③ 分隔符切分取长段（:149-154，len>5 字节≈中文 2 字以上）
    ④ 去疑问词前缀（:156-160）。与原查询/改写查询重复的变体丢弃。"""
    query = rewrite_query.strip()
    if not query:
        return []
    expansions, seen = [], {query.lower(), original_query.lower()}

    def add_if_new(s):
        s = s.strip()
        if not s or _blen(s) < 3:  # :125 len(s)<3（字节）
            return
        if s.lower() in seen:
            return
        seen.add(s.lower())
        expansions.append(s)

    keywords = [w for w in tokenize(query)
                if w.lower() not in STOPWORDS and _blen(w) > 1]  # :189-199
    if len(keywords) >= 2:
        add_if_new(" ".join(keywords))

    for m in re.finditer(r'["\'"「」『』]([^"\'"「」『』]+)["\'"「」『』]', query):  # :201-212
        if _blen(m.group(1)) > 2:
            add_if_new(m.group(1))

    for seg in re.split(r"[,，;；、。！？!?\s]+", query):  # :214-226
        if seg and _blen(seg) > 5:
            add_if_new(seg)

    cleaned = QUESTION_WORDS.sub("", query).strip()  # :228-230
    if cleaned != query:
        add_if_new(cleaned)

    return expansions[:EXPANSION_MAX_VARIANTS]  # :163-165


# ══════════════════ 插件：混合检索（v3 重写） ══════════════════

class PluginSearch(Plugin):
    """两路并行 → 融合 → 低召回扩展。结构对照 search.go 的 OnEvent 主干：
    并行检索（:97-120）→ 触发扩展（:127-132）→ 空结果报 ErrSearchNothing（:148）。"""

    def __init__(self, chunks, chunk_mat, bm25):
        self.chunks = chunks
        self.chunk_mat = chunk_mat
        self.bm25 = bm25

    def activation_events(self):
        return [CHUNK_SEARCH]

    def hybrid_search(self, query, state, kw_threshold=KEYWORD_THRESHOLD):
        """一次混合检索：两路并行（对照 search.go:97-120 的 goroutine 对）→
        各自归一化 → 融合或退化去重。"""
        embed = mock_embed if state.mock else real_embed
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_vec = pool.submit(lambda: vector_search(
                embed([query])[0], self.chunk_mat, self.chunks,
                EMBEDDING_TOP_K, VECTOR_THRESHOLD))
            f_kw = pool.submit(lambda: keyword_search(
                self.bm25, self.chunks, query, EMBEDDING_TOP_K, kw_threshold))
            return fuse_or_deduplicate(f_vec.result(), f_kw.result())

    def on_event(self, event_type, state, next):
        if not needs_retrieval(state.intent):
            return next()
        query = state.rewrite_query or state.query
        results = self.hybrid_search(query, state)

        # 低召回 → 查询扩展（对照 search.go:127-132 的触发条件）
        if state.enable_query_expansion and len(results) < max(1, EMBEDDING_TOP_K):
            variants = expand_queries(query, state.query)
            if variants:
                added = self._expansion_search(variants, state)
                seen = {r["id"] for r in results}
                extra = [r for r in added if r["id"] not in seen and not seen.add(r["id"])]
                # 课程简化：扩展命中只追加在尾部——主结果是 RRF 分、扩展结果
                # 可能是另一次融合甚至单路的原始分，两种量纲不可比。真实系统
                # 把它们一起丢给 v4 的精排统一重打分，排序问题在那里解决
                # （rerank.go 会对全体候选重新打分）。
                results = results + extra
                state.debug["expansion"] = {"variants": variants, "added": len(extra)}

        state.search_result = results
        if not state.search_result:
            return ERR_SEARCH_NOTHING
        return next()

    def _expansion_search(self, variants, state):
        """对每个变体并发重检索（对照 query_expansion.go:31-98：并发上限 16、
        关键词阈值 ×0.8、向量阈值不变）。"""
        kw_th = KEYWORD_THRESHOLD * EXPANSION_KW_DISCOUNT
        added = []
        with ThreadPoolExecutor(max_workers=min(EXPANSION_CONCURRENCY, len(variants))) as pool:
            for res in pool.map(lambda v: self.hybrid_search(v, state, kw_threshold=kw_th),
                                variants):
                added.extend(res)
        return added


class PluginLoggingBoost(Plugin):
    """与 v1/v2 相同的后置装饰器（对照 wiki_boost.go:42-97）。"""

    def activation_events(self):
        return [CHUNK_SEARCH]

    def on_event(self, event_type, state, next):
        err = next()
        if err is not None:
            return err
        if state.search_result:
            state.debug["search_trace"] = {
                "candidates": len(state.search_result),
                "top5_ids": [r["id"] for r in state.search_result[:5]],
                "decorated_by": "PluginLoggingBoost",
            }
        return None


class PluginGenerate(Plugin):
    """【模型调用】prompt 只取候选池前 k 条（截断逻辑在 v5 会独立成
    FILTER_TOP_K 事件，此处先内联）。"""

    def activation_events(self):
        return [CHAT_COMPLETION]

    def on_event(self, event_type, state, next):
        if state.search_result:
            prompt = build_prompt(state.query, state.search_result[:state.k])
        else:
            prompt = state.query
        state.answer = generate(prompt, state.mock)
        return next()


# ══════════════════ 驱动与入口 ══════════════════

PIPELINE_EVENTS = [QUERY_UNDERSTAND, CHUNK_SEARCH, CHAT_COMPLETION]


def build_manager(chunks, chunk_mat, mock):
    bm25 = BM25Okapi([tokenize(c["content"]) for c in chunks])
    manager = EventManager()
    manager.register(PluginQueryUnderstand(mock_understand_llm if mock else real_understand_llm))
    manager.register(PluginSearch(chunks, chunk_mat, bm25))
    manager.register(PluginLoggingBoost())
    manager.register(PluginGenerate())
    return manager


def run_pipeline(manager, state):
    for event_type in PIPELINE_EVENTS:
        err = manager.trigger(event_type, state)
        if err is not None:
            state.answer = f"[pipeline 中止] {err.description}（{err.error_type}）"
            return err
    return None


def run_eval(manager, k, mock):
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    passed = 0
    for q in queries:
        state = ChatState(query=q["query"], k=k, mock=mock, history=q.get("history", []))
        manager.trigger(QUERY_UNDERSTAND, state)
        manager.trigger(CHUNK_SEARCH, state)
        got = [r["id"] for r in state.search_result[:k]]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        passed += ok
        notes = []
        if state.rewrite_query != q["query"]:
            notes.append(f"改写：{state.rewrite_query}")
        if "expansion" in state.debug:
            notes.append(f"扩展×{len(state.debug['expansion']['variants'])}")
        suffix = f"（{'；'.join(notes)}）" if notes else ""
        print(f"{'PASS' if ok else 'MISS'}  {q['id']:<3s} [{q['scenario']}] {q['query']}{suffix}")
        print(f"      期望 {q['expect_chunk_ids']} | 实际 top-{k} {got}")
    print(f"\n检索命中：{passed}/{len(queries)}（对比 v2 的 6/9）")


def main():
    ap = argparse.ArgumentParser(description="v3 混合检索、RRF 融合与查询扩展")
    ap.add_argument("--query", help="单条提问")
    ap.add_argument("--eval", action="store_true", help="跑 eval/queries.json 全部问题")
    ap.add_argument("--k", type=int, default=3, help="top-k（默认 3）")
    ap.add_argument("--mock", action="store_true", help="用确定性假 embedding 与伪 LLM")
    args = ap.parse_args()
    if not args.mock:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))

    chunks = load_chunks()
    embed = mock_embed if args.mock else real_embed
    manager = build_manager(chunks, embed([c["content"] for c in chunks]), args.mock)

    if args.eval:
        run_eval(manager, args.k, args.mock)
    elif args.query:
        state = ChatState(query=args.query, k=args.k, mock=args.mock)
        run_pipeline(manager, state)
        print(f"改写结果   {state.rewrite_query!r}  意图 {state.intent or 'kb_search(默认)'}")
        for r in state.search_result[:args.k]:
            print(f"检索命中 {r['id']:<12s} score={r['score']:.6f} ({r['retriever']})")
        if "expansion" in state.debug:
            print(f"查询扩展   {state.debug['expansion']}")
        print("\n===== 回答 =====\n" + state.answer)
    else:
        ap.error("需要 --query 或 --eval")


if __name__ == "__main__":
    main()
