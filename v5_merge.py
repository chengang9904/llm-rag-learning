# -*- coding: utf-8 -*-
"""v5 —— 上下文合并四术（父块还原 / 重叠拼接 / FAQ 格式化 / 短块邻居扩展）

精排选完的是「碎片」，合并阶段把碎片重新变回人能读的上下文。新增两个事件：

CHUNK_MERGE（对照 merge.go:44-96 的八步流水）：
  选输入(rerank优先) → 去重(ID+内容签名) → 注入历史相关片段(Jaccard≥0.15,
  ×0.6, 最多3条) → 父块还原(子块→完整 parent_text) → 分组重叠拼接(同
  KnowledgeID+ChunkType 按 StartAt 排序) → FAQ 格式化 → 短块邻居扩展
  (<350 扩到 850 字符) → 扩展后再拼接一轮 → 终去重+部分重叠剔除

FILTER_TOP_K（对照 filter_top_k.go）：确定性排序后截断——整条链最朴素的一步。

三处手术都在维护 sub_chunk_ids（对照 SubChunkID）：记录被并入的成员块。
评测判定从本版起升级为「期望块 id ∈ 结果自身 id ∪ sub_chunk_ids」——
内容进了 prompt 才算数，这正是合并阶段存在的意义。

用法：
    python v5_merge.py --query "Python SDK 的 publish 怎么用？" --mock
    python v5_merge.py --eval --mock
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

# ── v3/v4 参数（出处见各版文档） ───────────────────────────────────────
RRF_K = 60
RRF_VECTOR_WEIGHT = 0.7
RRF_KEYWORD_WEIGHT = 0.3
EXPANSION_MAX_VARIANTS = 5
EXPANSION_CONCURRENCY = 16
EXPANSION_KW_DISCOUNT = 0.8
EMBEDDING_TOP_K = 10
VECTOR_THRESHOLD = 0.55
KEYWORD_THRESHOLD = 5.0
RERANK_THRESHOLD = 0.2
RERANK_TOP_K = 5
MMR_LAMBDA = 0.7
DEGRADE_FACTOR = 0.7
DEGRADE_FLOOR = 0.3
FALLBACK_MIN_SCORE = 0.15
COMPOSITE_MODEL_W = 0.6
COMPOSITE_BASE_W = 0.3
COMPOSITE_SOURCE_W = 0.1

# ── v5 参数（直接照抄 WeKnora） ────────────────────────────────────────
EXPAND_MIN_LEN = 350           # merge_expand.go:16（rune 数，非字节！）
EXPAND_MAX_LEN = 850           # merge_expand.go:17
HISTORY_MIN_SIMILARITY = 0.15  # merge_history.go:25
HISTORY_SCORE_DISCOUNT = 0.6   # merge_history.go:28
HISTORY_MAX_RESULTS = 3        # merge_history.go:31
HISTORY_REF_BASE_SCORE = 0.5   # 课程值：真实系统的历史引用携带上一轮的原始检索分
MIN_OVERLAP_RUNES = 12         # searchutil/chunkmerge.go:27（短于表格分隔行的后缀不参与匹配）
DEFAULT_SEARCH_SPAN = 400      # searchutil/chunkmerge.go:30
PARTIAL_OVERLAP_THRESHOLD = 0.85  # search.go:213


# ══════════════════ 与 v0-v4 相同的基础设施 ══════════════════

def load_chunks():
    """检索索引只放叶子块；父块经 load_all_chunks 进合并阶段的 chunk_map。"""
    with open(os.path.join(ROOT, "corpus", "chunks.json"), encoding="utf-8") as f:
        return [c for c in json.load(f) if c["chunk_type"] != "parent_text"]


def load_all_chunks():
    """合并阶段的「回表」数据源（对照 chunkRepo.ListChunksByID）：含父块。"""
    with open(os.path.join(ROOT, "corpus", "chunks.json"), encoding="utf-8") as f:
        return {c["id"]: c for c in json.load(f)}


def normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def mock_embed(texts):
    mat = np.zeros((len(texts), MOCK_DIM))
    for i, text in enumerate(texts):
        grams = [text[j:j + n] for n in (2, 3) for j in range(len(text) - n + 1)]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            mat[i, h % MOCK_DIM] += 1.0 if (h >> 16) % 2 else -1.0
    return normalize(mat)


def real_embed(texts):
    from openai import OpenAI
    resp = OpenAI().embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small"), input=texts)
    return normalize(np.array([d.embedding for d in resp.data]))


def build_prompt(query, chunks):
    context = "\n\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
    return ("请只根据以下资料回答问题；资料不足以回答时，明确说不知道。\n\n"
            f"===== 资料 =====\n{context}\n\n===== 问题 =====\n{query}\n")


def generate(prompt, mock):
    if mock:
        return "[mock 回答] 已跳过真实 LLM 调用；上面的 prompt 就是将喂给模型的最终内容。"
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("MODEL_ID", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


# ══════════════════ 与 v1-v4 相同的洋葱引擎 ══════════════════

@dataclass
class PluginError:
    description: str
    error_type: str
    err: Exception | None = None


ERR_SEARCH_NOTHING = PluginError("未检索到相关内容", "search_nothing")

QUERY_UNDERSTAND = "QUERY_UNDERSTAND"
CHUNK_SEARCH = "CHUNK_SEARCH"
CHUNK_RERANK = "CHUNK_RERANK"
CHUNK_MERGE = "CHUNK_MERGE"      # v5 新事件
FILTER_TOP_K = "FILTER_TOP_K"    # v5 新事件


CHAT_COMPLETION = "CHAT_COMPLETION"


@dataclass
class ChatState:
    """v5 新增 merge_result（对照 ChatManage.MergeResult）。
    下游取数：merge → rerank → search 逐级回退。"""
    query: str
    k: int = 3
    mock: bool = True
    history: list = field(default_factory=list)
    enable_rewrite: bool = True
    enable_query_expansion: bool = True
    rewrite_query: str = ""
    intent: str = ""
    search_result: list = field(default_factory=list)
    rerank_result: list = field(default_factory=list)
    merge_result: list = field(default_factory=list)
    answer: str = ""
    debug: dict = field(default_factory=dict)


class Plugin:
    def activation_events(self):
        raise NotImplementedError

    def on_event(self, event_type, state, next):
        raise NotImplementedError


class EventManager:
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


# ══════════════════ 与 v2 相同的查询理解（正文见 v2 文档） ══════════════════

def needs_retrieval(intent):
    return intent in ("kb_search", "clarification", "summarize", "")


def format_conversation_history(history):
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
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("MODEL_ID", "gpt-4o-mini"), messages=messages,
        temperature=0.3, max_completion_tokens=150)
    return resp.choices[0].message.content


class PluginQueryUnderstand(Plugin):
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


# ══════════════════ 与 v3 相同的混合检索（正文见 v3 文档） ══════════════════

def tokenize(text):
    tokens, current = [], []
    for ch in text:
        if "一" <= ch <= "鿿":
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
    if s != s:
        return 0.0
    if s <= 0:
        return 0.0
    if s >= 1:
        return 1.0
    return s


def normalize_score(score, retriever_type, engine_type="memory"):
    if retriever_type != "vector":
        return score
    if engine_type in ("memory", "milvus"):
        return clamp01((score + 1) / 2)
    return clamp01(score)


def vector_search(query_vec, chunk_mat, chunks, top_n, threshold):
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
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores, kind="stable")
    scored = []
    for i in order:
        s = float(scores[i])
        if s <= threshold:
            break
        scored.append({**chunks[int(i)], "score": s, "retriever": "keyword"})
        if len(scored) >= top_n:
            break
    return scored


def deduplicate_by_score(results):
    best = {}
    for r in results:
        if r["id"] not in best or r["score"] > best[r["id"]]["score"]:
            best[r["id"]] = r
    return sorted(best.values(), key=lambda r: -r["score"])


def fuse_with_rrf(vector_results, keyword_results,
                  k=RRF_K, vector_weight=RRF_VECTOR_WEIGHT, keyword_weight=RRF_KEYWORD_WEIGHT):
    vector_ranks = {}
    for i, r in enumerate(vector_results):
        vector_ranks.setdefault(r["id"], i + 1)
    keyword_ranks = {}
    for i, r in enumerate(keyword_results):
        keyword_ranks.setdefault(r["id"], i + 1)

    info = {}
    for r in vector_results:
        if r["id"] not in info or r["score"] > info[r["id"]]["score"]:
            info[r["id"]] = r
    for r in keyword_results:
        info.setdefault(r["id"], r)

    fused = []
    for chunk_id, r in info.items():
        rrf = 0.0
        if chunk_id in vector_ranks:
            rrf += vector_weight / (k + vector_ranks[chunk_id])
        if chunk_id in keyword_ranks:
            rrf += keyword_weight / (k + keyword_ranks[chunk_id])
        fused.append({**r, "score": rrf, "retriever": "rrf"})
    return sorted(fused, key=lambda r: -r["score"])


def fuse_or_deduplicate(vector_results, keyword_results):
    if not keyword_results:
        return deduplicate_by_score(vector_results)
    if not vector_results:
        return deduplicate_by_score(keyword_results)
    return fuse_with_rrf(vector_results, keyword_results)


STOPWORDS = {"的", "是", "在", "了", "和", "与", "或",
             "a", "an", "the", "is", "are", "to", "of", "in", "for", "on",
             "what", "how", "why", "when", "where", "which", "who"}

QUESTION_WORDS = re.compile(
    r"^(什么是|什么|如何|怎么|怎样|为什么|为何|哪个|哪些|谁|何时|何地|请问|请告诉我|帮我|我想知道|我想了解)")


def _blen(s):
    return len(s.encode("utf-8"))


def expand_queries(rewrite_query, original_query):
    query = rewrite_query.strip()
    if not query:
        return []
    expansions, seen = [], {query.lower(), original_query.lower()}

    def add_if_new(s):
        s = s.strip()
        if not s or _blen(s) < 3:
            return
        if s.lower() in seen:
            return
        seen.add(s.lower())
        expansions.append(s)

    keywords = [w for w in tokenize(query)
                if w.lower() not in STOPWORDS and _blen(w) > 1]
    if len(keywords) >= 2:
        add_if_new(" ".join(keywords))

    for m in re.finditer(r'["\'"「」『』]([^"\'"「」『』]+)["\'"「」『』]', query):
        if _blen(m.group(1)) > 2:
            add_if_new(m.group(1))

    for seg in re.split(r"[,，;；、。！？!?\s]+", query):
        if seg and _blen(seg) > 5:
            add_if_new(seg)

    cleaned = QUESTION_WORDS.sub("", query).strip()
    if cleaned != query:
        add_if_new(cleaned)

    return expansions[:EXPANSION_MAX_VARIANTS]


class PluginSearch(Plugin):
    def __init__(self, chunks, chunk_mat, bm25):
        self.chunks = chunks
        self.chunk_mat = chunk_mat
        self.bm25 = bm25

    def activation_events(self):
        return [CHUNK_SEARCH]

    def hybrid_search(self, query, state, kw_threshold=KEYWORD_THRESHOLD):
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

        if state.enable_query_expansion and len(results) < max(1, EMBEDDING_TOP_K):
            variants = expand_queries(query, state.query)
            if variants:
                added = self._expansion_search(variants, state)
                seen = {r["id"] for r in results}
                extra = [r for r in added if r["id"] not in seen and not seen.add(r["id"])]
                results = results + extra
                state.debug["expansion"] = {"variants": variants, "added": len(extra)}

        state.search_result = results
        if not state.search_result:
            return ERR_SEARCH_NOTHING
        return next()

    def _expansion_search(self, variants, state):
        kw_th = KEYWORD_THRESHOLD * EXPANSION_KW_DISCOUNT
        added = []
        with ThreadPoolExecutor(max_workers=min(EXPANSION_CONCURRENCY, len(variants))) as pool:
            for res in pool.map(lambda v: self.hybrid_search(v, state, kw_threshold=kw_th),
                                variants):
                added.extend(res)
        return added


# ══════════════════ 与 v4 相同的精排（正文见 v4 文档） ══════════════════

def tokenize_simple(text):
    text = text.lower().strip()
    tokens = set()
    for run in re.findall(r"[一-鿿]+|[a-z0-9]+", text):
        if run[0].isascii():
            if len(run) > 1:
                tokens.add(run)
        else:
            for j in range(len(run) - 1):
                tokens.add(run[j:j + 2])
    return tokens


def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def mock_rerank_model(query, passages):
    q_tokens = tokenize_simple(query)
    results = []
    for i, p in enumerate(passages):
        p_tokens = tokenize_simple(p)
        score = len(q_tokens & p_tokens) / max(1, len(q_tokens))
        results.append((i, score))
    return sorted(results, key=lambda r: (-r[1], r[0]))


def real_rerank_model(query, passages):
    from openai import OpenAI
    resp = OpenAI().post("/rerank", cast_to=dict, body={
        "model": os.environ["RERANK_MODEL_ID"], "query": query,
        "documents": passages})
    rows = sorted(resp["results"], key=lambda r: -r["relevance_score"])
    return [(r["index"], r["relevance_score"]) for r in rows]


def composite_score(result, model_score, base_score):
    source_weight = 0.95 if result.get("knowledge_source", "").lower() == "web_search" else 1.0
    position_prior = 1.0
    if result.get("start_at", -1) >= 0:
        raw = 1.0 - result["start_at"] / (result["end_at"] + 1)
        position_prior += min(0.05, max(-0.05, raw))
    composite = (COMPOSITE_MODEL_W * model_score
                 + COMPOSITE_BASE_W * clamp01(base_score)
                 + COMPOSITE_SOURCE_W * source_weight)
    composite *= position_prior
    return min(1.0, max(0.0, composite))


def apply_mmr(results, k, lam=MMR_LAMBDA):
    if k <= 0 or not results:
        return []
    token_sets = [tokenize_simple(r["content"]) for r in results]
    selected, selected_sets, selected_idx = [], [], set()
    while len(selected) < k and len(selected_idx) < len(results):
        best_idx, best_score = -1, -1.0
        for i, r in enumerate(results):
            if i in selected_idx:
                continue
            redundancy = max((jaccard(token_sets[i], s) for s in selected_sets), default=0.0)
            mmr = lam * r["score"] - (1.0 - lam) * redundancy
            if mmr > best_score:
                best_score, best_idx = mmr, i
        if best_idx < 0:
            break
        selected.append(results[best_idx])
        selected_sets.append(token_sets[best_idx])
        selected_idx.add(best_idx)
    return selected


class PluginRerank(Plugin):
    def __init__(self, model, threshold=RERANK_THRESHOLD, top_k=RERANK_TOP_K):
        self.model = model
        self.threshold = threshold
        self.top_k = top_k

    def activation_events(self):
        return [CHUNK_RERANK]

    def on_event(self, event_type, state, next):
        if not needs_retrieval(state.intent):
            return next()
        if not state.search_result:
            return next()
        if self.model is None:
            state.debug["rerank_skip"] = "empty_model_id"
            return next()

        candidates = [r for r in state.search_result if r["content"].strip()]
        passages = [r["content"] for r in candidates]

        try:
            ranked = self._rank_with_threshold(state, passages, self.threshold)
        except Exception as exc:
            state.debug["rerank_api_error"] = repr(exc)
            return next()

        if not ranked and self.threshold > DEGRADE_FLOOR:
            degraded = max(DEGRADE_FLOOR, self.threshold * DEGRADE_FACTOR)
            state.debug["rerank_degraded"] = {"from": self.threshold, "to": degraded}
            try:
                ranked = self._rank_with_threshold(state, passages, degraded)
            except Exception as exc:
                state.debug["rerank_api_error"] = repr(exc)
                return next()

        reranked = []
        for idx, model_score in ranked:
            if idx >= len(candidates):
                continue
            r = dict(candidates[idx])
            r["base_score"] = r["score"]
            r["model_score"] = model_score
            r["score"] = composite_score(r, model_score, r["base_score"])
            reranked.append(r)

        state.rerank_result = apply_mmr(
            reranked, min(len(reranked), max(1, self.top_k)))

        if not state.rerank_result:
            return ERR_SEARCH_NOTHING
        return next()

    def _rank_with_threshold(self, state, passages, threshold):
        query = state.rewrite_query or state.query
        resp = self.model(query, passages)
        kept = [(i, s) for i, s in resp if s >= threshold]
        if not kept and resp and resp[0][1] >= FALLBACK_MIN_SCORE:
            kept = resp[:1]
            state.debug["rerank_fallback_top1"] = resp[0][1]
        return kept


class PluginLoggingBoost(Plugin):
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


# ══════════════════ v5 新增：合并的工具函数 ══════════════════

def build_content_signature(content):
    """【代码算法】对照 searchutil.BuildContentSignature（textutil.go:14-24）：
    小写、压空白后取 MD5——按「内容长相」而不是 id 去重。"""
    c = " ".join(content.lower().strip().split())
    return hashlib.md5(c.encode("utf-8")).hexdigest() if c else ""


def remove_duplicate_results(results):
    """【代码算法】对照 removeDuplicateResults（search.go:169-193）：
    先按 chunk id、再按内容签名去重（不同 id 相同内容也算重复）。
    刻意**不**把同 parent 视作重复——同父的不同子块内容各不相同。"""
    seen_ids, seen_sigs, out = set(), {}, []
    for r in results:
        if r["id"] in seen_ids:
            continue
        sig = build_content_signature(r["content"])
        if sig and sig in seen_sigs:
            continue
        if sig:
            seen_sigs[sig] = r["id"]
        seen_ids.add(r["id"])
        out.append(r)
    return out


def append_with_overlap(acc, nxt, position_overlap):
    """【代码算法】对照 searchutil.AppendWithOverlap（chunkmerge.go:38-68）：
    重叠去重按**文本匹配**而非位置裁剪——offset 可能因补写表头/HTML 实体而
    错位，StartAt/EndAt 只用来估算搜索窗口。从 acc 的最长后缀开始往短找，
    后缀短于 12 字符不再参与匹配（表格分隔行这类模式会导致误配）。"""
    if not acc:
        return nxt
    if not nxt:
        return acc
    span = max(0, position_overlap)
    max_k = min(len(acc), len(nxt), max(span * 3, DEFAULT_SEARCH_SPAN))
    head_slack = max(span * 2, 320)
    for k in range(max_k, MIN_OVERLAP_RUNES - 1, -1):
        needle = acc[-k:]
        pos = nxt.find(needle)
        if 0 <= pos <= head_slack:
            return acc + nxt[pos + k:]
    return acc + nxt


def merge_overlapping_chunks(chunks):
    """【代码算法】对照 mergeOverlappingChunks（merge_overlap.go:15-78）：
    输入必须已按 (StartAt, EndAt) 升序。三种关系：不相邻→并列；部分重叠→
    拼接非重叠后缀并吞并 id；完全包含→只记 id。合并块保留最高分，最后按分排序。"""
    if not chunks:
        return []
    merged = [chunks[0]]
    for cur in chunks[1:]:
        last = merged[-1]
        if cur["start_at"] > last["end_at"]:
            merged.append(cur)
            continue
        if cur["end_at"] > last["end_at"]:  # 部分重叠
            last["content"] = append_with_overlap(
                last["content"], cur["content"], last["end_at"] - cur["start_at"])
            last["end_at"] = cur["end_at"]
            last["sub_chunk_ids"].append(cur["id"])
        else:  # 完全包含
            if cur["id"] not in last["sub_chunk_ids"]:
                last["sub_chunk_ids"].append(cur["id"])
        if cur["score"] > last["score"]:
            last["score"] = cur["score"]
    return sorted(merged, key=lambda r: -r["score"])


def remove_partial_overlaps(results):
    """【代码算法】对照 removePartialOverlaps（search.go:199-260 附近）：
    内容被高分块「大体包含」的低分块剔除——两种判定：规范化子串包含、
    token 重叠系数 ≥0.85（以小块 token 数为分母）。课程附加：被剔除块的
    id 记入幸存者的 sub_chunk_ids（评测溯源用，真实系统直接丢弃）。"""
    if len(results) <= 1:
        return results
    norm = [" ".join(r["content"].lower().split()) for r in results]
    toks = [tokenize_simple(r["content"]) for r in results]
    removed = set()
    for i in range(len(results)):
        if i in removed:
            continue
        for j in range(i + 1, len(results)):
            if j in removed:
                continue
            small, large = (i, j) if len(norm[i]) <= len(norm[j]) else (j, i)
            contained = norm[small] != "" and norm[small] in norm[large]
            if not contained and toks[small]:
                ratio = len(toks[small] & toks[large]) / len(toks[small])
                contained = ratio >= PARTIAL_OVERLAP_THRESHOLD
            if not contained:
                continue
            # 低分者被移除；同分时短的让位（对照「longer wins」）
            drop = small if results[small]["score"] <= results[large]["score"] else large
            keep = large if drop == small else small
            removed.add(drop)
            if results[drop]["id"] not in results[keep]["sub_chunk_ids"]:
                results[keep]["sub_chunk_ids"].append(results[drop]["id"])
            if drop == i:
                break
    return [r for idx, r in enumerate(results) if idx not in removed]


def concat_no_overlap(a, b):
    """【代码算法】对照 concatNoOverlap（merge_expand.go:276-293）：
    去掉 a 的后缀与 b 的前缀的最长重合再拼接（任意长度，不设 12 字符下限）。"""
    if not a:
        return b
    if not b:
        return a
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return a + b[k:]
    return a + b


def merge_ordered_content(prev, base, nxt, max_len):
    """【代码算法】对照 mergeOrderedContent（merge_expand.go:260-273）。"""
    content = base
    if prev:
        content = concat_no_overlap(prev, content)
    if nxt:
        content = concat_no_overlap(content, nxt)
    return content[:max_len] if len(content) > max_len else content


def build_faq_content(content):
    """【代码算法】对照 buildFAQAnswerContent（merge_faq.go:100-134）的渲染
    格式。生产版从结构化 FAQMetadata（标准问 + 多答案）构建；课程语料的
    FAQ 块存的是「Q：…\\n\\nA：…」文本，先解析再按同一格式渲染。"""
    m = re.match(r"Q：(.+?)\n+A：(.+)", content, re.S)
    if not m:
        return content
    question, answer = m.group(1).strip(), m.group(2).strip()
    return f"Q: {question}\nAnswer:\n- {answer}"


# ══════════════════ v5 新增：合并插件（八步流水） ══════════════════

class PluginMerge(Plugin):
    """对照 merge.go:44-96。chunk_map 是「回表」数据源（含父块与邻居），
    对照生产版的 chunkRepo.ListChunksByID 批量取块。"""

    def __init__(self, chunk_map):
        self.chunk_map = chunk_map

    def activation_events(self):
        return [CHUNK_MERGE]

    def on_event(self, event_type, state, next):
        if not needs_retrieval(state.intent):
            return next()  # merge.go:47-49

        # 1. 选输入：精排结果优先，否则检索结果按分排序（:101-113）。
        #    深拷贝 + 初始化 sub_chunk_ids——合并要改写 content，不能污染上游。
        source = state.rerank_result or sorted(state.search_result, key=lambda r: -r["score"])
        results = [{**r, "sub_chunk_ids": list(r.get("sub_chunk_ids", []))} for r in source]

        # 2. 初次去重：id + 内容签名（:59）
        results = remove_duplicate_results(results)

        # 3. 注入历史相关片段（:62）
        results = self._inject_history(state, results)
        if not results:
            return next()  # :68-74

        # 4. 父块还原（:77）
        results = self._resolve_parent_chunks(results)

        # 5. 分组 + 重叠拼接（:80）
        results = self._group_and_merge(results)

        # 6. FAQ 格式化（:83）
        for r in results:
            if r["chunk_type"] == "faq":
                r["content"] = build_faq_content(r["content"])

        # 7. 短块邻居扩展（:86）
        results = self._expand_short_chunks(results, state)

        # 7.5 扩展可能制造新的重叠，再拼一轮（:89）
        results = self._group_and_merge(results)

        # 8. 终去重 + 部分重叠剔除（:92-93）
        results = remove_duplicate_results(results)
        results = remove_partial_overlaps(results)

        state.merge_result = results
        return next()  # :96

    # ── 历史相关片段（merge_history.go:17-85 + search.go:152-167） ──

    def _inject_history(self, state, current):
        refs = self._history_references(state)
        if not refs:
            return current
        existing = {r["id"] for r in current}
        query_tokens = tokenize_simple(state.rewrite_query or state.query)
        kept = []
        for r in refs:
            if r["id"] in existing:
                continue  # 本轮已检回的不重复注入
            sim = jaccard(query_tokens, tokenize_simple(r["content"]))
            if sim < HISTORY_MIN_SIMILARITY:
                continue  # 与本轮问题无关的旧引用，别拿进来占地方
            r["score"] *= HISTORY_SCORE_DISCOUNT  # 旧货打折：让位给新检索的同分结果
            r["match_type"] = "history"
            r["history_similarity"] = round(sim, 4)
            kept.append(r)
            if len(kept) >= HISTORY_MAX_RESULTS:
                break
        if kept:
            state.debug["history_injected"] = [r["id"] for r in kept]
            return remove_duplicate_results(current + kept)
        return current

    def _history_references(self, state):
        """对照 getSearchResultFromHistory（search.go:152-167）：从最近往前找，
        第一轮带知识引用的对话贡献其引用块。"""
        for msg in reversed(state.history):
            ids = msg.get("retrieved_chunk_ids") or []
            refs = []
            for cid in ids:
                chunk = self.chunk_map.get(cid)
                if chunk:
                    refs.append({**chunk, "score": HISTORY_REF_BASE_SCORE,
                                 "sub_chunk_ids": []})
            if refs:
                return refs
        return []

    # ── 父块还原（merge.go:215-341，只实现 text 分支） ──

    def _resolve_parent_chunks(self, results):
        for r in results:
            if r["chunk_type"] != "text" or not r.get("parent_chunk_id"):
                continue  # image 子块的祖父链条（:343-385）课程不涉及
            parent = self.chunk_map.get(r["parent_chunk_id"])
            if not parent or not parent["content"] or parent["chunk_type"] != "parent_text":
                continue  # :316-318 防御：父块缺失/类型不对就保持原样
            r["content"] = parent["content"]  # 子块只是命中的证据，语境在父块里
            r["start_at"] = parent["start_at"]
            r["end_at"] = parent["end_at"]
            if r["id"] not in r["sub_chunk_ids"]:
                r["sub_chunk_ids"].append(r["id"])  # :339-341 记住是谁命中的
        return results

    # ── 分组 + 重叠拼接（merge.go:149-208） ──

    def _group_and_merge(self, results):
        groups = {}
        for r in results:
            groups.setdefault((r["knowledge_id"], r["chunk_type"]), []).append(r)
        merged = []
        for group in groups.values():
            group.sort(key=lambda r: (r["start_at"], r["end_at"]))  # :183-188
            merged.extend(merge_overlapping_chunks(group))
        return merged

    # ── 短块邻居扩展（merge_expand.go:10-252） ──

    def _expand_short_chunks(self, results, state):
        for r in results:
            if r["chunk_type"] != "text" or not r["content"]:
                continue
            if len(r["content"]) >= EXPAND_MIN_LEN:  # rune 长度（Python len 即字符数）
                continue
            base = self.chunk_map.get(r["id"])
            if not base or base["chunk_type"] != "text":
                continue

            prev_content, next_content, prev_ids, next_ids = "", "", [], []
            prev_cursor, next_cursor = base.get("pre_chunk_id"), base.get("next_chunk_id")

            # 首轮：先各取一个邻居（:141-159），之后交替向两侧伸展（:161-209）
            prev_cursor, prev_content, prev_ids = self._take_prev(
                base, prev_cursor, prev_content, prev_ids)
            next_cursor, next_content, next_ids = self._take_next(
                base, next_cursor, next_content, next_ids)

            merged = merge_ordered_content(prev_content, base["content"],
                                           next_content, EXPAND_MAX_LEN)
            while merged and len(merged) < EXPAND_MIN_LEN and (prev_cursor or next_cursor):
                taken_before = len(prev_ids) + len(next_ids)  # :174 的 expanded 标志
                if prev_cursor:
                    prev_cursor, prev_content, prev_ids = self._take_prev(
                        base, prev_cursor, prev_content, prev_ids)
                merged = merge_ordered_content(prev_content, base["content"],
                                               next_content, EXPAND_MAX_LEN)
                if len(merged) >= EXPAND_MIN_LEN:
                    break
                if next_cursor:
                    next_cursor, next_content, next_ids = self._take_next(
                        base, next_cursor, next_content, next_ids)
                merged = merge_ordered_content(prev_content, base["content"],
                                               next_content, EXPAND_MAX_LEN)
                if len(prev_ids) + len(next_ids) == taken_before:
                    break  # :206-208 两侧都没真正取到新邻居，防死循环

            if not merged:
                continue
            r["content"] = merged
            for cid in prev_ids + next_ids:
                if cid and cid not in r["sub_chunk_ids"]:
                    r["sub_chunk_ids"].append(cid)  # :218-227
            if prev_content:
                r["start_at"] = max(0, base["start_at"] - len(prev_content))  # :229-234
            r["end_at"] = r["start_at"] + len(r["content"])  # :235
            state.debug.setdefault("expanded", []).append(
                {"id": r["id"], "with": prev_ids + next_ids, "len": len(merged)})
        return results

    def _take_prev(self, base, cursor, content, ids):
        """向前取一个邻居：内容前置拼接（:176-186）。跨知识源即停。"""
        if not cursor:
            return "", content, ids
        chunk = self.chunk_map.get(cursor)
        if not chunk or chunk["knowledge_id"] != base["knowledge_id"]:
            return "", content, ids
        return (chunk.get("pre_chunk_id") or "",
                concat_no_overlap(chunk["content"], content),
                [chunk["id"]] + ids)

    def _take_next(self, base, cursor, content, ids):
        """向后取一个邻居：内容后置拼接（:193-204）。"""
        if not cursor:
            return "", content, ids
        chunk = self.chunk_map.get(cursor)
        if not chunk or chunk["knowledge_id"] != base["knowledge_id"]:
            return "", content, ids
        return (chunk.get("next_chunk_id") or "",
                concat_no_overlap(content, chunk["content"]),
                ids + [chunk["id"]])


# ══════════════════ v5 新增：FILTER_TOP_K 插件 ══════════════════

class PluginFilterTopK(Plugin):
    """对照 filter_top_k.go：整条链最朴素的一步——确定性排序后切片。
    排序的多级 tiebreaker（:76-99）是为了可复现：合并阶段经过了 map 分组，
    没有稳定排序的话同分结果的顺序会随机漂移。"""

    def __init__(self, top_k=RERANK_TOP_K):
        self.top_k = top_k

    def activation_events(self):
        return [FILTER_TOP_K]

    def on_event(self, event_type, state, next):
        if not needs_retrieval(state.intent):
            return next()

        def sort_and_cut(results):
            results.sort(key=lambda r: (-r["score"], r["knowledge_id"],
                                        r["chunk_type"], r["start_at"],
                                        r["end_at"], r["id"]))
            return results[:self.top_k] if 0 < self.top_k < len(results) else results

        if state.merge_result:
            state.merge_result = sort_and_cut(state.merge_result)
        elif state.rerank_result:
            state.rerank_result = sort_and_cut(state.rerank_result)
        elif state.search_result:
            state.search_result = sort_and_cut(state.search_result)
        return next()


class PluginGenerate(Plugin):
    """merge → rerank → search 逐级回退取数（对照下游插件的取数惯例）。"""

    def activation_events(self):
        return [CHAT_COMPLETION]

    def on_event(self, event_type, state, next):
        pool = state.merge_result or state.rerank_result or state.search_result
        if pool:
            prompt = build_prompt(state.query, pool[:state.k])
        else:
            prompt = state.query
        state.answer = generate(prompt, state.mock)
        return next()


# ══════════════════ 驱动与入口 ══════════════════

PIPELINE_EVENTS = [QUERY_UNDERSTAND, CHUNK_SEARCH, CHUNK_RERANK,
                   CHUNK_MERGE, FILTER_TOP_K, CHAT_COMPLETION]


def build_manager(chunks, chunk_mat, mock):
    bm25 = BM25Okapi([tokenize(c["content"]) for c in chunks])
    if mock:
        rerank_model = mock_rerank_model
    else:
        rerank_model = real_rerank_model if os.getenv("RERANK_MODEL_ID") else None
    manager = EventManager()
    manager.register(PluginQueryUnderstand(mock_understand_llm if mock else real_understand_llm))
    manager.register(PluginSearch(chunks, chunk_mat, bm25))
    manager.register(PluginLoggingBoost())
    manager.register(PluginRerank(rerank_model))
    manager.register(PluginMerge(load_all_chunks()))
    manager.register(PluginFilterTopK())
    manager.register(PluginGenerate())
    return manager


def run_pipeline(manager, state):
    for event_type in PIPELINE_EVENTS:
        err = manager.trigger(event_type, state)
        if err is not None:
            state.answer = f"[pipeline 中止] {err.description}（{err.error_type}）"
            return err
    return None


def covered_ids(result):
    """一条合并结果覆盖的全部 chunk id：自身 + 被并入的成员。"""
    return {result["id"], *result.get("sub_chunk_ids", [])}


def run_eval(manager, k, mock):
    """v5 起判定升级：期望块的**内容**进入 top-k 结果（含被合并的成员）即算
    命中——合并阶段的产出是拼好的上下文，按检索 id 判会漏记它的功劳。"""
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    passed = 0
    for q in queries:
        state = ChatState(query=q["query"], k=k, mock=mock, history=q.get("history", []))
        for event in (QUERY_UNDERSTAND, CHUNK_SEARCH, CHUNK_RERANK, CHUNK_MERGE, FILTER_TOP_K):
            manager.trigger(event, state)
        pool = (state.merge_result or state.rerank_result or state.search_result)[:k]
        got = set()
        for r in pool:
            got |= covered_ids(r)
        ok = set(q["expect_chunk_ids"]) <= got
        passed += ok
        shown = [r["id"] + (f"(+{','.join(r['sub_chunk_ids'])})" if r.get("sub_chunk_ids") else "")
                 for r in pool]
        print(f"{'PASS' if ok else 'MISS'}  {q['id']:<3s} [{q['scenario']}] {q['query']}")
        print(f"      期望 {q['expect_chunk_ids']} | 实际 top-{k} {shown}")
    print(f"\n覆盖命中：{passed}/{len(queries)}（对比 v4 的 7/9；判定含合并成员）")


def main():
    ap = argparse.ArgumentParser(description="v5 上下文合并四术")
    ap.add_argument("--query", help="单条提问")
    ap.add_argument("--eval", action="store_true", help="跑 eval/queries.json 全部问题")
    ap.add_argument("--k", type=int, default=3, help="top-k（默认 3）")
    ap.add_argument("--mock", action="store_true", help="用确定性假 embedding/伪 LLM/伪精排")
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
        pool = state.merge_result or state.rerank_result or state.search_result
        for r in pool[:args.k]:
            subs = f" ⊕{r['sub_chunk_ids']}" if r.get("sub_chunk_ids") else ""
            print(f"合并结果 {r['id']:<12s} score={r['score']:.4f} len={len(r['content'])}{subs}")
        if "expanded" in state.debug:
            print(f"短块扩展   {state.debug['expanded']}")
        if "history_injected" in state.debug:
            print(f"历史注入   {state.debug['history_injected']}")
        print("\n===== 回答 =====\n" + state.answer)
    else:
        ap.error("需要 --query 或 --eval")


if __name__ == "__main__":
    main()
