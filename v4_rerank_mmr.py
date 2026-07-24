# -*- coding: utf-8 -*-
"""v4 —— 精排（复合分数）与 MMR 多样性选择

在检索与生成之间插入 CHUNK_RERANK 事件（真实链序：精排在合并**之前**——
先从候选池筛出小集合，v5 的合并阶段才不用对陪跑结果做昂贵手术）：

  cross-encoder 重打分（对照 rerank.go:345）
    → 阈值过滤 + top1 兜底（:379-421）
    → 全滤空且阈值够高时降级重试一次（阈值 ×0.7、下限 0.3，:148-179）
    → 复合分数 0.6*模型分 + 0.3*基础分 + 0.1*来源权重，乘位置先验（:439-460）
    → MMR（λ*相关性 - (1-λ)*冗余度，λ=0.7）做多样性选择（:223,463-543）

用法：
    python v4_rerank_mmr.py --query "青鸟怎么保证消息不丢？" --mock
    python v4_rerank_mmr.py --eval --mock
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

# ── v3 参数（出处见 v3 文档） ──────────────────────────────────────────
RRF_K = 60
RRF_VECTOR_WEIGHT = 0.7
RRF_KEYWORD_WEIGHT = 0.3
EXPANSION_MAX_VARIANTS = 5
EXPANSION_CONCURRENCY = 16
EXPANSION_KW_DISCOUNT = 0.8
EMBEDDING_TOP_K = 10
VECTOR_THRESHOLD = 0.55
KEYWORD_THRESHOLD = 5.0

# ── v4 参数（直接照抄 WeKnora） ────────────────────────────────────────
RERANK_THRESHOLD = 0.2        # retrieval_config.go:72-78 默认值
RERANK_TOP_K = 5              # 课程值（生产默认 10，retrieval_config.go:64-70）
MMR_LAMBDA = 0.7              # rerank.go:223
DEGRADE_FACTOR = 0.7          # rerank.go:150（降级阈值 = 原阈值 × 0.7）
DEGRADE_FLOOR = 0.3           # rerank.go:148-153（触发条件 >0.3，降级下限 0.3）
FALLBACK_MIN_SCORE = 0.15     # rerank.go:394-421（top1 兜底的最低分）
COMPOSITE_MODEL_W = 0.6       # rerank.go:451
COMPOSITE_BASE_W = 0.3
COMPOSITE_SOURCE_W = 0.1


# ══════════════════ 与 v0-v3 相同的基础设施 ══════════════════

def load_chunks():
    with open(os.path.join(ROOT, "corpus", "chunks.json"), encoding="utf-8") as f:
        return [c for c in json.load(f) if c["chunk_type"] != "parent_text"]


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


# ══════════════════ 与 v1-v3 相同的洋葱引擎 ══════════════════

@dataclass
class PluginError:
    description: str
    error_type: str
    err: Exception | None = None


ERR_SEARCH_NOTHING = PluginError("未检索到相关内容", "search_nothing")

QUERY_UNDERSTAND = "QUERY_UNDERSTAND"
CHUNK_SEARCH = "CHUNK_SEARCH"
CHUNK_RERANK = "CHUNK_RERANK"  # v4 新事件
CHAT_COMPLETION = "CHAT_COMPLETION"


@dataclass
class ChatState:
    """v4 新增 rerank_result（对照 ChatManage.RerankResult）：精排后的小集合。
    下游取数规则：rerank_result 非空用它，否则退回 search_result——
    精排被跳过（如未配模型）时链路照常工作。"""
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


# ══════════════════ 与 v2 相同的查询理解（正文见 v2） ══════════════════

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


# ══════════════════ 与 v3 相同的混合检索（正文见 v3） ══════════════════

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


# ══════════════════ v4 新增：分词/Jaccard（MMR 的冗余度度量） ══════════════════

def tokenize_simple(text):
    """【代码算法】对照 searchutil.TokenizeSimple（textutil.go:40-63）：
    产出多字 token 集合（单字 token 被过滤 :58）。生产版中文走 jieba
    CutForSearch；课程无 jieba 依赖，用「汉字连续段的 bigram + ASCII 词」
    近似——对近重复文本，两种分词的 Jaccard 都会显著偏高，教学效果等价。"""
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
    """【代码算法】对照 searchutil.Jaccard（textutil.go:77-102）。"""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


# ══════════════════ v4 新增：精排模型（真实/Mock cross-encoder） ══════════════════

def mock_rerank_model(query, passages):
    """【代码算法·伪模型】确定性 mock cross-encoder：得分 = 查询 token 被
    passage 覆盖的比例（加权召回）。它模拟真实精排模型的关键性质——
    针对「查询-文档对」整体打分、量纲统一 [0,1]，从而能把 v3 候选池里
    RRF 分/BM25 原始分的量纲混杂一举抹平。返回 [(index, score)] 降序。"""
    q_tokens = tokenize_simple(query)
    results = []
    for i, p in enumerate(passages):
        p_tokens = tokenize_simple(p)
        score = len(q_tokens & p_tokens) / max(1, len(q_tokens))
        results.append((i, score))
    return sorted(results, key=lambda r: (-r[1], r[0]))


def real_rerank_model(query, passages):
    """【模型调用】OpenAI 兼容 /rerank 端点（Jina/Cohere 风格返回结构）。
    RERANK_MODEL_ID 未配置时上层直接跳过精排（对照 rerank.go:57-62）。"""
    from openai import OpenAI
    resp = OpenAI().post("/rerank", cast_to=dict, body={
        "model": os.environ["RERANK_MODEL_ID"], "query": query,
        "documents": passages})
    rows = sorted(resp["results"], key=lambda r: -r["relevance_score"])
    return [(r["index"], r["relevance_score"]) for r in rows]


# ══════════════════ v4 新增：复合分数与 MMR ══════════════════

def composite_score(result, model_score, base_score):
    """【代码算法】对照 compositeScore（rerank.go:439-460）：
    0.6*模型分 + 0.3*基础分 + 0.1*来源权重，再乘位置先验，最后 clamp [0,1]。
    - 来源权重：web_search 0.95、其余 1.0（:440-446）——外部网页可信度略降级；
    - 位置先验：1 + clamp(1 - start/(end+1), ±0.05)（:447-450）——文档开头的
      块最多 +5%（开头通常是定义/总述）；
    - 基础分是检索阶段的原始分，课程先 clamp01 防住 BM25 原始分（>1）炸穿
      权重——真实系统同样存在量纲混杂，靠 0.6 的模型分权重压制。"""
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
    """【代码算法】对照 applyMMR（rerank.go:463-543）：贪心选择——每轮在
    未选集合里挑 mmr = λ*相关性 - (1-λ)*冗余度 最大者；冗余度 = 与已选
    结果的**最大** Jaccard（不是平均：只要跟任何一条已选高度重合就该罚）。
    直接取分数前 N 会把近重复内容一起放进来，MMR 用 λ 在相关与多样之间调节。"""
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


# ══════════════════ v4 新增：精排插件 ══════════════════

class PluginRerank(Plugin):
    """对照 rerank.go:38-266。三条纪律贯穿全程：
    ① 每条退出路径都 return next()（除了拿不到模型 :71 和结果全空 :250）；
    ② 模型失败 → 回退原始候选继续走（:130-144），精排绝不当单点故障；
    ③ 阈值宁松勿空：全滤空先 top1 兜底（:390-408），还空且阈值 >0.3 就
       降级重试一次（×0.7、下限 0.3，:148-179）。"""

    def __init__(self, model, threshold=RERANK_THRESHOLD, top_k=RERANK_TOP_K):
        self.model = model  # None 表示未配置精排模型
        self.threshold = threshold
        self.top_k = top_k

    def activation_events(self):
        return [CHUNK_RERANK]

    def on_event(self, event_type, state, next):
        if not needs_retrieval(state.intent):
            return next()  # rerank.go:41-43
        if not state.search_result:
            return next()  # :51-56
        if self.model is None:
            state.debug["rerank_skip"] = "empty_model_id"  # :57-62
            return next()

        candidates = [r for r in state.search_result if r["content"].strip()]  # :74-88
        passages = [r["content"] for r in candidates]

        try:
            ranked = self._rank_with_threshold(state, passages, self.threshold)
        except Exception as exc:
            # :130-144 API 失败 → 回退原始候选继续走
            state.debug["rerank_api_error"] = repr(exc)
            return next()

        # :148-179 全滤空且阈值够高 → 降级重试一次
        if not ranked and self.threshold > DEGRADE_FLOOR:
            degraded = max(DEGRADE_FLOOR, self.threshold * DEGRADE_FACTOR)
            state.debug["rerank_degraded"] = {"from": self.threshold, "to": degraded}
            try:
                ranked = self._rank_with_threshold(state, passages, degraded)
            except Exception as exc:
                state.debug["rerank_api_error"] = repr(exc)
                return next()

        # :188-221 复合分数：模型分 0.6 + 基础分 0.3 + 来源权重 0.1，乘位置先验
        reranked = []
        for idx, model_score in ranked:
            if idx >= len(candidates):
                continue  # :195-197 防御模型返回越界索引
            r = dict(candidates[idx])
            r["base_score"] = r["score"]
            r["model_score"] = model_score
            r["score"] = composite_score(r, model_score, r["base_score"])
            reranked.append(r)

        # :223 MMR 多样性选择，k = min(候选数, max(1, RerankTopK))
        state.rerank_result = apply_mmr(
            reranked, min(len(reranked), max(1, self.top_k)))

        if not state.rerank_result:
            return ERR_SEARCH_NOTHING  # :237-251
        return next()  # :265

    def _rank_with_threshold(self, state, passages, threshold):
        """一次模型调用 + 阈值过滤 + top1 兜底，对照 p.rerank（rerank.go:318-411）。"""
        query = state.rewrite_query or state.query
        resp = self.model(query, passages)  # :345【模型调用】
        kept = [(i, s) for i, s in resp if s >= threshold]  # :379-388
        if not kept and resp and resp[0][1] >= FALLBACK_MIN_SCORE:
            kept = resp[:1]  # :390-408 top1 兜底：强扭的瓜总比没有瓜好——但太烂的瓜不要
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


class PluginGenerate(Plugin):
    """精排后的小集合优先；精排被跳过时退回检索结果（对照下游各插件的
    RerankResult-else-SearchResult 取数惯例）。"""

    def activation_events(self):
        return [CHAT_COMPLETION]

    def on_event(self, event_type, state, next):
        pool = state.rerank_result or state.search_result
        if pool:
            prompt = build_prompt(state.query, pool[:state.k])
        else:
            prompt = state.query
        state.answer = generate(prompt, state.mock)
        return next()


# ══════════════════ 驱动与入口 ══════════════════

PIPELINE_EVENTS = [QUERY_UNDERSTAND, CHUNK_SEARCH, CHUNK_RERANK, CHAT_COMPLETION]


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
        manager.trigger(CHUNK_RERANK, state)
        pool = state.rerank_result or state.search_result
        got = [r["id"] for r in pool[:k]]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        passed += ok
        notes = []
        if state.rewrite_query != q["query"]:
            notes.append(f"改写：{state.rewrite_query}")
        if "rerank_degraded" in state.debug:
            notes.append("阈值降级")
        suffix = f"（{'；'.join(notes)}）" if notes else ""
        print(f"{'PASS' if ok else 'MISS'}  {q['id']:<3s} [{q['scenario']}] {q['query']}{suffix}")
        print(f"      期望 {q['expect_chunk_ids']} | 实际 top-{k} {got}")
    print(f"\n检索命中：{passed}/{len(queries)}（对比 v3 的 7/9）")


def main():
    ap = argparse.ArgumentParser(description="v4 精排与 MMR 多样性选择")
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
        pool = state.rerank_result or state.search_result
        for r in pool[:args.k]:
            extra = (f" 模型分={r.get('model_score', 0):.3f} 基础分={clamp01(r.get('base_score', 0)):.3f}"
                     if "model_score" in r else "")
            print(f"精排命中 {r['id']:<12s} 复合分={r['score']:.4f}{extra}")
        print("\n===== 回答 =====\n" + state.answer)
    else:
        ap.error("需要 --query 或 --eval")


if __name__ == "__main__":
    main()
