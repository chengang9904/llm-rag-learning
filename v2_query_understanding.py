# -*- coding: utf-8 -*-
"""v2 —— 查询理解（Query Understanding）

在 v1 的洋葱链上挂第一个真正做事的新插件：QUERY_UNDERSTAND 事件先于检索触发，
一次 LLM 调用同时完成「查询改写（指代消解+省略补全）」和「意图分类」，输出
结构化 {"rewrite_query": ..., "intent": ...}（对照 query_understand.go:58-161、
提示词模板 config/prompt_templates/rewrite.yaml 的 default_rewrite）。
下游 Search 插件改用 rewrite_query 检索；非检索意图（问候/闲聊）直接跳过检索。

用法：
    python v2_query_understanding.py --query "如何重置青鸟的 API 密钥？" --mock
    python v2_query_understanding.py --query "你好" --mock      # 观察意图门控
    python v2_query_understanding.py --eval --mock              # q8 应从 MISS 变 PASS
"""
import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MOCK_DIM = 256


# ══════════════════ 与 v0/v1 完全相同的基础设施 ══════════════════

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


def cosine_top_k(query_vec, chunk_mat, k):
    """【代码算法】余弦 top-k。kind="stable" 保证同分时按语料顺序取。"""
    scores = chunk_mat @ query_vec
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def build_prompt(query, chunks):
    """【代码算法】把检索结果原样拼进 prompt。注意用的是用户**原话**——
    改写只喂给检索，不进最终生成，避免改写偏差污染答案。"""
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


# ══════════════════ 与 v1 完全相同的洋葱引擎 ══════════════════

@dataclass
class PluginError:
    """对照 chat_pipeline.go:81-85。返回 None 表示成功。"""
    description: str
    error_type: str
    err: Exception | None = None


ERR_SEARCH_NOTHING = PluginError("未检索到相关内容", "search_nothing")

QUERY_UNDERSTAND = "QUERY_UNDERSTAND"  # v2 新事件
CHUNK_SEARCH = "CHUNK_SEARCH"
CHAT_COMPLETION = "CHAT_COMPLETION"


@dataclass
class ChatState:
    """对照 types.ChatManage。v2 新增：history / enable_rewrite（输入），
    rewrite_query / intent（PipelineState，chat_manage.go:113-116）。"""
    query: str
    k: int = 3
    mock: bool = True
    history: list = field(default_factory=list)  # [{"role": ..., "content": ...}]
    enable_rewrite: bool = True
    rewrite_query: str = ""
    intent: str = ""
    search_result: list = field(default_factory=list)
    answer: str = ""
    debug: dict = field(default_factory=dict)


class Plugin:
    """对照 chat_pipeline.go:11-21。"""

    def activation_events(self):
        raise NotImplementedError

    def on_event(self, event_type, state, next):
        raise NotImplementedError


class EventManager:
    """对照 chat_pipeline.go:24-78（详见 v1 文档，此处不再赘述）。"""

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


# ══════════════════ v2 新增：查询理解 ══════════════════

def needs_retrieval(intent):
    """【代码算法】意图门控。对照 QueryIntent.NeedsKBRetrieval
    （chat_manage.go:102-109）：kb_search / clarification / summarize / 空值
    需要检索，其余（greeting/chitchat/follow_up/...）跳过。空值视为需要检索
    是安全兜底——意图分类失败时宁可多检索。（web_search 分支还要看
    WebSearchEnabled 开关，chat_manage.go:153-158，课程不涉及。）"""
    return intent in ("kb_search", "clarification", "summarize", "")


def format_conversation_history(history):
    """【代码算法】对照 formatConversationHistory（query_understand.go:444-459）：
    把 [{"role","content"}] 消息列表配成问答对，渲染成 BEGIN/END 块。"""
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


# 对照 config/prompt_templates/rewrite.yaml 的 default_rewrite 模板（压缩版）：
# 系统提示词 = 改写规则 + 意图分类表 + JSON-only 要求 + few-shot + {{conversation}}
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
    """【代码算法】对照 types.RenderPromptPlaceholders：{{name}} 逐个替换。
    不能用 str.format——模板里的 JSON 示例带字面量大括号，format 会把它们
    当成占位符直接抛异常（WeKnora 选 {{}} 替换而非模板引擎，同样是这个原因）。"""
    for name, value in values.items():
        template = template.replace("{{" + name + "}}", value)
    return template


def build_understand_messages(query, history):
    """【代码算法】对照 buildPrompts（query_understand.go:283-315）：
    历史进 system 的 {{conversation}} 占位符，当前问题进 user。"""
    conversation = format_conversation_history(history)
    return [
        {"role": "system",
         "content": render_placeholders(UNDERSTAND_SYSTEM, {"conversation": conversation})},
        {"role": "user", "content": render_placeholders(UNDERSTAND_USER, {"query": query})},
    ]


def parse_structured_output(raw):
    """【代码算法】对照 parseStructuredQueryOutput（query_understand.go:343-364）：
    先整体按 JSON 解析；失败则截取首个 { 到最后一个 }（容忍 markdown 包裹/
    多余文字）再试。别名键容错对照 :373-374。解析彻底失败返回 None。"""
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
    for key in ("rewrite_query", "rewritten_query", "query", "question"):  # :373-374
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
    """【代码算法·伪模型】确定性 Mock LLM，模拟真实模型的三类行为：
    问候/闲聊 → 对应意图；带指代或「…呢」式省略的追问且有历史 → 把最近一轮
    用户问题拼进改写（一个粗糙但确定的省略补全，真实 LLM 会改写得更自然）；
    其余 → 原样透传 + kb_search。它从 prompt 文本反向解析 query 与历史，
    顺带充当了对 prompt 组装格式的检验。"""
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
    """【模型调用】真实查询理解调用。temperature=0.3 / max_tokens=150 照抄
    query_understand.go:121-122（低温度：改写要稳定，不要发挥）。"""
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("MODEL_ID", "gpt-4o-mini"), messages=messages,
        temperature=0.3, max_completion_tokens=150)
    return resp.choices[0].message.content


class PluginQueryUnderstand(Plugin):
    """查询理解插件，对照 query_understand.go:58-161。要点：
    - 进门先把 rewrite_query 设成原查询（:61）——后面任何一步失败，
      下游拿到的都是可用的默认值；
    - 所有失败路径（模型不可用 :95-100、调用出错 :125-131）一律
      return next() 降级，绝不让「改写锦上添花」变成「改写单点故障」。"""

    def __init__(self, llm):
        self.llm = llm  # 依赖注入：mock_understand_llm 或 real_understand_llm

    def activation_events(self):
        return [QUERY_UNDERSTAND]

    def on_event(self, event_type, state, next):
        state.rewrite_query = state.query  # :61 安全默认值先行
        if not state.enable_rewrite:
            return next()  # :64-71 未启用改写，静默跳过
        try:
            raw = self.llm(build_understand_messages(state.query, state.history))
        except Exception as exc:  # :125-131 模型调用失败 → 降级用原查询
            state.debug["understand_error"] = repr(exc)  # 对照 pipelineError 日志
            return next()
        parsed = parse_structured_output(raw)
        if parsed is None:
            if raw.strip():
                state.rewrite_query = raw.strip()  # :336-340 拿原文当改写结果
            return next()
        if parsed["rewrite_query"]:
            state.rewrite_query = parsed["rewrite_query"]  # :327-331 空改写不覆盖默认值
        state.intent = parsed["intent"]
        return next()  # :160


# ══════════════════ 插件：检索（v2 起用 rewrite_query + 意图门控） ══════════════════

class PluginSearch(Plugin):
    """与 v1 的差异只有两处：入口的意图门控（对照 rerank.go:41-43 每个检索段
    插件都做的 NeedsRetrieval 检查），以及检索输入换成 rewrite_query。"""

    def __init__(self, chunks, chunk_mat):
        self.chunks = chunks
        self.chunk_mat = chunk_mat

    def activation_events(self):
        return [CHUNK_SEARCH]

    def on_event(self, event_type, state, next):
        if not needs_retrieval(state.intent):
            return next()  # 非检索意图：整段检索静默跳过，链条继续
        query = state.rewrite_query or state.query  # v2 核心：检索吃的是改写后的查询
        embed = mock_embed if state.mock else real_embed
        top = cosine_top_k(embed([query])[0], self.chunk_mat, state.k)
        state.search_result = [{**self.chunks[i], "score": s} for i, s in top]
        if not state.search_result:
            return ERR_SEARCH_NOTHING
        return next()


class PluginLoggingBoost(Plugin):
    """与 v1 相同的后置装饰器（对照 wiki_boost.go:42-97）。"""

    def activation_events(self):
        return [CHUNK_SEARCH]

    def on_event(self, event_type, state, next):
        err = next()
        if err is not None:
            return err
        if state.search_result:
            state.debug["search_trace"] = {
                "hit_ids": [r["id"] for r in state.search_result],
                "top_score": round(state.search_result[0]["score"], 4),
                "decorated_by": "PluginLoggingBoost",
            }
        return None


class PluginGenerate(Plugin):
    """【模型调用】v2 起要处理两种情形：有检索结果 → 拼资料回答；
    无检索结果（非检索意图）→ 直接对话。"""

    def activation_events(self):
        return [CHAT_COMPLETION]

    def on_event(self, event_type, state, next):
        if state.search_result:
            prompt = build_prompt(state.query, state.search_result)
        else:
            prompt = state.query  # greeting/chitchat：无资料直接回答
        state.answer = generate(prompt, state.mock)
        return next()


# ══════════════════ 驱动与入口 ══════════════════

# v2 起事件多了一个；顺序即真实链路顺序：理解先于检索
PIPELINE_EVENTS = [QUERY_UNDERSTAND, CHUNK_SEARCH, CHAT_COMPLETION]


def build_manager(chunks, chunk_mat, mock):
    manager = EventManager()
    manager.register(PluginQueryUnderstand(mock_understand_llm if mock else real_understand_llm))
    manager.register(PluginSearch(chunks, chunk_mat))
    manager.register(PluginLoggingBoost())
    manager.register(PluginGenerate())
    return manager


def run_pipeline(manager, state):
    """对照 KnowledgeQAByEvent（session_knowledge_qa.go:658-795）。"""
    for event_type in PIPELINE_EVENTS:
        err = manager.trigger(event_type, state)
        if err is not None:
            state.answer = f"[pipeline 中止] {err.description}（{err.error_type}）"
            return err
    return None


def run_eval(manager, k, mock):
    """跑 eval/queries.json：触发 QUERY_UNDERSTAND + CHUNK_SEARCH（评测不生成）。
    q8 的 history 字段在这里进入 state——v2 第一次真正用上它。"""
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    passed = 0
    for q in queries:
        state = ChatState(query=q["query"], k=k, mock=mock, history=q.get("history", []))
        manager.trigger(QUERY_UNDERSTAND, state)
        manager.trigger(CHUNK_SEARCH, state)
        got = [r["id"] for r in state.search_result]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        passed += ok
        rewrite_note = f"（改写：{state.rewrite_query}）" if state.rewrite_query != q["query"] else ""
        print(f"{'PASS' if ok else 'MISS'}  {q['id']:<3s} [{q['scenario']}] {q['query']}{rewrite_note}")
        print(f"      期望 {q['expect_chunk_ids']} | 实际 top-{k} {got}")
    print(f"\n检索命中：{passed}/{len(queries)}（对比 v0/v1 的 5/9——q8 应被查询改写救回）")


def main():
    ap = argparse.ArgumentParser(description="v2 查询理解")
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
        for r in state.search_result:
            print(f"检索命中 {r['id']:<12s} score={r['score']:.4f}")
        if not state.search_result:
            print("（非检索意图，已跳过检索）")
        print("\n===== 回答 =====\n" + state.answer)
    else:
        ap.error("需要 --query 或 --eval")


if __name__ == "__main__":
    main()
