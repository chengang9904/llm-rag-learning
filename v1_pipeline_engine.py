# -*- coding: utf-8 -*-
"""v1 —— 洋葱式中间件插件引擎

检索逻辑与 v0 完全相同（评测结果也必须完全相同——这是刻意的）。
唯一的变化：把 v0 硬编码在 main 里的顺序调用，重构成「事件 + 插件链」——
Plugin 接口、EventManager、右折叠构建的洋葱式调用链（对照 WeKnora
chat_pipeline.go:11-68），外加一个展示「先 next() 后处理」装饰器写法的
LoggingBoost 插件（对照 wiki_boost.go:42-97）。

用法（与 v0 一致）：
    python v1_pipeline_engine.py --query "如何重置青鸟的 API 密钥？" --mock
    python v1_pipeline_engine.py --eval --mock
"""
import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MOCK_DIM = 256


# ══════════════════ 与 v0 完全相同的基础设施（可直接 diff 验证） ══════════════════

def load_chunks():
    """【代码算法】读入预切好的语料块。只索引叶子块——真实系统同样不索引父块，
    父块（parent_text）要到 v5 的合并阶段才被「还原」出来。"""
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
    """【模型调用】真实 embedding（OpenAI 兼容接口，读 .env 的 EMBEDDING_MODEL_ID）。"""
    from openai import OpenAI
    resp = OpenAI().embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small"), input=texts)
    return normalize(np.array([d.embedding for d in resp.data]))


def cosine_top_k(query_vec, chunk_mat, k):
    """【代码算法】余弦 top-k。kind="stable" 保证同分时按语料顺序取，结果可复现。"""
    scores = chunk_mat @ query_vec
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def build_prompt(query, chunks):
    """【代码算法】把检索结果原样拼进 prompt——合并/格式化手术要等 v5。"""
    context = "\n\n".join(f"[{c['id']}] {c['content']}" for c in chunks)
    return ("请只根据以下资料回答问题；资料不足以回答时，明确说不知道。\n\n"
            f"===== 资料 =====\n{context}\n\n===== 问题 =====\n{query}\n")


def generate(prompt, mock):
    """【模型调用】最终生成。--mock 返回占位回答，让无 Key 环境也能跑通整条链路。"""
    if mock:
        return "[mock 回答] 已跳过真实 LLM 调用；上面的 prompt 就是将喂给模型的最终内容。"
    from openai import OpenAI
    resp = OpenAI().chat.completions.create(
        model=os.getenv("MODEL_ID", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


# ══════════════════ v1 新增：洋葱式中间件引擎 ══════════════════

@dataclass
class PluginError:
    """对照 chat_pipeline.go:81-85：插件执行错误。返回 None 表示成功。"""
    description: str
    error_type: str
    err: Exception | None = None


# 对照 chat_pipeline.go:88-125 的预定义错误（v1 只需要这一个，后续版本再补）
ERR_SEARCH_NOTHING = PluginError("未检索到相关内容", "search_nothing")

# 事件类型。真实系统还有 LOAD_HISTORY / QUERY_UNDERSTAND / CHUNK_RERANK /
# CHUNK_MERGE / FILTER_TOP_K 等十余种，v2 起逐个补上。
CHUNK_SEARCH = "CHUNK_SEARCH"
CHAT_COMPLETION = "CHAT_COMPLETION"


@dataclass
class ChatState:
    """对照 types.ChatManage：贯穿整条链的可变状态，插件靠原地修改它来交流。"""
    query: str
    k: int = 3
    mock: bool = True
    search_result: list = field(default_factory=list)  # [{**chunk, "score": float}]
    answer: str = ""
    debug: dict = field(default_factory=dict)


class Plugin:
    """对照 chat_pipeline.go:11-21 的 Plugin 接口。Go 版签名多一个 ctx
    （取消/追踪用的管道设施），教学版省略；event_type 保留——一个插件可以
    注册到多个事件，回调时需要知道是哪个事件触发了它。"""

    def activation_events(self):
        raise NotImplementedError

    def on_event(self, event_type, state, next):
        raise NotImplementedError


class EventManager:
    """对照 chat_pipeline.go:24-78。register 的顺序就是链的嵌套顺序：
    同一事件上先注册的插件在洋葱的外层。"""

    def __init__(self):
        self.listeners = {}  # event_type -> [Plugin]（注册顺序）
        self.handlers = {}   # event_type -> 已折叠好的调用链

    def register(self, plugin):
        """对照 Register（chat_pipeline.go:40-51）：每注册一个插件，
        重新折叠一次该事件的调用链。"""
        for event_type in plugin.activation_events():
            self.listeners.setdefault(event_type, []).append(plugin)
            self.handlers[event_type] = self._build_handler(self.listeners[event_type])

    def _build_handler(self, plugins):
        """对照 buildHandler（chat_pipeline.go:53-68）：右折叠。
        从终止 no-op 开始，倒序把每个插件包在外面——最终 plugins[0] 在最外层，
        调用它的 next() 才会进入 plugins[1]，层层嵌套直到 no-op。
        默认参数 plugin=plugin / prev_next=nxt 是 Python 的闭包快照写法，
        与 Go 版第 59 行 `current := plugins[i]` 解决的是同一个循环变量捕获问题。"""
        def terminal(event_type, state):
            return None  # 链条尽头的 no-op（chat_pipeline.go:57）

        nxt = terminal
        for plugin in reversed(plugins):
            def wrapped(event_type, state, plugin=plugin, prev_next=nxt):
                return plugin.on_event(event_type, state,
                                       lambda: prev_next(event_type, state))
            nxt = wrapped
        return nxt

    def trigger(self, event_type, state):
        """对照 Trigger（chat_pipeline.go:71-78）：未注册的事件静默返回 None。"""
        handler = self.handlers.get(event_type)
        return handler(event_type, state) if handler else None


# ══════════════════ v1 的三个插件：复刻 v0 逻辑 + 一个装饰器 ══════════════════

class PluginSearch(Plugin):
    """【代码算法+模型调用】v0 的 retrieve 搬进插件。写法对照 rerank.go:38-266：
    先做自己的事，最后 `return next()` 把控制权交给同事件链上的下一个插件——
    「前置逻辑」风格。若检索为空则直接返回错误、不调 next()，即短路整条链。"""

    def __init__(self, chunks, chunk_mat):
        self.chunks = chunks        # 依赖注入进插件，对照 Go 版构造函数注入 service
        self.chunk_mat = chunk_mat

    def activation_events(self):
        return [CHUNK_SEARCH]

    def on_event(self, event_type, state, next):
        embed = mock_embed if state.mock else real_embed
        top = cosine_top_k(embed([state.query])[0], self.chunk_mat, state.k)
        state.search_result = [{**self.chunks[i], "score": s} for i, s in top]
        if not state.search_result:
            return ERR_SEARCH_NOTHING  # 短路：下游插件（含装饰器）一个都不会执行
        return next()  # 对照 rerank.go:265


class PluginLoggingBoost(Plugin):
    """【代码算法】装饰器插件：注册在 PluginSearch 之后、监听同一事件。
    写法逐行对照 wiki_boost.go:42-97——先调 next()（错误原样上抛），
    再对 state 里已就位的结果做后处理。v1 只加一条 debug 字段；v6 会把它
    换成真正的 Boost（特定类型分数 ×1.3 后重排）。

    注意一个容易读错的细节：它的 next() 打到的是链条尽头的 no-op（它是最内层），
    「后置」地位来自『注册在后 + next() 先行』的组合，而不是 next() 本身
    有什么魔法——WeKnora 里 WikiBoost（container.go:336，注册在 Rerank :323
    之后）与此完全同构。"""

    def activation_events(self):
        return [CHUNK_SEARCH]

    def on_event(self, event_type, state, next):
        err = next()  # 对照 wiki_boost.go:49-51：先让下游跑完
        if err is not None:
            return err
        state.debug["search_trace"] = {  # 后处理：不改上游一行代码，往结果上叠信息
            "hit_ids": [r["id"] for r in state.search_result],
            "top_score": round(state.search_result[0]["score"], 4),
            "decorated_by": "PluginLoggingBoost",
        }
        return None


class PluginGenerate(Plugin):
    """【模型调用】v0 的 build_prompt + generate 搬进插件，挂在生成事件上。"""

    def activation_events(self):
        return [CHAT_COMPLETION]

    def on_event(self, event_type, state, next):
        state.answer = generate(build_prompt(state.query, state.search_result), state.mock)
        return next()


# ══════════════════ 驱动与入口 ══════════════════

PIPELINE_EVENTS = [CHUNK_SEARCH, CHAT_COMPLETION]


def build_manager(chunks, chunk_mat):
    manager = EventManager()
    manager.register(PluginSearch(chunks, chunk_mat))
    manager.register(PluginLoggingBoost())  # 注册在 Search 之后 → 同事件链的内层
    manager.register(PluginGenerate())
    return manager


def run_pipeline(manager, state):
    """对照 KnowledgeQAByEvent（session_knowledge_qa.go:658-795）：按事件列表
    依次 Trigger（:717），任一插件报错即中止（真实系统此处走兜底回答）。"""
    for event_type in PIPELINE_EVENTS:
        err = manager.trigger(event_type, state)
        if err is not None:
            state.answer = f"[pipeline 中止] {err.description}（{err.error_type}）"
            return err
    return None


def run_eval(manager, k, mock):
    """跑 eval/queries.json。只触发 CHUNK_SEARCH 事件——事件粒度的好处：
    评测不需要生成，就不触发生成（对照 WeKnora 按需拼 eventList 的 AddIf 写法，
    session_knowledge_qa.go:181-187）。"""
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    passed = 0
    for q in queries:
        state = ChatState(query=q["query"], k=k, mock=mock)
        manager.trigger(CHUNK_SEARCH, state)
        got = [r["id"] for r in state.search_result]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        passed += ok
        print(f"{'PASS' if ok else 'MISS'}  {q['id']:<3s} [{q['scenario']}] {q['query']}")
        print(f"      期望 {q['expect_chunk_ids']} | 实际 top-{k} {got}")
    print(f"\n检索命中：{passed}/{len(queries)}（应与 v0 完全一致——v1 是纯结构重构）")


def main():
    ap = argparse.ArgumentParser(description="v1 洋葱式中间件插件引擎")
    ap.add_argument("--query", help="单条提问")
    ap.add_argument("--eval", action="store_true", help="跑 eval/queries.json 全部问题")
    ap.add_argument("--k", type=int, default=3, help="top-k（默认 3）")
    ap.add_argument("--mock", action="store_true", help="用确定性假 embedding，不调真实 API")
    args = ap.parse_args()
    if not args.mock:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))

    chunks = load_chunks()
    embed = mock_embed if args.mock else real_embed
    manager = build_manager(chunks, embed([c["content"] for c in chunks]))

    if args.eval:
        run_eval(manager, args.k, args.mock)
    elif args.query:
        state = ChatState(query=args.query, k=args.k, mock=args.mock)
        run_pipeline(manager, state)
        for r in state.search_result:
            print(f"检索命中 {r['id']:<12s} score={r['score']:.4f}")
        print(f"装饰器痕迹 {state.debug.get('search_trace')}")
        print("\n===== 回答 =====\n" + state.answer)
    else:
        ap.error("需要 --query 或 --eval")


if __name__ == "__main__":
    main()
