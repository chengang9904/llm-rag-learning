# -*- coding: utf-8 -*-
"""v6 测试：TypeBoost 装饰器（next() 先行、×1.3、稳定重排、快速路径、错误上抛）、
上下文模板渲染、fire-and-forget 流式（门控生成器证明插件不等流跑完）、
以及与 v5 的最终回归：boost 不改变覆盖命中集合。全部不依赖网络与 API Key。"""
import json
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v5_merge as v5  # noqa: E402
import v6_boost_stream as v6  # noqa: E402


# ── PluginTypeBoost（对照 wiki_boost.go:37-97） ────────────────────────

class FakeRerank(v6.Plugin):
    """往 rerank_result 里塞固定结果的假精排，用于隔离测试装饰器。"""

    def __init__(self, results, trace=None, error=None):
        self.results, self.trace, self.error = results, trace, error

    def activation_events(self):
        return [v6.CHUNK_RERANK]

    def on_event(self, event_type, state, next):
        if self.trace is not None:
            self.trace.append("rerank")
        if self.error is not None:
            return self.error
        state.rerank_result = [dict(r) for r in self.results]
        return next()


def _boost_manager(results, trace=None, error=None):
    manager = v6.EventManager()
    manager.register(FakeRerank(results, trace, error))
    manager.register(v6.PluginTypeBoost())  # 注册在后 → 内层
    return manager


def _r(cid, ctype, score):
    return {"id": cid, "chunk_type": ctype, "score": score}


def test_boost_multiplies_target_type_and_resorts():
    """faq ×1.3 后稳定重排：0.5×1.3=0.65 > 0.6，faq 反超。"""
    trace = []
    manager = _boost_manager([_r("d1", "text", 0.6), _r("f1", "faq", 0.5)], trace)
    state = v6.ChatState(query="x")
    assert manager.trigger(v6.CHUNK_RERANK, state) is None
    assert [(r["id"], round(r["score"], 4)) for r in state.rerank_result] == \
        [("f1", 0.65), ("d1", 0.6)]
    assert state.debug["type_boost"]["boosted"] == 1
    assert trace == ["rerank"]  # 精排先做完工作才交棒进装饰器

def test_boost_fast_path_when_no_target_type():
    """对照 :53-64：没有目标类型 → 分数与顺序都不动（连重排都不做——
    这保护了 MMR 的多样性次序）。"""
    original = [_r("a", "text", 0.3), _r("b", "text", 0.9)]  # 故意非降序（MMR 序）
    manager = _boost_manager(original)
    state = v6.ChatState(query="x")
    manager.trigger(v6.CHUNK_RERANK, state)
    assert [r["id"] for r in state.rerank_result] == ["a", "b"]  # 未被重排
    assert "type_boost" not in state.debug

def test_boost_resort_overrides_mmr_order():
    """boost 一旦生效，重排是**全列表**按分数排序——MMR 次序被覆盖
    （wiki_boost.go:90-93 的真实副作用）。"""
    original = [_r("a", "text", 0.3), _r("b", "text", 0.9), _r("f", "faq", 0.1)]
    manager = _boost_manager(original)
    state = v6.ChatState(query="x")
    manager.trigger(v6.CHUNK_RERANK, state)
    assert [r["id"] for r in state.rerank_result] == ["b", "a", "f"]  # 全按分数了

def test_boost_propagates_inner_error():
    """对照 :49-51：next() 报错 → 原样上抛、后置逻辑不执行。"""
    boom = v6.PluginError("精排炸了", "boom")
    manager = _boost_manager([_r("f1", "faq", 0.5)], error=boom)
    state = v6.ChatState(query="x")
    assert manager.trigger(v6.CHUNK_RERANK, state) is boom
    assert state.rerank_result == [] and "type_boost" not in state.debug


# ── PluginIntoChatMessage（对照 into_chat_message.go:31-199） ──────────

def _render(state):
    manager = v6.EventManager()
    manager.register(v6.PluginIntoChatMessage())
    manager.trigger(v6.INTO_CHAT_MESSAGE, state)
    return state.user_content

def test_context_rendering_wraps_results():
    state = v6.ChatState(query="问题？", k=2)
    state.merge_result = [
        {"id": "a", "content": "资料甲", "score": 0.9},
        {"id": "b", "content": "资料乙", "score": 0.8},
        {"id": "c", "content": "不该出现（超出 k）", "score": 0.7},
    ]
    out = _render(state)
    assert '<context id="1">资料甲</context>' in out
    assert '<context id="2">资料乙</context>' in out
    assert "不该出现" not in out          # 只渲染 top-k
    assert "问题？" in out and "{{" not in out  # 占位符全部替换干净

def test_no_retrieval_intent_renders_with_empty_contexts():
    """对照 :80-110：非检索意图也走模板，query 用改写后的版本。"""
    state = v6.ChatState(query="你好呀", intent="greeting", rewrite_query="你好")
    out = _render(state)
    assert "<context" not in out and "你好" in out


# ── fire-and-forget 流式（对照 chat_completion_stream.go:41-238） ──────

def _stream_manager(stream_fn):
    manager = v6.EventManager()
    manager.register(v6.PluginChatCompletionStream(stream_fn))
    return manager

def test_stream_plugin_returns_before_stream_finishes():
    """fire-and-forget 的直接证明：流被闸门卡住，OnEvent 却已经返回。"""
    gate = threading.Event()

    def gated_stream(messages):
        gate.wait(timeout=5)  # 卡住：插件若同步等流，trigger 将卡在这
        yield {"content": "迟到的", "done": False}
        yield {"content": "内容", "done": True}

    state = v6.ChatState(query="x", event_bus=v6.EventBus())
    err = _stream_manager(gated_stream).trigger(v6.CHAT_COMPLETION_STREAM, state)
    assert err is None            # OnEvent 已返回（:237 return next()）
    assert state.answer == ""     # 而流还一个字都没产出
    gate.set()                    # 放行
    events = list(state.event_bus.drain())
    state.debug["stream_thread"].join(timeout=5)
    assert "".join(e["content"] for e in events) == "迟到的内容"
    assert events[-1]["done"] is True
    assert state.answer == "迟到的内容"  # 后台线程拼装完整答案

def test_stream_mock_chunks_reassemble():
    state = v6.ChatState(query="x", mock=True, event_bus=v6.EventBus(),
                         user_content="渲染好的 prompt")
    err = _stream_manager(None).trigger(v6.CHAT_COMPLETION_STREAM, state)
    assert err is None
    events = list(state.event_bus.drain())
    state.debug["stream_thread"].join(timeout=5)
    assert len(events) > 1                       # 确实是分块到达
    assert "".join(e["content"] for e in events) == state.answer
    assert events[-1]["done"] is True

def test_stream_requires_event_bus():
    """对照 :71-76：EventBus 缺失是硬错误——流式没有出口就没有意义。"""
    state = v6.ChatState(query="x", event_bus=None)
    err = _stream_manager(None).trigger(v6.CHAT_COMPLETION_STREAM, state)
    assert err is v6.ERR_MODEL_CALL

def test_stream_setup_failure_is_synchronous_error():
    """对照 :88-94：建流失败在 OnEvent 内同步可见。"""
    def broken(messages):
        raise RuntimeError("连不上")
    state = v6.ChatState(query="x", event_bus=v6.EventBus())
    err = _stream_manager(broken).trigger(v6.CHAT_COMPLETION_STREAM, state)
    assert err is v6.ERR_MODEL_CALL

def test_stream_mid_error_emits_error_event():
    """对照 :182-198：流中途出错 → 发 error 事件，已产出的部分保留。"""
    def flaky(messages):
        yield {"content": "前半", "done": False}
        raise RuntimeError("断流")
    state = v6.ChatState(query="x", event_bus=v6.EventBus())
    _stream_manager(flaky).trigger(v6.CHAT_COMPLETION_STREAM, state)
    events = list(state.event_bus.drain())
    state.debug["stream_thread"].join(timeout=5)
    assert events[0]["content"] == "前半"
    assert events[-1]["type"] == "error"
    assert state.answer == "前半"


# ── 端到端：与 v5 的最终回归 + 全链路流式 ──────────────────────────────

def _managers():
    chunks = v6.load_chunks()
    mat = v6.mock_embed([c["content"] for c in chunks])
    return (v5.build_manager(chunks, mat, mock=True),
            v6.build_manager(chunks, mat, mock=True))

def _queries():
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

def _top3_covered(module, manager, q):
    state = module.ChatState(query=q["query"], k=3, mock=True, history=q.get("history", []))
    for ev in (module.QUERY_UNDERSTAND, module.CHUNK_SEARCH, module.CHUNK_RERANK,
               module.CHUNK_MERGE, module.FILTER_TOP_K):
        manager.trigger(ev, state)
    covered = set()
    for r in state.merge_result[:3]:
        covered |= module.covered_ids(r)
    return covered, state

def test_e2e_pass_set_unchanged_from_v5():
    """v6 是架构收官：boost 只微调名次，覆盖命中集合与 v5 逐题相同（8/9）。"""
    m5, m6 = _managers()
    for qid, q in _queries().items():
        c5, _ = _top3_covered(v5, m5, q)
        c6, _ = _top3_covered(v6, m6, q)
        assert c5 == c6, f"{qid}: v5={c5} v6={c6}"

def test_e2e_boost_fires_on_faq_queries():
    _, m6 = _managers()
    queries = _queries()
    _, s2 = _top3_covered(v6, m6, queries["q2"])
    assert s2.debug["type_boost"]["boosted"] >= 1  # faq-02 被加成
    _, s5 = _top3_covered(v6, m6, queries["q5"])
    assert "type_boost" not in s5.debug  # q5 池里没有 faq → 快速路径

def test_e2e_full_pipeline_streams_answer():
    _, m6 = _managers()
    state = v6.ChatState(query="如何重置青鸟的 API 密钥？", mock=True,
                         event_bus=v6.EventBus())
    assert v6.run_pipeline(m6, state) is None
    assert "<context" in state.user_content  # 模板渲染发生在生成之前
    events = list(state.event_bus.drain())
    state.debug["stream_thread"].join(timeout=5)
    assert state.answer and "".join(e["content"] for e in events) == state.answer
