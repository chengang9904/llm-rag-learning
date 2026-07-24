# -*- coding: utf-8 -*-
"""v1 测试：洋葱链机制本身（右折叠嵌套顺序、注册顺序即链序、短路、错误上抛、
装饰器后置语义），以及最重要的重构不变量——v1 的检索结果必须与 v0 逐题一致。
全部不依赖网络与 API Key。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v0_naive_rag as v0  # noqa: E402
import v1_pipeline_engine as v1  # noqa: E402


# ── 探针插件：把执行轨迹写进 trace，用来观测洋葱的嵌套顺序 ─────────────

class Probe(v1.Plugin):
    def __init__(self, name, trace, events=("EV",), error=None, call_next=True):
        self.name, self.trace, self.events = name, trace, list(events)
        self.error, self.call_next = error, call_next

    def activation_events(self):
        return self.events

    def on_event(self, event_type, state, next):
        self.trace.append(f"{self.name}:pre")
        if self.error is not None:
            return self.error       # 短路：不调 next()
        err = next() if self.call_next else None
        self.trace.append(f"{self.name}:post")
        return err


# ── 洋葱链机制 ──────────────────────────────────────────────────────────

def test_right_fold_produces_onion_order():
    """对照 buildHandler（chat_pipeline.go:53-68）：先注册的在外层，
    前置逻辑按注册顺序执行，后置逻辑按注册逆序执行。"""
    trace, m = [], v1.EventManager()
    for name in ("A", "B", "C"):
        m.register(Probe(name, trace))
    assert m.trigger("EV", None) is None
    assert trace == ["A:pre", "B:pre", "C:pre", "C:post", "B:post", "A:post"]

def test_registration_order_is_chain_order():
    trace, m = [], v1.EventManager()
    m.register(Probe("B", trace))
    m.register(Probe("A", trace))
    m.trigger("EV", None)
    assert trace == ["B:pre", "A:pre", "A:post", "B:post"]

def test_plugin_error_short_circuits_downstream():
    """插件返回错误且不调 next() → 下游插件完全不执行，错误原样返回。"""
    trace, m = [], v1.EventManager()
    boom = v1.PluginError("炸了", "boom")
    m.register(Probe("A", trace))
    m.register(Probe("B", trace, error=boom))
    m.register(Probe("C", trace))
    assert m.trigger("EV", None) is boom
    assert trace == ["A:pre", "B:pre", "A:post"]  # C 从未运行；A 的后置仍会执行

def test_decorator_propagates_inner_error_and_skips_post():
    """对照 wiki_boost.go:49-51：next() 报错时装饰器必须原样上抛、跳过后处理。"""
    class Decorator(v1.Plugin):
        def __init__(self, trace):
            self.trace = trace
        def activation_events(self):
            return ["EV"]
        def on_event(self, event_type, state, next):
            err = next()
            if err is not None:
                return err
            self.trace.append("decorated")  # 不应执行
            return None

    trace, m = [], v1.EventManager()
    boom = v1.PluginError("下游失败", "inner_boom")
    m.register(Decorator(trace))
    m.register(Probe("X", trace, error=boom))
    assert m.trigger("EV", None) is boom
    assert "decorated" not in trace

def test_trigger_unknown_event_returns_none():
    """对照 Trigger（chat_pipeline.go:74-77）：未注册的事件静默返回。"""
    assert v1.EventManager().trigger("NO_SUCH_EVENT", None) is None

def test_plugin_can_activate_on_multiple_events():
    trace, m = [], v1.EventManager()
    m.register(Probe("A", trace, events=["EV1", "EV2"]))
    m.trigger("EV1", None)
    m.trigger("EV2", None)
    assert trace == ["A:pre", "A:post", "A:pre", "A:post"]

def test_register_after_trigger_rebuilds_chain():
    """对照 Register（chat_pipeline.go:47-50）：每次注册都重建该事件的调用链。"""
    trace, m = [], v1.EventManager()
    m.register(Probe("A", trace))
    m.trigger("EV", None)
    m.register(Probe("B", trace))
    trace.clear()
    m.trigger("EV", None)
    assert trace == ["A:pre", "B:pre", "B:post", "A:post"]


# ── v1 插件行为 ─────────────────────────────────────────────────────────

def _mock_manager():
    chunks = v1.load_chunks()
    return v1.build_manager(chunks, v1.mock_embed([c["content"] for c in chunks]))

def test_logging_boost_decorates_search_result():
    state = v1.ChatState(query="如何重置青鸟的 API 密钥？", mock=True)
    assert v1.run_pipeline(_mock_manager(), state) is None
    trace = state.debug["search_trace"]
    assert trace["decorated_by"] == "PluginLoggingBoost"
    assert trace["hit_ids"] == [r["id"] for r in state.search_result]
    assert state.answer.startswith("[mock 回答]")

def test_empty_corpus_short_circuits_with_search_nothing():
    """检索为空 → PluginSearch 返回 ERR_SEARCH_NOTHING 且不调 next()：
    装饰器不执行、生成事件被驱动循环跳过、state.answer 是兜底文案。"""
    manager = v1.EventManager()
    manager.register(v1.PluginSearch([], v1.mock_embed(["占位"])[:0]))
    manager.register(v1.PluginLoggingBoost())
    manager.register(v1.PluginGenerate())
    state = v1.ChatState(query="随便问点什么", mock=True)
    err = v1.run_pipeline(manager, state)
    assert err is v1.ERR_SEARCH_NOTHING
    assert "search_trace" not in state.debug
    assert "pipeline 中止" in state.answer


# ── 重构不变量：v1 检索结果必须与 v0 逐题一致 ──────────────────────────

def test_v1_retrieval_identical_to_v0_on_all_eval_queries():
    v0_chunks = v0.load_chunks()
    v0_mat = v0.mock_embed([c["content"] for c in v0_chunks])
    manager = _mock_manager()
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    assert len(queries) == 9
    for q in queries:
        v0_ids = [c["id"] for c, _ in v0.retrieve(q["query"], v0_chunks, v0_mat, 3, mock=True)]
        state = v1.ChatState(query=q["query"], k=3, mock=True)
        manager.trigger(v1.CHUNK_SEARCH, state)
        v1_ids = [r["id"] for r in state.search_result]
        assert v1_ids == v0_ids, f"{q['id']}: v1 {v1_ids} != v0 {v0_ids}"
