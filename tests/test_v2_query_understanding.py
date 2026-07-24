# -*- coding: utf-8 -*-
"""v2 测试：结构化输出解析的各层容错（对照 query_understand.go:321-407）、
意图门控（对照 chat_manage.go:102-109）、插件降级路径（模型失败绝不阻断链条）、
Mock LLM 行为，以及端到端：q8 被查询改写救回、无历史的题检索结果与 v1 一致。
全部不依赖网络与 API Key。"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v1_pipeline_engine as v1  # noqa: E402
import v2_query_understanding as v2  # noqa: E402


# ── parse_structured_output：对照 :343-364 的三层容错 ──────────────────

def test_parse_clean_json():
    parsed = v2.parse_structured_output('{"rewrite_query":"改写后","intent":"kb_search"}')
    assert parsed == {"rewrite_query": "改写后", "intent": "kb_search"}

def test_parse_alias_keys():
    """对照 :373-374：rewrite_query 缺席时依次尝试别名键。"""
    for key in ("rewritten_query", "query", "question"):
        parsed = v2.parse_structured_output(json.dumps({key: "别名改写", "intent": "kb_search"}))
        assert parsed["rewrite_query"] == "别名改写", key

def test_parse_markdown_wrapped_json():
    """对照 :353-359：容忍 markdown 代码块/多余文字，截取 { 到 } 再解析。"""
    raw = '好的，输出如下：\n```json\n{"rewrite_query":"包着的","intent":"greeting"}\n```'
    parsed = v2.parse_structured_output(raw)
    assert parsed == {"rewrite_query": "包着的", "intent": "greeting"}

def test_parse_garbage_returns_none():
    assert v2.parse_structured_output("这不是 JSON") is None
    assert v2.parse_structured_output("") is None
    assert v2.parse_structured_output("[1, 2, 3]") is None  # 非 dict 也算失败

def test_parse_empty_rewrite_field():
    parsed = v2.parse_structured_output('{"rewrite_query":"  ","intent":"chitchat"}')
    assert parsed == {"rewrite_query": "", "intent": "chitchat"}


# ── needs_retrieval：对照 chat_manage.go:102-109 ───────────────────────

def test_needs_retrieval_matrix():
    for intent in ("kb_search", "clarification", "summarize", ""):
        assert v2.needs_retrieval(intent), intent
    for intent in ("greeting", "chitchat", "follow_up", "image_only", "doc_only"):
        assert not v2.needs_retrieval(intent), intent


# ── prompt 组装 ─────────────────────────────────────────────────────────

def test_format_conversation_history_blocks():
    """对照 formatConversationHistory（:444-459）。"""
    history = [{"role": "user", "content": "问A"}, {"role": "assistant", "content": "答A"},
               {"role": "user", "content": "问B（无回答，应被忽略）"}]
    text = v2.format_conversation_history(history)
    assert text == ("------BEGIN------\nUser question: 问A\n"
                    "Assistant answer: 答A\n------END------\n")
    assert v2.format_conversation_history([]) == ""

def test_render_placeholders_tolerates_literal_braces():
    """回归测试：模板里的 JSON 示例带字面量大括号，str.format 会在此抛异常——
    这是 v2 开发时真实踩过的坑（WeKnora 用 {{}} 替换渲染，同因）。"""
    out = v2.render_placeholders('示例 {"a":"b"} 与 {{query}}', {"query": "Q"})
    assert out == '示例 {"a":"b"} 与 Q'

def test_understand_messages_contain_history_and_query():
    msgs = v2.build_understand_messages(
        "那超出之后呢？", [{"role": "user", "content": "免费版配额？"},
                           {"role": "assistant", "content": "10 万条。"}])
    assert "User question: 免费版配额？" in msgs[0]["content"]
    assert "## 用户问题\n那超出之后呢？" in msgs[1]["content"]


# ── PluginQueryUnderstand：降级路径永不阻断链条 ────────────────────────

def _run_understand(state, llm):
    plugin = v2.PluginQueryUnderstand(llm)
    manager = v2.EventManager()
    manager.register(plugin)
    return manager.trigger(v2.QUERY_UNDERSTAND, state)

def test_rewrite_disabled_skips_llm():
    calls = []
    state = v2.ChatState(query="原问题", enable_rewrite=False)
    assert _run_understand(state, lambda m: calls.append(m)) is None
    assert state.rewrite_query == "原问题" and calls == []  # :61 默认值 + :64-71 跳过

def test_llm_exception_degrades_to_original_query():
    """对照 :125-131：模型调用失败 → 原查询继续走，链条不断。"""
    def boom(messages):
        raise RuntimeError("网络炸了")
    state = v2.ChatState(query="原问题")
    assert _run_understand(state, boom) is None  # 返回 None：链条正常走完
    assert state.rewrite_query == "原问题"
    assert "understand_error" in state.debug

def test_llm_garbage_output_used_as_rewrite():
    """对照 :336-340：JSON 解析彻底失败时，把原始输出当改写结果。"""
    state = v2.ChatState(query="原问题")
    _run_understand(state, lambda m: "免费版配额超出之后的行为")
    assert state.rewrite_query == "免费版配额超出之后的行为"

def test_empty_rewrite_keeps_default_but_applies_intent():
    """对照 :327-331：空 rewrite 不覆盖默认值，intent 照常生效。"""
    state = v2.ChatState(query="原问题")
    _run_understand(state, lambda m: '{"rewrite_query":"","intent":"chitchat"}')
    assert state.rewrite_query == "原问题" and state.intent == "chitchat"


# ── Mock LLM 行为 ───────────────────────────────────────────────────────

def _mock_response(query, history=()):
    return json.loads(v2.mock_understand_llm(
        v2.build_understand_messages(query, list(history))))

def test_mock_llm_greeting_and_chitchat():
    assert _mock_response("你好")["intent"] == "greeting"
    assert _mock_response("你是谁？")["intent"] == "chitchat"

def test_mock_llm_ellipsis_completion_with_history():
    history = [{"role": "user", "content": "免费版每月的消息配额是多少？"},
               {"role": "assistant", "content": "10 万条。"}]
    out = _mock_response("那超出之后呢？", history)
    assert out["intent"] == "kb_search"
    assert "免费版每月的消息配额是多少" in out["rewrite_query"]
    assert "那超出之后呢" in out["rewrite_query"]

def test_mock_llm_passthrough_without_history():
    out = _mock_response("那超出之后呢？")  # 有指代特征但没有历史 → 不补全
    assert out["rewrite_query"] == "那超出之后呢？"


# ── 端到端 ─────────────────────────────────────────────────────────────

def _v2_manager():
    chunks = v2.load_chunks()
    return v2.build_manager(chunks, v2.mock_embed([c["content"] for c in chunks]), mock=True), chunks

def _queries():
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

def test_e2e_greeting_skips_retrieval_but_still_answers():
    manager, _ = _v2_manager()
    state = v2.ChatState(query="你好", mock=True)
    assert v2.run_pipeline(manager, state) is None
    assert state.intent == "greeting" and state.search_result == []
    assert state.answer.startswith("[mock 回答]")

def test_e2e_q8_rescued_by_rewrite():
    """v2 的核心战果：历史依赖题 q8 靠检索前的查询改写从 MISS 翻到 PASS。"""
    manager, _ = _v2_manager()
    q8 = _queries()["q8"]
    state = v2.ChatState(query=q8["query"], mock=True, history=q8["history"])
    manager.trigger(v2.QUERY_UNDERSTAND, state)
    manager.trigger(v2.CHUNK_SEARCH, state)
    assert state.rewrite_query != q8["query"]  # 确实发生了改写
    got = [r["id"] for r in state.search_result]
    assert set(q8["expect_chunk_ids"]) <= set(got), got

def test_e2e_queries_without_history_identical_to_v1():
    """回归不变量：无历史的题，v2 的改写是透传，检索结果必须与 v1 逐题一致。"""
    v1_chunks = v1.load_chunks()
    v1_manager = v1.build_manager(v1_chunks, v1.mock_embed([c["content"] for c in v1_chunks]))
    v2_manager, _ = _v2_manager()
    for qid, q in _queries().items():
        if q.get("history"):
            continue
        s1 = v1.ChatState(query=q["query"], k=3, mock=True)
        v1_manager.trigger(v1.CHUNK_SEARCH, s1)
        s2 = v2.ChatState(query=q["query"], k=3, mock=True)
        v2_manager.trigger(v2.QUERY_UNDERSTAND, s2)
        v2_manager.trigger(v2.CHUNK_SEARCH, s2)
        assert s2.rewrite_query == q["query"], qid  # 透传
        assert [r["id"] for r in s2.search_result] == [r["id"] for r in s1.search_result], qid
