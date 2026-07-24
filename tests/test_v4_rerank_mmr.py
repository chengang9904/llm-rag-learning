# -*- coding: utf-8 -*-
"""v4 测试：复合分数的精确数值（0.6/0.3/0.1、来源权重、位置先验、双重 clamp）、
MMR 贪心选择与 λ 敏感性、阈值过滤/top1 兜底/降级重试的交互、全失败路径降级，
以及端到端：7/9 基线、q5 翻正、q8 排序债还清、q4 的诚实回归。
全部不依赖网络与 API Key。"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v4_rerank_mmr as v4  # noqa: E402


# ── composite_score（对照 rerank.go:439-460） ──────────────────────────

def test_composite_base_formula():
    """0.6*模型分 + 0.3*基础分 + 0.1*来源权重（默认 1.0），无位置信息时先验=1。"""
    r = {"knowledge_source": ""}
    assert v4.composite_score(r, 0.5, 0.4) == pytest.approx(0.6 * 0.5 + 0.3 * 0.4 + 0.1)

def test_composite_web_search_source_discount():
    """对照 :440-446：web_search 来源权重 0.95。"""
    r = {"knowledge_source": "web_search"}
    assert v4.composite_score(r, 0.5, 0.4) == pytest.approx(0.3 + 0.12 + 0.095)

def test_composite_position_prior_bounds():
    """对照 :447-450：位置先验 = 1 + clamp(1 - start/(end+1), ±0.05)。
    文档开头的块顶格 +5%；深处的块趋近 +0%。"""
    head = {"knowledge_source": "", "start_at": 0, "end_at": 100}
    assert v4.composite_score(head, 0.5, 0.4) == pytest.approx(0.52 * 1.05)
    deep = {"knowledge_source": "", "start_at": 740, "end_at": 750}
    prior = 1 + (1 - 740 / 751)
    assert v4.composite_score(deep, 0.5, 0.4) == pytest.approx(0.52 * min(prior, 1.05))

def test_composite_clamps_base_and_final():
    """基础分 clamp01 防 BM25 原始分（>1）炸穿权重；最终分 clamp 到 [0,1]。"""
    r = {"knowledge_source": ""}
    assert v4.composite_score(r, 0.5, 4.2) == pytest.approx(0.3 + 0.3 + 0.1)  # base→1.0
    top = {"knowledge_source": "", "start_at": 0, "end_at": 100}
    assert v4.composite_score(top, 1.0, 1.0) == 1.0  # 1.0*1.05 → clamp 1.0


# ── tokenize_simple / jaccard（对照 searchutil/textutil.go） ───────────

def test_tokenize_simple_multi_char_tokens_only():
    """对照 textutil.go:58：单字 token 被过滤（生产 jieba，课程 bigram 近似）。"""
    assert v4.tokenize_simple("青鸟消息") == {"青鸟", "鸟消", "消息"}
    assert v4.tokenize_simple("qos=2 x abc") == {"qos", "abc"}  # 单字符 token 丢弃
    assert v4.tokenize_simple("") == set()

def test_jaccard_basics():
    a = {"青鸟", "消息"}
    assert v4.jaccard(a, a) == 1.0
    assert v4.jaccard(a, {"部署", "集群"}) == 0.0
    assert v4.jaccard(set(), set()) == 0.0
    assert v4.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


# ── apply_mmr（对照 rerank.go:463-543） ────────────────────────────────

def _cand(cid, content, score):
    return {"id": cid, "content": content, "score": score}

def test_mmr_suppresses_near_duplicate():
    """复制式近重复的教科书案例：B 与 A 几乎逐字相同，分数只差 0.01，
    直接取前 2 会选 A、B；MMR 用冗余罚把 C 顶上来。"""
    a = _cand("A", "青鸟通过预写日志刷盘和消费者确认保证消息可靠投递", 0.90)
    b = _cand("B", "青鸟通过预写日志刷盘和消费者确认保证消息可靠投递。", 0.89)
    c = _cand("C", "接入网关负责维持长连接并处理协议解析与心跳保活", 0.55)
    selected = v4.apply_mmr([a, b, c], 2)
    assert [r["id"] for r in selected] == ["A", "C"]  # B 被冗余罚压下去

def test_mmr_k_and_empty():
    assert v4.apply_mmr([], 3) == []
    assert v4.apply_mmr([_cand("A", "青鸟消息", 0.5)], 0) == []
    sel = v4.apply_mmr([_cand("A", "青鸟消息", 0.5)], 5)
    assert [r["id"] for r in sel] == ["A"]  # k 大于候选数时全选

def test_mmr_redundancy_is_max_not_avg():
    """冗余度取与已选集合的**最大** Jaccard（rerank.go:501-506）：
    只要与任何一条已选高度重合就该罚，平均值会稀释惩罚。"""
    a = _cand("A", "青鸟通过预写日志刷盘保证消息可靠投递", 0.9)
    x = _cand("X", "网关限流令牌桶与背压水位线控制生产者速率", 0.8)
    dup = _cand("D", "青鸟通过预写日志刷盘保证消息可靠投递", 0.79)  # 与 A 完全相同
    fresh = _cand("F", "会话层一致性哈希分片维护设备在线状态路由表", 0.60)
    sel = v4.apply_mmr([a, x, dup, fresh], 3)
    assert "D" not in [r["id"] for r in sel]  # 若用平均冗余，D(与X无重合)会稀释进选


# ── 阈值 / top1 兜底 / 降级重试（对照 rerank.go:148-179,379-421） ──────

def _run_rerank(model, search_result, threshold=v4.RERANK_THRESHOLD):
    plugin = v4.PluginRerank(model, threshold=threshold)
    manager = v4.EventManager()
    manager.register(plugin)
    state = v4.ChatState(query="测试查询", mock=True)
    state.rewrite_query = state.query
    state.search_result = search_result
    err = manager.trigger(v4.CHUNK_RERANK, state)
    return state, err

_POOL = [_cand("A", "内容甲甲甲", 0.01), _cand("B", "内容乙乙乙", 0.01)]

def test_threshold_filters_low_scores():
    state, err = _run_rerank(lambda q, p: [(0, 0.8), (1, 0.1)], _POOL)
    assert err is None
    assert [r["id"] for r in state.rerank_result] == ["A"]  # 0.1 < 0.2 被滤

def test_fallback_top1_when_all_below_threshold():
    """对照 :390-408：全滤空但 top1 ≥ 0.15 → 保 top1。"""
    state, err = _run_rerank(lambda q, p: [(1, 0.18), (0, 0.16)], _POOL)
    assert err is None
    assert [r["id"] for r in state.rerank_result] == ["B"]
    assert state.debug["rerank_fallback_top1"] == 0.18

def test_no_fallback_when_top1_too_low_default_threshold():
    """top1 < 0.15 且默认阈值 0.2 不满足降级条件（0.2 > 0.3 为假）→ 空结果报错。
    对照 :148 的触发条件与 :250 的 ErrSearchNothing。"""
    state, err = _run_rerank(lambda q, p: [(0, 0.1), (1, 0.05)], _POOL)
    assert err is v4.ERR_SEARCH_NOTHING
    assert "rerank_degraded" not in state.debug

def test_degradation_fires_only_on_empty_response():
    """代码考古发现：top1 兜底（≥0.15）让降级重试几乎成为死路径——模型返回
    非空时兜底总是先接住；只有模型返回**空列表**才走到降级分支（:148-179）。"""
    calls = []
    def empty_model(q, p):
        calls.append(1)
        return []
    state, err = _run_rerank(empty_model, _POOL, threshold=0.5)
    assert len(calls) == 2  # 原阈值一次 + 降级重试一次
    assert state.debug["rerank_degraded"] == {"from": 0.5, "to": pytest.approx(0.35)}
    assert err is v4.ERR_SEARCH_NOTHING

def test_degradation_floor_at_0_3():
    """对照 :151-153：降级阈值 = max(0.3, 原阈值×0.7)。"""
    state, _ = _run_rerank(lambda q, p: [], _POOL, threshold=0.35)
    assert state.debug["rerank_degraded"]["to"] == pytest.approx(0.3)

def test_api_error_falls_back_to_search_result():
    """对照 :130-144：模型炸了 → 保留原始候选继续走，精排不当单点故障。"""
    def boom(q, p):
        raise RuntimeError("rerank api down")
    state, err = _run_rerank(boom, _POOL)
    assert err is None and state.rerank_result == []
    assert "rerank_api_error" in state.debug

def test_missing_model_skips_rerank():
    """对照 :57-62：未配置精排模型 → 静默跳过，下游退回 search_result。"""
    state, err = _run_rerank(None, _POOL)
    assert err is None and state.rerank_result == []
    assert state.debug["rerank_skip"] == "empty_model_id"


# ── 端到端 ─────────────────────────────────────────────────────────────

def _manager():
    chunks = v4.load_chunks()
    return v4.build_manager(chunks, v4.mock_embed([c["content"] for c in chunks]), mock=True)

def _queries():
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

def _retrieve(manager, q, k=3):
    state = v4.ChatState(query=q["query"], k=k, mock=True, history=q.get("history", []))
    manager.trigger(v4.QUERY_UNDERSTAND, state)
    manager.trigger(v4.CHUNK_SEARCH, state)
    manager.trigger(v4.CHUNK_RERANK, state)
    return state

def test_e2e_pass_set_is_7_of_9():
    """v4 基线：q5 翻正、q4 诚实回归（sdk-03 模型分 0.158 < 阈值 0.2——
    它只回答了问题的一半，v0 起的双命中本来就是侥幸；真正的修复在 v5
    短块邻居扩展，且不依赖 sdk-03 被检索到）。"""
    manager, queries = _manager(), _queries()
    expected_pass = {"q1", "q2", "q5", "q6", "q7", "q8", "q9"}
    for qid, q in queries.items():
        pool = _retrieve(manager, q).rerank_result
        got = [r["id"] for r in pool[:3]]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        assert ok == (qid in expected_pass), f"{qid}: top3={got}"

def test_e2e_q5_rescued_and_q8_order_fixed():
    manager, queries = _manager(), _queries()
    q5_ids = [r["id"] for r in _retrieve(manager, queries["q5"]).rerank_result[:3]]
    assert "rel-v2-01" in q5_ids  # v0-v3 一直 MISS 的题
    q8_ids = [r["id"] for r in _retrieve(manager, queries["q8"]).rerank_result]
    assert q8_ids.index("faq-03") < q8_ids.index("faq-04")  # v2/v3 的排序债还清

def test_e2e_lambda_sensitivity_on_q5():
    """λ 在相关性与多样性之间调节：λ 越小、近重复 rel-v2 被压得越深
    （实测 λ=0.7 → 第 3；λ=0.3 → 第 5）。WeKnora 取 0.7 = 相关性优先。"""
    manager, queries = _manager(), _queries()
    state = _retrieve(manager, queries["q5"])
    pool = sorted(state.rerank_result, key=lambda r: -r["score"])
    rank_07 = [r["id"] for r in v4.apply_mmr(pool, 5, lam=0.7)].index("rel-v2-01")
    rank_03 = [r["id"] for r in v4.apply_mmr(pool, 5, lam=0.3)].index("rel-v2-01")
    assert rank_07 < rank_03

def test_e2e_rerank_pool_is_subset_of_search_result():
    manager, queries = _manager(), _queries()
    for q in queries.values():
        state = _retrieve(manager, q)
        search_ids = {r["id"] for r in state.search_result}
        assert all(r["id"] in search_ids for r in state.rerank_result), q["id"]
        for r in state.rerank_result:  # 复合分数三要素都已记录
            assert "model_score" in r and "base_score" in r
