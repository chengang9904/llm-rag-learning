# -*- coding: utf-8 -*-
"""v3 测试：归一化（clamp01/按引擎公式/BM25 直通）、RRF 融合的精确数值、
单路退化去重、分词器、查询扩展的四种变体规则（含 Go 字节长度语义）、
低召回触发，以及端到端：7/9 基线与「近重复对已同时进入候选池」的 v4 伏笔。
全部不依赖网络与 API Key。"""
import json
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v3_hybrid_fusion as v3  # noqa: E402


# ── clamp01 / normalize_score（对照 normalizer.go） ─────────────────────

def test_clamp01_handles_nan_inf_and_bounds():
    """对照 normalizer.go:163-174：NaN/Inf 必须安全落进 [0,1]。"""
    assert v3.clamp01(float("nan")) == 0.0
    assert v3.clamp01(float("-inf")) == 0.0
    assert v3.clamp01(float("inf")) == 1.0
    assert v3.clamp01(-0.5) == 0.0
    assert v3.clamp01(1.5) == 1.0
    assert v3.clamp01(0.42) == 0.42

def test_keyword_scores_pass_through_unchanged():
    """对照 normalizer.go:117-121：BM25 分不归一化（RRF 只看排名）。"""
    assert v3.normalize_score(7.73, "keyword") == 7.73
    assert v3.normalize_score(-2.0, "keyword") == -2.0  # 连 clamp 都不做

def test_vector_raw_cosine_normalized_like_milvus():
    """对照 normalizer.go:124-130：原始余弦 [-1,1] → (s+1)/2 → clamp01。
    本课程内存引擎暴露的正是原始余弦，与 Milvus 同组。"""
    assert v3.normalize_score(-1.0, "vector", "memory") == 0.0
    assert v3.normalize_score(0.0, "vector", "memory") == 0.5
    assert v3.normalize_score(1.0, "vector", "memory") == 1.0
    assert v3.normalize_score(0.0, "vector", "milvus") == 0.5

def test_vector_zero_one_engines_only_clamp():
    """对照 normalizer.go:131-153：到手已是 [0,1] 的引擎只做 clamp。"""
    assert v3.normalize_score(0.7, "vector", "pgvector") == 0.7
    assert v3.normalize_score(-0.2, "vector", "pgvector") == 0.0
    assert v3.normalize_score(1.0000002, "vector", "pgvector") == 1.0


# ── tokenize（对照 query_expansion.go:232-259） ────────────────────────

def test_tokenize_han_unigram_ascii_runs():
    assert v3.tokenize("E-4703 错误") == ["E", "4703", "错", "误"]
    assert v3.tokenize("QN_MAX_INFLIGHT") == ["QN", "MAX", "INFLIGHT"]
    assert v3.tokenize("abc123中文x") == ["abc123", "中", "文", "x"]
    assert v3.tokenize("，。！") == []


# ── RRF 融合（对照 knowledgebase_search_fusion.go:84-142） ─────────────

def _r(cid, score, src="vec"):
    return {"id": cid, "score": score, "src": src}

def test_rrf_exact_scores_and_order():
    """固定输入验证公式：score = 0.7/(60+vecRank) + 0.3/(60+kwRank)。"""
    fused = v3.fuse_with_rrf(
        [_r("A", 0.9), _r("B", 0.8), _r("C", 0.7)],   # 向量路排名 1/2/3
        [_r("B", 9.0, "kw"), _r("D", 5.0, "kw")])      # 关键词路排名 1/2
    scores = {r["id"]: r["score"] for r in fused}
    assert scores["A"] == pytest.approx(0.7 / 61)
    assert scores["B"] == pytest.approx(0.7 / 62 + 0.3 / 61)
    assert scores["C"] == pytest.approx(0.7 / 63)
    assert scores["D"] == pytest.approx(0.3 / 62)
    assert [r["id"] for r in fused] == ["B", "A", "C", "D"]  # 两路都命中的 B 居首

def test_rrf_rank_uses_first_occurrence():
    """对照 :89-100：同 id 重复出现时取首次出现的名次。"""
    fused = v3.fuse_with_rrf([_r("A", 0.9), _r("A", 0.5), _r("B", 0.4)],
                             [_r("B", 3.0, "kw")])
    scores = {r["id"]: r["score"] for r in fused}
    assert scores["A"] == pytest.approx(0.7 / 61)   # rank 1，不是 2
    assert scores["B"] == pytest.approx(0.7 / 63 + 0.3 / 61)

def test_rrf_metadata_prefers_vector_entry():
    """对照 :103-113：同 id 的元数据优先取向量路的那份。"""
    fused = v3.fuse_with_rrf([_r("A", 0.9, "vec")], [_r("A", 9.9, "kw")])
    assert fused[0]["src"] == "vec"

def test_fuse_or_deduplicate_degrades_to_single_path():
    """对照 :33-50：任一路为空 → 退化为按分去重，保留原始分数（不做 RRF）。"""
    vec = [_r("A", 0.9), _r("A", 0.6), _r("B", 0.7)]
    out = v3.fuse_or_deduplicate(vec, [])
    assert [(r["id"], r["score"]) for r in out] == [("A", 0.9), ("B", 0.7)]
    kw = [_r("X", 8.0, "kw"), _r("Y", 6.0, "kw")]
    out2 = v3.fuse_or_deduplicate([], kw)
    assert [r["id"] for r in out2] == ["X", "Y"]
    assert out2[0]["score"] == 8.0  # BM25 原始分保留


# ── 查询扩展（对照 query_expansion.go:110-171） ────────────────────────

def test_expand_quoted_phrase():
    variants = v3.expand_queries("「静默挂起」是在什么情况下发生的", "x")
    assert "静默挂起" in variants

def test_expand_question_word_prefix_removed():
    variants = v3.expand_queries("如何重置密钥", "如何重置密钥")
    assert "重置密钥" in variants

def test_expand_keywords_variant_strips_stopwords():
    variants = v3.expand_queries("消息的投递是可靠的吗", "x")
    kw = [v for v in variants if " " in v]
    assert kw and "的" not in kw[0]  # 停用词「的」被去掉

def test_expand_segment_needs_more_than_5_bytes():
    """对照 :151 len(seg)>5——Go 字节长度：2 个汉字=6 字节可通过，纯 ASCII 短词不行。"""
    variants = v3.expand_queries("部署，ab", "x")
    assert "部署" in variants and "ab" not in variants

def test_expand_dedups_against_original_and_caps_at_5():
    assert v3.expand_queries("abc", "abc") == []  # 唯一关键词 <2 个、段太短 → 无变体
    many = v3.expand_queries("一二三四五六，七八九十甲乙，丙丁戊己庚辛，壬癸子丑寅卯，辰巳午未申酉，戌亥金木水火", "x")
    assert len(many) <= v3.EXPANSION_MAX_VARIANTS


# ── 两路检索与低召回触发 ───────────────────────────────────────────────

def _corpus():
    chunks = v3.load_chunks()
    return chunks, v3.mock_embed([c["content"] for c in chunks])

def test_keyword_search_threshold_cuts_noise():
    """本语料实测：真命中 6-19 分、噪声 2-5 分，阈值 5.0 应只留强命中。"""
    chunks, _ = _corpus()
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([v3.tokenize(c["content"]) for c in chunks])
    hits = v3.keyword_search(bm25, chunks, "固件 OTA 升级失败怎么办",
                             v3.EMBEDDING_TOP_K, v3.KEYWORD_THRESHOLD)
    assert [r["id"] for r in hits] == ["deploy-03"]  # 7.77 分；4.23 分的 err-02 被切掉

def test_expansion_triggers_on_small_corpus():
    """4 块小语料下任何查询都凑不满 EMBEDDING_TOP_K → 扩展必然触发，
    扩展命中只追加在尾部、不与主结果重复。"""
    chunks, mat = _corpus()
    small = chunks[:4]
    from rank_bm25 import BM25Okapi
    plugin = v3.PluginSearch(small, mat[:4], BM25Okapi([v3.tokenize(c["content"]) for c in small]))
    manager = v3.EventManager()
    manager.register(plugin)
    state = v3.ChatState(query="如何理解设计目标与总体分层", mock=True)
    state.rewrite_query = state.query
    manager.trigger(v3.CHUNK_SEARCH, state)
    assert "expansion" in state.debug
    ids = [r["id"] for r in state.search_result]
    assert len(ids) == len(set(ids))  # 扩展不引入重复


# ── 端到端：7/9 基线与 v4 伏笔 ────────────────────────────────────────

def _manager():
    chunks, mat = _corpus()
    return v3.build_manager(chunks, mat, mock=True)

def _queries():
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

def _retrieve(manager, q, k=3):
    state = v3.ChatState(query=q["query"], k=k, mock=True, history=q.get("history", []))
    manager.trigger(v3.QUERY_UNDERSTAND, state)
    manager.trigger(v3.CHUNK_SEARCH, state)
    return state

def test_e2e_pass_set_is_7_of_9():
    manager, queries = _manager(), _queries()
    expected_pass = {"q1", "q2", "q4", "q6", "q7", "q8", "q9"}
    for qid, q in queries.items():
        got = [r["id"] for r in _retrieve(manager, q).search_result[:3]]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        assert ok == (qid in expected_pass), f"{qid}: top3={got}"

def test_e2e_q7_rescued_by_hybrid():
    """v3 的核心战果之一：语义题 q7 靠「弱词面向量 + 弱 unigram BM25」的
    RRF 叠加进入 top-3——单路都不够强，融合才够。"""
    manager, queries = _manager(), _queries()
    got = [r["id"] for r in _retrieve(manager, queries["q7"]).search_result[:3]]
    assert "bp-01" in got

def test_e2e_q5_redundant_pair_now_both_in_pool():
    """v4 伏笔：近重复对 rel-v1/rel-v2 现在同时进入候选池（第 3、5 名附近），
    「top-k 被近重复挤占」的问题从这里开始真实存在。"""
    manager, queries = _manager(), _queries()
    ids = [r["id"] for r in _retrieve(manager, queries["q5"]).search_result]
    assert "rel-v1-01" in ids and "rel-v2-01" in ids

def test_e2e_no_duplicate_ids_in_candidates():
    manager, queries = _manager(), _queries()
    for q in queries.values():
        ids = [r["id"] for r in _retrieve(manager, q).search_result]
        assert len(ids) == len(set(ids)), q["id"]
