# -*- coding: utf-8 -*-
"""v0 测试：余弦 top-k 的确定性（核心）、mock embedding 的确定性、
chunks.json 的结构完整性（后续版本的合并算法依赖这些约束）、
以及 mock 模式下的端到端检索命中/预期落空。全部不依赖网络与 API Key。"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v0_naive_rag as v0  # noqa: E402


# ── 余弦 top-k：确定性排序 ──────────────────────────────────────────────

def test_cosine_top_k_orders_by_similarity():
    query = np.array([1.0, 0.0])
    mat = v0.normalize(np.array([
        [0.0, 1.0],   # 正交，cos=0
        [1.0, 0.0],   # 同向，cos=1
        [1.0, 1.0],   # 45°，cos≈0.707
        [-1.0, 0.0],  # 反向，cos=-1
    ]))
    result = v0.cosine_top_k(query, mat, k=4)
    assert [i for i, _ in result] == [1, 2, 0, 3]
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[0] - 1.0) < 1e-9 and abs(scores[2]) < 1e-9

def test_cosine_top_k_ties_break_by_corpus_order():
    query = np.array([1.0, 0.0])
    mat = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # 三条同分
    assert [i for i, _ in v0.cosine_top_k(query, mat, k=3)] == [0, 1, 2]

def test_cosine_top_k_truncates_to_k():
    query = np.array([1.0, 0.0])
    mat = v0.normalize(np.random.default_rng(7).normal(size=(10, 2)))
    assert len(v0.cosine_top_k(query, mat, k=3)) == 3


# ── mock embedding：确定性、单位范数 ────────────────────────────────────

def test_mock_embed_is_deterministic():
    a = v0.mock_embed(["青鸟消息推送服务", "背压与限流"])
    b = v0.mock_embed(["青鸟消息推送服务", "背压与限流"])
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a[0], a[1])  # 不同文本向量不同

def test_mock_embed_rows_are_unit_norm():
    mat = v0.mock_embed(["如何重置 API 密钥", "E-4703", "背压"])
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-9)

def test_mock_embed_degenerate_text_is_zero_vector():
    # 单字符文本没有任何 2-gram，normalize 防除零后返回零向量（余弦恒为 0）
    assert np.linalg.norm(v0.mock_embed(["x"])[0]) == 0.0


# ── chunks.json 结构完整性（v5 的合并手术依赖这些不变量） ────────────────

def _all_chunks():
    with open(os.path.join(ROOT, "corpus", "chunks.json"), encoding="utf-8") as f:
        return json.load(f)

def test_chunk_ids_unique_and_fields_complete():
    chunks = _all_chunks()
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    required = {"id", "knowledge_id", "chunk_type", "content", "start_at",
                "end_at", "pre_chunk_id", "next_chunk_id", "parent_chunk_id"}
    for c in chunks:
        assert required <= set(c), f"{c['id']} 缺字段"

def test_chunk_content_matches_source_offsets():
    """content 必须与源文档 text[start_at:end_at] 严格一致——重叠拼接（v5）
    依赖偏移量是真实的。"""
    chunks = _all_chunks()
    for c in chunks:
        path = os.path.join(ROOT, "corpus", c["source"])
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert text[c["start_at"]:c["end_at"]] == c["content"], c["id"]

def test_chunk_links_are_reciprocal_and_valid():
    chunks = _all_chunks()
    by_id = {c["id"]: c for c in chunks}
    for c in chunks:
        for key in ("pre_chunk_id", "next_chunk_id", "parent_chunk_id"):
            if c[key] is not None:
                assert c[key] in by_id, f"{c['id']}.{key} 指向不存在的 {c[key]}"
        if c["next_chunk_id"]:
            assert by_id[c["next_chunk_id"]]["pre_chunk_id"] == c["id"]
        if c["parent_chunk_id"]:
            assert by_id[c["parent_chunk_id"]]["chunk_type"] == "parent_text"

def test_corpus_scenario_invariants():
    """SPEC 第四节的场景约束：短块 <350 字、部署块真实重叠、近重复对存在。"""
    by_id = {c["id"]: c for c in _all_chunks()}
    for cid in ("sdk-01", "sdk-02", "sdk-03", "sdk-04"):
        assert len(by_id[cid]["content"]) < 350, f"{cid} 不是短块"
    assert by_id["deploy-01"]["end_at"] > by_id["deploy-02"]["start_at"]
    assert by_id["deploy-02"]["end_at"] > by_id["deploy-03"]["start_at"]
    assert by_id["rel-v1-01"]["knowledge_id"] != by_id["rel-v2-01"]["knowledge_id"]

def test_load_chunks_excludes_parent_text():
    ids = {c["id"] for c in v0.load_chunks()}
    assert "arch-parent" not in ids and "arch-02" in ids


# ── 端到端（mock）：eval 固定问题的检索命中与「预期落空」 ───────────────

def _mock_retrieval_ids(query, k=3):
    chunks = v0.load_chunks()
    mat = v0.mock_embed([c["content"] for c in chunks])
    return [c["id"] for c, _ in v0.retrieve(query, chunks, mat, k, mock=True)]

def _queries():
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

def test_e2e_mock_hits_expected_chunks():
    """mock（词面 n-gram）下 v0 应稳定命中的题。语料若变动，此处会先报警。"""
    qs = _queries()
    for qid in ("q1", "q2", "q4", "q6", "q9"):
        got = _mock_retrieval_ids(qs[qid]["query"])
        assert set(qs[qid]["expect_chunk_ids"]) <= set(got), f"{qid}: {got}"

def test_e2e_mock_documented_gaps():
    """v0 的既定失败——这些断言记录的是缺陷本身，后续版本修复后会在各自
    版本的测试里翻转（docs/v0-朴素检索问答.md 有逐题解释）。"""
    qs = _queries()
    assert "bp-01" not in _mock_retrieval_ids(qs["q7"]["query"])      # 语义题：词面检索必挂
    assert "faq-03" not in _mock_retrieval_ids(qs["q8"]["query"])     # 历史题：不读历史必跑偏
    got_q3 = _mock_retrieval_ids(qs["q3"]["query"])
    assert not set(qs["q3"]["expect_chunk_ids"]) <= set(got_q3)       # 三块答案凑不齐
