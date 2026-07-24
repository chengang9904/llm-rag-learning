# -*- coding: utf-8 -*-
"""v5 测试：合并四术各自独立可测（父块还原/重叠拼接/FAQ 格式化/短块扩展）+
历史相关片段过滤 + FILTER_TOP_K + 端到端 8/9（覆盖判定含合并成员）。
重叠拼接直接用真实语料的 deploy 三连块验证「拼回等于原文切片」。
全部不依赖网络与 API Key。"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import v5_merge as v5  # noqa: E402

ALL = v5.load_all_chunks()


def _res(chunk_id, **over):
    r = {**ALL[chunk_id], "sub_chunk_ids": [], "score": 0.5}
    r.update(over)
    return r


# ── 内容签名去重（对照 textutil.go:14-24 + search.go:169-193） ─────────

def test_signature_normalizes_case_and_whitespace():
    assert v5.build_content_signature("Hello   World") == v5.build_content_signature("hello world")
    assert v5.build_content_signature("  ") == ""

def test_remove_duplicates_by_id_and_signature():
    a = {"id": "x1", "content": "同一段内容"}
    b = {"id": "x1", "content": "别的"}             # 同 id
    c = {"id": "x2", "content": "  同一段内容\n"}    # 不同 id，签名相同（首尾空白）
    d = {"id": "x3", "content": "独有内容"}
    out = v5.remove_duplicate_results([a, b, c, d])
    assert [r["id"] for r in out] == ["x1", "x3"]
    # 注意：签名只「压缩」空白不「删除」空白（strings.Fields 语义）——
    # "同一段内容" 与 "同一段 内容" 是不同签名
    assert (v5.build_content_signature("同一段内容")
            != v5.build_content_signature("同一段 内容"))


# ── append_with_overlap（对照 chunkmerge.go:38-68） ────────────────────

def test_append_with_overlap_merges_long_overlap():
    overlap = "乙丙丁戊己庚辛壬癸子丑寅"  # 恰好 12 字 = minOverlapRunes
    acc, nxt = "甲" * 20 + overlap, overlap + "卯辰"
    assert v5.append_with_overlap(acc, nxt, len(overlap)) == "甲" * 20 + overlap + "卯辰"

def test_append_with_overlap_short_overlap_falls_back_to_concat():
    """后缀短于 12 字符不参与匹配（表格分隔行之类会误配）→ 直接拼接。"""
    acc, nxt = "甲" * 20 + "乙丙丁", "乙丙丁戊"
    assert v5.append_with_overlap(acc, nxt, 3) == acc + nxt

def test_append_with_overlap_empty_sides():
    assert v5.append_with_overlap("", "内容", 0) == "内容"
    assert v5.append_with_overlap("内容", "", 0) == "内容"


# ── 重叠拼接：真实语料的 deploy 三连块（对照 merge_overlap.go:15-78） ──

def _deploy_doc_text():
    with open(os.path.join(ROOT, "corpus", "docs", "03-集群部署指南.md"),
              encoding="utf-8") as f:
        return f.read()

def test_merge_deploy_trio_reconstructs_source_slice():
    """课程语料 offset 逐字符真实——三个重叠块拼回后必须等于原文切片。"""
    chunks = [_res("deploy-01", score=0.9), _res("deploy-02", score=0.6),
              _res("deploy-03", score=0.7)]
    chunks.sort(key=lambda r: (r["start_at"], r["end_at"]))
    merged = v5.merge_overlapping_chunks(chunks)
    assert len(merged) == 1
    m = merged[0]
    doc = _deploy_doc_text()
    assert m["content"] == doc[ALL["deploy-01"]["start_at"]:ALL["deploy-03"]["end_at"]]
    assert m["sub_chunk_ids"] == ["deploy-02", "deploy-03"]
    assert m["score"] == 0.9  # 保留最高分
    assert m["end_at"] == ALL["deploy-03"]["end_at"]

def test_merge_keeps_non_overlapping_apart():
    a, b = _res("sdk-01"), _res("sdk-03")  # sdk-01 [20,214) 与 sdk-03 [392,624) 不相邻
    merged = v5.merge_overlapping_chunks(sorted([a, b], key=lambda r: r["start_at"]))
    assert len(merged) == 2

def test_merge_fully_contained_only_records_id():
    outer = _res("arch-parent", score=0.4)
    inner = _res("arch-02", score=0.8)
    merged = v5.merge_overlapping_chunks(
        sorted([outer, inner], key=lambda r: (r["start_at"], r["end_at"])))
    assert len(merged) == 1
    assert merged[0]["sub_chunk_ids"] == ["arch-02"]
    assert merged[0]["score"] == 0.8  # 分数取大者，内容保持外层


# ── FAQ 格式化（对照 merge_faq.go:100-134 的渲染格式） ─────────────────

def test_faq_content_rendered_as_q_answer():
    out = v5.build_faq_content(ALL["faq-02"]["content"])
    assert out.startswith("Q: 如何重置 API 密钥？\nAnswer:\n- ")
    assert "登录控制台" in out

def test_faq_non_matching_content_passthrough():
    assert v5.build_faq_content("不是 FAQ 格式的内容") == "不是 FAQ 格式的内容"


# ── 父块还原（对照 merge.go:306-341 的 text 分支） ─────────────────────

def _plugin():
    return v5.PluginMerge(ALL)

def test_parent_restore_expands_child_to_full_parent():
    r = _res("arch-02")
    out = _plugin()._resolve_parent_chunks([r])[0]
    assert out["content"] == ALL["arch-parent"]["content"]
    assert out["start_at"] == ALL["arch-parent"]["start_at"]
    assert out["end_at"] == ALL["arch-parent"]["end_at"]
    assert out["sub_chunk_ids"] == ["arch-02"]  # :339-341 记住命中者

def test_parent_restore_skips_orphans_and_faq():
    faq = _res("faq-02")     # 无 parent
    out = _plugin()._resolve_parent_chunks([faq])[0]
    assert out["content"] == ALL["faq-02"]["content"]


# ── 短块邻居扩展（对照 merge_expand.go:10-252） ────────────────────────

def test_expand_short_chunk_pulls_both_neighbors():
    """sdk-02（174 字）< 350 → 取 sdk-01 + sdk-03，600 字 ≥ 350 停。"""
    state = v5.ChatState(query="x")
    r = _res("sdk-02")
    out = _plugin()._expand_short_chunks([r], state)[0]
    assert set(out["sub_chunk_ids"]) == {"sdk-01", "sdk-03"}
    assert v5.EXPAND_MIN_LEN <= len(out["content"]) <= v5.EXPAND_MAX_LEN
    assert "publish(topic, payload, qos=1, ttl=None)" in out["content"]  # 本体
    assert "qos=0 表示至多一次" in out["content"]  # 邻居 sdk-03 的 qos 详解进来了

def test_expand_skips_long_and_non_text():
    state = v5.ChatState(query="x")
    long_text = _res("arch-02", content="长" * 400)
    faq = _res("faq-03")  # 96 字但 chunk_type=faq：FAQ 是自包含 Q/A，不扩展
    outs = _plugin()._expand_short_chunks([long_text, faq], state)
    assert outs[0]["sub_chunk_ids"] == [] and outs[1]["sub_chunk_ids"] == []

def test_expand_truncates_at_max_len():
    """merge_ordered_content 把拼接结果截断到 850（merge_expand.go:268-271）。"""
    chunk_map = {
        "b": {"id": "b", "knowledge_id": "k", "chunk_type": "text",
              "content": "基" * 100, "start_at": 900, "end_at": 1000,
              "pre_chunk_id": "p", "next_chunk_id": "n", "parent_chunk_id": None},
        "p": {"id": "p", "knowledge_id": "k", "chunk_type": "text",
              "content": "前" * 900, "start_at": 0, "end_at": 900,
              "pre_chunk_id": None, "next_chunk_id": "b", "parent_chunk_id": None},
        "n": {"id": "n", "knowledge_id": "k", "chunk_type": "text",
              "content": "后" * 900, "start_at": 1000, "end_at": 1900,
              "pre_chunk_id": "b", "next_chunk_id": None, "parent_chunk_id": None},
    }
    state = v5.ChatState(query="x")
    r = {**chunk_map["b"], "sub_chunk_ids": [], "score": 0.5}
    out = v5.PluginMerge(chunk_map)._expand_short_chunks([r], state)[0]
    assert len(out["content"]) == v5.EXPAND_MAX_LEN

def test_concat_no_overlap_strips_suffix_prefix():
    assert v5.concat_no_overlap("青鸟部署指南", "指南第二步") == "青鸟部署指南第二步"
    assert v5.concat_no_overlap("甲乙", "丙丁") == "甲乙丙丁"


# ── 历史相关片段（对照 merge_history.go:17-85） ────────────────────────

def _history_state(query, ids):
    return v5.ChatState(query=query, rewrite_query=query, history=[
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答", "retrieved_chunk_ids": ids},
    ])

def test_history_injects_similar_ref_with_discount():
    """与本轮问题 Jaccard≥0.15 的历史引用注入，分数打 6 折。"""
    query = ALL["faq-03"]["content"][:40]  # 与 faq-03 高度相似的查询
    state = _history_state(query, ["faq-03"])
    out = _plugin()._inject_history(state, [_res("err-01")])
    ids = [r["id"] for r in out]
    assert "faq-03" in ids
    injected = next(r for r in out if r["id"] == "faq-03")
    assert injected["score"] == pytest.approx(v5.HISTORY_REF_BASE_SCORE * 0.6)
    assert injected["match_type"] == "history"
    assert injected["history_similarity"] >= 0.15

def test_history_drops_dissimilar_ref():
    state = _history_state("和历史引用毫无词面关系的一句话哈哈哈", ["faq-03"])
    out = _plugin()._inject_history(state, [_res("err-01")])
    assert [r["id"] for r in out] == ["err-01"]

def test_history_skips_refs_already_retrieved():
    query = ALL["faq-03"]["content"][:40]
    state = _history_state(query, ["faq-03"])
    out = _plugin()._inject_history(state, [_res("faq-03")])
    assert len([r for r in out if r["id"] == "faq-03"]) == 1  # 不重复注入

def test_history_caps_at_three():
    """最多注入 3 条（merge_history.go:31,80-82）。"""
    query = "青鸟 消息 配额 部署 集群 网关 密钥 限流 背压 可靠 投递"
    many = ["faq-01", "faq-02", "faq-03", "faq-04", "faq-05", "bp-01", "bp-02"]
    state = _history_state(query, many)
    out = _plugin()._inject_history(state, [_res("err-01")])
    assert len(out) - 1 <= v5.HISTORY_MAX_RESULTS


# ── 部分重叠剔除 + FILTER_TOP_K ────────────────────────────────────────

def test_remove_partial_overlaps_drops_contained_lower_score():
    big = {"id": "big", "content": "青鸟网关限流令牌桶与背压水位线控制细节说明",
           "score": 0.9, "sub_chunk_ids": []}
    small = {"id": "small", "content": "限流令牌桶与背压水位线", "score": 0.5,
             "sub_chunk_ids": []}
    out = v5.remove_partial_overlaps([big, small])
    assert [r["id"] for r in out] == ["big"]
    assert "small" in out[0]["sub_chunk_ids"]  # 课程附加：溯源记录

def test_filter_top_k_sorts_deterministically_and_cuts():
    plugin = v5.PluginFilterTopK(top_k=2)
    state = v5.ChatState(query="x")
    state.merge_result = [
        {"id": "b", "score": 0.5, "knowledge_id": "k", "chunk_type": "text",
         "start_at": 10, "end_at": 20, "sub_chunk_ids": []},
        {"id": "a", "score": 0.5, "knowledge_id": "k", "chunk_type": "text",
         "start_at": 10, "end_at": 20, "sub_chunk_ids": []},  # 同分同位 → 按 id
        {"id": "c", "score": 0.9, "knowledge_id": "k", "chunk_type": "text",
         "start_at": 0, "end_at": 5, "sub_chunk_ids": []},
    ]
    manager = v5.EventManager()
    manager.register(plugin)
    manager.trigger(v5.FILTER_TOP_K, state)
    assert [r["id"] for r in state.merge_result] == ["c", "a"]


# ── 端到端：8/9 与覆盖判定 ─────────────────────────────────────────────

def _manager():
    chunks = v5.load_chunks()
    return v5.build_manager(chunks, v5.mock_embed([c["content"] for c in chunks]), mock=True)

def _queries():
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}

def _run(manager, q, k=3):
    state = v5.ChatState(query=q["query"], k=k, mock=True, history=q.get("history", []))
    for ev in (v5.QUERY_UNDERSTAND, v5.CHUNK_SEARCH, v5.CHUNK_RERANK,
               v5.CHUNK_MERGE, v5.FILTER_TOP_K):
        manager.trigger(ev, state)
    return state

def test_e2e_pass_set_is_8_of_9():
    """唯一剩下的 MISS 是 q3：mock cross-encoder 认不出「第二/三步」与
    「完整步骤」的语义关联，deploy-03 在精排就被滤掉——词面 mock 的边界。"""
    manager, queries = _manager(), _queries()
    for qid, q in queries.items():
        pool = _run(manager, q).merge_result[:3]
        got = set()
        for r in pool:
            got |= v5.covered_ids(r)
        ok = set(q["expect_chunk_ids"]) <= got
        assert ok == (qid != "q3"), f"{qid}: covered={got}"

def test_e2e_q4_healed_by_neighbor_expansion():
    """v4 的诚实回归在 v5 治愈：sdk-03 没被检索到，但作为邻居被拼进来了。"""
    manager, queries = _manager(), _queries()
    pool = _run(manager, queries["q4"]).merge_result
    top = pool[0]
    assert top["id"] == "sdk-02"
    assert "sdk-03" in top["sub_chunk_ids"]
    assert "qos=0 表示至多一次" in top["content"]

def test_e2e_q1_answers_from_full_parent():
    manager, queries = _manager(), _queries()
    pool = _run(manager, queries["q1"]).merge_result
    top = pool[0]
    assert len(top["content"]) == len(ALL["arch-parent"]["content"])

def test_e2e_final_pool_has_no_duplicate_or_unknown_ids():
    manager, queries = _manager(), _queries()
    for q in queries.values():
        pool = _run(manager, q).merge_result
        ids = [r["id"] for r in pool]
        assert len(ids) == len(set(ids)), q["id"]
        for r in pool:
            assert all(cid in ALL for cid in r["sub_chunk_ids"]), q["id"]
            assert len(pool) <= v5.RERANK_TOP_K
