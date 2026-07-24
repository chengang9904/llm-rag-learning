# -*- coding: utf-8 -*-
"""v0 —— 朴素检索问答（Naive RAG）

最小骨架：embedding 语料 → embedding 查询 → 余弦 top-k → 拼进 prompt → 调 LLM。
无框架、无插件链、无任何检索后处理。每个函数都标注了它是【代码算法】还是【模型调用】。

用法：
    python v0_naive_rag.py --query "如何重置青鸟的 API 密钥？" --mock
    python v0_naive_rag.py --eval --mock     # 跑 eval/queries.json 全部问题（只判检索命中）
去掉 --mock 则走真实 API，需要 .env 配置（见 .env.example）。
"""
import argparse
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MOCK_DIM = 256


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
    """【代码算法】Mock embedding：字符 2/3-gram 特征哈希，确定性、无需 API Key。
    本质是词面重合度的向量化伪装——它答对的题恰是真实语义向量可能翻车的题（q6），
    它答错的题恰是语义向量的强项（q7）。详见 docs/v0-朴素检索问答.md 的评测记录。"""
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
    """【代码算法】把检索结果原样拼进 prompt——v0 没有任何合并/格式化手术。"""
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


def retrieve(query, chunks, chunk_mat, k, mock):
    """【代码算法】单条查询的检索：embedding 查询 → 余弦 top-k → 返回命中块。"""
    embed = mock_embed if mock else real_embed
    top = cosine_top_k(embed([query])[0], chunk_mat, k)
    return [(chunks[i], score) for i, score in top]


def run_eval(chunks, chunk_mat, k, mock):
    """跑 eval/queries.json 全部问题。只判定「预期 chunk 是否全部出现在 top-k」，
    不判定生成文本（生成文本不适合精确断言，见 SPEC 第七节）。"""
    with open(os.path.join(ROOT, "eval", "queries.json"), encoding="utf-8") as f:
        queries = json.load(f)
    passed = 0
    for q in queries:
        got = [c["id"] for c, _ in retrieve(q["query"], chunks, chunk_mat, k, mock)]
        ok = set(q["expect_chunk_ids"]) <= set(got)
        passed += ok
        print(f"{'PASS' if ok else 'MISS'}  {q['id']:<3s} [{q['scenario']}] {q['query']}")
        print(f"      期望 {q['expect_chunk_ids']} | 实际 top-{k} {got}")
    print(f"\n检索命中：{passed}/{len(queries)}（答不好的题正是 v1–v6 的存在理由）")


def main():
    ap = argparse.ArgumentParser(description="v0 朴素检索问答")
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
    chunk_mat = embed([c["content"] for c in chunks])  # embedding 语料：一次性建内存索引

    if args.eval:
        run_eval(chunks, chunk_mat, args.k, args.mock)
    elif args.query:
        hits = retrieve(args.query, chunks, chunk_mat, args.k, args.mock)
        for c, score in hits:
            print(f"检索命中 {c['id']:<12s} score={score:.4f}")
        print("\n===== 回答 =====\n" + generate(build_prompt(args.query, [c for c, _ in hits]), args.mock))
    else:
        ap.error("需要 --query 或 --eval")


if __name__ == "__main__":
    main()
