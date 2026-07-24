# corpus/

本目录存放贯穿 v0–v6 所有版本、保持不变的共享教学语料。设计要求见
[`../docs/SPEC.md`](../docs/SPEC.md) 第四节，实现时与 v0 一并创建：

- `docs/*.md` — 人工设计的原始知识库文档（非真实抓取数据），需覆盖：长文档的
  父子块结构、FAQ、相邻重叠块、短块+邻居、近重复内容、仅关键词可命中的内容、
  仅语义可命中的内容、多轮对话历史样例。
- `chunks.json` — 预切分好的 chunk 列表（跳过真实切分/Embedding 生成流程），
  每条含 `id, knowledge_id, chunk_type, content, start_at, end_at,
  pre_chunk_id, next_chunk_id, parent_chunk_id`。

尚未实现——占位目录。
