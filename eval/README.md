# eval/

本目录存放贯穿 v0–v6 所有版本、保持不变的固定评测问题集。设计要求见
[`../docs/SPEC.md`](../docs/SPEC.md) 第四节。

- `queries.json` — 每条至少含 `{query, expect_chunk_ids, note}`，覆盖
  `corpus/` 里设计的 8 类场景各至少一题（关键词命中、语义命中、历史依赖、
  MMR 去冗余等），用于让学习者在同一组问题上对比 v0 到 v6 的答案质量变化。

尚未实现——占位目录。
