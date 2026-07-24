# tests/

测试要求见 [`../docs/SPEC.md`](../docs/SPEC.md) 第七节：

- 纯代码路径（归一化、RRF、合并四术、MMR）——不依赖网络/API Key 的单元测试，
  用固定输入断言具体数值。
- 模型调用路径（查询理解、精排、生成）——提供 Mock LLM，覆盖结构化输出解析、
  精排降级重试、流式分块拼接。
- 每个版本至少一个端到端测试，跑 `eval/queries.json` 断言检索到预期 chunk id。

尚未实现——占位目录，测试文件随每个版本一并添加（`test_v0_naive_rag.py` 等）。
