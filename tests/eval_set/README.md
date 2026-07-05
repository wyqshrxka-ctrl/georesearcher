# RAG 评估集（eval_set）

用于 M2 检索/生成评估的问答集。**paper 级 ground truth**（决策 D2）：
一条 case 的 `expected_paper_ids` 是"能回答该问题的论文 id 列表"，只要检索命中其中
任意论文的任意 chunk 即算命中。

## 文件

- `eval_set.json` — **定稿**（入 git，评估时加载）。由草稿人工校验后另存而来。
- `eval_set.draft.json` — 脚本产出的**草稿**（可不入 git）。**未经校验，禁止直接当定稿跑指标。**

## 格式

顶层是 JSON 数组，每个元素对应一条 `EvalCase`（见
`src/georesearcher/capabilities/evaluation/schemas.py`）：

```json
[
  {
    "id": "q001",
    "question": "……具体、可被论文回答的研究问题……",
    "expected_paper_ids": ["doi:10.xxxx/yyyy"],
    "reference_answer": "简短参考答案（可选，生成层评估用）",
    "tags": ["教育不平等/学校隔离", "空间分析/空间自相关"]
  }
]
```

校验规则（`dataset.load_eval_set`）：`question` 非空、`expected_paper_ids` 非空、`id` 不重复。

## 重造 / 校验流程

1. **造草稿**（需 `DEEPSEEK_API_KEY`，从 `.env` 读）：

   ```bash
   uv run python scripts/gen_eval_set.py --target 40    # 造 ~40 条草稿
   uv run python scripts/gen_eval_set.py --dry-run      # 只看抽样计划，不调 LLM
   ```

   脚本按 tag 分层抽样论文（教育不平等 / 空间分析 / 研究方法 + 其它），
   取每篇父块正文喂 `get_judge()`（temperature=0.0）造题，输出
   `eval_set.draft.json`。

2. **人工校验**（决策 D1，不可跳过）：
   - 删掉不合理 / 过于简单 / 语料无法回答的题；
   - 核对每条 `expected_paper_ids` 是否真的是能回答该题的论文（必要时增删 id）；
   - 保留 ~30 条高质量题。

3. **定稿**：另存为 `tests/eval_set/eval_set.json` 并入 git。

> 为什么半自动而非全自动：LLM 造题会产生泛泛/答非所问/GT 错配的噪声，
> 直接用会让评估指标失真。人工校验是评估集可信度的前提。
