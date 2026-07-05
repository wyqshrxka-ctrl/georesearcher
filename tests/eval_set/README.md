# RAG 评估集（eval_set）

用于 M2 检索/生成评估的问答集。**paper 级 ground truth**（决策 D2）：
一条 case 的 `expected_paper_ids` 是"能回答该问题的论文 id 列表"。

评估集含两类题（用 tag `难度/单篇`、`难度/跨篇` 区分）：

- **单篇题**：`expected_paper_ids` 只有 1 篇。模拟"已知精确目标文档"场景，题目偏简单。
  注意：单篇 GT 下 `context_recall` 退化成 0/1（等价 hit_rate），`precision` 恒为 1/k，
  这两个指标失去分辨力——单篇题主要看 hit@k / MRR / NDCG。
- **跨篇题**：`expected_paper_ids` 含 2~4 篇，需综合多篇才能回答。此时
  `recall = 命中gold数 / gold总数`、`precision = 命中gold数 / top_k` **才有分辨力**
  （如 3 篇 gold 命中 2 篇 → recall 0.667），也更能反映真实检索质量、避免指标虚高。

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
   uv run python scripts/gen_eval_set.py                    # 默认 mixed（单篇+跨篇）
   uv run python scripts/gen_eval_set.py --mode group       # 只造跨篇多GT难题
   uv run python scripts/gen_eval_set.py --mode single      # 只造单篇简单题
   uv run python scripts/gen_eval_set.py --dry-run          # 只看抽样/分组计划，不调 LLM
   ```

   三种模式（`--mode`）：
   - `single`：按主题分层抽样论文，逐篇造题（GT = 1 篇）。
   - `group`：把**同一主题 tag** 下的论文聚成簇（2~`--group-size` 篇），整组喂
     `get_judge()`（temperature=0.0）造"需综合多篇才能回答"的跨篇题，GT = 多篇。
     若 LLM 返回 `relevant_paper_idx`，取其指定子集作 GT（否则用整簇）。
   - `mixed`（默认）：single + group 混合，target 一分为二。

   **分组只用主题 tag**（`教育不平等/*`、`空间分析/*`、`研究方法/*`），
   排除纯地理/区域 tag（欧洲/北美/中国等语义太宽，不构成"能共同回答一题"的簇）。

2. **人工校验**（决策 D1，不可跳过）：
   - 删掉不合理 / 过于简单 / 语料无法回答的题；
   - 核对每条 `expected_paper_ids`——**跨篇题尤其要确认"这几篇是否真的都相关"**
     （多篇 GT 放大了 LLM 相关性判断的噪声，可能把不相关论文塞进 GT）；
   - 保留 ~30 条高质量题（单篇/跨篇兼顾）。

3. **定稿**：另存为 `tests/eval_set/eval_set.json` 并入 git。

> 为什么半自动而非全自动：LLM 造题会产生泛泛/答非所问/GT 错配的噪声，
> 直接用会让评估指标失真。人工校验是评估集可信度的前提。
