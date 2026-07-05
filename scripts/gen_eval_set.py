#!/usr/bin/env python3
"""T6：半自动造 RAG 评估集草稿（LLM 造题 → 人工校验，决策 D1）。

流程：
  1. 连 SQLite，按 tag 分层抽样论文（保证主题覆盖：教育不平等/空间分析/方法 等）。
  2. 对每篇抽中的论文，取若干 section 全文，喂给 get_judge() LLM，用造题 prompt
     生成 1~2 个"该论文能回答的、具体的研究问题" + 简短参考答案（JSON 结构化输出）。
  3. 每条 QA 的 expected_paper_ids = 造题所用论文的 id（这是 GT 的来源）。
  4. 输出 tests/eval_set/eval_set.draft.json（草稿，绝不覆盖定稿）。
  5. 打印人工校验提示后停下。

⚠️ 本脚本产出的是**草稿**。作者必须人工校验（删不合理的题、修 expected_paper_ids），
   另存为 tests/eval_set/eval_set.json 后才是定稿。**不要拿未校验草稿当定稿去跑指标。**

用法：
    uv run python scripts/gen_eval_set.py                 # 默认造 ~40 条草稿
    uv run python scripts/gen_eval_set.py --target 30     # 指定草稿目标条数
    uv run python scripts/gen_eval_set.py --dry-run       # 只打印抽样计划，不调 LLM
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ── 造题 prompt（写进常量，面试要讲：让问题"具体、可被该论文回答"，避免抄标题）──
_GEN_PROMPT = """你是一位学术检索评测专家。下面是一篇论文的若干正文段落。\
请基于这些内容，设计 {n} 个"这篇论文能够回答的、具体的研究问题"，用于评测检索系统。

要求：
1. 问题必须**具体、可被这篇论文回答**，避免泛泛而谈的大问题。
2. 问题**不要直接照抄论文标题句式**（否则检索太容易，指标虚高）。
3. 每个问题配一个**简短参考答案**（1~3 句，基于给定正文，不要编造）。
4. 严格输出 JSON 数组，每个元素形如：
   {{"question": "……", "reference_answer": "……"}}
5. 只输出 JSON，不要任何解释或 markdown 代码块标记。

论文正文片段：
---
{context}
---

现在输出 {n} 个问题的 JSON 数组："""

# 分层抽样的主题簇（前缀匹配 tag），保证覆盖三大方向。
_STRATA = {
    "教育不平等": "教育不平等",
    "空间分析": "空间分析",
    "研究方法": "研究方法",
}

_CTX_CHARS = 3000  # 每篇喂给 LLM 的正文上限，控制 token 与长度偏差


def _slugify_counter():
    n = {"i": 0}

    def nxt() -> str:
        n["i"] += 1
        return f"q{n['i']:03d}"

    return nxt


def _paper_context(store, paper_id: str) -> str:
    """取一篇论文的父块全文，按 section_idx 排序拼接并截断。"""
    parents = store.get_parent_chunks_for_paper(paper_id)
    if not parents:
        return ""
    ordered = [parents[k] for k in sorted(parents.keys())]
    return "\n\n".join(ordered)[:_CTX_CHARS]


def _stratified_sample(papers: list[dict], target_papers: int, seed: int) -> list[dict]:
    """按主题簇分层抽样，尽量均衡覆盖，再随机补足到目标篇数。"""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {name: [] for name in _STRATA}
    other: list[dict] = []
    for p in papers:
        placed = False
        for name, prefix in _STRATA.items():
            if any(t.startswith(prefix) for t in p["tags"]):
                buckets[name].append(p)
                placed = True
                break
        if not placed:
            other.append(p)

    per_bucket = max(1, target_papers // (len(_STRATA) + 1))
    selected: dict[str, dict] = {}
    for name in _STRATA:
        pool = buckets[name][:]
        rng.shuffle(pool)
        for p in pool[:per_bucket]:
            selected[p["id"]] = p
    # 其它主题也抽一些
    rng.shuffle(other)
    for p in other[:per_bucket]:
        selected[p["id"]] = p

    # 随机补足到目标篇数
    remaining = [p for p in papers if p["id"] not in selected]
    rng.shuffle(remaining)
    for p in remaining:
        if len(selected) >= target_papers:
            break
        selected[p["id"]] = p

    result = list(selected.values())
    rng.shuffle(result)
    return result[:target_papers]


def _parse_qa_json(raw: str) -> list[dict]:
    """健壮解析 LLM 输出的 QA JSON 数组（容忍 ```json 代码块、前后杂讯）。"""
    if not raw:
        return []
    text = raw.strip()
    # 去掉 markdown 代码块围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    # 抽取第一个 [...] 数组
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("question"):
            out.append(
                {
                    "question": str(item["question"]).strip(),
                    "reference_answer": str(item.get("reference_answer", "")).strip() or None,
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="半自动造 RAG 评估集草稿")
    ap.add_argument("--target", type=int, default=40, help="草稿目标条数（默认 40）")
    ap.add_argument("--per-paper", type=int, default=2, help="每篇论文造几个问题（默认 2）")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument("--dry-run", action="store_true", help="只打印抽样计划，不调 LLM")
    ap.add_argument(
        "--out",
        default="tests/eval_set/eval_set.draft.json",
        help="草稿输出路径",
    )
    args = ap.parse_args()

    sys.path.insert(0, "src")
    from georesearcher.config import load_config
    from georesearcher.models.llm import get_judge
    from georesearcher.storage.sqlite_store import get_sqlite_store

    cfg = load_config()
    store = get_sqlite_store(cfg)

    papers = store.list_papers_with_tags(limit=1000)
    if not papers:
        print("❌ SQLite 里没有论文，先跑 batch_ingest。", file=sys.stderr)
        return 1

    # 需要的论文篇数：target 条 / 每篇 per_paper 条，向上取整
    n_papers = max(1, -(-args.target // args.per_paper))
    sample = _stratified_sample(papers, n_papers, args.seed)

    print(f"📚 语料共 {len(papers)} 篇；抽样 {len(sample)} 篇造题，"
          f"每篇 {args.per_paper} 题，目标 ~{args.target} 条草稿。")
    for p in sample:
        print(f"   - {p['id']}  {p['title'][:50]}  {p['tags'][:2]}")

    if args.dry_run:
        print("\n(--dry-run：仅打印抽样计划，未调用 LLM。)")
        return 0

    judge = get_judge(cfg)
    print(f"\n🤖 造题用 judge model = {judge.model}（temperature=0.0）")

    next_id = _slugify_counter()
    drafts: list[dict] = []

    for idx, p in enumerate(sample, start=1):
        context = _paper_context(store, p["id"])
        if not context.strip():
            print(f"   ⚠️ [{idx}/{len(sample)}] {p['id']} 无正文父块，跳过。")
            continue
        prompt = _GEN_PROMPT.format(n=args.per_paper, context=context)
        try:
            raw = judge.complete(prompt, temperature=0.0)
        except Exception as e:  # 单篇失败不中断整轮
            print(f"   ⚠️ [{idx}/{len(sample)}] {p['id']} LLM 调用失败: {e}")
            continue
        qas = _parse_qa_json(raw)
        if not qas:
            print(f"   ⚠️ [{idx}/{len(sample)}] {p['id']} 解析不出 QA，跳过。")
            continue
        for qa in qas[: args.per_paper]:
            drafts.append(
                {
                    "id": next_id(),
                    "question": qa["question"],
                    "expected_paper_ids": [p["id"]],  # GT 来源
                    "reference_answer": qa["reference_answer"],
                    "tags": p["tags"],
                }
            )
        print(f"   ✅ [{idx}/{len(sample)}] {p['id']} → {len(qas)} 题（累计 {len(drafts)}）")
        if len(drafts) >= args.target:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print(f"📝 已写出草稿 {len(drafts)} 条 → {out_path}")
    print("=" * 64)
    print("⚠️  这是**草稿**，不是定稿。请人工校验：")
    print("    1. 删掉不合理 / 过于简单 / 语料无法回答的题；")
    print("    2. 核对每条 expected_paper_ids 是否真的是能回答该题的论文；")
    print("    3. 校验后另存为 tests/eval_set/eval_set.json（~30 条，入 git）。")
    print("    在人工校验定稿前，不要拿本草稿去跑评估指标（决策 D1）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
