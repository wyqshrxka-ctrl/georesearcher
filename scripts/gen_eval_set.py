#!/usr/bin/env python3
"""T6：半自动造 RAG 评估集草稿（LLM 造题 → 人工校验，决策 D1）。

支持三种造题模式（--mode）：
  - single：逐篇造题，expected_paper_ids = 1 篇。"已知精确目标"场景，题目偏简单。
  - group：按主题簇整组喂入多篇论文，造"需综合这几篇才能回答"的跨篇难题，
           expected_paper_ids = 该簇多篇。让 recall/precision 有分辨力、指标不虚高。
  - mixed（默认）：single + group 混合，两类题共存，用 tag "难度/单篇"|"难度/跨篇" 区分。

GT 来源：造题喂进去的论文 id 就是该题的 ground truth（paper 级，决策 D2）。
group 模式若 LLM 返回 relevant_paper_idx，则取其指定子集作为 GT（更精确）。

⚠️ 本脚本产出的是**草稿**。作者必须人工校验（删不合理的题、核对 expected_paper_ids
   ——多篇 GT 尤其要确认"这几篇是否真的都相关"），另存为 eval_set.json 才是定稿。
   **不要拿未校验草稿当定稿去跑指标。**

用法：
    uv run python scripts/gen_eval_set.py                      # 默认 mixed
    uv run python scripts/gen_eval_set.py --mode group         # 只造跨篇难题
    uv run python scripts/gen_eval_set.py --mode single --target 30
    uv run python scripts/gen_eval_set.py --dry-run            # 只打印抽样/分组计划
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# ── 单篇造题 prompt（面试要讲：让问题"具体、可被该论文回答"，避免抄标题）──
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

# ── 整组（跨篇）造题 prompt：造需综合多篇才能回答的难题，多篇 GT ──
_GROUP_PROMPT = """你是一位学术检索评测专家。下面是同一主题下 {k} 篇论文的正文片段，\
各篇以 [论文i] 标注。请设计 {n} 个"**需要综合其中至少 2 篇论文才能完整回答**"的\
研究问题，用于评测检索系统能否召回一组相关文献。

要求：
1. 问题必须**跨越多篇论文**（如比较不同研究的发现/方法/结论、或综述该主题的共识与分歧），
   不能只靠单独一篇就能完整回答。
2. 用**同义改写**表达，**不要照抄任何一篇的标题或原句**（否则检索太容易，指标虚高）。
3. 每个问题标注 relevant_paper_idx：一个整数数组，指出回答该问题真正需要哪几篇（用上面的 i，从 1 起）。
4. 每个问题配一个**简短参考答案**（1~3 句，综合多篇，不要编造）。
5. 严格输出 JSON 数组，每个元素形如：
   {{"question": "……", "reference_answer": "……", "relevant_paper_idx": [1, 3]}}
6. 只输出 JSON，不要任何解释或 markdown 代码块标记。

论文正文片段：
---
{context}
---

现在输出 {n} 个跨篇问题的 JSON 数组："""

# 只用**主题性** tag 分组/分层（前缀匹配）；排除纯地理/区域 tag（语义太宽，不成"能共同回答一题"的簇）。
_THEME_PREFIXES = ("教育不平等/", "空间分析/", "研究方法/")

# 分层抽样的三大方向（single 模式用）。
_STRATA = {
    "教育不平等": "教育不平等",
    "空间分析": "空间分析",
    "研究方法": "研究方法",
}

_CTX_CHARS = 3000  # 单篇喂给 LLM 的正文上限
_CTX_CHARS_GROUP = 1500  # 整组喂入时每篇的正文上限，避免多篇拼接后 token 爆


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


def _theme_clusters(
    papers: list[dict],
    *,
    group_size: int,
    group_count: int,
    seed: int,
) -> list[list[dict]]:
    """按**主题 tag** 把论文聚成簇，每簇随机抽 <=group_size 篇（>=2）。

    只用 _THEME_PREFIXES 的主题 tag 分组；纯地理/区域 tag 不成簇。
    同一主题 tag 下的论文进同一候选池；从各池随机抽样组成 group_count 个簇。
    返回 list of 论文组（每组 2~group_size 篇）。
    """
    rng = random.Random(seed)
    # tag -> 该 tag 下的论文列表（只保留主题 tag）
    pools: dict[str, list[dict]] = {}
    for p in papers:
        for t in p["tags"]:
            if t.startswith(_THEME_PREFIXES):
                pools.setdefault(t, []).append(p)
    # 只保留能凑够 2 篇的主题池
    usable = [(t, ps) for t, ps in pools.items() if len(ps) >= 2]
    if not usable:
        return []
    # 大池优先（覆盖主力主题），再轮转抽样
    usable.sort(key=lambda x: -len(x[1]))
    rng.shuffle(usable)  # 打散，避免总从同几个 tag 出题

    clusters: list[list[dict]] = []
    seen_signatures: set[frozenset[str]] = set()
    for tag, pool in usable:
        if len(clusters) >= group_count:
            break
        shuffled = pool[:]
        rng.shuffle(shuffled)
        size = min(group_size, len(shuffled))
        if size < 2:
            continue
        group = shuffled[:size]
        sig = frozenset(p["id"] for p in group)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        clusters.append(group)
    return clusters[:group_count]


def _parse_qa_json(raw: str) -> list[dict]:
    """健壮解析 LLM 输出的 QA JSON 数组（容忍 ```json 代码块、前后杂讯）。

    保留可选字段 relevant_paper_idx（整组造题用，指哪几篇相关，从 1 起）。
    """
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
            rec = {
                "question": str(item["question"]).strip(),
                "reference_answer": str(item.get("reference_answer", "")).strip() or None,
            }
            idx = item.get("relevant_paper_idx")
            if isinstance(idx, list):
                # 只保留能转成 int 的正整数
                clean = []
                for v in idx:
                    try:
                        iv = int(v)
                    except (TypeError, ValueError):
                        continue
                    if iv >= 1:
                        clean.append(iv)
                if clean:
                    rec["relevant_paper_idx"] = sorted(set(clean))
            out.append(rec)
    return out


def _resolve_group_gt(group: list[dict], relevant_idx: list[int] | None) -> list[str]:
    """把 relevant_paper_idx（1-based）映射成 paper id 列表。

    - 缺省或映射后不足 2 篇 → 退回整组全部 id（宁多勿漏，人工校验再删）。
    - 越界 idx 忽略。
    """
    all_ids = [p["id"] for p in group]
    if not relevant_idx:
        return all_ids
    picked = [all_ids[i - 1] for i in relevant_idx if 1 <= i <= len(all_ids)]
    picked = sorted(set(picked), key=all_ids.index)
    return picked if len(picked) >= 2 else all_ids


def _gen_single(store, judge, sample, per_paper, target, next_id) -> list[dict]:
    """single 模式：逐篇造题，GT = 1 篇。"""
    drafts: list[dict] = []
    for idx, p in enumerate(sample, start=1):
        context = _paper_context(store, p["id"])
        if not context.strip():
            print(f"   ⚠️ [single {idx}/{len(sample)}] {p['id']} 无正文父块，跳过。")
            continue
        prompt = _GEN_PROMPT.format(n=per_paper, context=context)
        try:
            raw = judge.complete(prompt, temperature=0.0)
        except Exception as e:
            print(f"   ⚠️ [single {idx}/{len(sample)}] {p['id']} LLM 调用失败: {e}")
            continue
        qas = _parse_qa_json(raw)
        if not qas:
            print(f"   ⚠️ [single {idx}/{len(sample)}] {p['id']} 解析不出 QA，跳过。")
            continue
        for qa in qas[:per_paper]:
            drafts.append(
                {
                    "id": next_id(),
                    "question": qa["question"],
                    "expected_paper_ids": [p["id"]],  # GT 来源（单篇）
                    "reference_answer": qa["reference_answer"],
                    "tags": [*p["tags"], "难度/单篇"],
                }
            )
        print(f"   ✅ [single {idx}/{len(sample)}] {p['id']} → {len(qas)} 题（累计 {len(drafts)}）")
        if len(drafts) >= target:
            break
    return drafts


def _gen_group(store, judge, clusters, per_group, target, next_id) -> list[dict]:
    """group 模式：整组喂入多篇论文造跨篇题，GT = 多篇。"""
    drafts: list[dict] = []
    for idx, group in enumerate(clusters, start=1):
        blocks = []
        for i, p in enumerate(group, start=1):
            ctx = _paper_context(store, p["id"])[:_CTX_CHARS_GROUP]
            if ctx.strip():
                blocks.append(f"[论文{i}] {p['title']}\n{ctx}")
        if len(blocks) < 2:
            print(f"   ⚠️ [group {idx}/{len(clusters)}] 有效正文不足 2 篇，跳过。")
            continue
        context = "\n\n".join(blocks)
        prompt = _GROUP_PROMPT.format(k=len(blocks), n=per_group, context=context)
        try:
            raw = judge.complete(prompt, temperature=0.0)
        except Exception as e:
            print(f"   ⚠️ [group {idx}/{len(clusters)}] LLM 调用失败: {e}")
            continue
        qas = _parse_qa_json(raw)
        if not qas:
            print(f"   ⚠️ [group {idx}/{len(clusters)}] 解析不出 QA，跳过。")
            continue
        # 簇的主题 tag（取交集里的主题 tag，便于分层）
        theme_tags = sorted(
            {t for p in group for t in p["tags"] if t.startswith(_THEME_PREFIXES)}
        )
        for qa in qas[:per_group]:
            gt = _resolve_group_gt(group, qa.get("relevant_paper_idx"))
            drafts.append(
                {
                    "id": next_id(),
                    "question": qa["question"],
                    "expected_paper_ids": gt,  # GT 来源（多篇）
                    "reference_answer": qa["reference_answer"],
                    "tags": [*theme_tags, "难度/跨篇"],
                }
            )
        print(
            f"   ✅ [group {idx}/{len(clusters)}] {len(group)}篇 → {len(qas)} 题"
            f"（累计 {len(drafts)}）"
        )
        if len(drafts) >= target:
            break
    return drafts


def main() -> int:
    ap = argparse.ArgumentParser(description="半自动造 RAG 评估集草稿")
    ap.add_argument(
        "--mode", choices=["single", "group", "mixed"], default="mixed",
        help="造题模式：single 单篇 / group 跨篇多GT / mixed 混合（默认）",
    )
    ap.add_argument("--target", type=int, default=40, help="草稿目标条数（默认 40）")
    ap.add_argument("--per-paper", type=int, default=2, help="single：每篇造几题（默认 2）")
    ap.add_argument("--group-size", type=int, default=3, help="group：每簇论文数上限（默认 3）")
    ap.add_argument("--group-count", type=int, default=12, help="group：造多少个跨篇簇（默认 12）")
    ap.add_argument("--per-group", type=int, default=2, help="group：每簇造几题（默认 2）")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument("--dry-run", action="store_true", help="只打印抽样/分组计划，不调 LLM")
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

    do_single = args.mode in ("single", "mixed")
    do_group = args.mode in ("group", "mixed")

    # mixed 模式下把 target 一分为二（单篇/跨篇各一半）
    if args.mode == "mixed":
        single_target = args.target // 2
        group_target = args.target - single_target
    elif args.mode == "single":
        single_target, group_target = args.target, 0
    else:
        single_target, group_target = 0, args.target

    print(f"📚 语料共 {len(papers)} 篇；模式={args.mode}，目标 ~{args.target} 条草稿。")

    sample: list[dict] = []
    clusters: list[list[dict]] = []

    if do_single:
        n_papers = max(1, -(-single_target // args.per_paper))
        sample = _stratified_sample(papers, n_papers, args.seed)
        print(f"\n[single] 抽 {len(sample)} 篇，每篇 {args.per_paper} 题 → ~{single_target} 条：")
        for p in sample:
            print(f"   - {p['id']}  {p['title'][:46]}  {p['tags'][:2]}")

    if do_group:
        clusters = _theme_clusters(
            papers, group_size=args.group_size,
            group_count=args.group_count, seed=args.seed,
        )
        print(f"\n[group] {len(clusters)} 个主题簇，每簇 {args.per_group} 题 → ~{group_target} 条：")
        for i, g in enumerate(clusters, start=1):
            theme = next(
                (t for p in g for t in p["tags"] if t.startswith(_THEME_PREFIXES)), "?"
            )
            print(f"   簇{i} [{theme}] {len(g)}篇: " + ", ".join(p["id"] for p in g))

    if args.dry_run:
        print("\n(--dry-run：仅打印抽样/分组计划，未调用 LLM。)")
        return 0

    judge = get_judge(cfg)
    print(f"\n🤖 造题用 judge model = {judge.model}（temperature=0.0）\n")

    next_id = _slugify_counter()
    drafts: list[dict] = []
    if do_single:
        drafts += _gen_single(store, judge, sample, args.per_paper, single_target, next_id)
    if do_group:
        drafts += _gen_group(store, judge, clusters, args.per_group, group_target, next_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_single = sum(1 for d in drafts if "难度/单篇" in d["tags"])
    n_group = sum(1 for d in drafts if "难度/跨篇" in d["tags"])
    multi_gt = sum(1 for d in drafts if len(d["expected_paper_ids"]) >= 2)

    print("\n" + "=" * 64)
    print(f"📝 已写出草稿 {len(drafts)} 条 → {out_path}")
    print(f"   单篇 {n_single} 条 / 跨篇 {n_group} 条；多篇 GT {multi_gt} 条")
    print("=" * 64)
    print("⚠️  这是**草稿**，不是定稿。请人工校验：")
    print("    1. 删掉不合理 / 过于简单 / 语料无法回答的题；")
    print("    2. 核对每条 expected_paper_ids——**跨篇题尤其确认这几篇是否真的都相关**；")
    print("    3. 校验后另存为 tests/eval_set/eval_set.json（~30 条，入 git）。")
    print("    在人工校验定稿前，不要拿本草稿去跑评估指标（决策 D1）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
