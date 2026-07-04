"""检索层指标（M2 / T4）——纯计算，不碰 LLM、不碰真实检索。

全部为 **paper 级、二元相关性** 指标（决策 D2）：
一个 GT paper 只要有任意 chunk 被检回，就算命中。

输入约定：
- ``retrieve_raw()`` 返回 ``list[Retrieved]``（按排名，含重复 paper 的多个 chunk）。
- 先用 ``dedup_paper_ranking`` 折叠成去重后的 paper_id 排序列表，再喂给各指标。
- 各指标只在 top_k 截断范围内计算。
"""

from __future__ import annotations

import math

from ...types import Retrieved
from .schemas import RetrievalMetrics


def dedup_paper_ranking(retrieved: list[Retrieved]) -> list[str]:
    """把 chunk 命中折叠成 paper_id 排序列表（首次出现的排名代表该 paper）。"""
    seen: set[str] = set()
    ranking: list[str] = []
    for r in retrieved:
        pid = r.chunk.paper_id
        if pid not in seen:
            seen.add(pid)
            ranking.append(pid)
    return ranking


def hit_rate(pred_papers: list[str], gold_papers: list[str], k: int) -> float:
    """top_k paper 里是否至少命中一个 gold → 1.0 / 0.0。"""
    gold = set(gold_papers)
    top = pred_papers[:k]
    return 1.0 if any(p in gold for p in top) else 0.0


def mrr(pred_papers: list[str], gold_papers: list[str], k: int) -> float:
    """第一个命中 gold 的 paper 的排名倒数（1/rank）；无命中记 0。"""
    gold = set(gold_papers)
    for idx, pid in enumerate(pred_papers[:k], start=1):
        if pid in gold:
            return 1.0 / idx
    return 0.0


def ndcg(pred_papers: list[str], gold_papers: list[str], k: int) -> float:
    """NDCG@k，paper 级二元相关性（命中 relevance=1）。

    DCG = Σ rel_i / log2(i+1)（i 从 1 起）；IDCG 为理想排序（gold 全排前）的 DCG。
    """
    gold = set(gold_papers)
    if not gold:
        return 0.0

    dcg = 0.0
    for idx, pid in enumerate(pred_papers[:k], start=1):
        if pid in gold:
            dcg += 1.0 / math.log2(idx + 1)

    # IDCG：理想情况下，min(len(gold), k) 个相关文档排在最前
    n_ideal = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    return dcg / idcg if idcg > 0 else 0.0


def context_precision(pred_papers: list[str], gold_papers: list[str], k: int) -> float:
    """top_k 去重 paper 中属于 gold 的比例。分母为实际检回数（≤ k）。"""
    gold = set(gold_papers)
    top = pred_papers[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if p in gold) / len(top)


def context_recall(pred_papers: list[str], gold_papers: list[str], k: int) -> float:
    """gold paper 中被检回（在 top_k 内）的比例。"""
    gold = set(gold_papers)
    if not gold:
        return 0.0
    top = set(pred_papers[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def aggregate_retrieval(per_case: list[dict], k: int) -> RetrievalMetrics:
    """对所有 case 的单条指标取平均，组装 RetrievalMetrics。

    per_case 每项需含键：hit_rate, mrr, ndcg, context_precision, context_recall。
    空列表时返回全 0 指标。
    """
    if not per_case:
        return RetrievalMetrics(
            hit_rate=0.0, mrr=0.0, ndcg=0.0,
            context_precision=0.0, context_recall=0.0, k=k,
        )

    n = len(per_case)

    def avg(key: str) -> float:
        return sum(c[key] for c in per_case) / n

    return RetrievalMetrics(
        hit_rate=avg("hit_rate"),
        mrr=avg("mrr"),
        ndcg=avg("ndcg"),
        context_precision=avg("context_precision"),
        context_recall=avg("context_recall"),
        k=k,
    )
