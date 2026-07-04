"""T4: 检索层指标数学正确性测试。

纯手工构造 pred/gold 列表，不碰 LLM、不碰真实检索。
每个断言的期望值都在注释里手算过。
"""

import math

from georesearcher.capabilities.evaluation.retrieval_metrics import (
    aggregate_retrieval,
    context_precision,
    context_recall,
    dedup_paper_ranking,
    hit_rate,
    mrr,
    ndcg,
)
from georesearcher.capabilities.evaluation.schemas import RetrievalMetrics
from georesearcher.types import Chunk, Retrieved


def _hit(paper_id: str, score: float = 1.0) -> Retrieved:
    return Retrieved(
        chunk=Chunk(id=f"{paper_id}::c", paper_id=paper_id, text="t", level="child"),
        score=score,
    )


# ---------- dedup_paper_ranking ----------

def test_dedup_keeps_first_occurrence_order():
    hits = [_hit("p1"), _hit("p2"), _hit("p1"), _hit("p3"), _hit("p2")]
    assert dedup_paper_ranking(hits) == ["p1", "p2", "p3"]


def test_dedup_empty():
    assert dedup_paper_ranking([]) == []


# ---------- hit_rate ----------

def test_hit_rate_hit():
    # gold p2 在 top3 内 → 1.0
    assert hit_rate(["p1", "p2", "p9"], ["p2"], k=3) == 1.0


def test_hit_rate_miss():
    assert hit_rate(["p1", "p9"], ["p2"], k=3) == 0.0


def test_hit_rate_respects_k():
    # gold 在第 3 位，但 k=2 → 未命中
    assert hit_rate(["p1", "p9", "p2"], ["p2"], k=2) == 0.0


# ---------- mrr ----------

def test_mrr_gold_at_rank_2():
    # gold 第 2 位 → 1/2 = 0.5
    assert mrr(["p1", "p2", "p3"], ["p2"], k=5) == 0.5


def test_mrr_gold_at_rank_1():
    assert mrr(["p2", "p1"], ["p2"], k=5) == 1.0


def test_mrr_no_hit():
    assert mrr(["p1", "p9"], ["p2"], k=5) == 0.0


def test_mrr_first_gold_wins():
    # 两个 gold，第一个命中在第 2 位 → 0.5
    assert mrr(["p1", "p2", "p3"], ["p2", "p3"], k=5) == 0.5


# ---------- ndcg ----------

def test_ndcg_perfect_single_gold_rank1():
    # gold 第 1 位：DCG = 1/log2(2)=1；IDCG=1 → 1.0
    assert ndcg(["p2", "p1"], ["p2"], k=5) == 1.0


def test_ndcg_single_gold_rank2():
    # DCG = 1/log2(3); IDCG = 1/log2(2)=1 → 1/log2(3)
    expected = 1.0 / math.log2(3)
    assert math.isclose(ndcg(["p1", "p2"], ["p2"], k=5), expected)


def test_ndcg_two_gold_perfect():
    # gold p1,p2 都在前两位：DCG = 1/log2(2)+1/log2(3)
    # IDCG 同（2 个相关排前两位）→ 1.0
    assert math.isclose(ndcg(["p1", "p2", "p9"], ["p1", "p2"], k=5), 1.0)


def test_ndcg_empty_gold():
    assert ndcg(["p1"], [], k=5) == 0.0


def test_ndcg_no_hit():
    assert ndcg(["p1", "p9"], ["p2"], k=5) == 0.0


# ---------- context_precision ----------

def test_context_precision_half():
    # top4 中 2 个是 gold → 2/4 = 0.5
    assert context_precision(["p1", "p2", "p8", "p9"], ["p1", "p2"], k=4) == 0.5


def test_context_precision_k_larger_than_list():
    # 只检回 2 个，1 个 gold → 1/2 = 0.5（分母是实际检回数）
    assert context_precision(["p1", "p9"], ["p1"], k=10) == 0.5


def test_context_precision_empty_pred():
    assert context_precision([], ["p1"], k=5) == 0.0


# ---------- context_recall ----------

def test_context_recall_full():
    # 2 个 gold 都在 top_k → 1.0
    assert context_recall(["p1", "p2", "p9"], ["p1", "p2"], k=5) == 1.0


def test_context_recall_partial():
    # 2 gold，只检回 1 个 → 0.5
    assert context_recall(["p1", "p9"], ["p1", "p2"], k=5) == 0.5


def test_context_recall_respects_k():
    # gold p2 在第 3 位，k=2 → recall 0
    assert context_recall(["p1", "p9", "p2"], ["p2"], k=2) == 0.0


def test_context_recall_empty_gold():
    assert context_recall(["p1"], [], k=5) == 0.0


# ---------- aggregate_retrieval ----------

def test_aggregate_averages():
    per_case = [
        {"hit_rate": 1.0, "mrr": 1.0, "ndcg": 1.0, "context_precision": 0.5, "context_recall": 1.0},
        {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "context_precision": 0.0, "context_recall": 0.0},
    ]
    m = aggregate_retrieval(per_case, k=5)
    assert isinstance(m, RetrievalMetrics)
    assert m.hit_rate == 0.5
    assert m.mrr == 0.5
    assert m.context_precision == 0.25
    assert m.k == 5


def test_aggregate_empty():
    m = aggregate_retrieval([], k=7)
    assert m.hit_rate == 0.0 and m.mrr == 0.0 and m.k == 7
