"""测试 T0/T1 修复：retrieve_raw()、HyDE/多查询容错、LLM judge 重排序 _rerank。

全部用 mock，不真调 DeepSeek / 不加载 bge-m3 / 不碰 Chroma。
"""
from __future__ import annotations

import pytest

from georesearcher.capabilities.rag.retriever import (
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalResult,
)
from georesearcher.types import Chunk, Retrieved


# ─── 假的依赖 ───────────────────────────────────────────────


class _FakeLLM:
    """可控的假 LLM。raise_on_call=True 时 complete() 抛异常，用于测容错。"""

    def __init__(self, response: str = "", raise_on_call: bool = False):
        self._response = response
        self._raise = raise_on_call
        self.calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        return self._response


class _FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class _FakeVectorStore:
    """返回固定命中，忽略 query_emb。"""

    def __init__(self, hits: list[Retrieved]):
        self._hits = hits

    def search(self, query_emb, top_k, filters=None) -> list[Retrieved]:
        return self._hits[:top_k]


class _FakeSqlite:
    def search_by_tags(self, tags):
        return []

    def get_parent_chunk(self, paper_id, section_idx):
        return f"parent-of-{paper_id}-{section_idx}"


def _mk_hits(n: int) -> list[Retrieved]:
    return [
        Retrieved(
            chunk=Chunk(id=f"p{i}_s0_p{i}", paper_id=f"paper{i}", text=f"passage {i}"),
            score=1.0 - i * 0.1,
        )
        for i in range(n)
    ]


def _make_pipeline(
    llm: _FakeLLM, hits: list[Retrieved]
) -> RetrievalPipeline:
    """绕过真实依赖构造 pipeline。"""
    p = RetrievalPipeline.__new__(RetrievalPipeline)
    p._llm = llm
    p._judge = llm
    p._embedder = _FakeEmbedder()
    p._vector_store = _FakeVectorStore(hits)
    p._sqlite_store = _FakeSqlite()
    # 关掉需要外部资源的功能：BM25 需拉全库、cross-encoder 需下模型
    from georesearcher.capabilities.rag.retriever import _TTLCache

    p._cache = _TTLCache(maxsize=8, ttl_seconds=1)
    p._bm25_index = None
    p._cross_encoder = None
    p._embedding_device = "cpu"
    return p


_CFG = RetrievalConfig(
    top_k=3,
    use_hyde=False,
    use_multi_query=False,
    use_tag_filter=False,
    use_bm25=False,
    use_cross_encoder=False,
    use_reranker=False,
)


# ─── T0: retrieve_raw ───────────────────────────────────────


def test_retrieve_raw_returns_list_of_retrieved():
    p = _make_pipeline(_FakeLLM(), _mk_hits(5))
    out = p.retrieve_raw("short", config=_CFG)
    assert isinstance(out, list)
    assert all(isinstance(r, Retrieved) for r in out)
    assert len(out) <= _CFG.top_k


def test_retrieve_returns_retrieval_results_with_parent():
    """retrieve() 保持原行为：返回 RetrievalResult（含父块）。"""
    p = _make_pipeline(_FakeLLM(), _mk_hits(5))
    out = p.retrieve("short", config=_CFG)
    assert all(isinstance(r, RetrievalResult) for r in out)
    assert len(out) <= _CFG.top_k
    # 父块被加载
    assert out[0].parent_text.startswith("parent-of-")
    # child_chunk 是原始命中
    assert isinstance(out[0].child_chunk, Retrieved)


def test_retrieve_and_retrieve_raw_same_ranking():
    """retrieve() 的 child_chunk 顺序应与 retrieve_raw() 一致（行为不变回归）。"""
    p = _make_pipeline(_FakeLLM(), _mk_hits(5))
    raw = p.retrieve_raw("short", config=_CFG, skip_cache=True)
    full = p.retrieve("short", config=_CFG, skip_cache=True)
    assert [r.chunk.id for r in raw] == [r.child_chunk.chunk.id for r in full]


# ─── T0: HyDE / 多查询容错 ──────────────────────────────────


def test_hyde_failure_does_not_crash():
    """HyDE LLM 抛异常时，检索仍应正常返回（退回纯 query）。"""
    cfg = RetrievalConfig(
        top_k=3, use_hyde=True, use_multi_query=False, use_tag_filter=False,
        use_bm25=False, use_cross_encoder=False, use_reranker=False,
    )
    p = _make_pipeline(_FakeLLM(raise_on_call=True), _mk_hits(5))
    out = p.retrieve_raw("some query", config=cfg)
    assert len(out) > 0  # 没崩，拿到结果


def test_multi_query_failure_falls_back_to_single():
    """多查询 LLM 抛异常时退回单查询，不崩。"""
    cfg = RetrievalConfig(
        top_k=3, use_hyde=False, use_multi_query=True, use_tag_filter=False,
        use_bm25=False, use_cross_encoder=False, use_reranker=False,
    )
    p = _make_pipeline(_FakeLLM(raise_on_call=True), _mk_hits(5))
    out = p.retrieve_raw("some query", config=cfg)
    assert len(out) > 0


def test_generate_hyde_returns_empty_on_failure():
    p = _make_pipeline(_FakeLLM(raise_on_call=True), _mk_hits(1))
    assert p._generate_hyde("q") == ""


def test_generate_multi_queries_returns_single_on_failure():
    p = _make_pipeline(_FakeLLM(raise_on_call=True), _mk_hits(1))
    assert p._generate_multi_queries("q", 3) == ["q"]


# ─── T1: LLM judge 重排序 _rerank ──────────────────────────


def test_rerank_normal_parsing_reorders():
    """judge 返回标准 passage_X: score，应据此重排序。"""
    # judge 把 passage_2 打最高分 → 它应排第一
    judge = _FakeLLM(response="passage_0: 3\npassage_1: 5\npassage_2: 9")
    p = _make_pipeline(judge, [])
    candidates = _mk_hits(3)  # ids: p0.., p1.., p2..
    out = p._rerank("q", candidates, top_k=3)
    assert out[0].chunk.id == candidates[2].chunk.id
    assert out[1].chunk.id == candidates[1].chunk.id
    assert out[2].chunk.id == candidates[0].chunk.id


def test_rerank_malformed_output_falls_back():
    """judge 返回畸形/无 passage_ 输出时，退回原顺序 candidates[:top_k]，不崩。"""
    judge = _FakeLLM(response="乱码没有分数\n???\n")
    p = _make_pipeline(judge, [])
    candidates = _mk_hits(3)
    out = p._rerank("q", candidates, top_k=2)
    assert [r.chunk.id for r in out] == [c.chunk.id for c in candidates[:2]]


def test_rerank_empty_output_falls_back():
    judge = _FakeLLM(response="")
    p = _make_pipeline(judge, [])
    candidates = _mk_hits(3)
    out = p._rerank("q", candidates, top_k=3)
    assert [r.chunk.id for r in out] == [c.chunk.id for c in candidates]
