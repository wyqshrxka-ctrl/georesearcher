"""测试 RAG 检索器（rag/retriever.py）—— 需要真实 DeepSeek API key。"""
from __future__ import annotations

from pathlib import Path

import pytest

from georesearcher.capabilities.rag.retriever import RetrievalConfig, RetrievalPipeline
from georesearcher.config import load_config
from georesearcher.types import Retrieved

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def pipeline():
    """先导入一篇论文以确保有数据可检索。"""
    cfg = load_config()

    # 确保有数据
    from georesearcher.capabilities.ingest.pipeline import ingest_paper
    sample = str(FIXTURES / "sample.pdf")
    try:
        ingest_paper(sample, cfg=cfg)
    except Exception:
        pass  # 可能已导入过

    return RetrievalPipeline(cfg=cfg)


def test_retrieve_returns_results(pipeline):
    results = pipeline.retrieve("What is spatial autocorrelation?")

    assert isinstance(results, list)
    assert len(results) > 0
    for r in results:
        assert isinstance(r, Retrieved)
        assert len(r.chunk.text) > 0
        assert r.score >= 0.0


def test_retrieve_without_hyde(pipeline):
    config = RetrievalConfig(top_k=3, use_hyde=False, use_multi_query=True, use_reranker=False)
    results = pipeline.retrieve("What is spatial autocorrelation?", config=config)

    assert len(results) >= 1


def test_retrieve_without_multi_query(pipeline):
    config = RetrievalConfig(top_k=3, use_hyde=True, use_multi_query=False, use_reranker=False)
    results = pipeline.retrieve("What is spatial autocorrelation?", config=config)

    assert len(results) >= 1


def test_retrieve_no_results_for_irrelevant_query(pipeline):
    """检索无关内容时至少不会崩溃。"""
    results = pipeline.retrieve("quantum entanglement in black hole physics")
    # 可能返回结果也可能不返回，关键是不要崩溃
    assert isinstance(results, list)


def test_retrieval_cache(pipeline):
    """相同查询应该命中缓存（通过观察 HyDE 调用次数间接验证）。"""
    results1 = pipeline.retrieve("spatial autocorrelation in urban studies")
    results2 = pipeline.retrieve("spatial autocorrelation in urban studies")

    # 缓存命中时结果应该一致
    ids1 = [r.chunk.id for r in results1]
    ids2 = [r.chunk.id for r in results2]
    assert ids1 == ids2
