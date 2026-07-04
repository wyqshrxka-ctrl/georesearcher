"""测试 RAG 生成器（rag/generator.py）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from georesearcher.capabilities.rag.generator import GeneratedAnswer, Generator
from georesearcher.capabilities.rag.retriever import RetrievalPipeline
from georesearcher.config import load_config
from georesearcher.types import Retrieved

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sample_results():
    """获取一些检索结果用于生成测试。"""
    cfg = load_config()

    # 确保有数据
    from georesearcher.capabilities.ingest.pipeline import ingest_paper
    sample = str(FIXTURES / "sample.pdf")
    try:
        ingest_paper(sample, cfg=cfg)
    except Exception:
        pass

    pipeline = RetrievalPipeline(cfg=cfg)
    return pipeline.retrieve("What is spatial autocorrelation?")


def test_generate_returns_answer(sample_results):
    generator = Generator()
    answer = generator.generate("What is spatial autocorrelation?", sample_results)

    assert isinstance(answer, GeneratedAnswer)
    assert len(answer.answer) > 50  # 回答至少 50 字符
    assert answer.model == "deepseek-chat"


def test_generate_empty_results():
    generator = Generator()
    answer = generator.generate("What is anything?", [])

    assert isinstance(answer, GeneratedAnswer)
    assert "No relevant passages" in answer.answer or "not found" in answer.answer.lower()


def test_generate_has_citations(sample_results):
    generator = Generator()
    answer = generator.generate("What is spatial autocorrelation?", sample_results)

    # 应该包含 inline citation（可能是 [1] 或 [1, 2, 3] 等格式）
    has_bracket_ref = any(
        f"[{i}]" in answer.answer or f"[{i}," in answer.answer
        for i in range(1, 6)
    )
    assert has_bracket_ref, f"Expected inline citations in answer: {answer.answer[:200]}"
