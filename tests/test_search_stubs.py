"""T3: 留桩测试（plan-m3 §3.3）。"""
from __future__ import annotations

import pytest

from georesearcher.capabilities.search.stubs import ArxivSource, SemanticScholarSource


def test_arxiv_stub_raises():
    source = ArxivSource()
    with pytest.raises(NotImplementedError, match="arXiv.*stub"):
        source.search("test")


def test_semantic_scholar_stub_raises():
    source = SemanticScholarSource()
    with pytest.raises(NotImplementedError, match="S2.*stub"):
        source.search("test")


def test_arxiv_stub_has_name():
    assert ArxivSource.name == "arxiv"


def test_semantic_scholar_stub_has_name():
    assert SemanticScholarSource.name == "semantic_scholar"
