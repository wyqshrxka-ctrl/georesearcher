"""测试 ingest 管道（ingest/pipeline.py）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from georesearcher.capabilities.ingest.pipeline import ingest_paper
from georesearcher.config import load_config
from georesearcher.storage.sqlite_store import get_sqlite_store
from georesearcher.types import Paper

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = str(FIXTURES / "sample.pdf")


def test_ingest_paper_returns_paper():
    cfg = load_config()
    paper = ingest_paper(SAMPLE_PDF, cfg=cfg)

    assert isinstance(paper, Paper)
    assert len(paper.id) > 0
    assert len(paper.title) > 0


def test_ingest_paper_persists_to_sqlite():
    cfg = load_config()
    store = get_sqlite_store(cfg)

    paper = ingest_paper(SAMPLE_PDF, cfg=cfg)

    # 验证 paper 已经存在于 SQLite 中（INSERT OR REPLACE 语义）
    fetched = store.get_paper(paper.id)
    assert fetched is not None
    assert fetched.title == paper.title
    store.close()


def test_ingest_paper_missing_file_raises():
    cfg = load_config()
    with pytest.raises(FileNotFoundError):
        ingest_paper("/nonexistent/file.pdf", cfg=cfg)
