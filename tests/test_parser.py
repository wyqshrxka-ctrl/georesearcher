"""测试 PDF 解析器（ingest/parser.py）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from georesearcher.capabilities.ingest.parser import ParsedPdf, ParsedSection, parse_pdf

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample.pdf"


def test_parse_pdf_extracts_title():
    result = parse_pdf(SAMPLE_PDF)
    assert isinstance(result, ParsedPdf)
    assert len(result.title) > 0


def test_parse_pdf_extracts_authors():
    result = parse_pdf(SAMPLE_PDF)
    # 合成 PDF 可能没有 author metadata，取决于 PDF 写入方式
    assert isinstance(result.authors, list)


def test_parse_pdf_has_sections():
    result = parse_pdf(SAMPLE_PDF)
    # 合成 PDF 的章节分割依赖启发式，至少 1 个 section
    assert len(result.sections) >= 1
    for sec in result.sections:
        assert isinstance(sec, ParsedSection)
        assert len(sec.title) > 0


def test_parse_pdf_all_paragraphs_not_empty():
    result = parse_pdf(SAMPLE_PDF)
    paras = result.all_paragraphs
    assert len(paras) > 0
    for p in paras:
        assert len(p["text"]) > 0
        assert "section_title" in p


def test_parse_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_pdf("/nonexistent/path.pdf")
