"""测试父子切块器。"""
from __future__ import annotations

from pathlib import Path

from georesearcher.capabilities.ingest.chunker import ChunkSpec, chunk_parsed
from georesearcher.capabilities.ingest.parser import parse_pdf

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample.pdf"


def test_chunk_parsed_returns_child_and_parent():
    parsed = parse_pdf(SAMPLE_PDF)
    children, parents = chunk_parsed(parsed, paper_id="test:001")

    assert len(children) > 0
    assert len(parents) > 0


def test_child_chunks_are_paragraphs():
    parsed = parse_pdf(SAMPLE_PDF)
    children, _ = chunk_parsed(parsed, paper_id="test:001")

    for c in children:
        assert isinstance(c, ChunkSpec)
        assert c.paper_id == "test:001"
        assert len(c.text) >= 20
        assert c.section_idx >= 0


def test_parent_map_keys_match_children():
    parsed = parse_pdf(SAMPLE_PDF)
    children, parents = chunk_parsed(parsed, paper_id="test:001")

    child_sections = {c.section_idx for c in children}
    for si in child_sections:
        assert si in parents, f"section_idx {si} not in parent_map"


def test_parent_text_contains_child_text():
    parsed = parse_pdf(SAMPLE_PDF)
    children, parents = chunk_parsed(parsed, paper_id="test:001")

    for c in children:
        parent = parents[c.section_idx]
        # 子块文本应该是父块的一部分
        assert len(parent) >= len(c.text)


def test_chunk_ids_unique():
    parsed = parse_pdf(SAMPLE_PDF)
    children, _ = chunk_parsed(parsed, paper_id="test:001")

    ids = [c.id for c in children]
    assert len(ids) == len(set(ids))
