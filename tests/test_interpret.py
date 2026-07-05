"""T7: interpret 模块测试（plan-m3 §3.6 / C1）。"""
from __future__ import annotations

import pytest

from georesearcher.capabilities.interpret.interpret import (
    _build_prompt,
    _parse_note_json,
    interpret_paper,
)
from georesearcher.types import Paper, StructuredNote


class FakeLLM:
    """Fake LLMClient for testing — returns a preset JSON string."""

    def __init__(self, response: str):
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self._response


# ── _build_prompt ──


def test_build_prompt_contains_title_and_context():
    prompt = _build_prompt("Test Title", "Full paper context", source_note="")
    assert "Test Title" in prompt
    assert "Full paper context" in prompt


def test_build_prompt_includes_source_note():
    prompt = _build_prompt("Title", "Context", source_note="（基于摘要）")
    assert "（基于摘要）" in prompt


# ── _parse_note_json ──


def test_parse_note_json_all_fields():
    raw = json.dumps({
        "research_question": "What is the impact?",
        "method": "Regression",
        "contribution": "Novel approach",
        "gap": "Lack of data",
        "key_findings": "Significant effect",
        "summary": "This paper studies...",
    })
    note = _parse_note_json(raw, "p1")
    assert note.paper_id == "p1"
    assert note.research_question == "What is the impact?"
    assert note.method == "Regression"
    assert note.contribution == "Novel approach"
    assert note.gap == "Lack of data"
    assert note.key_findings == "Significant effect"
    assert note.summary == "This paper studies..."


def test_parse_note_json_missing_fields_default_to_empty():
    raw = json.dumps({"research_question": "Only RQ"})
    note = _parse_note_json(raw, "p1")
    assert note.research_question == "Only RQ"
    assert note.method == ""
    assert note.contribution == ""
    assert note.gap == ""
    assert note.key_findings == ""
    assert note.summary == ""


def test_parse_note_json_strips_code_fence():
    raw = '```json\n{"research_question": "RQ", "method": "M"}\n```'
    note = _parse_note_json(raw, "p1")
    assert note.research_question == "RQ"
    assert note.method == "M"


def test_parse_note_json_invalid_json_returns_empty():
    note = _parse_note_json("not valid json at all", "p1")
    assert note.paper_id == "p1"
    assert note.research_question == ""
    assert note.summary == ""


def test_parse_note_json_finds_json_in_text():
    raw = 'Some preamble text\n{"research_question": "Found it"}\nMore text'
    note = _parse_note_json(raw, "p1")
    assert note.research_question == "Found it"


def test_parse_note_json_extra_keys_ignored():
    raw = json.dumps({"research_question": "RQ", "extra_field": "should be ignored"})
    note = _parse_note_json(raw, "p1")
    assert note.research_question == "RQ"


# ── interpret_paper ──


import json  # noqa: E402 — needed for test bodies above; already imported


def test_interpret_paper_with_full_text(tmp_path):
    """interpret_paper with parent chunks → generates note → upserts."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    paper = Paper(
        id="p1",
        title="Test Paper",
        authors=["A"],
        year=2024,
        oa_status="oa_full",
    )
    store.add_paper(paper)
    store.save_parent_chunks("p1", {0: "This is the full text of the paper."})

    response = json.dumps({
        "research_question": "Test RQ",
        "method": "Test Method",
        "contribution": "Test Contribution",
        "gap": "Test Gap",
        "key_findings": "Test Findings",
        "summary": "Test Summary",
    })
    fake_llm = FakeLLM(response)

    note = interpret_paper("p1", llm=fake_llm, sqlite_store=store)

    assert note.paper_id == "p1"
    assert note.research_question == "Test RQ"
    assert note.method == "Test Method"

    # Verify prompt contains context
    assert len(fake_llm.calls) == 1
    assert "full text" in fake_llm.calls[0]


def test_interpret_paper_metadata_only(tmp_path):
    """metadata_only paper → note with source_note, no parent chunks."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    paper = Paper(
        id="p2",
        title="Meta Only Paper",
        authors=["B"],
        year=2024,
        oa_status="metadata_only",
    )
    store.add_paper(paper)
    # No parent chunks

    response = json.dumps({
        "research_question": "",
        "method": "",
        "contribution": "",
        "gap": "",
        "key_findings": "",
        "summary": "Based on abstract only.",
    })
    fake_llm = FakeLLM(response)

    note = interpret_paper("p2", llm=fake_llm, sqlite_store=store)
    assert note.paper_id == "p2"
    assert note.summary == "Based on abstract only."

    # Prompt should include source note about abstract
    assert "仅有摘要" in fake_llm.calls[0]


def test_interpret_paper_not_found(tmp_path):
    """Non-existent paper_id → ValueError."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    fake_llm = FakeLLM("{}")
    with pytest.raises(ValueError, match="Paper not found"):
        interpret_paper("nonexistent", llm=fake_llm, sqlite_store=store)


def test_interpret_paper_empty_fields_allowed(tmp_path):
    """Empty JSON response → all fields empty, no crash."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    store.add_paper(Paper(
        id="p3",
        title="Empty Fields Paper",
        authors=["C"],
        year=2024,
        oa_status="oa_full",
    ))
    store.save_parent_chunks("p3", {0: "Some text."})

    fake_llm = FakeLLM("{}")
    note = interpret_paper("p3", llm=fake_llm, sqlite_store=store)

    assert note.paper_id == "p3"
    assert note.research_question == ""
    assert note.method == ""
