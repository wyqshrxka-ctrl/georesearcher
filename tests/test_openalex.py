"""T2: OpenAlex 实现测试（见 docs/plan-m3--20260705--v1.md）。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from georesearcher.capabilities.search.openalex import (
    OpenAlexSource,
    _make_paper_id,
    _parse_work,
    _reconstruct_abstract,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_works():
    with open(_FIXTURES / "openalex_works.json") as f:
        return json.load(f)["results"]


# ── _make_paper_id ──


def test_make_paper_id_with_doi():
    result = _make_paper_id("https://doi.org/10.1234/test", None, "A Title")
    assert result == "doi::10.1234/test"


def test_make_paper_id_with_doi_no_prefix():
    result = _make_paper_id("10.1234/test", None, "A Title")
    assert result == "doi::10.1234/test"


def test_make_paper_id_with_dx_doi_prefix():
    result = _make_paper_id("https://dx.doi.org/10.1234/test", None, "A Title")
    assert result == "doi::10.1234/test"


def test_make_paper_id_with_trailing_slash():
    result = _make_paper_id("https://doi.org/10.1234/test/", None, "A Title")
    assert result == "doi::10.1234/test"


def test_make_paper_id_with_empty_doi():
    result = _make_paper_id("", None, "A Title")
    assert result != "" and not result.startswith("doi::")


def test_make_paper_id_fallback_to_openalex():
    result = _make_paper_id(None, "https://openalex.org/W123", "A Title")
    assert result == "https://openalex.org/W123"


def test_make_paper_id_fallback_to_title_hash():
    result = _make_paper_id(None, None, "Unique Title String")
    assert result.startswith("title::")
    assert len(result) == len("title::") + 16  # sha256[:16]


def test_make_paper_id_same_title_same_hash():
    a = _make_paper_id(None, None, "Same Title")
    b = _make_paper_id(None, None, "Same Title")
    assert a == b


# ── _reconstruct_abstract ──


def test_reconstruct_abstract():
    inverted = {"Urban": [0], "vitality": [1], "measurement": [2]}
    result = _reconstruct_abstract(inverted)
    assert result == "Urban vitality measurement"


def test_reconstruct_abstract_empty():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


def test_reconstruct_abstract_with_gaps():
    inverted = {"Hello": [0], "world": [5]}
    result = _reconstruct_abstract(inverted)
    assert result == "Hello world"


# ── _parse_work ──


def test_parse_work_full():
    works = _load_works()
    work = works[0]  # Urban Vitality
    hit = _parse_work(work)
    assert hit is not None
    assert hit.paper.title == "Measuring Urban Vitality with POI Data: A Spatial Approach"
    assert hit.paper.authors == ["Jane Smith", "Li Wei"]
    assert hit.paper.year == 2023
    assert hit.paper.venue == "Urban Studies Journal"
    assert hit.paper.doi == "https://doi.org/10.1234/urban.vitality.2023"
    assert hit.paper.oa_status == "gold"
    assert hit.pdf_url == "https://example.com/oa/urban_vitality.pdf"
    assert hit.abstract == "Urban vitality measurement using POI data reveals significant spatial patterns"
    assert hit.source == "openalex"


def test_parse_work_no_oa():
    works = _load_works()
    work = works[1]  # School Segregation (closed)
    hit = _parse_work(work)
    assert hit is not None
    assert hit.paper.oa_status == "closed"
    assert hit.pdf_url is None
    assert hit.abstract == "This study examines school segregation in Chinese cities"


def test_parse_work_no_title_returns_none():
    assert _parse_work({"title": ""}) is None
    assert _parse_work({}) is None


def test_parse_work_missing_fields():
    """Missing optional fields should not crash."""
    work = {
        "title": "Minimal Paper",
        "id": "https://openalex.org/W99",
        "publication_date": "",
    }
    hit = _parse_work(work)
    assert hit is not None
    assert hit.paper.authors == []
    assert hit.paper.year is None
    assert hit.paper.venue is None
    assert hit.paper.doi is None
    assert hit.abstract == ""


# ── OpenAlexSource.search ──


def _fake_urlopen(body_dict):
    """Return a mock that responds with the given JSON body."""

    class _FakeResp:
        def __init__(self, data):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(body_dict).encode("utf-8")

        def decode(self, enc):
            return json.dumps(body_dict)

    return lambda req, timeout=None: _FakeResp(body_dict)


def test_openalex_search_returns_hits():
    data = _load_works()
    body = {"results": data}
    source = OpenAlexSource(mailto="test@example.com", _urlopen=_fake_urlopen(body))
    hits = source.search("urban vitality", limit=5)
    assert len(hits) == 2
    assert hits[0].paper.title.startswith("Measuring Urban Vitality")
    assert hits[1].paper.title.startswith("School Segregation")


def test_openalex_search_respects_limit():
    data = _load_works()
    body = {"results": data}
    source = OpenAlexSource(_urlopen=_fake_urlopen(body))
    hits = source.search("test", limit=1)
    assert len(hits) == 1


def test_openalex_search_empty_results():
    body = {"results": []}
    source = OpenAlexSource(_urlopen=_fake_urlopen(body))
    hits = source.search("nothing", limit=5)
    assert hits == []


def test_openalex_search_skips_no_title():
    body = {"results": [{"title": ""}, {"title": "Valid Paper"}]}
    source = OpenAlexSource(_urlopen=_fake_urlopen(body))
    hits = source.search("test", limit=5)
    assert len(hits) == 1
    assert hits[0].paper.title == "Valid Paper"


def test_rate_limit_does_not_block_first_call():
    """Rate limiter should allow first call immediately."""
    source = OpenAlexSource(rate_limit_per_sec=100, _urlopen=_fake_urlopen({"results": []}))
    hits = source.search("test", limit=1)
    assert hits == []
