"""T6: ingest_from_search 测试（plan-m3 §3.4）。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from georesearcher.capabilities.search.base import SearchHit
from georesearcher.capabilities.search.pipeline import IngestSearchResult, ingest_from_search
from georesearcher.types import Paper

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SAMPLE_PDF = str(_FIXTURES / "sample.pdf")


def _make_hit(paper_id="doi::10.1234/test", title="Test Paper", pdf_url=None, abstract="Abstract text"):
    """Helper: build a SearchHit for testing."""
    paper = Paper(
        id=paper_id,
        title=title,
        authors=["Author One"],
        year=2024,
        venue="Test Journal",
        doi="10.1234/test",
        oa_status="gold",
    )
    return SearchHit(paper=paper, abstract=abstract, pdf_url=pdf_url, source="test")


class FakeSource:
    """Fake PaperSource for testing."""
    name = "fake"

    def __init__(self, hits=None):
        self._hits = hits or []
        self.search_calls = []

    def search(self, query, *, limit=10):
        self.search_calls.append((query, limit))
        return self._hits[:limit]


def test_ingest_from_search_full_pipeline(tmp_path):
    """Full pipeline: hit with OA pdf_url → download → ingest_paper."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    hit = _make_hit(pdf_url="http://example.com/test.pdf")
    source = FakeSource([hit])

    # Fake downloader returns the real sample PDF fixture
    def fake_download(url):
        return _SAMPLE_PDF

    # Fake embedder returns fixed embeddings
    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10 for _ in texts]

    # Fake vector_store
    class FakeVecStore:
        def add(self, chunks):
            self.last_chunks = chunks

    fake_vec = FakeVecStore()

    result = ingest_from_search(
        "test query",
        limit=5,
        source=source,
        sqlite_store=store,
        vector_store=fake_vec,
        embedder=FakeEmbedder(),
        downloader=fake_download,
    )

    assert result.query == "test query"
    assert result.total_hits == 1
    assert len(result.ingested_full) == 1
    assert hit.paper.id in result.ingested_full
    assert len(result.ingested_meta) == 0
    assert len(result.skipped_existing) == 0
    assert len(result.updated_meta) == 0
    assert len(result.failed) == 0

    # Verify paper was persisted
    stored = store.get_paper(hit.paper.id)
    assert stored is not None
    assert stored.title == hit.paper.title


def test_ingest_from_search_dedup_updates_meta(tmp_path):
    """B1: hit already in DB → cover update meta, skip re-vectorization."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    # Pre-insert a paper with old title
    old_paper = Paper(
        id="doi::10.1234/test",
        title="Old Title",
        authors=["Old Author"],
        year=2020,
        venue="Old Venue",
        doi="10.1234/test",
        oa_status="closed",
    )
    store.add_paper(old_paper)

    hit = _make_hit(paper_id="doi::10.1234/test", title="New Title", pdf_url=None)
    source = FakeSource([hit])

    result = ingest_from_search(
        "test query", limit=5, source=source, sqlite_store=store,
        vector_store=MagicMock(), embedder=MagicMock(),
        downloader=lambda u: None,
    )

    assert result.total_hits == 1
    assert len(result.updated_meta) == 1
    assert "doi::10.1234/test" in result.updated_meta
    assert len(result.ingested_full) == 0
    assert len(result.ingested_meta) == 0

    # Verify meta was updated
    stored = store.get_paper("doi::10.1234/test")
    assert stored is not None
    assert stored.title == "New Title"
    assert stored.authors == ["Author One"]
    assert stored.year == 2024
    assert stored.venue == "Test Journal"


def test_ingest_from_search_metadata_only(tmp_path):
    """A2(a): hit with no pdf_url → ingest_metadata_only."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    hit = _make_hit(pdf_url=None, abstract="This is a test abstract")
    source = FakeSource([hit])

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10 for _ in texts]

    class FakeVecStore:
        def __init__(self):
            self.chunks = []
        def add(self, chunks):
            self.chunks.extend(chunks)

    fake_vec = FakeVecStore()

    result = ingest_from_search(
        "test query", limit=5, source=source, sqlite_store=store,
        vector_store=fake_vec, embedder=FakeEmbedder(),
        downloader=lambda u: None,
    )

    assert result.total_hits == 1
    assert len(result.ingested_meta) == 1
    assert len(result.ingested_full) == 0

    # Verify paper persisted with metadata_only
    stored = store.get_paper(hit.paper.id)
    assert stored is not None
    assert stored.oa_status == "metadata_only"

    # Verify abstract chunk was added to vector store
    assert len(fake_vec.chunks) == 1
    assert fake_vec.chunks[0].level == "abstract"
    assert "test abstract" in fake_vec.chunks[0].text


def test_ingest_from_search_download_failure_falls_back_to_meta(tmp_path):
    """Download fails → falls back to ingest_metadata_only."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    hit = _make_hit(pdf_url="http://example.com/missing.pdf", abstract="Abstract fallback")
    source = FakeSource([hit])

    def failing_download(url):
        return None  # simulate download failure

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10 for _ in texts]

    fake_vec = MagicMock()

    result = ingest_from_search(
        "test query", limit=5, source=source, sqlite_store=store,
        vector_store=fake_vec, embedder=FakeEmbedder(),
        downloader=failing_download,
    )

    assert result.total_hits == 1
    assert len(result.ingested_meta) == 1
    assert len(result.ingested_full) == 0
    assert len(result.failed) == 0

    stored = store.get_paper(hit.paper.id)
    assert stored.oa_status == "metadata_only"


def test_ingest_from_search_mixed_hits(tmp_path):
    """Mixed hits: some OA full, some meta-only, some already exist."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    # Pre-insert one that already exists
    store.add_paper(Paper(
        id="doi::10.1234/existing",
        title="Existing Paper",
        authors=["Old"],
        year=2020,
        oa_status="closed",
    ))

    # Three hits: OA full, meta-only, existing
    hits = [
        _make_hit(paper_id="doi::10.1234/full", pdf_url="http://e.com/full.pdf", title="Full OA"),
        _make_hit(paper_id="doi::10.1234/meta", pdf_url=None, title="Meta Only"),
        _make_hit(paper_id="doi::10.1234/existing", pdf_url=None, title="Updated Existing"),
    ]
    source = FakeSource(hits)

    def fake_download(url):
        return _SAMPLE_PDF

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 10 for _ in texts]

    fake_vec = MagicMock()

    result = ingest_from_search(
        "test query", limit=5, source=source, sqlite_store=store,
        vector_store=fake_vec, embedder=FakeEmbedder(),
        downloader=fake_download,
    )

    assert result.total_hits == 3
    assert "doi::10.1234/full" in result.ingested_full
    assert "doi::10.1234/meta" in result.ingested_meta
    assert "doi::10.1234/existing" in result.updated_meta
    assert len(result.failed) == 0


def test_ingest_from_search_empty_results(tmp_path):
    """No hits → empty result, no errors."""
    db_path = tmp_path / "test.db"
    from georesearcher.storage.sqlite_store import SqliteStore
    store = SqliteStore(str(db_path))

    source = FakeSource([])
    result = ingest_from_search(
        "nothing", limit=5, source=source, sqlite_store=store,
        vector_store=MagicMock(), embedder=MagicMock(),
        downloader=MagicMock(),
    )

    assert result.total_hits == 0
    assert len(result.ingested_full) == 0
    assert len(result.ingested_meta) == 0
    assert len(result.failed) == 0
