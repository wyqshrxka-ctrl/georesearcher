from georesearcher.storage.sqlite_store import SqliteStore
from georesearcher.types import Paper, StructuredNote


def test_paper_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    store = SqliteStore(str(db))
    assert store.count_papers() == 0

    p = Paper(id="p1", title="Urban vitality", authors=["Li", "Wang"], year=2024, venue="Nature Cities")
    store.add_paper(p)
    assert store.count_papers() == 1

    got = store.get_paper("p1")
    assert got is not None
    assert got.title == "Urban vitality"
    assert got.authors == ["Li", "Wang"]
    assert got.year == 2024

    store.upsert_note(StructuredNote(paper_id="p1", gap="lacks spatial autocorrelation"))
    store.add_citation("p1", "p1", context="self (test)")
    store.close()


def test_get_missing_paper(tmp_path):
    store = SqliteStore(str(tmp_path / "t.db"))
    assert store.get_paper("nope") is None
    store.close()
