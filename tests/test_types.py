from georesearcher.types import Chunk, Paper, Retrieved, StructuredNote


def test_paper_defaults():
    p = Paper(id="p1", title="A study on urban vitality")
    assert p.authors == []
    assert p.retracted is False
    assert p.year is None


def test_chunk_and_retrieved():
    c = Chunk(id="c1", paper_id="p1", text="hello", level="para")
    r = Retrieved(chunk=c, score=0.9)
    assert r.chunk.paper_id == "p1"
    assert r.score == 0.9


def test_note():
    n = StructuredNote(paper_id="p1", research_question="RQ", gap="no spatial control")
    assert n.paper_id == "p1"
    assert n.method == ""  # default empty
