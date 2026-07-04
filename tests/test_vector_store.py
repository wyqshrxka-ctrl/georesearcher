import pytest

from georesearcher.config import VectorStoreCfg
from georesearcher.storage.vector_store import (
    ChromaVectorStore,
    MilvusVectorStore,
    VectorStore,
)
from georesearcher.types import Chunk


def test_chroma_add_and_search(tmp_path):
    cfg = VectorStoreCfg(persist_dir=str(tmp_path / "chroma"), collection="test")
    vs = ChromaVectorStore(cfg)
    assert isinstance(vs, VectorStore)  # 满足协议

    chunks = [
        Chunk(id="c1", paper_id="p1", text="urban vitality and POI", embedding=[1.0, 0.0, 0.0]),
        Chunk(id="c2", paper_id="p1", text="spatial autocorrelation", embedding=[0.0, 1.0, 0.0]),
    ]
    ids = vs.add(chunks)
    assert set(ids) == {"c1", "c2"}

    res = vs.search([1.0, 0.0, 0.0], top_k=1)
    assert len(res) == 1
    assert res[0].chunk.id == "c1"

    vs.delete_by_doc("p1")
    assert vs.search([1.0, 0.0, 0.0], top_k=1) == []


def test_milvus_is_stub():
    vs = MilvusVectorStore(VectorStoreCfg(backend="milvus"))
    with pytest.raises(NotImplementedError):
        vs.add([])
