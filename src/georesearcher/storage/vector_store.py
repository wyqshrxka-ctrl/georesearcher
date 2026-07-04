"""VectorStore 接口 + Chroma 实现 + Milvus 桩（ADR-03）。

执行者：能力层只依赖 VectorStore 协议，不直接 new 具体实现（design §12.3）。
M0 只实现 Chroma；Milvus 留 NotImplementedError 桩。
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Config, VectorStoreCfg, load_config
from ..types import Chunk, Retrieved


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> list[str]:
        """写入分块（需已含 embedding），返回 chunk id 列表。"""
        ...

    def search(
        self, query_emb: list[float], top_k: int, filters: dict | None = None
    ) -> list[Retrieved]:
        """向量检索，可选元数据过滤。"""
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """按文献 id 删除其全部分块。"""
        ...


class ChromaVectorStore:
    """默认后端：嵌入式 Chroma（零运维、demo 友好）。"""

    def __init__(self, cfg: VectorStoreCfg):
        self._cfg = cfg
        self._collection = None

    def _ensure(self):
        if self._collection is None:
            import chromadb

            Path(self._cfg.persist_dir).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self._cfg.persist_dir)
            self._collection = client.get_or_create_collection(self._cfg.collection)
        return self._collection

    def add(self, chunks: list[Chunk]) -> list[str]:
        col = self._ensure()
        ids = [c.id for c in chunks]
        col.add(
            ids=ids,
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"paper_id": c.paper_id, "level": c.level} for c in chunks],
        )
        return ids

    def search(
        self, query_emb: list[float], top_k: int, filters: dict | None = None
    ) -> list[Retrieved]:
        col = self._ensure()
        res = col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=filters or None,
        )
        out: list[Retrieved] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, text, meta, dist in zip(ids, docs, metas, dists):
            chunk = Chunk(
                id=cid,
                paper_id=(meta or {}).get("paper_id", ""),
                text=text,
                level=(meta or {}).get("level", "para"),
            )
            # Chroma 返回距离，转成相似度分数（越大越相关）
            out.append(Retrieved(chunk=chunk, score=1.0 - float(dist)))
        return out

    def delete_by_doc(self, doc_id: str) -> None:
        col = self._ensure()
        col.delete(where={"paper_id": doc_id})


class MilvusVectorStore:
    """生产后端桩（ADR-03）。M0-M7 不实现，规模化时再补。"""

    def __init__(self, cfg: VectorStoreCfg):
        self._cfg = cfg

    def add(self, chunks: list[Chunk]) -> list[str]:
        raise NotImplementedError("Milvus 后端为预留桩，当前请用 chroma")

    def search(
        self, query_emb: list[float], top_k: int, filters: dict | None = None
    ) -> list[Retrieved]:
        raise NotImplementedError("Milvus 后端为预留桩，当前请用 chroma")

    def delete_by_doc(self, doc_id: str) -> None:
        raise NotImplementedError("Milvus 后端为预留桩，当前请用 chroma")


def get_vector_store(cfg: Config | None = None) -> VectorStore:
    cfg = cfg or load_config()
    vs_cfg = cfg.storage.vector_store
    backend = vs_cfg.backend.lower()
    if backend == "chroma":
        return ChromaVectorStore(vs_cfg)
    if backend == "milvus":
        return MilvusVectorStore(vs_cfg)
    raise ValueError(f"未知 vector_store backend: {vs_cfg.backend}")
