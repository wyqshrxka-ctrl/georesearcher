"""Ingest 管道：解析 → 父子分块 → 向量化子块 → 写入 vector store + SQLite。

设计（design §3.1）：单一入口 ingest_paper()，协调 parser/chunker/embedder/vector_store。
"""
from __future__ import annotations

import json
from pathlib import Path

from ...config import Config, load_config
from ...models.embedding import Embedder, get_embedder
from ...storage.sqlite_store import SqliteStore, get_sqlite_store
from ...storage.vector_store import VectorStore, get_vector_store
from ...types import Chunk, Paper
from .chunker import ChunkSpec, chunk_parsed
from .parser import ParsedPdf, parse_pdf


def _paper_id_from_pdf(parsed: ParsedPdf) -> str:
    """从解析结果推导 paper_id（优先 doi > arxiv > hash）。"""
    if parsed.doi:
        return f"doi:{parsed.doi}"
    import hashlib
    return f"pdf:{hashlib.sha256(parsed.file_path.encode()).hexdigest()[:16]}"


def _to_chunk(spec: ChunkSpec, embedding: list[float]) -> Chunk:
    return Chunk(
        id=spec.id,
        paper_id=spec.paper_id,
        text=spec.text,
        level="para",
        embedding=embedding,
    )


def ingest_paper(
    file_path: str | Path,
    *,
    cfg: Config | None = None,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    sqlite_store: SqliteStore | None = None,
) -> Paper:
    """将单篇 PDF 导入知���库。

    流程：
      1. parse_pdf() — 解析 PDF 文本
      2. chunk_parsed() — 父子分块（子块入向量库，父块存 SQLite）
      3. embedder.embed() — 向量化所有子块
      4. vector_store.add() — 写入向量库（子块）
      5. sqlite_store.add_paper() — 写入元数据
      6. sqlite_store.save_parent_chunks() — 写入父块（section 全文）

    返回：
      Paper 元数据对象
    """
    cfg = cfg or load_config()
    embedder = embedder or get_embedder(cfg)
    vector_store = vector_store or get_vector_store(cfg)
    sqlite_store = sqlite_store or get_sqlite_store(cfg)

    # 1. 解析
    parsed = parse_pdf(file_path)

    # 2. 推导 paper_id
    paper_id = _paper_id_from_pdf(parsed)

    # 3. 构造 Paper 元数据
    paper = Paper(
        id=paper_id,
        title=parsed.title or Path(file_path).stem,
        authors=parsed.authors,
        year=None,
        doi=parsed.doi or None,
        pdf_path=str(file_path),
        oa_status=None,
    )

    # 4. 父子分块
    child_specs, parent_map = chunk_parsed(parsed, paper_id)
    if not child_specs:
        raise RuntimeError(f"PDF 未提取到可用文本（可能是扫描件）: {file_path}")

    # 5. 写入 SQLite（元数据 + 父块）
    sqlite_store.add_paper(paper)
    sqlite_store.save_parent_chunks(paper_id, parent_map)

    # 6. 向量化子块
    texts = [cs.text for cs in child_specs]
    embeddings = embedder.embed(texts)

    # 7. 写入向量库（子块 + section_idx metadata）
    chunks = [
        _to_chunk(cs, emb) for cs, emb in zip(child_specs, embeddings)
    ]
    vector_store.add(chunks)

    return paper
