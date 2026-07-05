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
    paper_id: str | None = None,  # B1：外部传入的去重 id，传了就优先用
    paper_title: str | None = None,
    paper_authors: list[str] | None = None,
    paper_doi: str | None = None,
    paper_year: int | None = None,
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

    # 2. 推导 paper_id（B1：外部传入优先）
    if paper_id is not None:
        pid = paper_id
    else:
        pid = _paper_id_from_pdf(parsed)

    # 3. 构造 Paper 元数据（优先用外部传入的覆盖项）
    paper = Paper(
        id=pid,
        title=paper_title or parsed.title or Path(file_path).stem,
        authors=paper_authors if paper_authors is not None else parsed.authors,
        year=paper_year,
        doi=paper_doi or parsed.doi or None,
        pdf_path=str(file_path),
        oa_status="oa_full",
    )

    # 4. 父子分块
    child_specs, parent_map = chunk_parsed(parsed, pid)
    if not child_specs:
        raise RuntimeError(f"PDF 未提取到可用文本（可能是扫描件）: {file_path}")

    # 5. 写入 SQLite（元数据 + 父块）
    sqlite_store.add_paper(paper)
    sqlite_store.save_parent_chunks(pid, parent_map)

    # 6. 向量化子块
    texts = [cs.text for cs in child_specs]
    embeddings = embedder.embed(texts)

    # 7. 写入向量库（子块 + section_idx metadata）
    chunks = [
        _to_chunk(cs, emb) for cs, emb in zip(child_specs, embeddings)
    ]
    vector_store.add(chunks)

    return paper


def ingest_metadata_only(
    paper: Paper,
    abstract: str = "",
    *,
    cfg: Config | None = None,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    sqlite_store: SqliteStore | None = None,
) -> Paper:
    """无 OA 全文的论文入库（plan-m3 A2(a) / B2）。

    - paper.oa_status = "metadata_only"
    - sqlite_store.add_paper(paper)
    - 若 abstract 非空：构造 level="abstract" 的 Chunk → embed → vector_store.add
    - 不写 parent_chunks（摘要无父子结构）
    """
    cfg = cfg or load_config()
    embedder = embedder or get_embedder(cfg)
    vector_store = vector_store or get_vector_store(cfg)
    sqlite_store = sqlite_store or get_sqlite_store(cfg)

    # 标记 oa_status
    paper.oa_status = "metadata_only"

    # 写入元数据
    sqlite_store.add_paper(paper)

    # 若有摘要：构造单块入向量库
    if abstract.strip():
        chunk = Chunk(
            id=f"{paper.id}::abstract",
            paper_id=paper.id,
            text=abstract.strip(),
            level="abstract",
        )
        embeddings = embedder.embed([chunk.text])
        chunk.embedding = embeddings[0]
        vector_store.add([chunk])

    return paper
