"""capabilities/search/pipeline — 检索→判重→入库 编排（plan-m3 §3.4）。

不含 LangGraph（独立里程碑），纯函数编排。
"""
from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from georesearcher.capabilities.ingest.pipeline import ingest_metadata_only, ingest_paper
from georesearcher.capabilities.search.base import PaperSource, SearchHit
from georesearcher.config import Config, load_config
from georesearcher.models.embedding import Embedder, get_embedder
from georesearcher.storage.sqlite_store import SqliteStore, get_sqlite_store
from georesearcher.storage.vector_store import VectorStore, get_vector_store
from georesearcher.types import Paper


class IngestSearchResult(BaseModel):
    """ingest_from_search 的汇总结果。"""

    query: str
    total_hits: int
    ingested_full: list[str] = Field(default_factory=list)  # 走全文 ingest 的 paper_id
    ingested_meta: list[str] = Field(default_factory=list)  # 走摘要旁路的 paper_id
    skipped_existing: list[str] = Field(default_factory=list)  # 判重跳过、已存在的 paper_id
    updated_meta: list[str] = Field(default_factory=list)  # 命中已有、覆盖更新元数据的 paper_id
    failed: list[dict] = Field(default_factory=list)  # {paper_id, reason}


def _merge_paper_meta(existing: Paper, hit: SearchHit) -> Paper:
    """用 OpenAlex 元数据覆盖已有 Paper 的字段（B1：覆盖更新，不重向量化）。"""
    # Keep id unchanged (it's the same)
    existing.title = hit.paper.title
    existing.authors = hit.paper.authors
    existing.year = hit.paper.year
    existing.venue = hit.paper.venue
    if hit.paper.doi:
        existing.doi = hit.paper.doi
    if hit.paper.oa_status:
        existing.oa_status = hit.paper.oa_status
    return existing


def ingest_from_search(
    query: str,
    *,
    limit: int = 10,
    source: PaperSource | None = None,
    cfg: Config | None = None,
    sqlite_store: SqliteStore | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    downloader: Callable[[str], str | None] | None = None,
) -> IngestSearchResult:
    """检索 → 判重 → 下载/入库（全文或摘要旁路）。

    Args:
        query: 检索关键词
        limit: 最多检索数
        source: 检索源（默认 OpenAlexSource）
        cfg: 配置
        sqlite_store / vector_store / embedder: 可注入（测试）
        downloader: PDF 下载函数，签名为 (url) -> local_path | None。
                    默认使用 urllib 下载到 temp 文件。

    Returns:
        IngestSearchResult 汇总
    """
    from georesearcher.capabilities.search.openalex import OpenAlexSource

    cfg = cfg or load_config()
    sqlite_store = sqlite_store or get_sqlite_store(cfg)
    vector_store = vector_store or get_vector_store(cfg)
    embedder = embedder or get_embedder(cfg)

    if source is None:
        mailto = getattr(cfg, "search", None) and getattr(cfg.search, "mailto", None) or None
        rate = getattr(cfg, "search", None) and getattr(cfg.search, "rate_limit_per_sec", 3.0) or 3.0
        source = OpenAlexSource(mailto=mailto, rate_limit_per_sec=rate)

    if downloader is None:
        downloader = _default_downloader

    hits = source.search(query, limit=limit)
    result = IngestSearchResult(query=query, total_hits=len(hits))

    for hit in hits:
        pid = hit.paper.id

        # B1：判重
        existing = sqlite_store.get_paper(pid)
        if existing is not None:
            # 覆盖更新元数据，不重向量化
            merged = _merge_paper_meta(existing, hit)
            sqlite_store.add_paper(merged)
            result.updated_meta.append(pid)
            continue

        # 新论文：判断是否有 OA 全文
        if hit.pdf_url:
            local_path = None
            try:
                local_path = downloader(hit.pdf_url)
            except Exception:
                pass

            if local_path:
                try:
                    ingest_paper(
                        local_path,
                        paper_id=pid,
                        paper_title=hit.paper.title,
                        paper_authors=hit.paper.authors,
                        paper_doi=hit.paper.doi,
                        paper_year=hit.paper.year,
                        cfg=cfg,
                        embedder=embedder,
                        vector_store=vector_store,
                        sqlite_store=sqlite_store,
                    )
                    result.ingested_full.append(pid)
                except Exception:
                    result.failed.append({"paper_id": pid, "reason": "ingest_paper failed"})
            else:
                # 下载失败：走摘要旁路（A2(a)）
                try:
                    ingest_metadata_only(
                        hit.paper,
                        hit.abstract,
                        cfg=cfg,
                        embedder=embedder,
                        vector_store=vector_store,
                        sqlite_store=sqlite_store,
                    )
                    result.ingested_meta.append(pid)
                except Exception:
                    result.failed.append({"paper_id": pid, "reason": "ingest_metadata_only failed"})
        else:
            # 无 OA 全文：走摘要旁路（A2(a)）
            try:
                ingest_metadata_only(
                    hit.paper,
                    hit.abstract,
                    cfg=cfg,
                    embedder=embedder,
                    vector_store=vector_store,
                    sqlite_store=sqlite_store,
                )
                result.ingested_meta.append(pid)
            except Exception:
                result.failed.append({"paper_id": pid, "reason": "ingest_metadata_only failed"})

    return result


def _default_downloader(url: str) -> str | None:
    """默认 PDF 下载器：urllib → temp 文件。"""
    import tempfile
    import urllib.request

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "GeoResearcher/0.1")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if not data:
                return None
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data)
                return f.name
    except Exception:
        return None
