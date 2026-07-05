"""capabilities/search/openalex — OpenAlex 检索实现（plan-m3 §3.2）。"""
from __future__ import annotations

import json
import hashlib
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable

from georesearcher.capabilities.search.base import SearchHit
from georesearcher.types import Paper


# ── 纯函数（方便单测，无网络依赖） ──


def _make_paper_id(doi: str | None, openalex_id: str | None, title: str) -> str:
    """构造统一 paper_id（B1：DOI → OpenAlex id → title 哈希）。

    DOI 规范化：小写、去 https://doi.org/ 前缀。
    """
    if doi:
        normalized = doi.lower().strip()
        normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
        normalized = normalized.strip("/")
        if normalized:
            return f"doi::{normalized}"
    if openalex_id and openalex_id.startswith("https://openalex.org/"):
        return openalex_id.strip()
    # fallback: title hash
    digest = hashlib.sha256(title.strip().encode()).hexdigest()[:16]
    return f"title::{digest}"


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex inverted-index → 纯文本（B2）。"""
    if not inverted_index:
        return ""
    # Build list of (position, word) pairs
    entries: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            entries.append((pos, word))
    entries.sort(key=lambda x: x[0])
    return " ".join(w for _, w in entries)


def _parse_work(work: dict) -> SearchHit | None:
    """单条 OpenAlex work JSON → SearchHit（或 None，若不可用）。"""
    title = (work.get("title") or "").strip()
    if not title:
        return None

    # authors
    authorships = work.get("authorships") or []
    authors: list[str] = []
    for a in authorships:
        auth_name = (a.get("author", {}) or {}).get("display_name")
        if auth_name:
            authors.append(auth_name.strip())

    # doi
    doi = work.get("doi") or None

    # year
    pub_date = work.get("publication_date") or ""
    year: int | None = None
    try:
        year = int(pub_date[:4]) if pub_date else None
    except (ValueError, TypeError):
        year = None

    # venue
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    venue = source.get("display_name") or None

    # openalex id
    openalex_id = work.get("id") or None

    # paper id (B1)
    paper_id = _make_paper_id(doi, openalex_id, title)

    # OA status & pdf_url (A2)
    oa = work.get("open_access") or {}
    oa_status = oa.get("oa_status") or "closed"
    best_oa = work.get("best_oa_location") or {}
    pdf_url = (best_oa.get("pdf_url") or
               oa.get("any_repository_has_fulltext") and None or
               None)
    # Actually resolve pdf_url from best_oa_location
    if not pdf_url:
        pdf_url = best_oa.get("pdf_url") or None

    # abstract (B2: inverted-index → plain text)
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index") or None)

    paper = Paper(
        id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=None,
        pdf_path=None,
        oa_status=oa_status,
        retracted=False,
        tags=[],
    )

    return SearchHit(paper=paper, abstract=abstract, pdf_url=pdf_url, source="openalex")


# ── OpenAlex 实现 ──


class OpenAlexSource:
    """OpenAlex 检索源（A1/A2/B1/B2 落地）。"""

    name = "openalex"
    _BASE = "https://api.openalex.org"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        rate_limit_per_sec: float = 3.0,
        timeout: float = 20.0,
        max_retries: int = 2,
        _urlopen: Callable | None = None,  # 测试注入
    ):
        self._mailto = mailto
        self._rate = rate_limit_per_sec
        self._min_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0
        self._timeout = timeout
        self._max_retries = max_retries
        self._urlopen = _urlopen or urllib.request.urlopen
        self._last_call: float = 0.0
        # 已知问题：部分网络环境下 Python urllib 的 TLS 指纹可能被
        # Cloudflare 识别为自动化流量并阻断（见 plan-m3 §11.3-11.4）。
        # 如遇 SSL handshake timeout，可尝试换网络或降低请求频率。

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """调 OpenAlex /works?search=... 返回 SearchHit 列表。"""
        params = {
            "search": query,
            "per_page": min(limit, 200),
        }
        # Add mailto for polite pool (A2)
        if self._mailto:
            params["mailto"] = self._mailto

        url = f"{self._BASE}/works?{urllib.parse.urlencode(params)}"
        body = self._get_json(url)
        results = body.get("results") or []
        hits: list[SearchHit] = []
        for work in results:
            hit = _parse_work(work)
            if hit is not None:
                hits.append(hit)
                if len(hits) >= limit:
                    break
        return hits

    def _get_json(self, url: str) -> dict:
        """GET 请求 + 限速 + 重试。"""
        self._rate_limit()

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                req = urllib.request.Request(url)
                req.add_header("Accept", "application/json")
                req.add_header("User-Agent", f"GeoResearcher/0.1 (mailto:{self._mailto or 'none'})")
                with self._urlopen(req, timeout=self._timeout) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise RuntimeError(
            f"OpenAlex request failed after {self._max_retries + 1} attempts: {last_error}"
        )

    def _rate_limit(self):
        """客户端限速（A2）。"""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()
