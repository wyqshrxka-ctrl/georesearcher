"""capabilities/search/stubs — arXiv/S2 留桩（plan-m3 A1/§3.3）。"""
from __future__ import annotations

from georesearcher.capabilities.search.base import SearchHit


class ArxivSource:
    """arXiv 检索桩（M3 不实现，plan-m3 A1）。"""

    name = "arxiv"

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        raise NotImplementedError("arXiv source is a stub (see docs/plan-m3 A1).")


class SemanticScholarSource:
    """Semantic Scholar 检索桩（M3 不实现，plan-m3 A1）。"""

    name = "semantic_scholar"

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        raise NotImplementedError("S2 source is a stub (see docs/plan-m3 A1).")
