"""capabilities/search — 文献检索源抽象（见 docs/plan-m3--20260705--v1.md §3.1）。"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from georesearcher.types import Paper


class SearchHit(BaseModel):
    """一条检索命中（search 层内部结构，尚未入库）。"""

    paper: Paper  # 已填 id（去重键）/ title / authors / year / venue / doi
    abstract: str = ""  # 已从 inverted-index 重建为纯文本（B2）
    pdf_url: str | None = None  # OA 全文链接；None = 无全文（走 A2(a)）
    source: str = "openalex"  # 来源标识


class PaperSource(Protocol):
    """文献检索源统一接口（A1：默认 OpenAlex，arXiv/S2 留桩）。"""

    name: str

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """检索文献，返回 SearchHit 列表。"""
        ...
