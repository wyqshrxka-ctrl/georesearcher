"""共享数据结构 —— 全项目唯一真相源（design §12.3）。

执行者：字段名与 SQLite schema（storage/sqlite_store.py）保持一致。
改字段需同时改 schema，不要只改一处。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """一篇文献的元数据。"""

    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_path: str | None = None
    oa_status: str | None = None  # open-access 状态：green/gold/closed/None/oa_full/metadata_only
    retracted: bool = False
    tags: list[str] = Field(default_factory=list)  # 分类标签，如 ["教育不平等/学校隔离", "中国"]


class Chunk(BaseModel):
    """文献分块。level 表示三级粒度（保留上下文关联）。"""

    id: str
    paper_id: str
    text: str
    level: str = "para"  # "doc" | "section" | "para" | "abstract"
    embedding: list[float] | None = None


class Retrieved(BaseModel):
    """一次检索命中的结果。"""

    chunk: Chunk
    score: float


class StructuredNote(BaseModel):
    """逐篇结构化解读笔记（interpret 模块产出）。"""

    paper_id: str
    research_question: str = ""
    method: str = ""
    contribution: str = ""
    gap: str = ""
    key_findings: str = ""
    summary: str = ""
