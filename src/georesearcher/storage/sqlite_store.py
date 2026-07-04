"""SQLite 结构化存储：papers / notes / citations / parent_chunks（design §4.1、ADR-04）。

parent_chunks 表 = 父子切块的父块（section 全文）存储。
citations 表 = 未来导 Neo4j 知识图谱的钩子。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import Config, load_config
from ..types import Paper, StructuredNote

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT,            -- JSON 数组
    year INTEGER,
    venue TEXT,
    doi TEXT,
    arxiv_id TEXT,
    pdf_path TEXT,
    oa_status TEXT,
    retracted INTEGER DEFAULT 0,
    tags TEXT,               -- JSON 数组，分类标签
    added_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS notes (
    paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    research_question TEXT,
    method TEXT,
    contribution TEXT,
    gap TEXT,
    key_findings TEXT,
    summary TEXT
);

-- 父块：section 全文，子块命中后通过 paper_id + section_idx 查找
CREATE TABLE IF NOT EXISTS parent_chunks (
    paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
    section_idx INTEGER NOT NULL,
    section_title TEXT,
    full_text TEXT NOT NULL,
    PRIMARY KEY (paper_id, section_idx)
);

-- 引用关系边表：未来可导出为 (Paper)-[:CITES]->(Paper)
CREATE TABLE IF NOT EXISTS citations (
    src_paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
    dst_paper_id TEXT REFERENCES papers(id) ON DELETE CASCADE,
    context TEXT,
    PRIMARY KEY (src_paper_id, dst_paper_id)
);
"""


class SqliteStore:
    def __init__(self, db_path: str):
        self._path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ─── Papers ──────────────────────────────────────

    def add_paper(self, paper: Paper) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO papers
               (id, title, authors, year, venue, doi, arxiv_id, pdf_path, oa_status, retracted, tags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                paper.id,
                paper.title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.year,
                paper.venue,
                paper.doi,
                paper.arxiv_id,
                paper.pdf_path,
                paper.oa_status,
                int(paper.retracted),
                json.dumps(paper.tags, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        if row is None:
            return None
        return Paper(
            id=row["id"],
            title=row["title"],
            authors=json.loads(row["authors"]) if row["authors"] else [],
            year=row["year"],
            venue=row["venue"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            pdf_path=row["pdf_path"],
            oa_status=row["oa_status"],
            retracted=bool(row["retracted"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
        )

    def count_papers(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def search_by_tags(self, tags: list[str]) -> list[str]:
        """按标签查论文 ID 列表（AND 逻辑）。"""
        if not tags:
            rows = self._conn.execute("SELECT id FROM papers").fetchall()
            return [r["id"] for r in rows]

        conditions = " AND ".join(["tags LIKE ?" for _ in tags])
        params = [f"%{t}%" for t in tags]
        rows = self._conn.execute(
            f"SELECT id FROM papers WHERE {conditions}", params
        ).fetchall()
        return [r["id"] for r in rows]

    def list_papers_with_tags(self, limit: int = 100) -> list[dict]:
        """列出所有文献（含标签）。"""
        rows = self._conn.execute(
            "SELECT id, title, tags FROM papers ORDER BY added_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
            }
            for r in rows
        ]

    # ─── Parent Chunks ───────────────────────────────

    def save_parent_chunks(self, paper_id: str, parent_map: dict[int, str]) -> None:
        """写入父块（section 全文）。"""
        for section_idx, full_text in parent_map.items():
            self._conn.execute(
                """INSERT OR REPLACE INTO parent_chunks
                   (paper_id, section_idx, section_title, full_text)
                   VALUES (?,?,?,?)""",
                (paper_id, section_idx, f"section_{section_idx}", full_text),
            )
        self._conn.commit()

    def get_parent_chunk(self, paper_id: str, section_idx: int) -> str | None:
        """读取单个父块（section 全文）。"""
        row = self._conn.execute(
            "SELECT full_text FROM parent_chunks WHERE paper_id = ? AND section_idx = ?",
            (paper_id, section_idx),
        ).fetchone()
        return row["full_text"] if row else None

    def get_parent_chunks_for_paper(self, paper_id: str) -> dict[int, str]:
        """读取一篇论文的所有父块。"""
        rows = self._conn.execute(
            "SELECT section_idx, full_text FROM parent_chunks WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
        return {r["section_idx"]: r["full_text"] for r in rows}

    # ─── Notes / Citations ───────────────────────────

    def upsert_note(self, note: StructuredNote) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO notes
               (paper_id, research_question, method, contribution, gap, key_findings, summary)
               VALUES (?,?,?,?,?,?,?)""",
            (
                note.paper_id,
                note.research_question,
                note.method,
                note.contribution,
                note.gap,
                note.key_findings,
                note.summary,
            ),
        )
        self._conn.commit()

    def add_citation(self, src_id: str, dst_id: str, context: str = "") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO citations (src_paper_id, dst_paper_id, context) VALUES (?,?,?)",
            (src_id, dst_id, context),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def get_sqlite_store(cfg: Config | None = None) -> SqliteStore:
    cfg = cfg or load_config()
    return SqliteStore(cfg.storage.sqlite.path)
