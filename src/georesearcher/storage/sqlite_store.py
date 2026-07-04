"""SQLite 结构化存储：papers / notes / citations（design §4.1、ADR-04）。

citations 表 = 未来导 Neo4j 知识图谱的钩子（现在只作边表用）。
schema 字段与 types.py 保持一致。
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

    def add_paper(self, paper: Paper) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO papers
               (id, title, authors, year, venue, doi, arxiv_id, pdf_path, oa_status, retracted)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
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
        )

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

    def count_papers(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def get_sqlite_store(cfg: Config | None = None) -> SqliteStore:
    cfg = cfg or load_config()
    return SqliteStore(cfg.storage.sqlite.path)
