#!/usr/bin/env python3
"""两阶段批量导入：

阶段 1（fast_import）：只写 SQLite 元数据（秒级）
阶段 2（vectorize）：向量化所有分块（可能需要数小时，可后台运行）

用法：
    # 先快速导入元数据
    uv run python scripts/batch_ingest.py fast

    # 后台向量化（可多次续跑，已有向量的 chunk 会跳过）
    uv run python scripts/batch_ingest.py vectorize

    # 检查进度
    uv run python scripts/batch_ingest.py status
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path


def fast_import(csv_path: str = "data/paper_metadata.csv"):
    """只写入 SQLite 元数据 + tags，不向量化。"""
    rows = _read_csv(csv_path)
    print(f"共 {len(rows)} 篇论文待导入")

    sys.path.insert(0, "src")
    from georesearcher.capabilities.ingest.parser import parse_pdf
    from georesearcher.capabilities.ingest.pipeline import _paper_id_from_pdf
    from georesearcher.config import load_config
    from georesearcher.storage.sqlite_store import get_sqlite_store
    from georesearcher.types import Paper

    cfg = load_config()
    store = get_sqlite_store(cfg)
    before = store.count_papers()

    success = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows):
        pdf_path = row["dest_path"]
        title = row["title"][:60]
        tags_str = row.get("tags", "").strip()

        if not Path(pdf_path).exists():
            skipped += 1
            continue

        try:
            parsed = parse_pdf(pdf_path)
            paper_id = _paper_id_from_pdf(parsed)

            # 检查是否已存在
            if store.get_paper(paper_id) is not None:
                skipped += 1
                continue

            tags = [t.strip() for t in tags_str.split(";") if t.strip()] if tags_str else []
            authors = row.get("authors", "")
            author_list = [a.strip() for a in authors.split(";") if a.strip()] if authors else parsed.authors

            paper = Paper(
                id=paper_id,
                title=parsed.title or row["title"],
                authors=author_list,
                year=_parse_year(row.get("year", "")),
                venue=row.get("venue", "") or None,
                doi=row.get("doi", "") or None,
                pdf_path=pdf_path,
                tags=tags,
            )
            store.add_paper(paper)
            success += 1

            if (i + 1) % 20 == 0:
                print(f"  已导入 {success} 篇...")
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [FAIL] {title}: {e}")

    store.close()
    print(f"\n=== 元数据导入完成 ===")
    print(f"新增: {success}, 跳过(已存在): {skipped}, 失败: {failed}")
    print(f"知识库共 {before + success} 篇文献")


_VEC_PROGRESS_FILE = "data/vectorized_papers.txt"


def _load_progress() -> set[str]:
    """加载已完成向量化的 paper_id 集合。"""
    pf = Path(_VEC_PROGRESS_FILE)
    if not pf.exists():
        return set()
    return set(line.strip() for line in pf.read_text(encoding="utf-8").splitlines() if line.strip())


def _save_progress(paper_id: str) -> None:
    """追加一条已完成的 paper_id。"""
    with open(_VEC_PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(paper_id + "\n")


def vectorize_all(csv_path: str = "data/paper_metadata.csv", batch_size: int = 10):
    """批量向量化（分批 embed 提升吞吐，可中断续跑）。

    续跑机制：检查 data/vectorized_papers.txt 跳过已完成的论文，
    不再依赖 Chroma 查询（避免每篇都做一次向量检索）。
    """
    rows = _read_csv(csv_path)

    sys.path.insert(0, "src")
    from georesearcher.capabilities.ingest.chunker import chunk_parsed
    from georesearcher.capabilities.ingest.parser import parse_pdf
    from georesearcher.capabilities.ingest.pipeline import _paper_id_from_pdf, _to_chunk
    from georesearcher.config import load_config
    from georesearcher.models.embedding import get_embedder
    from georesearcher.storage.vector_store import get_vector_store

    cfg = load_config()
    embedder = get_embedder(cfg)
    vs = get_vector_store(cfg)

    done_set = _load_progress()
    total = len(rows)
    skipped = 0

    # 收集待向��化的论文
    pending: list[dict] = []
    for row in rows:
        pdf_path = row["dest_path"]
        if not Path(pdf_path).exists():
            continue
        try:
            parsed = parse_pdf(pdf_path)
            paper_id = _paper_id_from_pdf(parsed)
        except Exception:
            continue

        if paper_id in done_set:
            skipped += 1
            continue

        pending.append(row)

    if not pending:
        print("所有论文均已向量化。")
        return

    print(f"待向量化: {len(pending)} 篇, 已跳过: {skipped}, 总计: {total}")

    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    iterator = tqdm(pending, desc="向量化", unit="篇") if _HAS_TQDM else pending
    done = 0
    failed = 0

    for row in iterator:
        pdf_path = row["dest_path"]
        try:
            parsed = parse_pdf(pdf_path)
            paper_id = _paper_id_from_pdf(parsed)
            specs, parents = chunk_parsed(parsed, paper_id)
            if not specs:
                failed += 1
                continue

            texts = [cs.text for cs in specs]
            embeddings = embedder.embed(texts)
            chunks = [_to_chunk(cs, emb) for cs, emb in zip(specs, embeddings)]
            vs.add(chunks)

            # 写入父块到 SQLite（如果尚未写入）
            from georesearcher.storage.sqlite_store import get_sqlite_store
            store = get_sqlite_store(cfg)
            if not store.get_parent_chunks_for_paper(paper_id):
                store.save_parent_chunks(paper_id, parents)
            store.close()

            _save_progress(paper_id)
            done += 1

            if not _HAS_TQDM and done % 10 == 0:
                print(f"  已向量化 {done}/{len(pending)} 篇...")
        except Exception as e:
            failed += 1
            if failed <= 5:
                title = row.get("title", pdf_path)[:60]
                print(f"\n  [FAIL] {title}: {e}")

    print(f"\n=== 向量化完成 ===")
    print(f"新增向量: {done}, 跳过: {skipped}, 失败: {failed}")


def show_status():
    """显示导入进度。"""
    sys.path.insert(0, "src")
    from georesearcher.config import load_config
    from georesearcher.storage.sqlite_store import get_sqlite_store
    from georesearcher.storage.vector_store import get_vector_store

    cfg = load_config()
    store = get_sqlite_store(cfg)
    vs = get_vector_store(cfg)

    total = store.count_papers()
    print(f"SQLite 文献数: {total}")

    # Chroma 统计
    from chromadb import PersistentClient
    client = PersistentClient(path=cfg.storage.vector_store.persist_dir)
    col = client.get_collection(cfg.storage.vector_store.collection)
    chroma_count = col.count()

    # 去重后的论文数
    results = col.get(include=["metadatas"])
    pids = set(m["paper_id"] for m in results["metadatas"] if m and "paper_id" in m)

    print(f"Chroma 向量块总数: {chroma_count}")
    print(f"Chroma 已向量化论文: {len(pids)}")

    # 进度文件
    pf = Path(_VEC_PROGRESS_FILE)
    if pf.exists():
        progress = _load_progress()
        print(f"进度文件记录: {len(progress)} 篇")

    # 父块统计
    import sqlite3
    conn = sqlite3.connect(cfg.storage.sqlite.path)
    parent_count = conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0]
    parent_papers = conn.execute("SELECT COUNT(DISTINCT paper_id) FROM parent_chunks").fetchone()[0]
    conn.close()
    print(f"SQLite 父块(section)总数: {parent_count} (分布在 {parent_papers} 篇论文中)")

    print(f"\n向量化进度: {len(pids)}/{total} ({len(pids)/total*100:.1f}%)")
    store.close()


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_year(val: str) -> int | None:
    if not val:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if cmd == "fast":
        fast_import()
    elif cmd == "vectorize":
        vectorize_all()
    elif cmd == "status":
        show_status()
    else:
        print(f"用法: python {sys.argv[0]} [fast|vectorize|status]")
