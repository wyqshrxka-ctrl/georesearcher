#!/usr/bin/env python3
"""批量扫描 papers/ 目录，提取元数据，去重，复制到 data/pdfs/。

用法：
    uv run python scripts/scan_papers.py /Users/wwwyyyqqq/Documents/papers

产出：
    data/papers/ — 去重后的 PDF 副本（按 doi 命名）
    data/paper_metadata.csv — 元数据表（title, authors, doi, tags 等）
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path


def extract_metadata(pdf_path: Path) -> dict | None:
    """提取单篇 PDF 的元数据。"""
    try:
        import fitz
    except ImportError:
        print("需要安装 PyMuPDF: uv pip install pymupdf")
        sys.exit(1)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        print(f"  [WARN] 无法打开 {pdf_path.name}: {e}")
        return None

    meta = doc.metadata or {}
    title = meta.get("title", "").strip()
    author_str = meta.get("author", "")
    doi = ""
    year = None
    venue = ""

    # 从前几页文本提取 DOI、年份、期刊
    text = ""
    for page_no in range(min(3, len(doc))):
        text += doc[page_no].get_text()

    # DOI
    doi_match = re.search(r"10\.\d{4,}/[^\s]+", text)
    if doi_match:
        doi = doi_match.group(0).rstrip(".")

    # 年份
    year_match = re.search(r"(?:©|Published|Accepted).*?((?:19|20)\d{2})", text)
    if not year_match:
        year_match = re.search(r"((?:19|20)\d{2})\s*(?:;|,|Vol|\(|Nature|Science|EPB)", text)
    if year_match:
        try:
            year = int(year_match.group(1))
        except (ValueError, IndexError):
            pass

    # 如果 PDF 元数据没有标题，从文本第一段大字取
    if not title:
        for page_no in range(len(doc)):
            blocks = doc[page_no].get_text("blocks")
            for b in blocks:
                txt = b[4].strip()
                if len(txt) > 20 and len(txt) < 200:
                    title = txt.split("\n")[0]
                    break
            if title:
                break

    # 作者
    authors: list[str] = []
    if author_str:
        authors = [a.strip() for a in re.split(r"[,;]|\band\b", author_str) if a.strip()]

    # 期刊名（从首页文本提取）
    venue_match = re.search(
        r"(Nature Cities|Nature Communications|EPB|Urban Studies|"
        r"Social Forces|Sociology of Education|"
        r"Computers, Environment and Urban Systems|"
        r"Landscape and Urban Planning|"
        r"Journal of Transport Geography)",
        text[:500], re.IGNORECASE
    )
    if venue_match:
        venue = venue_match.group(1)

    pages = len(doc)
    doc.close()

    if not title:
        title = pdf_path.stem

    return {
        "title": title,
        "authors": authors,
        "doi": doi,
        "year": year,
        "venue": venue,
        "pages": pages,
        "pdf_path": str(pdf_path),
        "source_dir": str(pdf_path.parent),
    }


def _slugify(text: str, maxlen: int = 80) -> str:
    """把标题转为安全文件名。"""
    slug = re.sub(r"[^\w\s-]", "", text)
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:maxlen]


def scan_papers(source_dir: str, output_dir: str = "data/papers"):
    """扫描、去重、复制。"""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pdf_files = list(source.rglob("*.pdf"))
    print(f"找到 {len(pdf_files)} 篇 PDF")

    papers: list[dict] = []
    seen_doi: set[str] = set()
    seen_hash: set[str] = set()
    skipped = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}", end=" ... ")

        meta = extract_metadata(pdf_path)
        if meta is None:
            failed += 1
            print("FAILED")
            continue

        # 去重：按 DOI
        if meta["doi"] and meta["doi"] in seen_doi:
            skipped += 1
            print(f"SKIP (dup DOI: {meta['doi']})")
            continue

        # 去重：按文件内容哈希
        with open(pdf_path, "rb") as f:
            fhash = hashlib.sha256(f.read()).hexdigest()
        if fhash in seen_hash:
            skipped += 1
            print("SKIP (dup content)")
            continue

        if meta["doi"]:
            seen_doi.add(meta["doi"])
        seen_hash.add(fhash)

        # 命名：DOI > 标题
        if meta["doi"]:
            filename = meta["doi"].replace("/", "_") + ".pdf"
        else:
            filename = _slugify(meta["title"]) + ".pdf"

        # 复制
        dest = output / filename
        shutil.copy2(pdf_path, dest)
        meta["dest_path"] = str(dest)

        papers.append(meta)
        print(f"OK -> {filename}")

    # 写 CSV
    csv_path = Path("data/paper_metadata.csv")
    fieldnames = [
        "title", "authors", "doi", "year", "venue", "pages",
        "dest_path", "source_dir",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in papers:
            p_out = {k: p.get(k, "") for k in fieldnames}
            if isinstance(p_out["authors"], list):
                p_out["authors"] = "; ".join(p_out["authors"])
            writer.writerow(p_out)

    print(f"\n=== 汇总 ===")
    print(f"总 PDF: {len(pdf_files)}")
    print(f"成功:   {len(papers)}")
    print(f"跳过(重复): {skipped}")
    print(f"失败:   {failed}")
    print(f"CSV:    {csv_path}")
    print(f"PDF 副本: {output}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} /path/to/papers")
        sys.exit(1)
    scan_papers(sys.argv[1])
