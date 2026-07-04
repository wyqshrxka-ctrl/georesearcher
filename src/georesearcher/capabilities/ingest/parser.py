"""PDF 解析：使用 PyMuPDF (fitz) 提取结构化内容。

产出：
  - 论文元数据（title, authors, doi 等，从 PDF metadata / 首段尝试提取）
  - 按章节分段的段落列表
  - 每个段落的位置信息（页码、段落序号）

设计决策：不做复杂的 ML 章节检测，只用字号/空行/缩进启发式。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedSection:
    title: str  # 章节标题，如 "1. Introduction"
    paragraphs: list[str]  # 正文段落（按顺序）
    page_start: int = 1
    page_end: int = 1


@dataclass
class ParsedPdf:
    """PyMuPDF 解析后的结构化文档。"""

    file_path: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    doi: str = ""
    sections: list[ParsedSection] = field(default_factory=list)
    raw_metadata: dict = field(default_factory=dict)

    @property
    def all_paragraphs(self) -> list[dict]:
        """返回展平的段落列表，每条带 section 上下文。"""
        out: list[dict] = []
        for i, sec in enumerate(self.sections):
            for j, para in enumerate(sec.paragraphs):
                out.append({
                    "text": para,
                    "section_title": sec.title,
                    "section_idx": i,
                    "para_idx": j,
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                })
        return out


def _hash_id(text: str, prefix: str = "ch") -> str:
    return f"{prefix}_{hashlib.md5(text.encode()).hexdigest()[:12]}"


def parse_pdf(file_path: str | Path) -> ParsedPdf:
    """解析单个 PDF 文件，返回结构化结果。

    处理边界：
      - 加密 PDF：直接抛出异常（fitz 会先报 DocumentError）
      - 扫描件（无文字）：返回空 sections，调用方自行判断
      - 畸形文件：fitz.open 失败，异常上抛
    """
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError("需要安装 PyMuPDF：uv pip install pymupdf") from e

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    doc = fitz.open(str(path))

    # 1. 提取元数据
    meta = doc.metadata or {}
    title = meta.get("title", "").strip()
    author_str = meta.get("author", "")

    # 2. 提取全文文本，附带位置信息
    blocks: list[dict] = []  # {text, page, block_no, font_size_avg}
    for page_no in range(len(doc)):
        page = doc[page_no]
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            text = text.strip()
            if not text or len(text) < 10:
                continue  # 跳过页码、页眉等短块
            # 估算字号（block 不直接给字号，用 block 高度的 1/3 近似行高）
            line_count = max(1, text.count("\n") + 1)
            font_size_avg = (y1 - y0) / line_count / 3
            blocks.append({
                "text": text,
                "page": page_no + 1,
                "block_no": block_no,
                "font_size_avg": font_size_avg,
            })

    doc.close()

    # 3. 如果 PDF 元数据没有标题，从第一页的大字文本猜测
    if not title and blocks:
        # 取第一个字号较大的块作为标题
        largest = max(blocks, key=lambda b: b["font_size_avg"])
        title = largest["text"].split("\n")[0].strip()

    # 4. 作者解析
    authors: list[str] = []
    if author_str:
        authors = [a.strip() for a in re.split(r"[,;]|\band\b", author_str) if a.strip()]

    # 5. DOI 提取
    doi = ""
    if meta.get("subject", ""):
        doi_match = re.search(r"10\.\d{4,}/[^\s]+", meta.get("subject", ""))
        if doi_match:
            doi = doi_match.group(0)
    # 也扫前几段看是否有 DOI
    for b in blocks[:10]:
        doi_match = re.search(r"10\.\d{4,}/[^\s]+", b["text"])
        if doi_match:
            doi = doi_match.group(0)
            break

    # 6. 章节分割（启发式：字号较大 + 短文本 = 标题）
    sections: list[ParsedSection] = []
    current_section = ParsedSection(title="Abstract", paragraphs=[])

    for block in blocks:
        text = block["text"]
        is_large = block["font_size_avg"] > 10 and len(text.split("\n")) <= 3
        is_section_header = is_large and len(text) < 120

        if is_section_header and current_section.paragraphs:
            sections.append(current_section)
            current_section = ParsedSection(
                title=text.split("\n")[0].strip(),
                paragraphs=[],
                page_start=block["page"],
            )
        else:
            current_section.paragraphs.append(text)
            current_section.page_end = block["page"]

    # 不丢弃最后一个 section（即使空）
    if current_section.paragraphs:
        sections.append(current_section)

    # 如果分割失败，退化为单 section
    if not sections:
        all_text = [b["text"] for b in blocks]
        sections = [ParsedSection(title="Full Text", paragraphs=all_text)]

    result = ParsedPdf(
        file_path=str(path),
        title=title,
        authors=authors,
        doi=doi,
        sections=sections,
        raw_metadata=meta,
    )
    return result
