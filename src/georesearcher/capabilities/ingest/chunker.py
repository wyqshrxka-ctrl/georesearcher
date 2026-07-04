"""父子切块：paragraph 做子块（入向量库），section 做父块（入 SQLite）。

设计（两阶段检索）：
  子块 (paragraph):  细粒度向量检索，写入 Chroma
  父块 (section):    粗粒度上下文，按 section_idx 在 SQLite 中查找
  检索流程：向量命中子块 → 通过 section_idx 取父块 → 拼接完整上下文
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parser import ParsedPdf, ParsedSection


@dataclass
class ChunkSpec:
    """子块：一个自然段落，附 section 上下文引用。"""

    id: str
    paper_id: str
    text: str
    section_idx: int = 0  # 指向所属 section
    section_title: str = ""
    position: int = 0  # 段落在 section 内的序号


def chunk_parsed(parsed: ParsedPdf, paper_id: str) -> tuple[list[ChunkSpec], dict[int, str]]:
    """对已解析的 PDF 做父子切块。

    返回:
      child_chunks: 子块列表（paragraph 级，入向量库）
      parent_map:   {section_idx: section_full_text}（入 SQLite）
    """
    child_chunks: list[ChunkSpec] = []
    parent_map: dict[int, str] = {}

    for si, sec in enumerate(parsed.sections):
        # 父块：整节全文
        full_text = "\n\n".join(sec.paragraphs)
        if len(full_text) < 50:
            continue  # 太短跳过（如参考文献列表）
        parent_map[si] = full_text

        # 子块：逐段落
        for pi, para in enumerate(sec.paragraphs):
            if len(para) < 20:
                continue
            child_chunks.append(ChunkSpec(
                id=f"{paper_id}_s{si}_p{pi}",
                paper_id=paper_id,
                text=para,
                section_idx=si,
                section_title=sec.title,
                position=pi,
            ))

    return child_chunks, parent_map
