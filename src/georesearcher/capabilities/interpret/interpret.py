"""capabilities/interpret — 逐篇结构化解读（plan-m3 §3.6 / C1）。

每个函数都可独立测试：prompt 组装 / JSON 解析 / LLM 调用（mock）。
"""
from __future__ import annotations

import json

from georesearcher.config import Config, load_config
from georesearcher.models.llm import LLMClient, get_llm
from georesearcher.storage.sqlite_store import SqliteStore, get_sqlite_store
from georesearcher.types import StructuredNote

_INTERPRET_PROMPT = """你是一位 GIS / 社会科学领域的科研助理。请根据提供的论文内容，提取以下信息，并以严格的 JSON 格式返回。

**重要规则**：
- 如果原文中没有提及某个信息，该字段留空字符串 ""，**禁止编造或推断**。
- 只返回 JSON，不要任何额外说明文字。
- 对于只有摘要的论文，从摘要中尽可能提取信息，无法提取的字段留空。

返回的 JSON 格式：
{{
  "research_question": "该论文试图回答的核心研究问题",
  "method": "使用的研究方法、数据来源、模型",
  "contribution": "论文的主要贡献或创新点",
  "gap": "论文指出的已有研究不足或未来研究方向",
  "key_findings": "主要发现或结论",
  "summary": "用 2-3 句话概括整篇论文的核心内容"
}}

论文标题：{title}
{source_note}
论文内容：
{context}"""


def _build_prompt(title: str, context: str, *, source_note: str = "") -> str:
    """组装 interpret 的 LLM prompt（C1：纯函数，方便单测）。"""
    return _INTERPRET_PROMPT.format(
        title=title,
        context=context,
        source_note=source_note,
    )


def _parse_note_json(raw: str, paper_id: str) -> StructuredNote:
    """解析 LLM 返回的 JSON → StructuredNote（C1：缺失字段留空，多余字段忽略）。"""
    # Strip code fences
    text = raw.strip()
    if text.startswith("```"):
        # Find the first newline after opening ```
        idx = text.find("\n")
        if idx > 0:
            text = text[idx + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find the first { ... }
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if not isinstance(data, dict):
        data = {}

    return StructuredNote(
        paper_id=paper_id,
        research_question=str(data.get("research_question", "")),
        method=str(data.get("method", "")),
        contribution=str(data.get("contribution", "")),
        gap=str(data.get("gap", "")),
        key_findings=str(data.get("key_findings", "")),
        summary=str(data.get("summary", "")),
    )


def interpret_paper(
    paper_id: str,
    *,
    cfg: Config | None = None,
    llm: LLMClient | None = None,
    sqlite_store: SqliteStore | None = None,
) -> StructuredNote:
    """逐篇结构化提炼，落 SQLite notes（C1）。

    1. paper = store.get_paper(paper_id)；没有→ValueError
    2. 取上下文：优先父块全文；无父块（metadata_only）→ 标记"基于摘要"
    3. 组 prompt → llm.complete()
    4. _parse_note_json → StructuredNote
    5. store.upsert_note(note)
    """
    cfg = cfg or load_config()
    llm = llm or get_llm(cfg)
    sqlite_store = sqlite_store or get_sqlite_store(cfg)

    paper = sqlite_store.get_paper(paper_id)
    if paper is None:
        raise ValueError(f"Paper not found: {paper_id}")

    # 取上下文
    parent_chunks = sqlite_store.get_parent_chunks_for_paper(paper_id)
    source_note = ""

    if parent_chunks:
        # 有父块全文：拼接所有 section
        context = "\n\n".join(parent_chunks.values())
    else:
        # 无父块（metadata_only）：标记"基于摘要"
        context = ""  # 上下文将从 paper 元数据/摘要向量块获取
        source_note = "（注意：该论文仅有摘要，无全文，请基于摘要信息提取）"

    # 如果 context 为空，用标题兜底
    if not context.strip():
        context = f"（该论文仅有元数据，无全文和摘要。以下为已知信息。）\n标题：{paper.title}"

    prompt = _build_prompt(paper.title, context, source_note=source_note)
    raw = llm.complete(prompt)
    note = _parse_note_json(raw, paper_id)
    sqlite_store.upsert_note(note)

    return note
