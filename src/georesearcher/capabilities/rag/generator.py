"""RAG 生成：强制引用接地 + APA 格式输出 + 父块上下文。

设计（design §3.2）：
  - 优先使用父块（section 全文）作为生成上下文
  - 强制引用接地：每条事实标注来源
  - APA 格式引用列表
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...config import Config, load_config
from ...models.llm import LLMClient, get_llm
from .retriever import RetrievalResult

_GENERATION_SYSTEM_PROMPT = """You are a research assistant that answers questions based ONLY on the provided passages. Follow these rules strictly:

1. Every factual claim MUST include an inline citation like [1], [2], etc. corresponding to the passage numbers.
2. If the passages do NOT contain enough information to answer, say so clearly. Do NOT fabricate.
3. At the end of your answer, list all cited references in APA format using the metadata provided.
4. Keep answers concise but comprehensive. Use academic language.
5. If multiple passages support the same point, cite all of them."""


_GENERATION_USER_TEMPLATE = """Question: {question}

Relevant Passages:
{passages}

Please answer the question based on the passages above."""


@dataclass
class GeneratedAnswer:
    """RAG 生成结果。"""

    answer: str
    references: list[str] = field(default_factory=list)
    cited_chunks: list[str] = field(default_factory=list)
    model: str = ""


class Generator:
    """RAG 生成器。"""

    def __init__(self, cfg: Config | None = None, llm: LLMClient | None = None):
        cfg = cfg or load_config()
        self._llm = llm or get_llm(cfg)

    def generate(
        self,
        question: str,
        results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        if not results:
            return GeneratedAnswer(
                answer="No relevant passages found in the knowledge base.",
                model=self._llm.model,
            )

        # 构造 passage：优先用父块（section 全文），fallback 子块
        passage_blocks: list[str] = []
        for i, rr in enumerate(results, 1):
            text = rr.parent_text if rr.parent_text else rr.child_chunk.chunk.text
            paper_id = rr.child_chunk.chunk.paper_id
            block = f"[{i}] (paper: {paper_id})\n{text[:1500]}"  # 截断避免 token 溢出
            passage_blocks.append(block)

        passages_text = "\n\n".join(passage_blocks)

        messages = [
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": _GENERATION_USER_TEMPLATE.format(
                question=question, passages=passages_text
            )},
        ]

        answer = self._llm.chat(messages, temperature=0.3)

        # 收集引用
        seen_papers: set[str] = set()
        references: list[str] = []
        cited_chunks: list[str] = []

        for i, rr in enumerate(results, 1):
            ref_tag = f"[{i}]"
            if ref_tag in answer:
                chunk = rr.child_chunk.chunk
                cited_chunks.append(chunk.id)
                if chunk.paper_id not in seen_papers:
                    seen_papers.add(chunk.paper_id)
                    references.append(f"[{i}] {chunk.paper_id}")

        return GeneratedAnswer(
            answer=answer,
            references=references,
            cited_chunks=cited_chunks,
            model=self._llm.model,
        )
