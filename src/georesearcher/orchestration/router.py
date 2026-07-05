"""M4 Router — LLM 结构化意图识别（Intent-routing 范式）。"""

from __future__ import annotations

import json
from georesearcher.models.llm import LLMClient

VALID_INTENTS = ("SEARCH", "ASK", "PLOT", "GIS", "WRITE")

_ROUTER_PROMPT = """You are an intent classifier for a Research Copilot agent that helps researchers with GIS, spatial analysis, and smart-city research.

Your job: given a user message, classify it into exactly ONE of the following intents. Output ONLY a JSON object (no markdown, no explanation).

Intent definitions:

- SEARCH: User wants to find NEW literature/papers on a topic. Keywords: "find papers", "search for", "look up literature", "what papers exist about", "retrieve studies on", "literature search", "找文献", "检索".
- ASK: User wants to ask a question and get an answer from the existing knowledge base (already-ingested papers). Keywords: "what does the paper say", "summarize", "explain", "compare", "according to the knowledge base", "问", "总结", "解释".
- PLOT: User wants to create a visualization, chart, map, or figure. Keywords: "draw", "plot", "chart", "map", "visualize", "figure", "画图", "绘图", "可视化".
- GIS: User wants to perform GIS/spatial analysis (e.g., Moran's I, spatial autocorrelation, QGIS operations, spatial regression). Keywords: "spatial analysis", "Moran's I", "QGIS", "GIS", "geoprocessing", "空间分析", "空间自相关".
- WRITE: User wants to write something (literature review, paper section, summary). Keywords: "write a review", "draft", "literature review", "compose", "写作", "写综述".

Rules:
- If uncertain between SEARCH and ASK, pick the one that fits better and set confidence < 0.7.
- Return confidence as a float between 0 and 1, where 1.0 = perfectly clear.
- Include a one-sentence reason.

Output format (strict JSON):
{"intent": "ASK", "confidence": 0.95, "reason": "User is asking a question about existing knowledge base content."}

Examples:
User: "帮我找学校隔离对教育不平等影响的文献"
Output: {"intent": "SEARCH", "confidence": 0.95, "reason": "User explicitly asks to find literature on a research topic."}

User: "根据知识库里的论文，学校隔离主要有哪些表现？"
Output: {"intent": "ASK", "confidence": 0.92, "reason": "User asks a knowledge-base question referencing already-ingested papers."}

User: "帮我画一张城市活力的空间分布图"
Output: {"intent": "PLOT", "confidence": 0.93, "reason": "User asks to create a spatial visualization/chart."}

User: "用 Moran's I 分析一下这个数据集的空间自相关"
Output: {"intent": "GIS", "confidence": 0.96, "reason": "User explicitly requests a GIS spatial analysis method (Moran's I)."}

User: "帮我写一篇关于教育不平等的文献综述"
Output: {"intent": "WRITE", "confidence": 0.94, "reason": "User asks to compose a literature review."}

User: "最近有什么关于智慧城市的新研究"
Output: {"intent": "SEARCH", "confidence": 0.82, "reason": "User asks about new/recent research, implying a search for literature."}
"""


def _parse_intent_json(raw: str) -> dict:
    """Parse LLM intent JSON output.

    Strips ```json fences → json.loads → validates intent and confidence.
    Pure function: no network, no LLM call.
    Returns {"intent": str, "confidence": float, "reason": str}.
    On failure: {"intent": "ASK", "confidence": 0.0, "reason": "parse error"}.
    """
    text = raw.strip()
    # Strip ```json fences
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"intent": "ASK", "confidence": 0.0, "reason": "invalid JSON"}

    intent = parsed.get("intent", "ASK")
    if intent not in VALID_INTENTS:
        intent = "ASK"
    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))  # clamp
    reason = parsed.get("reason", "")

    return {"intent": intent, "confidence": confidence, "reason": reason}


def classify_intent(
    query: str,
    *,
    llm: LLMClient,
    prev_clarify: str | None = None,
) -> dict:
    """LLM structured intent classification.

    Returns {"intent": str, "confidence": float, "reason": str}.
    - Uses few-shot prompt to anchor 5 intent boundaries.
    - Forces JSON output; parse failure → fallback ASK, confidence 0.0 (triggers CLARIFY).
    - prev_clarify: user's clarification from CLARIFY node, appended to query.
    """
    full_query = query
    if prev_clarify:
        full_query = f"{query}\n\n[用户补充说明: {prev_clarify}]"

    messages = [
        {"role": "system", "content": _ROUTER_PROMPT},
        {"role": "user", "content": f"User message: {full_query}"},
    ]
    raw = llm.complete(messages)
    return _parse_intent_json(raw)
