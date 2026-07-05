"""M4 编排共享状态 — ResearchState（TypedDict + reducer）。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict, total=False):
    # ---- 输入 ----
    query: str                       # 用户原始输入
    config_path: str | None          # 透传给能力层的配置文件路径（--config）

    # ---- Router 层（Intent-routing 范式） ----
    intent: str | None               # "SEARCH" | "ASK" | "PLOT" | "GIS" | "WRITE"
    intent_confidence: float         # Router 置信度，< 阈值走 CLARIFY
    intent_reason: str               # Router 给出的判断理由（可观测/调试）
    clarify_round: int               # CLARIFY 反问轮次（防死循环）

    # ---- SEARCH / REFLECT 分支（循环①重搜） ----
    search_hits: list[dict]          # 轻量命中：[{paper_id, title, has_pdf}]
    search_round: int                # 重搜轮次
    refined_query: str | None        # REFLECT 改写后的检索词

    # ---- INGEST / INTERPRET ----
    ingested_ids: Annotated[list[str], operator.add]   # reducer 累加
    notes: Annotated[list[dict], operator.add]         # 结构化笔记（dict 化的 StructuredNote）

    # ---- RAG_QA + CRAG（Reflection 范式，循环②） ----
    rag_answer: str | None
    citations: list[dict]            # [{index, paper_id, reference}]
    retrieval_quality: float         # CRAG 门控信号 = 检索 top1 的 rerank 分数
    crag_round: int                  # CRAG 自主补充轮次

    # ---- HUMAN_REVIEW（中断，循环③） ----
    human_feedback: str | None       # 用户反馈；"" 或 None = 满意
    review_round: int                # 人机重答轮次

    # ---- 通用 ----
    error: str | None                # 节点失败降级信息（失败不崩，写进 state）
    trace: Annotated[list[str], operator.add]  # 节点执行轨迹（可观测/演示用）
