"""M4 编排图 — StateGraph 建图 + 条件边 + 编译。"""

from __future__ import annotations

import functools
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from georesearcher.orchestration.state import ResearchState
from georesearcher.orchestration.nodes import (
    router_node, clarify_node,
    search_node, reflect_node, interpret_node, rag_qa_node,
    plot_node, gis_node, write_node, human_review_node,
)


# ---------------------------------------------------------------------------
# 条件边路由函数（纯函数，单测重点）
# 接受 config 阈值参数，便于测试时注入不同值
# ---------------------------------------------------------------------------

def _route_after_router(
    state: ResearchState,
    *,
    confidence_threshold: float = 0.6,
    clarify_max_rounds: int = 2,
) -> str:
    """Router 之后的路由：按 intent + confidence 决定下一节点。

    Returns: "clarify" | "search" | "rag_qa" | "plot" | "gis" | "write"
    """
    confidence = state.get("intent_confidence", 0.0)
    intent = state.get("intent", "ASK")
    clarify_round = state.get("clarify_round", 0)

    # Low confidence → CLARIFY
    if confidence < confidence_threshold and clarify_round < clarify_max_rounds:
        return "clarify"

    # Route by intent
    intent_map = {
        "SEARCH": "search",
        "ASK": "rag_qa",
        "PLOT": "plot",
        "GIS": "gis",
        "WRITE": "write",
    }
    return intent_map.get(intent, "rag_qa")


def _route_after_search(
    state: ResearchState,
    *,
    min_hits: int = 3,
    max_rounds: int = 2,
) -> str:
    """Search 之后的路由：命中足够 → interpret；不足且未达上限 → reflect。

    Returns: "reflect" | "interpret"
    """
    hits = state.get("search_hits", [])
    search_round = state.get("search_round", 0)

    if len(hits) < min_hits and search_round < max_rounds:
        return "reflect"
    return "interpret"


def _route_after_rag(
    state: ResearchState,
    *,
    crag_max_rounds: int = 1,
) -> str:
    """RAG_QA 之后的路由：retrieval_quality 门控。

    Returns: "search" | "human_review"
    """
    # If rag_answer is set, go to human_review (generation happened)
    if state.get("rag_answer"):
        return "human_review"
    # Otherwise, go back to search for CRAG supplement
    return "search"


def _route_after_review(
    state: ResearchState,
    *,
    review_max_rounds: int = 3,
) -> str:
    """HumanReview 之后的路由：有 feedback → 回 rag_qa；空 → END。

    Returns: "rag_qa" | "__end__"
    """
    feedback = state.get("human_feedback")
    review_round = state.get("review_round", 0)

    if not feedback or review_round >= review_max_rounds:
        return END
    return "rag_qa"


# ---------------------------------------------------------------------------
# 建图
# ---------------------------------------------------------------------------

def build_graph(deps, *, checkpointer=None):
    """组装 StateGraph 并编译。

    Args:
        deps: NodeDeps 实例（含 cfg）
        checkpointer: 默认 MemorySaver()；可传 SqliteSaver 持久化
    """
    # Read thresholds from cfg (with defaults)
    ocfg = deps.cfg.orchestration
    rt_confidence_threshold = ocfg.router_confidence_threshold
    clarify_max_rounds = ocfg.clarify_max_rounds
    search_min_hits = ocfg.search_min_hits
    search_max_rounds = ocfg.search_max_rounds
    crag_max_rounds = ocfg.crag_max_rounds
    review_max_rounds = ocfg.review_max_rounds

    graph = StateGraph(ResearchState)

    # Bind deps to each node via functools.partial
    _n = lambda fn: functools.partial(fn, deps=deps)

    # Add all nodes
    graph.add_node("router", _n(router_node))
    graph.add_node("clarify", _n(clarify_node))
    graph.add_node("search", _n(search_node))
    graph.add_node("reflect", _n(reflect_node))
    graph.add_node("interpret", _n(interpret_node))
    graph.add_node("rag_qa", _n(rag_qa_node))
    graph.add_node("plot", _n(plot_node))
    graph.add_node("gis", _n(gis_node))
    graph.add_node("write", _n(write_node))
    graph.add_node("human_review", _n(human_review_node))

    # Edges
    graph.add_edge(START, "router")

    # Router → conditional (bind thresholds via partial)
    graph.add_conditional_edges(
        "router",
        functools.partial(
            _route_after_router,
            confidence_threshold=rt_confidence_threshold,
            clarify_max_rounds=clarify_max_rounds,
        ),
        {
            "clarify": "clarify",
            "search": "search",
            "rag_qa": "rag_qa",
            "plot": "plot",
            "gis": "gis",
            "write": "write",
        },
    )

    # Clarify → back to router (with human_feedback in state)
    graph.add_edge("clarify", "router")

    # Search → conditional
    graph.add_conditional_edges(
        "search",
        functools.partial(
            _route_after_search,
            min_hits=search_min_hits,
            max_rounds=search_max_rounds,
        ),
        {
            "reflect": "reflect",
            "interpret": "interpret",
        },
    )

    # Reflect → back to search (with refined_query)
    graph.add_edge("reflect", "search")

    # Interpret → rag_qa
    graph.add_edge("interpret", "rag_qa")

    # Plot / GIS / Write → human_review (placeholder paths)
    graph.add_edge("plot", "human_review")
    graph.add_edge("gis", "human_review")
    graph.add_edge("write", "human_review")

    # RAG_QA → conditional (CRAG)
    graph.add_conditional_edges(
        "rag_qa",
        functools.partial(_route_after_rag, crag_max_rounds=crag_max_rounds),
        {
            "search": "search",
            "human_review": "human_review",
        },
    )

    # HumanReview → conditional (loop or END)
    graph.add_conditional_edges(
        "human_review",
        functools.partial(_route_after_review, review_max_rounds=review_max_rounds),
        {
            "rag_qa": "rag_qa",
            END: END,
        },
    )

    checkpointer = checkpointer or MemorySaver()
    return graph.compile(checkpointer=checkpointer)
