"""M4 节点函数 — 薄壳复用能力层入口 + 占位节点 + 人机中断。

每个节点签名: def xxx_node(state: ResearchState, *, deps: NodeDeps) -> dict
返回要写回 state 的部分 key（LangGraph 节点约定：返回 dict merge 进 state）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from langgraph.types import interrupt
from georesearcher.orchestration.state import ResearchState


# ---------------------------------------------------------------------------
# 依赖容器（测试注入点）
# ---------------------------------------------------------------------------

@dataclass
class NodeDeps:
    """节点依赖容器：生产用真实现，测试注入 fake。避免节点内部直接 new。"""

    cfg: object                        # Config
    llm: object                        # LLMClient
    embedder: object = None
    vector_store: object = None
    sqlite_store: object = None
    retrieval_pipeline: object = None  # RetrievalPipeline
    generator: object = None           # Generator
    search_source: object = None       # PaperSource，可注入本地降级源
    downloader: object = None
    retrieval_cfg: object = None       # RetrievalConfig


# ---------------------------------------------------------------------------
# Router 层
# ---------------------------------------------------------------------------

def router_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """调用 classify_intent → 写 intent / intent_confidence / intent_reason。"""
    from georesearcher.orchestration.router import classify_intent

    query = state.get("query", "")
    clarify = state.get("human_feedback")

    result = classify_intent(query, llm=deps.llm, prev_clarify=clarify)
    return {
        "intent": result["intent"],
        "intent_confidence": result["confidence"],
        "intent_reason": result["reason"],
        "trace": [f"router: {result['intent']} (conf={result['confidence']:.2f})"],
    }


def clarify_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """置信度低时 interrupt 反问用户。

    拿到补充后 clarify_round+1，回 router 重判。
    """
    round_num = state.get("clarify_round", 0)
    intent = state.get("intent", "ASK")
    confidence = state.get("intent_confidence", 0.0)
    reason = state.get("intent_reason", "")

    feedback = interrupt({
        "question": (
            f"我不太确定你的意图（判为 {intent}，置信度 {confidence:.0%}）。"
            f"判断理由：{reason}\n\n"
            "请补充说明：你是想【查找新文献(SEARCH)】还是【问已有知识库的问题(ASK)】？"
        ),
        "intent": intent,
        "confidence": confidence,
    })
    return {
        "human_feedback": feedback if feedback else None,
        "clarify_round": round_num + 1,
        "trace": [f"clarify: round={round_num + 1}"],
    }


# ---------------------------------------------------------------------------
# SEARCH / REFLECT 分支（复用 M3：ingest_from_search）
# ---------------------------------------------------------------------------

def search_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """薄壳调用 ingest_from_search（M3）。

    - 优先用 refined_query，fallback query
    - 网络不可达时写 error 不崩
    """
    from georesearcher.capabilities.search.pipeline import ingest_from_search

    query = state.get("refined_query") or state.get("query", "")
    try:
        result = ingest_from_search(
            query,
            limit=deps.cfg.orchestration.search_min_hits,
            cfg=deps.cfg,
            sqlite_store=deps.sqlite_store,
            vector_store=deps.vector_store,
            embedder=deps.embedder,
            source=deps.search_source,
            downloader=deps.downloader,
        )
        hits = [
            {"paper_id": pid, "title": "", "has_pdf": True}
            for pid in result.ingested_full
        ] + [
            {"paper_id": pid, "title": "", "has_pdf": False}
            for pid in result.ingested_meta
        ]
        return {
            "search_hits": hits,
            "ingested_ids": result.ingested_full + result.ingested_meta,
            "search_round": state.get("search_round", 0) + 1,
            "trace": [f"search: {len(hits)} hits (full={len(result.ingested_full)}, meta={len(result.ingested_meta)})"],
        }
    except Exception as e:
        # Search failed — force route_after_search to go to interpret by setting search_round = max
        max_r = deps.cfg.orchestration.search_max_rounds
        return {
            "error": f"search_node: {e}",
            "search_hits": [],
            "search_round": max_r,
            "trace": [f"search: FAILED — {e}"],
        }


def reflect_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """检索命中不足时，LLM 改写检索词 → 写 refined_query、search_round+1。"""
    query = state.get("refined_query") or state.get("query", "")
    hits = state.get("search_hits", [])
    messages = [
        {"role": "system", "content": (
            "You are a research librarian. The user searched for literature but got too few results. "
            "Rewrite the search query to be broader or use alternative keywords. "
            "Return ONLY the new query string, nothing else."
        )},
        {"role": "user", "content": (
            f"Original query: {query}\n"
            f"Results found: {len(hits)} (too few).\n"
            f"Please rewrite to find more relevant literature."
        )},
    ]
    refined = deps.llm.complete(messages).strip().strip('"')
    return {
        "refined_query": refined,
        "trace": [f"reflect: refined query → {refined[:80]}{'...' if len(refined) > 80 else ''}"],
    }


# ---------------------------------------------------------------------------
# INGEST / INTERPRET（复用 M3：interpret_paper）
# ---------------------------------------------------------------------------

def interpret_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """对 ingested_ids 逐个调用 interpret_paper（M3）→ 写 notes。"""
    from georesearcher.capabilities.interpret.interpret import interpret_paper

    ingested = state.get("ingested_ids", [])
    notes: list[dict] = []
    trace_entries: list[str] = []

    for pid in ingested:
        try:
            note = interpret_paper(
                pid, cfg=deps.cfg, llm=deps.llm, sqlite_store=deps.sqlite_store,
            )
            notes.append(note.model_dump())
            trace_entries.append(f"interpret: {pid[:20]}... OK")
        except Exception as e:
            trace_entries.append(f"interpret: {pid[:20]}... FAILED — {e}")

    return {"notes": notes, "trace": trace_entries}


# ---------------------------------------------------------------------------
# RAG_QA + CRAG（复用 M1：RetrievalPipeline + Generator）
# ---------------------------------------------------------------------------

def rag_qa_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """检索 + 生成，含 CRAG 门控。

    1. results = retrieval_pipeline.retrieve(query)
    2. retrieval_quality = top1 rerank 分
    3. 若 quality >= 阈值 或 crag_round >= 上限 → 生成答案
       否则 → 只写 retrieval_quality（条件边路由去 SEARCH 联网补充）
    """
    query = state.get("query", "")
    crag_round = state.get("crag_round", 0)
    cfg = deps.cfg.orchestration

    # Retrieve
    results = deps.retrieval_pipeline.retrieve(query, config=deps.retrieval_cfg)
    if results:
        top_score = results[0].child_chunk.score
    else:
        top_score = 0.0

    retrieval_quality = top_score

    # CRAG 门控：质量不足且未达上限 → 只写 quality，触发联网补充
    quality_ok = (
        retrieval_quality >= cfg.crag_quality_threshold
        or crag_round >= cfg.crag_max_rounds
    )

    if not quality_ok:
        return {
            "retrieval_quality": retrieval_quality,
            "crag_round": crag_round + 1,
            "trace": [f"rag_qa: quality={retrieval_quality:.3f} < threshold={cfg.crag_quality_threshold}, triggering CRAG round {crag_round + 1}"],
        }

    # Generate
    answer = deps.generator.generate(query, results)
    citations = [
        {"index": i + 1, "reference": ref}
        for i, ref in enumerate(answer.references)
    ]

    return {
        "rag_answer": answer.answer,
        "citations": citations,
        "retrieval_quality": retrieval_quality,
        "trace": [f"rag_qa: quality={retrieval_quality:.3f}, {len(citations)} refs"],
    }


# ---------------------------------------------------------------------------
# 占位节点（诚实边界，不实现能力）
# ---------------------------------------------------------------------------

def plot_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    return {
        "rag_answer": "[占位] 绘图能力属 M5 里程碑，编排接口已预留。",
        "trace": ["plot_node: placeholder"],
    }


def gis_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    return {
        "rag_answer": "[占位] GIS/QGIS 能力属后续 MCP 里程碑（预留 ReAct 子编排）。",
        "trace": ["gis_node: placeholder"],
    }


def write_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    return {
        "rag_answer": "[占位] 学术写作能力属 M8 里程碑，可接外部 skill。",
        "trace": ["write_node: placeholder"],
    }


# ---------------------------------------------------------------------------
# 人机中断（决策 B / b+）
# ---------------------------------------------------------------------------

def human_review_node(state: ResearchState, *, deps: NodeDeps) -> dict:
    """interrupt 暂停 → 等待用户反馈 → resume 或 END。

    feedback 为空/None → 满意（条件边走 END）。
    否则 → 写 human_feedback、review_round+1（条件边回 RAG_QA）。
    """
    answer = state.get("rag_answer", "(no answer)")
    citations = state.get("citations", [])
    review_round = state.get("review_round", 0)

    feedback = interrupt({
        "answer": answer,
        "citations": citations,
        "prompt": "满意吗？回车通过，或输入追问让我重答",
    })

    if not feedback or not feedback.strip():
        return {
            "human_feedback": None,
            "trace": ["human_review: accepted"],
        }
    return {
        "human_feedback": feedback.strip(),
        "review_round": review_round + 1,
        "trace": [f"human_review: feedback → {feedback.strip()[:60]}..."],
    }
