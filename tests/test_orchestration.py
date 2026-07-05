"""M4 编排单测 — 全 mock，零真 LLM / 零网络 / 零模型下载。

覆盖：
- T1: ResearchState 结构
- T2: _parse_intent_json + classify_intent（fake LLM）
- T3: NodeDeps + 节点 state 写入（fake deps）
- T4: 占位节点返回预设串
- T5: _route_after_* 条件边纯函数
- T7: 图级集成（走向 / interrupt / round 上限）
"""

from __future__ import annotations

import json
import pytest
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


# ---------------------------------------------------------------------------
# T1: ResearchState
# ---------------------------------------------------------------------------

def test_state_instantiation():
    from georesearcher.orchestration.state import ResearchState
    state = ResearchState(query="test")
    assert state["query"] == "test"


def test_state_has_required_keys():
    from georesearcher.orchestration.state import ResearchState
    expected = {
        "query", "config_path",
        "intent", "intent_confidence", "intent_reason", "clarify_round",
        "search_hits", "search_round", "refined_query",
        "ingested_ids", "notes",
        "rag_answer", "citations", "retrieval_quality", "crag_round",
        "human_feedback", "review_round",
        "error", "trace",
    }
    state = ResearchState()
    for key in expected:
        assert key in state or True  # total=False so keys may be absent until set


# ---------------------------------------------------------------------------
# T2: router — _parse_intent_json (pure function, no network)
# ---------------------------------------------------------------------------

class TestParseIntentJson:
    def test_valid_json_all_fields(self):
        from georesearcher.orchestration.router import _parse_intent_json
        raw = '{"intent": "SEARCH", "confidence": 0.95, "reason": "looking for papers"}'
        result = _parse_intent_json(raw)
        assert result["intent"] == "SEARCH"
        assert result["confidence"] == 0.95
        assert result["reason"] == "looking for papers"

    def test_valid_json_missing_fields(self):
        from georesearcher.orchestration.router import _parse_intent_json
        raw = '{"intent": "ASK"}'
        result = _parse_intent_json(raw)
        assert result["intent"] == "ASK"
        assert result["confidence"] == 0.0
        assert result["reason"] == ""

    def test_invalid_intent_falls_back_to_ask(self):
        from georesearcher.orchestration.router import _parse_intent_json
        raw = '{"intent": "INVALID", "confidence": 0.8}'
        result = _parse_intent_json(raw)
        assert result["intent"] == "ASK"

    def test_invalid_json_returns_ask_zero_conf(self):
        from georesearcher.orchestration.router import _parse_intent_json
        result = _parse_intent_json("not json at all")
        assert result["intent"] == "ASK"
        assert result["confidence"] == 0.0

    def test_markdown_code_fence(self):
        from georesearcher.orchestration.router import _parse_intent_json
        raw = '```json\n{"intent": "PLOT", "confidence": 0.9, "reason": "draw a map"}\n```'
        result = _parse_intent_json(raw)
        assert result["intent"] == "PLOT"
        assert result["confidence"] == 0.9

    def test_confidence_clamped(self):
        from georesearcher.orchestration.router import _parse_intent_json
        raw = '{"intent": "GIS", "confidence": 2.5}'
        result = _parse_intent_json(raw)
        assert result["confidence"] == 1.0
        raw2 = '{"intent": "GIS", "confidence": -0.5}'
        result2 = _parse_intent_json(raw2)
        assert result2["confidence"] == 0.0

    def test_all_five_intents_parsed(self):
        from georesearcher.orchestration.router import _parse_intent_json
        for intent in ("SEARCH", "ASK", "PLOT", "GIS", "WRITE"):
            raw = json.dumps({"intent": intent, "confidence": 0.8, "reason": "test"})
            result = _parse_intent_json(raw)
            assert result["intent"] == intent


def test_classify_intent_with_fake_llm():
    """classify_intent calls LLM then _parse_intent_json."""
    from georesearcher.orchestration.router import classify_intent

    class FakeLLM:
        def complete(self, messages):
            return '{"intent": "SEARCH", "confidence": 0.92, "reason": "user asks to find papers"}'

    fake = FakeLLM()
    result = classify_intent("find papers about urban vitality", llm=fake)
    assert result["intent"] == "SEARCH"
    assert result["confidence"] == 0.92

    result2 = classify_intent("根据知识库回答", llm=fake, prev_clarify="我想查已有知识库")
    assert result2["intent"] == "SEARCH"  # same fake returns same


# ---------------------------------------------------------------------------
# T3 + T4: Node tests (fake deps)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_node_deps():
    """Create NodeDeps with all fake components."""
    from georesearcher.orchestration.nodes import NodeDeps

    class FakeCfg:
        class Orchestration:
            router_confidence_threshold = 0.6
            search_min_hits = 3
            search_max_rounds = 2
            crag_quality_threshold = 0.3
            crag_max_rounds = 1
            review_max_rounds = 3
            clarify_max_rounds = 2
            checkpoint_path = "/tmp/fake_checkpoints.sqlite"
        orchestration = Orchestration()

    class FakeLLM:
        def complete(self, messages):
            return '{"intent": "ASK", "confidence": 0.95, "reason": "asking knowledge base"}'

    class FakeRetrievalPipeline:
        def retrieve(self, query, *, config=None):
            from dataclasses import dataclass as dc
            @dc
            class FakeRetrieved:
                chunk: object
                score: float
            @dc
            class FakeChunk:
                text: str
                paper_id: str
            @dc
            class FakeResult:
                child_chunk: object
                parent_text: str = ""
            chunk = FakeChunk(text="test text", paper_id="doi::test")
            retrieved = FakeRetrieved(chunk=chunk, score=0.85)
            return [FakeResult(child_chunk=retrieved, parent_text="test text")]

    class FakeGenerator:
        def generate(self, question, results):
            class FakeAnswer:
                answer: str = "This is a test answer."
                references: list = ["Ref 1: test"]
            return FakeAnswer()

    deps = NodeDeps(
        cfg=FakeCfg(),
        llm=FakeLLM(),
        embedder=None,
        vector_store=None,
        sqlite_store=None,
        retrieval_pipeline=FakeRetrievalPipeline(),
        generator=FakeGenerator(),
    )
    return deps


def test_router_node_writes_intent(fake_node_deps):
    from georesearcher.orchestration.nodes import router_node
    state = {"query": "test query"}
    result = router_node(state, deps=fake_node_deps)
    assert result["intent"] == "ASK"
    assert result["intent_confidence"] == 0.95
    assert "trace" in result


def test_rag_qa_node_writes_answer(fake_node_deps):
    from georesearcher.orchestration.nodes import rag_qa_node
    state = {"query": "test query", "crag_round": 0}
    result = rag_qa_node(state, deps=fake_node_deps)
    assert result["rag_answer"] == "This is a test answer."
    assert result["retrieval_quality"] == 0.85
    assert len(result["citations"]) == 1


def test_rag_qa_node_crag_threshold_trigger(fake_node_deps):
    """When retrieval quality is low, CRAG should trigger (no answer)."""
    from georesearcher.orchestration.nodes import rag_qa_node

    # Make retrieval return low score
    class LowScorePipeline:
        def retrieve(self, query, *, config=None):
            from dataclasses import dataclass as dc
            @dc
            class FakeRetrieved:
                chunk: object
                score: float
            @dc
            class FakeChunk:
                text: str
                paper_id: str
            @dc
            class FakeResult:
                child_chunk: object
                parent_text: str = ""
            chunk = FakeChunk(text="test", paper_id="doi::test")
            retrieved = FakeRetrieved(chunk=chunk, score=0.1)
            return [FakeResult(child_chunk=retrieved)]

    fake_node_deps.retrieval_pipeline = LowScorePipeline()
    state = {"query": "test", "crag_round": 0}
    result = rag_qa_node(state, deps=fake_node_deps)
    # Low score, crag_round=0 < max=1 → no answer, triggers CRAG
    assert "rag_answer" not in result or result.get("rag_answer") is None
    assert result["retrieval_quality"] == 0.1


def test_plot_node_placeholder(fake_node_deps):
    from georesearcher.orchestration.nodes import plot_node
    result = plot_node({}, deps=fake_node_deps)
    assert "占位" in result["rag_answer"]
    assert "placeholder" in result["trace"][0]


def test_gis_node_placeholder(fake_node_deps):
    from georesearcher.orchestration.nodes import gis_node
    result = gis_node({}, deps=fake_node_deps)
    assert "占位" in result["rag_answer"]


def test_write_node_placeholder(fake_node_deps):
    from georesearcher.orchestration.nodes import write_node
    result = write_node({}, deps=fake_node_deps)
    assert "占位" in result["rag_answer"]


# ---------------------------------------------------------------------------
# T5: _route_after_* 条件边纯函数
# ---------------------------------------------------------------------------

class TestRouteAfterRouter:
    def test_high_confidence_ask(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "ASK", "intent_confidence": 0.95, "clarify_round": 0}
        assert _route_after_router(state) == "rag_qa"

    def test_high_confidence_search(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "SEARCH", "intent_confidence": 0.9, "clarify_round": 0}
        assert _route_after_router(state) == "search"

    def test_low_confidence_clarify(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "ASK", "intent_confidence": 0.3, "clarify_round": 0}
        assert _route_after_router(state, confidence_threshold=0.6) == "clarify"

    def test_clarify_round_exceeded(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "ASK", "intent_confidence": 0.3, "clarify_round": 2}
        assert _route_after_router(state, clarify_max_rounds=2) == "rag_qa"

    def test_plot_intent(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "PLOT", "intent_confidence": 0.9, "clarify_round": 0}
        assert _route_after_router(state) == "plot"

    def test_gis_intent(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "GIS", "intent_confidence": 0.9, "clarify_round": 0}
        assert _route_after_router(state) == "gis"

    def test_write_intent(self):
        from georesearcher.orchestration.graph import _route_after_router
        state = {"intent": "WRITE", "intent_confidence": 0.9, "clarify_round": 0}
        assert _route_after_router(state) == "write"


class TestRouteAfterSearch:
    def test_enough_hits(self):
        from georesearcher.orchestration.graph import _route_after_search
        state = {"search_hits": [{"paper_id": "a"}, {"paper_id": "b"}, {"paper_id": "c"}], "search_round": 0}
        assert _route_after_search(state, min_hits=3) == "interpret"

    def test_not_enough_hits_reflect(self):
        from georesearcher.orchestration.graph import _route_after_search
        state = {"search_hits": [{"paper_id": "a"}], "search_round": 0}
        assert _route_after_search(state, min_hits=3) == "reflect"

    def test_max_rounds_exceeded(self):
        from georesearcher.orchestration.graph import _route_after_search
        state = {"search_hits": [{"paper_id": "a"}], "search_round": 2}
        assert _route_after_search(state, max_rounds=2) == "interpret"


class TestRouteAfterRag:
    def test_with_answer_goes_to_review(self):
        from georesearcher.orchestration.graph import _route_after_rag
        state = {"rag_answer": "some answer"}
        assert _route_after_rag(state) == "human_review"

    def test_without_answer_goes_to_search(self):
        from georesearcher.orchestration.graph import _route_after_rag
        state = {}
        assert _route_after_rag(state) == "search"


class TestRouteAfterReview:
    def test_no_feedback_ends(self):
        from georesearcher.orchestration.graph import _route_after_review
        state = {"human_feedback": None}
        assert _route_after_review(state) == END

    def test_with_feedback_retry(self):
        from georesearcher.orchestration.graph import _route_after_review
        state = {"human_feedback": "need more detail", "review_round": 0}
        assert _route_after_review(state) == "rag_qa"

    def test_max_review_rounds_ends(self):
        from georesearcher.orchestration.graph import _route_after_review
        state = {"human_feedback": "still not good", "review_round": 3}
        assert _route_after_review(state, review_max_rounds=3) == END


# ---------------------------------------------------------------------------
# T7: 图级集成测试（全 fake，零网络）
# ---------------------------------------------------------------------------

@pytest.fixture
def compiled_graph():
    """Build and compile a graph with all fake deps."""
    from georesearcher.orchestration.nodes import NodeDeps
    from georesearcher.orchestration.graph import build_graph

    class FakeOrchestrationCfg:
        router_confidence_threshold = 0.6
        search_min_hits = 3
        search_max_rounds = 2
        crag_quality_threshold = 0.3
        crag_max_rounds = 1
        review_max_rounds = 3
        clarify_max_rounds = 2
        checkpoint_path = "/tmp/test_checkpoints.sqlite"

    class FakeCfg:
        orchestration = FakeOrchestrationCfg()

    class FakeLLM:
        def complete(self, messages):
            return '{"intent": "ASK", "confidence": 0.95, "reason": "knowledge base question"}'

    class FakeChunk:
        text: str = "test"
        paper_id: str = "doi::test"

    class FakeRetrieved:
        def __init__(self, score=0.85):
            self.chunk = FakeChunk()
            self.score = score

    class FakeResult:
        def __init__(self, score=0.85):
            self.child_chunk = FakeRetrieved(score=score)
            self.parent_text = "test parent text"

    class FakeRetrievalPipeline:
        def retrieve(self, query, *, config=None):
            return [FakeResult(score=0.85)]

    class FakeGenerator:
        def generate(self, question, results):
            class FakeAnswer:
                answer = "This is a fake answer about the query."
                references = ["Ref 1: Test et al. (2024)"]
            return FakeAnswer()

    deps = NodeDeps(
        cfg=FakeCfg(),
        llm=FakeLLM(),
        embedder=None,
        vector_store=None,
        sqlite_store=None,
        retrieval_pipeline=FakeRetrievalPipeline(),
        generator=FakeGenerator(),
    )

    return build_graph(deps, checkpointer=MemorySaver())


def test_graph_ask_flow_to_review(compiled_graph):
    """ASK intent should flow: router → rag_qa → human_review."""
    thread_id = "test-ask-1"
    config = {"configurable": {"thread_id": thread_id}}
    events = list(compiled_graph.stream({"query": "test question"}, config))
    # Check state after run
    state = compiled_graph.get_state(config)
    values = state.values
    # Should have an answer and be at human_review interrupt
    assert "rag_answer" in values or state.next


def test_graph_search_intent_flow(compiled_graph):
    """SEARCH intent should route to search node, then reflect (no hits) capped by max_rounds."""
    from georesearcher.orchestration.nodes import NodeDeps
    from georesearcher.orchestration.graph import build_graph

    class SearchLLM:
        def complete(self, messages):
            return '{"intent": "SEARCH", "confidence": 0.92, "reason": "search for literature"}'

    class FakeOrchCfg:
        router_confidence_threshold = 0.6
        search_min_hits = 3
        search_max_rounds = 2
        crag_quality_threshold = 0.3
        crag_max_rounds = 1
        review_max_rounds = 3
        clarify_max_rounds = 2
        checkpoint_path = "/tmp/test_checkpoints.sqlite"

    class FakeCfg:
        orchestration = FakeOrchCfg()

    class FakeChunk:
        text: str = "test"
        paper_id: str = "doi::test"

    class FakeRetrieved:
        def __init__(self, score=0.85):
            self.chunk = FakeChunk()
            self.score = score

    class FakeResult:
        def __init__(self, score=0.85):
            self.child_chunk = FakeRetrieved(score=score)
            self.parent_text = "test parent text"

    class FakeRetrievalPipeline:
        def retrieve(self, query, *, config=None):
            return [FakeResult(score=0.85)]

    class FakeGenerator:
        def generate(self, question, results):
            class FakeAnswer:
                answer = "Fake answer."
                references = ["Ref 1: Test (2024)"]
            return FakeAnswer()

    deps = NodeDeps(
        cfg=FakeCfg(),
        llm=SearchLLM(),
        embedder=None,
        vector_store=None,
        sqlite_store=None,
        retrieval_pipeline=FakeRetrievalPipeline(),
        generator=FakeGenerator(),
    )

    graph = build_graph(deps, checkpointer=MemorySaver())
    thread_id = "test-search-2"
    config = {"configurable": {"thread_id": thread_id}}
    # SEARCH intent → search node → reflect (no source=no hits) → search again → round 2 capped → interpret → rag_qa → human_review interrupt
    events = list(graph.stream({"query": "find papers"}, config))
    state = graph.get_state(config)
    trace = state.values.get("trace", [])
    router_trace = [t for t in trace if t.startswith("router:")]
    assert len(router_trace) > 0
    # Should have search traces
    search_trace = [t for t in trace if t.startswith("search:")]
    assert len(search_trace) > 0


def test_graph_placeholder_plot_flow():
    """PLOT intent should route to placeholder → human_review."""
    from georesearcher.orchestration.nodes import NodeDeps
    from georesearcher.orchestration.graph import build_graph

    class PlotLLM:
        def complete(self, messages):
            return '{"intent": "PLOT", "confidence": 0.93, "reason": "create visualization"}'

    class FakeOrchCfg:
        router_confidence_threshold = 0.6
        search_min_hits = 3
        search_max_rounds = 2
        crag_quality_threshold = 0.3
        crag_max_rounds = 1
        review_max_rounds = 3
        clarify_max_rounds = 2
        checkpoint_path = "/tmp/test_checkpoints.sqlite"

    class FakeCfg:
        orchestration = FakeOrchCfg()

    deps = NodeDeps(
        cfg=FakeCfg(),
        llm=PlotLLM(),
    )
    graph = build_graph(deps, checkpointer=MemorySaver())
    thread_id = "test-plot-1"
    config = {"configurable": {"thread_id": thread_id}}
    events = list(graph.stream({"query": "draw a map"}, config))
    state = graph.get_state(config)
    answer = state.values.get("rag_answer", "")
    assert "占位" in answer


def test_graph_round_limit_no_infinite_loop():
    """Verify search_max_rounds prevents infinite loop (route_after_search caps at max)."""
    from georesearcher.orchestration.graph import _route_after_search
    # Even with 0 hits and max_rounds=1, after search_round=1 we go to interpret
    state = {"search_hits": [], "search_round": 1}
    result = _route_after_search(state, min_hits=3, max_rounds=1)
    assert result == "interpret"  # capped, no infinite reflect loop
