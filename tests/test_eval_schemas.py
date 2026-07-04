"""T2: 评估数据结构 roundtrip 测试（纯 pydantic，无 LLM）。"""

from georesearcher.capabilities.evaluation.schemas import (
    CaseResult,
    EvalCase,
    EvalReport,
    GenerationMetrics,
    RetrievalMetrics,
)


def test_eval_case_roundtrip():
    c = EvalCase(id="q001", question="Q?", expected_paper_ids=["p1", "p2"])
    d = c.model_dump()
    assert EvalCase.model_validate(d) == c
    assert c.reference_answer is None
    assert c.tags == []


def test_eval_case_with_optionals():
    c = EvalCase(
        id="q002",
        question="Q2?",
        expected_paper_ids=["p3"],
        reference_answer="ref",
        tags=["方法", "教育"],
    )
    assert EvalCase.model_validate(c.model_dump()) == c


def test_retrieval_metrics_roundtrip():
    m = RetrievalMetrics(
        hit_rate=0.8, mrr=0.5, ndcg=0.6, context_precision=0.4, context_recall=0.9, k=5
    )
    assert RetrievalMetrics.model_validate(m.model_dump()) == m


def test_generation_metrics_roundtrip():
    m = GenerationMetrics(faithfulness=0.67, answer_relevancy=0.5, n_evaluated=3)
    assert GenerationMetrics.model_validate(m.model_dump()) == m


def test_case_result_roundtrip():
    r = CaseResult(
        case_id="q001",
        question="Q?",
        retrieved_paper_ids=["p1", "p9"],
        expected_paper_ids=["p1"],
        hit=True,
        reciprocal_rank=1.0,
    )
    assert CaseResult.model_validate(r.model_dump()) == r
    assert r.faithfulness is None


def test_eval_report_roundtrip():
    rep = EvalReport(
        timestamp="2026-07-05T00:00:00",
        n_cases=1,
        retrieval=RetrievalMetrics(
            hit_rate=1.0, mrr=1.0, ndcg=1.0, context_precision=1.0, context_recall=1.0, k=5
        ),
        generation=None,
        cases=[
            CaseResult(
                case_id="q001",
                question="Q?",
                retrieved_paper_ids=["p1"],
                expected_paper_ids=["p1"],
                hit=True,
                reciprocal_rank=1.0,
            )
        ],
        diagnosis="检索层良好",
    )
    dumped = rep.model_dump()
    assert EvalReport.model_validate(dumped) == rep
    assert rep.generation is None
    assert rep.ragas is None
    assert rep.config_snapshot == {}
