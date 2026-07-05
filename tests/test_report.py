"""T9: 报告渲染测试（终端不报错 + markdown/json 落盘 tmp_path）。"""

import json

from rich.console import Console

from georesearcher.capabilities.evaluation.report import (
    render_terminal,
    write_json,
    write_markdown,
)
from georesearcher.capabilities.evaluation.schemas import (
    CaseResult,
    EvalReport,
    GenerationMetrics,
    RetrievalMetrics,
)


def _report(with_generation=True):
    return EvalReport(
        timestamp="2026-07-05T12:34:56.789+00:00",
        n_cases=2,
        retrieval=RetrievalMetrics(
            hit_rate=0.75, mrr=0.6, ndcg=0.7, context_precision=0.4, context_recall=0.66, k=5
        ),
        generation=GenerationMetrics(faithfulness=0.8, answer_relevancy=0.9, n_evaluated=2)
        if with_generation
        else None,
        config_snapshot={"top_k": 5, "judge_model": "deepseek-chat", "eval_generation": with_generation},
        cases=[
            CaseResult(
                case_id="q1", question="Q1", retrieved_paper_ids=["p1", "p9"],
                expected_paper_ids=["p1"], hit=True, reciprocal_rank=1.0,
                answer="ans" if with_generation else None,
                faithfulness=0.8 if with_generation else None,
                answer_relevancy=0.9 if with_generation else None,
            ),
            CaseResult(
                case_id="q2", question="Q2", retrieved_paper_ids=["p8"],
                expected_paper_ids=["p2", "p3"], hit=False, reciprocal_rank=0.0,
            ),
        ],
        diagnosis="检索层达标。生成层达标。",
    )


def test_render_terminal_no_error():
    render_terminal(_report(), console=Console(file=open("/dev/null", "w")))
    render_terminal(_report(with_generation=False), console=Console(file=open("/dev/null", "w")))


def test_write_markdown(tmp_path):
    path = write_markdown(_report(), str(tmp_path))
    assert path.endswith(".md")
    content = (tmp_path / path.split("/")[-1]).read_text(encoding="utf-8")
    assert "RAG 评估报告" in content
    assert "检索层指标" in content
    assert "生成层指标" in content
    assert "诊断" in content
    assert "q1" in content and "q2" in content


def test_write_markdown_no_generation(tmp_path):
    path = write_markdown(_report(with_generation=False), str(tmp_path))
    content = open(path, encoding="utf-8").read()
    assert "检索层指标" in content
    assert "生成层指标" not in content  # 未评估生成层


def test_write_json_roundtrip(tmp_path):
    path = write_json(_report(), str(tmp_path))
    assert path.endswith(".json")
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["n_cases"] == 2
    assert data["retrieval"]["hit_rate"] == 0.75
    assert data["generation"]["faithfulness"] == 0.8
    # 能被 EvalReport 读回
    EvalReport.model_validate(data)


def test_filename_slug_is_safe(tmp_path):
    path = write_json(_report(), str(tmp_path))
    name = path.split("/")[-1]
    assert ":" not in name and "+" not in name
