"""capabilities/evaluation — RAG 评估模块（M2）。

对外公共入口。evaluator/report 等在后续任务补齐后加入导出。
"""

from .dataset import EvalSetError, load_eval_set
from .evaluator import RAGEvaluator
from .generation_metrics import GenerationJudge, aggregate_generation
from .report import render_terminal, write_json, write_markdown
from .retrieval_metrics import aggregate_retrieval, dedup_paper_ranking
from .schemas import (
    CaseResult,
    EvalCase,
    EvalReport,
    GenerationMetrics,
    RetrievalMetrics,
)

__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "EvalSetError",
    "GenerationJudge",
    "GenerationMetrics",
    "RAGEvaluator",
    "RetrievalMetrics",
    "aggregate_generation",
    "aggregate_retrieval",
    "dedup_paper_ranking",
    "load_eval_set",
    "render_terminal",
    "write_json",
    "write_markdown",
]
