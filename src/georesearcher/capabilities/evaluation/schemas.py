"""评估数据结构（M2）。

放在评估 package 内部，不污染 types.py（types.py 只放跨能力共享的核心结构）。
所有指标为 paper 级（决策 D2）。字段说明见 docs/plan-m2 §4 T2。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """一条评估用例：问题 + paper 级 ground truth。"""

    id: str  # 唯一 id，如 "q001"
    question: str
    expected_paper_ids: list[str]  # ground truth（paper 级，决策 D2）
    reference_answer: str | None = None  # 可选：生成层评估参考答案
    tags: list[str] = Field(default_factory=list)  # 可选：便于分层看指标


class RetrievalMetrics(BaseModel):
    """检索层聚合指标（对全部 case 取均值）。"""

    hit_rate: float  # 命中率（top_k 内是否有 GT paper）
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # NDCG@k
    context_precision: float  # top_k 去重 paper 中 GT 的占比
    context_recall: float  # GT paper 被检回的比例
    k: int  # 评估用的 top_k


class GenerationMetrics(BaseModel):
    """生成层聚合指标（自研 LLM-as-judge）。"""

    faithfulness: float  # 答案是否忠于检索上下文（0-1）
    answer_relevancy: float  # 答案是否切题（0-1）
    n_evaluated: int


class CaseResult(BaseModel):
    """单条 case 的明细（便于报告/debug）。"""

    case_id: str
    question: str
    retrieved_paper_ids: list[str]  # 实际检回（去重、按排名）
    expected_paper_ids: list[str]
    hit: bool
    reciprocal_rank: float
    answer: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None


class EvalReport(BaseModel):
    """一次评估运行的完整报告。"""

    timestamp: str  # ISO8601
    n_cases: int
    retrieval: RetrievalMetrics
    generation: GenerationMetrics | None = None  # 未跑生成层时为 None
    ragas: dict | None = None  # 可选 RAGAS 对照
    config_snapshot: dict = Field(default_factory=dict)  # 本次 retrieval 配置（可复现）
    cases: list[CaseResult] = Field(default_factory=list)
    diagnosis: str = ""  # "检索层弱还是生成层弱"的文字定位
