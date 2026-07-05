"""评估编排器（M2 / T8）——串起检索层 + 生成层评估，产出 EvalReport。

流程（每条 case）：
  1. retrieve_raw(question) → dedup_paper_ranking → 算检索指标（paper 级）。
  2. 若 eval_generation：retrieve(question)（含父块）→ generate → GenerationJudge
     算 faithfulness / answer_relevancy。
  3. 聚合检索/生成指标 → 生成 diagnosis 文字定位 → 返回 EvalReport。

diagnosis 阈值来自 config.evaluation.diagnosis（经验值，可覆盖）。
所有 LLM 调用走注入的 pipeline/generator/judge，测试可 mock，不真调网络。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ...config import Config, load_config
from .generation_metrics import GenerationJudge, aggregate_generation
from .retrieval_metrics import (
    aggregate_retrieval,
    context_precision,
    context_recall,
    dedup_paper_ranking,
    hit_rate,
    mrr,
    ndcg,
)
from .schemas import CaseResult, EvalCase, EvalReport

if TYPE_CHECKING:
    from ...models.llm import LLMClient
    from ..rag.generator import Generator
    from ..rag.retriever import RetrievalConfig, RetrievalPipeline


class RAGEvaluator:
    """RAG 分层评估编排器。"""

    def __init__(
        self,
        cfg: Config | None = None,
        pipeline: "RetrievalPipeline | None" = None,
        generator: "Generator | None" = None,
        judge: "LLMClient | None" = None,
    ):
        self._cfg = cfg or load_config()
        self._pipeline = pipeline  # 延迟构造：真跑时才需要
        self._generator = generator
        self._judge_client = judge

    # ── 依赖延迟构造（避免测试时加载重依赖）────────────────
    def _get_pipeline(self) -> "RetrievalPipeline":
        if self._pipeline is None:
            from ..rag.retriever import RetrievalPipeline

            self._pipeline = RetrievalPipeline(cfg=self._cfg)
        return self._pipeline

    def _get_generator(self) -> "Generator":
        if self._generator is None:
            from ..rag.generator import Generator

            self._generator = Generator(cfg=self._cfg)
        return self._generator

    def _get_judge(self) -> GenerationJudge:
        return GenerationJudge(judge=self._judge_client)

    def run(
        self,
        cases: list[EvalCase],
        *,
        top_k: int = 5,
        eval_generation: bool = True,
        run_ragas: bool = False,
        retrieval_config: "RetrievalConfig | None" = None,
    ) -> EvalReport:
        pipeline = self._get_pipeline()
        gen_judge = self._get_judge() if eval_generation else None
        generator = self._get_generator() if eval_generation else None

        per_retrieval: list[dict] = []
        per_generation: list[dict] = []
        case_results: list[CaseResult] = []

        for case in cases:
            # ── 检索层 ──
            raw = pipeline.retrieve_raw(case.question, config=retrieval_config)
            pred_papers = dedup_paper_ranking(raw)
            gold = case.expected_paper_ids

            m = {
                "hit_rate": hit_rate(pred_papers, gold, top_k),
                "mrr": mrr(pred_papers, gold, top_k),
                "ndcg": ndcg(pred_papers, gold, top_k),
                "context_precision": context_precision(pred_papers, gold, top_k),
                "context_recall": context_recall(pred_papers, gold, top_k),
            }
            per_retrieval.append(m)

            cr = CaseResult(
                case_id=case.id,
                question=case.question,
                retrieved_paper_ids=pred_papers[:top_k],
                expected_paper_ids=gold,
                hit=m["hit_rate"] > 0,
                reciprocal_rank=m["mrr"],
            )

            # ── 生成层（可选）──
            if eval_generation and generator is not None and gen_judge is not None:
                results = pipeline.retrieve(case.question, config=retrieval_config)
                answer = generator.generate(case.question, results)
                contexts = [
                    rr.parent_text if rr.parent_text else rr.child_chunk.chunk.text
                    for rr in results
                ]
                faith = gen_judge.faithfulness(answer.answer, contexts)
                rel = gen_judge.answer_relevancy(case.question, answer.answer)
                per_generation.append({"faithfulness": faith, "answer_relevancy": rel})
                cr.answer = answer.answer
                cr.faithfulness = faith
                cr.answer_relevancy = rel

            case_results.append(cr)

        # ── 聚合 ──
        retrieval_metrics = aggregate_retrieval(per_retrieval, k=top_k)
        generation_metrics = (
            aggregate_generation(per_generation) if eval_generation else None
        )

        # ── RAGAS 对照（可选，缺失优雅跳过）──
        ragas_result = None
        if run_ragas:
            ragas_result = self._maybe_run_ragas(case_results)

        diagnosis = self._diagnose(retrieval_metrics, generation_metrics)

        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_cases=len(cases),
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            ragas=ragas_result,
            config_snapshot=self._snapshot(top_k, eval_generation, retrieval_config),
            cases=case_results,
            diagnosis=diagnosis,
        )

    # ── diagnosis：检索 vs 生成 相对强弱定位（面试锚点）────────
    def _diagnose(self, retrieval, generation) -> str:
        d = self._cfg.evaluation.diagnosis
        parts: list[str] = []

        if retrieval.hit_rate < d.hit_rate_low:
            parts.append(
                f"检索层偏弱（hit_rate={retrieval.hit_rate:.2f} < {d.hit_rate_low}）："
                f"建议调分块粒度 / embedding 模型 / 混合检索权重 / HyDE / 多查询。"
            )
        else:
            parts.append(f"检索层达标（hit_rate={retrieval.hit_rate:.2f}）。")

        if generation is not None:
            if generation.faithfulness < d.faithfulness_low:
                parts.append(
                    f"生成层偏弱（faithfulness={generation.faithfulness:.2f} < "
                    f"{d.faithfulness_low}）：建议强化引用接地 / 缩短上下文 / 降温 / 改 prompt。"
                )
            else:
                parts.append(f"生成层达标（faithfulness={generation.faithfulness:.2f}）。")

        # 相对定位：谁更是瓶颈
        if generation is not None:
            if (
                retrieval.hit_rate >= d.hit_rate_low
                and generation.faithfulness < d.faithfulness_low
            ):
                parts.append("→ 瓶颈在生成层：检索能召回，但答案对上下文的忠实度不足。")
            elif (
                retrieval.hit_rate < d.hit_rate_low
                and generation.faithfulness >= d.faithfulness_low
            ):
                parts.append("→ 瓶颈在检索层：召回不足，生成层已尽力。")

        return " ".join(parts)

    def _snapshot(self, top_k, eval_generation, retrieval_config) -> dict:
        snap = {
            "top_k": top_k,
            "eval_generation": eval_generation,
            "judge_model": self._cfg.models.judge.model,
        }
        if retrieval_config is not None:
            # dataclass → dict（只存能序列化的标量）
            from dataclasses import asdict

            snap["retrieval_config"] = asdict(retrieval_config)
        return snap

    def _maybe_run_ragas(self, case_results) -> dict | None:
        try:
            from .ragas_adapter import run_ragas_crosscheck
        except ImportError:
            return {"status": "RAGAS adapter 不可用（未实现或未装 eval extra），跳过"}
        try:
            return run_ragas_crosscheck(case_results)
        except Exception as e:  # noqa: BLE001
            return {"status": f"RAGAS 运行失败，跳过：{e}"}
