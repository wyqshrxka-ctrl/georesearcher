"""T8: RAGEvaluator 编排器测试（全 mock，不真调网络/模型）。"""

from georesearcher.capabilities.evaluation.evaluator import RAGEvaluator
from georesearcher.capabilities.evaluation.schemas import EvalCase, EvalReport
from georesearcher.types import Chunk, Retrieved


def _hit(paper_id, score=1.0):
    return Retrieved(
        chunk=Chunk(id=f"{paper_id}::c", paper_id=paper_id, text=f"text of {paper_id}"),
        score=score,
    )


class _FakePipeline:
    """按 question 返回预设 raw 命中。"""

    def __init__(self, mapping):
        self._mapping = mapping  # question -> list[Retrieved]

    def retrieve_raw(self, query, *, config=None, skip_cache=False):
        return self._mapping.get(query, [])

    def retrieve(self, query, *, config=None, skip_cache=False):
        # 复用 raw 包成 RetrievalResult（父块 = chunk text）
        from georesearcher.capabilities.rag.retriever import RetrievalResult

        return [RetrievalResult(child_chunk=r, parent_text=r.chunk.text) for r in self._mapping.get(query, [])]


class _FakeGenerator:
    def generate(self, question, results):
        from georesearcher.capabilities.rag.generator import GeneratedAnswer

        return GeneratedAnswer(answer="这是一个基于上下文的答案。", model="fake-llm")


class _FakeJudge:
    """总说 yes（faithfulness=1）/ yes（relevancy=1）。"""

    @property
    def model(self):
        return "fake-judge"

    def complete(self, prompt, **kwargs):
        return "yes"


def _cfg():
    from georesearcher.config import load_config

    return load_config()


def test_run_retrieval_only():
    cases = [
        EvalCase(id="q1", question="Q1", expected_paper_ids=["p1"]),
        EvalCase(id="q2", question="Q2", expected_paper_ids=["p2", "p3"]),
    ]
    pipe = _FakePipeline({
        "Q1": [_hit("p1"), _hit("p9")],           # 命中 p1 于第 1 位
        "Q2": [_hit("p9"), _hit("p2"), _hit("p8")],  # 命中 p2 于第 2 位（gold p2,p3）
    })
    ev = RAGEvaluator(cfg=_cfg(), pipeline=pipe)
    report = ev.run(cases, top_k=5, eval_generation=False)

    assert isinstance(report, EvalReport)
    assert report.n_cases == 2
    assert report.generation is None
    assert report.diagnosis  # 非空
    # q1 hit=1.0, q2 hit=1.0 → 平均 hit_rate=1.0
    assert report.retrieval.hit_rate == 1.0
    # q1 mrr=1.0, q2 mrr=0.5 → 0.75
    assert report.retrieval.mrr == 0.75
    # q2 recall: gold 2 篇命中 1 → 0.5；q1 recall=1.0 → 平均 0.75
    assert report.retrieval.context_recall == 0.75
    assert len(report.cases) == 2
    assert report.cases[0].hit is True


def test_run_with_generation():
    cases = [EvalCase(id="q1", question="Q1", expected_paper_ids=["p1"])]
    pipe = _FakePipeline({"Q1": [_hit("p1")]})
    ev = RAGEvaluator(
        cfg=_cfg(), pipeline=pipe, generator=_FakeGenerator(), judge=_FakeJudge()
    )
    report = ev.run(cases, top_k=5, eval_generation=True)

    assert report.generation is not None
    assert report.generation.faithfulness == 1.0  # judge 全说 yes
    assert report.generation.answer_relevancy == 1.0
    assert report.cases[0].answer is not None
    assert report.cases[0].faithfulness == 1.0


def test_diagnosis_flags_weak_retrieval():
    # 全部未命中 → hit_rate=0 < 阈值 → 诊断应指出检索层弱
    cases = [EvalCase(id="q1", question="Q1", expected_paper_ids=["pX"])]
    pipe = _FakePipeline({"Q1": [_hit("p1"), _hit("p2")]})
    ev = RAGEvaluator(cfg=_cfg(), pipeline=pipe)
    report = ev.run(cases, top_k=5, eval_generation=False)
    assert report.retrieval.hit_rate == 0.0
    assert "检索层偏弱" in report.diagnosis


def test_config_snapshot_present():
    cases = [EvalCase(id="q1", question="Q1", expected_paper_ids=["p1"])]
    pipe = _FakePipeline({"Q1": [_hit("p1")]})
    ev = RAGEvaluator(cfg=_cfg(), pipeline=pipe)
    report = ev.run(cases, top_k=3, eval_generation=False)
    assert report.config_snapshot["top_k"] == 3
    assert report.config_snapshot["eval_generation"] is False
    assert "judge_model" in report.config_snapshot
