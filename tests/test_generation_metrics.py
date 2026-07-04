"""T5: 生成层 LLM-as-judge 测试。

mock judge.complete 返回固定文本，断言解析与聚合正确。不真调 DeepSeek。
"""

import math

from georesearcher.capabilities.evaluation.generation_metrics import (
    GenerationJudge,
    aggregate_generation,
    split_sentences,
)


class _FakeJudge:
    """按调用顺序返回预设输出；可设为抛异常。"""

    def __init__(self, outputs, raise_on_call=False):
        self._outputs = list(outputs)
        self._i = 0
        self.raise_on_call = raise_on_call
        self.calls = []

    @property
    def model(self):
        return "fake-judge"

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.raise_on_call:
            raise RuntimeError("boom")
        out = self._outputs[self._i % len(self._outputs)]
        self._i += 1
        return out


# ---------- split_sentences ----------

def test_split_sentences_mixed():
    s = split_sentences("这是第一句。This is second! 第三句？")
    assert len(s) == 3


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# ---------- faithfulness ----------

def test_faithfulness_two_of_three_supported():
    # 3 句，judge 返回 yes/no/yes → 2/3 ≈ 0.667
    judge = _FakeJudge(["yes", "no", "yes"])
    gj = GenerationJudge(judge=judge)
    score = gj.faithfulness("句子一。句子二。句子三。", ["some context"])
    assert math.isclose(score, 2 / 3)
    assert len(judge.calls) == 3
    # 坑3：temperature=0.0 传下去了
    assert judge.calls[0][1]["temperature"] == 0.0


def test_faithfulness_all_supported():
    judge = _FakeJudge(["Yes.", "yes", "YES"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness("a。b。c。", ["ctx"]) == 1.0


def test_faithfulness_empty_answer_returns_none():
    judge = _FakeJudge(["yes"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness("", ["ctx"]) is None


def test_faithfulness_all_unparseable_returns_none():
    # judge 全部返回垃圾 → 无可判句 → None（不崩）
    judge = _FakeJudge(["maybe", "???", "garbage"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness("a。b。c。", ["ctx"]) is None


def test_faithfulness_partial_unparseable_ignored():
    # 3 句：yes / 垃圾 / no → 只 2 句可判，1 支持 → 0.5
    judge = _FakeJudge(["yes", "hmm", "no"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness("a。b。c。", ["ctx"]) == 0.5


def test_faithfulness_judge_exception_returns_none():
    judge = _FakeJudge(["yes"], raise_on_call=True)
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness("a。b。", ["ctx"]) is None


def test_faithfulness_truncates_context():
    judge = _FakeJudge(["yes"])
    gj = GenerationJudge(judge=judge)
    long_ctx = "z" * 5000  # 用 z 避免与 prompt 模板里的字母冲突
    gj.faithfulness("aaa bbb ccc", [long_ctx])
    prompt = judge.calls[0][0]
    # 上下文被截断到 1500 → prompt 里恰好 1500 个 z（坑2：长度一致，非全部 5000）
    assert prompt.count("z") == 1500


# ---------- answer_relevancy ----------

def test_relevancy_yes():
    gj = GenerationJudge(judge=_FakeJudge(["yes"]))
    assert gj.answer_relevancy("Q?", "full answer") == 1.0


def test_relevancy_partial():
    gj = GenerationJudge(judge=_FakeJudge(["partial"]))
    assert gj.answer_relevancy("Q?", "half answer") == 0.5


def test_relevancy_no():
    gj = GenerationJudge(judge=_FakeJudge(["no"]))
    assert gj.answer_relevancy("Q?", "off topic") == 0.0


def test_relevancy_empty_answer():
    gj = GenerationJudge(judge=_FakeJudge(["yes"]))
    assert gj.answer_relevancy("Q?", "") == 0.0


def test_relevancy_unparseable_returns_none():
    gj = GenerationJudge(judge=_FakeJudge(["dunno"]))
    assert gj.answer_relevancy("Q?", "answer") is None


def test_relevancy_exception_returns_none():
    gj = GenerationJudge(judge=_FakeJudge(["yes"], raise_on_call=True))
    assert gj.answer_relevancy("Q?", "answer") is None


# ---------- aggregate_generation ----------

def test_aggregate_ignores_none():
    per_case = [
        {"faithfulness": 1.0, "answer_relevancy": 1.0},
        {"faithfulness": 0.5, "answer_relevancy": None},
        {"faithfulness": None, "answer_relevancy": 0.0},
    ]
    m = aggregate_generation(per_case)
    # faithfulness: (1.0+0.5)/2 = 0.75
    assert m.faithfulness == 0.75
    # relevancy: (1.0+0.0)/2 = 0.5
    assert m.answer_relevancy == 0.5
    # 3 case 都至少有一个指标非 None
    assert m.n_evaluated == 3


def test_aggregate_all_none():
    per_case = [{"faithfulness": None, "answer_relevancy": None}]
    m = aggregate_generation(per_case)
    assert m.faithfulness == 0.0
    assert m.answer_relevancy == 0.0
    assert m.n_evaluated == 0


def test_aggregate_empty():
    m = aggregate_generation([])
    assert m.n_evaluated == 0
