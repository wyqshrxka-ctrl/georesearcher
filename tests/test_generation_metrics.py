"""T5: 生成层 LLM-as-judge 测试。

mock judge.complete 返回固定文本，断言解析与聚合正确。不真调 DeepSeek。
"""

import math

from georesearcher.capabilities.evaluation.generation_metrics import (
    GenerationJudge,
    aggregate_generation,
    claim_sentences,
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


# ---------- split_sentences / claim_sentences ----------

def test_split_sentences_mixed():
    s = split_sentences("这是第一句。This is second! 第三句？")
    assert len(s) == 3


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_claim_sentences_drops_references_section():
    answer = (
        "流动儿童被分到较差学校。优质学校学位更少。\n\n"
        "**参考文献**\n"
        "张玉清, & Chung, H. (2025). 某标题. *某期刊*. https://doi.org/10.1/x\n"
    )
    claims = claim_sentences(answer)
    # 只保留 2 条事实断言，参考文献区块整段被截掉
    assert len(claims) == 2
    assert all("参考文献" not in c and "doi.org" not in c for c in claims)


def test_claim_sentences_filters_boilerplate():
    answer = "综上所述，流动儿童被分到较差学校。首先，优质学校学位更少。"
    claims = claim_sentences(answer)
    # "综上所述，" 和 "首先，" 这种结构话术开头的短句被过滤，只留真正断言
    assert any("流动儿童" in c for c in claims)
    assert not any(c.startswith("综上所述") and len(c) < 6 for c in claims)


def test_claim_sentences_filters_inline_citation_line():
    answer = "流动儿童被分到较差学校。Smith (2020) 提出了某理论。"
    claims = claim_sentences(answer)
    # 含 (2020) 的疑似引用句被过滤
    assert not any("(2020)" in c for c in claims)


def test_claim_sentences_empty():
    assert claim_sentences("") == []
    assert claim_sentences("**参考文献**\nSmith (2020).") == []


# ---------- faithfulness ----------

# 事实断言句需 >=4 字符且非结构话术/引用，才会被送去判定。
_S1, _S2, _S3 = "流动儿童被分到较差学校。", "优质学校学位更少。", "择校政策效果不明确。"


def test_faithfulness_two_of_three_supported():
    # 3 句，judge 返回 yes/no/yes → 2/3 ≈ 0.667
    judge = _FakeJudge(["yes", "no", "yes"])
    gj = GenerationJudge(judge=judge)
    score = gj.faithfulness(_S1 + _S2 + _S3, ["some context"])
    assert math.isclose(score, 2 / 3)
    assert len(judge.calls) == 3
    # 坑3：temperature=0.0 传下去了
    assert judge.calls[0][1]["temperature"] == 0.0


def test_faithfulness_all_supported():
    judge = _FakeJudge(["Yes.", "yes", "YES"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness(_S1 + _S2 + _S3, ["ctx"]) == 1.0


def test_faithfulness_empty_answer_returns_none():
    judge = _FakeJudge(["yes"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness("", ["ctx"]) is None


def test_faithfulness_all_unparseable_returns_none():
    # judge 全部返回垃圾 → 无可判句 → None（不崩）
    judge = _FakeJudge(["maybe", "???", "garbage"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness(_S1 + _S2 + _S3, ["ctx"]) is None


def test_faithfulness_partial_unparseable_ignored():
    # 3 句：yes / 垃圾 / no → 只 2 句可判，1 支持 → 0.5
    judge = _FakeJudge(["yes", "hmm", "no"])
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness(_S1 + _S2 + _S3, ["ctx"]) == 0.5


def test_faithfulness_judge_exception_returns_none():
    judge = _FakeJudge(["yes"], raise_on_call=True)
    gj = GenerationJudge(judge=judge)
    assert gj.faithfulness(_S1 + _S2, ["ctx"]) is None


def test_faithfulness_gives_full_context_not_truncated():
    # 缺陷修复：faithfulness 判定应给完整上下文（不截到 1500），保证支持证据不被漏。
    judge = _FakeJudge(["yes"])
    gj = GenerationJudge(judge=judge)
    long_ctx = "z" * 5000
    gj.faithfulness(_S1, [long_ctx])
    prompt = judge.calls[0][0]
    # 5000 个 z 应全部进入 prompt（远大于旧的 1500 截断）
    assert prompt.count("z") == 5000


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
