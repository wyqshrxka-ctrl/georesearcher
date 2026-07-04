"""生成层指标（M2 / T5）——自研 LLM-as-judge。

为什么自研（决策 D3）：可控、能配 DeepSeek、能针对性处理 LLM-as-judge 三个坑
（design §5.2）。RAGAS 作为可选对照放 T7。

两个指标：
- **Faithfulness（忠实度）**：答案的每个句子是否被检索上下文支持。
  逐句让 judge 判 yes/no，忠实度 = 支持句数 / 总句数。
- **Answer Relevancy（答案相关性）**：答案是否直接、完整回答了问题。
  judge 判 yes / partial / no → 1.0 / 0.5 / 0.0。

LLM-as-judge 三个坑的对策（面试要讲）：
  坑1 自我偏好偏差 → judge model 可配置（config.models.judge），验证期切强模型交叉。
  坑2 位置/长度偏差 → 单条评估、上下文统一截断（_CTX_TRUNC，复用 M1 的 1500 字符）。
  坑3 评分不一致 → judge temperature=0.0（默认）、用 yes/no 二元判断而非 1-5 主观分。

健壮性（学 T0/T1 教训）：judge 输出解析失败 / 调用异常时，该项记为 None（缺失），
不计入均值，绝不让整轮评估崩。
"""

from __future__ import annotations

import re

from ...models.llm import LLMClient, get_judge
from .schemas import GenerationMetrics

# 坑2：上下文统一截断长度，与 M1 生成阶段保持一致，避免长度偏差。
_CTX_TRUNC = 1500

# 逐句切分（中英文句末标点）。简单可控，避免引入 nltk 等新依赖。
_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")

_FAITHFULNESS_PROMPT = """You are a strict fact-checker. Decide whether the CLAIM is \
directly supported by the CONTEXT below. Answer with exactly one word: "yes" or "no".

CONTEXT:
{context}

CLAIM:
{claim}

Answer (yes/no):"""

_RELEVANCY_PROMPT = """You are evaluating whether an ANSWER addresses a QUESTION. \
Answer with exactly one word:
- "yes" if the answer directly and completely addresses the question,
- "partial" if it only partially addresses it,
- "no" if it does not address the question.

QUESTION:
{question}

ANSWER:
{answer}

Answer (yes/partial/no):"""


def split_sentences(text: str) -> list[str]:
    """把答案切成句子（去空白空句）。"""
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "")]
    return [s for s in parts if s]


def _parse_yes_no(raw: str) -> bool | None:
    """从 judge 输出解析 yes/no。解析不出返回 None（缺失，不计入）。"""
    if not raw:
        return None
    low = raw.strip().lower()
    # 取首个明确 token；容忍 "Yes." / "answer: no" 等杂讯
    if re.search(r"\byes\b", low):
        return True
    if re.search(r"\bno\b", low):
        return False
    return None


def _parse_relevancy(raw: str) -> float | None:
    """解析 yes/partial/no → 1.0 / 0.5 / 0.0；解析失败返回 None。"""
    if not raw:
        return None
    low = raw.strip().lower()
    if re.search(r"\bpartial\b", low):
        return 0.5
    if re.search(r"\byes\b", low):
        return 1.0
    if re.search(r"\bno\b", low):
        return 0.0
    return None


class GenerationJudge:
    """自研 LLM-as-judge，走 config 里的 judge model。"""

    def __init__(self, judge: LLMClient | None = None):
        self._judge = judge if judge is not None else get_judge()

    @property
    def model(self) -> str:
        return self._judge.model

    def _truncate_context(self, contexts: list[str]) -> str:
        """拼接并统一截断上下文（坑2：长度一致）。"""
        joined = "\n\n".join(c for c in contexts if c)
        return joined[:_CTX_TRUNC]

    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        """忠实度 = 被上下文支持的句子数 / 总句子数。

        无可判句子或全部解析失败 → 返回 None（缺失，不计入均值）。
        """
        sentences = split_sentences(answer)
        if not sentences:
            return None
        context = self._truncate_context(contexts)

        supported = 0
        judged = 0
        for sent in sentences:
            prompt = _FAITHFULNESS_PROMPT.format(context=context, claim=sent)
            try:
                raw = self._judge.complete(prompt, temperature=0.0)
            except Exception:
                continue  # 单句失败不计入，别让整轮崩
            verdict = _parse_yes_no(raw)
            if verdict is None:
                continue
            judged += 1
            if verdict:
                supported += 1

        if judged == 0:
            return None
        return supported / judged

    def answer_relevancy(self, question: str, answer: str) -> float | None:
        """答案相关性：judge 判 yes/partial/no → 1.0/0.5/0.0。"""
        if not (answer or "").strip():
            return 0.0
        prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
        try:
            raw = self._judge.complete(prompt, temperature=0.0)
        except Exception:
            return None
        return _parse_relevancy(raw)


def aggregate_generation(per_case: list[dict]) -> GenerationMetrics:
    """对所有 case 的生成指标取均值（None 不计入分母）。

    per_case 每项含键 faithfulness / answer_relevancy（可为 None）。
    """
    faith_vals = [c["faithfulness"] for c in per_case if c.get("faithfulness") is not None]
    rel_vals = [c["answer_relevancy"] for c in per_case if c.get("answer_relevancy") is not None]

    faithfulness = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0
    relevancy = sum(rel_vals) / len(rel_vals) if rel_vals else 0.0
    # n_evaluated：至少有一个指标非缺失的 case 数
    n_eval = sum(
        1
        for c in per_case
        if c.get("faithfulness") is not None or c.get("answer_relevancy") is not None
    )

    return GenerationMetrics(
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        n_evaluated=n_eval,
    )
