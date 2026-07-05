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

# faithfulness 判定时给 judge 的上下文上限。faithfulness 需要"能否找到支持证据"，
# 截太短会漏证据把分数压低（实测缺陷来源），故给足；仅在极端超长时兜底截断。
_FAITH_CTX_MAX = 12000
# relevancy 不看上下文，无需此常量。保留旧名兼容（answer_relevancy 未用）。
_CTX_TRUNC = 1500

# 逐句切分（中英文句末标点）。简单可控，避免引入 nltk 等新依赖。
_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")

# ── 非事实断言句过滤（faithfulness 只该评"可被上下文支持的事实断言"）──
# 参考文献区块标题：命中后其后所有内容都不算断言（APA 列表逐条判必然 no，会虚压分数）。
_REFERENCES_HEADER = re.compile(r"^\**\s*(参考文献|references|引用文献|bibliography)\s*\**\s*$", re.I)
# 纯结构/元话术句（不携带可核查事实）：开头是这些连接词且很短。
_BOILERPLATE = re.compile(
    r"^(综上所述|总的来说|总而言之|首先|其次|再次|此外|另外|最后|具体而言|"
    r"根据(提供的|上述|以上)?文献|基于(提供的|上述|以上)?(文献|上下文)|"
    r"in summary|in conclusion|overall|first|second|finally|moreover|furthermore)"
    r"[，,：:、\s]*$",
    re.I,
)
# 疑似单条参考文献（含 年份括号 + DOI/期刊斜体等）——逐条判会误伤。
_LOOKS_LIKE_CITATION = re.compile(r"\(\d{4}\)|https?://|doi\.org|\*[^*]+\*")

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


def _strip_references_section(text: str) -> str:
    """截掉"参考文献/References"标题及其之后的全部内容。

    APA 引用列表逐条送去判 faithfulness 必然被判 no（上下文里没有引用格式），
    会把分母撑大、分数虚低。它们不是答案的事实断言，应排除。
    """
    lines = (text or "").splitlines()
    kept: list[str] = []
    for line in lines:
        if _REFERENCES_HEADER.match(line.strip()):
            break
        kept.append(line)
    return "\n".join(kept)


def claim_sentences(answer: str) -> list[str]:
    """从答案抽取"可被上下文支持的事实断言"句，供 faithfulness 逐句判定。

    过滤：① 参考文献区块；② 纯结构/元话术句（"综上所述"等）；
    ③ 疑似单条引用（含年份括号/DOI/斜体期刊名）。
    """
    body = _strip_references_section(answer)
    out: list[str] = []
    for s in split_sentences(body):
        if _BOILERPLATE.match(s):
            continue
        if _LOOKS_LIKE_CITATION.search(s):
            continue
        # 太短（如残留的 "首先"）也跳过
        if len(s) < 4:
            continue
        out.append(s)
    return out


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

    def _join_context(self, contexts: list[str]) -> str:
        """拼接检索上下文；仅在极端超长时兜底截断（faithfulness 需要足够证据）。"""
        joined = "\n\n".join(c for c in contexts if c)
        return joined[:_FAITH_CTX_MAX]

    def faithfulness(self, answer: str, contexts: list[str]) -> float | None:
        """忠实度 = 被上下文支持的事实断言句数 / 事实断言句总数。

        只评"可被上下文支持的事实断言"（过滤参考文献列表、结构话术、单条引用），
        避免这些非断言句把分数虚压（实测缺陷来源）。
        judge 拿到的是**完整检索上下文**（不做 1500 截断），保证支持证据不被漏掉。
        无可判句子或全部解析失败 → 返回 None（缺失，不计入均值）。
        """
        sentences = claim_sentences(answer)
        if not sentences:
            return None
        context = self._join_context(contexts)

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
