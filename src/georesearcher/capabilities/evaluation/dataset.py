"""评估集加载与校验（M2 / T3）。

评估集是一个 JSON 数组，每个元素对应一条 EvalCase。
校验规则：question 非空、expected_paper_ids 非空（否则无法算检索指标）。
可选：传入 sqlite_store 时，校验 expected_paper_ids 里的 id 真实存在（不存在则 warn）。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from .schemas import EvalCase


class EvalSetError(ValueError):
    """评估集格式/校验错误。"""


def load_eval_set(
    path: str | Path,
    *,
    sqlite_store: Any | None = None,
) -> list[EvalCase]:
    """读 JSON（list[dict]）→ list[EvalCase]，逐条校验。

    Args:
        path: 评估集 JSON 路径。顶层必须是数组。
        sqlite_store: 可选。传入时校验 expected_paper_ids 在 papers 表真实存在，
                      不存在的 id 仅 warn（不报错，方便离线跑）。

    Raises:
        EvalSetError: 文件不存在、非数组、或某条 case 校验失败（会指明是哪条）。
    """
    p = Path(path)
    if not p.exists():
        raise EvalSetError(f"评估集文件不存在: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvalSetError(f"评估集 JSON 解析失败 ({p}): {e}") from e

    if not isinstance(raw, list):
        raise EvalSetError(f"评估集顶层必须是数组，实际是 {type(raw).__name__} ({p})")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        where = f"第 {i} 条 case"
        if not isinstance(item, dict):
            raise EvalSetError(f"{where} 不是对象: {item!r}")
        try:
            case = EvalCase.model_validate(item)
        except Exception as e:  # pydantic ValidationError 等
            raise EvalSetError(f"{where} 结构非法: {e}") from e

        if not case.question.strip():
            raise EvalSetError(f"{where} (id={case.id}) 的 question 为空")
        if not case.expected_paper_ids:
            raise EvalSetError(
                f"{where} (id={case.id}) 的 expected_paper_ids 为空，"
                f"无法计算检索指标"
            )
        if case.id in seen_ids:
            raise EvalSetError(f"{where} 的 id 重复: {case.id}")
        seen_ids.add(case.id)

        cases.append(case)

    if sqlite_store is not None:
        _warn_missing_papers(cases, sqlite_store)

    return cases


def _warn_missing_papers(cases: list[EvalCase], sqlite_store: Any) -> None:
    """校验 expected_paper_ids 在 SQLite 里真实存在，不存在则 warn。"""
    for case in cases:
        for pid in case.expected_paper_ids:
            try:
                paper = sqlite_store.get_paper(pid)
            except Exception:  # 存储层异常不应阻断加载
                paper = None
            if paper is None:
                warnings.warn(
                    f"case {case.id}: expected_paper_id {pid!r} 在 SQLite 中不存在",
                    stacklevel=2,
                )
