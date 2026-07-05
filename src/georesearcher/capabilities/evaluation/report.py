"""报告渲染（M2 / T9）——终端表格 + markdown + JSON 落盘。

文件名 eval--{timestamp}.md / .json，存 config.files.report_dir（默认 data/reports/）。
终端用 rich.Table，与现有 CLI 风格一致。
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .schemas import EvalReport

_console = Console()


def _ts_slug(timestamp: str) -> str:
    """把 ISO8601 时间戳转成安全文件名片段。"""
    return timestamp.replace(":", "").replace("-", "").replace(".", "").replace("+", "z")


def render_terminal(report: EvalReport, console: Console | None = None) -> None:
    """rich 表格打印分层指标 + 诊断。"""
    c = console or _console

    c.print(f"\n[bold]RAG 评估报告[/bold]  （{report.n_cases} 条 case，{report.timestamp}）")
    c.print(f"[dim]judge model = {report.config_snapshot.get('judge_model', '?')}，"
            f"top_k = {report.config_snapshot.get('top_k', '?')}[/dim]")

    # 检索层
    rt = Table(title="检索层指标", show_header=True)
    rt.add_column("指标")
    rt.add_column("值", justify="right")
    r = report.retrieval
    rt.add_row("Hit@k", f"{r.hit_rate:.3f}")
    rt.add_row("MRR", f"{r.mrr:.3f}")
    rt.add_row("NDCG@k", f"{r.ndcg:.3f}")
    rt.add_row("Context Precision", f"{r.context_precision:.3f}")
    rt.add_row("Context Recall", f"{r.context_recall:.3f}")
    rt.add_row("k", str(r.k))
    c.print(rt)

    # 生成层
    if report.generation is not None:
        gt = Table(title="生成层指标（LLM-as-judge）", show_header=True)
        gt.add_column("指标")
        gt.add_column("值", justify="right")
        g = report.generation
        gt.add_row("Faithfulness", f"{g.faithfulness:.3f}")
        gt.add_row("Answer Relevancy", f"{g.answer_relevancy:.3f}")
        gt.add_row("已评估条数", str(g.n_evaluated))
        c.print(gt)
    else:
        c.print("[dim]（未评估生成层：--no-generation）[/dim]")

    if report.ragas:
        c.print(f"[dim]RAGAS 对照：{report.ragas}[/dim]")

    c.print(f"\n[bold yellow]诊断：[/bold yellow]{report.diagnosis}\n")


def _markdown(report: EvalReport) -> str:
    r = report.retrieval
    lines = [
        f"# RAG 评估报告",
        "",
        f"- 时间：{report.timestamp}",
        f"- Case 数：{report.n_cases}",
        f"- Judge model：{report.config_snapshot.get('judge_model', '?')}",
        f"- 配置快照：`{json.dumps(report.config_snapshot, ensure_ascii=False)}`",
        "",
        "## 检索层指标",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| Hit@{r.k} | {r.hit_rate:.3f} |",
        f"| MRR | {r.mrr:.3f} |",
        f"| NDCG@{r.k} | {r.ndcg:.3f} |",
        f"| Context Precision | {r.context_precision:.3f} |",
        f"| Context Recall | {r.context_recall:.3f} |",
        "",
    ]

    if report.generation is not None:
        g = report.generation
        lines += [
            "## 生成层指标（LLM-as-judge）",
            "",
            "| 指标 | 值 |",
            "| --- | --- |",
            f"| Faithfulness | {g.faithfulness:.3f} |",
            f"| Answer Relevancy | {g.answer_relevancy:.3f} |",
            f"| 已评估条数 | {g.n_evaluated} |",
            "",
        ]

    if report.ragas:
        lines += ["## RAGAS 对照", "", f"```\n{json.dumps(report.ragas, ensure_ascii=False, indent=2)}\n```", ""]

    lines += ["## 诊断", "", report.diagnosis, "", "## 逐条明细", ""]
    lines += ["| case | hit | RR | faithful | relevancy | 检回(去重) | 期望 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for cr in report.cases:
        faith = f"{cr.faithfulness:.2f}" if cr.faithfulness is not None else "-"
        rel = f"{cr.answer_relevancy:.2f}" if cr.answer_relevancy is not None else "-"
        retrieved = ", ".join(cr.retrieved_paper_ids[:3]) + ("…" if len(cr.retrieved_paper_ids) > 3 else "")
        expected = ", ".join(cr.expected_paper_ids)
        lines.append(
            f"| {cr.case_id} | {'✅' if cr.hit else '❌'} | {cr.reciprocal_rank:.2f} | "
            f"{faith} | {rel} | {retrieved} | {expected} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(report: EvalReport, out_dir: str) -> str:
    """写 markdown 报告，返回文件路径。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"eval--{_ts_slug(report.timestamp)}.md"
    path.write_text(_markdown(report), encoding="utf-8")
    return str(path)


def write_json(report: EvalReport, out_dir: str) -> str:
    """写 JSON 报告（可被 eval-diff 读回），返回文件路径。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"eval--{_ts_slug(report.timestamp)}.json"
    path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(path)
