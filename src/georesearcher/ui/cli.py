"""CLI 入口（内核，UI 只是薄壳）。ADR-08。

M0 提供：
- version：打印版本
- doctor：健康检查，跑通"配置→存储层"空链路，不需要 API key / 重依赖

M1 新增：
- ingest：导入单篇 PDF 到知识库
- ask：基于知识库的 RAG 问答
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .. import __version__
from ..config import load_config

app = typer.Typer(add_completion=False, help="GeoResearcher — 科研 Copilot CLI")
console = Console()


@app.command()
def version():
    """打印版本号。"""
    console.print(f"GeoResearcher v{__version__}")


@app.command()
def doctor(config: str = typer.Option(None, help="配置文件路径，默认 config.yaml")):
    """健康检查：加载配置、初始化存储层、验证 M0 骨架可跑通。"""
    table = Table(title="GeoResearcher Doctor（M0 骨架自检）")
    table.add_column("检查项")
    table.add_column("结果")
    table.add_column("详情")

    # 1) 配置加载
    try:
        cfg = load_config(config)
        table.add_row("配置加载", "[green]OK[/green]", f"LLM={cfg.models.llm.model}, VS={cfg.storage.vector_store.backend}")
    except Exception as e:  # noqa: BLE001
        table.add_row("配置加载", "[red]FAIL[/red]", str(e))
        console.print(table)
        raise typer.Exit(1)

    # 2) SQLite 存储层
    try:
        from ..storage import get_sqlite_store

        store = get_sqlite_store(cfg)
        n = store.count_papers()
        store.close()
        table.add_row("SQLite", "[green]OK[/green]", f"当前文献数={n}")
    except Exception as e:  # noqa: BLE001
        table.add_row("SQLite", "[red]FAIL[/red]", str(e))

    # 3) 向量库后端可构造（不触发真正的模型/网络）
    try:
        from ..storage import get_vector_store

        vs = get_vector_store(cfg)
        table.add_row("VectorStore", "[green]OK[/green]", f"backend={type(vs).__name__}")
    except Exception as e:  # noqa: BLE001
        table.add_row("VectorStore", "[red]FAIL[/red]", str(e))

    # 4) 模型层可构造（不发请求；缺 key 只提示）
    try:
        from ..models import get_llm

        llm = get_llm(cfg)
        has_key = cfg.api_key(cfg.models.llm) is not None
        note = "已配置 API key" if has_key else "未配置 key（.env），仅骨架可用"
        table.add_row("LLM 接口", "[green]OK[/green]", f"{llm.model} — {note}")
    except Exception as e:  # noqa: BLE001
        table.add_row("LLM 接口", "[red]FAIL[/red]", str(e))

    # 5) Embedding 可构造
    try:
        from ..models import get_embedder

        emb = get_embedder(cfg)
        table.add_row("Embedding", "[green]OK[/green]", f"{emb.model_name} (device={cfg.models.embedding.device})")
    except Exception as e:  # noqa: BLE001
        table.add_row("Embedding", "[yellow]WARN[/yellow]", str(e))

    console.print(table)
    console.print("[bold green]M0 骨架自检完成。[/bold green]")


# ─── M1 命令 ─────────────────────────────────────────────────

@app.command()
def ingest(
    pdf_path: str = typer.Argument(..., help="PDF 文件路径"),
    config: str = typer.Option(None, help="配置文件路径"),
):
    """将单篇 PDF 导入知识库。

    流程：解析 → 三级分块 → 向量化 → 写入 Chroma + SQLite
    """
    from ..capabilities.ingest.pipeline import ingest_paper

    cfg = load_config(config)

    console.print(f"[bold]正在导入:[/bold] {pdf_path}")
    console.print("[dim]步骤 1/3: 解析 PDF（PyMuPDF）...[/dim]")

    try:
        paper = ingest_paper(pdf_path, cfg=cfg)
    except FileNotFoundError:
        console.print(f"[red]错误: 文件不存在 — {pdf_path}[/red]")
        raise typer.Exit(1)
    except RuntimeError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]导入成功![/green]")
    console.print(f"  Paper ID: {paper.id}")
    console.print(f"  标题:     {paper.title[:80]}{'...' if len(paper.title) > 80 else ''}")
    console.print(f"  作者:     {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
    if paper.doi:
        console.print(f"  DOI:      {paper.doi}")

    from ..storage import get_sqlite_store
    store = get_sqlite_store(cfg)
    total = store.count_papers()
    store.close()
    console.print(f"\n[dim]知识库共有 {total} 篇文献。[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="你想问的问题"),
    top_k: int = typer.Option(5, help="检索 chunk 数量"),
    no_hyde: bool = typer.Option(False, "--no-hyde", help="禁用 HyDE"),
    no_multi_query: bool = typer.Option(False, "--no-multi", help="禁用多查询"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="禁用重排序"),
    no_bm25: bool = typer.Option(False, "--no-bm25", help="禁用 BM25 稀疏检索"),
    no_ce: bool = typer.Option(False, "--no-ce", help="禁用交叉编码器重排序"),
    config: str = typer.Option(None, help="配置文件路径"),
):
    """基于知识库的 RAG 问答。

    流程：HyDE → 多查询 → [稠密+BM25]混合检索 → RRF融合 → 交叉编码器重排序 → 引用接地生成
    """
    from ..capabilities.rag.retriever import RetrievalConfig, RetrievalPipeline
    from ..capabilities.rag.generator import Generator

    cfg = load_config(config)

    console.print(f"[bold]问题:[/bold] {question}")
    console.print()

    # 1) 检索
    console.print("[dim]正在检索...[/dim]")
    retrieval_config = RetrievalConfig(
        top_k=top_k,
        use_hyde=not no_hyde,
        use_multi_query=not no_multi_query,
        use_reranker=not no_rerank,
        use_bm25=not no_bm25,
        use_cross_encoder=not no_ce,
    )
    pipeline = RetrievalPipeline(cfg=cfg)
    results = pipeline.retrieve(question, config=retrieval_config)

    if not results:
        console.print("[yellow]未找到相关文献片段。请先导入论文。[/yellow]")
        raise typer.Exit(0)

    console.print(f"[dim]找到 {len(results)} 个相关片段。[/dim]")

    # 2) 生成
    console.print("[dim]正在生成回答...[/dim]")
    console.print()
    generator = Generator(cfg=cfg)
    answer = generator.generate(question, results)

    # 3) 输出
    console.print(Panel.fit(
        answer.answer,
        title="回答",
        border_style="green",
    ))

    if answer.references:
        console.print()
        console.print("[bold]引用文献:[/bold]")
        for ref in answer.references:
            console.print(f"  {ref}")

    console.print(f"\n[dim]模型: {answer.model}[/dim]")


# ─── M2 命令：RAG 评估 ───────────────────────────────────────

@app.command(name="eval")
def eval_cmd(
    dataset: str = typer.Option(None, help="评估集路径，默认取 config.evaluation.eval_set_path"),
    top_k: int = typer.Option(None, help="检索 top_k，默认取 config"),
    no_generation: bool = typer.Option(False, "--no-generation", help="只评检索层，跳过生成层"),
    ragas: bool = typer.Option(False, "--ragas", help="额外跑 RAGAS 对照（需装 eval extra）"),
    config: str = typer.Option(None, "--config", help="配置文件路径"),
):
    """跑 RAG 分层评估 → 终端表格 + markdown/json 报告到 report_dir。"""
    from ..capabilities.evaluation import (
        RAGEvaluator,
        load_eval_set,
        render_terminal,
        write_json,
        write_markdown,
    )

    cfg = load_config(config)
    ecfg = cfg.evaluation
    ds_path = dataset or ecfg.eval_set_path
    k = top_k or ecfg.top_k
    eval_generation = ecfg.eval_generation and not no_generation

    console.print(f"[bold]加载评估集:[/bold] {ds_path}")
    try:
        cases = load_eval_set(ds_path)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]评估集加载失败: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[dim]{len(cases)} 条 case；生成层评估={'开' if eval_generation else '关'}[/dim]")

    console.print("[dim]正在评估（检索每条 case…）...[/dim]")
    evaluator = RAGEvaluator(cfg=cfg)
    report = evaluator.run(
        cases, top_k=k, eval_generation=eval_generation, run_ragas=ragas
    )

    render_terminal(report, console=console)

    report_dir = ecfg.report_dir or cfg.storage.files.report_dir
    md_path = write_markdown(report, report_dir)
    json_path = write_json(report, report_dir)
    console.print(f"[green]报告已写出:[/green]\n  {md_path}\n  {json_path}")


@app.command(name="eval-diff")
def eval_diff(
    run_a: str = typer.Argument(..., help="旧的 eval JSON 报告路径"),
    run_b: str = typer.Argument(..., help="新的 eval JSON 报告路径"),
):
    """对比两次 eval 的 JSON 结果，打印指标 diff（防跷跷板/回归）。"""
    import json

    def _load(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    try:
        a, b = _load(run_a), _load(run_b)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]读取报告失败: {e}[/red]")
        raise typer.Exit(1)

    table = Table(title="Eval Diff（B - A）")
    table.add_column("指标")
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("Δ", justify="right")

    def _row(label, va, vb):
        delta = (vb or 0) - (va or 0)
        arrow = "↑" if delta > 1e-9 else ("↓" if delta < -1e-9 else "＝")
        color = "green" if delta > 1e-9 else ("red" if delta < -1e-9 else "dim")
        table.add_row(label, f"{va:.3f}", f"{vb:.3f}", f"[{color}]{arrow} {delta:+.3f}[/{color}]")

    for key, label in [
        ("hit_rate", "Hit@k"), ("mrr", "MRR"), ("ndcg", "NDCG@k"),
        ("context_precision", "Context Precision"), ("context_recall", "Context Recall"),
    ]:
        _row(label, a["retrieval"].get(key, 0.0), b["retrieval"].get(key, 0.0))

    if a.get("generation") and b.get("generation"):
        for key, label in [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy")]:
            _row(label, a["generation"].get(key, 0.0), b["generation"].get(key, 0.0))

    console.print(table)


if __name__ == "__main__":
    app()
