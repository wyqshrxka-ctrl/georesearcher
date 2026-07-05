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
    config: str = typer.Option(None, help="配置文件路径，默认 config.yaml"),
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


# ─── M3 命令：检索入库 + 解读归档 ──────────────────────────────


@app.command()
def search(
    query: str = typer.Argument(..., help="检索关键词"),
    limit: int = typer.Option(10, help="最多返回条数"),
    config: str = typer.Option(None, help="配置文件路径，默认 config.yaml"),
):
    """检索 OpenAlex 并打印候选文献（不入库）。"""
    from ..capabilities.search.openalex import OpenAlexSource

    cfg = load_config(config)
    scfg = cfg.search

    source = OpenAlexSource(
        mailto=scfg.mailto,
        rate_limit_per_sec=scfg.rate_limit_per_sec,
        timeout=scfg.timeout,
        max_retries=scfg.max_retries,
    )

    console.print(f"[bold]检索:[/bold] {query}")
    console.print("[dim]正在查询 OpenAlex...[/dim]")

    try:
        hits = source.search(query, limit=limit)
    except Exception as e:
        console.print(f"[red]检索失败: {e}[/red]")
        raise typer.Exit(1)

    if not hits:
        console.print("[yellow]未找到匹配的文献。[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"OpenAlex 检索结果（{len(hits)} 条）")
    table.add_column("#", justify="right")
    table.add_column("标题")
    table.add_column("年份", justify="right")
    table.add_column("来源")
    table.add_column("OA")

    for i, hit in enumerate(hits, 1):
        oa_icon = ":white_check_mark:" if hit.pdf_url else ":x:"
        title = hit.paper.title[:60] + ("..." if len(hit.paper.title) > 60 else "")
        table.add_row(
            str(i), title,
            str(hit.paper.year or "-"),
            hit.paper.venue or "-",
            oa_icon,
        )

    console.print(table)

    # Print details for each
    for hit in hits:
        console.print()
        console.print(f"[bold]{hit.paper.title}[/bold]")
        console.print(f"  ID:    {hit.paper.id}")
        console.print(f"  作者:  {', '.join(hit.paper.authors[:5])}{'...' if len(hit.paper.authors) > 5 else ''}")
        if hit.paper.year:
            console.print(f"  年份:  {hit.paper.year}")
        if hit.paper.venue:
            console.print(f"  来源:  {hit.paper.venue}")
        if hit.paper.doi:
            console.print(f"  DOI:   {hit.paper.doi}")
        console.print(f"  OA:    {hit.paper.oa_status} {'(有全文)' if hit.pdf_url else '(仅元数据)'}")
        if hit.abstract:
            abbr = hit.abstract[:200] + ("..." if len(hit.abstract) > 200 else "")
            console.print(f"  摘要:  {abbr}")


@app.command(name="ingest-search")
def ingest_search(
    query: str = typer.Argument(..., help="检索关键词"),
    limit: int = typer.Option(10, help="最多入库条数"),
    config: str = typer.Option(None, help="配置文件路径，默认 config.yaml"),
):
    """检索 + 判重 + 入库（全文或摘要旁路）。打印汇总。"""
    from ..capabilities.search.pipeline import ingest_from_search

    cfg = load_config(config)

    console.print(f"[bold]检索+入库:[/bold] {query}")
    console.print("[dim]正在检索 OpenAlex → 判重 → 入库...[/dim]")

    try:
        result = ingest_from_search(query, limit=limit, cfg=cfg)
    except Exception as e:
        console.print(f"[red]执行失败: {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold green]检索入库完成![/bold green]")
    console.print(f"  命中总数:   {result.total_hits}")
    console.print(f"  全文入库:   {len(result.ingested_full)}")
    console.print(f"  摘要入库:   {len(result.ingested_meta)}")
    console.print(f"  覆盖更新:   {len(result.updated_meta)}")
    console.print(f"  跳过已存在: {len(result.skipped_existing)}")
    console.print(f"  失败:       {len(result.failed)}")

    if result.failed:
        for f in result.failed:
            console.print(f"    [red]- {f['paper_id']}: {f['reason']}[/red]")

    from ..storage import get_sqlite_store
    store = get_sqlite_store(cfg)
    total = store.count_papers()
    store.close()
    console.print(f"\n[dim]知识库共有 {total} 篇文献。[/dim]")


@app.command()
def interpret(
    paper_id: str = typer.Argument(..., help="论文 ID（可用 search 命令查看）"),
    config: str = typer.Option(None, help="配置文件路径，默认 config.yaml"),
):
    """对指定论文生成结构化笔记（RQ/method/contribution/gap/findings）并落库。"""
    from ..capabilities.interpret.interpret import interpret_paper

    cfg = load_config(config)

    console.print(f"[bold]解读论文:[/bold] {paper_id}")
    console.print("[dim]正在调用 LLM 提取结构化笔记...[/dim]")

    try:
        note = interpret_paper(paper_id, cfg=cfg)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]解读失败: {e}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel.fit(note.research_question or "（未提及）", title="Research Question"))
    console.print(Panel.fit(note.method or "（未提及）", title="Method"))
    console.print(Panel.fit(note.contribution or "（未提及）", title="Contribution"))
    console.print(Panel.fit(note.gap or "（未提及）", title="Gap"))
    console.print(Panel.fit(note.key_findings or "（未提及）", title="Key Findings"))
    console.print(Panel.fit(note.summary or "（未提及）", title="Summary"))

    console.print("[green]结构化笔记已落库 (SQLite notes 表)。[/green]")


# ---------------------------------------------------------------------------
# M4: LangGraph 编排
# ---------------------------------------------------------------------------

@app.command()
def orchestrate(
    query: str = typer.Argument(None, help="研究主题或问题；--resume 时可省略"),
    resume: str = typer.Option(None, help="从 checkpoint 恢复的 thread_id（跨进程）"),
    feedback: str = typer.Option(None, help="--resume 时注入的人机反馈"),
    pause: bool = typer.Option(False, help="跑到人机中断即退出并打印 thread_id（演示持久化）"),
    config: str = typer.Option(None, help="配置文件路径，默认 config.yaml"),
):
    """运行 Agent 编排总控图。

    - 无 resume：新会话，stream 到 human_review 中断
        · 默认（无 --pause）：单命令内捕获中断 → input() 拿反馈 → Command(resume=...) 续跑
        · --pause：打印 answer + thread_id 后退出（状态存 SqliteSaver，可跨进程恢复）
    - 有 resume：全新进程，用 SqliteSaver 按 thread_id 恢复 → Command(resume=feedback) 续跑
    """
    import uuid
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    from ..config import load_config
    from ..orchestration import NodeDeps, build_graph
    from ..capabilities.rag.retriever import RetrievalConfig, RetrievalPipeline
    from ..capabilities.rag.generator import Generator

    cfg = load_config(config)

    # Build NodeDeps
    deps = NodeDeps(
        cfg=cfg,
        llm=None,  # lazily resolved by RetrievalPipeline / Generator / router
        embedder=None,
        vector_store=None,
        sqlite_store=None,
        retrieval_pipeline=None,
        generator=None,
        retrieval_cfg=RetrievalConfig(),
    )

    # Lazily build real deps (expensive imports)
    # Note: we defer heavy imports to avoid loading models unnecessarily
    from ..models.llm import get_llm, get_judge
    from ..models.embedding import get_embedder
    from ..storage.vector_store import get_vector_store
    from ..storage.sqlite_store import get_sqlite_store

    deps.llm = get_llm(cfg)
    deps.embedder = get_embedder(cfg)
    deps.vector_store = get_vector_store(cfg)
    deps.sqlite_store = get_sqlite_store(cfg)
    deps.retrieval_pipeline = RetrievalPipeline(cfg=cfg, llm=deps.llm, embedder=deps.embedder)
    deps.generator = Generator(cfg=cfg, llm=deps.llm)
    deps.retrieval_cfg = RetrievalConfig()

    thread_id = resume or str(uuid.uuid4())[:8]

    # Checkpointer: SqliteSaver for persistence (b+)
    import os
    ckpt_path = cfg.orchestration.checkpoint_path
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string(ckpt_path)

    graph = build_graph(deps, checkpointer=checkpointer)
    stream_config = {"configurable": {"thread_id": thread_id}}

    # --- Resume mode ---
    if resume:
        if not feedback:
            console.print("[red]--resume 必须配合 --feedback 使用[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]从 checkpoint 恢复:[/bold] thread_id={thread_id}")
        for event in graph.stream(Command(resume=feedback), stream_config):
            _print_event(event)
        console.print(f"\n[green]完成。thread_id={thread_id}（可再次 --resume）[/green]")
        return

    # --- New session ---
    if not query:
        console.print("[red]请提供查询主题[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]GeoResearcher Agent[/bold]")
    console.print(f"[dim]thread_id={thread_id}[/dim]")
    console.print()

    for event in graph.stream({"query": query}, stream_config):
        _print_event(event)

    # Handle interrupt for single-command resume (fluent demo)
    state = graph.get_state(stream_config)
    if state.next and not pause:
        # Interrupted at human_review
        task = state.tasks[0] if state.tasks else None
        if task and task.interrupts:
            interrupt_value = task.interrupts[0].value
            answer = interrupt_value.get("answer", "")
            citations = interrupt_value.get("citations", [])
            prompt_text = interrupt_value.get("prompt", "满意吗？回车通过，或输入追问让我重答")

            console.print()
            console.print("[bold]回答:[/bold]")
            console.print(Markdown(answer[:2000]))
            if citations:
                console.print("\n[bold]引用:[/bold]")
                for c in citations[:5]:
                    console.print(f"  [{c['index']}] {c['reference'][:120]}")
            console.print(f"\n[dim]{prompt_text}[/dim]")

            user_input = input("> ").strip()
            if user_input:
                console.print(f"[dim]收到反馈，正在重答...[/dim]")
                for event in graph.stream(Command(resume=user_input), stream_config):
                    _print_event(event)
                state = graph.get_state(stream_config)
                if not state.next:
                    # Reached END after review
                    pass

    # --- Pause mode (persistence demo) ---
    if pause and state.next:
        task = state.tasks[0] if state.tasks else None
        if task and task.interrupts:
            interrupt_value = task.interrupts[0].value
            answer = interrupt_value.get("answer", "")
            console.print()
            console.print(f"[bold yellow]已暂停（可跨进程恢复）[/bold yellow]")
            console.print(f"thread_id = [bold]{thread_id}[/bold]")
            console.print()
            console.print(f"恢复命令: [dim]uv run georesearcher --config {config or 'config.m4.yaml'} orchestrate --resume {thread_id} --feedback '你的追问'[/dim]")
            return

    # Final summary
    final_state = graph.get_state(stream_config)
    if final_state.values:
        trace = final_state.values.get("trace", [])
        if trace:
            console.print("\n[bold dim]执行轨迹:[/bold dim]")
            for t in trace:
                console.print(f"  [dim]{t}[/dim]")

    console.print(f"\n[green]完成。[/green]")


def _print_event(event: dict) -> None:
    """Print a graph stream event with rich formatting."""
    for node_name, node_output in event.items():
        if not node_output:
            continue
        trace_msgs = node_output.get("trace", [])
        for msg in trace_msgs:
            console.print(f"[dim][{node_name}][/dim] {msg}")
        error = node_output.get("error")
        if error:
            console.print(f"[red][{node_name}] ERROR: {error}[/red]")


if __name__ == "__main__":
    app()
