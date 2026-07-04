"""CLI 入口（内核，UI 只是薄壳）。ADR-08。

M0 提供：
- version：打印版本
- doctor：健康检查，跑通"配置→存储层"空链路，不需要 API key / 重依赖
"""
from __future__ import annotations

import typer
from rich.console import Console
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

    console.print(table)
    console.print("[bold green]M0 骨架自检完成。[/bold green]")


if __name__ == "__main__":
    app()
