#!/usr/bin/env python3
"""
初始化本地数据与产物目录

创建并校验项目所需的持久化目录结构：
- data/imports/
- data/persistence/
- data/reference_samples/
- vector/chroma/
- graph/snapshots/
- ide-packages/
"""

import os
from pathlib import Path
import sys

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()
app = typer.Typer(add_completion=False)


REQUIRED_DIRS = [
    "data/imports",
    "data/persistence",
    "data/reference_samples",
    "vector/chroma",
    "graph/snapshots",
    "ide-packages",
]


@app.command("init")
def init_dirs(
    base: str = typer.Option(".", "--base", "-b", help="仓库根目录（默认当前目录）"),
) -> None:
    """创建并校验所需目录结构"""
    base_path = Path(base).resolve()
    created = []
    existed = []

    for rel in REQUIRED_DIRS:
        p = base_path / rel
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))
        else:
            existed.append(str(p))

    table = Table(title="目录初始化结果")
    table.add_column("状态", justify="center", style="cyan", no_wrap=True)
    table.add_column("路径", style="magenta")

    for path in created:
        table.add_row("创建", path)
    for path in existed:
        table.add_row("存在", path)

    console.print(
        Panel.fit(
            "[bold blue]🗂️ 初始化本地数据与产物目录[/bold blue]\n确保 RAG/图/导入样本等目录可用",
            border_style="blue",
        )
    )
    console.print(table)
    console.print("[green]✅ 完成[/green]")


if __name__ == "__main__":
    # 允许直接调用脚本而不带子命令时执行 init，便于统一入口调用
    if len(sys.argv) == 1:
        init_dirs()  # type: ignore[misc]
    else:
        app()

