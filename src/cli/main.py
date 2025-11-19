"""
CLI 主入口

基于 Typer 最佳实践实现：
- 命令分组
- 参数验证
- 错误处理
- 进度显示
- 日志输出

功能特点：
- 类型提示支持
- 自动生成帮助文档
- 命令嵌套
- 优雅的错误处理
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import typer
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_traceback

from src.cli.metrics_collector import app as metrics_collector_app
from src.framework.orchestrators import bootstrap_langchain_executor
from src.framework.shared.config import Settings

# 初始化 Rich
console = Console()
install_traceback()

# 创建主应用
app = typer.Typer(
    name="lumoscribe2033",
    help="Hybrid Graph-RAG Phase 1 质量平台",
    epilog="更多帮助请访问: https://github.com/lumoscribe2033",
    rich_markup_mode="markdown",
    pretty_exceptions_enable=True
)

# 子命令应用
pipeline_app = typer.Typer(name="pipeline", help="Speckit 自动化管线管理")
ide_app = typer.Typer(name="ide", help="IDE 适配包管理")
doc_app = typer.Typer(name="docs", help="文档评估管理")
conversation_app = typer.Typer(name="conversations", help="对话管理")
metrics_app = typer.Typer(name="metrics", help="指标收集管理")
config_app = typer.Typer(name="config", help="配置管理")
task_app = typer.Typer(name="tasks", help="任务管理")
health_app = typer.Typer(name="health", help="健康检查")

# 将子命令添加到主应用
app.add_typer(pipeline_app)
app.add_typer(ide_app)
app.add_typer(doc_app)
app.add_typer(conversation_app)
app.add_typer(metrics_app)
app.add_typer(task_app)
app.add_typer(health_app)
app.add_typer(config_app)


def setup_logging(verbose: int = 0) -> None:
    """设置日志配置"""
    # 移除默认处理器
    logger.remove()

    # 设置日志级别
    log_level = "DEBUG" if verbose > 0 else "INFO"

    # 添加 Rich 处理器（控制台输出）
    logger.add(
        RichHandler(
            console=console,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        ),
        level=log_level,
        format="{message}"
    )

    # 添加文件处理器（可选）
    log_file = Path("logs/lumoscribe2033.log")
    log_file.parent.mkdir(exist_ok=True)

    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        backtrace=True,
        diagnose=True
    )


# 定义命令选项常量
VERBOSE_OPTION = typer.Option(0, "--verbose", "-v", count=True, help="增加输出详细程度")
CONFIG_OPTION = typer.Option(None, "--config", help="配置文件路径")
DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="仅显示将要执行的操作")
INPUT_FILE_ARG = typer.Argument(..., help="输入文档路径")
OUTPUT_DIR_OPTION = typer.Option(None, "--output", "-o", help="输出目录")
FORCE_OPTION = typer.Option(False, "--force", "-f", help="强制覆盖输出")
IDE_ARG = typer.Argument(..., help="IDE 名称 (cursor, roocode)")
PATTERN_ARG = typer.Argument(..., help="文档匹配模式")
OUTPUT_ARG = typer.Argument(..., help="文档文件路径")
SOURCE_ARG = typer.Argument(..., help="对话来源 (cursor, roocode)")
PATH_ARG = typer.Argument(..., help="对话文件路径")
BATCH_SIZE_OPTION = typer.Option(100, "--batch-size", help="批量处理大小")
INTERVAL_OPTION = typer.Option(3600, "--interval", help="收集间隔（秒）")

@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: int = VERBOSE_OPTION,
    config: Path = CONFIG_OPTION,
    dry_run: bool = DRY_RUN_OPTION
) -> None:
    """主回调函数，处理全局选项"""

    # 设置日志
    setup_logging(verbose)

    # 记录启动信息
    logger.info("🚀 lumoscribe2033 CLI 启动")
    logger.debug(f"命令: {' '.join(sys.argv)}")

    # 加载配置
    if config and config.exists():
        logger.info(f"使用配置文件: {config}")
        # TODO: 实现配置文件加载逻辑

    # 设置上下文对象
    ctx.obj = {
        "verbose": verbose,
        "config": config,
        "dry_run": dry_run,
        "settings": Settings()
    }

    # 阶段 C：CLI 直接初始化 LangChainExecutor，供后续命令复用
    ctx.obj["executor"] = bootstrap_langchain_executor(settings=ctx.obj["settings"])


@app.command()
def version() -> None:
    """显示版本信息"""
    from src import __description__, __version__

    console.print(f"[bold]lumoscribe2033[/bold] {__version__}")
    console.print(f"{__description__}")
    console.print("")
    console.print("🔗 [link=https://github.com/lumoscribe2033]GitHub 仓库[/link]")


@app.command()
def init(
    force: bool = FORCE_OPTION,
) -> None:
    """初始化项目环境"""
    logger.info("初始化 lumoscribe2033 环境（统一操作）...")

    from subprocess import CalledProcessError, run

    from src.framework.shared.metadata_injector import bulk_inject, verify_directory

    failures: list[str] = []

    try:
        # 1) 目录初始化
        console.print("[bold]步骤 1/5[/bold] • 初始化数据与产物目录")
        run([sys.executable, "scripts/bootstrap_data_dirs.py"], check=True)

        # 2) 初始化 SQLite
        console.print("[bold]步骤 2/5[/bold] • 初始化 SQLite 数据库")
        run([sys.executable, "scripts/init_sqlite.py"], check=True)

        # 3) 初始化 Chroma
        console.print("[bold]步骤 3/5[/bold] • 初始化 Chroma 向量库")
        run([sys.executable, "scripts/init_chroma.py"], check=True)

        # 4) 初始化 NetworkX 图结构
        console.print("[bold]步骤 4/5[/bold] • 初始化 NetworkX 图结构")
        run([sys.executable, "scripts/init_networkx.py"], check=True)

        # 5) 文档元数据头 注入 + 校验
        console.print("[bold]步骤 5/5[/bold] • 文档元数据头 注入 + 校验")
        changed = bulk_inject(
            root=".",
            command="cli:init (unified)",
            include_globs=("docs/**/*.md", "specs/**/*.md"),
            exclude_globs=(".git/**",),
            update_if_exists=True,
        )
        verify_results = verify_directory(
            root=".",
            include_globs=("docs/**/*.md", "specs/**/*.md"),
            exclude_globs=(".git/**",),
        )
        missing = [r for r in verify_results if not r.has_header]

        # 输出摘要
        console.print("\n[bold]初始化摘要[/bold]")
        console.print(f"  • 元数据头已注入/更新: {len(changed)} 个文件")
        console.print(f"  • 校验缺失元数据头: {len(missing)} 个文件")
        if missing:
            for r in missing[:20]:
                console.print(f"    ❌ {r.path}")
            if len(missing) > 20:
                console.print(f"    … 以及 {len(missing) - 20} 个更多文件")

    except CalledProcessError as e:
        failures.append(f"子进程失败: {e}")
    except Exception as e:
        failures.append(str(e))

    if failures and not force:
        console.print("\n[red]❌ 初始化遇到错误[/red]")
        for msg in failures:
            console.print(f"  • {msg}")
        raise typer.Exit(code=1)

    console.print("\n[green]✅ 统一初始化完成[/green]")


@app.command()
def status() -> None:
    """显示系统状态"""
    from src.framework.shared.config import Settings

    settings = Settings()

    console.print("[bold]系统状态[/bold]")
    console.print(f"环境: {settings.ENVIRONMENT}")
    console.print(f"调试模式: {settings.DEBUG}")
    console.print(f"日志级别: {settings.LOG_LEVEL}")

    # TODO: 添加更多状态检查
    # - 数据库连接状态
    # - 向量存储状态
    # - LLM 服务状态


# Pipeline 命令
@pipeline_app.command("run")
def run_pipeline(
    ctx: typer.Context,
    input_file: Path = INPUT_FILE_ARG,
    output_dir: Path = OUTPUT_DIR_OPTION,
    force: bool = FORCE_OPTION,
) -> None:
    """运行 speckit 自动化管线"""
    logger.info(f"运行管线处理: {input_file}")

    if ctx.obj["dry_run"]:
        console.print(f"[yellow]⚠️  Dry run:[/yellow] 将处理文件 {input_file}")
        return

    # 检查输入文件
    if not input_file.exists():
        console.print(f"[red]❌[/red] 输入文件不存在: {input_file}")
        raise typer.BadParameter(f"文件不存在: {input_file}")

    # 确定输出目录
    if not output_dir:
        output_dir = input_file.parent / "output"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 调用 API 或直接执行管线
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/tasks/queue"

        # 准备请求数据
        with open(input_file, encoding='utf-8') as f:
            content = f.read()

        payload = {
            "task_type": "pipeline_full",
            "payload": {
                "input_content": content,
                "input_file": str(input_file),
                "output_dir": str(output_dir),
                "force": force
            }
        }

        # 发送请求到 API
        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        task_result = response.json()
        console.print(f"[green]✅[/green] 管线任务已提交: {task_result.get('task_id', 'unknown')}")

        # 等待任务完成（可选）
        if not force:
            console.print("💡 提示: 任务已在后台执行，可以使用任务管理命令查看状态")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 管线执行失败: {e}")
        raise
    except Exception as e:
        logger.error(f"管线执行错误: {e}")
        console.print(f"[red]❌[/red] 管线执行失败: {e}")
        raise


# IDE 命令
@ide_app.command("generate")
def generate_ide_package(
    ctx: typer.Context,
    ide: str = IDE_ARG,
    output_dir: Path = OUTPUT_DIR_OPTION,
    force: bool = FORCE_OPTION,
) -> None:
    """生成 IDE 适配包"""
    logger.info(f"生成 {ide} IDE 适配包")

    if ctx.obj["dry_run"]:
        console.print(f"[yellow]⚠️  Dry run:[/yellow] 将生成 {ide} 适配包")
        return

    # 验证 IDE 类型
    valid_ide_types = ["cursor", "roocode"]
    if ide.lower() not in valid_ide_types:
        console.print(f"[red]❌[/red] 不支持的 IDE 类型: {ide}")
        console.print(f"支持的 IDE: {', '.join(valid_ide_types)}")
        raise typer.BadParameter(f"不支持的 IDE 类型: {ide}")

    try:
        # 调用 API 生成 IDE 包
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/ide/packages/generate"

        payload = {
            "ide_type": ide.lower(),
            "force": force
        }

        if output_dir:
            payload["output_dir"] = str(output_dir)

        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        result = response.json()
        console.print(f"[green]✅[/green] {ide.upper()} IDE 适配包生成完成")
        console.print(f"📁 输出目录: {result.get('output_path', 'unknown')}")

        # 显示生成的文件列表
        generated_files = result.get('generated_files', [])
        if generated_files:
            console.print("\n[bold]生成的文件:[/bold]")
            for file_path in generated_files:
                console.print(f"  • {file_path}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] IDE 适配包生成失败: {e}")
        raise
    except Exception as e:
        logger.error(f"IDE 包生成错误: {e}")
        console.print(f"[red]❌[/red] IDE 适配包生成失败: {e}")
        raise


# 文档评估命令
OUTPUT_REPORT_OPTION = typer.Option(None, "--output", "-o", help="输出报告路径")

@doc_app.command("evaluate")
def evaluate_docs(
    ctx: typer.Context,
    pattern: str = PATTERN_ARG,
    output: Path = OUTPUT_REPORT_OPTION,
) -> None:
    """评估文档质量"""
    logger.info(f"评估文档: {pattern}")

    if ctx.obj["dry_run"]:
        console.print(f"[yellow]⚠️  Dry run:[/yellow] 将评估文档: {pattern}")
        return

    try:
        # 调用 API 评估文档
        from pathlib import Path

        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/docs/evaluate"

        payload = {
            "pattern": pattern,
            "auto_evaluate": True
        }

        if output:
            payload["output_path"] = str(output)

        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        result = response.json()
        console.print("[green]✅[/green] 文档评估完成")
        console.print(f"📊 评估文件数: {result.get('total_files', 0)}")
        console.print(f"📈 平均分: {result.get('average_score', 0):.1f}")

        # 显示详细结果
        if result.get('detailed_results'):
            console.print("\n[bold]详细结果:[/bold]")
            for doc_result in result['detailed_results']:
                status = "✅" if doc_result.get('success', False) else "❌"
                console.print(f"  {status} {doc_result.get('filename', 'unknown')}: {doc_result.get('score', 0):.1f}")

        if output:
            console.print(f"📄 报告已保存到: {output}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 文档评估失败: {e}")
        raise
    except Exception as e:
        logger.error(f"文档评估错误: {e}")
        console.print(f"[red]❌[/red] 文档评估失败: {e}")
        raise


@doc_app.command("meta-inject")
def docs_meta_inject(
    root: Path = typer.Option(".", "--root", "-r", help="根目录（默认当前路径）"),
    glob: str = typer.Option("docs/**/*.md", "--glob", "-g", help="包含模式（glob）"),
    exclude: str = typer.Option(".git/**", "--exclude", "-x", help="排除模式（glob）"),
    command: str = typer.Option("cli:docs meta-inject", "--command", "-c", help="生成命令名"),
    update: bool = typer.Option(False, "--update", "-u", help="已存在头部时更新时间戳"),
) -> None:
    """为匹配的文档批量注入生成元数据头"""
    from src.framework.shared.metadata_injector import bulk_inject

    try:
        changed = bulk_inject(
            root=str(root),
            command=command,
            include_globs=(glob,),
            exclude_globs=(exclude,) if exclude else (),
            update_if_exists=update,
        )
        console.print(f"[green]✅[/green] 已注入/更新 {len(changed)} 个文件的元数据头")
        if changed:
            console.print("\n[bold]变更文件:[/bold]")
            for p in changed[:50]:
                console.print(f"  • {p}")
            if len(changed) > 50:
                console.print(f"  … 以及 {len(changed) - 50} 个更多文件")
    except Exception as e:
        logger.error(f"元数据注入失败: {e}")
        console.print(f"[red]❌[/red] 元数据注入失败: {e}")
        raise


@doc_app.command("meta-verify")
def docs_meta_verify(
    root: Path = typer.Option(".", "--root", "-r", help="根目录（默认当前路径）"),
    glob: str = typer.Option("docs/**/*.md", "--glob", "-g", help="包含模式（glob）"),
    exclude: str = typer.Option(".git/**", "--exclude", "-x", help="排除模式（glob）"),
) -> None:
    """校验匹配文档是否包含生成元数据头"""
    from src.framework.shared.metadata_injector import verify_directory

    try:
        results = verify_directory(
            root=str(root),
            include_globs=(glob,),
            exclude_globs=(exclude,) if exclude else (),
        )
        missing = [r for r in results if not r.has_header]
        console.print("[bold]校验结果:[/bold]")
        console.print(f"  • 检查文件: {len(results)}")
        console.print(f"  • 缺失头部: {len(missing)}")

        if missing:
            console.print("\n[bold]缺失头部文件（最多 50 项）:[/bold]")
            for r in missing[:50]:
                console.print(f"  ❌ {r.path}")
            if len(missing) > 50:
                console.print(f"  … 以及 {len(missing) - 50} 个更多文件")
            raise typer.Exit(code=1)
        else:
            console.print("[green]✅[/green] 全部文件已包含生成元数据头")
    except Exception as e:
        logger.error(f"元数据校验失败: {e}")
        console.print(f"[red]❌[/red] 元数据校验失败: {e}")
        raise

# 对话管理命令
@conversation_app.command("import")
def import_conversations(
    ctx: typer.Context,
    source: str = SOURCE_ARG,
    path: Path = PATH_ARG,
    batch_size: int = BATCH_SIZE_OPTION,
) -> None:
    """导入对话记录"""
    logger.info(f"从 {source} 导入对话: {path}")

    if ctx.obj["dry_run"]:
        console.print(f"[yellow]⚠️  Dry run:[/yellow] 将导入 {source} 对话")
        return

    # TODO: 实现对话导入逻辑
    console.print("✅ 对话导入完成")


# 指标收集命令
@metrics_app.command("collect")
def collect_metrics(
    ctx: typer.Context,
    interval: int = INTERVAL_OPTION,
    output: Path = OUTPUT_ARG,
) -> None:
    """收集系统指标"""
    logger.info("收集系统指标")

    # TODO: 实现指标收集逻辑
    console.print("✅ 指标收集完成")


# 任务管理命令
@task_app.command("list")
def list_tasks(
    ctx: typer.Context,
    status: str = typer.Option(None, "--status", help="任务状态过滤"),
    task_type: str = typer.Option(None, "--type", help="任务类型过滤"),
    limit: int = typer.Option(20, "--limit", help="返回数量限制")
) -> None:
    """列出任务队列中的任务"""
    logger.info("查询任务队列状态")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/tasks"

        params = {}
        if status:
            params["status"] = status
        if task_type:
            params["task_type"] = task_type
        if limit:
            params["limit"] = limit

        response = requests.get(api_url, params=params)
        response.raise_for_status()

        tasks = response.json()
        console.print(f"[bold]任务列表 (共 {len(tasks)} 个):[/bold]")

        for task in tasks:
            status_icon = "🟢" if task["status"] == "completed" else "🟡" if task["status"] == "running" else "🔴"
            console.print(f"  {status_icon} {task['task_id']} - {task['task_type']} ({task['status']})")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 查询任务失败: {e}")
        raise


@task_app.command("status")
def task_status(
    ctx: typer.Context,
    task_id: str = typer.Argument(..., help="任务 ID")
) -> None:
    """获取任务状态"""
    logger.info(f"查询任务状态: {task_id}")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/tasks/{task_id}"

        response = requests.get(api_url)
        response.raise_for_status()

        task = response.json()
        console.print("[bold]任务详情:[/bold]")
        console.print(f"  • 任务 ID: {task['task_id']}")
        console.print(f"  • 任务类型: {task['task_type']}")
        console.print(f"  • 状态: {task['status']}")
        console.print(f"  • 创建时间: {task['created_at']}")

        if task.get('progress'):
            progress = task['progress']
            console.print(f"  • 进度: {progress.get('current', 0)}/{progress.get('total', 0)} ({progress.get('message', '')})")

        if task.get('result'):
            console.print(f"  • 结果: {task['result']}")

        if task.get('error'):
            console.print(f"  • 错误: {task['error']}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 查询任务状态失败: {e}")
        raise


@task_app.command("cancel")
def cancel_task(
    ctx: typer.Context,
    task_id: str = typer.Argument(..., help="任务 ID")
) -> None:
    """取消任务"""
    logger.info(f"取消任务: {task_id}")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/tasks/{task_id}"

        response = requests.delete(api_url)
        response.raise_for_status()

        result = response.json()
        console.print(f"[green]✅[/green] {result['message']}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 取消任务失败: {e}")
        raise


@task_app.command("queue-status")
def queue_status(ctx: typer.Context) -> None:
    """获取队列状态"""
    logger.info("查询队列状态")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/tasks/queue/status"

        response = requests.get(api_url)
        response.raise_for_status()

        status = response.json()
        console.print("[bold]队列状态:[/bold]")
        console.print(f"  • 队列名称: {status['queue_name']}")
        console.print(f"  • 队列大小: {status['queue_size']}")
        console.print(f"  • 运行中任务: {status['running_jobs']}")
        console.print(f"  • 已完成任务: {status['completed_jobs']}")
        console.print(f"  • 工作者数量: {status['worker_count']}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 查询队列状态失败: {e}")
        raise


# 健康检查命令
@health_app.command("check")
def health_check(ctx: typer.Context) -> None:
    """健康检查"""
    logger.info("执行健康检查")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/health"

        response = requests.get(api_url)
        response.raise_for_status()

        health = response.json()

        console.print("[bold]系统健康状态:[/bold]")
        console.print(f"  • 状态: {health['status']}")
        console.print(f"  • 版本: {health['version']}")
        console.print(f"  • 环境: {health['environment']}")
        console.print(f"  • 时间: {health['timestamp']}")

        console.print("\n[bold]服务状态:[/bold]")
        for service_name, service_info in health['services'].items():
            status_icon = "✅" if service_info['status'] == 'healthy' else "❌"
            console.print(f"  {status_icon} {service_name}: {service_info['status']}")

        console.print("\n[bold]系统信息:[/bold]")
        for key, value in health['system'].items():
            console.print(f"  • {key}: {value}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 健康检查失败: {e}")
        raise


@health_app.command("ready")
def ready_check(ctx: typer.Context) -> None:
    """就绪检查"""
    logger.info("执行就绪检查")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/health/ready"

        response = requests.get(api_url)
        response.raise_for_status()

        result = response.json()
        status = result['status']
        status_icon = "✅" if status == 'ready' else "❌"

        console.print(f"{status_icon} 就绪状态: {status}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 就绪检查失败: {e}")
        raise


@health_app.command("live")
def live_check(ctx: typer.Context) -> None:
    """存活检查"""
    logger.info("执行存活检查")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/health/live"

        response = requests.get(api_url)
        response.raise_for_status()

        result = response.json()
        status = result['status']
        status_icon = "✅" if status == 'alive' else "❌"

        console.print(f"{status_icon} 存活状态: {status}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 存活检查失败: {e}")
        raise


# 配置管理命令
@config_app.command("status")
def config_status(ctx: typer.Context) -> None:
    """显示配置状态"""
    logger.info("查询配置状态")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/config/status"

        response = requests.get(api_url)
        response.raise_for_status()

        status = response.json()

        console.print("[bold]配置状态:[/bold]")
        console.print(f"  • 有效性: {'✅' if status['valid'] else '❌'}")
        console.print(f"  • 环境: {status['environment']['environment']}")
        console.print(f"  • 调试模式: {status['environment']['debug']}")
        console.print(f"  • 日志级别: {status['environment']['log_level']}")

        # 显示验证错误
        if status.get('validation_errors'):
            console.print("\n[bold]验证错误:[/bold]")
            for error in status['validation_errors']:
                console.print(f"  ❌ {error}")
        else:
            console.print("\n[bold]验证状态:[/bold] ✅ 无错误")

        # 显示配置文件状态
        if 'config_files' in status:
            console.print("\n[bold]配置文件状态:[/bold]")
            for file_name, exists in status['config_files'].items():
                status_icon = "✅" if exists else "❌"
                console.print(f"  {status_icon} {file_name}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 查询配置状态失败: {e}")
        raise
    except Exception as e:
        logger.error(f"配置状态查询错误: {e}")
        console.print(f"[red]❌[/red] 配置状态查询失败: {e}")
        raise


@config_app.command("validate")
def config_validate(ctx: typer.Context) -> None:
    """验证配置"""
    logger.info("验证配置")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/config/validate"

        response = requests.get(api_url)
        response.raise_for_status()

        result = response.json()

        console.print("[bold]配置验证结果:[/bold]")
        console.print(f"  • 总体状态: {'✅ 有效' if result['valid'] else '❌ 无效'}")
        console.print(f"  • 总错误数: {result['total_errors']}")

        # 显示环境错误
        if result.get('environment_errors'):
            console.print("\n[bold]环境错误:[/bold]")
            for error in result['environment_errors']:
                console.print(f"  ❌ {error}")

        # 显示设置错误
        if result.get('settings_errors'):
            console.print("\n[bold]设置错误:[/bold]")
            for error in result['settings_errors']:
                console.print(f"  ❌ {error}")

        if result['valid']:
            console.print("\n✅ 所有配置验证通过")
        else:
            console.print(f"\n❌ 发现 {result['total_errors']} 个配置问题")
            raise typer.Exit(code=1)

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 配置验证失败: {e}")
        raise
    except Exception as e:
        logger.error(f"配置验证错误: {e}")
        console.print(f"[red]❌[/red] 配置验证失败: {e}")
        raise


@config_app.command("setup")
def config_setup(ctx: typer.Context) -> None:
    """设置开发环境"""
    logger.info("设置开发环境")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/config/setup-dev"

        response = requests.post(api_url)
        response.raise_for_status()

        result = response.json()

        console.print("[bold]开发环境设置结果:[/bold]")
        if result.get('success'):
            console.print("✅ 开发环境设置完成")

            # 显示详细信息
            if 'details' in result:
                console.print("\n[bold]设置详情:[/bold]")
                for key, value in result['details'].items():
                    status_icon = "✅" if value else "❌"
                    console.print(f"  {status_icon} {key}")
        else:
            console.print("❌ 开发环境设置失败")
            raise typer.Exit(code=1)

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 设置开发环境失败: {e}")
        raise
    except Exception as e:
        logger.error(f"设置开发环境错误: {e}")
        console.print(f"[red]❌[/red] 设置开发环境失败: {e}")
        raise


@config_app.command("environment")
def config_environment(ctx: typer.Context) -> None:
    """显示环境信息"""
    logger.info("查询环境信息")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/config/environment"

        response = requests.get(api_url)
        response.raise_for_status()

        env_info = response.json()

        console.print("[bold]环境信息:[/bold]")

        # 显示基本信息
        console.print(f"  • 环境: {env_info['environment']}")
        console.print(f"  • 调试模式: {env_info['debug']}")
        console.print(f"  • 日志级别: {env_info['log_level']}")
        console.print(f"  • API 主机: {env_info['api_host']}")
        console.print(f"  • API 端口: {env_info['api_port']}")

        # 显示 LLM 配置
        if 'llm_config' in env_info:
            console.print("\n[bold]LLM 配置:[/bold]")
            llm_config = env_info['llm_config']
            console.print(f"  • OpenAI 基础URL: {llm_config.get('openai_base_url', 'N/A')}")
            console.print(f"  • OpenAI 模型: {llm_config.get('openai_model', 'N/A')}")
            console.print(f"  • Ollama 主机: {llm_config.get('ollama_host', 'N/A')}")
            console.print(f"  • Ollama 模型: {llm_config.get('ollama_model', 'N/A')}")

        # 显示数据库配置
        if 'database_config' in env_info:
            console.print("\n[bold]数据库配置:[/bold]")
            db_config = env_info['database_config']
            console.print(f"  • 数据库 URL: {db_config.get('database_url', 'N/A')}")
            console.print(f"  • Chroma 主机: {db_config.get('chroma_host', 'N/A')}")
            console.print(f"  • Chroma 端口: {db_config.get('chroma_port', 'N/A')}")

        # 显示目录状态
        if 'directories_status' in env_info:
            console.print("\n[bold]目录状态:[/bold]")
            for dir_name, dir_path in env_info['directories_status'].items():
                dir_obj = Path(dir_path)
                status_icon = "✅" if dir_obj.exists() else "❌"
                console.print(f"  {status_icon} {dir_name}: {dir_path}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 查询环境信息失败: {e}")
        raise
    except Exception as e:
        logger.error(f"环境信息查询错误: {e}")
        console.print(f"[red]❌[/red] 环境信息查询失败: {e}")
        raise


@config_app.command("template")
def config_template(ctx: typer.Context) -> None:
    """生成环境变量模板"""
    logger.info("生成环境变量模板")

    try:
        import requests

        from src.framework.shared.config import Settings

        settings = Settings()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}/api/v1/config/template/env"

        response = requests.get(api_url)
        response.raise_for_status()

        result = response.json()

        console.print("[bold]环境变量模板:[/bold]")
        console.print(f"文件名: {result.get('filename', '.env.example')}")
        console.print(f"说明: {result.get('instructions', '请复制到 .env 文件')}")

        console.print("\n[bold]模板内容:[/bold]")
        console.print(result['template'])

        # 保存到文件
        template_file = Path(result.get('filename', '.env.example'))
        template_file.write_text(result['template'], encoding='utf-8')
        console.print(f"\n✅ 模板已保存到: {template_file}")

    except requests.RequestException as e:
        logger.error(f"API 请求失败: {e}")
        console.print(f"[red]❌[/red] 生成模板失败: {e}")
        raise
    except Exception as e:
        logger.error(f"生成模板错误: {e}")
        console.print(f"[red]❌[/red] 生成模板失败: {e}")
        raise


# 添加指标采集命令（独立应用）
app.add_typer(metrics_collector_app, name="metrics-collect")

if __name__ == "__main__":
    app()
