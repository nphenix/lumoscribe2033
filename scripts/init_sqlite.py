#!/usr/bin/env python3
"""
SQLite 数据库初始化脚本

基于 SQLModel + SQLite 最佳实践，为 lumoscribe2033 项目创建初始数据库结构。
包含 speckit 工件、对话记录、合规报告等核心数据模型。

使用方法:
    python scripts/init_sqlite.py [--database-url sqlite:///data/lumoscribe.db] [--drop-existing]

环境变量:
    DATABASE_URL: 数据库连接字符串，默认为 sqlite:///data/lumoscribe.db
    LOG_LEVEL: 日志级别，默认为 INFO
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from sqlmodel import SQLModel, create_engine, text
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.logging import RichHandler

from src.framework.shared.config import settings

# 配置 Rich 控制台
console = Console()

# 配置 Rich 日志处理器
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("lumoscribe.init")


async def create_database_tables(
    database_url: str,
    drop_existing: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None
) -> bool:
    """
    创建数据库表结构
    
    Args:
        database_url: 数据库连接字符串
        drop_existing: 是否删除现有表
        progress: Rich 进度条对象
        task_id: 进度条任务ID
    
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        # 创建数据库引擎
        engine = create_engine(
            database_url,
            echo=False,  # 生产环境关闭 SQL 日志
            pool_pre_ping=True,  # 连接池预检查
            pool_recycle=3600,   # 连接回收时间（秒）
            connect_args={
                "check_same_thread": False,  # 允许多线程访问
                "timeout": 30.0,              # 连接超时
            }
        )
        
        if progress and task_id:
            progress.update(task_id, description="🔧 创建数据库引擎...", advance=10)
        
        logger.info(f"🔗 连接到数据库: {database_url}")
        
        # 创建数据目录（如果不存在）
        if database_url.startswith("sqlite:///"):
            db_path = Path(database_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 确保数据目录存在: {db_path.parent}")
        
        if progress and task_id:
            progress.update(task_id, description="🗂️ 检查现有表结构...", advance=10)
        
        # 检查现有表
        from sqlmodel import Session
        with Session(engine) as session:
            # 检查是否已有数据
            try:
                result = session.exec(text("SELECT name FROM sqlite_master WHERE type='table'"))
                existing_tables = [row[0] for row in result]
                logger.info(f"📋 现有表: {existing_tables}")
            except Exception as e:
                logger.warning(f"⚠️ 检查现有表时出错: {e}")
                existing_tables = []
        
        if drop_existing:
            if progress and task_id:
                progress.update(task_id, description="🗑️ 删除现有表...", advance=15)
            
            logger.warning("⚠️ 删除现有表结构...")
            SQLModel.metadata.drop_all(engine)
        
        if progress and task_id:
            progress.update(task_id, description="🏗️ 创建表结构...", advance=25)
        
        # 导入所有模型并创建表
        logger.info("🏗️ 创建表结构...")
        
        # 导入领域模型
        try:
            from src.domain.pipeline.models import PipelineExecution, PipelineStep
            from src.domain.doc_review.models import DocumentReview, ReviewMetric
            from src.domain.compliance.models import ComplianceReport, ConversationRecord
            from src.domain.knowledge.models import BestPractice, PracticeReference
            
            logger.info("✅ 成功导入领域模型")
        except ImportError as e:
            logger.warning(f"⚠️ 部分模型导入失败，使用基础 SQLModel: {e}")
        
        # 创建所有表
        SQLModel.metadata.create_all(engine)
        
        if progress and task_id:
            progress.update(task_id, description="📊 验证表结构...", advance=20)
        
        # 验证表创建
        with Session(engine) as session:
            result = session.exec(text("SELECT name FROM sqlite_master WHERE type='table'"))
            created_tables = [row[0] for row in result]
            logger.info(f"✅ 创建的表: {created_tables}")
        
        if progress and task_id:
            progress.update(task_id, description="✨ 初始化基础数据...", advance=20)
        
        # 初始化基础数据
        await _initialize_base_data(engine)
        
        if progress and task_id:
            progress.update(task_id, description="✅ 完成!", advance=10)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建数据库表失败: {e}")
        if progress and task_id:
            progress.update(task_id, description=f"❌ 失败: {str(e)}", completed=100)
        return False


async def _initialize_base_data(engine) -> None:
    """初始化基础数据"""
    from sqlmodel import Session
    
    logger.info("📊 初始化基础数据...")
    
    try:
        with Session(engine) as session:
            # 这里可以添加初始数据
            # 例如：默认配置、系统用户等
            
            # 创建数据库信息记录
            session.exec(text("""
                CREATE TABLE IF NOT EXISTS database_info (
                    id INTEGER PRIMARY KEY,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # 插入版本信息
            session.exec(text("""
                INSERT OR REPLACE INTO database_info (id, version)
                VALUES (1, '0.1.0')
            """))
            
            session.commit()
            logger.info("✅ 基础数据初始化完成")
            
    except Exception as e:
        logger.error(f"⚠️ 基础数据初始化失败: {e}")
        # 不抛出异常，让主流程继续


def main(
    database_url: str = typer.Option(
        None,
        "--database-url",
        "-d",
        help="数据库连接字符串 (默认: 从配置文件读取)"
    ),
    drop_existing: bool = typer.Option(
        False,
        "--drop-existing",
        "-D",
        help="删除现有表结构（谨慎使用）"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细日志"
    )
):
    """
    主函数 - 初始化 SQLite 数据库
    
    Args:
        database_url: 数据库连接字符串
        drop_existing: 是否删除现有表
        verbose: 是否显示详细日志
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        "[bold blue]🚀 lumoscribe2033 SQLite 数据库初始化[/bold blue]\n"
        "为 Hybrid Graph-RAG 质量平台创建数据库结构",
        border_style="blue"
    ))
    
    # 获取配置
    try:
        config = settings
        if database_url:
            db_url = database_url
        else:
            db_url = config.DATABASE_URL
            if not db_url:
                db_url = "sqlite:///data/lumoscribe.db"
        
        logger.info(f"📁 使用数据库: {db_url}")
        
    except Exception as e:
        logger.error(f"❌ 获取配置失败: {e}")
        console.print(f"[red]❌ 配置错误: {e}[/red]")
        raise typer.Exit(1)
    
    # 创建进度条
    with Progress(
        "[progress.description]{task.description}",
        "[progress.bar]{task.completed:>3d}/{task.total:>3d}",
        "• [progress.percentage]{task.percentage:>3.0f}%",
        console=console,
        transient=True
    ) as progress:
        task_id = progress.add_task("初始化数据库", total=100, start=False)
        
        # 执行数据库初始化
        success = asyncio.run(create_database_tables(
            database_url=db_url,
            drop_existing=drop_existing,
            progress=progress,
            task_id=task_id
        ))
        
        if success:
            console.print("\n[green]✅ 数据库初始化成功！[/green]")
            console.print(f"[blue]📍 数据库位置: {db_url}[/blue]")
            
            # 显示下一步操作
            console.print("\n[bold]下一步操作：[/bold]")
            console.print("• 运行 RAG 系统: [cyan]python -m src.cli.main pipeline run[/cyan]")
            console.print("• 启动 API 服务: [cyan]uvicorn src.api.main:app --reload --port 8080[/cyan]")
            console.print("• 运行异步任务: [cyan]arq workers.settings.WorkerSettings[/cyan]")
        else:
            console.print("\n[red]❌ 数据库初始化失败！[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)