#!/usr/bin/env python3
"""
ChromaDB 向量数据库初始化脚本

基于 ChromaDB 最佳实践，为 lumoscribe2033 项目创建向量存储结构。
支持本地持久化存储和云服务配置，包含 RAG 文档、代码片段、最佳实践等集合。

使用方法:
    python scripts/init_chroma.py [--path vector/chroma] [--reset] [--model-name MODEL_NAME]

环境变量:
    CHROMA_SERVER_HOST: ChromaDB 服务器主机
    CHROMA_SERVER_HTTP_PORT: ChromaDB 服务器端口
    CHROMA_API_KEY: ChromaDB Cloud API 密钥
    CHROMA_TENANT: ChromaDB 租户名称
    CHROMA_DATABASE: ChromaDB 数据库名称
    OPENAI_API_KEY: OpenAI API 密钥（用于嵌入）
    OPENAI_BASE_URL: OpenAI 兼容 API 基础 URL（可选）
    LOG_LEVEL: 日志级别
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import chromadb
from chromadb import ClientAPI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.logging import RichHandler
from rich.table import Table

# 配置 Rich 控制台
console = Console()

# 配置 Rich 日志处理器
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("lumoscribe.chroma.init")


async def initialize_chroma_collections(
    chroma_client: ClientAPI,
    collection_configs: Dict[str, Dict[str, Any]],
    reset: bool = False,
    skip_sample_data: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None
) -> bool:
    """
    初始化 ChromaDB 集合
    
    Args:
        chroma_client: ChromaDB 客户端
        collection_configs: 集合配置
        reset: 是否重置现有集合
        progress: Rich 进度条对象
        task_id: 进度条任务ID
    
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        if progress and task_id:
            progress.update(task_id, description="🔍 检查现有集合...", advance=10)
        
        # 获取现有集合
        existing_collections = chroma_client.list_collections()
        existing_names = {coll.name: coll for coll in existing_collections}
        logger.info(f"📋 现有集合: {[coll.name for coll in existing_collections]}")
        
        total_collections = len(collection_configs)
        collection_progress = 20 / total_collections if total_collections > 0 else 0
        
        for i, (collection_name, config) in enumerate(collection_configs.items()):
            if progress and task_id:
                progress.update(task_id, description=f"🏗️ 创建集合: {collection_name}...", advance=collection_progress)
            
            logger.info(f"🏗️ 创建/验证集合: {collection_name}")
            
            # 如果存在且需要重置，删除现有集合
            if collection_name in existing_names and reset:
                logger.warning(f"🗑️ 删除现有集合: {collection_name}")
                chroma_client.delete_collection(collection_name)
                existing_names.pop(collection_name)
            
            # 创建集合（如果不存在）
            if collection_name not in existing_names:
                try:
                    collection = chroma_client.create_collection(
                        name=collection_name,
                        metadata=config.get("metadata", {}),
                        embedding_function=config.get("embedding_function"),
                        get_or_create=True
                    )
                    logger.info(f"✅ 创建集合: {collection_name}")
                except Exception as e:
                    logger.error(f"❌ 创建集合 {collection_name} 失败: {e}")
                    return False
            else:
                logger.info(f"✅ 集合已存在: {collection_name}")
        
        if progress and task_id:
            progress.update(task_id, description="📊 验证集合结构...", advance=15)
        
        # 验证所有集合
        final_collections = chroma_client.list_collections()
        created_names = set(collection_configs.keys())
        existing_final_names = {coll.name for coll in final_collections}
        
        missing_collections = created_names - existing_final_names
        if missing_collections:
            logger.warning(f"⚠️ 缺失集合: {missing_collections}")
        else:
            logger.info("✅ 所有集合创建成功")
        
        if not skip_sample_data:
            if progress and task_id:
                progress.update(task_id, description="✨ 初始化示例数据...", advance=25)
            
            # 初始化示例数据
            await _initialize_sample_data(chroma_client, collection_configs)
        else:
            if progress and task_id:
                progress.update(task_id, description="⏭️ 跳过示例数据初始化", advance=25)
            logger.info("⏭️ 跳过示例数据初始化")
        
        if progress and task_id:
            progress.update(task_id, description="✅ 完成!", advance=20)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 初始化 ChromaDB 集合失败: {e}")
        if progress and task_id:
            progress.update(task_id, description=f"❌ 失败: {str(e)}", completed=100)
        return False


async def _initialize_sample_data(
    chroma_client: ClientAPI,
    collection_configs: Dict[str, Dict[str, Any]]
) -> None:
    """初始化示例数据"""
    logger.info("📊 初始化示例数据...")
    
    try:
        # 为文档集合添加示例数据
        if "documents" in collection_configs:
            doc_collection = chroma_client.get_collection("documents")
            
            sample_docs = [
                {
                    "id": "sample_001",
                    "document": "lumoscribe2033 是一个基于 speckit 的 Hybrid Graph-RAG 质量平台，支持多 IDE 适配、文档评估和对话溯源。",
                    "metadata": {
                        "source": "README.md",
                        "type": "documentation",
                        "language": "zh-CN",
                        "created_at": "2025-11-14T00:00:00Z"
                    }
                },
                {
                    "id": "sample_002", 
                    "document": "FastAPI 是一个现代、快速、基于标准 Python 类型提示的 Web 框架，用于构建 API。",
                    "metadata": {
                        "source": "docs/external/fastapi.md",
                        "type": "documentation",
                        "language": "en",
                        "created_at": "2025-11-14T00:00:00Z"
                    }
                },
                {
                    "id": "sample_003",
                    "document": "LangChain 1.0 是一个用于构建基于大语言模型的应用程序的框架，支持代理、链和记忆功能。",
                    "metadata": {
                        "source": "docs/external/langchain.md",
                        "type": "documentation", 
                        "language": "en",
                        "created_at": "2025-11-14T00:00:00Z"
                    }
                }
            ]
            
            for doc in sample_docs:
                try:
                    doc_collection.add(
                        ids=[doc["id"]],
                        documents=[doc["document"]],
                        metadatas=[doc["metadata"]]
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 添加示例文档失败: {e}")
            
            logger.info(f"✅ 添加 {len(sample_docs)} 个示例文档")
        
        # 为代码片段集合添加示例数据
        if "code_snippets" in collection_configs:
            code_collection = chroma_client.get_collection("code_snippets")
            
            sample_codes = [
                {
                    "id": "code_sample_001",
                    "document": '''
from typing import List
from sqlmodel import SQLModel, Field

class DocumentReview(SQLModel, table=True):
    """文档评估记录模型"""
    id: int = Field(default=None, primary_key=True)
    document_path: str = Field(index=True)
    review_score: float
    review_metrics: dict = Field(default={})
''',
                    "metadata": {
                        "source": "src/domain/doc_review/models.py",
                        "type": "code",
                        "language": "python",
                        "function": "data_model",
                        "created_at": "2025-11-14T00:00:00Z"
                    }
                }
            ]
            
            for code in sample_codes:
                try:
                    code_collection.add(
                        ids=[code["id"]],
                        documents=[code["document"]],
                        metadatas=[code["metadata"]]
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 添加示例代码失败: {e}")
            
            logger.info(f"✅ 添加 {len(sample_codes)} 个示例代码片段")
        
        logger.info("✅ 示例数据初始化完成")
        
    except Exception as e:
        logger.error(f"❌ 示例数据初始化失败: {e}")


def create_collection_configs(embedding_function=None) -> Dict[str, Dict[str, Any]]:
    """创建集合配置"""
    return {
        "documents": {
            "metadata": {
                "description": "RAG 文档集合 - 存储项目文档、规范、说明等",
                "hnsw:space": "cosine",  # 使用余弦相似度
            },
            "embedding_function": embedding_function,
        },
        "code_snippets": {
            "metadata": {
                "description": "代码片段集合 - 存储代码示例、最佳实践代码",
                "hnsw:space": "cosine",
            },
            "embedding_function": embedding_function,
        },
        "best_practices": {
            "metadata": {
                "description": "最佳实践集合 - 存储最佳实践、模式、指导原则",
                "hnsw:space": "cosine",
            },
            "embedding_function": embedding_function,
        },
        "conversation_records": {
            "metadata": {
                "description": "对话记录集合 - 存储 AI 对话历史和上下文",
                "hnsw:space": "cosine",
            },
            "embedding_function": embedding_function,
        },
        "compliance_reports": {
            "metadata": {
                "description": "合规报告集合 - 存储静态检查报告和合规性分析",
                "hnsw:space": "cosine",
            },
            "embedding_function": embedding_function,
        }
    }


def main(
    path: str = typer.Option(
        "vector/chroma",
        "--path",
        "-p",
        help="ChromaDB 数据目录路径"
    ),
    host: str = typer.Option(
        None,
        "--host",
        "-h",
        help="ChromaDB 服务器主机"
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-P",
        help="ChromaDB 服务器端口"
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="重置现有集合（删除并重建）"
    ),
    use_openai: bool = typer.Option(
        True,
        "--use-openai",
        "-o",
        help="使用 OpenAI 嵌入函数"
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        "-m",
        help="嵌入模型名称（当使用 OpenAI 时有效，默认: text-embedding-3-small）"
    ),
    skip_sample_data: bool = typer.Option(
        False,
        "--skip-sample-data",
        help="跳过示例数据初始化（避免下载默认模型）"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细日志"
    )
):
    """
    主函数 - 初始化 ChromaDB
    
    Args:
        path: 数据目录路径
        host: 服务器主机
        port: 服务器端口
        reset: 是否重置
        use_openai: 是否使用 OpenAI 嵌入
        model_name: 嵌入模型名称
        skip_sample_data: 是否跳过示例数据初始化
        verbose: 是否显示详细日志
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        "[bold blue]🚀 lumoscribe2033 ChromaDB 向量数据库初始化[/bold blue]\n"
        "为 Hybrid Graph-RAG 质量平台创建向量存储结构",
        border_style="blue"
    ))
    
    # 确定嵌入函数
    embedding_function = None
    if use_openai:
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_base_url = os.getenv("OPENAI_BASE_URL")
            
            if not openai_api_key:
                logger.warning("⚠️ OPENAI_API_KEY 环境变量未设置，使用默认嵌入函数")
            else:
                # 确定模型名称
                model = model_name or "text-embedding-3-small"
                
                # 构建参数
                openai_kwargs = {
                    "api_key": openai_api_key,
                    "model_name": model
                }
                
                # 添加基础 URL（如果设置）
                if openai_base_url:
                    openai_kwargs["api_base"] = openai_base_url
                
                embedding_function = OpenAIEmbeddingFunction(**openai_kwargs)
                logger.info(f"✅ 使用 OpenAI 嵌入函数: {model}")
                if openai_base_url:
                    logger.info(f"🔗 API 基础 URL: {openai_base_url}")
        except ImportError:
            logger.warning("⚠️ OpenAI 嵌入函数不可用，使用默认函数")
    
    # 创建客户端
    try:
        if host and port:
            # 连接远程服务器
            chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=False
            )
            logger.info(f"🔗 连接到远程 ChromaDB: {host}:{port}")
        else:
            # 使用本地持久化客户端
            chroma_path = Path(path)
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            logger.info(f"📁 使用本地 ChromaDB: {chroma_path}")
        
    except Exception as e:
        logger.error(f"❌ 创建 ChromaDB 客户端失败: {e}")
        console.print(f"[red]❌ 客户端创建错误: {e}[/red]")
        raise typer.Exit(1)
    
    # 获取集合配置
    collection_configs = create_collection_configs(embedding_function)
    
    # 创建进度条
    with Progress(
        "[progress.description]{task.description}",
        "[progress.bar]{task.completed:>3d}/{task.total:>3d}",
        "• [progress.percentage]{task.percentage:>3.0f}%",
        console=console,
        transient=True
    ) as progress:
        task_id = progress.add_task("初始化 ChromaDB", total=100, start=False)
        
        # 执行初始化
        success = asyncio.run(initialize_chroma_collections(
            chroma_client=chroma_client,
            collection_configs=collection_configs,
            reset=reset,
            skip_sample_data=skip_sample_data,
            progress=progress,
            task_id=task_id
        ))
        
        if success:
            console.print("\n[green]✅ ChromaDB 初始化成功！[/green]")
            
            # 显示集合信息
            collections = chroma_client.list_collections()
            table = Table(title="📊 创建的集合")
            table.add_column("集合名称", style="cyan", justify="left")
            table.add_column("描述", style="magenta", justify="left")
            table.add_column("文档数量", style="yellow", justify="right")
            
            for collection in collections:
                count = collection.count() if hasattr(collection, 'count') else "未知"
                description = collection_configs.get(collection.name, {}).get("metadata", {}).get("description", "无描述")
                table.add_row(collection.name, description, str(count))
            
            console.print(table)
            
            # 显示下一步操作
            console.print("\n[bold]下一步操作：[/bold]")
            if skip_sample_data:
                console.print("• 查询空集合: [cyan]python -c \"from chromadb import PersistentClient; c=PersistentClient(); print(c.get_collection('documents').count())\"[/cyan]")
                console.print("• 添加文档: [cyan]使用 RAG API 或 CLI 工具[/cyan]")
            else:
                console.print("• 查询文档: [cyan]python -c \"from chromadb import PersistentClient; c=PersistentClient(); print(c.get_collection('documents').query(query_texts=['speckit'], n_results=3))\"[/cyan]")
                console.print("• 添加文档: [cyan]使用 RAG API 或 CLI 工具[/cyan]")
            console.print("• 启动 API 服务: [cyan]uvicorn src.api.main:app --reload --port 8080[/cyan]")
        else:
            console.print("\n[red]❌ ChromaDB 初始化失败！[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)