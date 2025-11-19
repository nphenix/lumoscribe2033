#!/usr/bin/env python3
"""
NetworkX 图数据库初始化脚本

基于 NetworkX 最佳实践，为 lumoscribe2033 项目创建图结构存储。
支持代码依赖图、文档关系图、最佳实践关联图等，用于分析和可视化复杂关系。

使用方法:
    python scripts/init_networkx.py [--path graph/snapshots] [--format gexf] [--reset]

环境变量:
    GRAPH_FORMAT: 图文件格式 (gexf, gml, graphml, json)
    LOG_LEVEL: 日志级别
    PERSISTENCE_ENABLED: 是否启用持久化
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import typer
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.panel import Panel
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree

# 配置 Rich 控制台
console = Console()

# 配置 Rich 日志处理器
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("lumoscribe.graph.init")


async def create_lumoscribe_graph(
    graph_path: Path,
    graph_format: str = "gexf",
    reset: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None
) -> bool:
    """
    创建 lumoscribe2033 图结构
    
    Args:
        graph_path: 图文件存储路径
        graph_format: 图文件格式
        reset: 是否重置现有图
        progress: Rich 进度条对象
        task_id: 进度条任务ID
    
    Returns:
        bool: 成功返回 True，失败返回 False
    """
    try:
        # 确保目录存在
        graph_path.mkdir(parents=True, exist_ok=True)
        
        if progress and task_id:
            progress.update(task_id, description="🏗️ 创建核心图结构...", advance=20)
        
        logger.info("🏗️ 创建 lumoscribe2033 核心图结构...")
        
        # 创建主图
        main_graph = nx.DiGraph()
        
        # 设置图属性
        main_graph.graph.update({
            "name": "lumoscribe2033-hybrid-graph-rag",
            "version": "0.1.0",
            "description": "Hybrid Graph-RAG 质量平台关系图",
            "created_at": "2025-11-14T00:00:00Z",
            "last_modified": "2025-11-14T00:00:00Z",
            "node_count": 0,
            "edge_count": 0,
            "graph_type": "directed",
            "schema_version": "1.0"
        })
        
        if progress and task_id:
            progress.update(task_id, description="🏗️ 添加节点类型定义...", advance=15)
        
        # 添加节点类型定义
        node_types = {
            "project": {"description": "项目根节点", "icon": "📁"},
            "module": {"description": "Python 模块", "icon": "📄"},
            "class": {"description": "Python 类", "icon": "🏛️"},
            "function": {"description": "Python 函数", "icon": "⚙️"},
            "document": {"description": "文档文件", "icon": "📖"},
            "speckit_artifact": {"description": "speckit 工件", "icon": "🔧"},
            "compliance_report": {"description": "合规报告", "icon": "📋"},
            "conversation": {"description": "对话记录", "icon": "💬"},
            "best_practice": {"description": "最佳实践", "icon": "⭐"},
            "vector_collection": {"description": "向量集合", "icon": "📊"}
        }
        
        # 添加节点类型作为特殊节点
        for node_type, metadata in node_types.items():
            main_graph.add_node(
                f"type:{node_type}",
                type="node_type",
                name=node_type,
                description=metadata["description"],
                icon=metadata["icon"],
                is_meta=True
            )
        
        if progress and task_id:
            progress.update(task_id, description="🏗️ 添加项目结构节点...", advance=20)
        
        # 添加项目结构节点
        project_nodes = _create_project_structure()
        for node_id, node_data in project_nodes.items():
            main_graph.add_node(node_id, **node_data)
        
        if progress and task_id:
            progress.update(task_id, description="🏗️ 添加关系边...", advance=20)
        
        # 添加项目结构关系
        project_edges = _create_project_relationships()
        logger.debug(f"📋 项目关系边数量: {len(project_edges)}")
        for i, edge in enumerate(project_edges):
            logger.debug(f"📋 边 {i}: {edge} (长度: {len(edge)})")
            if len(edge) == 3:
                source, target, edge_data = edge
                main_graph.add_edge(source, target, **edge_data)
            else:
                logger.warning(f"⚠️ 跳过格式错误的项目边 {i}: {edge}")
        
        if progress and task_id:
            progress.update(task_id, description="🏗️ 添加语义关系...", advance=15)
        
        # 添加语义关系
        semantic_edges = _create_semantic_relationships()
        logger.debug(f"🏷️ 语义关系边数量: {len(semantic_edges)}")
        for i, edge in enumerate(semantic_edges):
            logger.debug(f"🏷️ 语义边 {i}: {edge} (长度: {len(edge)})")
            if len(edge) == 3:
                source, target, edge_data = edge
                main_graph.add_edge(source, target, **edge_data)
            else:
                logger.warning(f"⚠️ 跳过格式错误的语义边 {i}: {edge}")
        
        # 更新统计信息
        main_graph.graph["node_count"] = main_graph.number_of_nodes()
        main_graph.graph["edge_count"] = main_graph.number_of_edges()
        main_graph.graph["last_modified"] = "2025-11-14T00:00:00Z"
        
        if progress and task_id:
            progress.update(task_id, description="💾 保存图数据...", advance=10)
        
        # 保存图数据
        logger.debug(f"💾 开始保存图数据，节点数: {main_graph.number_of_nodes()}, 边数: {main_graph.number_of_edges()}")
        await save_graph_to_file(main_graph, graph_path, graph_format)
        
        if progress and task_id:
            progress.update(task_id, description="✅ 完成!", advance=0)
        
        logger.info(f"✅ 图结构创建成功: {main_graph.number_of_nodes()} 节点, {main_graph.number_of_edges()} 边")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建图结构失败: {e}")
        if progress and task_id:
            progress.update(task_id, description=f"❌ 失败: {str(e)}", completed=100)
        return False


def _create_project_structure() -> Dict[str, Dict[str, Any]]:
    """创建项目结构节点"""
    nodes = {}
    
    # 项目根节点
    nodes["project:lumoscribe2033"] = {
        "type": "project",
        "name": "lumoscribe2033",
        "display_name": "Hybrid Graph-RAG 质量平台",
        "description": "基于 speckit 的 AI 驱动质量提升平台",
        "path": ".",
        "language": "python",
        "status": "active",
        "created_at": "2025-11-14T00:00:00Z",
        "tags": ["ai", "rag", "quality", "speckit"]
    }
    
    # 框架层节点
    framework_nodes = [
        ("module:src.framework", "src/framework", "框架层基础设施"),
        ("module:src.framework.shared", "src/framework/shared", "共享工具和配置"),
        ("module:src.framework.storage", "src/framework/storage", "存储抽象层"),
        ("module:src.framework.adapters", "src/framework/adapters", "适配器层"),
        ("class:Config", "src/framework/shared/config.py", "配置管理类"),
        ("class:Logger", "src/framework/shared/logging.py", "日志管理类"),
    ]
    
    for node_id, path, description in framework_nodes:
        nodes[node_id] = {
            "type": node_id.split(":")[0],
            "name": node_id.split(":")[1],
            "display_name": description,
            "path": path,
            "description": description,
            "layer": "framework",
            "status": "active",
            "created_at": "2025-11-14T00:00:00Z"
        }
    
    # 领域层节点
    domain_nodes = [
        ("module:src.domain", "src/domain", "领域层业务逻辑"),
        ("module:src.domain.pipeline", "src/domain/pipeline", "speckit 自动化管线"),
        ("module:src.domain.doc_review", "src/domain/doc_review", "文档三分法评估"),
        ("module:src.domain.compliance", "src/domain/compliance", "静态检查与可追溯性"),
        ("module:src.domain.knowledge", "src/domain/knowledge", "最佳实践与对话溯源"),
    ]
    
    for node_id, path, description in domain_nodes:
        nodes[node_id] = {
            "type": node_id.split(":")[0],
            "name": node_id.split(":")[1],
            "display_name": description,
            "path": path,
            "description": description,
            "layer": "domain",
            "status": "active",
            "created_at": "2025-11-14T00:00:00Z"
        }
    
    # API 层节点
    api_nodes = [
        ("module:src.api", "src/api", "FastAPI 接口层"),
        ("module:src.api.routes", "src/api/routes", "API 路由"),
        ("function:create_agent", "src/api/routes/speckit.py", "创建 AI 代理函数"),
        ("function:run_pipeline", "src/api/routes/speckit.py", "运行 speckit 管线函数"),
    ]
    
    for node_id, path, description in api_nodes:
        nodes[node_id] = {
            "type": node_id.split(":")[0],
            "name": node_id.split(":")[1],
            "display_name": description,
            "path": path,
            "description": description,
            "layer": "api",
            "status": "active",
            "created_at": "2025-11-14T00:00:00Z"
        }
    
    # 文档节点
    doc_nodes = [
        ("document:README.md", "README.md", "项目说明文档"),
        ("document:specs/001-hybrid-rag-platform/spec.md", "specs/001-hybrid-rag-platform/spec.md", "项目规格文档"),
        ("document:docs/internal/logs.md", "docs/internal/logs.md", "内部日志文档"),
        ("document:docs/external/user-guide.md", "docs/external/user-guide.md", "用户指南"),
    ]
    
    for node_id, path, description in doc_nodes:
        nodes[node_id] = {
            "type": "document",
            "name": node_id.split(":")[1],
            "display_name": description,
            "path": path,
            "description": description,
            "format": "markdown" if path.endswith(".md") else "text",
            "status": "active",
            "created_at": "2025-11-14T00:00:00Z"
        }
    
    # RAG 相关节点
    rag_nodes = [
        ("speckit_artifact:constitution", "artifacts/constitution", "项目章程"),
        ("speckit_artifact:specify", "artifacts/specify", "需求规格"),
        ("speckit_artifact:plan", "artifacts/plan", "项目计划"),
        ("speckit_artifact:tasks", "artifacts/tasks", "任务分解"),
        ("speckit_artifact:analyze", "artifacts/analyze", "分析报告"),
        ("speckit_artifact:implement", "artifacts/implement", "实现工件"),
    ]
    
    for node_id, path, description in rag_nodes:
        nodes[node_id] = {
            "type": "speckit_artifact",
            "name": node_id.split(":")[1],
            "display_name": description,
            "path": path,
            "description": description,
            "speckit_phase": node_id.split(":")[1],
            "status": "active",
            "created_at": "2025-11-14T00:00:00Z"
        }
    
    # 向量集合节点
    vector_nodes = [
        ("vector_collection:documents", "vector/chroma/documents", "文档向量集合"),
        ("vector_collection:code_snippets", "vector/chroma/code_snippets", "代码片段向量集合"),
        ("vector_collection:best_practices", "vector/chroma/best_practices", "最佳实践向量集合"),
        ("vector_collection:conversations", "vector/chroma/conversation_records", "对话记录向量集合"),
    ]
    
    for node_id, path, description in vector_nodes:
        nodes[node_id] = {
            "type": "vector_collection",
            "name": node_id.split(":")[1],
            "display_name": description,
            "path": path,
            "description": description,
            "embedding_model": "text-embedding-3-small",
            "vector_count": 0,
            "status": "active",
            "created_at": "2025-11-14T00:00:00Z"
        }
    
    return nodes


def _create_project_relationships() -> List[tuple]:
    """创建项目结构关系"""
    edges = []
    
    # 项目包含模块
    edges.extend([
        ("project:lumoscribe2033", "module:src.framework", {
            "type": "contains",
            "description": "项目包含框架层",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("project:lumoscribe2033", "module:src.domain", {
            "type": "contains",
            "description": "项目包含领域层",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("project:lumoscribe2033", "module:src.api", {
            "type": "contains",
            "description": "项目包含接口层",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("project:lumoscribe2033", "document:README.md", {
            "type": "contains",
            "description": "项目包含说明文档",
            "weight": 0.8,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    # 框架层内部关系
    edges.extend([
        ("module:src.framework", "module:src.framework.shared", {
            "type": "contains",
            "description": "框架层包含共享模块",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("module:src.framework.shared", "class:Config", {
            "type": "contains",
            "description": "共享模块包含配置类",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    # 领域层内部关系
    edges.extend([
        ("module:src.domain", "module:src.domain.pipeline", {
            "type": "contains",
            "description": "领域层包含管线模块",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("module:src.domain", "module:src.domain.compliance", {
            "type": "contains",
            "description": "领域层包含合规模块",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    # API 层内部关系
    edges.extend([
        ("module:src.api", "module:src.api.routes", {
            "type": "contains",
            "description": "API 层包含路由模块",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("module:src.api.routes", "function:create_agent", {
            "type": "contains",
            "description": "路由模块包含创建代理函数",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    # 依赖关系
    edges.extend([
        ("module:src.api", "module:src.framework", {
            "type": "depends_on",
            "description": "API 层依赖框架层",
            "weight": 0.9,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("module:src.domain", "module:src.framework", {
            "type": "depends_on",
            "description": "领域层依赖框架层",
            "weight": 0.9,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    return edges


def _create_semantic_relationships() -> List[tuple]:
    """创建语义关系"""
    edges = []
    
    # 文档与向量集合的关系
    edges.extend([
        ("document:README.md", "vector_collection:documents", {
            "type": "stored_in",
            "description": "文档存储在向量集合中",
            "weight": 0.8,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("document:specs/001-hybrid-rag-platform/spec.md", "vector_collection:documents", {
            "type": "stored_in",
            "description": "规格文档存储在向量集合中",
            "weight": 0.8,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    # speckit 工件与领域模块的关系
    edges.extend([
        ("speckit_artifact:constitution", "module:src.domain.pipeline", {
            "type": "processed_by",
            "description": "章程工件由管线模块处理",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("speckit_artifact:analyze", "module:src.domain.compliance", {
            "type": "analyzed_by",
            "description": "分析工件由合规模块分析",
            "weight": 1.0,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    # 配置与模块的关系
    edges.extend([
        ("class:Config", "module:src.framework.storage", {
            "type": "configures",
            "description": "配置类配置存储模块",
            "weight": 0.9,
            "created_at": "2025-11-14T00:00:00Z"
        }),
        ("class:Config", "vector_collection:documents", {
            "type": "configures",
            "description": "配置类配置向量集合",
            "weight": 0.9,
            "created_at": "2025-11-14T00:00:00Z"
        }),
    ])
    
    return edges


async def save_graph_to_file(
    graph: nx.Graph,
    graph_path: Path,
    graph_format: str = "gexf"
) -> None:
    """保存图到文件"""
    timestamp = "20251114_000000"
    
    logger.debug(f"💾 开始保存图文件，格式: {graph_format}, 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    
    if graph_format.lower() == "gexf":
        file_path = graph_path / f"lumoscribe_graph_{timestamp}.gexf"
        logger.debug(f"💾 保存为 GEXF 格式: {file_path}")
        try:
            nx.write_gexf(graph, file_path)
            logger.info(f"✅ 保存为 GEXF 格式: {file_path}")
        except Exception as e:
            logger.error(f"❌ GEXF 保存失败: {e}")
            raise
    
    elif graph_format.lower() == "gml":
        file_path = graph_path / f"lumoscribe_graph_{timestamp}.gml"
        logger.debug(f"💾 保存为 GML 格式: {file_path}")
        try:
            nx.write_gml(graph, file_path)
            logger.info(f"✅ 保存为 GML 格式: {file_path}")
        except Exception as e:
            logger.error(f"❌ GML 保存失败: {e}")
            raise
    
    elif graph_format.lower() == "graphml":
        file_path = graph_path / f"lumoscribe_graph_{timestamp}.graphml"
        logger.debug(f"💾 保存为 GraphML 格式: {file_path}")
        try:
            nx.write_graphml(graph, file_path)
            logger.info(f"✅ 保存为 GraphML 格式: {file_path}")
        except Exception as e:
            logger.error(f"❌ GraphML 保存失败: {e}")
            raise
    
    elif graph_format.lower() == "json":
        file_path = graph_path / f"lumoscribe_graph_{timestamp}.json"
        logger.debug(f"💾 保存为 JSON 格式: {file_path}")
        
        # 转换为 JSON 格式
        graph_data = {
            "graph": dict(graph.graph),
            "nodes": [],
            "edges": []
        }
        
        # 处理节点
        logger.debug(f"📝 处理 {graph.number_of_nodes()} 个节点")
        for node_id, node_data in graph.nodes(data=True):
            node_data = node_data.copy()  # 避免修改原始数据
            node_data["id"] = node_id
            graph_data["nodes"].append(node_data)
        
        # 处理边
        logger.debug(f"🔗 处理 {graph.number_of_edges()} 个边")
        edges_list = list(graph.edges(data=True))
        logger.debug(f"🔗 边列表长度: {len(edges_list)}")
        
        for i, edge in enumerate(edges_list):
            logger.debug(f"🔗 处理边 {i}: {edge} (类型: {type(edge)}, 长度: {len(edge) if hasattr(edge, '__len__') else 'N/A'})")
            if len(edge) == 3:
                source, target, edge_data = edge
                edge_data = edge_data.copy()  # 避免修改原始数据
                edge_data.update({
                    "source": source,
                    "target": target
                })
                graph_data["edges"].append(edge_data)
            else:
                logger.warning(f"⚠️ 跳过格式错误的边 {i}: {edge}")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 保存为 JSON 格式: {file_path}")
        except Exception as e:
            logger.error(f"❌ JSON 保存失败: {e}")
            raise
    
    else:
        raise ValueError(f"不支持的格式: {graph_format}")


def display_graph_info(graph_path: Path, graph_format: str = "gexf") -> None:
    """显示图信息"""
    console.print("\n[bold]📊 图结构信息：[/bold]")
    
    # 显示文件信息
    tree = Tree(f"📁 图文件目录: {graph_path}")
    
    for file_path in graph_path.glob("*.gexf"):
        tree.add(f"📄 {file_path.name} ({file_path.stat().st_size} bytes)")
    
    for file_path in graph_path.glob("*.gml"):
        tree.add(f"📄 {file_path.name} ({file_path.stat().st_size} bytes)")
    
    for file_path in graph_path.glob("*.graphml"):
        tree.add(f"📄 {file_path.name} ({file_path.stat().st_size} bytes)")
    
    for file_path in graph_path.glob("*.json"):
        tree.add(f"📄 {file_path.name} ({file_path.stat().st_size} bytes)")
    
    console.print(tree)
    
    # 加载并显示图统计信息
    try:
        # 查找最新的图文件
        graph_files = list(graph_path.glob(f"lumoscribe_graph_*.{graph_format}"))
        if graph_files:
            latest_file = max(graph_files, key=lambda x: x.stat().st_mtime)
            
            if graph_format == "gexf":
                graph = nx.read_gexf(latest_file)
            elif graph_format == "gml":
                graph = nx.read_gml(latest_file)
            elif graph_format == "graphml":
                graph = nx.read_graphml(latest_file)
            elif graph_format == "json":
                with open(latest_file, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                graph = nx.DiGraph()
                graph.graph.update(graph_data["graph"])
                
                for node_data in graph_data["nodes"]:
                    node_id = node_data.pop("id")
                    graph.add_node(node_id, **node_data)
                
                for edge_data in graph_data["edges"]:
                    source = edge_data.pop("source")
                    target = edge_data.pop("target")
                    graph.add_edge(source, target, **edge_data)
            
            # 显示统计信息
            table = Table(title="📈 图统计信息")
            table.add_column("指标", style="cyan")
            table.add_column("值", style="magenta")
            
            table.add_row("节点数量", str(graph.number_of_nodes()))
            table.add_row("边数量", str(graph.number_of_edges()))
            table.add_row("图类型", str(type(graph).__name__))
            table.add_row("连通分量", str(nx.number_weakly_connected_components(graph)))
            
            if hasattr(graph, 'graph') and graph.graph:
                graph_info = graph.graph
                table.add_row("版本", graph_info.get("version", "未知"))
                table.add_row("描述", graph_info.get("description", "无"))
            
            console.print(table)
            
            # 显示节点类型分布
            node_types = {}
            for _, data in graph.nodes(data=True):
                node_type = data.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_types:
                type_table = Table(title="🏷️ 节点类型分布")
                type_table.add_column("类型", style="cyan")
                type_table.add_column("数量", style="yellow")
                
                for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                    type_table.add_row(node_type, str(count))
                
                console.print(type_table)
    
    except Exception as e:
        logger.warning(f"⚠️ 读取图文件失败: {e}")


def main(
    path: str = typer.Option(
        "graph/snapshots",
        "--path",
        "-p",
        help="图文件存储路径"
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="图文件格式 (gexf, gml, graphml, json)"
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="重置现有图结构"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="显示详细日志"
    )
):
    """
    主函数 - 初始化 NetworkX 图结构
    
    Args:
        path: 图文件存储路径
        format: 图文件格式
        reset: 是否重置
        verbose: 是否显示详细日志
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(Panel.fit(
        "[bold blue]🚀 lumoscribe2033 NetworkX 图数据库初始化[/bold blue]\n"
        "为 Hybrid Graph-RAG 质量平台创建图关系结构",
        border_style="blue"
    ))
    
    graph_path = Path(path)
    
    # 创建进度条
    with Progress(
        "[progress.description]{task.description}",
        "[progress.bar]{task.completed:>3d}/{task.total:>3d}",
        "• [progress.percentage]{task.percentage:>3.0f}%",
        console=console,
        transient=True
    ) as progress:
        task_id = progress.add_task("初始化图结构", total=100, start=False)
        
        # 执行初始化
        success = asyncio.run(create_lumoscribe_graph(
            graph_path=graph_path,
            graph_format=format,
            reset=reset,
            progress=progress,
            task_id=task_id
        ))
        
        if success:
            console.print("\n[green]✅ NetworkX 图结构初始化成功！[/green]")
            
            # 显示图信息
            display_graph_info(graph_path, format)
            
            # 显示下一步操作
            console.print("\n[bold]下一步操作：[/bold]")
            console.print("• 可视化图结构: [cyan]python -c \"import networkx as nx; g=nx.read_gexf('graph/snapshots/lumoscribe_graph_20251114_000000.gexf'); print('节点:', list(g.nodes())[:10])\"[/cyan]")
            console.print("• 分析图结构: [cyan]使用 NetworkX 分析工具包[/cyan]")
            console.print("• 添加数据: [cyan]通过 RAG 系统自动构建关系图[/cyan]")
        else:
            console.print("\n[red]❌ NetworkX 图结构初始化失败！[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)