"""
增强版图存储管理器

基于 LlamaIndex GraphStore 最佳实践，提供：
- NetworkX 图存储后端支持
- LlamaIndex GraphStore 接口集成
- 知识图谱索引支持
- 图分析和可视化功能
"""

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
from llama_index.core import StorageContext
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.indices import KnowledgeGraphIndex, PropertyGraphIndex
from llama_index.core.schema import NodeWithScore
from pyvis.network import Network

from src.framework.shared.exceptions import GraphStoreError
from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class GraphStoreBackend(ABC):
    """图存储后端抽象接口"""

    @abstractmethod
    def get_storage_context(self) -> StorageContext:
        """获取存储上下文"""
        pass

    @abstractmethod
    def create_knowledge_graph_index(self, documents: list[Any], **kwargs) -> KnowledgeGraphIndex:
        """创建知识图谱索引"""
        pass

    @abstractmethod
    def create_property_graph_index(self, documents: list[Any], **kwargs) -> PropertyGraphIndex:
        """创建属性图索引"""
        pass

    @abstractmethod
    def get_graph_stats(self) -> dict[str, Any]:
        """获取图统计信息"""
        pass


class NetworkXGraphBackend:
    """NetworkX 图存储后端"""

    def __init__(self, db_path: str = "./graph/snapshots/graph.gexf"):
        self.db_path = db_path
        self.graph = nx.Graph()
        self._load_graph()

    def _load_graph(self):
        """加载图数据"""
        try:
            if os.path.exists(self.db_path):
                self.graph = nx.read_gexf(self.db_path)
                logger.info(f"从 {self.db_path} 加载图数据成功")
            else:
                logger.info("创建新的空图")
        except Exception as e:
            logger.warning(f"加载图数据失败，创建空图: {e}")
            self.graph = nx.Graph()

    def _save_graph(self):
        """保存图数据"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            nx.write_gexf(self.graph, self.db_path)
            logger.info(f"图数据保存成功: {self.db_path}")
        except Exception as e:
            logger.error(f"保存图数据失败: {e}")

    def get_storage_context(self) -> StorageContext:
        """NetworkX 不直接支持 StorageContext，返回空上下文"""
        return StorageContext.from_defaults()

    def create_knowledge_graph_index(self, documents: list[Any], **kwargs) -> KnowledgeGraphIndex:
        """创建知识图谱索引"""
        # NetworkX 后端的知识图谱索引需要特殊处理
        # 这里返回一个简化的实现
        logger.info("NetworkX 后端的知识图谱索引功能有限")
        return None

    def create_property_graph_index(self, documents: list[Any], **kwargs) -> PropertyGraphIndex:
        """创建属性图索引"""
        # NetworkX 后端的属性图索引需要特殊处理
        logger.info("NetworkX 后端的属性图索引功能有限")
        return None

    def get_graph_stats(self) -> dict[str, Any]:
        """获取图统计信息"""
        try:
            stats = {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "is_connected": nx.is_connected(self.graph),
                "avg_clustering": nx.average_clustering(self.graph),
                "diameter": None,
                "avg_shortest_path": None,
                "backend": "networkx",
                "db_path": self.db_path
            }

            # 计算直径和平均最短路径（仅对连通图）
            if nx.is_connected(self.graph) and self.graph.number_of_nodes() > 1:
                stats["diameter"] = nx.diameter(self.graph)
                stats["avg_shortest_path"] = nx.average_shortest_path_length(self.graph)

            return stats

        except Exception as e:
            logger.error(f"计算图统计信息失败: {e}")
            return {
                "nodes": 0,
                "edges": 0,
                "error": str(e),
                "backend": "networkx"
            }


class EnhancedGraphStoreManager:
    """增强版图存储管理器"""

    def __init__(self, **backend_config):
        """初始化增强版图存储管理器

        Args:
            **backend_config: 后端配置参数
        """
        self.backend_config = backend_config
        self.backends: dict[str, GraphStoreBackend] = {}
        self.index_cache: dict[str, Any] = {}

        # 初始化默认后端
        self._init_backend("default")

        logger.info("增强版图存储管理器初始化完成，使用 NetworkX 后端")

    def _init_backend(self, name: str):
        """初始化后端"""
        try:
            backend = NetworkXGraphBackend(**self.backend_config)
            self.backends[name] = backend
            logger.info(f"后端 {name} 初始化成功: networkx")

        except Exception as e:
            logger.error(f"后端 {name} 初始化失败: {e}")
            raise

    def get_backend(self, name: str = "default") -> GraphStoreBackend:
        """获取后端"""
        if name not in self.backends:
            self._init_backend(name)
        return self.backends[name]

    def get_storage_context(self, backend_name: str = "default") -> StorageContext:
        """获取存储上下文"""
        backend = self.get_backend(backend_name)
        return backend.get_storage_context()

    def create_knowledge_graph_index(
        self,
        documents: list[Any],
        backend_name: str = "default",
        **kwargs
    ) -> KnowledgeGraphIndex:
        """创建知识图谱索引"""
        try:
            backend = self.get_backend(backend_name)
            index = backend.create_knowledge_graph_index(documents, **kwargs)

            if index:
                cache_key = f"kg_{backend_name}"
                self.index_cache[cache_key] = index

            return index

        except Exception as e:
            logger.error(f"创建知识图谱索引失败: {e}")
            raise GraphStoreError(f"创建知识图谱索引失败: {e}")

    def create_property_graph_index(
        self,
        documents: list[Any],
        backend_name: str = "default",
        **kwargs
    ) -> PropertyGraphIndex:
        """创建属性图索引"""
        try:
            backend = self.get_backend(backend_name)
            index = backend.create_property_graph_index(documents, **kwargs)

            if index:
                cache_key = f"pg_{backend_name}"
                self.index_cache[cache_key] = index

            return index

        except Exception as e:
            logger.error(f"创建属性图索引失败: {e}")
            raise GraphStoreError(f"创建属性图索引失败: {e}")

    def get_graph_stats(self, backend_name: str = "default") -> dict[str, Any]:
        """获取图统计信息"""
        backend = self.get_backend(backend_name)
        stats = backend.get_graph_stats()
        stats["backend_name"] = backend_name
        return stats

    def visualize_graph(
        self,
        backend_name: str = "default",
        output_file: str = "graph_visualization.html",
        **kwargs
    ) -> str:
        """可视化图"""
        try:
            backend = self.get_backend(backend_name)

            # NetworkX 图可视化
            network = Network(
                notebook=False,
                directed=True,
                cdn_resources="in_line",
                **kwargs
            )

            # 转换 NetworkX 图为 PyVis 网络
            network.from_nx(backend.graph)

            # 保存可视化文件
            network.show(output_file)

            logger.info(f"图可视化文件已保存: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"图可视化失败: {e}")
            raise GraphStoreError(f"图可视化失败: {e}")

    def list_backends(self) -> list[str]:
        """列出所有后端"""
        return list(self.backends.keys())

    def add_backend(
        self,
        name: str,
        **backend_config
    ) -> bool:
        """添加新的后端"""
        try:
            # 保存当前配置
            original_config = self.backend_config

            # 临时切换到新配置
            self.backend_config = backend_config

            # 初始化新后端
            self._init_backend(name)

            # 恢复原始配置
            self.backend_config = original_config

            return True

        except Exception as e:
            logger.error(f"添加后端失败: {e}")
            return False

    @contextmanager
    def transaction(self, backend_name: str = "default"):
        """事务上下文管理器"""
        try:
            yield
        except Exception as e:
            logger.error(f"图存储事务失败: {e}")
            raise

    # NetworkX 原生方法 - 直接操作图
    def add_node(self, node_id: str, node_data: dict[str, Any] = None, backend_name: str = "default"):
        """添加节点"""
        backend = self.get_backend(backend_name)
        backend.graph.add_node(node_id, **(node_data or {}))
        backend._save_graph()

    def add_edge(self, source_id: str, target_id: str, edge_data: dict[str, Any] = None, backend_name: str = "default"):
        """添加边"""
        backend = self.get_backend(backend_name)
        backend.graph.add_edge(source_id, target_id, **(edge_data or {}))
        backend._save_graph()

    def get_neighbors(self, node_id: str, backend_name: str = "default") -> list[str]:
        """获取邻居节点"""
        backend = self.get_backend(backend_name)
        return list(backend.graph.neighbors(node_id))

    def shortest_path(self, source_id: str, target_id: str, backend_name: str = "default") -> list[str]:
        """最短路径"""
        backend = self.get_backend(backend_name)
        try:
            return nx.shortest_path(backend.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, nodes: list[str], backend_name: str = "default") -> nx.Graph:
        """获取子图"""
        backend = self.get_backend(backend_name)
        return backend.graph.subgraph(nodes)

    def remove_node(self, node_id: str, backend_name: str = "default"):
        """移除节点"""
        backend = self.get_backend(backend_name)
        backend.graph.remove_node(node_id)
        backend._save_graph()

    def remove_edge(self, source_id: str, target_id: str, backend_name: str = "default"):
        """移除边"""
        backend = self.get_backend(backend_name)
        backend.graph.remove_edge(source_id, target_id)
        backend._save_graph()
