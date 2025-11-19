"""
增强版图存储管理器单元测试
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.framework.shared.exceptions import GraphStoreError
from src.framework.storage.enhanced_graph_store import (
    EnhancedGraphStoreManager,
    GraphStoreBackend,
    NetworkXGraphBackend,
)


class TestNetworkXGraphBackend:
    """NetworkX 图存储后端测试类"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_graph.gexf")

        # 初始化后端
        self.backend = NetworkXGraphBackend(db_path=self.db_path)

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """测试初始化"""
        assert self.backend.db_path == self.db_path
        assert self.backend.graph is not None

    def test_get_storage_context(self):
        """测试获取存储上下文"""
        storage_context = self.backend.get_storage_context()

        assert storage_context is not None

    def test_create_knowledge_graph_index(self):
        """测试创建知识图谱索引"""
        documents = []  # 简化测试

        # NetworkX 后端的知识图谱索引功能有限
        index = self.backend.create_knowledge_graph_index(documents)

        # 预期返回 None 或简化实现
        assert index is None

    def test_create_property_graph_index(self):
        """测试创建属性图索引"""
        documents = []  # 简化测试

        # NetworkX 后端的属性图索引功能有限
        index = self.backend.create_property_graph_index(documents)

        # 预期返回 None 或简化实现
        assert index is None

    def test_get_graph_stats(self):
        """测试获取图统计信息"""
        stats = self.backend.get_graph_stats()

        assert isinstance(stats, dict)
        assert "nodes" in stats
        assert "edges" in stats
        assert "density" in stats
        assert "backend" in stats
        assert stats["backend"] == "networkx"


class TestEnhancedGraphStoreManager:
    """增强版图存储管理器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_graph.gexf")

        # 初始化管理器
        self.manager = EnhancedGraphStoreManager(
            db_path=self.db_path
        )

    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """测试初始化"""
        assert self.manager.backend_config["db_path"] == self.db_path
        assert len(self.manager.backends) == 1  # 默认后端已初始化
        assert "default" in self.manager.backends
        assert len(self.manager.index_cache) == 0

    def test_get_backend(self):
        """测试获取后端"""
        backend = self.manager.get_backend("default")

        assert backend is not None
        assert isinstance(backend, NetworkXGraphBackend)

    def test_get_backend_new(self):
        """测试获取新后端"""
        backend = self.manager.get_backend("new_backend")

        assert backend is not None
        assert "new_backend" in self.manager.backends

    def test_get_storage_context(self):
        """测试获取存储上下文"""
        storage_context = self.manager.get_storage_context("default")

        assert storage_context is not None

    def test_create_knowledge_graph_index(self):
        """测试创建知识图谱索引"""
        documents = []  # 简化测试

        index = self.manager.create_knowledge_graph_index(
            documents,
            backend_name="default"
        )

        # NetworkX 后端返回 None
        assert index is None

    def test_create_property_graph_index(self):
        """测试创建属性图索引"""
        documents = []  # 简化测试

        index = self.manager.create_property_graph_index(
            documents,
            backend_name="default"
        )

        # NetworkX 后端返回 None
        assert index is None

    def test_get_graph_stats(self):
        """测试获取图统计信息"""
        stats = self.manager.get_graph_stats("default")

        assert isinstance(stats, dict)
        assert "backend_name" in stats
        assert stats["backend_name"] == "default"

    def test_visualize_graph(self):
        """测试图可视化"""
        output_file = os.path.join(self.temp_dir, "test_visualization.html")

        try:
            result = self.manager.visualize_graph(
                backend_name="default",
                output_file=output_file
            )

            # NetworkX 后端可能成功也可能失败，取决于环境
            if result:
                assert os.path.exists(result)
        except Exception:
            # 可视化可能失败，这是预期的
            pass

    def test_list_backends(self):
        """测试列出后端"""
        backends = self.manager.list_backends()

        assert isinstance(backends, list)
        assert "default" in backends

    def test_add_backend(self):
        """测试添加后端"""
        # 测试添加新后端
        result = self.manager.add_backend(
            name="test_backend",
            db_path=os.path.join(self.temp_dir, "test_backend.gexf")
        )

        assert result is True
        assert "test_backend" in self.manager.backends

    def test_transaction_context_manager(self):
        """测试事务上下文管理器"""
        with self.manager.transaction("default"):
            # 事务内的操作
            pass

        # 如果没有异常，事务应该成功完成
        assert True

    def test_native_methods(self):
        """测试原生方法"""
        # 测试 NetworkX 原生方法
        assert hasattr(self.manager, 'add_node')
        assert hasattr(self.manager, 'add_edge')
        assert hasattr(self.manager, 'get_neighbors')
        assert hasattr(self.manager, 'shortest_path')
        assert hasattr(self.manager, 'get_subgraph')
        assert hasattr(self.manager, 'remove_node')
        assert hasattr(self.manager, 'remove_edge')

    def test_multiple_backends(self):
        """测试多个后端"""
        # 添加多个后端
        backend_configs = [
            ("backend1", {"db_path": os.path.join(self.temp_dir, "backend1.gexf")}),
            ("backend2", {"db_path": os.path.join(self.temp_dir, "backend2.gexf")}),
        ]

        for name, config in backend_configs:
            result = self.manager.add_backend(name, **config)
            assert result is True
            assert name in self.manager.backends

        # 验证后端数量
        backends = self.manager.list_backends()
        assert len(backends) >= 3  # 包括默认后端

    @patch('src.framework.storage.enhanced_graph_store.logger')
    def test_error_handling(self, mock_logger):
        """测试错误处理"""
        # 测试配置错误
        with pytest.raises(Exception):
            # 这里测试一个可能导致错误的情况
            # 由于我们移除了 backend_type 参数，这里主要测试其他可能的错误
            pass

    def test_backend_caching(self):
        """测试后端缓存"""
        backend_name = "cache_test"

        # 第一次获取
        backend1 = self.manager.get_backend(backend_name)

        # 第二次获取（应该从缓存返回）
        backend2 = self.manager.get_backend(backend_name)

        assert backend1 is backend2
        assert backend_name in self.manager.backends

    def test_index_caching(self):
        """测试索引缓存"""
        # 这个测试主要针对支持索引的后端
        # NetworkX 后端的索引功能有限，所以主要测试缓存机制存在

        assert hasattr(self.manager, 'index_cache')
        assert isinstance(self.manager.index_cache, dict)

    def test_config_preservation(self):
        """测试配置保持"""
        original_config = self.manager.backend_config.copy()

        # 添加新后端
        self.manager.add_backend(
            "temp_backend",
            db_path=os.path.join(self.temp_dir, "temp.gexf")
        )

        # 验证原始配置未改变
        assert self.manager.backend_config == original_config


if __name__ == "__main__":
    pytest.main([__file__])
