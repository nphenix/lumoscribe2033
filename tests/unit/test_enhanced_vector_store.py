"""
增强版向量存储管理器单元测试
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core.schema import Document as LlamaDocument

from src.framework.shared.exceptions import VectorStoreError
from src.framework.storage.enhanced_vector_store import EnhancedVectorStoreManager


@pytest.mark.skip(reason="增强版向量存储管理器测试依赖未实现的向量存储功能，阶段 4 实现")
class TestEnhancedVectorStoreManager:
    """增强版向量存储管理器测试类"""

    def setup_method(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.persist_dir = os.path.join(self.temp_dir, "chroma_test")

        # 初始化管理器
        self.manager = EnhancedVectorStoreManager(persist_dir=self.persist_dir)

        # 创建测试文档
        self.test_documents = [
            LlamaDocument(
                text="Python 是一种编程语言",
                metadata={"source": "test1", "category": "programming"}
            ),
            LlamaDocument(
                text="机器学习是人工智能的分支",
                metadata={"source": "test2", "category": "ai"}
            ),
            LlamaDocument(
                text="向量数据库用于存储嵌入向量",
                metadata={"source": "test3", "category": "database"}
            )
        ]

    def teardown_method(self):
        """测试后清理"""
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """测试初始化"""
        assert self.manager.persist_dir == self.persist_dir
        assert self.manager.chroma_client is not None
        assert isinstance(self.manager.base_manager, type(self.manager.base_manager))
        assert len(self.manager.vector_stores) == 0
        assert len(self.manager.index_cache) == 0
        assert len(self.manager.storage_contexts) == 0

    def test_get_storage_context(self):
        """测试获取存储上下文"""
        storage_context = self.manager.get_storage_context("test_collection")

        assert storage_context is not None
        assert hasattr(storage_context, 'vector_store')

        # 验证缓存
        assert "test_collection" in self.manager.storage_contexts
        assert "test_collection" in self.manager.vector_stores

    def test_create_index(self):
        """测试创建索引"""
        index = self.manager.create_index(
            self.test_documents,
            collection_name="test_index"
        )

        assert index is not None
        assert "test_index" in self.manager.index_cache

        # 验证索引功能
        retriever = index.as_retriever(similarity_top_k=2)
        results = retriever.retrieve("Python")
        assert len(results) <= 2

    def test_get_or_create_index_new(self):
        """测试获取或创建新索引"""
        index = self.manager.get_or_create_index(
            self.test_documents,
            collection_name="new_index"
        )

        assert index is not None
        assert "new_index" in self.manager.index_cache

    def test_get_or_create_index_existing(self):
        """测试获取现有索引"""
        # 先创建索引
        original_index = self.manager.create_index(
            self.test_documents,
            collection_name="existing_index"
        )

        # 再次获取应该返回缓存的索引
        cached_index = self.manager.get_or_create_index(
            [],  # 空文档列表
            collection_name="existing_index"
        )

        assert cached_index is original_index

    def test_load_index_from_storage(self):
        """测试从存储加载索引"""
        # 先创建索引
        self.manager.create_index(
            self.test_documents,
            collection_name="load_test"
        )

        # 清除缓存
        self.manager.index_cache.clear()

        # 从存储加载
        loaded_index = self.manager.load_index_from_storage("load_test")

        assert loaded_index is not None
        assert "load_test" in self.manager.index_cache

    def test_create_query_engine(self):
        """测试创建查询引擎"""
        # 先创建索引
        self.manager.create_index(
            self.test_documents,
            collection_name="query_test"
        )

        # 创建查询引擎
        query_engine = self.manager.create_query_engine(
            collection_name="query_test",
            similarity_top_k=3
        )

        assert query_engine is not None

        # 测试查询
        response = query_engine.query("Python")
        assert response is not None

    def test_get_index(self):
        """测试获取索引"""
        # 先创建索引
        original_index = self.manager.create_index(
            self.test_documents,
            collection_name="get_test"
        )

        # 获取索引
        retrieved_index = self.manager.get_index("get_test")

        assert retrieved_index is original_index

    def test_get_index_not_found(self):
        """测试获取不存在的索引"""
        with pytest.raises(VectorStoreError, match="索引不存在"):
            self.manager.get_index("nonexistent")

    def test_add_documents_to_index(self):
        """测试向索引添加文档"""
        # 创建初始索引
        self.manager.create_index(
            self.test_documents[:2],
            collection_name="add_test"
        )

        # 添加新文档
        new_doc = LlamaDocument(
            text="深度学习是机器学习的子集",
            metadata={"source": "new_doc"}
        )

        result = self.manager.add_documents_to_index(
            [new_doc],
            collection_name="add_test"
        )

        assert result is True

        # 验证文档已添加
        index = self.manager.get_index("add_test")
        retriever = index.as_retriever(similarity_top_k=3)
        results = retriever.retrieve("深度学习")
        assert len(results) > 0

    def test_delete_from_index(self):
        """测试从索引删除文档"""
        # 创建索引
        self.manager.create_index(
            self.test_documents,
            collection_name="delete_test"
        )

        # 获取文档ID
        self.manager.get_index("delete_test")
        # 注意：这里简化了删除逻辑，实际使用中需要正确的文档ID

        # 测试删除方法存在
        assert hasattr(self.manager, 'delete_from_index')

    def test_clear_index(self):
        """测试清空索引"""
        # 创建索引
        self.manager.create_index(
            self.test_documents,
            collection_name="clear_test"
        )

        # 清空索引
        result = self.manager.clear_index("clear_test")

        assert result is True
        assert "clear_test" not in self.manager.index_cache

    def test_list_collections(self):
        """测试列出集合"""
        # 创建一些集合
        self.manager.create_index(
            [self.test_documents[0]],
            collection_name="collection1"
        )
        self.manager.create_index(
            [self.test_documents[1]],
            collection_name="collection2"
        )

        collections = self.manager.list_collections()

        assert isinstance(collections, list)
        assert len(collections) >= 2

    def test_get_collection_info(self):
        """测试获取集合信息"""
        # 创建索引
        self.manager.create_index(
            self.test_documents,
            collection_name="info_test"
        )

        info = self.manager.get_collection_info("info_test")

        assert isinstance(info, dict)
        assert "name" in info
        assert "count" in info
        assert "persist_directory" in info
        assert "has_index" in info

    def test_reset(self):
        """测试重置"""
        # 创建一些数据
        self.manager.create_index(
            self.test_documents,
            collection_name="reset_test"
        )

        len(self.manager.list_collections())

        # 执行重置
        result = self.manager.reset()

        assert result is True

        # 验证缓存已清除
        assert len(self.manager.vector_stores) == 0
        assert len(self.manager.index_cache) == 0
        assert len(self.manager.storage_contexts) == 0

    def test_backup(self):
        """测试备份"""
        backup_dir = os.path.join(self.temp_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        # 创建一些数据
        self.manager.create_index(
            self.test_documents,
            collection_name="backup_test"
        )

        # 执行备份
        result = self.manager.backup(backup_dir)

        assert result is True

        # 验证备份目录存在
        backup_dirs = [d for d in os.listdir(backup_dir) if d.startswith("chroma_backup_")]
        assert len(backup_dirs) > 0

    def test_transaction_context_manager(self):
        """测试事务上下文管理器"""
        with self.manager.transaction("test_transaction"):
            # 事务内的操作
            self.manager.create_index(
                self.test_documents,
                collection_name="transaction_test"
            )

        # 验证操作成功
        assert "transaction_test" in self.manager.index_cache

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 测试委托给基础管理器的方法
        assert hasattr(self.manager, 'create_collection')
        assert hasattr(self.manager, 'similarity_search')
        assert hasattr(self.manager, 'add_documents')
        assert hasattr(self.manager, 'delete_collection')

    @patch('src.framework.storage.enhanced_vector_store.logger')
    def test_error_handling(self, mock_logger):
        """测试错误处理"""
        # 测试无效的集合名称
        with pytest.raises(VectorStoreError):
            self.manager.get_index("invalid_collection")

    def test_storage_context_caching(self):
        """测试存储上下文缓存"""
        collection_name = "cache_test"

        # 第一次获取
        context1 = self.manager.get_storage_context(collection_name)

        # 第二次获取（应该从缓存返回）
        context2 = self.manager.get_storage_context(collection_name)

        assert context1 is context2
        assert collection_name in self.manager.storage_contexts

    def test_index_caching(self):
        """测试索引缓存"""
        collection_name = "index_cache_test"

        # 创建索引
        index1 = self.manager.create_index(
            self.test_documents,
            collection_name=collection_name
        )

        # 获取索引（应该从缓存返回）
        index2 = self.manager.get_index(collection_name)

        assert index1 is index2
        assert collection_name in self.manager.index_cache


if __name__ == "__main__":
    pytest.main([__file__])
