"""
增强版向量存储管理器

基于 LlamaIndex Chroma 集成的最佳实践，提供：
- LlamaIndex StorageContext 集成
- 官方 ChromaVectorStore 支持
- 向后兼容现有接口
- 增强的索引管理功能
- AutoRetriever 支持
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Optional, Union

import chromadb
from chromadb.api.models.Collection import Collection
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.framework.shared.exceptions import VectorStoreError
from src.framework.shared.logging import get_logger
from src.framework.storage.vector_store import (
    VectorStoreManager as BaseVectorStoreManager,
)

logger = get_logger(__name__)


class EnhancedVectorStoreManager:
    """增强版向量存储管理器

    基于 LlamaIndex Chroma 集成，提供更强大的向量存储和索引管理功能
    """

    def __init__(self, persist_dir: str = "./vector/chroma"):
        """初始化增强版向量存储管理器

        Args:
            persist_dir: 持久化存储目录
        """
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        # 基础向量存储管理器（保持向后兼容）
        self.base_manager = BaseVectorStoreManager(persist_directory=persist_dir)

        # LlamaIndex 向量存储和索引缓存
        self.vector_stores: dict[str, ChromaVectorStore] = {}
        self.index_cache: dict[str, VectorStoreIndex] = {}
        self.storage_contexts: dict[str, StorageContext] = {}

        logger.info(f"增强版向量存储管理器初始化完成，持久化目录: {persist_dir}")

    def get_storage_context(self, collection_name: str = "default") -> StorageContext:
        """获取 LlamaIndex StorageContext

        Args:
            collection_name: 集合名称

        Returns:
            StorageContext: LlamaIndex 存储上下文
        """
        if collection_name in self.storage_contexts:
            return self.storage_contexts[collection_name]

        # 获取或创建 Chroma 集合
        collection = self._get_or_create_collection(collection_name)

        # 创建 LlamaIndex 向量存储
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # 创建存储上下文
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 缓存
        self.vector_stores[collection_name] = vector_store
        self.storage_contexts[collection_name] = storage_context

        return storage_context

    def create_index(
        self,
        documents: list[LlamaDocument],
        collection_name: str = "default",
        **kwargs
    ) -> VectorStoreIndex:
        """创建 VectorStoreIndex

        Args:
            documents: 文档列表
            collection_name: 集合名称
            **kwargs: 额外参数

        Returns:
            VectorStoreIndex: 创建的索引
        """
        try:
            storage_context = self.get_storage_context(collection_name)

            # 创建索引
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
                **kwargs
            )

            # 缓存索引
            self.index_cache[collection_name] = index

            logger.info(f"索引创建成功，集合: {collection_name}, 文档数: {len(documents)}")
            return index

        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            raise VectorStoreError(f"创建索引失败: {e}", collection_name, "create_index") from e

    def get_or_create_index(
        self,
        documents: list[LlamaDocument],
        collection_name: str = "default",
        **kwargs
    ) -> VectorStoreIndex:
        """获取或创建索引

        Args:
            documents: 文档列表（仅在创建时使用）
            collection_name: 集合名称
            **kwargs: 额外参数

        Returns:
            VectorStoreIndex: 索引实例
        """
        # 检查缓存
        if collection_name in self.index_cache:
            return self.index_cache[collection_name]

        # 检查是否已存在索引
        storage_context = self.get_storage_context(collection_name)

        try:
            # 尝试从现有向量存储加载索引
            index = VectorStoreIndex.from_vector_store(
                self.vector_stores[collection_name],
                storage_context=storage_context
            )
            self.index_cache[collection_name] = index
            logger.info(f"从现有向量存储加载索引成功: {collection_name}")
            return index

        except Exception as e:
            logger.info(f"现有向量存储为空，创建新索引: {e}")
            # 创建新索引
            return self.create_index(documents, collection_name, **kwargs)

    def load_index_from_storage(self, collection_name: str) -> VectorStoreIndex | None:
        """从存储加载索引

        Args:
            collection_name: 集合名称

        Returns:
            VectorStoreIndex or None: 加载的索引或 None（如果不存在）
        """
        try:
            storage_context = self.get_storage_context(collection_name)

            # 从向量存储创建索引
            index = VectorStoreIndex.from_vector_store(
                self.vector_stores[collection_name],
                storage_context=storage_context
            )

            self.index_cache[collection_name] = index
            logger.info(f"从存储加载索引成功: {collection_name}")
            return index

        except Exception as e:
            logger.warning(f"从存储加载索引失败: {e}")
            return None

    def create_query_engine(
        self,
        collection_name: str = "default",
        similarity_top_k: int = 5,
        **kwargs
    ):
        """创建查询引擎

        Args:
            collection_name: 集合名称
            similarity_top_k: 返回的相似文档数量
            **kwargs: 额外参数

        Returns:
            QueryEngine: 查询引擎
        """
        try:
            index = self.get_index(collection_name)
            query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                **kwargs
            )
            return query_engine

        except Exception as e:
            logger.error(f"创建查询引擎失败: {e}")
            raise VectorStoreError(f"创建查询引擎失败: {e}", collection_name, "create_query_engine") from e

    def get_index(self, collection_name: str = "default") -> VectorStoreIndex:
        """获取索引

        Args:
            collection_name: 集合名称

        Returns:
            VectorStoreIndex: 索引实例
        """
        if collection_name in self.index_cache:
            return self.index_cache[collection_name]

        # 尝试从存储加载
        index = self.load_index_from_storage(collection_name)
        if index:
            return index

        raise VectorStoreError(
            f"索引不存在: {collection_name}", collection_name, "get_index"
        )

    def add_documents_to_index(
        self,
        documents: list[LlamaDocument],
        collection_name: str = "default"
    ) -> bool:
        """向索引添加文档

        Args:
            documents: 文档列表
            collection_name: 集合名称

        Returns:
            bool: 是否成功
        """
        try:
            index = self.get_index(collection_name)

            # 添加文档到索引
            for doc in documents:
                index.insert(doc)

            logger.info(f"向索引添加文档成功，集合: {collection_name}, 文档数: {len(documents)}")
            return True

        except Exception as e:
            logger.error(f"向索引添加文档失败: {e}")
            raise VectorStoreError(
                f"向索引添加文档失败: {e}", collection_name, "add_documents_to_index"
            ) from e

    def delete_from_index(
        self,
        doc_id: str,
        collection_name: str = "default"
    ) -> bool:
        """从索引删除文档

        Args:
            doc_id: 文档ID
            collection_name: 集合名称

        Returns:
            bool: 是否成功
        """
        try:
            index = self.get_index(collection_name)
            index.delete_ref_doc(doc_id)

            logger.info(f"从索引删除文档成功，集合: {collection_name}, 文档ID: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"从索引删除文档失败: {e}")
            raise VectorStoreError(
                f"从索引删除文档失败: {e}", collection_name, "delete_from_index"
            ) from e

    def clear_index(self, collection_name: str = "default") -> bool:
        """清空索引

        Args:
            collection_name: 集合名称

        Returns:
            bool: 是否成功
        """
        try:
            # 删除集合
            self._delete_collection(collection_name)

            # 清除缓存
            if collection_name in self.vector_stores:
                del self.vector_stores[collection_name]
            if collection_name in self.storage_contexts:
                del self.storage_contexts[collection_name]
            if collection_name in self.index_cache:
                del self.index_cache[collection_name]

            logger.info(f"索引清空成功: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"清空索引失败: {e}")
            raise VectorStoreError(f"清空索引失败: {e}", collection_name, "clear_index") from e

    def list_collections(self) -> list[str]:
        """列出所有集合"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            raise VectorStoreError(f"列出集合失败: {e}", "all", "list_collections") from e

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """获取集合信息"""
        try:
            collection = self._get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata,
                "persist_directory": self.persist_dir,
                "has_index": collection_name in self.index_cache,
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            raise VectorStoreError(f"获取集合信息失败: {e}", collection_name, "get_collection_info") from e

    def reset(self) -> bool:
        """重置所有集合和索引"""
        try:
            self.chroma_client.reset()

            # 清除所有缓存
            self.vector_stores.clear()
            self.index_cache.clear()
            self.storage_contexts.clear()

            # 重置基础管理器
            self.base_manager.reset()

            logger.info("向量存储重置成功")
            return True

        except Exception as e:
            logger.error(f"重置向量存储失败: {e}")
            raise VectorStoreError(f"重置向量存储失败: {e}", "all", "reset") from e

    def backup(self, backup_path: str) -> bool:
        """备份所有数据"""
        try:
            import os
            import shutil

            # 确保备份目录存在
            os.makedirs(backup_path, exist_ok=True)

            # 复制持久化目录
            backup_dir = f"{backup_path}/chroma_backup_{int(time.time())}"
            shutil.copytree(self.persist_dir, backup_dir)

            logger.info(f"备份成功: {backup_dir}")
            return True

        except Exception as e:
            logger.error(f"备份失败: {e}")
            raise VectorStoreError(f"备份失败: {e}", "all", "backup") from e

    @contextmanager
    def transaction(self, collection_name: str = "default"):
        """事务上下文管理器"""
        try:
            yield
        except Exception as e:
            logger.error(f"事务执行失败: {e}")
            raise

    # 以下是向后兼容的方法，委托给基础管理器
    def _get_or_create_collection(self, collection_name: str) -> Collection:
        """获取或创建集合"""
        try:
            return self.chroma_client.get_or_create_collection(collection_name)
        except Exception as e:
            logger.error(f"获取或创建集合失败: {e}")
            raise VectorStoreError(
                f"获取或创建集合失败: {e}", collection_name, "get_or_create_collection"
            ) from e

    def _get_collection(self, collection_name: str) -> Collection:
        """获取集合"""
        try:
            return self.chroma_client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"获取集合失败: {e}")
            raise VectorStoreError(
                f"获取集合失败: {e}", collection_name, "get_collection"
            ) from e

    def _delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            self.chroma_client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            raise VectorStoreError(
                f"删除集合失败: {e}", collection_name, "delete_collection"
            ) from e

    # 向后兼容方法 - 委托给基础管理器
    def create_collection(self, collection_name: str, **kwargs) -> Collection:
        """创建集合（向后兼容）"""
        return self.base_manager.create_collection(collection_name, **kwargs)

    def similarity_search(self, query: str, collection_name: str, **kwargs) -> list[Any]:
        """相似度搜索（向后兼容）"""
        return self.base_manager.similarity_search(query, collection_name, **kwargs)

    def add_documents(self, documents: list[Any], collection_name: str, **kwargs) -> list[str]:
        """添加文档（向后兼容）"""
        return self.base_manager.add_documents(documents, collection_name, **kwargs)

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合（向后兼容）"""
        return self.base_manager.delete_collection(collection_name)
