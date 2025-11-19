"""
向量存储统一接口

使用适配器模式统一基础版和增强版向量存储的接口，遵循 SOLID 原则：
- 接口隔离：定义清晰的向量存储操作接口
- 依赖倒置：客户端依赖抽象接口而非具体实现
- 开闭原则：可以轻松扩展新的向量存储实现

使用方式：
    # 基础实现
    store = VectorStore(implementation="basic")
    
    # 增强实现（需要 LlamaIndex）
    store = VectorStore(implementation="enhanced")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from src.framework.shared.exceptions import VectorStoreError
from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class IVectorStore(ABC):
    """向量存储接口
    
    定义向量存储的标准操作，所有实现必须遵循此接口。
    """
    
    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[Any],
        collection_name: str,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """向集合添加文档"""
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行相似度搜索"""
        pass
    
    @abstractmethod
    def delete_documents(
        self,
        ids: list[str],
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """创建集合"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass
    
    @abstractmethod
    def list_collections(self) -> list[str]:
        """列出所有集合"""
        pass


class BasicVectorStoreAdapter(IVectorStore):
    """基础向量存储适配器
    
    适配 VectorStoreManager 到统一接口。
    """
    
    def __init__(self, base_manager):
        """初始化适配器
        
        Args:
            base_manager: VectorStoreManager 实例
        """
        self.manager = base_manager
    
    def add_documents(
        self,
        documents: Sequence[Any],
        collection_name: str,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """向集合添加文档"""
        return self.manager.add_documents(documents, collection_name, ids, metadatas)
    
    def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行相似度搜索"""
        return self.manager.similarity_search(
            query, collection_name, k, where, where_document
        )
    
    def delete_documents(
        self,
        ids: list[str],
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> bool:
        """删除文档"""
        return self.manager.delete_documents(ids, collection_name, where)
    
    def create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """创建集合"""
        try:
            self.manager.create_collection(collection_name, metadata)
            return True
        except Exception:
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        return self.manager.delete_collection(collection_name)
    
    def list_collections(self) -> list[str]:
        """列出所有集合"""
        return self.manager.list_collections()


class EnhancedVectorStoreAdapter(IVectorStore):
    """增强向量存储适配器
    
    适配 EnhancedVectorStoreManager 到统一接口。
    提供对 LlamaIndex 功能的访问。
    """
    
    def __init__(self, enhanced_manager):
        """初始化适配器
        
        Args:
            enhanced_manager: EnhancedVectorStoreManager 实例
        """
        self.manager = enhanced_manager
        # 使用基础管理器处理标准操作
        self.base_adapter = BasicVectorStoreAdapter(enhanced_manager.base_manager)
    
    def add_documents(
        self,
        documents: Sequence[Any],
        collection_name: str,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """向集合添加文档"""
        # 使用基础适配器处理标准操作
        return self.base_adapter.add_documents(documents, collection_name, ids, metadatas)
    
    def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行相似度搜索"""
        return self.base_adapter.similarity_search(
            query, collection_name, k, where, where_document
        )
    
    def delete_documents(
        self,
        ids: list[str],
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> bool:
        """删除文档"""
        return self.base_adapter.delete_documents(ids, collection_name, where)
    
    def create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """创建集合"""
        return self.base_adapter.create_collection(collection_name, metadata)
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        return self.base_adapter.delete_collection(collection_name)
    
    def list_collections(self) -> list[str]:
        """列出所有集合"""
        return self.base_adapter.list_collections()
    
    # 增强功能（不在接口中，但可通过适配器访问）
    def get_storage_context(self, collection_name: str = "default"):
        """获取 LlamaIndex StorageContext（增强功能）"""
        return self.manager.get_storage_context(collection_name)
    
    def create_index(self, documents, collection_name: str = "default", **kwargs):
        """创建 VectorStoreIndex（增强功能）"""
        return self.manager.create_index(documents, collection_name, **kwargs)
    
    def get_or_create_index(self, documents, collection_name: str = "default", **kwargs):
        """获取或创建索引（增强功能）"""
        return self.manager.get_or_create_index(documents, collection_name, **kwargs)


class VectorStore:
    """统一的向量存储接口
    
    使用适配器模式统一基础版和增强版的接口。
    客户端代码只需使用此统一接口，无需关心具体实现。
    """
    
    _implementations: dict[str, type] = {
        "basic": BasicVectorStoreAdapter,
        "enhanced": EnhancedVectorStoreAdapter,
    }
    
    def __init__(
        self,
        implementation: str = "basic",
        persist_directory: str = "./vector/chroma",
        embedding_function: Any | None = None,
    ):
        """初始化向量存储
        
        Args:
            implementation: 实现类型，"basic" 或 "enhanced"
            persist_directory: 持久化目录
            embedding_function: 嵌入函数
        """
        if implementation not in self._implementations:
            raise ValueError(
                f"未知的实现类型: {implementation}，可选: {list(self._implementations.keys())}"
            )
        
        self.implementation_name = implementation
        
        # 创建底层管理器
        if implementation == "basic":
            from src.framework.storage.vector_store import VectorStoreManager
            manager = VectorStoreManager(persist_directory, embedding_function)
            self.adapter = BasicVectorStoreAdapter(manager)
        else:  # enhanced
            from src.framework.storage.enhanced_vector_store import EnhancedVectorStoreManager
            manager = EnhancedVectorStoreManager(persist_directory)
            self.adapter = EnhancedVectorStoreAdapter(manager)
        
        logger.info(f"向量存储初始化完成，实现: {implementation}")
    
    # 标准接口方法（委托给适配器）
    def add_documents(
        self,
        documents: Sequence[Any],
        collection_name: str,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """向集合添加文档"""
        return self.adapter.add_documents(documents, collection_name, ids, metadatas)
    
    def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行相似度搜索"""
        return self.adapter.similarity_search(
            query, collection_name, k, where, where_document
        )
    
    def delete_documents(
        self,
        ids: list[str],
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> bool:
        """删除文档"""
        return self.adapter.delete_documents(ids, collection_name, where)
    
    def create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """创建集合"""
        return self.adapter.create_collection(collection_name, metadata)
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        return self.adapter.delete_collection(collection_name)
    
    def list_collections(self) -> list[str]:
        """列出所有集合"""
        return self.adapter.list_collections()
    
    # 增强功能访问（仅增强实现可用）
    def get_storage_context(self, collection_name: str = "default"):
        """获取 LlamaIndex StorageContext（仅增强实现）"""
        if isinstance(self.adapter, EnhancedVectorStoreAdapter):
            return self.adapter.get_storage_context(collection_name)
        raise NotImplementedError("基础实现不支持 StorageContext")
    
    def create_index(self, documents, collection_name: str = "default", **kwargs):
        """创建 VectorStoreIndex（仅增强实现）"""
        if isinstance(self.adapter, EnhancedVectorStoreAdapter):
            return self.adapter.create_index(documents, collection_name, **kwargs)
        raise NotImplementedError("基础实现不支持索引创建")
    
    @classmethod
    def register_implementation(cls, name: str, adapter_class: type[IVectorStore]) -> None:
        """注册新的向量存储实现（扩展点）
        
        Args:
            name: 实现名称
            adapter_class: 适配器类
        """
        if not issubclass(adapter_class, IVectorStore):
            raise TypeError(f"适配器类必须实现 IVectorStore 接口")
        cls._implementations[name] = adapter_class


__all__ = [
    "IVectorStore",
    "VectorStore",
    "BasicVectorStoreAdapter",
    "EnhancedVectorStoreAdapter",
]

