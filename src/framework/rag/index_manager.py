"""
索引管理器

管理向量索引的创建、更新和查询。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class IndexConfig:
    """索引配置"""
    collection_name: str
    embedding_dim: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 100
    similarity_threshold: float = 0.7


class IndexManager(ABC):
    """索引管理器抽象基类"""

    @abstractmethod
    async def create_index(self, config: IndexConfig) -> bool:
        """创建索引"""
        pass

    @abstractmethod
    async def add_documents(
        self, documents: list[Document], collection_name: str
    ) -> bool:
        """添加文档到索引"""
        pass

    @abstractmethod
    async def update_document(
        self, doc_id: str, document: Document, collection_name: str
    ) -> bool:
        """更新文档"""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """删除文档"""
        pass

    @abstractmethod
    async def search(
        self, query: str, collection_name: str, k: int = 10
    ) -> list[Document]:
        """搜索文档"""
        pass

    @abstractmethod
    async def delete_index(self, collection_name: str) -> bool:
        """删除索引"""
        pass


class ChromaIndexManager(IndexManager):
    """ChromaDB 索引管理器"""

    def __init__(self, persist_directory: str = "vector/chroma") -> None:
        self.persist_directory = persist_directory
        self.collections: dict[str, Any] = {}

    async def create_index(self, config: IndexConfig) -> bool:
        """创建 ChromaDB 索引"""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)

            # 创建集合
            collection = client.create_collection(
                name=config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            self.collections[config.collection_name] = collection
            return True

        except Exception as e:
            print(f"创建索引失败: {e}")
            return False

    async def add_documents(
        self, documents: list[Document], collection_name: str
    ) -> bool:
        """添加文档到 ChromaDB"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                await self.create_index(IndexConfig(collection_name=collection_name))
                collection = self.collections[collection_name]

            # 生成文档ID
            doc_ids = [
                f"doc_{i}_{hash(doc.page_content[:100])}"
                for i, doc in enumerate(documents)
            ]

            # 添加文档
            collection.add(
                documents=[doc.page_content for doc in documents],
                metadatas=[doc.metadata for doc in documents],
                ids=doc_ids
            )

            return True

        except Exception as e:
            print(f"添加文档失败: {e}")
            return False

    async def update_document(
        self, doc_id: str, document: Document, collection_name: str
    ) -> bool:
        """更新 ChromaDB 文档"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                return False

            collection.update(
                ids=[doc_id],
                documents=[document.page_content],
                metadatas=[document.metadata]
            )

            return True

        except Exception as e:
            print(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: str, collection_name: str) -> bool:
        """删除 ChromaDB 文档"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                return False

            collection.delete(ids=[doc_id])
            return True

        except Exception as e:
            print(f"删除文档失败: {e}")
            return False

    async def search(
        self, query: str, collection_name: str, k: int = 10
    ) -> list[Document]:
        """搜索 ChromaDB 文档"""
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=k
            )

            documents = []
            for i, content in enumerate(results['documents'][0]):
                doc = Document(
                    page_content=content,
                    metadata=(
                        results['metadatas'][0][i]
                        if results['metadatas'][0]
                        else {}
                    )
                )
                documents.append(doc)

            return documents

        except Exception as e:
            print(f"搜索文档失败: {e}")
            return []

    async def delete_index(self, collection_name: str) -> bool:
        """删除 ChromaDB 索引"""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=collection_name)

            if collection_name in self.collections:
                del self.collections[collection_name]

            return True

        except Exception as e:
            print(f"删除索引失败: {e}")
            return False
