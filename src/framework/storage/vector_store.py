"""
基础向量存储管理器

提供对 ChromaDB 的轻量封装，供增强版向量存储与其它组件复用。
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Sequence
from typing import Any, Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from src.framework.shared.exceptions import VectorStoreError
from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Chroma 向量存储基础管理器"""

    def __init__(
        self,
        persist_directory: str = "./vector/chroma",
        embedding_function: Any | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client: ClientAPI = chromadb.PersistentClient(path=self.persist_directory)
        self.collections: dict[str, Collection] = {}

        logger.info(
            "VectorStoreManager initialized",
            extra={"persist_directory": self.persist_directory},
        )

    # -------------------------------------------------------------------------
    # 基础集合操作
    # -------------------------------------------------------------------------
    def create_collection(
        self,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> Collection:
        """创建或获取集合"""
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=metadata,
                embedding_function=embedding_function or self.embedding_function,
            )
            self.collections[collection_name] = collection
            return collection
        except Exception as exc:  # pragma: no cover - chroma raises runtime errors
            raise VectorStoreError(
                f"创建集合失败: {exc}", collection=collection_name, operation="create"
            ) from exc

    def get_collection(
        self,
        collection_name: str,
        create_if_missing: bool = True,
    ) -> Collection:
        """获取集合"""
        if collection_name in self.collections:
            return self.collections[collection_name]

        try:
            collection = self.client.get_collection(collection_name)
            self.collections[collection_name] = collection
            return collection
        except NotFoundError:
            if not create_if_missing:
                raise VectorStoreError(
                    f"集合不存在: {collection_name}",
                    collection=collection_name,
                    operation="get",
                )
            return self.create_collection(collection_name)
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"获取集合失败: {exc}", collection=collection_name, operation="get"
            ) from exc

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(collection_name)
            self.collections.pop(collection_name, None)
            return True
        except NotFoundError:
            logger.warning("尝试删除不存在的集合: %s", collection_name)
            return False
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"删除集合失败: {exc}", collection=collection_name, operation="delete"
            ) from exc

    def list_collections(self) -> list[str]:
        """列出所有集合"""
        try:
            return [collection.name for collection in self.client.list_collections()]
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"列出集合失败: {exc}", operation="list_collections"
            ) from exc

    def reset(self) -> bool:
        """重置所有集合"""
        try:
            self.client.reset()
            self.collections.clear()
            return True
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"重置向量存储失败: {exc}", operation="reset") from exc

    # -------------------------------------------------------------------------
    # 文档操作
    # -------------------------------------------------------------------------
    def add_documents(
        self,
        documents: Sequence[Any],
        collection_name: str,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """向集合添加文档"""
        if not documents:
            return []

        collection = self.get_collection(collection_name)
        payload = self._normalize_documents(documents, ids, metadatas)

        try:
            # 批量添加文档，分批处理大批次
            batch_size = 100
            total_ids = payload["ids"]
            total_documents = payload["documents"]
            total_metadatas = payload["metadatas"]

            added_ids = []
            for i in range(0, len(total_ids), batch_size):
                batch_ids = total_ids[i:i + batch_size]
                batch_docs = total_documents[i:i + batch_size]
                batch_meta = total_metadatas[i:i + batch_size]

                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
                added_ids.extend(batch_ids)

            logger.debug(
                "向量文档已添加",
                extra={"collection": collection_name, "count": len(added_ids), "batches": len(total_ids) // batch_size + 1},
            )
            return added_ids
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"添加文档失败: {exc}", collection=collection_name, operation="add"
            ) from exc

    def similarity_search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行相似度搜索"""
        collection = self.get_collection(collection_name, create_if_missing=False)
        try:
            return collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                where_document=where_document,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"相似度搜索失败: {exc}",
                collection=collection_name,
                operation="similarity_search",
            ) from exc

    def batch_similarity_search(
        self,
        queries: list[str],
        collection_name: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """批量相似度搜索"""
        collection = self.get_collection(collection_name, create_if_missing=False)
        try:
            return collection.query(
                query_texts=queries,
                n_results=k,
                where=where,
                where_document=where_document,
            )
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"批量相似度搜索失败: {exc}",
                collection=collection_name,
                operation="batch_similarity_search",
            ) from exc

    def delete_documents(
        self,
        ids: list[str],
        collection_name: str,
        where: dict[str, Any] | None = None,
    ) -> bool:
        """删除文档"""
        collection = self.get_collection(collection_name)
        try:
            collection.delete(
                ids=ids,
                where=where,
            )
            logger.debug(
                "向量文档已删除",
                extra={"collection": collection_name, "count": len(ids)},
            )
            return True
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(
                f"删除文档失败: {exc}",
                collection=collection_name,
                operation="delete",
            ) from exc

    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_documents(
        documents: Sequence[Any],
        ids: Sequence[str] | None,
        metadatas: Sequence[dict[str, Any]] | None,
    ) -> dict[str, list[Any]]:
        """将输入文档转换为 Chroma 需要的结构"""
        normalized_ids: list[str] = []
        normalized_documents: list[str] = []
        normalized_metadatas: list[dict[str, Any]] = []

        for idx, doc in enumerate(documents):
            text: str
            metadata: dict[str, Any] = {}

            if isinstance(doc, str):
                text = doc
            elif hasattr(doc, "text"):
                text = getattr(doc, "text")
                metadata = dict(getattr(doc, "metadata", {}) or {})
            elif isinstance(doc, dict):
                text = str(doc.get("text") or doc.get("content") or "")
                metadata = dict(doc.get("metadata") or {})
            else:
                text = str(doc)

            if not text:
                continue

            doc_id = (
                (ids[idx] if ids and idx < len(ids) else None)
                or getattr(doc, "doc_id", None)
                or getattr(doc, "id", None)
                or str(uuid.uuid4())
            )

            normalized_ids.append(str(doc_id))
            normalized_documents.append(text)
            normalized_metadatas.append(metadata)

        if not normalized_documents:
            raise VectorStoreError("没有可添加的文档内容", operation="normalize")

        # 性能优化：预分配列表大小
        estimated_size = len(normalized_documents)
        normalized_ids.extend([None] * (estimated_size - len(normalized_ids)))
        normalized_metadatas.extend([None] * (estimated_size - len(normalized_metadatas)))

        return {
            "ids": normalized_ids,
            "documents": normalized_documents,
            "metadatas": normalized_metadatas,
        }


__all__ = ["VectorStoreManager"]

