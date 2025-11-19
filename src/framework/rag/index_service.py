"""
基础索引服务

为 RAG 组件提供统一的 LlamaIndex 访问封装，并暴露同步/异步两套接口。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any, Optional

from src.framework.rag.llamaindex_service import LlamaIndexService
from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


def _run_sync(coro):
    """在同步上下文中执行协程；若处于事件循环中则提示使用异步接口。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    raise RuntimeError(
        "检测到正在运行的事件循环，请改用 `await IndexService.a*` 异步接口。"
    )


class IndexService:
    """LlamaIndex 基础封装"""

    def __init__(self, llamaindex_service: LlamaIndexService | None = None) -> None:
        self.llamaindex_service = llamaindex_service or LlamaIndexService()

    # ------------------------------------------------------------------
    # 文档与索引
    # ------------------------------------------------------------------
    async def aadd_documents(
        self,
        documents: Iterable[Any],
        index_name: str = "default",
        **kwargs: Any,
    ) -> bool:
        return await self.llamaindex_service.add_documents(
            list(documents), index_name, **kwargs
        )

    def add_documents(
        self,
        documents: Iterable[Any],
        index_name: str = "default",
        **kwargs: Any,
    ) -> bool:
        return _run_sync(self.aadd_documents(documents, index_name, **kwargs))

    async def adelete_document(
        self,
        doc_id: str,
        index_name: str = "default",
        **kwargs: Any,
    ) -> bool:
        return await self.llamaindex_service.delete_document(
            doc_id, index_name, **kwargs
        )

    def delete_document(
        self,
        doc_id: str,
        index_name: str = "default",
        **kwargs: Any,
    ) -> bool:
        return _run_sync(self.adelete_document(doc_id, index_name, **kwargs))

    # ------------------------------------------------------------------
    # 检索与查询
    # ------------------------------------------------------------------
    async def aretrieve(
        self,
        query: str,
        index_name: str = "default",
        similarity_top_k: int = 10,
        retrieval_strategy: str = "auto",
        **kwargs: Any,
    ) -> list[Any]:
        return await self.llamaindex_service.retrieve(
            query,
            index_name=index_name,
            similarity_top_k=similarity_top_k,
            retrieval_strategy=retrieval_strategy,
            **kwargs,
        )

    def retrieve(
        self,
        query: str,
        index_name: str = "default",
        similarity_top_k: int = 10,
        retrieval_strategy: str = "auto",
        **kwargs: Any,
    ) -> list[Any]:
        return _run_sync(
            self.aretrieve(
                query,
                index_name=index_name,
                similarity_top_k=similarity_top_k,
                retrieval_strategy=retrieval_strategy,
                **kwargs,
            )
        )

    async def aquery(
        self,
        query: str,
        index_name: str = "default",
        **kwargs: Any,
    ) -> Any:
        return await self.llamaindex_service.query(
            query_str=query, index_name=index_name, **kwargs
        )

    def query(
        self,
        query: str,
        index_name: str = "default",
        **kwargs: Any,
    ) -> Any:
        return _run_sync(self.aquery(query, index_name, **kwargs))

    async def akeyword_search(
        self,
        query: str,
        index_name: str = "default",
        similarity_top_k: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        return await self.llamaindex_service.keyword_search(
            query_str=query,
            index_name=index_name,
            similarity_top_k=similarity_top_k,
            **kwargs,
        )

    def keyword_search(
        self,
        query: str,
        index_name: str = "default",
        similarity_top_k: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        return _run_sync(
            self.akeyword_search(
                query,
                index_name=index_name,
                similarity_top_k=similarity_top_k,
                **kwargs,
            )
        )


__all__ = ["IndexService"]

