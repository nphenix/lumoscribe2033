"""
LlamaIndex 服务

基于 LlamaIndex 最佳实践构建的 RAG 服务，充分利用 LlamaIndex 的既有能力。
支持多种索引类型：VectorStoreIndex、TreeIndex、KnowledgeGraphIndex
支持智能检索器、重排序、相关性过滤等高级功能
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

# from llama_index.core.indices.query.base import BaseQueryEngine
from llama_index.core.indices.tree.base import TreeIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import (
    KeywordNodePostprocessor,
    LongContextReorder,
    SentenceEmbeddingOptimizer,
    SimilarityPostprocessor,
)
from llama_index.core.query_engine import RetrieverQueryEngine

# from llama_index.core.postprocessor.rankguru_rerank import RankGuruRerank
# from llama_index.core.postprocessor.cross_encoder_rerank import SentenceTransformerRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import BaseRetriever

from src.framework.shared.exceptions import IndexServiceError
from src.framework.shared.logging import get_logger
from src.framework.shared.models import RetrievalMetrics


@dataclass
class IndexInfo:
    """索引信息"""
    name: str
    index_type: str
    document_count: int
    node_count: int
    storage_dir: str
    created: bool = True
    error: str = ""


class LlamaIndexService:
    """LlamaIndex 服务

    基于 LlamaIndex 最佳实践构建的 RAG 服务，提供高效的文档索引和检索功能。
    支持多种索引类型和智能检索策略。
    """

    def __init__(
        self,
        persist_dir: str = "data/llamaindex_storage",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        embed_model: Any = None,
        llm: Any = None,
        enable_metrics: bool = True,
        enable_rerank: bool = True,
    ):
        """初始化 LlamaIndex 服务

        Args:
            persist_dir: 持久化目录
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            embed_model: 嵌入模型
            llm: 大语言模型
            enable_metrics: 是否启用性能指标
            enable_rerank: 是否启用重排序
        """
        self.logger = get_logger(__name__)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 文档处理配置
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # LLM 和嵌入模型配置
        if embed_model:
            Settings.embed_model = embed_model
        if llm:
            Settings.llm = llm

        # 功能开关
        self.enable_metrics = enable_metrics
        self.enable_rerank = enable_rerank

        # 存储的索引（支持多种索引类型）
        self.indices: dict[str, Any] = {}
        self.index_types: dict[str, str] = {}  # 记录索引类型
        self.query_engines: dict[str, Any] = {}
        self.retrievers: dict[str, BaseRetriever] = {}

        # 性能指标
        self.metrics_history: list[RetrievalMetrics] = []

        # 重排序器缓存
        self.rerankers: dict[str, Any] = {}

    async def create_index_from_documents(
        self,
        documents: list[Document],
        index_name: str = "default",
        index_type: str = "vector",  # 新增索引类型参数
        use_auto_retriever: bool = True,
        similarity_top_k: int = 10,
        **kwargs,
    ) -> Any:
        """从文档创建索引（基于 LlamaIndex 最佳实践）"""
        try:
            self.logger.info(f"正在为 {len(documents)} 个文档创建 {index_type} 索引: {index_name}")

            # 创建文本分割器
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            # 创建存储上下文
            storage_context = StorageContext.from_defaults()

            # 根据索引类型创建不同的索引
            if index_type == "tree":
                index = await self._create_tree_index(documents, storage_context, splitter, index_name, **kwargs)
            elif index_type == "knowledge_graph":
                index = await self._create_knowledge_graph_index(documents, storage_context, splitter, index_name, **kwargs)
            else:  # vector 索引
                index = await self._create_vector_index(documents, storage_context, splitter, index_name, **kwargs)

            # 存储索引信息
            self.indices[index_name] = index
            self.index_types[index_name] = index_type

            # 创建查询引擎
            query_engine = await self._create_query_engine(
                index,
                index_name,
                index_type,
                use_auto_retriever,
                similarity_top_k,
                **kwargs
            )
            self.query_engines[index_name] = query_engine

            self.logger.info(f"{index_type} 索引创建成功: {index_name}")
            return index

        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            raise IndexServiceError(f"创建索引失败: {e}", index_name, "create_index") from e

    async def _create_vector_index(
        self,
        documents: list[Document],
        storage_context: StorageContext,
        splitter: SentenceSplitter,
        index_name: str,
        **kwargs,
    ) -> VectorStoreIndex:
        """创建向量索引"""
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
            **kwargs,
        )

        # 持久化索引
        index.set_index_id(f"vector_index_{index_name}")
        index.storage_context.persist(str(self.persist_dir / index_name))

        return index

    async def _create_tree_index(
        self,
        documents: list[Document],
        storage_context: StorageContext,
        splitter: SentenceSplitter,
        index_name: str,
        **kwargs,
    ) -> Any:
        """创建树形索引"""
        try:
            from llama_index.core.indices.tree.base import TreeIndex

            index = TreeIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter],
                show_progress=True,
                **kwargs,
            )

            # 持久化索引
            index.set_index_id(f"tree_index_{index_name}")
            index.storage_context.persist(str(self.persist_dir / f"{index_name}_tree"))

            return index

        except ImportError:
            self.logger.warning("TreeIndex 不可用，回退到向量索引")
            return await self._create_vector_index(documents, storage_context, splitter, index_name, **kwargs)

    async def _create_knowledge_graph_index(
        self,
        documents: list[Document],
        storage_context: StorageContext,
        splitter: SentenceSplitter,
        index_name: str,
        **kwargs,
    ) -> Any:
        """创建知识图谱索引"""
        try:
            # 知识图谱索引需要额外的依赖
            from llama_index.core.indices.knowledge_graph.base import (
                KnowledgeGraphIndex,
            )

            # 这里需要图数据库支持，暂时使用基础实现
            index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter],
                show_progress=True,
                **kwargs,
            )

            # 持久化索引
            index.set_index_id(f"kg_index_{index_name}")
            index.storage_context.persist(str(self.persist_dir / f"{index_name}_kg"))

            return index

        except ImportError:
            self.logger.warning("KnowledgeGraphIndex 不可用，回退到向量索引")
            return await self._create_vector_index(documents, storage_context, splitter, index_name, **kwargs)

    async def _create_query_engine(
        self,
        index: Any,
        index_name: str,
        index_type: str,
        use_auto_retriever: bool,
        similarity_top_k: int,
        **kwargs,
    ) -> Any:
        """创建查询引擎"""
        # 获取后处理器
        postprocessors = await self._get_postprocessors(index_type, similarity_top_k)

        if use_auto_retriever and hasattr(index, 'as_retriever'):
            retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            self.retrievers[index_name] = retriever
            query_engine = index.as_query_engine(
                retriever=retriever,
                node_postprocessors=postprocessors,
                response_mode=kwargs.get("response_mode", "compact"),
            )
        else:
            query_engine = index.as_query_engine(
                similarity_top_k=similarity_top_k,
                node_postprocessors=postprocessors,
                response_mode=kwargs.get("response_mode", "compact"),
            )

        return query_engine

    async def _get_postprocessors(self, index_type: str, similarity_top_k: int) -> list[Any]:
        """获取后处理器列表"""
        postprocessors = []

        # 基础相似度过滤
        postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.7))

        # 长文本重排序
        postprocessors.append(LongContextReorder())

        # 句子嵌入优化
        postprocessors.append(SentenceEmbeddingOptimizer())

        # 重排序器（如果启用）
        if self.enable_rerank:
            reranker = await self._get_reranker(index_type)
            if reranker:
                postprocessors.append(reranker)

        return postprocessors

    async def _get_reranker(self, index_type: str) -> Any | None:
        """获取重排序器"""
        reranker_key = f"{index_type}_reranker"

        if reranker_key in self.rerankers:
            return self.rerankers[reranker_key]

        try:
            # 暂时禁用重排序器
            # reranker = SentenceTransformerRerank(
            #     model="BAAI/bge-reranker-base",
            #     top_n=10,
            # )
            # self.rerankers[reranker_key] = reranker
            # return reranker
            return None

        except ImportError:
            self.logger.warning("重排序器不可用")
            return None

    async def create_index_from_directory(
        self,
        directory_path: str,
        index_name: str = "default",
        file_extensions: list[str] | None = None,
        **kwargs,
    ) -> VectorStoreIndex:
        """从目录创建索引"""
        try:
            self.logger.info(f"正在从目录创建索引: {directory_path}")

            # 加载文档
            if file_extensions:
                documents = SimpleDirectoryReader(
                    directory_path,
                    required_exts=file_extensions,
                    recursive=True,
                ).load_data()
            else:
                documents = SimpleDirectoryReader(
                    directory_path,
                    recursive=True,
                ).load_data()

            # 创建索引
            return await self.create_index_from_documents(
                documents, index_name, **kwargs
            )

        except Exception as e:
            self.logger.error(f"从目录创建索引失败: {e}")
            raise IndexServiceError(f"从目录创建索引失败: {e}", index_name, "create_index_from_directory") from e

    async def load_index(
        self,
        index_name: str,
        index_id: str | None = None,
    ) -> VectorStoreIndex:
        """加载已保存的索引"""
        try:
            if index_name in self.indices:
                return self.indices[index_name]

            index_dir = self.persist_dir / index_name
            if not index_dir.exists():
                raise IndexServiceError(
                    f"索引目录不存在: {index_dir}",
                    index_name,
                    "load_index",
                )

            # 重建存储上下文
            storage_context = StorageContext.from_defaults(
                persist_dir=str(index_dir)
            )

            # 加载索引 ID
            if index_id is None:
                index_id = f"vector_index_{index_name}"

            # 加载索引
            index = load_index_from_storage(
                storage_context,
                index_id=index_id,
            )

            self.indices[index_name] = index

            # 创建查询引擎
            query_engine = index.as_query_engine()
            self.query_engines[index_name] = query_engine

            self.logger.info(f"索引加载成功: {index_name}")
            return index

        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            raise IndexServiceError(f"加载索引失败: {e}", index_name, "load_index") from e

    async def query(
        self,
        query_str: str,
        index_name: str = "default",
        similarity_top_k: int = 5,
        **kwargs,
    ) -> Any:
        """查询索引（增强版，包含性能监控）"""
        start_time = time.time()

        try:
            # 确保索引已加载
            if index_name not in self.indices:
                await self.load_index(index_name)

            # 获取索引类型
            index_type = self.index_types.get(index_name, "vector")

            # 获取查询引擎
            query_engine = self.query_engines[index_name]

            # 执行查询
            response = await query_engine.aquery(query_str, **kwargs)

            # 记录性能指标
            if self.enable_metrics:
                query_time = time.time() - start_time
                metrics = RetrievalMetrics(
                    retrieval_time=query_time,
                    retrieved_count=1,  # 查询通常返回一个响应
                    query_complexity=self._assess_query_complexity(query_str),
                    retrieval_strategy=f"{index_type}_query"
                )
                self.metrics_history.append(metrics)
                self.logger.info(f"查询完成，耗时: {query_time:.3f}s, 索引类型: {index_type}")

            self.logger.debug(f"查询完成: {query_str[:50]}...")
            return response

        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            raise IndexServiceError(f"查询失败: {e}", index_name, "query") from e

    async def retrieve(
        self,
        query_str: str,
        index_name: str = "default",
        similarity_top_k: int = 10,
        retrieval_strategy: str = "auto",  # 新增检索策略参数
        **kwargs,
    ) -> list[Document]:
        """检索相关文档（增强版，支持多种检索策略）"""
        start_time = time.time()

        try:
            # 确保索引已加载
            if index_name not in self.indices:
                await self.load_index(index_name)

            # 获取索引类型
            index_type = self.index_types.get(index_name, "vector")
            index = self.indices[index_name]

            # 根据策略选择检索器
            retriever = await self._get_retriever_by_strategy(
                index, index_name, index_type, retrieval_strategy, similarity_top_k
            )

            # 执行检索
            nodes = await retriever.aretrieve(query_str, **kwargs)

            # 转换为文档对象
            documents = []
            for node in nodes:
                doc = Document(
                    text=node.text,
                    metadata=node.metadata,
                    id_=node.node_id,
                )
                documents.append(doc)

            # 记录性能指标
            if self.enable_metrics:
                retrieval_time = time.time() - start_time
                metrics = RetrievalMetrics(
                    retrieval_time=retrieval_time,
                    retrieved_count=len(documents),
                    query_complexity=self._assess_query_complexity(query_str),
                    retrieval_strategy=f"{index_type}_{retrieval_strategy}",
                    final_results_count=len(documents)
                )
                self.metrics_history.append(metrics)
                self.logger.info(f"检索完成，策略: {retrieval_strategy}, 结果数: {len(documents)}, 耗时: {retrieval_time:.3f}s")

            self.logger.debug(f"检索完成，找到 {len(documents)} 个相关文档")
            return documents

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            raise IndexServiceError(f"检索失败: {e}", index_name, "retrieve") from e

    async def _get_retriever_by_strategy(
        self,
        index: Any,
        index_name: str,
        index_type: str,
        strategy: str,
        similarity_top_k: int,
    ) -> BaseRetriever:
        """根据策略获取检索器"""
        if strategy == "auto" and index_name in self.retrievers:
            return self.retrievers[index_name]
        elif strategy == "vector":
            return index.as_retriever(similarity_top_k=similarity_top_k)
        elif strategy == "keyword":
            return await self._get_keyword_retriever(index, similarity_top_k)
        else:  # 默认向量检索
            return index.as_retriever(similarity_top_k=similarity_top_k)

    async def _get_keyword_retriever(self, index: Any, similarity_top_k: int) -> BaseRetriever:
        """获取关键词检索器"""
        try:
            from llama_index.core.retrievers import KeywordTableSimpleRetriever

            # 尝试创建关键词检索器
            if hasattr(index, 'keyword_table'):
                return KeywordTableSimpleRetriever(
                    index.keyword_table,
                    index=index,
                    similarity_top_k=similarity_top_k
                )
            else:
                # 回退到基础检索器
                return index.as_retriever(similarity_top_k=similarity_top_k)

        except ImportError:
            return index.as_retriever(similarity_top_k=similarity_top_k)

    async def keyword_search(
        self,
        query_str: str,
        index_name: str = "default",
        similarity_top_k: int = 10,
        **kwargs,
    ) -> list[Any]:
        """基于关键词的检索（供增强服务复用）"""
        try:
            if index_name not in self.indices:
                await self.load_index(index_name)

            index = self.indices[index_name]
            retriever = await self._get_keyword_retriever(index, similarity_top_k)
            return await retriever.aretrieve(query_str, **kwargs)
        except Exception as e:
            self.logger.error(f"关键词检索失败: {e}")
            raise IndexServiceError(
                f"关键词检索失败: {e}", index_name, "keyword_search"
            ) from e

    def _assess_query_complexity(self, query_str: str) -> str:
        """评估查询复杂度"""
        word_count = len(query_str.split())
        if word_count <= 3:
            return "simple"
        elif word_count <= 10:
            return "medium"
        else:
            return "complex"

    async def add_documents(
        self,
        documents: list[Document],
        index_name: str = "default",
        **kwargs,
    ) -> bool:
        """向现有索引添加文档"""
        try:
            # 确保索引已加载
            if index_name not in self.indices:
                await self.load_index(index_name)

            index = self.indices[index_name]

            # 添加文档
            await index.aadd_documents(documents, **kwargs)

            # 重新持久化
            index.storage_context.persist(str(self.persist_dir / index_name))

            self.logger.info(f"成功添加 {len(documents)} 个文档到索引: {index_name}")
            return True

        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise IndexServiceError(f"添加文档失败: {e}", index_name, "add_documents") from e

    async def delete_document(
        self,
        doc_id: str,
        index_name: str = "default",
        **kwargs,
    ) -> bool:
        """从索引中删除文档"""
        try:
            # 确保索引已加载
            if index_name not in self.indices:
                await self.load_index(index_name)

            index = self.indices[index_name]

            # 删除文档
            await index.adelete_ref_doc(doc_id, **kwargs)

            # 重新持久化
            index.storage_context.persist(str(self.persist_dir / index_name))

            self.logger.info(f"成功删除文档: {doc_id}")
            return True

        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            raise IndexServiceError(f"删除文档失败: {e}", index_name, "delete_document") from e

    def list_indices(self) -> list[str]:
        """列出所有索引"""
        try:
            indices = []
            if self.persist_dir.exists():
                for item in self.persist_dir.iterdir():
                    if item.is_dir():
                        indices.append(item.name)
            return indices
        except Exception as e:
            self.logger.error(f"列出索引失败: {e}")
            return []

    async def get_index_info(self, index_name: str) -> dict[str, Any]:
        """获取索引信息（增强版）"""
        try:
            # 尝试加载索引以获取信息
            if index_name not in self.indices:
                await self.load_index(index_name)

            index = self.indices[index_name]
            index_type = self.index_types.get(index_name, "vector")

            # 获取索引统计信息
            info = {
                "name": index_name,
                "index_type": index_type,
                "index_id": getattr(index, "index_id", "unknown"),
                "document_count": len(index.docstore.docs),
                "node_count": len(index.index_struct.nodes),
                "storage_dir": str(self.persist_dir / index_name),
                "created": True,
                "has_query_engine": index_name in self.query_engines,
                "has_retriever": index_name in self.retrievers,
            }

            return info

        except Exception as e:
            self.logger.error(f"获取索引信息失败: {e}")
            return {
                "name": index_name,
                "created": False,
                "error": str(e),
            }

    def get_retrieval_metrics(self) -> dict[str, Any]:
        """获取检索性能指标"""
        try:
            if not self.metrics_history:
                return {"error": "没有检索历史数据"}

            total_queries = len(self.metrics_history)
            avg_retrieval_time = sum(m.retrieval_time for m in self.metrics_history) / total_queries
            strategy_distribution = {}
            complexity_distribution = {}

            for metrics in self.metrics_history:
                # 策略分布
                strategy = metrics.retrieval_strategy
                strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1

                # 复杂度分布
                complexity = metrics.query_complexity
                complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1

            return {
                "total_queries": total_queries,
                "avg_retrieval_time": avg_retrieval_time,
                "strategy_distribution": strategy_distribution,
                "complexity_distribution": complexity_distribution,
                "recent_metrics": [vars(m) for m in self.metrics_history[-10:]]
            }

        except Exception as e:
            self.logger.error(f"获取检索指标失败: {e}")
            return {"error": str(e)}

    async def delete_index(self, index_name: str) -> bool:
        """删除索引"""
        try:
            # 从内存中移除
            if index_name in self.indices:
                del self.indices[index_name]
            if index_name in self.query_engines:
                del self.query_engines[index_name]
            if index_name in self.retrievers:
                del self.retrievers[index_name]

            # 删除持久化文件
            index_dir = self.persist_dir / index_name
            if index_dir.exists():
                import shutil
                shutil.rmtree(index_dir)

            self.logger.info(f"索引删除成功: {index_name}")
            return True

        except Exception as e:
            self.logger.error(f"删除索引失败: {e}")
            raise IndexServiceError(f"删除索引失败: {e}", index_name, "delete_index") from e

    async def create_hybrid_index(
        self,
        documents: list[Document],
        index_name: str = "hybrid",
        **kwargs,
    ) -> VectorStoreIndex:
        """创建混合索引（向量 + 关键词）"""
        try:
            self.logger.info(f"正在创建混合索引: {index_name}")

            # 创建文本分割器
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            # 创建存储上下文
            storage_context = StorageContext.from_defaults()

            # 创建向量存储索引
            vector_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter],
                show_progress=True,
                **kwargs,
            )

            # 创建关键词表索引
            from llama_index.core import SimpleKeywordTableIndex

            keyword_index = SimpleKeywordTableIndex(
                documents,
                storage_context=storage_context,
                transformations=[splitter],
                show_progress=True,
            )

            # 组合索引

            # 持久化
            storage_context.persist(str(self.persist_dir / f"{index_name}_vector"))
            storage_context.persist(str(self.persist_dir / f"{index_name}_keyword"))

            # 存储索引
            self.indices[f"{index_name}_vector"] = vector_index
            self.indices[f"{index_name}_keyword"] = keyword_index

            # 创建混合查询引擎
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import (
                KeywordTableSimpleRetriever,
                VectorIndexRetriever,
            )

            vector_retriever = VectorIndexRetriever(vector_index, similarity_top_k=5)
            keyword_retriever = KeywordTableSimpleRetriever(keyword_index)

            # 组合检索器
            from llama_index.core.retrievers import RouterRetriever

            router_retriever = RouterRetriever(
                selector=None,  # LLMSingleSelector.from_defaults(),
                retrievers=[
                    ("vector", vector_retriever),
                    ("keyword", keyword_retriever),
                ],
            )

            hybrid_query_engine = RetrieverQueryEngine(router_retriever)
            self.query_engines[index_name] = hybrid_query_engine

            self.logger.info(f"混合索引创建成功: {index_name}")
            return vector_index  # 返回主要索引

        except Exception as e:
            self.logger.error(f"创建混合索引失败: {e}")
            raise IndexServiceError(f"创建混合索引失败: {e}", index_name, "create_hybrid_index") from e

    async def get_relevant_nodes(
        self,
        query_str: str,
        index_name: str = "default",
        top_k: int = 10,
        retrieval_strategy: str = "auto",
    ) -> list[Any]:
        """获取相关节点（用于调试和分析，增强版）"""
        try:
            start_time = time.time()

            # 确保索引已加载
            if index_name not in self.indices:
                await self.load_index(index_name)

            index = self.indices[index_name]

            # 根据策略选择检索器
            retriever = await self._get_retriever_by_strategy(
                index, index_name, self.index_types.get(index_name, "vector"),
                retrieval_strategy, top_k
            )

            nodes = await retriever.aretrieve(query_str)

            # 记录性能指标
            if self.enable_metrics:
                retrieval_time = time.time() - start_time
                metrics = RetrievalMetrics(
                    retrieval_time=retrieval_time,
                    retrieved_count=len(nodes),
                    query_complexity=self._assess_query_complexity(query_str),
                    retrieval_strategy=f"nodes_{retrieval_strategy}",
                    final_results_count=len(nodes)
                )
                self.metrics_history.append(metrics)

            return nodes

        except Exception as e:
            self.logger.error(f"获取相关节点失败: {e}")
            return []

    async def close(self):
        """关闭服务"""
        try:
            # 清理资源
            self.indices.clear()
            self.query_engines.clear()
            self.retrievers.clear()
            self.logger.info("LlamaIndex 服务已关闭")
        except Exception as e:
            self.logger.error(f"关闭服务时出错: {e}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 全局 LlamaIndex 服务实例
_llamaindex_service = None


def get_llamaindex_service(**kwargs) -> LlamaIndexService:
    """获取全局 LlamaIndex 服务实例"""
    global _llamaindex_service

    if _llamaindex_service is None:
        _llamaindex_service = LlamaIndexService(**kwargs)

    return _llamaindex_service


async def init_llamaindex_service(**kwargs) -> LlamaIndexService:
    """初始化 LlamaIndex 服务"""
    service = get_llamaindex_service(**kwargs)
    return service
