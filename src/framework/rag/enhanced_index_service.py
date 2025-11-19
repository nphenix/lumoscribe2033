"""
增强版索引服务

基于 LlamaIndex 最佳实践，提供：
- AutoRetriever 智能检索
- 多策略检索支持
- 查询重写和优化
- 混合检索（向量 + 图 + 关键词）
- 索引生命周期管理
- 性能监控和优化
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.schema import NodeWithScore

from src.framework.rag.index_service import IndexService as BaseIndexService
from src.framework.rag.llamaindex_service import LlamaIndexService
from src.framework.shared.exceptions import IndexServiceError
from src.framework.shared.logging import get_logger
from src.framework.shared.models import DocumentChunk
from src.framework.storage.enhanced_graph_store import EnhancedGraphStoreManager
from src.framework.storage.enhanced_vector_store import EnhancedVectorStoreManager
from src.framework.shared.models import RetrievalMetrics

logger = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """查询分析结果"""
    intent: str
    entities: list[str]
    keywords: list[str]
    complexity: str
    suggested_strategies: list[str]


class EnhancedIndexService(BaseIndexService):
    """增强版索引服务"""

    def __init__(
        self,
        vector_store_manager: EnhancedVectorStoreManager | None = None,
        graph_store_manager: EnhancedGraphStoreManager | None = None,
        llamaindex_service: LlamaIndexService | None = None,
        enable_auto_retriever: bool = True,
        enable_query_analysis: bool = True,
        enable_metrics: bool = True
    ):
        """初始化增强版索引服务

        Args:
            vector_store_manager: 向量存储管理器
            graph_store_manager: 图存储管理器
            llamaindex_service: LlamaIndex 服务
            enable_auto_retriever: 是否启用 AutoRetriever
            enable_query_analysis: 是否启用查询分析
            enable_metrics: 是否启用性能指标
        """
        super().__init__(llamaindex_service=llamaindex_service or LlamaIndexService())
        self.vector_store_manager = vector_store_manager or EnhancedVectorStoreManager()
        self.graph_store_manager = graph_store_manager or EnhancedGraphStoreManager()

        self.enable_auto_retriever = enable_auto_retriever
        self.enable_query_analysis = enable_query_analysis
        self.enable_metrics = enable_metrics

        # 缓存和指标
        self.query_cache: dict[str, Any] = {}
        self.metrics_history: list[RetrievalMetrics] = []
        self.query_analysis_cache: dict[str, QueryAnalysis] = {}

        # 检索策略配置
        self.retrieval_strategies = {
            "vector": self._vector_retrieval,
            "graph": self._graph_retrieval,
            "hybrid": self._hybrid_retrieval,
            "auto": self._auto_retrieval,
            "keyword": self._keyword_retrieval
        }

        logger.info("增强版索引服务初始化完成")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """分析查询意图和策略

        Args:
            query: 用户查询

        Returns:
            QueryAnalysis: 查询分析结果
        """
        if not self.enable_query_analysis:
            return QueryAnalysis(
                intent="general",
                entities=[],
                keywords=query.split(),
                complexity="medium",
                suggested_strategies=["auto"]
            )

        # 检查缓存
        cache_key = f"query_analysis_{hash(query)}"
        if cache_key in self.query_analysis_cache:
            return self.query_analysis_cache[cache_key]

        try:
            # 简单的查询分析逻辑
            intent = self._detect_query_intent(query)
            entities = self._extract_entities(query)
            keywords = self._extract_keywords(query)
            complexity = self._assess_complexity(query)
            strategies = self._suggest_strategies(query, intent, entities)

            analysis = QueryAnalysis(
                intent=intent,
                entities=entities,
                keywords=keywords,
                complexity=complexity,
                suggested_strategies=strategies
            )

            # 缓存结果
            self.query_analysis_cache[cache_key] = analysis

            logger.info(f"查询分析完成: {query[:50]}...")
            return analysis

        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return QueryAnalysis(
                intent="general",
                entities=[],
                keywords=query.split(),
                complexity="medium",
                suggested_strategies=["auto"]
            )

    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        strategy: str = "auto",
        top_k: int = 10,
        **kwargs
    ) -> list[NodeWithScore]:
        """检索文档

        Args:
            query: 查询文本
            collection_name: 集合名称
            strategy: 检索策略
            top_k: 返回结果数量
            **kwargs: 额外参数

        Returns:
            List[NodeWithScore]: 检索结果
        """
        start_time = time.time()

        try:
            # 查询缓存检查
            cache_key = f"{query}_{collection_name}_{strategy}_{top_k}"
            if cache_key in self.query_cache:
                logger.info(f"缓存命中: {query[:50]}...")
                return self.query_cache[cache_key]

            # 查询分析
            if self.enable_query_analysis:
                analysis = self.analyze_query(query)
                if strategy == "auto":
                    strategy = analysis.suggested_strategies[0] if analysis.suggested_strategies else "auto"

            # 执行检索
            if strategy not in self.retrieval_strategies:
                raise IndexServiceError(f"不支持的检索策略: {strategy}")

            retriever = self.retrieval_strategies[strategy]
            results = retriever(query, collection_name, top_k, **kwargs)

            # 缓存结果
            if len(results) > 0:
                self.query_cache[cache_key] = results

            # 记录指标
            retrieval_time = time.time() - start_time
            if self.enable_metrics:
                metrics = RetrievalMetrics(
                    retrieval_time=retrieval_time,
                    retrieved_count=len(results),
                    query_complexity=analysis.complexity if self.enable_query_analysis else "unknown",
                    retrieval_strategy=strategy
                )
                self.metrics_history.append(metrics)

            logger.info(f"检索完成，策略: {strategy}, 结果数: {len(results)}, 耗时: {retrieval_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise IndexServiceError(f"检索失败: {e}") from e

    def create_query_engine(
        self,
        collection_name: str = "default",
        strategy: str = "auto",
        **kwargs
    ):
        """创建查询引擎

        Args:
            collection_name: 集合名称
            strategy: 检索策略
            **kwargs: 额外参数

        Returns:
            QueryEngine: 查询引擎
        """
        try:
            # 获取检索器
            if strategy == "auto":
                retriever = self._create_auto_retriever(collection_name)
            else:
                retriever = self._create_retriever(strategy, collection_name)

            # 创建响应合成器
            response_synthesizer = get_response_synthesizer(
                response_mode=kwargs.get("response_mode", "compact")
            )

            # 创建查询引擎
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )

            logger.info(f"查询引擎创建成功，策略: {strategy}")
            return query_engine

        except Exception as e:
            logger.error(f"创建查询引擎失败: {e}")
            raise IndexServiceError(f"创建查询引擎失败: {e}") from e

    def get_retrieval_metrics(self) -> dict[str, Any]:
        """获取检索性能指标

        Returns:
            Dict[str, Any]: 性能指标
        """
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
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "recent_metrics": [vars(m) for m in self.metrics_history[-10:]]
        }

    def _create_auto_retriever(self, collection_name: str):
        """创建自动检索器"""
        try:
            index = self.vector_store_manager.get_index(collection_name)

            # 创建基础检索器作为自动检索器
            auto_retriever = index.as_retriever(
                similarity_top_k=5,
                vector_store_kwargs={"include_metadata": True}
            )

            return auto_retriever

        except Exception as e:
            logger.error(f"创建自动检索器失败: {e}")
            # 回退到基础检索器
            return self._create_retriever("vector", collection_name)

    def _create_retriever(self, strategy: str, collection_name: str):
        """创建检索器"""
        if strategy == "vector":
            index = self.vector_store_manager.get_index(collection_name)
            return index.as_retriever(similarity_top_k=5)
        elif strategy == "graph":
            # 图检索器需要特殊实现
            return self._create_graph_retriever(collection_name)
        else:
            index = self.vector_store_manager.get_index(collection_name)
            return index.as_retriever(similarity_top_k=5)

    def _create_graph_retriever(self, collection_name: str):
        """创建图检索器"""
        # 这里需要实现图检索器的逻辑
        # 暂时返回基础检索器
        return self._create_retriever("vector", collection_name)

    def _vector_retrieval(self, query: str, collection_name: str, top_k: int, **kwargs) -> list[NodeWithScore]:
        """向量检索"""
        try:
            index = self.vector_store_manager.get_index(collection_name)
            retriever = index.as_retriever(similarity_top_k=top_k)
            return retriever.retrieve(query)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def _graph_retrieval(self, query: str, collection_name: str, top_k: int, **kwargs) -> list[NodeWithScore]:
        """图检索"""
        try:
            # 图检索的实现
            # 这里可以实现基于图的检索逻辑
            return []
        except Exception as e:
            logger.error(f"图检索失败: {e}")
            return []

    def _hybrid_retrieval(self, query: str, collection_name: str, top_k: int, **kwargs) -> list[NodeWithScore]:
        """混合检索"""
        try:
            # 向量检索
            vector_results = self._vector_retrieval(query, collection_name, top_k * 2, **kwargs)

            # 图检索
            graph_results = self._graph_retrieval(query, collection_name, top_k * 2, **kwargs)

            # 融合结果
            return self._fuse_retrieval_results(vector_results, graph_results, top_k)

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return self._vector_retrieval(query, collection_name, top_k, **kwargs)

    def _auto_retrieval(self, query: str, collection_name: str, top_k: int, **kwargs) -> list[NodeWithScore]:
        """自动检索"""
        try:
            if self.enable_auto_retriever:
                index = self.vector_store_manager.get_index(collection_name)
                auto_retriever = index.as_retriever(
                    similarity_top_k=top_k,
                    **kwargs
                )
                return auto_retriever.retrieve(query)
            else:
                return self._vector_retrieval(query, collection_name, top_k, **kwargs)
        except Exception as e:
            logger.error(f"自动检索失败: {e}")
            return self._vector_retrieval(query, collection_name, top_k, **kwargs)

    def _keyword_retrieval(self, query: str, collection_name: str, top_k: int, **kwargs) -> list[NodeWithScore]:
        """关键词检索"""
        try:
            return self.keyword_search(
                query, index_name=collection_name, similarity_top_k=top_k, **kwargs
            )
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []

    def _fuse_retrieval_results(
        self,
        vector_results: list[NodeWithScore],
        graph_results: list[NodeWithScore],
        top_k: int
    ) -> list[NodeWithScore]:
        """融合检索结果"""
        try:
            # 简单的融合策略：基于分数加权
            all_results = vector_results + graph_results
            result_dict = {}

            for result in all_results:
                doc_id = result.node.node_id
                if doc_id in result_dict:
                    # 如果已存在，取最高分数
                    if result.score > result_dict[doc_id].score:
                        result_dict[doc_id] = result
                else:
                    result_dict[doc_id] = result

            # 按分数排序并返回 top_k
            sorted_results = sorted(result_dict.values(), key=lambda x: x.score, reverse=True)
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"融合检索结果失败: {e}")
            return vector_results[:top_k]

    def _detect_query_intent(self, query: str) -> str:
        """检测查询意图"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["who", "what", "when", "where"]):
            return "informational"
        elif any(word in query_lower for word in ["how", "why", "explain"]):
            return "explanatory"
        elif any(word in query_lower for word in ["list", "show", "find"]):
            return "navigational"
        elif any(word in query_lower for word in ["compare", "contrast", "difference"]):
            return "comparative"
        else:
            return "general"

    def _extract_entities(self, query: str) -> list[str]:
        """提取实体"""
        # 简单的实体提取逻辑
        # 在实际应用中可以使用 NER 模型
        return []

    def _extract_keywords(self, query: str) -> list[str]:
        """提取关键词"""
        # 简单的关键词提取
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = query.lower().split()
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _assess_complexity(self, query: str) -> str:
        """评估查询复杂度"""
        word_count = len(query.split())
        if word_count <= 3:
            return "simple"
        elif word_count <= 10:
            return "medium"
        else:
            return "complex"

    def _suggest_strategies(self, query: str, intent: str, entities: list[str]) -> list[str]:
        """建议检索策略"""
        strategies = ["auto"]

        if len(entities) > 0:
            strategies.append("hybrid")

        if intent in ["informational", "explanatory"]:
            strategies.append("vector")

        if intent == "navigational":
            strategies.append("keyword")

        return strategies[:3]  # 最多返回3种策略

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        if not hasattr(self, '_cache_stats'):
            return 0.0

        total_requests = getattr(self, '_total_requests', 0)
        cache_hits = getattr(self, '_cache_hits', 0)

        if total_requests == 0:
            return 0.0

        return cache_hits / total_requests

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """性能监控上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"{operation_name} 耗时: {duration:.3f}s")

    # 向后兼容方法
    def add_documents(self, documents: list[DocumentChunk], collection_name: str = "default") -> bool:
        """添加文档（向后兼容）"""
        return self.llamaindex_service.add_documents(documents, collection_name)

    def search(self, query: str, collection_name: str = "default", top_k: int = 10) -> list[DocumentChunk]:
        """搜索（向后兼容）"""
        results = self.retrieve(query, collection_name, strategy="auto", top_k=top_k)
        return [DocumentChunk(
            content=result.node.text,
            metadata=result.node.metadata or {},
            score=result.score
        ) for result in results]
