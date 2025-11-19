"""
检索策略模式实现

使用策略模式重构检索逻辑，遵循开闭原则：
- 可以轻松添加新的检索策略而无需修改现有代码
- 每个策略类职责单一，易于测试和维护

使用方式：
    strategy = VectorRetrievalStrategy(vector_store_manager)
    results = strategy.retrieve(query, collection_name, top_k)
"""

from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.schema import NodeWithScore

from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class RetrievalStrategy(ABC):
    """检索策略抽象基类"""
    
    def __init__(self, vector_store_manager=None, graph_store_manager=None):
        """初始化策略
        
        Args:
            vector_store_manager: 向量存储管理器
            graph_store_manager: 图存储管理器
        """
        self.vector_store_manager = vector_store_manager
        self.graph_store_manager = graph_store_manager
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        **kwargs
    ) -> list[NodeWithScore]:
        """执行检索
        
        Args:
            query: 查询字符串
            collection_name: 集合名称
            top_k: 返回结果数量
            **kwargs: 额外参数
        
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass


class VectorRetrievalStrategy(RetrievalStrategy):
    """向量检索策略"""
    
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        **kwargs
    ) -> list[NodeWithScore]:
        """执行向量检索"""
        try:
            if not self.vector_store_manager:
                logger.warning("向量存储管理器未配置")
                return []
            
            index = self.vector_store_manager.get_index(collection_name)
            retriever = index.as_retriever(similarity_top_k=top_k)
            return retriever.retrieve(query)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "vector"


class GraphRetrievalStrategy(RetrievalStrategy):
    """图检索策略"""
    
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        **kwargs
    ) -> list[NodeWithScore]:
        """执行图检索"""
        try:
            if not self.graph_store_manager:
                logger.warning("图存储管理器未配置")
                return []
            
            # 图检索的实现
            # TODO: 实现基于图的检索逻辑
            return []
        except Exception as e:
            logger.error(f"图检索失败: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "graph"


class KeywordRetrievalStrategy(RetrievalStrategy):
    """关键词检索策略"""
    
    def __init__(self, vector_store_manager=None, graph_store_manager=None, index_service=None):
        """初始化关键词检索策略
        
        Args:
            index_service: 索引服务（用于关键词搜索）
        """
        super().__init__(vector_store_manager, graph_store_manager)
        self.index_service = index_service
    
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        **kwargs
    ) -> list[NodeWithScore]:
        """执行关键词检索"""
        try:
            if not self.index_service:
                logger.warning("索引服务未配置")
                return []
            
            # 使用索引服务的关键词搜索
            results = self.index_service.keyword_search(
                query, index_name=collection_name, similarity_top_k=top_k, **kwargs
            )
            # 转换为 NodeWithScore 格式
            return self._convert_to_nodes(results)
        except Exception as e:
            logger.error(f"关键词检索失败: {e}")
            return []
    
    def _convert_to_nodes(self, results: list[Any]) -> list[NodeWithScore]:
        """转换结果为 NodeWithScore 格式"""
        # TODO: 实现转换逻辑
        return []
    
    def get_strategy_name(self) -> str:
        return "keyword"


class HybridRetrievalStrategy(RetrievalStrategy):
    """混合检索策略
    
    结合向量检索和图检索的结果。
    """
    
    def __init__(self, vector_store_manager=None, graph_store_manager=None, fusion_method="max"):
        """初始化混合检索策略
        
        Args:
            fusion_method: 融合方法，"max"（取最高分）或 "weighted"（加权融合）
        """
        super().__init__(vector_store_manager, graph_store_manager)
        self.fusion_method = fusion_method
        self.vector_strategy = VectorRetrievalStrategy(vector_store_manager, graph_store_manager)
        self.graph_strategy = GraphRetrievalStrategy(vector_store_manager, graph_store_manager)
    
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        **kwargs
    ) -> list[NodeWithScore]:
        """执行混合检索"""
        try:
            # 向量检索
            vector_results = self.vector_strategy.retrieve(
                query, collection_name, top_k * 2, **kwargs
            )
            
            # 图检索
            graph_results = self.graph_strategy.retrieve(
                query, collection_name, top_k * 2, **kwargs
            )
            
            # 融合结果
            return self._fuse_results(vector_results, graph_results, top_k)
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            # 降级到向量检索
            return self.vector_strategy.retrieve(query, collection_name, top_k, **kwargs)
    
    def _fuse_results(
        self,
        vector_results: list[NodeWithScore],
        graph_results: list[NodeWithScore],
        top_k: int
    ) -> list[NodeWithScore]:
        """融合检索结果"""
        try:
            if self.fusion_method == "max":
                return self._fuse_max(vector_results, graph_results, top_k)
            elif self.fusion_method == "weighted":
                return self._fuse_weighted(vector_results, graph_results, top_k)
            else:
                return self._fuse_max(vector_results, graph_results, top_k)
        except Exception as e:
            logger.error(f"融合检索结果失败: {e}")
            return vector_results[:top_k]
    
    def _fuse_max(
        self,
        vector_results: list[NodeWithScore],
        graph_results: list[NodeWithScore],
        top_k: int
    ) -> list[NodeWithScore]:
        """取最高分融合"""
        all_results = vector_results + graph_results
        result_dict = {}
        
        for result in all_results:
            doc_id = result.node.node_id
            if doc_id in result_dict:
                if result.score > result_dict[doc_id].score:
                    result_dict[doc_id] = result
            else:
                result_dict[doc_id] = result
        
        sorted_results = sorted(result_dict.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    def _fuse_weighted(
        self,
        vector_results: list[NodeWithScore],
        graph_results: list[NodeWithScore],
        top_k: int
    ) -> list[NodeWithScore]:
        """加权融合"""
        # TODO: 实现加权融合逻辑
        return self._fuse_max(vector_results, graph_results, top_k)
    
    def get_strategy_name(self) -> str:
        return "hybrid"


class AutoRetrievalStrategy(RetrievalStrategy):
    """自动检索策略
    
    根据查询特征自动选择最佳检索策略。
    """
    
    def __init__(self, vector_store_manager=None, graph_store_manager=None, index_service=None):
        """初始化自动检索策略"""
        super().__init__(vector_store_manager, graph_store_manager)
        self.index_service = index_service
        self.vector_strategy = VectorRetrievalStrategy(vector_store_manager, graph_store_manager)
        self.hybrid_strategy = HybridRetrievalStrategy(vector_store_manager, graph_store_manager)
    
    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int,
        **kwargs
    ) -> list[NodeWithScore]:
        """执行自动检索"""
        try:
            # 分析查询特征
            query_type = self._analyze_query(query)
            
            # 根据查询类型选择策略
            if query_type == "simple":
                return self.vector_strategy.retrieve(query, collection_name, top_k, **kwargs)
            elif query_type == "complex":
                return self.hybrid_strategy.retrieve(query, collection_name, top_k, **kwargs)
            else:
                return self.vector_strategy.retrieve(query, collection_name, top_k, **kwargs)
        except Exception as e:
            logger.error(f"自动检索失败: {e}")
            return self.vector_strategy.retrieve(query, collection_name, top_k, **kwargs)
    
    def _analyze_query(self, query: str) -> str:
        """分析查询类型"""
        query_lower = query.lower()
        
        # 简单查询：短且无复杂结构
        if len(query.split()) <= 5 and not any(
            word in query_lower for word in ["and", "or", "not", "关系", "关联"]
        ):
            return "simple"
        
        # 复杂查询：包含关系查询
        if any(word in query_lower for word in ["关系", "关联", "连接", "路径"]):
            return "complex"
        
        return "simple"
    
    def get_strategy_name(self) -> str:
        return "auto"


class RetrievalStrategyFactory:
    """检索策略工厂
    
    负责创建和管理检索策略实例。
    """
    
    _strategy_classes: dict[str, type[RetrievalStrategy]] = {
        "vector": VectorRetrievalStrategy,
        "graph": GraphRetrievalStrategy,
        "keyword": KeywordRetrievalStrategy,
        "hybrid": HybridRetrievalStrategy,
        "auto": AutoRetrievalStrategy,
    }
    
    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        vector_store_manager=None,
        graph_store_manager=None,
        index_service=None,
        **kwargs
    ) -> RetrievalStrategy:
        """创建检索策略
        
        Args:
            strategy_name: 策略名称
            vector_store_manager: 向量存储管理器
            graph_store_manager: 图存储管理器
            index_service: 索引服务
            **kwargs: 额外参数
        
        Returns:
            检索策略实例
        """
        if strategy_name not in cls._strategy_classes:
            raise ValueError(
                f"未知的检索策略: {strategy_name}，可选: {list(cls._strategy_classes.keys())}"
            )
        
        strategy_class = cls._strategy_classes[strategy_name]
        
        # 根据策略类型传递不同的参数
        if strategy_name == "keyword":
            return strategy_class(vector_store_manager, graph_store_manager, index_service)
        elif strategy_name == "auto":
            return strategy_class(vector_store_manager, graph_store_manager, index_service)
        else:
            return strategy_class(vector_store_manager, graph_store_manager, **kwargs)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[RetrievalStrategy]) -> None:
        """注册新的检索策略（扩展点）
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        if not issubclass(strategy_class, RetrievalStrategy):
            raise TypeError(f"策略类必须继承 RetrievalStrategy")
        cls._strategy_classes[name] = strategy_class
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """列出所有可用的策略"""
        return list(cls._strategy_classes.keys())


__all__ = [
    "RetrievalStrategy",
    "VectorRetrievalStrategy",
    "GraphRetrievalStrategy",
    "KeywordRetrievalStrategy",
    "HybridRetrievalStrategy",
    "AutoRetrievalStrategy",
    "RetrievalStrategyFactory",
]

