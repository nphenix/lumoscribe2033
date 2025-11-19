"""
检索器配置

定义 RAG 检索器的各种配置选项和参数。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    VECTOR = "vector"  # 向量检索
    KEYWORD = "keyword"  # 关键词检索
    HYBRID = "hybrid"  # 混合检索
    GRAPH = "graph"  # 图检索


class RerankStrategy(Enum):
    """重排序策略枚举"""
    CROSS_ENCODER = "cross_encoder"  # 交叉编码器
    BM25 = "bm25"  # BM25算法
    SIMPLE = "simple"  # 简单相关性评分
    LLM = "llm"  # LLM重排序


@dataclass
class VectorRetrieverConfig:
    """向量检索器配置"""
    collection_name: str = "default"
    embedding_model: str = "text-embedding-ada-002"
    search_k: int = 20  # 检索的文档数量
    similarity_threshold: float = 0.7  # 相似度阈值
    distance_metric: str = "cosine"  # 距离度量方式


@dataclass
class KeywordRetrieverConfig:
    """关键词检索器配置"""
    index_name: str = "default"
    search_k: int = 15
    bm25_k1: float = 1.5  # BM25参数
    bm25_b: float = 0.75  # BM25参数


@dataclass
class GraphRetrieverConfig:
    """图检索器配置"""
    graph_name: str = "default"
    max_hops: int = 2  # 最大跳数
    node_types: list[str] | None = field(default=None)  # 节点类型过滤
    relationship_types: list[str] | None = field(default=None)  # 关系类型过滤


@dataclass
class RerankConfig:
    """重排序配置"""
    strategy: RerankStrategy = RerankStrategy.SIMPLE
    rerank_k: int = 10  # 重排序后的文档数量
    cross_encoder_model: str = "cross-enoder-ms-marco-MiniLM-L-6-v2"
    llm_rerank_prompt: str | None = field(default=None)  # LLM重排序提示词


@dataclass
class HybridRetrieverConfig:
    """混合检索器配置"""
    vector_config: VectorRetrieverConfig | None = field(default=None)
    keyword_config: KeywordRetrieverConfig | None = field(default=None)
    graph_config: GraphRetrieverConfig | None = field(default=None)
    fusion_strategy: str = "rrf"  # 融合策略: rrf, weighted, reciprocal
    weights: dict[str, float] | None = field(default=None)  # 各检索器权重
    rerank_config: RerankConfig | None = field(default=None)

    def __post_init__(self):
        if self.vector_config is None:
            self.vector_config = VectorRetrieverConfig()
        if self.keyword_config is None:
            self.keyword_config = KeywordRetrieverConfig()
        if self.graph_config is None:
            self.graph_config = GraphRetrieverConfig()
        if self.rerank_config is None:
            self.rerank_config = RerankConfig()
        if self.weights is None:
            self.weights = {"vector": 0.5, "keyword": 0.3, "graph": 0.2}


@dataclass
class RetrievalPipelineConfig:
    """检索流水线配置"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    pre_processing: list[str] | None = field(default=None)  # 预处理步骤
    post_processing: list[str] | None = field(default=None)  # 后处理步骤
    max_retries: int = 3  # 最大重试次数
    timeout: int = 30  # 超时时间（秒）
    cache_enabled: bool = True  # 是否启用缓存
    cache_ttl: int = 3600  # 缓存TTL（秒）

    def __post_init__(self):
        if self.pre_processing is None:
            self.pre_processing = ["query_cleaning", "query_expansion"]
        if self.post_processing is None:
            self.post_processing = ["deduplication", "relevance_filtering"]


class RetrieverConfigManager:
    """检索器配置管理器"""

    def __init__(self):
        self.configs: dict[str, Any] = {}
        self.default_config = self._create_default_config()

    def _create_default_config(self) -> HybridRetrieverConfig:
        """创建默认配置"""
        return HybridRetrieverConfig(
            vector_config=VectorRetrieverConfig(
                collection_name="default",
                search_k=20,
                similarity_threshold=0.7
            ),
            keyword_config=KeywordRetrieverConfig(
                search_k=15
            ),
            graph_config=GraphRetrieverConfig(
                max_hops=2
            ),
            fusion_strategy="rrf",
            weights={"vector": 0.5, "keyword": 0.3, "graph": 0.2},
            rerank_config=RerankConfig(
                strategy=RerankStrategy.SIMPLE,
                rerank_k=10
            )
        )

    def get_config(self, config_name: str = "default") -> Any:
        """获取配置"""
        return self.configs.get(config_name, self.default_config)

    def set_config(self, config_name: str, config: Any) -> None:
        """设置配置"""
        self.configs[config_name] = config

    def create_vector_config(
        self,
        collection_name: str,
        embedding_model: str = "text-embedding-ada-002",
        search_k: int = 20,
        similarity_threshold: float = 0.7
    ) -> VectorRetrieverConfig:
        """创建向量检索配置"""
        return VectorRetrieverConfig(
            collection_name=collection_name,
            embedding_model=embedding_model,
            search_k=search_k,
            similarity_threshold=similarity_threshold
        )

    def create_hybrid_config(
        self,
        vector_config: VectorRetrieverConfig,
        keyword_config: KeywordRetrieverConfig = None,
        graph_config: GraphRetrieverConfig = None,
        fusion_strategy: str = "rrf",
        weights: dict[str, float] = None,
        rerank_config: RerankConfig = None
    ) -> HybridRetrieverConfig:
        """创建混合检索配置"""
        return HybridRetrieverConfig(
            vector_config=vector_config,
            keyword_config=keyword_config,
            graph_config=graph_config,
            fusion_strategy=fusion_strategy,
            weights=weights,
            rerank_config=rerank_config
        )

    def create_pipeline_config(
        self,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        pre_processing: list[str] = None,
        post_processing: list[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ) -> RetrievalPipelineConfig:
        """创建检索流水线配置"""
        return RetrievalPipelineConfig(
            strategy=strategy,
            pre_processing=pre_processing,
            post_processing=post_processing,
            max_retries=max_retries,
            timeout=timeout
        )
