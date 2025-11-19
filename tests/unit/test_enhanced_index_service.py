"""
增强版索引服务单元测试
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.framework.rag.enhanced_index_service import (
    EnhancedIndexService,
    QueryAnalysis,
    RetrievalMetrics,
)
from src.framework.shared.exceptions import IndexServiceError
from src.framework.shared.models import DocumentChunk


class TestRetrievalMetrics:
    """检索指标测试类"""

    def test_init(self):
        """测试初始化"""
        metrics = RetrievalMetrics(
            retrieval_time=0.123,
            retrieved_count=5,
            query_complexity="medium",
            retrieval_strategy="auto",
            hit_rate=0.8
        )

        assert metrics.retrieval_time == 0.123
        assert metrics.retrieved_count == 5
        assert metrics.query_complexity == "medium"
        assert metrics.retrieval_strategy == "auto"
        assert metrics.hit_rate == 0.8


class TestQueryAnalysis:
    """查询分析测试类"""

    def test_init(self):
        """测试初始化"""
        analysis = QueryAnalysis(
            intent="informational",
            entities=["Python", "programming"],
            keywords=["Python", "language"],
            complexity="simple",
            suggested_strategies=["vector", "auto"]
        )

        assert analysis.intent == "informational"
        assert analysis.entities == ["Python", "programming"]
        assert analysis.keywords == ["Python", "language"]
        assert analysis.complexity == "simple"
        assert analysis.suggested_strategies == ["vector", "auto"]


class TestEnhancedIndexService:
    """增强版索引服务测试类"""

    def setup_method(self):
        """测试前准备"""
        # 创建模拟组件
        self.mock_vector_manager = Mock()
        self.mock_graph_manager = Mock()
        self.mock_llamaindex_service = Mock()

        # 初始化服务
        self.service = EnhancedIndexService(
            vector_store_manager=self.mock_vector_manager,
            graph_store_manager=self.mock_graph_manager,
            llamaindex_service=self.mock_llamaindex_service,
            enable_auto_retriever=True,
            enable_query_analysis=True,
            enable_metrics=True
        )

    def test_init(self):
        """测试初始化"""
        assert self.service.vector_store_manager is self.mock_vector_manager
        assert self.service.graph_store_manager is self.mock_graph_manager
        assert self.service.llamaindex_service is self.mock_llamaindex_service
        assert self.service.enable_auto_retriever is True
        assert self.service.enable_query_analysis is True
        assert self.service.enable_metrics is True
        assert len(self.service.query_cache) == 0
        assert len(self.service.metrics_history) == 0
        assert len(self.service.query_analysis_cache) == 0

    def test_analyze_query_basic(self):
        """测试基本查询分析"""
        query = "什么是 Python？"
        analysis = self.service.analyze_query(query)

        assert isinstance(analysis, QueryAnalysis)
        assert analysis.intent in ["informational", "general"]
        assert isinstance(analysis.entities, list)
        assert isinstance(analysis.keywords, list)
        assert analysis.complexity in ["simple", "medium", "complex"]
        assert isinstance(analysis.suggested_strategies, list)

    def test_analyze_query_disabled(self):
        """测试禁用查询分析"""
        service = EnhancedIndexService(enable_query_analysis=False)
        query = "测试查询"

        analysis = service.analyze_query(query)

        assert analysis.intent == "general"
        assert analysis.complexity == "medium"
        assert analysis.suggested_strategies == ["auto"]

    def test_analyze_query_cache_hit(self):
        """测试查询分析缓存命中"""
        query = "Python 编程"

        # 第一次分析
        analysis1 = self.service.analyze_query(query)

        # 第二次分析（应该命中缓存）
        analysis2 = self.service.analyze_query(query)

        assert analysis1 is analysis2

    def test_retrieve_auto_strategy(self):
        """测试自动策略检索"""
        query = "Python 编程"
        collection_name = "test_collection"
        top_k = 3

        # 模拟检索结果
        mock_node = Mock(spec=TextNode)
        mock_node.text = "Python 是一种编程语言"
        mock_node.metadata = {}
        mock_node.node_id = "test_id"

        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = mock_node
        mock_result.score = 0.9

        self.mock_vector_manager.get_index.return_value.as_retriever.return_value.retrieve.return_value = [mock_result]

        # 执行检索
        results = self.service.retrieve(
            query,
            collection_name=collection_name,
            strategy="auto",
            top_k=top_k
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_retrieve_vector_strategy(self):
        """测试向量策略检索"""
        query = "机器学习"
        collection_name = "test_collection"
        top_k = 5

        # 模拟向量检索
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = Mock()
        mock_result.node.text = "机器学习是 AI 的分支"
        mock_result.score = 0.85

        self.mock_vector_manager.get_index.return_value.as_retriever.return_value.retrieve.return_value = [mock_result]

        results = self.service.retrieve(
            query,
            collection_name=collection_name,
            strategy="vector",
            top_k=top_k
        )

        assert len(results) == 1
        assert results[0].score == 0.85

    def test_retrieve_invalid_strategy(self):
        """测试无效检索策略"""
        query = "测试"

        with pytest.raises(IndexServiceError, match="不支持的检索策略"):
            self.service.retrieve(
                query,
                strategy="invalid_strategy"
            )

    def test_retrieve_cache_hit(self):
        """测试检索缓存命中"""
        query = "缓存测试"
        collection_name = "test_collection"
        strategy = "auto"
        top_k = 3

        # 模拟检索结果
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = Mock()
        mock_result.score = 0.9

        self.mock_vector_manager.get_index.return_value.as_retriever.return_value.retrieve.return_value = [mock_result]

        # 第一次检索
        results1 = self.service.retrieve(query, collection_name, strategy, top_k)

        # 第二次检索（应该命中缓存）
        results2 = self.service.retrieve(query, collection_name, strategy, top_k)

        assert results1 is results2

    def test_retrieve_with_metrics(self):
        """测试检索性能指标"""
        query = "性能测试"
        collection_name = "test_collection"

        # 模拟检索
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = Mock()
        mock_result.score = 0.8

        self.mock_vector_manager.get_index.return_value.as_retriever.return_value.retrieve.return_value = [mock_result]

        # 执行检索
        start_time = time.time()
        self.service.retrieve(query, collection_name, "auto")
        end_time = time.time()

        # 验证指标记录
        assert len(self.service.metrics_history) == 1
        metrics = self.service.metrics_history[0]

        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.retrieved_count == 1
        assert metrics.query_complexity in ["simple", "medium", "complex"]
        assert metrics.retrieval_strategy == "auto"
        assert metrics.retrieval_time <= (end_time - start_time)

    def test_create_query_engine_auto(self):
        """测试创建自动查询引擎"""
        collection_name = "test_collection"

        # 模拟索引和检索器
        mock_index = Mock()
        mock_retriever = Mock()
        mock_index.as_retriever.return_value = mock_retriever

        self.mock_vector_manager.get_index.return_value = mock_index
        self.mock_vector_manager.get_index.return_value.as_retriever.return_value = mock_retriever

        with patch('src.framework.rag.enhanced_index_service.AutoRetriever') as mock_auto_retriever:
            mock_auto_retriever.from_defaults.return_value = mock_retriever

            query_engine = self.service.create_query_engine(
                collection_name=collection_name,
                strategy="auto"
            )

            assert query_engine is not None

    def test_create_query_engine_vector(self):
        """测试创建向量查询引擎"""
        collection_name = "test_collection"

        # 模拟索引和检索器
        mock_index = Mock()
        mock_retriever = Mock()
        mock_index.as_retriever.return_value = mock_retriever

        self.mock_vector_manager.get_index.return_value = mock_index

        query_engine = self.service.create_query_engine(
            collection_name=collection_name,
            strategy="vector"
        )

        assert query_engine is not None

    def test_get_retrieval_metrics(self):
        """测试获取检索指标"""
        # 添加一些测试指标
        test_metrics = [
            RetrievalMetrics(0.1, 5, "simple", "auto"),
            RetrievalMetrics(0.2, 3, "medium", "vector"),
            RetrievalMetrics(0.15, 4, "simple", "auto")
        ]

        self.service.metrics_history = test_metrics

        metrics = self.service.get_retrieval_metrics()

        assert isinstance(metrics, dict)
        assert "total_queries" in metrics
        assert "avg_retrieval_time" in metrics
        assert "strategy_distribution" in metrics
        assert "complexity_distribution" in metrics
        assert metrics["total_queries"] == 3

    def test_get_retrieval_metrics_empty(self):
        """测试空指标"""
        metrics = self.service.get_retrieval_metrics()

        assert "error" in metrics
        assert "没有检索历史数据" in metrics["error"]

    def test_detect_query_intent(self):
        """测试查询意图检测"""
        test_cases = [
            ("什么是 Python？", "informational"),
            ("如何学习机器学习？", "explanatory"),
            ("列出 Python 库", "navigational"),
            ("Python 和 Java 的区别", "comparative"),
            ("随便看看", "general")
        ]

        for query, expected_intent in test_cases:
            intent = self.service._detect_query_intent(query)
            assert intent == expected_intent

    def test_extract_entities(self):
        """测试实体提取"""
        query = "Python 和机器学习的关系"

        # 简单的实体提取应该返回空列表（在实际实现中会使用 NER）
        entities = self.service._extract_entities(query)

        assert isinstance(entities, list)

    def test_extract_keywords(self):
        """测试关键词提取"""
        query = "什么是 Python 编程语言？"

        keywords = self.service._extract_keywords(query)

        assert isinstance(keywords, list)
        assert "python" in keywords
        assert "programming" in keywords
        # 停用词应该被过滤
        assert "什么" not in keywords

    def test_assess_complexity(self):
        """测试复杂度评估"""
        test_cases = [
            ("Python", "simple"),
            ("什么是 Python 编程语言？", "medium"),
            ("请详细解释 Python 编程语言的特点、历史和发展趋势", "complex")
        ]

        for query, expected_complexity in test_cases:
            complexity = self.service._assess_complexity(query)
            assert complexity == expected_complexity

    def test_suggest_strategies(self):
        """测试策略建议"""
        test_cases = [
            # (query, intent, entities, expected_strategies)
            ("Python", "informational", [], ["auto"]),
            ("Python", "informational", ["Python"], ["auto", "hybrid"]),
            ("列出 Python 库", "navigational", [], ["auto", "vector", "keyword"])
        ]

        for query, intent, entities, expected_strategies in test_cases:
            strategies = self.service._suggest_strategies(query, intent, entities)

            # 验证返回的策略在预期列表中
            for strategy in strategies:
                assert strategy in expected_strategies

    def test_fuse_retrieval_results(self):
        """测试融合检索结果"""
        # 创建模拟结果
        mock_node1 = Mock()
        mock_node1.node_id = "node1"

        mock_node2 = Mock()
        mock_node2.node_id = "node2"

        result1 = Mock(spec=NodeWithScore)
        result1.node = mock_node1
        result1.score = 0.9

        result2 = Mock(spec=NodeWithScore)
        result2.node = mock_node2
        result2.score = 0.8

        vector_results = [result1]
        graph_results = [result2]

        fused_results = self.service._fuse_retrieval_results(
            vector_results, graph_results, top_k=2
        )

        assert isinstance(fused_results, list)
        assert len(fused_results) == 2

    def test_performance_monitor(self):
        """测试性能监控"""
        operation_name = "test_operation"

        with self.service.performance_monitor(operation_name):
            time.sleep(0.01)  # 模拟操作

        # 如果没有异常，监控应该成功完成
        assert True

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 测试委托给基础服务的方法
        assert hasattr(self.service, 'add_documents')
        assert hasattr(self.service, 'search')

    def test_error_handling(self):
        """测试错误处理"""
        # 模拟检索失败
        self.mock_vector_manager.get_index.side_effect = Exception("检索失败")

        with pytest.raises(IndexServiceError, match="检索失败"):
            self.service.retrieve("测试查询", "test_collection")

    def test_metrics_calculation(self):
        """测试指标计算"""
        # 添加测试指标
        self.service.metrics_history = [
            RetrievalMetrics(0.1, 5, "simple", "auto"),
            RetrievalMetrics(0.2, 3, "medium", "vector"),
            RetrievalMetrics(0.15, 4, "simple", "auto")
        ]

        metrics = self.service.get_retrieval_metrics()

        # 验证计算结果
        assert abs(metrics["avg_retrieval_time"] - 0.15) < 0.01
        assert metrics["strategy_distribution"]["auto"] == 2
        assert metrics["complexity_distribution"]["simple"] == 2

    @patch('src.framework.rag.enhanced_index_service.logger')
    def test_logging(self, mock_logger):
        """测试日志记录"""
        query = "日志测试"

        # 模拟检索
        mock_result = Mock(spec=NodeWithScore)
        mock_result.node = Mock()
        mock_result.score = 0.9

        self.mock_vector_manager.get_index.return_value.as_retriever.return_value.retrieve.return_value = [mock_result]

        self.service.retrieve(query, "test_collection")

        # 验证日志调用
        assert mock_logger.info.called or mock_logger.error.called


if __name__ == "__main__":
    pytest.main([__file__])
