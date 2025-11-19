"""
增强路由系统测试

测试智能路由决策、健康检查、故障转移等功能
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.framework.orchestrators.health_checker import (
    HealthCheckConfig,
    LangChainHealthChecker,
)
from src.framework.orchestrators.langchain_runner import LangChainRunner


class TestEnhancedRouter:
    """增强路由系统测试类"""

    def setup_method(self):
        """测试前设置"""
        # 使用实际的katpro1模型配置
        try:
            import os

            from langchain_openai import ChatOpenAI

            # 从环境变量获取katpro1配置
            katpro1_api_key = os.environ.get("KATPRO1_API_KEY", "3jreEq8e3v0n7_sWEQAxYCRHB1UwwF7M3S6bqQOrnm8")
            katpro1_base_url = os.environ.get("KATPRO1_BASE_URL", "https://wanqing.streamlakeapi.com/api/gateway/v1/endpoints")
            os.environ.get("KATPRO1_MODEL", "ep-zd78oa-1761741032688717062")

            self.models = {
                "katpro1-doc": ChatOpenAI(
                    model="ep-zd78oa-1761741032688717062",
                    temperature=0.3,
                    max_tokens=256000,
                    timeout=60,
                    api_key=katpro1_api_key,
                    base_url=katpro1_base_url,
                    name="KATpro1-文档分析"
                ),
                "katpro1-code": ChatOpenAI(
                    model="ep-zd78oa-1761741032688717062",
                    temperature=0.1,
                    max_tokens=256000,
                    timeout=120,
                    api_key=katpro1_api_key,
                    base_url=katpro1_base_url,
                    name="KATpro1-代码分析"
                ),
                "katpro1-general": ChatOpenAI(
                    model="ep-zd78oa-1761741032688717062",
                    temperature=0.7,
                    max_tokens=4000,
                    timeout=30,
                    api_key=katpro1_api_key,
                    base_url=katpro1_base_url,
                    name="KATpro1-通用"
                ),
                "katpro1-fast": ChatOpenAI(
                    model="ep-zd78oa-1761741032688717062",
                    temperature=0.9,
                    max_tokens=2000,
                    timeout=15,
                    api_key=katpro1_api_key,
                    base_url=katpro1_base_url,
                    name="KATpro1-快速响应"
                )
            }

            # 创建路由执行器
            self.runner = LangChainRunner(self.models)

            # 使用配置管理器设置模型成本
            from src.framework.shared.config import config_manager

            # 更新模型配置中的成本信息
            if config_manager.get_model_by_name("katpro1-doc"):
                config_manager.get_model_by_name("katpro1-doc").cost_per_token = 0.015
            if config_manager.get_model_by_name("katpro1-code"):
                config_manager.get_model_by_name("katpro1-code").cost_per_token = 0.020
            if config_manager.get_model_by_name("katpro1-general"):
                config_manager.get_model_by_name("katpro1-general").cost_per_token = 0.010
            if config_manager.get_model_by_name("katpro1-fast"):
                config_manager.get_model_by_name("katpro1-fast").cost_per_token = 0.005

        except ImportError:
            # 如果无法导入langchain_openai，使用模拟
            from langchain_core.language_models.fake_chat_models import (
                GenericFakeChatModel,
            )
            from langchain_core.messages import AIMessage

            self.models = {
                "katpro1-doc": GenericFakeChatModel(messages=iter([
                    AIMessage(content="Response from katpro1-doc: 文档分析结果"),
                    AIMessage(content="Response from katpro1-doc: 详细文档处理")
                ])),
                "katpro1-code": GenericFakeChatModel(messages=iter([
                    AIMessage(content="Response from katpro1-code: 代码分析结果"),
                    AIMessage(content="Response from katpro1-code: 技术解决方案")
                ])),
                "katpro1-general": GenericFakeChatModel(messages=iter([
                    AIMessage(content="Response from katpro1-general: 通用回答"),
                    AIMessage(content="Response from katpro1-general: 标准响应")
                ])),
                "katpro1-fast": GenericFakeChatModel(messages=iter([
                    AIMessage(content="Response from katpro1-fast: 快速回答"),
                    AIMessage(content="Response from katpro1-fast: 简洁响应")
                ]))
            }

            self.runner = LangChainRunner(self.models)

            # 使用配置管理器设置模型成本
            from src.framework.shared.config import config_manager

            # 更新模型配置中的成本信息
            if config_manager.get_model_by_name("katpro1-doc"):
                config_manager.get_model_by_name("katpro1-doc").cost_per_token = 0.015
            if config_manager.get_model_by_name("katpro1-code"):
                config_manager.get_model_by_name("katpro1-code").cost_per_token = 0.020
            if config_manager.get_model_by_name("katpro1-general"):
                config_manager.get_model_by_name("katpro1-general").cost_per_token = 0.010
            if config_manager.get_model_by_name("katpro1-fast"):
                config_manager.get_model_by_name("katpro1-fast").cost_per_token = 0.005

    @pytest.mark.asyncio
    async def test_content_analysis(self):
        """测试内容分析功能"""
        test_cases = [
            ("请帮我分析这段Python代码", "code_analysis"),
            ("解决这个数学问题", "complex_reasoning"),
            ("写一个科幻故事", "creative_writing"),
            ("总结这篇文档", "text_processing"),
            ("什么是Python?", "simple_query"),
            ("你好", "general_conversation")
        ]

        for input_text, expected_type in test_cases:
            analysis = self.runner._analyze_content(input_text)
            assert analysis["content_type"] == expected_type
            assert "reason" in analysis
            assert "complexity_score" in analysis

    @pytest.mark.asyncio
    async def test_model_scoring(self):
        """测试模型评分功能"""
        content_analysis = {
            "content_type": "code_analysis",
            "complexity_score": 0.8
        }

        # 测试基础评分
        scores = self.runner._score_models(content_analysis)
        assert len(scores) == 4  # 4个模型
        assert scores["katpro1-general"] > 0
        assert scores["katpro1-code"] > scores["katpro1-fast"]  # 代码分析能力更强

        # 测试可用性影响
        model_availability = {
            "katpro1-doc": True,
            "katpro1-code": True,
            "katpro1-general": True,
            "katpro1-fast": True
        }

        scores_with_availability = self.runner._score_models(content_analysis, model_availability)
        assert scores_with_availability["katpro1-fast"] > 0  # 所有模型都可用

    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        """测试健康检查集成"""
        # 初始化健康检查器
        self.runner.initialize_health_checker()
        assert self.runner.health_checker is not None

        # 启动监控
        await self.runner.start_health_monitoring()

        # 模拟健康状态
        self.runner.health_checker.health_status = {
            "katpro1-doc": {
                "is_healthy": True,
                "success_rate": 0.95,
                "response_time": 2.0,
                "metadata": {"consecutive_failures": 0}
            },
            "katpro1-code": {
                "is_healthy": True,
                "success_rate": 0.90,
                "response_time": 3.0,
                "metadata": {"consecutive_failures": 0}
            }
        }

        # 测试可用性获取
        availability = await self.runner._get_model_availability()
        assert availability["katpro1-doc"]
        assert availability["katpro1-code"]

        # 停止监控
        await self.runner.stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_intelligent_routing(self):
        """测试智能路由决策"""
        test_input = "请帮我分析这段Python代码的性能问题"

        # 执行智能路由
        result = await self.runner._intelligent_route(test_input)

        assert "model_name" in result
        assert "reason" in result
        assert "confidence" in result
        assert result["success"]
        assert "analysis" in result
        assert "scores" in result

        # 验证路由决策合理性
        assert result["analysis"]["content_type"] == "code_analysis"
        assert result["confidence"] > 0.0
        assert result["confidence"] <= 1.0

        # 代码分析应该倾向于代码能力强的模型
        selected_model = result["model_name"]
        assert selected_model in ["katpro1-code", "katpro1-doc", "katpro1-general"]

    @pytest.mark.asyncio
    async def test_fallback_routing(self):
        """测试故障转移路由"""
        test_input = "简单的问题"
        content_analysis = {"content_type": "simple_query", "complexity_score": 0.2}

        # 模拟所有高质量模型不可用
        model_availability = {
            "katpro1-doc": True,
            "katpro1-code": False,  # 不可用
            "katpro1-general": True,
            "katpro1-fast": True
        }

        # 测试智能故障转移
        current_scores = {
            "katpro1-doc": 0.8,
            "katpro1-general": 0.7,
            "katpro1-fast": 0.6
        }

        result = await self.runner._intelligent_fallback_routing(
            test_input, content_analysis, current_scores, model_availability
        )

        model_name, score = result
        assert model_name in ["katpro1-doc", "katpro1-general", "katpro1-fast"]  # 应该选择可用的模型
        assert score > 0

    @pytest.mark.asyncio
    async def test_cost_optimization(self):
        """测试成本优化功能"""
        content_analysis = {
            "content_type": "simple_query",
            "complexity_score": 0.2
        }

        model_scores = {
            "katpro1-code": 0.9,    # 高成本
            "katpro1-doc": 0.8,     # 中等成本
            "katpro1-general": 0.7, # 中等成本
            "katpro1-fast": 0.6    # 低成本
        }

        # 测试高预算敏感度（简单查询通常是预算敏感的）
        optimized_scores = self.runner._apply_cost_optimization(model_scores, content_analysis)

        # 高成本模型分数应该被大幅降低
        assert optimized_scores["katpro1-fast"] > optimized_scores["katpro1-code"]

    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """测试负载均衡功能"""
        # 模拟性能指标
        self.runner.performance_metrics["katpro1-doc"]["total_requests"] = 100
        self.runner.performance_metrics["katpro1-code"]["total_requests"] = 10
        self.runner.performance_metrics["katpro1-general"]["total_requests"] = 50
        self.runner.performance_metrics["katpro1-fast"]["total_requests"] = 1

        model_scores = {
            "katpro1-doc": 0.8,
            "katpro1-code": 0.9,
            "katpro1-general": 0.7,
            "katpro1-fast": 0.6
        }

        balanced_scores = self.runner._apply_load_balancing(model_scores)

        # 高使用率的模型分数应该被降低
        assert balanced_scores["katpro1-fast"] > balanced_scores["katpro1-doc"]
        assert balanced_scores["katpro1-code"] > balanced_scores["katpro1-general"]

    @pytest.mark.asyncio
    async def test_performance_weights(self):
        """测试性能权重功能"""
        # 模拟性能指标
        self.runner.performance_metrics["katpro1-doc"]["success_rate"] = 0.95
        self.runner.performance_metrics["katpro1-doc"]["avg_response_time"] = 2.0

        self.runner.performance_metrics["katpro1-code"]["success_rate"] = 0.3
        self.runner.performance_metrics["katpro1-code"]["avg_response_time"] = 15.0

        model_scores = {
            "katpro1-doc": 0.8,
            "katpro1-code": 0.8
        }

        weighted_scores = self.runner._apply_performance_weights(model_scores)

        # 高性能模型分数应该更高
        assert weighted_scores["katpro1-doc"] > weighted_scores["katpro1-code"]

    @pytest.mark.asyncio
    async def test_complexity_analysis(self):
        """测试复杂度分析功能"""
        test_cases = [
            ("简单问题", 0.1),  # 简单
            ("中等复杂度的问题，包含一些技术术语", 0.4),  # 中等
            ("复杂的技术问题，涉及多个概念和算法，需要详细分析和推理", 0.8)  # 复杂
        ]

        for input_text, expected_range in test_cases:
            analysis = self.runner._analyze_complexity_depth(input_text)

            assert "complexity_score" in analysis
            assert "depth_level" in analysis
            assert "word_count" in analysis
            assert analysis["complexity_score"] >= 0
            assert analysis["complexity_score"] <= 1

            # 验证复杂度级别
            if analysis["complexity_score"] > 0.7:
                assert analysis["depth_level"] == "very_high"
            elif analysis["complexity_score"] > 0.5:
                assert analysis["depth_level"] == "high"

    @pytest.mark.asyncio
    async def test_urgency_assessment(self):
        """测试紧急程度评估"""
        test_cases = [
            ("紧急：需要立即解决", "critical"),
            ("urgent: 快速响应", "high"),
            ("重要问题", "medium"),
            ("一般问题", "low")
        ]

        for input_text, expected_level in test_cases:
            urgency = self.runner._assess_urgency(input_text)

            assert "urgency_score" in urgency
            assert "urgency_level" in urgency
            assert "urgency_weight" in urgency
            assert urgency["urgency_level"] == expected_level

    @pytest.mark.asyncio
    async def test_context_aware_analysis(self):
        """测试上下文感知分析"""
        test_cases = [
            ("请分步骤说明如何实现", True, True),  # 多步骤 + 精确性
            ("详细说明这个概念", False, True),      # 非多步骤 + 精确性
            ("简单回答即可", False, False),         # 非多步骤 + 非精确性
            ("首先做A，然后做B，最后做C", True, False)  # 多步骤 + 非精确性
        ]

        for input_text, expected_multi_step, expected_precision in test_cases:
            context = await self.runner._context_aware_analysis(input_text)

            assert "requires_multi_step" in context
            assert "requires_precision" in context
            assert "context_flags" in context

            assert context["requires_multi_step"] == expected_multi_step
            assert context["requires_precision"] == expected_precision

    @pytest.mark.asyncio
    async def test_routing_with_health_check(self):
        """测试集成健康检查的完整路由"""
        # 初始化健康检查器
        self.runner.initialize_health_checker()

        # 模拟健康状态
        self.runner.health_checker.health_status = {
            "katpro1-doc": {
                "is_healthy": True,
                "success_rate": 0.95,
                "response_time": 2.0,
                "metadata": {"consecutive_failures": 0}
            },
            "katpro1-code": {
                "is_healthy": False,  # 故障状态
                "success_rate": 0.1,
                "response_time": 20.0,
                "metadata": {"consecutive_failures": 5}
            }
        }

        # 执行路由
        test_input = "复杂的技术问题需要高质量回答"
        result = await self.runner._intelligent_route(test_input)

        # 应该选择健康的模型，而不是理论上更适合的故障模型
        assert result["success"]
        assert result["model_name"] != "katpro1-code"  # 不应该选择故障模型
        assert "健康状态" in result["reason"] or "available" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_routing_failure_handling(self):
        """测试路由失败处理"""
        # 模拟所有模型都不可用的情况
        failing_models = {"all-failing": GenericFakeChatModel(messages=iter([
            Exception("模拟模型故障"),
            Exception("持续故障")
        ]))}
        failing_models["all-failing"].cost_per_token = 0.01

        failing_runner = LangChainRunner(failing_models)

        # 测试路由失败时的降级处理
        test_input = "测试输入"
        result = await failing_runner._intelligent_route(test_input)

        assert not result["success"]
        assert "降级" in result["reason"] or "fallback" in result["reason"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
