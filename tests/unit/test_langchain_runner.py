"""
LangChain Runner 单元测试

测试 LangChain 1.0 RouterChain + RunnableSequence 多模型路由功能
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.framework.orchestrators.langchain_runner import LangChainRunner, RouteDecision


class MockChatModel:
    """模拟聊天模型"""

    def __init__(self, name: str, response: str = "Mock response", should_fail: bool = False):
        self.name = name
        self.response = response
        self.should_fail = should_fail

    async def ainvoke(self, messages, **kwargs):
        if self.should_fail:
            raise Exception("Mock model failed")
        return Mock(content=self.response, usage_metadata={})


class TestLangChainRunner:
    """LangChain Runner 测试类"""

    def setup_method(self):
        """测试前设置"""
        self.mock_models = {
            "openai-gpt4": MockChatModel("openai-gpt4", "GPT-4 response"),
            "openai-gpt35": MockChatModel("openai-gpt35", "GPT-3.5 response"),
            "ollama-mistral": MockChatModel("ollama-mistral", "Mistral response"),
            "ollama-llama2": MockChatModel("ollama-llama2", "Llama2 response")
        }
        self.runner = LangChainRunner(self.mock_models)

    def test_initialization(self):
        """测试初始化"""
        assert len(self.runner.models) == 4
        assert "openai-gpt4" in self.runner.models
        assert "openai-gpt35" in self.runner.models
        assert "ollama-mistral" in self.runner.models
        assert "ollama-llama2" in self.runner.models

    def test_model_configs(self):
        """测试模型配置"""
        configs = self.runner.model_configs

        assert len(configs) == 4
        assert "openai-gpt4" in configs
        assert configs["openai-gpt4"]["cost_per_token"] == 0.03
        assert "complex_reasoning" in configs["openai-gpt4"]["capabilities"]

    def test_performance_metrics_initialization(self):
        """测试性能指标初始化"""
        metrics = self.runner.performance_metrics

        assert len(metrics) == 4
        assert "openai-gpt4" in metrics
        assert metrics["openai-gpt4"]["success_rate"] == 1.0
        assert metrics["openai-gpt4"]["avg_response_time"] == 0.0

    @pytest.mark.asyncio
    async def test_route_request_success(self):
        """测试路由请求成功"""
        request = "请帮我分析这个复杂的数学问题"

        with patch.object(self.runner.routing_chain, 'arun') as mock_routing:
            mock_routing.return_value = """model_name: openai-gpt4
reason: 复杂推理任务需要高质量模型
confidence: 0.95"""

            result = await self.runner.route_request(request)

            assert result["success"] is True
            assert result["model_name"] == "openai-gpt4"
            assert result["reason"] == "复杂推理任务需要高质量模型"
            assert result["confidence"] == 0.95
            assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_route_request_failure(self):
        """测试路由请求失败"""
        request = "测试请求"

        with patch.object(self.runner.routing_chain, 'arun') as mock_routing:
            mock_routing.side_effect = Exception("Routing failed")

            result = await self.runner.route_request(request)

            assert result["success"] is False
            assert "error" in result
            assert result["model_name"] in self.runner.models  # 应该返回默认模型

    @pytest.mark.asyncio
    async def test_execute_request_success(self):
        """测试执行请求成功"""
        request = "写一个Python函数来计算斐波那契数列"

        with patch.object(self.runner, 'route_request') as mock_route, \
             patch.object(self.runner, '_update_metrics') as mock_update_metrics:

            mock_route.return_value = {
                "model_name": "openai-gpt4",
                "reason": "代码分析任务",
                "confidence": 0.9,
                "success": True
            }

            result = await self.runner.execute_request(request)

            assert result["success"] is True
            assert result["model"] == "openai-gpt4"
            assert "response" in result
            assert "execution_time" in result
            assert "routing_info" in result
            mock_update_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_request_routing_failure(self):
        """测试执行请求路由失败"""
        request = "测试请求"

        with patch.object(self.runner, 'route_request') as mock_route:
            mock_route.return_value = {
                "success": False,
                "error": "Routing failed"
            }

            with pytest.raises(Exception):
                await self.runner.execute_request(request)

    @pytest.mark.asyncio
    async def test_execute_request_model_failure(self):
        """测试执行请求模型失败"""
        # 创建会失败的模型
        failing_model = MockChatModel("failing-model", should_fail=True)
        runner = LangChainRunner({"failing-model": failing_model})

        with patch.object(runner, 'route_request') as mock_route, \
             patch.object(runner, '_update_metrics') as mock_update_metrics:

            mock_route.return_value = {
                "model_name": "failing-model",
                "reason": "测试失败",
                "confidence": 0.5,
                "success": True
            }

            with pytest.raises(Exception):
                await runner.execute_request("测试请求")

            # 验证失败指标被更新
            mock_update_metrics.assert_called_once_with(
                "failing-model", success=False, response_time=pytest.approx(0, abs=1.0)
            )

    @pytest.mark.asyncio
    async def test_execute_chain(self):
        """测试链式执行"""
        request = "分析这段代码"
        chain_config = {
            "system_prompt": "你是一个代码分析专家",
            "parse_output": True
        }

        with patch.object(self.runner, 'route_request') as mock_route, \
             patch.object(self.runner, '_update_metrics') as mock_update_metrics:

            mock_route.return_value = {
                "model_name": "ollama-mistral",
                "reason": "代码分析任务",
                "confidence": 0.85,
                "success": True
            }

            result = await self.runner.execute_chain(request, chain_config)

            assert result["success"] is True
            assert result["model"] == "ollama-mistral"
            assert "result" in result
            assert "execution_time" in result
            mock_update_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_metrics(self):
        """测试更新性能指标"""
        model_name = "openai-gpt4"
        initial_metrics = self.runner.performance_metrics[model_name].copy()

        await self.runner._update_metrics(model_name, success=True, response_time=1.5)

        updated_metrics = self.runner.performance_metrics[model_name]

        # 验证成功率更新 (指数加权平均)
        expected_success_rate = 0.9 * initial_metrics["success_rate"] + 0.1 * 1.0
        assert abs(updated_metrics["success_rate"] - expected_success_rate) < 0.001

        # 验证响应时间更新
        assert updated_metrics["avg_response_time"] == 1.5
        assert updated_metrics["total_requests"] == initial_metrics["total_requests"] + 1

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        metrics = self.runner.get_performance_metrics()

        assert len(metrics) == 4
        assert "openai-gpt4" in metrics
        assert "capabilities" in metrics["openai-gpt4"]
        assert "success_rate" in metrics["openai-gpt4"]
        assert "avg_response_time" in metrics["openai-gpt4"]

    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.runner.get_model_info()

        assert len(info) == 4
        assert "openai-gpt4" in info
        assert "config" in info["openai-gpt4"]
        assert "metrics" in info["openai-gpt4"]
        assert info["openai-gpt4"]["config"]["name"] == "openai-gpt4"

    @pytest.mark.asyncio
    async def test_complex_routing_scenarios(self):
        """测试复杂路由场景"""
        test_cases = [
            ("写一个复杂的故事", "openai-gpt4", "创意写作"),
            ("帮我调试Python代码", "ollama-mistral", "代码分析"),
            ("简单的问题", "ollama-llama2", "简单查询"),
            ("一般对话", "openai-gpt35", "一般对话")
        ]

        for request, expected_model, expected_reason in test_cases:
            with patch.object(self.runner.routing_chain, 'ainvoke') as mock_routing:
                mock_routing.return_value = RouteDecision(
                    model_name=expected_model,
                    reason=expected_reason,
                    confidence=0.8
                )

                result = await self.runner.route_request(request)

                assert result["model_name"] == expected_model
                assert expected_reason.lower() in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_default_model_selection(self):
        """测试默认模型选择"""
        # 测试路由失败时的默认模型选择
        with patch.object(self.runner.routing_chain, 'ainvoke') as mock_routing:
            mock_routing.side_effect = Exception("Routing failed")

            result = await self.runner.route_request("测试")

            # 应该返回一个有效的默认模型
            assert result["model_name"] in self.runner.models
            assert result["success"] is False

    def test_model_capabilities(self):
        """测试模型能力映射"""
        configs = self.runner.model_configs

        # 验证不同模型的能力
        assert "complex_reasoning" in configs["openai-gpt4"]["capabilities"]
        assert "code_analysis" in configs["ollama-mistral"]["capabilities"]
        assert "simple_queries" in configs["ollama-llama2"]["capabilities"]
        assert "general_conversation" in configs["openai-gpt35"]["capabilities"]

    def test_cost_mapping(self):
        """测试成本映射"""
        configs = self.runner.model_configs

        assert configs["openai-gpt4"]["cost_per_token"] == 0.03
        assert configs["openai-gpt35"]["cost_per_token"] == 0.005
        assert configs["ollama-llama2"]["cost_per_token"] == 0.001
        assert configs["ollama-mistral"]["cost_per_token"] == 0.002


class TestRouteDecision:
    """RouteDecision 模型测试"""

    def test_route_decision_creation(self):
        """测试路由决策创建"""
        decision = RouteDecision(
            model_name="openai-gpt4",
            reason="复杂推理任务",
            confidence=0.95
        )

        assert decision.model_name == "openai-gpt4"
        assert decision.reason == "复杂推理任务"
        assert decision.confidence == 0.95

    def test_route_decision_validation(self):
        """测试路由决策验证"""
        # 测试置信度范围验证
        with pytest.raises(Exception):
            RouteDecision(
                model_name="openai-gpt4",
                reason="测试",
                confidence=1.5  # 超出范围
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
