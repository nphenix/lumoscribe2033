"""
LLM 路由器适配器单元测试（阶段 B 版本）

主要验证：
- 适配器是否正确委托给 LangChainExecutor / LangChainRunner
- 执行结果结构是否符合预期
- 自适应适配器是否传递预算提示 / metadata
"""

from types import SimpleNamespace

import pytest

from src.framework.adapters.llm_router_adapter import (
    AdaptiveLLMRouterAdapter,
    LangChainV1RouterAdapter,
)


class DummyRunner:
    def __init__(self):
        self.models = {"model-a": SimpleNamespace()}
        self.route_calls = []

    async def route_request(self, request, context=None):
        self.route_calls.append((request, context))
        return {
            "model_name": "model-a",
            "reason": "mock",
            "confidence": 0.9,
            "success": True,
        }

    def get_performance_metrics(self):
        return {"model-a": {"success_rate": 1.0}}


class DummyExecutor:
    def __init__(self):
        self.runner = DummyRunner()
        self.execute_calls = []
        self.agent_calls = []

    async def execute(self, request, **kwargs):
        self.execute_calls.append((request, kwargs))
        return {
            "model": "model-a",
            "response": f"resp:{request}",
            "usage": {"prompt_tokens": 1},
            "execution_time": 0.1,
            "success": True,
            "routing_info": {"model_name": "model-a"},
        }

    async def run_agent(self, task, **kwargs):
        self.agent_calls.append((task, kwargs))
        return f"agent:{task}"


@pytest.fixture
def base_adapter(monkeypatch):
    executor = DummyExecutor()
    monkeypatch.setattr(
        "src.framework.adapters.llm_router_adapter.LangChainExecutor.from_models",
        lambda *args, **kwargs: executor,
    )
    adapter = LangChainV1RouterAdapter({"model-a": SimpleNamespace()})
    return adapter, executor


@pytest.fixture
def adaptive_adapter(monkeypatch):
    executor = DummyExecutor()
    monkeypatch.setattr(
        "src.framework.adapters.llm_router_adapter.LangChainExecutor.from_models",
        lambda *args, **kwargs: executor,
    )
    adapter = AdaptiveLLMRouterAdapter({"model-a": SimpleNamespace()})
    return adapter, executor


@pytest.mark.asyncio
async def test_route_request_delegates_runner(base_adapter):
    adapter, executor = base_adapter
    result = await adapter.route_request("hello")
    assert executor.runner.route_calls[0][0] == "hello"
    assert result["model_name"] == "model-a"


@pytest.mark.asyncio
async def test_execute_request_returns_shape(base_adapter):
    adapter, executor = base_adapter
    result = await adapter.execute_request("hello")
    assert result["model"] == "model-a"
    assert result["response"] == "resp:hello"
    assert result["success"] is True
    assert executor.execute_calls  # ensure executor 被调用


@pytest.mark.asyncio
async def test_execute_with_agent(base_adapter):
    adapter, executor = base_adapter
    result = await adapter.execute_with_agent("do something")
    assert result["response"] == "agent:do something"
    assert executor.agent_calls


@pytest.mark.asyncio
async def test_get_routing_stats(base_adapter):
    adapter, executor = base_adapter
    stats = await adapter.get_routing_stats()
    assert stats["total_models"] == 1
    assert stats["performance_metrics"] == executor.runner.get_performance_metrics()


@pytest.mark.asyncio
async def test_adaptive_adapter_passes_budget_hint(adaptive_adapter):
    adapter, executor = adaptive_adapter
    await adapter.execute_request("hello", budget_hint="low", metadata={})
    # route_request context
    assert executor.runner.route_calls[-1][1] == {"budget_hint": "low"}
    # metadata 替换
    _, kwargs = executor.execute_calls[-1]
    assert kwargs["metadata"]["budget_hint"] == "low"
