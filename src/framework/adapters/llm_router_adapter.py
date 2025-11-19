"""
LLM 路由器适配器

阶段 B：以 LangChainExecutor 为核心，替换手写 RouterChain/Agent 调度逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from ..orchestrators.langchain_executor import LangChainExecutor
from ..shared.logging import get_logger
from ..shared.telemetry import trace_method

logger = get_logger(__name__)


class ExecutionResult(TypedDict):
    model: str
    response: str
    usage: dict[str, Any]
    execution_time: float | None
    success: bool
    routing_info: dict[str, Any]
    error_message: str | None


class LLMRouterAdapter(ABC):
    """LLM 路由器适配器抽象基类"""

    @abstractmethod
    async def route_request(self, request: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    async def execute_request(self, request: str, **kwargs) -> ExecutionResult:
        ...

    @abstractmethod
    async def get_routing_stats(self) -> dict[str, Any]:
        ...


class LangChainV1RouterAdapter(LLMRouterAdapter):
    """
    使用 LangChainExecutor（RouterChain + AgentExecutor）的标准实现：
    - 路由：委托给 LangChainRunner，支持模型指标读取
    - 执行：统一调用 LangChainExecutor.execute
    - Agent：可选工具集直接复用 AgentFactory
    """

    def __init__(
        self,
        models: dict[str, BaseChatModel],
        tools: Sequence[BaseTool] | None = None,
        *,
        runnable_config: dict[str, Any] | None = None,
        agent_type: str = "speckit",
    ):
        self.executor = LangChainExecutor.from_models(
            models,
            agent_tools=tools,
            agent_type=agent_type,
            runnable_config=runnable_config,
        )
        self.tools = list(tools or [])
        logger.info("LangChainV1RouterAdapter initialized with models: %s", list(models.keys()))

    # ------------------------------------------------------------------ #
    # 基础能力
    # ------------------------------------------------------------------ #
    @trace_method
    async def route_request(self, request: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.executor.runner.route_request(request, context=context)

    @trace_method
    async def execute_request(self, request: str, **kwargs) -> ExecutionResult:
        result = await self.executor.execute(request, **kwargs)
        return {
            "model": result["model"],
            "response": result["response"],
            "usage": result.get("usage", {}),
            "execution_time": result.get("execution_time"),
            "success": result.get("success", True),
            "routing_info": result.get("routing_info", {}),
            "error_message": result.get("error"),
        }

    async def execute_with_agent(self, task: str, **kwargs) -> ExecutionResult:
        agent_output = await self.executor.run_agent(task, **kwargs)
        return {
            "model": "agent",
            "response": agent_output,
            "usage": {},
            "execution_time": None,
            "success": True,
            "routing_info": {"mode": "agent"},
            "error_message": None,
        }

    async def get_routing_stats(self) -> dict[str, Any]:
        metrics = self.executor.runner.get_performance_metrics()
        return {
            "total_models": len(metrics),
            "performance_metrics": metrics,
        }


class AdaptiveLLMRouterAdapter(LangChainV1RouterAdapter):
    """在基础能力上增加预算提示与中间件 hook"""

    def __init__(
        self,
        models: dict[str, BaseChatModel],
        tools: Sequence[BaseTool] | None = None,
        *,
        runnable_config: dict[str, Any] | None = None,
    ):
        super().__init__(models, tools, runnable_config=runnable_config, agent_type="speckit")

    async def execute_request(self, request: str, **kwargs) -> ExecutionResult:
        budget_hint = kwargs.pop("budget_hint", None)
        context = {"budget_hint": budget_hint} if budget_hint else None
        metadata = kwargs.setdefault("metadata", {})
        if budget_hint:
            metadata["budget_hint"] = budget_hint

        routing_info = await self.route_request(request, context=context)
        result = await super().execute_request(request, **kwargs)
        result["routing_info"] = routing_info
        return result
