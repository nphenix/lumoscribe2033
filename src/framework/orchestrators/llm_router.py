"""
LLM 路由器

以 LangChainRunner 为核心，提供统一的多模型路由执行入口。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from .langchain_runner import LangChainRunner


class LLMRouter:
    """LangChainRunner 的轻量封装，兼容旧接口"""

    def __init__(self, models: dict[str, BaseChatModel]):
        self.runner = LangChainRunner(models)

    async def route_request(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """返回 LangChainRunner 的路由结果"""
        return await self.runner.route_request(request, context=context)

    async def execute_request(self, request: str, **kwargs) -> dict[str, Any]:
        """执行请求并保持原有返回结构"""
        result = await self.runner.execute_request(request, **kwargs)
        return {
            "model": result["model"],
            "response": result["response"],
            "usage": result.get("usage", {}),
            "routing_info": result.get("routing_info"),
            "execution_time": result.get("execution_time"),
            "success": result.get("success", True),
            "timestamp": kwargs.get("timestamp"),
        }

    async def execute_chain(self, request: str, chain_config: dict[str, Any]) -> dict[str, Any]:
        """透传复杂链式执行能力"""
        return await self.runner.execute_chain(request, chain_config)

    def get_performance_metrics(self) -> dict[str, dict[str, Any]]:
        return self.runner.get_performance_metrics()


class AdaptiveLLMRouter(LLMRouter):
    """保留兼容接口，同时直接复用 LangChainRunner 的性能指标"""

    async def execute_with_adaptation(
        self,
        request: str,
        *,
        budget_hint: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """在执行前注入预算提示等上下文"""
        context = {"budget_hint": budget_hint} if budget_hint else None
        routing_result = await self.route_request(request, context=context)
        kwargs.setdefault("metadata", {}).update({"budget_hint": budget_hint})
        result = await self.runner.execute_request(request, **kwargs)

        # 合并路由决策，方便旧调用方读取
        result["routing_info"] = routing_result
        return {
            "model": result["model"],
            "response": result["response"],
            "usage": result.get("usage", {}),
            "execution_time": result.get("execution_time"),
            "success": result.get("success", True),
            "routing_info": routing_result,
        }
