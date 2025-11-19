"""
LangChainExecutor

统一的 LangChain 1.0 执行入口：负责
- 构造 / 复用 RouterChain（LangChainRunner）
- 构造 AgentExecutor（通过 AgentFactory）
- 传递 RunnableConfig（用于追踪、标签、metadata）
- 暴露 execute / stream / batch 等方法，供 API、CLI、Worker 复用
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import (
    Any,
    Optional,
)

from langchain_classic.agents import AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .agent_factory import AgentFactory
from .langchain_runner import LangChainRunner


class LangChainExecutor:
    """集中管理 RouterChain / AgentExecutor / RunnableConfig 的执行器"""

    def __init__(
        self,
        runner: LangChainRunner,
        *,
        agent_executor: AgentExecutor | None = None,
        base_config: RunnableConfig | None = None,
    ):
        self.runner = runner
        self.agent_executor = agent_executor
        self.base_config = base_config

    # ------------------------------------------------------------------ #
    # 工厂方法
    # ------------------------------------------------------------------ #
    @classmethod
    def from_models(
        cls,
        models: dict[str, BaseChatModel],
        *,
        agent_tools: Sequence[BaseTool] | None = None,
        agent_type: str = "speckit",
        runnable_config: RunnableConfig | None = None,
        agent_system_prompt: str | None = None,
    ) -> LangChainExecutor:
        runner = LangChainRunner(models)
        agent_executor = None
        primary_llm = next(iter(models.values())) if models else None

        if agent_tools and primary_llm:
            agent_executor = cls._build_agent(primary_llm, agent_tools, agent_type, agent_system_prompt)

        return cls(runner, agent_executor=agent_executor, base_config=runnable_config)

    @staticmethod
    def _build_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        agent_type: str,
        system_prompt: str | None,
    ) -> AgentExecutor:
        """根据 agent_type 调用 AgentFactory 的具体构造函数"""
        agent_type = agent_type.lower()
        if agent_type == "doc_review":
            return AgentFactory.create_doc_review_agent(llm, tools, system_prompt=system_prompt)
        if agent_type == "compliance":
            return AgentFactory.create_compliance_agent(llm, tools, system_prompt=system_prompt)
        # 默认 speckit
        return AgentFactory.create_speckit_agent(llm, tools, system_prompt=system_prompt)

    # ------------------------------------------------------------------ #
    # RunnableConfig 相关
    # ------------------------------------------------------------------ #
    def with_config(self, config: RunnableConfig) -> LangChainExecutor:
        """返回挂载新配置的执行器实例"""
        merged = self._merge_config(self.base_config, config)
        return LangChainExecutor(self.runner, agent_executor=self.agent_executor, base_config=merged)

    def _resolve_config(self, overrides: RunnableConfig | None) -> RunnableConfig | None:
        if overrides and self.base_config:
            return self._merge_config(self.base_config, overrides)
        return overrides or self.base_config

    @staticmethod
    def _merge_config(
        base: RunnableConfig | None,
        override: RunnableConfig | None,
    ) -> RunnableConfig | None:
        if not base:
            return override
        if not override:
            return base
        merged = dict(base)
        merged.update(override)
        return merged  # RunnableConfig 本质为 TypedDict，可直接返回

    # ------------------------------------------------------------------ #
    # Router 执行
    # ------------------------------------------------------------------ #
    async def execute(
        self,
        request: str,
        *,
        runnable_config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = self._resolve_config(runnable_config)
        if config:
            kwargs.setdefault("config", config)
        return await self.runner.execute_request(request, **kwargs)

    async def stream(
        self,
        request: str,
        *,
        runnable_config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """流式返回选中模型的输出"""
        routing = await self.runner.route_request(request)
        model_name = routing["model_name"]
        model = self.runner.models[model_name]

        config = self._resolve_config(runnable_config)
        message = HumanMessage(content=request)
        stream = model.astream([message], config=config, **kwargs)
        async for chunk in stream:
            yield getattr(chunk, "content", str(chunk))

    async def batch(
        self,
        requests: Iterable[str],
        *,
        runnable_config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """串行执行多个请求；若后续需要可替换为真正的 batch"""
        results: list[dict[str, Any]] = []
        for item in requests:
            results.append(await self.execute(item, runnable_config=runnable_config, **kwargs))
        return results

    # ------------------------------------------------------------------ #
    # Agent 执行
    # ------------------------------------------------------------------ #
    async def run_agent(
        self,
        task: str,
        *,
        input_key: str = "input",
        runnable_config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.agent_executor:
            raise RuntimeError("未配置 AgentExecutor，无法执行 Agent 任务")

        config = self._resolve_config(runnable_config)
        payload = {input_key: task}
        return await self.agent_executor.ainvoke(payload, config=config, **kwargs)

    # ------------------------------------------------------------------ #
    # 同步包装（供阻塞环境使用）
    # ------------------------------------------------------------------ #
    def sync_execute(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return asyncio.run(self.execute(request, **kwargs))

    def sync_run_agent(self, task: str, **kwargs: Any) -> Any:
        return asyncio.run(self.run_agent(task, **kwargs))


# ----------------------------------------------------------------------
# 全局执行器注册（供 API / Worker / CLI 共享）
# ----------------------------------------------------------------------
_global_executor: LangChainExecutor | None = None


def set_global_executor(executor: LangChainExecutor) -> None:
    global _global_executor
    _global_executor = executor


def init_global_executor(
    models: dict[str, BaseChatModel],
    *,
    agent_tools: Sequence[BaseTool] | None = None,
    agent_type: str = "speckit",
    runnable_config: RunnableConfig | None = None,
) -> LangChainExecutor:
    executor = LangChainExecutor.from_models(
        models,
        agent_tools=agent_tools,
        agent_type=agent_type,
        runnable_config=runnable_config,
    )
    set_global_executor(executor)
    return executor


def get_global_executor() -> LangChainExecutor:
    if _global_executor is None:
        raise RuntimeError("LangChainExecutor 尚未初始化，请先调用 init_global_executor")
    return _global_executor


def get_executor_with_config(config: RunnableConfig | None) -> LangChainExecutor:
    executor = get_global_executor()
    return executor.with_config(config) if config else executor

