"""
orchestrators/ - LangChain 1.0 运行器和 LLM router

负责：
- LangChain 1.0 代理创建和管理
- 多模型路由策略
- 中间件配置
- 执行上下文管理

基于 LangChain 1.0 最佳实践：
- 使用 create_agent 简化代理创建
- 支持结构化输出
- 统一内容块访问
- 中间件模式支持
"""

from .agent_factory import AgentFactory
from .langchain_executor import LangChainExecutor
from .llm_router import LLMRouter
from .middleware_manager import MiddlewareManager
from .model_registry import bootstrap_langchain_executor, build_default_models

__all__ = [
    "AgentFactory",
    "LangChainExecutor",
    "LLMRouter",
    "MiddlewareManager",
    "bootstrap_langchain_executor",
    "build_default_models",
]
