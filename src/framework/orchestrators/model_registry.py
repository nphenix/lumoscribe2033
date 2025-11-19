"""
模型注册与 LangChainExecutor 启动辅助工具

提供：
- 基于 Settings 的默认模型构建（OpenAI + Ollama）
- FakeListChatModel 兜底，确保系统在无密钥/无服务情况下依旧可用
- 一键初始化全局 LangChainExecutor，供 API / Worker / CLI 公用
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

try:  # 依赖存在时才导入，便于离线测试
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:  # pragma: no cover
    ChatOllama = None  # type: ignore

from src.framework.shared.config import Settings
from src.framework.shared.logging import get_logger

from .langchain_executor import LangChainExecutor, init_global_executor

logger = get_logger(__name__)


def build_default_models(settings: Settings | None = None) -> dict[str, BaseChatModel]:
    """根据当前配置构建默认模型集合"""
    settings = settings or Settings()
    models: dict[str, BaseChatModel] = {}

    if ChatOpenAI is not None:
        try:
            models["openai-primary"] = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_API_BASE,
                temperature=0.1,
                timeout=60,
            )
            logger.info("✅ 已加载 OpenAI 模型: %s", settings.OPENAI_MODEL)
        except Exception as exc:  # pragma: no cover
            logger.warning("⚠️ 初始化 OpenAI 模型失败: %s", exc)

    if ChatOllama is not None:
        try:
            models["ollama-primary"] = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_HOST,
            )
            logger.info("✅ 已加载 Ollama 模型: %s", settings.OLLAMA_MODEL)
        except Exception as exc:  # pragma: no cover
            logger.warning("⚠️ 初始化 Ollama 模型失败: %s", exc)

    if not models:
        logger.warning("⚠️ 未能加载真实模型，使用 GenericFakeChatModel 作为兜底。")
        models["mock-llm"] = GenericFakeChatModel(responses=iter(["目前未配置真实 LLM，返回占位答案。"]))

    return models


def bootstrap_langchain_executor(
    *,
    settings: Settings | None = None,
    tools: Sequence | None = None,
    agent_type: str = "speckit",
) -> LangChainExecutor:
    """
    构建默认模型并初始化全局 LangChainExecutor。

    Args:
        settings: 可选 Settings 实例
        tools: Agent 所需工具
        agent_type: Agent 类型（speckit / doc_review / compliance）
    """
    models = build_default_models(settings)
    executor = init_global_executor(
        models,
        agent_tools=list(tools) if tools else None,
        agent_type=agent_type,
    )
    logger.info("✅ LangChainExecutor 已初始化，模型数量: %d", len(models))
    return executor


__all__ = ["build_default_models", "bootstrap_langchain_executor"]

