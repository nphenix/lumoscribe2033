"""
LLM 服务管理器

管理不同类型的 LLM 适配器实例，提供文档分析和代码分析专用接口。
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from ..shared.config import ModelConfig, config_manager
from ..shared.logging import get_logger
from .adapter_factory import AdapterType, get_adapter_factory
from .llm_adapter import (
    KATpro1Adapter,
    KATpro1LangChainAdapter,
    LangChainLLMAdapter,
    LLMAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)

logger = get_logger(__name__)


@dataclass
class LLMServiceConfig:
    """LLM 服务配置"""
    adapter_type: str  # "katpro1", "openai", "ollama"
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 60
    capabilities: list[str] = None


class LLMServiceManager:
    """LLM 服务管理器"""

    def __init__(self):
        self._services: dict[str, LLMAdapter] = {}
        self._configs: dict[str, LLMServiceConfig] = {}
        self._adapter_factory = get_adapter_factory()

        # 初始化默认服务
        self._initialize_default_services()

    def _initialize_default_services(self):
        """初始化默认服务"""
        try:
            # 从配置管理器获取模型配置
            models = config_manager.get_enabled_models()

            for name, model_config in models.items():
                if "katpro1" in name:
                    self._create_katpro1_service(name, model_config)
                elif model_config.provider.value == "openai":
                    self._create_openai_service(name, model_config)
                elif model_config.provider.value == "ollama":
                    self._create_ollama_service(name, model_config)

            logger.info(f"LLM 服务管理器初始化完成，已创建 {len(self._services)} 个服务")

        except Exception as e:
            logger.error(f"初始化默认服务失败: {e}")

    def _create_katpro1_service(self, name: str, model_config: ModelConfig):
        """创建 KATpro1 服务"""
        try:
            api_key = os.getenv(model_config.api_key_env or "KATPRO1_API_KEY", "")
            base_url = model_config.base_url or os.getenv("KATPRO1_BASE_URL", "")
            model = model_config.model_id

            if not api_key or not base_url:
                logger.warning(f"KATpro1 服务 {name} 缺少必要的配置，跳过创建")
                return

            # 创建适配器
            adapter = KATpro1Adapter(
                api_key=api_key,
                base_url=base_url,
                model=model
            )

            # 保存配置
            config = LLMServiceConfig(
                adapter_type="katpro1",
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                capabilities=[cap.value for cap in model_config.capabilities]
            )

            self._services[name] = adapter
            self._configs[name] = config

            logger.info(f"KATpro1 服务创建成功: {name}")

        except Exception as e:
            logger.error(f"创建 KATpro1 服务失败 {name}: {e}")

    def _create_openai_service(self, name: str, model_config: ModelConfig):
        """创建 OpenAI 服务"""
        try:
            api_key = os.getenv(model_config.api_key_env or "OPENAI_API_KEY", "")
            base_url = model_config.base_url or "https://api.openai.com/v1"
            model = model_config.model_id

            if not api_key:
                logger.warning(f"OpenAI 服务 {name} 缺少 API 密钥，跳过创建")
                return

            # 创建适配器
            adapter = OpenAIAdapter(
                api_key=api_key,
                base_url=base_url
            )

            # 保存配置
            config = LLMServiceConfig(
                adapter_type="openai",
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                capabilities=[cap.value for cap in model_config.capabilities]
            )

            self._services[name] = adapter
            self._configs[name] = config

            logger.info(f"OpenAI 服务创建成功: {name}")

        except Exception as e:
            logger.error(f"创建 OpenAI 服务失败 {name}: {e}")

    def _create_ollama_service(self, name: str, model_config: ModelConfig):
        """创建 Ollama 服务"""
        try:
            base_url = model_config.base_url or "http://localhost:11434"
            model = model_config.model_id

            # 创建适配器
            adapter = OllamaAdapter(base_url=base_url)

            # 保存配置
            config = LLMServiceConfig(
                adapter_type="ollama",
                api_key="",
                base_url=base_url,
                model=model,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=model_config.timeout,
                capabilities=[cap.value for cap in model_config.capabilities]
            )

            self._services[name] = adapter
            self._configs[name] = config

            logger.info(f"Ollama 服务创建成功: {name}")

        except Exception as e:
            logger.error(f"创建 Ollama 服务失败 {name}: {e}")

    def get_service(self, name: str) -> LLMAdapter | None:
        """获取服务"""
        return self._services.get(name)

    def get_doc_analysis_service(self) -> LLMAdapter | None:
        """获取文档分析服务"""
        # 优先使用 KATpro1 文档分析服务
        doc_service = self.get_service("katpro1-doc")
        if doc_service:
            return doc_service

        # 其次使用 LangChain 兼容的文档分析服务
        doc_langchain_service = self.get_service("katpro1-doc-langchain")
        if doc_langchain_service:
            return doc_langchain_service

        # 其次使用支持文本处理的其他服务
        for name, service in self._services.items():
            config = self._configs.get(name)
            if config and "text_processing" in config.capabilities:
                return service

        # 返回第一个可用服务
        return next(iter(self._services.values())) if self._services else None

    def get_code_analysis_service(self) -> LLMAdapter | None:
        """获取代码分析服务"""
        # 优先使用 KATpro1 代码分析服务
        code_service = self.get_service("katpro1-code")
        if code_service:
            return code_service

        # 其次使用 LangChain 兼容的代码分析服务
        code_langchain_service = self.get_service("katpro1-code-langchain")
        if code_langchain_service:
            return code_langchain_service

        # 其次使用支持代码分析的其他服务
        for name, service in self._services.items():
            config = self._configs.get(name)
            if config and "code_analysis" in config.capabilities:
                return service

        # 返回第一个可用服务
        return next(iter(self._services.values())) if self._services else None

    def get_langchain_doc_service(self) -> LangChainLLMAdapter | None:
        """获取 LangChain 文档分析服务"""
        service = self.get_service("katpro1-doc-langchain")
        if isinstance(service, LangChainLLMAdapter):
            return service
        return None

    def get_langchain_code_service(self) -> LangChainLLMAdapter | None:
        """获取 LangChain 代码分析服务"""
        service = self.get_service("katpro1-code-langchain")
        if isinstance(service, LangChainLLMAdapter):
            return service
        return None

    def list_services(self) -> list[str]:
        """列出所有服务"""
        return list(self._services.keys())

    def get_service_info(self, name: str) -> dict[str, Any] | None:
        """获取服务信息"""
        adapter = self._services.get(name)
        config = self._configs.get(name)

        if not adapter or not config:
            return None

        return {
            "name": name,
            "adapter_type": config.adapter_type,
            "model": config.model,
            "base_url": config.base_url,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "capabilities": config.capabilities,
            "health_status": "unknown"  # 可以添加健康检查
        }

    def create_custom_service(self, name: str, config: LLMServiceConfig) -> bool:
        """创建自定义服务"""
        try:
            if name in self._services:
                logger.warning(f"服务已存在: {name}")
                return False

            # 根据类型创建适配器
            if config.adapter_type == "katpro1":
                adapter = KATpro1Adapter(
                    api_key=config.api_key,
                    base_url=config.base_url,
                    model=config.model
                )
            elif config.adapter_type == "openai":
                adapter = OpenAIAdapter(
                    api_key=config.api_key,
                    base_url=config.base_url
                )
            elif config.adapter_type == "ollama":
                adapter = OllamaAdapter(base_url=config.base_url)
            else:
                logger.error(f"不支持的适配器类型: {config.adapter_type}")
                return False

            # 保存服务和配置
            self._services[name] = adapter
            self._configs[name] = config

            logger.info(f"自定义服务创建成功: {name}")
            return True

        except Exception as e:
            logger.error(f"创建自定义服务失败 {name}: {e}")
            return False

    def remove_service(self, name: str) -> bool:
        """移除服务"""
        try:
            if name not in self._services:
                logger.warning(f"服务不存在: {name}")
                return False

            # 关闭适配器（如果支持）
            adapter = self._services[name]
            if hasattr(adapter, 'close'):
                import asyncio
                try:
                    asyncio.create_task(adapter.close())
                except Exception:
                    pass  # 忽略关闭错误

            # 移除服务
            del self._services[name]
            if name in self._configs:
                del self._configs[name]

            logger.info(f"服务移除成功: {name}")
            return True

        except Exception as e:
            logger.error(f"移除服务失败 {name}: {e}")
            return False

    def get_available_models(self) -> list[dict[str, Any]]:
        """获取可用模型列表"""
        models = []

        for name, config in self._configs.items():
            models.append({
                "name": name,
                "adapter_type": config.adapter_type,
                "model": config.model,
                "capabilities": config.capabilities,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            })

        return models

    async def health_check(self) -> dict[str, Any]:
        """健康检查"""
        results = {}

        for name, adapter in self._services.items():
            try:
                # 检查适配器是否有健康检查方法
                if hasattr(adapter, 'health_check'):
                    if hasattr(adapter.health_check, '__await__'):
                        result = await adapter.health_check()
                    else:
                        result = adapter.health_check()

                    results[name] = result
                else:
                    results[name] = {"status": "healthy", "message": "No health check method"}

            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}

        return results


# 全局 LLM 服务管理器实例
_llm_service_manager: LLMServiceManager | None = None


def get_llm_service_manager() -> LLMServiceManager:
    """获取全局 LLM 服务管理器"""
    global _llm_service_manager
    if _llm_service_manager is None:
        _llm_service_manager = LLMServiceManager()
    return _llm_service_manager
