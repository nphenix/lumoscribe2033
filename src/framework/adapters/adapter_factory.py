"""
适配器工厂和注册机制

提供统一的适配器创建、注册和管理功能：
- 适配器工厂模式
- 自动注册机制
- 依赖注入支持
- 生命周期管理
- 插件系统支持
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, Union

from ..shared.config import ConfigManager, config_manager
from ..shared.logging import get_logger

logger = get_logger(__name__)


class AdapterType(Enum):
    """适配器类型枚举"""
    CONVERSATION = "conversation"
    LLM_ROUTER = "llm_router"
    CONFIG = "config"
    IDE = "ide"
    LLM = "llm"
    STORAGE = "storage"
    API = "api"


class AdapterLifecycle(Enum):
    """适配器生命周期状态"""
    CREATED = "created"
    INITIALIZED = "initialized"
    STARTED = "started"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AdapterMetadata:
    """适配器元数据"""
    name: str
    type: AdapterType
    version: str
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "capabilities": self.capabilities,
            "config_schema": self.config_schema,
            "tags": self.tags
        }


@dataclass
class AdapterInstance:
    """适配器实例信息"""
    adapter: Any
    metadata: AdapterMetadata
    lifecycle: AdapterLifecycle = AdapterLifecycle.CREATED
    created_at: str = ""
    last_health_check: str = ""
    error_count: int = 0
    performance_metrics: dict[str, Any] = field(default_factory=dict)


class AdapterRegistry:
    """适配器注册表"""

    def __init__(self):
        self._adapters: dict[str, type] = {}
        self._instances: dict[str, AdapterInstance] = {}
        self._metadata: dict[str, AdapterMetadata] = {}
        self._tag_index: dict[str, list[str]] = {}

    def register(self, name: str, adapter_class: type, metadata: AdapterMetadata) -> bool:
        """注册适配器"""
        try:
            # 验证适配器类
            if not self._validate_adapter_class(adapter_class):
                logger.error(f"适配器类验证失败: {name}")
                return False

            self._adapters[name] = adapter_class
            self._metadata[name] = metadata

            # 更新标签索引
            for tag in metadata.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = []
                self._tag_index[tag].append(name)

            logger.info(f"适配器注册成功: {name} (类型: {metadata.type.value})")
            return True

        except Exception as e:
            logger.error(f"注册适配器失败 {name}: {str(e)}")
            return False

    def unregister(self, name: str) -> bool:
        """注销适配器"""
        try:
            if name in self._adapters:
                del self._adapters[name]
            if name in self._metadata:
                del self._metadata[name]
            if name in self._instances:
                del self._instances[name]

            # 更新标签索引
            if name in self._metadata:
                for tag in self._metadata[name].tags:
                    if tag in self._tag_index and name in self._tag_index[tag]:
                        self._tag_index[tag].remove(name)

            logger.info(f"适配器注销成功: {name}")
            return True

        except Exception as e:
            logger.error(f"注销适配器失败 {name}: {str(e)}")
            return False

    def get_adapter_class(self, name: str) -> type | None:
        """获取适配器类"""
        return self._adapters.get(name)

    def get_metadata(self, name: str) -> AdapterMetadata | None:
        """获取适配器元数据"""
        return self._metadata.get(name)

    def get_instance(self, name: str) -> AdapterInstance | None:
        """获取适配器实例"""
        return self._instances.get(name)

    def list_adapters(self, adapter_type: AdapterType | None = None, tag: str | None = None) -> list[str]:
        """列出适配器"""
        adapters = list(self._adapters.keys())

        if adapter_type:
            adapters = [name for name in adapters
                       if self._metadata.get(name, AdapterMetadata("", AdapterType.CONFIG, "")).type == adapter_type]

        if tag:
            adapters = self._tag_index.get(tag, [])

        return adapters

    def get_adapter_by_tag(self, tag: str) -> list[str]:
        """根据标签获取适配器"""
        return self._tag_index.get(tag, [])

    def _validate_adapter_class(self, adapter_class: type) -> bool:
        """验证适配器类"""
        try:
            # 检查是否是类
            if not isinstance(adapter_class, type):
                return False

            # 检查是否有必要的方法
            required_methods = ['__init__']
            for method in required_methods:
                if not hasattr(adapter_class, method):
                    logger.warning(f"适配器类缺少必要方法: {adapter_class.__name__}.{method}")

            return True

        except Exception as e:
            logger.error(f"验证适配器类失败: {str(e)}")
            return False


class AdapterFactory:
    """适配器工厂"""

    def __init__(self, registry: AdapterRegistry | None = None):
        self.registry = registry or AdapterRegistry()
        self.config_manager = config_manager
        self._lifecycle_callbacks: dict[str, list[Callable]] = {
            "pre_create": [],
            "post_create": [],
            "pre_init": [],
            "post_init": [],
            "pre_start": [],
            "post_start": [],
            "pre_stop": [],
            "post_stop": []
        }

        # 注册内置适配器
        self._register_builtin_adapters()

    def register_adapter(self, name: str, adapter_class: type, metadata: AdapterMetadata) -> bool:
        """注册适配器"""
        return self.registry.register(name, adapter_class, metadata)

    def create_adapter(self, name: str, **kwargs) -> Any | None:
        """创建适配器实例"""
        try:
            # 触发创建前回调
            self._trigger_callbacks("pre_create", name, kwargs)

            adapter_class = self.registry.get_adapter_class(name)
            if not adapter_class:
                logger.error(f"适配器未找到: {name}")
                return None

            # 获取配置
            config = self._get_adapter_config(name, kwargs)

            # 创建实例
            adapter = adapter_class(**config)

            # 创建实例信息
            metadata = self.registry.get_metadata(name)
            instance = AdapterInstance(
                adapter=adapter,
                metadata=metadata,
                lifecycle=AdapterLifecycle.CREATED
            )

            self.registry._instances[name] = instance

            # 触发创建后回调
            self._trigger_callbacks("post_create", name, adapter, instance)

            logger.info(f"适配器创建成功: {name}")
            return adapter

        except Exception as e:
            logger.error(f"创建适配器失败 {name}: {str(e)}")
            return None

    async def initialize_adapter(self, name: str) -> bool:
        """初始化适配器"""
        try:
            instance = self.registry.get_instance(name)
            if not instance:
                logger.error(f"适配器实例未找到: {name}")
                return False

            # 触发初始化前回调
            self._trigger_callbacks("pre_init", name, instance)

            adapter = instance.adapter

            # 检查适配器是否有初始化方法
            if hasattr(adapter, 'initialize'):
                if inspect.iscoroutinefunction(adapter.initialize):
                    await adapter.initialize()
                else:
                    adapter.initialize()

            instance.lifecycle = AdapterLifecycle.INITIALIZED

            # 触发初始化后回调
            self._trigger_callbacks("post_init", name, instance)

            logger.info(f"适配器初始化成功: {name}")
            return True

        except Exception as e:
            logger.error(f"初始化适配器失败 {name}: {str(e)}")
            instance.error_count += 1
            return False

    async def start_adapter(self, name: str) -> bool:
        """启动适配器"""
        try:
            instance = self.registry.get_instance(name)
            if not instance:
                logger.error(f"适配器实例未找到: {name}")
                return False

            # 触发启动前回调
            self._trigger_callbacks("pre_start", name, instance)

            adapter = instance.adapter

            # 检查适配器是否有启动方法
            if hasattr(adapter, 'start'):
                if inspect.iscoroutinefunction(adapter.start):
                    await adapter.start()
                else:
                    adapter.start()

            instance.lifecycle = AdapterLifecycle.STARTED
            instance.created_at = instance.created_at or self._get_timestamp()

            # 触发启动后回调
            self._trigger_callbacks("post_start", name, instance)

            logger.info(f"适配器启动成功: {name}")
            return True

        except Exception as e:
            logger.error(f"启动适配器失败 {name}: {str(e)}")
            instance.error_count += 1
            instance.lifecycle = AdapterLifecycle.ERROR
            return False

    async def stop_adapter(self, name: str) -> bool:
        """停止适配器"""
        try:
            instance = self.registry.get_instance(name)
            if not instance:
                logger.warning(f"适配器实例未找到: {name}")
                return True

            # 触发停止前回调
            self._trigger_callbacks("pre_stop", name, instance)

            adapter = instance.adapter

            # 检查适配器是否有停止方法
            if hasattr(adapter, 'stop'):
                if inspect.iscoroutinefunction(adapter.stop):
                    await adapter.stop()
                else:
                    adapter.stop()

            instance.lifecycle = AdapterLifecycle.STOPPED

            # 触发停止后回调
            self._trigger_callbacks("post_stop", name, instance)

            logger.info(f"适配器停止成功: {name}")
            return True

        except Exception as e:
            logger.error(f"停止适配器失败 {name}: {str(e)}")
            return False

    def get_adapter(self, name: str) -> Any | None:
        """获取适配器实例"""
        instance = self.registry.get_instance(name)
        return instance.adapter if instance else None

    def list_adapters(self, adapter_type: AdapterType | None = None, tag: str | None = None) -> list[str]:
        """列出适配器"""
        return self.registry.list_adapters(adapter_type, tag)

    def get_adapter_metadata(self, name: str) -> AdapterMetadata | None:
        """获取适配器元数据"""
        return self.registry.get_metadata(name)

    def register_callback(self, event: str, callback: Callable) -> bool:
        """注册生命周期回调"""
        if event not in self._lifecycle_callbacks:
            logger.error(f"无效的生命周期事件: {event}")
            return False

        self._lifecycle_callbacks[event].append(callback)
        return True

    def unregister_callback(self, event: str, callback: Callable) -> bool:
        """注销生命周期回调"""
        if event in self._lifecycle_callbacks and callback in self._lifecycle_callbacks[event]:
            self._lifecycle_callbacks[event].remove(callback)
            return True
        return False

    async def health_check(self, name: str) -> dict[str, Any]:
        """适配器健康检查"""
        instance = self.registry.get_instance(name)
        if not instance:
            return {"status": "not_found", "error": f"适配器 {name} 未找到"}

        try:
            adapter = instance.adapter
            health_status = "healthy"
            error_info = None

            # 检查适配器是否有健康检查方法
            if hasattr(adapter, 'health_check'):
                if inspect.iscoroutinefunction(adapter.health_check):
                    health_result = await adapter.health_check()
                else:
                    health_result = adapter.health_check()

                if isinstance(health_result, dict):
                    health_status = health_result.get("status", "healthy")
                    error_info = health_result.get("error")
                else:
                    health_status = "healthy" if health_result else "unhealthy"

            # 更新实例信息
            instance.last_health_check = self._get_timestamp()
            instance.performance_metrics.update({
                "health_status": health_status,
                "error_count": instance.error_count,
                "lifecycle": instance.lifecycle.value
            })

            return {
                "status": health_status,
                "lifecycle": instance.lifecycle.value,
                "error_count": instance.error_count,
                "last_health_check": instance.last_health_check,
                "error": error_info
            }

        except Exception as e:
            instance.error_count += 1
            instance.lifecycle = AdapterLifecycle.ERROR
            return {
                "status": "error",
                "error": str(e),
                "error_count": instance.error_count
            }

    @asynccontextmanager
    async def adapter_context(self, name: str):
        """适配器上下文管理器"""
        adapter = self.get_adapter(name)
        if not adapter:
            raise ValueError(f"适配器未找到: {name}")

        try:
            yield adapter
        except Exception as e:
            logger.error(f"适配器上下文中发生错误 {name}: {str(e)}")
            raise

    def _register_builtin_adapters(self):
        """注册内置适配器"""
        # 这里可以注册系统内置的适配器
        # 实际项目中可以通过配置文件或插件系统动态注册
        pass

    def _get_adapter_config(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """获取适配器配置"""
        config = {}

        # 从配置管理器获取配置
        adapter_config = self.config_manager.get(f"adapters.{name}", {})
        config.update(adapter_config)

        # 从参数获取配置
        config.update(kwargs)

        return config

    def _trigger_callbacks(self, event: str, *args):
        """触发回调"""
        for callback in self._lifecycle_callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"回调执行失败 {event}: {str(e)}")

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# 全局适配器工厂实例
_adapter_factory: AdapterFactory | None = None


def get_adapter_factory() -> AdapterFactory:
    """获取全局适配器工厂"""
    global _adapter_factory
    if _adapter_factory is None:
        _adapter_factory = AdapterFactory()
    return _adapter_factory


def set_adapter_factory(factory: AdapterFactory):
    """设置全局适配器工厂"""
    global _adapter_factory
    _adapter_factory = factory


async def init_adapter_factory() -> AdapterFactory:
    """初始化适配器工厂"""
    factory = get_adapter_factory()

    # 注册内置适配器
    from .config_adapter import (
        ConfigAdapter,
        EnvironmentConfigAdapter,
        FastAPIConfigAdapter,
        FileConfigAdapter,
    )
    from .conversation_adapter import (
        ConversationAdapter,
        CursorConversationAdapter,
        RooCodeConversationAdapter,
    )
    from .ide_adapter import IDEAdapter
    from .llm_adapter import (
        KATpro1Adapter,
        KATpro1LangChainAdapter,
        LangChainLLMAdapter,
        LLMAdapter,
        OllamaAdapter,
        OpenAIAdapter,
    )
    from .llm_router_adapter import (
        AdaptiveLLMRouterAdapter,
        LangChainV1RouterAdapter,
        LLMRouterAdapter,
    )

    # 注册对话适配器
    factory.register_adapter(
        "cursor_conversation",
        CursorConversationAdapter,
        AdapterMetadata(
            name="cursor_conversation",
            type=AdapterType.CONVERSATION,
            version="1.0.0",
            description="Cursor IDE 对话日志解析适配器",
            capabilities=["parse_cursor_logs", "export_json", "export_csv"],
            tags=["cursor", "conversation", "logs"]
        )
    )

    factory.register_adapter(
        "roocode_conversation",
        RooCodeConversationAdapter,
        AdapterMetadata(
            name="roocode_conversation",
            type=AdapterType.CONVERSATION,
            version="1.0.0",
            description="RooCode IDE 对话日志解析适配器",
            capabilities=["parse_roocode_logs", "export_json", "export_csv"],
            tags=["roocode", "conversation", "logs"]
        )
    )

    # 注册 LLM 路由器适配器
    factory.register_adapter(
        "langchain_v1_router",
        LangChainV1RouterAdapter,
        AdapterMetadata(
            name="langchain_v1_router",
            type=AdapterType.LLM_ROUTER,
            version="1.0.0",
            description="LangChain v1.0 路由器适配器",
            capabilities=["route_llm_requests", "performance_monitoring", "cost_tracking"],
            tags=["langchain", "router", "llm"]
        )
    )

    factory.register_adapter(
        "adaptive_llm_router",
        AdaptiveLLMRouterAdapter,
        AdapterMetadata(
            name="adaptive_llm_router",
            type=AdapterType.LLM_ROUTER,
            version="1.0.0",
            description="自适应 LLM 路由器适配器",
            capabilities=["adaptive_routing", "load_balancing", "cost_optimization"],
            tags=["adaptive", "router", "llm"]
        )
    )

    # 注册配置适配器
    factory.register_adapter(
        "file_config",
        FileConfigAdapter,
        AdapterMetadata(
            name="file_config",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="文件配置适配器",
            capabilities=["load_yaml", "load_json", "file_watching", "config_validation"],
            tags=["file", "config", "yaml", "json"]
        )
    )

    factory.register_adapter(
        "fastapi_config",
        FastAPIConfigAdapter,
        AdapterMetadata(
            name="fastapi_config",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="FastAPI 配置适配器",
            capabilities=["api_config_management", "config_validation", "schema_generation"],
            tags=["fastapi", "config", "api"]
        )
    )

    factory.register_adapter(
        "environment_config",
        EnvironmentConfigAdapter,
        AdapterMetadata(
            name="environment_config",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="环境变量配置适配器",
            capabilities=["env_var_mapping", "config_export"],
            tags=["environment", "config", "env"]
        )
    )

    # 注册 LangChain 兼容的 LLM 适配器
    factory.register_adapter(
        "katpro1_langchain",
        KATpro1LangChainAdapter,
        AdapterMetadata(
            name="katpro1_langchain",
            type=AdapterType.LLM,
            version="1.0.0",
            description="KATpro1 LangChain 1.0 兼容适配器",
            capabilities=["chat_completion", "embeddings", "langchain_compatible"],
            tags=["katpro1", "langchain", "llm", "api", "custom"]
        )
    )

    logger.info("适配器工厂初始化完成，已注册内置适配器")
    return factory
