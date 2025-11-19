"""
adapters/ - 接口适配器

提供统一的接口抽象，支持：
- IDE 工具适配 (Cursor, RooCode)
- 对话存储适配
- LLM 服务适配
- 配置管理适配
- 外部服务集成

适配器模式：
- 统一接口定义
- 具体实现分离
- 易于扩展和替换
- 支持多租户
"""

from .adapter_factory import (
    AdapterFactory,
    AdapterInstance,
    AdapterLifecycle,
    AdapterMetadata,
    AdapterRegistry,
    AdapterType,
    get_adapter_factory,
    init_adapter_factory,
    set_adapter_factory,
)
from .config_adapter import (
    ConfigAdapter,
    ConfigAdapterFactory,
    EnvironmentConfigAdapter,
    FastAPIConfigAdapter,
    FileConfigAdapter,
    get_config_adapter,
    init_config_adapter,
    set_config_adapter,
)
from .conversation_adapter import (
    ConversationAdapter,
    CursorConversationAdapter,
    RooCodeConversationAdapter,
)
from .ide_adapter import IDEAdapter
from .llm_adapter import LLMAdapter
from .llm_router_adapter import (
    AdaptiveLLMRouterAdapter,
    LangChainV1RouterAdapter,
    LLMRouterAdapter,
)

__all__ = [
    # IDE 适配器
    "IDEAdapter",

    # 对话适配器
    "ConversationAdapter",
    "CursorConversationAdapter",
    "RooCodeConversationAdapter",

    # LLM 适配器
    "LLMAdapter",

    # LLM 路由器适配器
    "LLMRouterAdapter",
    "LangChainV1RouterAdapter",
    "AdaptiveLLMRouterAdapter",

    # 配置管理适配器
    "ConfigAdapter",
    "FileConfigAdapter",
    "FastAPIConfigAdapter",
    "EnvironmentConfigAdapter",
    "ConfigAdapterFactory",
    "get_config_adapter",
    "set_config_adapter",
    "init_config_adapter",

    # 适配器工厂和注册机制
    "AdapterFactory",
    "AdapterRegistry",
    "AdapterMetadata",
    "AdapterInstance",
    "AdapterType",
    "AdapterLifecycle",
    "get_adapter_factory",
    "set_adapter_factory",
    "init_adapter_factory"
]
