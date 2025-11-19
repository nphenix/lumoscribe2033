"""
适配器工厂集成测试

测试适配器工厂的完整功能，包括：
- 适配器注册和创建
- 生命周期管理
- 配置管理
- 健康检查
- 回调机制
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from src.framework.adapters import (
    AdapterFactory,
    AdapterLifecycle,
    AdapterMetadata,
    AdapterRegistry,
    AdapterType,
    get_adapter_factory,
    init_adapter_factory,
)
from src.framework.adapters.config_adapter import (
    EnvironmentConfigAdapter,
    FastAPIConfigAdapter,
    FileConfigAdapter,
)
from src.framework.adapters.conversation_adapter import (
    CursorConversationAdapter,
    RooCodeConversationAdapter,
)
from src.framework.adapters.llm_router_adapter import (
    AdaptiveLLMRouterAdapter,
    LangChainV1RouterAdapter,
)
from src.framework.shared.config import ConfigManager


class TestAdapterFactoryIntegration:
    """适配器工厂集成测试"""

    @pytest.fixture
    async def adapter_factory(self):
        """创建适配器工厂实例"""
        factory = AdapterFactory()
        await init_adapter_factory()
        yield factory

    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_config(self):
        """示例配置数据"""
        return {
            "llm": {
                "models": {
                    "openai-gpt4": {
                        "provider": "openai",
                        "model_id": "gpt-4",
                        "api_key_env": "OPENAI_API_KEY",
                        "capabilities": ["chat", "completion"]
                    }
                }
            },
            "database": {
                "url": "sqlite:///test.db",
                "echo": False
            },
            "api": {
                "port": 8080,
                "cors_origins": ["http://localhost:3000"]
            }
        }

    async def test_adapter_registration(self, adapter_factory):
        """测试适配器注册功能"""
        # 创建测试适配器元数据
        metadata = AdapterMetadata(
            name="test_adapter",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="测试适配器",
            capabilities=["test"],
            tags=["test"]
        )

        # 注册适配器
        result = adapter_factory.register_adapter(
            "test_adapter",
            FileConfigAdapter,
            metadata
        )

        assert result is True
        assert "test_adapter" in adapter_factory.registry.list_adapters()

        # 验证元数据
        registered_metadata = adapter_factory.registry.get_metadata("test_adapter")
        assert registered_metadata.name == "test_adapter"
        assert registered_metadata.type == AdapterType.CONFIG
        assert registered_metadata.version == "1.0.0"

    async def test_adapter_creation_and_lifecycle(self, adapter_factory, temp_config_dir):
        """测试适配器创建和生命周期管理"""
        # 注册配置适配器
        metadata = AdapterMetadata(
            name="file_config_test",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="文件配置适配器测试",
            capabilities=["load", "save"],
            tags=["file", "config"]
        )

        adapter_factory.register_adapter(
            "file_config_test",
            FileConfigAdapter,
            metadata
        )

        # 创建适配器
        adapter = adapter_factory.create_adapter(
            "file_config_test",
            config_dir=temp_config_dir
        )

        assert adapter is not None
        assert isinstance(adapter, FileConfigAdapter)

        # 验证实例信息
        instance = adapter_factory.registry.get_instance("file_config_test")
        assert instance is not None
        assert instance.lifecycle == AdapterLifecycle.CREATED

        # 初始化适配器
        result = await adapter_factory.initialize_adapter("file_config_test")
        assert result is True
        assert instance.lifecycle == AdapterLifecycle.INITIALIZED

        # 启动适配器
        result = await adapter_factory.start_adapter("file_config_test")
        assert result is True
        assert instance.lifecycle == AdapterLifecycle.STARTED

    async def test_adapter_health_check(self, adapter_factory, temp_config_dir):
        """测试适配器健康检查"""
        # 注册并创建适配器
        metadata = AdapterMetadata(
            name="health_test_adapter",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="健康检查测试适配器",
            capabilities=["health_check"],
            tags=["test"]
        )

        adapter_factory.register_adapter(
            "health_test_adapter",
            FileConfigAdapter,
            metadata
        )

        adapter_factory.create_adapter(
            "health_test_adapter",
            config_dir=temp_config_dir
        )

        # 执行健康检查
        health_result = await adapter_factory.health_check("health_test_adapter")

        assert "status" in health_result
        assert health_result["status"] in ["healthy", "not_found", "error"]
        assert "lifecycle" in health_result
        assert "error_count" in health_result

    async def test_adapter_context_manager(self, adapter_factory, temp_config_dir):
        """测试适配器上下文管理器"""
        # 注册适配器
        metadata = AdapterMetadata(
            name="context_test_adapter",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="上下文管理器测试适配器",
            capabilities=["context"],
            tags=["test"]
        )

        adapter_factory.register_adapter(
            "context_test_adapter",
            FileConfigAdapter,
            metadata
        )

        adapter = adapter_factory.create_adapter(
            "context_test_adapter",
            config_dir=temp_config_dir
        )

        # 测试上下文管理器
        async with adapter_factory.adapter_context("context_test_adapter") as ctx_adapter:
            assert ctx_adapter is adapter
            assert isinstance(ctx_adapter, FileConfigAdapter)

    async def test_callback_mechanism(self, adapter_factory):
        """测试回调机制"""
        # 创建回调函数
        callback_calls = []

        def test_callback(*args):
            callback_calls.append(args)

        # 注册回调
        result = adapter_factory.register_callback("post_create", test_callback)
        assert result is True

        # 创建适配器触发回调
        metadata = AdapterMetadata(
            name="callback_test_adapter",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="回调测试适配器",
            capabilities=["callback"],
            tags=["test"]
        )

        adapter_factory.register_adapter(
            "callback_test_adapter",
            FileConfigAdapter,
            metadata
        )

        adapter = adapter_factory.create_adapter("callback_test_adapter")
        assert adapter is not None

        # 验证回调被调用
        assert len(callback_calls) > 0
        assert "callback_test_adapter" in str(callback_calls)

    async def test_config_adapter_integration(self, adapter_factory, temp_config_dir, sample_config):
        """测试配置适配器集成"""
        # 注册文件配置适配器
        metadata = AdapterMetadata(
            name="config_integration_test",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="配置集成测试适配器",
            capabilities=["load", "save", "validate"],
            tags=["config", "integration"]
        )

        adapter_factory.register_adapter(
            "config_integration_test",
            FileConfigAdapter,
            metadata
        )

        # 创建适配器
        adapter = adapter_factory.create_adapter(
            "config_integration_test",
            config_dir=temp_config_dir
        )

        # 保存配置
        config_file = "test_config.yaml"
        save_result = await adapter.save_config(config_file, sample_config)
        assert save_result is True

        # 加载配置
        loaded_config = await adapter.load_config(config_file)
        assert loaded_config == sample_config

        # 验证配置
        errors = await adapter.validate_config(sample_config)
        assert len(errors) == 0

    async def test_conversation_adapter_integration(self, adapter_factory, temp_config_dir):
        """测试对话适配器集成"""
        # 注册 Cursor 对话适配器
        metadata = AdapterMetadata(
            name="cursor_integration_test",
            type=AdapterType.CONVERSATION,
            version="1.0.0",
            description="Cursor 对话集成测试适配器",
            capabilities=["parse", "export"],
            tags=["cursor", "conversation"]
        )

        adapter_factory.register_adapter(
            "cursor_integration_test",
            CursorConversationAdapter,
            metadata
        )

        # 创建适配器
        adapter = adapter_factory.create_adapter("cursor_integration_test")
        assert adapter is not None
        assert isinstance(adapter, CursorConversationAdapter)

        # 测试对话解析功能
        sample_log = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there!
"""

        # 测试解析
        result = await adapter.parse_conversation(sample_log)
        assert result is not None
        assert len(result.messages) >= 1

    async def test_llm_router_adapter_integration(self, adapter_factory):
        """测试 LLM 路由器适配器集成"""
        # 注册 LangChain 路由器适配器
        metadata = AdapterMetadata(
            name="router_integration_test",
            type=AdapterType.LLM_ROUTER,
            version="1.0.0",
            description="LLM 路由器集成测试适配器",
            capabilities=["route", "monitor"],
            tags=["router", "llm"]
        )

        adapter_factory.register_adapter(
            "router_integration_test",
            LangChainV1RouterAdapter,
            metadata
        )

        # 创建适配器
        adapter = adapter_factory.create_adapter("router_integration_test")
        assert adapter is not None
        assert isinstance(adapter, LangChainV1RouterAdapter)

        # 测试路由功能
        # 注意：这里只是测试适配器创建，实际路由需要完整的 LLM 配置

    async def test_adapter_listing_and_filtering(self, adapter_factory):
        """测试适配器列表和过滤"""
        # 注册多个适配器用于测试
        adapters_to_register = [
            ("test_config_1", FileConfigAdapter, AdapterType.CONFIG, ["config", "test"]),
            ("test_config_2", FileConfigAdapter, AdapterType.CONFIG, ["config", "test"]),
            ("test_conversation", CursorConversationAdapter, AdapterType.CONVERSATION, ["conversation", "test"]),
            ("test_router", LangChainV1RouterAdapter, AdapterType.LLM_ROUTER, ["router", "test"])
        ]

        for name, adapter_class, adapter_type, tags in adapters_to_register:
            metadata = AdapterMetadata(
                name=name,
                type=adapter_type,
                version="1.0.0",
                description=f"测试 {adapter_type.value} 适配器",
                capabilities=["test"],
                tags=tags
            )

            adapter_factory.register_adapter(name, adapter_class, metadata)

        # 测试按类型过滤
        config_adapters = adapter_factory.list_adapters(adapter_type=AdapterType.CONFIG)
        assert "test_config_1" in config_adapters
        assert "test_config_2" in config_adapters
        assert "test_conversation" not in config_adapters

        # 测试按标签过滤
        test_adapters = adapter_factory.list_adapters(tag="test")
        for name, _, _, _ in adapters_to_register:
            assert name in test_adapters

        # 测试标签索引
        config_tag_adapters = adapter_factory.registry.get_adapter_by_tag("config")
        assert "test_config_1" in config_tag_adapters
        assert "test_config_2" in config_tag_adapters

    async def test_adapter_unregistration(self, adapter_factory):
        """测试适配器注销"""
        # 注册适配器
        metadata = AdapterMetadata(
            name="unregister_test",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="注销测试适配器",
            capabilities=["unregister"],
            tags=["test"]
        )

        adapter_factory.register_adapter("unregister_test", FileConfigAdapter, metadata)
        assert "unregister_test" in adapter_factory.registry.list_adapters()

        # 注销适配器
        result = adapter_factory.registry.unregister("unregister_test")
        assert result is True

        assert "unregister_test" not in adapter_factory.registry.list_adapters()
        assert adapter_factory.registry.get_adapter_class("unregister_test") is None
        assert adapter_factory.registry.get_metadata("unregister_test") is None

    async def test_error_handling(self, adapter_factory):
        """测试错误处理"""
        # 测试创建不存在的适配器
        adapter = adapter_factory.create_adapter("nonexistent_adapter")
        assert adapter is None

        # 测试操作不存在的适配器
        result = await adapter_factory.initialize_adapter("nonexistent_adapter")
        assert result is False

        result = await adapter_factory.start_adapter("nonexistent_adapter")
        assert result is False

        health_result = await adapter_factory.health_check("nonexistent_adapter")
        assert health_result["status"] == "not_found"

    async def test_performance_metrics_tracking(self, adapter_factory, temp_config_dir):
        """测试性能指标跟踪"""
        # 注册适配器
        metadata = AdapterMetadata(
            name="metrics_test",
            type=AdapterType.CONFIG,
            version="1.0.0",
            description="性能指标测试适配器",
            capabilities=["metrics"],
            tags=["test"]
        )

        adapter_factory.register_adapter("metrics_test", FileConfigAdapter, metadata)

        adapter_factory.create_adapter(
            "metrics_test",
            config_dir=temp_config_dir
        )

        # 执行多次健康检查
        for _ in range(3):
            await adapter_factory.health_check("metrics_test")

        # 验证性能指标
        instance = adapter_factory.registry.get_instance("metrics_test")
        assert "health_status" in instance.performance_metrics
        assert "error_count" in instance.performance_metrics
        assert "lifecycle" in instance.performance_metrics

    async def test_global_factory_instance(self):
        """测试全局工厂实例"""
        # 获取全局实例
        factory1 = get_adapter_factory()
        factory2 = get_adapter_factory()

        # 应该是同一个实例
        assert factory1 is factory2

        # 测试设置自定义实例
        custom_factory = AdapterFactory()
        from src.framework.adapters.adapter_factory import set_adapter_factory
        set_adapter_factory(custom_factory)

        factory3 = get_adapter_factory()
        assert factory3 is custom_factory

    async def test_builtin_adapters_registration(self, adapter_factory):
        """测试内置适配器注册"""
        # 验证内置适配器是否已注册
        builtin_adapters = [
            "cursor_conversation",
            "roocode_conversation",
            "langchain_v1_router",
            "adaptive_llm_router",
            "file_config",
            "fastapi_config",
            "environment_config"
        ]

        for adapter_name in builtin_adapters:
            assert adapter_name in adapter_factory.registry.list_adapters()

            metadata = adapter_factory.registry.get_metadata(adapter_name)
            assert metadata is not None
            assert metadata.name == adapter_name
            assert metadata.version == "1.0.0"
            assert len(metadata.capabilities) > 0


if __name__ == "__main__":
    # 运行集成测试
    pytest.main([__file__, "-v"])
