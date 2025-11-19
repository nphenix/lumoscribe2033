"""
配置管理单元测试

测试配置管理器的各项功能，包括：
- 配置验证
- 环境变量处理
- 模型配置管理
- 路由配置
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.framework.shared.config import (
    Settings,
    ConfigManager,
    ModelConfig,
    ModelProvider,
    ModelCapability,
    RoutingConfig,
    get_settings,
    get_config_manager
)


class TestSettings:
    """测试设置类"""

    def test_settings_initialization(self):
        """测试设置初始化"""
        settings = Settings()
        
        assert settings.ENVIRONMENT == "development"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "INFO"
        assert settings.OPENAI_API_KEY == "dummy"

    def test_settings_properties(self):
        """测试设置属性"""
        settings = Settings()
        
        # 测试环境属性
        assert settings.is_development is True
        assert settings.is_production is False
        assert settings.is_testing is False

    def test_get_cors_origins(self):
        """测试获取CORS源"""
        settings = Settings()
        origins = settings.get_cors_origins()
        
        assert isinstance(origins, list)
        assert "http://localhost:8080" in origins

    def test_get_database_kwargs(self):
        """测试获取数据库参数"""
        settings = Settings()
        db_kwargs = settings.get_database_kwargs()
        
        assert isinstance(db_kwargs, dict)
        assert "echo" in db_kwargs
        assert "connect_args" in db_kwargs

    def test_get_redis_kwargs(self):
        """测试获取Redis参数"""
        settings = Settings()
        redis_kwargs = settings.get_redis_kwargs()
        
        assert isinstance(redis_kwargs, dict)
        assert "host" in redis_kwargs
        assert "port" in redis_kwargs

    def test_get_llm_kwargs(self):
        """测试获取LLM参数"""
        settings = Settings()
        llm_kwargs = settings.get_llm_kwargs()
        
        assert isinstance(llm_kwargs, dict)
        assert "model" in llm_kwargs
        assert "api_key" in llm_kwargs

    def test_validate_config(self):
        """测试配置验证"""
        settings = Settings()
        errors = settings.validate_config()
        
        assert isinstance(errors, list)

    def test_is_valid(self):
        """测试配置有效性检查"""
        settings = Settings()
        is_valid = settings.is_valid()
        
        assert isinstance(is_valid, bool)

    def test_get_environment_info(self):
        """测试获取环境信息"""
        settings = Settings()
        env_info = settings.get_environment_info()
        
        assert isinstance(env_info, dict)
        assert "environment" in env_info
        assert "python_version" in env_info


class TestModelConfig:
    """测试模型配置"""

    def test_model_config_creation(self):
        """测试模型配置创建"""
        config = ModelConfig(
            name="test-model",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4",
            capabilities=[ModelCapability.GENERAL_CONVERSATION],
            cost_per_token=0.03
        )
        
        assert config.name == "test-model"
        assert config.provider == ModelProvider.OPENAI
        assert config.model_id == "gpt-4"
        assert ModelCapability.GENERAL_CONVERSATION in config.capabilities
        assert config.cost_per_token == 0.03

    def test_model_config_defaults(self):
        """测试模型配置默认值"""
        config = ModelConfig(
            name="test-model",
            provider=ModelProvider.OPENAI,
            model_id="gpt-4"
        )
        
        assert config.cost_per_token == 0.01
        assert config.max_tokens == 4000
        assert config.temperature == 0.1
        assert config.timeout == 30
        assert config.enabled is True


class TestRoutingConfig:
    """测试路由配置"""

    def test_routing_config_defaults(self):
        """测试路由配置默认值"""
        config = RoutingConfig()
        
        assert config.enable_performance_routing is True
        assert config.enable_cost_optimization is True
        assert config.confidence_threshold == 0.7
        assert config.fallback_to_default is True
        assert config.max_retries == 3
        assert config.retry_delay == 1.0


class TestConfigManager:
    """测试配置管理器"""

    def setup_method(self):
        """测试前设置"""
        # 使用延迟初始化避免配置验证问题
        self.config_manager = ConfigManager()

    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        manager = ConfigManager()
        
        assert manager.settings is not None
        assert manager._models is not None
        assert manager._routing is not None
        assert manager._monitoring is not None

    def test_get_model_by_name(self):
        """测试根据名称获取模型"""
        model = self.config_manager.get_model_by_name("openai-gpt4")
        
        if model:
            assert model.name == "openai-gpt4"
            assert model.provider == ModelProvider.OPENAI

    def test_get_enabled_models(self):
        """测试获取启用的模型"""
        enabled_models = self.config_manager.get_enabled_models()
        
        assert isinstance(enabled_models, dict)
        # 应该有默认启用的模型

    def test_enable_disable_model(self):
        """测试启用/禁用模型"""
        # 测试启用模型
        result = self.config_manager.enable_model("openai-gpt4")
        assert result is True
        
        # 测试禁用模型
        result = self.config_manager.disable_model("openai-gpt4")
        assert result is True

    def test_update_model_config(self):
        """测试更新模型配置"""
        result = self.config_manager.update_model_config(
            "openai-gpt4",
            temperature=0.5,
            max_tokens=2000
        )
        assert result is True

    def test_get_routing_config(self):
        """测试获取路由配置"""
        routing = self.config_manager.get_routing_config()
        
        assert isinstance(routing, RoutingConfig)
        assert routing.enable_performance_routing is True

    def test_update_routing_config(self):
        """测试更新路由配置"""
        self.config_manager.update_routing_config(
            confidence_threshold=0.8,
            max_retries=5
        )
        
        routing = self.config_manager.get_routing_config()
        assert routing.confidence_threshold == 0.8
        assert routing.max_retries == 5

    def test_get_setting(self):
        """测试获取配置设置"""
        value = self.config_manager.get_setting("LOG_LEVEL", "DEFAULT")
        assert value == "INFO"
        
        # 测试不存在的设置
        value = self.config_manager.get_setting("NON_EXISTENT", "default_value")
        assert value == "default_value"

    def test_get_models_by_provider(self):
        """测试根据提供商获取模型"""
        openai_models = self.config_manager.get_models_by_provider(ModelProvider.OPENAI)
        
        assert isinstance(openai_models, dict)
        # 应该包含OpenAI模型

    def test_get_models_by_capability(self):
        """测试根据能力获取模型"""
        reasoning_models = self.config_manager.get_models_by_capability(
            ModelCapability.COMPLEX_REASONING
        )
        
        assert isinstance(reasoning_models, dict)

    def test_validate_config(self):
        """测试配置验证"""
        errors = self.config_manager.validate_config()
        
        assert isinstance(errors, list)

    def test_to_dict(self):
        """测试转换为字典"""
        config_dict = self.config_manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "models" in config_dict
        assert "routing" in config_dict
        assert "monitoring" in config_dict

    @patch('src.framework.shared.config.Path')
    def test_generate_env_template(self, mock_path):
        """测试生成环境变量模板"""
        mock_file = MagicMock()
        mock_path.return_value = mock_file
        
        template_content = self.config_manager.generate_env_template()
        
        assert isinstance(template_content, str)
        assert "OPENAI_API_KEY" in template_content
        assert "DATABASE_URL" in template_content
        mock_file.write_text.assert_called_once()


class TestGlobalInstances:
    """测试全局实例"""

    def test_get_settings_singleton(self):
        """测试获取设置单例"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_get_config_manager_singleton(self):
        """测试获取配置管理器单例"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2

    def test_validate_global_config(self):
        """测试验证全局配置"""
        errors = validate_global_config()
        
        assert isinstance(errors, list)


class TestConfigManagerIntegration:
    """测试配置管理器集成"""

    def setup_method(self):
        """测试前设置"""
        self.config_manager = ConfigManager()

    @patch('src.framework.shared.config.Path')
    def test_setup_development_environment(self, mock_path):
        """测试设置开发环境"""
        mock_dir = MagicMock()
        mock_path.return_value = mock_dir
        
        self.config_manager.setup_development_environment()
        
        # 验证目录创建调用
        assert mock_dir.mkdir.call_count > 0

    def test_get_config_status(self):
        """测试获取配置状态"""
        status = self.config_manager.get_config_status()
        
        assert isinstance(status, dict)
        assert "valid" in status
        assert "environment" in status
        assert "validation_errors" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])