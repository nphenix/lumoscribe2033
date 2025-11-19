"""
配置适配器单元测试

测试配置适配器的各种功能，包括：
- 文件配置适配器
- FastAPI 配置适配器
- 环境变量配置适配器
- 配置验证
- 配置热更新
"""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from src.framework.adapters.config_adapter import (
    ConfigAdapter,
    ConfigAdapterFactory,
    EnvironmentConfigAdapter,
    FastAPIConfigAdapter,
    FileConfigAdapter,
    get_config_adapter,
    init_config_adapter,
    set_config_adapter,
)


class TestFileConfigAdapter:
    """文件配置适配器单元测试"""

    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def file_adapter(self, temp_config_dir):
        """创建文件配置适配器实例"""
        return FileConfigAdapter(temp_config_dir)

    @pytest.fixture
    def sample_config_data(self):
        """示例配置数据"""
        return {
            "llm": {
                "models": {
                    "openai-gpt4": {
                        "provider": "openai",
                        "model_id": "gpt-4",
                        "api_key_env": "OPENAI_API_KEY",
                        "capabilities": ["chat", "completion"],
                        "cost_per_token": 0.00003,
                        "enabled": True
                    },
                    "anthropic-claude": {
                        "provider": "anthropic",
                        "model_id": "claude-3-sonnet",
                        "api_key_env": "ANTHROPIC_API_KEY",
                        "capabilities": ["chat"],
                        "cost_per_token": 0.00002,
                        "enabled": True
                    }
                }
            },
            "database": {
                "url": "sqlite:///test.db",
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080,
                "cors_origins": ["http://localhost:3000", "https://localhost:3000"],
                "debug": False
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/app.log"
            }
        }

    def test_yaml_config_loading(self, file_adapter, temp_config_dir, sample_config_data):
        """测试 YAML 配置加载"""
        config_file = Path(temp_config_dir) / "test_config.yaml"

        # 保存 YAML 配置
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config_data, f, default_flow_style=False, allow_unicode=True)

        # 加载配置
        import asyncio
        loaded_config = asyncio.run(file_adapter.load_config("test_config.yaml"))

        assert loaded_config == sample_config_data
        assert loaded_config["llm"]["models"]["openai-gpt4"]["provider"] == "openai"
        assert loaded_config["database"]["url"] == "sqlite:///test.db"
        assert loaded_config["api"]["port"] == 8080

    def test_json_config_loading(self, file_adapter, temp_config_dir, sample_config_data):
        """测试 JSON 配置加载"""
        config_file = Path(temp_config_dir) / "test_config.json"

        # 保存 JSON 配置
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config_data, f, indent=2, ensure_ascii=False)

        # 加载配置
        import asyncio
        loaded_config = asyncio.run(file_adapter.load_config("test_config.json"))

        assert loaded_config == sample_config_data

    def test_config_saving_yaml(self, file_adapter, temp_config_dir, sample_config_data):
        """测试 YAML 配置保存"""
        config_file = "save_test.yaml"

        import asyncio
        result = asyncio.run(file_adapter.save_config(config_file, sample_config_data))

        assert result is True

        # 验证文件是否被正确保存
        saved_file = Path(temp_config_dir) / config_file
        assert saved_file.exists()

        # 重新加载验证
        loaded_config = asyncio.run(file_adapter.load_config(config_file))
        assert loaded_config == sample_config_data

    def test_config_saving_json(self, file_adapter, temp_config_dir, sample_config_data):
        """测试 JSON 配置保存"""
        config_file = "save_test.json"

        import asyncio
        result = asyncio.run(file_adapter.save_config(config_file, sample_config_data))

        assert result is True

        # 验证文件是否被正确保存
        saved_file = Path(temp_config_dir) / config_file
        assert saved_file.exists()

        # 重新加载验证
        loaded_config = asyncio.run(file_adapter.load_config(config_file))
        assert loaded_config == sample_config_data

    def test_config_validation_success(self, file_adapter, sample_config_data):
        """测试配置验证成功"""
        import asyncio
        errors = asyncio.run(file_adapter.validate_config(sample_config_data))

        assert len(errors) == 0

    def test_config_validation_llm_errors(self, file_adapter):
        """测试 LLM 配置验证错误"""
        invalid_config = {
            "llm": {
                "models": {
                    "invalid-model": {
                        # 缺少 provider
                        "model_id": "gpt-4"
                        # 缺少 capabilities
                    }
                }
            }
        }

        import asyncio
        errors = asyncio.run(file_adapter.validate_config(invalid_config))

        assert len(errors) > 0
        assert any("provider" in error for error in errors)
        assert any("capabilities" in error for error in errors)

    def test_config_validation_database_errors(self, file_adapter):
        """测试数据库配置验证错误"""
        invalid_config = {
            "database": {
                # 缺少 url
                "echo": False
            }
        }

        import asyncio
        errors = asyncio.run(file_adapter.validate_config(invalid_config))

        assert len(errors) > 0
        assert any("database" in error for error in errors)
        assert any("url" in error for error in errors)

    def test_config_validation_api_errors(self, file_adapter):
        """测试 API 配置验证错误"""
        invalid_config = {
            "api": {
                "port": 99999,  # 无效端口
                "cors_origins": "not-a-list"  # 应该是列表
            }
        }

        import asyncio
        errors = asyncio.run(file_adapter.validate_config(invalid_config))

        assert len(errors) > 0
        assert any("port" in error for error in errors)
        assert any("cors_origins" in error for error in errors)

    def test_config_validation_logging_errors(self, file_adapter):
        """测试日志配置验证错误"""
        invalid_config = {
            "logging": {
                "level": "INVALID_LEVEL"  # 无效日志级别
            }
        }

        import asyncio
        errors = asyncio.run(file_adapter.validate_config(invalid_config))

        assert len(errors) > 0
        assert any("level" in error for error in errors)

    async def test_get_config_by_path(self, file_adapter, temp_config_dir, sample_config_data):
        """测试通过路径获取配置"""
        # 保存配置
        config_file = Path(temp_config_dir) / "test.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config_data, f, default_flow_style=False, allow_unicode=True)

        # 测试获取整个配置
        config = await file_adapter.get_config("test.yaml")
        assert config == sample_config_data

        # 测试获取子配置
        llm_config = await file_adapter.get_config("llm.models.openai-gpt4")
        assert llm_config["provider"] == "openai"
        assert llm_config["model_id"] == "gpt-4"

        # 测试获取不存在的配置
        nonexistent = await file_adapter.get_config("nonexistent.path")
        assert nonexistent is None

    async def test_set_config_by_path(self, file_adapter, temp_config_dir, sample_config_data):
        """测试通过路径设置配置"""
        # 保存初始配置
        config_file = Path(temp_config_dir) / "test.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config_data, f, default_flow_style=False, allow_unicode=True)

        # 更新配置
        new_port = 9090
        result = await file_adapter.set_config("api.port", new_port)

        assert result is True

        # 验证更新
        updated_config = await file_adapter.get_config("api.port")
        assert updated_config == new_port

    def test_config_caching(self, file_adapter, temp_config_dir, sample_config_data):
        """测试配置缓存"""
        config_file = Path(temp_config_dir) / "cache_test.yaml"

        # 保存配置
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config_data, f, default_flow_style=False, allow_unicode=True)

        # 第一次加载（应该缓存）
        import asyncio
        config1 = asyncio.run(file_adapter.load_config("cache_test.yaml"))

        # 第二次加载（应该从缓存读取）
        config2 = asyncio.run(file_adapter.load_config("cache_test.yaml"))

        assert config1 == config2
        assert config1 is config2  # 应该是同一个对象（缓存）

    def test_unsupported_file_format(self, file_adapter, temp_config_dir):
        """测试不支持的文件格式"""
        config_file = Path(temp_config_dir) / "test.xml"
        config_file.write_text("<config><test>value</test></config>")

        import asyncio
        with pytest.raises(ValueError, match="不支持的配置文件格式"):
            asyncio.run(file_adapter.load_config("test.xml"))

    def test_nonexistent_config_file(self, file_adapter):
        """测试不存在的配置文件"""
        import asyncio
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_adapter.load_config("nonexistent.yaml"))


class TestEnvironmentConfigAdapter:
    """环境变量配置适配器单元测试"""

    @pytest.fixture
    def env_adapter(self):
        """创建环境变量配置适配器实例"""
        return EnvironmentConfigAdapter()

    def test_environment_config_loading(self, env_adapter):
        """测试环境变量配置加载"""
        test_env_vars = {
            "LUMOSCRIBE_DATABASE_URL": "sqlite:///test.db",
            "LUMOSCRIBE_API_PORT": "8080",
            "LUMOSCRIBE_DEBUG": "true",
            "LUMOSCRIBE_LOG_LEVEL": "DEBUG"
        }

        import asyncio
        with patch.dict('os.environ', test_env_vars):
            config = asyncio.run(env_adapter.load_config(""))

        assert config["database"]["url"] == "sqlite:///test.db"
        assert config["api"]["port"] == 8080
        assert config["api"]["debug"] is True
        assert config["logging"]["level"] == "DEBUG"

    def test_models_config_parsing(self, env_adapter):
        """测试模型配置解析"""
        models_config = json.dumps({
            "openai-gpt4": {
                "provider": "openai",
                "model_id": "gpt-4",
                "api_key_env": "OPENAI_API_KEY"
            }
        })

        import asyncio
        with patch.dict('os.environ', {
            "LUMOSCRIBE_LLM_MODELS": models_config
        }):
            config = asyncio.run(env_adapter.load_config(""))

        assert "llm" in config
        assert "models" in config["llm"]
        assert config["llm"]["models"]["openai-gpt4"]["provider"] == "openai"

    def test_environment_config_validation(self, env_adapter):
        """测试环境变量配置验证"""
        import asyncio
        with patch.dict('os.environ', {}):  # 清空环境变量
            errors = asyncio.run(env_adapter.validate_config({}))

        assert len(errors) > 0
        assert any("DATABASE_URL" in error for error in errors)
        assert any("LOG_LEVEL" in error for error in errors)

    async def test_get_environment_config_value(self, env_adapter):
        """测试获取环境变量配置值"""
        with patch.dict('os.environ', {
            "LUMOSCRIBE_API_PORT": "9090",
            "LUMOSCRIBE_LOG_LEVEL": "INFO"
        }):
            port = await env_adapter.get_config("api.port")
            level = await env_adapter.get_config("logging.level")

            assert port == 9090
            assert level == "INFO"

    async def test_set_environment_config_value(self, env_adapter):
        """测试设置环境变量配置值"""
        result = await env_adapter.set_config("api.port", 8888)

        assert result is True
        import os
        assert os.environ.get("LUMOSCRIBE_API_PORT") == "8888"

    def test_config_to_environment_string(self, env_adapter):
        """测试配置转换为环境变量字符串"""
        config_data = {
            "api": {
                "port": 8080,
                "debug": True
            },
            "database": {
                "url": "sqlite:///test.db"
            }
        }

        env_string = env_adapter._config_to_env_string(config_data)

        assert "LUMOSCRIBE_API_PORT=8080" in env_string
        assert "LUMOSCRIBE_API_DEBUG=True" in env_string  # 修复：应该是 True 而不是 true
        assert "LUMOSCRIBE_DATABASE_URL=sqlite:///test.db" in env_string

    def test_nested_config_parsing(self, env_adapter):
        """测试嵌套配置解析"""
        config = {}
        env_adapter._set_nested_value(config, "api.cors_origins", ["http://localhost:3000"])

        assert config["api"]["cors_origins"] == ["http://localhost:3000"]

    def test_config_path_to_env_key(self, env_adapter):
        """测试配置路径转换为环境变量键"""
        env_key = env_adapter._config_path_to_env_key("api.port")

        assert env_key == "LUMOSCRIBE_API_PORT"


class TestFastAPIConfigAdapter:
    """FastAPI 配置适配器单元测试"""

    @pytest.fixture
    def mock_fastapi_app(self):
        """创建模拟的 FastAPI 应用"""
        return Mock()

    @pytest.fixture
    def fastapi_adapter(self, mock_fastapi_app):
        """创建 FastAPI 配置适配器实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_adapter = FileConfigAdapter(temp_dir)
            adapter = FastAPIConfigAdapter(mock_fastapi_app)
            adapter.config_adapter = file_adapter
            return adapter

    def test_fastapi_routes_setup(self, mock_fastapi_app):
        """测试 FastAPI 路由设置"""
        # 设置 Mock 对象的属性
        mock_fastapi_app.router.routes = []
        
        FastAPIConfigAdapter(mock_fastapi_app)

        # 验证路由是否被正确添加
        assert len(mock_fastapi_app.router.routes) > 0

        # 检查是否有配置相关的路由
        route_paths = [route.path for route in mock_fastapi_app.router.routes]
        assert any("/api/config/" in path for path in route_paths)

    def test_config_schema_generation(self, fastapi_adapter):
        """测试配置模式生成"""
        # 测试 LLM 配置模式
        llm_schema = fastapi_adapter._generate_config_schema("llm")

        assert "type" in llm_schema
        assert "properties" in llm_schema
        assert "models" in llm_schema["properties"]

        # 测试数据库配置模式
        db_schema = fastapi_adapter._generate_config_schema("database")

        assert "type" in db_schema
        assert "properties" in db_schema
        assert "url" in db_schema["properties"]

    def test_unknown_config_schema(self, fastapi_adapter):
        """测试未知配置模式"""
        schema = fastapi_adapter._generate_config_schema("unknown")

        assert schema == {}


class TestConfigAdapterFactory:
    """配置适配器工厂单元测试"""

    def test_create_file_adapter(self):
        """测试创建文件适配器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = ConfigAdapterFactory.create_adapter("file", config_dir=temp_dir)
            
            assert isinstance(adapter, FileConfigAdapter)
            assert adapter.config_dir == Path(temp_dir)

    def test_create_fastapi_adapter(self):
        """测试创建 FastAPI 适配器"""
        mock_app = Mock()
        adapter = ConfigAdapterFactory.create_adapter("fastapi", app=mock_app)

        assert isinstance(adapter, FastAPIConfigAdapter)
        assert adapter.app == mock_app

    def test_create_environment_adapter(self):
        """测试创建环境变量适配器"""
        adapter = ConfigAdapterFactory.create_adapter("environment")

        assert isinstance(adapter, EnvironmentConfigAdapter)

    def test_create_invalid_adapter(self):
        """测试创建无效适配器"""
        with pytest.raises(ValueError, match="不支持的配置适配器类型"):
            ConfigAdapterFactory.create_adapter("invalid_type")

    def test_get_available_adapters(self):
        """测试获取可用适配器类型"""
        available_types = ConfigAdapterFactory.get_available_adapters()

        assert "file" in available_types
        assert "fastapi" in available_types
        assert "environment" in available_types
        assert len(available_types) == 3


class TestGlobalConfigAdapter:
    """全局配置适配器测试"""

    def test_get_and_set_global_adapter(self):
        """测试获取和设置全局适配器"""
        # 获取初始适配器
        adapter1 = get_config_adapter()

        # 设置新的适配器
        new_adapter = Mock()
        set_config_adapter(new_adapter)

        # 获取新的适配器
        adapter2 = get_config_adapter()

        assert adapter2 is new_adapter
        assert adapter1 is not adapter2

    def test_init_config_adapter(self):
        """测试初始化配置适配器"""
        import asyncio
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = asyncio.run(init_config_adapter("file", config_dir=temp_dir))
            
            assert isinstance(adapter, FileConfigAdapter)
            assert adapter.config_dir == Path(temp_dir)


class TestConfigAdapterAbstract:
    """配置适配器抽象接口测试"""

    def test_adapter_interface_compliance(self):
        """测试适配器接口合规性"""
        # 测试所有适配器都实现了抽象方法
        with tempfile.TemporaryDirectory() as temp_dir:
            adapters = [
                FileConfigAdapter(temp_dir),
                EnvironmentConfigAdapter(),
                Mock(spec=ConfigAdapter)  # Mock 适配器
            ]

        for adapter in adapters:
            # 验证接口方法存在
            assert hasattr(adapter, 'load_config')
            assert hasattr(adapter, 'save_config')
            assert hasattr(adapter, 'validate_config')
            assert hasattr(adapter, 'get_config')
            assert hasattr(adapter, 'set_config')


if __name__ == "__main__":
    # 运行单元测试
    pytest.main([__file__, "-v"])
