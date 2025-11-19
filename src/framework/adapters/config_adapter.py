"""
配置管理适配器

基于 FastAPI/FastMCP 最佳实践实现配置管理接口，支持：
- 动态配置加载
- 环境变量管理
- 配置验证
- 配置热更新
- 多环境支持
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ValidationError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from ..shared.config import ConfigManager, Settings, config_manager
from ..shared.logging import get_logger

logger = get_logger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更处理器"""

    def __init__(self, callback: Callable[[], None]):
        self.callback = callback

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            logger.info(f"配置文件变更: {event.src_path}")
            self.callback()


class ConfigAdapter(ABC):
    """配置管理适配器抽象基类"""

    @abstractmethod
    async def load_config(self, config_path: str) -> dict[str, Any]:
        """加载配置"""
        pass

    @abstractmethod
    async def save_config(self, config_path: str, config_data: dict[str, Any]) -> bool:
        """保存配置"""
        pass

    @abstractmethod
    async def validate_config(self, config_data: dict[str, Any]) -> list[str]:
        """验证配置"""
        pass

    @abstractmethod
    async def get_config(self, key_path: str = "") -> Any:
        """获取配置值"""
        pass

    @abstractmethod
    async def set_config(self, key_path: str, value: Any) -> bool:
        """设置配置值"""
        pass


class FileConfigAdapter(ConfigAdapter):
    """文件配置适配器"""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_cache: dict[str, dict[str, Any]] = {}
        self.file_watchers: dict[str, Observer] = {}

    async def load_config(self, config_path: str) -> dict[str, Any]:
        """加载配置文件"""
        config_file = self.config_dir / config_path

        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        # 检查缓存
        cache_key = str(config_file)
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]

        try:
            with open(config_file, encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")

            # 缓存配置
            self.config_cache[cache_key] = config_data

            # 设置文件监听
            self._setup_file_watcher(config_file)

            return config_data

        except Exception as e:
            logger.error(f"加载配置文件失败 {config_file}: {str(e)}")
            raise

    async def save_config(self, config_path: str, config_data: dict[str, Any]) -> bool:
        """保存配置文件"""
        config_file = self.config_dir / config_path
        config_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                elif config_file.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")

            # 更新缓存
            cache_key = str(config_file)
            self.config_cache[cache_key] = config_data

            logger.info(f"配置文件保存成功: {config_file}")
            return True

        except Exception as e:
            logger.error(f"保存配置文件失败 {config_file}: {str(e)}")
            return False

    async def validate_config(self, config_data: dict[str, Any]) -> list[str]:
        """验证配置数据"""
        errors = []

        # 基础结构验证
        if not isinstance(config_data, dict):
            errors.append("配置必须是字典格式")
            return errors

        # 验证各模块配置
        if "llm" in config_data:
            errors.extend(self._validate_llm_config(config_data["llm"]))

        if "database" in config_data:
            errors.extend(self._validate_database_config(config_data["database"]))

        if "api" in config_data:
            errors.extend(self._validate_api_config(config_data["api"]))

        if "logging" in config_data:
            errors.extend(self._validate_logging_config(config_data["logging"]))

        return errors

    def _validate_llm_config(self, llm_config: dict[str, Any]) -> list[str]:
        """验证 LLM 配置"""
        errors = []

        if "models" in llm_config:
            for model_name, model_config in llm_config["models"].items():
                if not isinstance(model_config, dict):
                    errors.append(f"模型配置 {model_name} 必须是字典格式")
                    continue

                if "provider" not in model_config:
                    errors.append(f"模型 {model_name} 缺少 provider 配置")

                if "model_id" not in model_config:
                    errors.append(f"模型 {model_name} 缺少 model_id 配置")

        return errors

    def _validate_database_config(self, db_config: dict[str, Any]) -> list[str]:
        """验证数据库配置"""
        errors = []

        required_fields = ["url"]
        for required_field in required_fields:
            if required_field not in db_config:
                errors.append(f"数据库配置缺少必需字段: {required_field}")

        return errors

    def _validate_api_config(self, api_config: dict[str, Any]) -> list[str]:
        """验证 API 配置"""
        errors = []

        if "port" in api_config:
            port = api_config["port"]
            if not isinstance(port, int) or not (1 <= port <= 65535):
                errors.append("API 端口配置无效")

        if "cors_origins" in api_config:
            if not isinstance(api_config["cors_origins"], list):
                errors.append("CORS 源配置必须是列表格式")

        return errors

    def _validate_logging_config(self, logging_config: dict[str, Any]) -> list[str]:
        """验证日志配置"""
        errors = []

        if "level" in logging_config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if logging_config["level"] not in valid_levels:
                errors.append(f"无效的日志级别: {logging_config['level']}")

        return errors

    async def get_config(self, key_path: str = "") -> Any:
        """获取配置值"""
        # 如果是文件路径，直接加载文件
        if key_path.endswith(('.yaml', '.yml', '.json')):
            return await self.load_config(key_path)

        # 解析键路径 (例如: "llm.models.openai-gpt4")
        if not key_path:
            return {}

        keys = key_path.split('.')
        config_file = keys[0]
        config_data = await self.load_config(f"{config_file}.yaml")

        # 遍历配置路径
        current = config_data
        for key in keys[1:]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    async def set_config(self, key_path: str, value: Any) -> bool:
        """设置配置值"""
        if '.' not in key_path:
            return False

        keys = key_path.split('.')
        config_file = keys[0]
        config_path = f"{config_file}.yaml"

        try:
            config_data = await self.load_config(config_path)

            # 更新配置
            current = config_data
            for key in keys[1:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = value

            # 保存配置
            return await self.save_config(config_path, config_data)

        except Exception as e:
            logger.error(f"设置配置失败 {key_path}: {str(e)}")
            return False

    def _setup_file_watcher(self, config_file: Path):
        """设置文件监听器"""
        if str(config_file) in self.file_watchers:
            return

        def on_file_change():
            # 清除缓存
            cache_key = str(config_file)
            if cache_key in self.config_cache:
                del self.config_cache[cache_key]
            logger.info(f"配置文件缓存已清除: {config_file}")

        event_handler = ConfigChangeHandler(on_file_change)
        observer = Observer()
        observer.schedule(event_handler, str(config_file.parent), recursive=False)
        observer.start()

        self.file_watchers[str(config_file)] = observer


class FastAPIConfigAdapter(ConfigAdapter):
    """FastAPI 配置适配器"""

    def __init__(self, app: FastAPI):
        self.app = app
        self.config_adapter = FileConfigAdapter()
        self._setup_routes()

    def _setup_routes(self):
        """设置配置管理路由"""
        router = self.app.router

        @router.get("/api/config/{config_file}")
        async def get_config(config_file: str):
            """获取配置"""
            try:
                config_data = await self.config_adapter.load_config(config_file)
                return {"success": True, "data": config_data}
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))

        @router.put("/api/config/{config_file}")
        async def update_config(config_file: str, config_data: dict[str, Any]):
            """更新配置"""
            try:
                # 验证配置
                errors = await self.config_adapter.validate_config(config_data)
                if errors:
                    return {"success": False, "errors": errors}

                # 保存配置
                success = await self.config_adapter.save_config(config_file, config_data)
                return {"success": success}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @router.post("/api/config/validate")
        async def validate_config(config_data: dict[str, Any]):
            """验证配置"""
            try:
                errors = await self.config_adapter.validate_config(config_data)
                return {"valid": len(errors) == 0, "errors": errors}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @router.get("/api/config/schema/{config_type}")
        async def get_config_schema(config_type: str):
            """获取配置模式"""
            try:
                schema = await self._generate_config_schema(config_type)
                return {"success": True, "schema": schema}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def load_config(self, config_path: str) -> dict[str, Any]:
        """加载配置"""
        return await self.config_adapter.load_config(config_path)

    async def save_config(self, config_path: str, config_data: dict[str, Any]) -> bool:
        """保存配置"""
        return await self.config_adapter.save_config(config_path, config_data)

    async def validate_config(self, config_data: dict[str, Any]) -> list[str]:
        """验证配置"""
        return await self.config_adapter.validate_config(config_data)

    async def get_config(self, key_path: str = "") -> Any:
        """获取配置值"""
        return await self.config_adapter.get_config(key_path)

    async def set_config(self, key_path: str, value: Any) -> bool:
        """设置配置值"""
        return await self.config_adapter.set_config(key_path, value)

    async def _generate_config_schema(self, config_type: str) -> dict[str, Any]:
        """生成配置模式"""
        schemas = {
            "llm": {
                "type": "object",
                "properties": {
                    "models": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "provider": {"type": "string", "enum": ["openai", "anthropic", "ollama", "groq"]},
                            "model_id": {"type": "string"},
                            "api_key_env": {"type": "string"},
                            "base_url": {"type": "string"},
                            "capabilities": {"type": "array", "items": {"type": "string"}},
                            "cost_per_token": {"type": "number"},
                            "enabled": {"type": "boolean"}
                        },
                        "required": ["provider", "model_id"]
                    }
                }
            },
            "database": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "echo": {"type": "boolean"},
                    "pool_size": {"type": "integer"},
                    "max_overflow": {"type": "integer"}
                },
                "required": ["url"]
            },
            "api": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    "cors_origins": {"type": "array", "items": {"type": "string"}},
                    "debug": {"type": "boolean"}
                }
            }
        }

        return schemas.get(config_type, {})


class EnvironmentConfigAdapter(ConfigAdapter):
    """环境变量配置适配器"""

    def __init__(self):
        self.env_prefix = "LUMOSCRIBE_"
        self.config_mapping = {
            "LLM_MODELS": ("llm.models", self._parse_models_config),
            "DATABASE_URL": ("database.url", lambda x: x),
            "API_PORT": ("api.port", lambda x: int(x)),
            "DEBUG": ("api.debug", lambda x: x.lower() == "true"),
            "LOG_LEVEL": ("logging.level", lambda x: x.upper())
        }

    async def load_config(self, config_path: str) -> dict[str, Any]:
        """从环境变量加载配置"""
        config_data = {}

        for env_key, (config_path, parser) in self.config_mapping.items():
            env_value = os.getenv(f"{self.env_prefix}{env_key}")
            if env_value:
                try:
                    parsed_value = parser(env_value)
                    self._set_nested_value(config_data, config_path, parsed_value)
                except Exception as e:
                    logger.warning(f"解析环境变量 {env_key} 失败: {str(e)}")

        return config_data

    async def save_config(self, config_path: str, config_data: dict[str, Any]) -> bool:
        """将配置保存为环境变量文件"""
        env_file_path = Path(config_path)

        try:
            env_content = self._config_to_env_string(config_data)

            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write(env_content)

            logger.info(f"环境变量文件生成成功: {env_file_path}")
            return True

        except Exception as e:
            logger.error(f"生成环境变量文件失败: {str(e)}")
            return False

    async def validate_config(self, config_data: dict[str, Any]) -> list[str]:
        """验证环境变量配置"""
        errors = []

        # 检查必需的环境变量
        required_envs = [
            f"{self.env_prefix}DATABASE_URL",
            f"{self.env_prefix}LOG_LEVEL"
        ]

        for env_var in required_envs:
            if not os.getenv(env_var):
                errors.append(f"缺少必需的环境变量: {env_var}")

        return errors

    async def get_config(self, key_path: str = "") -> Any:
        """获取环境变量配置值"""
        if not key_path:
            return await self.load_config("")

        env_key = self._config_path_to_env_key(key_path)
        env_value = os.getenv(env_key)

        if env_value and key_path in self.config_mapping:
            parser = self.config_mapping[key_path][1]
            try:
                return parser(env_value)
            except Exception as e:
                logger.warning(f"解析环境变量 {env_key} 失败: {str(e)}")
                return None

        return env_value

    async def set_config(self, key_path: str, value: Any) -> bool:
        """设置环境变量配置值"""
        env_key = self._config_path_to_env_key(key_path)

        try:
            os.environ[env_key] = str(value)
            return True
        except Exception as e:
            logger.error(f"设置环境变量 {env_key} 失败: {str(e)}")
            return False

    def _parse_models_config(self, models_str: str) -> dict[str, Any]:
        """解析模型配置字符串"""
        try:
            return json.loads(models_str)
        except json.JSONDecodeError:
            logger.warning("模型配置 JSON 解析失败")
            return {}

    def _set_nested_value(self, config_data: dict[str, Any], path: str, value: Any):
        """设置嵌套配置值"""
        keys = path.split('.')
        current = config_data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _config_to_env_string(self, config_data: dict[str, Any]) -> str:
        """将配置转换为环境变量字符串"""
        env_lines = []

        def flatten_config(data: dict[str, Any], prefix: str = ""):
            for key, value in data.items():
                env_key = f"{self.env_prefix}{prefix}{key}".upper()

                if isinstance(value, dict):
                    flatten_config(value, f"{prefix}{key}_")
                else:
                    env_lines.append(f"{env_key}={value}")

        flatten_config(config_data)
        return "\n".join(env_lines)

    def _config_path_to_env_key(self, config_path: str) -> str:
        """将配置路径转换为环境变量键"""
        return f"{self.env_prefix}{config_path.upper().replace('.', '_')}"


class ConfigAdapterFactory:
    """配置适配器工厂"""

    @staticmethod
    def create_adapter(adapter_type: str, **kwargs) -> ConfigAdapter:
        """创建配置适配器"""
        adapter_classes = {
            "file": FileConfigAdapter,
            "fastapi": FastAPIConfigAdapter,
            "environment": EnvironmentConfigAdapter
        }

        if adapter_type not in adapter_classes:
            raise ValueError(f"不支持的配置适配器类型: {adapter_type}")

        adapter_class = adapter_classes[adapter_type]
        return adapter_class(**kwargs)

    @staticmethod
    def get_available_adapters() -> list[str]:
        """获取可用的配置适配器类型"""
        return ["file", "fastapi", "environment"]


# 全局配置适配器实例
_config_adapter: ConfigAdapter | None = None


def get_config_adapter() -> ConfigAdapter:
    """获取全局配置适配器"""
    global _config_adapter
    if _config_adapter is None:
        # 默认使用文件配置适配器
        _config_adapter = FileConfigAdapter()
    return _config_adapter


def set_config_adapter(adapter: ConfigAdapter):
    """设置全局配置适配器"""
    global _config_adapter
    _config_adapter = adapter


async def init_config_adapter(adapter_type: str = "file", **kwargs) -> ConfigAdapter:
    """初始化配置适配器"""
    adapter = ConfigAdapterFactory.create_adapter(adapter_type, **kwargs)
    set_config_adapter(adapter)
    return adapter
