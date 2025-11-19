"""
配置管理组件

将 ConfigManager 的职责拆分为多个单一职责的组件，遵循单一职责原则：
- ModelConfigManager: 负责模型配置管理
- EnvironmentValidator: 负责环境验证
- ConfigTemplateGenerator: 负责配置模板生成
- RoutingConfigManager: 负责路由配置管理
"""

import os
from pathlib import Path
from typing import Any

from src.framework.shared.config import (
    ModelCapability,
    ModelConfig,
    ModelProvider,
    RoutingConfig,
    Settings,
)
from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class ModelConfigManager:
    """模型配置管理器
    
    单一职责：管理所有模型配置相关操作。
    """
    
    def __init__(self):
        """初始化模型配置管理器"""
        self._models: dict[str, ModelConfig] = self._initialize_default_models()
    
    def _initialize_default_models(self) -> dict[str, ModelConfig]:
        """初始化默认模型配置"""
        return {
            "openai-gpt4": ModelConfig(
                name="openai-gpt4",
                provider=ModelProvider.OPENAI,
                model_id="gpt-4",
                capabilities=[
                    ModelCapability.COMPLEX_REASONING,
                    ModelCapability.CREATIVE_WRITING,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.HIGH_QUALITY
                ],
                cost_per_token=0.03,
                api_key_env="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1"
            ),
            "openai-gpt35": ModelConfig(
                name="openai-gpt35",
                provider=ModelProvider.OPENAI,
                model_id="gpt-3.5-turbo",
                capabilities=[
                    ModelCapability.GENERAL_CONVERSATION,
                    ModelCapability.TEXT_PROCESSING,
                    ModelCapability.MODERATE_QUALITY
                ],
                cost_per_token=0.005,
                api_key_env="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1"
            ),
            "ollama-llama2": ModelConfig(
                name="ollama-llama2",
                provider=ModelProvider.OLLAMA,
                model_id="llama2",
                capabilities=[
                    ModelCapability.SIMPLE_QUERIES,
                    ModelCapability.FAST_RESPONSE,
                    ModelCapability.LOW_COST
                ],
                cost_per_token=0.001,
                base_url="http://localhost:11434"
            ),
            "ollama-mistral": ModelConfig(
                name="ollama-mistral",
                provider=ModelProvider.OLLAMA,
                model_id="mistral",
                capabilities=[
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.TECHNICAL_TASKS,
                    ModelCapability.BALANCED_PERFORMANCE
                ],
                cost_per_token=0.002,
                base_url="http://localhost:11434"
            ),
        }
    
    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """根据名称获取模型配置"""
        return self._models.get(name)
    
    def get_enabled_models(self) -> dict[str, ModelConfig]:
        """获取启用的模型配置"""
        # 过滤出已配置 API Key 的模型
        enabled = {}
        for name, config in self._models.items():
            if config.api_key_env:
                api_key = os.getenv(config.api_key_env)
                if api_key:
                    enabled[name] = config
            else:
                # 没有 API Key 要求的模型（如 Ollama）默认启用
                enabled[name] = config
        return enabled
    
    def get_models_by_provider(self, provider: ModelProvider) -> dict[str, ModelConfig]:
        """根据提供商获取模型配置"""
        return {
            name: config
            for name, config in self._models.items()
            if config.provider == provider
        }
    
    def get_models_by_capability(self, capability: ModelCapability) -> dict[str, ModelConfig]:
        """根据能力获取模型配置"""
        return {
            name: config
            for name, config in self._models.items()
            if capability in config.capabilities
        }
    
    def get_default_model(self) -> str | None:
        """获取默认模型名称"""
        enabled = self.get_enabled_models()
        if not enabled:
            return None
        
        # 优先选择低成本模型
        for name in ["ollama-llama2", "openai-gpt35", "ollama-mistral"]:
            if name in enabled:
                return name
        
        # 返回第一个启用的模型
        return next(iter(enabled.keys()))
    
    def add_model(self, name: str, config: ModelConfig) -> None:
        """添加模型配置"""
        self._models[name] = config
    
    def remove_model(self, name: str) -> bool:
        """移除模型配置"""
        if name in self._models:
            del self._models[name]
            return True
        return False
    
    def list_models(self) -> list[str]:
        """列出所有模型名称"""
        return list(self._models.keys())


class EnvironmentValidator:
    """环境验证器
    
    单一职责：验证环境配置和目录结构。
    """
    
    def __init__(self, settings: Settings):
        """初始化环境验证器
        
        Args:
            settings: 应用设置
        """
        self.settings = settings
    
    def validate_environment(self) -> list[str]:
        """验证环境配置
        
        Returns:
            验证错误列表
        """
        errors = []
        
        # 检查必要的环境变量
        errors.extend(self._validate_env_vars())
        
        # 检查目录结构
        errors.extend(self._validate_directories())
        
        return errors
    
    def _validate_env_vars(self) -> list[str]:
        """验证环境变量"""
        errors = []
        required_env_vars = [
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "ARQ_REDIS_URL"
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                errors.append(f"环境变量 {var} 未设置")
        
        return errors
    
    def _validate_directories(self) -> list[str]:
        """验证目录结构"""
        errors = []
        required_dirs = [
            ("UPLOAD_DIR", self.settings.UPLOAD_DIR),
            ("PERSISTENCE_DIR", self.settings.PERSISTENCE_DIR),
            ("VECTOR_DIR", self.settings.VECTOR_DIR),
            ("GRAPH_DIR", self.settings.GRAPH_DIR)
        ]
        
        for name, path in required_dirs:
            dir_path = Path(path)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"创建目录: {dir_path}")
                except Exception as e:
                    errors.append(f"无法创建目录 {name} ({path}): {e}")
        
        return errors


class ConfigTemplateGenerator:
    """配置模板生成器
    
    单一职责：生成配置模板文件。
    """
    
    def generate_env_template(self) -> str:
        """生成环境变量模板文件
        
        Returns:
            模板文件内容
        """
        template_content = """# lumoscribe2033 环境配置文件

# 应用配置
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO

# API 配置
API_HOST=127.0.0.1
API_PORT=8080
API_CORS_ORIGINS=http://localhost:8080

# LLM 配置
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

LLM_ROUTING_MODE=auto
LLM_FALLBACK_ENABLED=true

# 数据库配置
DATABASE_URL=sqlite:///./data/persistence/lumoscribe2033.db
DATABASE_ECHO=false

CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_DB_PATH=./vector/chroma
CHROMA_PERSIST=true

# Arq 配置
ARQ_REDIS_URL=redis://localhost:6379/0
ARQ_QUEUE_NAME=lumoscribe2033
ARQ_WORKER_COUNT=4

# FastMCP 配置
MCP_SERVER_PORT=8081
MCP_AUTH_TOKEN=your-mcp-auth-token-here
MCP_CERT_FILE=
MCP_KEY_FILE=

# 存储路径配置
UPLOAD_DIR=./data/imports
PERSISTENCE_DIR=./data/persistence
REFERENCE_SAMPLES_DIR=./data/reference_samples
VECTOR_DIR=./vector/chroma
GRAPH_DIR=./graph/snapshots
IDE_PACKAGES_DIR=./ide-packages

# 文档评估配置
AGENT_DOC_MAX_TOKENS=2000
DEVELOPER_DOC_MAX_TOKENS=5000
EXTERNAL_DOC_MAX_TOKENS=10000
DOC_CLASSIFICATION_CONFIDENCE=0.8
DOC_STRUCTURE_SCORE_THRESHOLD=70

# 静态检查配置
RUFF_ENABLED=true
MYPY_ENABLED=true
ESLINT_ENABLED=true
COMPLIANCE_CHECK_ENABLED=true

# 指标收集配置
METRICS_ENABLED=true
METRICS_COLLECTION_INTERVAL=3600
COMPLIANCE_REPORT_PATH=./docs/internal/logs/metrics.md

# IDE 集成配置
CURSOR_COMMANDS_DIR=.cursor/commands
ROOCODE_COMMANDS_DIR=.roo/commands
IDE_PACKAGE_TEMPLATES_DIR=./templates/ide

# 对话导入配置
CURSOR_LOGS_DIR=C:\\\\logs\\\\cursor
ROOCODE_LOGS_DIR=C:\\\\logs\\\\roocode
CONVERSATION_BATCH_SIZE=100
VECTOR_INDEX_REFRESH_INTERVAL=300

# 安全配置
SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# 临时文件配置
TEMP_DIR=./temp
CLEANUP_TEMP_FILES=true
"""
        return template_content


class RoutingConfigManager:
    """路由配置管理器
    
    单一职责：管理路由配置。
    """
    
    def __init__(self):
        """初始化路由配置管理器"""
        self._routing = RoutingConfig()
    
    def get_routing_config(self) -> RoutingConfig:
        """获取路由配置"""
        return self._routing
    
    def update_routing_config(self, **kwargs) -> None:
        """更新路由配置"""
        for key, value in kwargs.items():
            if hasattr(self._routing, key):
                setattr(self._routing, key, value)


class MonitoringConfigManager:
    """监控配置管理器
    
    单一职责：管理监控配置。
    """
    
    def __init__(self):
        """初始化监控配置管理器"""
        self._monitoring: dict[str, Any] = {
            "enabled": True,
            "metrics_collection_interval": 3600,
            "performance_tracking": True,
            "error_tracking": True
        }
    
    def get_monitoring_config(self) -> dict[str, Any]:
        """获取监控配置"""
        return self._monitoring.copy()
    
    def update_monitoring_config(self, **kwargs) -> None:
        """更新监控配置"""
        self._monitoring.update(kwargs)


__all__ = [
    "ModelConfigManager",
    "EnvironmentValidator",
    "ConfigTemplateGenerator",
    "RoutingConfigManager",
    "MonitoringConfigManager",
]

