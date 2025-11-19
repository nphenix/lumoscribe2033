"""
配置管理

基于 pydantic-settings 实现：
- 环境变量支持
- 配置验证
- 类型安全
- 默认值管理
- 模型配置管理
- 环境验证和目录创建
- 配置模板生成

设计原则：
- 单一职责
- 类型安全
- 环境隔离
- 易于测试
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypedDict, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # 如果 pydantic-settings 不可用，提供基础实现
    class BaseSettings(BaseModel):
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False


class ModelProvider(str, Enum):
    """模型提供商枚举"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLAUDE = "claude"
    GROQ = "groq"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """模型能力枚举"""
    GENERAL_CONVERSATION = "general_conversation"
    COMPLEX_REASONING = "complex_reasoning"
    CREATIVE_WRITING = "creative_writing"
    CODE_ANALYSIS = "code_analysis"
    TEXT_PROCESSING = "text_processing"
    FAST_RESPONSE = "fast_response"
    LOW_COST = "low_cost"
    HIGH_QUALITY = "high_quality"
    MODERATE_QUALITY = "moderate_quality"
    TECHNICAL_TASKS = "technical_tasks"
    BALANCED_PERFORMANCE = "balanced_performance"
    SIMPLE_QUERIES = "simple_queries"
    GENERAL_PURPOSE = "general_purpose"
    LANGCHAIN_COMPATIBLE = "langchain_compatible"


class ModelConfig(BaseModel):
    """模型配置"""
    name: str = Field(..., description="模型名称")
    provider: ModelProvider = Field(..., description="模型提供商")
    model_id: str = Field(..., description="模型ID")
    capabilities: list[ModelCapability] = Field(default_factory=list, description="模型能力")
    cost_per_token: float = Field(default=0.01, description="每令牌成本")
    max_tokens: int = Field(default=4000, description="最大令牌数")
    temperature: float = Field(default=0.1, description="温度参数")
    timeout: int = Field(default=30, description="超时时间(秒)")
    enabled: bool = Field(default=True, description="是否启用")
    api_key_env: str | None = Field(default=None, description="API密钥环境变量名")
    base_url: str | None = Field(default=None, description="API基础URL")
    custom_params: dict[str, Any] = Field(default_factory=dict, description="自定义参数")


class RoutingConfig(BaseModel):
    """路由配置"""
    enable_performance_routing: bool = Field(default=True, description="启用性能路由")
    enable_cost_optimization: bool = Field(default=True, description="启用成本优化")
    confidence_threshold: float = Field(default=0.7, description="置信度阈值")
    fallback_to_default: bool = Field(default=True, description="回退到默认模型")
    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟(秒)")
    performance_decay_factor: float = Field(default=0.95, description="性能衰减因子")


class Settings(BaseSettings):
    """应用配置类"""

    # 通用配置
    ENVIRONMENT: str = Field(default="development", description="运行环境")
    DEBUG: bool = Field(default=False, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")

    # LLM 配置
    OPENAI_API_BASE: str = Field(
        default="http://localhost:11434/v1", description="OpenAI API 基础地址"
    )
    OPENAI_API_KEY: str = Field(default="dummy", description="OpenAI API 密钥")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI 模型名称")
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="OpenAI 嵌入模型"
    )

    OLLAMA_HOST: str = Field(
        default="http://localhost:11434", description="Ollama 主机地址"
    )
    OLLAMA_MODEL: str = Field(default="llama2", description="Ollama 模型名称")
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text", description="Ollama 嵌入模型"
    )

    LLM_ROUTING_MODE: str = Field(default="auto", description="LLM 路由模式")
    LLM_FALLBACK_ENABLED: bool = Field(default=True, description="LLM 回退启用")

    # 数据库配置
    DATABASE_URL: str = Field(
        default="sqlite:///./data/persistence/lumoscribe2033.db", description="数据库连接地址"
    )
    DATABASE_ECHO: bool = Field(default=False, description="数据库调试输出")

    CHROMA_HOST: str = Field(default="localhost", description="ChromaDB 主机")
    CHROMA_PORT: int = Field(default=8000, description="ChromaDB 端口")
    CHROMA_DB_PATH: str = Field(default="./vector/chroma", description="ChromaDB 路径")
    CHROMA_PERSIST: bool = Field(default=True, description="ChromaDB 持久化")

    # FastAPI 配置
    API_HOST: str = Field(default="127.0.0.1", description="API 主机地址")
    API_PORT: int = Field(default=8080, description="API 端口")
    API_CORS_ORIGINS: str = Field(
        default="http://localhost:8080", description="CORS 允许的源"
    )

    # Arq 配置
    ARQ_REDIS_URL: str = Field(
        default="redis://localhost:6379/0", description="Arq Redis 连接地址"
    )
    ARQ_QUEUE_NAME: str = Field(
        default="lumoscribe2033", description="Arq 队列名称"
    )
    ARQ_WORKER_COUNT: int = Field(default=4, description="Arq 工作者数量")

    # FastMCP 配置
    MCP_SERVER_PORT: int = Field(default=8081, description="MCP 服务器端口")
    MCP_AUTH_TOKEN: str = Field(default="", description="MCP 认证令牌")
    MCP_CERT_FILE: str = Field(default="", description="MCP 证书文件")
    MCP_KEY_FILE: str = Field(default="", description="MCP 私钥文件")

    # 存储路径配置
    UPLOAD_DIR: str = Field(default="./data/imports", description="上传目录")
    PERSISTENCE_DIR: str = Field(default="./data/persistence", description="持久化目录")
    REFERENCE_SAMPLES_DIR: str = Field(
        default="./data/reference_samples", description="参考样本目录"
    )
    VECTOR_DIR: str = Field(default="./vector/chroma", description="向量存储目录")
    GRAPH_DIR: str = Field(default="./graph/snapshots", description="图存储目录")
    IDE_PACKAGES_DIR: str = Field(default="./ide-packages", description="IDE 包目录")

    # 文档评估配置
    AGENT_DOC_MAX_TOKENS: int = Field(default=2000, description="Agent 文档最大令牌数")
    DEVELOPER_DOC_MAX_TOKENS: int = Field(
        default=5000, description="开发者文档最大令牌数"
    )
    EXTERNAL_DOC_MAX_TOKENS: int = Field(
        default=10000, description="外部文档最大令牌数"
    )

    DOC_CLASSIFICATION_CONFIDENCE: float = Field(
        default=0.8, description="文档分类置信度"
    )
    DOC_STRUCTURE_SCORE_THRESHOLD: int = Field(
        default=70, description="文档结构评分阈值"
    )

    # 静态检查配置
    RUFF_ENABLED: bool = Field(default=True, description="Ruff 检查启用")
    MYPY_ENABLED: bool = Field(default=True, description="Mypy 检查启用")
    ESLINT_ENABLED: bool = Field(default=True, description="ESLint 检查启用")
    COMPLIANCE_CHECK_ENABLED: bool = Field(default=True, description="合规检查启用")

    # 指标收集配置
    METRICS_ENABLED: bool = Field(default=True, description="指标收集启用")
    METRICS_COLLECTION_INTERVAL: int = Field(
        default=3600, description="指标收集间隔（秒）"
    )
    COMPLIANCE_REPORT_PATH: str = Field(
        default="./docs/internal/logs/metrics.md", description="合规报告路径"
    )

    # IDE 集成配置
    CURSOR_COMMANDS_DIR: str = Field(
        default=".cursor/commands", description="Cursor 命令目录"
    )
    ROOCODE_COMMANDS_DIR: str = Field(
        default=".roo/commands", description="RooCode 命令目录"
    )
    IDE_PACKAGE_TEMPLATES_DIR: str = Field(
        default="./templates/ide", description="IDE 包模板目录"
    )

    # 对话导入配置
    CURSOR_LOGS_DIR: str = Field(
        default="C:\\logs\\cursor", description="Cursor 日志目录"
    )
    ROOCODE_LOGS_DIR: str = Field(
        default="C:\\logs\\roocode", description="RooCode 日志目录"
    )
    CONVERSATION_BATCH_SIZE: int = Field(
        default=100, description="对话批量处理大小"
    )
    VECTOR_INDEX_REFRESH_INTERVAL: int = Field(
        default=300, description="向量索引刷新间隔（秒）"
    )

    # 安全配置
    SECRET_KEY: str = Field(
        default="your-secret-key-here-change-in-production", description="密钥"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT 算法")
    JWT_EXPIRATION_HOURS: int = Field(default=24, description="JWT 过期时间（小时）")

    # 临时文件配置
    TEMP_DIR: str = Field(default="./temp", description="临时文件目录")
    CLEANUP_TEMP_FILES: bool = Field(default=True, description="清理临时文件")

    # 速率限制配置
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="速率限制请求数")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="速率限制时间窗口")
    MAX_REQUEST_SIZE: int = Field(default=10485760, description="最大请求大小")

    # LLM 追踪配置
    LLM_TRACING_ENABLED: bool = Field(default=True, description="LLM 追踪启用")
    PROJECT_NAME: str = Field(default="lumoscribe2033", description="项目名称")

    # KATpro1 配置
    KATPRO1_API_KEY: str = Field(default="", description="KATpro1 API 密钥")
    KATPRO1_BASE_URL: str = Field(default="https://wanqing.streamlakeapi.com/api/gateway/v1/endpoints", description="KATpro1 API 基础地址")
    KATPRO1_MODEL: str = Field(default="ep-zd78oa-1761741032688717062", description="KATpro1 模型名称")

    # LangChain 模型配置前缀
    LANGCHAIN_MODEL_PREFIX: str = Field(default="LC_MODEL_", description="LangChain 模型配置前缀")

    # LangChain 路由配置
    LANGCHAIN_ROUTING_ENABLE_PERFORMANCE: bool = Field(default=True, description="启用 LangChain 性能路由")
    LANGCHAIN_ROUTING_ENABLE_COST_OPTIMIZATION: bool = Field(default=True, description="启用 LangChain 成本优化")
    LANGCHAIN_ROUTING_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="LangChain 置信度阈值")
    LANGCHAIN_ROUTING_MAX_RETRIES: int = Field(default=2, description="LangChain 最大重试次数")

    # LangChain 监控配置
    LANGCHAIN_MONITORING_COLLECT_METRICS: bool = Field(default=True, description="LangChain 指标收集启用")
    LANGCHAIN_MONITORING_LOG_ROUTING: bool = Field(default=True, description="LangChain 路由日志启用")
    LANGCHAIN_MONITORING_METRIC_WINDOW_SIZE: int = Field(default=100, description="LangChain 指标窗口大小")

    # 向量存储配置
    VECTOR_STORE_PERSIST_DIR: str = Field(default="./vector/chroma", description="向量存储持久化目录")
    VECTOR_STORE_COLLECTION_NAME: str = Field(default="default", description="向量存储集合名称")

    # LangChain 模型配置 - 文档分析
    LC_MODEL_KATPRO1DOC_TEMPERATURE: float = Field(default=0.3, description="LangChain KATpro1 文档模型温度")
    LC_MODEL_KATPRO1DOC_MAX_TOKENS: int = Field(default=256000, description="LangChain KATpro1 文档模型最大令牌数")
    LC_MODEL_KATPRO1DOC_TIMEOUT: int = Field(default=60, description="LangChain KATpro1 文档模型超时时间")
    LC_MODEL_KATPRO1DOC_ENABLED: bool = Field(default=True, description="LangChain KATpro1 文档模型启用")
    LC_MODEL_KATPRO1DOC_PRIORITY: int = Field(default=100, description="LangChain KATpro1 文档模型优先级")
    LC_MODEL_KATPRO1DOC_API_KEY: str = Field(default="", description="LangChain KATpro1 文档模型 API 密钥")
    LC_MODEL_KATPRO1DOC_BASE_URL: str = Field(default="https://wanqing.streamlakeapi.com/api/gateway/v1/endpoints", description="LangChain KATpro1 文档模型基础 URL")
    LC_MODEL_KATPRO1DOC_MODEL: str = Field(default="ep-zd78oa-1761741032688717062", description="LangChain KATpro1 文档模型名称")

    # LangChain 模型配置 - 代码分析
    LC_MODEL_KATPRO1CODE_TEMPERATURE: float = Field(default=0.1, description="LangChain KATpro1 代码模型温度")
    LC_MODEL_KATPRO1CODE_MAX_TOKENS: int = Field(default=256000, description="LangChain KATpro1 代码模型最大令牌数")
    LC_MODEL_KATPRO1CODE_TIMEOUT: int = Field(default=120, description="LangChain KATpro1 代码模型超时时间")
    LC_MODEL_KATPRO1CODE_ENABLED: bool = Field(default=True, description="LangChain KATpro1 代码模型启用")
    LC_MODEL_KATPRO1CODE_PRIORITY: int = Field(default=90, description="LangChain KATpro1 代码模型优先级")
    LC_MODEL_KATPRO1CODE_API_KEY: str = Field(default="", description="LangChain KATpro1 代码模型 API 密钥")
    LC_MODEL_KATPRO1CODE_BASE_URL: str = Field(default="https://wanqing.streamlakeapi.com/api/gateway/v1/endpoints", description="LangChain KATpro1 代码模型基础 URL")
    LC_MODEL_KATPRO1CODE_MODEL: str = Field(default="ep-zd78oa-1761741032688717062", description="LangChain KATpro1 代码模型名称")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.ENVIRONMENT == "testing"

    def get_cors_origins(self) -> list[str]:
        """获取 CORS 允许的源列表"""
        if not self.API_CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.API_CORS_ORIGINS.split(",")]

    def get_database_kwargs(self) -> dict:
        """获取数据库连接参数"""
        return {
            "echo": self.DATABASE_ECHO,
            "connect_args": {"check_same_thread": not self.is_production}
        }

    def get_redis_kwargs(self) -> dict:
        """获取 Redis 连接参数"""
        parsed = urlparse(self.ARQ_REDIS_URL)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 6379,
            "db": int(parsed.path.lstrip("/")) if parsed.path else 0,
            "password": parsed.password,
            "ssl": parsed.scheme in ("rediss", "rediss"),
            "decode_responses": True
        }

    def get_llm_kwargs(self) -> dict:
        """获取 LLM 连接参数"""
        return {
            "model": self.OPENAI_MODEL,
            "api_key": self.OPENAI_API_KEY,
            "base_url": self.OPENAI_API_BASE,
            "temperature": 0.1,
            "timeout": 60,
            "max_retries": 3
        }

    def get_chroma_kwargs(self) -> dict:
        """获取 ChromaDB 连接参数"""
        return {
            "host": self.CHROMA_HOST,
            "port": self.CHROMA_PORT,
            "path": self.CHROMA_DB_PATH if not self.CHROMA_HOST else None
        }

    def validate_config(self) -> list[str]:
        """验证配置项"""
        errors = []

        # 验证数据库 URL
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL 不能为空")

        # 验证 Redis URL
        if not self.ARQ_REDIS_URL:
            errors.append("ARQ_REDIS_URL 不能为空")

        # 验证 API 端口范围
        if not (1024 <= self.API_PORT <= 65535):
            errors.append("API_PORT 必须在 1024-65535 范围内")

        # 验证 LLM 配置
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 不能为空")

        # 验证存储路径
        required_paths = [
            ("UPLOAD_DIR", self.UPLOAD_DIR),
            ("PERSISTENCE_DIR", self.PERSISTENCE_DIR),
            ("VECTOR_DIR", self.VECTOR_DIR),
            ("GRAPH_DIR", self.GRAPH_DIR)
        ]

        for name, path in required_paths:
            if not path or not path.strip():
                errors.append(f"{name} 不能为空")

        return errors

    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return len(self.validate_config()) == 0

    def get_environment_info(self) -> dict:
        """获取环境信息"""
        import platform
        import sys

        return {
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "python_version": sys.version,
            "platform": platform.platform(),
            "config_source": "pydantic-settings"
        }


class ConfigManager:
    """
    配置管理器（协调者）
    
    使用组合模式整合各个配置组件，遵循单一职责原则。
    每个组件负责自己的领域，ConfigManager 负责协调。
    """

    def __init__(self):
        self.settings = Settings()
        
        # 使用组合模式，将职责委托给专门的组件
        from src.framework.shared.config_components import (
            ModelConfigManager,
            EnvironmentValidator,
            ConfigTemplateGenerator,
            RoutingConfigManager,
            MonitoringConfigManager,
        )
        
        self.model_manager = ModelConfigManager()
        self.env_validator = EnvironmentValidator(self.settings)
        self.template_generator = ConfigTemplateGenerator()
        self.routing_manager = RoutingConfigManager()
        self.monitoring_manager = MonitoringConfigManager()

    # 委托方法：将职责委托给专门的组件

    def validate_environment(self) -> list[str]:
        """验证环境配置（委托给 EnvironmentValidator）"""
        return self.env_validator.validate_environment()

    def generate_env_template(self) -> str:
        """生成环境变量模板文件（委托给 ConfigTemplateGenerator）"""
        template_content = self.template_generator.generate_env_template()
        
        # 可选：写入文件
        env_file = Path(".env.example")
        env_file.write_text(template_content, encoding="utf-8")
        from src.framework.shared.logging import get_logger
        logger = get_logger(__name__)
        logger.info(f"环境变量模板已生成: {env_file}")
        
        return template_content

    def get_config_status(self) -> dict[str, Any]:
        """
        获取配置状态

        Returns:
            配置状态信息
        """
        return {
            "valid": self.settings.is_valid(),
            "environment": self.settings.get_environment_info(),
            "validation_errors": self.settings.validate_config(),
            "env_file_exists": Path(".env").exists(),
            "env_example_exists": Path(".env.example").exists()
        }

    def setup_development_environment(self) -> None:
        """
        设置开发环境
        """
        logger = self._get_logger()
        if logger:
            logger.info("设置开发环境...")

        # 生成环境变量模板
        self.generate_env_template()

        # 创建必要的目录
        directories = [
            self.settings.UPLOAD_DIR,
            self.settings.PERSISTENCE_DIR,
            self.settings.REFERENCE_SAMPLES_DIR,
            self.settings.VECTOR_DIR,
            self.settings.GRAPH_DIR,
            self.settings.IDE_PACKAGES_DIR,
            "./logs",
            "./temp",
            "./config"
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            if logger:
                logger.debug(f"确保目录存在: {dir_path}")

        # 验证环境
        errors = self.validate_environment()
        if errors:
            if logger:
                logger.warning(f"环境验证发现 {len(errors)} 个问题:")
                for error in errors:
                    logger.warning(f"  - {error}")
        else:
            if logger:
                logger.info("✅ 环境验证通过")

    def get_model_by_name(self, name: str) -> ModelConfig | None:
        """根据名称获取模型配置（委托给 ModelConfigManager）"""
        return self.model_manager.get_model_by_name(name)

    def get_enabled_models(self) -> dict[str, ModelConfig]:
        """获取启用的模型配置（委托给 ModelConfigManager）"""
        return self.model_manager.get_enabled_models()

    def enable_model(self, name: str) -> bool:
        """启用模型"""
        if name in self._models:
            self._models[name].enabled = True
            return True
        return False

    def disable_model(self, name: str) -> bool:
        """禁用模型"""
        if name in self._models:
            self._models[name].enabled = False
            return True
        return False

    def update_model_config(self, name: str, **kwargs) -> bool:
        """更新模型配置"""
        if name in self._models:
            model_config = self._models[name]
            for key, value in kwargs.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            return True
        return False

    def add_custom_model(self, config: ModelConfig) -> bool:
        """添加自定义模型"""
        if config.name in self._models:
            return False  # 模型已存在
        self._models[config.name] = config
        return True

    def remove_model(self, name: str) -> bool:
        """移除模型"""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def get_routing_config(self) -> RoutingConfig:
        """获取路由配置（委托给 RoutingConfigManager）"""
        return self.routing_manager.get_routing_config()

    def update_routing_config(self, **kwargs) -> None:
        """更新路由配置"""
        for key, value in kwargs.items():
            if hasattr(self._routing, key):
                setattr(self._routing, key, value)

    def get_monitoring_config(self) -> dict[str, Any]:
        """获取监控配置（委托给 MonitoringConfigManager）"""
        return self.monitoring_manager.get_monitoring_config()

    def update_monitoring_config(self, **kwargs) -> None:
        """更新监控配置"""
        self._monitoring.update(kwargs)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """获取配置设置"""
        if hasattr(self.settings, key):
            return getattr(self.settings, key)
        return default

    def get_models_by_provider(self, provider: ModelProvider) -> dict[str, ModelConfig]:
        """根据提供商获取模型（委托给 ModelConfigManager）"""
        return self.model_manager.get_models_by_provider(provider)

    def get_models_by_capability(self, capability: ModelCapability) -> dict[str, ModelConfig]:
        """根据能力获取模型（委托给 ModelConfigManager）"""
        return self.model_manager.get_models_by_capability(capability)

    def get_default_model(self) -> str | None:
        """获取默认模型（委托给 ModelConfigManager）"""
        return self.model_manager.get_default_model()

    def validate_config(self) -> list[str]:
        """验证配置的有效性"""
        errors = []

        # 检查是否有启用的模型
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            errors.append("没有启用任何模型")

        # 检查必需的环境变量
        for name, config in enabled_models.items():
            if config.api_key_env and not os.getenv(config.api_key_env):
                errors.append(f"模型 {name} 的 API 密钥环境变量 {config.api_key_env} 未设置")

        # 检查路由配置
        routing = self._routing
        if not (0.0 <= routing.confidence_threshold <= 1.0):
            errors.append("置信度阈值必须在 0.0 到 1.0 之间")

        if routing.max_retries < 0:
            errors.append("最大重试次数不能为负数")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            "models": {
                name: {
                    "name": config.name,
                    "provider": config.provider.value,
                    "model_id": config.model_id,
                    "capabilities": [cap.value for cap in config.capabilities],
                    "cost_per_token": config.cost_per_token,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "timeout": config.timeout,
                    "enabled": config.enabled,
                    "api_key_env": config.api_key_env,
                    "base_url": config.base_url,
                    "custom_params": config.custom_params
                }
                for name, config in self._models.items()
            },
            "routing": {
                "enable_performance_routing": self._routing.enable_performance_routing,
                "enable_cost_optimization": self._routing.enable_cost_optimization,
                "confidence_threshold": self._routing.confidence_threshold,
                "fallback_to_default": self._routing.fallback_to_default,
                "max_retries": self._routing.max_retries,
                "retry_delay": self._routing.retry_delay,
                "performance_decay_factor": self._routing.performance_decay_factor
            },
            "monitoring": self._monitoring
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'ConfigManager':
        """从字典创建配置管理器"""
        manager = cls()

        # 从字典重建配置
        if "models" in config_dict:
            models = {}
            for name, model_data in config_dict["models"].items():
                capabilities = [ModelCapability(cap) for cap in model_data["capabilities"]]
                provider = ModelProvider(model_data["provider"])

                config = ModelConfig(
                    name=model_data["name"],
                    provider=provider,
                    model_id=model_data["model_id"],
                    capabilities=capabilities,
                    cost_per_token=model_data["cost_per_token"],
                    max_tokens=model_data.get("max_tokens", 4000),
                    temperature=model_data.get("temperature", 0.1),
                    timeout=model_data.get("timeout", 30),
                    enabled=model_data.get("enabled", True),
                    api_key_env=model_data.get("api_key_env"),
                    base_url=model_data.get("base_url"),
                    custom_params=model_data.get("custom_params", {})
                )
                models[name] = config
            manager._models = models

        if "routing" in config_dict:
            routing_data = config_dict["routing"]
            manager._routing = RoutingConfig(**routing_data)

        if "monitoring" in config_dict:
            manager._monitoring = config_dict["monitoring"]

        return manager


# 全局配置实例 - 延迟初始化
_settings = None
_config_manager = None

def get_settings() -> Settings:
    """获取全局设置实例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# 全局实例
settings = get_settings()
config_manager = get_config_manager()

def validate_global_config() -> list[str]:
    """验证全局配置"""
    current_settings = get_settings()
    return current_settings.validate_config()
