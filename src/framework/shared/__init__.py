"""
shared/ - 共享工具和数据模型

提供跨层共享的组件：
- 数据模型定义 (Pydantic)
- 配置管理
- 工具函数
- 异常定义
- 常量定义

设计原则：
- 最小依赖
- 高重用性
- 类型安全
- 易于测试
"""

from .config import Settings
from .exceptions import (
    ComplianceError,
    ConfigurationError,
    LumoscribeError,
    PipelineError,
    RAGError,
    ValidationError,
)

__all__ = [
    "Settings",
    "LumoscribeError", "ValidationError", "ConfigurationError",
    "PipelineError", "RAGError", "ComplianceError"
]
