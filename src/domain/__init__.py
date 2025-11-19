"""
domain/ - 领域层

包含核心业务逻辑和领域模型：
- pipeline/: speckit 自动化管线
- doc_review/: 文档三分法评估
- compliance/: 静态检查与可追溯性
- knowledge/: 最佳实践与对话溯源

领域驱动设计原则：
- 业务逻辑集中
- 领域模型丰富
- 依赖倒置
- 持久化无关
"""

from . import compliance, doc_review, knowledge, pipeline

__all__ = ["pipeline", "doc_review", "compliance", "knowledge"]
