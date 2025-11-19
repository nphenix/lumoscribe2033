"""
framework/ - 框架层

提供共用基础设施和抽象层，包括：
- orchestrators/: LangChain 1.0 运行器和 LLM router
- rag/: RAG 核心组件
- adapters/: IDE/Conversation/LLM 接口适配器
- storage/: SQLite/Chroma/NetworkX 存储抽象
- shared/: 共享工具和数据模型

设计原则：
- 与业务逻辑解耦
- 支持多种实现方式
- 提供统一的接口抽象
- 便于测试和替换
"""

from . import adapters, orchestrators, rag, shared, storage

__all__ = ["orchestrators", "rag", "adapters", "storage", "shared"]
