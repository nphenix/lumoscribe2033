"""
lumoscribe2033 - Hybrid Graph-RAG Phase 1 质量平台

基于 speckit 的 AI 驱动质量提升平台，支持多 IDE 适配、文档评估和对话溯源。

主要功能：
- speckit 自动化管线
- 多 IDE 支持 (Cursor, RooCode)
- 文档三分法评估
- 最佳实践与对话溯源
- 静态检查与合规性验证

架构分层：
- framework/: 框架层 - LangChain 1.0, LlamaIndex, FastMCP 基础设施
- domain/: 领域层 - 业务逻辑和领域模型
- api/: 接口层 - FastAPI REST API
- workers/: 任务层 - Arq 异步任务
- cli/: 工具层 - Typer 命令行工具
"""

__version__ = "0.1.0"
__author__ = "lumoscribe2033 Team"
__email__ = "team@lumoscribe2033.com"
__description__ = "Hybrid Graph-RAG Phase 1 质量平台"

from . import domain, framework

__all__ = ["framework", "domain"]
