"""
workers/ - Arq 异步任务层

处理异步任务和后台作业：
- 任务定义
- 工作者配置
- 任务调度
- 结果处理

Arq 最佳实践：
- 异步任务定义
- 重试机制
- 任务队列管理
- 结果存储
"""

from .settings import WorkerSettings

__all__ = ["WorkerSettings"]
