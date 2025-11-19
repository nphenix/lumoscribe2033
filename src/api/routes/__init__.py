"""
API 路由模块

包含所有 API 路由的导入和注册。

主要路由：
- health: 健康检查和系统状态
- tasks: 任务队列管理
- docs: 文档管理
- speckit: speckit 管线相关接口
"""

# 导入所有路由模块以便在 main.py 中使用
__all__ = ["health", "tasks", "docs", "speckit"]
