"""
中间件管理器

管理各种中间件的注册、执行和生命周期。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MiddlewareContext:
    """中间件上下文"""
    request: Any
    response: Any | None = None
    metadata: dict[str, Any] | None = None
    error: Exception | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class BaseMiddleware(ABC):
    """中间件基类"""

    @abstractmethod
    async def before_request(self, context: MiddlewareContext) -> bool:
        """请求前处理，返回 False 阻止请求继续"""
        pass

    @abstractmethod
    async def after_request(self, context: MiddlewareContext) -> None:
        """请求后处理"""
        pass

    @abstractmethod
    async def on_error(self, context: MiddlewareContext) -> None:
        """错误处理"""
        pass


class TelemetryMiddleware(BaseMiddleware):
    """遥测中间件"""

    async def before_request(self, context: MiddlewareContext) -> bool:
        context.start_time = datetime.utcnow()
        return True

    async def after_request(self, context: MiddlewareContext) -> None:
        if context.start_time:
            duration = (datetime.utcnow() - context.start_time).total_seconds()
            # 这里可以记录指标
            print(f"Request duration: {duration}s")

    async def on_error(self, context: MiddlewareContext) -> None:
        print(f"Error occurred: {context.error}")


class LoggingMiddleware(BaseMiddleware):
    """日志中间件"""

    async def before_request(self, context: MiddlewareContext) -> bool:
        print(f"Processing request: {context.request}")
        return True

    async def after_request(self, context: MiddlewareContext) -> None:
        print(f"Request completed: {context.response}")

    async def on_error(self, context: MiddlewareContext) -> None:
        print(f"Request failed: {context.error}")


class RateLimitMiddleware(BaseMiddleware):
    """限流中间件"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts: dict[str, Any] = {}

    async def before_request(self, context: MiddlewareContext) -> bool:
        # 简单的限流逻辑
        client_id = getattr(context.request, 'client_id', 'default')
        current_time = datetime.utcnow()

        if client_id not in self.request_counts:
            self.request_counts[client_id] = []

        # 清理过期的请求记录
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if (current_time - req_time).total_seconds() < self.window_seconds
        ]

        # 检查是否超过限制
        if len(self.request_counts[client_id]) >= self.max_requests:
            context.response = {"error": "Rate limit exceeded"}
            return False

        self.request_counts[client_id].append(current_time)
        return True

    async def after_request(self, context: MiddlewareContext) -> None:
        pass

    async def on_error(self, context: MiddlewareContext) -> None:
        pass


class MiddlewareManager:
    """中间件管理器"""

    def __init__(self) -> None:
        self.middlewares: list[BaseMiddleware] = []
        self.enabled = True

    def add_middleware(self, middleware: BaseMiddleware) -> None:
        """添加中间件"""
        self.middlewares.append(middleware)

    def remove_middleware(self, middleware: BaseMiddleware) -> None:
        """移除中间件"""
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)

    async def execute_before_request(self, context: MiddlewareContext) -> bool:
        """执行请求前中间件"""
        if not self.enabled:
            return True

        for middleware in self.middlewares:
            try:
                result = await middleware.before_request(context)
                if not result:
                    return False
            except Exception as e:
                context.error = e
                await self.execute_on_error(context)
                return False

        return True

    async def execute_after_request(self, context: MiddlewareContext) -> None:
        """执行请求后中间件"""
        if not self.enabled:
            return

        for middleware in self.middlewares:
            try:
                await middleware.after_request(context)
            except Exception as e:
                context.error = e
                await self.execute_on_error(context)

    async def execute_on_error(self, context: MiddlewareContext) -> None:
        """执行错误处理中间件"""
        if not self.enabled:
            return

        for middleware in self.middlewares:
            try:
                await middleware.on_error(context)
            except Exception as e:
                # 错误处理中的错误，记录但不传播
                print(f"Error in middleware error handler: {e}")

    def enable(self) -> None:
        """启用中间件"""
        self.enabled = True

    def disable(self) -> None:
        """禁用中间件"""
        self.enabled = False

    def get_middleware_count(self) -> int:
        """获取中间件数量"""
        return len(self.middlewares)
