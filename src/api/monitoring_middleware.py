"""
API 监控中间件

为 FastAPI 应用提供详细的请求监控、错误处理和指标收集
集成 LangChain 1.0 最佳实践的错误处理机制
"""

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.framework.shared.logging import get_logger
from src.framework.shared.monitoring import get_metrics_collector, metrics_collector
from src.framework.shared.exceptions import LumoscribeError
from src.framework.shared.error_handler import error_handler

logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """API 监控中间件"""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list = None,
        sample_rate: float = 1.0
    ):
        """
        初始化监控中间件

        Args:
            app: ASGI 应用
            exclude_paths: 排除的路径列表
            sample_rate: 采样率 (0.0-1.0)
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico"
        ]
        self.sample_rate = sample_rate
        self.request_start_times: dict[str, float] = {}

        # 启动指标收集器
        metrics_collector.start(interval=60)
        logger.info("📊 API 监控中间件已启动")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求并收集指标"""
        # 检查是否需要排除此路径
        if self._should_exclude_path(request.url.path):
            return await call_next(request)

        # 采样检查
        if not self._should_sample():
            return await call_next(request)

        # 记录请求开始时间
        start_time = time.time()
        request_id = self._generate_request_id()

        # 获取请求信息
        client_ip = self._get_client_ip(request)
        request_size = self._get_request_size(request)

        try:
            # 处理请求
            response = await call_next(request)

            # 计算响应时间
            response_time = (time.time() - start_time) * 1000  # 转换为毫秒

            # 获取响应大小
            response_size = self._get_response_size(response)

            # 记录成功请求指标
            metrics_collector.record_api_metric(
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                response_time=response_time,
                request_size=request_size,
                response_size=response_size,
                client_ip=client_ip
            )

            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.2f}ms"

            return response

        except Exception as e:
            # 计算错误响应时间
            response_time = (time.time() - start_time) * 1000

            # 判断是否应该重试
            should_retry = self._should_retry_request(e, request)
            
            if should_retry and self._can_retry_request(request):
                logger.warning(f"🔄 请求重试: {request.method} {request.url.path} - {e}")
                # 这里可以实现重试逻辑
                # 暂时先记录指标，实际重试需要在更高层实现
                return await self._handle_retry_request(request, call_next, e)
            
            # 记录错误请求指标
            self._record_error_metrics(
                endpoint=str(request.url.path),
                method=request.method,
                response_time=response_time,
                request_size=request_size,
                client_ip=client_ip,
                error=e
            )

            logger.error(f"❌ API 请求错误: {request.method} {request.url.path} - {type(e).__name__}: {e}")
            
            # 增强错误信息
            if hasattr(e, 'error_code'):
                logger.error(f"错误代码: {e.error_code}")
            if hasattr(e, 'details'):
                logger.error(f"错误详情: {e.details}")
            
            raise

    def _should_exclude_path(self, path: str) -> bool:
        """检查是否应该排除此路径"""
        return any(path.startswith(exclude_path) for exclude_path in self.exclude_paths)

    def _should_sample(self) -> bool:
        """检查是否应该采样"""
        import random
        return random.random() < self.sample_rate

    def _generate_request_id(self) -> str:
        """生成请求 ID"""
        import uuid
        return str(uuid.uuid4())

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端 IP"""
        # 检查常见的代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # 回退到直接连接的 IP
        client = request.client
        return client.host if client else "unknown"

    def _get_request_size(self, request: Request) -> int:
        """获取请求大小"""
        try:
            content_length = request.headers.get("Content-Length")
            return int(content_length) if content_length else 0
        except (ValueError, TypeError):
            return 0

    def _get_response_size(self, response: Response) -> int:
        """获取响应大小"""
        try:
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length else 0
        except (ValueError, TypeError):
            return 0


class TaskMonitoring:
    """任务监控装饰器"""

    @staticmethod
    def monitor_task(task_name: str = None):
        """任务监控装饰器"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                func_name = task_name or f"{func.__module__}.{func.__name__}"

                try:
                    result = await func(*args, **kwargs)
                    execution_time = (time.time() - start_time) * 1000  # 毫秒

                    metrics_collector.record_task_metric(
                        task_name=func_name,
                        execution_time=execution_time,
                        status="success",
                        queue_time=0,  # 这里可以添加队列时间计算
                        worker_id="",   # 这里可以添加工作进程 ID
                        retries=0       # 这里可以添加重试次数
                    )

                    return result

                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000

                    metrics_collector.record_task_metric(
                        task_name=func_name,
                        execution_time=execution_time,
                        status="failed",
                        queue_time=0,
                        worker_id="",
                        retries=0
                    )

                    logger.error(f"任务执行失败: {func_name} - {e}")
                    raise

            return wrapper
        return decorator


def create_monitoring_dashboard() -> dict[str, Any]:
    """创建监控仪表板数据"""
    # 获取系统摘要
    system_summary = metrics_collector.get_system_summary(hours=24)

    # 获取任务摘要
    task_summary = metrics_collector.get_task_summary(hours=24)

    # 获取 API 摘要
    api_summary = metrics_collector.get_api_summary(hours=24)

    # 获取健康状态
    health_status = metrics_collector.get_health_status()

    dashboard = {
        "timestamp": health_status["timestamp"],
        "health_status": health_status,
        "system_overview": system_summary,
        "task_overview": task_summary,
        "api_overview": api_summary,
        "summary": {
            "total_system_samples": len(metrics_collector.system_metrics),
            "total_task_samples": len(metrics_collector.task_metrics),
            "total_api_samples": len(metrics_collector.api_metrics),
            "monitoring_duration": "24 hours"
        }
    }

    return dashboard


def export_monitoring_data(output_dir: str = "logs/monitoring") -> dict[str, str]:
    """导出监控数据"""
    return metrics_collector.export_metrics(output_dir)


class EnhancedMonitoringMiddleware(MonitoringMiddleware):
    """增强的监控中间件 - 集成错误处理和重试机制"""
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list = None,
        sample_rate: float = 1.0,
        enable_retry: bool = True,
        max_retries: int = 2,
        retryable_error_codes: list = None
    ):
        super().__init__(app, exclude_paths, sample_rate)
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retryable_error_codes = retryable_error_codes or [
            "LLM_ERROR", "NETWORK_ERROR", "DATABASE_ERROR", "RATE_LIMIT_ERROR"
        ]
    
    def _should_retry_request(self, error: Exception, request: Request) -> bool:
        """判断是否应该重试请求"""
        if not self.enable_retry:
            return False
        
        # 检查是否是可重试的错误类型
        if isinstance(error, LumoscribeError):
            return error.error_code in self.retryable_error_codes
        
        # 检查是否是网络错误
        if "connection" in str(error).lower() or "timeout" in str(error).lower():
            return True
        
        return False
    
    def _can_retry_request(self, request: Request) -> bool:
        """判断请求是否可以重试"""
        # POST、PUT、DELETE 请求通常不应该重试
        if request.method in ["POST", "PUT", "DELETE"]:
            return False
        
        # 检查请求头中的重试信息
        retry_count = request.headers.get("X-Retry-Count", "0")
        try:
            if int(retry_count) >= self.max_retries:
                return False
        except ValueError:
            pass
        
        return True
    
    async def _handle_retry_request(self, request: Request, call_next: Callable, error: Exception) -> Response:
        """处理重试请求"""
        # 添加重试头信息
        retry_count = int(request.headers.get("X-Retry-Count", "0")) + 1
        request.headers.__dict__["_list"].append(
            (b"x-retry-count", str(retry_count).encode())
        )
        
        logger.info(f"🔄 执行重试 {retry_count}/{self.max_retries}: {request.method} {request.url.path}")
        
        try:
            # 重试请求
            response = await call_next(request)
            
            # 记录重试成功指标
            metrics_collector.record_api_metric(
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                response_time=0,  # 重试时间单独记录
                request_size=0,
                response_size=0,
                client_ip="",
                retry_count=retry_count,
                retry_success=True
            )
            
            # 添加重试成功头
            response.headers["X-Retry-Count"] = str(retry_count)
            response.headers["X-Retry-Success"] = "true"
            
            return response
            
        except Exception as retry_error:
            logger.error(f"❌ 重试失败 {retry_count}/{self.max_retries}: {retry_error}")
            
            # 如果还有重试机会，继续重试
            if retry_count < self.max_retries and self._should_retry_request(retry_error, request):
                return await self._handle_retry_request(request, call_next, retry_error)
            
            # 所有重试都失败，返回原始错误
            raise error
    
    def _record_error_metrics(
        self,
        endpoint: str,
        method: str,
        response_time: float,
        request_size: int,
        client_ip: str,
        error: Exception
    ) -> None:
        """记录错误指标"""
        error_type = type(error).__name__
        error_code = getattr(error, 'error_code', 'UNKNOWN_ERROR')
        
        metrics_collector.record_api_metric(
            endpoint=endpoint,
            method=method,
            status_code=500,
            response_time=response_time,
            request_size=request_size,
            response_size=0,
            client_ip=client_ip,
            error_type=error_type,
            error_code=error_code,
            error_details=str(error)
        )
        
        # 记录错误统计
        metrics_collector.record_error_metric(
            error_type=error_type,
            error_code=error_code,
            endpoint=endpoint,
            method=method,
            severity="high" if error_code in ["LLM_ERROR", "DATABASE_ERROR"] else "medium"
        )


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """断路器中间件"""
    
    def __init__(
        self,
        app: ASGIApp,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        monitored_endpoints: list = None
    ):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.monitored_endpoints = set(monitored_endpoints or ["/api/"])
        
        # 断路器状态
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        endpoint = str(request.url.path)
        
        # 检查是否需要监控此端点
        if not any(endpoint.startswith(monitored) for monitored in self.monitored_endpoints):
            return await call_next(request)
        
        # 检查断路器状态
        if self.circuit_open:
            if self._should_attempt_reset():
                logger.info("🔄 断路器尝试重置")
                self.circuit_open = False
                self.failure_count = 0
            else:
                logger.warning(f"🚨 断路器开启，拒绝请求: {endpoint}")
                return Response(
                    content=json.dumps({
                        "error_code": "CIRCUIT_BREAKER_OPEN",
                        "message": "服务暂时不可用（断路器保护）",
                        "success": False,
                        "retry_after": self.recovery_timeout
                    }),
                    status_code=503,
                    media_type="application/json"
                )
        
        try:
            response = await call_next(request)
            
            # 如果是错误响应，增加失败计数
            if response.status_code >= 500:
                self._record_failure()
            else:
                self._record_success()
            
            return response
            
        except Exception as e:
            self._record_failure()
            logger.error(f"断路器记录失败: {endpoint} - {e}")
            raise
    
    def _record_failure(self) -> None:
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (self.failure_count >= self.failure_threshold and
            not self.circuit_open):
            self.circuit_open = True
            logger.error(f"🚨 断路器开启 - 失败次数: {self.failure_count}")
    
    def _record_success(self) -> None:
        """记录成功"""
        if self.failure_count > 0:
            self.failure_count = 0
            if self.circuit_open:
                self.circuit_open = False
                logger.info("✅ 断路器重置")
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        return time.time() - self.last_failure_time > self.recovery_timeout
