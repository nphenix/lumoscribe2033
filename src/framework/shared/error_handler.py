"""
增强的错误处理系统

基于 LangChain 1.0 最佳实践，提供统一的异常处理、重试机制和错误恢复功能。
支持 LLM 调用、网络请求、数据库操作等场景的错误处理。
"""

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any, TypeVar, Union, Optional, Dict, List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.framework.shared.exceptions import (
    LumoscribeError,
    LLMError,
    NetworkError,
    DatabaseError,
    RateLimitError,
    ValidationError,
    PipelineError
)

T = TypeVar("T")

# 可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    NetworkError,
    LLMError,
    DatabaseError,
    RateLimitError,
)

# 不可重试的异常类型
NON_RETRYABLE_EXCEPTIONS = (
    ValidationError,
    PipelineError,
    LumoscribeError,
)


class ErrorHandler:
    """错误处理器 - 提供统一的错误处理策略"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
        enable_circuit_breaker: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: int = 60
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout = circuit_breaker_recovery_timeout
        
        # 断路器状态
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open = False
    
    def is_circuit_open(self) -> bool:
        """检查断路器是否开启"""
        if not self.enable_circuit_breaker:
            return False
        
        if not self._circuit_open:
            return False
        
        # 检查是否应该尝试恢复
        if time.time() - self._last_failure_time > self.circuit_breaker_recovery_timeout:
            self._circuit_open = False
            self._failure_count = 0
            logger.info("🔄 断路器尝试恢复")
            return False
        
        return True
    
    def record_failure(self) -> None:
        """记录失败"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if (self.enable_circuit_breaker and 
            self._failure_count >= self.circuit_breaker_failure_threshold and 
            not self._circuit_open):
            self._circuit_open = True
            logger.error(f"🚨 断路器开启 - 失败次数: {self._failure_count}")
    
    def record_success(self) -> None:
        """记录成功"""
        if self._failure_count > 0:
            self._failure_count = 0
            if self._circuit_open:
                self._circuit_open = False
                logger.info("✅ 断路器恢复")
    
    def should_retry(self, exception: Exception) -> bool:
        """判断是否应该重试"""
        if self.is_circuit_open():
            return False
        
        # 检查异常类型是否可重试
        if not isinstance(exception, self.retryable_exceptions):
            return False
        
        # 特定异常的重试判断
        if isinstance(exception, RateLimitError):
            # 限流错误应该重试
            return True
        elif isinstance(exception, NetworkError):
            # 网络错误应该重试
            return True
        elif isinstance(exception, LLMError):
            # LLM 错误根据具体类型判断
            return self._should_retry_llm_error(exception)
        
        return True
    
    def _should_retry_llm_error(self, exception: LLMError) -> bool:
        """判断 LLM 错误是否应该重试"""
        # 模型不可用、超时等错误应该重试
        if "timeout" in str(exception).lower() or "unavailable" in str(exception).lower():
            return True
        # 认证错误、配额耗尽等不应该重试
        if "auth" in str(exception).lower() or "quota" in str(exception).lower():
            return False
        return True
    
    def create_retry_decorator(self, **override_kwargs) -> Callable:
        """创建重试装饰器"""
        retry_kwargs = {
            "retry": retry_if_exception_type(self.retryable_exceptions),
            "stop": stop_after_attempt(override_kwargs.get("max_retries", self.max_retries) + 1),  # 总共执行次数 = 初始 + 重试
            "wait": wait_exponential(
                multiplier=override_kwargs.get("base_delay", self.base_delay),
                max=override_kwargs.get("max_delay", self.max_delay)
            ),
            "before_sleep": self._before_sleep,
            "reraise": True
        }
        
        return retry(**retry_kwargs)
    
    def _before_sleep(self, retry_state) -> None:
        """重试前的回调"""
        logger.warning(
            f"🔄 重试操作 - 尝试次数: {retry_state.attempt_number}, "
            f"异常: {retry_state.outcome.exception()}"
        )
    
    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """执行带重试的函数"""
        if self.is_circuit_open():
            raise LumoscribeError(
                "服务暂时不可用（断路器开启）",
                "CIRCUIT_BREAKER_OPEN"
            )
        
        retry_decorator = self.create_retry_decorator()
        wrapped_func = retry_decorator(func)
        
        try:
            result = await wrapped_func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else wrapped_func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            logger.error(f"❌ 操作最终失败: {e}")
            raise


class LLMErrorsHandler:
    """LLM 错误处理器 - 专门处理 LLM 调用错误"""
    
    @staticmethod
    def handle_llm_error(error: Exception, context: str = "") -> LumoscribeError:
        """处理 LLM 错误并转换为 LumoscribeError"""
        error_msg = str(error).lower()
        
        if "timeout" in error_msg:
            return LLMError(
                f"LLM 调用超时: {context}",
                model=getattr(error, 'model', None)
            )
        elif "quota" in error_msg or "limit" in error_msg:
            return LLMError(
                f"LLM 配额不足: {context}",
                model=getattr(error, 'model', None)
            )
        elif "auth" in error_msg or "unauthorized" in error_msg:
            return LLMError(
                f"LLM 认证失败: {context}",
                model=getattr(error, 'model', None)
            )
        elif "not found" in error_msg:
            return LLMError(
                f"LLM 模型不存在: {context}",
                model=getattr(error, 'model', None)
            )
        else:
            return LLMError(
                f"LLM 调用失败: {context} - {str(error)}"
            )


class PipelineErrorHandler:
    """管线错误处理器 - 处理管线执行中的错误"""
    
    @staticmethod
    def handle_pipeline_error(error: Exception, stage: str = "", pipeline_id: str = "") -> LumoscribeError:
        """处理管线错误并转换为 LumoscribeError"""
        if isinstance(error, LumoscribeError):
            return error
        
        error_msg = str(error).lower()
        
        if "validation" in error_msg or "invalid" in error_msg:
            return ValidationError(
                f"管线验证失败 [{stage}]: {str(error)}"
            )
        elif "timeout" in error_msg:
            return PipelineError(
                f"管线执行超时 [{stage}]: {str(error)}",
                pipeline_id=pipeline_id,
                stage=stage
            )
        elif "resource" in error_msg or "memory" in error_msg:
            return PipelineError(
                f"管线资源不足 [{stage}]: {str(error)}",
                pipeline_id=pipeline_id,
                stage=stage
            )
        else:
            return PipelineError(
                f"管线执行错误 [{stage}]: {str(error)}",
                pipeline_id=pipeline_id,
                stage=stage
            )


class ErrorRecoveryStrategy:
    """错误恢复策略"""
    
    @staticmethod
    def graceful_degradation(error: LumoscribeError, fallback_data: Any = None):
        """优雅降级策略"""
        logger.warning(f"⚠️ 执行优雅降级 - 错误: {error.error_code}")
        
        if fallback_data is not None:
            logger.info("✅ 使用备用数据")
            return fallback_data
        
        # 根据错误类型提供不同的降级策略
        if isinstance(error, (NetworkError, LLMError)):
            return {"status": "degraded", "message": "服务降级，使用缓存数据"}
        elif isinstance(error, DatabaseError):
            return {"status": "degraded", "message": "数据库服务降级"}
        
        return {"status": "error", "message": "服务暂时不可用"}
    
    @staticmethod
    async def circuit_breaker_recovery():
        """断路器恢复策略"""
        logger.info("🔄 执行断路器恢复检查")
        # 这里可以实现具体的恢复逻辑
        await asyncio.sleep(1)  # 模拟恢复时间
        return True


# 全局错误处理器实例
error_handler = ErrorHandler()


def handle_llm_errors(context: str = ""):
    """LLM 错误处理装饰器"""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handled_error = LLMErrorsHandler.handle_llm_error(e, context)
                logger.error(f"LLM 错误处理 [{context}]: {handled_error}")
                raise handled_error
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handled_error = LLMErrorsHandler.handle_llm_error(e, context)
                logger.error(f"LLM 错误处理 [{context}]: {handled_error}")
                raise handled_error
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def handle_pipeline_errors(stage: str = "", pipeline_id: str = ""):
    """管线错误处理装饰器"""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handled_error = PipelineErrorHandler.handle_pipeline_error(e, stage, pipeline_id)
                logger.error(f"管线错误处理 [{stage}]: {handled_error}")
                raise handled_error
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handled_error = PipelineErrorHandler.handle_pipeline_error(e, stage, pipeline_id)
                logger.error(f"管线错误处理 [{stage}]: {handled_error}")
                raise handled_error
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def with_error_handling(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
    enable_circuit_breaker: bool = True
):
    """通用错误处理装饰器"""
    def decorator(func: Callable):
        eh = ErrorHandler(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            retryable_exceptions=retryable_exceptions,
            enable_circuit_breaker=enable_circuit_breaker
        )
        
        async def async_wrapper(*args, **kwargs):
            return await eh.execute_with_retry(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # 对于同步函数，直接使用同步的执行方法
            return eh.execute_with_retry(lambda: func(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class ErrorContext:
    """错误处理上下文管理器
    
    用于简化重复的 try-except 错误处理模式，遵循 DRY 原则。
    
    使用示例：
        with ErrorContext(DatabaseError, operation="query", table="users") as ctx:
            result = db.query(...)
            return result
    """
    
    def __init__(
        self,
        error_class: type[LumoscribeError],
        operation: str | None = None,
        **error_details
    ):
        """初始化错误上下文
        
        Args:
            error_class: 要抛出的错误类型
            operation: 操作名称
            **error_details: 错误详情（如 table, collection, path 等）
        """
        self.error_class = error_class
        self.operation = operation
        self.error_details = error_details
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        # 如果已经是目标错误类型，直接传播
        if isinstance(exc_val, self.error_class):
            return False
        
        # 如果已经是 LumoscribeError 的子类，直接传播
        if isinstance(exc_val, LumoscribeError):
            return False
        
        # 转换为目标错误类型
        error_message = str(exc_val)
        if self.operation:
            error_message = f"{self.operation} 操作失败: {error_message}"
        
        raise self.error_class(
            error_message,
            operation=self.operation,
            **self.error_details
        ) from exc_val


class DatabaseErrorContext(ErrorContext):
    """数据库错误上下文管理器
    
    专门用于数据库操作的错误处理。
    
    使用示例：
        with DatabaseErrorContext(operation="query", table="users"):
            return db.execute(...)
    """
    
    def __init__(self, operation: str, table: str | None = None):
        """初始化数据库错误上下文
        
        Args:
            operation: 数据库操作名称
            table: 表名
        """
        from src.framework.shared.exceptions import DatabaseError
        super().__init__(DatabaseError, operation=operation, table=table)


class VectorStoreErrorContext(ErrorContext):
    """向量存储错误上下文管理器
    
    专门用于向量存储操作的错误处理。
    
    使用示例：
        with VectorStoreErrorContext(operation="add", collection="documents"):
            return vector_store.add(...)
    """
    
    def __init__(self, operation: str, collection: str | None = None):
        """初始化向量存储错误上下文
        
        Args:
            operation: 操作名称
            collection: 集合名称
        """
        from src.framework.shared.exceptions import VectorStoreError
        super().__init__(VectorStoreError, operation=operation, collection=collection)


class GraphStoreErrorContext(ErrorContext):
    """图存储错误上下文管理器
    
    专门用于图存储操作的错误处理。
    """
    
    def __init__(self, operation: str, graph: str | None = None):
        """初始化图存储错误上下文
        
        Args:
            operation: 操作名称
            graph: 图名称
        """
        from src.framework.shared.exceptions import GraphStoreError
        super().__init__(GraphStoreError, operation=operation, graph=graph)