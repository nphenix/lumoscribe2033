"""
共享异常定义

定义项目中使用的自定义异常。
"""

from collections.abc import Callable
from typing import Any


class LumoscribeError(Exception):
    """Lumoscribe 基础异常"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None
    ):
        self.message = message
        self.error_code = error_code or "LUMOSCRIBE_ERROR"
        self.details = details or {}
        self.cause = cause

        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }


class ConfigurationError(LumoscribeError):
    """配置错误"""
    def __init__(self, message: str, config_key: str | None = None):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, "CONFIG_ERROR", details)


class ValidationError(LumoscribeError):
    """验证错误"""
    def __init__(self, message: str, field: str | None = None, value: Any = None):
        details = {"field": field, "value": value} if field else {}
        super().__init__(message, "VALIDATION_ERROR", details)


class DatabaseError(LumoscribeError):
    """数据库错误"""
    def __init__(
        self, message: str, operation: str | None = None, table: str | None = None
    ):
        details = {"operation": operation, "table": table} if operation else {}
        super().__init__(message, "DATABASE_ERROR", details)


class DatabaseGatewayError(LumoscribeError):
    """数据库网关错误"""
    def __init__(
        self, message: str, gateway_type: str | None = None, operation: str | None = None
    ):
        details = {"gateway_type": gateway_type, "operation": operation} if gateway_type else {}
        super().__init__(message, "DATABASE_GATEWAY_ERROR", details)


class StorageError(LumoscribeError):
    """存储错误"""
    def __init__(
        self, message: str, storage_type: str | None = None, path: str | None = None
    ):
        details = {"storage_type": storage_type, "path": path} if storage_type else {}
        super().__init__(message, "STORAGE_ERROR", details)


class VectorStoreError(LumoscribeError):
    """向量存储错误"""
    def __init__(
        self, message: str, collection: str | None = None, operation: str | None = None
    ):
        details = (
            {"collection": collection, "operation": operation} if collection else {}
        )
        super().__init__(message, "VECTOR_STORE_ERROR", details)


class GraphStoreError(LumoscribeError):
    """图存储错误"""
    def __init__(
        self, message: str, graph: str | None = None, operation: str | None = None
    ):
        details = {"graph": graph, "operation": operation} if graph else {}
        super().__init__(message, "GRAPH_STORE_ERROR", details)


class LLMError(LumoscribeError):
    """LLM 服务错误"""
    def __init__(
        self, message: str, model: str | None = None, provider: str | None = None
    ):
        details = {"model": model, "provider": provider} if model else {}
        super().__init__(message, "LLM_ERROR", details)


class APIError(LumoscribeError):
    """API 错误"""
    def __init__(
        self, message: str, status_code: int | None = None, endpoint: str | None = None
    ):
        details = (
            {"status_code": status_code, "endpoint": endpoint} if status_code else {}
        )
        super().__init__(message, "API_ERROR", details)


class AuthenticationError(LumoscribeError):
    """认证错误"""
    def __init__(self, message: str, auth_type: str | None = None):
        details = {"auth_type": auth_type} if auth_type else {}
        super().__init__(message, "AUTH_ERROR", details)


class AuthorizationError(LumoscribeError):
    """授权错误"""
    def __init__(
        self, message: str, permission: str | None = None, resource: str | None = None
    ):
        details = {"permission": permission, "resource": resource} if permission else {}
        super().__init__(message, "AUTHZ_ERROR", details)


class RateLimitError(LumoscribeError):
    """限流错误"""
    def __init__(self, message: str, retry_after: int | None = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class NetworkError(LumoscribeError):
    """网络错误"""
    def __init__(
        self, message: str, url: str | None = None, timeout: float | None = None
    ):
        details = {"url": url, "timeout": timeout} if url else {}
        super().__init__(message, "NETWORK_ERROR", details)


class FileError(LumoscribeError):
    """文件错误"""
    def __init__(
        self, message: str, filepath: str | None = None, operation: str | None = None
    ):
        details = {"filepath": filepath, "operation": operation} if filepath else {}
        super().__init__(message, "FILE_ERROR", details)


class PipelineError(LumoscribeError):
    """管线错误"""
    def __init__(
        self, message: str, pipeline_id: str | None = None, stage: str | None = None
    ):
        details = {"pipeline_id": pipeline_id, "stage": stage} if pipeline_id else {}
        super().__init__(message, "PIPELINE_ERROR", details)


class TaskError(LumoscribeError):
    """任务错误"""
    def __init__(
        self, message: str, task_id: str | None = None, task_name: str | None = None
    ):
        details = {"task_id": task_id, "task_name": task_name} if task_id else {}
        super().__init__(message, "TASK_ERROR", details)


class AgentError(LumoscribeError):
    """AI 代理错误"""
    def __init__(
        self, message: str, agent_type: str | None = None, tool: str | None = None
    ):
        details = {"agent_type": agent_type, "tool": tool} if agent_type else {}
        super().__init__(message, "AGENT_ERROR", details)


class RetrievalError(LumoscribeError):
    """检索错误"""
    def __init__(
        self, message: str, query: str | None = None, retriever: str | None = None
    ):
        details = {"query": query, "retriever": retriever} if query else {}
        super().__init__(message, "RETRIEVAL_ERROR", details)


class RerankError(LumoscribeError):
    """重排序错误"""
    def __init__(
        self, message: str, model: str | None = None, documents_count: int | None = None
    ):
        details = {"model": model, "documents_count": documents_count} if model else {}
        super().__init__(message, "RERANK_ERROR", details)


class RAGError(LumoscribeError):
    """RAG (检索增强生成) 错误"""
    def __init__(
        self, message: str, operation: str | None = None, component: str | None = None
    ):
        details = {"operation": operation, "component": component} if operation else {}
        super().__init__(message, "RAG_ERROR", details)


class ComplianceError(LumoscribeError):
    """合规性检查错误"""
    def __init__(
        self, message: str, check_type: str | None = None, severity: str | None = None
    ):
        details = {"check_type": check_type, "severity": severity} if check_type else {}
        super().__init__(message, "COMPLIANCE_ERROR", details)


class DocumentError(LumoscribeError):
    """文档处理错误"""
    def __init__(
        self, message: str, document_id: str | None = None, operation: str | None = None
    ):
        details = (
            {"document_id": document_id, "operation": operation} if document_id else {}
        )
        super().__init__(message, "DOCUMENT_ERROR", details)


class ConversationError(LumoscribeError):
    """对话处理错误"""
    def __init__(
        self,
        message: str,
        conversation_id: str | None = None,
        source: str | None = None
    ):
        details = {
            "conversation_id": conversation_id, "source": source
        } if conversation_id else {}
        super().__init__(message, "CONVERSATION_ERROR", details)


class IDEError(LumoscribeError):
    """IDE 集成错误"""
    def __init__(
        self, message: str, ide_name: str | None = None, operation: str | None = None
    ):
        details = {"ide_name": ide_name, "operation": operation} if ide_name else {}
        super().__init__(message, "IDE_ERROR", details)


class TelemetryError(LumoscribeError):
    """遥测错误"""
    def __init__(
        self, message: str, component: str | None = None, metric: str | None = None
    ):
        details = {"component": component, "metric": metric} if component else {}
        super().__init__(message, "TELEMETRY_ERROR", details)


class MetricsError(LumoscribeError):
    """指标错误"""
    def __init__(
        self, message: str, metric_name: str | None = None, operation: str | None = None
    ):
        details = (
            {"metric_name": metric_name, "operation": operation} if metric_name else {}
        )
        super().__init__(message, "METRICS_ERROR", details)


class CacheError(LumoscribeError):
    """缓存错误"""
    def __init__(
        self, message: str, cache_key: str | None = None, operation: str | None = None
    ):
        details = {"cache_key": cache_key, "operation": operation} if cache_key else {}
        super().__init__(message, "CACHE_ERROR", details)


class IndexServiceError(LumoscribeError):
    """索引服务错误"""
    def __init__(
        self, message: str, service: str | None = None, operation: str | None = None
    ):
        details = {"service": service, "operation": operation} if service else {}
        super().__init__(message, "INDEX_SERVICE_ERROR", details)


# 异常处理装饰器
def handle_exceptions(
    exception_map: dict[type[Exception], type[LumoscribeError]] | None = None,
    default_exception: type[LumoscribeError] = LumoscribeError
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """异常处理装饰器"""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 查找匹配的异常映射
                exception_type = type(e)
                if exception_map and exception_type in exception_map:
                    lumoscribe_exc_class = exception_map[exception_type]
                    raise lumoscribe_exc_class(str(e), cause=e) from e
                else:
                    # 检查是否是子类匹配
                    for exc_class, lumoscribe_class in (exception_map or {}).items():
                        if isinstance(e, exc_class):
                            raise lumoscribe_class(str(e), cause=e) from e

                    # 默认处理
                    raise default_exception(str(e), cause=e) from e
        return wrapper
    return decorator


# LangChain 兼容的异常处理装饰器
def handle_llm_exceptions(
    context: str = "",
    retryable: bool = True,
    max_retries: int = 3
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """LangChain LLM 调用异常处理装饰器"""
    from src.framework.shared.error_handler import handle_llm_errors
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return handle_llm_errors(context)(func)
    return decorator


def handle_pipeline_exceptions(
    stage: str = "",
    pipeline_id: str = "",
    enable_retry: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """管线执行异常处理装饰器"""
    from src.framework.shared.error_handler import handle_pipeline_errors
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if enable_retry:
            from src.framework.shared.error_handler import with_error_handling
            func = with_error_handling()(func)
        return handle_pipeline_errors(stage, pipeline_id)(func)
    return decorator


# 错误码常量
ERROR_CODES = {
    "CONFIG_ERROR": "配置错误",
    "VALIDATION_ERROR": "验证错误",
    "DATABASE_ERROR": "数据库错误",
    "STORAGE_ERROR": "存储错误",
    "VECTOR_STORE_ERROR": "向量存储错误",
    "GRAPH_STORE_ERROR": "图存储错误",
    "LLM_ERROR": "LLM 服务错误",
    "API_ERROR": "API 错误",
    "AUTH_ERROR": "认证错误",
    "AUTHZ_ERROR": "授权错误",
    "RATE_LIMIT_ERROR": "限流错误",
    "NETWORK_ERROR": "网络错误",
    "FILE_ERROR": "文件错误",
    "PIPELINE_ERROR": "管线错误",
    "TASK_ERROR": "任务错误",
    "AGENT_ERROR": "AI 代理错误",
    "RETRIEVAL_ERROR": "检索错误",
    "RERANK_ERROR": "重排序错误",
    "RAG_ERROR": "RAG (检索增强生成) 错误",
    "COMPLIANCE_ERROR": "合规性检查错误",
    "DOCUMENT_ERROR": "文档处理错误",
    "CONVERSATION_ERROR": "对话处理错误",
    "IDE_ERROR": "IDE 集成错误",
    "TELEMETRY_ERROR": "遥测错误",
    "METRICS_ERROR": "指标错误",
    "CACHE_ERROR": "缓存错误",
    "INDEX_SERVICE_ERROR": "索引服务错误"
}
