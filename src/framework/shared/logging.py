"""
结构化日志配置模块

基于 structlog + OpenTelemetry 最佳实践，为 lumoscribe2033 项目提供统一的日志解决方案。
支持本地开发和生产环境的不同输出格式，集成 OpenTelemetry 追踪上下文。

主要特性:
- 结构化日志输出（JSON/控制台）
- OpenTelemetry 追踪上下文集成
- 环境自适应格式化（本地 vs 生产）
- 性能优化和异步支持
- 丰富的处理器链配置

使用方法:
    from src.framework.shared.logging import get_logger

    logger = get_logger(__name__)
    logger.info("操作成功", user_id="123", action="login")
    logger.error("操作失败", error_code="E001", exc_info=True)

环境变量:
    LOG_LEVEL: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    LOG_FORMAT: 日志格式 (json, console, rich)
    LOG_ENABLE_TRACE_LOGGING: 是否启用追踪日志
    OTEL_LOG_LEVEL: OpenTelemetry 日志级别
"""

import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

import structlog
from structlog.processors import CallsiteParameter
from structlog.types import EventDict, Processor

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanContext, format_trace_id
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

    class MockSpanContext:
        def __getattr__(self, name: str) -> Any:
            return None

    SpanContext = MockSpanContext  # type: ignore[assignment]

# 线程本地存储用于追踪上下文
_local = threading.local()


class LumoscribeLoggingConfig:
    """lumoscribe2033 日志配置类"""

    def __init__(self) -> None:
        self.log_level = self._get_log_level()
        self.log_format = self._get_log_format()
        self.enable_trace_logging = self._get_trace_logging_flag()
        self.is_production = self._is_production_environment()

    def _get_log_level(self) -> int:
        """获取日志级别"""
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        return getattr(logging, level, logging.INFO)

    def _get_log_format(self) -> str:
        """获取日志格式"""
        return os.getenv("LOG_FORMAT", "console").lower()

    def _get_trace_logging_flag(self) -> bool:
        """获取是否启用追踪日志"""
        return os.getenv("LOG_ENABLE_TRACE_LOGGING", "false").lower() == "true"

    def _is_production_environment(self) -> bool:
        """判断是否为生产环境"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"


class OpenTelemetryContextProcessor:
    """OpenTelemetry 追踪上下文处理器"""

    def __call__(
        self, logger: logging.Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """添加追踪上下文到日志事件"""
        if not OTEL_AVAILABLE:
            return event_dict

        try:
            # 获取当前 span
            current_span = trace.get_current_span()
            if current_span and hasattr(current_span, 'get_span_context'):
                span_context: SpanContext = current_span.get_span_context()
                if span_context and span_context.is_valid:
                    # 添加追踪信息
                    event_dict["trace_id"] = format_trace_id(span_context.trace_id)
                    event_dict["span_id"] = format_trace_id(span_context.span_id)

                    # 尝试获取 span 名称
                    span_name = getattr(current_span, 'name', None)
                    if span_name:
                        event_dict["span_name"] = span_name

            # 添加线程信息
            thread_id = threading.get_ident()
            event_dict["thread_id"] = thread_id

            # 添加协程信息（如果可用）
            try:
                import asyncio
                task = asyncio.current_task()
                if task:
                    event_dict["asyncio_task"] = str(task.get_name())
            except (ImportError, RuntimeError):
                pass

        except Exception:
            # 静默失败，不影响主流程
            pass

        return event_dict


class PerformanceMetricsProcessor:
    """性能指标处理器"""

    def __init__(self, enable_timing: bool = True) -> None:
        self.enable_timing = enable_timing

    def __call__(
        self, logger: logging.Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """添加性能指标到日志事件"""
        if not self.enable_timing:
            return event_dict

        try:
            import time
            current_time = time.time()

            # 计算处理时间（如果之前有时间戳）
            if hasattr(_local, 'start_time'):
                processing_time = current_time - _local.start_time
                event_dict["processing_time_ms"] = round(processing_time * 1000, 2)
                delattr(_local, 'start_time')

            # 为下一个日志事件设置开始时间
            _local.start_time = current_time

        except Exception:
            pass

        return event_dict


class SensitiveDataFilter:
    """敏感数据过滤器"""

    SENSITIVE_KEYS = {
        'password', 'passwd', 'pwd', 'secret', 'token', 'api_key',
        'access_token', 'refresh_token', 'session_id', 'auth_token',
        'private_key', 'certificate', 'ssn', 'social_security',
        'credit_card', 'cc_number', 'cvv', 'cvv2'
    }

    def __call__(
        self, logger: logging.Logger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """过滤敏感数据"""
        for key, value in event_dict.items():
            if key.lower() in self.SENSITIVE_KEYS and value:
                if isinstance(value, str):
                    # 保留前4位，其余用*替换
                    if len(value) > 4:
                        if len(value) > 8:
                            event_dict[key] = (
                                value[:4] + "*" * (len(value) - 8) + value[-4:]
                            )
                        else:
                            event_dict[key] = value[:2] + "*" * (len(value) - 2)
                    else:
                        event_dict[key] = "*" * len(value)
                elif isinstance(value, (int, float)):
                    event_dict[key] = "***"

        return event_dict


def _get_shared_processors() -> list[Processor]:
    """获取共享处理器链"""
    processors = []

    # 添加日志级别
    processors.append(structlog.processors.add_log_level)

    # 添加时间戳
    processors.append(structlog.processors.TimeStamper(fmt="iso"))

    # 添加调用点信息（文件、行号、函数名）
    processors.append(structlog.processors.CallsiteParameterAdder({
        CallsiteParameter.FILENAME,
        CallsiteParameter.FUNC_NAME,
        CallsiteParameter.LINENO,
        CallsiteParameter.THREAD_NAME,
    }))

    # 添加 OpenTelemetry 追踪上下文
    processors.append(OpenTelemetryContextProcessor())

    # 添加性能指标
    processors.append(PerformanceMetricsProcessor())

    # 过滤敏感数据
    processors.append(SensitiveDataFilter())

    # 异常信息处理
    processors.append(structlog.processors.format_exc_info)

    # Unicode 解码
    processors.append(structlog.processors.UnicodeDecoder())

    return processors


def _get_json_processors() -> list[Processor]:
    """获取 JSON 输出处理器链"""
    processors = _get_shared_processors()

    # 添加结构化异常信息
    processors.append(structlog.processors.dict_tracebacks)

    # JSON 渲染器
    processors.append(structlog.processors.JSONRenderer())

    return processors


def _get_console_processors() -> list[Processor]:
    """获取控制台输出处理器链"""
    processors = _get_shared_processors()

    # 条件性使用彩色控制台渲染器
    if sys.stderr.isatty():
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    return processors


def _get_rich_processors() -> list[Processor]:
    """获取 Rich 输出处理器链"""
    try:
        from rich.traceback import install

        # 安装 Rich traceback
        install()

        processors = _get_shared_processors()

        # 移除调用点信息（RichHandler 会自己处理）
        processors = [
            p for p in processors
            if not isinstance(p, structlog.processors.CallsiteParameterAdder)
        ]

        return processors

    except ImportError:
        # 回退到控制台渲染器
        return _get_console_processors()


def configure_structlog() -> None:
    """配置 structlog"""
    config = LumoscribeLoggingConfig()

    # 配置标准 logging 模块
    logging.basicConfig(
        level=config.log_level,
        format="%(message)s",  # structlog 会处理格式
        stream=sys.stdout,
    )

    # 根据环境选择处理器
    if config.log_format == "json":
        processors = _get_json_processors()
    elif config.log_format == "rich":
        processors = _get_rich_processors()
    else:  # console
        processors = _get_console_processors()

    # 配置 structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
        context_class=dict,
    )

    # 配置 OpenTelemetry 日志级别
    if config.enable_trace_logging and OTEL_AVAILABLE:
        otel_log_level = os.getenv("OTEL_LOG_LEVEL", "WARNING").upper()
        otel_level = getattr(logging, otel_log_level, logging.WARNING)

        # 设置 OpenTelemetry 相关注释
        logging.getLogger("opentelemetry").setLevel(otel_level)
        logging.getLogger("azure").setLevel(otel_level)  # Azure OpenAI
        logging.getLogger("httpx").setLevel(otel_level)   # HTTP 客户端
        logging.getLogger("httpcore").setLevel(otel_level)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    获取配置好的日志记录器

    Args:
        name: 日志记录器名称，默认为调用者的模块名

    Returns:
        structlog.BoundLogger: 配置好的日志记录器
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame is None:
            name = "__main__"
        else:
            # 优先取调用者模块名
            caller_globals = frame.f_back.f_globals if frame.f_back else None  # type: ignore[union-attr]
            name = caller_globals['__name__'] if caller_globals and '__name__' in caller_globals else "__main__"

    return structlog.get_logger(name)


def get_typed_logger(logger_type: str, **kwargs: Any) -> structlog.BoundLogger:
    """
    获取带有类型标识的日志记录器

    Args:
        logger_type: 日志类型标识
        **kwargs: 额外的绑定参数

    Returns:
        structlog.BoundLogger: 带有类型标识的日志记录器
    """
    logger = get_logger()
    logger = logger.bind(logger_type=logger_type, **kwargs)
    return logger


def setup_request_logger() -> structlog.BoundLogger:
    """
    设置请求级别的日志记录器（用于 Web 请求）

    Returns:
        structlog.BoundLogger: 请求日志记录器
    """
    return get_typed_logger("request")


def setup_task_logger() -> structlog.BoundLogger:
    """
    设置任务级别的日志记录器（用于异步任务）

    Returns:
        structlog.BoundLogger: 任务日志记录器
    """
    return get_typed_logger("task")


def setup_agent_logger() -> structlog.BoundLogger:
    """
    设置 AI 代理级别的日志记录器

    Returns:
        structlog.BoundLogger: 代理日志记录器
    """
    return get_typed_logger("agent")


def bind_context(**kwargs: Any) -> None:
    """
    绑定上下文信息到当前线程的所有日志

    Args:
        **kwargs: 要绑定的上下文参数
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """清除当前线程的上下文信息"""
    structlog.contextvars.clear_contextvars()


def log_execution_time(
    func_name: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    装饰器：记录函数执行时间

    Args:
        func_name: 函数名称标识

    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_typed_logger("performance", function=func_name)
            logger.info(f"开始执行: {func_name}")

            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"执行完成: {func_name}",
                    execution_time_ms=round(execution_time * 1000, 2),
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"执行失败: {func_name}",
                    execution_time_ms=round(execution_time * 1000, 2),
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        return wrapper
    return decorator


# 导入时自动配置
configure_structlog()

# 创建模块级别的日志记录器
logger = get_logger(__name__)

# 记录配置信息
config = LumoscribeLoggingConfig()
logger.info(
    "日志系统初始化完成",
    log_level=logging.getLevelName(config.log_level),
    log_format=config.log_format,
    is_production=config.is_production,
    otel_available=OTEL_AVAILABLE
)
