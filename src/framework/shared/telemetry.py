"""
可观测性配置模块

基于 OpenTelemetry 最佳实践，为 lumoscribe2033 项目提供完整的可观测性解决方案。
集成分布式追踪、指标收集、日志关联，支持多种后端导出器。

主要特性:
- 分布式追踪（Traces）
- 指标收集（Metrics）
- 日志关联（Logs）
- 自动仪器化
- 多后端支持（Jaeger, Prometheus, OTLP）

使用方法:
    from src.framework.shared.telemetry import setup_telemetry

    # 在应用启动时调用
    setup_telemetry(service_name="lumoscribe-api",
                   service_version="0.1.0")

环境变量:
    OTEL_SERVICE_NAME: 服务名称
    OTEL_SERVICE_VERSION: 服务版本
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP 导出端点
    OTEL_EXPORTER_OTLP_HEADERS: OTLP 导出头部
    OTEL_TRACES_EXPORTER: 追踪导出器 (jaeger, otlp, console)
    OTEL_METRICS_EXPORTER: 指标导出器 (prometheus, otlp, console)
    OTEL_LOGS_EXPORTER: 日志导出器 (otlp, console)
    OTEL_RESOURCE_ATTRIBUTES: 资源属性
    JAEGER_ENDPOINT: Jaeger 端点
    PROMETHEUS_PORT: Prometheus 端口
"""

import os
import sys
import threading
from datetime import datetime, timedelta
from typing import Any

try:
    from opentelemetry import metrics, trace
    # 日志功能在 OpenTelemetry 1.38.0 中可能不可用或需要额外包
    # 先尝试导入，如果失败则设置为 None
    LoggerProvider = None
    BatchLogRecordProcessor = None
    ConsoleLogRecordExporter = None
    logs = None

    try:
        from opentelemetry.sdk.logs import LoggerProvider
        from opentelemetry.sdk.logs.export import (
            BatchLogRecordProcessor,
            ConsoleLogRecordExporter,
        )
        logs = True  # 标记日志功能可用
    except ImportError:
        # 尝试从其他可能的路径导入
        try:
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk.logs.export import (
                BatchLogRecordProcessor,
                ConsoleLogRecordExporter,
            )
            logs = True  # 标记日志功能可用
        except ImportError:
            # 日志功能不可用，但其他功能仍然可用
            logs = None
            LoggerProvider = None
            BatchLogRecordProcessor = None
            ConsoleLogRecordExporter = None

    try:
        from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter
    except ImportError:
        OTLPLogExporter = None

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
    except ImportError:
        OTLPMetricExporter = None

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        OTLPSpanExporter = None

    # 修复仪器化包的导入，使用正确的包名
    try:
        from opentelemetry.instrumentation.arq import ArqInstrumentor
    except ImportError:
        ArqInstrumentor = None

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        FastAPIInstrumentor = None

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        HTTPXClientInstrumentor = None

    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
    except ImportError:
        LoggingInstrumentor = None

    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    except ImportError:
        Psycopg2Instrumentor = None

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
    except ImportError:
        RedisInstrumentor = None

    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except ImportError:
        RequestsInstrumentor = None

    try:
        from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
    except ImportError:
        SQLite3Instrumentor = None

    try:
        from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
    except ImportError:
        URLLib3Instrumentor = None

    try:
        from opentelemetry.sdk.metrics import MeterProvider
    except ImportError:
        MeterProvider = None

    try:
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
        # 验证 ConsoleMetricExporter 是否有必要的属性
        if hasattr(ConsoleMetricExporter, '_instrument_class_temporality'):
            pass  # 已经有正确的属性
        else:
            # 在某些版本中，ConsoleMetricExporter 可能缺少某些属性
            # 我们可以创建一个简单的包装器或使用其他导出器
            ConsoleMetricExporter = None
    except ImportError:
        ConsoleMetricExporter = None

    try:
        from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    except ImportError:
        SERVICE_NAME = None
        SERVICE_VERSION = None
        Resource = None

    try:
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        TracerProvider = None

    try:
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
    except ImportError:
        BatchSpanProcessor = None
        ConsoleSpanExporter = None

    try:
        from opentelemetry.semconv.resource import ResourceAttributes
    except ImportError:
        ResourceAttributes = None

    try:
        from opentelemetry.trace.status import Status, StatusCode
    except ImportError:
        Status = None
        StatusCode = None

    # 检查核心组件是否都可用
    core_components_available = all([
        metrics, trace, MeterProvider, TracerProvider,
        BatchSpanProcessor, ConsoleSpanExporter, Resource
    ])

    if core_components_available:
        OTEL_AVAILABLE = True
    else:
        OTEL_AVAILABLE = False
        print("OpenTelemetry 核心组件不完整，可观测性功能将被禁用")
except ImportError as e:
    OTEL_AVAILABLE = False
    print(f"OpenTelemetry 导入失败: {e}")
    # 创建空的占位符类
    class MockClass:
        def __getattr__(self, name: str) -> Any:
            return lambda *args, **kwargs: None
    # 创建空的占位符类
    class MockTrace:
        def get_tracer(self, name: str) -> MockClass:
            return MockClass()

        def set_tracer_provider(self, provider: Any) -> None:
            pass

    class MockMetrics:
        def get_meter(self, name: str, version: str | None = None, description: str | None = None) -> MockClass:
            return MockClass()

        def set_meter_provider(self, provider: Any) -> None:
            pass

    class MockLogs:
        def set_logger_provider(self, provider: Any) -> None:
            pass

    trace = MockTrace()  # type: ignore[assignment]
    metrics = MockMetrics()  # type: ignore[assignment]
    logs = MockLogs()  # type: ignore[assignment]

from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class TelemetryConfig:
    """可观测性配置类"""

    def __init__(self) -> None:
        self.service_name = self._get_service_name()
        self.service_version = self._get_service_version()
        self.environment = self._get_environment()
        self.otlp_endpoint = self._get_otlp_endpoint()
        self.otlp_headers = self._get_otlp_headers()
        self.enable_tracing = self._get_flag("OTEL_TRACING_ENABLED", True)
        self.enable_metrics = self._get_flag("OTEL_METRICS_ENABLED", True)
        self.enable_logs = self._get_flag("OTEL_LOGS_ENABLED", True)
        self.traces_exporter = self._get_traces_exporter()
        self.metrics_exporter = self._get_metrics_exporter()
        self.logs_exporter = self._get_logs_exporter()
        self.prometheus_port = self._get_prometheus_port()

    def _get_service_name(self) -> str:
        """获取服务名称"""
        return os.getenv("OTEL_SERVICE_NAME", "lumoscribe2033")

    def _get_service_version(self) -> str:
        """获取服务版本"""
        return os.getenv("OTEL_SERVICE_VERSION", "0.1.0")

    def _get_environment(self) -> str:
        """获取环境"""
        return os.getenv("ENVIRONMENT", "development")

    def _get_otlp_endpoint(self) -> str | None:
        """获取 OTLP 端点"""
        return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    def _get_otlp_headers(self) -> str | None:
        """获取 OTLP 头部"""
        return os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

    def _get_flag(self, env_var: str, default: bool) -> bool:
        """获取布尔标志"""
        value = os.getenv(env_var, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    def _get_traces_exporter(self) -> str:
        """获取追踪导出器"""
        return os.getenv("OTEL_TRACES_EXPORTER", "console")

    def _get_metrics_exporter(self) -> str:
        """获取指标导出器"""
        return os.getenv("OTEL_METRICS_EXPORTER", "console")

    def _get_logs_exporter(self) -> str:
        """获取日志导出器"""
        return os.getenv("OTEL_LOGS_EXPORTER", "console")

    def _get_prometheus_port(self) -> int:
        """获取 Prometheus 端口"""
        return int(os.getenv("PROMETHEUS_PORT", "9464"))


class LumoscribeSpanProcessor:
    """lumoscribe2033 自定义 Span 处理器"""

    def __init__(self) -> None:
        self.config = TelemetryConfig()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        """Span 开始时的处理"""
        try:
            # 添加通用属性
            span.set_attribute("service.name", self.config.service_name)
            span.set_attribute("service.version", self.config.service_version)
            span.set_attribute("service.environment", self.config.environment)

            # 添加主机信息
            import socket
            span.set_attribute("host.name", socket.gethostname())

            # 添加进程信息
            import os
            span.set_attribute("process.pid", os.getpid())

            # 添加线程信息
            span.set_attribute("thread.id", threading.get_ident())
            span.set_attribute("thread.name", threading.current_thread().name)

        except Exception as e:
            logger.warning(f"Span 开始处理失败: {e}")

    def on_end(self, span: Any) -> None:
        """Span 结束时的处理"""
        try:
            # 添加结束时间戳
            span.set_attribute("span.end_time", span.end_time.isoformat())

        except Exception as e:
            logger.warning(f"Span 结束处理失败: {e}")


class TelemetryMetrics:
    """自定义指标收集器"""

    def __init__(self) -> None:
        self.config = TelemetryConfig()
        self._setup_custom_metrics()

    def _setup_custom_metrics(self) -> None:
        """设置自定义指标"""
        if not OTEL_AVAILABLE:
            return

        try:
            meter = metrics.get_meter(
                name="lumoscribe2033",
                version="0.1.0"
            )

            # 请求计数器
            self.request_counter = meter.create_counter(
                name="lumoscribe_requests_total",
                description="总请求数",
                unit="1"
            )

            # 请求持续时间直方图
            self.request_duration = meter.create_histogram(
                name="lumoscribe_request_duration_seconds",
                description="请求持续时间",
                unit="s"
            )

            # AI 代理调用计数器
            self.agent_counter = meter.create_counter(
                name="lumoscribe_agent_calls_total",
                description="AI 代理调用总数",
                unit="1"
            )

            # RAG 查询计数器
            self.rag_query_counter = meter.create_counter(
                name="lumoscribe_rag_queries_total",
                description="RAG 查询总数",
                unit="1"
            )

            # 向量搜索延迟直方图
            self.vector_search_duration = meter.create_histogram(
                name="lumoscribe_vector_search_duration_seconds",
                description="向量搜索持续时间",
                unit="s"
            )

            # 图查询计数器
            self.graph_query_counter = meter.create_counter(
                name="lumoscribe_graph_queries_total",
                description="图查询总数",
                unit="1"
            )

            # 错误计数器
            self.error_counter = meter.create_counter(
                name="lumoscribe_errors_total",
                description="错误总数",
                unit="1"
            )

            # 文档处理计数器
            self.document_processed_counter = meter.create_counter(
                name="lumoscribe_documents_processed_total",
                description="处理的文档总数",
                unit="1"
            )

        except Exception as e:
            logger.error(f"设置自定义指标失败: {e}")

    def record_request(
        self, method: str, route: str, duration: float, status_code: int
    ) -> None:
        """记录请求指标"""
        if not OTEL_AVAILABLE:
            return

        try:
            self.request_counter.add(
                1,
                {
                    "http.method": method,
                    "http.route": route,
                    "http.status_code": status_code
                }
            )
            self.request_duration.record(
                duration, {"http.method": method, "http.route": route}
            )
        except Exception as e:
            logger.warning(f"记录请求指标失败: {e}")

    def record_agent_call(self, agent_type: str, success: bool) -> None:
        """记录 AI 代理调用"""
        if not OTEL_AVAILABLE:
            return

        try:
            self.agent_counter.add(
                1,
                {
                    "agent.type": agent_type,
                    "agent.success": str(success)
                }
            )
        except Exception as e:
            logger.warning(f"记录代理调用失败: {e}")

    def record_rag_query(self, collection: str, duration: float, success: bool) -> None:
        """记录 RAG 查询"""
        if not OTEL_AVAILABLE:
            return

        try:
            self.rag_query_counter.add(
                1,
                {
                    "rag.collection": collection,
                    "rag.success": str(success)
                }
            )
            self.vector_search_duration.record(duration, {"rag.collection": collection})
        except Exception as e:
            logger.warning(f"记录 RAG 查询失败: {e}")

    def record_error(self, error_type: str, component: str) -> None:
        """记录错误"""
        if not OTEL_AVAILABLE:
            return

        try:
            self.error_counter.add(
                1,
                {
                    "error.type": error_type,
                    "error.component": component
                }
            )
        except Exception as e:
            logger.warning(f"记录错误失败: {e}")

    def record_document_processed(self, document_type: str, success: bool) -> None:
        """记录文档处理"""
        if not OTEL_AVAILABLE:
            return

        try:
            self.document_processed_counter.add(
                1,
                {
                    "document.type": document_type,
                    "document.success": str(success)
                }
            )
        except Exception as e:
            logger.warning(f"记录文档处理失败: {e}")


class SystemMetricsCollector:
    """系统指标收集器"""

    def __init__(self) -> None:
        self._metrics_history: dict[str, list] = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_bytes_sent": [],
            "network_bytes_recv": [],
        }
        self._max_history_size = 1000

    def collect_system_metrics(self) -> dict[str, Any]:
        """收集系统资源指标"""
        try:
            import psutil

            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用情况
            memory = psutil.virtual_memory()

            # 磁盘使用情况
            disk = psutil.disk_usage('/')

            # 网络统计
            network = psutil.net_io_counters()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "disk_usage_percent": disk.percent,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "process_count": len(psutil.pids()),
            }

            # 更新历史记录
            for key in self._metrics_history:
                if key in metrics:
                    self._metrics_history[key].append(metrics[key])
                    if len(self._metrics_history[key]) > self._max_history_size:
                        self._metrics_history[key].pop(0)

            return metrics

        except ImportError:
            logger.warning("psutil 未安装，无法收集系统指标")
            return {}

    def get_system_health(self) -> dict[str, Any]:
        """获取系统健康状态"""
        current_metrics = self.collect_system_metrics()

        if not current_metrics:
            return {"status": "unknown", "reason": "无法收集系统指标"}

        # 健康检查规则
        health_issues = []

        if current_metrics["cpu_percent"] > 90:
            health_issues.append("CPU 使用率过高")

        if current_metrics["memory_percent"] > 80:
            health_issues.append("内存使用率过高")

        if current_metrics["disk_usage_percent"] > 90:
            health_issues.append("磁盘使用率过高")

        status = "healthy" if not health_issues else "warning" if len(health_issues) == 1 else "critical"

        return {
            "status": status,
            "issues": health_issues,
            "metrics": current_metrics,
            "timestamp": current_metrics["timestamp"]
        }


class RequestMetricsCollector:
    """请求指标收集器"""

    def __init__(self) -> None:
        self._request_history: list[dict[str, Any]] = []
        self._max_history_size = 10000
        self._rate_limiter: dict[str, list[float]] = {}

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: float,
        client_ip: str = ""
    ) -> None:
        """记录请求指标"""
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": response_time,
            "client_ip": client_ip,
        }

        self._request_history.append(request_data)

        if len(self._request_history) > self._max_history_size:
            self._request_history.pop(0)

    def check_rate_limit(self, client_ip: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
        """检查速率限制"""
        now = datetime.now().timestamp()

        if client_ip not in self._rate_limiter:
            self._rate_limiter[client_ip] = []

        # 清理过期请求
        self._rate_limiter[client_ip] = [
            req_time for req_time in self._rate_limiter[client_ip]
            if now - req_time < window_seconds
        ]

        # 检查是否超过限制
        if len(self._rate_limiter[client_ip]) >= max_requests:
            return False

        # 记录当前请求
        self._rate_limiter[client_ip].append(now)
        return True

    def get_request_stats(self, hours: int = 24) -> dict[str, Any]:
        """获取请求统计"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_requests = [
            req for req in self._request_history
            if datetime.fromisoformat(req["timestamp"]) > cutoff
        ]

        if not recent_requests:
            return {"total_requests": 0}

        # 统计数据
        status_codes = {}
        endpoints = {}
        response_times = []

        for req in recent_requests:
            # 状态码统计
            status = str(req["status_code"])
            status_codes[status] = status_codes.get(status, 0) + 1

            # 端点统计
            endpoint = req["endpoint"]
            endpoints[endpoint] = endpoints.get(endpoint, 0) + 1

            # 响应时间
            response_times.append(req["response_time"])

        # 计算成功率
        success_count = sum(count for status, count in status_codes.items()
                          if status.startswith(('2', '3')))
        success_rate = (success_count / len(recent_requests)) * 100

        return {
            "total_requests": len(recent_requests),
            "success_rate": success_rate,
            "status_codes": status_codes,
            "top_endpoints": dict(sorted(endpoints.items(), key=lambda x: x[1], reverse=True)[:10]),
            "response_time_stats": {
                "avg": sum(response_times) / len(response_times) if response_times else 0,
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
            } if response_times else {},
        }


class TelemetryManager:
    """可观测性管理器"""

    def __init__(self) -> None:
        self.config = TelemetryConfig()
        self.metrics_collector = TelemetryMetrics()
        self.system_collector = SystemMetricsCollector()
        self.request_collector = RequestMetricsCollector()
        self._setup_done = False

    def setup_telemetry(self) -> None:
        """设置完整的可观测性"""
        if not OTEL_AVAILABLE or self._setup_done:
            return

        try:
            # 创建资源
            resource = self._create_resource()

            # 设置追踪
            if self.config.enable_tracing:
                self._setup_tracing(resource)

            # 设置指标
            if self.config.enable_metrics:
                self._setup_metrics(resource)

            # 设置日志
            if self.config.enable_logs:
                self._setup_logs(resource)

            # 启动 Prometheus 服务器（如果需要）
            if self.config.metrics_exporter == "prometheus":
                self._start_prometheus_server()

            # 设置自动仪器化
            self._setup_auto_instrumentation()

            self._setup_done = True

            logger.info(
                "可观测性系统初始化完成",
                service_name=self.config.service_name,
                service_version=self.config.service_version,
                tracing_enabled=self.config.enable_tracing,
                metrics_enabled=self.config.enable_metrics,
                logs_enabled=self.config.enable_logs
            )

        except Exception as e:
            logger.error(f"可观测性系统初始化失败: {e}")
            raise

    def _create_resource(self) -> Any:
        """创建资源对象"""
        resource_attributes = {
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
        }

        # 添加额外的资源属性
        resource_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES")
        if resource_attrs:
            try:
                for attr in resource_attrs.split(","):
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        resource_attributes[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"解析资源属性失败: {e}")

        return Resource.create(resource_attributes)

    def _setup_tracing(self, resource: Any) -> None:
        """设置分布式追踪"""
        try:
            # 检查必要组件
            if not TracerProvider:
                logger.error("TracerProvider 不可用，无法设置追踪")
                return

            if not BatchSpanProcessor:
                logger.error("BatchSpanProcessor 不可用，无法设置追踪")
                return

            # 创建 TracerProvider
            trace_provider = TracerProvider(resource=resource)

            # 添加自定义处理器
            # span_processor = LumoscribeSpanProcessor()  # 暂时注释掉
            span_processor = None

            # 添加导出器
            trace_exporter = None
            if self.config.traces_exporter == "otlp" and self.config.otlp_endpoint and OTLPSpanExporter:
                try:
                    trace_exporter = OTLPSpanExporter(
                        endpoint=self.config.otlp_endpoint,
                        headers=self.config.otlp_headers
                    )
                except Exception as e:
                    logger.warning(f"OTLP 追踪导出器创建失败: {e}")

            elif self.config.traces_exporter == "jaeger":
                try:
                    from opentelemetry.exporter.jaeger import JaegerExporter
                    jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268")
                    trace_exporter = JaegerExporter(
                        agent_host_name=jaeger_endpoint.split(":")[0],
                        agent_port=int(jaeger_endpoint.split(":")[1])
                    )
                except (ImportError, Exception) as e:
                    logger.warning(f"Jaeger 导出器不可用: {e}，使用控制台导出器")
                    trace_exporter = ConsoleSpanExporter()
            else:  # console
                trace_exporter = ConsoleSpanExporter()

            # 如果导出器创建失败，使用控制台导出器
            if not trace_exporter:
                trace_exporter = ConsoleSpanExporter()

            if span_processor:
                trace_provider.add_span_processor(span_processor)
            else:
                batch_processor = BatchSpanProcessor(trace_exporter)
                trace_provider.add_span_processor(batch_processor)

            # 设置全局 TracerProvider
            trace.set_tracer_provider(trace_provider)

            logger.info(f"追踪已启用，导出器: {self.config.traces_exporter}")

        except Exception as e:
            logger.error(f"设置追踪失败: {e}")

    def _setup_metrics(self, resource: Any) -> None:
        """设置指标收集"""
        try:
            # 检查必要组件
            if not MeterProvider:
                logger.error("MeterProvider 不可用，无法设置指标")
                return

            # 创建指标读取器
            metric_reader = None
            if self.config.metrics_exporter == "otlp" and self.config.otlp_endpoint and OTLPMetricExporter:
                try:
                    metric_reader = OTLPMetricExporter(
                        endpoint=self.config.otlp_endpoint,
                        headers=self.config.otlp_headers
                    )
                except Exception as e:
                    logger.warning(f"OTLP 指标导出器创建失败: {e}")

            elif self.config.metrics_exporter == "prometheus":
                try:
                    from opentelemetry.exporter.prometheus import PrometheusMetricReader
                    metric_reader = PrometheusMetricReader()
                except (ImportError, Exception) as e:
                    logger.warning(f"Prometheus 导出器不可用: {e}，跳过指标设置")
                    return
            else:  # console
                # 对于控制台导出，我们可以使用 ConsoleMetricExporter（如果可用）
                # 或者创建一个简单的控制台导出器
                if ConsoleMetricExporter:
                    try:
                        metric_reader = ConsoleMetricExporter()
                    except Exception as e:
                        logger.warning(f"ConsoleMetricExporter 创建失败: {e}，跳过指标设置")
                        return
                else:
                    logger.warning("控制台指标导出器不可用，跳过指标设置")
                    return

            # 创建 MeterProvider
            if self.config.metrics_exporter == "prometheus" and metric_reader:
                # Prometheus 需要特殊的设置
                try:
                    from prometheus_client import start_http_server
                    start_http_server(port=self.config.prometheus_port)
                    provider = MeterProvider(
                        resource=resource,
                        metric_readers=[metric_reader]
                    )
                except (ImportError, Exception) as e:
                    logger.warning(f"Prometheus 服务器启动失败: {e}，跳过指标设置")
                    return
            elif metric_reader:
                provider = MeterProvider(
                    resource=resource,
                    metric_readers=[metric_reader]
                )
            else:
                logger.warning("没有可用的指标读取器，跳过指标设置")
                return

            metrics.set_meter_provider(provider)

            logger.info(f"指标已启用，导出器: {self.config.metrics_exporter}")

        except Exception as e:
            logger.error(f"设置指标失败: {e}")

    def _setup_logs(self, resource: Any) -> None:
        """设置日志收集"""
        try:
            # 检查日志功能是否可用
            if not logs or LoggerProvider is None or BatchLogRecordProcessor is None or ConsoleLogRecordExporter is None:
                logger.warning("日志功能不可用，跳过日志设置")
                return

            # 创建 LoggerProvider
            logger_provider = LoggerProvider(resource=resource)

            # 添加导出器
            log_exporter = None
            if self.config.logs_exporter == "otlp" and self.config.otlp_endpoint and OTLPLogExporter:
                try:
                    log_exporter = OTLPLogExporter(
                        endpoint=self.config.otlp_endpoint,
                        headers=self.config.otlp_headers
                    )
                except Exception as e:
                    logger.warning(f"OTLP 日志导出器创建失败: {e}")

            if not log_exporter:  # console
                log_exporter = ConsoleLogRecordExporter()

            log_processor = BatchLogRecordProcessor(log_exporter)
            logger_provider.add_log_record_processor(log_processor)

            # 设置全局 LoggerProvider
            try:
                from opentelemetry import logs as otel_logs
                otel_logs.set_logger_provider(logger_provider)
            except (ImportError, AttributeError) as e:
                logger.warning(f"无法设置全局 LoggerProvider: {e}")

            # 设置 logging 模块的仪器化
            try:
                if LoggingInstrumentor:
                    LoggingInstrumentor().instrument()
            except Exception as e:
                logger.warning(f"设置日志仪器化失败: {e}")

            logger.info(f"日志已启用，导出器: {self.config.logs_exporter}")

        except Exception as e:
            logger.error(f"设置日志失败: {e}")

    def _start_prometheus_server(self) -> None:
        """启动 Prometheus HTTP 服务器"""
        try:
            from prometheus_client import start_http_server
            start_http_server(port=self.config.prometheus_port)
            logger.info(f"Prometheus 服务器已启动: http://localhost:{self.config.prometheus_port}")
        except Exception as e:
            logger.error(f"启动 Prometheus 服务器失败: {e}")

    def _setup_auto_instrumentation(self) -> None:
        """设置自动仪器化"""
        try:
            instrumented_components = []

            # Web 框架仪器化
            if "fastapi" in sys.modules and FastAPIInstrumentor:
                FastAPIInstrumentor().instrument()
                instrumented_components.append("FastAPI")
            elif "fastapi" in sys.modules:
                logger.warning("FastAPI 已加载但仪器化不可用")

            # 任务队列仪器化
            if "arq" in sys.modules and ArqInstrumentor:
                ArqInstrumentor().instrument()
                instrumented_components.append("Arq")
            elif "arq" in sys.modules:
                logger.warning("Arq 已加载但仪器化不可用")

            # HTTP 客户端仪器化
            if RequestsInstrumentor:
                RequestsInstrumentor().instrument()
                instrumented_components.append("Requests")
            else:
                logger.warning("Requests 仪器化不可用")

            if URLLib3Instrumentor:
                URLLib3Instrumentor().instrument()
                instrumented_components.append("Urllib3")
            else:
                logger.warning("Urllib3 仪器化不可用")

            if "httpx" in sys.modules and HTTPXClientInstrumentor:
                HTTPXClientInstrumentor().instrument()
                instrumented_components.append("HTTPX")
            elif "httpx" in sys.modules:
                logger.warning("HTTPX 已加载但仪器化不可用")

            # 数据库仪器化
            if SQLite3Instrumentor:
                SQLite3Instrumentor().instrument()
                instrumented_components.append("SQLite3")
            else:
                logger.warning("SQLite3 仪器化不可用")

            if "psycopg2" in sys.modules and Psycopg2Instrumentor:
                Psycopg2Instrumentor().instrument()
                instrumented_components.append("Psycopg2")
            elif "psycopg2" in sys.modules:
                logger.warning("Psycopg2 已加载但仪器化不可用")

            if "redis" in sys.modules and RedisInstrumentor:
                RedisInstrumentor().instrument()
                instrumented_components.append("Redis")
            elif "redis" in sys.modules:
                logger.warning("Redis 已加载但仪器化不可用")

            # 日志仪器化
            if LoggingInstrumentor:
                LoggingInstrumentor().instrument()
                instrumented_components.append("Logging")
            else:
                logger.warning("Logging 仪器化不可用")

            if instrumented_components:
                logger.info(f"自动仪器化已启用: {', '.join(instrumented_components)}")
            else:
                logger.warning("没有可用的自动仪器化组件")

        except Exception as e:
            logger.warning(f"自动仪器化设置失败: {e}")


# 全局实例
_telemetry_manager = TelemetryManager()


def setup_telemetry(
    service_name: str | None = None, service_version: str | None = None
) -> None:
    """
    设置可观测性系统

    Args:
        service_name: 服务名称
        service_version: 服务版本
    """
    if service_name:
        os.environ["OTEL_SERVICE_NAME"] = service_name
    if service_version:
        os.environ["OTEL_SERVICE_VERSION"] = service_version

    _telemetry_manager.setup_telemetry()


def get_tracer(name: str, version: str | None = None) -> Any:
    """
    获取追踪器

    Args:
        name: 追踪器名称
        version: 追踪器版本

    Returns:
        追踪器实例
    """
    if not OTEL_AVAILABLE:
        # 返回一个空的追踪器
        class MockTracer:
            def start_as_current_span(self, name, kind=None, **kwargs):
                class MockSpan:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def set_attribute(self, key, value):
                        pass
                    def set_status(self, status):
                        pass
                    def record_exception(self, exception):
                        pass
                return MockSpan()
        return MockTracer()

    try:
        return trace.get_tracer(name, version)
    except Exception as e:
        logger.warning(f"获取追踪器失败: {e}")
        # 返回一个空的追踪器
        class MockTracer:
            def start_as_current_span(self, name, kind=None, **kwargs):
                class MockSpan:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def set_attribute(self, key, value):
                        pass
                    def set_status(self, status):
                        pass
                    def record_exception(self, exception):
                        pass
                return MockSpan()
        return MockTracer()


def get_meter(name: str, version: str | None = None, description: str | None = None) -> Any:
    """
    获取计量器

    Args:
        name: 计量器名称
        version: 计量器版本
        description: 计量器描述

    Returns:
        计量器实例
    """
    if not OTEL_AVAILABLE:
        # 返回一个空的计量器
        class MockMeter:
            def create_counter(self, name, description=None, unit=None):
                class MockCounter:
                    def add(self, amount, attributes=None):
                        pass
                return MockCounter()

            def create_histogram(self, name, description=None, unit=None):
                class MockHistogram:
                    def record(self, value, attributes=None):
                        pass
                return MockHistogram()
        return MockMeter()

    try:
        return metrics.get_meter(name, version, description)
    except Exception as e:
        logger.warning(f"获取计量器失败: {e}")
        # 返回一个空的计量器
        class MockMeter:
            def create_counter(self, name, description=None, unit=None):
                class MockCounter:
                    def add(self, amount, attributes=None):
                        pass
                return MockCounter()

            def create_histogram(self, name, description=None, unit=None):
                class MockHistogram:
                    def record(self, value, attributes=None):
                        pass
                return MockHistogram()
        return MockMeter()


def get_telemetry_metrics() -> TelemetryMetrics:
    """获取指标收集器"""
    return _telemetry_manager.metrics_collector


def create_span(
    name: str,
    kind: Any = None,
    attributes: dict[str, Any] | None = None
) -> Any:
    """
    创建自定义 Span

    Args:
        name: Span 名称
        kind: Span 类型
        attributes: Span 属性

    Returns:
        可用时返回 Span，否则返回 None
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            return span
    except Exception as e:
        logger.warning(f"创建 Span 失败: {e}")
        return None


def record_exception(
    span: Any, exception: Exception, description: str | None = None
) -> None:
    """
    在 Span 中记录异常

    Args:
        span: Span 对象
        exception: 异常对象
        description: 异常描述
    """
    if not OTEL_AVAILABLE or not span:
        return

    try:
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, description or str(exception)))
    except Exception as e:
        logger.warning(f"记录异常失败: {e}")


def add_span_attributes(span: Any, **attributes: Any) -> None:
    """
    向 Span 添加属性

    Args:
        span: Span 对象
        **attributes: 属性键值对
    """
    if not OTEL_AVAILABLE or not span:
        return

    try:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    except Exception as e:
        logger.warning(f"添加 Span 属性失败: {e}")


def get_system_health_status() -> dict[str, Any]:
    """获取系统健康状态"""
    return _telemetry_manager.system_collector.get_system_health()


def get_request_stats(hours: int = 24) -> dict[str, Any]:
    """获取请求统计"""
    return _telemetry_manager.request_collector.get_request_stats(hours)


def record_request_metric(
    method: str,
    endpoint: str,
    status_code: int,
    response_time: float,
    client_ip: str = ""
) -> None:
    """记录请求指标"""
    _telemetry_manager.request_collector.record_request(
        method, endpoint, status_code, response_time, client_ip
    )


def check_rate_limit(client_ip: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
    """检查速率限制"""
    return _telemetry_manager.request_collector.check_rate_limit(client_ip, max_requests, window_seconds)


def get_monitoring_dashboard() -> dict[str, Any]:
    """获取监控仪表板数据"""
    system_health = get_system_health_status()
    request_stats = get_request_stats(hours=1)

    return {
        "timestamp": datetime.now().isoformat(),
        "system_health": system_health,
        "request_stats": request_stats,
        "telemetry_config": {
            "tracing_enabled": _telemetry_manager.config.enable_tracing,
            "metrics_enabled": _telemetry_manager.config.enable_metrics,
            "logs_enabled": _telemetry_manager.config.enable_logs,
        }
    }


def trace_method(operation_name: str | None = None):
    """
    方法追踪装饰器

    Args:
        operation_name: 操作名称，如果为 None 则使用函数名

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)

            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            try:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(op_name) as span:
                    # 添加函数信息
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)

                    # 记录参数数量（不记录具体值以保护隐私）
                    if args:
                        span.set_attribute("function.args_count", len(args))
                    if kwargs:
                        span.set_attribute("function.kwargs_count", len(kwargs))

                    # 执行函数
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                logger.warning(f"追踪方法 {op_name} 失败: {e}")
                return func(*args, **kwargs)

        return wrapper
    return decorator


# 导入时自动设置（仅在可用时）
if OTEL_AVAILABLE:
    try:
        setup_telemetry()
    except Exception as e:
        logger.warning(f"自动设置可观测性失败: {e}")
else:
    logger.warning("OpenTelemetry 不可用，可观测性功能已禁用")
