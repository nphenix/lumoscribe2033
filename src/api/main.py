"""
FastAPI 主应用入口

基于 FastAPI 最佳实践实现：
- 统一的依赖注入
- 中间件配置
- 路由注册
- 错误处理
- CORS 支持

架构特点：
- 异步支持
- 自动 API 文档生成
- 类型安全
- 可扩展的中间件系统
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING
import traceback

if TYPE_CHECKING:
    pass

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# 导入异常类型
from src.framework.shared.exceptions import (
    LumoscribeError,
    LLMError,
    NetworkError,
    ValidationError,
    RateLimitError,
)

# 导入监控中间件模块，用于 configure_middleware 函数
import src.api.monitoring_middleware as monitoring_middleware
from src.api import langchain_middleware, middleware
from src.api.monitoring_middleware import MonitoringMiddleware
from src.api.routes import (
    config,
    docs,
    health,
    monitoring,
    performance,
    security,
    speckit,
    tasks,
)
from src.api.security_middleware import SecurityMiddleware
from src.framework.orchestrators import bootstrap_langchain_executor
from src.framework.shared.config import Settings
from src.framework.shared.exceptions import LumoscribeError
from src.framework.shared.langchain_security_middleware import (
    create_security_middleware_stack,
)


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """应用生命周期管理"""
        # 启动时的初始化
        logger.info("🚀 lumoscribe2033 API 服务启动中...")
        await startup_event()
        yield
        # 关闭时的清理
        await shutdown_event()

    app = FastAPI(
        title="lumoscribe2033 Hybrid Graph-RAG API",
        description="基于 speckit 的 AI 驱动质量提升平台",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        openapi_url="/openapi.json"
    )

    # 配置 LangChain 中间件
    configure_langchain_middleware(app)

    # 配置安全中间件（最优先）
    configure_security_middleware(app)

    # 配置监控中间件（需要在其他中间件之前）
    configure_monitoring_middleware(app)

    # 配置其他中间件
    configure_middleware(app)

    # 注册路由
    configure_routes(app)

    # 全局异常处理
    configure_exception_handlers(app)

    return app


def configure_langchain_middleware(app: FastAPI) -> None:
    """配置 LangChain 中间件"""
    settings = Settings()

    # 初始化 LangChain 中间件
    langchain_middleware.initialize_langchain_middleware(
        project_name=getattr(settings, 'PROJECT_NAME', 'lumoscribe2033'),
        tracing_enabled=getattr(settings, 'LLM_TRACING_ENABLED', True)
    )

    # 创建安全中间件栈
    security_middleware_stack = create_security_middleware_stack(
        enable_pii=settings.ENVIRONMENT == "production",  # 生产环境启用PII检测
        enable_validation=True,  # 启用输入验证
        enable_audit=settings.METRICS_ENABLED,  # 启用审计日志
        enable_rate_limit=True,  # 启用速率限制
        pii_strategy="redact"  # PII脱敏策略
    )

    # 添加 LangChain 中间件（包含安全中间件）
    app.add_middleware(
        langchain_middleware.LangChainMiddleware,
        dispatch=langchain_middleware.get_langchain_middleware(),
        security_middleware_stack=security_middleware_stack
    )

    logger.info("✅ LangChain 中间件配置完成（包含安全中间件）")

    # 初始化全局 LangChainExecutor（阶段 C 接入 API）
    bootstrap_langchain_executor(settings=settings)


def configure_security_middleware(app: FastAPI) -> None:
    """配置安全中间件"""
    settings = Settings()
    # 添加安全中间件
    app.add_middleware(
        SecurityMiddleware,
        enable_auth=settings.ENVIRONMENT == "production",  # 生产环境启用认证
        enable_rate_limit=True,
        enable_input_validation=True,
        excluded_paths=[
            "/health",
            "/ready",
            "/live",
            "/version",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/metrics"
        ]
    )

    logger.info("✅ 安全中间件配置完成")


async def startup_event() -> None:
    """应用启动事件"""
    logger.info("✅ lumoscribe2033 API 服务启动完成")


async def shutdown_event() -> None:
    """应用关闭事件"""
    logger.info("🛑 lumoscribe2033 API 服务正在关闭...")

    # 停止监控收集器
    from src.framework.shared.monitoring import get_metrics_collector
    get_metrics_collector().stop()


def configure_monitoring_middleware(app: FastAPI) -> None:
    """配置增强的监控中间件"""
    settings = Settings()

    # 获取排除路径和采样率配置
    exclude_paths = getattr(settings, 'MONITORING_EXCLUDE_PATHS', [
        "/health", "/metrics", "/docs", "/openapi.json", "/favicon.ico"
    ])
    sample_rate = getattr(settings, 'MONITORING_SAMPLE_RATE', 1.0)

    # 添加增强监控中间件
    app.add_middleware(
        monitoring_middleware.EnhancedMonitoringMiddleware,
        exclude_paths=exclude_paths,
        sample_rate=sample_rate,
        enable_retry=True,
        max_retries=2,
        retryable_error_codes=["LLM_ERROR", "NETWORK_ERROR", "DATABASE_ERROR", "RATE_LIMIT_ERROR"]
    )

    # 添加断路器中间件
    app.add_middleware(
        monitoring_middleware.CircuitBreakerMiddleware,
        failure_threshold=5,
        recovery_timeout=60,
        monitored_endpoints=["/api/v1/"]
    )

    logger.info("✅ 增强监控中间件配置完成（包含断路器和重试机制）")


def configure_middleware(app: FastAPI) -> None:
    """配置中间件"""

    # 1. 通用 API 中间件（安全验证、请求大小限制等）
    app.add_middleware(
        middleware.EnhancedValidationMiddleware,
        enable_security_check=True
    )

    # 请求大小限制中间件
    app.add_middleware(
        middleware.ContentValidationMiddleware,
        max_request_size=10 * 1024 * 1024  # 10MB
    )

    # 2. LangChain 中间件（LLM 调用追踪）
    if langchain_middleware.get_langchain_middleware():
        app.add_middleware(
            langchain_middleware.LangChainMiddleware,
            tracing_enabled=True,
            capture_io=True,
            capture_metadata=True
        )

    # 3. 基础中间件（日志、性能、安全头等）
    app.add_middleware(
        middleware.log_requests.__class__,
        dispatch=middleware.log_requests
    )

    app.add_middleware(
        middleware.add_process_time_header.__class__,
        dispatch=middleware.add_process_time_header
    )

    app.add_middleware(
        middleware.create_security_middleware(),
    )

    # 4. CORS 中间件
    configure_cors(app)


def configure_cors(app: FastAPI) -> None:
    """配置 CORS 中间件"""
    settings = Settings()
    origins = []

    if settings.API_CORS_ORIGINS:
        origins = [origin.strip() for origin in settings.API_CORS_ORIGINS.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def configure_routes(app: FastAPI) -> None:
    """配置路由"""
    # 健康检查路由
    app.include_router(
        health.router,
        prefix="/api/v1/health",
        tags=["health"]
    )

    # 任务管理路由
    app.include_router(
        tasks.router,
        prefix="/api/v1/tasks",
        tags=["tasks"]
    )

    # 文档管理路由
    app.include_router(
        docs.router,
        prefix="/api/v1/docs",
        tags=["docs"]
    )

    # 配置管理路由
    app.include_router(
        config.router,
        prefix="/api/v1/config",
        tags=["config"]
    )

    # Speckit 相关路由
    app.include_router(
        speckit.router,
        prefix="/api/v1/speckit",
        tags=["speckit"]
    )

    # 监控相关路由
    app.include_router(
        monitoring.router,
        prefix="",
        tags=["monitoring"]
    )

    # 安全管理路由
    app.include_router(
        security.router,
        tags=["security"]
    )

    # 性能监控路由
    app.include_router(
        performance.router,
        prefix="/api/v1",
        tags=["performance"]
    )



def configure_exception_handlers(app: FastAPI) -> None:
    """配置全局异常处理器"""

    @app.exception_handler(LumoscribeError)
    async def lumoscribe_exception_handler(
        request: Request, exc: LumoscribeError
    ) -> JSONResponse:
        """Lumoscribe 自定义异常处理"""
        logger.error(f"Lumoscribe 异常: {exc.error_code} - {exc.message}")
        
        # 根据异常类型返回不同的状态码
        status_code = _get_status_code_for_exception(exc)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "cause": str(exc.cause) if exc.cause else None,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @app.exception_handler(LLMError)
    async def llm_exception_handler(
        request: Request, exc: LLMError
    ) -> JSONResponse:
        """LLM 异常专门处理"""
        logger.error(f"LLM 异常: {exc.error_code} - {exc.message}")
        
        # LLM 异常通常返回 503（服务不可用）或 429（限流）
        if "quota" in str(exc).lower() or "limit" in str(exc).lower():
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error_code": exc.error_code,
                "message": f"AI 服务暂时不可用: {exc.message}",
                "details": exc.details,
                "model": exc.details.get("model", "unknown") if hasattr(exc, 'details') else None,
                "success": False,
                "retry_after": 5 if status_code == status.HTTP_503_SERVICE_UNAVAILABLE else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @app.exception_handler(NetworkError)
    async def network_exception_handler(
        request: Request, exc: NetworkError
    ) -> JSONResponse:
        """网络异常专门处理"""
        logger.error(f"网络异常: {exc.error_code} - {exc.message}")
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": exc.error_code,
                "message": "网络连接异常，请检查网络设置",
                "details": exc.details,
                "url": exc.details.get("url", "unknown") if hasattr(exc, 'details') else None,
                "success": False,
                "retry_after": 3,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """验证异常专门处理"""
        logger.warning(f"验证异常: {exc.error_code} - {exc.message}")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": exc.error_code,
                "message": "请求参数验证失败",
                "details": exc.details,
                "field": exc.details.get("field", "unknown") if hasattr(exc, 'details') else None,
                "success": False,
                "help": "请检查请求参数格式和值范围",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(
        request: Request, exc: RateLimitError
    ) -> JSONResponse:
        """限流异常专门处理"""
        logger.warning(f"限流异常: {exc.error_code} - {exc.message}")
        
        retry_after = exc.details.get("retry_after", 60) if hasattr(exc, 'details') else 60
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error_code": exc.error_code,
                "message": "请求频率过高，请稍后重试",
                "details": exc.details,
                "retry_after": retry_after,
                "success": False,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={"Retry-After": str(retry_after)}
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """全局异常处理"""
        logger.error(f"全局异常: {exc}")
        logger.error(f"异常追踪: {traceback.format_exc()}")

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_code": "INTERNAL_ERROR",
                "error": "Internal Server Error",
                "message": "服务暂时不可用，请稍后重试",
                "success": False,
                "request_id": getattr(request.state, 'request_id', 'unknown'),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


def _get_status_code_for_exception(exc: LumoscribeError) -> int:
    """根据异常类型获取对应的 HTTP 状态码"""
    from fastapi import status
    
    # 导入异常类型映射
    status_code_map = {
        "AUTH_ERROR": status.HTTP_401_UNAUTHORIZED,
        "AUTHZ_ERROR": status.HTTP_403_FORBIDDEN,
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "RATE_LIMIT_ERROR": status.HTTP_429_TOO_MANY_REQUESTS,
        "LLM_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "NETWORK_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "DATABASE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "PIPELINE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "TASK_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "AGENT_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "RETRIEVAL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "COMPLIANCE_ERROR": status.HTTP_400_BAD_REQUEST,
        "DOCUMENT_ERROR": status.HTTP_400_BAD_REQUEST,
        "CONVERSATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "IDE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "CACHE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "INDEX_SERVICE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    
    return status_code_map.get(exc.error_code, status.HTTP_400_BAD_REQUEST)


# 创建应用实例
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
