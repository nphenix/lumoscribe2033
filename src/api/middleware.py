"""
API 中间件

整合了所有 API 中间件功能，包括：
- 请求日志记录
- 性能监控和指标收集
- LLM 调用追踪
- 安全验证
- 速率限制
- 请求大小限制
- CORS 和安全头
- 监控仪表板和 API 端点
"""

import hashlib
import json
import random
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.framework.shared.logging import get_logger
from src.framework.shared.monitoring import get_metrics_collector, metrics_collector
from src.framework.shared.telemetry import check_rate_limit, record_request_metric

# 全局日志记录器
logger = get_logger(__name__)


async def log_requests(request: Request, call_next: Callable) -> Response:
    """
    请求日志中间件

    记录所有请求的详细信息，包括请求方法、路径、响应时间等
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # 添加请求 ID 到请求状态
    request.state.request_id = request_id

    # 记录请求开始
    logger.info(
        f"📤 [REQ-{request_id}] {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )

    try:
        # 处理请求
        response = await call_next(request)

        # 计算响应时间
        process_time = time.time() - start_time

        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # 记录请求完成
        logger.info(
            f"📥 [REQ-{request_id}] {response.status_code} "
            f"{request.method} {request.url.path} "
            f"({process_time:.4f}s)"
        )

        return response

    except Exception as e:
        # 记录错误
        process_time = time.time() - start_time
        logger.error(
            f"❌ [REQ-{request_id}] {request.method} {request.url.path} "
            f"错误: {str(e)} ({process_time:.4f}s)"
        )
        raise


async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """
    处理时间头中间件

    在响应头中添加请求处理时间
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


class RateLimitMiddleware:
    """
    速率限制中间件

    基于内存的简单速率限制实现
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # 清理过期的请求记录
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []

        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"🚫 速率限制触发: {client_ip}")
            return Response(
                content='{"error": "Too Many Requests", "message": "请求过于频繁，请稍后再试"}',
                status_code=429,
                media_type="application/json",
                headers={"X-Rate-Limit": "exceeded"}
            )

        # 记录当前请求
        self.requests[client_ip].append(current_time)

        return await call_next(request)


def create_cors_middleware(allow_origins: list[str] | None = None):
    """
    创建 CORS 中间件

    根据配置生成 CORS 中间件
    """
    from fastapi.middleware.cors import CORSMiddleware

    if allow_origins is None:
        allow_origins = ["http://localhost:8080"]

    return CORSMiddleware(
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_security_middleware():
    """
    创建安全中间件

    添加基本的安全头信息
    """
    async def security_headers(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # 添加安全头
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        })

        return response

    return security_headers


class RequestSizeLimitMiddleware:
    """
    请求大小限制中间件

    限制上传文件的大小
    """

    def __init__(self, max_size: int = 10 * 1024 * 1024):  # 10MB 默认
        self.max_size = max_size

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size:
                logger.warning(f"🚫 请求大小超限: {content_length} bytes")
                return Response(
                    content='{"error": "Payload Too Large", "message": "请求体过大"}',
                    status_code=413,
                    media_type="application/json"
                )

        return await call_next(request)


class SecurityValidator:
    """安全验证器"""

    # 常见的恶意模式
    MALICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',               # JavaScript 协议
        r'data:',                     # Data URI
        r'vbscript:',                 # VBScript
        r'on\w+\s*=',                 # 事件处理器
        r'union\s+select',            # SQL 注入
        r'drop\s+table',              # SQL 注入
        r'insert\s+into',             # SQL 注入
        r'update\s+',                 # SQL 注入
        r'exec\s*\(',                 # 执行命令
        r'eval\s*\(',                 # 执行代码
        r'file\s*:',                  # 文件协议
        r'\\\\',                      # 网络路径
        r'\.\./',                     # 路径遍历
        r'/etc/',                     # 系统路径
        r'/proc/',                    # 系统路径
        r'/dev/',                     # 系统路径
    ]

    @classmethod
    def validate_input_safety(cls, value: str, field_name: str = "input") -> str:
        """验证输入安全性"""
        if not isinstance(value, str):
            value = str(value)

        # 检查恶意模式
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"检测到潜在恶意输入 {field_name}: {pattern}")
                raise HTTPException(
                    status_code=400,
                    detail=f"输入包含不安全内容: {field_name}"
                )

        # 检查输入长度
        if len(value) > 10000:  # 10KB 限制
            raise HTTPException(
                status_code=400,
                detail=f"输入过长: {field_name}"
            )

        return value

    @classmethod
    def sanitize_html(cls, content: str) -> str:
        """HTML 内容清理"""
        # 移除潜在危险的标签和属性
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'link', 'meta']
        dangerous_attrs = ['onclick', 'onload', 'onerror', 'onmouseover']

        cleaned = content

        # 移除危险标签
        for tag in dangerous_tags:
            pattern = f'<{tag}[^>]*>.*?</{tag}>'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # 移除危险属性
        for attr in dangerous_attrs:
            pattern = f'\\s{attr}=[^\\s>]*'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        return cleaned


class EnhancedValidationMiddleware:
    """增强验证中间件 - 专注于安全验证"""

    def __init__(self, enable_security_check: bool = True):
        self.enable_security_check = enable_security_check

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # 安全检查
        if self.enable_security_check:
            await self._perform_security_checks(request)

        # 处理请求
        response = await call_next(request)

        return response

    async def _perform_security_checks(self, request: Request) -> None:
        """执行安全检查"""
        # 检查请求头
        for header_name, header_value in request.headers.items():
            if header_name.lower() not in ['host', 'user-agent', 'accept', 'content-type']:
                SecurityValidator.validate_input_safety(header_value, f"header_{header_name}")

        # 检查查询参数
        for param_name, param_value in request.query_params.items():
            SecurityValidator.validate_input_safety(param_value, f"query_param_{param_name}")

        # 检查路径参数
        path = str(request.url.path)
        path_segments = [seg for seg in path.split('/') if seg]
        for segment in path_segments:
            SecurityValidator.validate_input_safety(segment, "path_segment")


class ContentValidationMiddleware:
    """内容验证中间件"""

    def __init__(self, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        self.max_request_size = max_request_size

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # 检查请求大小
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_request_size:
                logger.warning(f"🚫 请求大小超限: {content_length} bytes")
                return Response(
                    content='{"error": "Payload Too Large", "message": "请求体过大"}',
                    status_code=413,
                    media_type="application/json"
                )

        # 处理请求
        response = await call_next(request)
        return response


def create_enhanced_validation_middleware():
    """创建增强验证中间件"""
    return EnhancedValidationMiddleware()


def create_content_validation_middleware():
    """创建内容验证中间件"""
    return ContentValidationMiddleware()
