"""
API安全中间件

为FastAPI应用提供安全功能，包括：
- 身份验证
- API密钥验证
- 速率限制
- 安全头设置
- 输入验证
"""

from collections.abc import Callable
from typing import Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from src.framework.shared.logging import get_logger
from src.framework.shared.security import (
    check_rate_limit,
    get_security_headers,
    get_security_manager,
    validate_input,
    verify_api_key,
    verify_token,
)

logger = get_logger(__name__)

# HTTP Bearer 认证
security = HTTPBearer(auto_error=False)


class SecurityMiddleware(BaseHTTPMiddleware):
    """安全中间件"""

    def __init__(
        self,
        app,
        enable_auth: bool = True,
        enable_rate_limit: bool = True,
        enable_input_validation: bool = True,
        excluded_paths: list = None
    ):
        super().__init__(app)
        self.enable_auth = enable_auth
        self.enable_rate_limit = enable_rate_limit
        self.enable_input_validation = enable_input_validation
        self.excluded_paths = excluded_paths or [
            "/health",
            "/ready",
            "/live",
            "/version",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico"
        ]

        # 需要认证的路径模式
        self.protected_paths = [
            "/api/v1/",
            "/api/v1/config/",
            "/api/v1/speckit/",
            "/api/v1/tasks/"
        ]

        # 速率限制配置
        self.rate_limits = {
            "default": {"limit": 100, "window": 60},  # 每分钟100次
            "api": {"limit": 50, "window": 60},       # API调用每分钟50次
            "upload": {"limit": 10, "window": 60}      # 上传每分钟10次
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求"""
        path = request.url.path

        # 检查是否排除路径
        if self._is_excluded_path(path):
            return await call_next(request)

        # 获取客户端标识符
        client_ip = self._get_client_ip(request)

        try:
            # 速率限制检查
            if self.enable_rate_limit:
                if not self._check_rate_limit(request, client_ip):
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="请求过于频繁，请稍后再试",
                        headers={"Retry-After": "60"}
                    )

            # 身份验证检查
            user_info = None
            if self.enable_auth and self._requires_auth(path):
                user_info = await self._authenticate_request(request)
                if not user_info:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="身份验证失败",
                        headers={"WWW-Authenticate": "Bearer"}
                    )

                # 将用户信息添加到请求状态
                request.state.user = user_info

            # 输入验证
            if self.enable_input_validation:
                await self._validate_request_input(request)

            # 处理请求
            response = await call_next(request)

            # 添加安全头
            self._add_security_headers(response)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"安全中间件处理失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="服务器内部错误"
            )

    def _is_excluded_path(self, path: str) -> bool:
        """检查是否为排除路径"""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)

    def _requires_auth(self, path: str) -> bool:
        """检查路径是否需要认证"""
        return any(path.startswith(protected) for protected in self.protected_paths)

    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP"""
        # 检查代理头
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # 回退到直接连接的IP
        client = request.client
        return client.host if client else "unknown"

    def _check_rate_limit(self, request: Request, client_ip: str) -> bool:
        """检查速率限制"""
        try:
            path = request.url.path

            # 确定速率限制类型
            if "/api/v1/" in path:
                limit_type = "api"
            elif "/upload" in path or "/import" in path:
                limit_type = "upload"
            else:
                limit_type = "default"

            rate_config = self.rate_limits.get(limit_type, self.rate_limits["default"])

            # 检查速率限制
            return check_rate_limit(
                identifier=client_ip,
                limit=rate_config["limit"],
                window=rate_config["window"]
            )

        except Exception as e:
            logger.error(f"速率限制检查失败: {e}")
            return True  # 出错时允许通过

    async def _authenticate_request(self, request: Request) -> dict | None:
        """认证请求"""
        try:
            # 检查API密钥
            api_key = request.headers.get("X-API-Key")
            if api_key:
                key_info = verify_api_key(api_key)
                if key_info:
                    return {
                        "type": "api_key",
                        "name": key_info["name"],
                        "permissions": key_info["permissions"]
                    }

            # 检查Bearer令牌
            authorization = request.headers.get("Authorization")
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]
                token_payload = verify_token(token)
                if token_payload:
                    return {
                        "type": "jwt_token",
                        "user_id": token_payload.get("sub"),
                        "scopes": token_payload.get("scopes", [])
                    }

            return None

        except Exception as e:
            logger.error(f"请求认证失败: {e}")
            return None

    async def _validate_request_input(self, request: Request):
        """验证请求输入"""
        try:
            # 验证查询参数
            for param_name, param_value in request.query_params.items():
                if not validate_input(param_value, max_length=1000):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"查询参数 {param_name} 包含不安全内容"
                    )

            # 验证路径参数
            path_segments = [seg for seg in request.url.path.split('/') if seg]
            for segment in path_segments:
                if not validate_input(segment, max_length=100):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="路径参数包含不安全内容"
                    )

            # 验证请求头（排除常见的安全头）
            for header_name, header_value in request.headers.items():
                if header_name.lower() not in [
                    'host', 'user-agent', 'accept', 'content-type',
                    'authorization', 'x-api-key', 'x-forwarded-for',
                    'x-real-ip', 'content-length'
                ]:
                    if not validate_input(header_value, max_length=500):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"请求头 {header_name} 包含不安全内容"
                        )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"输入验证失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="输入验证失败"
            )

    def _add_security_headers(self, response: Response):
        """添加安全头"""
        security_headers = get_security_headers()
        for header_name, header_value in security_headers.items():
            response.headers[header_name] = header_value


class OptionalAuth:
    """可选认证依赖"""

    def __init__(self):
        self.security = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request) -> dict | None:
        """可选认证"""
        try:
            # 检查请求状态中是否已有用户信息
            if hasattr(request.state, 'user'):
                return request.state.user

            # 尝试API密钥认证
            api_key = request.headers.get("X-API-Key")
            if api_key:
                key_info = verify_api_key(api_key)
                if key_info:
                    return {
                        "type": "api_key",
                        "name": key_info["name"],
                        "permissions": key_info["permissions"]
                    }

            # 尝试JWT令牌认证
            credentials = await self.security(request)
            if credentials:
                token_payload = verify_token(credentials.credentials)
                if token_payload:
                    return {
                        "type": "jwt_token",
                        "user_id": token_payload.get("sub"),
                        "scopes": token_payload.get("scopes", [])
                    }

            return None

        except Exception as e:
            logger.error(f"可选认证失败: {e}")
            return None


class RequiredAuth:
    """必需认证依赖"""

    def __init__(self):
        self.security = HTTPBearer(auto_error=True)

    async def __call__(self, request: Request) -> dict:
        """必需认证"""
        # 检查请求状态中是否已有用户信息
        if hasattr(request.state, 'user'):
            return request.state.user

        # 尝试API密钥认证
        api_key = request.headers.get("X-API-Key")
        if api_key:
            key_info = verify_api_key(api_key)
            if key_info:
                return {
                    "type": "api_key",
                    "name": key_info["name"],
                    "permissions": key_info["permissions"]
                }

        # 尝试JWT令牌认证
        try:
            credentials = await self.security(request)
            if credentials:
                token_payload = verify_token(credentials.credentials)
                if token_payload:
                    return {
                        "type": "jwt_token",
                        "user_id": token_payload.get("sub"),
                        "scopes": token_payload.get("scopes", [])
                    }
        except Exception as e:
            logger.error(f"JWT认证失败: {e}")

        # 认证失败
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="身份验证失败",
            headers={"WWW-Authenticate": "Bearer"}
        )


# 便捷依赖
optional_auth = OptionalAuth()
required_auth = RequiredAuth()


def require_permissions(required_permissions: list[str]):
    """权限检查装饰器"""
    def permission_checker(user_info: dict = required_auth):
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="身份验证失败"
            )

        user_permissions = user_info.get("permissions", [])

        # 检查是否有所需权限
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"缺少权限: {permission}"
                )

        return user_info

    return permission_checker


def get_current_user(request: Request) -> dict | None:
    """获取当前用户"""
    return getattr(request.state, 'user', None)


def is_authenticated(request: Request) -> bool:
    """检查是否已认证"""
    return hasattr(request.state, 'user') and request.state.user is not None
