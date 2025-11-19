"""
安全模块

针对单机个人用户的简化安全实践，包括：
- 基本身份验证
- API密钥管理
- 数据加密
- 输入验证
- 安全头设置
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from src.framework.shared.config import settings
from src.framework.shared.logging import get_logger

logger = get_logger(__name__)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """安全管理器 - 简化版，适合单机使用"""

    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire_minutes = settings.JWT_EXPIRATION_HOURS * 60

        # API密钥存储（内存中，单机使用）
        self._api_keys: dict[str, dict[str, Any]] = {}

        # 会话存储（内存中，单机使用）
        self._sessions: dict[str, dict[str, Any]] = {}

        # 速率限制存储（内存中，单机使用）
        self._rate_limits: dict[str, dict[str, Any]] = {}

        logger.info("🔐 安全管理器已初始化（单机模式）")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False

    def get_password_hash(self, password: str) -> str:
        """生成密码哈希"""
        try:
            return pwd_context.hash(password)
        except Exception as e:
            logger.error(f"密码哈希生成失败: {e}")
            raise

    def create_access_token(self, data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
        """创建访问令牌"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire})

        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"JWT令牌创建失败: {e}")
            raise

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"JWT令牌验证失败: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT令牌验证异常: {e}")
            return None

    def generate_api_key(self, name: str, permissions: list[str] = None) -> dict[str, Any]:
        """生成API密钥"""
        try:
            # 生成随机密钥
            api_key = f"ls_{secrets.token_urlsafe(32)}"

            # 存储密钥信息
            key_info = {
                "name": name,
                "key": api_key,
                "permissions": permissions or ["read", "write"],
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "usage_count": 0
            }

            self._api_keys[api_key] = key_info

            logger.info(f"🔑 API密钥已生成: {name}")
            return {
                "api_key": api_key,
                "name": name,
                "permissions": key_info["permissions"],
                "created_at": key_info["created_at"]
            }
        except Exception as e:
            logger.error(f"API密钥生成失败: {e}")
            raise

    def verify_api_key(self, api_key: str) -> dict[str, Any] | None:
        """验证API密钥"""
        try:
            key_info = self._api_keys.get(api_key)
            if not key_info:
                return None

            # 更新使用记录
            key_info["last_used"] = datetime.now().isoformat()
            key_info["usage_count"] += 1

            return key_info
        except Exception as e:
            logger.error(f"API密钥验证失败: {e}")
            return None

    def create_session(self, user_id: str, user_data: dict[str, Any] = None) -> str:
        """创建会话"""
        try:
            session_id = secrets.token_urlsafe(32)

            session_data = {
                "user_id": user_id,
                "user_data": user_data or {},
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }

            self._sessions[session_id] = session_data

            logger.info(f"🔐 会话已创建: {user_id}")
            return session_id
        except Exception as e:
            logger.error(f"会话创建失败: {e}")
            raise

    def verify_session(self, session_id: str) -> dict[str, Any] | None:
        """验证会话"""
        try:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return None

            # 检查过期
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                # 清理过期会话
                del self._sessions[session_id]
                return None

            # 更新最后访问时间
            session_data["last_accessed"] = datetime.now().isoformat()

            return session_data
        except Exception as e:
            logger.error(f"会话验证失败: {e}")
            return None

    def check_rate_limit(self, identifier: str, limit: int = 100, window: int = 60) -> bool:
        """检查速率限制"""
        try:
            now = time.time()
            window_start = now - window

            # 获取或创建速率限制记录
            if identifier not in self._rate_limits:
                self._rate_limits[identifier] = {"requests": [], "blocked_until": 0}

            rate_limit_data = self._rate_limits[identifier]

            # 检查是否被封禁
            if now < rate_limit_data["blocked_until"]:
                return False

            # 清理过期的请求记录
            rate_limit_data["requests"] = [
                req_time for req_time in rate_limit_data["requests"]
                if req_time > window_start
            ]

            # 检查是否超过限制
            if len(rate_limit_data["requests"]) >= limit:
                # 封禁一段时间
                rate_limit_data["blocked_until"] = now + window
                logger.warning(f"🚫 速率限制触发: {identifier}")
                return False

            # 记录当前请求
            rate_limit_data["requests"].append(now)
            return True

        except Exception as e:
            logger.error(f"速率限制检查失败: {e}")
            return True  # 出错时允许通过

    def encrypt_data(self, data: str, key: str | None = None) -> str:
        """简单数据加密（基于HMAC）"""
        try:
            encryption_key = key or self.secret_key
            encrypted = hmac.new(
                encryption_key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            return encrypted
        except Exception as e:
            logger.error(f"数据加密失败: {e}")
            raise

    def validate_input(self, input_data: str, max_length: int = 10000) -> bool:
        """输入验证"""
        try:
            if not isinstance(input_data, str):
                return False

            # 长度检查
            if len(input_data) > max_length:
                return False

            # 基本的安全检查
            dangerous_patterns = [
                '<script', '</script>', 'javascript:', 'vbscript:',
                'onload=', 'onerror=', 'onclick=', 'onmouseover=',
                'eval(', 'exec(', 'system(', 'file://', '../'
            ]

            input_lower = input_data.lower()
            for pattern in dangerous_patterns:
                if pattern in input_lower:
                    logger.warning(f"检测到潜在危险输入: {pattern}")
                    return False

            return True

        except Exception as e:
            logger.error(f"输入验证失败: {e}")
            return False

    def get_security_headers(self) -> dict[str, str]:
        """获取安全头"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }

    def cleanup_expired_data(self):
        """清理过期数据"""
        try:
            now = datetime.now()

            # 清理过期会话
            expired_sessions = []
            for session_id, session_data in self._sessions.items():
                expires_at = datetime.fromisoformat(session_data["expires_at"])
                if now > expires_at:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                del self._sessions[session_id]

            # 清理过期的速率限制记录
            cutoff_time = time.time() - 3600  # 1小时前
            expired_rate_limits = []
            for identifier, rate_data in self._rate_limits.items():
                if rate_data["blocked_until"] < cutoff_time and not rate_data["requests"]:
                    expired_rate_limits.append(identifier)

            for identifier in expired_rate_limits:
                del self._rate_limits[identifier]

            if expired_sessions or expired_rate_limits:
                logger.info(f"🧹 清理过期数据: {len(expired_sessions)} 会话, {len(expired_rate_limits)} 速率限制")

        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")

    def get_security_status(self) -> dict[str, Any]:
        """获取安全状态"""
        return {
            "active_sessions": len(self._sessions),
            "active_api_keys": len(self._api_keys),
            "rate_limits": len(self._rate_limits),
            "security_headers": list(self.get_security_headers().keys()),
            "last_cleanup": datetime.now().isoformat()
        }


# 全局安全管理器实例 - 延迟初始化
_security_manager = None


def get_security_manager() -> SecurityManager:
    """获取全局安全管理器实例"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


# 设置全局实例
security_manager = get_security_manager()


# 便捷函数
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return security_manager.verify_password(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希"""
    return security_manager.get_password_hash(password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """创建访问令牌"""
    return security_manager.create_access_token(data, expires_delta)


def verify_token(token: str) -> dict[str, Any] | None:
    """验证令牌"""
    return security_manager.verify_token(token)


def generate_api_key(name: str, permissions: list[str] = None) -> dict[str, Any]:
    """生成API密钥"""
    return security_manager.generate_api_key(name, permissions)


def verify_api_key(api_key: str) -> dict[str, Any] | None:
    """验证API密钥"""
    return security_manager.verify_api_key(api_key)


def check_rate_limit(identifier: str, limit: int = 100, window: int = 60) -> bool:
    """检查速率限制"""
    return security_manager.check_rate_limit(identifier, limit, window)


def validate_input(input_data: str, max_length: int = 10000) -> bool:
    """输入验证"""
    return security_manager.validate_input(input_data, max_length)


def get_security_headers() -> dict[str, str]:
    """获取安全头"""
    return security_manager.get_security_headers()


def get_security_status() -> dict[str, Any]:
    """获取安全状态"""
    return security_manager.get_security_status()


# 简化的用户认证和权限检查（单机版）
def get_current_user():
    """获取当前用户（单机版，返回默认用户）"""
    return {"user_id": "local_user", "username": "Local User", "role": "admin"}


def require_permission(permission: str):
    """权限检查装饰器（单机版，允许所有访问）"""
    def decorator(func):
        return func
    return decorator
