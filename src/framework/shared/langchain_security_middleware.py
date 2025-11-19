"""
LangChain v1.0 兼容的安全中间件

基于LangChain v1.0中间件模式实现的安全功能，包括：
- PII检测和脱敏
- 输入验证和清理
- 访问控制
- 审计日志
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.framework.shared.logging import get_logger
from src.framework.shared.security import get_security_manager

logger = get_logger(__name__)


@dataclass
class SecurityContext:
    """安全上下文"""
    user_id: str | None = None
    permissions: list[str] = None
    client_ip: str = ""
    request_id: str = ""
    risk_level: str = "low"  # low, medium, high
    pii_detected: bool = False
    blocked_content: bool = False


class PIIMiddleware(AgentMiddleware):
    """
    个人身份信息(PII)检测和脱敏中间件

    基于LangChain v1.0中间件模式实现
    """

    def __init__(
        self,
        pii_types: list[str] = None,
        strategy: str = "redact",  # redact, block, log
        apply_to_input: bool = True,
        apply_to_output: bool = False
    ):
        self.pii_types = pii_types or ["email", "phone_number", "ssn", "credit_card"]
        self.strategy = strategy
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output

        # PII检测模式
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "api_key": r'\b[A-Za-z0-9]{32,}\b',
            "password": r'(?i)password\s*[:=]\s*[^\s]+'
        }

        logger.info(f"🔒 PII中间件已初始化，策略: {strategy}")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """包装模型调用以进行PII检测和处理"""
        try:
            # 获取安全上下文
            security_context = self._get_security_context(request)

            # 处理输入
            if self.apply_to_input:
                processed_input = self._process_pii(
                    request.messages,
                    security_context
                )
                request.messages = processed_input

            # 调用原始处理器
            response = handler(request)

            # 处理输出
            if self.apply_to_output:
                processed_output = self._process_pii(
                    response.messages if hasattr(response, 'messages') else [response],
                    security_context
                )
                if hasattr(response, 'messages'):
                    response.messages = processed_output

            # 记录PII检测事件
            if security_context.pii_detected:
                logger.warning(
                    f"🚨 PII检测触发 - 用户: {security_context.user_id}, "
                    f"风险级别: {security_context.risk_level}, "
                    f"请求ID: {security_context.request_id}"
                )

            return response

        except Exception as e:
            logger.error(f"PII中间件处理失败: {e}")
            return handler(request)

    def _get_security_context(self, request: ModelRequest) -> SecurityContext:
        """获取安全上下文"""
        context = getattr(request.runtime, 'context', {})

        return SecurityContext(
            user_id=getattr(context, 'user_id', None),
            permissions=getattr(context, 'permissions', []),
            client_ip=getattr(context, 'client_ip', ''),
            request_id=getattr(context, 'request_id', ''),
            risk_level=getattr(context, 'risk_level', 'low')
        )

    def _process_pii(self, messages: list[Any], context: SecurityContext) -> list[Any]:
        """处理PII信息"""
        processed_messages = []

        for message in messages:
            if hasattr(message, 'content') and isinstance(message.content, str):
                processed_content, pii_detected = self._detect_and_handle_pii(
                    message.content,
                    context
                )

                if pii_detected:
                    context.pii_detected = True
                    context.risk_level = "medium"

                # 创建新消息对象
                if hasattr(message, 'model_copy'):
                    processed_message = message.model_copy()
                    processed_message.content = processed_content
                else:
                    processed_message = message
                    processed_message.content = processed_content

                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)

        return processed_messages

    def _detect_and_handle_pii(self, text: str, context: SecurityContext) -> tuple[str, bool]:
        """检测和处理PII"""
        pii_detected = False
        processed_text = text

        for pii_type in self.pii_types:
            if pii_type in self.pii_patterns:
                pattern = self.pii_patterns[pii_type]
                matches = re.findall(pattern, processed_text, re.IGNORECASE)

                if matches:
                    pii_detected = True
                    logger.debug(f"检测到PII类型 {pii_type}: {len(matches)} 个匹配")

                    if self.strategy == "redact":
                        processed_text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", processed_text, flags=re.IGNORECASE)
                    elif self.strategy == "block":
                        context.blocked_content = True
                        context.risk_level = "high"
                        return "内容包含敏感信息，已被阻止", True
                    elif self.strategy == "log":
                        logger.warning(f"PII检测 - 类型: {pii_type}, 内容: {matches[:2]}")  # 只记录前2个匹配

        return processed_text, pii_detected


class InputValidationMiddleware(AgentMiddleware):
    """
    输入验证中间件

    基于LangChain v1.0中间件模式实现
    """

    def __init__(
        self,
        max_length: int = 10000,
        allowed_patterns: list[str] = None,
        blocked_patterns: list[str] = None
    ):
        self.max_length = max_length
        self.allowed_patterns = allowed_patterns or []
        self.blocked_patterns = blocked_patterns or [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'file\s*:',
            r'\.\./',
            r'/etc/',
            r'/proc/',
            r'/dev/'
        ]

        logger.info("🔍 输入验证中间件已初始化")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """包装模型调用以进行输入验证"""
        try:
            # 验证输入
            validation_result = self._validate_input(request)

            if not validation_result["valid"]:
                logger.warning(
                    f"🚫 输入验证失败 - 原因: {validation_result['reason']}, "
                    f"请求ID: {getattr(request.runtime, 'request_id', 'unknown')}"
                )

                # 返回错误响应
                return self._create_error_response(validation_result["reason"])

            return handler(request)

        except Exception as e:
            logger.error(f"输入验证中间件处理失败: {e}")
            return handler(request)

    def _validate_input(self, request: ModelRequest) -> dict[str, Any]:
        """验证输入"""
        try:
            for message in request.messages:
                if hasattr(message, 'content') and isinstance(message.content, str):
                    content = message.content

                    # 长度检查
                    if len(content) > self.max_length:
                        return {
                            "valid": False,
                            "reason": f"输入长度超过限制 ({self.max_length} 字符)"
                        }

                    # 阻止模式检查
                    for pattern in self.blocked_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            return {
                                "valid": False,
                                "reason": f"输入包含不允许的内容模式: {pattern}"
                            }

                    # 允许模式检查（如果有）
                    if self.allowed_patterns:
                        allowed = False
                        for pattern in self.allowed_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                allowed = True
                                break

                        if not allowed:
                            return {
                                "valid": False,
                                "reason": "输入不符合允许的内容模式"
                            }

            return {"valid": True}

        except Exception as e:
            logger.error(f"输入验证异常: {e}")
            return {
                "valid": False,
                "reason": f"输入验证异常: {str(e)}"
            }

    def _create_error_response(self, reason: str) -> ModelResponse:
        """创建错误响应"""
        error_message = f"输入验证失败: {reason}"

        # 这里需要根据实际的ModelResponse结构来创建错误响应
        # 具体实现取决于LangChain v1.0的API
        try:
            from langchain.messages import AIMessage
            return ModelResponse(messages=[AIMessage(content=error_message)])
        except ImportError:
            # 如果无法导入，返回基本的错误响应
            return ModelResponse(content=error_message)


class AuditLoggingMiddleware(AgentMiddleware):
    """
    审计日志中间件

    基于LangChain v1.0中间件模式实现
    """

    def __init__(
        self,
        log_requests: bool = True,
        log_responses: bool = True,
        log_pii: bool = False,
        max_content_length: int = 1000
    ):
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_pii = log_pii
        self.max_content_length = max_content_length

        logger.info("📝 审计日志中间件已初始化")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """包装模型调用以进行审计日志记录"""
        start_time = time.time()

        try:
            # 获取上下文信息
            context = self._get_context_info(request)

            # 记录请求日志
            if self.log_requests:
                self._log_request(request, context)

            # 执行请求
            response = handler(request)

            # 记录响应日志
            if self.log_responses:
                execution_time = time.time() - start_time
                self._log_response(response, context, execution_time)

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"审计日志中间件处理失败: {e}, 执行时间: {execution_time:.3f}s")
            return handler(request)

    def _get_context_info(self, request: ModelRequest) -> dict[str, Any]:
        """获取上下文信息"""
        context = getattr(request.runtime, 'context', {})

        return {
            "user_id": getattr(context, 'user_id', 'anonymous'),
            "client_ip": getattr(context, 'client_ip', 'unknown'),
            "request_id": getattr(context, 'request_id', ''),
            "permissions": getattr(context, 'permissions', []),
            "timestamp": time.time()
        }

    def _log_request(self, request: ModelRequest, context: dict[str, Any]):
        """记录请求日志"""
        try:
            log_data = {
                "event": "model_request",
                "user_id": context["user_id"],
                "client_ip": context["client_ip"],
                "request_id": context["request_id"],
                "permissions": context["permissions"],
                "timestamp": context["timestamp"],
                "message_count": len(request.messages)
            }

            # 记录消息内容（截断长内容）
            if self.log_pii:
                log_data["messages"] = [
                    {
                        "role": getattr(msg, 'role', 'unknown'),
                        "content": self._truncate_content(getattr(msg, 'content', ''))
                    }
                    for msg in request.messages
                ]
            else:
                log_data["message_types"] = [
                    getattr(msg, 'role', 'unknown') for msg in request.messages
                ]

            logger.info(f"📋 审计日志: {log_data}")

        except Exception as e:
            logger.error(f"记录请求日志失败: {e}")

    def _log_response(self, response: ModelResponse, context: dict[str, Any], execution_time: float):
        """记录响应日志"""
        try:
            log_data = {
                "event": "model_response",
                "user_id": context["user_id"],
                "client_ip": context["client_ip"],
                "request_id": context["request_id"],
                "execution_time": execution_time,
                "timestamp": time.time()
            }

            # 记录响应内容（截断长内容）
            if hasattr(response, 'messages') and self.log_pii:
                log_data["response_messages"] = [
                    {
                        "role": getattr(msg, 'role', 'unknown'),
                        "content": self._truncate_content(getattr(msg, 'content', ''))
                    }
                    for msg in response.messages
                ]
            elif hasattr(response, 'content'):
                log_data["response_type"] = "content"
                log_data["response_length"] = len(str(response.content))

            logger.info(f"📋 审计日志: {log_data}")

        except Exception as e:
            logger.error(f"记录响应日志失败: {e}")

    def _truncate_content(self, content: str) -> str:
        """截断内容"""
        if len(content) <= self.max_content_length:
            return content

        return content[:self.max_content_length] + "...[截断]"


class RateLimitMiddleware(AgentMiddleware):
    """
    速率限制中间件

    基于LangChain v1.0中间件模式实现
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_limit = burst_limit

        # 内存存储速率限制数据（单机使用）
        self._rate_data: dict[str, dict[str, Any]] = {}

        logger.info(f"⏱️ 速率限制中间件已初始化 - 每分钟: {requests_per_minute}, 每小时: {requests_per_hour}")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """包装模型调用以进行速率限制"""
        try:
            # 获取用户标识
            user_id = self._get_user_id(request)

            # 检查速率限制
            if not self._check_rate_limit(user_id):
                logger.warning(f"🚫 速率限制触发 - 用户: {user_id}")
                return self._create_rate_limit_response()

            # 更新速率计数
            self._update_rate_count(user_id)

            return handler(request)

        except Exception as e:
            logger.error(f"速率限制中间件处理失败: {e}")
            return handler(request)

    def _get_user_id(self, request: ModelRequest) -> str:
        """获取用户标识"""
        context = getattr(request.runtime, 'context', {})
        user_id = getattr(context, 'user_id', None)

        if user_id:
            return user_id

        # 回退到客户端IP
        return getattr(context, 'client_ip', 'unknown')

    def _check_rate_limit(self, user_id: str) -> bool:
        """检查速率限制"""
        now = time.time()

        if user_id not in self._rate_data:
            self._rate_data[user_id] = {
                "requests": [],
                "minute_count": 0,
                "hour_count": 0,
                "last_minute_reset": now,
                "last_hour_reset": now
            }

        rate_data = self._rate_data[user_id]

        # 重置计数器
        if now - rate_data["last_minute_reset"] > 60:
            rate_data["minute_count"] = 0
            rate_data["last_minute_reset"] = now

        if now - rate_data["last_hour_reset"] > 3600:
            rate_data["hour_count"] = 0
            rate_data["last_hour_reset"] = now

        # 检查限制
        if rate_data["minute_count"] >= self.requests_per_minute:
            return False

        if rate_data["hour_count"] >= self.requests_per_hour:
            return False

        # 检查突发限制
        recent_requests = [
            req_time for req_time in rate_data["requests"]
            if now - req_time < 10  # 10秒内
        ]

        if len(recent_requests) >= self.burst_limit:
            return False

        return True

    def _update_rate_count(self, user_id: str):
        """更新速率计数"""
        now = time.time()
        rate_data = self._rate_data[user_id]

        rate_data["requests"].append(now)
        rate_data["minute_count"] += 1
        rate_data["hour_count"] += 1

        # 清理旧请求记录
        rate_data["requests"] = [
            req_time for req_time in rate_data["requests"]
            if now - req_time < 3600  # 保留1小时内的记录
        ]

    def _create_rate_limit_response(self) -> ModelResponse:
        """创建速率限制响应"""
        error_message = "请求过于频繁，请稍后再试"

        try:
            from langchain.messages import AIMessage
            return ModelResponse(messages=[AIMessage(content=error_message)])
        except ImportError:
            return ModelResponse(content=error_message)


def create_security_middleware_stack(
    enable_pii: bool = True,
    enable_validation: bool = True,
    enable_audit: bool = True,
    enable_rate_limit: bool = True,
    pii_strategy: str = "redact"
) -> list[AgentMiddleware]:
    """
    创建安全中间件栈

    Args:
        enable_pii: 是否启用PII检测
        enable_validation: 是否启用输入验证
        enable_audit: 是否启用审计日志
        enable_rate_limit: 是否启用速率限制
        pii_strategy: PII处理策略

    Returns:
        中间件列表
    """
    middleware_stack = []

    if enable_pii:
        middleware_stack.append(
            PIIMiddleware(
                pii_types=["email", "phone_number", "api_key", "password"],
                strategy=pii_strategy,
                apply_to_input=True,
                apply_to_output=False
            )
        )

    if enable_validation:
        middleware_stack.append(
            InputValidationMiddleware(
                max_length=10000,
                blocked_patterns=[
                    r'<script[^>]*>.*?</script>',
                    r'javascript:',
                    r'eval\s*\(',
                    r'exec\s*\('
                ]
            )
        )

    if enable_rate_limit:
        middleware_stack.append(
            RateLimitMiddleware(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_limit=10
            )
        )

    if enable_audit:
        middleware_stack.append(
            AuditLoggingMiddleware(
                log_requests=True,
                log_responses=True,
                log_pii=False,  # 不记录PII内容
                max_content_length=500
            )
        )

    logger.info(f"🛡️ 安全中间件栈已创建，包含 {len(middleware_stack)} 个中间件")
    return middleware_stack
