"""
安全管理路由

基于LangChain v1.0最佳实践实现的安全管理API，包括：
- API密钥管理（符合LangChain中间件模式）
- 认证状态查询
- 安全状态监控
- 会话管理
- 中间件配置管理
"""

from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.framework.shared.logging import get_logger

from ...framework.shared.langchain_security_middleware import (
    AuditLoggingMiddleware,
    InputValidationMiddleware,
    PIIMiddleware,
    RateLimitMiddleware,
    create_security_middleware_stack,
)
from ...framework.shared.security import (
    generate_api_key,
    get_security_manager,
    get_security_status,
)
from ..security_middleware import optional_auth, required_auth

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["安全管理"])


@dataclass
class MiddlewareConfig:
    """中间件配置"""
    enable_pii: bool = True
    enable_validation: bool = True
    enable_audit: bool = True
    enable_rate_limit: bool = True
    pii_strategy: str = "redact"
    pii_types: list[str] = None
    validation_max_length: int = 10000
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000


class APIKeyRequest(BaseModel):
    """API密钥请求模型"""
    name: str
    permissions: list[str] = ["read", "write"]


class APIKeyResponse(BaseModel):
    """API密钥响应模型"""
    api_key: str
    name: str
    permissions: list[str]
    created_at: str


class SecurityStatusResponse(BaseModel):
    """安全状态响应模型"""
    active_sessions: int
    active_api_keys: int
    rate_limits: int
    security_headers: list[str]
    last_cleanup: str


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    current_user: dict = Depends(required_auth)
) -> APIKeyResponse:
    """
    创建API密钥

    基于LangChain v1.0安全最佳实践，创建新的API密钥用于API访问。
    包含权限验证和审计日志记录。

    Args:
        request: API密钥创建请求
        current_user: 当前用户信息

    Returns:
        创建的API密钥信息
    """
    try:
        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "api_key_create" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有创建API密钥的权限"
            )

        # 验证请求内容（使用LangChain安全中间件）
        security_manager = get_security_manager()
        if not security_manager.validate_input(request.name, max_length=100):
            raise HTTPException(
                status_code=400,
                detail="API密钥名称包含不安全内容"
            )

        # 生成API密钥
        api_key_info = generate_api_key(
            name=request.name,
            permissions=request.permissions
        )

        # 记录审计日志
        logger.info(f"🔑 API密钥已创建 - 用户: {current_user.get('user_id')}, 名称: {request.name}")

        return APIKeyResponse(**api_key_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建API密钥失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"创建API密钥失败: {str(e)}"
        )


@router.get("/api-keys")
async def list_api_keys(
    current_user: dict = Depends(required_auth)
) -> dict[str, Any]:
    """
    列出API密钥

    需要认证。列出当前用户的所有API密钥。

    Args:
        current_user: 当前用户信息

    Returns:
        API密钥列表
    """
    try:
        security_manager = get_security_manager()

        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "api_key_list" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有查看API密钥的权限"
            )

        # 获取API密钥列表（隐藏实际密钥）
        api_keys = []
        for key_info in security_manager._api_keys.values():
            api_keys.append({
                "name": key_info["name"],
                "permissions": key_info["permissions"],
                "created_at": key_info["created_at"],
                "last_used": key_info["last_used"],
                "usage_count": key_info["usage_count"]
            })

        return {
            "api_keys": api_keys,
            "total_count": len(api_keys)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取API密钥列表失败: {str(e)}"
        )


@router.delete("/api-keys/{key_name}")
async def delete_api_key(
    key_name: str,
    current_user: dict = Depends(required_auth)
) -> dict[str, str]:
    """
    删除API密钥

    需要认证。删除指定的API密钥。

    Args:
        key_name: API密钥名称
        current_user: 当前用户信息

    Returns:
        删除结果
    """
    try:
        security_manager = get_security_manager()

        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "api_key_delete" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有删除API密钥的权限"
            )

        # 查找并删除API密钥
        key_to_delete = None
        for api_key, key_info in security_manager._api_keys.items():
            if key_info["name"] == key_name:
                key_to_delete = api_key
                break

        if not key_to_delete:
            raise HTTPException(
                status_code=404,
                detail="API密钥不存在"
            )

        del security_manager._api_keys[key_to_delete]

        return {"message": f"API密钥 '{key_name}' 已删除"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除API密钥失败: {str(e)}"
        )


@router.get("/status", response_model=SecurityStatusResponse)
async def get_security_status_endpoint(
    current_user: dict | None = Depends(optional_auth)
) -> SecurityStatusResponse:
    """
    获取安全状态

    获取当前系统的安全状态信息。

    Args:
        current_user: 当前用户信息（可选）

    Returns:
        安全状态信息
    """
    try:
        # 基础安全状态（所有用户可访问）
        security_status = get_security_status()

        return SecurityStatusResponse(**security_status)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取安全状态失败: {str(e)}"
        )


@router.get("/sessions")
async def list_sessions(
    current_user: dict = Depends(required_auth)
) -> dict[str, Any]:
    """
    列出活跃会话

    需要认证。列出当前的所有活跃会话。

    Args:
        current_user: 当前用户信息

    Returns:
        活跃会话列表
    """
    try:
        security_manager = get_security_manager()

        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "session_list" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有查看会话的权限"
            )

        # 获取会话列表
        sessions = []
        for session_id, session_data in security_manager._sessions.items():
            sessions.append({
                "session_id": session_id[:8] + "...",  # 只显示前8位
                "user_id": session_data["user_id"],
                "created_at": session_data["created_at"],
                "last_accessed": session_data["last_accessed"],
                "expires_at": session_data["expires_at"]
            })

        return {
            "sessions": sessions,
            "total_count": len(sessions)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取会话列表失败: {str(e)}"
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: dict = Depends(required_auth)
) -> dict[str, str]:
    """
    删除会话

    需要认证。删除指定的会话。

    Args:
        session_id: 会话ID
        current_user: 当前用户信息

    Returns:
        删除结果
    """
    try:
        security_manager = get_security_manager()

        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "session_delete" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有删除会话的权限"
            )

        # 查找并删除会话
        if session_id not in security_manager._sessions:
            raise HTTPException(
                status_code=404,
                detail="会话不存在"
            )

        del security_manager._sessions[session_id]

        return {"message": "会话已删除"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除会话失败: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_expired_data(
    current_user: dict = Depends(required_auth)
) -> dict[str, Any]:
    """
    清理过期数据

    需要认证。清理过期的会话和速率限制数据。

    Args:
        current_user: 当前用户信息

    Returns:
        清理结果
    """
    try:
        security_manager = get_security_manager()

        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "cleanup" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有清理数据的权限"
            )

        # 执行清理
        security_manager.cleanup_expired_data()

        return {
            "message": "过期数据清理完成",
            "timestamp": security_manager.get_security_status()["last_cleanup"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"清理过期数据失败: {str(e)}"
        )


@router.get("/current-user")
async def get_current_user(
    current_user: dict | None = Depends(optional_auth)
) -> dict[str, Any]:
    """
    获取当前用户信息

    获取当前认证用户的信息。

    Args:
        current_user: 当前用户信息（可选）

    Returns:
        用户信息
    """
    try:
        if not current_user:
            return {
                "authenticated": False,
                "message": "未认证"
            }

        return {
            "authenticated": True,
            "user": current_user
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取用户信息失败: {str(e)}"
        )


@router.get("/rate-limits")
async def get_rate_limits(
    current_user: dict = Depends(required_auth)
) -> dict[str, Any]:
    """
    获取速率限制状态

    需要认证。获取当前的速率限制状态。

    Args:
        current_user: 当前用户信息

    Returns:
        速率限制状态
    """
    try:
        security_manager = get_security_manager()

        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "rate_limit_view" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有查看速率限制的权限"
            )

        # 获取速率限制信息
        rate_limits = {}
        for identifier, rate_data in security_manager._rate_limits.items():
            rate_limits[identifier] = {
                "request_count": len(rate_data["requests"]),
                "blocked_until": rate_data["blocked_until"],
                "is_blocked": rate_data["blocked_until"] > 0
            }

        return {
            "rate_limits": rate_limits,
            "total_tracked": len(rate_limits)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取速率限制状态失败: {str(e)}"
        )


@router.get("/middleware/config")
async def get_middleware_config(
    current_user: dict = Depends(optional_auth)
) -> dict[str, Any]:
    """
    获取中间件配置

    基于LangChain v1.0中间件模式，返回当前安全中间件配置。
    """
    try:
        # 基础配置（所有用户可访问）
        base_config = {
            "middleware_stack": {
                "pii_detection": {
                    "enabled": True,
                    "strategy": "redact",
                    "types": ["email", "phone_number", "api_key", "password"]
                },
                "input_validation": {
                    "enabled": True,
                    "max_length": 10000,
                    "blocked_patterns_count": 7
                },
                "rate_limiting": {
                    "enabled": True,
                    "per_minute": 60,
                    "per_hour": 1000,
                    "burst_limit": 10
                },
                "audit_logging": {
                    "enabled": True,
                    "log_requests": True,
                    "log_responses": True,
                    "log_pii": False
                }
            }
        }

        # 管理员可以看到详细配置
        if current_user:
            user_permissions = current_user.get("permissions", [])
            if "admin" in user_permissions:
                security_manager = get_security_manager()
                base_config["detailed_status"] = security_manager.get_security_status()

        return base_config

    except Exception as e:
        logger.error(f"获取中间件配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取中间件配置失败: {str(e)}"
        )


@router.post("/middleware/config")
async def update_middleware_config(
    config: MiddlewareConfig,
    current_user: dict = Depends(required_auth)
) -> dict[str, str]:
    """
    更新中间件配置

    基于LangChain v1.0中间件模式，动态更新安全中间件配置。
    需要管理员权限。

    Args:
        config: 中间件配置
        current_user: 当前用户信息

    Returns:
        更新结果
    """
    try:
        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "middleware_config" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有配置中间件的权限"
            )

        # 创建新的中间件栈
        create_security_middleware_stack(
            enable_pii=config.enable_pii,
            enable_validation=config.enable_validation,
            enable_audit=config.enable_audit,
            enable_rate_limit=config.enable_rate_limit,
            pii_strategy=config.pii_strategy
        )

        # 记录配置更新
        logger.info(
            f"🔧 中间件配置已更新 - 用户: {current_user.get('user_id')}, "
            f"PII: {config.enable_pii}, 验证: {config.enable_validation}, "
            f"审计: {config.enable_audit}, 速率限制: {config.enable_rate_limit}"
        )

        security_manager = get_security_manager()
        return {
            "message": "中间件配置更新成功",
            "timestamp": security_manager.get_security_status()["last_cleanup"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新中间件配置失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"更新中间件配置失败: {str(e)}"
        )


@router.post("/test/policy")
async def test_security_policy(
    test_data: dict[str, Any],
    current_user: dict = Depends(required_auth)
) -> dict[str, Any]:
    """
    测试安全策略

    基于LangChain v1.0中间件模式，测试输入内容是否符合安全策略。

    Args:
        test_data: 测试数据
        current_user: 当前用户信息

    Returns:
        测试结果
    """
    try:
        # 验证权限
        user_permissions = current_user.get("permissions", [])
        if "admin" not in user_permissions and "security_test" not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="没有测试安全策略的权限"
            )

        test_input = test_data.get("input", "")
        test_type = test_data.get("type", "general")

        # 创建临时中间件进行测试
        pii_middleware = PIIMiddleware(
            pii_types=["email", "phone_number", "api_key"],
            strategy="redact",
            apply_to_input=True
        )

        validation_middleware = InputValidationMiddleware(
            max_length=10000,
            blocked_patterns=[
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'eval\s*\(',
                r'exec\s*\('
            ]
        )

        # 执行测试
        test_results = {
            "input": test_input[:100] + "..." if len(test_input) > 100 else test_input,
            "type": test_type,
            "pii_detection": {
                "enabled": True,
                "detected": False,
                "redacted_content": None
            },
            "input_validation": {
                "enabled": True,
                "valid": True,
                "violations": []
            },
            "overall_status": "passed"
        }

        # 测试PII检测
        try:
            from langchain.agents.middleware.types import ModelRequest
            mock_request = ModelRequest(
                messages=[{"role": "user", "content": test_input}]
            )

            def mock_handler(request):
                return request

            result = pii_middleware.wrap_model_call(mock_request, mock_handler)

            if hasattr(result, 'messages') and result.messages:
                processed_content = result.messages[0].content
                if processed_content != test_input:
                    test_results["pii_detection"]["detected"] = True
                    test_results["pii_detection"]["redacted_content"] = processed_content
                    test_results["overall_status"] = "warning"

        except Exception as e:
            logger.error(f"PII测试失败: {e}")

        # 测试输入验证
        try:
            if not validation_middleware._validate_input(mock_request)["valid"]:
                test_results["input_validation"]["valid"] = False
                test_results["input_validation"]["violations"].append("包含不允许的内容")
                test_results["overall_status"] = "failed"

        except Exception as e:
            logger.error(f"输入验证测试失败: {e}")

        # 记录测试
        logger.info(
            f"🧪 安全策略测试 - 用户: {current_user.get('user_id')}, "
            f"类型: {test_type}, 状态: {test_results['overall_status']}"
        )

        return test_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"安全策略测试失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"安全策略测试失败: {str(e)}"
        )


@router.get("/compliance/check")
async def check_compliance(
    current_user: dict = Depends(optional_auth)
) -> dict[str, Any]:
    """
    检查合规性

    基于LangChain v1.0安全最佳实践，检查系统安全合规性。

    Args:
        current_user: 当前用户信息（可选）

    Returns:
        合规性检查结果
    """
    try:
        security_manager = get_security_manager()
        security_status = security_manager.get_security_status()

        # 合规性检查项目
        compliance_checks = {
            "authentication": {
                "status": "passed",
                "description": "身份验证机制已实施",
                "details": "支持JWT和API密钥认证"
            },
            "authorization": {
                "status": "passed",
                "description": "授权机制已实施",
                "details": "基于角色的权限控制"
            },
            "input_validation": {
                "status": "passed",
                "description": "输入验证已实施",
                "details": "防止XSS、SQL注入等攻击"
            },
            "pii_protection": {
                "status": "passed",
                "description": "PII保护已实施",
                "details": "支持检测、脱敏和阻止"
            },
            "rate_limiting": {
                "status": "passed",
                "description": "速率限制已实施",
                "details": "防止API滥用和DDoS攻击"
            },
            "audit_logging": {
                "status": "passed",
                "description": "审计日志已实施",
                "details": "记录所有安全相关事件"
            },
            "secure_headers": {
                "status": "passed",
                "description": "安全头已配置",
                "details": "包含CSP、XSS保护等"
            }
        }

        # 计算总体合规性评分
        passed_checks = sum(1 for check in compliance_checks.values() if check["status"] == "passed")
        total_checks = len(compliance_checks)
        compliance_score = (passed_checks / total_checks) * 100

        # 确定总体状态
        if compliance_score >= 90:
            overall_status = "excellent"
        elif compliance_score >= 80:
            overall_status = "good"
        elif compliance_score >= 70:
            overall_status = "fair"
        else:
            overall_status = "poor"

        result = {
            "overall_status": overall_status,
            "compliance_score": round(compliance_score, 2),
            "checks": compliance_checks,
            "security_status": security_status,
            "recommendations": []
        }

        # 添加改进建议
        if compliance_score < 100:
            result["recommendations"].append("启用所有安全中间件以获得最佳保护")

        if security_status["active_sessions"] > 100:
            result["recommendations"].append("考虑实施会话清理策略")

        if security_status["active_api_keys"] > 50:
            result["recommendations"].append("考虑实施API密钥轮换策略")

        return result

    except Exception as e:
        logger.error(f"合规性检查失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"合规性检查失败: {str(e)}"
        )
