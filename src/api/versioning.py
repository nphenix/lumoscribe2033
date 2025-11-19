"""
API 版本控制

基于 FastAPI 最佳实践实现 API 版本控制，支持：
- 路径版本控制 (/v1/, /v2/)
- 请求头版本控制
- 向后兼容性
- 版本迁移
- API 文档版本化
"""

from collections.abc import Callable
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.routing import APIRoute
from starlette.status import HTTP_400_BAD_REQUEST


class VersionedAPIRouter(APIRouter):
    """版本化 API 路由器"""

    def __init__(
        self,
        version: str,
        default: bool = False,
        deprecated_versions: list | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.version = version
        self.default = default
        self.deprecated_versions = deprecated_versions or []

        # 添加版本前缀
        if not kwargs.get("prefix"):
            kwargs["prefix"] = f"/{version}"

        # 版本特定的标签
        if not kwargs.get("tags"):
            kwargs["tags"] = [f"api-{version}"]

    def add_api_route(self, path: str, endpoint: Callable, **kwargs) -> None:
        """添加 API 路由，自动添加版本信息"""
        # 添加版本到响应模型的描述中
        if "response_model" in kwargs:
            response_model = kwargs["response_model"]
            if hasattr(response_model, "__name__"):
                version_suffix = f" (API {self.version})"
                if hasattr(response_model, "Config"):
                    if hasattr(response_model.Config, "title"):
                        response_model.Config.title += version_suffix

        # 添加版本到路由标签
        if "tags" not in kwargs or kwargs["tags"] is None:
            kwargs["tags"] = []
        kwargs["tags"].append(self.version)

        super().add_api_route(path, endpoint, **kwargs)


class APIVersionMiddleware:
    """API 版本中间件"""

    def __init__(
        self,
        supported_versions: list,
        default_version: str = "v1",
        deprecated_versions: dict | None = None
    ):
        self.supported_versions = supported_versions
        self.default_version = default_version
        self.deprecated_versions = deprecated_versions or {}

    async def __call__(self, request: Request, call_next):
        """处理请求版本控制"""
        # 从路径提取版本
        path_version = self._extract_version_from_path(request.url.path)

        # 从请求头提取版本
        header_version = request.headers.get("X-API-Version")

        # 确定最终版本
        final_version = self._determine_version(path_version, header_version)

        # 验证版本
        if final_version not in self.supported_versions:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"不支持的 API 版本: {final_version}. 支持的版本: {', '.join(self.supported_versions)}"
            )

        # 检查是否已弃用
        if final_version in self.deprecated_versions:
            deprecated_info = self.deprecated_versions[final_version]
            request.state.api_version_warning = {
                "version": final_version,
                "message": deprecated_info.get("message", "此版本已弃用"),
                "replacement": deprecated_info.get("replacement"),
                "deprecation_date": deprecated_info.get("deprecation_date")
            }

        # 添加版本信息到请求状态
        request.state.api_version = final_version

        # 继续处理请求
        response = await call_next(request)

        # 添加版本信息到响应头
        response.headers["X-API-Version"] = final_version
        if final_version in self.deprecated_versions:
            response.headers["X-API-Deprecated"] = "true"

        return response

    def _extract_version_from_path(self, path: str) -> str | None:
        """从路径提取版本"""
        path_parts = path.strip("/").split("/")
        if path_parts and path_parts[0].startswith("v"):
            return path_parts[0]
        return None

    def _determine_version(self, path_version: str | None, header_version: str | None) -> str:
        """确定最终使用的 API 版本"""
        if header_version and header_version in self.supported_versions:
            return header_version
        elif path_version and path_version in self.supported_versions:
            return path_version
        else:
            return self.default_version


class VersionCompatibility:
    """版本兼容性管理"""

    def __init__(self):
        self.version_mappings = {}
        self.deprecated_endpoints = {}

    def add_version_mapping(
        self,
        from_version: str,
        to_version: str,
        endpoint_mapping: dict[str, str]
    ) -> None:
        """添加版本映射"""
        if from_version not in self.version_mappings:
            self.version_mappings[from_version] = {}
        self.version_mappings[from_version][to_version] = endpoint_mapping

    def add_deprecated_endpoint(
        self,
        version: str,
        endpoint: str,
        replacement: str | None = None,
        removal_date: str | None = None
    ) -> None:
        """添加已弃用的端点"""
        if version not in self.deprecated_endpoints:
            self.deprecated_endpoints[version] = {}
        self.deprecated_endpoints[version][endpoint] = {
            "replacement": replacement,
            "removal_date": removal_date
        }

    def get_compatible_endpoint(
        self,
        version: str,
        endpoint: str,
        target_version: str
    ) -> str | None:
        """获取兼容的端点"""
        if version in self.version_mappings:
            if target_version in self.version_mappings[version]:
                mappings = self.version_mappings[version][target_version]
                return mappings.get(endpoint)
        return None


# 全局版本管理器
version_compatibility = VersionCompatibility()


# 支持的 API 版本
SUPPORTED_VERSIONS = ["v1", "v2"]
DEFAULT_API_VERSION = "v1"
DEPRECATED_VERSIONS = {
    "beta": {
        "message": "此版本为测试版本，将在 2025-12-31 废弃",
        "replacement": "v1",
        "deprecation_date": "2025-12-31"
    }
}


def create_versioned_router(version: str, **kwargs) -> VersionedAPIRouter:
    """创建版本化路由器"""
    return VersionedAPIRouter(
        version=version,
        default=(version == DEFAULT_API_VERSION),
        deprecated_versions=list(DEPRECATED_VERSIONS.keys()),
        **kwargs
    )


def get_api_version(request: Request) -> str:
    """从请求中获取 API 版本"""
    return getattr(request.state, "api_version", DEFAULT_API_VERSION)


def get_version_warning(request: Request) -> dict[str, Any] | None:
    """获取版本警告信息"""
    return getattr(request.state, "api_version_warning", None)


# 版本迁移函数
def migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """v1 到 v2 数据迁移"""
    # 这里实现具体的迁移逻辑
    migrated = data.copy()

    # 示例迁移：字段重命名
    if "old_field" in migrated:
        migrated["new_field"] = migrated.pop("old_field")

    # 示例迁移：数据结构调整
    if "metadata" in migrated and isinstance(migrated["metadata"], dict):
        if "legacy_info" in migrated["metadata"]:
            migrated["metadata"]["info"] = migrated["metadata"].pop("legacy_info")

    return migrated


def validate_api_version(version: str) -> bool:
    """验证 API 版本是否有效"""
    return version in SUPPORTED_VERSIONS


def is_version_deprecated(version: str) -> bool:
    """检查版本是否已弃用"""
    return version in DEPRECATED_VERSIONS


def get_version_info() -> dict[str, Any]:
    """获取版本信息"""
    return {
        "supported_versions": SUPPORTED_VERSIONS,
        "default_version": DEFAULT_API_VERSION,
        "deprecated_versions": DEPRECATED_VERSIONS,
        "current_version": DEFAULT_API_VERSION
    }
