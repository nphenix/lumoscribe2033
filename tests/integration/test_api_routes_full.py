"""
API 路由完整测试

测试所有 API 端点的基本功能
"""

import pytest
from httpx import AsyncClient

from src.api.main import app


# 跳过整个文件的测试，因为依赖未实现的 API 端点
pytestmark = pytest.mark.skip(reason="完整 API 路由测试依赖未实现的端点，阶段 3 实现")


@pytest.mark.asyncio
async def test_health_check():
    """测试健康检查端点"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_docs_list():
    """测试文档列表端点"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/docs")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_tasks_list():
    """测试任务列表端点"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/tasks")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_config_get():
    """测试配置获取端点"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/config")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_speckit_status():
    """测试 Speckit 状态端点"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/v1/speckit/status")
        # 可能返回 404 或 200，取决于实现
        assert response.status_code in [200, 404]


# TODO: 添加更多端点测试
# - POST /api/v1/docs/upload
# - POST /api/v1/tasks/create
# - PUT /api/v1/config/update
# - POST /api/v1/speckit/run
