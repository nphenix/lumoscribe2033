"""
监控中间件单元测试

测试监控中间件的各种功能，包括：
- 请求处理
- 错误处理
- 性能监控
- 中间件集成
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from src.api.monitoring_middleware import MonitoringMiddleware


class TestMonitoringMiddleware:
    """测试监控中间件"""

    def setup_method(self):
        """测试前设置"""
        self.app = Mock()
        self.app.state = {}
        self.middleware = MonitoringMiddleware(self.app)

    @pytest.mark.asyncio
    async def test_request_processing(self):
        """测试请求处理"""
        # 创建模拟请求
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/test"
        request.url.query_string = ""
        request.headers = {"user-agent": "test-agent"}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        # 创建模拟响应
        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        
        # 处理请求
        await self.middleware.process_request(request, response)
        
        # 验证指标已记录
        assert len(self.middleware.metrics_collector.api_metrics) == 1
        metric = self.middleware.metrics_collector.api_metrics[0]
        assert metric.method == "GET"
        assert metric.endpoint == "/test"
        assert metric.status_code == 200

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        # 创建模拟请求
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/error"
        
        # 创建模拟响应
        response = Mock(spec=Response)
        response.status_code = 500
        
        # 处理请求
        await self.middleware.process_request(request, response)
        
        # 验证错误指标已记录
        assert len(self.middleware.metrics_collector.api_metrics) == 1
        metric = self.middleware.metrics_collector.api_metrics[0]
        assert metric.status_code == 500

    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """测试性能监控"""
        # 创建模拟请求
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/slow"
        
        # 创建模拟响应
        response = Mock(spec=Response)
        response.status_code = 200
        
        # 处理请求（模拟慢处理）
        start_time = time.time()
        await self.middleware.process_request(request, response)
        end_time = time.time()
        
        # 验证性能监控
        assert len(self.middleware.performance_monitor.active_operations) == 0  # 应该已结束
        stats = self.middleware.performance_monitor.get_performance_stats()
        assert stats["total_operations"] >= 1

    def test_get_monitoring_dashboard_data(self):
        """测试获取监控仪表板数据"""
        # 添加一些测试数据
        self.middleware.metrics_collector.record_api_metric(
            "/dashboard", "GET", 200, 0.1, 100, 500, "127.0.0.1"
        )
        self.middleware.health_checker.add_check(
            "dashboard_check",
            lambda: {"status": "healthy", "message": "OK"}
        )
        
        dashboard_data = self.middleware.get_monitoring_dashboard_data()
        
        assert "metrics" in dashboard_data
        assert "health" in dashboard_data
        assert "performance" in dashboard_data
        assert "timestamp" in dashboard_data

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="中间件集成测试依赖未实现的完整 FastAPI 应用，阶段 3 实现")
    async def test_middleware_integration(self):
        """测试中间件集成"""
        from fastapi import FastAPI
        
        # 创建FastAPI应用
        app = FastAPI()
        
        # 添加中间件
        app.add_middleware(MonitoringMiddleware)
        
        # 添加测试路由
        @app.get("/test-monitoring")
        async def test_route():
            return {"message": "test"}
        
        # 创建测试客户端
        client = TestClient(app)
        
        # 发送请求
        response = client.get("/test-monitoring")
        
        assert response.status_code == 200
        assert response.json()["message"] == "test"
        
        # 验证监控数据已收集
        dashboard_data = app.middleware_stack[0].get_monitoring_dashboard_data()
        assert len(dashboard_data["metrics"]["api_summary"]) > 0

    @pytest.mark.asyncio
    async def test_request_with_query_params(self):
        """测试带查询参数的请求"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/users"
        request.url.query_string = "page=1&limit=10"
        request.headers = {}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        response = Mock(spec=Response)
        response.status_code = 200
        
        await self.middleware.process_request(request, response)
        
        metric = self.middleware.metrics_collector.api_metrics[0]
        assert metric.endpoint == "/api/users"
        # 查询参数应该被记录但不影响端点识别

    @pytest.mark.asyncio
    async def test_request_size_calculation(self):
        """测试请求大小计算"""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/upload"
        request.headers = {"content-length": "1024"}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        response = Mock(spec=Response)
        response.status_code = 201
        response.headers = {"content-length": "512"}
        
        await self.middleware.process_request(request, response)
        
        metric = self.middleware.metrics_collector.api_metrics[0]
        assert metric.request_size == 1024
        assert metric.response_size == 512

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """测试并发请求处理"""
        async def process_request_async(path, status_code):
            request = Mock(spec=Request)
            request.method = "GET"
            request.url = Mock()
            request.url.path = path
            request.headers = {}
            request.client = Mock()
            request.client.host = "127.0.0.1"
            
            response = Mock(spec=Response)
            response.status_code = status_code
            
            await self.middleware.process_request(request, response)
        
        # 并发处理多个请求
        tasks = [
            process_request_async(f"/api/{i}", 200)
            for i in range(10)
        ]
        
        await asyncio.gather(*tasks)
        
        # 验证所有请求都被记录
        assert len(self.middleware.metrics_collector.api_metrics) == 10
        
        # 验证性能统计
        stats = self.middleware.performance_monitor.get_performance_stats()
        assert stats["total_operations"] == 10

    @pytest.mark.asyncio
    async def test_request_headers_logging(self):
        """测试请求头记录"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/headers"
        request.headers = {
            "user-agent": "TestAgent/1.0",
            "x-request-id": "req-123",
            "authorization": "Bearer token123"
        }
        request.client = Mock()
        request.client.host = "127.0.0.1"
        
        response = Mock(spec=Response)
        response.status_code = 200
        
        await self.middleware.process_request(request, response)
        
        metric = self.middleware.metrics_collector.api_metrics[0]
        # 验证敏感头被过滤
        assert "authorization" not in metric.headers
        # 验证普通头被记录
        assert metric.headers.get("user-agent") == "TestAgent/1.0"
        assert metric.headers.get("x-request-id") == "req-123"

    @pytest.mark.asyncio
    async def test_middleware_error_handling(self):
        """测试中间件错误处理"""
        # 模拟指标收集器抛出异常
        with patch.object(self.middleware.metrics_collector, 'record_api_metric', side_effect=Exception("Test error")):
            request = Mock(spec=Request)
            request.method = "GET"
            request.url = Mock()
            request.url.path = "/error-test"
            request.headers = {}
            request.client = Mock()
            request.client.host = "127.0.0.1"
            
            response = Mock(spec=Response)
            response.status_code = 200
            
            # 中间件应该优雅处理错误，不抛出异常
            await self.middleware.process_request(request, response)
            
            # 即使指标记录失败，也不应该影响请求处理
            assert response.status_code == 200

    def test_middleware_configuration(self):
        """测试中间件配置"""
        # 测试默认配置
        assert self.middleware.enable_performance_monitoring is True
        assert self.middleware.enable_metrics_collection is True
        assert self.middleware.enable_health_checks is True
        
        # 测试自定义配置
        custom_middleware = MonitoringMiddleware(
            self.app,
            enable_performance_monitoring=False,
            enable_metrics_collection=False,
            enable_health_checks=False
        )
        
        assert custom_middleware.enable_performance_monitoring is False
        assert custom_middleware.enable_metrics_collection is False
        assert custom_middleware.enable_health_checks is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])