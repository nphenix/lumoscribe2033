"""
系统集成测试

测试系统各组件之间的集成，包括：
- API与数据库集成
- 缓存与存储集成
- 错误处理集成
- 监控系统集成
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.framework.shared.error_handler import ErrorHandler
from src.framework.shared.monitoring import MetricsCollector
from src.framework.storage.enhanced_vector_store import EnhancedVectorStoreManager
from src.framework.shared.config import get_settings, get_config_manager


@pytest.mark.skip(reason="系统集成测试依赖未实现的 API 端点和完整功能，阶段 3-4 实现")
class TestSystemIntegration:
    """系统集成测试"""

    def setup_method(self):
        """测试前设置"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.persist_dir = Path(self.temp_dir) / "chroma"
        self.persist_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """测试后清理"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    @pytest.mark.asyncio
    async def test_api_database_integration(self):
        """测试API与数据库集成"""
        # 创建应用
        app = create_app()
        
        # 使用内存数据库进行测试
        with patch.object(get_settings(), 'DATABASE_URL', 'sqlite:///:memory:'):
            client = TestClient(app)
            
            # 测试健康检查端点
            response = client.get("/health")
            assert response.status_code == 200
            assert "status" in response.json()

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """测试错误处理集成"""
        # 创建错误处理器
        error_handler = ErrorHandler(max_retries=2)
        
        # 测试集成错误处理
        call_count = 0
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("模拟连接失败")
            return "success"
        
        # 使用错误处理器执行操作
        result = await error_handler.execute_with_retry(failing_operation)
        
        assert result == "success"
        assert call_count == 3  # 初始调用 + 2次重试

    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """测试监控集成"""
        # 创建指标收集器
        collector = MetricsCollector()
        
        # 记录一些测试指标
        collector.record_api_metric(
            endpoint="/test/integration",
            method="GET",
            status_code=200,
            response_time=0.1,
            request_size=100,
            response_size=500
        )
        
        collector.record_system_metric(
            cpu_percent=45.5,
            memory_percent=67.2,
            disk_usage_percent=78.1,
            network_sent_bytes=1024,
            network_recv_bytes=2048
        )
        
        # 获取摘要
        api_summary = collector.get_api_summary(hours=1)
        system_summary = collector.get_system_summary(hours=1)
        
        assert len(api_summary) > 0
        assert len(system_summary) > 0

    @pytest.mark.asyncio
    async def test_config_integration(self):
        """测试配置集成"""
        # 获取配置管理器
        config_manager = get_config_manager()
        
        # 测试模型配置
        enabled_models = config_manager.get_enabled_models()
        assert isinstance(enabled_models, dict)
        
        # 测试路由配置
        routing_config = config_manager.get_routing_config()
        assert routing_config is not None
        assert hasattr(routing_config, 'enable_performance_routing')

    @pytest.mark.asyncio
    async def test_vector_storage_integration(self):
        """测试向量存储集成"""
        # 创建向量存储管理器
        vector_manager = EnhancedVectorStoreManager(persist_dir=str(self.persist_dir))
        
        # 创建测试文档
        from llama_index.core.schema import Document as LlamaDocument
        test_docs = [
            LlamaDocument(
                text="集成测试文档1",
                metadata={"source": "integration_test", "type": "test"}
            ),
            LlamaDocument(
                text="集成测试文档2",
                metadata={"source": "integration_test", "type": "test"}
            )
        ]
        
        # 创建索引
        index = vector_manager.create_index(test_docs, collection_name="integration_test")
        assert index is not None
        
        # 测试查询
        retriever = index.as_retriever(similarity_top_k=2)
        results = retriever.retrieve("测试")
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 创建应用
        app = create_app()
        
        with patch.object(get_settings(), 'DATABASE_URL', 'sqlite:///:memory:'):
            client = TestClient(app)
            
            # 1. 上传文档
            upload_data = {
                "content": "端到端测试文档",
                "metadata": {"source": "e2e_test", "category": "test"}
            }
            
            upload_response = client.post("/api/documents", json=upload_data)
            assert upload_response.status_code == 200
            doc_id = upload_response.json()["id"]
            
            # 2. 处理文档
            process_response = client.post(f"/api/documents/{doc_id}/process")
            assert process_response.status_code == 200
            
            # 3. 查询文档
            query_response = client.get(f"/api/documents/{doc_id}")
            assert query_response.status_code == 200
            assert query_response.json()["content"] == "端到端测试文档"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """测试并发操作"""
        # 创建多个并发任务
        async def concurrent_task(task_id: int):
            # 模拟一些处理时间
            await asyncio.sleep(0.1)
            return {"task_id": task_id, "status": "completed"}
        
        # 并发执行多个任务
        tasks = [concurrent_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务完成
        assert len(results) == 10
        for i, result in enumerate(results):
            assert result["task_id"] == i
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """测试负载下性能"""
        # 创建指标收集器
        collector = MetricsCollector()
        
        # 模拟多个并发请求
        async def simulate_request():
            start_time = asyncio.get_event_loop().time()
            
            # 模拟请求处理
            await asyncio.sleep(0.05)
            
            end_time = asyncio.get_event_loop().time()
            
            # 记录指标
            collector.record_api_metric(
                endpoint="/api/test",
                method="GET",
                status_code=200,
                response_time=end_time - start_time,
                request_size=100,
                response_size=500
            )
            
            return {"status": "ok"}
        
        # 并发执行多个请求
        tasks = [simulate_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有请求成功
        assert len(results) == 20
        assert all(result["status"] == "ok" for result in results)
        
        # 检查性能指标
        api_summary = collector.get_api_summary(hours=1)
        assert api_summary["/api/test"]["total_requests"] == 20

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """测试错误恢复工作流"""
        # 创建错误处理器
        error_handler = ErrorHandler(
            max_retries=2,
            enable_circuit_breaker=True,
            circuit_breaker_failure_threshold=3
        )
        
        # 模拟服务故障和恢复
        failure_count = 0
        async def unreliable_service():
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 3:
                raise ConnectionError("服务暂时不可用")
            
            return "recovered"
        
        # 测试断路器开启
        with pytest.raises(Exception):  # 应该抛出断路器异常
            await error_handler.execute_with_retry(unreliable_service)
        
        assert error_handler.is_circuit_open() is True
        
        # 等待恢复时间
        await asyncio.sleep(0.1)
        
        # 测试恢复
        result = await error_handler.execute_with_retry(unreliable_service)
        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_data_consistency(self):
        """测试数据一致性"""
        # 创建应用
        app = create_app()
        
        with patch.object(get_settings(), 'DATABASE_URL', 'sqlite:///:memory:'):
            client = TestClient(app)
            
            # 创建文档
            doc_data = {
                "content": "数据一致性测试",
                "metadata": {"version": 1, "type": "consistency_test"}
            }
            
            create_response = client.post("/api/documents", json=doc_data)
            assert create_response.status_code == 200
            doc_id = create_response.json()["id"]
            
            # 更新文档
            update_data = {
                "content": "数据一致性测试 - 已更新",
                "metadata": {"version": 2, "type": "consistency_test"}
            }
            
            update_response = client.put(f"/api/documents/{doc_id}", json=update_data)
            assert update_response.status_code == 200
            
            # 验证数据一致性
            get_response = client.get(f"/api/documents/{doc_id}")
            assert get_response.status_code == 200
            
            doc_data = get_response.json()
            assert doc_data["content"] == "数据一致性测试 - 已更新"
            assert doc_data["metadata"]["version"] == 2

    @pytest.mark.asyncio
    async def test_security_integration(self):
        """测试安全集成"""
        # 创建应用
        app = create_app()
        
        with patch.object(get_settings(), 'DATABASE_URL', 'sqlite:///:memory:'):
            client = TestClient(app)
            
            # 测试未认证请求
            response = client.get("/api/protected")
            assert response.status_code == 401
            
            # 测试认证请求
            # 注意：这里假设有认证端点，实际实现可能不同
            auth_response = client.post("/api/auth/login", json={
                "username": "test_user",
                "password": "test_password"
            })
            
            # 如果认证端点存在，应该返回token
            if auth_response.status_code == 200:
                token = auth_response.json().get("token")
                if token:
                    # 使用token访问受保护资源
                    protected_response = client.get(
                        "/api/protected",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    assert protected_response.status_code == 200

    @pytest.mark.asyncio
    async def test_monitoring_dashboard_integration(self):
        """测试监控仪表板集成"""
        # 创建应用
        app = create_app()
        
        # 添加一些监控数据
        collector = MetricsCollector()
        
        # 模拟一些API请求
        for i in range(5):
            collector.record_api_metric(
                endpoint=f"/api/endpoint{i}",
                method="GET",
                status_code=200 if i < 4 else 500,
                response_time=0.1 + i * 0.02,
                request_size=100,
                response_size=500
            )
        
        with patch.object(get_settings(), 'DATABASE_URL', 'sqlite:///:memory:'):
            client = TestClient(app)
            
            # 获取监控仪表板数据
            response = client.get("/api/monitoring/dashboard")
            assert response.status_code == 200
            
            dashboard_data = response.json()
            assert "metrics" in dashboard_data
            assert "health" in dashboard_data
            assert "performance" in dashboard_data


class TestErrorHandlingIntegration:
    """错误处理集成测试"""

    @pytest.mark.asyncio
    async def test_cascade_failure_handling(self):
        """测试级联失败处理"""
        # 创建错误处理器
        error_handler = ErrorHandler(enable_circuit_breaker=True)
        
        # 模拟级联失败
        failure_count = 0
        async def cascading_failure():
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:
                raise ConnectionError(f"级联失败 {failure_count}")
            
            return "recovered"
        
        # 多个并发任务应该都失败
        tasks = [cascading_failure() for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证断路器开启
        assert error_handler.is_circuit_open() is True
        
        # 后续请求应该快速失败
        with pytest.raises(Exception):
            await error_handler.execute_with_retry(cascading_failure)

    @pytest.mark.asyncio
    async def test_graceful_degradation_integration(self):
        """测试优雅降级集成"""
        from src.framework.shared.error_handler import ErrorRecoveryStrategy
        
        # 测试网络错误降级
        fallback_data = {"status": "ok", "data": "cached_data"}
        result = ErrorRecoveryStrategy.graceful_degradation(
            ConnectionError("网络不可用"),
            fallback_data
        )
        
        assert result == fallback_data
        
        # 测试数据库错误降级
        result = ErrorRecoveryStrategy.graceful_degradation(
            Exception("数据库连接失败")
        )
        
        assert result["status"] == "degraded"
        assert "数据库" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])