"""
指标收集器单元测试

测试指标收集器的各种功能，包括：
- API 指标记录
- 系统指标记录
- 指标聚合
- 指标摘要
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.framework.shared.monitoring import MetricsCollector, get_metrics_collector


class TestMetricsCollector:
    """测试指标收集器"""

    def setup_method(self):
        """测试前设置"""
        self.collector = MetricsCollector()

    def test_record_api_metric(self):
        """测试记录API指标"""
        self.collector.record_api_metric(
            endpoint="/test/api",
            method="GET",
            status_code=200,
            response_time=0.1,
            request_size=100,
            response_size=500,
            client_ip="127.0.0.1"
        )
        
        assert len(self.collector.api_metrics) == 1
        metric = self.collector.api_metrics[0]
        assert metric.endpoint == "/test/api"
        assert metric.method == "GET"
        assert metric.status_code == 200
        assert metric.response_time == 0.1

    def test_record_system_metric(self):
        """测试记录系统指标"""
        self.collector.record_system_metric(
            cpu_percent=45.5,
            memory_percent=67.2,
            disk_usage_percent=78.1,
            network_sent_bytes=1024,
            network_recv_bytes=2048
        )
        
        assert len(self.collector.system_metrics) == 1
        metric = self.collector.system_metrics[0]
        assert metric.cpu_percent == 45.5
        assert metric.memory_percent == 67.2

    def test_get_metrics_summary(self):
        """测试获取指标摘要"""
        # 添加一些测试数据
        self.collector.record_api_metric("/api1", "GET", 200, 0.1, 100, 500, "127.0.0.1")
        self.collector.record_api_metric("/api2", "POST", 201, 0.2, 200, 1000, "127.0.0.1")
        self.collector.record_api_metric("/api3", "GET", 404, 0.05, 50, 200, "127.0.0.1")
        
        summary = self.collector.get_metrics_summary()
        
        assert "api_summary" in summary
        assert "system_summary" in summary
        assert "performance_summary" in summary
        
        api_summary = summary["api_summary"]
        assert api_summary["total_requests"] == 3
        assert api_summary["success_rate"] == 2/3 * 100  # 2成功，1失败

    def test_metrics_aggregation(self):
        """测试指标聚合"""
        # 添加多个相同端点的指标
        for i in range(10):
            self.collector.record_api_metric(
                endpoint="/repeated/api",
                method="GET",
                status_code=200 if i < 8 else 500,
                response_time=0.1 + i * 0.01,
                request_size=100,
                response_size=500,
                client_ip="127.0.0.1"
            )
        
        aggregated = self.collector.get_aggregated_metrics("/repeated/api")
        
        assert aggregated["total_requests"] == 10
        assert aggregated["success_count"] == 7
        assert aggregated["error_count"] == 3
        assert aggregated["success_rate"] == 70.0
        assert aggregated["average_response_time"] > 0.1

    def test_metrics_by_status_code(self):
        """测试按状态码分组指标"""
        # 添加不同状态码的请求
        status_codes = [200, 200, 404, 500, 200, 201]
        for status_code in status_codes:
            self.collector.record_api_metric(
                endpoint="/test/api",
                method="GET",
                status_code=status_code,
                response_time=0.1,
                request_size=100,
                response_size=500,
                client_ip="127.0.0.1"
            )
        
        by_status = self.collector.get_metrics_by_status_code()
        
        assert by_status[200] == 3
        assert by_status[201] == 1
        assert by_status[404] == 1
        assert by_status[500] == 1

    def test_metrics_by_endpoint(self):
        """测试按端点分组指标"""
        endpoints = ["/api/users", "/api/posts", "/api/users", "/api/comments"]
        for endpoint in endpoints:
            self.collector.record_api_metric(
                endpoint=endpoint,
                method="GET",
                status_code=200,
                response_time=0.1,
                request_size=100,
                response_size=500,
                client_ip="127.0.0.1"
            )
        
        by_endpoint = self.collector.get_metrics_by_endpoint()
        
        assert by_endpoint["/api/users"] == 2
        assert by_endpoint["/api/posts"] == 1
        assert by_endpoint["/api/comments"] == 1

    def test_performance_metrics_calculation(self):
        """测试性能指标计算"""
        # 添加不同响应时间的请求
        response_times = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for response_time in response_times:
            self.collector.record_api_metric(
                endpoint="/performance/test",
                method="GET",
                status_code=200,
                response_time=response_time,
                request_size=100,
                response_size=500,
                client_ip="127.0.0.1"
            )
        
        performance = self.collector.get_performance_metrics()
        
        assert performance["average_response_time"] == sum(response_times) / len(response_times)
        assert performance["min_response_time"] == min(response_times)
        assert performance["max_response_time"] == max(response_times)
        assert performance["p95_response_time"] == 0.475  # 95th percentile
        assert performance["p99_response_time"] == 0.495  # 99th percentile

    def test_error_rate_calculation(self):
        """测试错误率计算"""
        # 添加成功和失败的请求
        for i in range(100):
            status_code = 200 if i < 85 else 500  # 85% 成功率
            self.collector.record_api_metric(
                endpoint="/error/test",
                method="GET",
                status_code=status_code,
                response_time=0.1,
                request_size=100,
                response_size=500,
                client_ip="127.0.0.1"
            )
        
        error_rate = self.collector.get_error_rate()
        assert error_rate == 15.0  # 15% 错误率

    def test_metrics_cleanup(self):
        """测试指标清理"""
        # 添加一些指标
        self.collector.record_api_metric("/test", "GET", 200, 0.1, 100, 500, "127.0.0.1")
        self.collector.record_system_metric(50.0, 60.0, 70.0, 1024, 2048)
        
        assert len(self.collector.api_metrics) == 1
        assert len(self.collector.system_metrics) == 1
        
        # 清理指标
        self.collector.clear_metrics()
        
        assert len(self.collector.api_metrics) == 0
        assert len(self.collector.system_metrics) == 0

    def test_metrics_export(self):
        """测试指标导出"""
        # 添加测试数据
        self.collector.record_api_metric("/export/test", "GET", 200, 0.1, 100, 500, "127.0.0.1")
        
        # 导出到临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "metrics.json"
            self.collector.export_metrics(export_path)
            
            assert export_path.exists()
            
            # 验证导出的数据
            import json
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "api_metrics" in exported_data
            assert "system_metrics" in exported_data
            assert len(exported_data["api_metrics"]) == 1

    def test_concurrent_metric_recording(self):
        """测试并发指标记录"""
        import threading
        import time
        
        def record_metrics(thread_id):
            for i in range(10):
                self.collector.record_api_metric(
                    endpoint=f"/thread/{thread_id}/api/{i}",
                    method="GET",
                    status_code=200,
                    response_time=0.1,
                    request_size=100,
                    response_size=500,
                    client_ip=f"127.0.0.{thread_id}"
                )
                time.sleep(0.001)  # 小延迟
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_metrics, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有指标都被记录
        assert len(self.collector.api_metrics) == 50  # 5 threads * 10 metrics each


if __name__ == "__main__":
    pytest.main([__file__, "-v"])