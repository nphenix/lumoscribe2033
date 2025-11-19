"""
健康检查器单元测试

测试健康检查器的各种功能，包括：
- 健康检查注册
- 健康检查执行
- 超时处理
- 整体健康状态
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.framework.shared.monitoring import SystemStats, TaskStats, ApiStats


class TestHealthChecker:
    """测试健康检查器"""

    def setup_method(self):
        """测试前设置"""
        self.health_checker = HealthChecker()

    def test_add_health_check(self):
        """测试添加健康检查"""
        def check_function():
            return {"status": "healthy", "message": "All good"}
        
        self.health_checker.add_check("test_check", check_function)
        
        assert "test_check" in self.health_checker.checks
        assert self.health_checker.checks["test_check"]["function"] == check_function

    def test_remove_health_check(self):
        """测试移除健康检查"""
        def check_function():
            return {"status": "healthy", "message": "All good"}
        
        self.health_checker.add_check("test_check", check_function)
        assert "test_check" in self.health_checker.checks
        
        self.health_checker.remove_check("test_check")
        assert "test_check" not in self.health_checker.checks

    def test_run_health_checks(self):
        """测试运行健康检查"""
        # 添加一些检查
        def healthy_check():
            return {"status": "healthy", "message": "OK"}
        
        def warning_check():
            return {"status": "warning", "message": "High memory usage"}
        
        def error_check():
            return {"status": "error", "message": "Database connection failed"}
        
        self.health_checker.add_check("healthy", healthy_check)
        self.health_checker.add_check("warning", warning_check)
        self.health_checker.add_check("error", error_check)
        
        # 运行检查
        results = self.health_checker.run_checks()
        
        assert len(results) == 3
        assert results["healthy"]["status"] == "healthy"
        assert results["warning"]["status"] == "warning"
        assert results["error"]["status"] == "error"

    def test_get_overall_health(self):
        """测试获取整体健康状态"""
        # 添加检查
        self.health_checker.add_check("check1", lambda: {"status": "healthy"})
        self.health_checker.add_check("check2", lambda: {"status": "healthy"})
        
        overall = self.health_checker.get_overall_health()
        
        assert overall["status"] == "healthy"
        assert "checks" in overall
        assert "timestamp" in overall

    def test_health_check_timeout(self):
        """测试健康检查超时"""
        def slow_check():
            time.sleep(2)  # 模拟慢检查
            return {"status": "healthy"}
        
        self.health_checker.add_check("slow_check", slow_check)
        
        # 运行检查（应该超时）
        results = self.health_checker.run_checks(timeout=1.0)
        
        assert results["slow_check"]["status"] == "error"
        assert "timeout" in results["slow_check"]["message"].lower()

    def test_health_check_exception_handling(self):
        """测试健康检查异常处理"""
        def failing_check():
            raise Exception("Check failed")
        
        self.health_checker.add_check("failing_check", failing_check)
        
        results = self.health_checker.run_checks()
        
        assert results["failing_check"]["status"] == "error"
        assert "exception" in results["failing_check"]["message"].lower()

    def test_health_status_aggregation(self):
        """测试健康状态聚合"""
        # 测试不同状态组合
        test_cases = [
            ([], "healthy"),  # 无检查 = 健康
            ([("check1", "healthy")], "healthy"),  # 全部健康
            ([("check1", "healthy"), ("check2", "warning")], "warning"),  # 有警告
            ([("check1", "healthy"), ("check2", "error")], "error"),  # 有错误
            ([("check1", "warning"), ("check2", "warning")], "warning"),  # 全部警告
            ([("check1", "error"), ("check2", "error")], "error"),  # 全部错误
            ([("check1", "healthy"), ("check2", "warning"), ("check3", "error")], "error"),  # 混合状态
        ]
        
        for checks, expected_status in test_cases:
            # 清除现有检查
            self.health_checker.checks.clear()
            
            # 添加测试检查
            for check_name, status in checks:
                self.health_checker.add_check(check_name, lambda s=status: {"status": s})
            
            # 获取整体状态
            overall = self.health_checker.get_overall_health()
            assert overall["status"] == expected_status

    def test_health_check_with_metadata(self):
        """测试带元数据的健康检查"""
        def check_with_metadata():
            return {
                "status": "healthy",
                "message": "Service is running",
                "metadata": {
                    "version": "1.0.0",
                    "uptime": 3600,
                    "connections": 42
                }
            }
        
        self.health_checker.add_check("service_check", check_with_metadata)
        
        results = self.health_checker.run_checks()
        
        check_result = results["service_check"]
        assert check_result["status"] == "healthy"
        assert check_result["message"] == "Service is running"
        assert check_result["metadata"]["version"] == "1.0.0"
        assert check_result["metadata"]["uptime"] == 3600
        assert check_result["metadata"]["connections"] == 42

    def test_health_check_dependency(self):
        """测试健康检查依赖"""
        def dependency_check():
            return {"status": "healthy", "message": "Dependency is available"}
        
        def main_check():
            # 检查依赖状态
            dependency_results = self.health_checker.run_checks()
            dep_status = dependency_results.get("dependency", {}).get("status", "error")
            
            if dep_status != "healthy":
                return {"status": "error", "message": "Dependency not available"}
            
            return {"status": "healthy", "message": "Main service is running"}
        
        self.health_checker.add_check("dependency", dependency_check)
        self.health_checker.add_check("main", main_check)
        
        results = self.health_checker.run_checks()
        
        assert results["dependency"]["status"] == "healthy"
        assert results["main"]["status"] == "healthy"

    def test_health_check_caching(self):
        """测试健康检查缓存"""
        call_count = 0
        
        def cached_check():
            nonlocal call_count
            call_count += 1
            return {"status": "healthy", "message": f"Call {call_count}"}
        
        self.health_checker.add_check("cached_check", cached_check)
        
        # 第一次运行
        results1 = self.health_checker.run_checks()
        assert results1["cached_check"]["message"] == "Call 1"
        assert call_count == 1
        
        # 第二次运行（应该使用缓存）
        results2 = self.health_checker.run_checks(use_cache=True, cache_ttl=1.0)
        assert results2["cached_check"]["message"] == "Call 1"
        assert call_count == 1  # 没有增加调用
        
        # 等待缓存过期
        time.sleep(1.1)
        results3 = self.health_checker.run_checks(use_cache=True, cache_ttl=1.0)
        assert results3["cached_check"]["message"] == "Call 2"
        assert call_count == 2

    def test_health_check_custom_timeout(self):
        """测试自定义超时设置"""
        def slow_check():
            time.sleep(0.5)
            return {"status": "healthy"}
        
        def fast_check():
            return {"status": "healthy"}
        
        # 添加不同超时设置的检查
        self.health_checker.add_check("slow_check", slow_check, timeout=1.0)
        self.health_checker.add_check("fast_check", fast_check, timeout=0.1)
        
        results = self.health_checker.run_checks()
        
        # 两个检查都应该成功（因为全局超时足够长）
        assert results["slow_check"]["status"] == "healthy"
        assert results["fast_check"]["status"] == "healthy"

    def test_health_check_metrics(self):
        """测试健康检查指标"""
        def healthy_check():
            return {"status": "healthy"}
        
        def error_check():
            return {"status": "error", "message": "Failed"}
        
        self.health_checker.add_check("healthy", healthy_check)
        self.health_checker.add_check("error", error_check)
        
        # 运行多次检查
        for _ in range(5):
            self.health_checker.run_checks()
        
        metrics = self.health_checker.get_health_metrics()
        
        assert metrics["total_checks"] == 2
        assert metrics["healthy_checks"] == 1
        assert metrics["unhealthy_checks"] == 1
        assert metrics["total_runs"] == 5
        assert metrics["success_rate"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
