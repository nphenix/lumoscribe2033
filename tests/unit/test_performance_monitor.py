"""
性能监控器单元测试

测试性能监控器的各种功能，包括：
- 操作计时
- 性能统计
- 慢操作检测
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.framework.shared.monitoring import PerformanceMetric, SystemResourceMetric, TaskMetric, ApiMetric


class TestPerformanceMonitor:
    """测试性能监控器"""

    def setup_method(self):
        """测试前设置"""
        self.monitor = PerformanceMonitor()

    def test_start_operation_timing(self):
        """测试开始操作计时"""
        operation_id = self.monitor.start_operation("test_operation")
        
        assert operation_id is not None
        assert operation_id in self.monitor.active_operations
        assert self.monitor.active_operations[operation_id]["name"] == "test_operation"
        assert "start_time" in self.monitor.active_operations[operation_id]

    def test_end_operation_timing(self):
        """测试结束操作计时"""
        # 先开始一个操作
        operation_id = self.monitor.start_operation("timing_test")
        
        # 模拟一些处理时间
        time.sleep(0.01)
        
        # 结束操作
        result = self.monitor.end_operation(operation_id, success=True)
        
        assert result is not None
        assert result["operation"] == "timing_test"
        assert result["success"] is True
        assert result["duration"] > 0
        assert operation_id not in self.monitor.active_operations

    def test_get_performance_stats(self):
        """测试获取性能统计"""
        # 添加一些性能数据
        self.monitor.start_operation("op1")
        time.sleep(0.01)
        self.monitor.end_operation("op1", success=True)
        
        self.monitor.start_operation("op2")
        time.sleep(0.02)
        self.monitor.end_operation("op2", success=False)
        
        stats = self.monitor.get_performance_stats()
        
        assert "total_operations" in stats
        assert "successful_operations" in stats
        assert "failed_operations" in stats
        assert "average_duration" in stats
        assert "operations" in stats
        
        assert stats["total_operations"] == 2
        assert stats["successful_operations"] == 1
        assert stats["failed_operations"] == 1

    def test_slow_operation_detection(self):
        """测试慢操作检测"""
        # 添加一个慢操作
        self.monitor.start_operation("slow_op")
        time.sleep(0.1)  # 模拟慢操作
        self.monitor.end_operation("slow_op", success=True)
        
        slow_operations = self.monitor.get_slow_operations(threshold=0.05)
        
        assert len(slow_operations) == 1
        assert slow_operations[0]["operation"] == "slow_op"
        assert slow_operations[0]["duration"] > 0.05

    def test_concurrent_operations(self):
        """测试并发操作"""
        # 开始多个并发操作
        operation_ids = []
        for i in range(5):
            op_id = self.monitor.start_operation(f"concurrent_op_{i}")
            operation_ids.append(op_id)
        
        # 确保所有操作都在活跃状态
        assert len(self.monitor.active_operations) == 5
        
        # 结束所有操作
        for i, op_id in enumerate(operation_ids):
            self.monitor.end_operation(op_id, success=True)
        
        # 确保没有活跃操作
        assert len(self.monitor.active_operations) == 0
        
        stats = self.monitor.get_performance_stats()
        assert stats["total_operations"] == 5
        assert stats["successful_operations"] == 5

    def test_operation_error_handling(self):
        """测试操作错误处理"""
        # 开始一个操作
        operation_id = self.monitor.start_operation("error_test")
        
        # 结束操作时标记为失败
        result = self.monitor.end_operation(operation_id, success=False, error="Test error")
        
        assert result["success"] is False
        assert result["error"] == "Test error"
        
        stats = self.monitor.get_performance_stats()
        assert stats["failed_operations"] == 1

    def test_performance_metrics_calculation(self):
        """测试性能指标计算"""
        # 添加已知持续时间的操作
        operations = [
            ("fast_op", 0.01, True),
            ("medium_op", 0.05, True),
            ("slow_op", 0.1, False)
        ]
        
        for name, duration, success in operations:
            op_id = self.monitor.start_operation(name)
            time.sleep(duration)
            self.monitor.end_operation(op_id, success=success)
        
        stats = self.monitor.get_performance_stats()
        
        # 验证平均持续时间
        expected_avg = (0.01 + 0.05 + 0.1) / 3
        assert abs(stats["average_duration"] - expected_avg) < 0.01
        
        # 验证成功率
        expected_success_rate = 2/3 * 100
        assert abs(stats["success_rate"] - expected_success_rate) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])