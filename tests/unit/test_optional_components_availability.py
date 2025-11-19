"""
可选组件可用性测试

测试 OpenTelemetry 可选组件的安装和可用性状态
"""

import sys
import unittest
from unittest.mock import patch


class TestOptionalComponentsAvailability(unittest.TestCase):
    """可选组件可用性测试类"""

    def test_opentelemetry_core_available(self):
        """测试 OpenTelemetry 核心功能可用性"""
        try:
            from src.framework.shared.telemetry import OTEL_AVAILABLE
            self.assertTrue(OTEL_AVAILABLE, "OpenTelemetry 核心功能应该可用")
        except ImportError as e:
            self.fail(f"无法导入 telemetry 模块: {e}")

    def test_instrumentation_packages_import(self):
        """测试仪器化包导入"""
        # 测试 Requests 仪器化
        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            self.assertIsNotNone(RequestsInstrumentor)
        except ImportError:
            self.skipTest("Requests 仪器化包不可用")

        # 测试 Urllib3 仪器化
        try:
            from opentelemetry.instrumentation.urllib3 import Urllib3Instrumentor
            self.assertIsNotNone(Urllib3Instrumentor)
        except ImportError:
            self.skipTest("Urllib3 仪器化包不可用")

        # 测试 HTTPX 仪器化
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            self.assertIsNotNone(HTTPXClientInstrumentor)
        except ImportError:
            self.skipTest("HTTPX 仪器化包不可用")

        # 测试 Redis 仪器化
        try:
            from opentelemetry.instrumentation.redis import RedisInstrumentor
            self.assertIsNotNone(RedisInstrumentor)
        except ImportError:
            self.skipTest("Redis 仪器化包不可用")

        # 测试 Psycopg2 仪器化
        try:
            from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
            self.assertIsNotNone(Psycopg2Instrumentor)
        except ImportError:
            self.skipTest("Psycopg2 仪器化包不可用")

        # 测试 SQLite3 仪器化
        try:
            from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
            self.assertIsNotNone(SQLite3Instrumentor)
        except ImportError:
            self.skipTest("SQLite3 仪器化包不可用")

        # 测试 FastAPI 仪器化
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            self.assertIsNotNone(FastAPIInstrumentor)
        except ImportError:
            self.skipTest("FastAPI 仪器化包不可用")

        # 测试 Arq 仪器化
        try:
            from opentelemetry.instrumentation.arq import ArqInstrumentor
            self.assertIsNotNone(ArqInstrumentor)
        except ImportError:
            self.skipTest("Arq 仪器化包不可用")

        # 测试 Logging 仪器化
        try:
            from opentelemetry.instrumentation.logging import LoggingInstrumentor
            self.assertIsNotNone(LoggingInstrumentor)
        except ImportError:
            self.skipTest("Logging 仪器化包不可用")

    def test_system_monitoring_packages(self):
        """测试系统监控包可用性"""
        # 测试 psutil
        try:
            import psutil
            self.assertIsNotNone(psutil)
            # 测试基本功能
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.assertIsInstance(cpu_percent, (int, float))
        except ImportError:
            self.skipTest("psutil 包不可用")

    def test_telemetry_functions_available(self):
        """测试 telemetry 模块中的功能函数可用性"""
        from src.framework.shared.telemetry import (
            get_meter,
            get_request_stats,
            get_system_health_status,
            get_tracer,
            setup_telemetry,
        )

        # 测试 get_tracer 函数
        self.assertTrue(callable(get_tracer))

        # 测试 get_meter 函数
        self.assertTrue(callable(get_meter))

        # 测试 setup_telemetry 函数
        self.assertTrue(callable(setup_telemetry))

        # 测试系统健康状态函数
        self.assertTrue(callable(get_system_health_status))

        # 测试请求统计函数
        self.assertTrue(callable(get_request_stats))

    def test_telemetry_initialization(self):
        """测试 telemetry 初始化"""
        from src.framework.shared.telemetry import OTEL_AVAILABLE, setup_telemetry

        if OTEL_AVAILABLE:
            # 测试初始化不抛出异常
            try:
                setup_telemetry(service_name="test-service", service_version="1.0.0")
            except Exception as e:
                self.fail(f"Telemetry 初始化失败: {e}")
        else:
            self.skipTest("OpenTelemetry 不可用")

    def test_telemetry_functions_with_mock_data(self):
        """测试 telemetry 函数在模拟数据下的行为"""
        from src.framework.shared.telemetry import OTEL_AVAILABLE, get_meter, get_tracer

        if OTEL_AVAILABLE:
            # 测试获取 tracer
            tracer = get_tracer("test-tracer")
            self.assertIsNotNone(tracer)

            # 测试获取 meter
            meter = get_meter("test-meter")
            self.assertIsNotNone(meter)

            # 测试创建自定义 span
            with tracer.start_as_current_span("test-span") as span:
                span.set_attribute("test", "value")
                self.assertIsNotNone(span)
        else:
            self.skipTest("OpenTelemetry 不可用")

    def test_auto_instrumentation_components(self):
        """测试自动仪器化组件的可用性"""
        # 这些测试验证 telemetry 模块中的导入是否正确
        from src.framework.shared.telemetry import OTEL_AVAILABLE

        if OTEL_AVAILABLE:
            # 测试 telemetry 模块是否能正确处理各种仪器化包的导入
            # 即使某些包不可用，系统也应该能正常工作

            # 模拟不同的模块加载状态
            with patch('sys.modules', {}):
                # 清空 sys.modules 来测试自动仪器化的容错性
                from src.framework.shared.telemetry import setup_telemetry
                try:
                    setup_telemetry()
                except Exception as e:
                    self.fail(f"自动仪器化容错失败: {e}")


if __name__ == '__main__':
    unittest.main()
