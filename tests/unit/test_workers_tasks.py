"""
Workers任务单元测试

测试 Arq 任务处理器的各种功能和边界情况，基于实际实现的任务框架。
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

from src.workers.tasks.compliance import run_static_check
from src.workers.tasks.knowledge import import_conversations, generate_ide_package
from src.workers.tasks.metrics import collect_comprehensive_metrics, collect_real_time_metrics
from src.workers.tasks.pipeline import run_full_pipeline


class TestComplianceTasks:
    """合规性任务测试"""

    @pytest.mark.asyncio
    async def test_run_static_check_success(self):
        """测试静态检查成功"""
        ctx = {"job_id": "test_job_123"}
        request_data = {
            "document_id": "doc_123",
            "check_type": "ruff_mypy",
            "config": {"strict": True}
        }

        result = await run_static_check(ctx, request_data)

        assert result["success"]
        assert "report_file" in result
        assert "violations" in result
        assert "score" in result
        assert result["message"] == "静态检查完成"
        assert result["score"] == 100

    @pytest.mark.asyncio
    async def test_run_static_check_failure(self):
        """测试静态检查失败"""
        ctx = {"job_id": "test_job_456"}
        request_data = {
            "document_id": "invalid_doc",
            "check_type": "invalid_type"
        }

        # 模拟处理异常
        with patch('src.workers.tasks.compliance.asyncio.sleep',
                   side_effect=Exception("模拟异常")):
            result = await run_static_check(ctx, request_data)

            assert not result["success"]
            assert "error" in result
            assert result["message"] == "静态检查失败"


class TestKnowledgeTasks:
    """知识处理任务测试"""

    @pytest.mark.asyncio
    async def test_import_conversations_success(self):
        """测试对话导入成功"""
        ctx = {"job_id": "test_job_knowledge_123"}
        request_data = {
            "source": "cursor",
            "path": "logs/cursor",
            "format": "json"
        }

        result = await import_conversations(ctx, request_data)

        assert result["success"]
        assert "imported_count" in result
        assert "source" in result
        assert result["message"] == "对话导入完成"
        assert result["imported_count"] == 150
        assert result["source"] == "cursor"

    @pytest.mark.asyncio
    async def test_import_conversations_with_different_sources(self):
        """测试不同来源的对话导入"""
        ctx = {"job_id": "test_job_knowledge_456"}

        sources = ["cursor", "roocode", "generic"]

        for source in sources:
            request_data = {
                "source": source,
                "path": f"logs/{source}",
                "format": "json"
            }

            result = await import_conversations(ctx, request_data)

            assert result["success"]
            assert result["source"] == source

    @pytest.mark.asyncio
    async def test_generate_ide_package_success(self):
        """测试IDE包生成成功"""
        ctx = {"job_id": "test_job_ide_123"}
        request_data = {
            "ide": "cursor",
            "template": "commands",
            "output_path": "ide-packages/cursor"
        }

        result = await generate_ide_package(ctx, request_data)

        assert result["success"]
        assert "ide_name" in result
        assert "package_path" in result
        assert result["message"] == "IDE 适配包生成完成"
        assert result["ide_name"] == "cursor"

    @pytest.mark.asyncio
    async def test_generate_ide_package_failure(self):
        """测试IDE包生成失败"""
        ctx = {"job_id": "test_job_ide_456"}
        request_data = {
            "ide": "invalid_ide",
            "template": "invalid_template"
        }

        with patch('src.workers.tasks.knowledge.asyncio.sleep',
                   side_effect=Exception("模板不存在")):
            result = await generate_ide_package(ctx, request_data)

            assert not result["success"]
            assert "error" in result
            assert result["message"] == "IDE 适配包生成失败"


class TestMetricsTasks:
    """指标收集任务测试"""

    @pytest.mark.asyncio
    async def test_collect_comprehensive_metrics_success(self):
        """测试综合指标收集成功"""
        ctx = {"job_id": "test_job_metrics_123"}
        request_data = {
            "metric_types": ["system", "application", "cache"],
            "time_range": "1h",
            "include_health": True
        }

        # 模拟依赖服务
        with patch('src.workers.tasks.metrics.get_enhanced_metrics_collector') as mock_get_collector, \
             patch('src.workers.tasks.metrics.get_cache_manager') as mock_get_cache:
            
            mock_cache_health = {"status": "healthy", "hit_rate": 85}
            mock_cache_manager = AsyncMock()
            mock_cache_manager.health_check.return_value = mock_cache_health
            mock_get_cache.return_value = mock_cache_manager
            
            # 创建模拟的指标收集器
            mock_metrics_collector = AsyncMock()
            
            # 模拟 collect_comprehensive_metrics 方法
            mock_comprehensive_metrics = {
                "system": {"cpu_usage": 50, "memory_usage": 60},
                "application": {"active_tasks": 5, "completed_tasks": 100},
                "cache": {"hit_rate": 85}
            }
            mock_metrics_collector.collect_comprehensive_metrics.return_value = mock_comprehensive_metrics
            
            mock_get_collector.return_value = mock_metrics_collector

            result = await collect_comprehensive_metrics(ctx, request_data)

            assert result["success"]
            assert "metrics" in result
            assert "comprehensive" in result["metrics"]
            assert "cache_health" in result["metrics"]
            assert "system_resources" in result["metrics"]
            assert "message" in result

    @pytest.mark.asyncio
    async def test_collect_real_time_metrics_success(self):
        """测试实时指标收集成功"""
        ctx = {"job_id": "test_job_realtime_123"}
        request_data = {
            "interval": "5s",
            "metrics": ["cpu", "memory", "cache"]
        }

        result = await collect_real_time_metrics(ctx, request_data)

        assert result["success"]
        assert "metrics" in result
        assert "timestamp" in result["metrics"]
        assert "system_load" in result["metrics"]
        assert "cache_performance" in result["metrics"]
        assert "message" in result

    @pytest.mark.asyncio
    async def test_collect_metrics_failure(self):
        """测试指标收集失败"""
        ctx = {"job_id": "test_job_metrics_failure"}
        request_data = {
            "metric_types": ["invalid_type"],
            "time_range": "invalid_range"
        }

        with patch('src.workers.tasks.metrics._collect_system_resources',
                   side_effect=Exception("系统资源访问失败")):
            result = await collect_comprehensive_metrics(ctx, request_data)

            assert not result["success"]
            assert "error" in result
            assert result["message"] == "指标收集失败"


class TestPipelineTasks:
    """管道任务测试"""

    @pytest.mark.asyncio
    async def test_run_full_pipeline_success(self):
        """测试完整管线执行成功"""
        ctx = {"job_id": "test_job_pipeline_123"}
        request_data = {
            "document_path": "samples/test.md",
            "pipeline_type": "speckit_full",
            "output_dir": "specs/test_output"
        }

        result = await run_full_pipeline(ctx, request_data)

        assert result["success"]
        assert "artifacts" in result
        assert "execution_time" in result
        assert result["message"] == "完整管线执行成功"
        assert len(result["artifacts"]) == 5
        assert all("specs/" in artifact for artifact in result["artifacts"])

    @pytest.mark.asyncio
    async def test_run_full_pipeline_failure(self):
        """测试完整管线执行失败"""
        ctx = {"job_id": "test_job_pipeline_failure"}
        request_data = {
            "document_path": "nonexistent.md",
            "pipeline_type": "invalid_type"
        }

        with patch('src.workers.tasks.pipeline.asyncio.sleep',
                   side_effect=Exception("文档不存在")):
            result = await run_full_pipeline(ctx, request_data)

            assert not result["success"]
            assert "error" in result
            assert result["message"] == "完整管线执行失败"


class TestTaskIntegration:
    """任务集成测试"""

    @pytest.mark.asyncio
    async def test_task_basic_functionality(self):
        """测试任务基本功能"""
        # 测试静态检查
        ctx = {"job_id": "integration_test"}
        static_result = await run_static_check(ctx, {"document_id": "test"})
        assert static_result["success"]

        # 测试对话导入
        import_result = await import_conversations(ctx, {"source": "cursor"})
        assert import_result["success"]

        # 测试指标收集
        metrics_result = await collect_comprehensive_metrics(ctx, {})
        assert metrics_result["success"]

        # 测试管线执行
        pipeline_result = await run_full_pipeline(ctx, {"document_path": "test.md"})
        assert pipeline_result["success"]

    @pytest.mark.asyncio
    async def test_task_error_handling(self):
        """测试任务错误处理"""
        ctx = {"job_id": "error_test"}

        # 测试空请求数据
        error_cases = [
            {},  # 空数据
            {"invalid_field": "invalid_value"},  # 无效数据
        ]

        for error_data in error_cases:
            # 测试各种任务的错误处理
            static_result = await run_static_check(ctx, error_data)
            # 应该仍然返回成功，因为有默认处理
            
            import_result = await import_conversations(ctx, error_data)
            # 应该仍然返回成功，因为有默认处理

    @pytest.mark.asyncio
    async def test_task_execution_timing(self):
        """测试任务执行时间"""
        ctx = {"job_id": "timing_test"}

        # 测试各个任务的执行时间
        start_time = asyncio.get_event_loop().time()
        
        # 静态检查（模拟2秒）
        static_result = await run_static_check(ctx, {"document_id": "test"})
        static_duration = static_result.get("execution_time", 0)
        
        # 对话导入（模拟3秒）
        import_result = await import_conversations(ctx, {"source": "cursor"})
        import_duration = import_result.get("execution_time", 0)
        
        # 管线执行（模拟5秒）
        pipeline_result = await run_full_pipeline(ctx, {"document_path": "test.md"})
        pipeline_duration = pipeline_result.get("execution_time", 0)

        # 验证执行时间合理性（允许一定误差）
        assert 1.5 <= static_duration <= 2.5  # 静态检查约2秒
        assert 2.5 <= import_duration <= 3.5  # 对话导入约3秒
        assert 12.0 <= pipeline_duration <= 13.0  # 管线执行约12.5秒


class TestTaskContext:
    """任务上下文测试"""

    @pytest.mark.asyncio
    async def test_task_context_preservation(self):
        """测试任务上下文保持"""
        ctx = {
            "job_id": "context_test_123",
            "user_id": "test_user",
            "timestamp": "2025-11-19T10:00:00Z",
            "metadata": {"source": "test"}
        }
        
        request_data = {"document_id": "test_doc"}

        # 测试上下文在任务中的传递
        result = await run_static_check(ctx, request_data)
        
        # 验证上下文信息被正确使用
        assert result["success"]
        # 上下文信息应该在日志或内部处理中被使用

    @pytest.mark.asyncio
    async def test_task_with_different_context_scenarios(self):
        """测试不同上下文场景"""
        scenarios = [
            {"job_id": "dev_test", "environment": "development"},
            {"job_id": "prod_test", "environment": "production"},
            {"job_id": "debug_test", "debug": True}
        ]

        for scenario_ctx in scenarios:
            request_data = {"document_id": f"test_{scenario_ctx['job_id']}"}
            
            result = await run_static_check(scenario_ctx, request_data)
            assert result["success"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
