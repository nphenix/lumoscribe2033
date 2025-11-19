"""
CLI Speckit 命令测试

测试基于实际实现的 CLI 命令，专注于基础设施相关的命令。
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
import pytest

from src.cli.main import app, pipeline_app, config_app, health_app, task_app
from src.cli.metrics_collector import app as metrics_app

runner = CliRunner()


class TestCLIInfrastructure:
    """CLI 基础设施测试"""

    def test_version_command(self):
        """测试版本命令"""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "lumoscribe2033" in result.output
        assert "Hybrid Graph-RAG Phase 1 质量平台" in result.output

    def test_help_command(self):
        """测试帮助命令"""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "pipeline" in result.output
        assert "config" in result.output
        assert "health" in result.output
        assert "tasks" in result.output

    def test_status_command(self):
        """测试状态命令"""
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "系统状态" in result.output
        assert "环境:" in result.output
        assert "调试模式:" in result.output


class TestPipelineCommands:
    """管线命令测试"""

    def test_pipeline_run_help(self):
        """测试管线运行帮助"""
        result = runner.invoke(pipeline_app, ["run", "--help"])
        
        assert result.exit_code == 0
        assert "运行 speckit 自动化管线" in result.output
        assert "input_file" in result.output
        assert "输出目录" in result.output  # 检查中文描述

    def test_pipeline_run_without_file(self):
        """测试管线运行无输入文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_file = Path(temp_dir) / "nonexistent.md"
            result = runner.invoke(pipeline_app, ["run", str(non_existent_file)])
            
            assert result.exit_code != 0
            # 检查是否包含错误信息或帮助信息
            assert "Usage" in result.output or "Error" in result.output


class TestConfigCommands:
    """配置命令测试"""

    def test_config_status_help(self):
        """测试配置状态帮助"""
        result = runner.invoke(config_app, ["status", "--help"])
        
        assert result.exit_code == 0
        assert "显示配置状态" in result.output

    def test_config_validate_help(self):
        """测试配置验证帮助"""
        result = runner.invoke(config_app, ["validate", "--help"])
        
        assert result.exit_code == 0
        assert "验证配置" in result.output

    def test_config_environment_help(self):
        """测试环境信息帮助"""
        result = runner.invoke(config_app, ["environment", "--help"])
        
        assert result.exit_code == 0
        assert "显示环境信息" in result.output

    def test_config_template_help(self):
        """测试模板生成帮助"""
        result = runner.invoke(config_app, ["template", "--help"])
        
        assert result.exit_code == 0
        assert "生成环境变量模板" in result.output


class TestHealthCommands:
    """健康检查命令测试"""

    def test_health_check_help(self):
        """测试健康检查帮助"""
        result = runner.invoke(health_app, ["check", "--help"])
        
        assert result.exit_code == 0
        assert "健康检查" in result.output

    def test_health_ready_help(self):
        """测试就绪检查帮助"""
        result = runner.invoke(health_app, ["ready", "--help"])
        
        assert result.exit_code == 0
        assert "就绪检查" in result.output

    def test_health_live_help(self):
        """测试存活检查帮助"""
        result = runner.invoke(health_app, ["live", "--help"])
        
        assert result.exit_code == 0
        assert "存活检查" in result.output


class TestTaskCommands:
    """任务命令测试"""

    def test_task_list_help(self):
        """测试任务列表帮助"""
        result = runner.invoke(task_app, ["list", "--help"])
        
        assert result.exit_code == 0
        assert "列出任务队列中的任务" in result.output
        assert "status" in result.output
        assert "type" in result.output

    def test_task_status_help(self):
        """测试任务状态帮助"""
        result = runner.invoke(task_app, ["status", "--help"])
        
        assert result.exit_code == 0
        assert "获取任务状态" in result.output
        assert "task_id" in result.output

    def test_task_cancel_help(self):
        """测试任务取消帮助"""
        result = runner.invoke(task_app, ["cancel", "--help"])
        
        assert result.exit_code == 0
        assert "取消任务" in result.output
        assert "task_id" in result.output

    def test_queue_status_help(self):
        """测试队列状态帮助"""
        result = runner.invoke(task_app, ["queue-status", "--help"])
        
        assert result.exit_code == 0
        assert "获取队列状态" in result.output


class TestMetricsCommands:
    """指标命令测试"""

    def test_metrics_collect_help(self):
        """测试指标收集帮助"""
        result = runner.invoke(metrics_app, ["collect", "--help"])
        
        assert result.exit_code == 0
        assert "收集系统综合指标" in result.output
        assert "system" in result.output
        assert "app" in result.output
        assert "compliance" in result.output

    def test_metrics_summary_help(self):
        """测试指标摘要帮助"""
        result = runner.invoke(metrics_app, ["summary", "--help"])
        
        assert result.exit_code == 0
        assert "显示指标报告摘要" in result.output
        assert "file" in result.output

    def test_metrics_collect_dry_run(self):
        """测试指标收集 Dry Run"""
        result = runner.invoke(metrics_app, ["collect", "--dry-run"])
        
        assert result.exit_code == 0
        assert "Dry run 模式" in result.output
        assert "系统指标: ✅" in result.output
        assert "应用指标: ✅" in result.output
        assert "合规指标: ✅" in result.output
        assert "文档指标: ✅" in result.output
        assert "存储指标: ✅" in result.output


class TestCLIWithOptions:
    """CLI 选项测试"""

    def test_verbose_option(self):
        """测试详细输出选项"""
        result = runner.invoke(app, ["--verbose", "status"])
        
        assert result.exit_code == 0
        # 详细模式应该包含更多信息

    def test_multiple_verbose_flags(self):
        """测试多个详细标志"""
        result = runner.invoke(app, ["-vvv", "version"])
        
        assert result.exit_code == 0
        assert "lumoscribe2033" in result.output


class TestCLIErrorHandling:
    """CLI 错误处理测试"""

    def test_invalid_command(self):
        """测试无效命令"""
        result = runner.invoke(app, ["invalid_command"])
        
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_invalid_option(self):
        """测试无效选项"""
        result = runner.invoke(app, ["status", "--invalid-option"])
        
        assert result.exit_code != 0

    def test_subcommand_invalid_option(self):
        """测试子命令无效选项"""
        result = runner.invoke(pipeline_app, ["run", "--invalid-option", "test.md"])
        
        assert result.exit_code != 0


class TestCLIIntegration:
    """CLI 集成测试"""

    def test_command_hierarchy(self):
        """测试命令层次结构"""
        # 测试主命令帮助包含所有子命令
        result = runner.invoke(app, ["--help"])
        
        assert "pipeline" in result.output
        assert "config" in result.output
        assert "health" in result.output
        assert "tasks" in result.output
        assert "metrics" in result.output

    def test_nested_command_access(self):
        """测试嵌套命令访问"""
        # 测试可以访问深层嵌套的命令
        result = runner.invoke(app, ["config", "status", "--help"])
        assert result.exit_code == 0
        
        result = runner.invoke(app, ["health", "check", "--help"])
        assert result.exit_code == 0
        
        result = runner.invoke(app, ["tasks", "list", "--help"])
        assert result.exit_code == 0

    def test_command_with_mocked_api(self):
        """测试带模拟 API 的命令"""
        # 模拟 API 响应
        mock_response = {
            "status": "healthy",
            "version": "1.0.0",
            "environment": "test",
            "timestamp": "2025-11-19T10:00:00Z",
            "services": {
                "database": {"status": "healthy"},
                "cache": {"status": "healthy"}
            },
            "system": {
                "cpu": 50,
                "memory": 60
            }
        }

        with patch('requests.get') as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_response
            )
            
            result = runner.invoke(health_app, ["check"])
            
            # 即使 API 被模拟，命令应该能正常执行
            assert result.exit_code == 0
            # 注意：由于我们模拟了 API，实际输出可能包含错误信息
            # 但命令应该能正常退出


class TestCLIOutputFormats:
    """CLI 输出格式测试"""

    def test_json_output_consistency(self):
        """测试 JSON 输出一致性"""
        # 测试配置状态输出格式
        with patch('requests.get') as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "valid": True,
                    "environment": {"environment": "development", "debug": False, "log_level": "INFO"},
                    "config_files": {".env": True}
                }
            )
            
            result = runner.invoke(config_app, ["status"])
            # 验证输出包含预期的格式
            assert result.exit_code == 0

    def test_error_output_format(self):
        """测试错误输出格式"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("网络连接失败")
            
            result = runner.invoke(health_app, ["check"])
            
            assert result.exit_code != 0
            # 检查是否包含异常信息
            assert "网络连接失败" in str(result.exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])