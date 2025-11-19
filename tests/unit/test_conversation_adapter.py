"""
对话适配器单元测试

测试对话适配器的各种功能，包括：
- Cursor 日志解析
- RooCode 日志解析
- 格式识别
- 导出功能
- 错误处理
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.framework.adapters.conversation_adapter import (
    ConversationExportFormat,
    ConversationParseError,
    ConversationParseResult,
    CursorConversationAdapter,
    IDELogFormat,
    RooCodeConversationAdapter,
)


class TestCursorConversationAdapter:
    """Cursor 对话适配器单元测试"""

    @pytest.fixture
    def cursor_adapter(self):
        """创建 Cursor 适配器实例"""
        return CursorConversationAdapter()

    def test_basic_log_parsing(self, cursor_adapter):
        """测试基础日志解析"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there! How can I help you?
[2024-01-01 10:00:10] User: I need help with Python code
"""

        result = cursor_adapter.parse_conversation(log_content)

        assert result.success is True
        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "Hi there! How can I help you?"
        assert result.messages[2].role == "user"
        assert result.messages[2].content == "I need help with Python code"

    def test_json_format_parsing(self, cursor_adapter):
        """测试 JSON 格式解析"""
        json_log = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "role": "user",
                "content": "Hello",
                "type": "message"
            },
            {
                "timestamp": "2024-01-01T10:00:05",
                "role": "assistant",
                "content": "Hi there!",
                "type": "message"
            }
        ]

        result = cursor_adapter.parse_conversation(str(json_log))

        assert result.success is True
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_code_edit_events(self, cursor_adapter):
        """测试代码编辑事件解析"""
        log_content = """
[2024-01-01 10:00:00] Code Edit: Inserted text at line 5
[2024-01-01 10:00:01] User: Please explain this code
[2024-01-01 10:00:05] Assistant: This code does X
[2024-01-01 10:00:10] File Operation: Opened file.py
"""

        result = cursor_adapter.parse_conversation(log_content)

        assert result.success is True
        # 应该只包含对话消息，不包含代码编辑和文件操作
        assert len(result.messages) == 2
        assert all(msg.role in ["user", "assistant"] for msg in result.messages)

    def test_malformed_log_handling(self, cursor_adapter):
        """测试错误日志处理"""
        malformed_log = """
Invalid log format
Some random text
Another invalid line
"""

        result = cursor_adapter.parse_conversation(malformed_log)

        assert result.success is False
        assert isinstance(result.error, ConversationParseError)
        assert "无法解析任何对话消息" in str(result.error)

    def test_empty_log_handling(self, cursor_adapter):
        """测试空日志处理"""
        result = cursor_adapter.parse_conversation("")

        assert result.success is False
        assert isinstance(result.error, ConversationParseError)

    def test_format_detection(self, cursor_adapter):
        """测试格式检测"""
        # 测试结构化格式
        structured_log = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi
"""
        format_type = cursor_adapter.detect_format(structured_log)
        assert format_type == IDELogFormat.STRUCTURED

        # 测试 JSON 格式
        json_log = '[{"timestamp": "2024-01-01T10:00:00", "role": "user"}]'
        format_type = cursor_adapter.detect_format(json_log)
        assert format_type == IDELogFormat.JSON

        # 测试简单文本格式
        simple_log = "User: Hello\nAssistant: Hi"
        format_type = cursor_adapter.detect_format(simple_log)
        assert format_type == IDELogFormat.SIMPLE

    def test_export_json(self, cursor_adapter):
        """测试 JSON 导出"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there!
"""

        result = cursor_adapter.parse_conversation(log_content)
        exported = cursor_adapter.export_conversation(result, ConversationExportFormat.JSON)

        assert isinstance(exported, str)
        assert "messages" in exported
        assert "user" in exported
        assert "assistant" in exported

    def test_export_csv(self, cursor_adapter):
        """测试 CSV 导出"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there!
"""

        result = cursor_adapter.parse_conversation(log_content)
        exported = cursor_adapter.export_conversation(result, ConversationExportFormat.CSV)

        assert isinstance(exported, str)
        assert "role,content,timestamp" in exported
        assert "user,Hello," in exported

    def test_export_text(self, cursor_adapter):
        """测试纯文本导出"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there!
"""

        result = cursor_adapter.parse_conversation(log_content)
        exported = cursor_adapter.export_conversation(result, ConversationExportFormat.TEXT)

        assert isinstance(exported, str)
        assert "User:" in exported
        assert "Assistant:" in exported

    def test_time_window_merge(self, cursor_adapter):
        """测试时间窗口合并"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:01] User: How are you?
[2024-01-01 10:00:10] Assistant: I'm good
"""

        result = cursor_adapter.parse_conversation(log_content)

        # 在时间窗口内，应该合并为一条消息
        assert len(result.messages) == 2  # user(合并) + assistant

    def test_session_segmentation(self, cursor_adapter):
        """测试会话分割"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi
Session: New Session Started
[2024-01-01 14:00:00] User: Back again
[2024-01-01 14:00:05] Assistant: Welcome back
"""

        result = cursor_adapter.parse_conversation(log_content)

        # 应该识别会话分割
        assert len(result.messages) == 4
        # 可以通过消息内容或时间戳来验证会话分割

    def test_consecutive_messages_filtering(self, cursor_adapter):
        """测试连续消息过滤"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:01] User: Hello again
[2024-01-01 10:00:02] User: And again
[2024-01-01 10:00:05] Assistant: Hi there!
"""

        result = cursor_adapter.parse_conversation(log_content)

        # 连续的用户消息应该被合并
        assert len(result.messages) == 2  # user(合并) + assistant


class TestRooCodeConversationAdapter:
    """RooCode 对话适配器单元测试"""

    @pytest.fixture
    def roocode_adapter(self):
        """创建 RooCode 适配器实例"""
        return RooCodeConversationAdapter()

    def test_chinese_role_parsing(self, roocode_adapter):
        """测试中文角色解析"""
        log_content = """
[2024-01-01 10:00:00] 用户: 你好
[2024-01-01 10:00:05] 助手: 你好！有什么可以帮助你？
[2024-01-01 10:00:10] 用户: 我需要 Python 代码帮助
"""

        result = roocode_adapter.parse_conversation(log_content)

        assert result.success is True
        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "你好"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "你好！有什么可以帮助你？"

    def test_json_format_with_chinese(self, roocode_adapter):
        """测试包含中文的 JSON 格式"""
        json_log = [
            {
                "timestamp": "2024-01-01T10:00:00",
                "role": "用户",
                "content": "你好",
                "type": "message"
            },
            {
                "timestamp": "2024-01-01T10:00:05",
                "role": "助手",
                "content": "你好！",
                "type": "message"
            }
        ]

        result = roocode_adapter.parse_conversation(str(json_log))

        assert result.success is True
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_mixed_chinese_english_roles(self, roocode_adapter):
        """测试中英文混合角色"""
        log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] 助手: 你好！
[2024-01-01 10:00:10] 用户: Hi
[2024-01-01 10:00:15] Assistant: Hello
"""

        result = roocode_adapter.parse_conversation(log_content)

        assert result.success is True
        assert len(result.messages) == 4
        # 所有消息都应该被正确解析
        expected_roles = ["user", "assistant", "user", "assistant"]
        for i, expected_role in enumerate(expected_roles):
            assert result.messages[i].role == expected_role

    def test_roocode_specific_events(self, roocode_adapter):
        """测试 RooCode 特有事件"""
        log_content = """
[2024-01-01 10:00:00] 用户: 请帮我写代码
[2024-01-01 10:00:05] 代码生成: 生成了 Python 代码
[2024-01-01 10:00:06] 助手: 这是你要的代码
[2024-01-01 10:00:10] 用户: 谢谢
"""

        result = roocode_adapter.parse_conversation(log_content)

        assert result.success is True
        # 应该只包含对话消息
        assert len(result.messages) == 3
        assert all(msg.role in ["user", "assistant"] for msg in result.messages)

    def test_unicode_handling(self, roocode_adapter):
        """测试 Unicode 字符处理"""
        log_content = """
[2024-01-01 10:00:00] 用户: 你好 🌟
[2024-01-01 10:00:05] 助手: 你好！欢迎使用 RooCode ✨
"""

        result = roocode_adapter.parse_conversation(log_content)

        assert result.success is True
        assert len(result.messages) == 2
        # Unicode 字符应该被正确保留
        assert "🌟" in result.messages[0].content
        assert "✨" in result.messages[1].content

    def test_chinese_format_detection(self, roocode_adapter):
        """测试中文格式检测"""
        # 测试中文结构化格式
        chinese_log = """
[2024-01-01 10:00:00] 用户: 你好
[2024-01-01 10:00:05] 助手: 你好！
"""
        format_type = roocode_adapter.detect_format(chinese_log)
        assert format_type == IDELogFormat.STRUCTURED

        # 测试英文格式（应该也能处理）
        english_log = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi
"""
        format_type = roocode_adapter.detect_format(english_log)
        assert format_type == IDELogFormat.STRUCTURED

    def test_chinese_time_window_merge(self, roocode_adapter):
        """测试中文时间窗口合并"""
        log_content = """
[2024-01-01 10:00:00] 用户: 你好
[2024-01-01 10:00:01] 用户: 在吗
[2024-01-01 10:00:10] 助手: 在的
"""

        result = roocode_adapter.parse_conversation(log_content)

        # 在时间窗口内，应该合并为一条消息
        assert len(result.messages) == 2  # user(合并) + assistant

    def test_export_with_chinese_content(self, roocode_adapter):
        """测试包含中文内容的导出"""
        log_content = """
[2024-01-01 10:00:00] 用户: 你好
[2024-01-01 10:00:05] 助手: 你好！欢迎使用
"""

        result = roocode_adapter.parse_conversation(log_content)

        # 测试 JSON 导出
        json_export = roocode_adapter.export_conversation(result, ConversationExportFormat.JSON)
        assert isinstance(json_export, str)
        assert "你好" in json_export

        # 测试 CSV 导出
        csv_export = roocode_adapter.export_conversation(result, ConversationExportFormat.CSV)
        assert isinstance(csv_export, str)
        assert "你好" in csv_export

        # 测试文本导出
        text_export = roocode_adapter.export_conversation(result, ConversationExportFormat.TEXT)
        assert isinstance(text_export, str)
        assert "你好" in text_export


class TestConversationParseResult:
    """对话解析结果测试"""

    def test_result_creation(self):
        """测试结果创建"""
        from src.framework.adapters.conversation_adapter import ConversationMessage

        messages = [
            ConversationMessage(role="user", content="Hello", timestamp="2024-01-01T10:00:00")
        ]

        result = ConversationParseResult(
            success=True,
            messages=messages,
            format_detected="structured",
            processing_time=0.1
        )

        assert result.success is True
        assert len(result.messages) == 1
        assert result.messages[0].content == "Hello"
        assert result.format_detected == "structured"
        assert result.processing_time == 0.1

    def test_error_result_creation(self):
        """测试错误结果创建"""
        error = ConversationParseError("测试错误", "invalid_format")

        result = ConversationParseResult(
            success=False,
            messages=[],
            format_detected="unknown",
            error=error
        )

        assert result.success is False
        assert result.error is error
        assert str(result.error) == "测试错误"


class TestConversationParseError:
    """对话解析错误测试"""

    def test_error_creation(self):
        """测试错误创建"""
        error = ConversationParseError("解析失败", "invalid_format", details={"line": 10})

        assert error.message == "解析失败"
        assert error.error_type == "invalid_format"
        assert error.details == {"line": 10}

    def test_error_str_representation(self):
        """测试错误字符串表示"""
        error = ConversationParseError("测试错误", "format_error")
        error_str = str(error)

        assert "测试错误" in error_str
        assert "format_error" in error_str


class TestIDELogFormat:
    """IDE 日志格式枚举测试"""

    def test_format_values(self):
        """测试格式值"""
        assert IDELogFormat.STRUCTURED.value == "structured"
        assert IDELogFormat.JSON.value == "json"
        assert IDELogFormat.SIMPLE.value == "simple"

    def test_format_comparison(self):
        """测试格式比较"""
        assert IDELogFormat.STRUCTURED == IDELogFormat.STRUCTURED
        assert IDELogFormat.STRUCTURED != IDELogFormat.JSON


if __name__ == "__main__":
    # 运行单元测试
    pytest.main([__file__, "-v"])
