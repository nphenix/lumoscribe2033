"""
错误处理系统单元测试

测试增强的错误处理系统、重试机制和断路器功能
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from src.framework.shared.error_handler import (
    ErrorHandler,
    LLMErrorsHandler,
    PipelineErrorHandler,
    ErrorRecoveryStrategy,
    error_handler
)
from src.framework.shared.exceptions import (
    LumoscribeError,
    LLMError,
    NetworkError,
    DatabaseError,
    RateLimitError,
    ValidationError,
    PipelineError
)


class TestErrorHandler:
    """测试错误处理器"""
    
    def test_should_retry_llm_error_timeout(self):
        """测试 LLM 超时错误是否应该重试"""
        handler = ErrorHandler()
        error = LLMError("Request timeout after 30s")
        assert handler.should_retry(error) is True
    
    def test_should_retry_llm_error_auth(self):
        """测试 LLM 认证错误是否不应该重试"""
        handler = ErrorHandler()
        error = LLMError("Unauthorized: Invalid API key")
        assert handler.should_retry(error) is False
    
    def test_should_retry_network_error(self):
        """测试网络错误是否应该重试"""
        handler = ErrorHandler()
        error = NetworkError("Connection refused")
        assert handler.should_retry(error) is True
    
    def test_should_not_retry_validation_error(self):
        """测试验证错误是否不应该重试"""
        handler = ErrorHandler()
        error = ValidationError("Invalid field value")
        assert handler.should_retry(error) is False
    
    def test_circuit_breaker_open(self):
        """测试断路器开启逻辑"""
        handler = ErrorHandler(
            enable_circuit_breaker=True,
            circuit_breaker_failure_threshold=3
        )
        
        # 模拟连续失败
        for _ in range(3):
            handler.record_failure()
        
        assert handler.is_circuit_open() is True
    
    def test_circuit_breaker_recovery(self):
        """测试断路器恢复逻辑"""
        handler = ErrorHandler(
            enable_circuit_breaker=True,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=0.1
        )
        
        # 开启断路器
        handler.record_failure()
        assert handler.is_circuit_open() is True
        
        # 等待恢复时间
        import time
        time.sleep(0.2)
        
        # 检查是否应该恢复
        assert handler.is_circuit_open() is False
    
    def test_record_success_resets_failure_count(self):
        """测试成功记录重置失败计数"""
        handler = ErrorHandler()
        
        # 记录失败
        handler.record_failure()
        assert handler._failure_count == 1
        
        # 记录成功
        handler.record_success()
        assert handler._failure_count == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """测试带重试的执行成功场景"""
        handler = ErrorHandler(max_retries=3)
        
        async def mock_success_func():
            return "success"
        
        result = await handler.execute_with_retry(mock_success_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_eventually_fails(self):
        """测试带重试的执行最终失败场景"""
        handler = ErrorHandler(max_retries=2)
        
        call_count = 0
        async def mock_failing_func():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Persistent network error")
        
        with pytest.raises(NetworkError):
            await handler.execute_with_retry(mock_failing_func)
        
        assert call_count == 3  # 初始调用 + 2次重试


class TestLLMErrorsHandler:
    """测试 LLM 错误处理器"""
    
    def test_handle_timeout_error(self):
        """测试超时错误处理"""
        error = Exception("Request timeout after 30s")
        handled = LLMErrorsHandler.handle_llm_error(error, "test_context")
        
        assert isinstance(handled, LLMError)
        assert "调用超时" in handled.message
        # 移除 cause 属性的检查，因为我们已经移除了这个参数
    
    def test_handle_quota_error(self):
        """测试配额错误处理"""
        error = Exception("You have exceeded your quota")
        handled = LLMErrorsHandler.handle_llm_error(error, "test_context")
        
        assert isinstance(handled, LLMError)
        assert "配额不足" in handled.message
    
    def test_handle_auth_error(self):
        """测试认证错误处理"""
        error = Exception("Unauthorized access")
        handled = LLMErrorsHandler.handle_llm_error(error, "test_context")
        
        assert isinstance(handled, LLMError)
        assert "认证失败" in handled.message
    
    def test_handle_general_error(self):
        """测试通用错误处理"""
        error = Exception("Unknown error occurred")
        handled = LLMErrorsHandler.handle_llm_error(error, "test_context")
        
        assert isinstance(handled, LLMError)
        assert "调用失败" in handled.message


class TestPipelineErrorHandler:
    """测试管线错误处理器"""
    
    def test_handle_validation_error(self):
        """测试验证错误处理"""
        error = Exception("Invalid input format")
        handled = PipelineErrorHandler.handle_pipeline_error(
            error, stage="validation", pipeline_id="test_pipeline"
        )
        
        assert isinstance(handled, ValidationError)
        assert "validation" in handled.message.lower()
    
    def test_handle_timeout_error(self):
        """测试超时错误处理"""
        error = Exception("Operation timeout")
        handled = PipelineErrorHandler.handle_pipeline_error(
            error, stage="processing", pipeline_id="test_pipeline"
        )
        
        assert isinstance(handled, PipelineError)
        assert "timeout" in handled.message.lower()
        # 移除 details 属性的检查，因为我们已经移除了这个参数
    
    def test_handle_existing_lumoscribe_error(self):
        """测试已存在的 LumoscribeError 处理"""
        original_error = LLMError("Original error")
        handled = PipelineErrorHandler.handle_pipeline_error(
            original_error, stage="test", pipeline_id="test"
        )
        
        # 应该直接返回原错误
        assert handled is original_error


class TestErrorRecoveryStrategy:
    """测试错误恢复策略"""
    
    def test_graceful_degradation_with_fallback(self):
        """测试带备用数据的优雅降级"""
        fallback_data = {"status": "ok", "data": "fallback"}
        result = ErrorRecoveryStrategy.graceful_degradation(
            NetworkError("Service unavailable"),
            fallback_data
        )
        
        assert result == fallback_data
    
    def test_graceful_degradation_llm_error(self):
        """测试 LLM 错误的优雅降级"""
        result = ErrorRecoveryStrategy.graceful_degradation(
            LLMError("Model unavailable")
        )
        
        assert result["status"] == "degraded"
        assert "缓存数据" in result["message"]
    
    def test_graceful_degradation_database_error(self):
        """测试数据库错误的优雅降级"""
        result = ErrorRecoveryStrategy.graceful_degradation(
            DatabaseError("Connection failed")
        )
        
        assert result["status"] == "degraded"
        assert "数据库" in result["message"]
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """测试断路器恢复策略"""
        result = await ErrorRecoveryStrategy.circuit_breaker_recovery()
        assert result is True


class TestErrorHandlingDecorators:
    """测试错误处理装饰器"""
    
    def test_handle_llm_errors_decorator(self):
        """测试 LLM 错误处理装饰器"""
        from src.framework.shared.error_handler import handle_llm_errors
        
        @handle_llm_errors("test_context")
        def mock_llm_func():
            raise Exception("LLM timeout")
        
        with pytest.raises(LLMError) as exc_info:
            mock_llm_func()
        
        assert "test_context" in str(exc_info.value)
    
    def test_handle_pipeline_errors_decorator(self):
        """测试管线错误处理装饰器"""
        from src.framework.shared.error_handler import handle_pipeline_errors
        
        @handle_pipeline_errors(stage="test_stage", pipeline_id="test_id")
        def mock_pipeline_func():
            raise Exception("Pipeline failed")
        
        with pytest.raises(PipelineError) as exc_info:
            mock_pipeline_func()
        
        assert exc_info.value.details["stage"] == "test_stage"
        assert exc_info.value.details["pipeline_id"] == "test_id"
    
    @pytest.mark.asyncio
    async def test_with_error_handling_decorator(self):
        """测试通用错误处理装饰器"""
        from src.framework.shared.error_handler import with_error_handling
        
        call_count = 0
        @with_error_handling(max_retries=2, enable_circuit_breaker=False)
        async def mock_retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary error")
            return "success"
        
        result = await mock_retry_func()
        assert result == "success"
        assert call_count == 3


class TestIntegration:
    """集成测试"""
    
    def test_error_hierarchy_consistency(self):
        """测试错误层次结构一致性"""
        # 所有自定义异常都应该继承自 LumoscribeError
        error_types = [
            LLMError, NetworkError, DatabaseError, RateLimitError,
            ValidationError, PipelineError
        ]
        
        for error_type in error_types:
            error = error_type("Test message")
            assert isinstance(error, LumoscribeError)
            assert hasattr(error, 'error_code')
            assert hasattr(error, 'message')
            assert hasattr(error, 'details')
    
    def test_error_to_dict_method(self):
        """测试错误转字典方法"""
        error = LLMError(
            "Test error message",
            model="gpt-4"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "LLM_ERROR"
        assert error_dict["message"] == "Test error message"
        assert "model" in error_dict["details"]
    
    def test_error_code_constants(self):
        """测试错误代码常量"""
        from src.framework.shared.exceptions import ERROR_CODES
        
        # 验证所有错误类型都有对应的中文描述
        expected_codes = [
            "LLM_ERROR", "NETWORK_ERROR", "DATABASE_ERROR",
            "VALIDATION_ERROR", "PIPELINE_ERROR"
        ]
        
        for code in expected_codes:
            assert code in ERROR_CODES
            assert isinstance(ERROR_CODES[code], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])