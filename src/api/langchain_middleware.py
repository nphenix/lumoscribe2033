"""
LangChain 集成中间件

基于 LangChain 1.0 Runnable/Callback 体系的本地化追踪方案：
- 自定义 AsyncCallbackHandler 捕获 LLM/Runnable 事件
- 本地 JSONL 日志（无需 LangSmith 或任何云服务）
- 请求级 RunnableConfig 注入，方便在 API/任务中统一追踪
- LLM 输入输出、提示词、性能指标完整归档
"""

from __future__ import annotations

import asyncio
import datetime
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from fastapi import Request, Response
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from src.framework.orchestrators.langchain_executor import (
    LangChainExecutor,
    get_executor_with_config,
)


class LangChainMiddleware:
    """
    LangChain 追踪中间件

    为所有 LLM 调用提供本地追踪和监控功能
    """

    def __init__(
        self,
        project_name: str = "lumoscribe2033",
        tracing_enabled: bool = True,
        capture_io: bool = True,
        capture_metadata: bool = True,
        log_file: str = "logs/llm_traces.log",
        tracker: LLMCallTracker | None = None,
        prompt_logger: PromptLogger | None = None,
        default_tags: list[str] | None = None,
    ):
        self.project_name = project_name
        self.tracing_enabled = tracing_enabled
        self.capture_io = capture_io
        self.capture_metadata = capture_metadata
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.default_tags = default_tags or ["api-request"]
        self.tracker = tracker
        self.prompt_logger = prompt_logger

        if tracing_enabled:
            logger.info("✅ LangChain 本地追踪已启用")

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        中间件调用逻辑

        为请求添加 LangChain 追踪上下文
        """
        if not self.tracing_enabled:
            return await call_next(request)

        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        callback_handler = LocalTraceCallbackHandler(
            trace_id=trace_id,
            event_log=self.log_file,
            tracker=self.tracker,
            prompt_logger=self.prompt_logger,
            project_name=self.project_name,
        )

        runnable_config = RunnableConfig(
            callbacks=[callback_handler],
            tags=self.default_tags + [request.url.path],
            metadata={
                "trace_id": trace_id,
                "project": self.project_name,
                "path": request.url.path,
                "method": request.method,
            },
        )

        request.state.langchain_callback_handler = callback_handler
        request.state.langchain_runnable_config = runnable_config
        request.state.langchain_executor = _resolve_request_executor(runnable_config)

        start_time = time.time()

        try:
            response = await call_next(request)
            if self.capture_io:
                process_time = time.time() - start_time
                logger.debug(
                    f"🔍 [Trace-{trace_id}] 请求完成: {request.method} {request.url.path} ({process_time:.4f}s)"
                )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"❌ [Trace-{trace_id}] 请求失败: {request.method} {request.url.path} - {str(e)} ({process_time:.4f}s)"
            )
            raise


class LLMCallTracker:
    """
    LLM 调用追踪器

    专门用于本地追踪 LLM 调用的详细信息
    """

    def __init__(self, log_file: str = "logs/llm_calls.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def track_llm_call(
        self,
        model_name: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        trace_id: str | None = None
    ) -> None:
        """
        追踪 LLM 调用

        Args:
            model_name: 模型名称
            prompt: 输入提示词
            response: 模型响应
            metadata: 额外元数据
            tags: 标签列表
            trace_id: 追踪 ID
        """
        try:
            # 创建追踪记录
            trace_data = {
                "timestamp": time.time(),
                "trace_id": trace_id or str(uuid.uuid4()),
                "model_name": model_name,
                "prompt": prompt,
                "response": response,
                "metadata": metadata or {},
                "tags": tags or [],
                "prompt_length": len(prompt),
                "response_length": len(response) if response else 0
            }

            # 保存到本地日志文件
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")

            logger.debug(f"📊 LLM 调用已追踪: {model_name}")

        except Exception as e:
            logger.error(f"❌ LLM 调用追踪失败: {e}")

    def track_chain_execution(
        self,
        chain_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        steps: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None
    ) -> None:
        """
        追踪链执行

        Args:
            chain_name: 链名称
            inputs: 输入数据
            outputs: 输出数据
            steps: 执行步骤
            metadata: 元数据
            trace_id: 追踪 ID
        """
        try:
            # 创建链追踪记录
            chain_data = {
                "timestamp": time.time(),
                "trace_id": trace_id or str(uuid.uuid4()),
                "chain_name": chain_name,
                "run_type": "chain",
                "inputs": inputs,
                "outputs": outputs,
                "metadata": metadata or {},
                "steps": steps,
                "step_count": len(steps)
            }

            # 保存到本地日志文件
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(chain_data, ensure_ascii=False) + "\n")

            logger.debug(f"🔗 链执行已追踪: {chain_name}")

        except Exception as e:
            logger.error(f"❌ 链执行追踪失败: {e}")


class PromptLogger:
    """
    提示词日志记录器

    专门用于记录和分析提示词使用情况
    """

    def __init__(self, log_file: str = "logs/prompts.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_prompt(
        self,
        prompt_type: str,
        prompt_content: str,
        model: str,
        response_time: float,
        token_usage: dict[str, int],
        success: bool = True,
        error: str | None = None,
        trace_id: str | None = None
    ) -> None:
        """
        记录提示词调用

        Args:
            prompt_type: 提示词类型
            prompt_content: 提示词内容
            model: 使用的模型
            response_time: 响应时间
            token_usage: 令牌使用情况
            success: 是否成功
            error: 错误信息（如果失败）
            trace_id: 追踪 ID
        """
        import datetime

        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "trace_id": trace_id or str(uuid.uuid4()),
            "prompt_type": prompt_type,
            "prompt_content": prompt_content,
            "model": model,
            "response_time": response_time,
            "token_usage": token_usage,
            "success": success,
            "error": error,
            "prompt_length": len(prompt_content),
            "response_preview": prompt_content[:100] + "..." if len(prompt_content) > 100 else prompt_content
        }

        # 保存到日志文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"❌ 提示词日志记录失败: {e}")

        logger.info(f"📝 提示词调用: {prompt_type} - {model} - {response_time:.2f}s - {token_usage}")


class LLMRetryHandler:
    """
    LLM 重试处理器

    为 LLM 调用提供智能重试机制
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retryable_errors: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        执行带重试的函数

        Args:
            func: 要执行的函数
            *args: 函数位置参数
            retryable_errors: 可重试的错误类型
            **kwargs: 函数关键字参数

        Returns:
            函数执行结果
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except retryable_errors as e:
                last_exception = e

                if attempt == self.max_retries:
                    logger.error(f"❌ 重试次数已达上限 ({self.max_retries}): {e}")
                    raise

                # 计算重试延迟
                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )

                logger.warning(f"⚠️ LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                logger.info(f"🔄 {delay:.1f} 秒后重试...")

                await asyncio.sleep(delay)

        # 如果所有重试都失败，抛出最后一次的异常
        raise last_exception


class LocalTraceCallbackHandler(AsyncCallbackHandler):
    """基于 LangChain 1.0 Callback 的本地追踪实现"""

    def __init__(
        self,
        trace_id: str | None = None,
        event_log: Path | None = None,
        tracker: LLMCallTracker | None = None,
        prompt_logger: PromptLogger | None = None,
        project_name: str = "lumoscribe2033",
    ) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self.event_log = event_log or Path("logs/llm_traces.log")
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        self.prompt_logger = prompt_logger
        self.project_name = project_name
        self._runs: dict[str, dict[str, Any]] = {}

    async def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], run_id: str, parent_run_id: str | None, **kwargs: Any) -> None:
        self._write_event(
            "chain_start",
            {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "chain": serialized.get("name"),
                "inputs": inputs,
            },
        )

    async def on_chain_end(self, outputs: dict[str, Any], run_id: str, parent_run_id: str | None, **kwargs: Any) -> None:
        self._write_event(
            "chain_end",
            {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "outputs": outputs,
            },
        )

    async def on_chain_error(self, error: Exception, run_id: str, parent_run_id: str | None, **kwargs: Any) -> None:
        self._write_event(
            "chain_error",
            {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "error": str(error),
            },
        )

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], run_id: str, parent_run_id: str | None, **kwargs: Any) -> None:
        prompt_text = "\n\n".join(prompts)
        self._runs[str(run_id)] = {
            "prompt": prompt_text,
            "model": serialized.get("id") or serialized.get("name") or "unknown",
            "start_time": time.time(),
            "metadata": kwargs.get("metadata") or {},
        }
        self._write_event(
            "llm_start",
            {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "model": serialized.get("id") or serialized.get("name"),
                "prompt": prompt_text,
            },
        )

    async def on_llm_end(self, response: LLMResult, run_id, parent_run_id, **kwargs):
        context = self._runs.pop(str(run_id), {})
        response_text = self._extract_text(response)
        elapsed = time.time() - context.get("start_time", time.time())
        token_usage = self._extract_token_usage(response)

        if self.tracker:
            self.tracker.track_llm_call(
                model_name=context.get("model", "unknown"),
                prompt=context.get("prompt", ""),
                response=response_text,
                metadata={
                    "elapsed": elapsed,
                    "token_usage": token_usage,
                    "project": self.project_name,
                },
                trace_id=self.trace_id,
            )

        if self.prompt_logger and context.get("prompt"):
            self.prompt_logger.log_prompt(
                prompt_type=context.get("metadata", {}).get("type", "llm"),
                prompt_content=context["prompt"],
                model=context.get("model", "unknown"),
                response_time=elapsed,
                token_usage=token_usage,
                success=True,
                trace_id=self.trace_id,
            )

        self._write_event(
            "llm_end",
            {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "response": response_text,
                "token_usage": token_usage,
                "elapsed": elapsed,
            },
        )

    async def on_llm_error(self, error: Exception, run_id: str, parent_run_id: str | None, **kwargs: Any) -> None:
        context = self._runs.pop(str(run_id), {})
        self._write_event(
            "llm_error",
            {
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "model": context.get("model"),
                "prompt": context.get("prompt"),
                "error": str(error),
            },
        )

    # Helper methods -----------------------------------------------------
    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "trace_id": self.trace_id,
            "event": event_type,
            **payload,
        }
        with open(self.event_log, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_text(result: LLMResult) -> str:
        texts: list[str] = []
        for generations in result.generations:
            for generation in generations:
                if hasattr(generation, "text") and generation.text:
                    texts.append(generation.text)
                elif hasattr(generation, "message") and getattr(generation.message, "content", None):
                    content = generation.message.content
                    if isinstance(content, list):
                        texts.extend(str(item) for item in content)
                    else:
                        texts.append(str(content))
        return "\n".join(texts)

    @staticmethod
    def _extract_token_usage(result: LLMResult) -> dict[str, int]:
        llm_output = result.llm_output or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        return {k: int(v) for k, v in token_usage.items()} if isinstance(token_usage, dict) else {}


# 全局实例
_langchain_middleware: LangChainMiddleware | None = None
_llm_tracker: LLMCallTracker | None = None
_prompt_logger: PromptLogger | None = None
_retry_handler: LLMRetryHandler | None = None


def get_langchain_middleware() -> LangChainMiddleware | None:
    """获取 LangChain 中间件实例"""
    return _langchain_middleware


def get_llm_tracker() -> LLMCallTracker | None:
    """获取 LLM 追踪器实例"""
    return _llm_tracker


def get_prompt_logger() -> PromptLogger | None:
    """获取提示词日志记录器实例"""
    return _prompt_logger


def get_retry_handler() -> LLMRetryHandler | None:
    """获取重试处理器实例"""
    return _retry_handler


def get_request_runnable_config(request: Request) -> RunnableConfig | None:
    """从 FastAPI Request 中获取 LangChain RunnableConfig"""
    return getattr(request.state, "langchain_runnable_config", None)


def get_request_executor(request: Request) -> LangChainExecutor | None:
    """从 FastAPI Request 中获取 LangChainExecutor"""
    return getattr(request.state, "langchain_executor", None)


def initialize_langchain_middleware(
    project_name: str = "lumoscribe2033",
    tracing_enabled: bool = True,
    trace_log_file: str = "logs/llm_traces.log",
    llm_call_log: str = "logs/llm_calls.log",
    prompt_log_file: str = "logs/prompts.log",
    default_tags: list[str] | None = None,
) -> None:
    """
    初始化 LangChain 中间件组件（纯本地实现）
    """
    global _langchain_middleware, _llm_tracker, _prompt_logger, _retry_handler

    _llm_tracker = LLMCallTracker(log_file=llm_call_log)
    _prompt_logger = PromptLogger(log_file=prompt_log_file)
    _retry_handler = LLMRetryHandler()
    _langchain_middleware = LangChainMiddleware(
        project_name=project_name,
        tracing_enabled=tracing_enabled,
        log_file=trace_log_file,
        tracker=_llm_tracker,
        prompt_logger=_prompt_logger,
        default_tags=default_tags,
    )

    logger.info("🚀 LangChain 本地中间件组件初始化完成")


def create_langchain_middleware_factory(
    project_name: str = "lumoscribe2033",
    tracing_enabled: bool = True,
    trace_log_file: str = "logs/llm_traces.log",
    llm_call_log: str = "logs/llm_calls.log",
    prompt_log_file: str = "logs/prompts.log",
    default_tags: list[str] | None = None,
) -> LangChainMiddleware:
    """
    创建新的 LangChainMiddleware 实例
    """
    tracker = LLMCallTracker(log_file=llm_call_log)
    prompt_logger = PromptLogger(log_file=prompt_log_file)
    return LangChainMiddleware(
        project_name=project_name,
        tracing_enabled=tracing_enabled,
        log_file=trace_log_file,
        tracker=tracker,
        prompt_logger=prompt_logger,
        default_tags=default_tags,
    )


def build_local_runnable_config(
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """
    构建可在任务/脚本中复用的 RunnableConfig，自动接入本地追踪
    """
    base_log = _langchain_middleware.log_file if _langchain_middleware else Path("logs/llm_traces.log")
    handler = LocalTraceCallbackHandler(
        trace_id=trace_id,
        event_log=base_log,
        tracker=_llm_tracker,
        prompt_logger=_prompt_logger,
    )
    merged_metadata = {"trace_id": handler.trace_id}
    if metadata:
        merged_metadata.update(metadata)
    return RunnableConfig(
        callbacks=[handler],
        metadata=merged_metadata,
        tags=tags or [],
    )


def _resolve_request_executor(config: RunnableConfig | None) -> LangChainExecutor | None:
    try:
        return get_executor_with_config(config)
    except RuntimeError:
        return None
