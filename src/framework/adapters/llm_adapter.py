"""
LLM 适配器

提供统一的 LLM 服务接口，支持 OpenAI 兼容 API 和本地模型。
支持 KATpro1 OpenAI 兼容 API。
符合 LangChain 1.0 最佳实践。
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Mock classes for when LangChain is not available
    class BaseMessage:
        def __init__(self, content: str, role: str = "user", **kwargs):
            self.content = content
            self.role = role
            self.kwargs = kwargs

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    class ToolMessage(BaseMessage):
        pass

    class ChatMessage(BaseMessage):
        pass

    class BaseChatModel:
        pass

    class ChatGeneration:
        def __init__(self, message: BaseMessage, **kwargs):
            self.message = message
            self.generation_info = kwargs

    class ChatResult:
        def __init__(self, generations: list[ChatGeneration], message: BaseMessage | None = None):
            self.generations = generations
            self.message = message

    class LLMResult:
        pass

    class CallbackManagerForLLMRun:
        pass


@dataclass
class LLMMessage:
    """LLM 消息数据类"""
    role: str  # system, user, assistant
    content: str
    name: str | None = None


@dataclass
class LLMResponse:
    """LLM 响应数据类"""
    content: str
    usage: dict[str, int]
    model: str
    finish_reason: str


class MessageConverter:
    """消息转换器 - 在不同消息格式间转换"""

    @staticmethod
    def to_openai_format(messages: list[BaseMessage | LLMMessage | dict]) -> list[dict]:
        """转换为 OpenAI 格式"""
        openai_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                # 已经是字典格式
                openai_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "name": msg.get("name")
                })
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # 有 role 和 content 属性的对象
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "name": getattr(msg, 'name', None)
                })
            else:
                # LLMMessage 对象
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name
                })

        return openai_messages

    @staticmethod
    def to_langchain_format(response_data: dict, messages: list[BaseMessage | dict]) -> AIMessage:
        """转换为 LangChain 格式"""
        choice = response_data["choices"][0]
        content = choice["message"].get("content", "")
        tool_calls = choice.get("tool_calls", [])

        # 构建 AIMessage
        ai_message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            response_metadata={
                "model": response_data.get("model", "unknown"),
                "usage": response_data.get("usage", {}),
                "finish_reason": choice.get("finish_reason", "stop")
            }
        )

        return ai_message


class LLMAdapter(ABC):
    """LLM 适配器抽象基类"""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[BaseMessage | LLMMessage | dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False
    ) -> LLMResponse:
        """聊天补全"""
        pass

    @abstractmethod
    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取文本嵌入"""
        pass

    @abstractmethod
    async def list_models(self) -> list[dict[str, Any]]:
        """列出可用模型"""
        pass


class LangChainLLMAdapter(BaseChatModel):
    """LangChain 1.0 兼容的 LLM 适配器基类"""

    model_name: str = "custom"
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 60
    base_url: str = ""
    api_key: str = ""

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "custom"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成方法"""
        # 转换为异步调用
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            self._agenerate(messages, stop, run_manager, **kwargs)
        )
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成方法"""
        # 转换消息格式并调用具体实现
        converted_messages = MessageConverter.to_openai_format(messages)

        # 获取参数
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # 调用具体的聊天补全方法
        response = await self._chat_completion_impl(
            converted_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 创建 AIMessage
        ai_message = AIMessage(
            content=response.content,
            response_metadata={
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason
            }
        )

        # 创建 ChatGeneration
        generation = ChatGeneration(
            message=ai_message,
            generation_info={
                "usage": response.usage,
                "finish_reason": response.finish_reason
            }
        )

        return ChatResult(generations=[generation])

    @abstractmethod
    async def _chat_completion_impl(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """具体的聊天补全实现"""
        pass

    def get_token_ids(self, text: str) -> list[int]:
        """获取文本的 token ID 列表"""
        # 简单的实现，实际应该使用正确的 tokenizer
        return list(range(len(text.split())))

    def get_num_tokens(self, text: str) -> int:
        """获取文本的 token 数量"""
        return len(text.split())


class OpenAIAdapter(LLMAdapter):
    """OpenAI 适配器 - 支持 OpenAI 兼容 API"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )

    async def chat_completion(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False
    ) -> LLMResponse:
        """真实的聊天补全实现"""
        try:
            # 转换消息格式
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name if msg.name else None
                })

            # 构建请求数据
            request_data = {
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4000
            }

            if stream:
                request_data["stream"] = True

            # 发送请求
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            # 解析响应
            choice = result["choices"][0]
            content = choice["message"]["content"]
            usage = result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            model = result.get("model", "unknown")
            finish_reason = choice.get("finish_reason", "stop")

            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                finish_reason=finish_reason
            )

        except Exception as e:
            # 返回错误信息作为内容
            return LLMResponse(
                content=f"API调用失败: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="error",
                finish_reason="error"
            )

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取文本嵌入"""
        try:
            request_data = {
                "input": texts,
                "model": "text-embedding-ada-002"  # 使用标准嵌入模型
            }

            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            response.raise_for_status()

            result = response.json()
            return [item["embedding"] for item in result["data"]]

        except Exception:
            # 返回模拟嵌入向量
            return [[0.1] * 1536 for _ in texts]

    async def list_models(self) -> list[dict[str, Any]]:
        """列出可用模型"""
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()

            result = response.json()
            return result.get("data", [])

        except Exception:
            # 返回默认模型列表
            return [
                {"id": "gpt-3.5-turbo", "type": "chat"},
                {"id": "gpt-4", "type": "chat"},
                {"id": "text-embedding-ada-002", "type": "embedding"}
            ]

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

    async def _chat_completion_impl(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """OpenAI 聊天补全实现"""
        try:
            request_data = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4000
            }

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            choice = result["choices"][0]
            content = choice["message"]["content"]
            usage = result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            model = result.get("model", "unknown")
            finish_reason = choice.get("finish_reason", "stop")

            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                finish_reason=finish_reason
            )

        except Exception as e:
            return LLMResponse(
                content=f"API调用失败: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="error",
                finish_reason="error"
            )


class OllamaAdapter(LLMAdapter):
    """Ollama 本地模型适配器"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )

    async def chat_completion(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False
    ) -> LLMResponse:
        """真实的Ollama聊天补全实现"""
        try:
            # 转换消息格式为Ollama格式
            ollama_messages = []
            for msg in messages:
                ollama_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            request_data = {
                "messages": ollama_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4000
            }

            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            return LLMResponse(
                content=result.get("message", {}).get("content", ""),
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                model=result.get("model", "unknown"),
                finish_reason="stop"
            )

        except Exception as e:
            return LLMResponse(
                content=f"Ollama API调用失败: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="error",
                finish_reason="error"
            )

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        """获取Ollama嵌入"""
        try:
            embeddings = []
            for text in texts:
                response = await self.client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"input": text}
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result.get("embedding", [0.1] * 384))

            return embeddings

        except Exception:
            return [[0.1] * 384 for _ in texts]

    async def list_models(self) -> list[dict[str, Any]]:
        """列出Ollama可用模型"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            result = response.json()
            models = []
            for model in result.get("models", []):
                models.append({
                    "id": model.get("name", ""),
                    "type": "chat",
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", "")
                })

            return models

        except Exception:
            return [
                {"id": "llama2", "type": "chat"},
                {"id": "mistral", "type": "chat"},
                {"id": "nomic-embed-text", "type": "embedding"}
            ]

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()

    async def _chat_completion_impl(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """Ollama 聊天补全实现"""
        try:
            request_data = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4000
            }

            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            return LLMResponse(
                content=result.get("message", {}).get("content", ""),
                usage={
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                },
                model=result.get("model", "unknown"),
                finish_reason="stop"
            )

        except Exception as e:
            return LLMResponse(
                content=f"Ollama API调用失败: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="error",
                finish_reason="error"
            )


class KATpro1Adapter(LLMAdapter):
    """KATpro1 OpenAI 兼容 API 适配器"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),  # KATpro1可能需要更长超时
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10)
        )

    async def chat_completion(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False
    ) -> LLMResponse:
        """KATpro1聊天补全实现"""
        try:
            # 转换消息格式
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name if msg.name else None
                })

            # 构建请求数据
            request_data = {
                "messages": openai_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4000,
                "model": self.model
            }

            if stream:
                request_data["stream"] = True

            # 发送请求到KATpro1
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            # 解析响应
            choice = result["choices"][0]
            content = choice["message"]["content"]
            usage = result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            model = result.get("model", self.model)
            finish_reason = choice.get("finish_reason", "stop")

            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                finish_reason=finish_reason
            )

        except Exception as e:
            return LLMResponse(
                content=f"KATpro1 API调用失败: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="error",
                finish_reason="error"
            )

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        """KATpro1嵌入实现"""
        try:
            request_data = {
                "input": texts,
                "model": self.model
            }

            response = await self.client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            response.raise_for_status()

            result = response.json()
            return [item["embedding"] for item in result["data"]]

        except Exception:
            # 返回模拟嵌入向量
            return [[0.1] * 1536 for _ in texts]

    async def list_models(self) -> list[dict[str, Any]]:
        """列出KATpro1可用模型"""
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()

            result = response.json()
            return result.get("data", [])

        except Exception:
            # 返回KATpro1模型信息
            return [
                {"id": self.model, "type": "chat", "provider": "KATpro1"},
                {"id": "text-embedding", "type": "embedding", "provider": "KATpro1"}
            ]

    async def close(self):
        """关闭HTTP客户端"""
        await self.client.aclose()


# 导出类
__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMAdapter",
    "LangChainLLMAdapter",
    "OpenAIAdapter",
    "OllamaAdapter",
    "KATpro1Adapter",
    "KATpro1LangChainAdapter",
    "StructuredOutputWrapper",
    "ToolBoundAdapter",
    "MessageConverter"
]


class KATpro1LangChainAdapter(LangChainLLMAdapter):
    """KATpro1 LangChain 1.0 兼容适配器"""

    api_key: str = ""
    client: Any = None

    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.7, max_tokens: int | None = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = 120
        # 使用 object.__setattr 避免 Pydantic 验证
        object.__setattr__(self, 'client', httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10)
        ))

    def with_structured_output(
        self,
        schema: dict | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """支持结构化输出的实现"""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.runnables import Runnable

            # 检查是否支持 function calling 或 tool calling
            if hasattr(self, '_supports_function_calling') and self._supports_function_calling:
                # 如果支持 function calling，使用原生方法
                return super().with_structured_output(schema, include_raw=include_raw, **kwargs)

            # 否则返回一个包装器，使用 JSON 解析实现结构化输出
            return StructuredOutputWrapper(self, schema, include_raw)

        except ImportError:
            # 如果没有 langchain_core，返回一个简单的包装器
            return StructuredOutputWrapper(self, schema, include_raw)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        """绑定工具的方法"""
        # 创建一个新的适配器实例，绑定工具
        from langchain_core.language_models.chat_models import BaseChatModel
        return ToolBoundAdapter(self, tools, **kwargs)

    async def _chat_completion_impl(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None
    ) -> LLMResponse:
        """KATpro1 LangChain 聊天补全实现"""
        try:
            # 构建请求数据
            request_data = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.max_tokens or 4000,
                "model": self.model_name
            }

            # 发送请求到KATpro1
            client = object.__getattribute__(self, 'client')
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            response.raise_for_status()

            result = response.json()

            # 解析响应
            choice = result["choices"][0]
            content = choice["message"]["content"]
            usage = result.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
            model = result.get("model", self.model_name)
            finish_reason = choice.get("finish_reason", "stop")

            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                finish_reason=finish_reason
            )

        except Exception as e:
            return LLMResponse(
                content=f"KATpro1 API调用失败: {str(e)}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model="error",
                finish_reason="error"
            )

    async def close(self):
        """关闭HTTP客户端"""
        client = object.__getattribute__(self, 'client')
        if client:
            await client.aclose()


class StructuredOutputWrapper:
    """结构化输出包装器"""

    def __init__(self, adapter: KATpro1LangChainAdapter, schema: dict | type, include_raw: bool = False):
        self.adapter = adapter
        self.schema = schema
        self.include_raw = include_raw

    async def invoke(self, input_data: str | list[BaseMessage], **kwargs: Any) -> Any | dict:
        """调用适配器并解析结构化输出"""
        try:
            # 调用原始适配器
            if isinstance(input_data, str):
                messages = [HumanMessage(content=input_data)]
            else:
                messages = input_data

            result = await self.adapter._agenerate(messages, **kwargs)

            # 获取响应内容
            content = result.generations[0].message.content

            # 尝试解析为 JSON
            import json
            parsed_data = json.loads(content)

            # 验证数据结构
            if isinstance(self.schema, type) and hasattr(self.schema, 'parse_obj'):
                # Pydantic 模型
                structured_result = self.schema.parse_obj(parsed_data)
            else:
                structured_result = parsed_data

            if self.include_raw:
                return {
                    "raw": result.generations[0].message,
                    "parsed": structured_result,
                    "parsing_error": None
                }
            else:
                return structured_result

        except Exception as e:
            if self.include_raw:
                return {
                    "raw": None,
                    "parsed": None,
                    "parsing_error": e
                }
            else:
                raise e

    def __call__(self, *args, **kwargs):
        """使包装器可调用"""
        return self.invoke(*args, **kwargs)


class ToolBoundAdapter:
    """工具绑定适配器"""

    def __init__(self, adapter: KATpro1LangChainAdapter, tools: list[Any], **kwargs: Any):
        self.adapter = adapter
        self.tools = tools
        self.kwargs = kwargs

    async def invoke(self, input_data: str | list[BaseMessage], **kwargs: Any) -> Any:
        """调用绑定工具的适配器"""
        # 这里可以实现工具调用逻辑
        # 目前简单地调用原始适配器
        if isinstance(input_data, str):
            messages = [HumanMessage(content=input_data)]
        else:
            messages = input_data

        return await self.adapter._agenerate(messages, **{**self.kwargs, **kwargs})

    def __call__(self, *args, **kwargs):
        """使适配器可调用"""
        return self.invoke(*args, **kwargs)
