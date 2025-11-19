"""
LangChain 1.0 RouterChain + RunnableSequence 多模型路由执行器

基于 LangChain 1.0 的 RouterChain 和 RunnableSequence 实现智能多模型路由，
支持动态模型选择、链式执行和性能监控。
"""

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal, Optional, TypedDict, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence
from pydantic import BaseModel, Field

from ..shared.config import config_manager
from ..shared.logging import get_logger
from ..shared.telemetry import trace_method

logger = get_logger(__name__)


class RouteDecision(BaseModel):
    """路由决策的结构化输出"""
    model_name: str = Field(
        ...,
        description="选择的模型名称",
        json_schema_extra={"example": "openai-gpt4"}
    )
    reason: str = Field(
        ...,
        description="选择该模型的原因",
        json_schema_extra={"example": "复杂推理任务需要高质量模型"}
    )
    confidence: float = Field(
        ...,
        description="路由决策的置信度 (0-1)",
        json_schema_extra={"example": 0.95}
    )


class ExecutionState(TypedDict):
    """执行状态"""
    input: str
    route_decision: dict[str, Any] | None
    selected_model: str | None
    response: str | None
    execution_time: float | None
    success: bool
    error_message: str | None
    usage_metadata: dict[str, Any] | None
    performance_metrics: dict[str, Any] | None


class ModelConfig(TypedDict):
    """模型配置"""
    name: str
    model: BaseChatModel
    cost_per_token: float
    avg_response_time: float
    success_rate: float
    capabilities: list[str]


class LangChainRunner:
    """LangChain 1.0 多模型路由执行器"""

    def __init__(self, models: dict[str, BaseChatModel] | None = None):
        """
        初始化 LangChain 执行器

        Args:
            models: 可选的模型字典，如果不提供则使用配置管理器中的启用模型
        """
        # 如果提供了模型，使用提供的模型；否则使用配置管理器
        if models is not None:
            self.models = models
        else:
            # 从配置管理器获取启用的模型配置，但需要外部提供模型实例
            enabled_configs = config_manager.get_enabled_models()
            self.models = {}  # 需要外部注入模型实例
            logger.info(f"LangChainRunner initialized with config from ConfigManager: {list(enabled_configs.keys())}")

        self.model_configs = self._create_model_configs()
        self.performance_metrics = self._initialize_metrics()
        self.routing_chain = self._create_routing_chain()
        self.execution_chain = self._create_execution_chain()

        # 初始化健康检查器
        self.health_checker = None
        self._enable_health_checking = config_manager.get_setting("enable_health_checking", True)

        # 初始化高级路由缓存（基于LangChain最佳实践）
        self._routing_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._cache_ttl = config_manager.get_setting("routing_cache_ttl", 300)  # 5分钟默认缓存
        self._cache_cleanup_interval = config_manager.get_setting("routing_cache_cleanup_interval", 3600)  # 1小时清理
        self._cache_max_size = config_manager.get_setting("routing_cache_max_size", 1000)  # 最大缓存条目数
        self._last_cleanup_time = time.time()

        logger.info(f"LangChainRunner initialized with models: {list(self.models.keys())}")

    @classmethod
    def from_config_manager(cls, model_instances: dict[str, BaseChatModel]) -> 'LangChainRunner':
        """
        基于配置管理器创建 LangChainRunner

        Args:
            model_instances: 模型实例字典，key 为模型名称，value 为模型实例

        Returns:
            LangChainRunner 实例
        """
        # 验证配置
        errors = config_manager.validate_config()
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")

        # 获取启用的模型配置
        enabled_configs = config_manager.get_enabled_models()
        logger.info(f"Enabled models from config: {list(enabled_configs.keys())}")

        # 过滤出有实例的模型
        available_models = {}
        for model_name, config in enabled_configs.items():
            if model_name in model_instances:
                available_models[model_name] = model_instances[model_name]
            else:
                logger.warning(f"Model {model_name} is enabled in config but no instance provided")

        if not available_models:
            raise ValueError("No models available - check configuration and model instances")

        return cls(available_models)

    def initialize_health_checker(self):
        """初始化健康检查器"""
        if not self._enable_health_checking:
            logger.info("Health checking is disabled")
            return

        if self.health_checker is None and self.models:
            from .health_checker import HealthCheckConfig, LangChainHealthChecker

            config = HealthCheckConfig(
                timeout=config_manager.get_setting("health_check_timeout", 30.0),
                max_retries=config_manager.get_setting("health_check_retries", 3),
                check_interval=config_manager.get_setting("health_check_interval", 60.0),
                failure_threshold=config_manager.get_setting("health_failure_threshold", 5),
                enable_auto_recovery=config_manager.get_setting("enable_auto_recovery", True),
                enable_tracing=config_manager.get_setting("enable_health_tracing", True)
            )

            self.health_checker = LangChainHealthChecker(self.models, config)
            logger.info("LangChain health checker initialized")

    async def start_health_monitoring(self):
        """启动健康监控"""
        if self.health_checker:
            await self.health_checker.start_monitoring()
            logger.info("Health monitoring started")

    async def stop_health_monitoring(self):
        """停止健康监控"""
        if self.health_checker:
            await self.health_checker.stop_monitoring()
            logger.info("Health monitoring stopped")

    def _create_model_configs(self) -> dict[str, ModelConfig]:
        """创建模型配置"""
        configs = {}
        for name, model in self.models.items():
            # 从配置管理器获取配置
            config_manager_model = config_manager.get_model_by_name(name)
            if config_manager_model:
                # 使用配置管理器中的配置
                configs[name] = ModelConfig(
                    name=name,
                    model=model,
                    cost_per_token=config_manager_model.cost_per_token,
                    avg_response_time=0.0,
                    success_rate=1.0,
                    capabilities=[cap.value for cap in config_manager_model.capabilities]
                )
            else:
                # 使用默认配置
                configs[name] = ModelConfig(
                    name=name,
                    model=model,
                    cost_per_token=self._get_default_cost(name),
                    avg_response_time=0.0,
                    success_rate=1.0,
                    capabilities=self._get_model_capabilities(name)
                )
        return configs

    def _get_default_cost(self, model_name: str) -> float:
        """获取模型的默认成本"""
        cost_mapping = {
            "openai-gpt4": 0.03,
            "openai-gpt35": 0.005,
            "ollama-llama2": 0.001,
            "ollama-mistral": 0.002,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003
        }
        return cost_mapping.get(model_name, 0.01)

    def _get_model_capabilities(self, model_name: str) -> list[str]:
        """获取模型能力"""
        capabilities_mapping = {
            "openai-gpt4": ["complex_reasoning", "creative_writing", "code_analysis", "high_quality"],
            "openai-gpt35": ["general_conversation", "text_processing", "moderate_quality"],
            "ollama-llama2": ["simple_queries", "fast_response", "low_cost"],
            "ollama-mistral": ["code_analysis", "technical_tasks", "balanced_performance"],
            "claude-3-opus": ["complex_reasoning", "creative_writing", "high_quality"],
            "claude-3-sonnet": ["general_conversation", "code_analysis", "moderate_quality"]
        }
        return capabilities_mapping.get(model_name, ["general_purpose"])

    def _initialize_metrics(self) -> dict[str, dict[str, float]]:
        """初始化性能指标"""
        return {
            name: {
                "success_rate": 1.0,
                "avg_response_time": 0.0,
                "cost_per_token": self._get_default_cost(name),
                "total_requests": 0,
                "total_cost": 0.0
            }
            for name in self.models
        }

    def _create_routing_chain(self) -> Runnable:
        """创建基于 LangChain 1.0 最佳实践的路由链"""
        # 基于 LangChain 最佳实践的路由提示词模板
        routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是一个智能 LLM 路由器专家。根据用户请求的内容、复杂度、成本预算、系统状态和上下文信息，选择最适合的模型。

## 路由原则（基于 LangChain 1.0 最佳实践）
1. **动态上下文感知**：根据运行时状态调整路由策略
2. **多维度评估**：内容类型、性能指标、成本效益、系统负载
3. **故障转移机制**：自动降级到可用模型
4. **负载均衡**：避免过度使用单一模型

## 可用模型及其特点：
{model_descriptions}

## 当前系统状态：
{system_context}

## 路由决策框架：
### 内容类型映射：
- **复杂推理**：数学计算、逻辑分析、算法设计 → openai-gpt4, claude-3-opus
- **创意写作**：故事生成、文案创作、内容生成 → openai-gpt4, claude-3-opus
- **代码分析**：编程问题、技术调试、代码审查 → ollama-mistral, openai-gpt4
- **文本处理**：文档分析、总结翻译、信息提取 → openai-gpt35, claude-3-sonnet
- **简单查询**：快速问答、基本信息、状态查询 → ollama-llama2, openai-gpt35

### 性能权重因子：
- **成功率**：>90% (+20%), 80-90% (+10%), <80% (-30%)
- **响应时间**：>10s (-25%), 5-10s (-10%), <5s (+15%)
- **负载状态**：高负载 (-20%), 中等 (-5%), 低负载 (+10%)

### 成本优化策略：
- **预算充足**：优先质量，选择高性能模型
- **预算中等**：平衡性能与成本
- **预算有限**：优先成本，选择经济模型

## 上下文信息：
{routing_context}

## 分析流程：
1. **内容识别**：分析请求类型、复杂度、紧急程度
2. **模型匹配**：根据能力映射找到候选模型
3. **状态评估**：检查模型健康状态和性能指标
4. **权重计算**：综合性能、成本、负载因素
5. **最终选择**：选择综合得分最高的可用模型

## 输出格式要求：
请严格按照以下 JSON 格式输出，不要包含任何其他内容：

{{"model_name": "模型名称","reason": "详细决策理由","confidence": 0.85}}
"""),
            ("user", "用户请求：{input}")
        ])

        # 使用最强的模型作为路由模型
        routing_model = self._get_routing_model()

        # 创建结构化输出解析器（基于 LangChain 1.0 最佳实践）
        structured_llm = routing_model.with_structured_output(RouteDecision)

        # 创建路由链（使用 LangChain 1.0 推荐的链式组合）
        routing_chain = routing_prompt | structured_llm

        return routing_chain

    def _get_routing_model(self) -> BaseChatModel:
        """获取用于路由的模型"""
        # 优先使用高质量模型进行路由决策
        high_quality_models = ["openai-gpt4", "claude-3-opus"]
        for model_name in high_quality_models:
            if model_name in self.models:
                return self.models[model_name]

        # 降级到其他可用模型
        for model_name, model in self.models.items():
            if "gpt" in model_name or "claude" in model_name:
                return model

        # 最后使用第一个可用模型
        return list(self.models.values())[0]

    def _create_execution_chain(self) -> RunnableSequence:
        """创建执行链"""
        # 创建模型执行链
        model_chains = {}

        for model_name, model in self.models.items():
            # 为每个模型创建执行链
            model_chain = (
                RunnableLambda(self._prepare_model_input) |
                model |
                RunnableLambda(self._extract_response)
            )
            model_chains[model_name] = model_chain

        # 创建多路由执行器
        # 在 LangChain v1.0 中，我们需要手动实现路由逻辑
        return self._create_manual_router(model_chains)

    @trace_method
    async def route_request(self, request: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        路由请求到合适的模型

        Args:
            request: 用户请求
            context: 上下文信息

        Returns:
            路由决策结果
        """
        start_time = time.time()

        try:
            # 准备路由上下文
            routing_context = self._prepare_routing_context(context)

            # 执行路由决策
            decision = await self.routing_chain.ainvoke({
                "input": request,
                **routing_context
            })

            execution_time = time.time() - start_time

            result = {
                "model_name": decision.model_name,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "execution_time": execution_time,
                "success": True
            }

            logger.info(f"Routing decision: {decision.model_name} (confidence: {decision.confidence})")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Routing failed: {str(e)}")

            # 返回默认模型
            default_model = self._get_default_model()
            return {
                "model_name": default_model,
                "reason": f"路由失败，使用默认模型: {str(e)}",
                "confidence": 0.5,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }

    def _prepare_routing_context(self, context: dict[str, Any] | None) -> dict[str, Any]:
        """准备增强的路由上下文"""
        # 生成模型描述
        model_descriptions = []
        for name, config in self.model_configs.items():
            capabilities_str = ", ".join(config["capabilities"])
            cost_str = f"${config['cost_per_token']:.4f}/token"
            latency_str = f"{config.get('avg_response_time', 0):.2f}s"
            model_descriptions.append(
                f"- {name}: {capabilities_str} (成本: {cost_str}, 延迟: {latency_str})"
            )

        # 生成系统上下文
        system_context = self._generate_system_context()

        # 生成路由上下文
        routing_context = self._generate_routing_context(context)

        return {
            "model_descriptions": "\n".join(model_descriptions),
            "system_context": system_context,
            "routing_context": routing_context
        }

    def _generate_system_context(self) -> str:
        """生成系统状态上下文"""
        system_info = []

        # 模型性能状态
        model_statuses = []
        for name, metrics in self.performance_metrics.items():
            status = self._assess_model_status(name, metrics)
            model_statuses.append(f"- {name}: {status}")

        system_info.append(f"**模型状态**: {'; '.join(model_statuses)}")

        # 系统负载情况
        load_info = self._assess_system_load()
        system_info.append(f"**系统负载**: {load_info}")

        # 成本概览
        cost_info = self._generate_cost_summary()
        system_info.append(f"**成本概览**: {cost_info}")

        return "\n".join(system_info)

    def _assess_model_status(self, model_name: str, metrics: dict[str, Any]) -> str:
        """评估模型状态"""
        success_rate = metrics.get('success_rate', 0)
        response_time = metrics.get('avg_response_time', 0)
        total_requests = metrics.get('total_requests', 0)

        # 状态评估逻辑
        if success_rate < 0.8:
            status = "⚠️ 故障"
        elif success_rate < 0.9:
            status = "🟡 不稳定"
        elif response_time > 10:
            status = "🐌 响应慢"
        elif total_requests == 0:
            status = "⚪ 未使用"
        else:
            status = "✅ 正常"

        return f"{status} (成功率:{success_rate:.1%}, 响应:{response_time:.1f}s, 请求:{total_requests})"

    def _assess_system_load(self) -> str:
        """评估系统负载"""
        total_requests = sum(m.get('total_requests', 0) for m in self.performance_metrics.values())
        avg_response_time = sum(m.get('avg_response_time', 0) for m in self.performance_metrics.values()) / len(self.performance_metrics)

        if total_requests > 1000:
            load_level = "高负载"
        elif total_requests > 100:
            load_level = "中等负载"
        else:
            load_level = "低负载"

        return f"{load_level} (总请求:{total_requests}, 平均响应:{avg_response_time:.1f}s)"

    def _generate_cost_summary(self) -> str:
        """生成成本概览"""
        total_cost = sum(m.get('total_cost', 0) for m in self.performance_metrics.values())
        avg_cost_per_request = total_cost / max(sum(m.get('total_requests', 1) for m in self.performance_metrics.values()), 1)

        return f"总成本:${total_cost:.2f}, 平均成本:${avg_cost_per_request:.4f}/请求"

    def _generate_routing_context(self, context: dict[str, Any] | None) -> str:
        """生成路由上下文"""
        context_info = []

        if context:
            # 预算提示
            if budget_hint := context.get('budget_hint'):
                context_info.append(f"**预算提示**: {budget_hint}")

            # 优先级
            if priority := context.get('priority'):
                context_info.append(f"**优先级**: {priority}")

            # 响应时间要求
            if max_response_time := context.get('max_response_time'):
                context_info.append(f"**最大响应时间**: {max_response_time}s")

        # 添加默认上下文
        if not context_info:
            context_info.append("**默认路由**: 无特殊要求，使用智能路由")

        return "\n".join(context_info)

    def _get_default_model(self) -> str:
        """获取默认模型"""
        # 使用配置管理器的默认模型选择逻辑
        default_model_name = config_manager.get_default_model()
        if default_model_name and default_model_name in self.models:
            return default_model_name

        # 降级到优先级选择
        preferred_models = ["openai-gpt35-turbo", "ollama-mistral", "groq-llama3"]
        for model_name in preferred_models:
            if model_name in self.models:
                return model_name

        # 最后使用第一个可用模型
        return list(self.models.keys())[0]

    @trace_method
    async def execute_request(self, request: str, **kwargs) -> dict[str, Any]:
        """
        执行请求并返回结果

        Args:
            request: 用户请求
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        start_time = time.time()

        try:
            # 路由决策
            routing_result = await self.route_request(request)

            if not routing_result["success"]:
                raise Exception(f"路由失败: {routing_result.get('error', 'Unknown error')}")

            model_name = routing_result["model_name"]
            selected_model = self.models[model_name]

            # 执行请求
            response = await selected_model.ainvoke([
                SystemMessage(content="你是一个专业的 AI 助手，请提供准确、有用的回答。"),
                HumanMessage(content=request)
            ], **kwargs)

            execution_time = time.time() - start_time

            # 更新性能指标
            await self._update_metrics(model_name, success=True, response_time=execution_time)

            result = {
                "model": model_name,
                "response": response.content if hasattr(response, 'content') else str(response),
                "usage": getattr(response, 'usage_metadata', {}),
                "execution_time": execution_time,
                "success": True,
                "routing_info": routing_result
            }

            logger.info(f"Request executed successfully with {model_name} in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Request execution failed: {str(e)}")

            # 更新失败指标
            if "model_name" in locals():
                await self._update_metrics(model_name, success=False, response_time=execution_time)

            raise e

    @trace_method
    async def execute_chain(self, request: str, chain_config: dict[str, Any]) -> dict[str, Any]:
        """
        执行链式请求

        Args:
            request: 初始请求
            chain_config: 链配置

        Returns:
            链执行结果
        """
        start_time = time.time()

        try:
            # 执行路由决策
            routing_result = await self.route_request(request)

            if not routing_result["success"]:
                raise Exception(f"路由失败: {routing_result.get('error', 'Unknown error')}")

            model_name = routing_result["model_name"]
            selected_model = self.models[model_name]

            # 构建执行链
            execution_chain = self._build_chain(selected_model, chain_config)

            # 执行链
            result = await execution_chain.ainvoke({"input": request})

            execution_time = time.time() - start_time

            # 更新性能指标
            await self._update_metrics(model_name, success=True, response_time=execution_time)

            return {
                "model": model_name,
                "result": result,
                "execution_time": execution_time,
                "success": True,
                "routing_info": routing_result
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Chain execution failed: {str(e)}")
            raise e

    def _build_chain(self, model: BaseChatModel, chain_config: dict[str, Any]) -> RunnableSequence:
        """构建执行链"""
        chain_steps = []

        # 添加系统提示词
        if "system_prompt" in chain_config:
            chain_steps.append(RunnableLambda(
                lambda x: [SystemMessage(content=chain_config["system_prompt"])] + x
            ))

        # 添加模型
        chain_steps.append(model)

        # 添加输出解析器
        if chain_config.get("parse_output", True):
            chain_steps.append(StrOutputParser())

        return RunnableSequence(*chain_steps)

    async def _update_metrics(self, model_name: str, success: bool, response_time: float):
        """更新模型性能指标"""
        metrics = self.performance_metrics[model_name]

        # 更新成功率 (指数加权平均)
        metrics["success_rate"] = (
            0.9 * metrics["success_rate"] + 0.1 * (1.0 if success else 0.0)
        )

        # 更新平均响应时间 (指数加权平均)
        if metrics["avg_response_time"] == 0:
            metrics["avg_response_time"] = response_time
        else:
            metrics["avg_response_time"] = (
                0.9 * metrics["avg_response_time"] + 0.1 * response_time
            )

        # 更新总请求次数
        metrics["total_requests"] += 1

        # 更新总成本
        if success:
            cost = self.model_configs[model_name]["cost_per_token"] * response_time
            metrics["total_cost"] += cost

    def get_performance_metrics(self) -> dict[str, dict[str, Any]]:
        """获取性能指标"""
        return {
            name: {
                **metrics,
                "capabilities": self.model_configs[name]["capabilities"]
            }
            for name, metrics in self.performance_metrics.items()
        }

    def get_model_info(self) -> dict[str, dict[str, Any]]:
        """获取模型信息"""
        return {
            name: {
                "config": config,
                "metrics": self.performance_metrics[name]
            }
            for name, config in self.model_configs.items()
        }

    def _prepare_model_input(self, inputs: str | dict | list) -> list[BaseMessage]:
        """准备模型输入"""
        if isinstance(inputs, str):
            return [HumanMessage(content=inputs)]
        elif isinstance(inputs, dict):
            input_text = inputs.get("input", "")
            return [HumanMessage(content=input_text)]
        elif isinstance(inputs, list):
            # 如果已经是消息列表，直接返回
            if all(isinstance(msg, BaseMessage) for msg in inputs):
                return inputs
            else:
                # 将字符串列表转换为消息列表
                return [HumanMessage(content=str(msg)) for msg in inputs]
        else:
            return [HumanMessage(content=str(inputs))]

    def _extract_response(self, response: AIMessage | dict | str) -> str | dict:
        """提取模型响应"""
        if isinstance(response, AIMessage):
            return response.content
        elif isinstance(response, dict):
            # 如果是字典，保持原样
            return response
        else:
            # 其他情况转换为字符串
            return str(response)

    def _create_manual_router(self, model_chains: dict[str, Runnable]) -> Runnable:
        """手动创建智能路由器"""
        async def route_and_execute(inputs):
            """智能路由和执行逻辑"""
            try:
                # 获取输入内容
                if isinstance(inputs, dict):
                    input_text = inputs.get("input", "")
                elif isinstance(inputs, str):
                    input_text = inputs
                else:
                    input_text = str(inputs)

                # 智能路由决策
                routing_decision = await self._intelligent_route(input_text)

                if not routing_decision["success"]:
                    # 路由失败时使用默认模型
                    model_name = self._get_default_model()
                    logger.warning(f"智能路由失败，使用默认模型: {model_name}")
                else:
                    model_name = routing_decision["model_name"]
                    logger.info(f"智能路由选择模型: {model_name} (置信度: {routing_decision['confidence']})")

                # 检查模型是否可用
                if model_name not in model_chains:
                    raise ValueError(f"模型 {model_name} 不可用")

                # 执行模型链
                chain = model_chains[model_name]

                # 添加路由上下文
                if isinstance(inputs, dict):
                    inputs["routing_context"] = {
                        "selected_model": model_name,
                        "routing_reason": routing_decision.get("reason", ""),
                        "confidence": routing_decision.get("confidence", 0.5)
                    }

                # 执行并返回结果
                result = await chain.ainvoke(inputs)

                # 返回增强的结果
                return {
                    "result": result,
                    "model": model_name,
                    "routing_info": routing_decision,
                    "success": True
                }

            except Exception as e:
                logger.error(f"路由执行失败: {str(e)}")
                raise ValueError(f"路由执行失败: {str(e)}")

        return RunnableLambda(route_and_execute)

    async def _intelligent_route(self, input_text: str) -> dict[str, Any]:
        """基于 LangChain 1.0 最佳实践的智能路由决策"""
        try:
            # 1. 动态内容分析（使用 LangChain 推荐的上下文感知）
            content_analysis = await self._dynamic_content_analysis(input_text)

            # 2. 获取模型可用性状态
            model_availability = await self._get_model_availability()

            # 3. 模型评分（考虑可用性）
            model_scores = self._score_models(content_analysis, model_availability)

            # 4. 性能加权（基于实时指标）
            weighted_scores = self._apply_performance_weights(model_scores)

            # 5. 动态健康权重调整
            health_weighted_scores = self._apply_dynamic_health_weights(weighted_scores)

            # 6. 成本优化分析
            cost_optimized_scores = self._apply_cost_optimization(health_weighted_scores, content_analysis)

            # 7. 负载均衡调整
            balanced_scores = self._apply_load_balancing(cost_optimized_scores)

            # 8. 智能故障转移
            final_model = await self._intelligent_fallback_routing(
                input_text, content_analysis, balanced_scores, model_availability
            )
            model_name, final_score = final_model

            # 9. 计算置信度
            confidence = self._calculate_confidence(balanced_scores, final_score)

            # 10. 生成详细理由
            detailed_reason = self._generate_detailed_reason(
                content_analysis, model_name, confidence, balanced_scores
            )

            return {
                "model_name": model_name,
                "reason": detailed_reason,
                "confidence": confidence,
                "success": True,
                "scores": balanced_scores,
                "availability": model_availability,
                "analysis": content_analysis,
                "final_score": final_score
            }

        except Exception as e:
            logger.error(f"智能路由决策失败: {str(e)}")
            # 使用降级策略
            fallback_model = await self._get_fallback_model()
            return {
                "model_name": fallback_model,
                "reason": f"路由系统异常，使用降级模型: {str(e)}",
                "confidence": 0.3,
                "success": False
            }

    async def _get_model_availability(self) -> dict[str, bool]:
        """获取模型可用性状态"""
        availability = {}

        for model_name in self.models.keys():
            # 默认可用
            is_available = True

            # 检查健康状态
            if self.health_checker and model_name in self.health_checker.health_status:
                health = self.health_checker.health_status[model_name]
                is_available = health.get("is_healthy", True)

                # 检查连续失败次数
                consecutive_failures = health.get("metadata", {}).get("consecutive_failures", 0)
                if consecutive_failures > 3:
                    is_available = False
                    logger.warning(f"Model {model_name} marked as unavailable due to {consecutive_failures} consecutive failures")

            availability[model_name] = is_available

        logger.debug(f"Model availability: {availability}")
        return availability

    def _get_routing_cache_key(self, input_text: str, context: dict[str, Any] | None) -> str:
        """生成路由缓存键（基于LangChain最佳实践）"""
        import hashlib
        import json

        # 创建缓存键的组件（包含更多上下文信息）
        key_components = {
            "input_text": input_text,
            "context": context,
            "model_configs": {name: {
                "capabilities": config["capabilities"],
                "cost_per_token": config["cost_per_token"]
            } for name, config in self.model_configs.items()},
            "performance_metrics": {name: {
                "success_rate": metrics["success_rate"],
                "avg_response_time": metrics["avg_response_time"]
            } for name, metrics in self.performance_metrics.items()}
        }

        # 序列化并生成哈希
        key_string = json.dumps(key_components, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_time: float, model_health: dict[str, Any]) -> bool:
        """检查缓存是否有效（增强版，包含模型健康状态检查）"""
        # 检查TTL过期
        if (time.time() - cache_time) > self._cache_ttl:
            return False

        # 检查模型健康状态变化（如果有缓存的健康状态信息）
        cached_health = model_health.get("health_snapshot", {})
        if self.health_checker:
            current_health = {}
            for model_name in self.models.keys():
                if model_name in self.health_checker.health_status:
                    health = self.health_checker.health_status[model_name]
                    current_health[model_name] = {
                        "is_healthy": health.get("is_healthy", True),
                        "success_rate": health.get("success_rate", 1.0),
                        "consecutive_failures": health.get("metadata", {}).get("consecutive_failures", 0)
                    }

            # 如果健康状态发生显著变化，使缓存失效
            for model_name, current in current_health.items():
                cached = cached_health.get(model_name, {})
                if (current.get("is_healthy", True) != cached.get("is_healthy", True) or
                    abs(current.get("success_rate", 1.0) - cached.get("success_rate", 1.0)) > 0.2):
                    return False

        return True

    def _cleanup_expired_cache(self) -> int:
        """清理过期缓存（增强版，支持LRU和容量管理）"""
        current_time = time.time()
        cleaned_count = 0

        if current_time - self._last_cleanup_time > self._cache_cleanup_interval:
            expired_keys = []
            healthy_models = set()

            # 获取当前健康模型列表
            if self.health_checker:
                healthy_models = {
                    name for name, health in self.health_checker.health_status.items()
                    if health.get("is_healthy", True)
                }

            # 检查过期和无效缓存项
            for key, (result, cache_time, model_health) in self._routing_cache.items():
                # 检查TTL过期
                if not self._is_cache_valid(cache_time, model_health):
                    expired_keys.append(key)
                # 检查模型健康状态（如果选择了不健康的模型）
                elif result.get("model_name") not in healthy_models:
                    expired_keys.append(key)

            # 清理过期项
            for key in expired_keys:
                del self._routing_cache[key]
                cleaned_count += 1

            # 如果缓存大小超过限制，执行LRU清理
            while len(self._routing_cache) > self._cache_max_size:
                # 找到最老的缓存项（这里简化为随机删除，实际可以维护LRU队列）
                oldest_key = next(iter(self._routing_cache))
                del self._routing_cache[oldest_key]
                cleaned_count += 1
                self._cache_stats["evictions"] += 1

            self._last_cleanup_time = current_time
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个过期/无效缓存项，当前缓存大小: {len(self._routing_cache)}")

        return cleaned_count

    def _get_cached_routing_result(self, cache_key: str) -> dict[str, Any] | None:
        """获取缓存的路由结果（增强版）"""
        if cache_key in self._routing_cache:
            result, cache_time, model_health = self._routing_cache[cache_key]

            # 检查缓存有效性
            if self._is_cache_valid(cache_time, model_health):
                self._cache_stats["hits"] += 1
                logger.debug(f"路由缓存命中: {cache_key[:8]}...")

                # 记录缓存命中时的健康状态变化
                if self.health_checker:
                    current_health = {}
                    for model_name in self.models.keys():
                        if model_name in self.health_checker.health_status:
                            health = self.health_checker.health_status[model_name]
                            current_health[model_name] = {
                                "is_healthy": health.get("is_healthy", True),
                                "success_rate": health.get("success_rate", 1.0)
                            }

                    # 更新缓存项的健康状态快照
                    self._routing_cache[cache_key] = (result, cache_time, {"health_snapshot": current_health})

                return result
            else:
                # 缓存过期，删除
                del self._routing_cache[cache_key]
                self._cache_stats["misses"] += 1
                logger.debug(f"路由缓存过期: {cache_key[:8]}...")

        return None

    def _cache_routing_result(self, cache_key: str, result: dict[str, Any]) -> None:
        """缓存路由结果（增强版）"""
        # 获取当前模型健康状态快照
        model_health = {}
        if self.health_checker:
            for model_name in self.models.keys():
                if model_name in self.health_checker.health_status:
                    health = self.health_checker.health_status[model_name]
                    model_health[model_name] = {
                        "is_healthy": health.get("is_healthy", True),
                        "success_rate": health.get("success_rate", 1.0),
                        "consecutive_failures": health.get("metadata", {}).get("consecutive_failures", 0)
                    }

        health_snapshot = {"health_snapshot": model_health}

        # 如果已存在相同键，先删除（保持LRU顺序）
        if cache_key in self._routing_cache:
            del self._routing_cache[cache_key]

        # 检查缓存容量限制
        while len(self._routing_cache) >= self._cache_max_size:
            # 简化的LRU：删除最老的项
            oldest_key = next(iter(self._routing_cache))
            del self._routing_cache[oldest_key]
            self._cache_stats["evictions"] += 1

        # 存储新缓存项
        self._routing_cache[cache_key] = (result, time.time(), health_snapshot)

        # 定期清理过期缓存
        self._cleanup_expired_cache()

    @lru_cache(maxsize=1000)
    def _cached_content_analysis(self, input_text: str) -> dict[str, Any]:
        """缓存内容分析结果（基于LangChain最佳实践）"""
        return self._analyze_content(input_text)

    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计信息（基于LangChain最佳实践）"""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0

        # 获取缓存内存使用情况
        cache_size_bytes = len(str(self._routing_cache))
        memory_usage_mb = cache_size_bytes / (1024 * 1024)

        stats = {
            "routing_cache": {
                "total_requests": total_requests,
                "cache_hits": self._cache_stats["hits"],
                "cache_misses": self._cache_stats["misses"],
                "hit_rate": hit_rate,
                "cache_size": len(self._routing_cache),
                "max_size": self._cache_max_size,
                "evictions": self._cache_stats["evictions"],
                "memory_usage_mb": round(memory_usage_mb, 2),
                "is_healthy": hit_rate > 0.3,  # 缓存命中率>30%认为健康
                "recommendation": self._get_cache_optimization_recommendation(hit_rate)
            }
        }

        return stats

    def _get_cache_optimization_recommendation(self, hit_rate: float) -> str:
        """获取缓存优化建议"""
        if hit_rate > 0.7:
            return "缓存效果优秀，可以考虑增加缓存TTL"
        elif hit_rate > 0.5:
            return "缓存效果良好，保持当前配置"
        elif hit_rate > 0.3:
            return "缓存效果一般，建议优化缓存键生成策略"
        else:
            return "缓存效果较差，建议检查缓存键生成或增加缓存容量"

    def clear_cache(self) -> None:
        """清空所有缓存"""
        cache_size = len(self._routing_cache)
        self._routing_cache.clear()
        logger.info(f"已清空路由缓存，共清理 {cache_size} 个缓存项")

    async def warmup_cache(self, warmup_data: list[tuple[str, dict[str, Any] | None]]) -> int:
        """预热缓存（基于LangChain最佳实践）"""
        warmed_count = 0

        for input_text, context in warmup_data:
            cache_key = self._get_routing_cache_key(input_text, context)

            # 检查是否已缓存
            if cache_key not in self._routing_cache:
                try:
                    # 执行路由决策并缓存
                    routing_result = await self._intelligent_route(input_text)
                    self._cache_routing_result(cache_key, routing_result)
                    warmed_count += 1
                    logger.debug(f"缓存预热成功: {cache_key[:8]}...")
                except Exception as e:
                    logger.warning(f"缓存预热失败: {e}")

        logger.info(f"缓存预热完成，共预热 {warmed_count} 个缓存项")
        return warmed_count

    async def _dynamic_content_analysis(self, input_text: str) -> dict[str, Any]:
        """动态内容分析（基于 LangChain 最佳实践）"""
        # 使用缓存的内容分析
        basic_analysis = self._cached_content_analysis(input_text)

        # 深度分析
        depth_analysis = self._analyze_complexity_depth(input_text)

        # 紧急程度评估
        urgency = self._assess_urgency(input_text)

        # 上下文感知分析
        context_analysis = await self._context_aware_analysis(input_text)

        return {
            **basic_analysis,
            **depth_analysis,
            "urgency": urgency,
            **context_analysis,
            "analysis_timestamp": time.time(),
            "analysis_version": "1.0"
        }

    def _analyze_complexity_depth(self, input_text: str) -> dict[str, Any]:
        """分析复杂度深度"""
        word_count = len(input_text.split())
        line_count = len(input_text.split('\n'))
        technical_terms = self._extract_technical_terms(input_text)

        # 复杂度计算（基于多维度）
        complexity_score = min(1.0, (
            word_count * 0.001 +  # 长度因素
            line_count * 0.05 +    # 结构复杂度
            len(technical_terms) * 0.1 +  # 技术术语
            (1 if '?' in input_text else 0) * 0.2  # 问题复杂度
        ))

        # 深度级别
        if complexity_score > 0.8:
            depth_level = "very_high"
        elif complexity_score > 0.6:
            depth_level = "high"
        elif complexity_score > 0.4:
            depth_level = "medium"
        elif complexity_score > 0.2:
            depth_level = "low"
        else:
            depth_level = "very_low"

        return {
            "complexity_score": complexity_score,
            "depth_level": depth_level,
            "word_count": word_count,
            "line_count": line_count,
            "technical_terms_count": len(technical_terms),
            "technical_terms": technical_terms
        }

    def _assess_urgency(self, input_text: str) -> dict[str, Any]:
        """评估紧急程度"""
        urgency_keywords = [
            "紧急", "urgent", "立即", "马上", "现在", " ASAP ",
            "crucial", "critical", "important", "必须", "需要"
        ]

        urgency_score = sum(1 for keyword in urgency_keywords
                          if keyword.lower() in input_text.lower())

        if urgency_score >= 3:
            urgency_level = "critical"
            urgency_weight = 1.5
        elif urgency_score >= 2:
            urgency_level = "high"
            urgency_weight = 1.3
        elif urgency_score >= 1:
            urgency_level = "medium"
            urgency_weight = 1.1
        else:
            urgency_level = "low"
            urgency_weight = 1.0

        return {
            "urgency_score": urgency_score,
            "urgency_level": urgency_level,
            "urgency_weight": urgency_weight
        }

    async def _context_aware_analysis(self, input_text: str) -> dict[str, Any]:
        """上下文感知分析"""
        # 分析是否需要多步骤处理
        multi_step_indicators = [
            "步骤", "step", "首先", "然后", "最后", "流程",
            "过程", "procedure", "method", "approach"
        ]

        is_multi_step = any(indicator in input_text.lower()
                          for indicator in multi_step_indicators)

        # 分析是否需要精确性
        precision_indicators = [
            "精确", "准确", "详细", "具体", "exact", "precise",
            "详细说明", "具体步骤", "准确答案"
        ]

        requires_precision = any(indicator in input_text.lower()
                               for indicator in precision_indicators)

        return {
            "requires_multi_step": is_multi_step,
            "requires_precision": requires_precision,
            "context_flags": {
                "multi_step": is_multi_step,
                "precision": requires_precision
            }
        }

    def _extract_technical_terms(self, text: str) -> list[str]:
        """提取技术术语"""
        tech_keywords = [
            "API", "database", "algorithm", "function", "class",
            "method", "variable", "parameter", "query", "request",
            "response", "error", "exception", "debug", "test",
            "deploy", "server", "client", "network", "security"
        ]

        found_terms = []
        text_lower = text.lower()
        for term in tech_keywords:
            if term.lower() in text_lower:
                found_terms.append(term)

        return list(set(found_terms))  # 去重

    def _apply_dynamic_health_weights(self, model_scores: dict[str, float]) -> dict[str, float]:
        """应用动态健康权重"""
        weighted_scores = {}

        for model_name, score in model_scores.items():
            # 获取实时健康状态
            health_multiplier = self._calculate_dynamic_health_multiplier(model_name)

            # 动态调整权重
            weighted_score = score * health_multiplier
            weighted_scores[model_name] = weighted_score

            logger.debug(f"Model {model_name}: dynamic_health_multiplier={health_multiplier:.3f}, weighted_score={weighted_score:.3f}")

        return weighted_scores

    def _calculate_dynamic_health_multiplier(self, model_name: str) -> float:
        """计算动态健康乘数"""
        if not self.health_checker:
            return 1.0

        health = self.health_checker.health_status.get(model_name)
        if not health:
            return 1.0

        # 基于多个健康指标计算动态乘数
        success_rate = health.get("success_rate", 1.0)
        response_time = health.get("response_time", 0.0)
        consecutive_failures = health.get("metadata", {}).get("consecutive_failures", 0)
        total_checks = health.get("metadata", {}).get("check_count", 1)

        # 成功率权重（指数衰减）
        success_weight = success_rate

        # 响应时间权重（反向指数）
        response_weight = max(0.1, 1.0 - (response_time / 15.0))

        # 连续失败惩罚
        failure_penalty = 0.1 ** consecutive_failures if consecutive_failures > 0 else 1.0

        # 检查次数置信度
        confidence_factor = min(1.0, total_checks / 5.0)

        # 综合健康乘数
        health_multiplier = (
            success_weight * 0.4 +
            response_weight * 0.3 +
            confidence_factor * 0.2
        ) * failure_penalty

        # 如果模型不健康，应用额外惩罚
        if not health.get("is_healthy", True):
            health_multiplier *= 0.05

        return health_multiplier

    def _apply_cost_optimization(self, model_scores: dict[str, float], content_analysis: dict[str, Any]) -> dict[str, float]:
        """应用成本优化"""
        optimized_scores = {}

        # 获取内容的预算敏感度
        budget_sensitivity = self._assess_budget_sensitivity(content_analysis)

        for model_name, score in model_scores.items():
            # 获取模型成本
            cost_per_token = self.model_configs[model_name].get("cost_per_token", 0.01)

            # 成本权重调整
            if budget_sensitivity == "high":
                # 高预算敏感度，大幅降低高成本模型分数
                cost_weight = max(0.1, 1.0 - (cost_per_token / 0.05))
            elif budget_sensitivity == "medium":
                # 中等预算敏感度，适度调整
                cost_weight = max(0.3, 1.0 - (cost_per_token / 0.1))
            else:
                # 低预算敏感度，轻微调整
                cost_weight = max(0.7, 1.0 - (cost_per_token / 0.2))

            # 应用成本优化
            optimized_score = score * cost_weight
            optimized_scores[model_name] = optimized_score

            logger.debug(f"Model {model_name}: cost_weight={cost_weight:.3f}, optimized_score={optimized_score:.3f}")

        return optimized_scores

    def _assess_budget_sensitivity(self, content_analysis: dict[str, Any]) -> str:
        """评估预算敏感度"""
        content_type = content_analysis.get("content_type", "")
        urgency_level = content_analysis.get("urgency", {}).get("urgency_level", "low")

        # 简单查询通常是预算敏感的
        if content_type == "simple_query":
            return "high"

        # 高紧急程度降低预算敏感度
        if urgency_level in ["critical", "high"]:
            return "low"

        # 一般内容为中等敏感度
        return "medium"

    def _apply_load_balancing(self, model_scores: dict[str, float]) -> dict[str, float]:
        """应用负载均衡"""
        balanced_scores = {}

        # 计算模型使用率
        model_usage = self._calculate_model_usage()

        for model_name, score in model_scores.items():
            # 获取模型使用率
            usage_rate = model_usage.get(model_name, 0.0)

            # 负载均衡权重（使用率越高，权重越低）
            if usage_rate > 0.8:
                load_weight = 0.6  # 高负载，大幅降低分数
            elif usage_rate > 0.6:
                load_weight = 0.8  # 中等负载，适度降低
            elif usage_rate > 0.4:
                load_weight = 0.9  # 低负载，轻微降低
            else:
                load_weight = 1.0  # 未使用，保持原分

            # 应用负载均衡
            balanced_score = score * load_weight
            balanced_scores[model_name] = balanced_score

            logger.debug(f"Model {model_name}: usage_rate={usage_rate:.3f}, load_weight={load_weight:.3f}, balanced_score={balanced_score:.3f}")

        return balanced_scores

    def _calculate_model_usage(self) -> dict[str, float]:
        """计算模型使用率"""
        usage_rates = {}
        total_requests = sum(m.get("total_requests", 0) for m in self.performance_metrics.values())

        if total_requests == 0:
            # 如果没有请求，所有模型使用率为0
            for model_name in self.models:
                usage_rates[model_name] = 0.0
        else:
            for model_name, metrics in self.performance_metrics.items():
                request_count = metrics.get("total_requests", 0)
                usage_rates[model_name] = request_count / total_requests

        return usage_rates

    async def _intelligent_fallback_routing(
        self,
        input_text: str,
        content_analysis: dict[str, Any],
        current_scores: dict[str, float],
        model_availability: dict[str, bool]
    ) -> tuple[str, float]:
        """智能故障转移路由"""
        logger.warning("主路由失败，启用智能故障转移")

        # 获取可用模型列表
        available_models = [name for name, available in model_availability.items() if available]

        if not available_models:
            # 所有模型都不可用，返回错误
            raise Exception("所有模型都不可用")

        # 根据内容类型选择备用策略
        content_type = content_analysis.get("content_type", "general_conversation")

        # 定义降级策略
        fallback_strategies = {
            "complex_reasoning": ["openai-gpt35", "claude-3-sonnet", "ollama-mistral"],
            "creative_writing": ["openai-gpt35", "claude-3-sonnet", "ollama-mistral"],
            "code_analysis": ["ollama-mistral", "openai-gpt35", "claude-3-sonnet"],
            "text_processing": ["openai-gpt35", "ollama-llama2", "ollama-mistral"],
            "simple_query": ["ollama-llama2", "openai-gpt35", "ollama-mistral"],
            "general_conversation": ["openai-gpt35", "ollama-llama2", "claude-3-sonnet"]
        }

        # 获取降级策略
        strategy = fallback_strategies.get(content_type, fallback_strategies["general_conversation"])

        # 选择第一个可用的降级模型
        for model_name in strategy:
            if model_name in available_models:
                logger.info(f"故障转移选择模型: {model_name}")
                return (model_name, current_scores.get(model_name, 0.5))

        # 如果策略中没有可用模型，选择分数最高的可用模型
        available_scores = {name: score for name, score in current_scores.items() if name in available_models}
        if available_scores:
            best_fallback = max(available_scores.items(), key=lambda x: x[1])
            logger.info(f"策略外故障转移选择模型: {best_fallback[0]}")
            return (best_fallback[0], best_fallback[1])

        # 最后的降级选择
        fallback_model = available_models[0]
        logger.warning(f"最后降级选择模型: {fallback_model}")
        return (fallback_model, 0.3)

    async def _get_fallback_model(self) -> str:
        """获取降级模型"""
        # 检查是否有健康检查器
        if self.health_checker:
            healthy_models = [
                name for name, health in self.health_checker.health_status.items()
                if health.get("is_healthy", True)
            ]
            if healthy_models:
                # 返回最简单的模型
                simple_models = ["ollama-llama2", "openai-gpt35", "ollama-mistral"]
                for model in simple_models:
                    if model in healthy_models:
                        return model
                return healthy_models[0]

        # 没有健康检查器时的降级逻辑
        preferred_models = ["openai-gpt35", "ollama-llama2", "ollama-mistral"]
        for model in preferred_models:
            if model in self.models:
                return model

        # 最后的降级
        return list(self.models.keys())[0]

    def _generate_detailed_reason(
        self,
        content_analysis: dict[str, Any],
        selected_model: str,
        confidence: float,
        all_scores: dict[str, float]
    ) -> str:
        """生成详细的决策理由"""
        content_type = content_analysis.get("content_type", "unknown")
        complexity = content_analysis.get("complexity_score", 0)
        urgency = content_analysis.get("urgency", {}).get("urgency_level", "low")

        # 获取模型信息
        model_config = self.model_configs.get(selected_model, {})
        capabilities = model_config.get("capabilities", [])

        # 分析为什么选择这个模型
        reasons = []

        # 内容匹配原因
        if content_type in capabilities:
            reasons.append(f"内容类型匹配 ({content_type})")
        else:
            reasons.append(f"内容类型适配 ({content_type} → {capabilities})")

        # 复杂度适配原因
        if complexity > 0.7:
            reasons.append("高复杂度任务")
        elif complexity < 0.3:
            reasons.append("简单查询优化")
        else:
            reasons.append("中等复杂度")

        # 紧急程度原因
        if urgency in ["critical", "high"]:
            reasons.append(f"高紧急度 ({urgency})")

        # 性能原因
        if self.health_checker:
            health = self.health_checker.health_status.get(selected_model, {})
            success_rate = health.get("success_rate", 0)
            if success_rate > 0.9:
                reasons.append("高成功率")
            elif success_rate > 0.8:
                reasons.append("稳定性能")

        # 成本原因
        cost_per_token = model_config.get("cost_per_token", 0)
        if cost_per_token < 0.01:
            reasons.append("成本优化")
        elif cost_per_token > 0.02:
            reasons.append("高质量优先")

        # 综合理由
        detailed_reason = f"选择 {selected_model} 模型，因为：{', '.join(reasons)}。"
        detailed_reason += f" 内容类型：{content_type}，复杂度：{complexity:.2f}，紧急度：{urgency}。"
        detailed_reason += f" 模型置信度：{confidence:.2%}"

        return detailed_reason

    def _analyze_content(self, input_text: str) -> dict[str, Any]:
        """内容分析"""
        input_lower = input_text.lower()

        # 检测内容类型 - 按优先级顺序检查，避免误判
        # 1. 首先检查明确的代码相关关键词（需要更精确的匹配）
        code_keywords = ["代码", "code", "编程", "program", "编程", "debug", "调试", "开发", "development"]
        if any(keyword in input_lower for keyword in code_keywords):
            # 额外检查是否真的是代码相关（避免"什么是Python?"被误判）
            # 扩展代码相关检测词汇，包括"分析"、"这段"等上下文词汇
            code_context_terms = [
                "编程语言", "programming language", "写代码", "开发", "编程", "coding",
                "分析", "这段", "函数", "function", "类", "class", "方法", "method",
                "变量", "variable", "算法", "algorithm", "数据结构", "data structure"
            ]
            if any(term in input_lower for term in code_context_terms):
                content_type = "code_analysis"
                reason = "检测到代码相关内容，选择代码分析模型"
            else:
                # 可能是简单的概念询问，降级为简单查询
                content_type = "simple_query"
                reason = "检测到技术概念询问，选择简单查询模型"
        # 2. 检查复杂推理相关内容
        elif any(keyword in input_lower for keyword in ["数学", "math", "计算", "calculation", "逻辑", "logic", "推理", "reasoning", "算法", "algorithm", "证明", "proof"]):
            content_type = "complex_reasoning"
            reason = "检测到复杂推理内容，选择高质量模型"
        # 3. 检查创意写作相关内容
        elif any(keyword in input_lower for keyword in ["写作", "write", "创作", "creative", "故事", "story", "文案", "copywriting", "写诗", "作曲"]):
            content_type = "creative_writing"
            reason = "检测到创意写作内容，选择创意能力强的模型"
        # 4. 检查文本处理相关内容
        elif any(keyword in input_lower for keyword in ["文档", "document", "text", "分析", "analyze", "总结", "summarize", "翻译", "translate", "提取", "extract"]):
            content_type = "text_processing"
            reason = "检测到文本处理内容，选择文本处理模型"
        # 5. 检查简单查询相关内容
        elif any(keyword in input_lower for keyword in ["简单", "simple", "快速", "quick", "基本", "basic", "什么是", "什么是", "what is", "how to", "如何", "怎么"]):
            content_type = "simple_query"
            reason = "检测到简单查询，选择快速响应模型"
        else:
            content_type = "general_conversation"
            reason = "通用对话内容，选择平衡性能模型"

        return {
            "content_type": content_type,
            "reason": reason,
            "input_length": len(input_text),
            "complexity_score": self._estimate_complexity(input_text)
        }

    def _estimate_complexity(self, input_text: str) -> float:
        """估算内容复杂度"""
        length = len(input_text)
        lines = len(input_text.split('\n'))
        technical_words = len([w for w in input_text.split() if any(c in w for c in ['api', 'database', 'algorithm', 'function'])])

        # 复杂度评分 (0-1)
        complexity = min(1.0, (length * 0.001 + lines * 0.1 + technical_words * 0.05))
        return complexity

    def _score_models(self, content_analysis: dict[str, Any], model_availability: dict[str, bool] | None = None) -> dict[str, float]:
        """为模型评分（增强版，支持模型可用性检查）"""
        content_type = content_analysis["content_type"]
        complexity = content_analysis["complexity_score"]

        scores = {}

        for model_name, config in self.model_configs.items():
            # 检查模型可用性
            if model_availability and not model_availability.get(model_name, True):
                scores[model_name] = 0.0  # 不可用的模型得分为0
                continue

            model_capabilities = set(config["capabilities"])

            # 基础能力匹配分
            capability_score = 0.0

            if content_type == "code_analysis":
                if "code_analysis" in model_capabilities:
                    capability_score = 0.9
                elif "technical_tasks" in model_capabilities:
                    capability_score = 0.7
                else:
                    capability_score = 0.3

            elif content_type == "complex_reasoning":
                if "complex_reasoning" in model_capabilities:
                    capability_score = 0.9
                elif "high_quality" in model_capabilities:
                    capability_score = 0.8
                else:
                    capability_score = 0.4

            elif content_type == "creative_writing":
                if "creative_writing" in model_capabilities:
                    capability_score = 0.9
                elif "high_quality" in model_capabilities:
                    capability_score = 0.7
                else:
                    capability_score = 0.3

            elif content_type == "text_processing":
                if "text_processing" in model_capabilities:
                    capability_score = 0.8
                elif "general_conversation" in model_capabilities:
                    capability_score = 0.6
                else:
                    capability_score = 0.4

            elif content_type == "simple_query":
                if "fast_response" in model_capabilities:
                    capability_score = 0.9
                elif "low_cost" in model_capabilities:
                    capability_score = 0.8
                else:
                    capability_score = 0.5

            else:  # general_conversation
                if "general_conversation" in model_capabilities:
                    capability_score = 0.8
                elif "balanced_performance" in model_capabilities:
                    capability_score = 0.7
                else:
                    capability_score = 0.5

            # 复杂度调整
            if complexity > 0.7:  # 高复杂度
                if "high_quality" in model_capabilities or "complex_reasoning" in model_capabilities:
                    capability_score *= 1.2
            elif complexity < 0.3:  # 低复杂度
                if "fast_response" in model_capabilities or "low_cost" in model_capabilities:
                    capability_score *= 1.1

            # 健康状态调整（如果有健康检查器）
            if self.health_checker and model_name in self.health_checker.health_status:
                health = self.health_checker.health_status[model_name]
                health_multiplier = health.get("success_rate", 1.0)
                capability_score *= health_multiplier

            scores[model_name] = capability_score

        return scores

    def _apply_performance_weights(self, model_scores: dict[str, float]) -> dict[str, float]:
        """应用性能权重"""
        weighted_scores = {}

        for model_name, base_score in model_scores.items():
            metrics = self.performance_metrics[model_name]

            # 性能权重
            success_rate_weight = metrics["success_rate"]
            response_time_weight = max(0.1, 1.0 - (metrics["avg_response_time"] / 10.0))  # 响应时间越长权重越低
            cost_weight = max(0.1, 1.0 - (self.model_configs[model_name]["cost_per_token"] / 0.05))  # 成本越高质量权越低

            # 综合权重
            performance_weight = (success_rate_weight * 0.5 + response_time_weight * 0.3 + cost_weight * 0.2)

            # 应用权重
            weighted_score = base_score * performance_weight
            weighted_scores[model_name] = weighted_score

            logger.debug(f"Model {model_name}: base_score={base_score:.3f}, performance_weight={performance_weight:.3f}, weighted_score={weighted_score:.3f}")

        return weighted_scores

    def _calculate_confidence(self, weighted_scores: dict[str, float], best_score: float) -> float:
        """计算置信度"""
        if len(weighted_scores) == 1:
            return 0.9

        # 获取第二高的分数
        sorted_scores = sorted(weighted_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            second_best_score = sorted_scores[1]
            # 置信度基于分数差距
            score_gap = best_score - second_best_score
            confidence = min(0.95, 0.5 + score_gap)
        else:
            confidence = 0.7

        return confidence

