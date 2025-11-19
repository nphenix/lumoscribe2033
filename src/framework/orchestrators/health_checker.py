"""
LLM 模型健康检查器

基于 LangChain 1.0 最佳实践的模型健康检查和监控系统。
提供实时健康检查、故障检测、自动恢复和智能故障转移功能。
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ..shared.config import config_manager
from ..shared.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(TypedDict):
    """健康状态数据结构"""
    model_name: str
    is_healthy: bool
    status_code: str
    response_time: float
    success_rate: float
    error_count: int
    last_check: float
    failure_reason: str | None
    recovery_attempts: int
    can_recover: bool
    metadata: dict[str, Any]


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    timeout: float = 30.0
    max_retries: int = 3
    check_interval: float = 60.0
    failure_threshold: int = 5
    recovery_timeout: float = 300.0
    enable_auto_recovery: bool = True
    enable_tracing: bool = True


class LangChainHealthChecker:
    """基于 LangChain 1.0 模式的模型健康检查器"""

    def __init__(
        self,
        models: dict[str, BaseChatModel],
        config: HealthCheckConfig | None = None,
        callback_handler: AsyncCallbackHandler | None = None
    ):
        """
        初始化健康检查器

        Args:
            models: 模型字典
            config: 健康检查配置
            callback_handler: LangChain 回调处理器
        """
        self.models = models
        self.config = config or HealthCheckConfig()
        self.callback_handler = callback_handler
        self.health_status: dict[str, HealthStatus] = {}
        self._check_tasks: dict[str, asyncio.Task] = {}
        self._is_running = False
        self._check_counter = 0

        # 初始化健康状态
        for model_name in self.models:
            self.health_status[model_name] = self._create_initial_health_status(model_name)

        logger.info(f"LangChainHealthChecker initialized with {len(models)} models")

    def _create_initial_health_status(self, model_name: str) -> HealthStatus:
        """创建初始健康状态"""
        return HealthStatus(
            model_name=model_name,
            is_healthy=True,
            status_code="initial",
            response_time=0.0,
            success_rate=1.0,
            error_count=0,
            last_check=0.0,
            failure_reason=None,
            recovery_attempts=0,
            can_recover=True,
            metadata={
                "created_at": time.time(),
                "check_count": 0,
                "total_response_time": 0.0,
                "consecutive_failures": 0,
                "last_recovery_attempt": 0.0
            }
        )

    async def start_monitoring(self, runnable_config: RunnableConfig | None = None):
        """开始监控所有模型"""
        if self._is_running:
            logger.warning("Health monitoring is already running")
            return

        self._is_running = True
        logger.info("Starting LangChain model health monitoring")

        # 为每个模型启动健康检查任务
        for model_name in self.models:
            self._check_tasks[model_name] = asyncio.create_task(
                self._monitor_model(model_name, runnable_config)
            )

    async def stop_monitoring(self):
        """停止监控"""
        if not self._is_running:
            return

        self._is_running = False
        logger.info("Stopping model health monitoring")

        # 取消所有检查任务
        for task in self._check_tasks.values():
            task.cancel()

        # 等待任务完成
        await asyncio.gather(*self._check_tasks.values(), return_exceptions=True)
        self._check_tasks.clear()

    async def _monitor_model(self, model_name: str, runnable_config: RunnableConfig | None):
        """监控单个模型"""
        while self._is_running:
            try:
                # 执行健康检查
                health = await self.check_model_health(model_name, runnable_config)
                self.health_status[model_name] = health

                # 检查是否需要故障转移
                if not health["is_healthy"]:
                    await self._handle_model_failure(model_name, health, runnable_config)

                # 等待下次检查
                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring model {model_name}: {e}")
                await asyncio.sleep(self.config.check_interval)

    async def check_model_health(
        self,
        model_name: str,
        runnable_config: RunnableConfig | None = None
    ) -> HealthStatus:
        """
        检查模型健康状态

        Args:
            model_name: 模型名称
            runnable_config: LangChain 运行配置

        Returns:
            健康状态信息
        """
        start_time = time.time()
        self._check_counter += 1
        check_id = self._check_counter

        try:
            model = self.models[model_name]

            # 执行健康检查请求
            await self._perform_health_check(
                model, model_name, check_id, runnable_config
            )

            response_time = time.time() - start_time

            # 更新健康状态
            current_health = self.health_status.get(model_name)
            if current_health:
                # 更新现有状态
                current_health.update({
                    "is_healthy": True,
                    "status_code": "healthy",
                    "response_time": response_time,
                    "success_rate": self._calculate_success_rate(current_health, True),
                    "error_count": 0,  # 重置错误计数
                    "last_check": start_time,
                    "failure_reason": None,
                    "can_recover": True,
                    "metadata": {
                        **current_health["metadata"],
                        "check_count": current_health["metadata"].get("check_count", 0) + 1,
                        "total_response_time": current_health["metadata"].get("total_response_time", 0) + response_time,
                        "consecutive_failures": 0,  # 重置连续失败次数
                        "last_success": start_time
                    }
                })
                health_status = current_health
            else:
                # 创建新状态
                health_status = HealthStatus(
                    model_name=model_name,
                    is_healthy=True,
                    status_code="healthy",
                    response_time=response_time,
                    success_rate=1.0,
                    error_count=0,
                    last_check=start_time,
                    failure_reason=None,
                    recovery_attempts=0,
                    can_recover=True,
                    metadata={
                        "check_count": 1,
                        "total_response_time": response_time,
                        "consecutive_failures": 0,
                        "last_success": start_time
                    }
                )
                self.health_status[model_name] = health_status

            logger.debug(f"Model {model_name} health check passed ({response_time:.2f}s)")
            return health_status

        except Exception as e:
            response_time = time.time() - start_time

            # 更新失败状态
            current_health = self.health_status.get(model_name)
            if current_health:
                consecutive_failures = current_health["metadata"].get("consecutive_failures", 0) + 1
                error_count = current_health["error_count"] + 1
                can_recover = self._can_model_recover(model_name, error_count)

                current_health.update({
                    "is_healthy": False,
                    "status_code": "unhealthy",
                    "response_time": response_time,
                    "success_rate": self._calculate_success_rate(current_health, False),
                    "error_count": error_count,
                    "last_check": start_time,
                    "failure_reason": str(e),
                    "can_recover": can_recover,
                    "metadata": {
                        **current_health["metadata"],
                        "consecutive_failures": consecutive_failures,
                        "last_failure": start_time,
                        "failure_reason": str(e)
                    }
                })
                health_status = current_health
            else:
                health_status = HealthStatus(
                    model_name=model_name,
                    is_healthy=False,
                    status_code="unhealthy",
                    response_time=response_time,
                    success_rate=0.0,
                    error_count=1,
                    last_check=start_time,
                    failure_reason=str(e),
                    recovery_attempts=0,
                    can_recover=True,
                    metadata={
                        "check_count": 1,
                        "consecutive_failures": 1,
                        "last_failure": start_time,
                        "failure_reason": str(e)
                    }
                )
                self.health_status[model_name] = health_status

            logger.warning(f"Model {model_name} health check failed: {e} (error_count: {health_status['error_count']})")
            return health_status

    async def _perform_health_check(
        self,
        model: BaseChatModel,
        model_name: str,
        check_id: int,
        runnable_config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """执行实际的健康检查"""
        try:
            # 健康检查提示词 - 使用更简洁的提示以避免token限制
            health_check_prompt = [
                HumanMessage(content="OK?")
            ]

            # 设置超时
            async with asyncio.timeout(self.config.timeout):
                # 执行模型调用，可选地使用 LangChain 配置
                if runnable_config and self.config.enable_tracing:
                    response = await model.ainvoke(health_check_prompt, config=runnable_config)
                else:
                    response = await model.ainvoke(health_check_prompt)

                # 验证响应
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)

                # 更灵活的响应验证 - 检查是否包含任何积极响应
                positive_indicators = ["ok", "hello", "hi", "working", "ready", "available", "assistant", "ai", "kwaipilot"]
                if not any(indicator in content.lower() for indicator in positive_indicators):
                    raise Exception(f"Model response validation failed: {content[:100]}")

                return {
                    "status": "ok",
                    "response": content,
                    "check_id": check_id,
                    "model_name": model_name
                }

        except TimeoutError:
            raise Exception(f"Model health check timeout after {self.config.timeout}s")
        except Exception as e:
            raise Exception(f"Model health check failed: {str(e)}")

    def _calculate_success_rate(self, current_health: HealthStatus, is_success: bool) -> float:
        """计算成功率（指数加权移动平均）"""
        current_success_rate = current_health.get("success_rate", 1.0)
        alpha = 0.1  # 平滑因子

        if is_success:
            return current_success_rate * (1 - alpha) + alpha
        else:
            return current_success_rate * (1 - alpha)

    def _can_model_recover(self, model_name: str, error_count: int) -> bool:
        """判断模型是否可以恢复"""
        # 检查配置管理器中的恢复策略
        model_config = config_manager.get_model_by_name(model_name)
        if model_config and hasattr(model_config, 'recovery_strategy'):
            recovery_strategy = model_config.recovery_strategy
        else:
            recovery_strategy = "auto"

        # 基于错误次数和策略判断
        if error_count >= self.config.failure_threshold:
            return False
        if recovery_strategy == "none":
            return False
        if recovery_strategy == "auto":
            return True

        return True

    async def _handle_model_failure(
        self,
        model_name: str,
        health: HealthStatus,
        runnable_config: RunnableConfig | None = None
    ):
        """处理模型故障"""
        logger.error(f"Model {model_name} failed health check: {health['failure_reason']}")

        # 触发故障转移
        await self._trigger_failover(model_name, health, runnable_config)

        # 尝试自动恢复（如果启用且模型可以恢复）
        if (self.config.enable_auto_recovery and
            health["can_recover"] and
            health["metadata"].get("consecutive_failures", 0) <= 3):
            await self._attempt_recovery(model_name, runnable_config)

    async def _trigger_failover(
        self,
        model_name: str,
        health: HealthStatus,
        runnable_config: RunnableConfig | None = None
    ):
        """触发故障转移"""
        logger.info(f"Triggering failover for model {model_name}")

        # 通知路由系统该模型不可用
        # 可以集成到事件总线或消息队列中
        if runnable_config:
            # 使用 LangChain 追踪记录故障转移事件
            if hasattr(runnable_config, 'callbacks') and runnable_config.callbacks:
                for callback in runnable_config.callbacks:
                    if hasattr(callback, 'on_chain_start'):
                        try:
                            await callback.on_chain_start(
                                serialized={"name": "failover_handler"},
                                inputs={"model_name": model_name, "health": health},
                                run_id=model_name
                            )
                        except Exception:
                            pass

    async def _attempt_recovery(
        self,
        model_name: str,
        runnable_config: RunnableConfig | None = None
    ):
        """尝试自动恢复"""
        logger.info(f"Attempting recovery for model {model_name}")

        # 记录恢复尝试时间
        health = self.health_status.get(model_name)
        if health:
            health["recovery_attempts"] += 1
            health["metadata"]["last_recovery_attempt"] = time.time()

        # 简单的恢复策略：等待一段时间后重新检查
        await asyncio.sleep(10)

        # 重新检查健康状态
        recovery_health = await self.check_model_health(model_name, runnable_config)
        if recovery_health["is_healthy"]:
            logger.info(f"Model {model_name} recovered successfully after {recovery_health['recovery_attempts']} attempts")
        else:
            logger.error(f"Model {model_name} recovery failed")

    def get_health_summary(self) -> dict[str, Any]:
        """获取健康状态摘要"""
        total_models = len(self.health_status)
        healthy_models = sum(1 for h in self.health_status.values() if h["is_healthy"])
        unhealthy_models = total_models - healthy_models

        # 计算平均响应时间
        total_response_time = sum(h["response_time"] for h in self.health_status.values())
        avg_response_time = total_response_time / total_models if total_models > 0 else 0

        return {
            "total_models": total_models,
            "healthy_models": healthy_models,
            "unhealthy_models": unhealthy_models,
            "health_rate": healthy_models / total_models if total_models > 0 else 0,
            "avg_response_time": avg_response_time,
            "models": self.health_status,
            "monitoring": {
                "is_running": self._is_running,
                "total_checks": self._check_counter,
                "check_interval": self.config.check_interval
            }
        }

    def is_model_healthy(self, model_name: str) -> bool:
        """检查模型是否健康"""
        health = self.health_status.get(model_name)
        return health["is_healthy"] if health else False

    async def force_health_check(
        self,
        model_name: str,
        runnable_config: RunnableConfig | None = None
    ) -> HealthStatus:
        """强制执行健康检查"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        health = await self.check_model_health(model_name, runnable_config)
        self.health_status[model_name] = health
        return health

    def get_unhealthy_models(self) -> list[str]:
        """获取不健康模型列表"""
        return [name for name, health in self.health_status.items() if not health["is_healthy"]]

    def get_model_metrics(self, model_name: str) -> dict[str, Any]:
        """获取模型详细指标"""
        health = self.health_status.get(model_name)
        if not health:
            return {}

        metadata = health["metadata"]
        avg_response_time = (
            metadata.get("total_response_time", 0) /
            max(metadata.get("check_count", 1), 1)
        )

        return {
            "model_name": model_name,
            "is_healthy": health["is_healthy"],
            "success_rate": health["success_rate"],
            "avg_response_time": avg_response_time,
            "error_count": health["error_count"],
            "consecutive_failures": metadata.get("consecutive_failures", 0),
            "uptime_ratio": health["success_rate"],  # 简单的正常运行时间计算
            "last_check": health["last_check"],
            "total_checks": metadata.get("check_count", 0)
        }
