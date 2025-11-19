"""
系统监控和可观测性

提供详细的性能监控、指标收集和可观测性功能
"""

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import psutil

from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    tags: dict[str, str]
    metadata: dict[str, Any]


@dataclass
class SystemResourceMetric:
    """系统资源指标"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_sent_bytes: int
    network_recv_bytes: int


@dataclass
class TaskMetric:
    """任务执行指标"""
    timestamp: str
    task_name: str
    execution_time: float
    status: str  # success, failed, running
    queue_time: float
    worker_id: str
    retries: int


@dataclass
class ApiMetric:
    """API 请求指标"""
    timestamp: str
    endpoint: str
    method: str
    status_code: int
    response_time: float
    request_size: int
    response_size: int
    client_ip: str


class MetricsCollector:
    """指标收集器"""

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.performance_metrics: deque = deque(maxlen=max_metrics)
        self.system_metrics: deque = deque(maxlen=1000)
        self.task_metrics: deque = deque(maxlen=max_metrics)
        self.api_metrics: deque = deque(maxlen=max_metrics)

        self.system_stats = SystemStats()
        self.task_stats = TaskStats()
        self.api_stats = ApiStats()

        self._collectors: list[Callable] = []
        self._is_running = False
        self._monitoring_thread: threading.Thread | None = None

    def start(self, interval: int = 60):
        """启动监控收集"""
        if self._is_running:
            return

        self._is_running = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("📊 指标收集器已启动")

    def stop(self):
        """停止监控收集"""
        self._is_running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("📊 指标收集器已停止")

    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self._is_running:
            try:
                self._collect_system_metrics()
                self._collect_custom_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"监控收集错误: {e}")
                time.sleep(5)  # 错误后短暂等待

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU 和内存使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # 磁盘使用率
            disk = psutil.disk_usage('/')

            # 网络统计
            network = psutil.net_io_counters()

            metric = SystemResourceMetric(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_sent_bytes=network.bytes_sent,
                network_recv_bytes=network.bytes_recv
            )

            self.system_metrics.append(metric)
            self.system_stats.update(metric)

        except Exception as e:
            logger.error(f"系统指标收集失败: {e}")

    def _collect_custom_metrics(self):
        """收集自定义指标"""
        for collector in self._collectors:
            try:
                collector(self)
            except Exception as e:
                logger.error(f"自定义指标收集器错误: {e}")

    def add_collector(self, collector: Callable):
        """添加自定义指标收集器"""
        self._collectors.append(collector)

    def record_task_metric(
        self,
        task_name: str,
        execution_time: float,
        status: str,
        queue_time: float = 0,
        worker_id: str = "",
        retries: int = 0
    ):
        """记录任务执行指标"""
        metric = TaskMetric(
            timestamp=datetime.now().isoformat(),
            task_name=task_name,
            execution_time=execution_time,
            status=status,
            queue_time=queue_time,
            worker_id=worker_id,
            retries=retries
        )

        self.task_metrics.append(metric)
        self.task_stats.update(metric)

    def record_api_metric(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_size: int = 0,
        response_size: int = 0,
        client_ip: str = ""
    ):
        """记录 API 请求指标"""
        metric = ApiMetric(
            timestamp=datetime.now().isoformat(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            request_size=request_size,
            response_size=response_size,
            client_ip=client_ip
        )

        self.api_metrics.append(metric)
        self.api_stats.update(metric)

    def record_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: dict[str, str] = None,
        metadata: dict[str, Any] = None
    ):
        """记录性能指标"""
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {}
        )

        self.performance_metrics.append(metric)

    def get_system_summary(self, hours: int = 24) -> dict[str, Any]:
        """获取系统资源摘要"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.system_metrics
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]

        if not recent_metrics:
            return {}

        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]

        return {
            "time_range": f"最近 {hours} 小时",
            "system_stats": {
                "avg_cpu": sum(cpu_values) / len(cpu_values),
                "max_cpu": max(cpu_values),
                "avg_memory": sum(memory_values) / len(memory_values),
                "max_memory": max(memory_values),
                "total_samples": len(recent_metrics)
            },
            "current_resources": self.system_stats.get_current_stats()
        }

    def get_task_summary(self, hours: int = 24) -> dict[str, Any]:
        """获取任务执行摘要"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.task_metrics
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]

        return self.task_stats.get_summary(recent_metrics)

    def get_api_summary(self, hours: int = 24) -> dict[str, Any]:
        """获取 API 请求摘要"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.api_metrics
            if datetime.fromisoformat(m.timestamp) > cutoff
        ]

        return self.api_stats.get_summary(recent_metrics)

    def get_health_status(self) -> dict[str, Any]:
        """获取系统健康状态"""
        current_time = datetime.now()

        # 检查最近 5 分钟的系统指标
        recent_system_metrics = [
            m for m in self.system_metrics
            if datetime.fromisoformat(m.timestamp) > current_time - timedelta(minutes=5)
        ]

        # 检查最近 5 分钟的任务指标
        recent_task_metrics = [
            m for m in self.task_metrics
            if datetime.fromisoformat(m.timestamp) > current_time - timedelta(minutes=5)
        ]

        health_status = {
            "timestamp": current_time.isoformat(),
            "system_health": "healthy",
            "task_health": "healthy",
            "overall_health": "healthy"
        }

        # 系统健康检查
        if recent_system_metrics:
            latest_system = recent_system_metrics[-1]
            if latest_system.cpu_percent > 90:
                health_status["system_health"] = "critical"
            elif latest_system.memory_percent > 80:
                health_status["system_health"] = "warning"

        # 任务健康检查
        if recent_task_metrics:
            failed_tasks = [m for m in recent_task_metrics if m.status == "failed"]
            if failed_tasks:
                failure_rate = len(failed_tasks) / len(recent_task_metrics)
                if failure_rate > 0.1:  # 失败率超过 10%
                    health_status["task_health"] = "critical"
                elif failure_rate > 0.05:  # 失败率超过 5%
                    health_status["task_health"] = "warning"

        # 综合健康状态
        if health_status["system_health"] == "critical" or health_status["task_health"] == "critical":
            health_status["overall_health"] = "critical"
        elif health_status["system_health"] == "warning" or health_status["task_health"] == "warning":
            health_status["overall_health"] = "warning"

        return health_status

    def export_metrics(self, output_dir: str = "logs/metrics") -> dict[str, str]:
        """导出指标数据到文件"""
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = {}

        # 导出系统指标
        if self.system_metrics:
            system_data = [asdict(m) for m in self.system_metrics]
            system_file = os.path.join(output_dir, f"system_metrics_{current_time}.json")
            with open(system_file, "w", encoding="utf-8") as f:
                json.dump(system_data, f, ensure_ascii=False, indent=2)
            exported_files["system"] = system_file

        # 导出任务指标
        if self.task_metrics:
            task_data = [asdict(m) for m in self.task_metrics]
            task_file = os.path.join(output_dir, f"task_metrics_{current_time}.json")
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
            exported_files["task"] = task_file

        # 导出 API 指标
        if self.api_metrics:
            api_data = [asdict(m) for m in self.api_metrics]
            api_file = os.path.join(output_dir, f"api_metrics_{current_time}.json")
            with open(api_file, "w", encoding="utf-8") as f:
                json.dump(api_data, f, ensure_ascii=False, indent=2)
            exported_files["api"] = api_file

        logger.info(f"📊 指标数据已导出: {exported_files}")
        return exported_files


class SystemStats:
    """系统统计信息"""

    def __init__(self):
        self._cpu_history = deque(maxlen=100)
        self._memory_history = deque(maxlen=100)

    def update(self, metric: SystemResourceMetric):
        """更新系统统计"""
        self._cpu_history.append(metric.cpu_percent)
        self._memory_history.append(metric.memory_percent)

    def get_current_stats(self) -> dict[str, float]:
        """获取当前系统统计"""
        return {
            "cpu_percent": self._cpu_history[-1] if self._cpu_history else 0,
            "memory_percent": self._memory_history[-1] if self._memory_history else 0,
            "cpu_avg_10": sum(list(self._cpu_history)[-10:]) / min(10, len(self._cpu_history)),
            "memory_avg_10": sum(list(self._memory_history)[-10:]) / min(10, len(self._memory_history))
        }


class TaskStats:
    """任务统计信息"""

    def __init__(self):
        self._task_counts = defaultdict(int)
        self._task_times = defaultdict(list)
        self._task_errors = defaultdict(list)

    def update(self, metric: TaskMetric):
        """更新任务统计"""
        self._task_counts[f"{metric.task_name}_{metric.status}"] += 1
        self._task_times[metric.task_name].append(metric.execution_time)
        if metric.status == "failed":
            self._task_errors[metric.task_name].append(metric)

    def get_summary(self, metrics: list[TaskMetric]) -> dict[str, Any]:
        """获取任务摘要"""
        if not metrics:
            return {}

        # 按任务名称分组
        task_groups = defaultdict(list)
        for metric in metrics:
            task_groups[metric.task_name].append(metric)

        summary = {}
        for task_name, task_metrics in task_groups.items():
            total_count = len(task_metrics)
            success_count = len([m for m in task_metrics if m.status == "success"])
            failed_count = len([m for m in task_metrics if m.status == "failed"])

            execution_times = [m.execution_time for m in task_metrics]

            summary[task_name] = {
                "total_count": total_count,
                "success_count": success_count,
                "failed_count": failed_count,
                "success_rate": (success_count / total_count * 100) if total_count > 0 else 0,
                "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "total_execution_time": sum(execution_times)
            }

        return summary


class ApiStats:
    """API 统计信息"""

    def __init__(self):
        self._endpoint_counts = defaultdict(int)
        self._status_counts = defaultdict(int)
        self._response_times = defaultdict(list)

    def update(self, metric: ApiMetric):
        """更新 API 统计"""
        endpoint_key = f"{metric.method} {metric.endpoint}"
        self._endpoint_counts[endpoint_key] += 1
        self._status_counts[str(metric.status_code)] += 1
        self._response_times[endpoint_key].append(metric.response_time)

    def get_summary(self, metrics: list[ApiMetric]) -> dict[str, Any]:
        """获取 API 摘要"""
        if not metrics:
            return {}

        # 按端点分组
        endpoint_groups = defaultdict(list)
        for metric in metrics:
            endpoint_key = f"{metric.method} {metric.endpoint}"
            endpoint_groups[endpoint_key].append(metric)

        summary = {}
        for endpoint, endpoint_metrics in endpoint_groups.items():
            total_count = len(endpoint_metrics)
            status_codes = [m.status_code for m in endpoint_metrics]
            response_times = [m.response_time for m in endpoint_metrics]

            # 状态码统计
            status_counts = defaultdict(int)
            for code in status_codes:
                status_counts[str(code)] += 1

            summary[endpoint] = {
                "total_requests": total_count,
                "status_codes": dict(status_counts),
                "success_rate": (len([c for c in status_codes if 200 <= c < 300]) / total_count * 100) if total_count > 0 else 0,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "total_response_time": sum(response_times)
            }

        return summary


# 全局指标收集器实例
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器实例"""
    return metrics_collector


class EnhancedMetricsCollector:
    """增强的指标收集器，集成 Redis 缓存和性能监控"""

    def __init__(self):
        self.base_collector = metrics_collector
        self.cache_manager = None
        self.performance_metrics = defaultdict(list)
        self.alert_thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 75.0,
            "memory_critical": 85.0,
            "disk_warning": 80.0,
            "disk_critical": 90.0,
            "response_time_warning": 2.0,
            "response_time_critical": 5.0,
            "error_rate_warning": 5.0,
            "error_rate_critical": 10.0
        }
        self.alert_history = deque(maxlen=1000)

    async def initialize(self):
        """初始化增强指标收集器"""
        try:
            from src.framework.shared.redis_cache import get_cache_manager
            self.cache_manager = await get_cache_manager()
            logger.info("✅ 增强指标收集器已初始化")
        except Exception as e:
            logger.error(f"❌ 增强指标收集器初始化失败: {e}")

    async def collect_comprehensive_metrics(self) -> dict[str, Any]:
        """收集综合指标"""
        try:
            # 基础系统指标
            system_metrics = self.base_collector.get_system_summary(hours=1)

            # 缓存指标
            cache_metrics = {}
            if self.cache_manager:
                cache_metrics = self.cache_manager.get_metrics()

            # 应用性能指标
            app_metrics = self._collect_application_metrics()

            # 健康状态
            health_status = self.base_collector.get_health_status()
            cache_health = {}
            if self.cache_manager:
                cache_health = await self.cache_manager.health_check()

            # 警报检查
            alerts = self._check_alerts(system_metrics, cache_metrics)

            # 综合指标
            comprehensive_metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "cache_metrics": cache_metrics,
                "application_metrics": app_metrics,
                "health_status": {
                    "overall": health_status["overall_health"],
                    "system": health_status["system_health"],
                    "tasks": health_status["task_health"],
                    "cache": cache_health.get("status", "unknown")
                },
                "alerts": alerts,
                "performance_trends": self._calculate_performance_trends(),
                "resource_utilization": self._calculate_resource_utilization()
            }

            # 存储到 Redis 缓存
            if self.cache_manager:
                await self.cache_manager.set(
                    "comprehensive_metrics",
                    comprehensive_metrics,
                    ttl=300  # 5分钟缓存
                )

            return comprehensive_metrics

        except Exception as e:
            logger.error(f"❌ 综合指标收集失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _collect_application_metrics(self) -> dict[str, Any]:
        """收集应用级指标"""
        try:
            # 任务执行指标
            task_summary = self.base_collector.get_task_summary(hours=1)

            # API 请求指标
            api_summary = self.base_collector.get_api_summary(hours=1)

            # 性能计数器
            performance_counters = {
                "total_requests": sum(
                    summary.get("total_requests", 0)
                    for summary in api_summary.values()
                ),
                "successful_requests": sum(
                    summary.get("success_rate", 0) * summary.get("total_requests", 0) / 100
                    for summary in api_summary.values()
                ),
                "failed_requests": sum(
                    summary.get("total_requests", 0) * (100 - summary.get("success_rate", 0)) / 100
                    for summary in api_summary.values()
                ),
                "average_response_time": sum(
                    summary.get("avg_response_time", 0)
                    for summary in api_summary.values()
                ) / max(len(api_summary), 1),
                "total_tasks": sum(
                    summary.get("total_count", 0)
                    for summary in task_summary.values()
                ),
                "successful_tasks": sum(
                    summary.get("success_count", 0)
                    for summary in task_summary.values()
                ),
                "failed_tasks": sum(
                    summary.get("failed_count", 0)
                    for summary in task_summary.values()
                )
            }

            return {
                "task_summary": task_summary,
                "api_summary": api_summary,
                "performance_counters": performance_counters
            }

        except Exception as e:
            logger.error(f"❌ 应用指标收集失败: {e}")
            return {}

    def _check_alerts(self, system_metrics: dict, cache_metrics: dict) -> list[dict[str, Any]]:
        """检查警报条件"""
        alerts = []
        current_time = datetime.now()

        try:
            # 系统 CPU 警报
            system_stats = system_metrics.get("system_stats", {})
            if system_stats.get("avg_cpu", 0) > self.alert_thresholds["cpu_critical"]:
                alerts.append({
                    "type": "system",
                    "level": "critical",
                    "metric": "cpu_usage",
                    "value": system_stats.get("avg_cpu"),
                    "threshold": self.alert_thresholds["cpu_critical"],
                    "message": "CPU 使用率过高",
                    "timestamp": current_time.isoformat()
                })
            elif system_stats.get("avg_cpu", 0) > self.alert_thresholds["cpu_warning"]:
                alerts.append({
                    "type": "system",
                    "level": "warning",
                    "metric": "cpu_usage",
                    "value": system_stats.get("avg_cpu"),
                    "threshold": self.alert_thresholds["cpu_warning"],
                    "message": "CPU 使用率较高",
                    "timestamp": current_time.isoformat()
                })

            # 内存警报
            if system_stats.get("avg_memory", 0) > self.alert_thresholds["memory_critical"]:
                alerts.append({
                    "type": "system",
                    "level": "critical",
                    "metric": "memory_usage",
                    "value": system_stats.get("avg_memory"),
                    "threshold": self.alert_thresholds["memory_critical"],
                    "message": "内存使用率过高",
                    "timestamp": current_time.isoformat()
                })
            elif system_stats.get("avg_memory", 0) > self.alert_thresholds["memory_warning"]:
                alerts.append({
                    "type": "system",
                    "level": "warning",
                    "metric": "memory_usage",
                    "value": system_stats.get("avg_memory"),
                    "threshold": self.alert_thresholds["memory_warning"],
                    "message": "内存使用率较高",
                    "timestamp": current_time.isoformat()
                })

            # 缓存警报
            cache_stats = cache_metrics.get("cache_metrics", {})
            if cache_stats.get("hit_rate", 0) < 50:
                alerts.append({
                    "type": "cache",
                    "level": "warning",
                    "metric": "hit_rate",
                    "value": cache_stats.get("hit_rate"),
                    "threshold": 50,
                    "message": "缓存命中率过低",
                    "timestamp": current_time.isoformat()
                })

            # 错误率警报
            operations = cache_metrics.get("operations", {})
            if operations:
                total_ops = operations.get("total", 0)
                cache_errors = cache_stats.get("errors", 0)
                if total_ops > 0:
                    error_rate = (cache_errors / total_ops) * 100
                    if error_rate > self.alert_thresholds["error_rate_critical"]:
                        alerts.append({
                            "type": "cache",
                            "level": "critical",
                            "metric": "error_rate",
                            "value": error_rate,
                            "threshold": self.alert_thresholds["error_rate_critical"],
                            "message": "缓存错误率过高",
                            "timestamp": current_time.isoformat()
                        })
                    elif error_rate > self.alert_thresholds["error_rate_warning"]:
                        alerts.append({
                            "type": "cache",
                            "level": "warning",
                            "metric": "error_rate",
                            "value": error_rate,
                            "threshold": self.alert_thresholds["error_rate_warning"],
                            "message": "缓存错误率较高",
                            "timestamp": current_time.isoformat()
                        })

            # 记录警报历史
            for alert in alerts:
                self.alert_history.append(alert)

            return alerts

        except Exception as e:
            logger.error(f"❌ 警报检查失败: {e}")
            return []

    def _calculate_performance_trends(self) -> dict[str, Any]:
        """计算性能趋势"""
        try:
            # 获取最近的性能数据
            recent_metrics = list(self.base_collector.performance_metrics)[-100:]  # 最近100个数据点

            if len(recent_metrics) < 10:
                return {"trend": "insufficient_data"}

            # CPU 趋势
            cpu_values = [m.value for m in recent_metrics if m.metric_name == "cpu_usage"]
            memory_values = [m.value for m in recent_metrics if m.metric_name == "memory_usage"]

            trends = {}

            # CPU 趋势分析
            if len(cpu_values) >= 10:
                recent_cpu = cpu_values[-10:]
                earlier_cpu = cpu_values[-20:-10] if len(cpu_values) >= 20 else cpu_values[:-10]

                if earlier_cpu:
                    recent_avg = sum(recent_cpu) / len(recent_cpu)
                    earlier_avg = sum(earlier_cpu) / len(earlier_cpu)

                    if recent_avg > earlier_avg * 1.1:  # 10% 增长
                        trends["cpu_trend"] = "increasing"
                    elif recent_avg < earlier_avg * 0.9:  # 10% 减少
                        trends["cpu_trend"] = "decreasing"
                    else:
                        trends["cpu_trend"] = "stable"
                else:
                    trends["cpu_trend"] = "insufficient_data"

            # 内存趋势分析
            if len(memory_values) >= 10:
                recent_mem = memory_values[-10:]
                earlier_mem = memory_values[-20:-10] if len(memory_values) >= 20 else memory_values[:-10]

                if earlier_mem:
                    recent_avg = sum(recent_mem) / len(recent_mem)
                    earlier_avg = sum(earlier_mem) / len(earlier_mem)

                    if recent_avg > earlier_avg * 1.1:
                        trends["memory_trend"] = "increasing"
                    elif recent_avg < earlier_avg * 0.9:
                        trends["memory_trend"] = "decreasing"
                    else:
                        trends["memory_trend"] = "stable"
                else:
                    trends["memory_trend"] = "insufficient_data"

            return trends

        except Exception as e:
            logger.error(f"❌ 性能趋势计算失败: {e}")
            return {"trend": "calculation_error"}

    def _calculate_resource_utilization(self) -> dict[str, Any]:
        """计算资源利用率"""
        try:
            system_summary = self.base_collector.get_system_summary(hours=24)
            system_stats = system_summary.get("system_stats", {})

            if not system_stats:
                return {"error": "no_system_data"}

            # 资源利用率分类
            cpu_util = system_stats.get("avg_cpu", 0)
            memory_util = system_stats.get("avg_memory", 0)

            # 利用率等级
            def get_utilization_level(value, warning_threshold, critical_threshold):
                if value >= critical_threshold:
                    return "critical"
                elif value >= warning_threshold:
                    return "warning"
                elif value >= warning_threshold * 0.7:
                    return "moderate"
                else:
                    return "good"

            utilization = {
                "cpu": {
                    "current": cpu_util,
                    "level": get_utilization_level(
                        cpu_util,
                        self.alert_thresholds["cpu_warning"],
                        self.alert_thresholds["cpu_critical"]
                    ),
                    "status": "healthy" if cpu_util < self.alert_thresholds["cpu_warning"] else "degraded"
                },
                "memory": {
                    "current": memory_util,
                    "level": get_utilization_level(
                        memory_util,
                        self.alert_thresholds["memory_warning"],
                        self.alert_thresholds["memory_critical"]
                    ),
                    "status": "healthy" if memory_util < self.alert_thresholds["memory_warning"] else "degraded"
                }
            }

            # 综合健康状态
            critical_count = sum(
                1 for resource in utilization.values()
                if resource["level"] == "critical"
            )
            warning_count = sum(
                1 for resource in utilization.values()
                if resource["level"] == "warning"
            )

            if critical_count > 0:
                overall_status = "critical"
            elif warning_count > 0 or critical_count > 0:
                overall_status = "warning"
            else:
                overall_status = "healthy"

            utilization["overall"] = {
                "status": overall_status,
                "critical_resources": [name for name, resource in utilization.items() if resource["level"] == "critical"],
                "warning_resources": [name for name, resource in utilization.items() if resource["level"] == "warning"]
            }

            return utilization

        except Exception as e:
            logger.error(f"❌ 资源利用率计算失败: {e}")
            return {"error": str(e)}

    def get_alert_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """获取警报历史"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]

    def update_alert_thresholds(self, **thresholds):
        """更新警报阈值"""
        for key, value in thresholds.items():
            if key in self.alert_thresholds:
                self.alert_thresholds[key] = value
                logger.info(f"🔔 警报阈值已更新: {key} = {value}")


# 全局增强指标收集器实例
enhanced_metrics_collector = EnhancedMetricsCollector()


async def get_enhanced_metrics_collector() -> EnhancedMetricsCollector:
    """获取全局增强指标收集器实例"""
    await enhanced_metrics_collector.initialize()
    return enhanced_metrics_collector
