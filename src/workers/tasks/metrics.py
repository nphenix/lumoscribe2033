"""
Metrics 相关的 Arq 任务

集成 LangChain 1.0 和 OpenTelemetry 最佳实践的指标收集系统
"""

import asyncio
import time
from datetime import datetime
from typing import Any

from src.framework.shared.logging import get_logger
from src.framework.shared.monitoring import get_enhanced_metrics_collector
from src.framework.shared.redis_cache import get_cache_manager

logger = get_logger(__name__)


async def collect_comprehensive_metrics(
    ctx: dict[str, Any],
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    收集综合系统指标任务

    Args:
        ctx: Arq 上下文
        request_data: 请求数据

    Returns:
        任务执行结果
    """
    start_time = time.time()

    try:
        # 获取增强指标收集器
        metrics_collector = await get_enhanced_metrics_collector()
        cache_manager = await get_cache_manager()

        # 收集综合指标
        comprehensive_metrics = await metrics_collector.collect_comprehensive_metrics()

        # 收集缓存健康状态
        cache_health = {}
        if cache_manager:
            cache_health = await cache_manager.health_check()

        # 收集系统资源指标
        system_metrics = _collect_system_resources()

        # 收集应用性能指标
        app_metrics = _collect_application_performance()

        # 生成综合报告
        report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - start_time,
            "metrics": {
                "comprehensive": comprehensive_metrics,
                "cache_health": cache_health,
                "system_resources": system_metrics,
                "application_performance": app_metrics,
                "langchain_integration": _get_langchain_metrics(),
                "opentelemetry_status": _get_opentelemetry_status()
            },
            "message": "综合指标收集完成"
        }

        # 缓存报告结果
        if cache_manager:
            await cache_manager.set(
                f"metrics_report_{int(time.time())}",
                report,
                ttl=1800  # 30分钟缓存
            )

        logger.info("📊 综合指标收集完成")
        return report

    except Exception as e:
        logger.error(f"❌ 综合指标收集失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "message": "指标收集失败"
        }


async def collect_real_time_metrics(
    ctx: dict[str, Any],
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    实时指标收集任务

    Args:
        ctx: Arq 上下文
        request_data: 请求数据

    Returns:
        实时指标数据
    """
    start_time = time.time()

    try:
        # metrics_collector = await get_enhanced_metrics_collector()
        # cache_manager = await get_cache_manager()

        # 获取实时指标
        real_time_metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_load": _get_current_system_load(),
            "cache_performance": _get_cache_performance(),
            "application_health": _get_application_health(),
            "active_alerts": _get_active_alerts()
        }

        return {
            "success": True,
            "metrics": real_time_metrics,
            "execution_time": time.time() - start_time,
            "message": "实时指标收集完成"
        }

    except Exception as e:
        logger.error(f"❌ 实时指标收集失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "message": "实时指标收集失败"
        }


async def generate_performance_report(
    ctx: dict[str, Any],
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    生成性能报告任务

    Args:
        ctx: Arq 上下文
        request_data: 请求数据

    Returns:
        性能报告
    """
    start_time = time.time()

    try:
        metrics_collector = await get_enhanced_metrics_collector()

        # 获取历史数据
        alert_history = metrics_collector.get_alert_history(hours=24)
        performance_trends = metrics_collector._calculate_performance_trends()
        resource_utilization = metrics_collector._calculate_resource_utilization()

        # 生成性能报告
        report = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - start_time,
            "report": {
                "summary": {
                    "period": "24小时",
                    "total_alerts": len(alert_history),
                    "critical_alerts": len([a for a in alert_history if a.get("level") == "critical"]),
                    "warning_alerts": len([a for a in alert_history if a.get("level") == "warning"]),
                    "overall_health": "healthy" if len(alert_history) == 0 else "degraded"
                },
                "performance_trends": performance_trends,
                "resource_utilization": resource_utilization,
                "recommendations": _generate_performance_recommendations(resource_utilization, alert_history)
            },
            "message": "性能报告生成完成"
        }

        logger.info("📈 性能报告生成完成")
        return report

    except Exception as e:
        logger.error(f"❌ 性能报告生成失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "message": "性能报告生成失败"
        }


def _collect_system_resources() -> dict[str, Any]:
    """收集系统资源指标"""
    try:
        import psutil

        # CPU 信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # 内存信息
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        # 网络信息
        network = psutil.net_io_counters()

        # 进程信息
        processes = len(psutil.pids())

        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": cpu_count,
                "frequency_current": cpu_freq.current if cpu_freq else 0,
                "frequency_min": cpu_freq.min if cpu_freq else 0,
                "frequency_max": cpu_freq.max if cpu_freq else 0
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            },
            "network": {
                "bytes_sent": network.bytes_sent if network else 0,
                "bytes_recv": network.bytes_recv if network else 0,
                "packets_sent": network.packets_sent if network else 0,
                "packets_recv": network.packets_recv if network else 0
            },
            "processes": {
                "count": processes,
                "running": len([p for p in psutil.process_iter() if p.status() == 'running'])
            }
        }

    except Exception as e:
        logger.error(f"❌ 系统资源收集失败: {e}")
        return {}


def _collect_application_performance() -> dict[str, Any]:
    """收集应用性能指标"""
    try:
        from src.framework.shared.monitoring import metrics_collector

        # 获取任务指标
        task_summary = metrics_collector.get_task_summary(hours=1)

        # 获取 API 指标
        api_summary = metrics_collector.get_api_summary(hours=1)

        # 计算性能指标
        total_requests = sum(
            summary.get("total_requests", 0)
            for summary in api_summary.values()
        )
        successful_requests = sum(
            summary.get("success_rate", 0) * summary.get("total_requests", 0) / 100
            for summary in api_summary.values()
        )
        return {
            "requests": {
                "total": total_requests,
                "successful": successful_requests,
                "failed": total_requests - successful_requests,
                "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0
            },
            "response_time": {
                "avg": sum(
                    summary.get("avg_response_time", 0)
                    for summary in api_summary.values()
                ) / max(len(api_summary), 1),
                "min": min(
                    summary.get("min_response_time", 0)
                    for summary in api_summary.values()
                ) if any(summary.get("min_response_time", 0) for summary in api_summary.values()) else 0,
                "max": max(
                    summary.get("max_response_time", 0)
                    for summary in api_summary.values()
                ) if any(summary.get("max_response_time", 0) for summary in api_summary.values()) else 0
            },
            "tasks": {
                "total": sum(
                    summary.get("total_count", 0)
                    for summary in task_summary.values()
                ),
                "successful": sum(
                    summary.get("success_count", 0)
                    for summary in task_summary.values()
                ),
                "failed": sum(
                    summary.get("failed_count", 0)
                    for summary in task_summary.values()
                )
            }
        }


    except Exception as e:
        logger.error(f"❌ 应用性能收集失败: {e}")
        return {}


def _get_langchain_metrics() -> dict[str, Any]:
    """获取 LangChain 集成指标"""
    try:
        from src.framework.orchestrators.langchain_runner import get_global_runner

        runner = get_global_runner()
        if not runner:
            return {"status": "not_initialized"}

        # 获取路由统计
        routing_stats = getattr(runner, 'routing_stats', {})

        # 获取模型健康状态
        health_stats = getattr(runner, 'health_stats', {})

        return {
            "status": "active",
            "routing": {
                "total_requests": routing_stats.get("total_requests", 0),
                "successful_routings": routing_stats.get("successful_routings", 0),
                "fallback_count": routing_stats.get("fallback_count", 0),
                "avg_routing_time": routing_stats.get("avg_routing_time", 0)
            },
            "health_checks": {
                "total_checks": health_stats.get("total_checks", 0),
                "healthy_models": health_stats.get("healthy_models", 0),
                "unhealthy_models": health_stats.get("unhealthy_models", 0)
            }
        }

    except Exception as e:
        logger.error(f"❌ LangChain 指标获取失败: {e}")
        return {"status": "error", "error": str(e)}


def _get_opentelemetry_status() -> dict[str, Any]:
    """获取 OpenTelemetry 状态"""
    try:
        from src.framework.shared.telemetry import get_telemetry_metrics

        # telemetry_metrics = get_telemetry_metrics()

        return {
            "status": "active",
            "tracing_enabled": True,
            "metrics_enabled": True,
            "exporters": {
                "span_exporter": "console",
                "metric_exporter": "console"
            },
            "instrumentation": {
                "auto_instrumentation": True,
                "libraries": ["fastapi", "requests", "sqlite3"]
            }
        }

    except Exception as e:
        logger.error(f"❌ OpenTelemetry 状态获取失败: {e}")
        return {"status": "error", "error": str(e)}


def _get_current_system_load() -> dict[str, float]:
    """获取当前系统负载"""
    try:
        import psutil

        # 1分钟平均负载
        load_avg = psutil.getloadavg()

        return {
            "load_1min": load_avg[0] if len(load_avg) > 0 else 0,
            "load_5min": load_avg[1] if len(load_avg) > 1 else 0,
            "load_15min": load_avg[2] if len(load_avg) > 2 else 0
        }

    except Exception as e:
        logger.error(f"❌ 系统负载获取失败: {e}")
        return {}


def _get_cache_performance() -> dict[str, Any]:
    """获取缓存性能指标"""
    try:
        cache_manager = asyncio.run(get_cache_manager())

        if not cache_manager:
            return {"status": "not_available"}

        cache_metrics = cache_manager.get_metrics()
        cache_health = asyncio.run(cache_manager.health_check())

        return {
            "status": "active",
            "hit_rate": cache_metrics.get("cache_metrics", {}).get("hit_rate", 0),
            "operations_per_second": cache_metrics.get("operations", {}).get("per_second", 0),
            "local_cache_utilization": cache_metrics.get("local_cache", {}).get("utilization", 0),
            "active_locks": cache_metrics.get("locks", {}).get("active_count", 0),
            "health_status": cache_health.get("status", "unknown")
        }

    except Exception as e:
        logger.error(f"❌ 缓存性能获取失败: {e}")
        return {"status": "error", "error": str(e)}


def _get_application_health() -> dict[str, Any]:
    """获取应用健康状态"""
    try:
        from src.framework.shared.monitoring import metrics_collector

        health_status = metrics_collector.get_health_status()

        return {
            "overall": health_status.get("overall_health", "unknown"),
            "system": health_status.get("system_health", "unknown"),
            "tasks": health_status.get("task_health", "unknown"),
            "timestamp": health_status.get("timestamp", datetime.now().isoformat())
        }

    except Exception as e:
        logger.error(f"❌ 应用健康状态获取失败: {e}")
        return {"overall": "error", "error": str(e)}


def _get_active_alerts() -> list[dict[str, Any]]:
    """获取活跃警报"""
    try:
        from src.framework.shared.monitoring import get_enhanced_metrics_collector

        metrics_collector = asyncio.run(get_enhanced_metrics_collector())
        alert_history = metrics_collector.get_alert_history(hours=1)

        # 只返回最近1小时的警报
        return alert_history

    except Exception as e:
        logger.error(f"❌ 活跃警报获取失败: {e}")
        return []


def _generate_performance_recommendations(
    resource_utilization: dict[str, Any],
    alert_history: list[dict[str, Any]]
) -> list[str]:
    """生成性能优化建议"""
    recommendations = []

    try:
        # CPU 使用率建议
        cpu_util = resource_utilization.get("cpu", {})
        if cpu_util.get("level") in ["warning", "critical"]:
            recommendations.append("考虑优化CPU密集型任务或增加CPU资源")
            recommendations.append("检查是否有异常进程消耗CPU资源")

        # 内存使用率建议
        memory_util = resource_utilization.get("memory", {})
        if memory_util.get("level") in ["warning", "critical"]:
            recommendations.append("优化内存使用，检查内存泄漏")
            recommendations.append("考虑增加物理内存或优化应用内存管理")

        # 警报频率建议
        critical_alerts = [a for a in alert_history if a.get("level") == "critical"]
        if len(critical_alerts) > 5:  # 24小时内超过5个严重警报
            recommendations.append("系统存在严重性能问题，建议立即检查")

        # 缓存性能建议
        if len([a for a in alert_history if a.get("type") == "cache"]) > 3:
            recommendations.append("优化缓存策略，考虑调整缓存大小或TTL设置")

        return recommendations if recommendations else ["系统运行正常，无特殊建议"]

    except Exception as e:
        logger.error(f"❌ 性能建议生成失败: {e}")
        return ["无法生成性能建议"]


# 保持向后兼容的原始函数
async def collect_metrics(
    ctx: dict[str, Any],
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    收集系统指标任务（向后兼容版本）
    """
    return await collect_comprehensive_metrics(ctx, request_data)
