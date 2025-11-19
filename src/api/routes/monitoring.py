"""
监控数据路由

提供系统监控、性能指标和可观测性数据的 API 接口
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.monitoring_middleware import (
    create_monitoring_dashboard,
    export_monitoring_data,
)
from src.framework.shared.monitoring import get_metrics_collector

router = APIRouter(
    prefix="/api/v1/monitoring",
    tags=["monitoring", "metrics", "observability"]
)


@router.get("/health")
async def get_health_status() -> dict[str, Any]:
    """
    获取系统健康状态

    返回系统的整体健康状况，包括：
    - 系统资源使用情况
    - 任务执行状态
    - API 服务状态
    """
    try:
        metrics_collector = get_metrics_collector()
        health_status = metrics_collector.get_health_status()

        return {
            "status": "success",
            "data": health_status,
            "timestamp": health_status["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取健康状态失败: {e}")


@router.get("/system")
async def get_system_metrics(hours: int = 24) -> dict[str, Any]:
    """
    获取系统资源指标

    Args:
        hours: 查询时间范围（小时）

    Returns:
        系统资源使用情况，包括 CPU、内存、磁盘、网络等指标
    """
    try:
        metrics_collector = get_metrics_collector()
        system_summary = metrics_collector.get_system_summary(hours=hours)

        return {
            "status": "success",
            "data": system_summary,
            "time_range": f"最近 {hours} 小时"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {e}")


@router.get("/tasks")
async def get_task_metrics(hours: int = 24) -> dict[str, Any]:
    """
    获取任务执行指标

    Args:
        hours: 查询时间范围（小时）

    Returns:
        任务执行统计信息，包括成功率、执行时间等
    """
    try:
        metrics_collector = get_metrics_collector()
        task_summary = metrics_collector.get_task_summary(hours=hours)

        return {
            "status": "success",
            "data": task_summary,
            "time_range": f"最近 {hours} 小时"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取任务指标失败: {e}")


@router.get("/api")
async def get_api_metrics(hours: int = 24) -> dict[str, Any]:
    """
    获取 API 请求指标

    Args:
        hours: 查询时间范围（小时）

    Returns:
        API 请求统计信息，包括响应时间、状态码分布等
    """
    try:
        metrics_collector = get_metrics_collector()
        api_summary = metrics_collector.get_api_summary(hours=hours)

        return {
            "status": "success",
            "data": api_summary,
            "time_range": f"最近 {hours} 小时"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 API 指标失败: {e}")


@router.get("/dashboard")
async def get_monitoring_dashboard() -> dict[str, Any]:
    """
    获取监控仪表板数据

    返回综合的监控仪表板数据，包括：
    - 系统健康状态
    - 资源使用情况
    - 任务执行统计
    - API 请求统计
    """
    try:
        dashboard = create_monitoring_dashboard()

        return {
            "status": "success",
            "data": dashboard,
            "timestamp": dashboard["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取监控仪表板失败: {e}")


@router.get("/export")
async def export_monitoring_data_api(output_dir: str = "logs/monitoring") -> dict[str, Any]:
    """
    导出监控数据

    Args:
        output_dir: 输出目录路径

    Returns:
        导出的文件路径信息
    """
    try:
        exported_files = export_monitoring_data(output_dir)

        return {
            "status": "success",
            "data": {
                "exported_files": exported_files,
                "output_dir": output_dir,
                "export_time": exported_files.get("system", "").split("_")[-1].replace(".json", "") if "system" in exported_files else ""
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出监控数据失败: {e}")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics() -> str:
    """
    获取 Prometheus 格式的指标

    返回 Prometheus 兼容的指标格式，便于集成到 Prometheus 监控系统
    """
    try:
        metrics_collector = get_metrics_collector()

        # 构建 Prometheus 格式的指标
        prometheus_metrics = []

        # 系统指标
        if metrics_collector.system_metrics:
            latest_system = metrics_collector.system_metrics[-1]
            prometheus_metrics.extend([
                '# HELP system_cpu_usage CPU 使用率百分比',
                '# TYPE system_cpu_usage gauge',
                f'system_cpu_usage {latest_system.cpu_percent}',
                '# HELP system_memory_usage 内存使用率百分比',
                '# TYPE system_memory_usage gauge',
                f'system_memory_usage {latest_system.memory_percent}',
                '# HELP system_memory_used_mb 内存使用量（MB）',
                '# TYPE system_memory_used_mb gauge',
                f'system_memory_used_mb {latest_system.memory_used_mb}',
                '# HELP system_disk_usage 磁盘使用率百分比',
                '# TYPE system_disk_usage gauge',
                f'system_disk_usage {latest_system.disk_usage_percent}',
            ])

        # 任务指标统计
        task_summary = metrics_collector.get_task_summary(hours=1)
        if task_summary:
            for task_name, stats in task_summary.items():
                safe_task_name = task_name.replace("-", "_").replace(" ", "_")
                prometheus_metrics.extend([
                    f'# HELP task_total_count_{safe_task_name} 任务总数',
                    f'# TYPE task_total_count_{safe_task_name} counter',
                    f'task_total_count_{safe_task_name} {stats.get("total_count", 0)}',
                    f'# HELP task_success_rate_{safe_task_name} 任务成功率百分比',
                    f'# TYPE task_success_rate_{safe_task_name} gauge',
                    f'task_success_rate_{safe_task_name} {stats.get("success_rate", 0)}',
                    f'# HELP task_avg_execution_time_{safe_task_name} 平均执行时间（毫秒）',
                    f'# TYPE task_avg_execution_time_{safe_task_name} gauge',
                    f'task_avg_execution_time_{safe_task_name} {stats.get("avg_execution_time", 0)}',
                ])

        # API 指标统计
        api_summary = metrics_collector.get_api_summary(hours=1)
        if api_summary:
            for endpoint, stats in api_summary.items():
                safe_endpoint = endpoint.replace(" ", "_").replace("/", "_")
                prometheus_metrics.extend([
                    f'# HELP api_request_count_{safe_endpoint} API 请求总数',
                    f'# TYPE api_request_count_{safe_endpoint} counter',
                    f'api_request_count_{safe_endpoint} {stats.get("total_requests", 0)}',
                    f'# HELP api_success_rate_{safe_endpoint} API 成功率百分比',
                    f'# TYPE api_success_rate_{safe_endpoint} gauge',
                    f'api_success_rate_{safe_endpoint} {stats.get("success_rate", 0)}',
                    f'# HELP api_avg_response_time_{safe_endpoint} 平均响应时间（毫秒）',
                    f'# TYPE api_avg_response_time_{safe_endpoint} gauge',
                    f'api_avg_response_time_{safe_endpoint} {stats.get("avg_response_time", 0)}',
                ])

        # 健康状态指标
        health_status = metrics_collector.get_health_status()
        health_mapping = {"healthy": 1, "warning": 0.5, "critical": 0}
        prometheus_metrics.extend([
            '# HELP system_health Overall system health status',
            '# TYPE system_health gauge',
            f'system_health {health_mapping.get(health_status.get("overall_health", "warning"), 0.5)}',
            '# HELP system_health_cpu CPU health status',
            '# TYPE system_health_cpu gauge',
            f'system_health_cpu {health_mapping.get(health_status.get("system_health", "warning"), 0.5)}',
            '# HELP system_health_tasks Task health status',
            '# TYPE system_health_tasks gauge',
            f'system_health_tasks {health_mapping.get(health_status.get("task_health", "warning"), 0.5)}',
        ])

        return "\n".join(prometheus_metrics) + "\n"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 Prometheus 指标失败: {e}")


@router.get("/stats/summary")
async def get_monitoring_summary() -> dict[str, Any]:
    """
    获取监控摘要信息

    返回简洁的监控摘要，适用于快速状态检查
    """
    try:
        metrics_collector = get_metrics_collector()

        # 获取健康状态
        health_status = metrics_collector.get_health_status()

        # 获取最近 1 小时的统计数据
        system_summary = metrics_collector.get_system_summary(hours=1)
        api_summary = metrics_collector.get_api_summary(hours=1)

        summary = {
            "overall_health": health_status.get("overall_health", "unknown"),
            "timestamp": health_status["timestamp"],
            "system": {
                "cpu_usage": system_summary.get("current_resources", {}).get("cpu_percent", 0),
                "memory_usage": system_summary.get("current_resources", {}).get("memory_percent", 0),
                "health": health_status.get("system_health", "unknown")
            },
            "tasks": {
                "total_samples": len(metrics_collector.task_metrics),
                "health": health_status.get("task_health", "unknown")
            },
            "api": {
                "total_samples": len(metrics_collector.api_metrics),
                "total_endpoints": len(api_summary) if api_summary else 0
            }
        }

        return {
            "status": "success",
            "data": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取监控摘要失败: {e}")
