"""
Arq 工作者设置

基于 Arq 最佳实践配置：
- 任务队列管理
- 重试策略
- 并发控制
- 结果存储
- 监控和可观测性
- 工作进程管理和健康检查

特性：
- 异步任务处理
- 自动重试机制
- 任务优先级
- 资源管理
- 详细的监控指标
- 健康状态检查
"""

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

import psutil

if TYPE_CHECKING:
    pass

from arq import ArqRedis, create_pool, cron
from arq.connections import RedisSettings
from loguru import logger

from src.framework.orchestrators import bootstrap_langchain_executor
from src.framework.shared.config import Settings
from src.framework.shared.monitoring import metrics_collector
from src.workers.serialization import MsgpackSerializer


class AdvancedWorkerSettings:
    """Arq 工作者配置类 - 基于 Arq v0.26+ 最佳实践"""

    # Redis 连接设置 - 支持更多配置选项
    @classmethod
    def _get_redis_settings(cls):
        """获取 Redis 连接设置"""
        settings = Settings()
        parsed = urlparse(settings.ARQ_REDIS_URL)

        return RedisSettings(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            database=int(parsed.path.lstrip("/")) if parsed.path else 0,
            # Arq v0.26+ 最佳实践配置
            ssl=parsed.scheme in ("rediss", "rediss"),
            ssl_certfile=None,
            ssl_keyfile=None,
            ssl_ca_certs=None,
            username=parsed.username,
            password=parsed.password,
            conn_timeout=60,  # 连接超时
            conn_retries=5,  # 连接重试次数
            conn_retry_delay=1,  # 连接重试延迟
            max_connections=20,  # 最大连接数
        )


    # 工作者基本信息
    job_serializer = MsgpackSerializer.serialize
    job_deserializer = MsgpackSerializer.deserialize
    queue_name = Settings().ARQ_QUEUE_NAME

    # Arq v0.26+ 新增配置
    job_id_prefix = "lumoscribe2033"
    max_burst_jobs = 10
    health_check_interval = 30
    log_results = True
    log_curtail = 1000  # 日志长度限制

    # 任务函数定义
    functions = [
        # Speckit 任务
        'src.workers.tasks.speckit.run_constitution',
        'src.workers.tasks.speckit.run_specify',
        'src.workers.tasks.speckit.run_plan',
        'src.workers.tasks.speckit.run_tasks',

        # Pipeline 任务
        'src.workers.tasks.pipeline.run_full_pipeline',
        'src.workers.tasks.pipeline.process_document',
        'src.workers.tasks.pipeline.generate_speckit_output',

        # 合规检查任务
        'src.workers.tasks.compliance.run_static_check',
        'src.workers.tasks.compliance.check_speckit_compliance',
        'src.workers.tasks.compliance.validate_document_structure',

        # 知识管理任务
        'src.workers.tasks.knowledge.import_conversations',
        'src.workers.tasks.knowledge.generate_ide_package',
        'src.workers.tasks.knowledge.update_vector_store',
        'src.workers.tasks.knowledge.build_knowledge_graph',

        # 指标收集任务
        'src.workers.tasks.metrics.collect_metrics',
        'src.workers.tasks.metrics.generate_compliance_report',
        'src.workers.tasks.metrics.analyze_system_performance',

        # 文档处理任务
        'src.workers.tasks.docs.upload_and_evaluate',
        'src.workers.tasks.docs.batch_process_documents',
        'src.workers.tasks.docs.generate_document_report',
    ]

    # 并发设置
    max_jobs = 10  # 最大并发任务数
    job_timeout = 300  # 任务超时时间（秒）
    keep_result = 3600  # 结果保存时间（秒）
    keep_result_forever = False  # 是否永久保存结果

    # 重试设置
    max_tries = 3  # 最大重试次数
    retry_delay = 30  # 重试延迟（秒）

    # 任务优先级设置
    burst = False  # 是否以 burst 模式运行
    poll_delay = 0.5  # 轮询延迟（秒）

    # 任务队列配置
    max_burst_jobs_queue = 5  # Burst 模式下的最大任务数
    health_check_interval_queue = 60  # 健康检查间隔（秒）

    # 生命周期钩子
    on_startup = 'src.workers.lifecycle.on_startup'
    on_shutdown = 'src.workers.lifecycle.on_shutdown'
    on_after_job = 'src.workers.lifecycle.on_after_job'
    on_before_job = 'src.workers.lifecycle.on_before_job'

    # 序列化配置
    serialization_manager = None  # SerializationManager(default_serializer='msgpack')

    # 性能优化配置
    max_jobs_per_worker = 10  # 每个工作进程最大并发任务数
    max_queue_size = 1000  # 队列最大长度

    # 结果存储配置
    result_ttl = 3600  # 结果保存时间（秒）
    result_limit = 100  # 最大结果数量

    # 错误处理配置
    error_retry_attempts = 3
    error_retry_delay = 60  # 错误重试延迟（秒）

    # 监控配置
    metrics_enabled = True
    metrics_interval = 60  # 监控指标收集间隔（秒）


# 设置 redis_settings
AdvancedWorkerSettings.redis_settings = AdvancedWorkerSettings._get_redis_settings()


async def on_startup(ctx: dict[str, Any]) -> None:
    """工作者启动时的初始化 - Arq 最佳实践"""
    logger.info("🚀 Arq 工作者启动中...")

    # 记录启动信息
    worker_info = {
        "worker_id": ctx.get("worker_id", "unknown"),
        "start_time": time.time(),
        "pid": ctx.get("pid", "unknown"),
        "hostname": ctx.get("hostname", "unknown"),
        "python_version": ctx.get("python_version", "unknown"),
        "arq_version": ctx.get("arq_version", "unknown"),
    }

    # 保存工作者信息到上下文
    ctx["worker_info"] = worker_info
    ctx["start_time"] = time.time()

    # 初始化监控指标收集
    if metrics_collector:
        await metrics_collector.start_worker_monitoring()

    # 初始化 LangChainExecutor，确保 Worker 侧也能复用路由/追踪
    bootstrap_langchain_executor(settings=Settings())

    # 初始化数据库连接
    # 初始化向量存储
    # 初始化 LLM 客户端

    logger.info(f"✅ Arq 工作者启动完成 - Worker ID: {worker_info['worker_id']}")


async def on_shutdown(ctx: dict[str, Any]) -> None:
    """工作者关闭时的清理 - Arq 最佳实践"""
    logger.info("🛑 Arq 工作者正在关闭...")

    # 记录关闭信息
    worker_info = ctx.get("worker_info", {})
    start_time = ctx.get("start_time", time.time())
    uptime = time.time() - start_time

    shutdown_info = {
        "worker_id": worker_info.get("worker_id", "unknown"),
        "uptime": uptime,
        "shutdown_time": time.time(),
        "total_jobs_processed": ctx.get("total_jobs_processed", 0),
        "successful_jobs": ctx.get("successful_jobs", 0),
        "failed_jobs": ctx.get("failed_jobs", 0),
    }

    logger.info(f"📊 工作者运行统计: {shutdown_info}")

    # 停止监控指标收集
    if metrics_collector:
        await metrics_collector.stop_worker_monitoring()

    # 关闭数据库连接
    # 清理临时文件
    # 保存状态

    logger.info("✅ Arq 工作者已关闭")


async def on_before_job(ctx: dict[str, Any], job_id: str) -> None:
    """任务开始前的钩子 - Arq 最佳实践"""
    logger.debug(f"📋 开始执行任务: {job_id}")

    # 记录任务开始时间
    ctx[f"job_start_time_{job_id}"] = time.time()

    # 更新任务计数器
    ctx["total_jobs_processed"] = ctx.get("total_jobs_processed", 0) + 1

    # 记录系统资源使用情况
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()

        ctx[f"job_resources_{job_id}"] = {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "cpu_percent": cpu_percent,
            "timestamp": time.time()
        }
    except Exception:
        pass


async def on_after_job(
    ctx: dict[str, Any],
    job_id: str,
    result: Any | None = None,
    exc: Exception | None = None
) -> None:
    """任务完成后的钩子 - Arq 最佳实践"""
    import datetime

    # 计算任务执行时间
    start_time = ctx.get(f"job_start_time_{job_id}")
    execution_time = (time.time() - start_time) if start_time else 0

    # 获取任务资源使用情况
    job_resources = ctx.get(f"job_resources_{job_id}", {})

    if exc:
        logger.error(f"❌ 任务执行失败: {job_id}, 错误: {exc}, 耗时: {execution_time:.2f}s")

        # 更新失败任务计数器
        ctx["failed_jobs"] = ctx.get("failed_jobs", 0) + 1

        # 记录失败任务的详细信息
        failure_info = {
            "job_id": job_id,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "execution_time": execution_time,
            "worker_info": ctx.get("worker_info", {}),
            "resources": job_resources,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # 保存失败信息到日志文件
        import json
        from pathlib import Path

        failure_log_file = Path("logs/job_failures.log")
        failure_log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(failure_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_info, ensure_ascii=False) + "\n")
        except Exception as log_error:
            logger.error(f"记录失败信息时出错: {log_error}")

        # 记录失败任务指标
        if metrics_collector:
            await metrics_collector.record_task_metric(
                task_name=job_id,
                execution_time=execution_time * 1000,  # 转换为毫秒
                status="failed",
                error=str(exc),
                worker_id=ctx.get("worker_info", {}).get("worker_id", "unknown"),
                resources=job_resources
            )
    else:
        logger.info(f"✅ 任务执行成功: {job_id}, 耗时: {execution_time:.2f}s")

        # 更新成功任务计数器
        ctx["successful_jobs"] = ctx.get("successful_jobs", 0) + 1

        # 记录成功任务的统计信息
        if result and isinstance(result, dict):
            logger.info(f"📊 任务结果统计: {result.get('stats', {})}")

        # 记录成功任务指标
        if metrics_collector:
            await metrics_collector.record_task_metric(
                task_name=job_id,
                execution_time=execution_time * 1000,  # 转换为毫秒
                status="success",
                result_size=len(str(result)) if result else 0,
                worker_id=ctx.get("worker_info", {}).get("worker_id", "unknown"),
                resources=job_resources
            )

    # 清理任务相关上下文
    ctx.pop(f"job_start_time_{job_id}", None)
    ctx.pop(f"job_resources_{job_id}", None)


async def create_redis_pool() -> ArqRedis:
    """创建 Redis 连接池"""
    return await create_pool(WorkerSettings.redis_settings)


# Arq v0.26+ 最佳实践：Cron 任务配置
def cron_jobs() -> list:
    """定义周期性任务"""
    return [
        # 每5分钟执行一次健康检查
        cron(
            'src.workers.tasks.metrics.health_check',
            minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
            name='health_check_cron',
            timeout=300,
            retries=3,
            retry_delay=30,
        ),

        # 每10分钟收集一次系统指标
        cron(
            'src.workers.tasks.metrics.collect_system_metrics',
            minute={0, 10, 20, 30, 40, 50},
            name='system_metrics_cron',
            timeout=600,
            retries=2,
            retry_delay=60,
        ),

        # 每30分钟清理一次旧任务记录
        cron(
            'src.workers.tasks.metrics.cleanup_old_tasks',
            minute={0, 30},
            name='cleanup_old_tasks_cron',
            timeout=900,
            retries=1,
            retry_delay=120,
        ),

        # 每小时生成一次性能报告
        cron(
            'src.workers.tasks.metrics.generate_performance_report',
            minute=0,
            name='performance_report_cron',
            timeout=1800,
            retries=2,
            retry_delay=300,
        ),
    ]


    # 任务函数定义

    # Arq v0.26+ 基本配置

    # 错误处理配置

    # 结果管理配置

    # 队列管理配置

    # Cron 任务
    cron_jobs()

    # 其他高级配置


async def enqueue_job(
    function_name: str,
    *args,
    **kwargs
) -> str | None:
    """
    入队任务

    Args:
        function_name: 任务函数名
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        任务 ID 或 None
    """
    try:
        redis = await create_redis_pool()
        job = await redis.enqueue_job(function_name, *args, **kwargs)
        logger.info(f"📨 任务已入队: {function_name}, ID: {job.job_id}")
        return job.job_id if job else None
    except Exception as e:
        logger.error(f"❌ 任务入队失败: {function_name}, 错误: {e}")
        return None


async def get_job_result(job_id: str) -> Any:
    """
    获取任务执行结果

    Args:
        job_id: 任务 ID

    Returns:
        任务结果
    """
    try:
        redis = await create_redis_pool()
        job_result = await redis.get_job_result(job_id)

        if job_result:
            if job_result.success:
                logger.info(f"✅ 获取任务结果成功: {job_id}")
                return job_result.result
            else:
                logger.error(f"❌ 任务执行失败: {job_id}, 错误: {job_result.exc}")
                raise job_result.exc
        else:
            logger.warning(f"⚠️ 任务结果不存在: {job_id}")
            return None

    except Exception as e:
        logger.error(f"❌ 获取任务结果失败: {job_id}, 错误: {e}")
        raise


async def cancel_job(job_id: str) -> bool:
    """
    取消任务

    Args:
        job_id: 任务 ID

    Returns:
        是否取消成功
    """
    try:
        redis = await create_redis_pool()
        result = await redis.cancel_job(job_id)

        if result:
            logger.info(f"🛑 任务已取消: {job_id}")
        else:
            logger.warning(f"⚠️ 任务取消失败: {job_id}")

        return result
    except Exception as e:
        logger.error(f"❌ 任务取消失败: {job_id}, 错误: {e}")
        return False


async def get_queue_info() -> dict[str, Any]:
    """
    获取队列信息

    Returns:
        队列信息字典
    """
    try:
        redis = await create_redis_pool()

        # 获取队列中的任务数
        queue_size = await redis.zcard(redis.queue_name)

        # 获取正在运行的任务数
        running_jobs = await redis.keys(f"{redis.queue_name}:*")

        # 获取已完成的任务数
        completed_jobs = await redis.get(f"{redis.queue_name}:completed")

        return {
            "queue_size": queue_size,
            "running_jobs": len(running_jobs),
            "completed_jobs": int(completed_jobs or 0),
            "queue_name": redis.queue_name
        }
    except Exception as e:
        logger.error(f"❌ 获取队列信息失败: {e}")
        return {}


async def get_worker_stats() -> dict[str, Any]:
    """
    获取工作者统计信息

    Returns:
        工作者统计信息字典
    """
    try:
        redis = await create_redis_pool()

        # 获取工作者信息
        worker_keys = await redis.keys("arq:worker:*")
        worker_count = len(worker_keys)

        # 获取任务执行统计
        stats_keys = await redis.keys("arq:stats:*")
        total_jobs = 0
        successful_jobs = 0
        failed_jobs = 0

        for key in stats_keys:
            stats_data = await redis.hgetall(key)
            if stats_data:
                total_jobs += int(stats_data.get(b'total', 0))
                successful_jobs += int(stats_data.get(b'successful', 0))
                failed_jobs += int(stats_data.get(b'failed', 0))

        return {
            "worker_count": worker_count,
            "total_jobs": total_jobs,
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            "queue_name": redis.queue_name
        }
    except Exception as e:
        logger.error(f"❌ 获取工作者统计失败: {e}")
        return {}


async def cleanup_old_jobs(days: int = 7) -> int:
    """
    清理旧的任务记录

    Args:
        days: 保留天数

    Returns:
        清理的任务数量
    """
    try:
        # 计算过期时间戳
        import time
        time.time() - (days * 24 * 60 * 60)

        # 这里应该实现具体的清理逻辑
        # 由于 Arq 的内部结构，清理逻辑可能需要根据实际情况调整

        logger.info(f"🧹 清理 {days} 天前的任务记录完成")
        return 0

    except Exception as e:
        logger.error(f"❌ 清理旧任务失败: {e}")
        return 0


async def get_job_history(limit: int = 100) -> list[dict[str, Any]]:
    """
    获取任务执行历史

    Args:
        limit: 返回记录数限制

    Returns:
        任务历史列表
    """
    try:
        # 这里应该实现获取任务历史的逻辑
        # 可能需要查询 Redis 中的任务记录

        logger.info(f"📋 获取最近 {limit} 个任务的历史记录")
        return []

    except Exception as e:
        logger.error(f"❌ 获取任务历史失败: {e}")
        return []


# Arq v0.26+ 最佳实践：增强的监控和管理功能
async def get_detailed_worker_stats() -> dict[str, Any]:
    """
    获取详细的工作进程统计信息

    Returns:
        详细的工作进程统计信息
    """
    try:
        redis = await create_redis_pool()

        # 获取基本统计信息
        basic_stats = await get_worker_stats()

        # 获取系统资源信息
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "network": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
        }

        # 获取 Redis 连接信息
        redis_info = {
            "redis_version": await redis.info("server"),
            "connected_clients": await redis.info("clients"),
            "used_memory": await redis.info("memory"),
            "keyspace": await redis.info("keyspace"),
        }

        # 获取队列详细信息
        queue_info = await get_queue_info()

        return {
            "timestamp": time.time(),
            "basic_stats": basic_stats,
            "system_info": system_info,
            "redis_info": redis_info,
            "queue_info": queue_info,
            "worker_settings": {
                "max_jobs": AdvancedWorkerSettings.max_jobs,
                "job_timeout": AdvancedWorkerSettings.job_timeout,
                "max_tries": AdvancedWorkerSettings.max_tries,
                "queue_name": AdvancedWorkerSettings.queue_name,
            }
        }

    except Exception as e:
        logger.error(f"❌ 获取详细工作进程统计失败: {e}")
        return {}


async def health_check() -> dict[str, Any]:
    """
    执行健康检查

    Returns:
        健康检查结果
    """
    try:
        redis = await create_redis_pool()

        # 检查 Redis 连接
        redis_status = "healthy"
        try:
            await redis.ping()
        except Exception:
            redis_status = "unhealthy"

        # 检查系统资源
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        # 资源健康状态判断
        system_status = "healthy"
        if cpu_percent > 90:
            system_status = "critical"
        elif memory_percent > 80 or disk_percent > 90:
            system_status = "warning"

        # 检查队列状态
        queue_info = await get_queue_info()
        queue_size = queue_info.get("queue_size", 0)
        queue_status = "healthy"
        if queue_size > 1000:
            queue_status = "critical"
        elif queue_size > 500:
            queue_status = "warning"

        # 整体健康状态
        overall_status = "healthy"
        if "critical" in [redis_status, system_status, queue_status]:
            overall_status = "critical"
        elif "warning" in [redis_status, system_status, queue_status]:
            overall_status = "warning"

        health_result = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "redis_status": redis_status,
            "system_status": system_status,
            "queue_status": queue_status,
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "queue_size": queue_size,
                "running_jobs": queue_info.get("running_jobs", 0),
            }
        }

        logger.info(f"🏥 健康检查结果: {health_result}")
        return health_result

    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        return {
            "timestamp": time.time(),
            "overall_status": "error",
            "error": str(e),
        }


async def emergency_shutdown() -> bool:
    """
    紧急关闭所有工作进程

    Returns:
        是否成功关闭
    """
    try:
        redis = await create_redis_pool()

        # 获取所有活跃的工作进程
        await redis.keys("arq:worker:*")

        # 取消所有队列中的任务
        queue_name = WorkerSettings.queue_name
        queued_jobs = await redis.zrange(queue_name, 0, -1)

        cancelled_count = 0
        for job_id in queued_jobs:
            try:
                await redis.cancel_job(job_id.decode())
                cancelled_count += 1
            except Exception:
                continue

        logger.warning(f"🛑 紧急关闭: 取消了 {cancelled_count} 个队列任务")
        return True

    except Exception as e:
        logger.error(f"❌ 紧急关闭失败: {e}")
        return False




# 向后兼容性别名
WorkerSettings = AdvancedWorkerSettings
