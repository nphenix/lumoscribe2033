"""
任务管理路由

提供 Arq 任务队列的管理和监控功能。
基于 FastAPI 最佳实践实现：
- 任务状态查询
- 任务提交和取消
- 队列监控
- 执行历史
"""

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from src.workers.settings import get_queue_info

router = APIRouter(tags=["tasks"])


class TaskRequest(BaseModel):
    """任务请求模型"""
    task_type: str
    payload: dict[str, Any]
    priority: int = 0
    delay: int = 0


class TaskResponse(BaseModel):
    """任务响应模型"""
    task_id: str
    status: str
    task_type: str
    created_at: str
    scheduled_for: str | None = None


class QueueStatusResponse(BaseModel):
    """队列状态响应模型"""
    queue_name: str
    queue_size: int
    running_jobs: int
    completed_jobs: int
    worker_count: int


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    status: str
    task_type: str
    progress: dict[str, Any] | None = None
    result: Any | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status() -> QueueStatusResponse:
    """
    获取队列状态

    返回当前任务队列的统计信息
    """
    queue_info = await get_queue_info()

    return QueueStatusResponse(
        queue_name=queue_info.get("queue_name", "lumoscribe2033"),
        queue_size=queue_info.get("queue_size", 0),
        running_jobs=queue_info.get("running_jobs", 0),
        completed_jobs=queue_info.get("completed_jobs", 0),
        worker_count=4  # 从配置获取
    )


@router.post("/queue/push", response_model=TaskResponse)
async def push_task(
    task: TaskRequest,
    background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    提交新任务

    将新任务加入队列并返回任务信息
    """
    import datetime
    import uuid

    # 验证任务类型
    valid_task_types = [
        "speckit_constitution",
        "speckit_specify",
        "speckit_plan",
        "speckit_tasks",
        "pipeline_full",
        "compliance_check",
        "knowledge_import",
        "ide_package",
        "metrics_collect"
    ]

    if task.task_type not in valid_task_types:
        raise HTTPException(
            status_code=400,
            detail=f"无效的任务类型。支持的类型: {', '.join(valid_task_types)}"
        )

    # 生成任务 ID
    task_id = str(uuid.uuid4())

    # 模拟任务提交
    background_tasks.add_task(_simulate_task_execution, task_id, task)

    return TaskResponse(
        task_id=task_id,
        status="queued",
        task_type=task.task_type,
        created_at=datetime.datetime.now().isoformat(),
        scheduled_for=(
            (datetime.datetime.now() + datetime.timedelta(seconds=task.delay)).isoformat()
            if task.delay > 0
            else None
        )
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    获取任务状态

    根据任务 ID 查询任务执行状态
    """
    # 这里应该从 Redis 或数据库获取实际任务状态
    # 目前返回模拟数据

    return TaskStatusResponse(
        task_id=task_id,
        status="completed",
        task_type="speckit_constitution",
        progress={"current": 100, "total": 100, "message": "任务完成"},
        result={"output_path": "./specs/001-hybrid-rag-platform", "files": ["constitution.md"]},
        created_at="2025-11-15T08:00:00",
        started_at="2025-11-15T08:00:10",
        completed_at="2025-11-15T08:05:00"
    )


@router.get("/tasks", response_model=list[TaskStatusResponse])
async def list_tasks(
    status: str | None = Query(None, description="任务状态过滤"),
    task_type: str | None = Query(None, description="任务类型过滤"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
) -> list[TaskStatusResponse]:
    """
    列出任务

    分页查询任务列表，支持状态和类型过滤
    """
    # 这里应该从 Redis 或数据库获取实际任务列表
    # 目前返回模拟数据

    tasks = [
        TaskStatusResponse(
            task_id="task_001",
            status="completed",
            task_type="speckit_constitution",
            progress={"current": 100, "total": 100, "message": "任务完成"},
            result={"output_path": "./specs/001-hybrid-rag-platform"},
            created_at="2025-11-15T08:00:00",
            started_at="2025-11-15T08:00:10",
            completed_at="2025-11-15T08:05:00"
        ),
        TaskStatusResponse(
            task_id="task_002",
            status="running",
            task_type="pipeline_full",
            progress={"current": 50, "total": 100, "message": "处理中..."},
            created_at="2025-11-15T08:10:00",
            started_at="2025-11-15T08:10:30"
        ),
        TaskStatusResponse(
            task_id="task_003",
            status="failed",
            task_type="compliance_check",
            progress={"current": 0, "total": 100, "message": "任务失败"},
            error="数据库连接失败",
            created_at="2025-11-15T08:15:00"
        )
    ]

    # 应用过滤条件
    if status:
        tasks = [t for t in tasks if t.status == status]

    if task_type:
        tasks = [t for t in tasks if t.task_type == task_type]

    # 应用分页
    return tasks[offset:offset + limit]


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str) -> dict[str, str]:
    """
    取消任务

    根据任务 ID 取消正在执行的任务
    """
    # 这里应该调用 Arq 的取消任务 API
    # 目前返回模拟结果

    return {"message": f"任务 {task_id} 已取消", "task_id": task_id}


@router.post("/tasks/{task_id}/retry")
async def retry_task(task_id: str) -> dict[str, str]:
    """
    重试任务

    对失败的任务进行重试
    """
    # 这里应该创建新的任务实例
    # 目前返回模拟结果

    return {"message": f"任务 {task_id} 已重新加入队列", "task_id": task_id}


@router.get("/tasks/stats")
async def get_task_stats() -> dict[str, Any]:
    """
    获取任务统计信息

    返回任务执行的统计摘要
    """
    return {
        "total_tasks": 1234,
        "completed_tasks": 1100,
        "failed_tasks": 89,
        "running_tasks": 45,
        "average_execution_time": "2m 34s",
        "success_rate": 89.2,
        "tasks_by_type": {
            "speckit_constitution": 200,
            "speckit_specify": 180,
            "speckit_plan": 175,
            "speckit_tasks": 160,
            "pipeline_full": 150,
            "compliance_check": 140,
            "knowledge_import": 120,
            "ide_package": 80,
            "metrics_collect": 29
        },
        "queue_stats": {
            "avg_queue_time": "30s",
            "max_queue_time": "5m",
            "queue_depth": 25
        }
    }


async def _simulate_task_execution(task_id: str, task: TaskRequest) -> None:
    """
    模拟任务执行

    实际实现中应该调用相应的 Arq 任务函数
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"开始执行任务 {task_id}: {task.task_type}")

    # 模拟任务执行时间
    await asyncio.sleep(2)

    logger.info(f"任务 {task_id} 执行完成")
