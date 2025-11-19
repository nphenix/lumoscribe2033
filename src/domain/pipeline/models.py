"""
Pipeline 领域模型

定义 Pipeline 执行相关的数据模型。
"""

from datetime import datetime
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class PipelineExecution(SQLModel, table=True):
    """Pipeline 执行记录"""
    __tablename__ = "pipeline_executions"

    id: int | None = Field(default=None, primary_key=True)
    pipeline_id: str = Field(index=True)
    name: str
    status: str = Field(default="pending")  # pending, running, completed, failed
    input_data: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PipelineStep(SQLModel, table=True):
    """Pipeline 步骤记录"""
    __tablename__ = "pipeline_steps"

    id: int | None = Field(default=None, primary_key=True)
    execution_id: int = Field(foreign_key="pipeline_executions.id")
    step_name: str
    step_order: int
    status: str = Field(default="pending")
    input_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    execution_time_ms: float | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
