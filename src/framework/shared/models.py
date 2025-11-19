"""
共享数据模型

定义跨层共享的数据模型和 Pydantic 模型。
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class MessageType(Enum):
    """消息类型枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SeverityLevel(Enum):
    """严重程度级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LLMMessage(SQLModel):
    """LLM 消息模型"""
    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    tool_call_id: str | None = None


class ChatRequest(SQLModel):
    """聊天请求模型"""
    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = Field(default=0.7)
    max_tokens: int | None = None
    stream: bool | None = Field(default=False)
    request_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


class ChatResponse(SQLModel):
    """聊天响应模型"""
    content: str
    usage: dict[str, int] = Field(default={}, sa_column=Column(JSON))
    model: str
    finish_reason: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EmbeddingRequest(SQLModel):
    """嵌入请求模型"""
    texts: list[str]
    model: str | None = None
    embedding_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


class EmbeddingResponse(SQLModel):
    """嵌入响应模型"""
    embeddings: list[list[float]]
    usage: dict[str, int] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskInfo(SQLModel):
    """任务信息模型"""
    task_id: str
    task_name: str
    status: TaskStatus
    progress: float | None = None
    total_steps: int | None = None
    current_step: int | None = None
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TaskResult(SQLModel):
    """任务结果模型"""
    success: bool
    message: str
    data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    execution_time: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ComplianceCheck(SQLModel):
    """合规性检查模型"""
    check_id: str
    check_type: str  # static_check, doc_review, etc.
    severity: SeverityLevel
    category: str
    title: str
    description: str
    file_path: str | None = None
    line_number: int | None = None
    column_number: int | None = None
    context: str | None = None
    suggested_fix: str | None = None
    is_resolved: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(SQLModel):
    """文档块模型"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    total_chunks: int
    title: str | None = None
    chunk_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    score: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentMetadata(SQLModel):
    """文档元数据模型"""
    document_id: str
    title: str
    document_type: str
    author: str | None = None
    created_date: datetime | None = None
    modified_date: datetime | None = None
    tags: list[str] = Field(default=[], sa_column=Column(JSON))
    size: int | None = None  # 字节数
    word_count: int | None = None
    language: str | None = None
    version: str | None = None
    checksum: str | None = None
    custom_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


class SearchResult(SQLModel):
    """搜索结果模型"""
    document_id: str
    title: str
    content: str
    score: float
    file_path: str | None = None
    search_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    snippet: str | None = None
    rank: int


class PipelineEvent(SQLModel):
    """管线事件模型"""
    event_id: str
    pipeline_id: str
    event_type: str  # start, progress, complete, error, etc.
    stage: str
    message: str
    data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MetricPoint(SQLModel):
    """指标数据点模型"""
    metric_name: str
    value: int | float
    tags: dict[str, str] = Field(default={}, sa_column=Column(JSON))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Alert(SQLModel):
    """告警模型"""
    alert_id: str
    alert_type: str
    severity: SeverityLevel
    title: str
    message: str
    source: str
    alert_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    is_acknowledged: bool = Field(default=False)
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None


class UserContext(SQLModel):
    """用户上下文模型"""
    user_id: str
    session_id: str
    conversation_history: list[dict[str, str]] = Field(
        default=[], sa_column=Column(JSON)
    )
    preferences: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    user_metadata: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)


# RAG 相关模型


@dataclass
class RetrievalMetrics:
    """检索指标 - 统一的数据模型
    
    用于记录 RAG 检索过程中的性能指标和统计信息。
    支持多种检索策略（向量、图、混合等）的指标收集。
    """
    retrieval_time: float
    retrieved_count: int
    query_complexity: str
    retrieval_strategy: str
    rerank_time: float = 0.0
    final_results_count: int = 0
    hit_rate: float = 0.0