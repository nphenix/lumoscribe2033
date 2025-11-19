"""
Document Review 领域模型

定义文档审查相关的数据模型。
"""

from datetime import datetime

from sqlmodel import Field, SQLModel


class ReviewComment(SQLModel):
    """审查评论"""
    comment_id: str
    review_id: int
    comment_type: str  # question, suggestion, issue, praise
    content: str
    page_number: int | None = None
    line_number: int | None = None
    is_resolved: bool = Field(default=False)
    resolved_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentReview(SQLModel, table=True):
    """文档审查记录"""
    __tablename__ = "document_reviews"

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(index=True)
    document_type: str
    document_content: str
    review_status: str = Field(
        default="pending"  # pending, reviewing, completed, rejected
    )
    reviewer: str | None = Field(default=None)
    review_notes: str | None = Field(default=None)
    confidence_score: float | None = Field(default=None)
    quality_score: float | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ReviewMetric(SQLModel, table=True):
    """审查指标记录"""
    __tablename__ = "review_metrics"

    id: int | None = Field(default=None, primary_key=True)
    review_id: int = Field(foreign_key="document_reviews.id")
    metric_name: str
    metric_value: float
    metric_unit: str | None = Field(default=None)
    threshold_min: float | None = Field(default=None)
    threshold_max: float | None = Field(default=None)
    is_passed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
