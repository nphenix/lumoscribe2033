"""
Knowledge 领域模型

定义知识管理相关的数据模型。
"""

from datetime import datetime
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class BestPractice(SQLModel, table=True):
    """最佳实践记录"""
    __tablename__ = "best_practices"

    id: int | None = Field(default=None, primary_key=True)
    practice_id: str = Field(index=True)
    title: str
    description: str
    category: str  # coding, architecture, testing, etc.
    tags: list[str] = Field(default=[], sa_column=Column(JSON))
    content: str
    examples: list[dict[str, Any]] | None = Field(default=None, sa_column=Column(JSON))
    related_practices: list[str] = Field(default=[], sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

class PracticeReference(SQLModel, table=True):
    """实践引用记录"""
    __tablename__ = "practice_references"

    id: int | None = Field(default=None, primary_key=True)
    practice_id: str = Field(index=True)
    reference_type: str  # conversation, document, code, etc.
    reference_id: str
    context: str | None = Field(default=None)
    confidence_score: float | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
