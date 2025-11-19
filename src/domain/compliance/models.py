"""
Compliance 领域模型

定义合规性检查相关的数据模型。
"""

from datetime import datetime
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class ComplianceIssue(SQLModel):
    """合规性问题"""
    issue_id: str
    check_type: str
    severity: str  # low, medium, high, critical
    description: str
    file_path: str | None = None
    line_number: int | None = None
    suggestion: str | None = None
    related_commit: str | None = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class ComplianceReport(SQLModel, table=True):
    """合规性报告"""
    __tablename__ = "compliance_reports"

    id: int | None = Field(default=None, primary_key=True)
    report_id: str = Field(index=True)
    report_type: str  # speckit, static_check, doc_review, etc.
    status: str = Field(default="pending")  # pending, running, completed, failed
    total_checks: int = Field(default=0)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    severity_counts: dict[str, int] = Field(default={}, sa_column=Column(JSON))
    summary: str | None = Field(default=None)
    details: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ConversationRecord(SQLModel, table=True):
    """对话记录"""
    __tablename__ = "conversation_records"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(index=True)
    source: str  # cursor, roocode, etc.
    user_message: str
    assistant_message: str | None = Field(default=None)
    context_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    meta_info: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = Field(default=None)
