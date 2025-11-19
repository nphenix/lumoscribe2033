"""
领域模型测试

测试所有领域模型的数据完整性和约束
"""

from datetime import datetime

import pytest

from src.domain.compliance.models import ComplianceIssue, ComplianceReport
from src.domain.doc_review.models import DocumentReview, ReviewComment
from src.domain.knowledge.models import BestPractice, PracticeReference
from src.domain.pipeline.models import PipelineExecution, PipelineStep


class TestPipelineModels:
    """Pipeline 模型测试"""

    def test_pipeline_execution_creation(self):
        """测试 PipelineExecution 创建"""
        execution = PipelineExecution(
            pipeline_id="test-pipeline",
            name="Test Pipeline",
            status="pending"
        )
        assert execution.pipeline_id == "test-pipeline"
        assert execution.status == "pending"
        assert execution.created_at is not None

    def test_pipeline_step_creation(self):
        """测试 PipelineStep 创建"""
        step = PipelineStep(
            execution_id=1,
            step_name="Test Step",
            step_order=1,
            status="pending"
        )
        assert step.step_name == "Test Step"
        assert step.step_order == 1


class TestKnowledgeModels:
    """Knowledge 模型测试"""

    def test_best_practice_creation(self):
        """测试 BestPractice 创建"""
        practice = BestPractice(
            practice_id="bp-001",
            title="Test Practice",
            description="Test Description",
            category="coding",
            content="Test Content"
        )
        assert practice.practice_id == "bp-001"
        assert practice.category == "coding"
        assert practice.is_active is True

    def test_practice_reference_creation(self):
        """测试 PracticeReference 创建"""
        ref = PracticeReference(
            practice_id="bp-001",
            reference_type="conversation",
            reference_id="conv-001"
        )
        assert ref.practice_id == "bp-001"
        assert ref.reference_type == "conversation"


# TODO: 添加更多模型测试
# - ComplianceReport
# - ComplianceIssue
# - DocumentReview
# - ReviewComment
