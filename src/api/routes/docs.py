"""
文档管理路由 (v2)

提供文档上传、分类、评估和管理功能。
基于 FastAPI 最佳实践实现：
- 文件上传处理
- 文档分类评估
- 质量检查
- 批量操作
- API 版本控制
- 增强的请求验证
"""

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from pydantic.v1 import validator

from src.api.versioning import (
    create_versioned_router,
    get_api_version,
    get_version_warning,
)

# v1 版本的兼容模型定义

class BatchOperationRequest(BaseModel):
    """批量操作请求模型"""
    document_ids: list[str] = Field(..., min_length=1, max_length=100, description="文档 ID 列表")
    operation: str = Field(..., description="操作类型")
    parameters: dict[str, Any] | None = Field(default=None, description="操作参数")

    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ["evaluate", "delete", "classify", "export"]
        if v not in valid_operations:
            raise ValueError(f"不支持的操作类型: {v}")
        return v


class DocumentMetadata(BaseModel):
    """文档元数据模型 v1（兼容版本）"""
    filename: str = Field(..., description="文件名")
    file_type: str = Field(..., description="文件 MIME 类型")
    file_size: int = Field(..., ge=0, description="文件大小（字节）")
    upload_time: str = Field(..., description="上传时间 ISO 格式")
    document_type: str = Field(..., description="文档类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分类置信度")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    status: str = Field(..., description="文档状态")


class DocumentEvaluation(BaseModel):
    """文档评估模型 v1（兼容版本）"""
    document_id: str = Field(..., description="文档 ID")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="总体评分")
    category_scores: dict[str, float] = Field(..., description="分类评分")
    issues: list[str] = Field(default_factory=list, description="问题列表")
    recommendations: list[str] = Field(default_factory=list, description="建议列表")
    token_usage: dict[str, int] | None = Field(default=None, description="Token 使用情况")

    @validator('category_scores')
    def validate_category_scores(cls, v):
        for score in v.values():
            if not 0.0 <= score <= 100.0:
                raise ValueError("分类评分必须在 0-100 之间")
        return v


class DocumentResponse(BaseModel):
    """文档响应模型 v1（兼容版本）"""
    id: str = Field(..., description="文档唯一标识")
    metadata: DocumentMetadata = Field(..., description="文档元数据")
    evaluation: DocumentEvaluation | None = Field(default=None, description="评估结果")
    analysis: dict[str, Any] | None = Field(default=None, description="分析数据")




class BatchOperationResponse(BaseModel):
    """批量操作响应模型 v1（兼容版本）"""
    operation_id: str = Field(..., description="操作唯一标识")
    total_count: int = Field(..., description="总操作数")
    success_count: int = Field(..., description="成功操作数")
    failed_count: int = Field(..., description="失败操作数")
    results: list[dict[str, Any]] = Field(..., description="操作结果")

# 创建 v2 版本的路由器
router = create_versioned_router(
    version="v2",
    prefix="/docs",
    tags=["documents", "v2"]
)


# v1 版本的路由器（保持向后兼容）
v1_router = APIRouter(
    prefix="/v1/docs",
    tags=["documents", "v1", "deprecated"]
)


# v2 版本的模型（增强验证）
class DocumentMetadataV2(BaseModel):
    """文档元数据模型 v2"""
    filename: str = Field(..., min_length=1, max_length=255, description="文件名")
    file_type: str = Field(..., description="文件 MIME 类型")
    file_size: int = Field(..., ge=0, le=50*1024*1024, description="文件大小（字节）")
    upload_time: str = Field(..., description="上传时间 ISO 格式")
    document_type: str = Field(..., description="文档类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分类置信度")
    tags: list[str] = Field(default_factory=list, max_length=20, description="标签列表")
    status: str = Field(..., description="文档状态")
    version: str = Field(default="v2", description="API 版本")

    @validator('file_type')
    def validate_file_type(cls, v):
        allowed_types = [
            "text/markdown", "text/plain", "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if v not in allowed_types:
            raise ValueError(f"不支持的文件类型: {v}")
        return v

    @validator('document_type')
    def validate_document_type(cls, v):
        allowed_types = ["agent", "developer", "external", "internal"]
        if v not in allowed_types:
            raise ValueError(f"无效的文档类型: {v}")
        return v


class DocumentEvaluationV2(BaseModel):
    """文档评估模型 v2"""
    document_id: str = Field(..., description="文档 ID")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="总体评分")
    category_scores: dict[str, float] = Field(..., description="分类评分")
    issues: list[str] = Field(default_factory=list, description="问题列表")
    recommendations: list[str] = Field(default_factory=list, description="建议列表")
    token_usage: dict[str, int] | None = Field(default=None, description="Token 使用情况")
    evaluation_time: str | None = Field(default=None, description="评估时间")

    @validator('category_scores')
    def validate_category_scores(cls, v):
        for score in v.values():
            if not 0.0 <= score <= 100.0:
                raise ValueError("分类评分必须在 0-100 之间")
        return v


class DocumentResponseV2(BaseModel):
    """文档响应模型 v2"""
    id: str = Field(..., description="文档唯一标识")
    metadata: DocumentMetadataV2 = Field(..., description="文档元数据")
    evaluation: DocumentEvaluationV2 | None = Field(default=None, description="评估结果")
    analysis: dict[str, Any] | None = Field(default=None, description="分析数据")
    api_version: str = Field(default="v2", description="API 版本")


class BatchOperationRequestV2(BaseModel):
    """批量操作请求模型 v2"""
    document_ids: list[str] = Field(..., min_length=1, max_length=100, description="文档 ID 列表")
    operation: str = Field(..., description="操作类型")
    parameters: dict[str, Any] | None = Field(default=None, description="操作参数")
    priority: int = Field(default=0, ge=0, le=10, description="操作优先级")

    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = ["evaluate", "delete", "classify", "export", "migrate"]
        if v not in valid_operations:
            raise ValueError(f"不支持的操作类型: {v}")
        return v


class BatchOperationResponseV2(BaseModel):
    """批量操作响应模型 v2"""
    operation_id: str = Field(..., description="操作唯一标识")
    total_count: int = Field(..., description="总操作数")
    success_count: int = Field(..., description="成功操作数")
    failed_count: int = Field(..., description="失败操作数")
    results: list[dict[str, Any]] = Field(..., description="操作结果")
    processing_time: float = Field(..., description="处理时间（秒）")
    api_version: str = Field(default="v2", description="API 版本")


# v1 版本的兼容模型
class DocumentMetadataV1(BaseModel):
    """文档元数据模型 v1（兼容版本）"""
    filename: str
    file_type: str
    file_size: int
    upload_time: str
    document_type: str
    confidence: float
    tags: list[str]
    status: str


class DocumentEvaluationV1(BaseModel):
    """文档评估模型 v1（兼容版本）"""
    document_id: str
    overall_score: float
    category_scores: dict[str, float]
    issues: list[str]
    recommendations: list[str]
    token_usage: dict[str, int] | None = None


class DocumentResponseV1(BaseModel):
    """文档响应模型 v1（兼容版本）"""
    id: str
    metadata: DocumentMetadataV1
    evaluation: DocumentEvaluationV1 | None = None
    analysis: dict[str, Any] | None = None


@router.post("/upload", response_model=DocumentResponseV2)
async def upload_document_v2(
    request: Request,
    file: UploadFile = File(..., description="要上传的文档文件"),
    auto_evaluate: bool = Query(True, description="是否自动进行文档评估"),
    document_type: str | None = Query(None, description="手动指定文档类型"),
    validate_content: bool = Query(False, description="是否进行内容验证")
) -> DocumentResponseV2:
    """
    上传文档 v2

    支持上传各种格式的文档文件并进行自动分类和评估
    新增功能：
    - 增强的文件验证
    - 内容质量检查
    - 更详细的元数据
    - API 版本控制
    """
    import datetime
    import mimetypes
    import time
    import uuid

    start_time = time.time()
    api_version = get_api_version(request)

    # 验证文件类型
    allowed_types = [
        "text/markdown",
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    file_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "unknown"

    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。支持的类型: {', '.join(allowed_types)}"
        )

    # 读取文件内容
    content = await file.read()
    file_size = len(content)

    # 验证文件大小（v2 版本增加到 50MB）
    max_file_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小不能超过 {max_file_size // (1024 * 1024)}MB"
        )

    document_id = str(uuid.uuid4())

    # 内容验证（如果启用）
    content_issues = []
    if validate_content and file_type == "text/markdown":
        content_str = content.decode('utf-8', errors='ignore')
        if len(content_str.strip()) < 10:
            content_issues.append("文档内容过短")
        if not any(char.isupper() for char in content_str):
            content_issues.append("缺少大写字母，可能格式不正确")

    # 模拟文档分类和评估
    metadata = DocumentMetadataV2(
        filename=file.filename,
        file_type=file_type,
        file_size=file_size,
        upload_time=datetime.datetime.now().isoformat(),
        document_type=document_type or "developer",
        confidence=0.85,
        tags=["speckit", "documentation"],
        status="uploaded" if not content_issues else "validation_failed",
        version=api_version
    )

    evaluation = None
    if auto_evaluate:
        issues = [
            "文档结构不够清晰",
            "缺少必要的示例代码",
            "术语使用不一致"
        ]
        issues.extend(content_issues)  # 添加内容验证问题

        evaluation = DocumentEvaluationV2(
            document_id=document_id,
            overall_score=78.5,
            category_scores={
                "structure": 85.0,
                "clarity": 75.0,
                "completeness": 80.0,
                "consistency": 72.0,
                "content_quality": 88.0 if not content_issues else 60.0
            },
            issues=issues,
            recommendations=[
                "添加目录结构",
                "补充代码示例",
                "统一术语使用"
            ],
            token_usage={
                "total_tokens": 1250,
                "prompt_tokens": 1200,
                "completion_tokens": 50
            },
            evaluation_time=datetime.datetime.now().isoformat()
        )

    processing_time = time.time() - start_time

    response = DocumentResponseV2(
        id=document_id,
        metadata=metadata,
        evaluation=evaluation,
        analysis={
            "processing_time": processing_time,
            "file_hash": str(hash(content)),
            "content_issues": content_issues
        },
        api_version=api_version
    )

    # 添加版本警告（如果有）
    version_warning = get_version_warning(request)
    if version_warning:
        response.analysis["version_warning"] = version_warning

    return response


# v1 兼容版本的上传接口
@v1_router.post("/upload", response_model=DocumentResponseV1)
async def upload_document_v1(
    file: UploadFile = File(..., description="要上传的文档文件"),
    auto_evaluate: bool = Query(True, description="是否自动进行文档评估"),
    document_type: str | None = Query(None, description="手动指定文档类型")
) -> DocumentResponseV1:
    """
    上传文档 v1（兼容版本）

    保持与 v1 API 的向后兼容性
    """
    import datetime
    import mimetypes
    import uuid

    # 验证文件类型
    allowed_types = [
        "text/markdown",
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    file_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "unknown"

    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。支持的类型: {', '.join(allowed_types)}"
        )

    # 读取文件内容
    content = await file.read()
    file_size = len(content)

    # v1 版本的文件大小限制（10MB）
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="文件大小不能超过 10MB"
        )

    document_id = str(uuid.uuid4())

    # 模拟文档分类和评估
    metadata = DocumentMetadataV1(
        filename=file.filename,
        file_type=file_type,
        file_size=file_size,
        upload_time=datetime.datetime.now().isoformat(),
        document_type=document_type or "developer",
        confidence=0.85,
        tags=["speckit", "documentation"],
        status="uploaded"
    )

    evaluation = None
    if auto_evaluate:
        evaluation = DocumentEvaluationV1(
            document_id=document_id,
            overall_score=78.5,
            category_scores={
                "structure": 85.0,
                "clarity": 75.0,
                "completeness": 80.0,
                "consistency": 72.0
            },
            issues=[
                "文档结构不够清晰",
                "缺少必要的示例代码",
                "术语使用不一致"
            ],
            recommendations=[
                "添加目录结构",
                "补充代码示例",
                "统一术语使用"
            ],
            token_usage={
                "total_tokens": 1250,
                "prompt_tokens": 1200,
                "completion_tokens": 50
            }
        )

    return DocumentResponseV1(
        id=document_id,
        metadata=metadata,
        evaluation=evaluation
    )


@router.get("/{document_id}", response_model=DocumentResponseV2)
async def get_document_v2(
    request: Request,
    document_id: str,
    include_analysis: bool = Query(False, description="是否包含详细分析")
) -> DocumentResponseV2:
    """
    获取文档信息 v2

    根据文档 ID 获取文档的详细信息和评估结果
    新增功能：
    - 详细的分析数据
    - API 版本信息
    - 性能指标
    """
    import datetime

    api_version = get_api_version(request)

    # 这里应该从数据库获取实际文档信息
    # 目前返回模拟数据

    metadata = DocumentMetadataV2(
        filename="speckit_specification.md",
        file_type="text/markdown",
        file_size=15678,
        upload_time="2025-11-15T08:00:00",
        document_type="developer",
        confidence=0.92,
        tags=["speckit", "specification", "requirements"],
        status="evaluated",
        version=api_version
    )

    evaluation = DocumentEvaluationV2(
        document_id=document_id,
        overall_score=85.2,
        category_scores={
            "structure": 90.0,
            "clarity": 85.0,
            "completeness": 82.0,
            "consistency": 84.0,
            "technical_accuracy": 88.0
        },
        issues=["缺少实现细节"],
        recommendations=["添加具体的实现步骤"],
        token_usage={
            "total_tokens": 2100,
            "prompt_tokens": 2000,
            "completion_tokens": 100
        },
        evaluation_time="2025-11-15T08:15:00"
    )

    analysis = None
    if include_analysis:
        analysis = {
            "access_count": 156,
            "last_accessed": "2025-11-15T08:30:00",
            "similarity_score": 0.87,
            "related_documents": ["doc_001", "doc_002"],
            "quality_trend": "improving",
            "processing_metrics": {
                "evaluation_time": 2.5,
                "analysis_time": 1.8,
                "total_time": 4.3
            }
        }

    response = DocumentResponseV2(
        id=document_id,
        metadata=metadata,
        evaluation=evaluation,
        analysis=analysis,
        api_version=api_version
    )

    # 添加版本警告（如果有）
    version_warning = get_version_warning(request)
    if version_warning:
        if response.analysis is None:
            response.analysis = {}
        response.analysis["version_warning"] = version_warning

    return response


# v1 兼容版本
@v1_router.get("/{document_id}", response_model=DocumentResponseV1)
async def get_document_v1(document_id: str) -> DocumentResponseV1:
    """
    获取文档信息 v1（兼容版本）
    """
    return DocumentResponseV1(
        id=document_id,
        metadata=DocumentMetadataV1(
            filename="speckit_specification.md",
            file_type="text/markdown",
            file_size=15678,
            upload_time="2025-11-15T08:00:00",
            document_type="developer",
            confidence=0.92,
            tags=["speckit", "specification", "requirements"],
            status="evaluated"
        ),
        evaluation=DocumentEvaluationV1(
            document_id=document_id,
            overall_score=85.2,
            category_scores={
                "structure": 90.0,
                "clarity": 85.0,
                "completeness": 82.0,
                "consistency": 84.0
            },
            issues=["缺少实现细节"],
            recommendations=["添加具体的实现步骤"],
            token_usage={
                "total_tokens": 2100,
                "prompt_tokens": 2000,
                "completion_tokens": 100
            }
        )
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    document_type: str | None = Query(None, description="文档类型过滤"),
    status: str | None = Query(None, description="文档状态过滤"),
    tags: str | None = Query(None, description="标签过滤，逗号分隔"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
) -> list[DocumentResponse]:
    """
    列出文档

    分页查询文档列表，支持类型、状态和标签过滤
    """
    # 这里应该从数据库获取实际文档列表
    # 目前返回模拟数据

    documents = [
        DocumentResponse(
            id="doc_001",
            metadata=DocumentMetadata(
                filename="speckit_constitution.md",
                file_type="text/markdown",
                file_size=8900,
                upload_time="2025-11-15T07:30:00",
                document_type="agent",
                confidence=0.95,
                tags=["speckit", "constitution"],
                status="evaluated"
            ),
            evaluation=DocumentEvaluation(
                document_id="doc_001",
                overall_score=92.3,
                category_scores={"structure": 95.0, "clarity": 90.0, "completeness": 90.0, "consistency": 92.0},
                issues=[],
                recommendations=["继续保持"],
                token_usage={"total_tokens": 800, "prompt_tokens": 750, "completion_tokens": 50}
            )
        ),
        DocumentResponse(
            id="doc_002",
            metadata=DocumentMetadata(
                filename="api_design_docs.md",
                file_type="text/markdown",
                file_size=23400,
                upload_time="2025-11-15T07:45:00",
                document_type="developer",
                confidence=0.88,
                tags=["api", "design", "documentation"],
                status="evaluated"
            ),
            evaluation=DocumentEvaluation(
                document_id="doc_002",
                overall_score=76.8,
                category_scores={"structure": 80.0, "clarity": 75.0, "completeness": 70.0, "consistency": 82.0},
                issues=["结构不够清晰", "缺少示例"],
                recommendations=["添加目录", "补充代码示例"],
                token_usage={"total_tokens": 3200, "prompt_tokens": 3000, "completion_tokens": 200}
            )
        ),
        DocumentResponse(
            id="doc_003",
            metadata=DocumentMetadata(
                filename="user_guide.pdf",
                file_type="application/pdf",
                file_size=45600,
                upload_time="2025-11-15T08:00:00",
                document_type="external",
                confidence=0.82,
                tags=["user", "guide", "manual"],
                status="pending"
            )
        )
    ]

    # 应用过滤条件
    if document_type:
        documents = [d for d in documents if d.metadata.document_type == document_type]

    if status:
        documents = [d for d in documents if d.metadata.status == status]

    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        documents = [d for d in documents if any(tag in d.metadata.tags for tag in tag_list)]

    return documents[offset:offset + limit]


@router.post("/{document_id}/evaluate", response_model=DocumentEvaluation)
async def evaluate_document(document_id: str) -> DocumentEvaluation:
    """
    评估文档质量

    对指定文档进行质量评估并返回详细报告
    """
    # 这里应该调用实际的文档评估逻辑
    # 目前返回模拟数据

    return DocumentEvaluation(
        document_id=document_id,
        overall_score=82.7,
        category_scores={
            "structure": 85.0,
            "clarity": 80.0,
            "completeness": 82.0,
            "consistency": 84.0,
            "readability": 78.0,
            "technical_accuracy": 88.0
        },
        issues=[
            "部分章节缺少详细说明",
            "代码示例不够完整"
        ],
        recommendations=[
            "补充章节详细说明",
            "完善代码示例",
            "添加错误处理说明"
        ],
        token_usage={
            "total_tokens": 1800,
            "prompt_tokens": 1700,
            "completion_tokens": 100
        }
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict[str, str]:
    """
    删除文档

    根据文档 ID 删除文档及其相关数据
    """
    # 这里应该执行实际的删除操作
    # 目前返回模拟结果

    return {"message": f"文档 {document_id} 已删除", "document_id": document_id}


@router.post("/batch", response_model=BatchOperationResponse)
async def batch_operation(request: BatchOperationRequest) -> BatchOperationResponse:
    """
    批量操作

    对多个文档执行批量操作（评估、删除、分类等）
    """
    import datetime
    import uuid

    operation_id = str(uuid.uuid4())

    # 验证操作类型
    valid_operations = ["evaluate", "delete", "classify", "export"]
    if request.operation not in valid_operations:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的操作类型。支持的类型: {', '.join(valid_operations)}"
        )

    # 模拟批量操作结果
    results = []
    success_count = 0
    failed_count = 0

    for doc_id in request.document_ids:
        try:
            # 模拟操作成功
            results.append({
                "document_id": doc_id,
                "status": "success",
                "message": f"文档 {doc_id} {request.operation} 操作成功"
            })
            success_count += 1
        except Exception:
            # 模拟操作失败
            results.append({
                "document_id": doc_id,
                "status": "failed",
                "message": f"文档 {doc_id} {request.operation} 操作失败"
            })
            failed_count += 1

    return BatchOperationResponse(
        operation_id=operation_id,
        total_count=len(request.document_ids),
        success_count=success_count,
        failed_count=failed_count,
        results=results
    )


@router.get("/stats")
async def get_document_stats() -> dict[str, Any]:
    """
    获取文档统计信息

    返回文档管理的统计摘要
    """
    return {
        "total_documents": 1234,
        "documents_by_type": {
            "agent": 234,
            "developer": 654,
            "external": 346
        },
        "documents_by_status": {
            "uploaded": 45,
            "evaluated": 987,
            "pending": 156,
            "failed": 46
        },
        "total_size": "2.3GB",
        "average_score": 78.5,
        "most_common_tags": ["speckit", "documentation", "api", "guide", "tutorial"],
        "evaluation_stats": {
            "total_evaluated": 987,
            "average_score": 78.5,
            "high_quality_count": 342,
            "low_quality_count": 89
        }
    }


@router.get("/export/{document_id}")
async def export_document(document_id: str) -> dict[str, Any]:
    """
    导出文档

    将文档导出为指定格式
    """
    return {
        "document_id": document_id,
        "export_url": f"/api/v1/docs/{document_id}/download",
        "formats": ["pdf", "markdown", "html", "docx"],
        "estimated_size": "2.5MB",
        "export_time": "2025-11-15T08:20:00"
    }
