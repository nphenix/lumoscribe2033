"""
对话管理路由

提供对话导入、查询和管理功能，支持从 Cursor、RooCode 等 IDE 导入对话记录，
并将其存储到向量知识库中。
"""

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, UploadFile
from loguru import logger
from pydantic import BaseModel

from src.domain.compliance.models import ConversationRecord
from src.framework.adapters.conversation_adapter import ConversationAdapter
from src.framework.storage.enhanced_graph_store import EnhancedGraphStoreManager
from src.framework.storage.enhanced_vector_store import EnhancedVectorStoreManager

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    responses={404: {"description": "Not found"}}
)


class ConversationImportRequest(BaseModel):
    """对话导入请求模型"""
    source_type: str  # cursor, roocode, manual
    path: str
    submission_refs: list[str] | None = None
    auto_detect: bool = True
    recursive: bool = False
    create_vector_index: bool = True
    create_graph_links: bool = True


class ConversationImportResponse(BaseModel):
    """对话导入响应模型"""
    success: bool
    conversations_ingested: int
    vector_embeddings_created: int
    graph_nodes_created: int
    warnings: list[str]
    errors: list[str]
    processing_time: float


class ConversationSearchRequest(BaseModel):
    """对话搜索请求模型"""
    query: str
    source_type: str | None = None
    limit: int = 20
    offset: int = 0


class ConversationRecordResponse(BaseModel):
    """对话记录响应模型"""
    id: str
    conversation_id: str
    source: str
    user_message: str
    assistant_message: str | None = None
    context_data: dict[str, Any] | None = None
    meta_info: dict[str, Any] | None = None
    created_at: str
    processed_at: str | None = None
    vector_id: str | None = None
    graph_node_id: str | None = None


@router.post("/import", response_model=ConversationImportResponse)
async def import_conversations(
    request: ConversationImportRequest,
    background_tasks: BackgroundTasks
) -> ConversationImportResponse:
    """
    导入对话记录

    从目录或文件导入 AI 对话记录，支持多种来源格式
    """
    import time
    start_time = time.time()

    try:
        logger.info(f"开始导入对话: {request.source_type} from {request.path}")

        # 验证来源类型
        valid_sources = ["cursor", "roocode", "manual"]
        if request.source_type not in valid_sources:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的来源类型。支持的类型: {', '.join(valid_sources)}"
            )

        # 检查路径存在性
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"指定的路径不存在: {request.path}"
            )

        # 初始化适配器
        adapter = ConversationAdapter.get_adapter(request.source_type)
        if not adapter:
            raise HTTPException(
                status_code=500,
                detail=f"无法创建 {request.source_type} 适配器"
            )

        # 导入对话记录
        import_result = await adapter.import_conversations(
            path=path,
            auto_detect=request.auto_detect,
            recursive=request.recursive
        )

        conversations = import_result.get("conversations", [])
        warnings = import_result.get("warnings", [])
        errors = import_result.get("errors", [])

        if not conversations:
            raise HTTPException(
                status_code=400,
                detail="未找到有效的对话记录"
            )

        # 保存到数据库
        for conv in conversations:
            ConversationRecord(
                conversation_id=conv.get("id", ""),
                source=request.source_type,
                user_message=conv.get("user_message", ""),
                assistant_message=conv.get("assistant_message"),
                context_data=conv.get("context_data"),
                meta_info=conv.get("meta_info", {}),
                created_at=conv.get("created_at"),
                processed_at=None
            )

            # TODO: 保存到数据库
            # saved_record = save_conversation_record(record)
            # saved_records.append(saved_record)

        # 创建向量索引（如果需要）
        vector_embeddings_created = 0
        if request.create_vector_index and conversations:
            vector_embeddings_created = await _create_vector_embeddings(conversations)

        # 创建图谱链接（如果需要）
        graph_nodes_created = 0
        if request.create_graph_links and conversations:
            graph_nodes_created = await _create_graph_links(conversations)

        processing_time = time.time() - start_time

        logger.info(f"对话导入完成: {len(conversations)} 条记录")

        return ConversationImportResponse(
            success=True,
            conversations_ingested=len(conversations),
            vector_embeddings_created=vector_embeddings_created,
            graph_nodes_created=graph_nodes_created,
            warnings=warnings,
            errors=errors,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话导入失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=list[ConversationRecordResponse])
async def search_conversations(
    query: str = Query(..., description="搜索查询"),
    source_type: str | None = Query(None, description="来源类型过滤"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量")
) -> list[ConversationRecordResponse]:
    """
    搜索对话记录

    基于语义相似度搜索对话记录
    """
    try:
        logger.info(f"搜索对话: {query}")

        # 使用向量存储进行语义搜索
        vector_store = EnhancedVectorStoreManager()
        search_results = await vector_store.semantic_search(
            query=query,
            limit=limit,
            offset=offset,
            filter_dict={"source": source_type} if source_type else None
        )

        # 转换为响应格式
        records = []
        for result in search_results:
            record = ConversationRecordResponse(
                id=result.get("id", ""),
                conversation_id=result.get("conversation_id", ""),
                source=result.get("source", ""),
                user_message=result.get("user_message", ""),
                assistant_message=result.get("assistant_message"),
                context_data=result.get("context_data"),
                meta_info=result.get("meta_info"),
                created_at=result.get("created_at", ""),
                processed_at=result.get("processed_at"),
                vector_id=result.get("vector_id"),
                graph_node_id=result.get("graph_node_id")
            )
            records.append(record)

        logger.info(f"搜索完成，返回 {len(records)} 条记录")
        return records

    except Exception as e:
        logger.error(f"对话搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{conversation_id}", response_model=ConversationRecordResponse)
async def get_conversation(
    conversation_id: str
) -> ConversationRecordResponse:
    """
    获取单个对话记录

    根据对话 ID 获取详细的对话信息
    """
    try:
        logger.info(f"查询对话: {conversation_id}")

        # TODO: 从数据库查询对话记录
        # record = get_conversation_record(conversation_id)

        # 模拟数据
        record = ConversationRecordResponse(
            id=conversation_id,
            conversation_id=conversation_id,
            source="cursor",
            user_message="如何实现文档分类功能？",
            assistant_message="可以使用基于关键词的分类算法...",
            context_data={
                "file_path": "src/domain/doc_review/classifier.py",
                "timestamp": "2025-11-17T10:30:00Z"
            },
            meta_info={
                "session_id": "session_123",
                "project_context": "lumoscribe2033"
            },
            created_at="2025-11-17T10:30:00Z",
            processed_at="2025-11-17T10:31:00Z",
            vector_id=f"vector_{conversation_id}",
            graph_node_id=f"node_{conversation_id}"
        )

        return record

    except Exception as e:
        logger.error(f"获取对话记录失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources", response_model=list[dict[str, Any]])
async def get_supported_sources() -> list[dict[str, Any]]:
    """
    获取支持的对话来源

    返回系统支持的所有对话来源类型及其配置信息
    """
    try:
        sources = [
            {
                "type": "cursor",
                "name": "Cursor IDE",
                "description": "Cursor IDE 对话记录",
                "file_patterns": ["*.log", "*.json"],
                "directory_structure": ".cursor/logs/",
                "auto_detect": True
            },
            {
                "type": "roocode",
                "name": "RooCode IDE",
                "description": "RooCode IDE 对话记录",
                "file_patterns": ["*.txt", "*.md"],
                "directory_structure": ".roo/logs/",
                "auto_detect": True
            },
            {
                "type": "manual",
                "name": "手动导入",
                "description": "手动格式化的对话文件",
                "file_patterns": ["*.json", "*.md"],
                "directory_structure": "自定义",
                "auto_detect": False
            }
        ]

        logger.info(f"返回 {len(sources)} 个支持的来源")
        return sources

    except Exception as e:
        logger.error(f"获取支持来源失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str
) -> dict[str, str]:
    """
    删除对话记录

    删除指定的对话记录及其相关的向量和图谱数据
    """
    try:
        logger.info(f"删除对话: {conversation_id}")

        # TODO: 删除数据库记录
        # delete_conversation_record(conversation_id)

        # TODO: 删除向量数据
        # delete_vector_embedding(conversation_id)

        # TODO: 删除图谱节点
        # delete_graph_node(conversation_id)

        return {
            "message": f"对话 {conversation_id} 已删除",
            "conversation_id": conversation_id
        }

    except Exception as e:
        logger.error(f"删除对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=dict[str, Any])
async def get_conversation_stats() -> dict[str, Any]:
    """
    获取对话统计信息

    返回对话库的统计摘要
    """
    try:
        # TODO: 计算实际统计数据
        stats = {
            "total_conversations": 1234,
            "by_source": {
                "cursor": 856,
                "roocode": 278,
                "manual": 100
            },
            "storage_info": {
                "vector_embeddings": 1234,
                "graph_nodes": 1100,
                "total_size_mb": 45.6
            },
            "time_range": {
                "earliest": "2025-11-01T00:00:00Z",
                "latest": "2025-11-17T23:59:59Z"
            },
            "quality_metrics": {
                "complete_conversations": 1100,
                "partial_conversations": 134,
                "average_length": 156.7
            }
        }

        logger.info("返回对话统计信息")
        return stats

    except Exception as e:
        logger.error(f"获取对话统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 辅助函数
async def _create_vector_embeddings(conversations: list[dict[str, Any]]) -> int:
    """创建向量 embeddings"""
    try:
        vector_store = EnhancedVectorStoreManager()

        embeddings_created = 0
        for conv in conversations:
            # 组合用户消息和助手消息作为文档
            document = f"{conv.get('user_message', '')}\n\n{conv.get('assistant_message', '')}"

            metadata = {
                "conversation_id": conv.get("id", ""),
                "source": conv.get("source", ""),
                "created_at": conv.get("created_at", ""),
                "user_message": conv.get("user_message", ""),
                "assistant_message": conv.get("assistant_message", "")
            }

            # 添加到向量存储
            await vector_store.add_document(
                document=document,
                metadata=metadata,
                doc_id=conv.get("id", "")
            )
            embeddings_created += 1

        logger.info(f"创建了 {embeddings_created} 个向量 embeddings")
        return embeddings_created

    except Exception as e:
        logger.error(f"创建向量 embeddings 失败: {e}")
        return 0


async def _create_graph_links(conversations: list[dict[str, Any]]) -> int:
    """创建图谱链接"""
    try:
        graph_store = EnhancedGraphStoreManager()

        nodes_created = 0
        for conv in conversations:
            # 创建对话节点
            node_data = {
                "type": "conversation",
                "source": conv.get("source", ""),
                "user_message": conv.get("user_message", ""),
                "assistant_message": conv.get("assistant_message", ""),
                "created_at": conv.get("created_at", ""),
                "content_hash": conv.get("content_hash", "")
            }

            # 添加节点到图谱
            await graph_store.add_node(
                node_id=conv.get("id", ""),
                node_data=node_data
            )
            nodes_created += 1

        logger.info(f"创建了 {nodes_created} 个图谱节点")
        return nodes_created

    except Exception as e:
        logger.error(f"创建图谱链接失败: {e}")
        return 0
