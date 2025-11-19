"""
SQLite 网关

提供专门的 SQLite 数据访问接口，封装常见的数据库操作模式。
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional, Union

from sqlmodel import Field, Relationship, SQLModel, select

from src.framework.shared.exceptions import DatabaseGatewayError
from src.framework.shared.logging import get_logger


# 数据模型定义
class DocumentRecord(SQLModel, table=True):
    """文档记录模型"""
    __tablename__ = "documents"

    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True, unique=True)
    content: str = Field(sa_column_kwargs={"nullable": False})
    source: str | None = Field(default=None)
    doc_type: str | None = Field(default=None)
    doc_metadata: str | None = Field(default=None, alias="metadata")  # JSON 字符串
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    embedding_vector: str | None = Field(default=None)  # JSON 字符串

    # 关系
    chunks: list["DocumentChunk"] = Relationship(back_populates="document")
    entities: list["Entity"] = Relationship(back_populates="document")


class DocumentChunk(SQLModel, table=True):
    """文档块记录模型"""
    __tablename__ = "document_chunks"

    id: int | None = Field(default=None, primary_key=True)
    chunk_id: str = Field(index=True, unique=True)
    doc_id: str = Field(foreign_key="documents.doc_id")
    content: str = Field(sa_column_kwargs={"nullable": False})
    chunk_order: int = Field(default=0)
    chunk_metadata: str | None = Field(default=None, alias="metadata")  # JSON 字符串
    embedding_vector: str | None = Field(default=None)  # JSON 字符串
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # 关系
    document: DocumentRecord = Relationship(back_populates="chunks")


class Entity(SQLModel, table=True):
    """实体记录模型"""
    __tablename__ = "entities"

    id: int | None = Field(default=None, primary_key=True)
    entity_id: str = Field(index=True, unique=True)
    doc_id: str = Field(foreign_key="documents.doc_id")
    name: str = Field(sa_column_kwargs={"nullable": False})
    entity_type: str = Field(sa_column_kwargs={"nullable": False})
    description: str | None = Field(default=None)
    entity_metadata: str | None = Field(default=None, alias="metadata")  # JSON 字符串
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # 关系
    document: DocumentRecord = Relationship(back_populates="entities")
    relationships: list["EntityRelationship"] = Relationship(back_populates="source_entity")


class EntityRelationship(SQLModel, table=True):
    """实体关系记录模型"""
    __tablename__ = "entity_relationships"

    id: int | None = Field(default=None, primary_key=True)
    relationship_id: str = Field(index=True, unique=True)
    source_entity_id: str = Field(foreign_key="entities.entity_id")
    target_entity_id: str = Field(foreign_key="entities.entity_id")
    relationship_type: str = Field(sa_column_kwargs={"nullable": False})
    description: str | None = Field(default=None)
    strength: float = Field(default=1.0)
    relationship_metadata: str | None = Field(default=None, alias="metadata")  # JSON 字符串
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # 关系
    source_entity: Entity = Relationship(back_populates="relationships")


class SearchLog(SQLModel, table=True):
    """搜索日志记录模型"""
    __tablename__ = "search_logs"

    id: int | None = Field(default=None, primary_key=True)
    query: str = Field(sa_column_kwargs={"nullable": False})
    search_type: str = Field(sa_column_kwargs={"nullable": False})
    results_count: int = Field(default=0)
    response_time: float = Field(default=0.0)
    search_metadata: str | None = Field(default=None, alias="metadata")  # JSON 字符串
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserFeedback(SQLModel, table=True):
    """用户反馈记录模型"""
    __tablename__ = "user_feedback"

    id: int | None = Field(default=None, primary_key=True)
    query: str = Field(sa_column_kwargs={"nullable": False})
    feedback_type: str = Field(sa_column_kwargs={"nullable": False})  # positive, negative, correction
    feedback_text: str | None = Field(default=None)
    rating: int | None = Field(default=None)  # 1-5 星
    feedback_metadata: str | None = Field(default=None, alias="metadata")  # JSON 字符串
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SQLiteGateway:
    """SQLite 网关

    提供专门的 SQLite 数据访问接口，封装常见的数据库操作模式。
    """

    def __init__(self, database_manager):
        """初始化 SQLite 网关"""
        self.db_manager = database_manager
        self.logger = get_logger(__name__)

    # 文档操作
    async def create_document(
        self,
        doc_id: str,
        content: str,
        source: str = None,
        doc_type: str = None,
        metadata: dict[str, Any] = None,
        embedding_vector: list[float] = None,
    ) -> DocumentRecord:
        """创建文档记录"""
        try:
            # 序列化元数据和向量
            metadata_json = json.dumps(metadata) if metadata else None
            embedding_json = json.dumps(embedding_vector) if embedding_vector else None

            # 创建文档记录
            doc_record = DocumentRecord(
                doc_id=doc_id,
                content=content,
                source=source,
                doc_type=doc_type,
                metadata=metadata_json,
                embedding_vector=embedding_json,
                updated_at=datetime.utcnow(),
            )

            # 保存到数据库
            result = await self.db_manager.create(doc_record)
            self.logger.info(f"文档 {doc_id} 创建成功")
            return result

        except Exception as e:
            self.logger.error(f"创建文档失败: {e}")
            raise DatabaseGatewayError(f"创建文档失败: {e}", "documents", "create") from e

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        """获取文档记录"""
        try:
            result = await self.db_manager.get_by_id(DocumentRecord, doc_id)
            if result and result.doc_metadata:
                result.doc_metadata = json.loads(result.doc_metadata)
            if result and result.embedding_vector:
                result.embedding_vector = json.loads(result.embedding_vector)
            return result
        except Exception as e:
            self.logger.error(f"获取文档失败: {e}")
            raise DatabaseGatewayError(f"获取文档失败: {e}", "documents", "get") from e

    async def update_document(
        self,
        doc_id: str,
        update_data: dict[str, Any],
    ) -> DocumentRecord | None:
        """更新文档记录"""
        try:
            # 序列化特殊字段
            if "doc_metadata" in update_data and update_data["doc_metadata"]:
                update_data["doc_metadata"] = json.dumps(update_data["doc_metadata"])
            if "embedding_vector" in update_data and update_data["embedding_vector"]:
                update_data["embedding_vector"] = json.dumps(update_data["embedding_vector"])

            update_data["updated_at"] = datetime.utcnow()

            result = await self.db_manager.update(DocumentRecord, doc_id, update_data)
            if result and result.doc_metadata:
                result.doc_metadata = json.loads(result.doc_metadata)
            if result and result.embedding_vector:
                result.embedding_vector = json.loads(result.embedding_vector)
            return result

        except Exception as e:
            self.logger.error(f"更新文档失败: {e}")
            raise DatabaseGatewayError(f"更新文档失败: {e}", "documents", "update") from e

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档记录"""
        try:
            # 删除相关的块、实体和关系
            await self._delete_related_records(doc_id)

            # 删除文档
            return await self.db_manager.delete(DocumentRecord, doc_id)

        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            raise DatabaseGatewayError(f"删除文档失败: {e}", "documents", "delete") from e

    async def _delete_related_records(self, doc_id: str) -> None:
        """删除相关的记录"""
        try:
            # 删除文档块
            chunks = await self.get_document_chunks(doc_id)
            for chunk in chunks:
                await self.db_manager.delete(DocumentChunk, chunk.id)

            # 删除实体和关系
            entities = await self.get_document_entities(doc_id)
            for entity in entities:
                # 删除实体的关系
                relationships = await self.get_entity_relationships(entity.entity_id)
                for rel in relationships:
                    await self.db_manager.delete(EntityRelationship, rel.id)
                # 删除实体
                await self.db_manager.delete(Entity, entity.id)

        except Exception as e:
            self.logger.warning(f"删除相关记录时出错: {e}")

    async def list_documents(
        self,
        doc_type: str = None,
        source: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """列出文档"""
        try:
            query = select(DocumentRecord)

            if doc_type:
                query = query.where(DocumentRecord.doc_type == doc_type)
            if source:
                query = query.where(DocumentRecord.source == source)

            query = query.offset(offset).limit(limit)
            results = await self.db_manager.execute_query(query)
            documents = results.fetchall()

            # 反序列化元数据
            for doc in documents:
                if doc.doc_metadata:
                    doc.doc_metadata = json.loads(doc.doc_metadata)
                if doc.embedding_vector:
                    doc.embedding_vector = json.loads(doc.embedding_vector)

            return documents

        except Exception as e:
            self.logger.error(f"列出文档失败: {e}")
            raise DatabaseGatewayError(f"列出文档失败: {e}", "documents", "list") from e

    # 文档块操作
    async def create_document_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        chunk_order: int,
        metadata: dict[str, Any] = None,
        embedding_vector: list[float] = None,
    ) -> DocumentChunk:
        """创建文档块记录"""
        try:
            # 序列化元数据和向量
            metadata_json = json.dumps(metadata) if metadata else None
            embedding_json = json.dumps(embedding_vector) if embedding_vector else None

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                chunk_order=chunk_order,
                metadata=metadata_json,
                embedding_vector=embedding_json,
            )

            result = await self.db_manager.create(chunk)
            self.logger.info(f"文档块 {chunk_id} 创建成功")
            return result

        except Exception as e:
            self.logger.error(f"创建文档块失败: {e}")
            raise DatabaseGatewayError(f"创建文档块失败: {e}", "document_chunks", "create") from e

    async def get_document_chunks(self, doc_id: str) -> list[DocumentChunk]:
        """获取文档的所有块"""
        try:
            query = select(DocumentChunk).where(DocumentChunk.doc_id == doc_id)
            results = await self.db_manager.execute_query(query)
            chunks = results.fetchall()

            # 反序列化元数据
            for chunk in chunks:
                if chunk.metadata:
                    chunk.metadata = json.loads(chunk.metadata)
                if chunk.embedding_vector:
                    chunk.embedding_vector = json.loads(chunk.embedding_vector)

            return chunks

        except Exception as e:
            self.logger.error(f"获取文档块失败: {e}")
            raise DatabaseGatewayError(f"获取文档块失败: {e}", "document_chunks", "get") from e

    # 实体操作
    async def create_entity(
        self,
        entity_id: str,
        doc_id: str,
        name: str,
        entity_type: str,
        description: str = None,
        metadata: dict[str, Any] = None,
    ) -> Entity:
        """创建实体记录"""
        try:
            # 序列化元数据
            metadata_json = json.dumps(metadata) if metadata else None

            entity = Entity(
                entity_id=entity_id,
                doc_id=doc_id,
                name=name,
                entity_type=entity_type,
                description=description,
                metadata=metadata_json,
            )

            result = await self.db_manager.create(entity)
            self.logger.info(f"实体 {entity_id} 创建成功")
            return result

        except Exception as e:
            self.logger.error(f"创建实体失败: {e}")
            raise DatabaseGatewayError(f"创建实体失败: {e}", "entities", "create") from e

    async def get_document_entities(self, doc_id: str) -> list[Entity]:
        """获取文档的所有实体"""
        try:
            query = select(Entity).where(Entity.doc_id == doc_id)
            results = await self.db_manager.execute_query(query)
            entities = results.fetchall()

            # 反序列化元数据
            for entity in entities:
                if entity.metadata:
                    entity.metadata = json.loads(entity.metadata)

            return entities

        except Exception as e:
            self.logger.error(f"获取文档实体失败: {e}")
            raise DatabaseGatewayError(f"获取文档实体失败: {e}", "entities", "get") from e

    async def create_entity_relationship(
        self,
        relationship_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        description: str = None,
        strength: float = 1.0,
        metadata: dict[str, Any] = None,
    ) -> EntityRelationship:
        """创建实体关系记录"""
        try:
            # 序列化元数据
            metadata_json = json.dumps(metadata) if metadata else None

            relationship = EntityRelationship(
                relationship_id=relationship_id,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relationship_type=relationship_type,
                description=description,
                strength=strength,
                metadata=metadata_json,
            )

            result = await self.db_manager.create(relationship)
            self.logger.info(f"实体关系 {relationship_id} 创建成功")
            return result

        except Exception as e:
            self.logger.error(f"创建实体关系失败: {e}")
            raise DatabaseGatewayError(f"创建实体关系失败: {e}", "entity_relationships", "create") from e

    async def get_entity_relationships(self, entity_id: str) -> list[EntityRelationship]:
        """获取实体的所有关系"""
        try:
            query = select(EntityRelationship).where(
                (EntityRelationship.source_entity_id == entity_id) |
                (EntityRelationship.target_entity_id == entity_id)
            )
            results = await self.db_manager.execute_query(query)
            relationships = results.fetchall()

            # 反序列化元数据
            for rel in relationships:
                if rel.metadata:
                    rel.metadata = json.loads(rel.metadata)

            return relationships

        except Exception as e:
            self.logger.error(f"获取实体关系失败: {e}")
            raise DatabaseGatewayError(f"获取实体关系失败: {e}", "entity_relationships", "get") from e

    # 搜索日志操作
    async def log_search(
        self,
        query: str,
        search_type: str,
        results_count: int = 0,
        response_time: float = 0.0,
        metadata: dict[str, Any] = None,
    ) -> SearchLog:
        """记录搜索日志"""
        try:
            # 序列化元数据
            metadata_json = json.dumps(metadata) if metadata else None

            search_log = SearchLog(
                query=query,
                search_type=search_type,
                results_count=results_count,
                response_time=response_time,
                metadata=metadata_json,
            )

            result = await self.db_manager.create(search_log)
            self.logger.debug(f"搜索日志记录成功: {query}")
            return result

        except Exception as e:
            self.logger.error(f"记录搜索日志失败: {e}")
            # 搜索日志失败不应该影响主要功能，所以只记录错误
            return None

    # 用户反馈操作
    async def add_feedback(
        self,
        query: str,
        feedback_type: str,
        feedback_text: str = None,
        rating: int = None,
        metadata: dict[str, Any] = None,
    ) -> UserFeedback:
        """添加用户反馈"""
        try:
            # 序列化元数据
            metadata_json = json.dumps(metadata) if metadata else None

            feedback = UserFeedback(
                query=query,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                rating=rating,
                metadata=metadata_json,
            )

            result = await self.db_manager.create(feedback)
            self.logger.info(f"用户反馈记录成功: {feedback_type}")
            return result

        except Exception as e:
            self.logger.error(f"添加用户反馈失败: {e}")
            raise DatabaseGatewayError(f"添加用户反馈失败: {e}", "user_feedback", "create") from e

    # 批量操作
    async def bulk_create_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[DocumentRecord]:
        """批量创建文档"""
        try:
            doc_records = []
            for doc_data in documents:
                # 序列化元数据和向量
                if "metadata" in doc_data and doc_data["metadata"]:
                    doc_data["metadata"] = json.dumps(doc_data["metadata"])
                if "embedding_vector" in doc_data and doc_data["embedding_vector"]:
                    doc_data["embedding_vector"] = json.dumps(doc_data["embedding_vector"])

                doc_record = DocumentRecord(
                    updated_at=datetime.utcnow(),
                    **doc_data
                )
                doc_records.append(doc_record)

            results = await self.db_manager.bulk_create(doc_records)

            # 反序列化结果
            for result in results:
                if result.metadata:
                    result.metadata = json.loads(result.metadata)
                if result.embedding_vector:
                    result.embedding_vector = json.loads(result.embedding_vector)

            self.logger.info(f"批量创建 {len(results)} 个文档成功")
            return results

        except Exception as e:
            self.logger.error(f"批量创建文档失败: {e}")
            raise DatabaseGatewayError(f"批量创建文档失败: {e}", "documents", "bulk_create") from e

    # 统计和分析
    async def get_document_stats(self) -> dict[str, Any]:
        """获取文档统计信息"""
        try:
            # 文档总数
            total_docs = await self.db_manager.count(DocumentRecord)

            # 按类型统计
            query = select(DocumentRecord.doc_type, DocumentRecord.__table__.c.count())
            query = query.select_from(DocumentRecord.__table__)
            query = query.group_by(DocumentRecord.doc_type)
            results = await self.db_manager.execute_query(query)
            type_stats = dict(results.fetchall())

            # 按来源统计
            query = select(DocumentRecord.source, DocumentRecord.__table__.c.count())
            query = query.select_from(DocumentRecord.__table__)
            query = query.group_by(DocumentRecord.source)
            results = await self.db_manager.execute_query(query)
            source_stats = dict(results.fetchall())

            return {
                "total_documents": total_docs,
                "by_type": type_stats,
                "by_source": source_stats,
            }

        except Exception as e:
            self.logger.error(f"获取文档统计失败: {e}")
            raise DatabaseGatewayError(f"获取文档统计失败: {e}", "stats", "get") from e

    async def get_search_stats(self, days: int = 30) -> dict[str, Any]:
        """获取搜索统计信息"""
        try:
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # 总搜索次数
            query = select(SearchLog.__table__).where(SearchLog.created_at >= cutoff_date)
            results = await self.db_manager.execute_query(query)
            total_searches = len(results.fetchall())

            # 按搜索类型统计
            query = select(SearchLog.search_type, SearchLog.__table__.c.count())
            query = query.select_from(SearchLog.__table__)
            query = query.where(SearchLog.created_at >= cutoff_date)
            query = query.group_by(SearchLog.search_type)
            results = await self.db_manager.execute_query(query)
            type_stats = dict(results.fetchall())

            # 平均响应时间
            query = select(SearchLog.__table__).where(SearchLog.created_at >= cutoff_date)
            results = await self.db_manager.execute_query(query)
            search_logs = results.fetchall()
            avg_response_time = sum(log.response_time for log in search_logs) / len(search_logs) if search_logs else 0

            return {
                "total_searches": total_searches,
                "by_type": type_stats,
                "avg_response_time": avg_response_time,
                "period_days": days,
            }

        except Exception as e:
            self.logger.error(f"获取搜索统计失败: {e}")
            raise DatabaseGatewayError(f"获取搜索统计失败: {e}", "stats", "get") from e
