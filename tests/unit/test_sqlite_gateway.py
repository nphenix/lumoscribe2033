"""
SQLite 网关单元测试
"""

import pytest
from sqlmodel import SQLModel

from src.framework.storage.database import DatabaseManager
from src.framework.storage.sqlite_gateway import DocumentRecord, SQLiteGateway


class TestSQLiteGateway:
    """SQLite 网关测试类"""

    @pytest.fixture
    def db_manager(self):
        """创建数据库管理器实例"""
        # 使用内存数据库进行测试
        manager = DatabaseManager("sqlite:///:memory:")
        manager.create_tables()
        return manager

    @pytest.fixture
    def sqlite_gateway(self, db_manager):
        """创建 SQLite 网关实例"""
        return SQLiteGateway(db_manager)

    @pytest.mark.asyncio
    async def test_create_document(self, sqlite_gateway):
        """测试创建文档"""
        doc_id = "test_doc_1"
        content = "这是一个测试文档内容"
        metadata = {"author": "test", "version": "1.0"}

        result = await sqlite_gateway.create_document(
            doc_id=doc_id,
            content=content,
            source="test_source",
            doc_type="test_type",
            metadata=metadata
        )

        assert result.doc_id == doc_id
        assert result.content == content
        assert result.source == "test_source"
        assert result.doc_type == "test_type"
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_get_document(self, sqlite_gateway):
        """测试获取文档"""
        # 先创建文档
        doc_id = "test_doc_2"
        content = "测试文档内容"
        metadata = {"category": "test"}

        await sqlite_gateway.create_document(
            doc_id=doc_id,
            content=content,
            metadata=metadata
        )

        # 获取文档
        result = await sqlite_gateway.get_document(doc_id)

        assert result is not None
        assert result.doc_id == doc_id
        assert result.content == content
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_update_document(self, sqlite_gateway):
        """测试更新文档"""
        # 先创建文档
        doc_id = "test_doc_3"
        await sqlite_gateway.create_document(
            doc_id=doc_id,
            content="原始内容"
        )

        # 更新文档
        update_data = {
            "content": "更新后的内容",
            "source": "updated_source",
            "metadata": {"updated": True}
        }

        result = await sqlite_gateway.update_document(doc_id, update_data)

        assert result is not None
        assert result.content == "更新后的内容"
        assert result.source == "updated_source"
        assert result.metadata == {"updated": True}

    @pytest.mark.asyncio
    async def test_delete_document(self, sqlite_gateway):
        """测试删除文档"""
        # 先创建文档
        doc_id = "test_doc_4"
        await sqlite_gateway.create_document(
            doc_id=doc_id,
            content="要删除的文档"
        )

        # 删除文档
        result = await sqlite_gateway.delete_document(doc_id)
        assert result is True

        # 验证文档已被删除
        deleted_doc = await sqlite_gateway.get_document(doc_id)
        assert deleted_doc is None

    @pytest.mark.asyncio
    async def test_list_documents(self, sqlite_gateway):
        """测试列出文档"""
        # 创建多个文档
        for i in range(3):
            await sqlite_gateway.create_document(
                doc_id=f"doc_{i}",
                content=f"文档内容 {i}",
                doc_type="test",
                source="batch_test"
            )

        # 列出文档
        documents = await sqlite_gateway.list_documents(
            doc_type="test",
            source="batch_test"
        )

        assert isinstance(documents, list)
        assert len(documents) == 3

        # 验证文档类型过滤
        for doc in documents:
            assert doc.doc_type == "test"

    @pytest.mark.asyncio
    async def test_bulk_create_documents(self, sqlite_gateway):
        """测试批量创建文档"""
        documents_data = [
            {
                "doc_id": "bulk_doc_1",
                "content": "批量文档 1",
                "doc_type": "bulk",
                "metadata": {"batch": "test"}
            },
            {
                "doc_id": "bulk_doc_2",
                "content": "批量文档 2",
                "doc_type": "bulk",
                "metadata": {"batch": "test"}
            },
            {
                "doc_id": "bulk_doc_3",
                "content": "批量文档 3",
                "doc_type": "bulk",
                "metadata": {"batch": "test"}
            }
        ]

        results = await sqlite_gateway.bulk_create_documents(documents_data)

        assert isinstance(results, list)
        assert len(results) == 3

        # 验证创建的文档
        for i, result in enumerate(results):
            assert result.doc_id == f"bulk_doc_{i+1}"
            assert result.doc_type == "bulk"
            assert result.metadata == {"batch": "test"}

    @pytest.mark.asyncio
    async def test_document_stats(self, sqlite_gateway):
        """测试文档统计"""
        # 创建不同类型的文档
        await sqlite_gateway.create_document(
            doc_id="doc_type1_1",
            content="类型1文档",
            doc_type="type1"
        )
        await sqlite_gateway.create_document(
            doc_id="doc_type1_2",
            content="类型1文档2",
            doc_type="type1"
        )
        await sqlite_gateway.create_document(
            doc_id="doc_type2_1",
            content="类型2文档",
            doc_type="type2"
        )

        stats = await sqlite_gateway.get_document_stats()

        assert "total_documents" in stats
        assert "by_type" in stats
        assert "by_source" in stats

        assert stats["total_documents"] == 3
        assert stats["by_type"]["type1"] == 2
        assert stats["by_type"]["type2"] == 1

    @pytest.mark.asyncio
    async def test_log_search(self, sqlite_gateway):
        """测试搜索日志"""
        metadata = {"user_id": "test_user", "session_id": "test_session"}

        result = await sqlite_gateway.log_search(
            query="测试查询",
            search_type="vector",
            results_count=5,
            response_time=0.123,
            metadata=metadata
        )

        assert result.query == "测试查询"
        assert result.search_type == "vector"
        assert result.results_count == 5
        assert result.response_time == 0.123
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_add_feedback(self, sqlite_gateway):
        """测试添加反馈"""
        metadata = {"rating_scale": 5}

        result = await sqlite_gateway.add_feedback(
            query="测试查询",
            feedback_type="positive",
            feedback_text="搜索结果很好",
            rating=5,
            metadata=metadata
        )

        assert result.query == "测试查询"
        assert result.feedback_type == "positive"
        assert result.feedback_text == "搜索结果很好"
        assert result.rating == 5
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_document_with_embedding(self, sqlite_gateway):
        """测试带嵌入向量的文档"""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = await sqlite_gateway.create_document(
            doc_id="doc_with_embedding",
            content="带嵌入向量的文档",
            embedding_vector=embedding
        )

        assert result.doc_id == "doc_with_embedding"
        assert result.embedding_vector == embedding

        # 验证获取文档时嵌入向量被正确反序列化
        retrieved_doc = await sqlite_gateway.get_document("doc_with_embedding")
        assert retrieved_doc.embedding_vector == embedding

    @pytest.mark.asyncio
    async def test_nonexistent_document(self, sqlite_gateway):
        """测试获取不存在的文档"""
        result = await sqlite_gateway.get_document("nonexistent_doc")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_nonexistent_document(self, sqlite_gateway):
        """测试更新不存在的文档"""
        result = await sqlite_gateway.update_document(
            "nonexistent_doc",
            {"content": "新内容"}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, sqlite_gateway):
        """测试删除不存在的文档"""
        result = await sqlite_gateway.delete_document("nonexistent_doc")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
