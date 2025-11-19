"""
SQLite网关单元测试

测试SQLite数据库网关的各项功能，包括：
- 数据库连接管理
- CRUD操作
- 查询优化
- 事务处理
- 错误处理
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from src.framework.shared.exceptions import DatabaseError, ValidationError
from src.framework.storage.sqlite_gateway import SQLiteGateway


class TestSQLiteGateway:
    """测试SQLite网关"""

    def setup_method(self):
        """测试前设置"""
        # 创建临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        
        # 初始化网关
        self.gateway = SQLiteGateway(database_url=f"sqlite:///{self.db_path}")

    def teardown_method(self):
        """测试后清理"""
        # 关闭连接
        if hasattr(self.gateway, '_connection') and self.gateway._connection:
            self.gateway._connection.close()
        
        # 删除临时文件
        try:
            self.temp_db.close()
            Path(self.db_path).unlink(missing_ok=True)
        except:
            pass

    @pytest.mark.asyncio
    async def test_initialize(self):
        """测试初始化"""
        await self.gateway.initialize()
        
        assert self.gateway._connection is not None
        assert self.gateway._cursor is not None

    @pytest.mark.asyncio
    async def test_create_table(self):
        """测试创建表"""
        await self.gateway.initialize()
        
        # 创建测试表
        await self.gateway.create_table(
            "test_table",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        )
        
        # 验证表是否存在
        tables = await self.gateway.get_tables()
        assert "test_table" in tables

    @pytest.mark.asyncio
    async def test_insert(self):
        """测试插入数据"""
        await self.gateway.initialize()
        
        # 创建表
        await self.gateway.create_table(
            "test_insert",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        # 插入数据
        data = {"name": "test_item", "value": 42}
        result = await self.gateway.insert("test_insert", data)
        
        assert result is not None
        assert result["name"] == "test_item"
        assert result["value"] == 42
        assert "id" in result

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        """测试根据ID获取数据"""
        await self.gateway.initialize()
        
        # 创建表和插入数据
        await self.gateway.create_table(
            "test_get",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        data = {"name": "get_test", "value": 100}
        inserted = await self.gateway.insert("test_get", data)
        
        # 获取数据
        result = await self.gateway.get_by_id("test_get", inserted["id"])
        
        assert result is not None
        assert result["name"] == "get_test"
        assert result["value"] == 100

    @pytest.mark.asyncio
    async def test_update(self):
        """测试更新数据"""
        await self.gateway.initialize()
        
        # 创建表和插入数据
        await self.gateway.create_table(
            "test_update",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        data = {"name": "update_test", "value": 50}
        inserted = await self.gateway.insert("test_update", data)
        
        # 更新数据
        update_data = {"value": 75}
        result = await self.gateway.update("test_update", inserted["id"], update_data)
        
        assert result is not None
        assert result["value"] == 75
        assert result["name"] == "update_test"  # 未更新的字段保持不变

    @pytest.mark.asyncio
    async def test_delete(self):
        """测试删除数据"""
        await self.gateway.initialize()
        
        # 创建表和插入数据
        await self.gateway.create_table(
            "test_delete",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        data = {"name": "delete_test", "value": 25}
        inserted = await self.gateway.insert("test_delete", data)
        
        # 删除数据
        result = await self.gateway.delete("test_delete", inserted["id"])
        
        assert result is True
        
        # 验证数据已删除
        deleted_item = await self.gateway.get_by_id("test_delete", inserted["id"])
        assert deleted_item is None

    @pytest.mark.asyncio
    async def test_query(self):
        """测试查询数据"""
        await self.gateway.initialize()
        
        # 创建表和插入多条数据
        await self.gateway.create_table(
            "test_query",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER",
                "category": "TEXT"
            }
        )
        
        # 插入测试数据
        test_data = [
            {"name": "item1", "value": 10, "category": "A"},
            {"name": "item2", "value": 20, "category": "B"},
            {"name": "item3", "value": 30, "category": "A"}
        ]
        
        for data in test_data:
            await self.gateway.insert("test_query", data)
        
        # 测试查询所有数据
        all_results = await self.gateway.query("SELECT * FROM test_query")
        assert len(all_results) == 3
        
        # 测试带条件的查询
        filtered_results = await self.gateway.query(
            "SELECT * FROM test_query WHERE category = ?", 
            params=("A",)
        )
        assert len(filtered_results) == 2
        
        # 测试查询结果格式
        for result in filtered_results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "name" in result
            assert "value" in result
            assert "category" in result

    @pytest.mark.asyncio
    async def test_execute_many(self):
        """测试批量执行"""
        await self.gateway.initialize()
        
        # 创建表
        await self.gateway.create_table(
            "test_batch",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        # 批量插入数据
        batch_data = [
            {"name": "batch1", "value": 1},
            {"name": "batch2", "value": 2},
            {"name": "batch3", "value": 3}
        ]
        
        result = await self.gateway.execute_many(
            "INSERT INTO test_batch (name, value) VALUES (?, ?)",
            [(d["name"], d["value"]) for d in batch_data]
        )
        
        assert result is True
        
        # 验证数据已插入
        all_results = await self.gateway.query("SELECT * FROM test_batch")
        assert len(all_results) == 3

    @pytest.mark.asyncio
    async def test_transaction(self):
        """测试事务处理"""
        await self.gateway.initialize()
        
        # 创建表
        await self.gateway.create_table(
            "test_transaction",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        # 测试事务成功
        async with self.gateway.transaction():
            await self.gateway.insert("test_transaction", {"name": "tx_item1", "value": 100})
            await self.gateway.insert("test_transaction", {"name": "tx_item2", "value": 200})
        
        # 验证数据已插入
        results = await self.gateway.query("SELECT * FROM test_transaction")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_transaction_rollback(self):
        """测试事务回滚"""
        await self.gateway.initialize()
        
        # 创建表
        await self.gateway.create_table(
            "test_rollback",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER UNIQUE"  # 添加唯一约束
            }
        )
        
        # 先插入一条数据
        await self.gateway.insert("test_rollback", {"name": "unique_item", "value": 999})
        
        # 测试事务回滚（违反唯一约束）
        with pytest.raises(DatabaseError):
            async with self.gateway.transaction():
                await self.gateway.insert("test_rollback", {"name": "unique_item", "value": 999})
        
        # 验证只有原始数据存在
        results = await self.gateway.query("SELECT * FROM test_rollback")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_table_schema(self):
        """测试获取表结构"""
        await self.gateway.initialize()
        
        # 创建表
        await self.gateway.create_table(
            "test_schema",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER DEFAULT 0"
            }
        )
        
        # 获取表结构
        schema = await self.gateway.get_table_schema("test_schema")
        
        assert isinstance(schema, dict)
        assert "columns" in schema
        assert len(schema["columns"]) == 3
        
        # 验证列信息
        columns = {col["name"]: col for col in schema["columns"]}
        assert "id" in columns
        assert "name" in columns
        assert "value" in columns

    @pytest.mark.asyncio
    async def test_health_check(self):
        """测试健康检查"""
        await self.gateway.initialize()
        
        health = await self.gateway.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "database" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_backup(self):
        """测试备份功能"""
        await self.gateway.initialize()
        
        # 创建表和数据
        await self.gateway.create_table(
            "test_backup",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER"
            }
        )
        
        await self.gateway.insert("test_backup", {"name": "backup_item", "value": 123})
        
        # 执行备份
        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir) / "backup.db"
            result = await self.gateway.backup(str(backup_path))
            
            assert result is True
            assert backup_path.exists()
            
            # 验证备份文件包含数据
            backup_gateway = SQLiteGateway(database_url=f"sqlite:///{backup_path}")
            await backup_gateway.initialize()
            
            backup_results = await backup_gateway.query("SELECT * FROM test_backup")
            assert len(backup_results) == 1
            assert backup_results[0]["name"] == "backup_item"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        await self.gateway.initialize()
        
        # 测试表不存在错误
        with pytest.raises(DatabaseError):
            await self.gateway.get_by_id("nonexistent_table", 1)
        
        # 测试SQL语法错误
        with pytest.raises(DatabaseError):
            await self.gateway.query("INVALID SQL SYNTAX")

    @pytest.mark.asyncio
    async def test_connection_pooling(self):
        """测试连接池"""
        # 测试多个并发连接
        await self.gateway.initialize()
        
        async def concurrent_operation():
            async with self.gateway.get_connection() as conn:
                result = await conn.execute("SELECT 1").fetchone()
                return result[0]
        
        # 并发执行多个操作
        tasks = [concurrent_operation() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(result == 1 for result in results)

    @pytest.mark.asyncio
    async def test_query_optimization(self):
        """测试查询优化"""
        await self.gateway.initialize()
        
        # 创建表和索引
        await self.gateway.create_table(
            "test_optimization",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "name": "TEXT NOT NULL",
                "value": "INTEGER",
                "category": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        )
        
        # 创建索引
        await self.gateway.execute(
            "CREATE INDEX idx_category ON test_optimization(category)"
        )
        
        # 插入测试数据
        test_data = [
            {"name": f"item_{i}", "value": i, "category": "A" if i % 2 == 0 else "B"}
            for i in range(100)
        ]
        
        for data in test_data:
            await self.gateway.insert("test_optimization", data)
        
        # 测试带索引的查询性能
        import time
        start_time = time.time()
        
        results = await self.gateway.query(
            "SELECT * FROM test_optimization WHERE category = ? LIMIT 10",
            params=("A",)
        )
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert len(results) == 50  # 应该返回50条记录
        assert query_time < 1.0  # 查询应该在合理时间内完成


class TestSQLiteGatewayIntegration:
    """测试SQLite网关集成"""

    def setup_method(self):
        """测试前设置"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.gateway = SQLiteGateway(database_url=f"sqlite:///{self.db_path}")

    def teardown_method(self):
        """测试后清理"""
        try:
            self.temp_db.close()
            Path(self.db_path).unlink(missing_ok=True)
        except:
            pass

    @pytest.mark.asyncio
    async def test_full_crud_workflow(self):
        """测试完整CRUD工作流"""
        await self.gateway.initialize()
        
        # 创建表
        await self.gateway.create_table(
            "workflow_test",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "title": "TEXT NOT NULL",
                "content": "TEXT",
                "status": "TEXT DEFAULT 'pending'",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        )
        
        # 创建记录
        item_data = {
            "title": "Test Item",
            "content": "This is test content",
            "status": "pending"
        }
        
        created_item = await self.gateway.insert("workflow_test", item_data)
        assert created_item["id"] is not None
        
        # 读取记录
        read_item = await self.gateway.get_by_id("workflow_test", created_item["id"])
        assert read_item["title"] == "Test Item"
        assert read_item["status"] == "pending"
        
        # 更新记录
        update_data = {"status": "completed", "content": "Updated content"}
        updated_item = await self.gateway.update("workflow_test", created_item["id"], update_data)
        assert updated_item["status"] == "completed"
        assert updated_item["content"] == "Updated content"
        
        # 删除记录
        delete_result = await self.gateway.delete("workflow_test", created_item["id"])
        assert delete_result is True
        
        # 验证删除
        deleted_item = await self.gateway.get_by_id("workflow_test", created_item["id"])
        assert deleted_item is None

    @pytest.mark.asyncio
    async def test_data_integrity(self):
        """测试数据完整性"""
        await self.gateway.initialize()
        
        # 创建带约束的表
        await self.gateway.create_table(
            "integrity_test",
            {
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "email": "TEXT NOT NULL UNIQUE",
                "username": "TEXT NOT NULL UNIQUE",
                "age": "INTEGER CHECK(age >= 0)"
            }
        )
        
        # 测试唯一约束
        user_data = {"email": "test@example.com", "username": "testuser", "age": 25}
        await self.gateway.insert("integrity_test", user_data)
        
        # 尝试插入重复邮箱应该失败
        with pytest.raises(DatabaseError):
            await self.gateway.insert("integrity_test", {
                "email": "test@example.com",  # 重复邮箱
                "username": "another_user",
                "age": 30
            })
        
        # 测试检查约束
        with pytest.raises(DatabaseError):
            await self.gateway.insert("integrity_test", {
                "email": "another@example.com",
                "username": "another_user",
                "age": -5  # 无效年龄
            })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])