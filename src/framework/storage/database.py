"""
数据库管理器

基于 SQLModel 提供统一的数据库操作接口，使用线程池处理异步操作。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from src.framework.shared.exceptions import DatabaseError


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, database_url: str = "sqlite:///data/lumoscribe.db", **kwargs):
        self.database_url = database_url
        self.engine = None
        self._session_pool = []
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._init_engine(**kwargs)

    def _init_engine(self, **kwargs):
        """初始化数据库引擎"""
        # 默认配置
        default_config = {
            "echo": False,
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": 10,  # 连接池大小
            "max_overflow": 20,  # 最大溢出连接数
            "pool_timeout": 30,  # 连接超时时间
            "connect_args": {
                "check_same_thread": False,
                "timeout": 30.0,
                "isolation_level": None,  # 启用自动提交模式
            }
        }

        # 合并用户配置
        config = {**default_config, **kwargs}

        # SQLite 特殊处理
        if self.database_url.startswith("sqlite:///"):
            # 确保数据目录存在
            import os
            from pathlib import Path
            db_path = Path(self.database_url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # 测试环境使用内存数据库
            if os.getenv("TESTING") == "1":
                config["connect_args"] = {"check_same_thread": False}
                config["poolclass"] = StaticPool
                self.database_url = "sqlite:///:memory:"

        self.engine = create_engine(self.database_url, **config)

    def create_tables(self):
        """创建所有表"""
        try:
            SQLModel.metadata.create_all(self.engine)
        except Exception as e:
            raise DatabaseError(f"创建数据库表失败: {e}", "create_tables") from e

    def drop_tables(self):
        """删除所有表"""
        try:
            SQLModel.metadata.drop_all(self.engine)
        except Exception as e:
            raise DatabaseError(f"删除数据库表失败: {e}", "drop_tables") from e

    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话"""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"数据库操作失败: {e}", "session_operation") from e
        finally:
            session.close()

    async def execute_query(self, query, params: dict[str, Any] = None):
        """执行查询"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_execute_query,
            query,
            params
        )

    def _sync_execute_query(self, query, params: dict[str, Any] = None):
        """同步执行查询"""
        try:
            with self.get_session() as session:
                if params:
                    result = session.exec(query, params)
                else:
                    result = session.exec(query)
                return result
        except Exception as e:
            raise DatabaseError(f"执行查询失败: {e}", "execute_query") from e

    async def get_by_id(
        self, model_class: type[SQLModel], record_id: Any
    ) -> SQLModel | None:
        """根据ID获取记录"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_get_by_id,
            model_class,
            record_id
        )

    def _sync_get_by_id(self, model_class: type[SQLModel], record_id: Any) -> SQLModel | None:
        """同步根据ID获取记录"""
        try:
            with self.get_session() as session:
                return session.get(model_class, record_id)
        except Exception as e:
            raise DatabaseError(f"根据ID获取记录失败: {e}", "get_by_id") from e

    async def get_all(
        self, model_class: type[SQLModel], offset: int = 0, limit: int = 100
    ) -> list[SQLModel]:
        """获取所有记录"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_get_all,
            model_class,
            offset,
            limit
        )

    def _sync_get_all(
        self, model_class: type[SQLModel], offset: int = 0, limit: int = 100
    ) -> list[SQLModel]:
        """同步获取所有记录"""
        try:
            with self.get_session() as session:
                query = select(model_class).offset(offset).limit(limit)
                result = session.exec(query)
                return result.fetchall()
        except Exception as e:
            raise DatabaseError(f"获取所有记录失败: {e}", "get_all") from e

    async def create(self, record: SQLModel) -> SQLModel:
        """创建记录"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_create,
            record
        )

    def _sync_create(self, record: SQLModel) -> SQLModel:
        """同步创建记录"""
        try:
            with self.get_session() as session:
                session.add(record)
                session.commit()
                session.refresh(record)
                return record
        except Exception as e:
            raise DatabaseError(f"创建记录失败: {e}", "create") from e

    async def update(
        self,
        model_class: type[SQLModel],
        record_id: Any,
        update_data: dict[str, Any],
    ) -> SQLModel | None:
        """更新记录"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_update,
            model_class,
            record_id,
            update_data
        )

    def _sync_update(
        self,
        model_class: type[SQLModel],
        record_id: Any,
        update_data: dict[str, Any],
    ) -> SQLModel | None:
        """同步更新记录"""
        try:
            with self.get_session() as session:
                record = session.get(model_class, record_id)
                if not record:
                    return None

                for key, value in update_data.items():
                    setattr(record, key, value)

                session.add(record)
                session.commit()
                session.refresh(record)
                return record
        except Exception as e:
            raise DatabaseError(f"更新记录失败: {e}", "update") from e

    async def delete(self, model_class: type[SQLModel], record_id: Any) -> bool:
        """删除记录"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_delete,
            model_class,
            record_id
        )

    def _sync_delete(self, model_class: type[SQLModel], record_id: Any) -> bool:
        """同步删除记录"""
        try:
            with self.get_session() as session:
                record = session.get(model_class, record_id)
                if not record:
                    return False

                session.delete(record)
                session.commit()
                return True
        except Exception as e:
            raise DatabaseError(f"删除记录失败: {e}", "delete") from e

    async def bulk_create(self, records: list[SQLModel]) -> list[SQLModel]:
        """批量创建记录"""
        try:
            with self.get_session() as session:
                session.add_all(records)
                session.commit()

                # 刷新所有记录以获取ID
                for record in records:
                    session.refresh(record)

                return records
        except Exception as e:
            raise DatabaseError(f"批量创建记录失败: {e}", "bulk_create") from e

    async def bulk_update(
        self,
        model_class: type[SQLModel],
        updates: list[dict[str, Any]],
        id_field: str = "id",
    ) -> list[SQLModel]:
        """批量更新记录"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_bulk_update,
            model_class,
            updates,
            id_field
        )

    def _sync_bulk_update(
        self,
        model_class: type[SQLModel],
        updates: list[dict[str, Any]],
        id_field: str = "id",
    ) -> list[SQLModel]:
        """同步批量更新记录"""
        try:
            with self.get_session() as session:
                updated_records = []

                for update_data in updates:
                    record_id = update_data.pop(id_field)
                    record = session.get(model_class, record_id)
                    if record:
                        for key, value in update_data.items():
                            setattr(record, key, value)
                        session.add(record)
                        updated_records.append(record)

                session.commit()

                # 清理并重新获取更新的记录
                for _i, record in enumerate(updated_records):
                    session.refresh(record)

                return updated_records
        except Exception as e:
            raise DatabaseError(f"批量更新记录失败: {e}", "bulk_update") from e

    async def query(self, model_class: type[SQLModel], **filters) -> list[SQLModel]:
        """条件查询"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_query,
            model_class,
            filters
        )

    def _sync_query(self, model_class: type[SQLModel], **filters) -> list[SQLModel]:
        """同步条件查询"""
        try:
            with self.get_session() as session:
                query = select(model_class)

                # 添加过滤条件
                for field, value in filters.items():
                    if hasattr(model_class, field):
                        query = query.where(getattr(model_class, field) == value)

                result = session.exec(query)
                return result.fetchall()
        except Exception as e:
            raise DatabaseError(f"条件查询失败: {e}", "query") from e

    async def count(self, model_class: type[SQLModel], **filters) -> int:
        """计数查询"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_count,
            model_class,
            filters
        )

    def _sync_count(self, model_class: type[SQLModel], **filters) -> int:
        """同步计数查询"""
        try:
            with self.get_session() as session:
                query = select(model_class)

                # 添加过滤条件
                for field, value in filters.items():
                    if hasattr(model_class, field):
                        query = query.where(getattr(model_class, field) == value)

                result = session.exec(query)
                return len(result.fetchall())
        except Exception as e:
            raise DatabaseError(f"计数查询失败: {e}", "count") from e

    async def exists(self, model_class: type[SQLModel], **filters) -> bool:
        """检查记录是否存在"""
        count = await self.count(model_class, **filters)
        return count > 0

    async def close(self):
        """关闭数据库连接"""
        if self.engine:
            self.engine.dispose()
        if self._executor:
            self._executor.shutdown(wait=True)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 全局数据库实例
_database_manager = None


def get_database_manager(database_url: str = None) -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _database_manager

    if _database_manager is None:
        from src.framework.shared.config import get_config
        config = get_config()
        url = database_url or getattr(config, 'database_url', "sqlite:///data/lumoscribe.db")
        _database_manager = DatabaseManager(url)

    return _database_manager


async def init_database(database_url: str = None):
    """初始化数据库"""
    db_manager = get_database_manager(database_url)
    db_manager.create_tables()
    return db_manager
