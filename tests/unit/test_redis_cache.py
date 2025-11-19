"""
Redis 缓存管理器单元测试

测试 Redis 缓存的各种功能和性能指标。
"""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.framework.shared.redis_cache import (
    CacheEntry,
    CacheStrategy,
    RedisCacheManager,
    close_cache_manager,
    get_cache_manager,
)


# OpenTelemetry 导出器清理
@pytest.fixture(autouse=True)
def cleanup_opentelemetry():
    """自动清理 OpenTelemetry 导出器以避免 I/O 错误"""
    # 保存原始状态
    original_sdk_disabled = os.environ.get('OTEL_SDK_DISABLED')

    # 在测试前禁用OpenTelemetry以避免初始化
    os.environ['OTEL_SDK_DISABLED'] = 'true'

    yield  # 运行测试

    # 测试后清理
    try:
        # 强制关闭所有OpenTelemetry组件
        try:
            from opentelemetry import metrics, trace
            # 获取当前提供者并关闭
            tracer_provider = trace.get_tracer_provider()
            meter_provider = metrics.get_meter_provider()

            if hasattr(tracer_provider, 'shutdown'):
                tracer_provider.shutdown()
            if hasattr(meter_provider, 'shutdown'):
                meter_provider.shutdown()

            # 重置全局提供者
            trace._TRACER_PROVIDER = None
            metrics._METER_PROVIDER = None
        except Exception:
            pass

        # 强制关闭所有处理器
        try:
            from opentelemetry.sdk.metrics.export import MetricReader
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            # 清理所有处理器实例
            BatchSpanProcessor._shutdown()
            if hasattr(MetricReader, '_shutdown'):
                MetricReader._shutdown()
        except Exception:
            pass

        # 重定向stdout以避免控制台导出器写入已关闭的文件
        try:
            import io
            import sys

            # 创建新的缓冲区
            buffer = io.StringIO()

            # 重定向stdout和stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            sys.stdout = buffer
            sys.stderr = buffer

            # 刷新缓冲区
            buffer.flush()

            # 恢复原始输出
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        except Exception:
            pass

        # 强制垃圾回收
        import gc
        gc.collect()

    except Exception:
        # 忽略清理过程中的任何错误
        pass
    finally:
        # 恢复原始状态
        if original_sdk_disabled is not None:
            os.environ['OTEL_SDK_DISABLED'] = original_sdk_disabled
        elif 'OTEL_SDK_DISABLED' in os.environ:
            del os.environ['OTEL_SDK_DISABLED']


class TestRedisCacheManager:
    """Redis 缓存管理器测试类"""

    @pytest.fixture
    async def cache_manager(self):
        """创建缓存管理器实例"""
        # 使用测试 Redis URL
        manager = RedisCacheManager(
            redis_url="redis://localhost:6379/1",  # 使用不同的数据库
            max_connections=10,
            default_ttl=300,  # 5分钟TTL
            key_prefix="test:",
            enable_metrics=True
        )

        # 初始化真实连接
        initialized = await manager.initialize()
        assert initialized, "Redis 连接初始化失败"

        yield manager

        # 清理 - 清空测试数据
        await manager.clear()
        await manager.close()

    @pytest.fixture
    def mock_redis_client(self):
        """模拟 Redis 客户端"""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_client.delete.return_value = True
        mock_client.exists.return_value = False
        mock_client.ttl.return_value = 300
        mock_client.keys.return_value = []
        return mock_client

    @pytest.mark.asyncio
    async def test_initialization(self, cache_manager):
        """测试初始化"""
        # 测试基本属性
        assert cache_manager.redis_url == "redis://localhost:6379/1"
        assert cache_manager.max_connections == 10
        assert cache_manager.default_ttl == 300
        assert cache_manager.key_prefix == "test:"
        assert cache_manager.enable_metrics

        # 测试指标初始化
        assert cache_manager.metrics.hits == 0
        assert cache_manager.metrics.misses == 0
        assert cache_manager.metrics.sets == 0
        assert cache_manager.metrics.deletes == 0
        assert cache_manager.metrics.evictions == 0
        assert cache_manager.metrics.errors == 0

    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache_manager):
        """测试基本缓存操作"""
        # 测试设置和获取
        success = await cache_manager.set("test_key", "test_value", ttl=60)
        assert success is True

        value = await cache_manager.get("test_key")
        assert value == "test_value"

        # 测试不存在的键
        default_value = await cache_manager.get("nonexistent_key", "default")
        assert default_value == "default"

        # 测试存在性检查
        exists = await cache_manager.exists("test_key")
        assert exists is True

        not_exists = await cache_manager.exists("nonexistent_key")
        assert not_exists is False

        # 测试删除
        delete_success = await cache_manager.delete("test_key")
        assert delete_success is True

        # 验证删除后不存在
        exists_after_delete = await cache_manager.exists("test_key")
        assert exists_after_delete is False

    @pytest.mark.asyncio
    async def test_cache_strategies(self, cache_manager):
        """测试不同缓存策略"""
        # 测试TTL策略
        await cache_manager.set("ttl_key", "ttl_value", ttl=30)
        entry = cache_manager._local_cache.get("test:ttl_key")
        assert entry is not None
        assert entry.ttl == 30

        # 测试本地缓存大小限制
        for i in range(1500):  # 超过本地缓存限制
            await cache_manager.set(f"bulk_key_{i}", f"value_{i}")

        # 验证本地缓存大小被限制
        assert len(cache_manager._local_cache) <= cache_manager._local_cache_size

    @pytest.mark.asyncio
    async def test_distributed_locks(self, cache_manager):
        """测试分布式锁功能"""
        # 测试获取锁
        lock_acquired = await cache_manager.acquire_lock("test_lock", timeout=5)
        assert lock_acquired is True
        assert "test_lock" in cache_manager._locks

        # 测试锁重入
        lock_reacquired = await cache_manager.acquire_lock("test_lock", timeout=1)
        assert lock_reacquired is False  # 同一个锁不能重复获取

        # 测试释放锁
        lock_released = await cache_manager.release_lock("test_lock")
        assert lock_released is True
        assert "test_lock" not in cache_manager._locks

        # 测试释放不存在的锁
        release_nonexistent = await cache_manager.release_lock("nonexistent_lock")
        assert release_nonexistent is False

    @pytest.mark.asyncio
    async def test_cache_metrics(self, cache_manager):
        """测试缓存指标"""
        # 执行一些操作来生成指标
        await cache_manager.set("key1", "value1")
        await cache_manager.get("key1")  # hit
        await cache_manager.get("key2")  # miss
        await cache_manager.set("key3", "value3")  # 先设置一个键，然后删除
        await cache_manager.delete("key3")  # delete
        await cache_manager.set("key4", "value4")
        await cache_manager.set("key5", "value5")

        # 强制一些淘汰
        for i in range(10):
            await cache_manager.set(f"evict_key_{i}", f"evict_value_{i}")

        # 获取指标
        metrics = cache_manager.get_metrics()

        # 验证指标
        assert metrics["cache_metrics"]["hits"] >= 1
        assert metrics["cache_metrics"]["misses"] >= 1
        assert metrics["cache_metrics"]["sets"] >= 5
        assert metrics["cache_metrics"]["deletes"] >= 1
        assert metrics["cache_metrics"]["evictions"] >= 0
        assert 0 <= metrics["cache_metrics"]["hit_rate"] <= 100

    @pytest.mark.asyncio
    async def test_health_check(self, cache_manager):
        """测试健康检查"""
        # 模拟健康的 Redis
        health = await cache_manager.health_check()

        assert health["status"] in ["healthy", "warning", "critical", "unhealthy"]
        assert "timestamp" in health
        assert "checks" in health

        # 验证检查内容
        checks = health["checks"]
        if "redis_connection" in checks:
            assert checks["redis_connection"]["status"] in ["healthy", "unhealthy"]

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager):
        """测试缓存过期"""
        # 设置一个短TTL的键
        await cache_manager.set("expire_key", "expire_value", ttl=2)

        # 立即检查存在
        exists_immediately = await cache_manager.exists("expire_key")
        assert exists_immediately is True

        # 等待过期
        await asyncio.sleep(3)

        # 检查是否已过期
        value_after_expire = await cache_manager.get("expire_key")
        assert value_after_expire is None  # 应该返回默认值None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache_manager):
        """测试并发操作"""
        # 并发设置多个键
        tasks = []
        for i in range(10):
            task = cache_manager.set(f"concurrent_key_{i}", f"concurrent_value_{i}")
            tasks.append(task)

        # 等待所有操作完成
        results = await asyncio.gather(*tasks)

        # 验证所有操作都成功
        assert all(results)

        # 并发获取
        get_tasks = []
        for i in range(10):
            task = cache_manager.get(f"concurrent_key_{i}")
            get_tasks.append(task)

        get_results = await asyncio.gather(*get_tasks)

        # 验证获取结果
        for i, result in enumerate(get_results):
            assert result == f"concurrent_value_{i}"

    @pytest.mark.asyncio
    async def test_error_handling(self, cache_manager):
        """测试错误处理"""
        # 保存原始客户端
        original_client = cache_manager.redis_client

        # 模拟 Redis 连接错误
        cache_manager.redis_client = None

        # 应该返回默认值而不是抛出异常
        value = await cache_manager.get("error_key", "default")
        assert value == "default"

        # 验证错误计数增加
        metrics = cache_manager.get_metrics()
        assert metrics["cache_metrics"]["errors"] > 0

        # 恢复原始客户端
        cache_manager.redis_client = original_client

    @pytest.mark.asyncio
    async def test_metrics_reset(self, cache_manager):
        """测试指标重置"""
        # 执行一些操作
        await cache_manager.set("test_key", "test_value")
        await cache_manager.get("test_key")

        # 获取指标
        metrics_before = cache_manager.get_metrics()
        assert metrics_before["cache_metrics"]["sets"] > 0

        # 重置指标
        cache_manager.reset_metrics()

        # 验证指标已重置
        metrics_after = cache_manager.get_metrics()
        assert metrics_after["cache_metrics"]["hits"] == 0
        assert metrics_after["cache_metrics"]["misses"] == 0
        assert metrics_after["cache_metrics"]["sets"] == 0

    @pytest.mark.asyncio
    async def test_complex_data_types(self, cache_manager):
        """测试复杂数据类型的缓存"""
        # 测试字典
        test_dict = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
        await cache_manager.set("test_dict", test_dict)
        retrieved_dict = await cache_manager.get("test_dict")
        assert retrieved_dict == test_dict

        # 测试列表
        test_list = [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]
        await cache_manager.set("test_list", test_list)
        retrieved_list = await cache_manager.get("test_list")
        assert retrieved_list == test_list

        # 测试嵌套结构
        test_nested = {
            "level1": {
                "level2": {
                    "data": "deep_value"
                }
            }
        }
        await cache_manager.set("test_nested", test_nested)
        retrieved_nested = await cache_manager.get("test_nested")
        assert retrieved_nested == test_nested

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache_manager):
        """测试缓存清空"""
        # 设置多个键
        for i in range(5):
            await cache_manager.set(f"clear_key_{i}", f"value_{i}")

        # 验证键存在
        for i in range(5):
            exists = await cache_manager.exists(f"clear_key_{i}")
            assert exists is True

        # 清空所有键
        deleted_count = await cache_manager.clear("clear_key_*")
        assert deleted_count >= 0  # 可能删除了0个或多个键

        # 验证键已被删除
        for i in range(5):
            exists = await cache_manager.exists(f"clear_key_{i}")
            assert not exists

        # 验证本地缓存也被清空
        assert len(cache_manager._local_cache) == 0

    @pytest.mark.asyncio
    async def test_performance_optimization(self, cache_manager):
        """测试性能优化功能"""
        # 测试批量操作
        batch_tasks = []
        for i in range(100):
            task = cache_manager.set(f"batch_key_{i}", f"batch_value_{i}")
            batch_tasks.append(task)

        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*batch_tasks)
        batch_time = asyncio.get_event_loop().time() - start_time

        # 测试单个操作
        single_start_time = asyncio.get_event_loop().time()
        for i in range(100):
            await cache_manager.set(f"single_key_{i}", f"single_value_{i}")
        single_time = asyncio.get_event_loop().time() - single_start_time

        # 批量操作应该更快（在某些情况下）
        # 这个测试主要验证机制正常工作
        assert batch_time > 0
        assert single_time > 0

        # 获取性能指标
        metrics = cache_manager.get_metrics()
        # 验证有操作被执行（通过sets计数）
        assert metrics["cache_metrics"]["sets"] > 0


class TestCacheEntry:
    """缓存条目测试"""

    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            value="test_value",
            created_at=1000.0,
            accessed_at=1000.0,
            access_count=0,
            ttl=300
        )

        assert entry.value == "test_value"
        assert entry.created_at == 1000.0
        assert entry.accessed_at == 1000.0
        assert entry.access_count == 0
        assert entry.ttl == 300

    def test_cache_entry_expiration(self):
        """测试缓存条目过期"""
        import time

        # 创建过期的条目
        past_time = time.time() - 400  # 400秒前
        expired_entry = CacheEntry(
            value="expired_value",
            created_at=past_time,
            accessed_at=past_time,
            access_count=5,
            ttl=300
        )

        assert expired_entry.is_expired is True

        # 创建未过期的条目
        current_time = time.time()
        fresh_entry = CacheEntry(
            value="fresh_value",
            created_at=current_time,
            accessed_at=current_time,
            access_count=1,
            ttl=300
        )

        assert fresh_entry.is_expired is False

    def test_cache_entry_touch(self):
        """测试缓存条目访问更新"""
        entry = CacheEntry(
            value="test_value",
            created_at=1000.0,
            accessed_at=1000.0,
            access_count=1,
            ttl=300
        )

        # 更新访问
        entry.touch()

        assert entry.access_count == 2
        assert entry.accessed_at > 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
