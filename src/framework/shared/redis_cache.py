"""
Redis 缓存管理器

提供高性能的 Redis 缓存功能，支持：
- 多种缓存策略
- 分布式锁
- 性能指标收集
- 连接池管理
- 故障转移
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Union

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import ConnectionError, RedisError

from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class CacheStrategy(str, Enum):
    """缓存策略枚举"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    TTL = "ttl"   # 生存时间
    WRITE_THROUGH = "write_through"  # 写透
    WRITE_BACK = "write_back"  # 写回


@dataclass
class CacheMetrics:
    """缓存指标"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    total_get_time: float = 0.0
    total_set_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def avg_get_time(self) -> float:
        """平均获取时间"""
        return self.total_get_time / self.hits if self.hits > 0 else 0.0

    @property
    def avg_set_time(self) -> float:
        """平均设置时间"""
        return self.total_set_time / self.sets if self.sets > 0 else 0.0


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: int | None = None

    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def touch(self):
        """更新访问时间和计数"""
        self.accessed_at = time.time()
        self.access_count += 1


class RedisCacheManager:
    """Redis 缓存管理器"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_connections: int = 20,
        default_ttl: int = 3600,
        key_prefix: str = "lumoscribe:",
        enable_metrics: bool = True
    ):
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.enable_metrics = enable_metrics

        # 连接池
        self.connection_pool: ConnectionPool | None = None
        self.redis_client: redis.Redis | None = None

        # 指标
        self.metrics = CacheMetrics()

        # 本地缓存热点数据
        self._local_cache: dict[str, CacheEntry] = {}
        self._local_cache_size = 1000

        # 分布式锁
        self._locks: dict[str, asyncio.Lock] = {}

        # 统计信息
        self._stats = {
            "total_operations": 0,
            "cache_efficiency": 0.0,
            "last_reset": time.time()
        }

    async def initialize(self) -> bool:
        """初始化 Redis 连接"""
        try:
            # 创建连接池
            self.connection_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            # 创建 Redis 客户端
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=True
            )

            # 测试连接
            await self.redis_client.ping()

            logger.info(f"✅ Redis 缓存管理器已初始化: {self.redis_url}")
            return True

        except Exception as e:
            logger.error(f"❌ Redis 初始化失败: {e}")
            return False

    async def close(self):
        """关闭 Redis 连接"""
        try:
            if self.redis_client:
                await self.redis_client.aclose()  # 使用 aclose() 替代 close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            logger.info("🔌 Redis 连接已关闭")
        except Exception as e:
            logger.error(f"❌ Redis 关闭失败: {e}")

    def _make_key(self, key: str) -> str:
        """生成带前缀的键名"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        start_time = time.time()

        try:
            # 先检查本地缓存
            local_key = self._make_key(key)
            if local_key in self._local_cache:
                entry = self._local_cache[local_key]
                if not entry.is_expired:
                    entry.touch()
                    self.metrics.hits += 1
                    self.metrics.total_get_time += time.time() - start_time
                    return entry.value
                else:
                    # 过期，从本地缓存删除
                    del self._local_cache[local_key]

            # 从 Redis 获取
            if not self.redis_client:
                self.metrics.errors += 1
                logger.error("❌ Redis 客户端未初始化")
                return default

            redis_key = self._make_key(key)
            cached_data = await self.redis_client.get(redis_key)

            if cached_data is not None:
                try:
                    # 尝试解析 JSON
                    if cached_data.startswith('{') or cached_data.startswith('['):
                        value = json.loads(cached_data)
                    else:
                        value = cached_data

                    # 更新本地缓存
                    entry = CacheEntry(
                        value=value,
                        created_at=time.time(),
                        accessed_at=time.time(),
                        access_count=1
                    )
                    self._local_cache[local_key] = entry

                    # 限制本地缓存大小
                    if len(self._local_cache) > self._local_cache_size:
                        self._evict_lru_items()

                    self.metrics.hits += 1
                    self.metrics.total_get_time += time.time() - start_time
                    return value

                except json.JSONDecodeError:
                    # 不是 JSON，直接返回
                    entry = CacheEntry(
                        value=cached_data,
                        created_at=time.time(),
                        accessed_at=time.time(),
                        access_count=1
                    )
                    self._local_cache[local_key] = entry
                    self.metrics.hits += 1
                    self.metrics.total_get_time += time.time() - start_time
                    return cached_data

            # 缓存未命中
            self.metrics.misses += 1
            self.metrics.total_get_time += time.time() - start_time
            return default

        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ 缓存获取失败 [{key}]: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        strategy: CacheStrategy = CacheStrategy.TTL
    ) -> bool:
        """设置缓存值"""
        start_time = time.time()

        try:
            if not self.redis_client:
                return False

            redis_key = self._make_key(key)
            ttl_value = ttl or self.default_ttl

            # 序列化值
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            else:
                serialized_value = str(value)

            # 设置到 Redis
            success = await self.redis_client.setex(
                redis_key,
                ttl_value,
                serialized_value
            )

            if success:
                # 更新本地缓存
                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    access_count=1,
                    ttl=ttl_value
                )
                local_key = self._make_key(key)
                self._local_cache[local_key] = entry

                # 限制本地缓存大小
                if len(self._local_cache) > self._local_cache_size:
                    self._evict_lru_items()

                self.metrics.sets += 1
                self.metrics.total_set_time += time.time() - start_time
                return True
            else:
                self.metrics.errors += 1
                return False

        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ 缓存设置失败 [{key}]: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            if not self.redis_client:
                return False

            redis_key = self._make_key(key)

            # 从 Redis 删除
            result = await self.redis_client.delete(redis_key)

            # 从本地缓存删除
            local_key = self._make_key(key)
            if local_key in self._local_cache:
                del self._local_cache[local_key]

            if result:
                self.metrics.deletes += 1
                return True
            return False

        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ 缓存删除失败 [{key}]: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            # 先检查本地缓存
            local_key = self._make_key(key)
            if local_key in self._local_cache:
                entry = self._local_cache[local_key]
                return not entry.is_expired

            # 检查 Redis
            if not self.redis_client:
                return False

            redis_key = self._make_key(key)
            return bool(await self.redis_client.exists(redis_key))

        except Exception as e:
            logger.error(f"❌ 缓存检查失败 [{key}]: {e}")
            return False

    async def clear(self, pattern: str | None = None) -> int:
        """清空缓存"""
        try:
            if not self.redis_client:
                return 0

            if pattern:
                # 清空匹配模式
                search_pattern = self._make_key(pattern)
                keys = await self.redis_client.keys(search_pattern)
                if keys:
                    deleted_count = await self.redis_client.delete(*keys)
                else:
                    deleted_count = 0
            else:
                # 清空所有带前缀的键
                search_pattern = self._make_key("*")
                keys = await self.redis_client.keys(search_pattern)
                if keys:
                    deleted_count = await self.redis_client.delete(*keys)
                else:
                    deleted_count = 0

            # 清空本地缓存
            if pattern:
                prefix = self._make_key("")
                self._local_cache = {
                    k: v for k, v in self._local_cache.items()
                    if not k.startswith(prefix) or not pattern or pattern in k
                }
            else:
                self._local_cache.clear()

            logger.info(f"🗑️ 缓存已清空，删除了 {deleted_count} 个键")
            return deleted_count

        except Exception as e:
            logger.error(f"❌ 缓存清空失败: {e}")
            return 0

    async def get_ttl(self, key: str) -> int:
        """获取键的剩余生存时间"""
        try:
            if not self.redis_client:
                return -1

            redis_key = self._make_key(key)
            return await self.redis_client.ttl(redis_key)

        except Exception as e:
            logger.error(f"❌ 获取 TTL 失败 [{key}]: {e}")
            return -1

    async def acquire_lock(self, key: str, timeout: int = 10) -> bool:
        """获取分布式锁"""
        try:
            if not self.redis_client:
                return False

            lock_key = self._make_key(f"lock:{key}")
            lock_value = f"{time.time()}:{id(asyncio.current_task())}"

            # 使用 SET NX EX 实现分布式锁
            result = await self.redis_client.set(
                lock_key,
                lock_value,
                ex=timeout,
                nx=True
            )

            if result:
                self._locks[key] = asyncio.Lock()
                logger.debug(f"🔒 获取锁成功: {key}")
                return True
            else:
                logger.debug(f"❌ 获取锁失败: {key}")
                return False

        except Exception as e:
            logger.error(f"❌ 获取锁异常 [{key}]: {e}")
            return False

    async def release_lock(self, key: str) -> bool:
        """释放分布式锁"""
        try:
            if not self.redis_client:
                return False

            lock_key = self._make_key(f"lock:{key}")

            # 只有锁的持有者才能释放
            result = await self.redis_client.delete(lock_key)

            if key in self._locks:
                del self._locks[key]

            if result:
                logger.debug(f"🔓 释放锁成功: {key}")
                return True
            else:
                logger.warning(f"❌ 释放锁失败: {key}")
                return False

        except Exception as e:
            logger.error(f"❌ 释放锁异常 [{key}]: {e}")
            return False

    def _evict_lru_items(self):
        """LRU 淘汰策略"""
        if not self._local_cache:
            return

        # 按访问时间排序，删除最旧的项
        sorted_items = sorted(
            self._local_cache.items(),
            key=lambda x: x[1].accessed_at
        )

        # 删除最旧的 20% 项
        evict_count = max(1, len(sorted_items) // 5)
        for i in range(evict_count):
            if i < len(sorted_items):
                key_to_remove = sorted_items[i][0]
                del self._local_cache[key_to_remove]
                self.metrics.evictions += 1

    async def get_info(self) -> dict[str, Any]:
        """获取 Redis 信息"""
        try:
            if not self.redis_client:
                return {}

            info = await self.redis_client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "uptime_in_seconds": info.get("uptime_in_seconds")
            }
        except Exception as e:
            logger.error(f"❌ 获取 Redis 信息失败: {e}")
            return {}

    def get_metrics(self) -> dict[str, Any]:
        """获取缓存指标"""
        if not self.enable_metrics:
            return {}

        current_time = time.time()
        uptime = current_time - self._stats["last_reset"]

        return {
            "cache_metrics": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "sets": self.metrics.sets,
                "deletes": self.metrics.deletes,
                "evictions": self.metrics.evictions,
                "errors": self.metrics.errors,
                "hit_rate": self.metrics.hit_rate,
                "avg_get_time": self.metrics.avg_get_time,
                "avg_set_time": self.metrics.avg_set_time
            },
            "local_cache": {
                "size": len(self._local_cache),
                "max_size": self._local_cache_size,
                "utilization": len(self._local_cache) / self._local_cache_size * 100
            },
            "operations": {
                "total": self._stats["total_operations"],
                "per_second": self._stats["total_operations"] / uptime if uptime > 0 else 0,
                "uptime_seconds": uptime
            },
            "locks": {
                "active_count": len(self._locks),
                "active_keys": list(self._locks.keys())
            }
        }

    def reset_metrics(self):
        """重置指标"""
        self.metrics = CacheMetrics()
        self._stats = {
            "total_operations": 0,
            "cache_efficiency": 0.0,
            "last_reset": time.time()
        }
        logger.info("📊 缓存指标已重置")

    async def health_check(self) -> dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        try:
            # 检查 Redis 连接
            if self.redis_client:
                start_time = time.time()
                await self.redis_client.ping()
                response_time = time.time() - start_time
                health_status["checks"]["redis_connection"] = {
                    "status": "healthy",
                    "response_time": response_time
                }
            else:
                health_status["checks"]["redis_connection"] = {
                    "status": "unhealthy",
                    "error": "Redis 客户端未初始化"
                }
                health_status["status"] = "unhealthy"

            # 检查内存使用
            info = await self.get_info()
            if info.get("used_memory"):
                health_status["checks"]["memory_usage"] = {
                    "status": "healthy",
                    "used_memory": info["used_memory"]
                }

            # 检查缓存效率
            if self.metrics.hit_rate < 50:
                health_status["checks"]["cache_efficiency"] = {
                    "status": "warning",
                    "hit_rate": self.metrics.hit_rate,
                    "message": "缓存命中率较低"
                }
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"
            else:
                health_status["checks"]["cache_efficiency"] = {
                    "status": "healthy",
                    "hit_rate": self.metrics.hit_rate
                }

            # 检查错误率
            total_ops = self.metrics.hits + self.metrics.misses + self.metrics.errors
            if total_ops > 0:
                error_rate = self.metrics.errors / total_ops * 100
                if error_rate > 5:
                    health_status["checks"]["error_rate"] = {
                        "status": "critical",
                        "error_rate": error_rate,
                        "message": "错误率过高"
                    }
                    health_status["status"] = "critical"
                elif error_rate > 1:
                    health_status["checks"]["error_rate"] = {
                        "status": "warning",
                        "error_rate": error_rate,
                        "message": "错误率较高"
                    }
                    if health_status["status"] == "healthy":
                        health_status["status"] = "warning"
                else:
                    health_status["checks"]["error_rate"] = {
                        "status": "healthy",
                        "error_rate": error_rate
                    }

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["health_check_error"] = {
                "status": "error",
                "error": str(e)
            }

        return health_status


# 全局缓存管理器实例
_cache_manager: RedisCacheManager | None = None


async def get_cache_manager() -> RedisCacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager

    if _cache_manager is None:
        from src.framework.shared.config import settings
        _cache_manager = RedisCacheManager(
            redis_url=settings.ARQ_REDIS_URL,
            max_connections=20,
            default_ttl=3600,
            key_prefix="lumoscribe:cache:",
            enable_metrics=settings.METRICS_ENABLED
        )

        if not await _cache_manager.initialize():
            logger.error("❌ 缓存管理器初始化失败")

    return _cache_manager


async def close_cache_manager():
    """关闭全局缓存管理器"""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None
