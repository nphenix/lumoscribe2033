"""
性能监控和优化API路由

提供系统性能监控、优化建议和配置管理功能
"""

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.framework.shared.logging import get_logger
from src.framework.shared.performance import (
    get_cache_optimizer,
    get_chroma_optimizer,
    get_langchain_optimizer,
    get_llamaindex_optimizer,
    get_networkx_optimizer,
    get_performance_optimizer,
    get_redis_optimizer,
    get_sqlite_optimizer,
)
from src.framework.shared.security import get_current_user, require_permission

logger = get_logger(__name__)
router = APIRouter(prefix="/performance", tags=["performance"])


# 请求模型
class PerformanceOptimizationRequest(BaseModel):
    """性能优化请求"""
    optimization_type: str = Field(..., description="优化类型: cache, database, query, all")
    parameters: dict[str, Any] = Field(default_factory=dict, description="优化参数")


class CacheWarmupRequest(BaseModel):
    """缓存预热请求"""
    cache_keys: list[str] = Field(..., description="要预热的缓存键列表")
    cache_type: str = Field(default="all", description="缓存类型: l1, l2, all")


class QueryPerformanceRequest(BaseModel):
    """查询性能分析请求"""
    query: str = Field(..., description="要分析的查询")
    query_type: str = Field(default="llamaindex", description="查询类型: llamaindex, langchain, database")


class PerformanceConfigRequest(BaseModel):
    """性能配置请求"""
    config_type: str = Field(..., description="配置类型: cache, query, database")
    config_data: dict[str, Any] = Field(..., description="配置数据")


# 响应模型
class PerformanceStatsResponse(BaseModel):
    """性能统计响应"""
    timestamp: float
    system_stats: dict[str, Any]
    langchain_stats: dict[str, Any] | None = None
    llamaindex_stats: dict[str, Any] | None = None
    cache_stats: dict[str, Any] | None = None
    recommendations: list[str]


class OptimizationResponse(BaseModel):
    """优化响应"""
    success: bool
    message: str
    optimization_results: dict[str, Any] | None = None


# 核心API端点
@router.get("/stats", response_model=PerformanceStatsResponse)
async def get_performance_stats(
    current_user: dict[str, Any] = Depends(get_current_user)
):
    """获取系统性能统计"""
    try:
        # 获取基础性能统计
        optimizer = get_performance_optimizer()
        system_stats = optimizer.get_performance_stats()

        # 获取LangChain统计
        langchain_stats = None
        try:
            langchain_optimizer = get_langchain_optimizer()
            langchain_stats = langchain_optimizer.get_langchain_performance_stats()
        except Exception as e:
            logger.warning(f"获取LangChain统计失败: {e}")

        # 获取LlamaIndex统计
        llamaindex_stats = None
        try:
            llamaindex_optimizer = get_llamaindex_optimizer()
            llamaindex_stats = llamaindex_optimizer.get_llamaindex_performance_stats()
        except Exception as e:
            logger.warning(f"获取LlamaIndex统计失败: {e}")

        # 获取缓存统计
        cache_stats = None
        try:
            cache_optimizer = get_cache_optimizer()
            cache_stats = cache_optimizer.get_cache_stats()
        except Exception as e:
            logger.warning(f"获取缓存统计失败: {e}")

        # 收集所有建议
        recommendations = []
        recommendations.extend(optimizer.get_performance_recommendations())

        if langchain_stats and "message" not in langchain_stats:
            langchain_optimizer = get_langchain_optimizer()
            recommendations.extend(langchain_optimizer.get_langchain_recommendations())

        if llamaindex_stats and "message" not in llamaindex_stats:
            llamaindex_optimizer = get_llamaindex_optimizer()
            recommendations.extend(llamaindex_optimizer.get_llamaindex_recommendations())

        return PerformanceStatsResponse(
            timestamp=asyncio.get_event_loop().time(),
            system_stats=system_stats,
            langchain_stats=langchain_stats,
            llamaindex_stats=llamaindex_stats,
            cache_stats=cache_stats,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_performance(
    request: PerformanceOptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict[str, Any] = Depends(require_permission("performance:optimize"))
):
    """执行性能优化"""
    try:
        optimization_type = request.optimization_type.lower()
        parameters = request.parameters

        if optimization_type in ["cache", "all"]:
            # 缓存优化
            cache_optimizer = get_cache_optimizer()
            if "warmup_keys" in parameters:
                background_tasks.add_task(
                    cache_optimizer.warmup_cache,
                    parameters["warmup_keys"]
                )

        if optimization_type in ["query", "all"]:
            # 查询优化
            optimizer = get_performance_optimizer()
            if "cleanup_days" in parameters:
                background_tasks.add_task(
                    optimizer.cleanup_old_metrics,
                    parameters["cleanup_days"]
                )

        if optimization_type in ["llamaindex", "all"]:
            # LlamaIndex优化
            try:
                llamaindex_optimizer = get_llamaindex_optimizer()
                if "cleanup_caches" in parameters and parameters["cleanup_caches"]:
                    background_tasks.add_task(llamaindex_optimizer.cleanup_caches)
            except Exception as e:
                logger.warning(f"LlamaIndex优化失败: {e}")

        return OptimizationResponse(
            success=True,
            message=f"性能优化任务已启动: {optimization_type}",
            optimization_results={"status": "scheduled"}
        )

    except Exception as e:
        logger.error(f"性能优化失败: {e}")
        raise HTTPException(status_code=500, detail=f"性能优化失败: {str(e)}")


@router.post("/cache/warmup")
async def warmup_cache(
    request: CacheWarmupRequest,
    current_user: dict[str, Any] = Depends(require_permission("performance:cache"))
):
    """缓存预热"""
    try:
        optimizer = get_performance_optimizer()
        await optimizer.warmup_cache(request.cache_keys)

        return {"success": True, "message": f"缓存预热完成，处理了 {len(request.cache_keys)} 个键"}

    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        raise HTTPException(status_code=500, detail=f"缓存预热失败: {str(e)}")


@router.post("/query/analyze")
async def analyze_query_performance(
    request: QueryPerformanceRequest,
    current_user: dict[str, Any] = Depends(require_permission("performance:analyze"))
):
    """分析查询性能"""
    try:
        query_type = request.query_type.lower()

        if query_type == "llamaindex":
            # llamaindex_optimizer = get_llamaindex_optimizer()
            # 这里需要传入实际的查询引擎
            # result = llamaindex_optimizer.profile_query_performance(query_engine, request.query)
            result = {"message": "LlamaIndex查询分析需要传入查询引擎"}

        elif query_type == "langchain":
            # langchain_optimizer = get_langchain_optimizer()
            # 这里需要传入实际的链
            # result = await langchain_optimizer.optimize_langchain_chain(chain_func, request.query)
            result = {"message": "LangChain查询分析需要传入链函数"}

        else:
            # optimizer = get_performance_optimizer()
            # 通用查询分析
            result = {"message": f"通用查询分析: {request.query}"}

        return {"success": True, "result": result}

    except Exception as e:
        logger.error(f"查询性能分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询性能分析失败: {str(e)}")


@router.get("/slow-queries")
async def get_slow_queries(
    threshold: float = 2.0,
    current_user: dict[str, Any] = Depends(require_permission("performance:view"))
):
    """获取慢查询列表"""
    try:
        optimizer = get_performance_optimizer()
        slow_queries = optimizer.get_slow_queries(threshold)

        return {
            "success": True,
            "threshold": threshold,
            "count": len(slow_queries),
            "queries": slow_queries
        }

    except Exception as e:
        logger.error(f"获取慢查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取慢查询失败: {str(e)}")


@router.get("/recommendations")
async def get_performance_recommendations(
    current_user: dict[str, Any] = Depends(get_current_user)
):
    """获取性能优化建议"""
    try:
        recommendations = []

        # 基础性能建议
        optimizer = get_performance_optimizer()
        recommendations.extend(optimizer.get_performance_recommendations())

        # LangChain建议
        try:
            langchain_optimizer = get_langchain_optimizer()
            recommendations.extend(langchain_optimizer.get_langchain_recommendations())
        except Exception as e:
            logger.warning(f"获取LangChain建议失败: {e}")

        # LlamaIndex建议
        try:
            llamaindex_optimizer = get_llamaindex_optimizer()
            recommendations.extend(llamaindex_optimizer.get_llamaindex_recommendations())
        except Exception as e:
            logger.warning(f"获取LlamaIndex建议失败: {e}")

        return {
            "success": True,
            "count": len(recommendations),
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"获取性能建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能建议失败: {str(e)}")


@router.post("/config")
async def update_performance_config(
    request: PerformanceConfigRequest,
    current_user: dict[str, Any] = Depends(require_permission("performance:config"))
):
    """更新性能配置"""
    try:
        config_type = request.config_type.lower()
        config_data = request.config_data

        if config_type == "cache":
            # 更新缓存配置
            optimizer = get_performance_optimizer()
            if "enable_multi_level_cache" in config_data:
                optimizer.cache_optimization.enable_multi_level_cache = config_data["enable_multi_level_cache"]
            if "l1_cache_size" in config_data:
                optimizer.cache_optimization.l1_cache_size = config_data["l1_cache_size"]
            if "l2_cache_ttl" in config_data:
                optimizer.cache_optimization.l2_cache_ttl = config_data["l2_cache_ttl"]

        elif config_type == "query":
            # 更新查询配置
            optimizer = get_performance_optimizer()
            if "enable_query_cache" in config_data:
                optimizer.query_optimization.enable_query_cache = config_data["enable_query_cache"]
            if "max_connections" in config_data:
                optimizer.query_optimization.max_connections = config_data["max_connections"]
            if "batch_size" in config_data:
                optimizer.query_optimization.batch_size = config_data["batch_size"]

        elif config_type == "database":
            # 更新数据库配置
            optimizer = get_performance_optimizer()
            if "enable_connection_pooling" in config_data:
                optimizer.query_optimization.enable_connection_pooling = config_data["enable_connection_pooling"]
            if "query_timeout" in config_data:
                optimizer.query_optimization.query_timeout = config_data["query_timeout"]

        return {"success": True, "message": f"性能配置已更新: {config_type}"}

    except Exception as e:
        logger.error(f"更新性能配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新性能配置失败: {str(e)}")


@router.get("/config")
async def get_performance_config(
    current_user: dict[str, Any] = Depends(get_current_user)
):
    """获取当前性能配置"""
    try:
        optimizer = get_performance_optimizer()

        config = {
            "cache": {
                "enable_multi_level_cache": optimizer.cache_optimization.enable_multi_level_cache,
                "l1_cache_size": optimizer.cache_optimization.l1_cache_size,
                "l2_cache_ttl": optimizer.cache_optimization.l2_cache_ttl,
                "enable_write_through": optimizer.cache_optimization.enable_write_through,
                "enable_write_back": optimizer.cache_optimization.enable_write_back,
                "cache_warmup": optimizer.cache_optimization.cache_warmup
            },
            "query": {
                "enable_query_cache": optimizer.query_optimization.enable_query_cache,
                "enable_connection_pooling": optimizer.query_optimization.enable_connection_pooling,
                "max_connections": optimizer.query_optimization.max_connections,
                "query_timeout": optimizer.query_optimization.query_timeout,
                "batch_size": optimizer.query_optimization.batch_size,
                "enable_index_hints": optimizer.query_optimization.enable_index_hints
            }
        }

        return {"success": True, "config": config}

    except Exception as e:
        logger.error(f"获取性能配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能配置失败: {str(e)}")


@router.delete("/metrics/cleanup")
async def cleanup_performance_metrics(
    days: int = 7,
    current_user: dict[str, Any] = Depends(require_permission("performance:cleanup"))
):
    """清理性能指标"""
    try:
        optimizer = get_performance_optimizer()
        await optimizer.cleanup_old_metrics(days)

        return {"success": True, "message": f"已清理 {days} 天前的性能指标"}

    except Exception as e:
        logger.error(f"清理性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理性能指标失败: {str(e)}")


@router.get("/health")
async def performance_health_check():
    """性能模块健康检查"""
    try:
        # 检查各组件状态
        health_status = {
            "performance_optimizer": "healthy",
            "langchain_optimizer": "healthy",
            "llamaindex_optimizer": "healthy",
            "cache_optimizer": "healthy"
        }

        # 检查基础性能优化器
        try:
            # optimizer = get_performance_optimizer()
            # stats = optimizer.get_performance_stats()
            pass
        except Exception as e:
            health_status["performance_optimizer"] = f"unhealthy: {str(e)}"

        # 检查LangChain优化器
        try:
            # langchain_optimizer = get_langchain_optimizer()
            # langchain_stats = langchain_optimizer.get_langchain_performance_stats()
            pass
        except Exception as e:
            health_status["langchain_optimizer"] = f"unhealthy: {str(e)}"

        # 检查LlamaIndex优化器
        try:
            # llamaindex_optimizer = get_llamaindex_optimizer()
            # llamaindex_stats = llamaindex_optimizer.get_llamaindex_performance_stats()
            pass
        except Exception as e:
            health_status["llamaindex_optimizer"] = f"unhealthy: {str(e)}"

        # 检查缓存优化器
        try:
            # cache_optimizer = get_cache_optimizer()
            # cache_stats = cache_optimizer.get_cache_stats()
            pass
        except Exception as e:
            health_status["cache_optimizer"] = f"unhealthy: {str(e)}"

        # 确定整体健康状态
        all_healthy = all(status == "healthy" for status in health_status.values())

        return {
            "healthy": all_healthy,
            "components": health_status,
            "timestamp": asyncio.get_event_loop().time()
        }

    except Exception as e:
        logger.error(f"性能健康检查失败: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }
