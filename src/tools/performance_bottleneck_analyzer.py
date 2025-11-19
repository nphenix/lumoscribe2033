"""
性能瓶颈分析工具

用于识别和分析系统中的性能瓶颈，提供具体的优化建议
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

from src.framework.shared.logging import get_logger
from src.framework.shared.monitoring import get_metrics_collector
from src.framework.shared.performance import (
    get_performance_optimizer,
    get_redis_optimizer,
    get_chroma_optimizer,
    get_sqlite_optimizer,
    get_networkx_optimizer
)

logger = get_logger(__name__)


@dataclass
class BottleneckReport:
    """性能瓶颈报告"""
    timestamp: str
    component: str
    severity: str  # critical, high, medium, low
    issue: str
    impact: str
    recommendation: str
    metrics: Dict[str, Any]
    estimated_gain: str  # 预估性能提升


class PerformanceBottleneckAnalyzer:
    """性能瓶颈分析器"""

    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.performance_optimizer = get_performance_optimizer()
        
        # 初始化各个优化器
        self.redis_optimizer = get_redis_optimizer()
        self.chroma_optimizer = get_chroma_optimizer()
        self.sqlite_optimizer = get_sqlite_optimizer()
        self.networkx_optimizer = get_networkx_optimizer()
        
        # 分析结果缓存
        self.analysis_cache = {}
        self.last_analysis_time = None
        
        logger.info("🔍 性能瓶颈分析器已初始化")

    async def comprehensive_analysis(self) -> Dict[str, Any]:
        """执行全面的性能瓶颈分析"""
        start_time = time.time()
        logger.info("🔍 开始全面性能瓶颈分析...")
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_duration": 0,
            "system_resources": await self._analyze_system_resources(),
            "database_performance": await self._analyze_database_performance(),
            "cache_performance": await self._analyze_cache_performance(),
            "vector_search_performance": await self._analyze_vector_search_performance(),
            "graph_computation_performance": await self._analyze_graph_computation_performance(),
            "api_performance": await self._analyze_api_performance(),
            "bottlenecks": [],
            "recommendations": [],
            "performance_score": 0
        }
        
        # 收集所有瓶颈
        all_bottlenecks = []
        for category, data in analysis_results.items():
            if isinstance(data, dict) and "bottlenecks" in data:
                all_bottlenecks.extend(data["bottlenecks"])
        
        # 按严重程度排序
        all_bottlenecks.sort(key=lambda x: self._severity_score(x.severity), reverse=True)
        analysis_results["bottlenecks"] = all_bottlenecks[:10]  # 取前10个最严重的瓶颈
        
        # 生成综合建议
        analysis_results["recommendations"] = self._generate_comprehensive_recommendations(all_bottlenecks)
        
        # 计算性能评分
        analysis_results["performance_score"] = self._calculate_performance_score(all_bottlenecks)
        
        analysis_results["analysis_duration"] = time.time() - start_time
        self.last_analysis_time = datetime.now()
        
        logger.info(f"✅ 性能瓶颈分析完成，耗时: {analysis_results['analysis_duration']:.2f}s")
        return analysis_results

    async def _analyze_system_resources(self) -> Dict[str, Any]:
        """分析系统资源使用情况"""
        bottlenecks = []
        recommendations = []
        
        try:
            # CPU使用率分析
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="system",
                    severity="critical" if cpu_percent > 90 else "high",
                    issue=f"CPU使用率过高: {cpu_percent:.1f}%",
                    impact="系统响应变慢，可能影响所有操作",
                    recommendation="优化CPU密集型操作，考虑增加处理能力或负载均衡",
                    metrics={"cpu_percent": cpu_percent},
                    estimated_gain="20-40%性能提升"
                ))
            
            # 内存使用率分析
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="system",
                    severity="critical" if memory.percent > 90 else "high",
                    issue=f"内存使用率过高: {memory.percent:.1f}%",
                    impact="可能导致内存不足错误和系统不稳定",
                    recommendation="优化内存使用，增加内存容量或实施内存缓存策略",
                    metrics={"memory_percent": memory.percent, "used_gb": memory.used / (1024**3)},
                    estimated_gain="15-30%稳定性提升"
                ))
            
            # 磁盘I/O分析
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # 简单的磁盘使用率检查
                disk_usage = psutil.disk_usage('/')
                if disk_usage.percent > 85:
                    bottlenecks.append(BottleneckReport(
                        timestamp=datetime.now().isoformat(),
                        component="system",
                        severity="high",
                        issue=f"磁盘空间不足: {disk_usage.percent:.1f}%",
                        impact="可能影响日志写入和数据存储",
                        recommendation="清理不必要的文件，扩展存储空间",
                        metrics={"disk_percent": disk_usage.percent, "free_gb": disk_usage.free / (1024**3)},
                        estimated_gain="避免服务中断"
                    ))
            
            # 网络I/O分析
            network_io = psutil.net_io_counters()
            if network_io:
                # 这里可以添加更复杂的网络分析逻辑
                pass
            
            if not bottlenecks:
                recommendations.append("系统资源使用正常，继续保持当前配置")
            
        except Exception as e:
            logger.error(f"系统资源分析失败: {e}")
            bottlenecks.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                component="system",
                severity="medium",
                issue="系统资源监控异常",
                impact="无法准确评估系统性能",
                recommendation="检查系统监控工具配置",
                metrics={"error": str(e)},
                estimated_gain="提升监控可靠性"
            ))
        
        return {
            "component": "system_resources",
            "status": "healthy" if not bottlenecks else "degraded",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk_usage.percent if 'disk_usage' in locals() else 0
            }
        }

    async def _analyze_database_performance(self) -> Dict[str, Any]:
        """分析数据库性能"""
        bottlenecks = []
        recommendations = []
        
        try:
            # 获取SQLite性能统计
            sqlite_stats = self.sqlite_optimizer.get_sqlite_performance_stats()
            
            # 检查查询缓存大小
            cache_size = sqlite_stats.get("query_cache_size", 0)
            if cache_size > 1000:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="database",
                    severity="medium",
                    issue=f"查询缓存过大: {cache_size}项",
                    impact="占用过多内存，可能影响性能",
                    recommendation="定期清理查询缓存，实施LRU策略",
                    metrics=sqlite_stats,
                    estimated_gain="5-15%内存优化"
                ))
            
            # 检查索引优化情况
            index_stats = sqlite_stats.get("index_stats", {})
            if not index_stats:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="database",
                    severity="high",
                    issue="缺少数据库索引",
                    impact="查询性能低下，特别是复杂查询",
                    recommendation="为常用查询字段创建索引",
                    metrics=sqlite_stats,
                    estimated_gain="30-70%查询性能提升"
                ))
            
            # 获取性能优化器的慢查询
            slow_queries = self.performance_optimizer.get_slow_queries(threshold=2.0)
            if slow_queries:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="database",
                    severity="high",
                    issue=f"发现{len(slow_queries)}个慢查询",
                    impact="数据库响应时间过长，影响整体性能",
                    recommendation="优化慢查询，添加索引，考虑查询重写",
                    metrics={"slow_query_count": len(slow_queries), "queries": slow_queries[:5]},
                    estimated_gain="20-50%查询性能提升"
                ))
            
            # 生成建议
            sqlite_recommendations = sqlite_stats.get("recommendations", [])
            recommendations.extend(sqlite_recommendations)
            
        except Exception as e:
            logger.error(f"数据库性能分析失败: {e}")
            bottlenecks.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                component="database",
                severity="medium",
                issue="数据库性能分析异常",
                impact="无法准确评估数据库性能",
                recommendation="检查数据库连接和配置",
                metrics={"error": str(e)},
                estimated_gain="提升数据库监控能力"
            ))
        
        return {
            "component": "database_performance",
            "status": "healthy" if not bottlenecks else "degraded",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": sqlite_stats if 'sqlite_stats' in locals() else {}
        }

    async def _analyze_cache_performance(self) -> Dict[str, Any]:
        """分析缓存性能"""
        bottlenecks = []
        recommendations = []
        
        try:
            # 获取Redis性能统计
            redis_stats = self.redis_optimizer.get_redis_performance_stats()
            
            # 检查连接池统计
            pool_stats = redis_stats.get("connection_pool_stats", {})
            pool_hits = pool_stats.get("pool_hits", 0)
            pool_misses = pool_stats.get("pool_misses", 0)
            
            if pool_misses > pool_hits:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="cache",
                    severity="high",
                    issue="Redis连接池命中率低",
                    impact="频繁创建连接，增加延迟",
                    recommendation="优化连接池配置，增加连接池大小",
                    metrics=pool_stats,
                    estimated_gain="10-25%缓存性能提升"
                ))
            
            # 获取性能优化器的缓存统计
            perf_stats = self.performance_optimizer.get_performance_stats()
            cache_stats = perf_stats.get("cache_stats", {})
            cache_hit_rate = cache_stats.get("cache_hit_rate", 0)
            
            if cache_hit_rate < 50:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="cache",
                    severity="high",
                    issue=f"缓存命中率过低: {cache_hit_rate:.1f}%",
                    impact="增加后端负载，降低响应速度",
                    recommendation="优化缓存策略，增加缓存预热，调整TTL",
                    metrics=cache_stats,
                    estimated_gain="20-40%响应速度提升"
                ))
            
            # 生成建议
            redis_recommendations = redis_stats.get("recommendations", [])
            recommendations.extend(redis_recommendations)
            
        except Exception as e:
            logger.error(f"缓存性能分析失败: {e}")
            bottlenecks.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                component="cache",
                severity="medium",
                issue="缓存性能分析异常",
                impact="无法准确评估缓存性能",
                recommendation="检查Redis连接和配置",
                metrics={"error": str(e)},
                estimated_gain="提升缓存监控能力"
            ))
        
        return {
            "component": "cache_performance",
            "status": "healthy" if not bottlenecks else "degraded",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": redis_stats if 'redis_stats' in locals() else {}
        }

    async def _analyze_vector_search_performance(self) -> Dict[str, Any]:
        """分析向量搜索性能"""
        bottlenecks = []
        recommendations = []
        
        try:
            # 获取ChromaDB性能统计
            chroma_stats = self.chroma_optimizer.get_chroma_performance_stats()
            
            # 检查批量查询比例
            perf_metrics = chroma_stats.get("performance_metrics", {})
            batch_ratio = perf_metrics.get("batch_query_ratio", 0)
            
            if batch_ratio < 0.7:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="vector_search",
                    severity="medium",
                    issue=f"批量查询比例过低: {batch_ratio:.1%}",
                    impact="增加网络开销，降低整体吞吐量",
                    recommendation="尽可能使用批量查询API，合并单独查询",
                    metrics=perf_metrics,
                    estimated_gain="15-35%搜索性能提升"
                ))
            
            # 检查平均查询时间
            avg_query_time = perf_metrics.get("avg_query_time", 0)
            if avg_query_time > 1.0:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="vector_search",
                    severity="high",
                    issue=f"平均查询时间过长: {avg_query_time:.3f}s",
                    impact="用户等待时间增加，影响体验",
                    recommendation="优化HNSW参数，减少返回结果数量，优化查询向量",
                    metrics=perf_metrics,
                    estimated_gain="25-50%查询速度提升"
                ))
            
            # 生成建议
            chroma_recommendations = chroma_stats.get("recommendations", [])
            recommendations.extend(chroma_recommendations)
            
        except Exception as e:
            logger.error(f"向量搜索性能分析失败: {e}")
            bottlenecks.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                component="vector_search",
                severity="medium",
                issue="向量搜索性能分析异常",
                impact="无法准确评估搜索性能",
                recommendation="检查ChromaDB连接和配置",
                metrics={"error": str(e)},
                estimated_gain="提升搜索监控能力"
            ))
        
        return {
            "component": "vector_search_performance",
            "status": "healthy" if not bottlenecks else "degraded",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": chroma_stats if 'chroma_stats' in locals() else {}
        }

    async def _analyze_graph_computation_performance(self) -> Dict[str, Any]:
        """分析图计算性能"""
        bottlenecks = []
        recommendations = []
        
        try:
            # 获取NetworkX性能统计
            networkx_stats = self.networkx_optimizer.get_networkx_performance_stats()
            
            # 检查缓存命中率
            cache_hit_rate = networkx_stats.get("cache_hit_rate", 0)
            if cache_hit_rate < 0.5:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="graph_computation",
                    severity="medium",
                    issue=f"图计算缓存命中率过低: {cache_hit_rate:.1%}",
                    impact="重复计算增加CPU负载",
                    recommendation="增加缓存容量，优化缓存键生成策略",
                    metrics=networkx_stats,
                    estimated_gain="20-40%计算性能提升"
                ))
            
            # 检查平均计算时间
            comp_stats = networkx_stats.get("computation_stats", {})
            avg_time = comp_stats.get("avg_computation_time", 0)
            if avg_time >= 1.0:
                bottlenecks.append(BottleneckReport(
                    timestamp=datetime.now().isoformat(),
                    component="graph_computation",
                    severity="high",
                    issue=f"平均图计算时间过长: {avg_time:.3f}s",
                    impact="影响依赖图计算的功能性能",
                    recommendation="使用更高效的算法，考虑图分割处理，使用并行计算",
                    metrics=comp_stats,
                    estimated_gain="30-60%计算速度提升"
                ))
            
            # 生成建议
            networkx_recommendations = networkx_stats.get("recommendations", [])
            recommendations.extend(networkx_recommendations)
            
        except Exception as e:
            logger.error(f"图计算性能分析失败: {e}")
            bottlenecks.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                component="graph_computation",
                severity="medium",
                issue="图计算性能分析异常",
                impact="无法准确评估图计算性能",
                recommendation="检查NetworkX配置和图数据结构",
                metrics={"error": str(e)},
                estimated_gain="提升图计算监控能力"
            ))
        
        return {
            "component": "graph_computation_performance",
            "status": "healthy" if not bottlenecks else "degraded",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": networkx_stats if 'networkx_stats' in locals() else {}
        }

    async def _analyze_api_performance(self) -> Dict[str, Any]:
        """分析API性能"""
        bottlenecks = []
        recommendations = []
        
        try:
            # 获取API指标摘要
            api_summary = self.metrics_collector.get_api_summary(hours=1)
            
            if not api_summary:
                recommendations.append("暂无足够的API性能数据，建议收集更多数据后重新分析")
                return {
                    "component": "api_performance",
                    "status": "unknown",
                    "bottlenecks": bottlenecks,
                    "recommendations": recommendations,
                    "metrics": {}
                }
            
            # 分析各端点的性能
            for endpoint, stats in api_summary.items():
                avg_response_time = stats.get("avg_response_time", 0)
                success_rate = stats.get("success_rate", 100)
                total_requests = stats.get("total_requests", 0)
                
                # 检查响应时间
                if avg_response_time > 3.0:
                    bottlenecks.append(BottleneckReport(
                        timestamp=datetime.now().isoformat(),
                        component="api",
                        severity="high",
                        issue=f"端点 {endpoint} 响应时间过长: {avg_response_time:.2f}s",
                        impact="用户体验差，可能影响系统可用性",
                        recommendation="优化端点逻辑，添加缓存，减少数据库查询",
                        metrics={"endpoint": endpoint, "avg_response_time": avg_response_time},
                        estimated_gain="30-50%响应速度提升"
                    ))
                elif avg_response_time > 1.5:
                    bottlenecks.append(BottleneckReport(
                        timestamp=datetime.now().isoformat(),
                        component="api",
                        severity="medium",
                        issue=f"端点 {endpoint} 响应时间较长: {avg_response_time:.2f}s",
                        impact="用户体验一般，有优化空间",
                        recommendation="分析端点性能瓶颈，考虑异步处理",
                        metrics={"endpoint": endpoint, "avg_response_time": avg_response_time},
                        estimated_gain="15-25%响应速度提升"
                    ))
                
                # 检查成功率
                if success_rate < 95:
                    bottlenecks.append(BottleneckReport(
                        timestamp=datetime.now().isoformat(),
                        component="api",
                        severity="high",
                        issue=f"端点 {endpoint} 成功率过低: {success_rate:.1f}%",
                        impact="服务不稳定，影响用户信任度",
                        recommendation="增强错误处理，改进输入验证，提高系统稳定性",
                        metrics={"endpoint": endpoint, "success_rate": success_rate},
                        estimated_gain="提升服务可靠性"
                    ))
            
            # 生成综合建议
            if not bottlenecks:
                recommendations.append("API性能表现良好，继续保持当前优化水平")
            else:
                recommendations.extend([
                    "实施API性能监控和告警",
                    "定期进行API性能测试和优化",
                    "考虑实施API网关进行统一管理和优化"
                ])
            
        except Exception as e:
            logger.error(f"API性能分析失败: {e}")
            bottlenecks.append(BottleneckReport(
                timestamp=datetime.now().isoformat(),
                component="api",
                severity="medium",
                issue="API性能分析异常",
                impact="无法准确评估API性能",
                recommendation="检查API监控配置和数据收集",
                metrics={"error": str(e)},
                estimated_gain="提升API监控能力"
            ))
        
        return {
            "component": "api_performance",
            "status": "healthy" if not bottlenecks else "degraded",
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": api_summary
        }

    def _severity_score(self, severity: str) -> int:
        """将严重程度转换为数值分数"""
        severity_map = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return severity_map.get(severity, 0)

    def _generate_comprehensive_recommendations(self, bottlenecks: List[BottleneckReport]) -> List[str]:
        """生成综合优化建议"""
        if not bottlenecks:
            return ["系统性能表现良好，继续保持当前配置"]
        
        # 按组件分组瓶颈
        component_bottlenecks = defaultdict(list)
        for bottleneck in bottlenecks:
            component_bottlenecks[bottleneck.component].append(bottleneck)
        
        recommendations = []
        
        # 为每个组件生成建议
        for component, component_bottlenecks in component_bottlenecks.items():
            critical_count = len([b for b in component_bottlenecks if b.severity == "critical"])
            high_count = len([b for b in component_bottlenecks if b.severity == "high"])
            
            if critical_count > 0:
                recommendations.append(f"🚨 {component}组件存在{critical_count}个严重问题，需要立即处理")
            
            if high_count > 0:
                recommendations.append(f"⚠️ {component}组件存在{high_count}个高优先级问题，建议优先解决")
        
        # 通用建议
        recommendations.extend([
            "📊 建立完善的性能监控体系，实时跟踪关键指标",
            "🔧 定期进行性能评估和优化，预防性能退化",
            "📈 实施性能预算和告警机制，确保服务质量",
            "🚀 考虑实施自动扩缩容和负载均衡，提升系统弹性"
        ])
        
        return recommendations

    def _calculate_performance_score(self, bottlenecks: List[BottleneckReport]) -> int:
        """计算性能评分 (0-100)"""
        if not bottlenecks:
            return 100
        
        # 根据瓶颈严重程度计算扣分
        total_deduction = 0
        for bottleneck in bottlenecks:
            if bottleneck.severity == "critical":
                total_deduction += 25
            elif bottleneck.severity == "high":
                total_deduction += 15
            elif bottleneck.severity == "medium":
                total_deduction += 8
            elif bottleneck.severity == "low":
                total_deduction += 3
        
        score = max(0, 100 - total_deduction)
        return score

    async def generate_optimization_plan(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """基于分析结果生成优化计划"""
        bottlenecks = analysis_results.get("bottlenecks", [])
        
        # 按优先级分组
        critical_issues = [b for b in bottlenecks if b.severity == "critical"]
        high_issues = [b for b in bottlenecks if b.severity == "high"]
        medium_issues = [b for b in bottlenecks if b.severity == "medium"]
        low_issues = [b for b in bottlenecks if b.severity == "low"]
        
        optimization_plan = {
            "timestamp": datetime.now().isoformat(),
            "performance_score": analysis_results.get("performance_score", 0),
            "phases": [],
            "estimated_total_gain": "全面提升系统性能",
            "implementation_priority": []
        }
        
        # 第一阶段：处理关键问题
        if critical_issues:
            phase1 = {
                "phase": 1,
                "name": "紧急修复",
                "duration": "1-2周",
                "focus": "解决关键性能问题",
                "tasks": [
                    {
                        "task": f"修复{issue.component}组件: {issue.issue}",
                        "priority": "critical",
                        "estimated_gain": issue.estimated_gain,
                        "implementation": issue.recommendation
                    }
                    for issue in critical_issues
                ]
            }
            optimization_plan["phases"].append(phase1)
        
        # 第二阶段：高优先级优化
        if high_issues:
            phase2 = {
                "phase": 2,
                "name": "性能优化",
                "duration": "2-4周",
                "focus": "解决高优先级性能问题",
                "tasks": [
                    {
                        "task": f"优化{issue.component}组件: {issue.issue}",
                        "priority": "high",
                        "estimated_gain": issue.estimated_gain,
                        "implementation": issue.recommendation
                    }
                    for issue in high_issues
                ]
            }
            optimization_plan["phases"].append(phase2)
        
        # 第三阶段：持续改进
        if medium_issues or low_issues:
            remaining_issues = medium_issues + low_issues
            phase3 = {
                "phase": 3,
                "name": "持续改进",
                "duration": "4-8周",
                "focus": "处理剩余性能问题",
                "tasks": [
                    {
                        "task": f"改进{issue.component}组件: {issue.issue}",
                        "priority": issue.severity,
                        "estimated_gain": issue.estimated_gain,
                        "implementation": issue.recommendation
                    }
                    for issue in remaining_issues
                ]
            }
            optimization_plan["phases"].append(phase3)
        
        # 生成实施优先级列表
        all_tasks = []
        for phase in optimization_plan["phases"]:
            all_tasks.extend(phase["tasks"])
        
        # 按影响程度排序
        all_tasks.sort(key=lambda x: self._priority_score(x["priority"]), reverse=True)
        optimization_plan["implementation_priority"] = all_tasks[:10]  # 取前10个任务
        
        return optimization_plan

    def _priority_score(self, priority: str) -> int:
        """将优先级转换为数值分数"""
        priority_map = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return priority_map.get(priority, 0)

    async def export_analysis_report(self, analysis_results: Dict[str, Any], 
                                   output_path: str = "logs/performance_analysis_report.json") -> str:
        """导出分析报告"""
        import json
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成优化计划
        optimization_plan = await self.generate_optimization_plan(analysis_results)
        analysis_results["optimization_plan"] = optimization_plan
        
        # 导出报告
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"📊 性能分析报告已导出: {output_path}")
        return output_path


# 全局性能瓶颈分析器实例
_bottleneck_analyzer = None


def get_bottleneck_analyzer() -> PerformanceBottleneckAnalyzer:
    """获取全局性能瓶颈分析器实例"""
    global _bottleneck_analyzer
    if _bottleneck_analyzer is None:
        _bottleneck_analyzer = PerformanceBottleneckAnalyzer()
    return _bottleneck_analyzer