"""
性能分析CLI命令

提供性能瓶颈分析和优化的命令行接口
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.framework.shared.logging import get_logger
from src.tools.performance_bottleneck_analyzer import get_bottleneck_analyzer

logger = get_logger(__name__)
app = typer.Typer(help="性能分析工具")
console = Console()


@app.command()
def analyze(
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="输出报告文件路径"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="详细输出"
    )
):
    """执行全面的性能瓶颈分析"""
    
    async def run_analysis():
        console.print("[bold blue]🔍 开始性能瓶颈分析...[/bold blue]")
        
        try:
            analyzer = get_bottleneck_analyzer()
            results = await analyzer.comprehensive_analysis()
            
            # 显示分析摘要
            _display_analysis_summary(results)
            
            # 显示详细瓶颈
            if verbose:
                _display_detailed_bottlenecks(results)
            
            # 显示优化建议
            _display_recommendations(results)
            
            # 生成优化计划
            optimization_plan = await analyzer.generate_optimization_plan(results)
            _display_optimization_plan(optimization_plan)
            
            # 导出报告
            if output:
                report_path = await analyzer.export_analysis_report(results, output)
                console.print(f"[green]📊 分析报告已导出: {report_path}[/green]")
            else:
                # 默认导出路径
                default_path = "logs/performance_analysis_report.json"
                report_path = await analyzer.export_analysis_report(results, default_path)
                console.print(f"[green]📊 分析报告已导出: {report_path}[/green]")
            
            # 返回性能评分
            score = results.get("performance_score", 0)
            if score >= 80:
                console.print(f"[green]✅ 性能评分: {score}/100 (优秀)[/green]")
            elif score >= 60:
                console.print(f"[yellow]⚠️ 性能评分: {score}/100 (良好)[/yellow]")
            else:
                console.print(f"[red]🚨 性能评分: {score}/100 (需要优化)[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ 分析失败: {e}[/red]")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    
    asyncio.run(run_analysis())


@app.command()
def quick_check():
    """快速性能检查"""
    
    async def run_quick_check():
        console.print("[bold blue]⚡ 执行快速性能检查...[/bold blue]")
        
        try:
            analyzer = get_bottleneck_analyzer()
            
            # 只分析关键组件
            tasks = [
                analyzer._analyze_system_resources(),
                analyzer._analyze_database_performance(),
                analyzer._analyze_cache_performance()
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 创建快速检查表格
            table = Table(title="快速性能检查结果")
            table.add_column("组件", style="cyan")
            table.add_column("状态", style="magenta")
            table.add_column("问题数", style="yellow")
            table.add_column("主要问题", style="red")
            
            total_issues = 0
            for result in results:
                component = result.get("component", "未知")
                status = result.get("status", "unknown")
                bottlenecks = result.get("bottlenecks", [])
                issue_count = len(bottlenecks)
                total_issues += issue_count
                
                # 获取主要问题
                main_issue = "无" if not bottlenecks else bottlenecks[0].issue
                
                # 状态样式
                status_style = {
                    "healthy": "[green]健康[/green]",
                    "degraded": "[yellow]降级[/yellow]",
                    "critical": "[red]严重[/red]",
                    "unknown": "[gray]未知[/gray]"
                }.get(status, status)
                
                table.add_row(component, status_style, str(issue_count), main_issue)
            
            console.print(table)
            
            # 总结
            if total_issues == 0:
                console.print("[green]✅ 系统性能良好，未发现明显问题[/green]")
            elif total_issues <= 3:
                console.print(f"[yellow]⚠️ 发现 {total_issues} 个性能问题，建议进一步分析[/yellow]")
            else:
                console.print(f"[red]🚨 发现 {total_issues} 个性能问题，需要立即优化[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ 快速检查失败: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(run_quick_check())


@app.command()
def monitor(
    interval: int = typer.Option(
        60, "--interval", "-i", help="监控间隔（秒）"
    ),
    duration: int = typer.Option(
        300, "--duration", "-d", help="监控持续时间（秒）"
    )
):
    """持续性能监控"""
    
    async def run_monitoring():
        console.print(f"[bold blue]📊 开始性能监控，间隔: {interval}s，持续时间: {duration}s[/bold blue]")
        
        try:
            analyzer = get_bottleneck_analyzer()
            start_time = asyncio.get_event_loop().time()
            end_time = start_time + duration
            
            # 创建监控历史
            history = []
            
            while asyncio.get_event_loop().time() < end_time:
                # 执行快速检查
                tasks = [
                    analyzer._analyze_system_resources(),
                    analyzer._analyze_database_performance(),
                    analyzer._analyze_cache_performance()
                ]
                
                results = await asyncio.gather(*tasks)
                
                # 计算当前性能评分
                all_bottlenecks = []
                for result in results:
                    all_bottlenecks.extend(result.get("bottlenecks", []))
                
                score = analyzer._calculate_performance_score(all_bottlenecks)
                timestamp = asyncio.get_event_loop().time()
                
                history.append({
                    "timestamp": timestamp,
                    "score": score,
                    "issues": len(all_bottlenecks)
                })
                
                # 显示当前状态
                status_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                console.print(f"{status_emoji} 时间: {int(timestamp - start_time)}s, 评分: {score}, 问题: {len(all_bottlenecks)}")
                
                # 等待下次检查
                await asyncio.sleep(interval)
            
            # 显示监控摘要
            _display_monitoring_summary(history)
            
        except Exception as e:
            console.print(f"[red]❌ 监控失败: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(run_monitoring())


@app.command()
def compare(
    report1: str = typer.Argument(..., help="第一个分析报告文件"),
    report2: str = typer.Argument(..., help="第二个分析报告文件")
):
    """比较两个性能分析报告"""
    
    try:
        # 读取报告文件
        with open(report1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        
        with open(report2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        # 比较性能评分
        score1 = data1.get("performance_score", 0)
        score2 = data2.get("performance_score", 0)
        score_change = score2 - score1
        
        console.print(f"[bold]性能评分比较[/bold]")
        console.print(f"报告1 ({Path(report1).name}): {score1}/100")
        console.print(f"报告2 ({Path(report2).name}): {score2}/100")
        
        if score_change > 0:
            console.print(f"[green]提升: +{score_change} 分[/green]")
        elif score_change < 0:
            console.print(f"[red]下降: {score_change} 分[/red]")
        else:
            console.print("[yellow]无变化[/yellow]")
        
        # 比较瓶颈数量
        bottlenecks1 = len(data1.get("bottlenecks", []))
        bottlenecks2 = len(data2.get("bottlenecks", []))
        bottleneck_change = bottlenecks2 - bottlenecks1
        
        console.print(f"\n[bold]瓶颈数量比较[/bold]")
        console.print(f"报告1: {bottlenecks1} 个瓶颈")
        console.print(f"报告2: {bottlenecks2} 个瓶颈")
        
        if bottleneck_change > 0:
            console.print(f"[red]增加: +{bottleneck_change} 个瓶颈[/red]")
        elif bottleneck_change < 0:
            console.print(f"[green]减少: {bottleneck_change} 个瓶颈[/green]")
        else:
            console.print("[yellow]无变化[/yellow]")
        
        # 比较组件状态
        console.print(f"\n[bold]组件状态比较[/bold]")
        
        components = ["system_resources", "database_performance", "cache_performance", 
                     "vector_search_performance", "graph_computation_performance", "api_performance"]
        
        for component in components:
            comp1 = data1.get(component, {})
            comp2 = data2.get(component, {})
            
            status1 = comp1.get("status", "unknown")
            status2 = comp2.get("status", "unknown")
            
            if status1 != status2:
                console.print(f"{component}: {status1} → {status2}")
        
    except Exception as e:
        console.print(f"[red]❌ 比较失败: {e}[/red]")
        sys.exit(1)


def _display_analysis_summary(results: dict):
    """显示分析摘要"""
    console.print("\n[bold]📊 分析摘要[/bold]")
    
    # 基本信息表格
    table = Table(title="性能分析概览")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="magenta")
    
    table.add_row("分析时间", results.get("timestamp", "未知"))
    table.add_row("分析耗时", f"{results.get('analysis_duration', 0):.2f}s")
    table.add_row("性能评分", f"{results.get('performance_score', 0)}/100")
    table.add_row("发现问题", f"{len(results.get('bottlenecks', []))} 个")
    
    console.print(table)


def _display_detailed_bottlenecks(results: dict):
    """显示详细瓶颈信息"""
    bottlenecks = results.get("bottlenecks", [])
    
    if not bottlenecks:
        console.print("\n[green]✅ 未发现性能瓶颈[/green]")
        return
    
    console.print("\n[bold red]🚨 性能瓶颈详情[/bold red]")
    
    # 按严重程度分组
    by_severity = {}
    for bottleneck in bottlenecks:
        severity = bottleneck.severity
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(bottleneck)
    
    # 显示各严重程度的瓶颈
    for severity in ["critical", "high", "medium", "low"]:
        if severity in by_severity:
            color = {
                "critical": "red",
                "high": "bright_red",
                "medium": "yellow",
                "low": "bright_yellow"
            }[severity]
            
            console.print(f"\n[{color}]{severity.upper()} 级瓶颈[/{color}]")
            
            for i, bottleneck in enumerate(by_severity[severity], 1):
                panel = Panel(
                    f"[bold]问题:[/bold] {bottleneck.issue}\n"
                    f"[bold]影响:[/bold] {bottleneck.impact}\n"
                    f"[bold]建议:[/bold] {bottleneck.recommendation}\n"
                    f"[bold]预估提升:[/bold] {bottleneck.estimated_gain}",
                    title=f"{i}. {bottleneck.component}",
                    border_style=color
                )
                console.print(panel)


def _display_recommendations(results: dict):
    """显示优化建议"""
    recommendations = results.get("recommendations", [])
    
    if not recommendations:
        return
    
    console.print("\n[bold blue]💡 优化建议[/bold blue]")
    
    for i, recommendation in enumerate(recommendations, 1):
        console.print(f"{i}. {recommendation}")


def _display_optimization_plan(plan: dict):
    """显示优化计划"""
    phases = plan.get("phases", [])
    
    if not phases:
        return
    
    console.print("\n[bold green]🚀 优化计划[/bold green]")
    
    # 创建优化计划树
    tree = Tree("优化计划")
    
    for phase in phases:
        phase_branch = tree.add(f"[bold]阶段 {phase['phase']}: {phase['name']} ({phase['duration']})[/bold]")
        phase_branch.add(f"重点: {phase['focus']}")
        
        tasks_branch = phase_branch.add("任务列表")
        for task in phase['tasks'][:3]:  # 只显示前3个任务
            tasks_branch.add(f"• {task['task']}")
        
        if len(phase['tasks']) > 3:
            tasks_branch.add(f"• ... 还有 {len(phase['tasks']) - 3} 个任务")
    
    console.print(tree)


def _display_monitoring_summary(history: list):
    """显示监控摘要"""
    if not history:
        return
    
    console.print("\n[bold]📈 监控摘要[/bold]")
    
    # 计算统计信息
    scores = [h["score"] for h in history]
    issues = [h["issues"] for h in history]
    
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    avg_issues = sum(issues) / len(issues)
    max_issues = max(issues)
    min_issues = min(issues)
    
    # 创建统计表格
    table = Table(title="监控统计")
    table.add_column("指标", style="cyan")
    table.add_column("平均值", style="magenta")
    table.add_column("最大值", style="green")
    table.add_column("最小值", style="red")
    
    table.add_row("性能评分", f"{avg_score:.1f}", str(max_score), str(min_score))
    table.add_row("问题数量", f"{avg_issues:.1f}", str(max_issues), str(min_issues))
    
    console.print(table)
    
    # 趋势分析
    if len(scores) >= 2:
        trend = scores[-1] - scores[0]
        if trend > 5:
            console.print("[green]📈 性能呈上升趋势[/green]")
        elif trend < -5:
            console.print("[red]📉 性能呈下降趋势[/red]")
        else:
            console.print("[yellow]➡️ 性能保持稳定[/yellow]")


if __name__ == "__main__":
    app()