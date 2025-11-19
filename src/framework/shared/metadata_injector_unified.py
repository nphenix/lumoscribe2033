"""
统一的元数据注入器

使用策略模式整合基础版和增强版元数据注入功能，遵循 SOLID 原则：
- 单一职责：每个策略负责自己的实现
- 开闭原则：可以轻松扩展新的注入策略
- 依赖倒置：客户端依赖抽象接口

使用方式：
    # 基础注入（默认）
    injector = MetadataInjector()
    injector.inject(file_path, command)
    
    # 增强注入（需要 LangChain）
    injector = MetadataInjector(strategy="enhanced")
    injector.inject(file_path, command)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from src.framework.shared.metadata_injector import (
    VerificationResult,
    format_header,
    inject_header as basic_inject_header,
    parse_header_line,
    verify_file as basic_verify_file,
    verify_directory as basic_verify_directory,
    bulk_inject as basic_bulk_inject,
)


@dataclass
class InjectResult:
    """注入结果"""
    success: bool
    file_path: Path
    message: str
    metadata: dict[str, Any] | None = None


class MetadataInjectionStrategy(ABC):
    """元数据注入策略抽象基类"""
    
    @abstractmethod
    def inject(self, file_path: str | Path, command: str, update_if_exists: bool = False) -> InjectResult:
        """注入元数据到文件"""
        pass
    
    @abstractmethod
    def verify(self, file_path: str | Path) -> dict[str, Any]:
        """验证文件元数据"""
        pass


class BasicMetadataStrategy(MetadataInjectionStrategy):
    """基础元数据注入策略
    
    提供简单的元数据注入功能，不依赖 LangChain。
    适用于所有场景，性能开销小。
    """
    
    def inject(self, file_path: str | Path, command: str, update_if_exists: bool = False) -> InjectResult:
        """注入基础元数据"""
        try:
            p = Path(file_path)
            if not p.exists() or not p.is_file():
                return InjectResult(
                    success=False,
                    file_path=p,
                    message=f"文件不存在: {p}"
                )
            
            changed = basic_inject_header(p, command, update_if_exists=update_if_exists)
            
            if changed:
                return InjectResult(
                    success=True,
                    file_path=p,
                    message="元数据注入成功",
                    metadata={"command": command, "strategy": "basic"}
                )
            else:
                return InjectResult(
                    success=True,
                    file_path=p,
                    message="元数据已存在，未更新",
                    metadata={"command": command, "strategy": "basic", "updated": False}
                )
        except Exception as e:
            return InjectResult(
                success=False,
                file_path=Path(file_path),
                message=f"注入失败: {str(e)}"
            )
    
    def verify(self, file_path: str | Path) -> dict[str, Any]:
        """验证基础元数据"""
        result = basic_verify_file(file_path)
        return {
            "valid": result.has_header,
            "file_path": str(result.path),
            "command": result.command,
            "timestamp": result.timestamp,
            "strategy": "basic"
        }


class EnhancedMetadataStrategy(MetadataInjectionStrategy):
    """增强元数据注入策略
    
    使用 LangChain 1.0 提供智能分析功能，包括：
    - 内容摘要生成
    - 质量评估
    - 问题检测
    
    需要 LangChain 依赖，性能开销较大。
    """
    
    def __init__(self):
        """初始化增强策略"""
        try:
            from src.framework.shared.metadata_injector_enhanced import EnhancedMetadataInjector
            self.injector = EnhancedMetadataInjector()
            self.available = True
        except ImportError:
            self.injector = None
            self.available = False
    
    def inject(self, file_path: str | Path, command: str, update_if_exists: bool = False) -> InjectResult:
        """注入增强元数据"""
        if not self.available:
            # 降级到基础策略
            basic = BasicMetadataStrategy()
            result = basic.inject(file_path, command, update_if_exists)
            result.message = f"增强策略不可用，已降级到基础策略: {result.message}"
            return result
        
        try:
            success = self.injector.inject_metadata(str(file_path), command)
            if success:
                return InjectResult(
                    success=True,
                    file_path=Path(file_path),
                    message="增强元数据注入成功",
                    metadata={"command": command, "strategy": "enhanced"}
                )
            else:
                return InjectResult(
                    success=False,
                    file_path=Path(file_path),
                    message="增强元数据注入失败"
                )
        except Exception as e:
            return InjectResult(
                success=False,
                file_path=Path(file_path),
                message=f"注入失败: {str(e)}"
            )
    
    def verify(self, file_path: str | Path) -> dict[str, Any]:
        """验证增强元数据"""
        if not self.available:
            # 降级到基础策略
            basic = BasicMetadataStrategy()
            return basic.verify(file_path)
        
        try:
            return self.injector.verify_metadata(str(file_path))
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "strategy": "enhanced"
            }


class MetadataInjector:
    """统一的元数据注入器
    
    使用策略模式，支持基础版和增强版注入策略。
    默认使用基础策略，可通过 strategy 参数切换。
    """
    
    _strategies: dict[str, type[MetadataInjectionStrategy]] = {
        "basic": BasicMetadataStrategy,
        "enhanced": EnhancedMetadataStrategy,
    }
    
    def __init__(self, strategy: str = "basic"):
        """初始化注入器
        
        Args:
            strategy: 注入策略，可选 "basic" 或 "enhanced"
        """
        if strategy not in self._strategies:
            raise ValueError(f"未知的策略: {strategy}，可选: {list(self._strategies.keys())}")
        
        self.strategy_name = strategy
        self.strategy = self._strategies[strategy]()
    
    def inject(
        self,
        file_path: str | Path,
        command: str,
        update_if_exists: bool = False
    ) -> InjectResult:
        """注入元数据到文件
        
        Args:
            file_path: 文件路径
            command: 生成命令
            update_if_exists: 如果元数据已存在，是否更新
        
        Returns:
            注入结果
        """
        return self.strategy.inject(file_path, command, update_if_exists)
    
    def verify(self, file_path: str | Path) -> dict[str, Any]:
        """验证文件元数据
        
        Args:
            file_path: 文件路径
        
        Returns:
            验证结果字典
        """
        return self.strategy.verify(file_path)
    
    def bulk_inject(
        self,
        root: str | Path,
        command: str,
        include_globs: Iterable[str] = ("**/*.md",),
        exclude_globs: Iterable[str] = (".git/**",),
        update_if_exists: bool = False,
    ) -> list[InjectResult]:
        """批量注入元数据
        
        Args:
            root: 根目录
            command: 生成命令
            include_globs: 包含的文件模式
            exclude_globs: 排除的文件模式
            update_if_exists: 如果元数据已存在，是否更新
        
        Returns:
            注入结果列表
        """
        root_path = Path(root)
        results: list[InjectResult] = []
        
        # 收集文件
        matched: set[Path] = set()
        for pattern in include_globs:
            matched.update(root_path.glob(pattern))
        
        excluded: set[Path] = set()
        for pattern in exclude_globs:
            excluded.update(root_path.glob(pattern))
        
        targets = [p for p in matched if p.is_file() and p not in excluded]
        
        # 批量注入
        for target in targets:
            result = self.inject(target, command, update_if_exists)
            results.append(result)
        
        return results
    
    def verify_directory(
        self,
        root: str | Path,
        include_globs: Iterable[str] = ("**/*.md",),
        exclude_globs: Iterable[str] = (".git/**",),
    ) -> list[dict[str, Any]]:
        """验证目录下文件的元数据
        
        Args:
            root: 根目录
            include_globs: 包含的文件模式
            exclude_globs: 排除的文件模式
        
        Returns:
            验证结果列表
        """
        root_path = Path(root)
        results: list[dict[str, Any]] = []
        
        # 收集文件
        matched: set[Path] = set()
        for pattern in include_globs:
            matched.update(root_path.glob(pattern))
        
        excluded: set[Path] = set()
        for pattern in exclude_globs:
            excluded.update(root_path.glob(pattern))
        
        targets = [p for p in matched if p.is_file() and p not in excluded]
        
        # 批量验证
        for target in targets:
            result = self.verify(target)
            results.append(result)
        
        return results
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[MetadataInjectionStrategy]) -> None:
        """注册新的注入策略（扩展点）
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        if not issubclass(strategy_class, MetadataInjectionStrategy):
            raise TypeError(f"策略类必须继承 MetadataInjectionStrategy")
        cls._strategies[name] = strategy_class


# 向后兼容：导出基础函数
__all__ = [
    "MetadataInjector",
    "MetadataInjectionStrategy",
    "BasicMetadataStrategy",
    "EnhancedMetadataStrategy",
    "InjectResult",
    # 向后兼容导出
    "VerificationResult",
    "format_header",
    "parse_header_line",
    "basic_inject_header",
    "basic_verify_file",
    "basic_verify_directory",
    "basic_bulk_inject",
]

