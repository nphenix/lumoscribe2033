"""
IDE 适配器

提供统一的 IDE 工具接口，支持 Cursor 和 RooCode 等 IDE。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class IDEAdapter(ABC):
    """IDE 适配器抽象基类"""

    @abstractmethod
    async def get_workspace_info(self) -> dict[str, Any]:
        """获取工作区信息"""
        pass

    @abstractmethod
    async def get_file_content(self, file_path: str) -> str | None:
        """获取文件内容"""
        pass

    @abstractmethod
    async def get_directory_tree(self, path: str = ".") -> dict[str, Any]:
        """获取目录树结构"""
        pass

    @abstractmethod
    async def search_files(
        self, pattern: str, content: bool = False
    ) -> list[dict[str, Any]]:
        """搜索文件"""
        pass

    @abstractmethod
    async def get_selection_context(self) -> dict[str, Any]:
        """获取选中内容的上下文"""
        pass


class CursorAdapter(IDEAdapter):
    """Cursor IDE 适配器"""

    async def get_workspace_info(self) -> dict[str, Any]:
        return {
            "ide": "cursor",
            "workspace_path": ".",
            "project_type": "python"
        }

    async def get_file_content(self, file_path: str) -> str | None:
        try:
            with open(file_path, encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None

    async def get_directory_tree(self, path: str = ".") -> dict[str, Any]:
        tree = {"name": path, "type": "directory", "children": []}
        try:
            for item in Path(path).iterdir():
                if item.is_file():
                    tree["children"].append({"name": item.name, "type": "file"})  # type: ignore[attr-defined]
                else:
                    tree["children"].append({"name": item.name, "type": "directory"})  # type: ignore[attr-defined]
        except Exception:
            pass
        return tree

    async def search_files(
        self, pattern: str, content: bool = False
    ) -> list[dict[str, Any]]:
        return []

    async def get_selection_context(self) -> dict[str, Any]:
        return {}


class RooCodeAdapter(IDEAdapter):
    """RooCode IDE 适配器"""

    async def get_workspace_info(self) -> dict[str, Any]:
        return {
            "ide": "roocode",
            "workspace_path": ".",
            "project_type": "python"
        }

    async def get_file_content(self, file_path: str) -> str | None:
        try:
            with open(file_path, encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None

    async def get_directory_tree(self, path: str = ".") -> dict[str, Any]:
        tree = {"name": path, "type": "directory", "children": []}
        try:
            for item in Path(path).iterdir():
                if item.is_file():
                    tree["children"].append({"name": item.name, "type": "file"})  # type: ignore[attr-defined]
                else:
                    tree["children"].append({"name": item.name, "type": "directory"})  # type: ignore[attr-defined]
        except Exception:
            pass
        return tree

    async def search_files(
        self, pattern: str, content: bool = False
    ) -> list[dict[str, Any]]:
        return []

    async def get_selection_context(self) -> dict[str, Any]:
        return {}
