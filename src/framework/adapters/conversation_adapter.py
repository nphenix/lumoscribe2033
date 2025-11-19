"""
对话适配器

提供统一的对话存储接口，支持不同格式的对话记录导入和导出。
"""

import csv
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ConversationExportFormat(str, Enum):
    """对话导出格式枚举"""
    JSON = "json"
    CSV = "csv"
    TEXT = "txt"
    XML = "xml"


class IDELogFormat(str, Enum):
    """IDE 日志格式枚举"""
    CURSOR = "cursor"
    ROOCODE = "roocode"
    VSCODE = "vscode"
    INTELLIJ = "intellij"
    ECLIPSE = "eclipse"


class ConversationParseError(Exception):
    """对话解析错误"""
    def __init__(self, message: str, source_file: str | None = None, line_number: int | None = None):
        self.source_file = source_file
        self.line_number = line_number
        super().__init__(message)


class ConversationParseResult(BaseModel):
    """对话解析结果"""
    success: bool
    processed_count: int = 0
    failed_count: int = 0
    total_lines: int = 0
    warnings: list[str] = []
    errors: list[str] = []
    metadata: dict[str, Any] = {}


class ConversationAdapter(ABC):
    """对话适配器抽象基类"""

    @abstractmethod
    async def import_conversations(
        self, source_path: str, batch_size: int = 100
    ) -> dict[str, Any]:
        """导入对话记录"""
        pass

    @abstractmethod
    async def export_conversations(
        self, output_path: str, format: str = "json"
    ) -> dict[str, Any]:
        """导出对话记录"""
        pass

    @abstractmethod
    async def search_conversations(
        self, query: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """搜索对话"""
        pass

    @abstractmethod
    async def get_conversation_stats(self) -> dict[str, Any]:
        """获取对话统计信息"""
        pass


class BaseIDEConversationAdapter(ConversationAdapter):
    """IDE 对话适配器基类
    
    提取 Cursor 和 RooCode 适配器的共同逻辑，遵循 DRY 原则。
    子类只需实现特定 IDE 的日志解析逻辑。
    """
    
    def __init__(self, source_name: str):
        """初始化基类适配器
        
        Args:
            source_name: IDE 名称（如 'cursor', 'roocode'）
        """
        self.source_name = source_name
        self.conversations: list[dict[str, Any]] = []
        self.logger = None
    
    async def import_conversations(
        self, source_path: str, batch_size: int = 100
    ) -> dict[str, Any]:
        """导入对话记录 - 通用实现
        
        子类需要实现 `_parse_log_file` 方法来处理特定格式的日志。
        """
        import time
        
        start_time = time.time()
        imported_count = 0
        failed_count = 0
        
        source_path_obj = Path(source_path)
        
        if not source_path_obj.exists():
            raise FileNotFoundError(f"{self.source_name} 日志文件不存在: {source_path}")
        
        # 支持单个文件或目录
        log_files = self._collect_log_files(source_path_obj)
        
        for log_file in log_files:
            try:
                await self._parse_log_file(log_file)
                imported_count += 1
            except Exception as e:
                failed_count += 1
                if self.logger:
                    self.logger.error(f"解析 {self.source_name} 日志失败 {log_file}: {str(e)}")
        
        # 合并对话记录
        self._merge_conversations()
        
        execution_time = time.time() - start_time
        
        result = {
            "source": self.source_name,
            "imported_count": imported_count,
            "failed_count": failed_count,
            "total_time": execution_time,
            "total_conversations": len(self.conversations)
        }
        
        if self.logger:
            self.logger.info(f"{self.source_name} 日志导入完成: {result}")
        
        return result
    
    def _collect_log_files(self, source_path: Path) -> list[Path]:
        """收集日志文件
        
        子类可以重写此方法以支持不同的文件扩展名。
        """
        if source_path.is_dir():
            log_files = list(source_path.glob("*.log"))
            log_files.extend(source_path.glob("*.txt"))
            log_files.extend(source_path.glob("*.json"))
            return log_files
        else:
            return [source_path]
    
    @abstractmethod
    async def _parse_log_file(self, log_file: Path) -> None:
        """解析单个日志文件 - 子类必须实现"""
        pass
    
    def _merge_conversations(self, time_window_seconds: int = 1800):
        """合并相关的对话记录 - 通用实现
        
        Args:
            time_window_seconds: 时间窗口（秒），默认 30 分钟
        """
        # 按时间戳排序所有对话
        self.conversations.sort(
            key=lambda x: x["messages"][0]["timestamp"] if x["messages"] else ""
        )
        
        # 简单的时间窗口合并逻辑
        merged = []
        current_batch = []
        
        for conversation in self.conversations:
            if not conversation["messages"]:
                continue
            
            if not current_batch:
                current_batch = [conversation]
            else:
                # 检查时间间隔
                last_msg_time = current_batch[-1]["messages"][-1]["timestamp"]
                first_msg_time = conversation["messages"][0]["timestamp"]
                
                last_time = datetime.fromisoformat(last_msg_time.replace('Z', '+00:00'))
                first_time = datetime.fromisoformat(first_msg_time.replace('Z', '+00:00'))
                
                if (first_time - last_time).total_seconds() < time_window_seconds:
                    current_batch.append(conversation)
                else:
                    # 合并当前批次
                    merged.append(self._combine_conversations(current_batch))
                    current_batch = [conversation]
        
        # 合并最后一个批次
        if current_batch:
            merged.append(self._combine_conversations(current_batch))
        
        self.conversations = merged
    
    def _combine_conversations(self, conversations: list[dict]) -> dict:
        """合并多个对话记录 - 通用实现"""
        if not conversations:
            return {}
        
        combined = conversations[0].copy()
        combined["messages"] = []
        combined["id"] = f"combined_{conversations[0]['id']}"
        
        for conv in conversations:
            combined["messages"].extend(conv["messages"])
        
        # 按时间戳排序消息
        combined["messages"].sort(key=lambda x: x["timestamp"])
        
        return combined
    
    async def export_conversations(
        self, output_path: str, format: str = "json"
    ) -> dict[str, Any]:
        """导出对话记录 - 通用实现"""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            return await self._export_json(output_path_obj)
        elif format.lower() == "csv":
            return await self._export_csv(output_path_obj)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    async def _export_json(self, output_path: Path) -> dict[str, Any]:
        """导出为 JSON 格式"""
        data = {
            "export_date": datetime.now().isoformat(),
            "source": self.source_name,
            "total_conversations": len(self.conversations),
            "conversations": self.conversations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {
            "format": "json",
            "exported_count": len(self.conversations),
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }
    
    async def _export_csv(self, output_path: Path) -> dict[str, Any]:
        """导出为 CSV 格式"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["conversation_id", "message_id", "role", "content", "timestamp"])
            
            for conv in self.conversations:
                for msg_idx, msg in enumerate(conv["messages"]):
                    writer.writerow([
                        conv["id"],
                        f"{conv['id']}_msg_{msg_idx}",
                        msg.get("role", "unknown"),
                        msg.get("content", ""),
                        msg.get("timestamp", "")
                    ])
        
        return {
            "format": "csv",
            "exported_count": len(self.conversations),
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }
    
    async def search_conversations(
        self, query: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """搜索对话 - 通用实现"""
        results = []
        query_lower = query.lower()
        
        for conv in self.conversations:
            for msg in conv["messages"]:
                content = msg.get("content", "").lower()
                if query_lower in content:
                    results.append({
                        "conversation_id": conv["id"],
                        "message": msg,
                        "match_score": content.count(query_lower)
                    })
                    break
        
        # 按匹配分数排序
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:limit]
    
    async def get_conversation_stats(self) -> dict[str, Any]:
        """获取对话统计信息 - 通用实现"""
        total_messages = sum(len(conv["messages"]) for conv in self.conversations)
        
        role_counts = {}
        for conv in self.conversations:
            for msg in conv["messages"]:
                role = msg.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "role_distribution": role_counts,
            "source": self.source_name
        }


class CursorConversationAdapter(BaseIDEConversationAdapter):
    """Cursor 对话适配器

    基于 Cursor IDE 日志格式解析对话记录。
    Cursor 日志通常包含：
    - 对话消息记录
    - 代码编辑操作
    - 文件操作历史
    - AI 助手交互
    """

    def __init__(self):
        import re
        super().__init__("cursor")
        
        self.log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(.*?)\] (.*?)'
        )
        self.message_pattern = re.compile(
            r'(User|Assistant|System): (.*)'
        )

    async def _parse_log_file(self, log_file: Path) -> None:
        """解析单个 Cursor 日志文件"""
        await self._parse_cursor_log(log_file)
    
    async def _parse_cursor_log(self, log_file: Path) -> None:
        """解析单个 Cursor 日志文件"""

        current_conversation = {
            "id": f"cursor_{log_file.stem}_{len(self.conversations)}",
            "source": "cursor",
            "file": str(log_file),
            "messages": [],
            "metadata": {
                "log_file": str(log_file),
                "parsed_at": datetime.now().isoformat()
            }
        }

        current_message = None

        with open(log_file, encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 尝试匹配日志行
                log_match = self.log_pattern.match(line)
                if log_match:
                    timestamp_str, level, content = log_match.groups()

                    # 解析时间戳
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue

                    # 尝试匹配消息
                    msg_match = self.message_pattern.match(content)
                    if msg_match:
                        role, message_content = msg_match.groups()

                        # 保存当前消息（如果有）
                        if current_message:
                            current_conversation["messages"].append(current_message)

                        # 开始新消息
                        current_message = {
                            "role": role.lower(),
                            "content": message_content,
                            "timestamp": timestamp.isoformat(),
                            "source_line": line_num
                        }
                    else:
                        # 处理其他类型的日志
                        self._process_cursor_event(current_conversation, timestamp, level, content)
                else:
                    # 处理多行消息内容
                    if current_message and line:
                        current_message["content"] += "\n" + line

        # 保存最后一条消息
        if current_message:
            current_conversation["messages"].append(current_message)

        # 只有当有消息时才添加对话
        if current_conversation["messages"]:
            self.conversations.append(current_conversation)

    def _process_cursor_event(self, conversation: dict, timestamp, level: str, content: str):
        """处理 Cursor 事件日志"""
        # 处理代码编辑事件
        if "edit" in content.lower() or "change" in content.lower():
            conversation.setdefault("code_edits", []).append({
                "timestamp": timestamp.isoformat(),
                "event": content,
                "type": "code_edit"
            })

        # 处理文件操作事件
        elif any(keyword in content.lower() for keyword in ["file", "open", "save", "create"]):
            conversation.setdefault("file_operations", []).append({
                "timestamp": timestamp.isoformat(),
                "event": content,
                "type": "file_operation"
            })

        # 处理 AI 助手交互
        elif any(keyword in content.lower() for keyword in ["ai", "assistant", "model", "llm"]):
            conversation.setdefault("ai_interactions", []).append({
                "timestamp": timestamp.isoformat(),
                "event": content,
                "type": "ai_interaction"
            })

    async def search_conversations(
        self, query: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """搜索对话"""

        results = []
        query_lower = query.lower()

        for conversation in self.conversations:
            conversation_matches = []

            for msg in conversation["messages"]:
                if query_lower in msg["content"].lower():
                    conversation_matches.append({
                        "message": msg,
                        "score": self._calculate_relevance_score(msg["content"], query)
                    })

            if conversation_matches:
                # 按相关性排序
                conversation_matches.sort(key=lambda x: x["score"], reverse=True)

                results.append({
                    "conversation": conversation,
                    "matches": conversation_matches[:10],  # 每个对话最多10个匹配
                    "total_matches": len(conversation_matches)
                })

        # 按总匹配数排序
        results.sort(key=lambda x: x["total_matches"], reverse=True)

        return results[:limit]

    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """计算相关性分数"""

        content_lower = content.lower()
        query_lower = query.lower()

        # 基础匹配分数
        score = 0.0

        # 完全匹配
        if query_lower in content_lower:
            score += 0.5

        # 词频分数
        query_words = query_lower.split()
        content_words = content_lower.split()
        word_matches = sum(1 for word in query_words if word in content_words)
        if query_words:
            score += word_matches / len(query_words) * 0.3

        # 位置分数（前面的匹配更有价值）
        if query_lower in content_lower:
            position = content_lower.find(query_lower)
            position_score = 1.0 - (position / len(content_lower))
            score += position_score * 0.2

        return min(score, 1.0)

    async def get_conversation_stats(self) -> dict[str, Any]:
        """获取对话统计信息"""

        if not self.conversations:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "avg_conversation_length": 0,
                "date_range": {"start": None, "end": None},
                "message_types": {},
                "source_files": []
            }

        total_messages = sum(len(conv["messages"]) for conv in self.conversations)
        message_lengths = [len(conv["messages"]) for conv in self.conversations]

        # 收集所有时间戳
        all_timestamps = []
        for conv in self.conversations:
            for msg in conv["messages"]:
                all_timestamps.append(msg["timestamp"])

        # 解析时间范围
        date_range = {"start": None, "end": None}
        if all_timestamps:
            all_timestamps.sort()
            date_range["start"] = all_timestamps[0]
            date_range["end"] = all_timestamps[-1]

        # 统计消息类型
        message_types = {}
        for conv in self.conversations:
            for msg in conv["messages"]:
                role = msg["role"]
                message_types[role] = message_types.get(role, 0) + 1

        # 统计源文件
        source_files = list(set(
            conv.get("metadata", {}).get("log_file", "")
            for conv in self.conversations
        ))

        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "avg_conversation_length": total_messages / len(self.conversations) if self.conversations else 0,
            "min_conversation_length": min(message_lengths) if message_lengths else 0,
            "max_conversation_length": max(message_lengths) if message_lengths else 0,
            "median_conversation_length": sorted(message_lengths)[len(message_lengths) // 2] if message_lengths else 0,
            "date_range": date_range,
            "message_types": message_types,
            "source_files": source_files,
            "avg_messages_per_day": self._calculate_avg_messages_per_day(all_timestamps)
        }

    def _calculate_avg_messages_per_day(self, timestamps: list[str]) -> float:
        """计算每日平均消息数"""
        if not timestamps:
            return 0.0

        from datetime import datetime

        # 解析时间戳
        dates = []
        for ts in timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                dates.append(dt.date())
            except (ValueError, TypeError):
                continue

        if not dates:
            return 0.0

        # 计算日期范围
        start_date = min(dates)
        end_date = max(dates)

        # 计算天数
        days = (end_date - start_date).days + 1

        return len(dates) / days if days > 0 else len(dates)


class RooCodeConversationAdapter(BaseIDEConversationAdapter):
    """RooCode 对话适配器

    基于 RooCode IDE 日志格式解析对话记录。
    RooCode 日志通常包含：
    - 结构化的对话消息
    - 代码操作记录
    - 文件变更历史
    - AI 助手交互事件
    """

    def __init__(self):
        import re
        super().__init__("roocode")

        # RooCode 日志格式模式
        self.json_log_pattern = re.compile(r'^\s*{\s*"timestamp"')
        self.simple_log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\s*-\s*(\w+)\s*-\s*(.+)'
        )
        self.message_pattern = re.compile(r'(用户|助手|系统)\s*[:：]\s*(.+)', re.UNICODE)

    async def _parse_log_file(self, log_file: Path) -> None:
        """解析单个 RooCode 日志文件"""
        await self._parse_roocode_log(log_file)
    
    async def _parse_roocode_log(self, log_file: Path) -> None:
        """解析单个 RooCode 日志文件"""

        current_conversation = {
            "id": f"roocode_{log_file.stem}_{len(self.conversations)}",
            "source": "roocode",
            "file": str(log_file),
            "messages": [],
            "metadata": {
                "log_file": str(log_file),
                "parsed_at": datetime.now().isoformat(),
                "file_format": "unknown"
            }
        }

        # 尝试不同的解析方法
        try:
            # 方法1: 尝试 JSON 格式解析
            if log_file.suffix.lower() in ['.json']:
                await self._parse_json_format(log_file, current_conversation)
                current_conversation["metadata"]["file_format"] = "json"
            else:
                # 方法2: 尝试结构化文本格式解析
                await self._parse_structured_text(log_file, current_conversation)
                current_conversation["metadata"]["file_format"] = "structured_text"

        except Exception:
            # 方法3: 尝试简单文本格式解析
            try:
                await self._parse_simple_text(log_file, current_conversation)
                current_conversation["metadata"]["file_format"] = "simple_text"
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"RooCode 日志解析失败，尝试备选方法: {e}")
                # 方法4: 通用文本解析
                await self._parse_generic_text(log_file, current_conversation)
                current_conversation["metadata"]["file_format"] = "generic_text"

    async def _parse_json_format(self, log_file: Path, conversation: dict) -> None:
        """解析 JSON 格式日志"""
        with open(log_file, encoding='utf-8') as f:
            try:
                # 尝试解析整个文件为一个 JSON
                data = json.load(f)

                if isinstance(data, list):
                    # 如果是数组，处理每个条目
                    for entry in data:
                        self._process_json_entry(entry, conversation)
                elif isinstance(data, dict):
                    # 如果是单个对象，可能需要处理嵌套的记录
                    if "records" in data or "logs" in data:
                        records = data.get("records", data.get("logs", []))
                        for entry in records:
                            self._process_json_entry(entry, conversation)
                    else:
                        self._process_json_entry(data, conversation)

            except json.JSONDecodeError:
                # 如果不是有效的 JSON，逐行解析
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        try:
                            entry = json.loads(line)
                            self._process_json_entry(entry, conversation)
                        except json.JSONDecodeError:
                            continue

    def _process_json_entry(self, entry: dict, conversation: dict) -> None:
        """处理 JSON 日志条目"""

        # 提取基本信息
        timestamp = entry.get("timestamp", entry.get("time", entry.get("date", "")))
        if isinstance(timestamp, (int, float)):
            # 时间戳格式
            dt = datetime.fromtimestamp(timestamp)
            timestamp_str = dt.isoformat()
        elif isinstance(timestamp, str):
            # ISO 格式时间字符串
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp_str = dt.isoformat()
            except ValueError:
                timestamp_str = timestamp
        else:
            timestamp_str = datetime.now().isoformat()

        # 提取消息内容
        role = entry.get("role", entry.get("type", "user")).lower()
        content = entry.get("content", entry.get("message", entry.get("text", "")))

        # 处理特殊字段
        if not content and "query" in entry:
            content = entry["query"]
            role = "user"
        elif not content and "response" in entry:
            content = entry["response"]
            role = "assistant"

        if content:
            message = {
                "role": role,
                "content": str(content),
                "timestamp": timestamp_str,
                "source_entry": entry
            }
            conversation["messages"].append(message)

    async def _parse_structured_text(self, log_file: Path, conversation: dict) -> None:
        """解析结构化文本格式"""
        with open(log_file, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 尝试匹配结构化日志行
                match = self.simple_log_pattern.match(line)
                if match:
                    timestamp_str, level, content = match.groups()

                    # 解析时间戳
                    try:
                        timestamp = self._parse_timestamp(timestamp_str)
                    except ValueError:
                        continue

                    # 提取消息
                    msg_match = self.message_pattern.match(content)
                    if msg_match:
                        role_cn, message_content = msg_match.groups()
                        role = self._convert_role(role_cn)

                        message = {
                            "role": role,
                            "content": message_content,
                            "timestamp": timestamp.isoformat(),
                            "source_line": line_num,
                            "log_level": level
                        }
                        conversation["messages"].append(message)

    async def _parse_simple_text(self, log_file: Path, conversation: dict) -> None:
        """解析简单文本格式"""
        with open(log_file, encoding='utf-8') as f:
            current_message = None

            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 检查是否是新的消息开始
                msg_match = self.message_pattern.match(line)
                if msg_match:
                    # 保存当前消息
                    if current_message:
                        conversation["messages"].append(current_message)

                    # 开始新消息
                    role_cn, message_content = msg_match.groups()
                    role = self._convert_role(role_cn)

                    current_message = {
                        "role": role,
                        "content": message_content,
                        "timestamp": conversation["messages"][-1]["timestamp"] if conversation["messages"] else datetime.now().isoformat(),
                        "source_line": line_num
                    }
                elif current_message:
                    # 继续当前消息
                    current_message["content"] += "\n" + line

            # 保存最后一条消息
            if current_message:
                conversation["messages"].append(current_message)

    async def _parse_generic_text(self, log_file: Path, conversation: dict) -> None:
        """通用文本解析方法"""
        with open(log_file, encoding='utf-8') as f:
            content = f.read()

            # 查找可能的消息模式
            patterns = [
                r'(?:用户|User|USER)[:：]\s*(.*?)(?=\n(?:助手|助理|Assistant|ASSISTANT)[:：]|\n(?:系统|System|SYSTEM)[:：]|\n$)',
                r'(?:助手|助理|Assistant|ASSISTANT)[:：]\s*(.*?)(?=\n(?:用户|User|USER)[:：]|\n(?:系统|System|SYSTEM)[:：]|\n$)',
                r'(?:系统|System|SYSTEM)[:：]\s*(.*?)(?=\n(?:用户|User|USER)[:：]|\n(?:助手|助理|Assistant|ASSISTANT)[:：]|\n$)'
            ]

            import re

            role_mapping = {
                '用户': 'user', 'User': 'user', 'USER': 'user',
                '助手': 'assistant', '助理': 'assistant', 'Assistant': 'assistant', 'ASSISTANT': 'assistant',
                '系统': 'system', 'System': 'system', 'SYSTEM': 'system'
            }

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.DOTALL)
                for match in matches:
                    message_content = match.group(1).strip()
                    if message_content:
                        # 从上下文中推断角色
                        context = content[max(0, match.start()-100):match.start()]
                        role = self._infer_role_from_context(context, role_mapping)

                        message = {
                            "role": role,
                            "content": message_content,
                            "timestamp": datetime.now().isoformat(),
                            "source_pattern": pattern
                        }
                        conversation["messages"].append(message)

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """解析时间戳字符串"""
        import re

        # 移除毫秒和时区信息的标准化
        timestamp_str = re.sub(r'\.\d+', '', timestamp_str)
        timestamp_str = re.sub(r'Z$', '+00:00', timestamp_str)

        # 尝试不同的时间格式
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # 如果都失败了，返回当前时间
        return datetime.now()

    def _convert_role(self, role_cn: str) -> str:
        """转换中文角色名为标准角色名"""
        role_mapping = {
            '用户': 'user',
            'User': 'user',
            'USER': 'user',
            '助手': 'assistant',
            '助理': 'assistant',
            'Assistant': 'assistant',
            'ASSISTANT': 'assistant',
            '系统': 'system',
            'System': 'system',
            'SYSTEM': 'system'
        }
        return role_mapping.get(role_cn, 'user')

    def _infer_role_from_context(self, context: str, role_mapping: dict) -> str:
        """从上下文推断角色"""
        for cn_role, std_role in role_mapping.items():
            if cn_role in context:
                return std_role
        return 'user'  # 默认角色

    def _merge_conversations(self, time_window_seconds: int = 3600):
        """合并相关的对话记录（RooCode 特定逻辑）
        
        重写基类方法，使用更宽松的合并策略：60分钟内且有相同的源文件。
        """
        # 按时间戳排序所有对话
        self.conversations.sort(key=lambda x: x["messages"][0]["timestamp"] if x["messages"] else "")

        # 基于时间窗口和会话 ID 合并
        merged = []
        current_batch = []

        for conversation in self.conversations:
            if not conversation["messages"]:
                continue

            if not current_batch:
                current_batch = [conversation]
            else:
                # 检查时间间隔和会话连续性
                last_msg_time = current_batch[-1]["messages"][-1]["timestamp"]
                first_msg_time = conversation["messages"][0]["timestamp"]

                last_time = datetime.fromisoformat(last_msg_time.replace('Z', '+00:00'))
                first_time = datetime.fromisoformat(first_msg_time.replace('Z', '+00:00'))

                # 更宽松的合并策略：60分钟内且有相同的源文件
                time_diff = (first_time - last_time).total_seconds()
                same_source = (
                    conversation.get("metadata", {}).get("log_file", "") ==
                    current_batch[0].get("metadata", {}).get("log_file", "")
                )

                if time_diff < time_window_seconds and same_source:
                    current_batch.append(conversation)
                else:
                    # 合并当前批次
                    merged.append(self._combine_conversations(current_batch))
                    current_batch = [conversation]

        # 合并最后一个批次
        if current_batch:
            merged.append(self._combine_conversations(current_batch))

        self.conversations = merged

    async def export_conversations(
        self, output_path: str, format: str = "json"
    ) -> dict[str, Any]:
        """导出对话记录"""

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            return await self._export_json(output_path_obj)
        elif format.lower() == "csv":
            return await self._export_csv(output_path_obj)
        elif format.lower() == "txt":
            return await self._export_text(output_path_obj)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    async def _export_json(self, output_path: Path) -> dict[str, Any]:
        """导出为 JSON 格式"""
        data = {
            "export_date": datetime.now().isoformat(),
            "source": "roocode",
            "format_version": "1.0",
            "total_conversations": len(self.conversations),
            "conversations": self.conversations
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {
            "format": "json",
            "exported_count": len(self.conversations),
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }

    async def _export_csv(self, output_path: Path) -> dict[str, Any]:
        """导出为 CSV 格式"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "conversation_id", "message_id", "role", "content",
                "timestamp", "source_file", "format_type"
            ])

            message_id = 0
            for conv in self.conversations:
                for msg in conv["messages"]:
                    message_id += 1
                    writer.writerow([
                        conv["id"],
                        f"msg_{message_id}",
                        msg["role"],
                        msg["content"][:1000],  # 限制长度
                        msg["timestamp"],
                        conv.get("metadata", {}).get("log_file", ""),
                        conv.get("metadata", {}).get("file_format", "unknown")
                    ])

        return {
            "format": "csv",
            "exported_count": len(self.conversations),
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }

    async def _export_text(self, output_path: Path) -> dict[str, Any]:
        """导出为纯文本格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"RooCode 对话导出 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for i, conv in enumerate(self.conversations, 1):
                f.write(f"对话 {i}: {conv['id']}\n")
                f.write(f"源文件: {conv.get('metadata', {}).get('log_file', 'unknown')}\n")
                f.write(f"格式类型: {conv.get('metadata', {}).get('file_format', 'unknown')}\n")
                f.write("-" * 40 + "\n")

                for msg in conv["messages"]:
                    timestamp = msg.get("timestamp", "")
                    role = msg["role"].upper()
                    content = msg["content"]

                    f.write(f"[{timestamp}] {role}:\n")
                    f.write(f"{content}\n\n")

                f.write("\n" + "=" * 80 + "\n\n")

        return {
            "format": "txt",
            "exported_count": len(self.conversations),
            "output_path": str(output_path),
            "file_size": output_path.stat().st_size if output_path.exists() else 0
        }

    async def search_conversations(
        self, query: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """搜索对话"""

        results = []
        query_lower = query.lower()

        for conversation in self.conversations:
            conversation_matches = []

            for msg in conversation["messages"]:
                if query_lower in msg["content"].lower():
                    conversation_matches.append({
                        "message": msg,
                        "score": self._calculate_relevance_score(msg["content"], query)
                    })

            if conversation_matches:
                # 按相关性排序
                conversation_matches.sort(key=lambda x: x["score"], reverse=True)

                results.append({
                    "conversation": conversation,
                    "matches": conversation_matches[:10],  # 每个对话最多10个匹配
                    "total_matches": len(conversation_matches),
                    "source_file": conversation.get("metadata", {}).get("log_file", ""),
                    "format_type": conversation.get("metadata", {}).get("file_format", "")
                })

        # 按总匹配数和相关性排序
        results.sort(key=lambda x: (
            -x["total_matches"],  # 匹配数降序
            -max(m["score"] for m in x["matches"]) if x["matches"] else 0  # 最高相关性降序
        ))

        return results[:limit]

    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """计算相关性分数"""

        content_lower = content.lower()
        query_lower = query.lower()

        # 基础匹配分数
        score = 0.0

        # 完全匹配
        if query_lower in content_lower:
            score += 0.5

        # 词频分数
        query_words = query_lower.split()
        content_words = content_lower.split()
        word_matches = sum(1 for word in query_words if word in content_words)
        if query_words:
            score += word_matches / len(query_words) * 0.3

        # 位置分数（前面的匹配更有价值）
        if query_lower in content_lower:
            position = content_lower.find(query_lower)
            position_score = 1.0 - (position / len(content_lower))
            score += position_score * 0.2

        return min(score, 1.0)

    async def get_conversation_stats(self) -> dict[str, Any]:
        """获取对话统计信息"""

        if not self.conversations:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "avg_conversation_length": 0,
                "date_range": {"start": None, "end": None},
                "message_types": {},
                "source_files": [],
                "format_types": {}
            }

        total_messages = sum(len(conv["messages"]) for conv in self.conversations)
        message_lengths = [len(conv["messages"]) for conv in self.conversations]

        # 收集所有时间戳
        all_timestamps = []
        for conv in self.conversations:
            for msg in conv["messages"]:
                all_timestamps.append(msg["timestamp"])

        # 解析时间范围
        date_range = {"start": None, "end": None}
        if all_timestamps:
            all_timestamps.sort()
            date_range["start"] = all_timestamps[0]
            date_range["end"] = all_timestamps[-1]

        # 统计消息类型
        message_types = {}
        for conv in self.conversations:
            for msg in conv["messages"]:
                role = msg["role"]
                message_types[role] = message_types.get(role, 0) + 1

        # 统计源文件
        source_files = list(set(
            conv.get("metadata", {}).get("log_file", "")
            for conv in self.conversations
        ))

        # 统计格式类型
        format_types = {}
        for conv in self.conversations:
            format_type = conv.get("metadata", {}).get("file_format", "unknown")
            format_types[format_type] = format_types.get(format_type, 0) + 1

        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "avg_conversation_length": total_messages / len(self.conversations) if self.conversations else 0,
            "min_conversation_length": min(message_lengths) if message_lengths else 0,
            "max_conversation_length": max(message_lengths) if message_lengths else 0,
            "median_conversation_length": sorted(message_lengths)[len(message_lengths) // 2] if message_lengths else 0,
            "date_range": date_range,
            "message_types": message_types,
            "source_files": source_files,
            "format_types": format_types,
            "avg_messages_per_day": self._calculate_avg_messages_per_day(all_timestamps),
            "parsing_success_rate": self._calculate_parsing_success_rate()
        }

    def _calculate_avg_messages_per_day(self, timestamps: list[str]) -> float:
        """计算每日平均消息数"""
        if not timestamps:
            return 0.0

        # 解析时间戳
        dates = []
        for ts in timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                dates.append(dt.date())
            except (ValueError, TypeError):
                continue

        if not dates:
            return 0.0

        # 计算日期范围
        start_date = min(dates)
        end_date = max(dates)

        # 计算天数
        days = (end_date - start_date).days + 1

        return len(dates) / days if days > 0 else len(dates)

    def _calculate_parsing_success_rate(self) -> float:
        """计算解析成功率"""
        if not self.conversations:
            return 0.0

        # 计算有效对话的比例（有消息的对话）
        valid_conversations = sum(1 for conv in self.conversations if conv["messages"])
        return valid_conversations / len(self.conversations)
