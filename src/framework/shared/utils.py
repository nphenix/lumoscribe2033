"""
共享工具函数

提供跨层使用的通用工具函数。
"""

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, TypeVar

import yaml

# 类型变量
T = TypeVar('T')


def generate_id(prefix: str = "id") -> str:
    """生成唯一ID"""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def generate_hash(content: str, algorithm: str = "sha256") -> str:
    """生成内容哈希"""
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(content.encode('utf-8'))
    return hash_obj.hexdigest()


def safe_get(obj: Any, path: str, default: Any | None = None) -> Any:
    """安全获取嵌套对象的值"""
    try:
        for key in path.split('.'):
            if isinstance(obj, dict):
                obj = obj[key]
            else:
                obj = getattr(obj, key)
        return obj
    except (KeyError, AttributeError, TypeError):
        return default


def deep_merge(base: dict[Any, Any], update: dict[Any, Any]) -> dict[Any, Any]:
    """深度合并字典"""
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def validate_json(json_str: str) -> bool:
    """验证JSON字符串"""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def parse_yaml(file_path: str) -> dict[Any, Any] | None:
    """解析YAML文件"""
    try:
        with open(file_path, encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """提取文本中的代码块"""
    pattern = r'```(?:' + (language or r'\w*') + r')?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    # 移除或替换非法字符
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '_', filename)

    # 移除前后空格和点
    sanitized = sanitized.strip(' .')

    # 限制长度
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext

    return sanitized


async def async_retry(
    func: Callable[..., Any],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Any:
    """异步重试装饰器"""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:  # type: ignore[misc]
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(delay * (backoff ** attempt))

    raise last_exception


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """函数执行时间装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    return wrapper


@lru_cache(maxsize=128)
def cached_property(func: Callable[..., Any]) -> Any:
    """缓存属性装饰器"""
    return func


def create_directory(path: str) -> bool:
    """创建目录"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_file_extension(filename: str) -> str:
    """获取文件扩展名"""
    return Path(filename).suffix.lower()


def is_text_file(filepath: str) -> bool:
    """判断是否为文本文件"""
    text_extensions = {
        '.txt', '.md', '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
        '.py', '.js', '.ts', '.css', '.csv', '.log', '.ini', '.cfg',
        '.toml', '.rst', '.tex', '.sql', '.sh', '.bat', '.ps1'
    }
    return get_file_extension(filepath) in text_extensions


def read_file_content(filepath: str, encoding: str = 'utf-8') -> str | None:
    """读取文件内容"""
    try:
        with open(filepath, encoding=encoding) as f:
            return f.read()
    except Exception:
        return None


def write_file_content(filepath: str, content: str, encoding: str = 'utf-8') -> bool:
    """写入文件内容"""
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False


def get_current_timestamp() -> int:
    """获取当前时间戳"""
    return int(time.time())


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化日期时间"""
    return dt.strftime(format_str)


def parse_datetime(
    dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> datetime | None:
    """解析日期时间字符串"""
    try:
        return datetime.strptime(dt_str, format_str)
    except ValueError:
        return None


def is_valid_email(email: str) -> bool:
    """验证邮箱格式"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def extract_urls(text: str) -> list[str]:
    """提取文本中的URL"""
    pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'
    return re.findall(pattern, text)


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """截断字符串"""
    if len(text) <= max_length:
        return text
    return text[:max_length-len(suffix)] + suffix


def flatten_list(nested_list: list[Any]) -> list[Any]:
    """展平嵌套列表"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    """将列表分块"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default


def get_env_var(key: str, default: Any = None, var_type: type = str) -> Any:
    """获取环境变量并转换类型"""
    value = os.getenv(key, default)
    if value is None:
        return default

    try:
        if var_type is bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif var_type is int:
            return int(value)
        elif var_type is float:
            return float(value)
        else:
            return str(value)
    except (ValueError, TypeError):
        return default


class SingletonMeta(type):
    """单例元类"""
    _instances: dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def create_progress_bar(total: int, description: str = "Processing") -> Any | None:
    """创建进度条（需要rich库）"""
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TimeRemainingColumn,
        )

        return Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
    except ImportError:
        return None
