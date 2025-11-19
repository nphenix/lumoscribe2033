"""
Arq 序列化配置

提供 msgpack 序列化支持，优化任务队列性能
"""

import json
import pickle
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Union

import msgpack
import msgpack_numpy as m

from src.framework.shared.logging import get_logger

logger = get_logger(__name__)


class MsgpackSerializer:
    """Msgpack 序列化器"""

    @staticmethod
    def serialize(data: Any) -> bytes:
        """
        序列化数据为 msgpack 格式

        Args:
            data: 要序列化的数据

        Returns:
            bytes: 序列化后的二进制数据
        """
        try:
            # 启用 numpy 支持
            m.patch()

            # 自定义编码器
            def default_encoder(obj: Any) -> dict[str, Any]:
                if isinstance(obj, datetime):
                    return {"__datetime__": obj.isoformat()}
                elif isinstance(obj, date):
                    return {"__date__": obj.isoformat()}
                elif isinstance(obj, Decimal):
                    return {"__decimal__": str(obj)}
                elif isinstance(obj, set):
                    return {"__set__": list(obj)}
                elif hasattr(obj, '__dict__'):
                    return {"__object__": obj.__dict__, "__class__": obj.__class__.__name__}
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            return msgpack.packb(data, default=default_encoder, use_bin_type=True)

        except Exception as e:
            logger.error(f"Msgpack 序列化失败: {e}")
            # 回退到 pickle
            return pickle.dumps(data)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        从 msgpack 格式反序列化数据

        Args:
            data: msgpack 二进制数据

        Returns:
            Any: 反序列化后的数据
        """
        try:
            # 启用 numpy 支持
            m.patch()

            # 自定义解码器
            def object_hook(obj: dict[str, Any]) -> Any:
                if "__datetime__" in obj:
                    return datetime.fromisoformat(obj["__datetime__"])
                elif "__date__" in obj:
                    return date.fromisoformat(obj["__date__"])
                elif "__decimal__" in obj:
                    return Decimal(obj["__decimal__"])
                elif "__set__" in obj:
                    return set(obj["__set__"])
                elif "__object__" in obj:
                    # 重建对象
                    class_name = obj.get("__class__")
                    if class_name:
                        # 尝试从已知模块导入类
                        try:
                            if class_name == "Document":
                                from src.domain.knowledge.models import Document
                                cls = Document
                            else:
                                cls = object
                        except ImportError:
                            cls = object
                        instance = cls.__new__(cls)
                        instance.__dict__.update(obj["__object__"])
                        return instance
                return obj

            return msgpack.unpackb(data, object_hook=object_hook, raw=False)

        except Exception as e:
            logger.error(f"Msgpack 反序列化失败: {e}")
            # 回退到 pickle
            return pickle.loads(data)


class JsonSerializer:
    """JSON 序列化器（备用）"""

    @staticmethod
    def serialize(data: Any) -> bytes:
        """序列化为 JSON 格式"""
        try:
            return json.dumps(data, default=str).encode('utf-8')
        except Exception as e:
            logger.error(f"JSON 序列化失败: {e}")
            return pickle.dumps(data)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """从 JSON 格式反序列化"""
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"JSON 反序列化失败: {e}")
            return pickle.loads(data)


class FallbackSerializer:
    """回退序列化器（使用 pickle）"""

    @staticmethod
    def serialize(data: Any) -> bytes:
        """使用 pickle 序列化"""
        return pickle.dumps(data)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """使用 pickle 反序列化"""
        return pickle.loads(data)


# 序列化器映射
SERIALIZERS = {
    'msgpack': MsgpackSerializer,
    'json': JsonSerializer,
    'pickle': FallbackSerializer
}


def get_serializer(serializer_type: str = 'msgpack') -> type[MsgpackSerializer] | type[JsonSerializer] | type[FallbackSerializer]:
    """
    获取指定类型的序列化器

    Args:
        serializer_type: 序列化器类型

    Returns:
        序列化器类
    """
    return SERIALIZERS.get(serializer_type, MsgpackSerializer)


class SerializationManager:
    """序列化管理器"""

    def __init__(self, default_serializer: str = 'msgpack'):
        self.default_serializer = get_serializer(default_serializer)
        self.fallback_serializer = get_serializer('pickle')

    def serialize(self, data: Any, serializer_type: str = None) -> bytes:
        """
        序列化数据

        Args:
            data: 要序列化的数据
            serializer_type: 序列化器类型，如果为 None 则使用默认的

        Returns:
            bytes: 序列化后的数据
        """
        serializer = get_serializer(serializer_type) if serializer_type else self.default_serializer

        try:
            return serializer.serialize(data)
        except Exception as e:
            logger.warning(f"主要序列化器失败，使用回退序列化器: {e}")
            return self.fallback_serializer.serialize(data)

    def deserialize(self, data: bytes, serializer_type: str = None) -> Any:
        """
        反序列化数据

        Args:
            data: 要反序列化的数据
            serializer_type: 序列化器类型，如果为 None 则使用默认的

        Returns:
            Any: 反序列化后的数据
        """
        serializer = get_serializer(serializer_type) if serializer_type else self.default_serializer

        try:
            return serializer.deserialize(data)
        except Exception as e:
            logger.warning(f"主要反序列化器失败，使用回退反序列化器: {e}")
            return self.fallback_serializer.deserialize(data)
