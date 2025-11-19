<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- classification: developer -->
# 适配器模块开发指南

## 概述

适配器模块是 lumoscribe2033 框架的核心组件之一，提供统一的接口抽象，支持各种外部服务和工具的集成。本指南详细介绍了适配器模块的设计理念、使用方法和最佳实践。

## 架构设计

### 适配器模式

适配器模块采用经典的适配器模式设计：

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   客户端代码     │───▶│   适配器接口      │───▶│   外部服务       │
│                │    │   (ConfigAdapter) │    │   (具体实现)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 核心组件

1. **适配器接口** - 定义统一的抽象接口
2. **具体适配器** - 实现特定服务的适配逻辑
3. **适配器工厂** - 负责适配器的创建和管理
4. **注册中心** - 管理适配器的注册和发现

## 主要适配器类型

### 1. 对话适配器 (Conversation Adapter)

处理 IDE 对话日志的解析和转换。

#### CursorConversationAdapter

支持 Cursor IDE 日志格式解析：

```python
from src.framework.adapters import CursorConversationAdapter

adapter = CursorConversationAdapter()
result = adapter.parse_conversation(log_content)

if result.success:
    for message in result.messages:
        print(f"{message.role}: {message.content}")
```

**支持的格式：**
- 结构化日志：`[2024-01-01 10:00:00] User: Hello`
- JSON 格式：`[{"timestamp": "...", "role": "user", "content": "..."}]`
- 简单文本：`User: Hello\nAssistant: Hi`

**导出格式：**
```python
from src.framework.adapters import ConversationExportFormat

# JSON 导出
json_export = adapter.export_conversation(result, ConversationExportFormat.JSON)

# CSV 导出
csv_export = adapter.export_conversation(result, ConversationExportFormat.CSV)

# 纯文本导出
text_export = adapter.export_conversation(result, ConversationExportFormat.TEXT)
```

#### RooCodeConversationAdapter

专门处理 RooCode IDE 的中文日志格式：

```python
from src.framework.adapters import RooCodeConversationAdapter

adapter = RooCodeConversationAdapter()
result = adapter.parse_conversation(chinese_log_content)

# 支持中文角色名
# [2024-01-01 10:00:00] 用户: 你好
# [2024-01-01 10:00:05] 助手: 你好！欢迎使用 RooCode
```

### 2. LLM 路由器适配器 (LLM Router Adapter)

基于 LangChain 1.0 实现智能 LLM 路由。

#### LangChainV1RouterAdapter

核心路由器适配器，支持多种路由策略：

```python
from src.framework.adapters import LangChainV1RouterAdapter, RouterConfig, RoutingStrategy

config = RouterConfig(
    models={
        "openai-gpt4": {
            "provider": "openai",
            "model_id": "gpt-4",
            "api_key_env": "OPENAI_API_KEY",
            "capabilities": ["chat", "completion"],
            "cost_per_token": 0.00003,
            "enabled": True
        },
        "anthropic-claude": {
            "provider": "anthropic", 
            "model_id": "claude-3-sonnet",
            "capabilities": ["chat"],
            "cost_per_token": 0.00002,
            "enabled": True
        }
    },
    routing_strategy=RoutingStrategy.PERFORMANCE
)

router = LangChainV1RouterAdapter(config)
decision = await router.route_request({"prompt": "Hello, how are you?"})
print(f"选择模型: {decision.selected_model}")
print(f"理由: {decision.reasoning}")
```

**路由策略：**
- `PERFORMANCE`: 性能优先
- `COST_OPTIMIZATION`: 成本优化
- `LOAD_BALANCE`: 负载均衡
- `ADAPTIVE`: 自适应

#### AdaptiveLLMRouterAdapter

增强版自适应路由器，支持动态学习和优化：

```python
from src.framework.adapters import AdaptiveLLMRouterAdapter

adaptive_router = AdaptiveLLMRouterAdapter(config)

# 自动负载均衡
balanced_model = adaptive_router.load_balancer.select_model()

# 成本优化
optimized_model = adaptive_router.cost_optimizer.select_optimal_model(costs)

# 中间件链处理
adaptive_router.middleware_chain.add_middleware(custom_middleware)
```

### 3. 配置管理适配器 (Config Adapter)

提供多种配置管理方式。

#### FileConfigAdapter

基于文件的配置管理：

```python
from src.framework.adapters import FileConfigAdapter

adapter = FileConfigAdapter("./config")

# YAML 配置
config = await adapter.load_config("app.yaml")

# 保存配置
await adapter.save_config("app.yaml", config_data)

# 验证配置
errors = await adapter.validate_config(config_data)
```

**支持的文件格式：**
- YAML: `app.yaml`, `app.yml`
- JSON: `app.json`

#### EnvironmentConfigAdapter

环境变量配置管理：

```python
from src.framework.adapters import EnvironmentConfigAdapter

adapter = EnvironmentConfigAdapter()

# 从环境变量加载
config = await adapter.load_config("")

# 设置环境变量
await adapter.set_config("api.port", 8080)

# 生成 .env 文件
await adapter.save_config(".env", config_data)
```

**环境变量映射：**
```
LUMOSCRIBE_API_PORT=8080
LUMOSCRIBE_DATABASE_URL=sqlite:///app.db
LUMOSCRIBE_LOG_LEVEL=INFO
```

#### FastAPIConfigAdapter

集成 FastAPI 的配置管理：

```python
from fastapi import FastAPI
from src.framework.adapters import FastAPIConfigAdapter

app = FastAPI()
config_adapter = FastAPIConfigAdapter(app)

# 自动添加配置管理路由
# GET /api/config/{config_file}
# PUT /api/config/{config_file}
# POST /api/config/validate
```

## 适配器工厂和注册机制

### AdapterFactory

统一的适配器创建和管理：

```python
from src.framework.adapters import AdapterFactory, AdapterMetadata, AdapterType

factory = AdapterFactory()

# 注册适配器
metadata = AdapterMetadata(
    name="custom_adapter",
    type=AdapterType.CONFIG,
    version="1.0.0",
    description="自定义配置适配器",
    capabilities=["load", "save", "validate"],
    tags=["custom", "config"]
)

factory.register_adapter("custom_adapter", CustomAdapter, metadata)

# 创建适配器
adapter = factory.create_adapter("custom_adapter", **kwargs)

# 生命周期管理
await factory.initialize_adapter("custom_adapter")
await factory.start_adapter("custom_adapter")

# 健康检查
health = await factory.health_check("custom_adapter")
```

### 自动注册

内置适配器自动注册：

```python
from src.framework.adapters import init_adapter_factory

# 初始化并注册内置适配器
factory = await init_adapter_factory()

# 获取所有适配器
all_adapters = factory.list_adapters()

# 按类型过滤
config_adapters = factory.list_adapters(adapter_type=AdapterType.CONFIG)

# 按标签过滤
test_adapters = factory.list_adapters(tag="test")
```

## 最佳实践

### 1. 适配器设计原则

#### 单一职责
每个适配器应该只负责一种特定的集成任务：

```python
# ✅ 好的设计
class CursorLogParserAdapter:
    """专门解析 Cursor 日志格式"""
    def parse_conversation(self, log_content):
        pass

# ❌ 不好的设计
class MultiPurposeAdapter:
    """同时处理多种格式和功能"""
    def parse_cursor_log(self, content):
        pass
    
    def parse_roocode_log(self, content):
        pass
    
    def validate_config(self, config):
        pass
```

#### 接口一致性
确保所有适配器遵循统一的接口规范：

```python
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    @abstractmethod
    async def load_config(self, config_path: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def save_config(self, config_path: str, config_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        pass
```

### 2. 错误处理

#### 统一错误类型
使用自定义异常类型：

```python
class AdapterError(Exception):
    """适配器基础异常"""
    pass

class ConfigurationError(AdapterError):
    """配置错误"""
    pass

class ParseError(AdapterError):
    """解析错误"""
    pass
```

#### 错误恢复机制
实现优雅的错误恢复：

```python
async def robust_parse(self, content):
    try:
        return await self.parse_strict(content)
    except ParseError:
        # 降级到宽松解析
        return await self.parse_lenient(content)
    except Exception as e:
        # 返回错误结果
        return ParseResult(
            success=False,
            error=ParseError(f"解析失败: {str(e)}")
        )
```

### 3. 性能优化

#### 缓存策略
实现智能缓存：

```python
class CachedConfigAdapter(FileConfigAdapter):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self._cache = {}
        self._cache_ttl = 300  # 5分钟
    
    def load_config(self, config_path):
        cache_key = config_path
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
        
        # 重新加载并缓存
        data = super().load_config(config_path)
        self._cache[cache_key] = (time.time(), data)
        return data
```

#### 异步处理
充分利用异步特性：

```python
class AsyncLogProcessor:
    async def process_logs_concurrently(self, log_files):
        tasks = []
        for log_file in log_files:
            task = self.process_single_log(log_file)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
```

### 4. 配置管理

#### 配置验证
实现严格的配置验证：

```python
class ConfigValidator:
    def validate_llm_config(self, config):
        required_fields = ["provider", "model_id"]
        errors = []
        
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")
        
        if config.get("provider") not in ["openai", "anthropic", "ollama"]:
            errors.append(f"不支持的提供商: {config.get('provider')}")
        
        return errors
```

#### 环境隔离
支持多环境配置：

```yaml
# config/development.yaml
api:
  debug: true
  port: 8080

# config/production.yaml  
api:
  debug: false
  port: 80

# config/test.yaml
api:
  debug: true
  port: 8081
```

### 5. 监控和日志

#### 性能监控
集成性能监控：

```python
class MonitoredAdapter:
    def __init__(self, adapter):
        self.adapter = adapter
        self.metrics = {}
    
    async def load_config(self, config_path):
        start_time = time.time()
        try:
            result = await self.adapter.load_config(config_path)
            self._record_success(config_path, time.time() - start_time)
            return result
        except Exception as e:
            self._record_error(config_path, time.time() - start_time, e)
            raise
    
    def _record_success(self, config_path, duration):
        self.metrics[config_path] = {
            "last_success": time.time(),
            "avg_duration": duration,
            "error_count": 0
        }
```

#### 结构化日志
使用结构化日志记录：

```python
import logging

logger = logging.getLogger(__name__)

class AdapterLogger:
    def log_parse_result(self, adapter_name, success, message_count):
        logger.info("适配器解析完成", extra={
            "adapter": adapter_name,
            "success": success,
            "message_count": message_count,
            "timestamp": "2024-01-01T10:00:00Z"
        })
```

## 扩展开发

### 自定义适配器开发

1. **继承基础接口**：
```python
from src.framework.adapters import ConfigAdapter

class CustomAdapter(ConfigAdapter):
    def __init__(self, custom_param):
        self.custom_param = custom_param
    
    async def load_config(self, config_path):
        # 实现自定义加载逻辑
        pass
    
    async def save_config(self, config_path, config_data):
        # 实现自定义保存逻辑
        pass
```

2. **注册到工厂**：
```python
from src.framework.adapters import AdapterFactory, AdapterMetadata, AdapterType

factory = AdapterFactory()

metadata = AdapterMetadata(
    name="custom_adapter",
    type=AdapterType.CONFIG,
    version="1.0.0",
    description="自定义适配器",
    capabilities=["custom_operation"],
    tags=["custom"]
)

factory.register_adapter("custom_adapter", CustomAdapter, metadata)
```

3. **添加测试**：
```python
def test_custom_adapter():
    adapter = CustomAdapter("test_param")
    # 添加测试用例
    pass
```

### 插件系统

支持动态插件加载：

```python
class PluginManager:
    def load_plugin_from_module(self, module_path):
        spec = importlib.util.spec_from_file_location("plugin", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 注册插件中的适配器
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, ConfigAdapter):
                self.factory.register_adapter(name.lower(), obj, self._infer_metadata(obj))
```

## 故障排除

### 常见问题

1. **解析失败**：
   - 检查日志格式是否正确
   - 验证时间戳格式
   - 确认角色名是否匹配

2. **配置验证错误**：
   - 检查必需字段
   - 验证数据类型
   - 确认环境变量设置

3. **性能问题**：
   - 启用缓存
   - 优化解析算法
   - 监控资源使用

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试模式
adapter = CursorConversationAdapter(debug=True)
result = adapter.parse_conversation(log_content)
print(f"解析详情: {result.debug_info}")

# 性能分析
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# 执行适配器操作
profiler.disable()
profiler.print_stats()
```

## 版本兼容性

### 向后兼容
确保新版本适配器与旧配置兼容：

```python
class BackwardCompatibleAdapter:
    def load_config(self, config_path):
        config = super().load_config(config_path)
        
        # 兼容旧版本配置格式
        if "old_format" in config:
            config = self._migrate_old_format(config)
        
        return config
    
    def _migrate_old_format(self, config):
        # 迁移逻辑
        return migrated_config
```

### 版本管理
使用语义化版本号：

```python
class AdapterMetadata:
    version = "2.1.0"  # 主版本.次版本.修订版本
```

- **主版本**: 不兼容的 API 修改
- **次版本**: 向后兼容的功能性新增
- **修订版本**: 向后兼容的问题修正

## 总结

适配器模块为 lumoscribe2033 框架提供了强大的集成能力，通过统一的接口抽象和灵活的扩展机制，支持各种外部服务和工具的无缝集成。遵循本指南的最佳实践，可以确保适配器的可靠性、性能和可维护性。

通过适配器工厂和注册机制，系统能够动态管理和配置各种适配器，为应用提供高度的灵活性和可扩展性。无论是处理 IDE 日志、管理 LLM 路由，还是配置管理，适配器模块都提供了完整的解决方案。