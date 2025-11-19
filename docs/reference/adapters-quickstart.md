<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- classification: developer -->
# 适配器模块快速入门

## 概述

本指南将帮助您快速了解和使用 lumoscribe2033 框架的适配器模块。通过几个简单的步骤，您就能掌握核心功能并开始集成各种外部服务。

## 快速开始

### 1. 基本使用

```python
# 导入适配器模块
from src.framework.adapters import (
    CursorConversationAdapter,
    LangChainV1RouterAdapter,
    FileConfigAdapter,
    AdapterFactory
)

# 解析 IDE 对话日志
cursor_adapter = CursorConversationAdapter()
log_content = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there!
"""
result = cursor_adapter.parse_conversation(log_content)
print(f"解析成功: {result.success}")
print(f"消息数量: {len(result.messages)}")

# 配置管理
config_adapter = FileConfigAdapter("./config")
config = await config_adapter.load_config("app.yaml")
config["api"]["port"] = 8080
await config_adapter.save_config("app.yaml", config)

# LLM 路由
router_config = RouterConfig(
    models={
        "openai-gpt4": {
            "provider": "openai",
            "model_id": "gpt-4",
            "api_key_env": "OPENAI_API_KEY"
        }
    },
    routing_strategy=RoutingStrategy.PERFORMANCE
)
router = LangChainV1RouterAdapter(router_config)
decision = await router.route_request({"prompt": "Hello"})
print(f"选择模型: {decision.selected_model}")
```

### 2. 适配器工厂使用

```python
# 初始化适配器工厂
factory = await init_adapter_factory()

# 创建适配器
adapter = factory.create_adapter("cursor_conversation")
await factory.initialize_adapter("cursor_conversation")
await factory.start_adapter("cursor_conversation")

# 健康检查
health = await factory.health_check("cursor_conversation")
print(f"适配器状态: {health['status']}")

# 列出所有适配器
all_adapters = factory.list_adapters()
print(f"可用适配器: {all_adapters}")
```

## 核心功能详解

### 对话日志解析

#### Cursor 日志解析

```python
# 支持多种格式
structured_log = """
[2024-01-01 10:00:00] User: Hello
[2024-01-01 10:00:05] Assistant: Hi there!
[2024-01-01 10:00:10] User: How are you?
"""

json_log = '''
[
    {
        "timestamp": "2024-01-01T10:00:00",
        "role": "user",
        "content": "Hello",
        "type": "message"
    }
]
'''

# 解析
adapter = CursorConversationAdapter()
result = adapter.parse_conversation(structured_log)

if result.success:
    for msg in result.messages:
        print(f"{msg.role}: {msg.content}")
        print(f"时间: {msg.timestamp}")
else:
    print(f"解析失败: {result.error}")
```

#### RooCode 日志解析

```python
# 中文日志格式
chinese_log = """
[2024-01-01 10:00:00] 用户: 你好
[2024-01-01 10:00:05] 助手: 你好！欢迎使用 RooCode
"""

adapter = RooCodeConversationAdapter()
result = adapter.parse_conversation(chinese_log)

# 导出为不同格式
json_export = adapter.export_conversation(result, ConversationExportFormat.JSON)
csv_export = adapter.export_conversation(result, ConversationExportFormat.CSV)
text_export = adapter.export_conversation(result, ConversationExportFormat.TEXT)
```

### LLM 路由器使用

#### 基础路由配置

```python
from src.framework.adapters import RouterConfig, RoutingStrategy

config = RouterConfig(
    models={
        "openai-gpt4": {
            "provider": "openai",
            "model_id": "gpt-4",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "capabilities": ["chat", "completion"],
            "cost_per_token": 0.00003,
            "enabled": True
        },
        "anthropic-claude": {
            "provider": "anthropic",
            "model_id": "claude-3-sonnet",
            "api_key_env": "ANTHROPIC_API_KEY",
            "capabilities": ["chat"],
            "cost_per_token": 0.00002,
            "enabled": True
        },
        "ollama-llama": {
            "provider": "ollama",
            "model_id": "llama2",
            "base_url": "http://localhost:11434",
            "capabilities": ["chat", "completion"],
            "cost_per_token": 0.0,
            "enabled": True
        }
    },
    routing_strategy=RoutingStrategy.PERFORMANCE,
    fallback_model="ollama-llama"
)

router = LangChainV1RouterAdapter(config)
await router.initialize()
```

#### 路由请求处理

```python
# 基础请求路由
request = {
    "prompt": "Hello, how are you?",
    "context": {"user_id": "test_user"}
}

decision = await router.route_request(request)
print(f"选择模型: {decision.selected_model}")
print(f"理由: {decision.reasoning}")
print(f"置信度: {decision.confidence}")

# Agent 模式
agent_request = {
    "prompt": "Hello",
    "agent_mode": True,
    "tools": ["calculator", "search"]
}

response = await router.process_with_agent(agent_request)
print(f"Agent 响应: {response}")
```

#### 自适应路由

```python
from src.framework.adapters import AdaptiveLLMRouterAdapter

adaptive_router = AdaptiveLLMRouterAdapter(config)

# 添加中间件
class LoggingMiddleware:
    async def process_request(self, request):
        print(f"处理请求: {request}")
        return request
    
    async def process_response(self, response):
        print(f"处理响应: {response}")
        return response

adaptive_router.middleware_chain.add_middleware(LoggingMiddleware())

# 自动学习和优化
await adaptive_router.perform_adaptive_learning()
decision = adaptive_router.make_adaptive_routing_decision(request)
```

### 配置管理

#### 文件配置

```python
import yaml

# YAML 配置文件
config_data = {
    "llm": {
        "models": {
            "openai-gpt4": {
                "provider": "openai",
                "model_id": "gpt-4",
                "api_key_env": "OPENAI_API_KEY",
                "capabilities": ["chat", "completion"]
            }
        }
    },
    "database": {
        "url": "sqlite:///app.db",
        "echo": False
    },
    "api": {
        "port": 8080,
        "cors_origins": ["http://localhost:3000"]
    }
}

# 保存配置
adapter = FileConfigAdapter("./config")
await adapter.save_config("app.yaml", config_data)

# 加载配置
config = await adapter.load_config("app.yaml")
print(f"API 端口: {config['api']['port']}")

# 验证配置
errors = await adapter.validate_config(config)
if errors:
    print(f"配置错误: {errors}")
else:
    print("配置验证通过")
```

#### 环境变量配置

```python
import os

# 设置环境变量
os.environ["LUMOSCRIBE_DATABASE_URL"] = "sqlite:///test.db"
os.environ["LUMOSCRIBE_API_PORT"] = "8080"
os.environ["LUMOSCRIBE_LOG_LEVEL"] = "INFO"

# 加载配置
adapter = EnvironmentConfigAdapter()
config = await adapter.load_config("")

print(f"数据库 URL: {config['database']['url']}")
print(f"API 端口: {config['api']['port']}")

# 生成 .env 文件
env_config = {
    "database": {"url": "sqlite:///production.db"},
    "api": {"port": 80, "debug": False}
}
await adapter.save_config(".env", env_config)
```

#### FastAPI 集成

```python
from fastapi import FastAPI
from src.framework.adapters import FastAPIConfigAdapter

app = FastAPI()

# 集成配置适配器
config_adapter = FastAPIConfigAdapter(app)

# 现在可以通过 API 管理配置
# GET /api/config/app.yaml
# PUT /api/config/app.yaml
# POST /api/config/validate
```

## 实用示例

### 1. IDE 日志分析工具

```python
async def analyze_ide_logs(log_files):
    """分析多个 IDE 日志文件"""
    results = []
    
    for log_file in log_files:
        # 自动检测格式并解析
        if "cursor" in log_file.lower():
            adapter = CursorConversationAdapter()
        else:
            adapter = RooCodeConversationAdapter()
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        result = adapter.parse_conversation(content)
        results.append({
            "file": log_file,
            "success": result.success,
            "message_count": len(result.messages),
            "duration": result.processing_time
        })
    
    return results

# 使用示例
log_files = ["cursor_log.txt", "roocode_log.txt"]
analysis = await analyze_ide_logs(log_files)
for result in analysis:
    print(f"{result['file']}: {result['message_count']} 条消息")
```

### 2. 智能 LLM 路由器

```python
class SmartLLMRouter:
    def __init__(self):
        self.router = AdaptiveLLMRouterAdapter(config)
        await self.router.initialize()
    
    async def route_intelligent(self, prompt, user_context=None):
        """智能路由请求"""
        # 分析请求类型
        request_type = self._analyze_request_type(prompt)
        
        # 根据上下文选择策略
        if user_context and user_context.get("budget_constrained"):
            strategy = RoutingStrategy.COST_OPTIMIZATION
        elif request_type == "creative":
            strategy = RoutingStrategy.PERFORMANCE
        else:
            strategy = RoutingStrategy.ADAPTIVE
        
        # 执行路由
        decision = await self.router.route_request({
            "prompt": prompt,
            "strategy": strategy,
            "context": user_context
        })
        
        return decision
    
    def _analyze_request_type(self, prompt):
        """分析请求类型"""
        if any(word in prompt.lower() for word in ["代码", "编程", "debug"]):
            return "technical"
        elif any(word in prompt.lower() for word in ["创意", "写作", "故事"]):
            return "creative"
        else:
            return "general"
```

### 3. 配置热更新

```python
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigHotReloader:
    def __init__(self, config_adapter):
        self.config_adapter = config_adapter
        self.watchers = {}
    
    def watch_config_file(self, config_path):
        """监视配置文件变化"""
        if config_path in self.watchers:
            return
        
        class ConfigChangeHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
                    print(f"配置文件变更: {event.src_path}")
                    # 触发重新加载逻辑
                    asyncio.create_task(self._reload_config(event.src_path))
        
        handler = ConfigChangeHandler()
        observer = Observer()
        observer.schedule(handler, str(Path(config_path).parent), recursive=False)
        observer.start()
        
        self.watchers[config_path] = observer
    
    async def _reload_config(self, config_path):
        """重新加载配置"""
        try:
            config = await self.config_adapter.load_config(config_path)
            print(f"配置重新加载成功: {config_path}")
            # 触发配置更新事件
            await self._notify_config_update(config)
        except Exception as e:
            print(f"配置重新加载失败: {e}")
```

## 故障排除

### 常见问题解决

1. **解析失败**
   ```python
   # 启用调试模式
   adapter = CursorConversationAdapter(debug=True)
   result = adapter.parse_conversation(log_content)
   if not result.success:
       print(f"调试信息: {result.debug_info}")
   ```

2. **配置验证错误**
   ```python
   errors = await config_adapter.validate_config(config_data)
   for error in errors:
       print(f"配置错误: {error}")
   ```

3. **LLM 路由失败**
   ```python
   try:
       decision = await router.route_request(request)
   except Exception as e:
       print(f"路由失败: {e}")
       # 使用备用模型
       decision = RoutingDecision(
           selected_model="fallback-model",
           reasoning="主路由失败，使用备用模型",
           confidence=0.5
       )
   ```

### 性能优化

```python
# 启用缓存
adapter = FileConfigAdapter("./config", enable_cache=True)

# 批量处理
async def batch_process_logs(log_files):
    tasks = [process_single_log(log_file) for log_file in log_files]
    results = await asyncio.gather(*tasks)
    return results

# 监控性能
import time
start_time = time.time()
result = await adapter.parse_conversation(log_content)
duration = time.time() - start_time
print(f"解析耗时: {duration:.3f}秒")
```

## 下一步

- 阅读 [适配器开发指南](./adapters-guide.md) 了解高级功能
- 查看 [API 参考文档](./api-reference.md) 了解完整接口
- 参与 [最佳实践讨论](./best-practices.md) 分享经验

通过本快速入门指南，您已经掌握了适配器模块的核心功能。现在可以开始在您的项目中使用这些强大的工具了！