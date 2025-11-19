<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- classification: developer -->
# LangChain 1.0 RouterChain + RunnableSequence 多模型路由执行器

## 概述

`LangChainRunner` 是基于 LangChain 1.0 的智能多模型路由执行器，实现了 RouterChain 和 RunnableSequence 的最佳实践，支持动态模型选择、链式执行和性能监控。

## 核心特性

### 1. 智能路由决策
- 基于任务类型和复杂度自动选择最适合的模型
- 支持结构化输出的路由决策
- 考虑模型性能指标和成本因素

### 2. 多模型支持
- 支持 OpenAI、Anthropic、Ollama 等多种模型
- 动态模型配置和能力映射
- 灵活的模型扩展机制

### 3. 性能监控
- 实时性能指标收集
- 成本跟踪和成功率监控
- 响应时间统计

### 4. 链式执行
- 支持复杂的执行链配置
- 灵活的系统提示词配置
- 输出解析和格式化

## 架构设计

```
LangChainRunner
├── RouterChain (路由决策)
│   ├── 路由提示词模板
│   ├── 结构化输出解析
│   └── 模型选择逻辑
├── MultiRouteRunnable (多路由执行)
│   ├── 模型执行链
│   ├── 输入准备
│   └── 响应提取
├── 性能监控系统
│   ├── 指标收集
│   ├── 成本计算
│   └── 成功率跟踪
└── 配置管理
    ├── 模型配置
    ├── 能力映射
    └── 成本设置
```

## 使用方法

### 基本用法

```python
from src.framework.orchestrators.langchain_runner import LangChainRunner
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 创建模型实例
models = {
    "openai-gpt4": ChatOpenAI(model="gpt-4"),
    "openai-gpt35": ChatOpenAI(model="gpt-3.5-turbo"),
    "claude-sonnet": ChatAnthropic(model="claude-3-sonnet-20240229")
}

# 初始化执行器
runner = LangChainRunner(models)

# 执行请求
result = await runner.execute_request("请分析这个复杂的数学问题")
print(f"使用的模型: {result['model']}")
print(f"响应内容: {result['response']}")
print(f"执行时间: {result['execution_time']}秒")
```

### 链式执行

```python
# 配置执行链
chain_config = {
    "system_prompt": "你是一个专业的代码分析专家",
    "parse_output": True
}

# 执行链式请求
result = await runner.execute_chain(
    "分析这段 Python 代码的复杂度",
    chain_config
)
print(result["result"])
```

### 路由决策

```python
# 获取路由决策信息
routing_info = await runner.route_request("写一个复杂的算法")
print(f"推荐模型: {routing_info['model_name']}")
print(f"选择原因: {routing_info['reason']}")
print(f"置信度: {routing_info['confidence']}")
```

### 性能监控

```python
# 获取性能指标
metrics = runner.get_performance_metrics()
for model_name, model_metrics in metrics.items():
    print(f"{model_name}:")
    print(f"  成功率: {model_metrics['success_rate']:.2%}")
    print(f"  平均响应时间: {model_metrics['avg_response_time']:.2f}s")
    print(f"  总成本: ${model_metrics['total_cost']:.2f}")
    print(f"  能力: {model_metrics['capabilities']}")
```

## 模型配置

### 支持的模型类型

| 模型类型 | 用途 | 成本 | 能力 |
|---------|------|------|------|
| openai-gpt4 | 复杂推理、创意写作 | 高 | complex_reasoning, creative_writing, code_analysis |
| openai-gpt35 | 一般对话、文本处理 | 中 | general_conversation, text_processing |
| claude-3-opus | 复杂推理、高质量输出 | 高 | complex_reasoning, creative_writing |
| claude-3-sonnet | 一般任务、代码分析 | 中 | general_conversation, code_analysis |
| ollama-mistral | 代码分析、技术任务 | 低 | code_analysis, technical_tasks |
| ollama-llama2 | 简单查询、快速响应 | 最低 | simple_queries, fast_response |

### 自定义模型配置

```python
# 自定义模型能力映射
def _get_model_capabilities(self, model_name: str) -> List[str]:
    custom_capabilities = {
        "my-custom-model": ["custom_task", "specialized_analysis"]
    }
    return custom_capabilities.get(model_name, ["general_purpose"])
```

## 性能优化

### 1. 路由策略优化
- 使用高质量模型进行路由决策
- 考虑历史性能指标
- 动态调整模型选择权重

### 2. 成本控制
- 实时成本跟踪
- 按需选择性价比最高的模型
- 支持成本阈值设置

### 3. 响应时间优化
- 指数加权平均响应时间计算
- 快速失败和重试机制
- 并发请求处理

## 错误处理

### 路由失败处理
```python
try:
    result = await runner.execute_request("用户请求")
except Exception as e:
    print(f"执行失败: {e}")
    # 可以降级到默认模型或返回错误信息
```

### 模型失败处理
- 自动记录失败指标
- 动态调整模型权重
- 支持故障转移机制

## 最佳实践

### 1. 模型初始化
```python
# 推荐：使用配置化的模型初始化
models = {
    "openai-gpt4": ChatOpenAI(
        model="gpt-4",
        temperature=0.1,
        max_tokens=4000
    ),
    "ollama-mistral": ChatOllama(
        model="mistral",
        temperature=0.7
    )
}
```

### 2. 监控和日志
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO)

# 定期检查性能指标
metrics = runner.get_performance_metrics()
for name, metrics in metrics.items():
    if metrics["success_rate"] < 0.8:
        print(f"警告: {name} 成功率过低")
```

### 3. 测试和验证
```python
# 测试不同类型的请求
test_cases = [
    ("数学问题", "openai-gpt4"),
    ("代码分析", "ollama-mistral"),
    ("一般对话", "openai-gpt35")
]

for request, expected_model in test_cases:
    routing_result = await runner.route_request(request)
    print(f"请求: {request}")
    print(f"推荐模型: {routing_result['model_name']}")
    print(f"置信度: {routing_result['confidence']}")
```

## 扩展开发

### 添加新模型支持
1. 在 `_get_model_capabilities` 中添加模型能力
2. 在 `_get_default_cost` 中设置默认成本
3. 更新路由提示词模板

### 自定义路由逻辑
```python
def _create_routing_chain(self) -> RouterChain:
    # 自定义路由提示词
    custom_prompt = ChatPromptTemplate.from_template("""
    [自定义路由逻辑]
    """)
    
    # 自定义路由模型
    custom_router = self._get_custom_router_model()
    
    # 创建自定义路由链
    return RouterChain(custom_prompt | custom_router)
```

## 版本兼容性

- 基于 LangChain 1.0 构建
- 支持最新的 Runnable 接口
- 兼容结构化输出功能
- 遵循 LangChain 最佳实践

## 故障排除

### 常见问题

1. **导入错误**
   - 确保 LangChain 1.0 已正确安装
   - 检查依赖版本兼容性

2. **路由失败**
   - 检查模型连接性
   - 验证 API 密钥配置
   - 查看日志输出

3. **性能问题**
   - 监控模型响应时间
   - 检查网络连接
   - 调整并发设置
### 调试模式

```python
# 启用调试模式
import logging
logging.getLogger('src.framework.orchestrators.langchain_runner').setLevel(logging.DEBUG)
```

## 配置管理

### 环境变量配置

基于 LangChain 1.0 最佳实践，支持完整的环境变量配置：

```bash
# .env 文件示例
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
XAI_API_KEY=your_xai_api_key_here
OLLAMA_BASE_URL=http://localhost:11434

# 模型参数配置
LANGCHAIN_MODEL_PREFIX=LC_MODEL_
LC_MODEL_OPENAI-GPT4_TEMPERATURE=0.1
LC_MODEL_OPENAI-GPT4_MAX_TOKENS=4000
LC_MODEL_OPENAI-GPT4_ENABLED=true
LC_MODEL_OPENAI-GPT4_PRIORITY=100

# 路由配置
LANGCHAIN_ROUTING_ENABLE_PERFORMANCE=true
LANGCHAIN_ROUTING_ENABLE_COST_OPTIMIZATION=true
LANGCHAIN_ROUTING_CONFIDENCE_THRESHOLD=0.7
LANGCHAIN_ROUTING_MAX_RETRIES=2

# 监控配置
LANGCHAIN_MONITORING_COLLECT_METRICS=true
LANGCHAIN_MONITORING_LOG_ROUTING=true
LANGCHAIN_MONITORING_METRIC_WINDOW_SIZE=100
```

### 配置管理器使用

```python
from src.framework.shared.config_manager import config_manager

# 获取配置
config = config_manager.get_config()
enabled_models = config_manager.get_enabled_models()
default_model = config_manager.get_default_model()

# 验证配置
errors = config_manager.validate_config()
if errors:
    print(f"配置错误: {errors}")

# 创建示例环境变量文件
config_manager.create_sample_env_file('.env.example')
```

### 基于配置管理器的初始化

```python
from src.framework.orchestrators.langchain_runner import LangChainRunner
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# 创建模型实例
model_instances = {
    "openai-gpt4": ChatOpenAI(model="gpt-4", temperature=0.1),
    "openai-gpt35-turbo": ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1),
    "ollama-mistral": ChatOllama(model="mistral", temperature=0.7),
}

# 基于配置管理器创建 LangChainRunner
runner = LangChainRunner.from_config_manager(model_instances)
```

## 参考资料

- [LangChain 1.0 官方文档](https://python.langchain.com/)
- [RouterChain 最佳实践](https://python.langchain.com/docs/concepts/rag/chains/)
- [RunnableSequence 使用指南](https://python.langchain.com/docs/concepts/runnables/)
- [结构化输出文档](https://python.langchain.com/docs/concepts/output_parsers/structured/)
- [LangChain 1.0 配置管理最佳实践](https://python.langchain.com/docs/concepts/configuration/)