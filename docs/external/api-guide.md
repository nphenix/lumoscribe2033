# API 使用指南

本指南详细介绍 lumoscribe2033 的 API 接口使用方法，帮助开发者快速集成和使用平台功能。

## 📋 目录

- [快速开始](#快速开始)
- [认证方式](#认证方式)
- [API 端点](#api-端点)
- [请求响应格式](#请求响应格式)
- [错误处理](#错误处理)
- [示例代码](#示例代码)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 环境准备

1. **启动服务**
```bash
# 启动 FastAPI 服务
uvicorn src.api.main:app --port 8080 --reload

# 访问 API 文档
# http://localhost:8080/docs - Swagger UI
# http://localhost:8080/redoc - ReDoc
```

2. **测试连接**
```bash
curl http://localhost:8080/health
```

### 基础配置

```python
import requests

# 基础配置
BASE_URL = "http://localhost:8080"
API_KEY = "your-api-key"  # 如果需要认证

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
```

## 🔐 认证方式

### API Key 认证

```python
headers = {
    "Authorization": "Bearer your-api-key"
}
```

### 无需认证（开发环境）

```python
headers = {
    "Content-Type": "application/json"
}
```

## 🌐 API 端点

### 健康检查

#### GET `/health`

检查服务状态

**响应示例：**
```json
{
    "status": "healthy",
    "timestamp": "2025-11-19T20:00:00Z",
    "version": "0.1.0",
    "dependencies": {
        "database": "connected",
        "redis": "connected",
        "llm": "available"
    }
}
```

### Speckit 工具

#### POST `/speckit/constitute`

生成项目章程

**请求体：**
```json
{
    "project_description": "基于 AI 的代码质量分析平台",
    "requirements": [
        "支持多 IDE 适配",
        "AI 驱动的代码审查",
        "自动化测试生成"
    ],
    "constraints": [
        "仅支持 Windows 11",
        "Python 3.12+"
    ]
}
```

**响应示例：**
```json
{
    "success": true,
    "output_files": [
        "specs/001-ai-code-review-platform/constitution.md"
    ],
    "message": "项目章程生成成功"
}
```

#### POST `/speckit/specify`

生成需求规格

**请求体：**
```json
{
    "constitution_file": "specs/001-ai-code-review-platform/constitution.md"
}
```

#### POST `/speckit/plan`

生成项目计划

**请求体：**
```json
{
    "specification_file": "specs/001-ai-code-review-platform/spec.md"
}
```

#### POST `/speckit/tasks`

生成任务清单

**请求体：**
```json
{
    "plan_file": "specs/001-ai-code-review-platform/plan.md"
}
```

### 文档评估

#### POST `/doc-review/evaluate`

评估文档质量

**请求体：**
```json
{
    "document_path": "docs/reference/architecture.md",
    "evaluation_type": "completeness",
    "criteria": ["clarity", "completeness", "accuracy"]
}
```

**响应示例：**
```json
{
    "file": "docs/reference/architecture.md",
    "score": 85,
    "category": "good",
    "feedback": [
        "文档结构清晰",
        "缺少具体的实现细节",
        "建议添加代码示例"
    ],
    "recommendations": [
        "添加详细的实现步骤",
        "补充性能指标说明"
    ]
}
```

### 对话管理

#### POST `/conversations`

创建对话记录

**请求体：**
```json
{
    "session_id": "session_001",
    "user_message": "如何实现用户认证功能？",
    "assistant_response": "可以使用 JWT 进行用户认证...",
    "metadata": {
        "ide": "cursor",
        "project": "lumoscribe2033"
    }
}
```

#### GET `/conversations/{session_id}`

获取对话历史

**响应示例：**
```json
{
    "session_id": "session_001",
    "messages": [
        {
            "role": "user",
            "content": "如何实现用户认证功能？",
            "timestamp": "2025-11-19T19:00:00Z"
        },
        {
            "role": "assistant",
            "content": "可以使用 JWT 进行用户认证...",
            "timestamp": "2025-11-19T19:00:01Z"
        }
    ]
}
```

### 合规检查

#### POST `/compliance/check`

执行合规性检查

**请求体：**
```json
{
    "target_path": "src/api/",
    "check_types": ["code_style", "security", "documentation"],
    "config": {
        "max_line_length": 100,
        "require_docstring": true
    }
}
```

**响应示例：**
```json
{
    "summary": {
        "total_files": 15,
        "passed": 12,
        "failed": 3,
        "success_rate": 80.0
    },
    "details": [
        {
            "file": "src/api/routes/auth.py",
            "issues": [
                {
                    "type": "code_style",
                    "line": 45,
                    "message": "行长度超过限制",
                    "severity": "warning"
                }
            ]
        }
    ]
}
```

### 监控指标

#### GET `/metrics`

获取系统指标

**响应示例：**
```json
{
    "api_calls": {
        "total": 1250,
        "successful": 1200,
        "failed": 50,
        "success_rate": 96.0
    },
    "performance": {
        "avg_response_time": 245.5,
        "p95_response_time": 500.0,
        "p99_response_time": 800.0
    },
    "llm_usage": {
        "total_tokens": 1500000,
        "requests_count": 1200
    }
}
```

#### GET `/metrics/health`

获取健康指标

**响应示例：**
```json
{
    "database": {
        "status": "healthy",
        "connection_pool": {
            "active": 5,
            "idle": 10,
            "max": 20
        }
    },
    "llm": {
        "status": "healthy",
        "providers": {
            "openai": "available",
            "ollama": "available"
        }
    },
    "storage": {
        "chroma": {
            "status": "healthy",
            "collections": 15
        },
        "sqlite": {
            "status": "healthy",
            "tables": 25
        }
    }
}
```

## 📊 请求响应格式

### 成功响应

```json
{
    "success": true,
    "data": {
        // 具体数据
    },
    "message": "操作成功",
    "metadata": {
        "timestamp": "2025-11-19T20:00:00Z",
        "version": "0.1.0"
    }
}
```

### 错误响应

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "请求参数验证失败",
        "details": {
            "field": "project_description",
            "reason": "字段不能为空"
        }
    }
}
```

## ⚠️ 错误处理

### 常见错误码

| 错误码 | HTTP 状态 | 说明 |
|--------|-----------|------|
| `VALIDATION_ERROR` | 400 | 请求参数验证失败 |
| `AUTHENTICATION_FAILED` | 401 | 认证失败 |
| `FORBIDDEN` | 403 | 权限不足 |
| `NOT_FOUND` | 404 | 资源不存在 |
| `RATE_LIMIT_EXCEEDED` | 429 | 请求频率超限 |
| `INTERNAL_ERROR` | 500 | 服务器内部错误 |

### 错误处理示例

```python
import requests

try:
    response = requests.post(f"{BASE_URL}/speckit/constitute", 
                           json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    if result.get("success"):
        print("操作成功:", result.get("data"))
    else:
        print("操作失败:", result.get("error", {}).get("message"))
        
except requests.exceptions.HTTPError as e:
    if response.status_code == 401:
        print("认证失败，请检查 API Key")
    elif response.status_code == 429:
        print("请求频率超限，请稍后重试")
    else:
        print(f"HTTP 错误: {e}")
except requests.exceptions.RequestException as e:
    print(f"网络错误: {e}")
```

## 💻 示例代码

### Python 示例

```python
import requests
import json
from typing import Dict, Any

class LumoscribeClient:
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def health_check(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def generate_constitution(self, project_description: str, 
                            requirements: list, constraints: list) -> Dict[str, Any]:
        """生成项目章程"""
        payload = {
            "project_description": project_description,
            "requirements": requirements,
            "constraints": constraints
        }
        response = requests.post(f"{self.base_url}/speckit/constitute", 
                               json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def evaluate_document(self, document_path: str, 
                         evaluation_type: str = "completeness") -> Dict[str, Any]:
        """评估文档质量"""
        payload = {
            "document_path": document_path,
            "evaluation_type": evaluation_type
        }
        response = requests.post(f"{self.base_url}/doc-review/evaluate", 
                               json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

# 使用示例
client = LumoscribeClient()

# 检查健康状态
health = client.health_check()
print(f"服务状态: {health['status']}")

# 生成项目章程
constitution = client.generate_constitution(
    project_description="AI 代码质量分析平台",
    requirements=["多 IDE 支持", "智能代码审查"],
    constraints=["Windows 11", "Python 3.12+"]
)
print(f"生成结果: {constitution}")
```

### JavaScript 示例

```javascript
class LumoscribeClient {
    constructor(baseURL = "http://localhost:8080", apiKey = null) {
        this.baseURL = baseURL;
        this.headers = { 'Content-Type': 'application/json' };
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseURL}/health`, {
            method: 'GET',
            headers: this.headers
        });
        return await response.json();
    }
    
    async generateConstitution(data) {
        const response = await fetch(`${this.baseURL}/speckit/constitute`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        return await response.json();
    }
    
    async evaluateDocument(documentPath, evaluationType = "completeness") {
        const response = await fetch(`${this.baseURL}/doc-review/evaluate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ document_path: documentPath, evaluation_type: evaluationType })
        });
        return await response.json();
    }
}

// 使用示例
const client = new LumoscribeClient();

async function main() {
    try {
        const health = await client.healthCheck();
        console.log('服务状态:', health.status);
        
        const constitution = await client.generateConstitution({
            project_description: "AI 代码质量分析平台",
            requirements: ["多 IDE 支持", "智能代码审查"],
            constraints: ["Windows 11", "Python 3.12+"]
        });
        console.log('生成结果:', constitution);
        
    } catch (error) {
        console.error('错误:', error);
    }
}

main();
```

## 🏆 最佳实践

### 1. 错误处理

```python
def robust_api_call(client, endpoint, payload, max_retries=3):
    """带重试机制的 API 调用"""
    for attempt in range(max_retries):
        try:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                # 速率限制，等待后重试
                time.sleep(2 ** attempt)
                continue
            raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise
```

### 2. 连接池管理

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    """创建带重试策略的会话"""
    session = requests.Session()
    
    # 重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session
```

### 3. 批量操作

```python
def batch_evaluate_documents(client, document_paths):
    """批量评估文档"""
    results = []
    for path in document_paths:
        try:
            result = client.evaluate_document(path)
            results.append({"path": path, "result": result, "status": "success"})
        except Exception as e:
            results.append({"path": path, "error": str(e), "status": "failed"})
    return results
```

### 4. 性能监控

```python
import time
import logging

def monitor_api_performance(func):
    """API 性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logging.info(f"API 调用成功: {func.__name__}, 耗时: {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logging.error(f"API 调用失败: {func.__name__}, 耗时: {duration:.2f}s, 错误: {e}")
            raise
    return wrapper
```

## 🔗 相关链接

- [Swagger API 文档](http://localhost:8080/docs)
- [ReDoc API 文档](http://localhost:8080/redoc)
- [快速开始指南](specs/001-hybrid-rag-platform/quickstart.md)
- [部署指南](docs/external/deployment.md)
- [项目架构](docs/reference/system-architecture.md)

---

**注意**: 本 API 指南基于技术栈预览版 v0.1.0，具体接口可能在后续版本中调整。建议定期查看最新文档。