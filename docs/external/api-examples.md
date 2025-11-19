<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- generated: python -m src.cli metadata-injector @ 2025-11-16T10:50:32.977Z -->
<!-- classification: external -->

# API 使用示例与交互式文档

## 概述

本文档提供了 lumoscribe2033 API 的详细使用示例，包括代码片段、测试用例和交互式演示。所有示例都可以直接复制使用。

## 快速开始

### 环境准备

```bash
# 1. 启动服务
conda activate lumoscribe2033
uvicorn src.api.main:app --reload --port 8080
arq workers.settings.WorkerSettings

# 2. 安装客户端依赖
pip install requests aiohttp httpx

# 3. 验证服务运行
curl http://localhost:8080/health
```

### 基础配置

```python
import requests
import json
import time
from typing import Dict, Any

# API 配置
BASE_URL = "http://localhost:8080/api/v1"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

# 工具函数
def api_request(method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
    """通用 API 请求函数"""
    url = f"{BASE_URL}{endpoint}"
    response = requests.request(method, url, headers=HEADERS, **kwargs)
    response.raise_for_status()
    return response.json()

def poll_job_status(job_id: str, max_wait: int = 600) -> Dict[Any, Any]:
    """轮询作业状态"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = api_request("GET", f"/pipeline/status/{job_id}")
        if status["status"] in ["completed", "failed"]:
            return status
        time.sleep(5)
    raise TimeoutError(f"作业 {job_id} 超时")
```

## 核心 API 示例

### 1. Speckit 管线 API

#### 基础使用

```python
# 示例 1: 提交简单的自然语言文档
def submit_basic_pipeline():
    """提交基础管线作业"""
    request_data = {
        "source_path": "/docs/requirements.md"
    }
    
    response = api_request("POST", "/pipeline/run", json=request_data)
    print(f"作业已提交: {response['job_id']}")
    print(f"消息: {response['message']}")
    
    return response["job_id"]

# 示例 2: 带 IDE 适配的管线
def submit_pipeline_with_ide():
    """提交包含 IDE 适配的管线"""
    request_data = {
        "source_path": "/docs/project_spec.md",
        "ide_targets": ["cursor", "roocode"]
    }
    
    response = api_request("POST", "/pipeline/run", json=request_data)
    print(f"作业 ID: {response['job_id']}")
    print(f"进度 URL: {response['progress_url']}")
    
    return response["job_id"]

# 示例 3: 重试执行
def retry_failed_pipeline(original_job_id: str):
    """重试失败的管线"""
    request_data = {
        "source_path": "/docs/updated_requirements.md",
        "retry_of": original_job_id
    }
    
    response = api_request("POST", "/pipeline/run", json=request_data)
    print(f"重试作业已提交: {response['job_id']}")
    
    return response["job_id"]
```

#### 状态监控

```python
def monitor_pipeline_execution(job_id: str):
    """监控管线执行过程"""
    print(f"开始监控作业: {job_id}")
    
    try:
        # 轮询状态
        final_status = poll_job_status(job_id)
        
        if final_status["status"] == "completed":
            print("✅ 管线执行成功!")
            print(f"执行时长: {final_status['end_time'] - final_status['start_time']}")
            
            # 获取结果
            results = api_request("GET", f"/pipeline/results/{job_id}")
            print("生成的工件:")
            for artifact, path in results["speckit_artifacts"].items():
                print(f"  - {artifact}: {path}")
                
        else:
            print("❌ 管线执行失败")
            print(f"错误信息: {final_status.get('error', {}).get('message', '未知错误')}")
            
    except TimeoutError as e:
        print(f"⏰ {e}")

def get_realtime_progress(job_id: str):
    """获取实时进度"""
    while True:
        status = api_request("GET", f"/pipeline/status/{job_id}")
        
        print(f"状态: {status['status']}")
        print(f"进度: {status['progress']}%")
        print(f"当前步骤: {status['current_step']}")
        print(f"消息: {status['message']}")
        
        if status["status"] in ["completed", "failed"]:
            break
            
        time.sleep(2)
```

#### 结果处理

```python
def download_pipeline_results(job_id: str, format_type: str = "json"):
    """下载管线执行结果"""
    endpoint = f"/pipeline/results/{job_id}"
    if format_type != "json":
        endpoint += f"?format={format_type}"
    
    if format_type == "json":
        results = api_request("GET", endpoint)
        print("执行指标:")
        print(f"  - 执行时长: {results['metrics']['duration_seconds']} 秒")
        print(f"  - Speckit 成功率: {results['metrics']['speckit_success_rate']}")
        print(f"  - 合规评分: {results['metrics']['compliance_score']}")
        
        return results
    
    else:
        # 下载文件
        import requests
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        filename = f"pipeline_results_{job_id}.{format_type}"
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"结果已下载到: {filename}")
        return filename
```

### 2. IDE 适配 API

#### 生成适配包

```python
def generate_cursor_package(submission_id: str):
    """为 Cursor IDE 生成适配包"""
    request_data = {
        "ide_type": "cursor",
        "submission_id": submission_id,
        "options": {
            "include_commands": True,
            "include_agents": True,
            "include_prompts": True
        }
    }
    
    response = api_request("POST", "/ide-packages/generate", json=request_data)
    
    print("Cursor 适配包已生成:")
    print(f"  - 包路径: {response['package_path']}")
    print(f"  - 验证状态: {response['validation_report']['status']}")
    
    if response["validation_report"]["errors"]:
        print("验证错误:")
        for error in response["validation_report"]["errors"]:
            print(f"  - {error}")
    
    return response

def generate_roocode_package(submission_id: str):
    """为 RooCode IDE 生成适配包"""
    request_data = {
        "ide_type": "roocode",
        "submission_id": submission_id,
        "options": {
            "include_commands": True,
            "include_agents": False,  # RooCode 不需要 agents 文件
            "include_prompts": True
        }
    }
    
    response = api_request("POST", "/ide-packages/generate", json=request_data)
    
    print("RooCode 适配包已生成:")
    print(f"  - 包路径: {response['package_path']}")
    
    return response
```

#### 验证适配包

```python
def validate_ide_package(package_path: str, ide_type: str):
    """验证 IDE 适配包"""
    validation_data = {
        "package_path": package_path,
        "ide_type": ide_type
    }
    
    response = api_request("POST", "/ide-packages/validate", json=validation_data)
    
    print(f"{ide_type} 适配包验证结果:")
    print(f"  - 状态: {response['status']}")
    print(f"  - 通过项: {response['passed']}/{response['total']}")
    
    if response["failed_items"]:
        print("失败项目:")
        for item in response["failed_items"]:
            print(f"  - {item['file']}: {item['error']}")
    
    return response
```

### 3. 文档评估 API

#### 基础评估

```python
def evaluate_single_document(file_path: str, submission_id: str):
    """评估单个文档"""
    request_data = {
        "glob": file_path,
        "submission_id": submission_id,
        "include_token_analysis": True,
        "max_files": 1
    }
    
    response = api_request("POST", "/documents/evaluate", json=request_data)
    
    print(f"文档评估完成:")
    print(f"  - 分析文档数: {response['profiles_created']}")
    
    if response["findings"]:
        print("发现的问题:")
        for finding in response["findings"]:
            print(f"  - {finding['type']}: {finding['message']}")
    
    return response

def batch_evaluate_documents(glob_pattern: str):
    """批量评估文档"""
    request_data = {
        "glob": glob_pattern,
        "include_token_analysis": True,
        "max_files": 20
    }
    
    response = api_request("POST", "/documents/evaluate", json=request_data)
    
    print(f"批量评估完成:")
    print(f"  - 分析文档数: {response['profiles_created']}")
    print(f"  - 发现问题数: {len(response['findings'])}")
    
    # 分析问题类型分布
    issue_types = {}
    for finding in response["findings"]:
        issue_type = finding["type"]
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    print("问题类型分布:")
    for issue_type, count in issue_types.items():
        print(f"  - {issue_type}: {count} 个")
    
    return response

def get_document_recommendations(file_path: str):
    """获取文档改进建议"""
    # 先评估文档
    eval_response = evaluate_single_document(file_path)
    
    # 获取详细建议
    recommendations = api_request("GET", f"/documents/recommendations/{file_path}")
    
    print(f"改进建议:")
    for rec in recommendations["recommendations"]:
        print(f"  - 优先级 {rec['priority']}: {rec['description']}")
        print(f"    实施难度: {rec['difficulty']}")
        print(f"    预估收益: {rec['benefit']}")
    
    return recommendations
```

### 4. 对话管理 API

#### 导入对话

```python
def import_cursor_conversations(logs_directory: str):
    """导入 Cursor 对话记录"""
    request_data = {
        "source_type": "cursor",
        "path": logs_directory,
        "auto_detect": True,
        "recursive": True
    }
    
    response = api_request("POST", "/conversations/import", json=request_data)
    
    print(f"Cursor 对话导入完成:")
    print(f"  - 导入对话数: {response['conversations_ingested']}")
    print(f"  - 存储路径: {response['storage_path']}")
    
    return response

def import_roocode_conversations(logs_directory: str):
    """导入 RooCode 对话记录"""
    request_data = {
        "source_type": "roocode",
        "path": logs_directory,
        "auto_detect": True,
        "recursive": False  # RooCode 日志通常在单个目录
    }
    
    response = api_request("POST", "/conversations/import", json=request_data)
    
    print(f"RooCode 对话导入完成:")
    print(f"  - 导入对话数: {response['conversations_ingested']}")
    
    return response

def import_manual_conversation(file_path: str, submission_refs: list = None):
    """手动导入单个对话文件"""
    request_data = {
        "source_type": "manual",
        "path": file_path,
        "submission_refs": submission_refs or []
    }
    
    response = api_request("POST", "/conversations/import", json=request_data)
    
    print(f"手动导入完成:")
    print(f"  - 文件: {file_path}")
    print(f"  - 关联提交: {len(submission_refs) if submission_refs else 0}")
    
    return response
```

#### 对话检索

```python
def search_conversations(query: str, top_k: int = 5):
    """搜索相关对话"""
    params = {
        "query": query,
        "top_k": top_k
    }
    
    response = api_request("GET", "/conversations/search", params=params)
    
    print(f"搜索结果 (查询: '{query}'):")
    for i, conversation in enumerate(response["results"], 1):
        print(f"  {i}. {conversation['title']}")
        print(f"     相关性: {conversation['similarity']:.3f}")
        print(f"     来源: {conversation['source_type']}")
        print(f"     摘要: {conversation['summary'][:100]}...")
        print()
    
    return response

def get_conversation_details(conversation_id: str):
    """获取对话详细信息"""
    response = api_request("GET", f"/conversations/{conversation_id}")
    
    print(f"对话详情:")
    print(f"  - 标题: {response['title']}")
    print(f"  - 来源: {response['source_type']}")
    print(f"  - 时间: {response['timestamp']}")
    print(f"  - 参与者: {', '.join(response['participants'])}")
    print(f"  - 消息数: {len(response['messages'])}")
    print(f"  - 向量 ID: {response['vector_id']}")
    
    print("\n对话内容:")
    for message in response["messages"][:5]:  # 显示前5条消息
        print(f"  {message['role']}: {message['content'][:100]}...")
    
    return response

def find_related_conversations(conversation_id: str):
    """查找相关对话"""
    response = api_request("GET", f"/conversations/{conversation_id}/related")
    
    print(f"相关对话:")
    for i, related in enumerate(response["related_conversations"], 1):
        print(f"  {i}. {related['title']} (相关性: {related['similarity']:.3f})")
    
    return response
```

### 5. 最佳实践 API

#### 搜索实践

```python
def search_best_practices(query: str, stage: str = None, pillar: str = None):
    """搜索最佳实践"""
    params = {"query": query}
    if stage:
        params["stage"] = stage
    if pillar:
        params["pillar"] = pillar
    
    response = api_request("GET", "/best-practices/search", params=params)
    
    print(f"最佳实践搜索结果 (查询: '{query}'):")
    for i, practice in enumerate(response["items"], 1):
        print(f"  {i}. {practice['title']}")
        print(f"     阶段: {practice['applicable_stage']}")
        print(f"     支柱: {practice['pillar']}")
        print(f"     描述: {practice['description'][:100]}...")
        print()
    
    return response

def get_practice_details(practice_id: str):
    """获取实践详情"""
    response = api_request("GET", f"/best-practices/{practice_id}")
    
    print(f"实践详情:")
    print(f"  - 标题: {response['title']}")
    print(f"  - 阶段: {response['applicable_stage']}")
    print(f"  - 支柱: {response['pillar']}")
    print(f"  - 版本: {response['version']}")
    print(f"  - 来源: {response['source']}")
    
    print(f"\n内容:")
    print(response['content'])
    
    if response.get("linked_conversations"):
        print(f"\n关联对话: {len(response['linked_conversations'])} 个")
    
    return response

def add_best_practice(title: str, stage: str, pillar: str, content: str):
    """添加最佳实践"""
    request_data = {
        "title": title,
        "applicable_stage": stage,
        "pillar": pillar,
        "content": content,
        "source": "user-upload"
    }
    
    response = api_request("POST", "/best-practices", json=request_data)
    
    print(f"最佳实践已添加:")
    print(f"  - ID: {response['id']}")
    print(f"  - 标题: {response['title']}")
    
    return response
```

### 6. 合规检查 API

#### 获取合规报告

```python
def get_compliance_report(submission_id: str):
    """获取合规检查报告"""
    response = api_request("GET", f"/compliance/reports/{submission_id}")
    
    print(f"合规报告 (ID: {submission_id}):")
    print(f"  - 状态: {response['status']}")
    print(f"  - 生成时间: {response['generated_at']}")
    
    # 静态检查结果
    if response.get("static_checks"):
        print(f"\n静态检查结果:")
        tool_stats = {}
        for check in response["static_checks"]:
            tool = check["tool"]
            tool_stats[tool] = tool_stats.get(tool, {"total": 0, "errors": 0, "warnings": 0})
            tool_stats[tool]["total"] += 1
            if check["severity"] == "error":
                tool_stats[tool]["errors"] += 1
            elif check["severity"] == "warning":
                tool_stats[tool]["warnings"] += 1
        
        for tool, stats in tool_stats.items():
            print(f"  - {tool}: {stats['total']} 个检查, {stats['errors']} 个错误, {stats['warnings']} 个警告")
    
    # 文档问题
    if response.get("doc_findings"):
        print(f"\n文档问题 ({len(response['doc_findings'])} 个):")
        for finding in response["doc_findings"][:5]:  # 显示前5个
            print(f"  - {finding['type']}: {finding['message']}")
    
    # 追溯性缺口
    if response.get("traceability_gaps"):
        print(f"\n追溯性缺口 ({len(response['traceability_gaps'])} 个):")
        for gap in response["traceability_gaps"][:3]:  # 显示前3个
            print(f"  - {gap['requirement_id']}: {gap['status']}")
    
    return response

def trigger_compliance_check(submission_id: str, check_types: list = None):
    """触发合规检查"""
    request_data = {
        "submission_id": submission_id,
        "check_types": check_types or ["static", "docs", "traceability"]
    }
    
    response = api_request("POST", "/compliance/check", json=request_data)
    
    print(f"合规检查已触发:")
    print(f"  - 作业 ID: {response['job_id']}")
    print(f"  - 检查类型: {', '.join(response['check_types'])}")
    
    return response
```

## 完整工作流示例

### 端到端示例

```python
def complete_workflow_example():
    """完整的端到端工作流示例"""
    print("=== Lumoscribe2033 完整工作流示例 ===\n")
    
    # 1. 提交文档并运行管线
    print("1. 提交文档并运行 Speckit 管线...")
    job_id = submit_pipeline_with_ide()
    
    # 2. 监控执行过程
    print("\n2. 监控管线执行...")
    monitor_pipeline_execution(job_id)
    
    # 3. 获取执行结果
    print("\n3. 获取执行结果...")
    results = download_pipeline_results(job_id)
    
    # 4. 生成 IDE 适配包
    print("\n4. 生成 IDE 适配包...")
    submission_id = job_id
    cursor_package = generate_cursor_package(submission_id)
    roocode_package = generate_roocode_package(submission_id)
    
    # 5. 评估生成的文档
    print("\n5. 评估生成的文档...")
    spec_files = "specs/**/*.md"
    eval_result = batch_evaluate_documents(spec_files)
    
    # 6. 导入对话记录
    print("\n6. 导入对话记录...")
    import_cursor_conversations("C:/Users/username/.cursor/logs")
    import_roocode_conversations("C:/Users/username/.roocode/logs")
    
    # 7. 搜索最佳实践
    print("\n7. 搜索相关最佳实践...")
    practices = search_best_practices("speckit pipeline optimization", stage="plan")
    
    # 8. 获取合规报告
    print("\n8. 获取合规报告...")
    compliance_report = get_compliance_report(submission_id)
    
    print("\n=== 工作流完成 ===")
    return {
        "pipeline_job": job_id,
        "evaluation_result": eval_result,
        "compliance_report": compliance_report
    }

if __name__ == "__main__":
    # 运行完整示例
    workflow_result = complete_workflow_example()
    
    # 打印总结
    print(f"\n工作流总结:")
    print(f"  - 管线作业: {workflow_result['pipeline_job']}")
    print(f"  - 评估文档数: {workflow_result['evaluation_result']['profiles_created']}")
    print(f"  - 合规状态: {workflow_result['compliance_report']['status']}")
```

## 错误处理示例

```python
def robust_api_call(endpoint: str, data: dict = None, max_retries: int = 3):
    """带重试机制的 API 调用"""
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            if data:
                response = api_request("POST", endpoint, json=data)
            else:
                response = api_request("GET", endpoint)
            
            return response
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            if attempt == max_retries - 1:
                print("所有重试尝试失败")
                raise
            
            # 指数退避
            wait_time = (2 ** attempt) + random.random()
            print(f"等待 {wait_time:.2f} 秒后重试...")
            time.sleep(wait_time)

def handle_api_errors():
    """API 错误处理示例"""
    try:
        # 尝试提交不存在的文件
        response = api_request("POST", "/pipeline/run", json={
            "source_path": "/nonexistent/file.md"
        })
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error_detail = e.response.json()
            print(f"请求错误: {error_detail['detail']}")
            
            # 处理具体字段错误
            if "errors" in error_detail:
                for field_error in error_detail["errors"]:
                    print(f"字段 {field_error['field']}: {field_error['message']}")
        elif e.response.status_code == 500:
            print("服务器内部错误，请稍后重试")
        else:
            print(f"未知错误: {e}")
    
    except requests.exceptions.ConnectionError:
        print("无法连接到服务器，请检查网络连接")
    
    except requests.exceptions.Timeout:
        print("请求超时，请稍后重试")
```

## 性能优化示例

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncAPIClient:
    """异步 API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            base_url=self.base_url,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def submit_multiple_jobs(self, file_paths: list):
        """并发提交多个作业"""
        tasks = []
        for file_path in file_paths:
            task = self.submit_job(file_path)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def submit_job(self, file_path: str):
        """提交单个作业"""
        data = {"source_path": file_path}
        
        async with self.session.post("/pipeline/run", json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def monitor_jobs_concurrently(self, job_ids: list):
        """并发监控多个作业"""
        tasks = []
        for job_id in job_ids:
            task = self.poll_job_status(job_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def poll_job_status(self, job_id: str):
        """轮询作业状态"""
        while True:
            async with self.session.get(f"/pipeline/status/{job_id}") as response:
                response.raise_for_status()
                status = await response.json()
                
                if status["status"] in ["completed", "failed"]:
                    return status
                
                await asyncio.sleep(2)

# 使用示例
async def async_workflow_example():
    """异步工作流示例"""
    file_paths = [
        "/docs/project1.md",
        "/docs/project2.md", 
        "/docs/project3.md"
    ]
    
    async with AsyncAPIClient() as client:
        # 并发提交作业
        print("提交多个作业...")
        submit_results = await client.submit_multiple_jobs(file_paths)
        
        job_ids = [result["job_id"] for result in submit_results]
        print(f"作业 IDs: {job_ids}")
        
        # 并发监控作业
        print("监控作业执行...")
        final_statuses = await client.monitor_jobs_concurrently(job_ids)
        
        for i, status in enumerate(final_statuses):
            print(f"作业 {file_paths[i]}: {status['status']}")

if __name__ == "__main__":
    asyncio.run(async_workflow_example())
```

这些示例展示了 lumoscribe2033 API 的完整使用方法，涵盖了从基础调用到复杂工作流的各种场景。可以根据具体需求选择合适的示例进行参考和修改。