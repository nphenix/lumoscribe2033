<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- generated: python -m src.cli metadata-injector @ 2025-11-16T10:49:00.274Z -->
<!-- classification: developer -->

# 最佳实践指南

## 概述

本文档收集了 lumoscribe2033 项目的核心最佳实践，涵盖 speckit 流程、适配器使用、架构设计和开发规范。所有实践都经过验证，能够显著提升开发效率和代码质量。

## Speckit 流程最佳实践

### 1. 项目启动阶段

#### 宪法制定 (Constitution) 最佳实践

**原则**: 建立清晰的项目治理框架

```markdown
# 好的项目宪法示例

## 代码质量原则
- 维护测试覆盖率 > 80%
- 使用有意义的变量名
- 文档化复杂算法
- 单文件代码行数 < 4000 行

## 架构指导原则
- 优先组合而非继承
- 函数长度 < 50 行
- 分离业务逻辑与 UI
- 模块职责单一

## 测试标准
- TDD 开发模式
- 所有 API 端点集成测试
- 关键用户流程端到端测试
- 性能回归测试

## 用户体验一致性
- 响应时间 < 3 秒
- 错误信息友好且可操作
- 一致的交互模式
- 无障碍设计支持
```

**错误示范**:
```markdown
# ❌ 不好的宪法
- 代码要写好
- 界面要美观
- 性能要快
```

#### 需求规范 (Specification) 最佳实践

**原则**: 具体、可测试、无歧义

```markdown
# 好的需求规范示例

## 用户故事 1 - 一键生成 speckit 全流程 (P1)

**场景**: 运营人员上传自然语言文档

**验收标准**:
1. 给定用户上传 txt/md 文档，当触发管线时，系统应生成完整的 speckit 工件
2. 所有生成文件必须包含章程标签和时间戳
3. 执行失败时应提供可定位的错误信息

**非功能性需求**:
- 支持最大 50MB 文档
- 处理时间 ≤ 10 分钟
- 错误恢复机制

**成功标准**:
- 95% 的输入能成功生成工件
- 平均处理时间 5 分钟
- 用户满意度 ≥ 4.5/5
```

**检查清单**:
- [ ] 所有用户故事都有明确的验收标准
- [ ] 非功能性需求具体可测量
- [ ] 成功标准可量化
- [ ] 没有 [NEEDS CLARIFICATION] 标记

### 2. 实施计划阶段

#### 技术选型最佳实践

**原则**: 最小化复杂性，最大化可维护性

```yaml
# 好的技术选型决策
技术栈选择:
  前端: 无（纯 API 服务）
  后端: FastAPI + Python 3.12
  数据库: SQLite (开发) + PostgreSQL (生产)
  缓存: Redis
  消息队列: Arq
  向量存储: Chroma
  图数据库: NetworkX + SQLite
  测试: pytest + coverage
  部署: Docker + Docker Compose

选型理由:
  - FastAPI: 高性能，自动生成 OpenAPI 文档
  - SQLite: 零配置，适合开发和小型部署
  - Chroma: 轻量级向量存储，易于集成
  - Arq: 基于 asyncio 的轻量级任务队列
```

**架构决策记录 (ADR)**:

```markdown
# ADR: 选择 FastAPI 而非 Django

## 决策背景
需要高性能 API 服务，支持异步处理，易于生成 OpenAPI 文档

## 选项评估
1. **Django REST Framework**: 成熟但重量级，同步为主
2. **FastAPI**: 现代，高性能，原生异步支持，自动生成文档
3. **Flask**: 轻量但需要手动配置较多

## 决策结果
选择 FastAPI，因为：
- ✅ 异步原生支持
- ✅ 自动生成 OpenAPI 文档
- ✅ 类型提示支持
- ✅ 高性能
- ✅ 生态丰富

## 影响
- 正面: 开发效率高，性能好
- 负面: 相对新，社区规模较小
```

#### 数据模型设计最佳实践

**原则**: 清晰的实体关系，支持扩展

```python
# 好的数据模型设计
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime

class SubmissionPackage(SQLModel, table=True):
    """用户提交的文档包"""
    id: Optional[int] = Field(default=None, primary_key=True)
    source_path: str = Field(index=True)
    uploader: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="pending", index=True)
    retry_of: Optional[int] = Field(default=None, foreign_key="submissionpackage.id")
    
    # 关系
    compliance_report: Optional["ComplianceReport"] = Relationship(back_populates="submission")
    document_profiles: List["DocumentProfile"] = Relationship(back_populates="submission")

class ComplianceReport(SQLModel, table=True):
    """合规检查报告"""
    id: Optional[int] = Field(default=None, primary_key=True)
    submission_id: int = Field(foreign_key="submissionpackage.id")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(index=True)  # pass, warn, fail
    
    # JSON 字段存储详细结果
    static_checks: str  # JSON
    doc_findings: str   # JSON
    traceability_gaps: str  # JSON
    
    submission: SubmissionPackage = Relationship(back_populates="compliance_report")
```

### 3. 任务分解最佳实践

#### 任务编写规范

**原则**: 清晰、可执行、可验证

```markdown
# 好的任务示例

## 阶段 1: 基础设施设置

- [ ] T001 初始化项目结构，创建标准目录
- [ ] T002 [P] 配置依赖管理，锁定关键版本
- [ ] T003 [P] 设置代码质量工具 (ruff, mypy, pytest)
- [ ] T004 [US1] 实现基础 API 框架

## 阶段 2: 核心功能开发

- [ ] T010 [P] [US1] 实现 speckit 管线执行器
- [ ] T011 [US1] 添加文档解析和验证
- [ ] T012 [P] [US1] 实现错误处理和重试机制
- [ ] T013 [US2] 添加进度追踪和状态查询
```

**任务编写检查清单**:
- [ ] 使用标准格式: `- [ ] T{数字} [P?] [US{数字}] 描述 @文件路径`
- [ ] 并行任务标记 [P]
- [ ] 关联用户故事 [US{数字}]
- [ ] 包含具体文件路径
- [ ] 任务大小适中 (0.5-2 人天)

## 适配器使用最佳实践

### 1. 适配器设计原则

#### 单一职责原则

```python
# ✅ 好的设计：每个适配器专注一个功能
class CursorLogParserAdapter:
    """专门解析 Cursor IDE 日志格式"""
    def parse_conversation(self, log_content: str) -> ParseResult:
        # 只处理解析逻辑
        pass

class CursorConfigAdapter:
    """专门处理 Cursor IDE 配置"""
    def load_config(self, config_path: str) -> Dict:
        # 只处理配置加载
        pass

# ❌ 不好的设计：多功能适配器
class MultiPurposeCursorAdapter:
    """同时处理多种功能"""
    def parse_log(self, content): pass
    def load_config(self, path): pass
    def validate_file(self, file): pass
    def generate_commands(self): pass
```

#### 接口一致性

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAdapter(ABC):
    """所有适配器的基础接口"""
    
    @abstractmethod
    async def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        pass
    
    @abstractmethod
    async def save_config(self, config_path: str, config_data: Dict[str, Any]) -> bool:
        """保存配置"""
        pass
    
    @abstractmethod
    async def validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        """验证配置，返回错误列表"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
```

### 2. 错误处理最佳实践

#### 统一异常体系

```python
class AdapterError(Exception):
    """适配器基础异常"""
    pass

class ConfigurationError(AdapterError):
    """配置错误"""
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)

class ParseError(AdapterError):
    """解析错误"""
    def __init__(self, message: str, line: Optional[int] = None):
        self.line = line
        super().__init__(message)

class ValidationError(AdapterError):
    """验证错误"""
    def __init__(self, message: str, errors: List[str]):
        self.errors = errors
        super().__init__(message)
```

#### 优雅的错误恢复

```python
class RobustAdapter:
    """支持错误恢复的适配器"""
    
    async def load_config_with_fallback(self, config_path: str) -> Dict[str, Any]:
        """带降级机制的配置加载"""
        try:
            # 尝试严格模式解析
            return await self.parse_strict(config_path)
        except ParseError as e:
            logger.warning(f"严格模式解析失败，使用宽松模式: {e}")
            try:
                # 降级到宽松模式
                return await self.parse_lenient(config_path)
            except Exception as e:
                # 最后使用默认配置
                logger.error(f"配置加载完全失败，使用默认配置: {e}")
                return await self.get_default_config()
    
    async def parse_strict(self, config_path: str) -> Dict[str, Any]:
        """严格模式解析"""
        # 严格验证逻辑
        pass
    
    async def parse_lenient(self, config_path: str) -> Dict[str, Any]:
        """宽松模式解析"""
        # 宽松验证逻辑
        pass
    
    async def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "timeout": 30,
            "retry_count": 3,
            "enable_cache": True
        }
```

### 3. 性能优化最佳实践

#### 缓存策略

```python
import time
from functools import lru_cache
from typing import Optional

class CachedConfigAdapter(BaseAdapter):
    """带缓存的配置适配器"""
    
    def __init__(self, config_dir: str, cache_ttl: int = 300):
        self.config_dir = config_dir
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}
    
    async def load_config(self, config_path: str) -> Dict[str, Any]:
        """带缓存的配置加载"""
        cache_key = config_path
        
        # 检查缓存
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"缓存命中: {config_path}")
                return cached_data
        
        # 重新加载并缓存
        logger.debug(f"加载配置: {config_path}")
        data = await self._load_from_file(config_path)
        
        # 更新缓存
        self._cache[cache_key] = (time.time(), data)
        
        # 定期清理过期缓存
        await self._cleanup_expired_cache()
        
        return data
    
    async def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, (cached_time, _) in self._cache.items()
            if current_time - cached_time > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
```

#### 异步处理

```python
import asyncio
from typing import List

class AsyncLogProcessor:
    """异步日志处理器"""
    
    async def process_logs_concurrently(self, log_files: List[str]) -> List[ParseResult]:
        """并发处理多个日志文件"""
        # 创建任务
        tasks = []
        for log_file in log_files:
            task = self._process_single_log(log_file)
            tasks.append(task)
        
        # 并发执行
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"处理文件 {log_files[i]} 时发生错误: {result}")
                    processed_results.append(ParseResult(
                        success=False,
                        error=str(result),
                        messages=[]
                    ))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            raise
    
    async def _process_single_log(self, log_file: str) -> ParseResult:
        """处理单个日志文件"""
        async with aiofiles.open(log_file, 'r') as f:
            content = await f.read()
        
        adapter = CursorConversationAdapter()
        return adapter.parse_conversation(content)
```

### 4. 配置管理最佳实践

#### 多环境配置

```yaml
# config/development.yaml - 开发环境
api:
  debug: true
  port: 8080
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"

database:
  url: "sqlite:///dev.db"
  echo: true

llm:
  models:
    openai-gpt4:
      provider: "openai"
      model_id: "gpt-4"
      api_key_env: "OPENAI_API_KEY"
      enabled: true

# config/production.yaml - 生产环境
api:
  debug: false
  port: 80
  cors_origins:
    - "https://app.lumoscribe2033.com"

database:
  url: "postgresql://user:pass@prod-db:5432/prod"
  echo: false

llm:
  models:
    openai-gpt4:
      provider: "openai"
      model_id: "gpt-4"
      api_key_env: "OPENAI_API_KEY"
      enabled: true
    ollama-llama:
      provider: "ollama"
      model_id: "llama2"
      base_url: "http://localhost:11434"
      enabled: true
```

#### 配置验证

```python
class ConfigValidator:
    """配置验证器"""
    
    def validate_llm_config(self, config: Dict[str, Any]) -> List[str]:
        """验证 LLM 配置"""
        errors = []
        
        # 检查必需字段
        required_fields = ["provider", "model_id"]
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")
        
        # 验证提供商
        valid_providers = ["openai", "anthropic", "ollama"]
        if config.get("provider") not in valid_providers:
            errors.append(f"不支持的提供商: {config.get('provider')}")
        
        # 验证环境变量
        if config.get("provider") in ["openai", "anthropic"]:
            api_key_env = config.get("api_key_env")
            if not api_key_env:
                errors.append("缺少 API 密钥环境变量配置")
            elif api_key_env not in os.environ:
                errors.append(f"环境变量未设置: {api_key_env}")
        
        # 验证模型启用状态
        if not config.get("enabled", False):
            logger.warning(f"模型 {config.get('model_id')} 已禁用")
        
        return errors
    
    def validate_api_config(self, config: Dict[str, Any]) -> List[str]:
        """验证 API 配置"""
        errors = []
        
        # 端口范围验证
        port = config.get("port", 8080)
        if not (1 <= port <= 65535):
            errors.append(f"端口号超出范围: {port}")
        
        # CORS 配置验证
        cors_origins = config.get("cors_origins", [])
        if not isinstance(cors_origins, list):
            errors.append("CORS origins 必须是数组")
        
        return errors
```

## 开发工作流最佳实践

### 1. Git 工作流

```bash
# 好的分支命名
feature/speckit-pipeline-optimization
bugfix/fix-config-validation-error
hotfix/critical-security-patch
docs/update-api-documentation

# 提交信息规范
feat: add speckit pipeline retry mechanism
fix: resolve configuration validation error
docs: update API documentation with examples
test: add unit tests for pipeline executor
refactor: simplify adapter factory implementation
```

### 2. 代码审查检查清单

```markdown
# 代码审查检查清单

## 功能正确性
- [ ] 功能按需求实现
- [ ] 边界条件处理正确
- [ ] 错误处理完善
- [ ] 测试覆盖率足够

## 代码质量
- [ ] 遵循项目编码规范
- [ ] 函数长度合理 (< 50 行)
- [ ] 变量命名有意义
- [ ] 注释清晰必要

## 性能考虑
- [ ] 避免不必要的数据库查询
- [ ] 适当的缓存策略
- [ ] 异步处理充分利用
- [ ] 内存使用合理

## 安全性
- [ ] 输入验证充分
- [ ] 敏感信息不泄露
- [ ] 权限控制正确
- [ ] SQL 注入防护

## 可维护性
- [ ] 代码结构清晰
- [ ] 依赖关系合理
- [ ] 配置外部化
- [ ] 日志记录充分
```

### 3. 测试策略

```python
# 好的测试实践
import pytest
from unittest.mock import Mock, patch

class TestPipelineExecutor:
    """Pipeline 执行器测试"""
    
    @pytest.fixture
    def mock_adapter(self):
        """模拟适配器"""
        adapter = Mock()
        adapter.parse_document.return_value = {"content": "test"}
        adapter.execute_command.return_value = {"success": True}
        return adapter
    
    @pytest.mark.asyncio
    async def test_successful_pipeline_execution(self, mock_adapter):
        """测试成功执行流程"""
        executor = PipelineExecutor(adapter=mock_adapter)
        
        result = await executor.execute_pipeline("test.md")
        
        assert result.success is True
        assert mock_adapter.parse_document.called
        assert mock_adapter.execute_command.called
    
    @pytest.mark.asyncio
    async def test_pipeline_retry_mechanism(self, mock_adapter):
        """测试重试机制"""
        # 模拟第一次失败，第二次成功
        mock_adapter.execute_command.side_effect = [
            Exception("Network error"),
            {"success": True}
        ]
        
        executor = PipelineExecutor(adapter=mock_adapter, max_retries=3)
        
        result = await executor.execute_pipeline("test.md")
        
        assert result.success is True
        assert mock_adapter.execute_command.call_count == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_timeout_handling(self, mock_adapter):
        """测试超时处理"""
        # 模拟超时
        async def slow_command(*args, **kwargs):
            await asyncio.sleep(10)
            return {"success": True}
        
        mock_adapter.execute_command.side_effect = slow_command
        
        executor = PipelineExecutor(adapter=mock_adapter, timeout=5)
        
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute_pipeline("test.md")
```

## 部署和运维最佳实践

### 1. 环境配置

```bash
# 开发环境
conda create -n lumoscribe2033 python=3.12
conda activate lumoscribe2033
pip install -e .

# 环境变量配置
export LUMOSCRIBE_ENV=development
export LUMOSCRIBE_LOG_LEVEL=DEBUG
export OPENAI_API_KEY=your-api-key
export DATABASE_URL=sqlite:///dev.db

# 启动服务
uvicorn src.api.main:app --reload --port 8080
arq workers.settings.WorkerSettings
```

### 2. 监控和告警

```python
# 健康检查端点
@app.get("/health")
async def health_check():
    """系统健康检查"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "database": await check_database_health(),
            "redis": await check_redis_health(),
            "llm_api": await check_llm_api_health(),
            "storage": await check_storage_health()
        }
    }
    
    # 检查是否有服务不健康
    unhealthy_services = [
        name for name, status in health_status["services"].items()
        if status["status"] != "healthy"
    ]
    
    if unhealthy_services:
        health_status["status"] = "degraded"
        logger.warning(f"不健康服务: {unhealthy_services}")
    
    return health_status

async def check_database_health():
    """检查数据库连接"""
    try:
        # 执行简单查询
        result = await database.fetch_one("SELECT 1")
        return {"status": "healthy", "response_time": 0.01}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_llm_api_health():
    """检查 LLM API 连接"""
    try:
        # 测试 API 调用
        response = await llm_client.chat_completion([
            {"role": "user", "content": "健康检查"}
        ], max_tokens=1)
        
        return {"status": "healthy", "response_time": response.time}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 总结

这些最佳实践是 lumoscribe2033 项目成功的关键要素。遵循这些指导原则能够：

1. **提高开发效率**: 标准化的流程和工具
2. **保证代码质量**: 严格的测试和审查机制
3. **降低维护成本**: 清晰的架构和文档
4. **增强系统可靠性**: 完善的错误处理和监控

定期回顾和更新这些最佳实践，确保它们始终符合项目的最新需求和技术发展趋势。