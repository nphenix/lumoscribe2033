<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- generated: python -m src.cli metadata-injector @ 2025-11-16T10:47:39.623Z -->
<!-- classification: agent -->

# 日志管理与追踪体系

## 概述

本文档定义了 lumoscribe2033 项目的日志管理策略、格式规范和追踪机制，确保所有系统活动可审计、可调试、可优化。

## 日志级别与分类

### 1. 日志级别

```python
class LogLevel:
    DEBUG = "DEBUG"      # 调试信息，详细执行过程
    INFO = "INFO"        # 一般信息，正常执行流程
    WARNING = "WARNING"  # 警告信息，潜在问题
    ERROR = "ERROR"      # 错误信息，功能异常
    CRITICAL = "CRITICAL" # 严重错误，系统故障
```

### 2. 日志分类

#### 系统日志 (System Logs)
- **功能**: 系统启动、配置加载、服务状态
- **级别**: INFO, WARNING, ERROR
- **存储**: `logs/system/`

#### Pipeline 日志 (Pipeline Logs)
- **功能**: Speckit 流程执行、步骤追踪、结果记录
- **级别**: INFO, WARNING, ERROR
- **存储**: `data/persistence/logs/pipeline/`

#### API 日志 (API Logs)
- **功能**: HTTP 请求、响应、认证、限流
- **级别**: INFO, WARNING
- **存储**: `logs/api/`

#### 错误日志 (Error Logs)
- **功能**: 异常堆栈、错误详情、修复建议
- **级别**: ERROR, CRITICAL
- **存储**: `logs/errors/`

#### 审计日志 (Audit Logs)
- **功能**: 用户操作、数据变更、合规检查
- **级别**: INFO, WARNING
- **存储**: `logs/audit/`

## 日志格式规范

### 1. 结构化日志格式

所有日志采用 JSON 格式，确保可解析和可查询：

```json
{
  "timestamp": "2025-11-16T10:47:39.623Z",
  "level": "INFO",
  "logger": "src.domain.pipeline.speckit_executor",
  "message": "Speckit pipeline started",
  "trace_id": "550e8c44-8e34-4b13-8c22-6fad58e2b39a",
  "span_id": "123e4567-e89b-12d3-a456-426614174000",
  "user_id": "user123",
  "session_id": "session456",
  "context": {
    "source_path": "/docs/requirements.md",
    "ide_targets": ["cursor", "roocode"],
    "retry_of": null
  },
  "performance": {
    "memory_usage": "150MB",
    "cpu_usage": "25%"
  },
  "tags": ["pipeline", "speckit", "start"]
}
```

### 2. 日志字段说明

| 字段 | 类型 | 描述 | 示例 |
|------|------|------|------|
| timestamp | string | ISO 8601 时间戳 | "2025-11-16T10:47:39.623Z" |
| level | string | 日志级别 | "INFO" |
| logger | string | 日志记录器名称 | "src.domain.pipeline.speckit_executor" |
| message | string | 日志消息 | "Speckit pipeline started" |
| trace_id | string | 追踪 ID（分布式追踪） | "550e8c44-8e34-4b13-8c22-6fad58e2b39a" |
| span_id | string | 跨度 ID | "123e4567-e89b-12d3-a456-426614174000" |
| user_id | string | 用户标识 | "user123" |
| session_id | string | 会话标识 | "session456" |
| context | object | 上下文信息 | 见上文 |
| performance | object | 性能指标 | 见上文 |
| tags | array | 标签列表 | ["pipeline", "speckit", "start"] |

### 3. 特定场景日志格式

#### Pipeline 执行日志
```json
{
  "timestamp": "2025-11-16T10:47:39.623Z",
  "level": "INFO",
  "logger": "PipelineExecutor",
  "message": "Speckit step completed",
  "trace_id": "550e8c44-8e34-4b13-8c22-6fad58e2b39a",
  "job_id": "550e8c44-8e34-4b13-8c22-6fad58e2b39a",
  "step": "speckit.constitution",
  "status": "completed",
  "duration": 15.5,
  "output_files": [
    "specs/001-hybrid-rag-platform/constitution.md"
  ],
  "error": null
}
```

#### API 请求日志
```json
{
  "timestamp": "2025-11-16T10:47:39.623Z",
  "level": "INFO",
  "logger": "APIGateway",
  "message": "API request processed",
  "trace_id": "550e8c44-8e34-4b13-8c22-6fad58e2b39a",
  "request": {
    "method": "POST",
    "path": "/api/v1/pipeline/run",
    "user_agent": "Mozilla/5.0...",
    "ip": "127.0.0.1",
    "content_length": 150
  },
  "response": {
    "status_code": 202,
    "content_length": 89,
    "duration": 0.15
  },
  "user_id": "user123"
}
```

#### 错误日志
```json
{
  "timestamp": "2025-11-16T10:47:39.623Z",
  "level": "ERROR",
  "logger": "PipelineExecutor",
  "message": "Speckit step failed",
  "trace_id": "550e8c44-8e34-4b13-8c22-6fad58e2b39a",
  "job_id": "550e8c44-8e34-4b13-8c22-6fad58e2b39a",
  "step": "speckit.plan",
  "error": {
    "type": "ValidationError",
    "message": "Invalid plan format",
    "details": {
      "field": "tech_stack",
      "value": null,
      "expected": "string"
    },
    "traceback": "Traceback (most recent call last)..."
  },
  "retryable": true,
  "suggested_fix": "Check the plan template format"
}
```

## 日志收集配置

### 1. Python 日志配置

```python
# src/framework/shared/logging.py
import structlog
import logging.config
from pythonjsonlogger import jsonlogger

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": jsonlogger.JsonFormatter(
                fmt="%(timestamp)s %(level)s %(logger)s %(message)s %(module)s %(funcName)s %(lineno)s"
            ),
            "foreign_pre_chain": [
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
            ],
        },
        "console": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "console",
        },
        "file_system": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/system/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "json",
        },
        "file_pipeline": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "data/persistence/logs/pipeline/pipeline.log",
            "maxBytes": 10485760,
            "backupCount": 10,
            "formatter": "json",
        },
        "file_error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/errors/error.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "formatter": "json",
        },
    },
    "loggers": {
        "src": {
            "level": "INFO",
            "handlers": ["console", "file_system"],
            "propagate": False,
        },
        "src.domain.pipeline": {
            "level": "INFO",
            "handlers": ["file_pipeline"],
            "propagate": False,
        },
        "src.api": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
        "src.domain.compliance": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

# 初始化日志配置
def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

### 2. 日志轮转策略

```python
# 日志轮转配置
LOG_ROTATION_CONFIG = {
    "system": {
        "max_size": "10MB",
        "backup_count": 5,
        "rotation_time": "midnight",
        "compression": "gzip"
    },
    "pipeline": {
        "max_size": "10MB", 
        "backup_count": 10,
        "per_job": True,  # 每个作业单独文件
        "compression": "gzip"
    },
    "errors": {
        "max_size": "10MB",
        "backup_count": 20,
        "compression": "gzip"
    },
    "api": {
        "max_size": "10MB",
        "backup_count": 7,
        "rotation_time": "daily",
        "compression": "gzip"
    }
}
```

## 日志分析与监控

### 1. 关键指标监控

```python
class LogMetricsAnalyzer:
    def analyze_pipeline_logs(self, job_id):
        """分析 Pipeline 日志，提取关键指标"""
        log_entries = self.read_job_logs(job_id)
        
        metrics = {
            "total_duration": self.calculate_total_duration(log_entries),
            "step_durations": self.calculate_step_durations(log_entries),
            "error_count": self.count_errors(log_entries),
            "warning_count": self.count_warnings(log_entries),
            "success_rate": self.calculate_success_rate(log_entries),
            "performance_metrics": self.extract_performance_metrics(log_entries)
        }
        
        return metrics
    
    def extract_error_patterns(self, time_range):
        """提取错误模式"""
        error_logs = self.get_error_logs(time_range)
        
        patterns = {
            "common_errors": self.identify_common_errors(error_logs),
            "error_trends": self.analyze_error_trends(error_logs),
            "peak_error_times": self.find_peak_error_times(error_logs),
            "suggested_fixes": self.generate_fix_suggestions(error_logs)
        }
        
        return patterns
```

### 2. 实时监控仪表板

```python
class LogMonitoringDashboard:
    def get_realtime_metrics(self):
        """获取实时监控指标"""
        return {
            "active_jobs": self.count_active_jobs(),
            "error_rate": self.calculate_error_rate(),
            "response_time_p95": self.calculate_response_time_p95(),
            "throughput": self.calculate_throughput(),
            "system_health": self.assess_system_health()
        }
    
    def generate_alerts(self):
        """生成告警"""
        alerts = []
        
        # 错误率告警
        if self.calculate_error_rate() > 0.05:
            alerts.append({
                "type": "high_error_rate",
                "severity": "warning",
                "message": f"错误率超过阈值: {self.calculate_error_rate():.2%}"
            })
        
        # 响应时间告警
        if self.calculate_response_time_p95() > 5.0:
            alerts.append({
                "type": "slow_response",
                "severity": "warning", 
                "message": f"P95 响应时间过长: {self.calculate_response_time_p95():.2f}s"
            })
        
        return alerts
```

## 日志查询与检索

### 1. 查询语法

```python
class LogQueryEngine:
    def query_logs(self, query):
        """执行日志查询"""
        # 支持的查询语法
        query_syntax = """
        SELECT * FROM logs 
        WHERE level IN ('ERROR', 'CRITICAL')
        AND timestamp >= '2025-11-16T00:00:00Z'
        AND logger LIKE 'src.domain.pipeline.%'
        AND message CONTAINS 'failed'
        ORDER BY timestamp DESC
        LIMIT 100
        """
        
        return self.execute_query(query)
    
    def search_by_trace_id(self, trace_id):
        """按追踪 ID 搜索"""
        return self.query_logs(f"""
        SELECT * FROM logs 
        WHERE trace_id = '{trace_id}'
        ORDER BY timestamp ASC
        """)
    
    def get_job_execution_log(self, job_id):
        """获取作业执行日志"""
        return self.query_logs(f"""
        SELECT * FROM logs 
        WHERE job_id = '{job_id}'
        ORDER BY timestamp ASC
        """)
```

### 2. 常用查询示例

```python
# 查询所有错误日志
error_logs = log_query.query_logs("""
SELECT * FROM logs 
WHERE level = 'ERROR' 
AND timestamp >= '2025-11-16T00:00:00Z'
ORDER BY timestamp DESC
""")

# 查询特定作业的完整执行过程
job_logs = log_query.query_logs(f"""
SELECT * FROM logs 
WHERE job_id = '{job_id}'
AND logger LIKE 'src.domain.pipeline.%'
ORDER BY timestamp ASC
""")

# 查询 API 性能指标
api_metrics = log_query.query_logs("""
SELECT 
    path,
    COUNT(*) as request_count,
    AVG(response_time) as avg_response_time,
    MAX(response_time) as max_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
FROM logs 
WHERE logger = 'APIGateway'
AND timestamp >= '2025-11-16T00:00:00Z'
GROUP BY path
ORDER BY request_count DESC
""")
```

## 日志保留策略

### 1. 保留期限

| 日志类型 | 保留期限 | 存储位置 | 压缩策略 |
|----------|----------|----------|----------|
| 系统日志 | 30 天 | `logs/system/` | gzip |
| Pipeline 日志 | 90 天 | `data/persistence/logs/pipeline/` | gzip |
| API 日志 | 60 天 | `logs/api/` | gzip |
| 错误日志 | 365 天 | `logs/errors/` | gzip |
| 审计日志 | 2555 天 (7年) | `logs/audit/` | gzip |

### 2. 清理脚本

```python
class LogCleanupManager:
    def cleanup_old_logs(self):
        """清理过期日志"""
        cleanup_rules = {
            "logs/system/": {"retention_days": 30, "compress_after": 7},
            "data/persistence/logs/pipeline/": {"retention_days": 90, "compress_after": 14},
            "logs/api/": {"retention_days": 60, "compress_after": 7},
            "logs/errors/": {"retention_days": 365, "compress_after": 30},
            "logs/audit/": {"retention_days": 2555, "compress_after": 30}
        }
        
        for log_dir, rules in cleanup_rules.items():
            self.cleanup_directory(log_dir, rules)
    
    def compress_old_logs(self, directory, days_old):
        """压缩指定天数之前的日志"""
        # 实现日志压缩逻辑
        pass
    
    def delete_expired_logs(self, directory, retention_days):
        """删除过期日志"""
        # 实现日志删除逻辑
        pass
```

## 最佳实践

### 1. 日志记录原则

- **有意义的消息**: 日志消息应该清晰描述发生了什么
- **适当的级别**: 根据事件的重要性选择合适的日志级别
- **结构化数据**: 使用结构化格式便于后续分析
- **避免敏感信息**: 不要在日志中记录密码、密钥等敏感信息
- **性能考虑**: 避免在高频路径中记录大量调试信息

### 2. 追踪上下文

```python
# 在请求开始时创建追踪上下文
def create_trace_context():
    trace_id = str(uuid.uuid4())
    span_id = str(uuid.uuid4())
    
    structlog.contextualize(
        trace_id=trace_id,
        span_id=span_id
    )
    
    return trace_id, span_id

# 在函数调用中传递上下文
def process_pipeline_step(step_name):
    with structlog.contextualize(step=step_name):
        logger.info("Processing pipeline step")
        # 执行业务逻辑
        logger.info("Pipeline step completed")
```

### 3. 错误处理与日志

```python
try:
    # 业务逻辑
    result = execute_speckit_command(command)
    logger.info("Command executed successfully", command=command, result=result)
except SpeckitError as e:
    logger.error(
        "Speckit command failed",
        command=command,
        error=str(e),
        error_type=type(e).__name__,
        retryable=e.retryable,
        suggested_fix=e.suggested_fix
    )
    raise
except Exception as e:
    logger.critical(
        "Unexpected error in speckit command",
        command=command,
        error=str(e),
        error_type=type(e).__name__,
        traceback=traceback.format_exc()
    )
    raise
```

这套日志管理体系确保了系统的可观测性，支持问题诊断、性能优化和合规审计。