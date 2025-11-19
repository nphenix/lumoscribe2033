<!-- generated: cli:docs meta-inject @ 2025-11-16T20:04:04Z -->
<!-- generated: python -m src.cli metadata-injector @ 2025-11-16T10:47:00.081Z -->
<!-- classification: agent -->

# 文档指标与度量体系

## 概述

本文档定义了 lumoscribe2033 项目的文档指标收集、计算和报告机制，确保所有文档活动可量化、可追踪、可优化。

## 指标分类

### 1. Speckit 流程指标 (SC-001)

#### 指标定义
- **成功率**: 成功完成完整 speckit 流程的作业比例
- **响应时间**: 从提交到完成的平均时间
- **重试率**: 需要重试的作业比例

#### 计算公式
```python
speckit_success_rate = successful_jobs / total_jobs
speckit_avg_duration = sum(job_durations) / successful_jobs
speckit_retry_rate = retry_jobs / total_jobs
```

#### 目标值
- ✅ 成功率 ≥ 95%
- ✅ 平均响应时间 ≤ 600 秒（10分钟）
- ✅ 重试率 ≤ 5%

### 2. IDE 适配指标 (SC-002)

#### 指标定义
- **支持率**: 成功生成适配包的 IDE 类型比例
- **验证通过率**: 适配包验证脚本通过率
- **更新及时性**: 适配包与最新章程的同步率

#### 计算公式
```python
ide_support_rate = supported_ide_types / total_ide_types
ide_validation_pass_rate = passed_validations / total_validations
ide_sync_rate = synced_packages / total_packages
```

#### 目标值
- ✅ 支持率 = 100%（所有目标 IDE）
- ✅ 验证通过率 = 100%
- ✅ 更新及时性 = 100%

### 3. 文档质量指标 (SC-003)

#### 指标定义
- **分类准确率**: 文档自动分类的准确率
- **Token 优化率**: Agent 文档 token 消耗降低比例
- **结构评分**: Developer/External 文档的平均结构评分

#### 计算公式
```python
classification_accuracy = correct_classifications / total_documents
token_optimization_rate = (initial_tokens - optimized_tokens) / initial_tokens
structure_score_avg = sum(structure_scores) / total_documents
```

#### 目标值
- ✅ 分类准确率 ≥ 98%
- ✅ Token 优化率 ≥ 30%
- ✅ 结构评分 ≥ 90/100

### 4. 静态检查指标 (SC-004)

#### 指标定义
- **CI 拦截率**: CI 中拦截的违规提交比例
- **覆盖率**: Spec-to-Code 追溯覆盖率
- **修复率**: 发现问题的修复比例

#### 计算公式
```python
ci_interception_rate = blocked_commits / total_violations
traceability_coverage = covered_requirements / total_requirements
fix_rate = fixed_issues / total_issues
```

#### 目标值
- ✅ CI 拦截率 ≥ 99%
- ✅ 覆盖率 = 100%
- ✅ 修复率 ≥ 95%

### 5. 对话检索指标 (SC-005)

#### 指标定义
- **语义召回率**: 相关对话的检索召回率
- **溯源覆盖率**: 严重告警的对话溯源比例
- **检索响应时间**: 平均检索响应时间

#### 计算公式
```python
semantic_recall_rate = relevant_retrieved / total_relevant
traceability_coverage = traced_alerts / severe_alerts
retrieval_response_time = sum(response_times) / total_queries
```

#### 目标值
- ✅ 语义召回率 ≥ 95%
- ✅ 溯源覆盖率 = 100%
- ✅ 响应时间 ≤ 2 秒

## 指标收集机制

### 1. 自动收集

#### Pipeline Metrics
```python
# 在 pipeline 执行过程中收集
class PipelineMetricsCollector:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.steps = []
        
    def start_pipeline(self, job_id):
        self.start_time = datetime.utcnow()
        self.job_id = job_id
        
    def record_step(self, step_name, duration, success):
        self.steps.append({
            'step': step_name,
            'duration': duration,
            'success': success,
            'timestamp': datetime.utcnow()
        })
        
    def end_pipeline(self, success):
        self.end_time = datetime.utcnow()
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # 记录到数据库
        metrics_store.save_pipeline_metrics({
            'job_id': self.job_id,
            'total_duration': total_duration,
            'success': success,
            'steps': self.steps,
            'timestamp': self.start_time
        })
```

#### Document Quality Metrics
```python
# 文档评估时收集
class DocumentMetricsCollector:
    def collect_document_metrics(self, file_path, doc_type, analysis_result):
        metrics = {
            'file_path': file_path,
            'doc_type': doc_type,
            'token_count': analysis_result.get('token_count', 0),
            'structure_score': analysis_result.get('structure_score', 0),
            'style_score': analysis_result.get('style_score', 0),
            'findings_count': len(analysis_result.get('findings', [])),
            'recommendations_count': len(analysis_result.get('recommendations', [])),
            'timestamp': datetime.utcnow()
        }
        
        metrics_store.save_document_metrics(metrics)
```

#### Compliance Metrics
```python
# 合规检查时收集
class ComplianceMetricsCollector:
    def collect_compliance_metrics(self, submission_id, report):
        static_check_count = len(report.get('static_checks', []))
        doc_finding_count = len(report.get('doc_findings', []))
        traceability_gap_count = len(report.get('traceability_gaps', []))
        
        metrics = {
            'submission_id': submission_id,
            'static_check_count': static_check_count,
            'doc_finding_count': doc_finding_count,
            'traceability_gap_count': traceability_gap_count,
            'status': report.get('status'),
            'timestamp': datetime.utcnow()
        }
        
        metrics_store.save_compliance_metrics(metrics)
```

### 2. 手动收集

#### 月度报告生成
```python
class MonthlyMetricsReporter:
    def generate_monthly_report(self, month, year):
        # 收集当月所有指标
        pipeline_metrics = self.get_pipeline_metrics(month, year)
        document_metrics = self.get_document_metrics(month, year)
        compliance_metrics = self.get_compliance_metrics(month, year)
        
        # 计算汇总指标
        summary = {
            'speckit_success_rate': self.calculate_speckit_success_rate(pipeline_metrics),
            'ide_support_rate': self.calculate_ide_support_rate(),
            'classification_accuracy': self.calculate_classification_accuracy(document_metrics),
            'ci_interception_rate': self.calculate_ci_interception_rate(compliance_metrics),
            'semantic_recall_rate': self.calculate_semantic_recall_rate(),
            
            # 趋势分析
            'trends': self.analyze_trends(month, year),
            
            # 改进建议
            'recommendations': self.generate_recommendations(summary)
        }
        
        return summary
```

## 指标报告格式

### 1. 实时仪表板

```json
{
  "timestamp": "2025-11-16T10:47:00Z",
  "overall_status": "healthy",
  "metrics": {
    "speckit": {
      "success_rate": 0.97,
      "avg_duration": 450.5,
      "retry_rate": 0.02,
      "status": "pass"
    },
    "ide": {
      "support_rate": 1.0,
      "validation_pass_rate": 1.0,
      "sync_rate": 1.0,
      "status": "pass"
    },
    "documents": {
      "classification_accuracy": 0.99,
      "token_optimization_rate": 0.35,
      "structure_score_avg": 92.5,
      "status": "pass"
    },
    "compliance": {
      "ci_interception_rate": 0.995,
      "traceability_coverage": 1.0,
      "fix_rate": 0.96,
      "status": "pass"
    },
    "conversations": {
      "semantic_recall_rate": 0.96,
      "traceability_coverage": 1.0,
      "avg_response_time": 1.8,
      "status": "pass"
    }
  },
  "alerts": [],
  "last_updated": "2025-11-16T10:47:00Z"
}
```

### 2. 详细报告

```markdown
# 月度指标报告 - 2025年11月

## 执行摘要
- 🟢 所有核心指标均达到目标值
- 📈 Speckit 成功率提升至 97%（上月：95%）
- 🎯 文档分类准确率保持 99%

## 详细指标

### Speckit 流程 (SC-001)
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| 成功率 | 97% | ≥95% | ✅ |
| 平均响应时间 | 7.5分钟 | ≤10分钟 | ✅ |
| 重试率 | 2% | ≤5% | ✅ |

### IDE 适配 (SC-002)
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| 支持率 | 100% | =100% | ✅ |
| 验证通过率 | 100% | =100% | ✅ |
| 更新及时性 | 100% | =100% | ✅ |

### 文档质量 (SC-003)
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| 分类准确率 | 99% | ≥98% | ✅ |
| Token 优化率 | 35% | ≥30% | ✅ |
| 结构评分 | 92.5/100 | ≥90 | ✅ |

### 合规检查 (SC-004)
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| CI 拦截率 | 99.5% | ≥99% | ✅ |
| 追溯覆盖率 | 100% | =100% | ✅ |
| 修复率 | 96% | ≥95% | ✅ |

### 对话检索 (SC-005)
| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| 语义召回率 | 96% | ≥95% | ✅ |
| 溯源覆盖率 | 100% | =100% | ✅ |
| 响应时间 | 1.8秒 | ≤2秒 | ✅ |

## 趋势分析
- Speckit 成功率连续3个月提升
- 文档 Token 优化效果显著
- CI 拦截率保持稳定高位

## 改进建议
1. 继续优化 pipeline 并发处理能力
2. 扩展更多 IDE 类型支持
3. 增强对话检索的上下文理解