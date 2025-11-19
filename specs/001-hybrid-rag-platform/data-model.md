# Data Model: Hybrid Graph-RAG Phase 1

## SubmissionPackage
- `id` (UUID) — 全局唯一作业 ID, 供 speckit 管线与追溯使用
- `source_path` (Path) — 用户提供的 txt/md 文件绝对路径
- `uploader` (String) — 触发 CLI/HTTP/CLI 用户
- `submitted_at` (Datetime) — 记录作业开始时间
- `status` (Enum: pending/running/succeeded/failed)
- `speckit_log_path` (Path) — 存放每次命令输出及章程校验报告
- `retry_of` (UUID | null) — 指向前一次失败作业

**Relationships**: 一对一关联 `ComplianceReport`; 多对多关联 `ConversationRecord` (基于 traceability)

## IDESupportPackage
- `id` (UUID)
- `ide_type` (Enum: cursor, roocode, custom)
- `generated_at` (Datetime)
- `command_dir` (Path) — e.g. `.cursor/commands`
- `agents_manifest` (Path) — 生成的 `agents.md` 或等效文件
- `spec_version` (String) — speckit 规范版本/commit
- `validation_report` (JSON) — 结构校验结果
- `related_submission` (UUID) — 追踪由哪个作业触发

## DocumentProfile
- `id` (UUID)
- `file_path` (Path)
- `doc_type` (Enum: agent, developer, external)
- `token_count` (Integer, 仅 agent 文档填充)
- `structure_score` (0-100)
- `style_findings` (JSON list)
- `last_reviewed_at` (Datetime)
- `submission_id` (UUID) — 与 `SubmissionPackage` 关联

## BestPracticeArtifact
- `id` (UUID)
- `title` (String)
- `applicable_stage` (Enum: constitution/specify/plan/tasks/check)
- `pillar` (Enum: pipeline, IDE, docs, compliance)
- `content` (Markdown)
- `source` (Enum: llm-auto, user-upload)
- `linked_conversations` (List[UUID])
- `version` (String) — 支持迭代

## ConversationRecord
- `id` (UUID)
- `source_tool` (Enum: cursor, roocode, manual)
- `storage_path` (Path)
- `import_type` (Enum: directory-scan, manual-upload)
- `timestamp` (Datetime)
- `participants` (JSON list)
- `vector_id` (String) — 对应 Chroma embedding
- `graph_node_id` (String) — NetworkX 节点标识
- `submission_refs` (List[UUID])

## VectorKnowledgeStore
- `id` (UUID)
- `conversation_id` (UUID)
- `embedding_checksum` (String)
- `graph_neighbors` (JSON list of node ids)
- `metadata` (JSON: topic, severity, model_used)

## ComplianceReport
- `id` (UUID)
- `submission_id` (UUID)
- `generated_at` (Datetime)
- `static_checks` (JSON list — Ruff/Mypy/ESLint 结果)
- `doc_findings` (JSON list — 三分法、token、结构建议)
- `traceability_gaps` (JSON list — Requirement vs code)
- `linked_pr` (String | null)
- `status` (Enum: pass/warn/fail)

## Relationships Overview
- `SubmissionPackage` 1—1 `ComplianceReport`
- `SubmissionPackage` 1—n `DocumentProfile`
- `SubmissionPackage` n—n `ConversationRecord` (通过 `submission_refs`)
- `ConversationRecord` 1—1 `VectorKnowledgeStore`
- `BestPracticeArtifact` n—n `ConversationRecord`
- `IDESupportPackage` n—1 `SubmissionPackage`

