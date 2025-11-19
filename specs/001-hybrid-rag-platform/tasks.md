# 任务: Hybrid Graph-RAG Phase 1 质量平台

**输入**: 来自 `/specs/001-hybrid-rag-platform/` 的设计文档  
**前置条件**: plan.md、spec.md、research.md、data-model.md、contracts/、quickstart.md  
**测试**: 任务中已标明必要的测试与静态检查  
**组织结构**: 任务按阶段 + 用户故事分组, 确保每个故事可独立交付

## 阶段 1: 设置 (共享基础设施)

- [x] T001 初始化仓库目录结构(`src/framework`, `src/domain`, `docs/*`, `data/*`, `vector`, `graph`, `ide-packages`)并在 README 中记录用途
- [x] T002 在 `requirements.txt`/`requirements-dev.txt` 中锁定 Python 3.12 依赖版本, 包含 LangChain 1.0、LlamaIndex、FastAPI、FastMCP、Arq、SQLModel、Ruff、Mypy、Typer、Rich
- [x] T002b 添加 `langchain-classic` 依赖以支持 LangChain 1.0 兼容性
- [x] T002c 添加 `loguru`, `structlog` 及完整 OpenTelemetry 生态系统依赖
- [x] T003 [P] 配置 `pyproject.toml` 或 `ruff.toml`、`mypy.ini`、`pytest.ini`, 使 lint/type/test 规则与章程一致
- [x] T004 搭建 `tooling/eslint` 目录, 添加 ESLint + TypeScript 配置, 供 IDE 命令脚本使用
- [x] T005 [P] 编写 `scripts/init_sqlite.py`, `scripts/init_chroma.py`, `scripts/init_networkx.py`, `scripts/bootstrap_data_dirs.py` 以初始化本地存储
- [x] T006 配置 `structlog` + 本地 OpenTelemetry 导出器, 在 `src/framework/shared/logging.py` 中提供统一日志入口
- [x] T006b 创建 `src/workers/tasks/` 目录及5个任务模块(`speckit.py`, `pipeline.py`, `compliance.py`, `knowledge.py`, `metrics.py`)，实现Arq异步任务
- [x] T006c 实现 `src/workers/lifecycle.py` 生命周期钩子，确保任务执行的完整性和错误处理

---

## 阶段 2: 基础 (阻塞前置条件)

- [x] T007 设计并实现 `src/framework/orchestrators/langchain_runner.py`, 集成 LangChain 1.0 RunnableSequence + RouterChain, 支持 OpenAI 兼容 API 与 Ollama
- [x] T008 [P] 在 `src/framework/rag/index_service.py` 中封装 LlamaIndex 索引/查询流程, 读取 `data/persistence` 与 `vector/chroma`
- [x] T009 在 `src/framework/storage/sqlite_gateway.py` 中实现 SQLModel 连接池、迁移脚本、实体 CRUD
- [x] T010 [P] 搭建 Chroma 客户端适配层(`src/framework/storage/vector_store.py`)与 NetworkX 快照工具(`src/framework/storage/graph_store.py`)
- [x] T011 在 `src/framework/adapters/conversations/` 下编写 Cursor 与 RooCode 日志解析器, 支持目录扫描、元数据提取
- [x] T012 创建 `src/framework/adapters/llm_router.py`, 定义多模型路由策略(任务类型/成本/隐私)与 fallback 行为
- [x] T013 建立 `src/framework/shared/settings.py`, 支持 `.env` + 环境变量, 区分开发/部署(Conda 可选)
- [x] T014 配置 Arq + RedisLite, 实现 `workers/settings.py` 与任务注册, 提供 CLI/HTTP 统一调度接口
- [x] T015 创建 FastAPI 应用(`src/api/main.py`), 注册 `/pipeline/run`, `/documents/evaluate`, `/conversations/import`, `/best-practices/search`, `/compliance/reports/{id}` 等契约
- [x] T016 [P] 实现 Typer CLI(`src/cli/__main__.py`)骨架, 含 `run-pipeline`, `generate-ide-package`, `evaluate-docs`, `import-conversations`, `search-best-practices`

**检查点**: 平台具备 orchestrator、存储、API/CLI 骨架, 进入用户故事阶段

---

## 阶段 3: 用户故事 1 - 一键生成 speckit 全流程 (P1) 🎯 MVP

### 测试/校验
- [ ] T017 [P] 在 `tests/integration/test_pipeline_flow.py` 编写用例, 模拟提交自然语言文档, 断言 speckit 四份工件生成并附带日志
- [ ] T018 配置 CLI snapshot 测试, 校验 `run-pipeline` 命令输出

### 实施
- [ ] T019 在 `src/domain/pipeline/parser.py` 实现 txt/md ingestion + 章节抽取
- [ ] T020 [P] 编写 `src/domain/pipeline/speckit_executor.py`, 调用 LangChain orchestrator 串行执行 `/speckit.constitution → specify → plan → tasks`, 记录日志到 `SubmissionPackage`
- [ ] T021 实现 speckit 命令失败的重试/回滚逻辑, 将错误上下文写入 `ComplianceReport`
- [ ] T022 [P] 在 CLI/FastAPI 层连通 orchestrator, 支持同步提交与异步 (Arq) 执行, 返回 job_id
- [ ] T023 完成 `quickstart.md` 示例命令, 确保 Speckit 管线可在 Windows 11 上运行

**检查点**: 用户可上传文档并获取完整 speckit 工件 + 日志

---

## 阶段 4: 用户故事 2 - IDE 适配工件生成 (P1)

- [ ] T024 [P] 在 `src/domain/knowledge/ide_package_service.py` 实现适配器, 依据模板生成 `.cursor/commands/*.md`, `.roo/commands/*.json`, `agents.md`
- [ ] T025 构建 `src/framework/adapters/ide_validator.py`, 校验路径、章程引用、命令语法
- [ ] T026 [P] CLI 命令 `generate-ide-package --ide <name>` 与 FastAPI `/ide-packages/generate` 调用适配器并输出验证报告
- [ ] T027 在 tests/contract/ 添加 IDE 适配快照/结构测试, 覆盖 Cursor 与 RooCode
- [ ] T028 记录适配包元数据到 `IDESupportPackage`, 并在 `docs/internal/IDE.md` 说明扩展流程

**检查点**: IDE 静态文件可自动生成并通过校验

---

## 阶段 5: 用户故事 3 - 文档三分法评估 (P2)

- [ ] T029 实现 `src/domain/doc_review/classifier.py`, 使用 LangChain LCEL + 多模型路由对现有文档进行 Agent/Developer/External 分类
- [ ] T030 [P] 在 `classifier.py` 中仅对 Agent 文档执行 token 估算, Output 精简建议; Developer/External 侧重结构/格式
- [ ] T031 构建 `src/domain/doc_review/report_builder.py`, 生成 `DocumentProfile` + 整改项
- [ ] T032 [P] CLI 命令 `evaluate-docs --glob` 与 FastAPI `/documents/evaluate` 调用评估逻辑, 支持手动触发
- [ ] T033 编写 `tests/unit/test_doc_classifier.py` 覆盖分类/评分/建议
- [ ] T034 在 `docs/internal/docs-policy.md` 记录三分法标准与触发方式

**检查点**: 文档评估报告可手动触发, 输出分类+整改建议

---

## 阶段 6: 用户故事 4 - 最佳实践 & 对话溯源 (P2)

- [ ] T035 实现 `src/domain/knowledge/best_practice_service.py`, 支持增删改查 + 场景/章程检索
- [ ] T036 [P] 在 `src/domain/knowledge/conversation_ingestor.py` 中实现目录扫描与手动上传兼容, 解析 Cursor/RooCode/通用 txt
- [ ] T037 将对话嵌入写入 Chroma, 图节点写入 NetworkX/SQLite, 建立 `VectorKnowledgeStore`
- [ ] T038 [P] 构建检索 API `/best-practices/search` 与 `/conversations/import`, 将结果关联到 `ComplianceReport`
- [ ] T039 在 `src/domain/compliance/traceability.py` 中实现静态检查告警与对话/最佳实践的联动
- [ ] T040 添加 `tests/integration/test_conversation_ingest.py` 与 `tests/unit/test_best_practice_search.py`

**检查点**: 对话存储与最佳实践库可导入/检索, 静态检查告警可追溯

---

## 阶段 7: 完善与横切关注点

- [ ] T041 [P] 打通 Spec-to-Code Traceability 报告(`src/domain/compliance/traceability.py`), 将任务/提交映射写入 `ComplianceReport`
- [ ] T042 集成 Ruff/Mypy/ESLint/pytest 到 CI, 阻止未更新文档或临时文件的提交; 更新 PR 模板
- [ ] T043 [P] 扩展 `quickstart.md` 与 `docs/external/overview.md`, 提供 CLI/HTTP/MCP 使用指南
- [ ] T044 实装 `docs/reference/` 与 `data/reference_samples/` 示例, 并在 README 中记录参考资料存放策略
- [ ] T045 [P] 运行 `/speckit.analyze`, 修复任何章程/文档/任务不一致项, 记录在 `ComplianceReport`
- [ ] T046 建立指标采集与可视化脚本(`src/domain/compliance/metrics.py`), 计算 SC-001~SC-005 所需的成功率/准确度/召回率, 并写入 `ComplianceReport` 与 `docs/internal/metrics.md`
- [ ] T047 [P] 实现生成产物头部注入器(`src/framework/shared/metadata_injector.py`)以及 CI 钩子, 确保 speckit 工件、IDE 包、评估报告等自动加上“生成命令 + 时间戳”并在 CI 中校验, 满足章程 P4

---

## 依赖关系与执行顺序

- 阶段 1 完成后才能启动阶段 2; 阶段 2 是各用户故事的基础.
- 用户故事 1、2 属于 P1, 在阶段 2 完成后可并行推进; 用户故事 3、4 依赖前两者的管线与适配能力.
- 阶段 7 在所有用户故事交付后进行, 聚焦合规与文档收尾.

