# Research: Hybrid Graph-RAG Phase 1

## Completed Tasks
- Research LangChain 1.0 与 LlamaIndex 的协同方式, 确保 agent orchestration 与 RAG 查询互不阻塞。
- 评估多模型路由策略, 同时兼容 OpenAI 兼容 API 与本地 Ollama。
- 定义 NetworkX + SQLite + Chroma 的分层存储方案, 便于 Windows 本地持久化。
- 设计文档三分法评估的策略(Agent/Developer/External)与 token 估算流程。
- 制定对话导入兼容矩阵, 支持 Cursor、RooCode 及手动 txt/md。
- 验证 Arq 在 Windows 环境下的异步执行可行性, 并与 FastAPI/CLI 协作。
- **新增**: 研究 LangChain 1.0 兼容性问题，发现需要使用 `langchain-classic` 包来保持向后兼容性。

## Decision Log

### 1. LangChain 1.0 + LlamaIndex Hand-off
- **Decision**: 使用 LangChain 1.0 RunnableSequence 作为顶层 orchestrator, 在生成 speckit 工件与执行检查时, 通过工具节点调用 LlamaIndex QueryEngine 处理 RAG 检索。
- **Rationale**: LangChain 1.0 提供稳定的 pydantic-style 配置、回调链路和多模型路由; LlamaIndex 对文档切片、检索和指标更成熟。分离 orchestrator 与 RAG 主干能保持组件解耦。
- **Alternatives considered**: 完全依靠 LangChain RetrievalQA (对复杂图谱检索支持不足); 直接使用 LlamaIndex Agent (LangChain 生态下 IDE/MCP 对接更困难)。

### 2. Multi-model Routing
- **Decision**: 通过 LangChain RouterChain + RunnableRouting, 以“任务类型 + 成本 + 隐私”规则在 ChatOpenAI(兼容模式) 与 ChatOllama 之间切换, 失败时回退到默认本地模型。
- **Rationale**: RouterChain 可在运行时基于 prompt 元数据选择模型, 支持将 Ollama 设为离线兜底; 统一接口降低 CLI 与 API 的复杂度。
- **Alternatives considered**: OpenAIProxy 的静态优先级(对本地模型支持弱); 手动 if/else 选择(难以扩展, 缺少可观察性)。

### 3. Storage Strategy (SQLite + NetworkX + Chroma)
- **Decision**: 采用 SQLite 记录 SubmissionPackage、DocumentProfile、ComplianceReport, 以 SQLModel 管理; NetworkX 仅在内存构建, 并把节点/边序列化回 SQLite; Chroma 持久化在 `vector/chroma`.
- **Rationale**: 全部组件均可在 Windows 本地运行, SQLite 便于备份, NetworkX 适合在内存构建知识图谱, Chroma 提供轻量向量库且无需 Docker。
- **Alternatives considered**: Neo4j/PGVector(需要额外服务); Weaviate/FAISS docker 镜像(违反不使用 Docker 的约束)。

### 4. Document Classification & Token Evaluation
- **Decision**: 使用 LangChain LCEL + OpenAI/Ollama 模型执行分类提示, 将已有文档按 Agent/Developer/External 标记; 仅对 Agent 文档调用 tiktoken 估算 token 并提供压缩建议, 其他文档聚焦结构/术语校验。
- **Rationale**: 手动分类成本高; LCEL PromptTemplate 容易插入章程原则; token 计算只在 Agent 文档进行可降低资源消耗。
- **Alternatives considered**: 纯规则正则(难以覆盖语义差异); 对所有文档做 token 统计(无必要, 增加噪声)。

### 5. Conversation Ingestion Compatibility
- **Decision**: 提供“目录扫描 + 手动上传”两种模式。Cursor、RooCode 默认目录采用解析器适配; 其余 txt/md 使用元数据表单补齐来源标签, 再写入 VectorKnowledgeStore。
- **Rationale**: IDE 已在本地生成日志文件, 直接扫描可避免手动操作; 手动导入用于外部对话补充。统一写入向量库可供检索。
- **Alternatives considered**: 仅允许人工上传(违背自动化目标); 直接抓取 IDE API(目前不公开, 不稳定)。

### 6. Async Execution on Windows
- **Decision**: 使用 Arq + RedisLite 替代; Arq 提供 asyncio worker, 可通过 in-process Redis 模拟器在 Windows 上运行。FastAPI 负责任务提交, CLI 提供 `arq run` 包装脚本, 确保 speckit 管线与文档评估异步执行。
- **Rationale**: 需要异步队列避免长任务阻塞; Arq 在纯 Python 环境下可运行, 配置简单。RedisLite/内嵌模式符合集成本地的原则。
- **Alternatives considered**: Celery + RabbitMQ/Redis(部署成本高); 自研线程池(缺少重试/监控机制)。

### 7. Observability & Logging
- **Decision**: 采用 structlog + OpenTelemetry exporter(本地文件)记录 LangChain 回调、Arq 任务状态与 CLI 操作; 关键指标写入 SQLite, 方便 speckit 分析。
- **Rationale**: 章程要求可追溯, structlog + OTEL 可记录 json 日志且在 Windows 上无需额外代理; 统一存储便于静态检查引用。
- **Alternatives considered**: 仅用 print/标准日志(难以与 LLM 调用关联); 商业 APM(不符离线要求)。

### 8. LangChain 1.0 兼容性策略
- **Decision**: 使用 `langchain-classic` 包替代直接导入 `langchain`, 确保所有 `langchain.chains` 和 `langchain.prompts` 导入正常工作。
- **Rationale**: LangChain 1.0 将部分模块移至单独的 `langchain-classic` 包中，直接导入会导致 ModuleNotFoundError。使用 `langchain-classic` 可以保持代码兼容性，同时享受 LangChain 1.0 的新特性。
- **Alternatives considered**: 手动重写所有导入路径(维护成本高); 使用旧版本 LangChain(无法获得新功能支持)。

