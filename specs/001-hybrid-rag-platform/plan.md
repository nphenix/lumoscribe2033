# 实施计划: Hybrid Graph-RAG Phase 1 质量平台

**分支**: `001-hybrid-rag-platform` | **日期**: 2025-11-14 | **规范**: [spec.md](./spec.md)
**输入**: 来自 `/specs/001-hybrid-rag-platform/spec.md` 的功能规范

## 摘要

Phase 1 聚焦“speckit 全流程自动化 + IDE 适配 + 文档评估 + 最佳实践与对话溯源”四大能力, 只负责 speckit 工件的导入、静态检查与可追溯性。技术方案以 Windows 11 本地环境为核心: 由 LangChain 1.0 统一编排, LlamaIndex 提供 RAG 主干, FastAPI + FastMCP + Arq 提供同步/异步服务, CLI 通过 Typer/Rich 输出, 并以 SQLite + Chroma + NetworkX 维护结构化、向量和图形数据。静态检查基于 Ruff/Mypy/ESLint, 多模型调用支持 OpenAI 兼容接口与本地 Ollama, 通过 LangChain 路由实现动态切换。阶段 7 还会交付“指标采集 + 文件元数据守卫”组件, 用于量化 SC-001~005 并确保所有自动生成工件遵循章程 P4。

## 技术背景

**Language/Version**: Python 3.12 (services/CLI/agents), Node.js 20 (ESLint toolchain)  
**Primary Dependencies**: LangChain 1.0, LlamaIndex, FastAPI, FastMCP, Arq, Typer, Rich, openai-compatible SDK, Ollama, NetworkX, Chroma, SQLModel/SQLite, Ruff, Mypy, ESLint  
**Storage**: SQLite (jobs + metadata), Chroma (vector store), NetworkX + SQLite (graph snapshots), filesystem artifacts  
**Project Type**: Single-repo services (HTTP API + CLI + workers)  
测试: Pytest + pytest-asyncio + coverage; Ruff/Mypy 作为 lint/type gate; ESLint 针对 IDE 命令脚本; CLI snapshot 测试  
目标平台: Windows 11 主机(部署与开发环境一致); Conda(`lumoscribe2033`) 仅作为推荐开发环境, 部署时可直接使用系统 Python, 不强制依赖 Conda  
项目类型: 单一仓库(services + CLI + workers), 暴露 HTTP API、CLI、MCP 服务  
性能目标: speckit 管线单次运行 ≤10 分钟(50 页文档); 文档评估一次可覆盖 ≥20 份文件; CLI 命令回响 ≤3 秒; IDE 适配包生成 ≤30 秒  
约束条件: 仅使用 Windows 友好型开源框架; 业务代码与框架代码分离以便升级; 禁止在仓库中留下临时文件/缓存; 所有输出需中文; 单文件 <4000 行; 运行全程离线可用(可选本地模型)  
规模/范围: 同时支持 5 个 speckit 作业、2 个 IDE 适配目标、10k+ 对话记录、100 条最佳实践, 后续可横向扩展

## 章程检查

1. **P1 代码质量**: 规划以模块化包(`core/`, `api/`, `cli/`, `workers/`)实现, 文件分层并规定生成器/检查器, 配合 Ruff/Mypy/ESLint 避免巨石文件。  
2. **P2 目录结构与工件定位**: IDE 适配包、speckit 产物、向量/图数据均有固定目录(`.cursor/commands`, `.roo/commands`, `vector/chroma`, `graph/snapshots`), 管线结束即校验路径。  
3. **P3 临时文件纪律**: Arq/CLI 运行在 `%TEMP%` 子目录, 结束后清理; CI 针对 `.tmp`、`debug.log` 设拒绝规则。  
4. **P4 文档与可维护性**: 所有自动生成的文档/命令头部带生成命令+时间戳; `quickstart.md` 和 README 记录使用步骤; PR 模板要求列出文档更新。  
5. **环境与语言要求**: 所有脚本默认 UTF-8, CLI/日志输出中文; 依赖通过 `lumoscribe2033` Conda 环境装载。  
6. **P5 架构设计**: 严格遵循 SOLID 设计原则组织代码, 以清晰职责和模块边界保证 speckit 管线长周期可演进。  
7. **P6 代码复用**: 应用 DRY 原则消除冗余代码, 遇到复杂逻辑时必须抽离为职责单一、可复用的组件或函数, 并在 speckit 章程中登记。  
→ 当前规划符合章程, 无需复杂度豁免; 阶段 1 结束后再复检。

## 项目结构

### 文档(此功能)

```
specs/001-hybrid-rag-platform/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── openapi.yaml
└── tasks.md              # 将在 /speckit.tasks 中生成
```

### 源代码(仓库根目录)

```
src/
├── framework/                # 共用框架层(可升级)
│   ├── orchestrators/        # LangChain 1.0 运行器、LLM router
│   ├── rag/
│   ├── adapters/             # IDE/Conversation/LLM 接口
│   ├── storage/              # SQLite/Chroma/NetworkX 抽象
│   └── shared/               # schemas/utils
├── domain/                   # 业务逻辑层
│   ├── pipeline/             # speckit 自动化管线
│   ├── doc_review/           # 文档三分法评估
│   ├── compliance/           # 静态检查 & traceability
│   └── knowledge/            # 最佳实践 & 对话溯源
├── api/
│   ├── main.py
│   └── routes/
├── workers/
│   ├── settings.py
│   ├── lifecycle.py
│   └── tasks/
│       ├── __init__.py
│       ├── speckit.py
│       ├── pipeline.py
│       ├── compliance.py
│       ├── knowledge.py
│       └── metrics.py
├── cli/
│   ├── commands/
│   └── styles/
└── tests/
    ├── unit/
    ├── integration/
    ├── contract/
    └── cli_snapshots/

docs/
├── internal/                 # 面向开发者
├── external/                 # 面向外部用户
└── reference/
    ├── vendor/               # 第三方规范/白皮书
    └── prompts/              # 提示词与最佳实践样例

data/
├── imports/                  # 原始导入文件
├── persistence/              # 持久化缓存、graph、向量快照
└── reference_samples/        # 示例数据、对话样本

vector/chroma/                # 由脚本自动维护
graph/snapshots/
ide-packages/
```

**结构决策**:  
- 维持单一仓库, 通过 `framework/` 与 `domain/` 分层隔离框架代码与业务代码, 方便后续升级/复用。  
- `docs/` 与 `data/` 采用二级目录, 额外包含 reference/reference_samples, 统一存储参考文档与示例数据, 保持结构清晰并满足章程。  
- `vector`、`graph`、`ide-packages` 等产物都有固定目录, 便于 CI 检查与清理。

## 复杂度跟踪

目前没有违反章程需要单独论证的复杂度项; 待 future 迭代若引入额外分层(例如前端)再补充。
