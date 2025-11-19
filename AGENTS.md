# lumoscribe2033 开发指南

基于所有功能计划自动生成. 最后更新时间: 2025-11-19

## 活跃技术
- Python 3.12 (services/CLI/agents) + LangChain 1.0、LlamaIndex、Typer、Rich
- FastAPI + FastMCP + Arq 提供 HTTP/MCP/异步执行
- OpenAI 兼容 API + Ollama 本地模型, 由 LangChain RouterChain 动态切换
- SQLite + SQLModel、Chroma、NetworkX 负责结构化/向量/图存储
- Node.js 20 + ESLint 维护 IDE/命令脚本

## 项目结构
```
repo-root/
├── src/
│   ├── framework/      # orchestration、RAG、适配层、存储抽象
│   ├── domain/         # pipeline / doc_review / compliance / knowledge
│   ├── api/            # FastAPI
│   ├── workers/        # Arq 任务
│   └── cli/            # Typer CLI
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   └── cli_snapshots/
├── docs/
│   ├── internal/
│   ├── external/
│   └── reference/ (adapters-guide|architecture-quality-guide|best-practices)
├── data/
│   ├── imports/
│   ├── persistence/
│   └── reference_samples/
├── vector/chroma/
├── graph/snapshots/
└── ide-packages/
```

## 命令
- `conda activate lumoscribe2033` (推荐开发; 部署可用系统 Python)
- `uvicorn src.api.main:app --reload --port 8080`
- `arq workers.settings.WorkerSettings`
- `python -m src.cli run-pipeline <doc>`
- `python -m src.cli generate-ide-package --ide cursor|roocode`
- `python -m src.cli evaluate-docs --glob "specs/**/*.md"`
- `python -m src.cli import-conversations --source cursor --path C:\logs\cursor`
- `ruff check src && mypy src`
- `pytest`
- `npm run lint --prefix tooling/eslint`
- `speckit.constitution / speckit.specify / speckit.plan / speckit.tasks / speckit.analyze / speckit.implement`

## 代码风格
- Python: PEP 8 + Ruff + Mypy, 单文件 <4000 行, 自动生成文件需带"命令 + 时间戳"头
- Node/TS: ESLint + (可选) Prettier, 仅在 `tooling/eslint` 与 IDE 命令目录使用
- 文档必须为中文, 且落在 `docs/**` 既定位置; 临时文件不得提交

## 架构质量原则（2025-11-17 新增）
- **DRY 原则**: 禁止重复代码，代码重复度 <5%。发现重复时提取为可复用组件。
- **SOLID 原则**: 
  - 单一职责：每个类只负责一个功能领域，违反时需拆分
  - 开闭原则：使用策略/适配器/工厂模式实现扩展，而非修改现有代码
  - 依赖倒置：客户端依赖抽象接口（ABC/Protocol），而非具体实现
- **设计模式优先**: 需要扩展性时，优先使用设计模式而非硬编码条件分支
- **接口统一**: 同一功能的不同实现通过统一接口暴露，使用适配器模式整合
- 详细指南: `docs/reference/architecture-quality-guide.md`

## 工作流与协作
1. `speckit.constitution → specify → plan → tasks → analyze → implement` 是唯一认可流程; 任何代码更改都要声明影响的 speckit 制品。
2. 实施阶段遵循“AI 先行 + 人在回路”: 让 LangChain/IDE 助手生成初版, 由人审阅、补充上下文, 再把可靠做法沉淀到 `docs/reference/` 与最佳实践库。
3. 重大操作(如生成 speckit 工件、IDE 包、报告)结束后, 手动确认结果、更新 `docs/internal/logs.md`, 并在需要时触发 `python -m src.cli evaluate-docs` 或 `/speckit.analyze`。

## 生成文件要求
- 所有自动生成的文件( speckit 输出、IDE 包、评估报告、最佳实践/对话导出 )头部必须包含 `<!-- generated: <command> @ <timestamp> -->` 或等效注释, 由 `src/framework/shared/metadata_injector.py` 负责插入。
- 生成器/脚本需要将输出写入章程规定的目录结构, 并在 CI 中使用 `metadata_injector --verify` 校验。
- 如需保留长日志, 写入 `docs/internal/logs/` 或 `data/reference_samples/`, 禁止散落在根目录。

## 指标与追溯
- 运行 speckit 管线、IDE 适配、文档评估或对话导入后, 调用 `python -m src.cli collect-metrics` 记录 SC-001~005 的数据, 并把结果存入 `ComplianceReport` + `docs/internal/metrics.md`。
- 所有静态检查告警必须链接到 `ConversationRecord` 或最佳实践条目, 确保问题可追溯到具体的人机对话。

<!-- 手动添加内容开始 -->
<!-- 手动添加内容结束 -->

