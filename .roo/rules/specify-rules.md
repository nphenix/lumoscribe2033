# lumoscribe2033 开发指南

基于所有功能计划自动生成. 最后更新时间: 2025-11-14

## 活跃技术
- Python 3.12 (services/CLI/agents) + LangChain 1.0、LlamaIndex、Typer、Rich
- FastAPI + FastMCP + Arq 组成 HTTP/MCP/异步执行层
- 多模型路由: OpenAI 兼容 API + 本地 Ollama (LangChain RouterChain)
- SQLite + SQLModel、Chroma、NetworkX 负责结构化/向量/图存储
- Node.js 20 + ESLint 维护 IDE 命令脚本

## 项目结构
```
repo-root/
├── src/
│   ├── framework/
│   ├── domain/
│   ├── api/
│   ├── workers/
│   └── cli/
├── tests/ (unit/integration/contract/cli_snapshots)
├── docs/ (internal/external/reference)
├── data/ (imports/persistence/reference_samples)
├── vector/chroma/
├── graph/snapshots/
└── ide-packages/
```

## 命令
- `conda activate lumoscribe2033` (开发推荐)
- `uvicorn src.api.main:app --reload --port 8080`
- `arq workers.settings.WorkerSettings`
- `python -m src.cli run-pipeline samples/phase1.md`
- `python -m src.cli generate-ide-package --ide roocode`
- `python -m src.cli evaluate-docs --glob "specs/**/*.md"`
- `python -m src.cli import-conversations --source roocode --path C:\logs\roocode`
- `ruff check src && mypy src`
- `pytest`
- `npm run lint --prefix tooling/eslint`
- `speckit.constitution / speckit.specify / speckit.plan / speckit.tasks / speckit.analyze`

## 代码风格
- Python: PEP 8 + Ruff/Mypy, 单文件 <4000 行, 自动生成文件需带命令+时间戳
- Node: ESLint 规则 (tooling/eslint), IDE 命令保持 JSON/Markdown 结构
- 文档全中文, 临时文件不得提交, 目录必须遵循章程

## 工作流与协作
- speckit 命令必须按顺序运行; 任何改动 spec/plan/tasks 后立即执行 `/speckit.analyze`。
- RooCode 与 Cursor 一样需要先由 AI 生成草稿, 然后由人工审核并将经验沉淀到 `docs/reference/` / 最佳实践库。
- 每次管线运行、IDE 适配、文档评估或对话导入后, 运行 `python -m src.cli collect-metrics` 更新 SC-001~005 指标和 `docs/internal/logs.md`。

## 生成文件要求
- 通过 `python -m src.cli metadata-injector --files ...` 为 speckit 工件、IDE 包、评估报告添加 `生成命令 + 时间戳` 头部; CI 将验证此标注。
- 所有生成文件必须落在 `docs/**`、`data/**`、`vector/chroma`、`graph/snapshots`、`ide-packages` 等既定目录; 禁止写入根目录或临时路径。

## 指标与追溯
- `ComplianceReport` 需要附带 speckit 成功率、IDE 校验通过率、文档分类准确率、CI 拦截率、对话检索召回率; 数据由阶段 7 的 metrics 任务产出。
- 静态检查告警要链接到 `ConversationRecord` 或最佳实践条目, 便于追查问题对话; 使用 `python -m src.cli import-conversations --source roocode` 定期同步 RooCode 日志。

## 最近变更
- 001-hybrid-rag-platform: 引入 LangChain 1.0 + LlamaIndex + FastAPI/FastMCP/Arq + 指标采集/元数据守卫任务

## IDE 专用提示
- RooCode 脚本位于 `.roo/commands`、`.roo/rules`; 若需要新命令, 使用 `python -m src.cli generate-ide-package --ide roocode`
- 运行 speckit 工具需保证命令在项目根目录执行, 输出写入 `specs/<feature>/`

<!-- 手动添加内容开始 -->
<!-- 手动添加内容结束 -->
