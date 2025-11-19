# lumoscribe2033 - Hybrid Graph-RAG 质量平台

> 基于 speckit 的 AI 驱动质量提升平台，支持多 IDE 适配、文档评估和对话溯源

![版本](https://img.shields.io/badge/version-0.1.0-blue)
![许可证](https://img.shields.io/badge/license-Apache%202.0-green)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![状态](https://img.shields.io/badge/status-技术栈预览版-yellow)
![构建状态](https://img.shields.io/github/actions/workflow/status/lumoscribe2033/lumoscribe2033/release.yml?label=CI%20Build)

[📖 快速开始](specs/001-hybrid-rag-platform/quickstart.md) • [🤝 贡献指南](CONTRIBUTING.md) • [⚠️ 安全政策](SECURITY.md) • [📚 API 文档](docs/external/api-guide.md)

## 🚀 技术栈预览版 (v0.1.0)

欢迎使用 lumoscribe2033 技术栈预览版！本版本展示了项目的基础架构和技术能力，为后续功能开发奠定坚实基础。

### 📋 版本特点

- **🏗️ 完整架构**: 展示了分层架构设计和模块化组织
- **🤖 AI 集成**: LangChain + LlamaIndex 双引擎支持
- **📊 质量保障**: 完善的测试体系和代码质量检查
- **🔧 开发体验**: 现代化开发工具链和自动化流程
- **📚 文档完善**: 全面的中文文档和使用指南

### 🎯 预览内容

本技术栈预览版包含：

1. **基础设施层**: 完整的项目结构和配置管理
2. **框架层**: 核心适配器、执行器和存储抽象
3. **API 层**: FastAPI 接口和路由系统
4. **任务层**: Arq 异步任务处理
5. **工具层**: CLI 命令行工具
6. **质量体系**: 测试、静态检查和 CI/CD

---

## 📁 项目结构

```
repo-root/
├── src/                          # 源代码目录
│   ├── framework/                # 框架层 - 共用基础设施
│   │   ├── orchestrators/        # LangChain 1.0 运行器、LLM router
│   │   ├── rag/                  # RAG 核心组件
│   │   ├── adapters/             # IDE/Conversation/LLM 接口适配器
│   │   ├── storage/              # SQLite/Chroma/NetworkX 存储抽象
│   │   └── shared/               # 共享工具和数据模型
│   ├── domain/                   # 领域层 - 业务逻辑
│   │   ├── pipeline/             # speckit 自动化管线
│   │   ├── doc_review/           # 文档三分法评估
│   │   ├── compliance/           # 静态检查与可追溯性
│   │   └── knowledge/            # 最佳实践与对话溯源
│   ├── api/                      # FastAPI 接口层
│   ├── workers/                  # Arq 异步任务
│   │   ├── tasks/                # 具体任务实现
│   │   │   ├── speckit.py        # Speckit 生成任务
│   │   │   ├── pipeline.py       # 管线执行任务
│   │   │   ├── compliance.py     # 合规性检查任务
│   │   │   ├── knowledge.py      # 知识管理任务
│   │   │   └── metrics.py        # 指标收集任务
│   │   └── lifecycle.py          # Arq 生命周期钩子
│   └── cli/                      # Typer 命令行工具
├── tests/                        # 测试目录
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   ├── contract/                 # 契约测试
│   └── cli_snapshots/            # CLI 快照测试
├── docs/                         # 文档目录
│   ├── internal/                 # 内部开发文档
│   ├── external/                 # 外部用户文档
│   └── reference/                # 参考资料
│       ├── adapters-guide.md   # 适配器开发指南
│       ├── architecture-quality-guide.md  # 架构质量指南
│       └── best-practices.md   # 最佳实践
├── data/                         # 数据目录
│   ├── imports/                  # 原始导入文件
│   ├── persistence/              # 持久化缓存
│   └── reference_samples/        # 示例数据和对话样本
├── vector/chroma/                # 向量存储 (ChromaDB)
├── graph/snapshots/              # 图存储 (NetworkX)
└── ide-packages/                 # IDE 适配包目录
```

## 🎯 预览版功能

本技术栈预览版展示了以下核心能力：

### 🤖 AI 与 RAG 能力
- **LangChain 1.0**: 代理创建和 LLM 编排
- **LlamaIndex**: RAG 解决方案和向量索引
- **多模型支持**: OpenAI 兼容 API + 本地 Ollama
- **智能路由**: 动态模型选择和负载均衡

### 🌐 API 与服务
- **FastAPI**: 高性能异步 Web 服务
- **FastMCP**: 模型上下文协议支持
- **RESTful API**: 完整的接口设计和文档
- **异步任务**: Arq 队列处理后台任务

### 🛠️ 开发工具链
- **CLI 工具**: Typer 命令行界面
- **质量检查**: Ruff + MyPy 代码质量保障
- **测试体系**: 单元测试、集成测试、契约测试
- **CI/CD**: GitHub Actions 自动化流程

### 📊 可观测性
- **监控指标**: OpenTelemetry 分布式追踪
- **日志系统**: Loguru + Structlog 结构化日志
- **性能分析**: 请求监控和性能指标
- **健康检查**: 服务状态监控

### 🗄️ 数据存储
- **关系数据库**: SQLModel + SQLite
- **向量数据库**: ChromaDB 向量存储
- **图数据库**: NetworkX 图分析
- **存储抽象**: 统一的数据访问接口

## 🚀 快速体验

### 📦 安装预览版

```bash
# 从 GitHub 安装技术栈预览版
pip install https://github.com/lumoscribe2033/lumoscribe2033/releases/download/v0.1.0/lumoscribe2033-0.1.0-py3-none-any.whl

# 或克隆源码
git clone https://github.com/lumoscribe2033/lumoscribe2033.git
cd lumoscribe2033
pip install -e .
```

### 🎯 体验功能

```bash
# 启动 API 服务
uvicorn src.api.main:app --port 8080

# 启动任务队列
arq workers.settings.WorkerSettings

# 查看 CLI 命令
python -m src.cli.main --help

# 运行 speckit 演示
python -m src.cli.main run-speckit

# 检查系统状态
curl http://localhost:8080/health
```

## 🛠️ 核心技术栈

### 🤖 AI 与 RAG
- **LangChain 1.0.6** + **langchain-classic 1.0.0** - 代理创建和 LLM 编排
- **LangChain Community 0.4.1** - LangChain 生态组件
- **LangChain OpenAI 1.0.2** - OpenAI 集成
- **LlamaIndex 0.14.8** - RAG 和向量索引管理
- **LlamaIndex OpenAI LLMs 0.6.9** - LLM 集成
- **LlamaIndex OpenAI Embeddings 0.5.1** - 嵌入模型集成

### 🌐 Web 与 API
- **FastAPI 0.121.2** - 高性能 REST API
- **FastMCP 2.13.0.2** - 模型上下文协议服务器
- **Uvicorn 0.38.0** - ASGI 服务器

### 📊 可观测性
- **OpenTelemetry SDK 1.21.0** - 分布式追踪与指标
- **Prometheus Client 0.20.0** - 指标收集
- **Loguru 0.7.2** - 日志记录
- **Structlog 24.1.0** - 结构化日志

### 🗄️ 数据存储
- **SQLModel 0.0.27** - 数据库抽象
- **ChromaDB 1.3.4** - 向量存储
- **NetworkX 3.2** - 图存储

### ⚡ 异步任务
- **Arq 0.26.3** - 异步任务队列

### 🛠️ 开发工具
- **Typer 0.20.0** - 命令行工具
- **Rich 14.2.0** - 终端输出美化
- **Ruff 0.14.5** + **MyPy 1.7.0** - 代码质量检查

## 📋 核心功能

**注意**: 以下功能在技术栈预览版中已实现基础架构，具体业务逻辑将在后续版本中完善：

- **speckit 自动化**: 基础框架已实现，业务逻辑待完善
- **多 IDE 支持**: 适配器架构已实现，具体 IDE 集成待完善
- **文档评估**: 分类器框架已实现，评估算法待优化
- **对话溯源**: 存储结构已实现，检索逻辑待完善
- **静态检查**: 检查框架已实现，规则库待扩展

## 📖 学习资源

### 📚 外部文档
- [📖 API 使用指南](docs/external/api-guide.md) - 完整的 API 使用说明
- [🚀 部署指南](docs/external/deployment.md) - 生产环境部署说明
- [🎯 快速开始](specs/001-hybrid-rag-platform/quickstart.md) - 5分钟上手指南

### 📋 内部文档
- [🏗️ 系统架构](docs/reference/system-architecture.md) - 架构设计和组件说明
- [🔧 最佳实践](docs/reference/best-practices.md) - 开发规范和最佳实践
- [📚 适配器指南](docs/reference/adapters-guide.md) - 扩展开发指南
- [✅ 质量指南](docs/reference/architecture-quality-guide.md) - 架构质量标准

### 🔗 相关链接
- [🌐 GitHub 仓库](https://github.com/lumoscribe2033/lumoscribe2033)
- [📦 PyPI 包](https://pypi.org/project/lumoscribe2033/)
- [🎯 项目计划](specs/001-hybrid-rag-platform/plan.md)
- [📋 任务清单](specs/001-hybrid-rag-platform/tasks.md)

## 🤝 参与贡献

技术栈预览版欢迎社区反馈和贡献！

### 📋 贡献指南

1. **环境准备**: 请使用 Windows 11 开发环境
2. **代码规范**: 遵循 PEP 8 和项目代码风格
3. **测试要求**: 确保所有测试通过
4. **文档更新**: 及时更新相关文档

### 🔄 开发流程

请遵循 [贡献指南](CONTRIBUTING.md) 中的工作流程：
1. 环境准备和代码规范
2. 代码更改需声明影响的功能模块
3. 运行静态检查并确保所有测试通过
4. 提交 PR 前请更新相关文档

### 📞 联系方式

- 📧 邮箱: 18210768480@139.com
- 🐛 问题报告: [GitHub Issues](https://github.com/lumoscribe2033/lumoscribe2033/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/lumoscribe2033/lumoscribe2033/discussions)


**注意**: 本项目遵循严格的目录结构和代码规范，请在开发前阅读完整的 [贡献指南](CONTRIBUTING.md)，开始工作前，让你的IDE快速阅读 [`AGENTS.md`](AGENTS.md) 文件遵循本项目开发，进入 vibe coding。

**版本**: v0.1.0 (技术栈预览版) - 2025年11月发布