# lumoscribe2033 项目贡献指南

欢迎为 lumoscribe2033 项目贡献代码！本指南将帮助您了解项目的贡献流程、代码规范和最佳实践。

## 📋 目录

- [贡献前必读](#贡献前必读)
- [开发环境设置](#开发环境设置)
- [项目结构](#项目结构)
- [代码规范](#代码规范)
- [工作流程](#工作流程)
- [测试要求](#测试要求)
- [文档规范](#文档规范)
- [提交规范](#提交规范)
- [代码审查](#代码审查)
- [常见问题](#常见问题)

## 📖 贡献前必读

### 项目理念

lumoscribe2033 是一个基于 speckit 的 AI 驱动质量提升平台，致力于：
- 提供标准化的 speckit 自动化流程
- 支持多 IDE 适配和文档评估
- 建立对话溯源和最佳实践库

### 开发原则

本项目严格遵循以下原则：
- **SOLID 原则**：单一职责、开闭原则、依赖倒置等
- **DRY 原则**：禁止重复代码，重复度 <5%
- **设计模式优先**：使用策略/适配器/工厂模式而非硬编码
- **接口统一**：同一功能的不同实现通过统一接口暴露

## 🛠️ 开发环境设置

### 系统要求

- **操作系统**：Windows 11
- **Python**：3.12+
- **Conda**：推荐使用（可选）

### 环境配置

1. **安装依赖**
   ```bash
   # 创建 Conda 环境
   conda create -n lumoscribe2033 python=3.12 -y
   conda activate lumoscribe2033
   
   # 安装项目依赖
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # 安装 Node.js 依赖（用于 ESLint）
   npm install --prefix tooling/eslint
   ```

2. **初始化存储**
   ```bash
   # 初始化数据库
   python scripts/init_sqlite.py
   
   # 初始化向量存储
   python scripts/init_chroma.py
   
   # 初始化图存储
   python scripts/init_networkx.py
   ```

3. **环境变量配置**
   ```bash
   # 复制环境变量模板
   cp .env.example .env
   
   # 编辑 .env 文件，配置您的环境
   # OPENAI_API_BASE=your_openai_api_base
   # OPENAI_API_KEY=your_openai_api_key
   # OLLAMA_HOST=http://localhost:11434
   ```

## 📁 项目结构

```
lumoscribe2033/
├── src/                          # 源代码目录
│   ├── framework/                # 框架层（可升级）
│   │   ├── orchestrators/        # LangChain 1.0 运行器、LLM router
│   │   ├── rag/                  # RAG 核心组件
│   │   ├── adapters/             # IDE/Conversation/LLM 接口适配器
│   │   ├── storage/              # SQLite/Chroma/NetworkX 抽象
│   │   └── shared/               # schemas/utils
│   ├── domain/                   # 业务逻辑层
│   │   ├── pipeline/             # speckit 自动化管线
│   │   ├── doc_review/           # 文档三分法评估
│   │   ├── compliance/           # 静态检查 & traceability
│   │   └── knowledge/            # 最佳实践 & 对话溯源
│   ├── api/                      # FastAPI 接口层
│   ├── workers/                  # Arq 异步任务
│   └── cli/                      # Typer 命令行工具
├── tests/                        # 测试目录
│   ├── unit/                     # 单元测试
│   ├── integration/              # 雐成测试
│   ├── contract/                 # 契约测试
│   └── cli_snapshots/            # CLI 快照测试
├── docs/                         # 文档目录
│   ├── internal/                 # 面向开发者
│   ├── external/                 # 面向外部用户
│   └── reference/                # 参考资料
└── specs/                        # 功能规范
    └── 001-hybrid-rag-platform/  # 当前功能规范
```

## 📏 代码规范

### Python 代码规范

- **代码风格**：遵循 PEP 8 + Ruff + Mypy
- **类型注解**：所有函数必须有类型注解
- **单文件大小**：不超过 4000 行
- **命名规范**：
  - 函数和变量：`snake_case`
  - 类名：`PascalCase`
  - 常量：`UPPER_SNAKE_CASE`

### 代码质量检查

```bash
# 运行代码检查
ruff check src/

# 运行类型检查
mypy src/

# 运行测试
pytest

# 运行 ESLint（IDE 命令脚本）
npm run lint --prefix tooling/eslint
```

### 文档规范

- **文档语言**：所有文档必须使用中文
- **自动生成文件**：必须包含生成头 `<!-- generated: <command> @ <timestamp> -->`
- **文档位置**：必须位于 `docs/**` 既定位置
- **临时文件**：禁止提交临时文件到仓库

## 🔄 工作流程

### 1. Speckit 工作流程

本项目遵循唯一的认可流程：

```
speckit.constitution → specify → plan → tasks → analyze → implement
```

**重要**：任何代码更改都要声明影响的 speckit 制品。

### 2. 开发流程

1. **分支策略**
   ```bash
   # 从 main 分支创建功能分支
   git checkout -b feature/your-feature-name
   ```

2. **代码实现**
   - 实施阶段遵循"AI 先行 + 人在回路"
   - 让 LangChain/IDE 助手生成初版
   - 由人审阅、补充上下文
   - 将可靠做法沉淀到 `docs/reference/` 与最佳实践库

3. **测试和验证**
   ```bash
   # 运行相关测试
   pytest tests/unit/test_your_module.py
   
   # 运行静态检查
   ruff check src/your_module/
   mypy src/your_module/
   ```

4. **文档更新**
   - 更新相关文档
   - 添加生成元数据头
   - 记录变更到 `docs/internal/logs.md`

5. **提交代码**
   ```bash
   # 提交前运行完整检查
   ruff check src/ && mypy src/ && pytest
   
   # 提交代码
   git add .
   git commit -m "feat: 添加新功能说明"
   ```

## 🧪 测试要求

### 测试分类

- **单元测试**：测试单个组件功能
- **集成测试**：测试组件间交互
- **契约测试**：验证 API 契约一致性
- **CLI 快照测试**：验证命令行工具输出

### 测试规范

- **覆盖率**：新代码覆盖率不低于 80%
- **命名规范**：`test_*.py` 或 `*_test.py`
- **异步测试**：使用 `pytest-asyncio`
- **夹具使用**：合理使用 pytest fixtures

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定类型测试
pytest tests/unit/
pytest tests/integration/
pytest tests/contract/

# 运行带覆盖率的测试
pytest --cov=src --cov-report=html
```

## 📚 文档规范

### 文档分类

- **内部文档**：`docs/internal/` - 面向开发者
- **外部文档**：`docs/external/` - 面向用户
- **参考文档**：`docs/reference/` - 最佳实践和指南

### 自动文档生成

所有自动生成的文档必须包含元数据头：

```markdown
<!-- generated: python -m src.cli generate-docs @ 2025-11-19T19:00:00Z -->
```

### 文档更新流程

1. 重大操作后手动确认结果
2. 更新 `docs/internal/logs.md`
3. 在需要时触发 `python -m src.cli evaluate-docs` 或 `/speckit.analyze`

## 📝 提交规范

### 提交信息格式

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### 提交类型

- **feat**：新功能
- **fix**：bug 修复
- **docs**：文档更新
- **style**：代码格式（不影响功能）
- **refactor**：重构
- **test**：测试相关
- **chore**：构建过程或辅助工具的变动

### 示例

```bash
# 好的提交信息
feat(api): 添加用户认证接口
fix(cli): 修复配置文件解析错误
docs(reference): 更新架构设计文档
test(unit): 添加用户模型单元测试

# 避免的提交信息
fix bug
update
add stuff
```

## 👀 代码审查

### 审查清单

提交 PR 时，请确保：

- [ ] 代码遵循项目编码规范
- [ ] 通过了 Ruff 代码检查
- [ ] 通过了 MyPy 类型检查
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] 新增的 `docs/**/*.md` 文件包含生成元数据头
- [ ] 通过了文档分类检查
- [ ] 更新了相关的 ComplianceReport
- [ ] 追溯性映射正确

### PR 模板

请使用项目提供的 PR 模板，完整填写：
- 关联的 Speckit 工件
- 修改类型和影响分析
- 自查清单
- 测试验证结果

## 🤝 贡献流程

1. **提出问题或建议**
   - 在 GitHub Issues 中描述您的想法
   - 使用相应的标签分类

2. **开始开发**
   - Fork 项目到您的账户
   - 创建功能分支
   - 实施功能

3. **测试和验证**
   - 运行所有相关测试
   - 确保通过所有 CI 检查
   - 更新文档

4. **提交 PR**
   - 提交 Pull Request
   - 填写完整的 PR 模板
   - 等待代码审查

5. **合并**
   - 根据审查意见修改代码
   - 通过所有检查后合并

## ❓ 常见问题

### Q: 如何处理代码冲突？

A: 在提交前先同步最新代码：
```bash
git checkout main
git pull origin main
git checkout your-branch
git rebase main
# 解决冲突后
git push origin your-branch --force-with-lease
```

### Q: 如何添加新的依赖？

A: 
1. 在 `requirements.txt` 或 `requirements-dev.txt` 中添加
2. 在 `pyproject.toml` 中添加版本约束
3. 提交依赖更新和相关代码更改

### Q: 如何处理大型功能开发？

A: 将大型功能拆分为多个小的 PR：
1. 先提交基础框架
2. 逐步添加功能实现
3. 最后完善测试和文档

### Q: 如何确保代码质量？

A: 
1. 运行本地检查：`ruff check src/ && mypy src/`
2. 运行完整测试套件：`pytest`
3. 检查 CI 结果
4. 遵循代码审查反馈

## 📞 联系我们

如果您有任何问题或需要帮助：
- 提交 GitHub Issue
- 查看项目文档
- 参与讨论

感谢您为 lumoscribe2033 项目贡献代码！

---

**注意**：本指南会随着项目发展而更新。贡献前请务必查看最新版本的指南。