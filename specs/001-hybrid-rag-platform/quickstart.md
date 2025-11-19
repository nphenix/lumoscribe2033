# Quickstart: Hybrid Graph-RAG Phase 1

## 1. 环境准备
1. 安装 [Miniconda](https://docs.conda.io/) 并创建项目环境:
   ```powershell
   conda create -n lumoscribe2033 python=3.12 -y
   conda activate lumoscribe2033
   ```
2. 安装系统依赖:
   ```powershell
   choco install git nodejs-lts -y    # Node.js 仅用于 ESLint
   ```
3. 复制 `.env.example` 为 `.env`, 配置本地 OpenAI 兼容地址与 Ollama 主机:
   ```
   OPENAI_API_BASE=http://localhost:11434/v1
   OPENAI_API_KEY=dummy
   OLLAMA_HOST=http://localhost:11434
   ```

## 2. 安装依赖
```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
npm install --prefix tooling/eslint
```

## 3. 初始化存储
```powershell
python scripts/init_sqlite.py              # 创建 SQLite schema
python scripts/init_chroma.py              # 初始化 Chroma 持久目录
python scripts/init_networkx.py            # 生成 graph/snapshots/seed.graphml
```

## 4. 运行核心服务
```powershell
# FastAPI + FastMCP 联合服务
uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8080

# Arq 异步 worker
arq workers.settings.WorkerSettings

# CLI 演示 speckit 管线
python -m src.cli run-pipeline samples/phase1.md
```

## 5. IDE 适配包生成
```powershell
python -m src.cli generate-ide-package --ide cursor
python -m src.cli generate-ide-package --ide roocode
```
生成结果位于 `ide-packages/<ide>/` 并自动同步到 `.cursor/commands`、`.roo/commands`。

## 6. 文档与对话评估
```powershell
# 手动触发三分法评估
python -m src.cli evaluate-docs --glob "specs/**/*.md"

# 导入对话
python -m src.cli import-conversations --source cursor --path C:\logs\cursor

# 查询最佳实践
python -m src.cli search-best-practices --query "speckit 管线失败"
```

## 7. 静态检查与测试
```powershell
ruff check src/
mypy src/
pytest
npm run lint --prefix tooling/eslint
```

## 8. 常见问题
- **LangChain 1.0 兼容性**: 请勿混用 0.x API, 若需扩展在 `src/core/orchestrators` 添加适配器。
- **离线模式**: 设置 `LLM_ROUTING_MODE=local` 强制使用 Ollama。
- **临时文件**: 所有命令会在 `%TEMP%/lumoscribe2033` 下写入缓存, 命令结束自动清理。若遇未清理, 可手动删除以满足章程的临时文件纪律。

