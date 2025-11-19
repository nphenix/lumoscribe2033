# 部署指南

本指南详细介绍 lumoscribe2033 的部署方法，包括开发环境、测试环境和生产环境的部署配置。

## 📋 目录

- [环境要求](#环境要求)
- [开发环境部署](#开发环境部署)
- [测试环境部署](#测试环境部署)
- [生产环境部署](#生产环境部署)
- [配置管理](#配置管理)
- [监控与维护](#监控与维护)
- [故障排除](#故障排除)

## 🏗️ 环境要求

### 系统要求

- **操作系统**: Windows 11 (推荐)
- **Python**: 3.12+
- **内存**: 8GB RAM (推荐 16GB)
- **存储**: 20GB 可用空间
- **网络**: 稳定的互联网连接

### 依赖服务

- **数据库**: SQLite (默认), PostgreSQL (生产环境推荐)
- **向量数据库**: ChromaDB
- **缓存**: Redis (可选，生产环境推荐)
- **AI 服务**: OpenAI API 或本地 Ollama

## 🚀 开发环境部署

### 1. 环境准备

```bash
# 1. 安装 Conda (推荐)
# 下载地址: https://docs.conda.io/en/latest/miniconda.html

# 2. 创建虚拟环境
conda create -n lumoscribe2033 python=3.12 -y
conda activate lumoscribe2033

# 3. 克隆项目
git clone https://github.com/lumoscribe2033/lumoscribe2033.git
cd lumoscribe2033
```

### 2. 安装依赖

```bash
# 安装生产依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt

# 或者安装为可编辑包
pip install -e .
```

### 3. 配置环境

```bash
# 复制环境配置模板
cp .env.example .env

# 编辑配置文件
notepad .env
```

**基础配置示例 (`.env`)**:
```env
# 应用配置
APP_ENV=development
APP_DEBUG=true
APP_LOG_LEVEL=DEBUG

# 服务器配置
API_HOST=0.0.0.0
API_PORT=8080
WORKER_CONCURRENCY=4

# 数据库配置
DATABASE_URL=sqlite:///./data/persistence/lumoscribe2033.db
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Redis 配置 (可选)
REDIS_URL=redis://localhost:6379/0

# LLM 配置
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OLLAMA_BASE_URL=http://localhost:11434

# 安全配置
SECRET_KEY=your-secret-key-here
API_KEY=dev-api-key-12345

# 监控配置
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/metrics
PROMETHEUS_ENABLED=true
```

### 4. 初始化数据目录

```bash
# 创建必要的数据目录
python scripts/bootstrap_data_dirs.py

# 初始化数据库
python scripts/init_sqlite.py

# 初始化向量存储
python scripts/init_chroma.py

# 初始化图存储
python scripts/init_networkx.py
```

### 5. 启动服务

```bash
# 启动 API 服务 (开发模式)
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# 启动任务队列
arq workers.settings.WorkerSettings

# 启动向量数据库 (如果需要)
chroma run --host 0.0.0.0 --port 8000

# 启动 Redis (如果需要)
redis-server --port 6379
```

### 6. 验证部署

```bash
# 检查服务状态
curl http://localhost:8080/health

# 查看 API 文档
# http://localhost:8080/docs
# http://localhost:8080/redoc
```

## 🧪 测试环境部署

### Docker 部署

#### 1. 构建镜像

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements-dev.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p data/persistence vector/chroma graph/snapshots

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### 2. Docker Compose 配置

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - APP_ENV=testing
      - DATABASE_URL=sqlite:///./data/persistence/test.db
      - CHROMA_HOST=ch而来
      - CHROMA_PORT=8000
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./vector:/app/vector
      - ./graph:/app/graph
    depends_on:
      - chroma
      - redis

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./vector/chroma:/chroma/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  worker:
    build: .
    command: arq workers.settings.WorkerSettings
    environment:
      - APP_ENV=testing
      - DATABASE_URL=sqlite:///./data/persistence/test.db
      - CHROMA_HOST=ch而来
      - CHROMA_PORT=8000
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./vector:/app/vector
      - ./graph:/app/graph
    depends_on:
      - app
      - redis

volumes:
  redis_data:
```

#### 3. 启动测试环境

```bash
# 构建并启动服务
docker-compose -f docker-compose.test.yml up --build -d

# 查看日志
docker-compose -f docker-compose.test.yml logs -f

# 运行测试
docker-compose -f docker-compose.test.yml exec app pytest tests/

# 停止服务
docker-compose -f docker-compose.test.yml down
```

## 🏭 生产环境部署

### 1. 环境准备

#### 系统配置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# 安装其他依赖
sudo apt install redis-server postgresql chromadb -y
```

#### 安全配置

```bash
# 配置防火墙
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable

# 创建应用用户
sudo adduser --system --group lumoscribe
sudo chown -R lumoscribe:lumoscribe /opt/lumoscribe2033
```

### 2. 应用部署

#### 使用 systemd

**应用服务配置** (`/etc/systemd/system/lumoscribe-api.service`):
```ini
[Unit]
Description=Lumoscribe2033 API Service
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=lumoscribe
Group=lumoscribe
WorkingDirectory=/opt/lumoscribe2033
Environment="PATH=/opt/lumoscribe2033/venv/bin"
EnvironmentFile=/opt/lumoscribe2033/.env
ExecStart=/opt/lumoscribe2033/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8080
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**任务队列服务配置** (`/etc/systemd/system/lumoscribe-worker.service`):
```ini
[Unit]
Description=Lumoscribe2033 Worker Service
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=lumoscribe
Group=lumoscribe
WorkingDirectory=/opt/lumoscribe2033
Environment="PATH=/opt/lumoscribe2033/venv/bin"
EnvironmentFile=/opt/lumoscribe2033/.env
ExecStart=/opt/lumoscribe2033/venv/bin/arq workers.settings.WorkerSettings
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 启动服务

```bash
# 重新加载 systemd
sudo systemctl daemon-reload

# 启用服务
sudo systemctl enable lumoscribe-api
sudo systemctl enable lumoscribe-worker

# 启动服务
sudo systemctl start lumoscribe-api
sudo systemctl start lumoscribe-worker

# 查看状态
sudo systemctl status lumoscribe-api
sudo systemctl status lumoscribe-worker
```

### 3. Nginx 配置

**反向代理配置** (`/etc/nginx/sites-available/lumoscribe2033`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL 配置
    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # API 代理
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时配置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # 静态文件
    location /static/ {
        alias /opt/lumoscribe2033/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### 启用站点

```bash
# 启用站点
sudo ln -s /etc/nginx/sites-available/lumoscribe2033 /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重新加载 Nginx
sudo systemctl reload nginx
```

### 4. 数据库配置

#### PostgreSQL 配置

```sql
-- 创建数据库
CREATE DATABASE lumoscribe2033_prod;

-- 创建用户
CREATE USER lumoscribe_user WITH PASSWORD 'secure_password';

-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE lumoscribe2033_prod TO lumoscribe_user;

-- 创建扩展
\c lumoscribe2033_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

#### 环境配置

```env
# 生产环境配置
APP_ENV=production
APP_DEBUG=false
APP_LOG_LEVEL=INFO

# 数据库配置
DATABASE_URL=postgresql://lumoscribe_user:secure_password@localhost:5432/lumoscribe2033_prod

# Redis 配置
REDIS_URL=redis://localhost:6379/0

# 安全配置
SECRET_KEY=your-super-secret-production-key
API_KEY=your-production-api-key

# 监控配置
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-monitoring-endpoint.com/v1/metrics
PROMETHEUS_ENABLED=true
```

## ⚙️ 配置管理

### 配置文件结构

```
config/
├── development.env     # 开发环境配置
├── testing.env         # 测试环境配置
├── production.env      # 生产环境配置
└── docker.env          # Docker 环境配置
```

### 环境变量说明

| 变量名 | 说明 | 默认值 | 生产环境 |
|--------|------|--------|----------|
| `APP_ENV` | 应用环境 | `development` | `production` |
| `APP_DEBUG` | 调试模式 | `true` | `false` |
| `APP_LOG_LEVEL` | 日志级别 | `DEBUG` | `INFO` |
| `API_HOST` | API 监听地址 | `0.0.0.0` | `127.0.0.1` |
| `API_PORT` | API 监听端口 | `8080` | `8080` |
| `DATABASE_URL` | 数据库连接字符串 | SQLite | PostgreSQL |
| `CHROMA_HOST` | ChromaDB 主机 | `localhost` | 内网地址 |
| `CHROMA_PORT` | ChromaDB 端口 | `8000` | `8000` |
| `REDIS_URL` | Redis 连接字符串 | 无 | 生产环境必需 |
| `SECRET_KEY` | 应用密钥 | 开发密钥 | 强密钥 |
| `API_KEY` | API 访问密钥 | 开发密钥 | 强密钥 |

### 配置验证

```python
# 配置验证脚本
python scripts/validate_config.py

# 输出示例
# ✅ 配置验证通过
# 📋 环境: production
# 🔍 数据库连接: OK
# 🔍 Redis 连接: OK
# 🔍 LLM 配置: OK
# 🔍 安全配置: OK
```

## 📊 监控与维护

### 1. 健康检查

```bash
# API 健康检查
curl -f http://localhost:8080/health || exit 1

# 数据库连接检查
curl -f http://localhost:8080/metrics/health || exit 1

# 任务队列检查
systemctl is-active --quiet lumoscribe-worker || exit 1
```

### 2. 日志管理

```bash
# 查看应用日志
sudo journalctl -u lumoscribe-api -f

# 查看任务日志
sudo journalctl -u lumoscribe-worker -f

# 日志轮转配置
sudo cp config/logrotate.d/lumoscribe2033 /etc/logrotate.d/
```

### 3. 备份策略

```bash
# 数据库备份脚本
#!/bin/bash
BACKUP_DIR="/backup/lumoscribe2033"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份数据库
pg_dump lumoscribe2033_prod > $BACKUP_DIR/db_$DATE.sql

# 备份向量存储
tar -czf $BACKUP_DIR/vector_$DATE.tgz vector/chroma/

# 清理旧备份 (保留7天)
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tgz" -mtime +7 -delete
```

### 4. 性能监控

```bash
# 系统资源监控
htop
iotop
nethogs

# 应用性能监控
curl http://localhost:8080/metrics

# 数据库性能
sudo -u postgres psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

## 🔧 故障排除

### 常见问题

#### 1. 服务启动失败

```bash
# 检查服务状态
sudo systemctl status lumoscribe-api

# 查看详细日志
sudo journalctl -u lumoscribe-api -n 50

# 常见错误解决
# - 端口占用: 修改 API_PORT
# - 权限问题: 检查文件权限
# - 依赖缺失: 重新安装依赖
```

#### 2. 数据库连接失败

```bash
# 检查数据库服务
sudo systemctl status postgresql

# 测试连接
psql -h localhost -U lumoscribe_user -d lumoscribe2033_prod

# 检查配置
python -c "from src.framework.shared.config import Config; print(Config().database_url)"
```

#### 3. LLM 服务不可用

```bash
# 检查 OpenAI API 密钥
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# 检查 Ollama 服务
curl http://localhost:11434/api/tags

# 查看 LLM 配置
python -c "from src.framework.shared.config import Config; print(Config().llm_config)"
```

#### 4. 性能问题

```bash
# 检查资源使用
top
iostat 1
netstat -tulpn

# 检查应用指标
curl http://localhost:8080/metrics/performance

# 数据库慢查询
sudo -u postgres psql -c "SELECT query, mean_time, calls FROM pg_stat_statements WHERE mean_time > 1000 ORDER BY mean_time DESC;"
```

### 恢复策略

#### 1. 应用恢复

```bash
# 重启应用服务
sudo systemctl restart lumoscribe-api

# 重启任务队列
sudo systemctl restart lumoscribe-worker

# 验证恢复
curl http://localhost:8080/health
```

#### 2. 数据恢复

```bash
# 从备份恢复数据库
psql lumoscribe2033_prod < /backup/lumoscribe2033/db_20251119_120000.sql

# 恢复向量存储
tar -xzf /backup/lumoscribe2033/vector_20251119_120000.tgz -C /

# 重启服务
sudo systemctl restart lumoscribe-api lumoscribe-worker
```

## 📞 支持与联系

- 📧 技术支持: 18210768480@139.com
- 📄 文档: [项目文档](docs/)
- 🐛 问题报告: [GitHub Issues](https://github.com/lumoscribe2033/lumoscribe2033/issues)

---

**注意**: 生产环境部署前请务必进行充分的测试，并根据实际需求调整配置参数。