# ESLint 配置

这是 lumoscribe2033 项目的 ESLint 配置，用于 JavaScript/TypeScript 代码的静态检查。

## 使用方法

### 安装依赖
```bash
npm install
```

### 运行检查
```bash
# 检查所有 JS/TS 文件
npm run lint

# 自动修复可修复的问题
npm run lint:fix

# 仅检查语法
npm run lint:check
```

### 在项目根目录运行
```bash
npm run lint --prefix tooling/eslint
```

## 配置说明

- 支持 ES2022 语法
- TypeScript 支持（如果项目中有 TS 文件）
- Node.js 环境配置
- Prettier 集成
- 针对开发工具的特殊规则

## 规则重点

- 禁用 `console` 警告（开发工具中允许使用）
- TypeScript 严格模式
- Node.js 最佳实践
- 自动格式化集成