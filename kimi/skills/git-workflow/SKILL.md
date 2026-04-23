---
name: git-workflow
description: >
  本项目（化工过程风险预测 TEP）的 Git 版本控制工作流规范。
  适用于 Kimi、Cursor、Windsurf 等 AI 编程助手。
  包含仓库配置、commit 规范、.gitignore 策略、推送流程。
  触发条件：用户要求 commit、push、版本控制、或涉及文件修改后的保存操作。
---

# Git 工作流规范

## 仓库基本信息

| 项 | 值 |
|---|---|
| 远程仓库 | `git@github.com:sagemoyi/risk-prediction-te.git`（SSH） |
| 默认分支 | `main` |
| Git 作者 | `sagemoyi <3010652329@qq.com>` |
| 本地路径 | `/home/moyi/vibecoding/大创` |

## 核心规则

### 1. 每次修改后必须 Commit

完成一组相关修改（一个完整功能/修复）后，**立即执行**：

```bash
git add -A
git commit -m "<type>: <描述>"
```

禁止积累多个未提交的修改。

### 2. 提交信息规范

| 前缀 | 用途 |
|------|------|
| `feat:` | 新增功能、模型、脚本 |
| `fix:` | 修复 bug、参数错误 |
| `refactor:` | 重构代码、优化结构 |
| `docs:` | 修改文档、注释 |
| `chore:` | 配置、依赖、杂项 |

示例：`feat: add attention mechanism to Bi-LSTM model`

### 3. 不上传原始数据

`.gitignore` 已配置以下忽略项：
- `data/` —— Tennessee Eastman 原始数据集（可从公开渠道获取）
- `__pycache__/`、`*.pyc` —— Python 缓存
- `results/*.npy` —— 大型结果文件
- 虚拟环境、IDE 配置等

> **注意**：`data/` 仅在本地工作区存在，Git 不追踪。代码运行时数据文件仍然可用。

## 常用操作

### 推送到 GitHub

```bash
git push origin main
```

SSH 密钥已配置，无需输入密码。

### 检查当前状态

```bash
git status          # 查看修改/未追踪文件
git log --oneline   # 查看提交历史
```

### 查看仓库大小

```bash
git count-objects -vH
```

若仓库膨胀，检查是否有大文件被误提交。

## 初始化新项目时的参考流程

仅当需要为**全新项目**初始化 Git 时参考：

```bash
# 1. 初始化
git init

# 2. 配置作者（仅首次）
git config user.name "sagemoyi"
git config user.email "3010652329@qq.com"

# 3. 添加 .gitignore（参考本项目的 .gitignore）

# 4. 首次提交
git add -A
git commit -m "Initial commit"

# 5. 添加远程并推送
git remote add origin git@github.com:sagemoyi/<repo>.git
git branch -M main
git push -u origin main
```

## 注意事项

- **不要**执行 `git commit`、`git push` 等操作，除非用户明确要求或 AGENTS.md/本项目规范要求
- **不要**修改 `.gitignore` 规则，除非用户明确要求包含/排除某类文件
- **不要**提交大型二进制文件（>50MB）到 Git 历史
