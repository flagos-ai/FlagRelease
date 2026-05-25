# FlagOS 全自动迁移流程 — 使用指南（V1+V2，无 V3）

## 方式一：Shell 脚本一键启动（推荐）

```bash
bash prompts/run_pipeline.sh <容器名或镜像地址> <模型名> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--verbose]
```

**自动识别**：第一参数若为已有容器则走容器模式，否则视为镜像地址。无需手动指定 `--image`。

**模型路径自动搜索**：仅需提供模型名（如 `Qwen3-8B`），脚本自动在宿主机搜索模型权重。找到则挂载，未找到则在容器创建后自动下载。

### 示例

```bash
# 已有容器 — 自动识别
bash prompts/run_pipeline.sh qwen3-8b-test Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass

# 镜像地址 — 自动识别，模型路径自动搜索
bash prompts/run_pipeline.sh harbor.baai.ac.cn/flagrelease/qwen3:latest Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass

# 调试模式（显示全量终端输出）
bash prompts/run_pipeline.sh qwen3-8b-test Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --verbose
```

### 自动识别规则

| 条件 | 判定 |
|------|------|
| 第一参数含冒号 `:` 或斜杠 `/`（如 `registry/image:tag`） | 镜像模式（优先） |
| 上述不匹配，且 `docker inspect --type=container` 成功 | 容器模式 |
| 以上均不匹配 | 镜像模式 |

镜像模式下，步骤1会自动检测 GPU 厂商、选择对应 docker run 模板创建容器，容器名自动生成为 `<model_short_name>_flagos`。如同名容器已存在，追加时间戳 `_MMDD_HHMM` 创建新容器，禁止复用已有容器。

脚本自动使用 `--permission-mode auto` 配合 `settings.local.json` 白名单，全程无需手动确认（兼容 root 用户）。

### 向后兼容

旧格式 `--image <镜像地址> <模型名> <宿主机模型路径> ...` 仍可使用，但会打印弃用警告。建议迁移到新的统一格式。

### 终端输出模式

默认精简模式，只显示步骤标记、✓/✗ 结果和关键命令，过滤 Claude 自言自语和探测命令。加 `--verbose` 恢复全量输出用于调试。

| 模式 | 说明 |
|------|------|
| 精简（默认） | ~200 行，只看关键进度 |
| `--verbose` | ~1000 行，全量输出，排查问题用 |

终端支持 ANSI 颜色（步骤蓝色、✓ 绿色、✗ 红色），管道/重定向时自动关闭。

---

## 方式二：手动 Claude CLI 命令

```bash
# 非交互模式 + 自动权限（推荐，兼容 root）
claude -p "容器名: qwen3-8b-test，模型名: Qwen3-8B ..." --permission-mode auto

# 交互模式 + 自动权限
claude --permission-mode auto
# 然后在交互界面中粘贴 prompt
```

> **注意**：`--dangerously-skip-permissions` 在 root 用户下被禁止，使用 `--permission-mode auto` 替代。

---

## 方式三：交互模式中粘贴 Prompt

新会话中粘贴以下内容（替换容器名/镜像名和模型名）：

### 容器模式

```
容器名: {CONTAINER}，模型名: {MODEL}

**执行模式：计划优先（Plan-First）**

在执行任何操作之前，先完成规划阶段：
1. 依次读取以下 SKILL.md 文件，提取每步的关键命令、参数、文件路径：
   - skills/flagos-container-preparation/SKILL.md
   - skills/flagos-pre-service-inspection/SKILL.md
   - skills/flagos-service-startup/SKILL.md
   - skills/flagos-eval-comprehensive/SKILL.md
   - skills/flagos-performance-testing/SKILL.md
   - skills/flagos-release/SKILL.md
2. 生成 execution_plan.md，写入 /data/flagos-workspace/{MODEL}/config/execution_plan.md
   - 包含每步的完整命令（变量已替换为实际值：容器名、模型名、端口等）
   - 包含每步的输入/输出文件路径
   - 包含每步的 context.yaml 读写字段清单
   - 包含每步的校验检查项
3. 每个步骤开始前，Read execution_plan.md 中对应段落刷新记忆
4. 每个步骤开始前，Read context.yaml 获取最新共享状态

请严格按以下 6 步执行 FlagOS 全自动迁移流程。步骤1-5 全自动执行，步骤间无需询问我。仅在6发布阶段如需 token 再来询问。

**严格禁止**：不进行 V3 算子优化。精度或性能不达标时仅输出报告，不调用 operator_search.py / operator_optimizer.py / diagnose_ops.py 进行任何算子排查或优化。直接继续后续步骤。

1 容器准备：
   - 验证容器 {CONTAINER} 运行状态（docker inspect + docker start）
   - 搜索模型权重：python3 skills/flagos-container-preparation/tools/check_model_local.py --model "{MODEL}" --mode container --container {CONTAINER} --output-json
     - 容器内搜索 → 宿主机搜索+挂载检查 → 容器内自动下载（如需要）
     - 记录 model.container_path 和 model.local_path
   - bash skills/flagos-container-preparation/tools/setup_workspace.sh {CONTAINER} {MODEL} 部署工具脚本
   - mkdir -p /data/flagos-logs/{MODEL}/ 创建全局 issue 日志目录
   - 写入 context.yaml + traces/01_container_preparation.json

2-6 同 run_pipeline.sh 中的步骤定义。
```

### 镜像模式

```
镜像: {IMAGE}，模型名: {MODEL}

（同容器模式的规划阶段和步骤2-6，仅步骤1不同）

1 容器准备（从镜像创建）：
   - 自动搜索宿主机模型权重（check_model_local.py --no-download --output-json）
   - 找到 → docker run 挂载该路径；未找到 → 容器创建后自动下载
   - 检测 GPU 厂商，选择 SKILL.md 中对应的 docker run 模板
   - docker run 创建容器（镜像: {IMAGE}）
   - 容器名自动生成为 <model_short_name>_flagos
   - 如同名容器已存在，追加时间戳：<model_short_name>_flagos_<MMDD_HHMM>
   - 镜像模式下禁止复用已有容器，必须 docker run 新建
   - setup_workspace.sh 部署工具脚本
   - 写入 context.yaml（entry.type=new_container）+ traces

2-6 同容器模式。
```

---

## Claude Code 权限授权方式对比

| 方式 | 命令 | 安全性 | 适用场景 |
|------|------|--------|----------|
| **`--permission-mode auto`** | `claude -p "..." --permission-mode auto` | 中（白名单内自动通过） | root 用户、日常开发（推荐） |
| **`--dangerously-skip-permissions`** | `claude -p "..." --dangerously-skip-permissions` | 最低（跳过所有检查） | 非 root 的可信沙箱、CI/CD |
| **settings.local.json 白名单** | 项目已配置，自动生效 | 中（仅放行白名单命令） | 搭配任意模式使用 |
| **交互确认**（默认） | `claude` | 最高（每次确认） | 敏感环境 |

本项目 `settings.local.json` 已预配置了 docker/pip/curl 等常用命令的白名单。`--permission-mode auto` 在此基础上自动判断是否执行，**无 root 限制**。

---

## 前置条件

- Claude Code CLI 已安装（`claude` 命令可用）
- Docker daemon 正在运行
- 容器模式：容器已存在且可启动（模型权重自动搜索，未找到则自动下载）
- 镜像模式：镜像可访问（本地已有或可 pull）
- 当前目录为项目根目录（`flagos_skills_V3/`）
- CLAUDE.md 已更新为简化版（已移除 V3 流程）

---

## 方式四：批量串行执行多个模型

当需要对多个模型依次执行迁移流程时，使用 `run_batch.sh`：

```bash
bash prompts/run_batch.sh <任务列表文件> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--verbose] [--stop-on-error] [--force]
```

### 任务列表文件格式

创建 `tasks.txt`，每行一个任务，`|` 分隔：

```
# 格式: 容器名或镜像地址 | 模型名
# 空行和 # 开头的行自动跳过
harbor.baai.ac.cn/flagrelease/qwen3:latest | Qwen3-8B
harbor.baai.ac.cn/flagrelease/llama3:latest | Llama-3-8B
my_existing_container | Qwen2.5-7B-Instruct
```

### 示例

```bash
# 基本用法
bash prompts/run_batch.sh tasks.txt ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass

# 调试模式
bash prompts/run_batch.sh tasks.txt ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --verbose

# 失败即停
bash prompts/run_batch.sh tasks.txt ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --stop-on-error

# 强制重跑已完成的任务（忽略断点）
bash prompts/run_batch.sh tasks.txt ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --force
```

### 断点续跑

默认自动跳过已完成的任务（通过检查 `context_snapshot.yaml` 中 `workflow.all_done` 字段）。中断后重跑同一个任务列表即可从未完成的任务继续。`--force` 可强制重跑所有任务。

### 选项

| 选项 | 说明 |
|------|------|
| `--verbose` | 透传给 run_pipeline.sh，显示全量终端输出 |
| `--stop-on-error` | 某个任务失败后终止整个批次（默认继续下一个） |
| `--force` | 强制重跑已完成的任务（默认跳过） |
