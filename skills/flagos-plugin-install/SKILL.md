---
name: flagos-plugin-install
description: vllm-plugin-FL 组件安装/验证/卸载，安装后复用 V1 基线进行精度性能验证
version: 1.0.0
triggers:
  - 安装 plugin
  - install plugin
  - plugin 安装
  - vllm-plugin
depends_on: []
provides:
  - plugin_install.installed
  - plugin_install.version
  - plugin_install.success
---

# Plugin 安装 Skill

在 flaggems+flagtree 环境精度性能双达标后，安装 vllm-plugin-FL 组件并验证。

**工具脚本**（已由 setup_workspace.sh 部署到容器）：
- `install_plugin.py` — plugin 安装/验证/卸载

**前置条件**：
- flaggems + flagtree 环境已就绪
- 精度性能双达标（`workflow.accuracy_ok=true && workflow.performance_ok=true`）

**目标仓库**：`https://github.com/flagos-ai/vllm-plugin-FL`

---

# 上下文集成

## 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
model:
  name: <来自 container-preparation>
gpu:
  vendor: <来自 container-preparation>
workflow:
  accuracy_ok: <来自 eval-comprehensive>
  performance_ok: <来自 performance-testing>
environment:
  has_plugin: <来自 pre-service-inspection>
inspection:
  vllm_plugin_installed: <来自 pre-service-inspection>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
plugin_install:
  installed: true|false
  version: "<version>"
  repo_url: "<仓库地址>"
  install_method: "source|editable"
  success: true|false
  timestamp: "<ISO 8601>"
```

---

# 工作流程

## 步骤 1 — 检查前置条件

确认 flaggems+flagtree 环境精度性能双达标：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 -c \"
import yaml
ctx = yaml.safe_load(open('/flagos-workspace/shared/context.yaml'))
wf = ctx.get('workflow', {})
print(f'accuracy_ok={wf.get(\"accuracy_ok\")}, performance_ok={wf.get(\"performance_ok\")}')
\""
```

## 步骤 2 — 检查当前 plugin 状态

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py --action verify --json"
```

## 步骤 3 — 安装 plugin

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py \
    --action install --json"
```

指定分支：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py \
    --action install --branch main --json"
```

Editable 安装（开发调试用）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py \
    --action install --editable --json"
```

带代理：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py \
    --action install --proxy http://proxy:port --json"
```

## 步骤 4 — 验证安装

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py --action verify --json"
```

## 步骤 5 — 安装后验证流程

安装成功后，复用已有 V1(native) 基线，只跑 plugin 版本：

1. 启动服务（plugin 模式）
2. 精度评测 — 与 V1 基线对比
3. 性能评测 — 与 V1 基线对比

遇到 plugin 相关报错时，调用 issue_reporter 提交到 `flagos-ai/vllm-plugin-FL`：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \
    --type plugin-error \
    --log-path /flagos-workspace/logs/startup_flagos.log \
    --context-yaml /flagos-workspace/shared/context.yaml \
    --repo flagos-ai/vllm-plugin-FL \
    --output-dir /flagos-workspace/results/ \
    --json"
```

## 步骤 6 — 卸载（如需回退）

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py --action uninstall --json"
```

---

# 完成条件

- plugin 安装成功，版本已确认
- context.yaml `plugin_install` 字段已更新
- 安装后服务可正常启动
- 精度/性能与 V1 基线对比完成
- 遇到 plugin 报错时 issue 已提交到 `flagos-ai/vllm-plugin-FL`

---

# 故障排查

| 问题 | 解决方案 |
|------|----------|
| git clone 失败 | 检查网络，使用 `--proxy` 参数 |
| pip install 编译失败 | 确认 `--no-build-isolation` 已使用，检查构建依赖 |
| import 失败 | 检查 Python 环境，确认 conda 环境激活 |
| 服务启动后 plugin 未生效 | 检查 `VLLM_FL_PREFER_ENABLED` 环境变量 |
| plugin 与 flaggems 冲突 | 卸载 plugin 回退到 flaggems+flagtree 环境 |

---

## 编排层指令（步骤9 Plugin 安装 — 固化决策）

### 触发条件

步骤 8（自动发布）完成后，检查 `workflow.qualified`：
- `qualified=true` → 进入步骤 9，设置 `plugin_workflow.triggered=true`
- `qualified=false` → 跳过步骤 9-13，设置 `plugin_workflow.skip_reason="主流程不达标"`

### 调用方式

```bash
# 安装 plugin
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py --action install --json"

# 验证安装
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_plugin.py --action verify --json"
```

### 失败处理（强制停止）

Plugin 安装失败（install 返回 `success=false` 或 verify 返回 `installed=false`）时：

1. 调用 `issue_reporter.py` 提交 issue 到 plugin 仓库：
   ```bash
   docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \
       --type plugin-error \
       --context-yaml /flagos-workspace/shared/context.yaml \
       --repo flagos-ai/vllm-plugin-FL \
       --output-dir /flagos-workspace/results/ \
       --json"
   ```
2. 设置 `plugin_workflow.crash_stopped=true`
3. 写入 `traces/09_plugin_install.json`（status=failed）
4. **停止任务**，不继续步骤 10-13

### 编排层后续操作

安装成功后：
- 更新 `context.yaml` 的 `plugin_install` 字段
- 更新 `environment.env_type` 为 `vllm_plugin_flaggems`（如果原来不是）
- 更新 `environment.has_plugin=true`
- 写入 `traces/09_plugin_install.json`
- 更新 `workflow_ledger` 步骤 09 状态
- 更新 `timing.steps.plugin_install`

### 步骤 10-12 概述（在此处统一说明）

步骤 10（Plugin 服务启动）、11（Plugin 精度评测）、12（Plugin 性能评测）分别复用 `flagos-service-startup`、`flagos-eval-comprehensive`、`flagos-performance-testing` 的 Skill 逻辑，区别如下：

| 差异项 | 主流程（步骤 3-7） | Plugin 流程（步骤 10-12） |
|--------|-------------------|--------------------------|
| 算子集 | 从全量开始，经调优得到最终集合 | 直接使用主流程最终算子集 |
| Issue 路由 | FlagGems 仓库 | `flagos-ai/vllm-plugin-FL` |
| 服务崩溃 | 尝试恢复（切 native / 翻倍 TP） | **写 issue + 停止任务** |
| 不达标处理 | 触发算子调优（步骤 5/7） | **三级递进**：①写 issue → ②plugin 模式关算子调优 → ③全关仍不达标才判框架问题 |
| 启动环境变量 | 按 env_type 决定 | 固定 `USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true` + blacklist |
| V1 基线 | 当前流程内测得 | 复用步骤 4/6 的 V1 结果 |
| Trace 文件 | `traces/03-07_*.json` | `traces/10-12_*.json` |
| 日志命名 | `startup_flagos.log` | `startup_plugin.log` |

**步骤 10 算子控制（必须使用环境变量，禁止控制文件）**：

```bash
# 1. 从 context.yaml 获取 disabled_ops 列表（逗号分隔）
# 2. 调用 apply_op_config.py 生成 env_inline
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/apply_op_config.py \
    --mode custom --flagos-blacklist '${DISABLED_OPS_COMMA_SEPARATED}'"
# 3. 使用输出的 env_inline 作为启动命令前缀
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH \
    USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_FLAGOS_BLACKLIST='${DISABLED_OPS}' \
    vllm serve ... > /flagos-workspace/logs/startup_plugin.log 2>&1"
```

> ⚠️ Plugin 模式下 `VLLM_FL_PREFER_ENABLED=true` 会使注入代码 `pass`，跳过控制文件逻辑。
> 必须通过 `VLLM_FL_FLAGOS_BLACKLIST` 环境变量传递禁用算子，`/root/flaggems_ops_control.json` 在此场景无效。

**步骤 10 服务崩溃处理**：
```bash
# 服务崩溃 → 写 issue + 停止
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \
    --type plugin-error \
    --log-path /flagos-workspace/logs/startup_plugin.log \
    --context-yaml /flagos-workspace/shared/context.yaml \
    --repo flagos-ai/vllm-plugin-FL \
    --output-dir /flagos-workspace/results/ \
    --json"
# 设置 plugin_workflow.crash_stopped=true → 停止任务
```

**步骤 11/12 不达标处理（三级递进，精度/性能同此逻辑）**：

V3 切 plugin 用的是主流程已达标的算子集，但 plugin 实现路径与注入模式不完全等价，可能出现精度/性能下降。发现不达标时**不直接放弃**，按三级递进逐级抢救，只有最后一级失败才认定框架问题：

**第一级 — 记录 issue**
- 精度不达标：写 `logs/issues_accuracy.log`，`plugin_workflow.accuracy_ok=false`
- 性能不达标：写 `logs/issues_performance.log`，`plugin_workflow.performance_ok=false`
- 用 `issue_reporter.py full --type plugin-error --repo flagos-ai/vllm-plugin-FL` 提交（仅本地文件）

**第二级 — plugin 模式关算子调优（当前缺失的关键补救步）**
- 在主流程已达标算子集基础上，用 plugin 模式继续关闭拖累精度/性能的算子：
  ```bash
  docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/operator_search.py run \
      --plugin-mode \
      --final-output-name v3_performance \
      --state-path /flagos-workspace/results/operator_config_v3.json \
      ...（精度不达标用精度评测入口，性能不达标用性能评测入口）"
  ```
- plugin 模式下走 env_inline（`VLLM_FL_FLAGOS_BLACKLIST`），不写控制文件（见步骤 10 说明）。
- 调优达标 → 记录最终算子集，`accuracy_ok/performance_ok=true`，正常继续下一步。

**第三级 — 全关仍不达标才判框架问题**
- 若把 flaggems 算子全部关闭后精度/性能仍不达标，说明问题不在算子层而在 plugin 框架本身：
  - 提交 `plugin-error` issue（标注"全关算子仍不达标，判定为框架问题"）到 `flagos-ai/vllm-plugin-FL`
  - 保持 `accuracy_ok/performance_ok=false`，继续步骤 13
- 步骤 13 检查 `plugin_workflow.qualified`，不达标则按"Plugin 不达标发布"（Harbor 私有，不更新 README）处理。

> 与 CLAUDE.md:98/101 一致：调优前先提交 issue，plugin 模式下继续关算子直到达标；调优后仍不达标才走不达标发布。
> 注：本表上方"Plugin 算子集：复用主流程已达标算子集，不重新调优"仅描述**达标情况下**的默认路径；一旦不达标即进入本三级递进，允许在 plugin 模式下继续调优。

### 步骤 13（Plugin 发布）

触发条件：`plugin_workflow.accuracy_ok=true AND plugin_workflow.performance_ok=true`

```bash
# 宿主机执行 plugin 发布
python3 skills/flagos-release/tools/main.py \
    --from-context /data/flagos-workspace/<model>/config/context_snapshot.yaml \
    --plugin-mode
```

`--plugin-mode` 标志使发布脚本：
- 镜像 tag 追加 `-plugin` 后缀
- 复用步骤8原仓库（不创建新仓库）
- 用 plugin 镜像地址和评测数据重新生成 README，覆盖上传到步骤8的 ModelScope/HuggingFace 仓库
