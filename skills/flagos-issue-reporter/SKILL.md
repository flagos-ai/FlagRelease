---
name: flagos-issue-reporter
description: FlagGems/FlagTree/Plugin 问题自动归因与 GitHub issue 提交，支持算子崩溃、精度/性能不佳、框架报错六种场景
version: 2.0.0
triggers:
  - 提交 issue
  - submit issue
  - report bug
  - 自动报告
  - issue reporter
depends_on: []
provides:
  - issues.submitted
---

# 问题自动提交 Skill

自动收集算子/框架问题数据，格式化为 Bug Report 并提交到 GitHub。

**工具脚本**（已由 setup_workspace.sh 部署到容器）：
- `issue_reporter.py` — 问题收集、格式化、提交（collect / format / submit / full）

**目标仓库**：`flagos-ai/FlagGems`（以用户身份提交 issue）

**认证方式**：`GITHUB_TOKEN` 环境变量（GitHub Personal Access Token，需 `public_repo` 权限）
- 默认只生成带类型标注的 markdown 文件保存到本地（`issue_{type}_{repo}_{YYYYMMDD_HHMMSS}.md`）
- 显式传入 `--submit` 且 Token 已设置 → 保存 markdown + 自动通过 GitHub API 提交
- 未传 `--submit` 或 Token 未设置 → 只保存 markdown 到工作目录，用户可手动提交

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
  type: <来自 container-preparation>
inspection:
  core_packages: <来自 pre-service-inspection>
  flag_packages: <来自 pre-service-inspection>
environment:
  env_type: <来自 pre-service-inspection>
  flaggems_code_path: <flaggems 代码文件路径>
  flaggems_enable_call: <flag_gems.enable() 完整调用>
  flaggems_txt_path: <gems.txt 路径>
eval:
  excluded_ops_accuracy: <来自 eval-comprehensive>
optimization:
  disabled_ops: <来自 operator-replacement>
  search_log: <来自 operator-replacement>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
issues:
  submitted:
    - type: "<issue 类型>"
      url: "<GitHub issue URL>"
      ops: [<相关算子>]
      timestamp: "<ISO 8601>"
```

---

# 六种 Issue 类型

| type | 触发场景 | 标签 | 目标仓库 |
|------|---------|------|---------|
| `operator-crash` | 算子导致服务崩溃（启动失败或推理中崩溃） | `bug` | flagos-ai/FlagGems |
| `accuracy-zero` | 精度结果为零或 < 5%（严重异常） | `bug` | flagos-ai/FlagGems |
| `accuracy-degraded` | 精度调优流程筛出的精度不佳算子 | `bug` | flagos-ai/FlagGems |
| `performance-degraded` | 性能调优流程筛出的性能不佳算子 | `bug` | flagos-ai/FlagGems |
| `flagtree-error` | FlagTree/Triton 框架报错 | `bug` | flagos-ai/FlagGems |
| `plugin-error` | vllm-plugin-FL 框架报错 | `bug` | flagos-ai/vllm-plugin-FL |

---

# Issue 内容强制要求

**所有 issue 必须包含**：

1. **环境信息表**（硬件、GPU、软件版本）
2. **FlagGems Integration Code**（强制）：
   - `flaggems.enable()` 代码文件路径和调用行
   - 相关代码片段（上下文 ±5 行）
   - 当前 `gems.txt` 内容（已替换算子列表）
3. **错误日志** / 精度对比数据 / 性能对比数据
4. **诊断建议**

脚本通过 `--flaggems-code-path` 和 `--gems-txt-path` 参数自动采集，或从 context.yaml 的 `environment.flaggems_code_path` / `environment.flaggems_enable_call` / `environment.flaggems_txt_path` 自动读取。

---

# 触发规则（按工作流步骤）

## 步骤3 启服务

| 触发条件 | issue type | 时机 |
|---------|-----------|------|
| FlagGems 模式启动崩溃 | `operator-crash` | 即时提交 |

**处理流程**：保存日志 → 提交 issue（含 flaggems.enable 代码）→ 排除操作失误 → 标记 `workflow.service_ok: false` → 跳过4/6到8发布（私有）

## 步骤4 精度评测

| 触发条件 | issue type | 时机 |
|---------|-----------|------|
| 评测中服务崩溃 | `operator-crash` | 即时提交 |
| V2精度下降 >5% | `accuracy-degraded` | 发现即提交，然后开始算子优化 |
| 3 轮优化后仍不达标 | `accuracy-degraded` | 最终汇总提交（含所有禁用算子及原因） |

## 步骤6 性能评测

| 触发条件 | issue type | 时机 |
|---------|-----------|------|
| 评测中服务崩溃 | `operator-crash` | 即时提交 |
| V2/V1 任一并发级别 < 80% | `performance-degraded` | 发现即提交，然后开始算子优化 |
| 算子优化后仍不达标 | `performance-degraded` | 最终汇总提交（含所有禁用算子及原因） |

---

# 工作流程

## 一步完成（推荐）

```bash
# 服务启动崩溃（宿主机执行）
python3 issue_reporter.py full \
    --type operator-crash \
    --log-path /data/flagos-workspace/<model>/logs/startup_flagos.log \
    --context-yaml /data/flagos-workspace/<model>/shared/context.yaml \
    --flaggems-code-path <container内flaggems代码文件路径> \
    --gems-txt-path /root/gems.txt \
    --repo flagos-ai/FlagGems \
    --output-dir /data/flagos-workspace/<model>/results/ \
    --json

# 精度不达标
python3 issue_reporter.py full \
    --type accuracy-degraded \
    --disabled-ops "softmax,layer_norm" \
    --disabled-reasons '{"softmax":"偏差 8.2%","layer_norm":"偏差 6.1%"}' \
    --context-yaml /data/flagos-workspace/<model>/shared/context.yaml \
    --flaggems-code-path <container内flaggems代码文件路径> \
    --gems-txt-path /root/gems.txt \
    --repo flagos-ai/FlagGems \
    --output-dir /data/flagos-workspace/<model>/results/ \
    --json

# 性能不达标
python3 issue_reporter.py full \
    --type performance-degraded \
    --disabled-ops "mm,addmm" \
    --disabled-reasons '{"mm":"ratio 62%","addmm":"ratio 71%"}' \
    --context-yaml /data/flagos-workspace/<model>/shared/context.yaml \
    --flaggems-code-path <container内flaggems代码文件路径> \
    --gems-txt-path /root/gems.txt \
    --repo flagos-ai/FlagGems \
    --output-dir /data/flagos-workspace/<model>/results/ \
    --json

# plugin 报错（提交到 vllm-plugin-FL 仓库）
python3 issue_reporter.py full \
    --type plugin-error \
    --log-path /data/flagos-workspace/<model>/logs/startup_flagos.log \
    --context-yaml /data/flagos-workspace/<model>/shared/context.yaml \
    --repo flagos-ai/vllm-plugin-FL \
    --output-dir /data/flagos-workspace/<model>/results/ \
    --json
```

## 分步执行

### 步骤 1 — 收集问题数据

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py collect \
    --type operator-crash \
    --log-path /flagos-workspace/logs/startup_flagos.log \
    --context-yaml /flagos-workspace/shared/context.yaml \
    --flaggems-code-path <flaggems代码文件路径> \
    --gems-txt-path /root/gems.txt \
    --output /flagos-workspace/results/issue_data.json \
    --json"
```

### 步骤 2 — 格式化为 Bug Report

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py format \
    --collected-file /flagos-workspace/results/issue_data.json \
    --output /flagos-workspace/results/issue_report.md \
    --json"
```

### 步骤 3 — 提交 issue

```bash
python3 issue_reporter.py submit \
    --issue-file /data/flagos-workspace/<model>/results/issue_report.md \
    --repo flagos-ai/FlagGems \
    --json
```

**提交方式**：
1. 始终保存 markdown 到本地（`issue_{repo}_{YYYYMMDD_HHMMSS}.md`）
2. 有 `GITHUB_TOKEN` → 自动通过 GitHub API 提交
3. 无 token → 只保存 markdown，用户手动提交

---

# 完成条件

- 问题数据已收集（`issue_data.json`），包含 flaggems 代码上下文
- Bug Report 已格式化并保存为带仓库名+时间戳的 markdown 文件（`issue_{repo}_{YYYYMMDD_HHMMSS}.md`）
- markdown 文件包含元信息头（目标仓库、生成时间、issue 类型）
- 有 `GITHUB_TOKEN` 时 issue 已通过 API 提交，无 token 时 markdown 已保存到宿主机工作目录
- context.yaml `issues.submitted` 已更新（编排层负责）
- 对应 trace 文件中记录了 issue 提交操作

---

# 故障排查

| 问题 | 解决方案 |
|------|----------|
| `GITHUB_TOKEN` 未设置 | markdown 已保存到本地，如需自动提交：`export GITHUB_TOKEN=ghp_xxx`（需 `public_repo` 权限） |
| Token 无权限 | 重新生成 PAT：GitHub Settings → Developer settings → Fine-grained tokens → Issues Read/Write |
| API 提交失败 | markdown 已保存到本地，可手动提交到对应仓库 issues 页面 |
| flaggems 代码路径未知 | 从 context.yaml `environment.flaggems_code_path` 读取，或用 `--flaggems-code-path` 指定 |
| gems.txt 不存在 | 服务未启动或 FlagGems 未启用，issue 中标注 "gems.txt not found" |
| 无法定位问题算子 | 手动在 `--disabled-ops` 参数中指定 |
