---
name: flagos-component-install
description: FlagOS 生态组件统一安装/升级/卸载，支持 FlagGems、FlagTree
version: 2.0.0
triggers:
  - 组件安装
  - install component
  - 安装 FlagGems
  - 安装 FlagTree
  - 升级 FlagGems
  - 升级 FlagTree
  - flag upgrade
  - flag install
depends_on: []
provides:
  - component_install.component
  - component_install.action
  - component_install.success
  - component_install.previous_version
  - component_install.current_version
  - flagtree.installed
  - flagtree.version
---

# 组件安装 Skill

通过 `install_component.py` 统一管理 FlagOS 生态组件的安装/升级/卸载。

**工具脚本**（已由 setup_workspace.sh 部署到容器）：
- `install_component.py` — 统一入口
- `install_flagtree.sh` — FlagTree 专用安装脚本（由 install_component.py 内部调用）

**支持组件**：

| 组件 | 安装方式 | 说明 |
|------|---------|------|
| `flaggems` | pip install（默认最新版） | FlagGems 算子库 |
| `flagtree` | pip wheel / 源码编译（委托 install_flagtree.sh） | 统一 Triton 编译器 |

---

# 上下文集成

## 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
execution:
  cmd_prefix: <来自 pre-service-inspection>
inspection:
  flag_packages: <来自 pre-service-inspection>
gpu:
  vendor: <来自 container-preparation，FlagTree 安装时用于自动选择后端>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
component_install:
  component: "<flaggems|flagtree>"
  action: "<install|uninstall|upgrade>"
  previous_version: "<old>"
  current_version: "<new>"
  install_method: "<pip|flagtree_pip|flagtree_source>"
  success: true|false
  timestamp: "<ISO>"

# FlagTree 安装后同步更新
flagtree:
  installed: true|false
  version: "<version>"
  triton_version: "<version>"
  install_method: "pip|source"
  backend: "<vendor>"
environment:
  has_flagtree: true|false
```

---

# 工作流程

## 步骤 1 — 查看当前版本

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH pip show flag-gems 2>/dev/null"
```

FlagTree 状态：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py --component flagtree --action verify --json"
```

## 步骤 2 — 停止服务（如在运行）

```bash
docker exec $CONTAINER bash -c "pkill -f 'vllm\|sglang\|flagscale' 2>/dev/null; sleep 3"
```

## 步骤 3 — 执行安装/升级

### FlagGems 安装

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flaggems --action install --json"
```

指定版本：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flaggems --action install --version 4.2.1rc0 --json"
```

### FlagGems 升级

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flaggems --action upgrade --json"
```

### FlagTree 安装

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flagtree --action install --vendor nvidia --json"
```

`--vendor` 根据 `gpu.vendor` 自动选择，支持：nvidia, iluvatar, mthreads, metax, ascend, tsingmicro, hcu, enflame, sunrise, amd, xpu。

源码编译（无预编译包时）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flagtree --action install --vendor ascend --source --json"
```

### FlagTree 卸载

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flagtree --action uninstall --json"
```

卸载时自动恢复原版 triton（从备份恢复）。

### FlagTree 验证

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flagtree --action verify --json"
```

## 步骤 4 — 验证安装结果

FlagGems 验证：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/install_component.py \
    --component flaggems --action verify --json"
```

输出包含版本和 API 兼容性信息（has_enable, has_only_enable, enable_params）。

## 步骤 5 — 写入 context.yaml

从 JSON 输出中提取字段，写入 `component_install`、`flagtree`、`environment` 等字段。

---

# 完成条件

- 目标组件已确定
- 安装/升级操作已完成
- 版本变化已记录
- API 兼容性已检查（FlagGems）
- context.yaml 已更新
- 对应 trace 文件已写入

---

# 故障排查

| 问题 | 解决方案 |
|------|----------|
| pip install flag-gems 失败 | 检查网络连通性和 pip 源配置 |
| FlagTree 安装后 import triton 失败 | `install_flagtree.sh verify` 检查，`uninstall` 恢复原版 |
| FlagTree 无预编译包 | 使用 `--source` 源码编译 |
| 版本冲突 | 先 `pip uninstall` 再重新安装 |
