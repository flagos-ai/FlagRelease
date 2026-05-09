---
name: flagos-operator-replacement
description: 算子替换与优化工具，支持被动排除（评测报错）和主动渐进排除搜索优化，适配 plugin 环境变量控制和非 plugin Layer 1-4 分层降级，算子列表自动发现，全自动搜索编排，反向二分搜索 + 框架验证 + 排序校验
version: 6.0.0
license: internal
triggers:
  - operator replacement
  - replace operator
  - 算子替换
  - gems replace
  - 算子优化
  - operator optimize
depends_on: []
provides:
  - operator_replacement.replaced_operators
  - operator_replacement.replacement_mode
  - operator_replacement.final_gems_txt
  - operator_replacement.config_file_path
  - operator_replacement.available_ops
  - operator_replacement.rollback_info
  - optimization.target_ratio
  - optimization.current_ratio
  - optimization.enabled_ops
  - optimization.disabled_ops
  - optimization.operator_config_path
  - optimization.search_log
---

# 算子替换与优化 Skill

独立工具，可在任何阶段按需调用。支持两种模式：

1. **被动排除模式**：根据评测报错信息排除问题算子（沿用 Layer 1-4 分层降级）
2. **主动优化模式**：渐进排除搜索，使 FlagOS 性能 ≥ 目标比率

**工具脚本**（已由 setup_workspace.sh 部署到容器）:
- `operator_optimizer.py` — 算子优化器（渐进排除搜索、算子列表自动发现、映射表生成）
- `operator_search.py` — 搜索编排（完整的 next→配置→重启→benchmark→update 自动循环，支持 plugin/非plugin）
- `apply_op_config.py` — Plugin 场景：生成算子替换环境变量 JSON（内联前缀方式）

**核心设计**：FlagGems / vllm-plugin-FL 处于持续迭代中，本 skill 不硬编码任何特定 API，而是根据 `pre-service-inspection` 探测到的 `flaggems_capabilities` 自动选择最优操作方式，逐层降级保证稳定性。

**强制约束**：不在 `flaggems_enable_oplist.txt` 中的算子必须被显式关闭，无论 plugin 还是非 plugin 场景。init 阶段通过 `--registered-ops` 传入 FlagGems 完整注册算子列表，黑名单计算时以注册表为基准，确保注册表中有但 oplist 中没有的算子也被加入 blacklist。

---

# 上下文集成

## 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
gpu:
  vendor: <来自 container-preparation>
execution:
  mode: <来自 pre-service-inspection>
  cmd_prefix: <来自 pre-service-inspection>
inspection:
  flaggems_control: <来自 pre-service-inspection>
  flaggems_logic: <来自 pre-service-inspection>
  flaggems_code_path: <来自 pre-service-inspection>
  flaggems_code_lines: <来自 pre-service-inspection>
  flaggems_capabilities: <来自 pre-service-inspection>
  vendor_config_path: <来自 pre-service-inspection>
  vllm_plugin_installed: <来自 pre-service-inspection>
  plugin_has_dispatch: <来自 pre-service-inspection>
  gpu_compute_capability: <来自 pre-service-inspection>
  gpu_arch: <来自 pre-service-inspection>
service:
  gems_txt_path: <来自 service-startup>
  enable_oplist_path: <来自 service-startup>    # flaggems_enable_oplist.txt 路径（权威算子列表）
  enable_oplist_count: <来自 service-startup>   # 当前生效算子数量
  initial_operator_list: <来自 service-startup>
native_perf:
  output_throughput: <来自 performance-testing>
flaggems_control:
  enable_method: <来自 pre-service-inspection>
  disable_method: <来自 pre-service-inspection>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
operator_replacement:
  replaced_operators: []
  replacement_mode: ""
  final_gems_txt: ""
  config_file_path: ""
  available_ops: []
  rollback_info: ""

optimization:
  target_ratio: 0.8
  current_ratio: <当前性能比>
  enabled_ops: [<最终启用的算子列表>]
  disabled_ops: [<最终禁用的算子列表>]
  operator_config_path: "/flagos-workspace/results/operator_config.json"
  search_log: [<搜索历史>]
```

---

# 环境场景适配

根据 `environment.env_type` 决定算子优化的方式和路径：

## env_type = native

纯 vllm 原生环境，无 FlagGems，**不进入算子优化流程**。

## env_type = vllm_flaggems（环境变量驱动算子控制）

环境检测阶段已自动注入环境变量驱动代码，后续算子控制通过修改控制文件 + 环境变量实现，不再修改源码。

- **算子列表来源**：`environment.flaggems_txt_path` 指向的 txt 文件（由 `flag_gems.enable()` 生成）
- **控制机制**：
  - `USE_FLAGGEMS`：控制 FlagGems 开关（`1`=开启，`0`=关闭）
  - `FLAGGEMS_CONTROL_MODE`：控制算子分支模式
    - `only_enable`：白名单模式，从控制文件读取 `include` 列表
    - `unused`：黑名单模式，从控制文件读取 `unused` 列表
  - `/root/flaggems_ops_control.json`：算子控制配置文件
- **优化方式**：通过 `toggle_flaggems.py --action modify-enable` 写入控制文件（已注入场景）或修改源码（未注入兜底）

**控制文件格式**：
```json
// only_enable 模式（白名单）
{"include": ["addmm", "bmm", "mm", "softmax"]}

// unused 模式（黑名单）
{"unused": ["softmax", "layer_norm"]}

// 全量开启
{"unused": [], "include": []}
```

**搜索循环**：
```
写控制文件(调整算子) → 重启服务 → benchmark → 读取 txt 确认生效算子 → 判断是否达标
```

**脚本调用示例**：
```bash
# 只启用指定算子（写入控制文件 include 列表）
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py \
    --action modify-enable --enabled-ops 'addmm,mm,bmm,softmax' --json"

# 禁用指定算子（写入控制文件 unused 列表）
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py \
    --action modify-enable --disabled-ops 'softmax,layer_norm' --json"
```

**`operator_search.py` 集成**：`_apply_non_plugin_config` 检测到已注入代码后，直接调用 `_apply_via_control_file()` 写控制文件 + 设环境变量，无需修改源码。未注入时降级为 Layer 1-4 策略。

## env_type = vllm_plugin_flaggems（环境变量算子控制）

- **算子列表来源**：`/tmp/flaggems_enable_oplist.txt`
- **优化方式**：环境变量黑白名单（已有逻辑不变）
- 详见下方"Plugin 场景的两层替换架构"

---

# GPU 架构预检

在开始算子替换之前，检查 GPU 架构信息（来自 inspect_env.py 的 `gpu_compute_capability` 和 `gpu_arch`）。

| GPU 架构 | Compute Capability | 已知限制 |
|----------|-------------------|----------|
| sm_80 (A100) | 8.0 | FlagGems 完整支持 |
| sm_89 (L40S/4090) | 8.9 | 部分算子可能不支持 |
| sm_90 (H100/H800) | 9.0 | DeepGemm 可用 |
| sm_70 (V100) | 7.0 | Triton 支持有限，多数算子不可用 |

**如果 `gpu_arch` 为 sm_70 或更低，警告用户 FlagGems 支持有限，算子问题可能较多。**

---

# Plugin 场景的两层替换架构

当 `vllm_plugin_installed=true` 且 `plugin_has_dispatch=true` 时，算子替换通过**环境变量**控制两个独立层：

```
┌─────────────────────────────────────────┐
│ OOT 层（5 个高层融合算子）               │
│   silu_and_mul, rms_norm,               │
│   rotary_embedding, fused_moe,          │
│   attention_backend                     │
│   控制: VLLM_FL_OOT_BLACKLIST           │
├─────────────────────────────────────────┤
│ flag_gems 层（几十个 torch 底层算子）     │
│   addmm, mm, softmax, cos, sin, ...    │
│   控制: VLLM_FL_FLAGOS_BLACKLIST        │
└─────────────────────────────────────────┘
```

> **注意**：主工作流中算子优化仅在 flag_gems 层（gems.txt 范围）进行渐进排除搜索，不单独排查 OOT 层。OOT 环境变量仅作为手动调试工具保留。

## 环境变量控制方式

服务启动时使用**内联环境变量**前缀方式（`VAR=val cmd`），无状态、无残留：

```bash
# 关闭所有 FlagGems（= native 模式）
USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false vllm serve ...

# 全量启用 FlagGems
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true vllm serve ...

# 禁用指定的 torch 底层算子（黑名单，FlagGems < 4.2.1rc0）
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_FLAGOS_BLACKLIST="softmax,layer_norm" vllm serve ...

# 只启用指定的 torch 底层算子（白名单，FlagGems >= 4.2.1rc0，优先使用）
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_FLAGOS_WHITELIST="addmm,mm,bmm" vllm serve ...

# 禁用指定的 OOT 高层算子（与白名单/黑名单独立）
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_OOT_BLACKLIST="fused_moe" vllm serve ...

# 指定单个算子使用 vendor 实现
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_PER_OP="rms_norm=vendor;attention_backend=vendor" vllm serve ...
```

**重要**：环境变量设置后需重启服务才生效。重启后 FlagGems 运行时会自动重新生成算子列表 txt 文件。

## 算子控制方式总结

| 场景 | 控制方式 | txt 文件角色 |
|------|----------|-------------|
| Plugin | `VLLM_FL_*` 环境变量 | **权威来源**（验证生效 + 读取实际算子列表） |
| Plugin (白名单) | `VLLM_FL_FLAGOS_WHITELIST`（FlagGems >= 4.2.1rc0 自动启用） | 同上 |
| 非 plugin (yaml_config) | YAML exclude 配置文件 | **权威来源**（验证生效 + 读取实际算子列表） |
| 非 plugin (only_enable) | `flag_gems.only_enable(include=[...])` | **权威来源**（验证生效 + 读取实际算子列表） |
| 非 plugin (enable_unused) | `flag_gems.enable(unused=[...])` | **权威来源**（验证生效 + 读取实际算子列表） |
| 非 plugin (兜底) | 直接写 txt 文件 | 读写 |

**核心原则**：`/tmp/flaggems_enable_oplist.txt` 是运行时自动生成的**唯一权威算子列表**。每次服务启动后 FlagGems 会重新生成此文件，内容反映 blacklist/whitelist 等配置生效后的实际算子集合。所有算子替换、搜索、对比操作必须以此文件为准，而非 API 枚举或静态文件。

## apply_op_config.py 使用方式

Plugin 场景专用工具，输出环境变量 JSON（含 `env_vars` 字典和 `env_inline` 内联前缀字符串）：

```bash
# Native 模式
${CMD_PREFIX} python3 /flagos-workspace/scripts/apply_op_config.py --mode native

# Full 模式
${CMD_PREFIX} python3 /flagos-workspace/scripts/apply_op_config.py --mode full

# 自定义 blacklist
${CMD_PREFIX} python3 /flagos-workspace/scripts/apply_op_config.py --mode custom \
  --oot-blacklist "fused_moe" \
  --flagos-blacklist "softmax,layer_norm"

# 从状态文件生成
${CMD_PREFIX} python3 /flagos-workspace/scripts/apply_op_config.py \
  --from-state /flagos-workspace/results/operator_config.json
```

输出示例：
```json
{
  "env_vars": {"USE_FLAGGEMS": "1", "VLLM_FL_PREFER_ENABLED": "true", "VLLM_FL_FLAGOS_BLACKLIST": "softmax,layer_norm"},
  "env_inline": "USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_FLAGOS_BLACKLIST=softmax,layer_norm"
}
```

在启动命令前使用内联前缀：
```bash
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_FLAGOS_BLACKLIST=softmax,layer_norm vllm serve ...
```

---

# 运行时算子名映射

FlagGems 注册的算子名（运行时函数名）与 PyTorch aten 算子名不完全一致。`operator_optimizer.py mapping` 子命令可以生成完整映射。

常见不一致项：

| 运行时函数名 | aten 算子名 | 说明 |
|-------------|------------|------|
| `arange_start` | `arange.start` | 点号变下划线 |
| `arange_start_step` | `arange.start_step` | 同上 |
| `add_scalar` | `add.Scalar` | 大小写 + 点号 |
| `fill_scalar_` | `fill_.Scalar` | 下划线位置不同 |
| `sort_stable` | `sort.stable` | 点号变下划线 |
| `to_copy` | `_to_copy` | 前缀下划线 |

```bash
# 生成映射表
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py mapping \
  --output /flagos-workspace/results/op_mapping.json
```

---

# 已知问题模式库

从实战中沉淀的 5 个高频问题模式，优先检查：

## 模式 1：SM 架构不支持

**症状**：`CUDA error: no kernel image is available for execution on the device`
**原因**：Triton kernel 未编译对应 SM 架构
**修复**：禁用报错算子，检查 `gpu_arch` 是否在 FlagGems 支持列表中

## 模式 2：算子参数不匹配

**症状**：`RuntimeError: xxx() got an unexpected keyword argument`
**原因**：FlagGems 实现的算子签名与 PyTorch 版本不一致
**修复**：禁用该算子

## 模式 3：精度问题导致 NaN

**症状**：输出中出现 NaN 或 Inf，精度评测失败
**原因**：FlagGems 算子在特定输入下精度不足
**修复**：禁用该算子，通常涉及 `softmax`、`layer_norm`、`rms_norm`

## 模式 4：DeepGemm 兼容性

**症状**：`VLLM_USE_DEEP_GEMM=1` 时启动崩溃
**原因**：DeepGemm 与某些 FlagGems 算子冲突
**修复**：设置 `VLLM_USE_DEEP_GEMM=0` 或禁用冲突算子

## 模式 5：dispatch 层遗漏

**症状**：gems.txt 中已移除算子但仍被调用
**原因**：vllm_fl dispatch 层有独立的算子注册，未同步修改
**修复**：同时修改 flag_gems 层和 dispatch 层（见两层替换架构）

---

# 快速诊断（搜索前置步骤）

**在进入搜索循环之前，先用 `diagnose_ops.py` 快速定位问题算子**，避免盲搜。

**工具脚本**（已由 setup_workspace.sh 部署到容器）:
- `diagnose_ops.py` — 三场景快速定位（崩溃日志解析、精度分组、性能热点预扫描）

## 场景 1：启动崩溃 → 日志自动解析（O(1) 定位）

FlagOS 模式启动失败时，**先解析崩溃日志再决定下一步**，不要直接进搜索：

```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/diagnose_ops.py crash-log \
  --log-path /flagos-workspace/logs/startup_flagos.log \
  --ops-file /flagos-workspace/results/ops_list.json \
  --json
```

输出示例：
```json
{
  "crashed_ops": ["softmax"],
  "evidence": [{"op": "softmax", "line_start": 142, "error_type": "sm_unsupported", "error_message": "CUDA error: no kernel image..."}],
  "suggestion": "建议禁用以下算子后重启: softmax"
}
```

**决策逻辑**：
- `crashed_ops` 非空 → 直接禁用这些算子 → 重启服务 → 不进搜索
- `crashed_ops` 为空但有 evidence → 人工查看日志
- 无 evidence → 非算子问题，检查环境配置

## 场景 2：精度不达标 → 逐组禁用测试（达标即停）

精度评测不通过时，**逐组禁用测试**，每次禁用一组算子（其余全开），快速定位问题组：

```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/diagnose_ops.py accuracy-groups \
  --ops-file /flagos-workspace/results/ops_list.json \
  --plugin-mode \
  --json
```

输出每组的 test_env 配置（含 `env_inline`），测试流程：

```
1. baseline: 全量启用（V2 配置）→ 已有步骤4的 V2 精度数据
2. 逐组禁用: 每次禁用一组，其余全开 → 重启服务 → fast_gpqa.py 评测
3. 禁用某组后精度恢复（下降 ≤5%）→ 该组有问题，达标即停
4. 问题组内逐个算子排查（可选，缩小禁用范围）
```

**达标即停**：任意一轮禁用后精度下降 ≤5% 即标记 accuracy_ok=true 并停止排查，不继续测试剩余组。主工作流中上限 3 轮（超限标记不合格进入下一步）

## 场景 3：性能不达标 → Profiling 预扫描（缩小搜索范围）

某用例或并发级别性能低于 80% 时，**先做 profiling 看哪些算子最慢**，再针对性搜索：

```bash
# 方式 A：配置 profiler 目录后采集
# 启动服务时加 VLLM_TORCH_PROFILER_DIR=/flagos-workspace/traces/profiler
${CMD_PREFIX} python3 /flagos-workspace/scripts/diagnose_ops.py profile \
  --profiler-dir /flagos-workspace/traces/profiler \
  --port $PORT --model-name $MODEL_NAME --json

# 方式 B：利用 vLLM 的 /start_profile API 自动采集
${CMD_PREFIX} python3 /flagos-workspace/scripts/diagnose_ops.py profile \
  --port $PORT --model-name $MODEL_NAME --json
```

输出示例：
```json
{
  "method": "torch_profiler",
  "hotspots": [
    {"op": "softmax", "avg_ms": 12.3, "calls": 1280, "total_ms": 15744},
    {"op": "rms_norm", "avg_ms": 3.1, "calls": 640, "total_ms": 1984}
  ],
  "suggestion": "性能热点 Top3: softmax(15744ms, 1280次); rms_norm(1984ms, 640次)。建议优先搜索这些算子"
}
```

**决策逻辑**：
- 有热点 → 只搜 hotspot 算子，跳过大部分搜索轮次
- 无法采集 profiler → 按 setup_instructions 配置后重试，或 fallback 到渐进排除搜索

## 诊断与搜索的衔接

```
服务启动 flagos 模式
    │
    ├── 启动失败 → crash-log 解析 → 禁用问题算子 → 重启
    │
    ├── 启动成功 → 精度评测
    │       ├── 不达标 → accuracy-groups 分组定位 → 禁用问题组/算子
    │       └── 通过 → 性能测试
    │               ├── 达标 → 完成
    │               └── 不达标 → profile 预扫描 → 缩小搜索空间 → 渐进排除
```

---

# 两种触发方式

| 触发方式 | 场景 | 模式 |
|----------|------|------|
| 评测报错 | eval-comprehensive 发现算子问题 | **被动排除**（Layer 1-4） |
| 性能不达标 | performance-testing 发现某用例/并发级别 < 80% of V1 | **主动渐进排除搜索** |

---

# 模式一：被动排除（沿用）

## 分层降级策略

根据 `flaggems_capabilities` 从最优方案逐层降级：

```
Layer 1 (最优): YAML 配置文件     ← 需要 capabilities 含 yaml_config
Layer 2:        only_enable API   ← 需要 capabilities 含 only_enable
Layer 3:        enable(unused=)   ← 需要 capabilities 含 enable_unused
Layer 4 (兜底): 源码直接修改       ← 任何版本都能用，但最脆弱
```

| Layer | 所需能力 | 操作方式 | 稳定性 | 回滚难度 |
|-------|----------|----------|--------|----------|
| 1 | `yaml_config` | 写入/修改 YAML 配置文件 | 最高 | 删除文件 |
| 2 | `only_enable` | 修改启动入口的 API 调用 | 高 | 改回原调用 |
| 3 | `enable_unused` | 修改 enable() 的 unused 参数 | 中 | 删除参数 |
| 4 | 无（兜底） | 修改源码中的算子列表 | 低 | 用备份还原 |

## 步骤 1 — 查询当前可用算子

```bash
${CMD_PREFIX} python3 -c "
import json
ops = []
error = ''

try:
    import flag_gems
    flag_gems.enable()

    if hasattr(flag_gems, 'all_registered_ops'):
        ops = list(flag_gems.all_registered_ops())
    elif hasattr(flag_gems, 'all_ops'):
        ops = list(flag_gems.all_ops())
    else:
        try:
            import flag_gems.ops as ops_module
            ops = [name for name in dir(ops_module) if not name.startswith('_')]
        except ImportError:
            error = 'unable to enumerate ops'
except ImportError:
    error = 'flag_gems not installed'
except Exception as e:
    error = str(e)

print(json.dumps({'registered_ops': sorted(ops), 'count': len(ops), 'error': error}, indent=2))
"
```

## 步骤 2 — 确定需要替换的算子

| 来源 | 说明 |
|------|------|
| 评测报错 | 服务端 CUDA error、算子不支持等报错 → 排除问题算子 |
| 已知模式库 | 对照上方 5 个已知模式快速定位 |
| 用户指定 | 用户明确指定需要替换/排除的算子 |
| 日志分析 | `flagos-log-analyzer` 识别出的问题算子 |

## 步骤 3 — 选择操作层级并执行

### Layer 1：YAML 配置文件（capabilities 含 `yaml_config`）

```bash
GEMS_PATH=$(${CMD_PREFIX} python3 -c "
import flag_gems, os
print(os.path.dirname(flag_gems.__file__))
")

${CMD_PREFIX} python3 -c "
import os
config_dir = '${GEMS_PATH}/runtime/backend/_<vendor>'
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, 'enable_configs.yaml')

content = '''exclude:
  - <problem_operator_1>
  - <problem_operator_2>
'''

with open(config_path, 'w') as f:
    f.write(content)
print('配置已写入:', config_path)
"
```

**回滚方式**：`${CMD_PREFIX} rm <config_file_path>`

### Layer 2：only_enable API（capabilities 含 `only_enable` 但无 `yaml_config`）

修改 FlagGems 启动入口代码，将 `enable()` 调用替换为 `only_enable(include=[...])`。

**先备份 → 展示 diff → 确认后执行**。

### Layer 3：enable(unused=) API（capabilities 含 `enable_unused` 但无 `only_enable`）

在现有 `enable()` 调用中添加 `unused` 参数。

### Layer 4：源码直接修改（兜底）

**先完整读取 → 理解结构 → 展示 diff → 确认后执行 → 验证**。

## 步骤 4 — Plugin 场景：通过环境变量同步禁用

如果 `vllm_plugin_installed=true`，使用内联环境变量控制而不是修改代码：

```bash
# 生成环境变量 JSON
${CMD_PREFIX} python3 /flagos-workspace/scripts/apply_op_config.py --mode custom \
  --oot-blacklist "fused_moe" \
  --flagos-blacklist "softmax"

# 在启动命令前使用内联环境变量
USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_OOT_BLACKLIST=fused_moe VLLM_FL_FLAGOS_BLACKLIST=softmax vllm serve ...
```

如果 `vllm_plugin_installed=false`，无需此步骤（Layer 1-4 已直接控制 flag_gems 层）。

## 步骤 5 — 报告替换详情并提醒重启

---

# 模式二：主动渐进排除搜索

## 触发条件

`performance-testing` 对比结果显示 FlagOS 性能在某个用例或并发级别 < 80% of V1。

## 搜索策略

**渐进排除**（默认策略 progressive，最多 3 轮）：

```
基于算子功能的先验知识，将算子按性能影响力分为 high/medium/low 三级。
逐轮排除，达标即停：

Round 1: 排除 high 风险算子（matmul/linear/softmax/rms_norm 等 ~8 个）
         → benchmark → 达标？→ 结束
         → 不达标？→ Round 2

Round 2: 追加排除 medium 风险算子（embedding/gelu/silu 等 ~10 个）
         → benchmark → 达标？→ 结束
         → 不达标？→ Round 3

Round 3: 追加排除 low 风险算子（copy_/add/mul 等基础运算）
         → benchmark → 达标？→ 结束
         → 不达标？→ 问题不在算子层面，标记失败

Plugin 场景：通过 VLLM_FL_FLAGOS_BLACKLIST 环境变量控制
非 plugin 场景：通过 Layer 1-4 策略控制
每轮重启后读取 txt 文件验证算子变化

备选策略：--search-strategy group（分组二分搜索）或 linear（线性逐个搜索）或 elimination（逐删策略）
```

**为什么用渐进排除而非分组二分？**
- 算子列表通常只有 20-30 个，分组二分轮次过多（最坏 ~22 轮）
- 渐进排除最多 3 轮即可定位问题所在的风险等级
- 达标即停，保留尽可能多的算子

**逐删策略**（elimination）：

```
从 txt 算子列表中按顺序逐个累积禁用，每禁用一个就测试，达标即停。
与 linear 的关键区别：linear 独立测试（禁用→测→恢复），elimination 累积禁用（禁用→测→保持禁用→禁下一个→测）。

适用场景：精度/性能不达标时的暴力削减，不关心具体哪个算子有问题，只要达标即可。
最坏情况：所有算子都禁用仍不达标 → 标记 failed。
不支持反向搜索（--reverse），语义冲突。
```

## 工作流程

### 步骤 O1 — 复制优化器到容器

```bash
docker cp skills/flagos-operator-replacement/tools/operator_optimizer.py \
  $CONTAINER:/flagos-workspace/scripts/
docker cp skills/flagos-operator-replacement/tools/operator_search.py \
  $CONTAINER:/flagos-workspace/scripts/
docker cp skills/flagos-operator-replacement/tools/apply_op_config.py \
  $CONTAINER:/flagos-workspace/scripts/
# 共享模块（上述脚本的依赖）
docker cp skills/shared/env_utils.py $CONTAINER:/flagos-workspace/scripts/
docker cp skills/shared/ops_constants.py $CONTAINER:/flagos-workspace/scripts/
```

### 步骤 O2 — 自动发现并导出算子列表

```bash
# 自动发现算子列表文件并保存
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py discover \
  --save-ops /flagos-workspace/results/ops_list.json
```

如果自动发现失败，回退到 API 枚举：

```bash
${CMD_PREFIX} python3 -c "
import json, flag_gems
flag_gems.enable()
ops = list(flag_gems.all_registered_ops()) if hasattr(flag_gems, 'all_registered_ops') else list(flag_gems.all_ops())
with open('/flagos-workspace/results/ops_list.json', 'w') as f:
    json.dump(sorted(ops), f, indent=2)
print(f'导出 {len(ops)} 个算子')
"
```

### 步骤 O2.5 — [可选] 获取运行时算子列表

如果有运行时 profiling 数据（如 torch.profiler trace），提取实际调用的算子：

```bash
# 运行时算子列表保存为 JSON
${CMD_PREFIX} python3 -c "
import json
# 从 profiler trace 或日志中提取
runtime_ops = [...]  # 实际调用的算子
with open('/flagos-workspace/results/runtime_ops.json', 'w') as f:
    json.dump(runtime_ops, f, indent=2)
"
```

### 步骤 O3 — 初始化优化器

```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py init \
  --ops-file /flagos-workspace/results/ops_list.json \
  --native-throughput <native_perf.output_throughput> \
  --native-benchmark /flagos-workspace/results/v1_benchmark.json \
  --target-ratio 0.8
```

Plugin 场景加 `--plugin-mode`：
```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py init \
  --ops-file /flagos-workspace/results/ops_list.json \
  --native-throughput <native_perf.output_throughput> \
  --native-benchmark /flagos-workspace/results/v1_benchmark.json \
  --target-ratio 0.8 \
  --plugin-mode
```

**备选：使用分组二分搜索**：
```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py init \
  --ops-file /flagos-workspace/results/ops_list.json \
  --native-throughput <native_perf.output_throughput> \
  --native-benchmark /flagos-workspace/results/v1_benchmark.json \
  --target-ratio 0.8 \
  --search-strategy group
```

**备选：使用逐删策略**（累积禁用算子直到达标）：
```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py init \
  --ops-file /flagos-workspace/results/ops_list.json \
  --native-throughput <native_perf.output_throughput> \
  --native-benchmark /flagos-workspace/results/v1_benchmark.json \
  --target-ratio 0.8 \
  --search-strategy elimination
```

### 步骤 O4 — 运行搜索循环

**推荐方式：使用 operator_search.py 全自动搜索**（减少 Claude Code 思考开销）：

**Plugin 场景**：
```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_search.py run \
  --state-path /flagos-workspace/results/operator_config.json \
  --perf-config /flagos-workspace/scripts/config/perf_config.yaml \
  --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh" \
  --plugin-mode \
  --max-rounds 3
```

**非 Plugin 场景**：
```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_search.py run \
  --state-path /flagos-workspace/results/operator_config.json \
  --perf-config /flagos-workspace/scripts/config/perf_config.yaml \
  --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh" \
  --capabilities "yaml_config,only_enable" \
  --gems-txt-path ${GEMS_TXT_PATH} \
  --max-rounds 3
```

搜索阶段每轮 benchmark **始终使用 quick**（只跑 `4k_input_1k_output` + max，`num_prompts=concurrency`），无需配置。quick 足以判断单算子对性能的影响。

脚本自动完成：next→应用算子配置→清除Triton cache→重启服务→验证 txt→quick benchmark→更新结果→循环。

**备选方式：手动逐步搜索**（需要更细粒度控制时使用）：

```
1. 获取下一步操作:
   ${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py next

2. 根据返回的 action 和 env_vars/env_inline，应用算子配置:
   - Plugin: 使用 env_inline 内联前缀启动
   - 非 plugin: 通过 Layer 1-4 策略

3. 清除 Triton cache + 重启服务（plugin 场景使用内联 env vars）

4. 运行快速 benchmark:
   ${CMD_PREFIX} python3 /flagos-workspace/scripts/benchmark_runner.py \
     --config /flagos-workspace/scripts/config/perf_config.yaml \
     --quick --output-name optimize_step_N --output-dir /flagos-workspace/results/

5. 更新优化器:
   ${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py update \
     --op-name <名称> --throughputs '{"64":900}' \
     --native-throughput <native_perf.output_throughput>

6. 检查状态，继续或结束
```

### 步骤 O5 — 生成优化报告

```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py report
```

### 步骤 O6 — 应用最终配置

使用优化器输出的最终 `enabled_ops` 列表，通过 Layer 1-4 策略应用配置。

Plugin 场景需同步修改 dispatch 层。

### 步骤 O7 — 验证最终性能

重启服务后运行完整 benchmark 验证优化后性能。

## 反向二分搜索（--reverse）

当全量 FlagGems 性能极低（如 <60%）时，**推荐使用反向搜索**。

**正向搜索**（默认）：从全量启用出发，逐组禁用 → 定位性能拖慢的组 → 组内二分禁用
**反向搜索**（--reverse）：从全禁用出发（= Native 性能），逐组启用 → 定位引入性能下降的组 → 组内二分启用

### 适用场景

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 性能 60-80% | 正向搜索 | 少量算子拖慢，逐步禁用高效 |
| 性能 <60% 或大量注册算子 | **反向搜索** | dispatch 注册干扰严重，正向每轮变化小定位困难 |
| 运行时执行算子远少于注册算子 | **反向搜索** | 问题来自注册开销而非执行慢 |

### 使用方式

初始化时加 `--reverse`：

```bash
${CMD_PREFIX} python3 /flagos-workspace/scripts/operator_optimizer.py init \
  --ops-file /flagos-workspace/results/ops_list.json \
  --native-throughput <native_perf.output_throughput> \
  --native-benchmark /flagos-workspace/results/v1_benchmark.json \
  --target-ratio 0.8 \
  --plugin-mode \
  --reverse
```

搜索编排无需改动，`operator_search.py` 自动读取状态中的 `search_direction` 字段。

### 反向搜索流程

```
1. 初始化: enabled=[], disabled=全部算子 (性能 ≈ Native)
2. 逐组启用: 将 compute 组全部启用 → benchmark
   ├── 达标(≥80%) → 该组安全，保留启用，测试下一组
   └── 不达标 → 该组有问题算子 → 整组回退 → 组内二分
3. 组内二分: 启用前半 → benchmark
   ├── 达标 → 前半安全保留，继续测试后半
   └── 不达标 → 前半有问题 → 回退前半 → 在前半内继续二分
4. 定位到单个问题算子 → 保持禁用，组内其余启用
```

### MetaX 实战参考

MetaX C500 + Qwen3-8B：309 个注册算子仅 26 个运行时执行，全量 FlagGems 性能仅 60%。反向搜索 11 轮定位到 `mm_out` 单个问题算子（未被调用但 dispatch 注册干扰原生 mm），Optimized 平均 96.1%。

## 搜索前框架验证（仅 Plugin 模式）

`operator_search.py run` 在搜索循环开始前自动执行框架验证：

1. 以 `USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false` 启动服务（禁用所有算子，仅保留 plugin 框架）
2. 运行 quick benchmark
3. 与 native_throughput 对比

| ratio | 结论 | 行为 |
|-------|------|------|
| ≥95% | 框架零开销 | PASS，正常搜索 |
| 80-95% | 框架轻微开销 | WARNING，搜索继续但摘要标注 |
| <80% | 框架严重问题 | ERROR，建议先排查 plugin 再搜索算子（不中断） |

预检结果保存在 `results/preflight_framework.json`。

## 排序一致性约束

**二分搜索全程必须使用同一排序的算子列表。** 初始化时每组的算子列表会被排序并固化到 state 中，后续每次 `next` 操作都会校验当前组内算子排序与初始化时一致。

如果检测到排序不一致（如外部修改了 state 文件），会抛出 `ValueError` 并中止搜索，防止产生不可比较的结果。

## 搜索限制

- 渐进排除搜索：最多 3 轮（high → medium → low）
- 线性搜索：遍历搜索范围内所有算子一轮
- **主工作流中上限 3 轮**：步骤3精度/步骤4性能的算子优化均限 3 轮，超限标记不合格进入下一步
- 每轮保存进度，支持断点续搜

---

# 写入 context.yaml

```yaml
operator_replacement:
  replaced_operators:
    - name: "softmax"
      reason: "optimization: ratio 95% without it"
      action: "disabled"
  replacement_mode: "plugin_env"      # 或 yaml_config / only_enable / source_edit
  final_gems_txt: "/path/to/gems.txt"
  config_file_path: "/path/to/enable_configs.yaml"
  env_vars: {}                        # Plugin 模式下的当前环境变量快照
  available_ops: [...]
  rollback_info: "rm /path/to/enable_configs.yaml"

optimization:
  target_ratio: 0.8
  current_ratio: 0.85
  search_mode: "group"
  search_phase: "done"           # group | done
  plugin_mode: true
  enabled_ops: [<最终启用列表>]
  disabled_ops: [<最终禁用列表>]
  oot_blacklist: []              # OOT 层禁用列表（仅手动调试时使用）
  flagos_blacklist: []            # torch 底层禁用列表
  operator_config_path: "/flagos-workspace/results/operator_config.json"
  search_log:
    - op: "softmax"
      decision: "blacklisted"
      throughput: 950.0
      ratio: 0.95
    - op: "memory"
      decision: "group_disabled"
      throughput: 950.0
      ratio: 0.95
```

---

# 累计替换报告格式

每次算子替换后，必须向用户输出以下格式的累计报告：

```
算子替换累计报告
========================================
已剔除算子 (共 N 个):
  精度问题:
    1. softmax    — 精度评测报错 (CUDA error)
    2. layer_norm — 精度下降 >5% (V1 vs V2 对比)
  性能问题:
    3. fused_moe  — 性能拖慢 (禁用后 +15%)

当前启用算子: 35/38 个
启用列表: [addmm, mm, bmm, cos, sin, ...]

当前禁用算子: 3 个
禁用列表: [softmax, layer_norm, fused_moe]

替换方式: plugin_env / yaml_config / only_enable / source_edit
V2 (Full) → V3 (Optimized) 性能比: 95.2% of V1 (Native)
========================================
```

**报告规则**：
- 每次替换后累计更新，不是只报告本次
- **分类展示**：精度问题和性能问题分开列出
- 性能原因标注禁用后的提升幅度
- **必须输出完整的启用算子列表和禁用算子列表**
- 标注当前 V3 相对 V1 的性能比

---

# 完成条件

- 替换操作已执行（含 dispatch 层同步）
- **累计替换报告已输出给用户**
- operator_config.json 已保存
- context.yaml 已更新
- 算子调优记录写入独立的 trace 文件：
  - 精度调优 → `traces/05_accuracy_tuning.json`
  - 性能调优 → `traces/07_performance_tuning.json`
- 精度调优耗时更新 `timing.steps.accuracy_tuning`
- 性能调优耗时更新 `timing.steps.performance_tuning`
- **算子列表 txt 备份**（调优完成、服务重启验证通过后保存）：
  - 精度调优完成后：`docker exec $CONTAINER cp /tmp/flaggems_enable_oplist.txt /flagos-workspace/results/accuracy_tuned_oplist.txt`
  - 性能调优完成后：`docker exec $CONTAINER cp /tmp/flaggems_enable_oplist.txt /flagos-workspace/results/final_oplist.txt`
- **精度调优后精度结果保存**（步骤5达标时必须执行）：
  - 评测时直接通过 `--output` 指定输出路径，无需事后 cp：
    - V3 精度文件：`fast_gpqa.py ... --output /flagos-workspace/results/gpqa_flagos_optimized.json`
    - 同时覆盖 V2 精度文件：`docker exec $CONTAINER cp /flagos-workspace/results/gpqa_flagos_optimized.json /flagos-workspace/results/gpqa_flagos.json`
  - 更新 context.yaml：`eval.v3_score` 设为调优后分数，`eval.accuracy_diff` 更新为调优后偏差

---

# 故障排查

| 问题 | 解决方案 |
|------|----------|
| gems.txt 不存在 | 服务可能未启动过，先执行 `flagos-service-startup` |
| 代码路径不存在 | 重新执行 `flagos-pre-service-inspection` 更新路径 |
| 替换后服务仍报错 | 检查是否 dispatch 层未同步修改（模式 5） |
| capabilities 为空 | FlagGems 版本过旧，将自动降级到 Layer 4（源码修改） |
| 贪心搜索中途服务挂掉 | 保存进度 → 恢复上一个可用配置 → 支持断点继续 |
| 优化后仍不达标 | 检查是否有硬件限制（gpu_arch），报告给用户 |
| YAML 配置写入后不生效 | 确认 FlagGems 启动时使用了 `resolve_user_setting()` |
| 运行时算子名不匹配 | 使用 `mapping` 子命令生成映射表，确认名称对应关系 |
| sm_70/sm_75 大量算子失败 | GPU 架构过旧，建议减少 FlagGems 算子使用范围 |

---

# 失败恢复

1. **算子优化中途失败**：`operator_optimizer.py` 自动保存进度到 `operator_config.json`
2. **恢复搜索**：下次调用 `next` 自动从上次位置继续
3. **回退到可用配置**：应用 `operator_config.json` 中 `enabled_ops` 的上一个快照
4. **dispatch 层回退**：从 `.bak` 备份文件还原

---

## 编排层指令（步骤7 性能算子调优 — 固化决策）

**触发条件**：`workflow.performance_ok = false` 且 `env_type ≠ native`
**跳过条件**：`performance_ok = true`（不触发时显示已完成）

固化选择：
- 搜索策略：`--search-strategy elimination`（逐删，不使用 group/linear）
- 必须通过 `operator_search.py run` 一次性执行完整自动循环，禁止手动拼凑 toggle→restart→benchmark 循环
- V3 验证 benchmark 使用 `--strategy quick`，output-name 为 `flagos_optimized`
- 达标即停，不继续优化

执行顺序（固定）：
1. `operator_optimizer.py discover` → 发现算子列表
2. 收集 FlagGems 注册算子列表 → `registered_ops.json`
3. `operator_optimizer.py init --search-strategy elimination --target-ratio 0.8`
4. `operator_search.py run`（容器内全自动循环）
5. 搜索完成 → `benchmark_runner.py --strategy quick --output-name flagos_optimized`
6. `performance_compare.py` 对比 V3/V1（含 `--flagos-initial` 和 `--flagos-optimized`）
7. V3/V1 ratio ≥ 80% → `performance_ok=true`；否则 `performance_ok=false`

执行后必须完成：
- 更新 `context.yaml` 的 `optimization` 和 `operator_replacement` 字段
- 写入 `traces/07_performance_tuning.json`
- 保存算子列表：`docker exec $CONTAINER cp /tmp/flaggems_enable_oplist.txt /flagos-workspace/results/final_oplist.txt`
- 更新报告：`docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"`

### 步骤7完成后：算子配置固化

**强制执行**：无论是否触发了步骤5/7，只要 `env_type` 不是 `native`，都必须执行：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/persist_op_config.py --auto --verify"
```

`--verify` 会重启服务检查运行时算子数量是否与预期一致。验证失败时标记 `workflow.config_persisted=false`，继续发布但标记为私有。
