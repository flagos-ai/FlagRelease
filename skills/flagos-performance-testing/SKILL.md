---
name: flagos-performance-testing
description: 两档性能基准测试（quick 只跑 4k_input_1k_output 并发 64 / comprehensive 全量用例全并发），标准 markdown 输出格式
version: 8.0.0
triggers:
  - 性能测试
  - benchmark
  - vllm bench
  - 吞吐量测试
  - performance test
depends_on:
  - flagos-service-startup
next_skill: null
provides:
  - native_perf.result_path
  - native_perf.output_throughput
  - native_perf.total_throughput
  - flagos_full_perf.result_path
  - flagos_optimized_perf.result_path
---

# 性能测试 Skill

支持三版自动化性能测试：V1 (Native) → V2 (Full FlagGems) → V3 (Optimized FlagGems)（如需优化），标准 markdown 表格输出。

**两档测试策略**（`--strategy` 参数）：

| strategy | 含义 | 用例选择 | 并发行为 |
|----------|------|----------|----------|
| `quick` | 快速验证（默认） | 只跑 `4k_input_1k_output` | 预热 2 请求 + 固定并发 64 |
| `comprehensive` | 全跑 | 所有 enabled 用例 | 所有并发全跑（1~256） |

**策略选择**：主流程步骤6自动使用 `quick`，独立调用全量测试使用 `comprehensive`。

**三版结果文件**：
- `native_performance.json` — V1 (Native，无 FlagGems)
- `flagos_performance.json` — V2 (Full FlagGems，全量算子)
- `flagos_optimized.json` — V3 (Optimized FlagGems，≥80% 组合，如需优化才产出)

**工具脚本**（已由 setup_workspace.sh 部署到容器）:
- `benchmark_runner.py` — 性能测试（`--strategy quick/comprehensive`）
- `performance_compare.py` — 性能对比（`--format markdown` 标准表格输出）

**对比规则（钉死）**：性能对比逐并发级别进行。quick 模式下只有一个数据点（4k_input_1k_output × 64），comprehensive 模式下每个 `(test_case, concurrency)` 组合独立计算 ratio。达标条件：所有数据点的 `min(output_ratio, total_ratio)` 均 ≥ 80%。

---

## 强制约束

**只能通过 `benchmark_runner.py` 执行性能测试**，禁止直接运行 `vllm bench serve`。

**启动前互斥检查**：性能测试启动前，必须确认没有正在运行的精度评测进程。并发执行会互相抢占 GPU 资源，导致结果不可信。

```bash
# 检查是否有评测进程在运行（fast_gpqa.py / eval_aime.py / eval_erqa.py / eval_monitor.py）
docker exec $CONTAINER bash -c "pgrep -f 'fast_gpqa\|eval_aime\|eval_erqa\|eval_monitor' && echo 'BLOCKED: 评测进程运行中，等待结束' && exit 1 || echo 'OK: 无评测进程'"
```

如果检测到评测进程，**必须等待其结束后再启动性能测试**，禁止强杀评测进程。

**策略触发规则**：
- 用户说"快速测试"/"走通流程"/"smoke test"/"验证流程"→ `--strategy quick`
- 用户说"完整测试"/"全量"/"所有并发"→ `--strategy comprehensive`
- 默认（或用户说"正常测试"/"标准"）→ `--strategy quick`

---

# Triton Cache 保护

**警告**：在算子替换后重启服务时，Triton JIT cache 可能导致旧的 kernel 被使用。

```bash
# 清除 Triton cache（在每次算子配置变更后）
${CMD_PREFIX} rm -rf ~/.triton/cache/ 2>/dev/null
${CMD_PREFIX} rm -rf /tmp/triton_cache/ 2>/dev/null
```

**何时需要清除**：
- 算子替换后重启服务前
- FlagGems 升级后重启服务前
- 性能测试结果异常时排查

---

# Plugin 场景的算子覆盖率检查

当 `vllm_plugin_installed=true` 时，在性能测试前检查算子覆盖率：

```bash
# 检查 FlagGems 实际覆盖了多少 aten 算子
${CMD_PREFIX} python3 -c "
import json
try:
    import flag_gems
    flag_gems.enable()
    ops = list(flag_gems.all_registered_ops()) if hasattr(flag_gems, 'all_registered_ops') else []
    print(json.dumps({'covered_ops': len(ops), 'ops': sorted(ops)}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"
```

如果覆盖率很低（< 20 个算子），FlagOS 加速效果可能有限，应在报告中注明。

---

# 工作流程

## 核心原则：三版测试 + 按需优化

新工作流在步骤6（快速性能评测）中按固定顺序执行：
1. **V1 Native** — 关闭 FlagGems 的基线性能
2. **V2 FlagGems** — 启用 FlagGems 的性能（使用步骤4精度达标后的算子列表）
3. **V3 Optimized FlagGems** — 仅在 V2 不达标时，通过算子优化找到 ≥80% 的组合

**算子列表必录**：只要 FlagGems 处于启用状态，必须记录算子列表到 ops_list.json，这是算子优化的基础。

最终需要三个结果文件：
1. **native_performance.json** — V1 性能
2. **flagos_performance.json** — V2 性能
3. **flagos_optimized.json** — V3 性能（仅在 V2 不达标时产出）

## 步骤 0：策略确定

策略由流程阶段自动决定，不在此处单独询问：

- **主流程步骤6**：自动使用 `--strategy quick`，无需询问用户
- **独立调用全量性能测试**：使用 `comprehensive`

| 选项 | 说明 | 触发时机 |
|------|------|----------|
| `quick`（默认） | 只跑 4k_input_1k_output 并发 64 | 主流程步骤6自动使用 |
| `comprehensive` | 所有用例，所有并发全跑 | 用户要求"完整测试" |

一旦选定，该阶段内所有性能测试统一使用同一策略。

## 步骤 1：同步配置

从容器内 `/flagos-workspace/shared/context.yaml` 读取服务信息，写入 `/flagos-workspace/scripts/config/perf_config.yaml`。

同时将配置快照保存到 `config/` 目录：
```bash
docker exec $CONTAINER cp /flagos-workspace/scripts/config/perf_config.yaml /flagos-workspace/config/perf_config.yaml
```

## 步骤 2：判断当前 FlagGems 状态

从容器内 `/flagos-workspace/shared/context.yaml` 的 `flaggems_control.integration_type` 和 `inspection` 字段判断当前环境中 FlagGems 是否已启用。

判断依据（按优先级）：
1. `flaggems_control.enable_method` 是否为 `auto`（plugin 自动启用）
2. 环境变量 `USE_FLAGGEMS=1` / `USE_FLAGOS=1`
3. 代码中是否有 `flag_gems.enable()` 被调用
4. 服务启动日志中是否有 FlagGems 相关输出

```
当前状态判定:
  ├── FlagGems 已启用 → 走路径 A（先测 V2）
  └── FlagGems 未启用 → 走路径 B（先测 V1）
```

---

## 步骤 3：运行 V1 基线测试

**前置条件**：关闭 FlagGems，以 native 模式启动服务。

```bash
# 关闭 FlagGems
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action disable"
# 以 native 模式启动服务
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=0 bash /flagos-workspace/scripts/start_service.sh --mode native > /flagos-workspace/logs/startup_native.log 2>&1"
# 等待服务就绪
docker exec $CONTAINER bash -c "bash /flagos-workspace/scripts/wait_for_service.sh --port $PORT --model-name '$MODEL_NAME' --timeout 120 --max-timeout 900 --log-path /flagos-workspace/logs/startup_native.log --mode native"
```

运行 V1 基线 benchmark：

```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH python3 scripts/benchmark_runner.py \
  --config scripts/config/perf_config.yaml \
  --strategy quick \
  --output-name native_performance \
  --output-dir /flagos-workspace/results/ \
  --mode native"
```

**V1 测试完成后，必须停止服务释放 GPU**：
```bash
docker restart $CONTAINER
sleep 5
```

## 步骤 4：启用 FlagGems，切换到 FlagOS 模式

**前置条件**：V1 服务已停止（步骤3末尾的 `docker restart`）。

通过 `toggle_flaggems.py` 启用 FlagGems，重启服务。

**强制规则**：V1 和 V2 必须使用相同的 GPU 配置（`CUDA_VISIBLE_DEVICES` 和 `TP_SIZE`），复用 context.yaml 中首次启动时写入的值，禁止重新检测 GPU。

## 步骤 5：记录算子列表（强制）

**FlagGems 启用状态下，必须先记录算子列表。**

```bash
${CMD_PREFIX} python3 -c "
import json, flag_gems
flag_gems.enable()
ops = list(flag_gems.all_registered_ops()) if hasattr(flag_gems, 'all_registered_ops') else list(flag_gems.all_ops())
with open('/flagos-workspace/results/ops_list.json', 'w') as f:
    json.dump(sorted(ops), f, indent=2)
print(f'已记录 {len(ops)} 个算子到 ops_list.json')
"
```

## 步骤 6：运行 V2 FlagGems 性能测试

```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH python3 scripts/benchmark_runner.py \
  --config scripts/config/perf_config.yaml \
  --strategy quick \
  --output-name flagos_performance \
  --output-dir /flagos-workspace/results/ \
  --mode flagos"
```

## 步骤 7：性能对比（强制执行）

**强制规则**：V1 和 V2 性能测试完成后，必须调用 `performance_compare.py` 生成对比。禁止跳过此步骤或手动计算比值。

```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH python3 scripts/performance_compare.py \
  --native results/native_performance.json \
  --flagos-full results/flagos_performance.json \
  --output results/performance_compare.csv \
  --target-ratio 0.8 \
  --format markdown"
```

**输出解读**：
- 工具会输出对比表格，必须完整保留到报告中
- 返回码 `0`：达标（ratio ≥ 80%），V2 已达标，不存在 V3，跳到步骤 9
- 返回码 `1`：不达标，触发步骤 8

**禁止行为**：不得自行从 JSON 中提取数据手动计算 ratio，必须使用 `performance_compare.py` 的输出。

## 步骤 8：[自动] 触发算子优化

前置条件：`ops_list.json` 已存在（步骤 5 中已记录）。

调用 `flagos-operator-replacement` 算子搜索优化。优化过程中以 `gems.txt`（或 `flaggems_enable_oplist.txt`）中已替换的算子为唯一候选范围。使用 elimination 逐删策略，轮次不限（由算子总数决定），达标即停；全部禁完仍不达标则标记 `workflow.performance_ok: false` 进入下一步。

### 性能不达标时的强制闭环（不可跳过）

ratio < 80% 时，编排层**必须**按以下逻辑执行：

```
IF ratio < 80%:
    # 1. 写 issue log（强制）
    追加写入 logs/issues_performance.log:
      "[时间] V2 | 性能不达标"
      "  详情: V2/V1 ratio=XX% (<80%)"

    # 2. 设置状态
    workflow.performance_ok = false

    # 3. 继续到步骤8发布（不终止流程）

ELSE:
    workflow.performance_ok = true

→ 继续步骤8 发布
```

**禁止行为**：ratio < 80% 时终止流程。必须写入 issue log、标记 `performance_ok=false`，然后继续到发布步骤（`qualified=false` → 私有发布）。

优化完成后重测：

```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH python3 scripts/benchmark_runner.py \
  --config scripts/config/perf_config.yaml \
  --strategy quick \
  --output-name flagos_optimized \
  --output-dir /flagos-workspace/results/ \
  --mode flagos_optimized"
```

## 步骤 9：性能对比 + 报告

```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH python3 scripts/performance_compare.py \
  --native results/native_performance.json \
  --flagos-optimized results/flagos_optimized.json \
  --flagos-full results/flagos_performance.json \
  --output results/performance_compare.csv \
  --target-ratio 0.8 \
  --format markdown"
```

当 V2 已达标（不存在 V3）时，只传 `--flagos-full`，不传 `--flagos-optimized`。

## 步骤 10：写入 context.yaml

写入字段：

```yaml
perf:
  strategy: quick          # 使用的测试策略
  test_case: 4k_input_1k_output
  concurrency: 64
  v1_output_tps: 6598.2
  v1_total_tps: 32991.0
  v2_output_tps: 5178.5
  v2_total_tps: 25892.5
  output_ratio_pct: 78.5
  total_ratio_pct: 78.5
  ratio_pct: 78.5          # min(output_ratio, total_ratio)
  pass: false              # ratio >= 80%?
```

comprehensive 模式下包含 `per_concurrency` 全量列表和 `summary` 统计。

---

## 阶段性反馈格式

每次性能测试完成后，必须向用户输出以下格式的总结：

```
性能测试结果
========================================
模式: native / flagos_full / flagos_optimized
测试用例: 4k_input_1k_output
并发: 64
Output throughput: 1850 tok/s
Total throughput: 5200 tok/s
Native 基线: 6500 tok/s（首次 native 测试时不显示此行）
性能比: 80.0% — 达标(≥80%) / 不达标(<80%)
========================================
```

**反馈规则**：
- Native 模式测试时不显示"性能比"
- FlagOS 模式测试时必须与 Native 基线对比并给出达标/不达标判断
- 不达标时自动提示"建议触发算子优化"

### 性能问题日志写入

V2/V1 ratio < 80% 时，必须追加写入 `logs/issues_performance.log`：

```bash
docker exec $CONTAINER bash -c "cat >> /flagos-workspace/logs/issues_performance.log << 'ISSUE_EOF'
[$(date '+%Y-%m-%d %H:%M:%S')] <版本> | <问题摘要>
  详情: <4k_input_1k_output conc=64, V1 TPS, V2 TPS, ratio>
  操作: <措施，如 operator_search.py 第 N 轮优化>
  结果: <优化后 ratio，达标/不达标>
ISSUE_EOF"
```

记录场景：
- V2/V1 性能对比 ratio < 80%
- 算子搜索优化每轮结果（禁用了哪些算子、优化后的 ratio）
- 最终结论（V2 达标 / V3 达标 / 不达标）

---

## benchmark_runner.py 参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径 |
| `--strategy` | 测试策略：`quick`(只跑4k_input_1k_output并发64,默认) / `comprehensive`(全跑) |
| `--quick` | (向后兼容别名) 等同于 `--strategy quick` |
| `--output-name` | 输出文件名（不含扩展名） |
| `--output-dir` | 输出目录 |
| `--mode` | 测试模式标记 |
| `--test-case` | 运行指定测试用例 |
| `--dry-run` | 仅打印命令不执行 |

**优先级**：`--strategy` > `--quick` > 默认 `quick`

## performance_compare.py 参数

| 参数 | 说明 |
|------|------|
| `--native` | 原生性能 JSON（必填） |
| `--flagos-initial` | FlagOS 初始性能 |
| `--flagos-optimized` | FlagOS 优化后性能 |
| `--flagos-full` | FlagOS 全量算子性能 |
| `--output` | CSV 输出路径 |
| `--target-ratio` | 目标比率（默认 0.8） |
| `--format` | 输出格式: `text`（默认） / `markdown` |

---

## 完成条件

- 测试脚本已在容器中就绪
- **算子列表已记录**（FlagGems 启用时 ops_list.json 必须存在）
- native_performance.json 已生成
- flagos_performance.json 已生成
- 对比结果已生成（performance_compare.csv）
- 性能比率已判断（ratio ≥ 80%，或触发算子优化）
- 性能不达标时，`logs/issues_performance.log` 已追加写入问题记录
- 如触发优化：flagos_optimized.json 已生成
- 最终对比已生成
- context.yaml 已更新
- 配置快照已保存到 `config/perf_config.yaml`
- 对应 trace 文件已写入：
  - `traces/06_quick_performance.json`（V1/V2 性能测试 + 对比）
  - `traces/07_performance_tuning.json`（算子优化记录，仅触发步骤7时）
- `timing.steps.quick_performance` 已更新为本步骤的 `duration_seconds`
- 更新报告：`docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"`

---

## 编排层指令（步骤6 性能评测 — 固化决策）

**前置条件**：步骤4（及5如触发）已完成，当前算子集为精度对齐后的最终集合。

固化选择：
- 主流程步骤6使用 `--strategy quick`（4k_input_1k_output 并发 64 单数据点）
- 性能对比必须通过 `performance_compare.py` 执行，禁止手动计算 ratio
- output-name 标准命名：V1=`native_performance`，V2=`flagos_performance`
- `benchmark_runner.py` 仅接受以下参数：`--config`、`--strategy`、`--output-name`、`--output-dir`、`--mode`、`--test-case`、`--dry-run`。`--quick` 为 `--strategy quick` 的向后兼容别名，优先使用 `--strategy`。禁止传入 `--host`、`--port`、`--model-name`、`--json` 等未定义参数，host/port/model 由 config 文件和 context.yaml 自动提供
- 禁止使用 `pgrep -f benchmark_runner` 轮询等待 benchmark 完成。benchmark_runner.py 是同步脚本，直接等待其返回即可。如必须后台轮询，使用 `pgrep -f '[b]enchmark_runner'` 避免自匹配

执行顺序（固定）：
1. 关闭 flaggems → 启动服务 → benchmark V1 → 停服务
2. 开启 flaggems → 启动服务 → benchmark V2（使用精度调优后的算子集）
3. `performance_compare.py` 对比，ratio ≥ 80%?
   - 达标 → `workflow.performance_ok=true`，进入步骤8
   - 不达标 → 必须按顺序：① 标记 `performance_ok=false` ② issue_reporter.py --type performance-degraded ③ 追加 `logs/issues_performance.log` → 触发步骤7
