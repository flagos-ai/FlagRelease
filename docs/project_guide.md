# FlagOS 自动化框架说明书

> 基于当前 CLAUDE.md + prompts/ 体系（调优后）与 CLAUDE.md.bak（调优前）的对比

---

## 一、当前项目如何使用进行迁移工作

### 1.1 一键启动（推荐）

```bash
bash prompts/run_pipeline.sh <容器名或镜像地址> <模型名> \
  <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> \
  [--verbose]
```

**自动识别规则**：
- 第一参数含 `:` 或 `/`（如 `harbor.baai.ac.cn/flagrelease/qwen3:latest`）→ 镜像模式
- 否则通过 `docker inspect --type=container` 检测 → 容器模式
- 模型路径自动搜索宿主机，未找到则容器内自动下载

**示例**：

```bash
# 已有容器
bash prompts/run_pipeline.sh qwen3-8b-test Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass

# 从镜像创建
bash prompts/run_pipeline.sh harbor.baai.ac.cn/flagrelease/qwen3:latest Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass

# 调试模式（全量终端输出）
bash prompts/run_pipeline.sh qwen3-8b-test Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --verbose
```

**特性**：
- 全程零交互，凭证通过参数传入后自动 export 为环境变量
- 默认精简终端输出（~200 行关键进度），`--verbose` 切换全量输出
- `--permission-mode auto` + `settings.local.json` 白名单，root 用户也能用
- Claude 退出后自动执行诊断、数据同步、缺失文件补生成

### 1.2 手动 Claude CLI

```bash
# 非交互模式
claude -p "容器名: qwen3-8b-test，模型名: Qwen3-8B ..." --permission-mode auto

# 交互模式
claude --permission-mode auto
# 然后粘贴 prompts/auto_v1v2_pipeline.md 中的 Prompt 模板
```

### 1.3 六步全自动流程

```
1 容器准备  → 容器/镜像就绪 + 模型权重搜索/下载 + 工具脚本部署
2 环境检测  → inspect_env.py 场景分类 + FlagGems 集成分析
3 启服务    → V1(native) + V2(flagos) 启动验证
4 精度评测  → V1/V2 GPQA Diamond 对比（偏差阈值 5%）
5 性能评测  → V1/V2 4k1k benchmark 对比（每并发级别 ≥ 80%）
6 自动发布  → Harbor + ModelScope + HuggingFace（qualified 判定公开/私有）
```

**关键规则**：
- 精度/性能不达标 → 记录 issue + 标记 `workflow.xxx_ok=false` → 继续后续步骤 → 最终私有发布
- 流程不可中途终止，唯一终止条件是 Claude API 本身不可用
- V1 = 不开启 FlagGems 的基线版本；V2 = 初始环境的 FlagGems 状态

**环境场景自动分类**：

| env_type | 判定条件 | FlagGems 控制方式 |
|----------|---------|------------------|
| `native` | 无 flaggems | 无 |
| `vllm_flaggems` | 有 flaggems，无 plugin | 代码注释/取消注释 |
| `vllm_plugin_flaggems` | 有 flaggems + plugin | 环境变量 |

### 1.4 产出物结构

```
/data/flagos-workspace/<model>/
├── results/                          # 最终交付物
│   ├── native_performance.json          # V1 性能
│   ├── flagos_performance.json          # V2 性能
│   ├── performance_compare.csv          # 性能对比
│   ├── gpqa_native.json                 # V1 精度
│   ├── gpqa_flagos.json                 # V2 精度
│   └── release_info.json               # 发布结果
│
├── traces/                           # 每步留痕 JSON
│   ├── 01_container_preparation.json
│   ├── 02_environment_inspection.json
│   ├── 03_service_startup.json
│   ├── 04_quick_accuracy.json
│   ├── 05_accuracy_tuning.json
│   ├── 06_quick_performance.json
│   ├── 07_performance_tuning.json
│   └── 08_release.json
│
├── logs/                             # 运行日志
│   ├── pipeline.log                     # 全流程执行记录（tail -f 可跟踪）
│   ├── issues_startup.log               # 服务启动异常
│   ├── issues_accuracy.log              # 精度异常
│   └── issues_performance.log           # 性能不达标
│
└── config/                           # 配置快照
    └── context_snapshot.yaml            # 流程结束时的完整 context
```

### 1.5 终端输出与日志体系

| 层级 | 文件/工具 | 用途 | 受众 |
|------|----------|------|------|
| 终端实时 | `stream_filter.py` 精简输出 | 步骤进度、✓/✗ 结果 | 人（实时观察） |
| 流水线日志 | `logs/pipeline.log` | 全流程概览，`tail -f` 可跟踪 | 人（事后查看） |
| Debug 日志 | `debug.log`（stream_to_debug_log.py 生成） | 全量工具调用、命令、输出 | 人（排查问题） |
| Trace JSON | `traces/*.json` | 单步详细留痕（命令、参数、输出） | 程序/审计 |
| 工作流台账 | `context.yaml` → `workflow_ledger` | 所有步骤的结构化状态 | 编排层/下游 |

### 1.6 异常处理与恢复

**流程中断后**：
1. `run_pipeline.sh` 自动调用 `diagnose_failure.py`，输出中断步骤、错误原因、恢复建议
2. 自动同步容器内数据到宿主机（防止数据丢失）
3. 检查 Harbor 发布是否完成，未完成则自动补执行
4. 新会话启动时自动检测 `workflow.all_done != true`，从中断点恢复

**网络失败**：
- pip install：自动依次尝试阿里云/清华/腾讯镜像（最多 4 次）
- 其他网络操作：自动重试 1 次
- 全部失败直接终止，不询问用户

---

## 二、相比调优前的改进

### 对比基准

| 维度 | 调优前（CLAUDE.md.bak） | 调优后（CLAUDE.md + prompts/） |
|------|------------------------|-------------------------------|
| 流程步骤 | 1-8（8 步） | 1-6（6 步） |
| 版本定义 | V1/V2/V3 | V1/V2 |
| 启动方式 | 手动拼 claude 命令或粘贴 prompt | `run_pipeline.sh` 一键启动 |
| 终端输出 | 全量 stream-json 事件流 | 精简过滤，~200 行关键进度 |
| 日志体系 | 仅 trace JSON | 四层：终端/pipeline.log/debug.log/trace |
| 异常处理 | 算子调优循环（不可控） | 记录 issue + 继续（可预期） |
| 中断恢复 | 人工排查 | 自动诊断 + 断点续跑 |
| 发布流程 | 打包+上传分两步，需确认 | 统一 main.py，零交互 |

### 2.1 流程简化：去除 V3 算子优化

调优前步骤4精度不达标时，会自动调用 `diagnose_ops.py` 进行分组排查 → 禁用问题算子 → 循环重测；步骤6性能不达标时，会调用 `operator_search.py` 最多进行 5 轮搜索优化。这个算子优化循环是流程中最不可控的部分，单次可能耗时数小时且结果不确定。

调优后去除了整个 V3 优化循环。不达标时仅记录 issue、标记 `workflow.xxx_ok=false`，继续后续步骤，最终以 `qualified=false` 私有发布。流程耗时可预期，失败模式清晰。

同时，6打包和7上传合并为统一的6自动发布（`main.py --from-context`），8可选正式评测被移除。

### 2.2 一键启动脚本 `run_pipeline.sh`

调优前没有启动脚本。调优后提供：

- **参数自动识别**：容器名 vs 镜像地址，无需 `--image` 标志
- **模型路径自动搜索**：只需模型名，自动在宿主机搜索权重
- **凭证统一管理**：7 个位置参数一次传入
- **权限自动配置**：`--permission-mode auto`，root 用户也能用
- **流程结束兜底**：Claude 退出后自动诊断、数据同步、缺失文件补生成
- **向后兼容**：旧 `--image` 格式仍可用（打印弃用警告）

### 2.3 实时终端输出过滤 `stream_filter.py`

调优前 Claude 的终端输出是全量 stream-json 事件流，包含大量自言自语（"Let me..."）、探测命令（ls/find/cat）、JSON 结构体，人无法实时跟踪进度。

调优后 `stream_filter.py`：
- 精简模式（默认）：~200 行，只显示 `[步骤1]` 标记、✓/✗ 结果、关键命令
- 终端顶部实时显示 6 步进度状态（⬜→🔵→✅/❌）
- ANSI 颜色：步骤蓝色、成功绿色、失败红色，管道时自动关闭
- `--verbose` 恢复全量输出用于调试

### 2.4 流水线日志 `pipeline.log`

调优前没有面向人的流程执行记录。调优后由 `stream_filter.py` 自动生成：

```
[2026-04-09 15:11:13] [步骤1] 容器准备 — 开始
[2026-04-09 15:12:22] [步骤1] 容器准备 — 完成 (1m 9s)
  结果: 容器 nv_gems_tree 就绪, 8x H20-3e, 工具脚本已部署
```

支持 `tail -f` 实时跟踪，流程首尾有分隔线和耗时统计。

### 2.5 全量 Debug 日志 `stream_to_debug_log.py`

调优前排查问题需要翻阅原始 stream-json。调优后转为人可读格式：

```
[15:30:05] ▶ Bash
  command: docker exec xxx nvidia-smi
  ── result (ok) ──
  ...

[15:31:00] ══ SESSION SUMMARY ══
  duration: 15m 0s
  tool_calls: 47
  cost: $1.23
```

### 2.6 问题日志体系

调优前遇到问题只写入 trace JSON。调优后新增三个专门的 issue log 文件：

| 文件 | 写入时机 |
|------|---------|
| `logs/issues_startup.log` | 服务启动失败、崩溃、超时 |
| `logs/issues_accuracy.log` | V2精度下降 >5% |
| `logs/issues_performance.log` | 任一并发级别 V2/V1 < 80% |

统一格式，支持 `tail -f`：
```
[2026-03-20 16:30:05] V2 | 精度下降超阈值
  详情: V1=68.2%, V2=61.5%, 下降=6.7% (阈值 5%)
  操作: 提交 accuracy-degraded issue, 标记 workflow.accuracy_ok=false
  结果: 继续进入5性能评测
```

### 2.7 中断自动诊断 `diagnose_failure.py`

调优前 Claude 中断后需人工检查容器状态、日志、context.yaml。

调优后 `run_pipeline.sh` 在 Claude 退出后自动调用 `diagnose_failure.py`：
- 输出中断步骤、错误原因、恢复建议、已完成/未完成步骤清单
- `--json` 输出供新会话自动读取
- 新会话启动时自动检测未完成流程，从中断点恢复

### 2.8 结构化错误持久化 `error_writer.py`

调优前工具脚本失败后，错误信息只在 stdout/stderr 中，Claude 可能因 context 压缩丢失。

调优后 `error_writer.py`（部署到容器内，工具脚本 import 使用）：
- `_last_error.json`：最新错误（覆盖写），含 tool、error_type、error_message、traceback
- `_error_history.jsonl`：错误历史（追加写）
- `checkpoint.json`：当前执行位置，中断后可精确定位
- 自动同步错误到 `context.yaml` 的 `workflow.last_error` 字段

### 2.9 工作流台账 `workflow_ledger`

调优前 `context.yaml` 只有 `workflow` 字段记录布尔状态（service_ok/accuracy_ok 等），无法一眼看出所有步骤的执行状态。

调优后新增 `workflow_ledger.steps[]`：
```yaml
workflow_ledger:
  steps:
  - name: container_preparation
    status: success
    started_at: '2026-04-11T02:25:00'
    finished_at: '2026-04-11T02:30:00'
    duration_seconds: 300
    notes: 容器就绪, 8x H20-3e, 工具脚本已部署
  - name: quick_accuracy
    status: success
    notes: V1=24.75 V2=20.20 dev=4.55 pass
```

状态流转：`pending → in_progress → success | failed | skipped`

### 2.10 网络失败自动处理

调优前网络失败没有统一策略，Claude 可能反复重试或询问用户。

调优后：
- pip install：自动依次尝试阿里云 → 清华 → 腾讯镜像（最多 4 次），全部失败直接终止
- 其他网络操作：自动重试 1 次，仍失败则终止
- 不询问用户，终止时输出失败原因和已尝试的镜像源列表

### 2.11 发布流程统一化

调优前 6打包（docker commit/tag/push）和 7上传（ModelScope/HuggingFace）分两步，docker push 等需要用户确认，可能需要补充 token。

调优后统一为一条命令：
```bash
python3 skills/flagos-release/tools/main.py --from-context /data/flagos-workspace/<model>/config/context_snapshot.yaml
```
自动完成 commit → tag → push Harbor → 上传 ModelScope/HuggingFace → 生成 README。凭证全部通过环境变量提供，docker push 也加入了权限预批准白名单。

### 2.12 新增关键约束

以下约束是调优后新增的，解决了实际运行中发现的问题：

| 约束 | 解决的问题 |
|------|-----------|
| 宿主机 mkdir 禁止花括号展开 | sandbox 拦截 `mkdir -p {a,b,c}` |
| 流程不可中途终止 | 防止 Claude 在不达标时自行停止 |
| workflow 状态必须与实际数据一致 | 防止数据不达标时误设 ok=true |
| 中间文件禁止写入项目源码目录 | 防止污染 git 仓库 |
| 工具脚本失败后必须读取 `_last_error.json` | 确保错误信息不丢失 |
| 编排层 JSON 必须包含 `_meta` 字段 | 确保产出文件自描述 |

### 2.13 `run_pipeline.sh` 兜底机制

Claude 退出后（无论正常完成还是中断），`run_pipeline.sh` 自动执行：
1. 容器内数据同步到宿主机（`docker cp results/traces/logs`）
2. 检查 Harbor 发布是否完成，未完成则自动补执行 `main.py`
3. 检查 `performance_compare.csv` 是否存在，缺失则自动生成
4. 运行 `diagnose_failure.py` 输出诊断结果

这确保了即使 Claude 因 context 超限或 API 异常退出，已产出的数据不会丢失，关键步骤有兜底补执行。

---

## 总结

调优后的核心改进归纳为三个方向：

1. **流程可预期**：去除算子优化循环，6 步线性执行，不达标 = 私有发布，不会陷入不可控的调优
2. **过程可观测**：从"只有 trace JSON"到四层日志体系（终端过滤 / pipeline.log / debug.log / issue 日志 + workflow_ledger 台账）
3. **故障可恢复**：从"中断后人工排查"到自动诊断 + 断点续跑（error_writer 结构化错误 + diagnose_failure 自动诊断 + 会话恢复检测 + run_pipeline.sh 兜底机制）
