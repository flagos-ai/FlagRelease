---
name: flagos-log-analyzer
description: 分析推理服务日志以诊断启动失败、运行时错误、GPU 问题或 FlagGems 集成问题，提供失败恢复指引
version: 3.0.0
triggers:
  - log analysis
  - analyze logs
  - 日志分析
depends_on: []
provides:
  - diagnosis.status
  - diagnosis.errors
  - diagnosis.suggestions
---

# 日志分析 Skill

分析推理服务日志，识别部署或运行时问题，提供恢复指引。

**工具脚本**（已由 setup_workspace.sh 部署到容器）：
- `log_analyzer.py` — 日志分析与诊断（analyze / scan）

---

# 上下文集成

## 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
model:
  name: <来自 container-preparation>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
diagnosis:
  status: "<ok | warning | error>"
  service_status: "<running | crashed | oom_killed | ...>"
  errors: [<错误分类列表>]
  suggestions: [<建议列表>]
```

---

# 错误分类

| category | 严重性 | 典型关键词 | 建议 |
|----------|--------|-----------|------|
| `cuda_error` | critical | `CUDA error`, `CUDAError` | 检查算子/驱动兼容性 |
| `oom` | critical | `out of memory`, `OOM` | 减小 TP / max-model-len |
| `triton_compile` | critical | `triton compile fail` | FlagTree/Triton 版本问题 |
| `operator_error` | high | `flag_gems error`, `operator not supported` | 禁用问题算子 |
| `model_load` | high | `model not found`, `tokenizer error` | 检查模型路径 |
| `port_conflict` | medium | `address already in use` | 更换端口或杀占用进程 |
| `dependency` | medium | `ModuleNotFoundError` | pip install 缺失包 |
| `timeout` | low | `timeout`, `connection refused` | 等待或检查网络 |
| `warning` | info | `WARNING`, `DeprecationWarning` | 记录但不阻塞 |

---

# 工作流程

## 步骤 1 — 分析单个日志文件

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/log_analyzer.py analyze \
    --log-path /flagos-workspace/logs/startup_flagos.log \
    --json"
```

输出包含：错误分类列表、服务状态推断、FlagGems 检测、启动序列完成度、诊断建议。

## 步骤 2 — 扫描整个日志目录

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/log_analyzer.py scan \
    --log-dir /flagos-workspace/logs/ \
    --json"
```

扫描所有 `*.log` 文件，按时间排序，汇总错误和建议。

## 宿主机直接访问（可选）

由于日志目录已挂载到宿主机，也可直接在宿主机执行：

```bash
python3 skills/flagos-log-analyzer/tools/log_analyzer.py analyze \
    --log-path /data/flagos-workspace/<model>/logs/startup_flagos.log \
    --json

python3 skills/flagos-log-analyzer/tools/log_analyzer.py scan \
    --log-dir /data/flagos-workspace/<model>/logs/ \
    --json
```

---

# 完成条件

- 日志文件已扫描分析
- 错误已分类（category + severity）
- 服务状态已推断（running / crashed / oom_killed / ...）
- FlagGems 加载情况已检测
- 启动序列完成度已识别
- 诊断摘要和建议已生成

---

# 失败恢复指引

## 服务启动失败

```
诊断: 启动失败
  │
  ├── FlagOS 模式失败
  │   → 保存日志到 /flagos-workspace/logs/
  │   → 自动切回 Native 模式验证
  │   │
  │   ├── Native 也失败 → 环境问题，需人工介入
  │   └── Native 成功 → FlagGems 问题，触发算子优化
  │
  └── Native 模式失败
      → 检查 GPU 驱动、显存、模型路径
      → 建议调整 tensor-parallel-size
```

## Benchmark 失败

```
诊断: Benchmark 失败
  │
  ├── 单次失败 → 自动重试 1 次
  ├── 重试后仍失败 → 跳过当前 case，继续下一个
  └── 服务在测试中挂掉 → 重启服务 → 从失败的 case 继续
```

## 算子优化中途失败

```
诊断: 优化中断
  │
  ├── 进度已自动保存到 operator_config.json
  ├── 恢复上一个可用配置
  └── 支持断点继续：operator_optimizer.py next
```
