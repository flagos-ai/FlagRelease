---
name: v4-v5-baseline-optimization-enabled-ops
description: V4/V5 算子基准实际来自 optimization.enabled_ops，versions.v3.enabled_ops 是有读无写的 dead field
metadata:
  type: project
---

V4(减算子)和 V5(扩算子)的算子基准，名义上应是 V3，但代码与实际行为有出入：

- `operator_reduction.py:get_enabled_ops`（V4）优先读 `versions.v3.enabled_ops`，兜底 `optimization.enabled_ops`，再兜底 `initial - disabled`。
- `operator_expansion.py:get_enabled_ops`（V5）直接读 `optimization.enabled_ops`，无 V3 兜底。

**关键事实：全仓库没有任何代码或 SKILL 指令写 `versions.v3.enabled_ops`**（只有 operator_reduction.py、generate_report.py 及报告回退函数在读它）。SKILL.md 只定义并维护 `optimization.enabled_ops`（无版本号的全局单一状态）。

因此：
1. `versions.v3.enabled_ops` 是 dead field（有读无写），V4 那条"优先读 V3"分支实际永远走不到，V4 和 V5 实际都以 `optimization.enabled_ops`（算子优化段最终态）为基准。
2. V4 减算子后**不回写** `optimization.enabled_ops`（只写自己的 state 文件 `current_enabled_ops`），所以 V5 读到的仍是 V4 之前的值 → V4/V5 是**同源并行**（都从优化段最终态出发，V4 往下减、V5 往上扩），不是 V5 继承 V4。这个并行设计本身合理。

**隐患**：现在能正常工作依赖两个巧合互相抵消——"没人写 versions.v3" + "V4 不回写 optimization"。若将来 V3 段开始写 `versions.v3.enabled_ops`，或 V4 改成回写 `optimization`，基准链就会漂移。

相关：worker 只读 whitelist env、V5 的 write_control_file 仍未按 plugin 环境分流（见 [[v5-operator-expansion-whitelist-bug]]）。
