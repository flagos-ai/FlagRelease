# Memory Index

- [V4/V5 算子基准](v4-v5-baseline-optimization-enabled-ops.md) — V4/V5 基准实际是 optimization.enabled_ops，versions.v3 是 dead field
- [V5 whitelist bug](v5-operator-expansion-whitelist-bug.md) — V5 operator_expansion 未按 plugin 环境分流，扩算子在 plugin 下不生效
- [plugin 精度调优 bug](plugin-accuracy-tuning-control-file-bug.md) — plugin 精度调优误用控制文件而非 whitelist env，调优空转
- [调优逻辑修复方案](tuning-logic-fixes-plan.md) — 五个调优问题的完整修复方案，用户诉求已定，待实施
- [海光 pipeline 缺陷修复](hygon-pipeline-bug-fixes.md) — V4被kill/V5误跳/DEBUG行污染 已修，报告缺算子待确认
- [统一入口重构清单](unified-op-config-refactor-plan.md) — ⑤第二批实施清单已批准待窗口，第一批V5 plugin bug已修
