---
name: v5-operator-expansion-whitelist-bug
description: V5 operator_expansion.py 的 write_control_file 未按 plugin 环境分流，plugin 下不更新 worker 读的 whitelist
metadata:
  type: project
---

plugin 环境下 worker 只读 `VLLM_FL_FLAGOS_WHITELIST` 环境变量决定启用哪些算子，从不读 `/root/flaggems_ops_control.json` 控制文件（用户已确认）。

- **V4** `operator_reduction.py:write_control_file`（约 line 181）已在提交 98c9fda 修复：`_is_plugin_env()` 检测 `VLLM_FL_PREFER_ENABLED`，plugin 环境写 `VLLM_FL_FLAGOS_WHITELIST` + 清 BLACKLIST，非 plugin 才写控制文件。
- **V5** `operator_expansion.py:write_control_file`（约 line 115）**已修复（2026-07，⑤重构第一批止血）**：照搬 V4 的 `_is_plugin_env()/_env_has/_clear_env` helper，改为双路分流——plugin 写 `VLLM_FL_FLAGOS_WHITELIST` + 清 BLACKLIST（空算子则 USE_FLAGGEMS=0），非 plugin 才写控制文件。三条路径(非plugin/plugin有算子/plugin空算子)单元自测通过，未真机验证。

后果（修复前）：V5 在 plugin 环境扩算子后，worker 读到的 whitelist 仍是上一版遗留值，V5 扩的算子不生效——镜像/tag 照常产出，但内容可能是 V3/V4 的翻版，且不会报错。

根治（第二批，待窗口）：本 bug 只是"应用算子配置逻辑复制多份且不一致"的一个实例，⑤ 统一入口重构会把 expansion/reduction/search 的 write/apply/restart 收敛为一份。详见 [[tuning-logic-fixes-plan]] ⑤ 与 [[plugin-accuracy-tuning-control-file-bug]]。

相关：V4/V5 基准问题见 [[v4-v5-baseline-optimization-enabled-ops]]。
