---
name: plugin-accuracy-tuning-control-file-bug
description: plugin 环境精度调优误用控制文件而非 whitelist env，导致每轮禁用算子不生效、调优空转
metadata:
  type: project
---

**已查实的严肃 bug**：plugin 环境下精度算子调优（步骤5）的每轮禁用不生效，调优空转。

链路事实（均已从代码确认）：
- plugin worker 只读 `VLLM_FL_FLAGOS_WHITELIST/BLACKLIST` env 决定算子，不读 `/root/flaggems_ops_control.json`（用户确认）。
- `start_service.sh` 148-174：只从 `/etc/environment` 透传 `VLLM_FL_*`，并从控制文件推断 `FLAGGEMS_CONTROL_MODE` 字符串（only_enable/unused），**不把控制文件的 include 列表转成 whitelist env**。
- 性能调优 `operator_search.py` plugin 分支 = 正确：`_apply_plugin_config` 生成 env_inline，`restart_service`（约 line 684）内联 `VLLM_FL_FLAGOS_WHITELIST=... nohup vllm serve` 启动。
- 精度调优 = 错误：SKILL.md(flagos-eval-comprehensive) 570-599 + diagnose_ops.py 的 `apply_method`(498-502) 指示 agent 每轮把白名单写进控制文件、用 start_service.sh 启动，且这一步没往 /etc/environment 写 whitelist。SKILL 581 行注明"与 operator_search.py 一致"，实际不一致（operator_search 用 env_inline）。

讽刺点：`diagnose_ops.py:_build_group_env`(530-545) 其实**已经生成了正确的 env_inline**（含 VLLM_FL_FLAGOS_BLACKLIST），就在 `cumulative_test_env` 里，但 apply_method 和 SKILL 指向了同结构里的 `control_file` 字段，用错了一半。

后果连锁：精度调优产出的 disabled_ops/excluded_ops_accuracy 不可信 → 传给性能调优(步骤6/7)基线 → 也传给 V5 扩算子(V5 读 optimization.disabled_ops 逐个重开)。见 [[v4-v5-baseline-optimization-enabled-ops]]。

修复方向（未动手，用户基调"先诊断"）：精度调优 plugin 场景改用 cumulative_test_env.env_inline 内联启动，或直接复用 operator_search.py 的 restart_service(env_inline=...)。改前建议在真机用 /proc/<vllm-pid>/environ 对拍验证。相关 [[v5-operator-expansion-whitelist-bug]]。
