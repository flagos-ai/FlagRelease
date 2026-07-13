---
name: rollback-overflow-fix
description: 段1越界回滚误删产出bug已修复(时间戳判据+新建rollback_overflow.py)
metadata:
  type: project
---

2026-07-13(test分支)修复的回滚机制bug。

**Bug根因(三重叠加)**:
1. `shared/rollback_overflow.py` 从未创建(git无记录),setup_workspace.sh:281 部署因 `[ -f ]` 静默跳过,容器内没有该脚本 → 段1/2/3 三处越界回滚调用全失败。
2. 只有段1 fallback 危险:调用失败后执行 `rm -f /root/flaggems_ops_control.json operator_config.json` 无差别删产出(段2/3 fallback 只打印继续)。
3. 段1无幂等 + 越界判据宽松(任何 step≥4 非pending 即越界)→ 断点续跑时上次 step4+ 的 success 状态被误判越界 → 触发 fallback rm → 删掉有效产出(用户观察到的"产出被覆盖")。

**修复(3改动,已验证bash -n + 双场景)**:
1. 新建 `shared/rollback_overflow.py`:`--overflow-from N` 按 STEP_OUTPUT_MAP 精细清理≥N的产物;`--preserve-step3-disabled-ops` 保留步骤3算子集(operator_config.json 回退为 disabled_ops 基线而非删除,保留 ops_control_initial.json/flaggems_ops_control.json);文件缺失安全跳过恒退出0。
2. run_pipeline.sh 段1(~800)持久化 `${LOG_DIR}/seg1_start_ts`(UTC ISO);越界检测(~880)改为只有 `finished_at > seg1_start` 的 step≥4 才算越界。seg1_start 缺失/finished_at 空 → 回退旧宽松判据(向后兼容)。仿段2 seg2_start_ts 范式。
3. 段1 fallback 去掉 `rm`,与段2/3 对齐仅告警继续(ledger 已回滚,越界步骤会重跑覆盖)。

未真机验证(容器内实际清理行为)。下一批工作:去阻断重构 [[no-block-full-chain-v2-v5]]。