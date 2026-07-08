---
name: unified-op-config-refactor-plan
description: ⑤统一入口重构第二批实施清单——flagos_op_config.py 收敛 env构建/双路应用/重启，6个调用点改造，需窗口+真机回归
metadata:
  type: project
---

⑤"抽统一入口"重构的第二批实施清单（第一批止血已完成：operator_expansion.py plugin 分支 bug 已修）。方案已获用户批准，等非跑批窗口执行。

## 现状（扫描确认，2026-07）
"应用算子配置+重启"散在 8 个 .py 的 12+ 处：
- **env_to_inline 4-5 份重复**：apply_op_config.py:29 / operator_search.py:344 / operator_optimizer.py / diagnose_ops.py:622 / toggle_flaggems.py:27
- **双路应用 3 份**：operator_search.py:313 apply_operator_config（最完整，含 Layer1-4 能力探测）/ operator_reduction.py:181 write_control_file（/etc/environment 持久化）/ operator_expansion.py:141（第一批刚补 plugin 分支）
- **restart 3 份且行为不同**：search:628（优雅停止→超时才-9，socket探测30次等端口）/ reduction:235（直接 pkill -9 含 multiprocessing.spawn，ss 探测15次）/ expansion:182（**pkill 无-9、不等端口直接 sleep 5——潜在隐患：旧进程未死透/端口未释放**，日志硬编码 startup_v5.log）
- 部署机制：setup_workspace.sh 的 SCRIPT_MAP 显式清单逐个 docker cp 到容器扁平 /flagos-workspace/scripts/，同级可 import。

## 实施步骤
1. 新建 skills/flagos-operator-replacement/tools/flagos_op_config.py，三个权威函数：
   - build_op_env(enabled/disabled_ops, plugin_mode, mode) → env dict + env_to_inline（唯一）
   - apply_op_config(...) → 双路分派（plugin→whitelist env / 非plugin→控制文件+FLAGGEMS_CONTROL_MODE），融合 search 的 Layer 探测 + reduction 的 /etc/environment 持久化
   - restart_service(service_cmd, wait_script, env_inline, ...) → 唯一重启：优雅停止→pkill -9 兜底（含 multiprocessing.spawn）→等端口释放→清 Triton/FlagGems 缓存→启动→wait_for_service
2. setup_workspace.sh SCRIPT_MAP 加一行部署该模块。
3. 逐个改 6 个调用点为 import 共享模块（每个单独 py_compile+自测）：operator_search / operator_reduction / operator_expansion / apply_op_config（保留薄 CLI 包装）/ diagnose_ops / persist_op_config（仅共享 env 构建，持久化场景不重启）。
4. 废弃 toggle_flaggems.py 的 modify_enable_call/_write_ops_control_file（SKILL/pipeline 已声明不使用）。
5. 指令层：eval-comprehensive/SKILL.md:581-592 精度调优手工应用段落改为"调用统一入口脚本"。

## 回归验证（必做）
- 每工具 py_compile + 单元自测
- 真机（plugin + 非plugin 各一台）对拍 /proc/<vllm-pid>/environ 的 VLLM_FL_FLAGOS_* + /tmp/flaggems_enable_oplist.txt，确认两条路算子真实生效
- 在非跑批窗口执行

## 效果
单一事实源（plugin/非plugin 判断从 3+ 处→1 处）；消灭 expansion restart 无-9/不等端口的隐患；精度调优不再靠 Claude 手工拼配置；4-5 份 env_to_inline + 3 份 restart + 3 份双路 → 各 1 份。

相关：[[tuning-logic-fixes-plan]] ⑤、[[v5-operator-expansion-whitelist-bug]]（第一批已修）、[[plugin-accuracy-tuning-control-file-bug]]、[[hygon-pipeline-bug-fixes]]。
