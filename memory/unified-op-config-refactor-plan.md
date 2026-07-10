---
name: unified-op-config-refactor-plan
description: ⑤统一入口重构——半程收敛已完成（flagos_op_config.py 落地），全量方案被否决（动 operator_search 是负收益）
metadata:
  type: project
---

⑤"抽统一入口"重构。**半程收敛已实施完成（2026-07），全量方案经风险评估后被有意缩减**。

## 风险评估结论（用户认可）
全量方案两个高危点收益极小：①统一三份 restart 会动 operator_search 的调用约定（stop_cmd 语义），而它是唯一真机验证过的 V2/V3 调优主链路，本来就是对的；②Layer 1-4 能力探测与 operator_search 的 action 结构深耦合，搬迁=重写+全链路回归。**"已经对的东西不动"**。toggle_flaggems 遗留 modify-enable 路径经验证：非 plugin 正确；plugin 下空转但 fail-safe（重试2轮失败→issue→service_ok=false 私有发布，不产出假成功），不污染结果——因此半程收敛顺手修它而非废弃。

## 半程收敛已完成内容
1. **新建 skills/flagos-operator-replacement/tools/flagos_op_config.py**（唯一权威）：
   env_to_inline / build_op_env(native|full|custom,白名单优先) / is_plugin_env / persist_env / clear_env / env_has / load_etc_environment / **write_op_config**(双路：plugin→WHITELIST env+清BLACKLIST+空算子USE_FLAGGEMS=0；非plugin→控制文件+only_enable，返回所走路径)。
2. setup_workspace.sh SCRIPT_MAP 加部署行；**顺带发现并修复 operator_reduction.py 根本不在部署清单**（V4 段调 /flagos-workspace/scripts/operator_reduction.py 会文件不存在——海光 V4 问题的隐藏因素）。
3. operator_reduction.py / operator_expansion.py：删本地 _is_plugin_env/write_control_file/_persist_env 等副本，改 `from flagos_op_config import ...`（别名保留原函数名，内部调用点零改动）。
4. expansion 的 restart_and_wait 对齐 reduction 的稳健版（pkill -9 含 multiprocessing.spawn + ss 等端口释放；原版 pkill 无-9、不等端口）。**未统一三份 restart，只修有隐患的这份**。
5. apply_op_config.py generate() 内部改调 build_op_env（CLI 行为回归通过，输出算子改为排序——语义等价）。
6. diagnose_ops.py _build_group_env plugin 分支改调 build_op_env+env_to_inline（OOT/flagos 双层拆分保留在本地）。
7. toggle_flaggems.modify_enable_call 加 **plugin 守卫**：detect_plugin_mode()（vllm_fl 可导入，比 VLLM_FL_PREFER_ENABLED 更适合步骤3时点）+ disabled_ops → 走 blacklist env 持久化（合并已有 blacklist、清冲突 whitelist），返回结构兼容 results[] 信封；ImportError 降级回控制文件路径（宿主机/旧容器）。修复步骤3崩溃恢复在 plugin 下空转。

## 刻意不做（边界）
- operator_search.py **零改动**（已验证）。
- 不统一三份 restart。
- SKILL 指令层"改调脚本"缓行（①修复文字已正确）。

## 验证状态
6 文件编译+setup_workspace 语法通过；共享模块/双路/CLI回归/diagnose/toggle守卫 全部单元自测通过。**未真机验证**——需在 plugin+非plugin 各一台对拍 /proc/<vllm-pid>/environ 与 /tmp/flaggems_enable_oplist.txt。

相关：[[tuning-logic-fixes-plan]]、[[v5-operator-expansion-whitelist-bug]]（已由共享模块彻底收敛）、[[plugin-accuracy-tuning-control-file-bug]]、[[hygon-pipeline-bug-fixes]]。
