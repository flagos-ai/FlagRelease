---
name: operator-search-nohup-source-fix
description: operator_search.py性能调优服务启动因nohup直接套source复合命令静默失败已修复
metadata:
  type: project
---

2026-07-13(test分支)修复的性能调优(步骤7)服务启动bug。

**现象**: operator_search.py 内部搜索循环重启服务时静默失败,服务没起来。Claude 曾被迫绕开它手动前台调 start_service.sh + wait_for_service.sh 应急。

**根因**: operator_search.py restart_service()(~685/688) 构造后台启动命令时把 nohup 直接拼在 startup_cmd 前:`nohup {startup_cmd} > log &`。当 startup_cmd 是 shell 复合命令(DTK 平台 `source /opt/dtk-*/env.sh && bash start_service.sh`)时,变成 `nohup source ...`——nohup 只能执行可执行文件,source 是 shell 内建 → 报 "无法运行命令 'source'" 退出127静默失败。实测复现:退出127+日志空+service_log_path未写。

**修复(最小,原设计不变)**: 在 nohup 与 startup_cmd 间插入 `bash -c {shlex.quote(startup_cmd)}` 包裹,让 source/&& 在子shell内合法执行。外层 nohup 后台化/nohup_log/env_inline 在前全部保留。import shlex。实测修复后 DTK env 正确 source、start_service 被调用并写 service_log_path。

**范围**: 只 operator_search.py 一处。operator_reduction.py/operator_expansion.py 的 restart_and_wait 用 subprocess.Popen(service_cmd)(不加nohup)不踩坑,无需改。

**背景**: start_service.sh 本身已 `nohup bash -c "...vllm..." &` 后台化vllm(第23行)并写service_log_path(第26行)+自己 source /opt/dtk/env.sh(第15行)。设计上是"前台调用、内部后台化、立即返回"。

关联: [[gpu-cleanup-restart-unify]] 同为服务重启相关；均 test 分支编排/调优层修复。