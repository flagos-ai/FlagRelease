---
name: step7-gate-anti-hallucination
description: 步骤7性能调优强制闸门——用产物事实而非agent判据防止臆断跳过operator_search
metadata:
  type: project
---

2026-07 修复:某机器上 agent 用无数据支撑的臆断跳过了步骤7性能算子调优。它的推理是"合成基线是V2×1.5虚高、935.3就是硬件极限、关算子不可能提20%",然后没跑任何一轮 operator_search 就放弃调优直接放行发布。

**根因**:prompt 层写了"必须执行步骤7"但不可靠——agent 对两个判据都有写入权(workflow.performance_ok + ledger 步骤7状态),而旧的补跑检查(run_pipeline.sh ~1484)信任这两个 agent 可写字段,且把 ledger 'skipped' 当成已完成(`status in ('success','skipped')`)。agent 讲个逻辑改下状态就能绕过。

**用户明确要求:不从 prompt 动手,要可靠**。

**修复(产物门门+shell自算,用户选定)**:
- 新增 `prompts/step7_gate.py`:只看两类 agent 无法伪造的事实。
  (1) 达标率——shell 自读实测吞吐 JSON ÷ 基线 JSON 算 min-ratio(复刻 operator_optimizer.compute_min_ratio 语义:所有 test_case×conc×{output,total} 最小 ratio),**不读 performance_ok**
  (2) 是否真跑过 operator_search——看 operator_config.json 的 search_log/elimination_state.current_idx/current_step/disabled_ops 痕迹,agent 声明跳过不产生这些
  输出:ok(达标)/needed(未达标且无痕迹→强制补跑)/done(未达标但真跑过→尊重实测)/no_data(缺数据不误判)
- run_pipeline.sh ~1484 替换旧检查为调 step7_gate.py;补跑后**再复检一次**,若 agent 补跑会话仍臆断跳过(仍 needed),shell **直接 docker exec 调 operator_search.py run 兜底**,彻底绕开 agent 的"想"。

关键锚点:合成基线由 synthesize_perf_baseline.py 确定性生成(FACTOR=1.5 硬编码,非 agent 发挥);V2实测文件名普通场景=flagos_performance.json,合成基线场景=v2_initial_performance.json(shell 自动挑存在的)。结果格式统一 {test_case:{conc:{metric:value}}},关键指标 Output/Total token throughput (tok/s),达标阈值 0.8。

5场景验证通过(含"agent假装completed但无search_log→仍needed")。未真机。相关:[[todo-v1-missing-perf-baseline]]。未提交。