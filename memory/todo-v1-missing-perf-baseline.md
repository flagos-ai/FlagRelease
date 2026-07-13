---
name: todo-v1-missing-perf-baseline
description: 已实施:V1 性能基线完全缺失时,以 V2 初始性能 ×1.2 合成基线(全芯片统一;2026-07 从1.5下调)
metadata:
  type: project
---

**已实施(2026-07-09,用户确认方案与全芯片统一标准)**:V1 性能基线完全缺失时(分支 B 三选=none 强依赖 flaggems,或 V1 服务无法启动),以 **V2 使能 flaggems 后首次可正常启动状态的初始性能 × FACTOR 作为性能基线**。**FACTOR 于 2026-07 从 1.5 下调至 1.2**(降低性能门槛;配合性能不阻断策略 [[perf-non-blocking-policy]])。

## 方案定稿要点(讨论结论)
- 测量时机:**步骤4 之前**(段2 开头),此时算子集=步骤3 幸存的初始状态(未被步骤5 精度调优削减)——流程中本没有任何环节保证测过"全开"性能,该口径是唯一可确定性执行的定义
- FACTOR 语义=经验推定的 V1/V2初始 性能比(合成一个虚拟 V1);×1.2 下 80% 判据等价于"调优后 ≥ 初始的 0.96 倍"(原 ×1.5 时是 1.2 倍),无参照场景的达标率会偏低,按既有"不达标不终止、私有发布"处理
- 成本中性:省掉的 V1 性能轮 ≈ 换来的初始测量轮
- 段1 有"禁止 benchmark"硬边界,故测量放段2 开头而非段1,不破坏段边界与约束19(已加"先性能后精度串行插入"例外)

## 实施内容
- **新脚本** `skills/flagos-performance-testing/tools/synthesize_perf_baseline.py`:吞吐(tok/s/throughput)×FACTOR、延迟((ms)/ttft/tpot/itl/latency)÷FACTOR、其余原样(FACTOR=1.2);输出 **native_performance.json 标准扁平格式** + `_meta.synthetic=true/baseline_source=v2_initial_x{FACTOR}/factor`(串已改为从 FACTOR 派生);**防覆盖护栏**(已存在且非 synthetic 的文件拒绝覆盖,exit 2,--force 可强制;合成文件自身可幂等重生成)
- **零改动消费**(已验证):performance_compare.py / operator_optimizer.py `_extract_native_throughputs` / operator_search.py 照常把合成文件当 V1 基线
- generate_report.py:识别 `_meta.synthetic` 在"## 性能评测"下加合成基线警示
- run_pipeline.sh:段2"无 V1 场景性能基线合成"条件块(V2 启动→quick 测 v2_initial_performance→合成→context 记 baseline.perf_baseline_source→停服→进步骤4);BRANCH_DIRECTIVE A/B 加兜底句
- CLAUDE.md:V1 基线定义、决策表行、约束19 例外;performance-testing SKILL.md 步骤3 加无 V1 分支
- setup_workspace.sh SCRIPT_MAP 部署行

## 验证
编译/语法全过;功能验证:缩放方向正确(1000→1500,TTFT 300→200,计数原样)、防覆盖 exit 2、幂等重生成、performance_compare 真实对比(1.25×初始→ratio 83.3% PASS)、optimizer 提取基线 1500。**未真机验证**。

相关:[[new-v1-v5-workflow]]
