---
name: tuning-logic-fixes-plan
description: 五个调优逻辑问题的完整修复方案（用户已定诉求，待实施）
metadata:
  type: project
---

用户提出的五个调优逻辑问题的修复方案。用户已就 ②④ 给出明确诉求，方案按用户诉求定稿，**待实施（尚未改代码）**。

## ① plugin 精度调优空转（前置项）
位置：eval-comprehensive SKILL 581-585、diagnose_ops.py apply_method 498-502、run_pipeline.sh 1075/1094。
现状：plugin 下每轮把白名单写 /root/flaggems_ops_control.json，但 worker 只读 VLLM_FL_* env，控制文件死代码，调优空转。详见 [[plugin-accuracy-tuning-control-file-bug]]。
方案：精度调优 plugin 场景改用 env_inline（diagnose_ops 的 cumulative_test_env.env_inline 已现成），复用 plugin-install SKILL 226-236 已写对的 apply_op_config.py --flagos-blacklist 那套。删除"写控制文件"表述，修正 581 行错误的"与 operator_search.py 一致"。
验证：真机对拍 /proc/<vllm-pid>/environ 的 VLLM_FL_FLAGOS_BLACKLIST + flaggems_enable_oplist.txt。

## ② V4 减算子：最少轮次最大效果（用户诉求已定）
位置：operator_reduction.py 402-446（贪心 + 443 行 best_gain<1% 提前 break）。
现状：贪心每轮逐个试删选增益最大者，首轮无 >1% 增益即停 → gemma3/hunyuan V4 卡在原点没减。
**用户诉求**：动用最少轮次实现最大效果；用户判断"很可能关到只剩一个算子时性能最好"。不要随机。
方案：
- 去掉 1% 提前终止（443 行不再 break）。
- 一路下探直到最小（min_ops，plugin 下可到 0/1），即使单步增益为负也继续。
- 全程记录每一步 (算子组合, 吞吐, 精度护栏结果)。
- 结束后**全局回溯**：从所有探测过的组合中，选"精度护栏通过 且 综合吞吐最高"的组合作为 V4 结果。
- 精度护栏保留：护栏不通过的组合不参与最优选择。
待权衡（实施时与用户确认）：贪心逐个试删是 O(N²) 次 benchmark，很慢，受 V4 单模型 15h 超时约束。若"很可能关到剩1个最好"成立，可考虑更省的下探顺序（一次定序后逐个关，O(N) 次），减少轮次。

## ③ V2 精度调优轮次不足（glm4）
位置：eval-comprehensive SKILL 728/778/884（硬编码"最多3轮/3组"）+ 870 行"(1/2/3)"输出格式。
现状：glm4 有5组24算子，3轮后仅累积禁用前3组，仍不达标就放弃。
方案：轮次上限改为"分组数"（分几组跑几轮，覆盖所有组），加绝对上限（如8）防失控，去掉死值3。同步改 728/778/884/870 四处。依赖 ① 先修（否则空转，加轮次无用）。

## ④ V3(plugin) 精度/性能不达标：三级递进（用户诉求已定）
位置：plugin-install SKILL 220/255-256；与 CLAUDE.md:98"Plugin 阶段允许算子调优"冲突。
场景：V2 未开 plugin（gems+tree），V3 切 plugin 用**相同算子列表**，此时出现精度/性能下降。
现状：写 issue、accuracy_ok=false、不调优、继续走发布。（注：现状"不调优"，非"结束阶段"；结束阶段实为服务崩溃处理 219 行）。
**用户诉求的新逻辑（三级递进）**：
1. 先写 issue（记录问题）。
2. 通过关算子调优（operator_search.py run --plugin-mode，走 env_inline）。
3. 若所有算子全关仍不达标 → 才判定为框架问题（plugin-error issue）。
方案：plugin-install SKILL 220 从"不调优"改为上述三级逻辑；与 CLAUDE.md:98 对齐消除冲突；保留"服务崩溃→issue+停止"不变。

## ⑤ 指令自相矛盾（根因，根治项）
plugin-install SKILL 226/239-240 明确"必须用 env、禁止控制文件、控制文件无效"（对），但 eval SKILL 581 + diagnose apply_method 说"写控制文件"（错）。同一件事两处相反。
根治：抽统一入口 apply_operator_config(ops, env_type, mode)，按 env_type 选控制文件 vs env_inline，让性能调优/精度调优/V4/V5/plugin步骤10 全走它（apply_op_config.py 已是雏形）。一次收敛 ①④⑤ + [[v5-operator-expansion-whitelist-bug]]。范围大，单独排期+回归测试，不在跑批期间动。

## 实施顺序
第一批（独立止血）：① ② ③ — **已完成**（见下方实施记录）
第二批（依赖①）：④ — **已完成**（见下方 ④ 实施记录）
第三批（根治，有窗口再做）：⑤ — 待实施（唯一剩余的调优逻辑修复）

## 第一批实施记录（已改，编译通过）
- ①：eval-comprehensive SKILL 模块C第4步 + 固化选择块改为按 env_type 分流（plugin 用 env_inline，禁写控制文件；非 plugin 才写控制文件），加第7步"验证生效"（对拍 /proc/<pid>/environ）。diagnose_ops.py apply_method 字段同步改。run_pipeline.sh 1075/1094 同步改。
- ②：operator_reduction.py run_reduction 重写为**两阶段**（用户最终定的算法，取代早先的"顺序累积下探+全局回溯"）：
  - **阶段1 性能搜索（全程不测精度）**：从 V3 基线起按 probe_order 逐个试禁用，仅当禁用后吞吐 > 当前基线才提交（基线动态推进），记入 improvements 序列；否则保留该算子。崩溃算子标 essential 跳过。O(N) 次 benchmark。
  - **阶段2 精度回溯**：improvements 按性能降序，用最优组合测精度，达标即产出；不达标回退次优；最坏回退 V3 基线（improvements[0]，等价 V3 仍成立）。精度只在此阶段按需测。
  - 断点续跑字段：improvements/probe_order/committed_removed/probe_idx/essential_ops 存 state。
  - 用户选定丙-1（改 operator_reduction.py 内部）。
- ②准则修正（用户明确）：**V4 追求性能绝对值最大化，达标基准是"超越 V3"，不与 V1 比较**（V1 仅报告参考）；**至少保留 1 个算子（plugin 也不例外，原 min_ops=0 已改为恒 1）**；精度是成立前提。success = beats_v3 and kept_at_least_one and (accuracy_ok is not False)。失败 reason 按三条准则分别给出。注：这推翻了 [[plugin-accuracy-tuning-control-file-bug]] 里"plugin 可减至 0"的旧记录。
- ③：eval-comprehensive SKILL 迭代控制 728/778/884/870 四处，"最多3轮"改为"轮次上限=分组数，绝对上限8轮"。
- 未做真机验证（需在 plugin 机器对拍 environ + 观察 V4 下探日志）。write_control_file 已按 plugin 分流（98c9fda），measure_config 复用它，下探链路 plugin/非plugin 抓手均正确。

## ④ 实施记录（已改，语法通过；用户诉求确认：精度+性能都递进）
V3(plugin) 精度/性能不达标从"写issue不调优"改为**三级递进**：①先写issue → ②plugin模式关算子调优(operator_search.py run --plugin-mode，走env_inline VLLM_FL_FLAGOS_BLACKLIST) → ③全关flaggems仍不达标才判框架问题(plugin-error issue)。精度和性能同此逻辑。
改动4处 + 消除矛盾：
- plugin-install SKILL：表格"不达标处理"行改三级递进；步骤11/12 处理段完整重写为三级；加注说明"不重新调优"仅指达标默认路径。
- run_pipeline.sh 段4 PROMPT_SEG4（1860附近，Claude运行时实际遵循）：1860行改三级递进；1863算子集表述加限定；1865 修正为"必须用env禁止控制文件"（原写控制文件与①矛盾）。
- CLAUDE.md：92/211 裸"不重新调优"加"达标则/不达标进三级递进"限定；98行补第三级"全关仍不达标判框架问题"终止条件。
- 关键发现：CLAUDE.md 80/81/98/125 本就说要调优、operator_search.py 已支持 plugin_mode+env_inline，矛盾在 CLAUDE.md(要调优) vs plugin-install SKILL(不调优) 两文件之间，SKILL 是过时错误的一方。
