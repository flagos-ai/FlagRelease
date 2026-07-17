---
name: no-block-full-chain-v2-v5
description: 用户要求V2-V5全链去阻断——性能不卡流程,精度≤5%仍是硬闸门
metadata:
  type: project
---

2026-07(test分支)用户提出的核心流程哲学变更。**已实施**(见 [[perf-non-blocking-policy]]):段4/4.5/5 门控**统一**改用 QUALIFIED_CORE=service_ok AND accuracy_ok(性能不阻断);段4 plugin 步骤12性能不达标照常继续步骤13;合成基线 FACTOR 1.5→1.2;operator_search/V4 reduction 调优设 20 轮上限(难调即停)。汇总表段4/段5显示均改用 QUALIFIED_CORE。

**V4/V5 门控重算修复(2026-07,V5盲目复用V3镜像 bug)**:根因——`QUALIFIED_CORE` 在 run_pipeline.sh:2014 **只算一次**,在段4(V3 plugin流程)**之前**,读的是段2(V2代码注入)的 `workflow.accuracy_ok`(通常达标)。段4 若 V3 plugin 精度不达标,SKILL 设的是 `plugin_workflow.accuracy_ok=false`(V3不产出,见 flagos-plugin-install/SKILL.md:260),**不是** `workflow.accuracy_ok`。段4.5(V4)和段5(V5)门控复用段4之前的旧 QUALIFIED_CORE、从不重算 → V3不合格却因旧门控True,V5从不合格V3基础扩展并复用其镜像发布 -v5。**修复**:段4 context 重同步后(约2249行)新增 `QUALIFIED_CORE_V3`——基础=service_ok AND accuracy_ok,若 `plugin_workflow.triggered` 为真则叠加 `plugin_workflow.accuracy_ok`(分支B),否则用基础值(分支A/v1.3同镜像不误关)。段4.5入口、段5入口、两处else提示、汇总表段5行改用 QUALIFIED_CORE_V3;段4入口门控(2039/2052/2217)和汇总表段4行**保留** QUALIFIED_CORE(进入plugin流程用pre-plugin值正确)。用户定稿:V3精度不达标 → V4和V5都不产出。5场景已验证(核心bug场景/分支A/v1.3同镜像均正确)。主修复从源头拦截:V5整段跳过 → 根本不调用 main.py --version-tag v5,无需在发布脚本打补丁。

**V5 门控修复(2026-07 关键)**:段5 原用 `HAS_DISABLED_OPS=yes` 作门控 → 两个 bug:①V3跑通但全开就达标/从没禁用过算子时 HAS_DISABLED_OPS=no → **漏产 V5**(违反"一定得到V5");②精度崩/服务挂时 HAS_DISABLED_OPS 可能=yes → **误触发 V5**。已改为 QUALIFIED_CORE 门控(run_pipeline.sh 段5 if + else 提示 + 汇总表行)。无禁用算子时 operator_expansion.py:552 走"无需扩展→success"分支,V5=当前最优版本仍照常发 -v5。HAS_DISABLED_OPS 变量保留但降级为参考/日志,不再门控。风险评估:选此方案(改门控复用已测脚本路径)而非"重打tag"方案,因后者涉及 docker tag+Harbor重推等外部不可逆操作、失败面更大。

**核心思路:V2→V5 每一步都不被阻断,竭尽全力跑完整链,各自产出"当前条件下最优镜像"。**

**性能:从流程闸门降级为尽力目标**
- 性能调优该做还做(累积关闭算子/逐轮搜索,即 step7 手段),但**调不动就带当前结果放行**,不停在原地
- V2 性能不达标 → 仍走 V3,不跳过
- V4 基准本是"超过V3",但若 V3 本身没到80% → 不纠结达标,取该场景**相对性能最优**镜像作为 V4
- V5 只要**精度对齐 + 尽量多开算子**,**完全不考虑性能**

**精度:永远是硬闸门,不放宽**
- 精度退化≤5%(rel_drop=(基线-当前)/基线)始终硬卡,精度崩才算真失败
- 答非所问的镜像绝不发布

**V3 使能方式按分支区分(易错点,用户纠正过两次)**
- 分支B(gems_tree_plugin):准入镜像本就带plugin → V3**不装plugin**,只 `VLLM_PLUGINS="fl"` 环境变量使能(同v1.3 fl路径)+放宽调优硬要求
- 分支A(gems_tree):准入镜像无plugin → V3**照常装plugin**/切白名单,保持原路线不动

**发布:V2/V3 无论性能达标都发布(-v2/-v3 tag)+ 打 qualified=true/false 标签**,下游靠标签区分,不达标不再等于不发布。

**待改的阻断点(初步,子agent在核实全貌)**:段2 check_seg_complete性能校验触发retry/阻断、step7_gate强制闸门(约1588行,我前几天刚加的——需从"阻断"改"尽力后放行只记录")、qualified计算(约1741)对下游段和发布的影响、段3/4/5前置门槛、发布逻辑accuracy_ok/performance_ok=false行为(约1829)。改动牵一发动全身,需先出完整方案再动手。相关:[[anti-freewheel-gates-v1-v2]]、[[step7-gate-anti-hallucination]]。