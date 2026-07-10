---
name: plugin-inheritance-cold-injection
description: gemma-3 V2 被强制切 fl plugin 事故的修复：plugin 状态机(持久化传承) + 冷注入能力(零 flag_gems 引用镜像)
metadata:
  type: project
---

真机事故(2026-07,gemma-3/metax):start_service.sh 在 USE_FLAGGEMS=1 且 vllm_fl 可导入时无条件强制 `VLLM_PLUGINS=fl`,覆盖 V1 三选选定的厂商插件(metax)→ V2 的 53% 性能损失混淆了 plugin 切换开销与 FlagGems 开销,V3 对比失义,分支 B"V2.1 保持 V1 plugin + 代码注入"设计被打破。根因:V1 三选把 --vllm-plugins 参数化后**只接了 V1 一个调用方**,V2-V5 启动都没传,全落旧 auto-fl 默认。

## 修复架构:plugin 选择 = 持久化状态机(全部脚本化)

- **写入端1** baseline_selector.py `_persist_plugin_state`:三选定案后 persist `VLLM_PLUGINS=<选中值>`(含空串)到 /etc/environment;v1.1/v1.2 场景 clear `VLLM_FL_PREFER_ENABLED`(V2.1 注入路径生效前提+调优工具走控制文件路径);`none` 场景不持久化(保留 auto-fl 兜底,强依赖镜像 V2 需要 fl);写 context `baseline.v1_variant/vllm_plugins/vendor_plugin/v1_available`。另:`--vendor-plugin` 默认 auto=从 vllm.platform_plugins entry points 排除 fl 自动推导(封掉"Claude 挑厂商插件"的自觉缝隙)
- **写入端2** persist_op_config.py `persist_env_vars`:V3 固化时 env_vars 增加 `VLLM_PLUGINS=fl`(状态推进)
- **读取端** start_service.sh:/etc/environment 加载白名单加 `VLLM_PLUGINS`;决策三级:显式 --vllm-plugins(含空串) > 持久化/继承值(含空串,`${VLLM_PLUGINS+x}` 判定) > 旧 auto-fl 兜底
- **调优路径判据**改为"当前实际控制方式":`grep -q '^VLLM_FL_PREFER_ENABLED=true' /etc/environment` 命中=env_inline 路径,未命中=控制文件路径(vllm_fl **包存在≠plugin 控制生效**,v1.2 场景 V2 装着包但走注入+控制文件)。run_pipeline.sh 步骤5指令与 eval-comprehensive SKILL 模块C已同步

## 冷注入(base 镜像 vllm 源码零 flag_gems 引用,plugin 镜像常态)

- 原理:flag_gems.enable() 必须在**每个 worker 进程**内、模型加载前执行(FlagGems 官方 how_to_use 文档);model runner 模块是每个 worker 都 import 的文件
- inspect_env.py 双模式:有调用点→replace(原位替换,原有);无调用点→**cold**(新增):`_find_cold_inject_anchors` 用 importlib.find_spec 解析(不硬编码 python 版本路径)已知模块表 [vllm_ascend.worker.model_runner_v1(华为), vllm.v1.worker.gpu_model_runner, vllm.worker.model_runner(V0兜底)] + vllm_* 厂商包泛化扫描 *model_runner*.py(排除 vllm_fl);命中全注(env 门控幂等,多注无害);`_cold_inject_single_file` 在**模块尾部**追加 try/except 包裹的 `_build_inject_block`(异常不炸 worker),.flagos_backup 可回滚
- plugin 环境(vllm_plugin_flaggems)步骤2也执行注入(共用能力),注入块自带 VLLM_FL_PREFER_ENABLED 门控,plugin 路径下静默;**不写** control_env(避免干扰 V1 三选)
- 用户决策:apply_gems_patches_to_vllm **不纳入**(不受 only_enable 控制,会造成调优盲区);冷注入两分支共用

## 修复后版本形态(v1.2/metax 例)
V1=metax+gems关 → V2=metax+冷注入gems(V2−V1=纯FlagGems开销) → V3=fl+白名单(V3−V2=plugin路径开销) → V4/V5 fl 基础上减/扩

## 验证
模拟验证全过:锚点发现(含 vllm_fl 排除)/注入后语法/幂等/备份/运行时四场景(plugin门控pass、USE_FLAGGEMS=0 pass、=1 enable、flag_gems缺失不炸)/only_enable白名单分支/baseline_selector四场景落盘/start_service三级决策五场景(含 gemma-3 修复场景 B)。**未真机验证**。
另修:baseline_selector.py 此前不在 setup_workspace SCRIPT_MAP(与 operator_reduction 同款部署缺失),已补。

## 追加:V5 扩展"先试顶失败降级"策略(2026-07-09,已实施)
- 背景:用户质疑 V5"以 V3 为标准开最多算子"——结论:V3 作**安全基/回退锚点**合理(理论收敛结果不依赖起点,逐个探测会救回分组误伤+性能原因禁用的算子),但纯逐个爬轮次低效(N 个禁用=N 轮,数小时/轮)
- operator_expansion.py 改为三级:**Tier0** 一次性开 V3+全部禁用(达标即 1 轮完成)→ **Tier1** 排除 eval.excluded_ops_accuracy(精度风险最高)批量开其余(性能/崩溃来源,大概率过;过则该批批量保留,仅剩精度算子逐个)→ **Tier2** 逐个增量兜底(原算法,排序:非精度禁用在前)。任一 tier 失败只损失一轮启动+精度,正确性与纯逐个一致
- 崩溃算子无单独 context 标签(混在 optimization.disabled_ops),故 Tier0 直接全开、启动失败即降级,不做崩溃预排除
- state 新增 tier_results/actual_rounds;旧 state(无 tier_results 且有 probed)→ 标 skipped_legacy 直接续跑 Tier2,断点兼容。result 新增 strategy/tier_results,expansion_rounds 改记真实启动轮次
- 四场景模拟验证过:Tier0 直过(1轮 vs 4轮)/Tier0精度崩→Tier1过→剩余逐个/无分类记录→Tier1 skipped→全逐个(含 incompatible)/旧state续跑。CLAUDE.md 与段5 prompt 措辞已同步。未真机验证

## 追加:2.2/3.2 同镜像双 tag 编排接线(2026-07-09,已实施)
- 缺口:main.py 早已支持 --also-tag(publish.py:571)与 --incompatible-tag(:613),但 run_pipeline 段3/段4 硬编码 --version-tag v2/v3,V1=v1.3 场景会发两个内容相同的镜像并白跑一遍段4 plugin 验证
- 修复(shell 确定性,消费 baseline.v1_variant 状态):段3 前读 v1_variant,=v1.3 → SEG3_RELEASE_ARGS="--version-tag v2 --also-tag v3"(一次 commit 双 tag)+ prompt 注入说明(also-tag 不算进入步骤13,ledger 仍只更新 08);段4 gating 用 QUALIFIED_SEG4(v1.3 时置 False 短路,ledger 9-13 由 shell 置 skipped 并同步 context);段4.5/段5 仍用原 QUALIFIED(v1.3 时 V4/V5 照常,V4 基线=optimization.enabled_ops 不依赖段4 产物);断点续跑保险:段4 gating 处重读 v1_variant。段4 步骤13 发布处补 --incompatible-tag 用法提示(3.1 厂商/fl 均不适配时)
- 模拟验证:SEG3_ARGS 四场景(v1.3/v1.2/空/qualified=False)+ gating 三场景全过。未真机验证
- 仍未接:V1 镜像 pipeline 不产出(需用户定 commit 时机,"五版本全上 Harbor"意图 vs 版本表"阶段一手动发布"的冲突待拍板)

相关:[[new-v1-v5-workflow]] [[unified-op-config-refactor-plan]] [[hygon-pipeline-bug-fixes]] [[determinism-in-scripts]]
