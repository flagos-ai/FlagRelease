# Memory Index

- [V4/V5 算子基准](v4-v5-baseline-optimization-enabled-ops.md) — V4/V5 基准实际是 optimization.enabled_ops，versions.v3 是 dead field
- [V5 whitelist bug](v5-operator-expansion-whitelist-bug.md) — V5 operator_expansion 未按 plugin 环境分流，扩算子在 plugin 下不生效
- [plugin 精度调优 bug](plugin-accuracy-tuning-control-file-bug.md) — plugin 精度调优误用控制文件而非 whitelist env，调优空转
- [调优逻辑修复方案](tuning-logic-fixes-plan.md) — 五个调优问题的完整修复方案，用户诉求已定，待实施
- [海光 pipeline 缺陷修复](hygon-pipeline-bug-fixes.md) — V4被kill/V5误跳/DEBUG行污染 已修，报告缺算子待确认
- [统一入口重构清单](unified-op-config-refactor-plan.md) — ⑤第二批实施清单已批准待窗口，第一批V5 plugin bug已修
- [plugin 传承+冷注入](plugin-inheritance-cold-injection.md) — gemma-3 强制切fl事故修复：VLLM_PLUGINS 持久化状态机 + 零引用镜像冷注入，已模拟验证待真机
- [V1性能基线合成](todo-v1-missing-perf-baseline.md) — 已实施：V2 初始×1.5 合成基线(全芯片统一)，下游零改动消费，待真机
- [镜像命名新规范](image-naming-new-spec.md) — 已实施：压缩版命名由 get_image_name.sh 权威生成，generate_image_tag 改调用它，V1-V5 后缀保留，待真机
- [步骤7防臆断闸门](step7-gate-anti-hallucination.md) — 已实施：step7_gate.py 用产物事实(达标率shell自算+搜索痕迹)判定，防agent臆断跳过operator_search，补跑后shell兜底直调脚本，待真机
- [V1.3残留服务误判](v1-3-stale-service-false-negative.md) — 已实施：国产厂商V1.3端口能响应但缺service_ready短语被误判残留失败，残留检测加"本次无进度信号(PHASES_OBSERVED空)"判据，待真机
- [防自由发挥闸门V1V2](anti-freewheel-gates-v1-v2.md) — 已实施：V1三选强制闸门(v1_gate.py+shell兜底调baseline_selector)+V2精度条件兜底(is_flaggems_service判模式,非FlagGems不硬测防污染)，待真机
- [V2-V5全链去阻断](no-block-full-chain-v2-v5.md) — 方案设计中：性能不卡流程(尽力后放行)、精度≤5%仍硬闸门、V3使能分支区分(B用VLLM_PLUGINS=fl不装/A照常装)、V2/V3不达标也发布打qualified标签
- [回滚误删产出修复](rollback-overflow-fix.md) — 已修复：段1越界回滚因rollback_overflow.py从未实现+fallback无差别rm+断点续跑误判→误删产出；新建脚本+时间戳判据(finished_at>seg1_start才算越界)+去掉危险rm
- [显存清理统一restart](gpu-cleanup-restart-unify.md) — cleanup_gpu_services改docker restart优先+注入.container_name；容器内无docker.sock无法restart的硬约束+分层清理点全貌
