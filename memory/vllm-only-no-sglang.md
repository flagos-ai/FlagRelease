---
name: vllm-only-no-sglang
description: 项目只支持 vllm 推理框架,已彻底移除 sglang;runtime.framework 固定为 vllm
metadata:
  type: project
---

2026-07(test分支)用户决定:**只支持 vllm,彻底移除 sglang 相关逻辑**。

**根因**(本次触发):context.template.yaml 的 `runtime.framework` 初始为空串 `""`(注释"由 service-startup 写入"),但没有任何组件真正回写它——setup_workspace.sh 只写 container/workspace/model 四字段,inspect_env.py 只输出检测报告不回写。V1 三选(baseline_selector→start_service.sh)在字段仍为空串时执行,而 start_service.sh 的 `.get('framework','vllm')` 对"存在但为空"的键默认值失效 → FRAMEWORK="" → `if [ "$FRAMEWORK" = "vllm" ]` 不匹配 → 落入 else 的 sglang 分支。这是循环依赖:start_service 要读该值决定框架,却又期望 start_service 自己写入。

**修复**:
1. context.template.yaml:`runtime.framework: "vllm"`(固定值,消除循环依赖,下游直接消费),并删掉 inspection.core_packages.sglang 字段。
2. start_service.sh:`framework = ctx.get('runtime',{}).get('framework') or 'vllm'`(空值兜底);删除 sglang else 启动分支,改为"非 vllm 直接 exit 1 报错"。
3. 全仓移除 sglang:进程清理 pkill/grep 模式删 sglang 保留 vllm(operator_reduction/search/expansion.py、persist_op_config.py、baseline_selector.py、wait_for_service.sh、setup_workspace.sh、run_pipeline.sh 等);日志正则(service_monitor.py、eval_wrapper.py、diagnose_failure.py 的进程检测 dict/list 去掉 sglang 键);检测(inspect_env.py 去掉 sglang 包检测和 framework 扫描);toggle_flaggems.py 删 `import sglang` 路径查找块;stream_filter.py 删 sglang 命令白名单和 SGLang 启动 banner;各 SKILL.md/CLAUDE.md/docs 文字。

**唯一保留**:examples/release_notes_flagscale.md:66 是第三方 FlagScale changelog 的 PR 标题引用,非本项目逻辑,不动。

**How to apply**:今后勿再引入 sglang 分支;runtime.framework 视为常量 vllm,勿改回空串或加 sglang 选项。相关:[[no-block-full-chain-v2-v5]]。
