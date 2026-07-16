---
name: v1-baseline-selector-monitors-wrong-log
description: 分支B V1三选恒判none的根因——baseline_selector监控的日志文件与start_service.sh实际写入的不是同一个
metadata:
  type: project
---

分支 B(gems_tree_plugin)V1 三选状态机 `baseline_selector.py` 存在"监控错文件"bug，导致有 V1 基线的环境（含 NV）被误判 `v1_variant=none / v1_available=false`，下游错误回退 NV 精度基线 + 合成性能基线。

**根因（真机 SmolLM2-135M-Instruct_flagos_0714_0922 上确认）**：
- `baseline_selector.py:179` 用 `Popen(..., stdout=open(startup_{variant}.log))` 启动 `start_service.sh`，并让 `wait_for_service.sh --log-path startup_{variant}.log` 监控它。
- 但 `start_service.sh:212` 日志名由 `--mode` 决定：`LOG_FILE=startup_${MODE}.log`；V1 三个 variant(v1.1/v1.2/v1.3)全部 `--mode native`(use_flaggems=False)。且 `start_service.sh:253` 用 `nohup ... > $LOG_FILE &` 后台化并二次重定向。
- 结果：`startup_v1.1.log` 只截获到 start_service.sh 的 6 行 echo(~585字节，含末行 `log=/flagos-workspace/logs/startup_native.log`)，真正的 vLLM 输出全进了 `startup_native.log`。`wait_for_service` 盯着一个永不增长的文件 → 无进度/致命信号 → 盲判。
- 附带：v1.1/v1.3/native 都写同一个 `startup_native.log`，互相覆盖，事后无法追溯各 variant 失败原因。

**验证产物特征**：`v1_baseline_selection.json` 里 attempts[] 全 `service_ok=false, reason="服务启动失败", smoke_answer=""`(空=没走到冒烟)；对应 `startup_v1.x.log` 只有几百字节 echo。

**注意**：那台机器这次的失败叠加了真·环境问题（GPU 显存被占，`ValueError: Free memory 10.1/139.81 GiB < desired 128.63`），但显存错误在 startup_native.log 里，wait_for_service 因监控错文件根本没看到。

**修复①（已实施，2026-07 真机验证日志落位成功）**：
- `start_service.sh` 新增 `--log-file` 参数（默认回退 `startup_${MODE}.log`，现存调用零影响），并让 `startup_default.log` 软链跟随实际 LOG_FILE。
- `baseline_selector.start_variant`：`cmd` 加 `--log-file '{log_path}'` 让 start_service.sh 把 vLLM 日志写进 variant 独立文件；Popen stdout 改 DEVNULL；wait_cmd 加 `--from-start`（offset 从 0 读）。
- 验证：真机 SmolLM2 上 `startup_v1.1.log` 从 585B 空壳变为 16KB 完整 vLLM 日志，软链/`service_log_path` 均正确跟随。日志落位 bug 彻底修复。

**修复②（已实施并真机验证，2026-07）——第二个独立 bug：wait_for_service 引号崩溃**：
修复①后 V1 服务真的起来了（日志含 `Application startup complete` + `GET /v1/models 200`），但仍判 `service_ok=false`。根因锁定在 `wait_for_service.sh:196`：
```python
if "vllm._C" in s or "Failed to import from vllm._C" in s:
```
这行 python 代码里的**内层双引号未转义**，而整段 python 嵌在第134行 `python3 -c "..."` 的**外层双引号**内。bash 解析时，`"vllm._C"` 的双引号提前闭合了外层 `-c "`，传给 python 的是损坏代码 → 执行到该行 `NameError: name 'vllm' is not defined` 非零退出 → `analyze_new_lines` 走 `2>/dev/null || echo` 兜底 → 恒返回 `progress:false, new_size=0` → `PHASES_OBSERVED` 恒空 → 端口响应也被判 `stale_service`(exit 2) → V1 恒失败。
- **修复**：`:196` 内层双引号改单引号 `if 'vllm._C' in s or 'Failed to import from vllm._C' in s:`（python 单双引号等价，功能不变），并加注释警示外层 -c 引号约束。
- **铁证**：提取该 python 代码成独立 .py 文件跑 → progress:true 成功；wait 里 `-c "..."` 方式跑 → NameError 兜底。最小复现确认报 `NameError: name 'vllm' is not defined`。
- **为何被修复①掩盖**：修复①前 wait 监控空的 585B 文件，`CURRENT>LAST` 极少成立，analyze 几乎不被调用，引号 bug 无从暴露；修复①让 analyze 开始运行后才浮现。两 bug 叠加。
- 该行本意是给良性 `vllm._C` ImportError 开白名单避免误判致命（FATAL 里有 `ImportError` 规则），意图正确但引号写法反成崩溃元凶。

**最终验证结论（2026-07 真机 SmolLM2）**：两 bug 修复后三选 `service_ok=true`、走到冒烟。但 SmolLM2-135M native 模式冒烟输出复读（"中国的首都是，但是…"答不出北京），冒烟正确拦截 → `v1_variant=none`。**这是模型真实能力表现，非 bug**：该模型强依赖 flaggems，native V1 无可用输出，回退 NV 基线是正确行为。至此 V1 判 none 基于真实就绪+真实冒烟，而非盲判。

相关：[[todo-v1-missing-perf-baseline]]（v1_available=false 触发合成基线）、[[v1-3-stale-service-false-negative]]
