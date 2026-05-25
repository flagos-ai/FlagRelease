# FlagOS 产出文件字段参考

> 所有产出文件的字段含义集中索引。各文件内部也包含 `_meta` 或 `#` 注释。

---

## 1. /flagos-workspace/shared/context.yaml — 工作流共享上下文（容器内，每个任务独立）

Skill 间共享状态，每步读写各自负责的字段。YAML 文件内已有 `#` 行内注释。项目目录下的 `shared/context.template.yaml` 为初始化模板。

| Section | 字段 | 说明 |
|---------|------|------|
| `container` | `name` | 容器名称或 ID |
| | `status` | 容器状态: running / stopped / not_found |
| `entry` | `source` | 来源标识（镜像地址或已有容器名） |
| | `type` | 入口类型: new_container / existing_container |
| `environment` | `env_type` | 场景类型: native / vllm_flaggems / vllm_plugin_flaggems |
| | `flaggems_code_path` | FlagGems import 所在源码路径 |
| | `flaggems_enable_call` | FlagGems enable() 调用签名 |
| | `flaggems_txt_path` | FlagGems 算子列表 txt 路径 |
| | `gems_txt_auto_detect` | 是否通过自动搜索找到 gems.txt |
| | `has_flagtree` | 是否安装 FlagTree |
| | `has_plugin` | 是否存在 vllm plugin |
| `eval` | `v1_score` | V1 (native) GPQA Diamond 正确率 (%) |
| | `v2_score` | V2 (flaggems) GPQA Diamond 正确率 (%) |
| | `deviation` | V1 vs V2 精度下降百分点（正值=V2低于V1） |
| | `threshold` | 精度下降阈值（百分点，默认 5.0） |
| | `accuracy_aligned` | 精度是否达标（下降 <= threshold，V2高于V1也算达标） |
| `gpu` | `count` | GPU 数量 |
| | `type` | GPU 型号 |
| | `vendor` | GPU 厂商: nvidia / ascend |
| | `visible_devices_env` | 指定卡的环境变量名（如 CUDA_VISIBLE_DEVICES、ASCEND_RT_VISIBLE_DEVICES） |
| `model` | `name` | HuggingFace 模型标识 |
| | `container_path` | 容器内模型路径 |
| | `local_path` | 宿主机模型路径 |
| `perf` | `strategy` | 测试策略: quick / fast / comprehensive / fixed |
| | `test_case` | 测试用例名称 |
| | `v1_native.c1_output_tps` | V1 并发 1 输出吞吐 (tok/s) |
| | `v1_native.max_output_tps` | V1 最大并发输出吞吐 (tok/s) |
| | `v2_flaggems.*` | V2 对应指标，结构同 v1_native |
| `performance` | `min_ratio` | V2/V1 最小吞吐比 (%)，达标判定依据 |
| | `target_ratio` | 达标阈值 (%)，默认 80.0 |
| | `pass` | 是否达标（min_ratio >= target_ratio） |
| `runtime` | `framework` | 推理框架: vllm / sglang |
| | `tp_size` | 张量并行度 |
| | `flaggems_enabled` | FlagGems 当前是否启用 |
| | `thinking_model` | 是否为 thinking 模型 |
| `service` | `port` | 服务端口 |
| | `healthy` | 服务是否健康 |
| | `max_model_len` | 模型最大上下文长度 |
| | `enable_oplist_count` | 当前生效算子数 |
| `workflow` | `service_ok` | 服务启动是否成功 |
| | `accuracy_ok` | 精度是否达标 |
| | `performance_ok` | 性能是否达标 |
| | `qualified` | 综合判定（service_ok AND accuracy_ok AND performance_ok） |
| | `all_done` | 全流程是否完成 |
| `workflow_ledger.steps[]` | `name` | 步骤名称 |
| | `status` | pending / in_progress / success / failed / skipped |
| | `started_at` | 开始时间 (ISO 8601) |
| | `finished_at` | 结束时间 (ISO 8601) |
| | `duration_seconds` | 耗时（秒） |
| | `notes` | 关键结果摘要 |
| | `fail_reason` | 失败原因（status=failed 时） |
| | `skip_reason` | 跳过原因（status=skipped 时） |

---

## 2. traces/*.json — 步骤留痕

每个步骤完成后写入，记录实际执行的命令和关键输出。JSON 内含 `_meta` 字段。

| 字段 | 说明 |
|------|------|
| `step` | 步骤编号（如 01_container_preparation） |
| `title` | 步骤中文名称 |
| `timestamp_start` | 步骤开始时间 (ISO 8601) |
| `timestamp_end` | 步骤结束时间 (ISO 8601) |
| `duration_seconds` | 步骤耗时（秒） |
| `status` | 执行状态: success / failed / skipped |
| `actions[]` | 该步骤中执行的关键操作列表 |
| `actions[].action` | 操作标识（如 docker_run / v1_eval / compare） |
| `actions[].command` | 实际执行的完整命令字符串 |
| `actions[].timestamp` | 操作执行时间 (ISO 8601) |
| `actions[].status` | 操作状态: success / failed |
| `actions[].output_summary` | 关键输出摘要（非全量 stdout） |
| `result_files` | 该步骤产出的结果文件路径 |
| `context_updates` | 该步骤写入 context.yaml 的字段及其值 |

---

## 3. results/gpqa_*.json — 精度评测结果

由 `fast_gpqa.py` 生成。JSON 内含 `_meta` 字段。

| 字段 | 说明 |
|------|------|
| `model` | 模型名称或路径 |
| `benchmark` | 评测基准名称（固定 gpqa_diamond） |
| `mode` | 评测模式: standard（普通模型）/ thinking（思维链模型） |
| `score` | GPQA Diamond 正确率百分比 |
| `total_questions` | 评测题目总数（GPQA Diamond 固定 198 题） |
| `eval_batch_size` | 评测并发数（自动探测选择） |
| `max_tokens` | 单次生成最大 token 数 |
| `max_model_len` | 模型支持的最大上下文长度 |
| `truncation_detected` | 是否检测到输出被截断（true 时分数可能偏低） |
| `temperature` | 采样温度（0.0=贪心解码） |
| `probe_time_seconds` | 并发探测阶段耗时（秒） |
| `eval_duration_seconds` | 实际评测阶段耗时（秒） |
| `total_duration_seconds` | 总耗时（含探测，秒） |
| `work_dir` | evalscope 原始输出目录（含预测、报告、日志） |

---

## 4. results/*_performance.json — 性能测试结果

由 `benchmark_runner.py` 生成。JSON 内含 `_meta` 字段。

结构：`{ "test_case_name": { "concurrency_level": { metrics }, ... }, "_timing": {...}, "_meta": {...} }`

### 每个并发级别的指标

| 字段 | 说明 |
|------|------|
| `Successful requests` | 成功完成的请求数 |
| `Failed requests` | 失败的请求数 |
| `Benchmark duration (s)` | 该并发级别测试总耗时（秒） |
| `Total input tokens` | 所有请求的输入 token 总数 |
| `Total generated tokens` | 所有请求的生成 token 总数 |
| `Request throughput (req/s)` | 请求吞吐量（每秒完成的请求数） |
| `Output token throughput (tok/s)` | 输出 token 吞吐量，**性能对比的核心指标** |
| `Total token throughput (tok/s)` | 总 token 吞吐量（input + output） |
| `Peak output token throughput (tok/s)` | 峰值输出 token 吞吐量 |
| `Peak concurrent requests` | 峰值并发请求数 |
| `Mean TTFT (ms)` | 平均首 token 延迟（Time To First Token） |
| `Median TTFT (ms)` | 中位数首 token 延迟 |
| `P99 TTFT (ms)` | 99 分位首 token 延迟 |
| `Mean TPOT (ms)` | 平均每 token 生成延迟（Time Per Output Token） |
| `Median TPOT (ms)` | 中位数每 token 生成延迟 |
| `P99 TPOT (ms)` | 99 分位每 token 生成延迟 |
| `Mean ITL (ms)` | 平均 token 间延迟（Inter-Token Latency） |
| `Median ITL (ms)` | 中位数 token 间延迟 |
| `P99 ITL (ms)` | 99 分位 token 间延迟 |

### `_timing` 字段

| 字段 | 说明 |
|------|------|
| `total_seconds` | 全部测试用例总耗时（秒） |
| `per_test_case` | 每个测试用例的耗时（秒） |
| `timestamp_start` | 测试开始时间 (ISO 8601) |
| `timestamp_end` | 测试结束时间 (ISO 8601) |

---

## 5. results/final_report.json — 最终报告

由编排层生成，汇总精度和性能结果。

| 字段 | 说明 |
|------|------|
| `report_version` | 报告格式版本 |
| `generated_at` | 报告生成时间 (ISO 8601) |
| `model` | 模型路径 |
| `container` | 容器名称 |
| `accuracy.benchmark` | 精度评测基准名称 |
| `accuracy.total_questions` | 评测题目总数 |
| `accuracy.v1_native.score_pct` | V1 正确率 (%) |
| `accuracy.v1_native.duration_s` | V1 评测耗时（秒） |
| `accuracy.v2_flagos.score_pct` | V2 正确率 (%) |
| `accuracy.v2_flagos.duration_s` | V2 评测耗时（秒） |
| `accuracy.diff_pct` | V1 vs V2 偏差百分点 |
| `accuracy.threshold_pct` | 偏差阈值百分点 |
| `accuracy.pass` | 精度是否达标 |
| `performance.test_case` | 性能测试用例名称 |
| `performance.strategy` | 测试策略 |
| `performance.v1_native.output_tps` | V1 输出吞吐 (tok/s) |
| `performance.v1_native.total_tps` | V1 总吞吐 (tok/s) |
| `performance.v1_native.ttft_ms` | V1 首 token 延迟 (ms) |
| `performance.v1_native.tpot_ms` | V1 每 token 延迟 (ms) |
| `performance.v2_flagos.*` | V2 对应指标，结构同 v1_native |
| `performance.ratio_pct` | V2/V1 吞吐比 (%) |

---

## 6. logs/_last_error.json — 最近一次错误

由 `error_writer.py` 生成，每次错误覆盖写入。JSON 内含 `_meta` 字段。

| 字段 | 说明 |
|------|------|
| `tool` | 产生错误的工具脚本名 |
| `timestamp` | 错误发生时间 (ISO 8601) |
| `exit_code` | 进程退出码（非零表示异常） |
| `error_type` | 错误分类（如 KeyError / service_unreachable / timeout） |
| `error_message` | 错误详情 |
| `traceback` | Python 完整堆栈（可选） |
| `partial_result` | 错误发生前已产出的部分结果（可选） |
| `context` | 错误发生时的上下文信息（可选） |

---

## 7. logs/checkpoint.json — 执行检查点

由 `error_writer.py` 生成，记录当前正在执行的操作，用于中断恢复。JSON 内含 `_meta` 字段。

| 字段 | 说明 |
|------|------|
| `step` | 当前步骤编号（如 04_accuracy_eval） |
| `step_name` | 步骤中文名 |
| `action` | 当前正在执行的操作标识 |
| `started_at` | 检查点创建时间 (ISO 8601) |
| `updated_at` | 检查点最后更新时间 (ISO 8601) |
| `pid` | 执行进程 ID |
| `action_detail` | 操作详情，如完整命令（可选） |
| `last_success` | 上一个成功完成的操作信息（可选） |

---

## 8. logs/issues_*.log — 问题日志

纯文本格式，追加写入。三个文件：

| 文件 | 记录内容 |
|------|---------|
| `issues_startup.log` | 服务启动失败、崩溃、超时 |
| `issues_accuracy.log` | V2精度下降 >5%、评测报错 |
| `issues_performance.log` | 任一并发级别 V2/V1 < 80% |

每条记录格式：

```
[YYYY-MM-DD HH:MM:SS] <版本(V1/V2)> | <问题摘要>
  详情: <错误信息/数值/不达标指标>
  操作: <采取的措施>
  结果: <措施结果>
```

---

## 9. logs/pipeline.log — 全流程执行日志

由 `stream_filter.py` 从 Claude 输出自动提取，面向人的实时概览。

关键格式：
- 时间戳：`[YYYY-MM-DD HH:MM:SS]`
- 步骤标记：`[步骤1]` ~ `[步骤8]`
- 事件关键词：`开始` / `完成` / `失败` / `跳过` / `异常`
- 结果标记：`✓`（成功）/ `✗`（失败）/ `⚠`（警告）
