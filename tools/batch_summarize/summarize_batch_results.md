# FlagOS 批量迁移结果汇总（输出模版）

> 本文件为汇总报告的格式参考模版，不包含真实运行数据。
> 实际报告由 `summarize.sh` 生成并输出到标准输出或指定文件。

---

## ⚠️ 达标判定规则（最高优先级，汇总时必须遵循）

> 新流程 v3.1：**V3 Max 为最终交付版本，已无 V5**（见 `run_pipeline.sh`："V5 算子扩展流程已移除"）。V4 Express 是 V3 之上的可选性能优化镜像，**不影响达标**。

**汇总表中所有"达标"列的判定标准**，仅由以下两个条件**同时满足**决定：

1. **V3 镜像已成功上传 Harbor**：`versions.v3.harbor_image` 非空，或 `traces/13_plugin_release.json` 中 harbor_push status=success
2. **V3 精度达标**：V3 镜像对应算子配置下，精度相对退化 ≤ 5%

**V3 精度取数规则**（按优先级）：
1. `context.yaml` 的 `versions.v3.accuracy_ok`（true/false）+ `eval.v3_score`（最直接的结构化结论）
2. `results/gpqa_v3.json`（回退 `results/gpqa_plugin.json`）的 `score` 字段，与 NV 基线（`nv_baseline.yaml` / `eval.nv_score`）或本地 V1 对比计算 rel_drop
3. `results/accuracy_compare.json` 的 `aligned`（bool，true=达标）与 `rel_drop`（**0~1 小数，×100 得百分比**）
4. 证据不足时，达标=否（无法确认视同未达标，但需在异常列注明"精度证据不足"）

rel_drop = (基线 - V3分数) / 基线 ≤ 5% 即精度达标；基线为本地 V1 或 NV 参考。注意 `accuracy_compare.json` 里 `rel_drop` 是小数（如 0.03=3%），字段名不是 `rel_drop_pct`。

**若产出 V4 Express 优化镜像**（`versions.v4.harbor_image` 非空）：
- V4 是 V3 之上的减算子性能优化，V4 精度≥V3 为其成立前提，故达标口径不变（仍看 V3 已上传 + V3 精度达标）
- 交付展示层可标注"最终交付=V4"，但达标判定不依赖 V4

**禁止使用以下中间状态作为模型"不达标"依据**：
- ❌ V2(注入)精度不达标 → 不代表模型不达标（V2 与 V3 是两套算子调度路径，V3 精度在步骤11单独判；只要 V3 已上传+V3 精度达标即达标）
- ❌ V3/V2 性能不达标（performance_ok=false）→ 性能从不阻断发布，不影响达标
- ❌ context.yaml 中 `plugin_workflow.accuracy_ok: false` 的历史中间标记 → 以 `versions.v3.accuracy_ok` / `accuracy_compare.json` 为准
- ❌ V4 减算子跳过/失败 → V3 已是最终交付版本，V4 不影响达标

**正确的判定数据来源**（按优先级）：
1. `shared/context.yaml` → `versions.v3.harbor_image` 字段非空 + `versions.v3.accuracy_ok=true`
2. `traces/13_plugin_release.json` → harbor_push status=success（含远端 curl tags/list 核验）
3. 批次日志中 `[步骤13] ... V3发布` 行显示 ✓ 完成 + Harbor 地址

**未达标的合法原因仅有**：
- V3 精度不达标（rel_drop > 5%）→ 仅推私有不合格交付镜像，判未达标
- V2 服务起不来（service_ok=false）→ 跳过 plugin，无 V3 镜像产出
- 流程中断/超时导致 V3 未能上传
- 网络问题导致 Harbor push 最终失败且无兜底成功

---

## 板块 A — 结果总表

### A1. 批次概览

```
批次时间: <start_time> ~ <end_time>
任务总数: <total> (已完成 <done>，进行中 <running>，待执行 <pending>)
达标: <pass_count>    完成(不达标): <fail_count>    失败/中断: <error_count>
GPU 厂商: <vendor> (<gpu_model> ×<count>)
已完成模型总耗时: <total_hours>
```

### A2. 主汇总表

> **"达标"列判定**：仅看 V3 镜像是否已上传 Harbor + V3 精度 rel_drop ≤ 5%。详见顶部「达标判定规则」。V4 为可选性能优化，不影响达标。

| # | 模型 | V1 | V2精度(得分/基线) | V2退化% | V2性能比 | V2调优 | V3精度(得分/退化%) | V3性能比 | V3调优 | V4(减算子优化) | 达标(V3上传+精度) | 问题标记 |
|---|------|----|--------------------|---------|----------|--------|---------------------|----------|--------|----------------|------|----------|
| 1 | <model_name> | <v1_status> | <score>/<baseline> | <rel_drop>% | <perf_ratio>% | <tuning_result> | <v3_score>%/<v3_rel_drop>% | <v3_perf>% | <v3_tuning> | <v4_status: 成功beats_v3/跳过/失败> | <Y/N> | <tags> |

"V3精度"列填写：
- 按上方「V3 精度取数规则」获取分数，计算 rel_drop = (基线 - V3分数) / 基线
- 优先取 `context.yaml versions.v3.accuracy_ok + eval.v3_score`，再取 `gpqa_v3.json`/`gpqa_plugin.json` 的 score

"V4(减算子优化)"列填写：
- **成功(beats_v3)** = V4 减算子后性能超越 V3 且精度达标，产出 -v4 镜像
- **跳过** = V3 精度不达标未触发 V4，或 V4 不适用
- **失败** = V4 流程执行但未成立（未超越 V3 / 精度回退 / 脚本错误）
- 注：V4 状态不影响"达标"列

"达标"列填写规则：
- **Y** = V3 Harbor 镜像已上传成功 且 V3 精度 rel_drop ≤ 5%
- **N** = V3 未上传 或 V3 精度不达标（service_ok=false / 流程中断 / 网络失败 / 精度超阈值）
- **N(进行中)** = 流程尚未执行到 V3

### A2.1 网络问题导致的失败

<如无网络失败则写"本批次无网络相关失败。">

### A3. 达标/阶段统计

逐模型 V1~V4 各阶段状态，标明是否由网络问题导致失败：

| # | 模型 | V1 | V2 | V3(交付版本, 精度+上传) | V4(减算子优化) | 达标 | 网络导致失败 |
|---|------|----|----|--------------------------|----------------|------|------------|
| 1 | <model> | <状态+原因> | <状态+原因> | <状态, 精度X%, rel_drop Y%, Harbor已上传/未上传> | <成功beats_v3/跳过/失败> | Y/N | 是/否 |

V3 列必须写明（V3 为最终交付版本）：
1. 状态（成功/失败/跳过+原因）
2. V3 精度分数与相对退化（rel_drop = (基线-分数)/基线）
3. Harbor 镜像是否已上传
4. 是否达标（V3 已上传 且 rel_drop ≤ 5%）

V4 列写明：成功(beats_v3产出-v4镜像) / 跳过(V3精度不达标未触发) / 失败(未成立)。V4 不影响达标。

总体统计：
- V2 精度达标率: <n>/<total> (<pct>%)
- V2 性能达标率: <n>/<total> (<pct>%)
- V3 发布成功率（V3 镜像已上传 Harbor）: <n>/<total> (<pct>%)
- **总体达标率（V3已上传+V3精度达标）**: <n>/<total> (<pct>%)
- V4 优化产出率（可选）: <n>/<total> (<pct>%)
- 网络导致失败: <count> 个模型

### A3.1 Harbor 镜像发布验证

数据来源：V2/V3 读各模型 `traces/08_release.json`（V2）、`traces/13_plugin_release.json`（V3，交付版本）的 harbor_push action；V4 无独立 release trace，读 `context.yaml` 的 `versions.v4.harbor_image`（或 `v4_reduction` 字段）判断 -v4 优化镜像是否产出。

| # | 模型 | V2 Harbor | V3 Harbor(交付) | V4 Harbor(优化) | 异常 |
|---|------|-----------|-----------------|-----------------|------|
| 1 | <model> | <status> <image_tag> | <status> <image_tag> | <status> <image_tag 或 "未产出"> | <异常描述或"无"> |

说明：
- V1 无发布步骤（基线测试阶段）
- V3 为最终交付版本；精度不达标时仍推送私有不合格交付镜像（flagrelease-project，不更新 README）
- "incompatible tag" 表示厂商 plugin 不适配但仍推送了不兼容标记镜像
- V4 为可选性能优化镜像，V3 精度不达标或 V4 未成立时为"未产出"
- 异常列标注"阶段成功但未上传"或"上传失败"的情况

### A4. 耗时汇总

按真实步骤名逐列统计：

| # | 模型 | 容器准备 | 环境检测 | 服务启动 | 精度评测 | 精度调优 | 性能评测 | 性能调优 | 打包发布 | Plugin安装 | Plugin服务 | Plugin精度 | Plugin性能 | Plugin发布 | V4减算子 | V4发布 | 总耗时 |
|---|------|---------|---------|---------|---------|---------|---------|---------|---------|-----------|-----------|-----------|-----------|-----------|---------|--------|--------|
| 1 | <model> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <total> |

汇总：
```
批次已完成耗时(wall-clock): <total>
平均每模型: <avg>
最长单模型: <model> <time>
```

### A5. 费用汇总

| # | 模型 | 段1(容器/检查/启动) | 段2(精度/性能) | 段3(打包发布V2) | 段4(Plugin V3) | 段5(V4优化) | 总计($) |
|---|------|---------------------|----------------|----------------|----------------|-------------|---------|
| 1 | <model> | $<cost> | $<cost> | $<cost> | $<cost> | $<cost 或 -> | $<total> |

汇总：
```
批次已完成费用: $<total>
平均每模型: $<avg>
最高单模型: <model> $<cost>
```

### A6. Issue 产出汇总

数据来源：`results/issue_data_*.json` 和 `results/issue_*_flagos-ai_*.md`

| # | 模型 | Issue 类型 | 目标组件 | 问题摘要 | 涉及算子 | 产出阶段 |
|---|------|-----------|---------|---------|---------|---------|
| 1 | <model> | <type: accuracy-degraded/plugin-error/performance-degraded> | <FlagGems/vllm-plugin-FL> | <摘要> | <算子列表> | <V阶段> |

按组件统计：
```
FlagGems:        <n> issues (<types>)
vllm-plugin-FL:  <n> issues (<types>)
```

---

## 板块 B — 逐模型流程详述

每个模型必须列出 V1~V4 全部四个阶段（新流程 v3.1 已无 V5）。未执行的阶段标注"跳过"并写明原因。V3 Max 为最终交付版本，V4 为可选性能优化。

---

### <model_name>

**环境**: 分支<branch> | GPU: <vendor> <gpu_model> ×<count> (TP=<tp>) | <env_type> | 总耗时 <time> | 费用 $<cost>

#### V1 基线阶段
- **状态**: <可用/不可用>
- **流程**: <V1三选结果描述>
- **根因**: <原因说明>

#### V2 Pro 阶段
- **状态**: <成功/失败(原因)>
- **精度**: V2=<score>% vs <基线类型>=<baseline>%，相对退化 <rel_drop>%，达标: <Y/N>
- **性能**: <throughput> tok/s vs 基线 <baseline_throughput> tok/s，ratio=<ratio>% >= 80%，达标: <Y/N>
- **流程**:
  - <逐步骤描述>
- **根因**: <总结>

#### V3 Max 阶段 (Plugin，⭐最终交付版本)
- **状态**: <成功/失败/跳过>
- **跳过原因**（如跳过）: <如"V2服务起不来service_ok=false，V3被跳过">
- **精度**: V3=<score>% vs 基线 <baseline>%，rel_drop=<X>%，达标: <Y/N>（数据源: context versions.v3.accuracy_ok+eval.v3_score / gpqa_v3.json(回退 gpqa_plugin.json) / accuracy_compare.json）
- **性能**: <throughput> tok/s，ratio=<ratio>%（性能不门控，不影响达标）
- **Harbor 上传**: <镜像 tag，flagrelease-project；达标则同步更新 ModelScope/HF README>
- **达标判定**: V3 已上传 <Y/N> 且 精度达标 <Y/N> → **模型达标: <Y/N>**
- **流程**: <描述>
- **根因**: <总结>

#### V4 Express 阶段 (减算子，可选性能优化)
- **状态**: <成功(beats_v3)/失败/跳过>
- **跳过原因**（如跳过）: <如"V3精度不达标，V4被跳过">
- **优化结果**（如成功）: 减算子 <reduced_ops>，保留 <kept_ops>，V4 性能比 <v4_ratio>% vs V3 <v3_ratio>%，是否超越 V3: <beats_v3>
- **V4 精度**: <score>% vs 基线 <baseline>%，rel_drop=<X>%（V4 精度达标为成立前提）
- **Harbor**: <V4 镜像 tag 或 "未产出">
- **说明**: V4 是 V3 之上的性能优化版，V3 已是最终交付版本，V4 状态不影响模型达标判定。

---

## 板块 C — 流程问题根因分析

### 1. 执行层问题

| 类别 | 影响模型数 | 描述 |
|------|-----------|------|
| 端口冲突 | <n>/<total> | <描述> |
| GPU占用 | <n>/<total> | <描述> |
| 网络问题 | <n>/<total> | <描述> |
| Plugin安装问题 | <n>/<total> | <描述> |
| 服务启动异常 | <n>/<total> | <描述> |

### 2. 业务层问题

#### 2.1 算子精度问题

| 算子 | 被禁模型数 | 涉及模型 | 根因分析 |
|------|-----------|---------|----------|
| <op_name> | <n> | <models> | <analysis> |

#### 2.2 算子性能问题

| 算子 | 影响模型数 | 涉及模型 | 根因分析 |
|------|-----------|---------|----------|
| <op_name> | <n> | <models> | <analysis> |

#### 2.3 算子兼容性/崩溃问题

| 算子 | 崩溃模型数 | 涉及模型 | 崩溃类型 | 根因分析 |
|------|-----------|---------|----------|----------|
| <op_name> | <n> | <models> | <crash_type> | <analysis> |

#### 2.4 基线问题

- <合成基线说明>
- <NV基线偏差说明>

#### 2.5 框架/Plugin 问题

- <问题描述及分析>

### 3. 算子禁用热力图（聚合统计）

| 算子名 | 被禁次数 | 涉及模型 | 主要原因 | 来源 |
|--------|----------|----------|----------|------|
| <op_name> | <count> | <models> | <reason> | <source_stage> |

---

数据采集时间: <timestamp>
批次状态: <status_description>
