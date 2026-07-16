# FlagOS 批量迁移结果汇总（输出模版）

> 本文件为汇总报告的格式参考模版，不包含真实运行数据。
> 实际报告由 `summarize.sh` 生成并输出到标准输出或指定文件。

---

## ⚠️ 达标判定规则（最高优先级，汇总时必须遵循）

**汇总表中所有"达标"列的判定标准**，仅由以下两个条件**同时满足**决定：

1. **V5 镜像已成功发布**：`versions.v5.harbor_image` 非空，或 `traces/15_v5_publish.json` / `traces/15_v5_release.json` 中 harbor_push status=success
2. **V5 精度达标**：V5 镜像对应算子配置下，精度相对退化 ≤ 5%

**V5 精度取数规则**（context.yaml 中 `eval.v5_score` 可能为 0，需要推断）：

按优先级取数：
1. `results/gpqa_v5.json` 中的 `score` 字段（V5 独立评测结果）
2. `eval.v5_score`（context.yaml，若 > 0）
3. `results/operator_config_v5.json` 中的 `v5_score`（若 > 0）
4. **Fallback 推断**：当上述均为 0 或不存在时，根据 V5 配置来源推断：
   - 若 `v5_expansion.note = "no_disabled_ops_v5_equals_v3"` 或无禁用算子 → **V5 精度 = V2 精度**（`eval.v2_score`），因为算子集未变化
   - 若 V5 扩展了 0 个算子（所有被禁算子仍不兼容）→ **V5 精度 = V2 精度**（算子集等于调优后的 V2）
   - 若 V5 成功扩展了部分算子 → `operator_expansion.py` 逐步验证了每步 ≤ 5%，精度达标

**V5 配置来源判定**：
- `no_disabled_ops_v5_equals_v3` → V5 = V2 配置（无算子被禁过）
- 有禁用但扩展 0 个成功 → V5 = V2 调优后配置（禁用算子不变）
- 有禁用且部分/全部扩展成功 → V5 = V2 + 重新启用的算子

**禁止使用以下中间状态作为模型"不达标"依据**：
- ❌ V3 Plugin 精度不达标 → 不代表模型任务不达标（V5 可能以 V2 配置发布，只要 V5 精度满足即达标）
- ❌ V3 性能不达标（performance_ok=false）→ 性能从不阻断发布
- ❌ context.yaml 中 `plugin_workflow.accuracy_ok: false` → 这是 V3 阶段的中间标记，不代表最终 V5 不达标
- ❌ V4 跳过/失败 → 不影响 V5 发布

**正确的判定数据来源**（按优先级）：
1. `shared/context.yaml` → `versions.v5.harbor_image` 字段非空
2. `traces/15_v5_publish.json` 或 `traces/15_v5_release.json` → status=success
3. 批次日志中 `[步骤15] V5发布` 行显示 ✓ 完成 + Harbor 地址

**未达标的合法原因仅有**：
- V2 精度不达标（`qualified_core=false`）→ V3/V4/V5 全部跳过，无 V5 镜像产出
- 流程中断/超时导致 V5 未能发布
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

> **"达标"列判定**：仅看 V5 镜像是否已发布到 Harbor + V5 精度 rel_drop ≤ 5%。详见顶部「达标判定规则」。

| # | 模型 | V1 | V2精度(得分/基线) | V2退化% | V2性能比 | V2调优 | V3精度 | V3性能比 | V3调优 | V4 | V5配置来源 | V5精度(得分/退化%) | 达标 | 问题标记 |
|---|------|----|--------------------|---------|----------|--------|--------|----------|--------|----|----|------|------|----------|
| 1 | <model_name> | <v1_status> | <score>/<baseline> | <rel_drop>% | <perf_ratio>% | <tuning_result> | <v3_score> | <v3_perf> | <v3_tuning> | <v4_status> | <V5=V2/V5=V2+扩展N个/跳过> | <score>%/<rel_drop>% | <Y/N> | <tags> |

"V5配置来源"列填写：
- **V5=V2** = 无禁用算子或扩展0个成功，V5与V2算子集相同
- **V5=V2+N** = V5在V2基础上成功重新启用了N个算子
- **跳过** = qualified_core=false，V5未触发

"V5精度"列填写：
- 按上方「V5 精度取数规则」获取分数，计算 rel_drop = (基线 - V5分数) / 基线
- 若为 fallback 推断（V5=V2），标注 "=V2" 表明精度继承自 V2

"达标"列填写规则：
- **Y** = V5 Harbor 镜像已推送成功 且 V5 精度 rel_drop ≤ 5%（按上方「V5 精度取数规则」确认）
- **N** = V5 未发布（qualified_core=false / 流程中断 / 网络失败）
- **N(进行中)** = 流程尚未执行到 V5

### A2.1 网络问题导致的失败

<如无网络失败则写"本批次无网络相关失败。">

### A3. 达标/阶段统计

逐模型 V1~V5 各阶段状态，标明是否由网络问题导致失败：

| # | 模型 | V1 | V2 | V3 | V4 | V5(配置来源+精度) | 达标 | 网络导致失败 |
|---|------|----|----|----|----|-------------------|------|------------|
| 1 | <model> | <状态+原因> | <状态+原因> | <状态+原因> | <状态+原因> | <V5=V2/V5=V2+N, 精度X%, rel_drop Y%> | Y/N | 是/否 |

V5 列必须写明：
1. 配置来源（V5=V2 / V5=V2+扩展N个 / 跳过）
2. V5 精度分数（实测或继承）
3. 相对退化百分比（rel_drop = (基线-分数)/基线）
4. 是否达标（rel_drop ≤ 5%）

总体统计：
- V2 精度达标率: <n>/<total> (<pct>%)
- V2 性能达标率: <n>/<total> (<pct>%)
- V5 发布成功率: <n>/<total> (<pct>%)
- **总体达标率（V5已发布+V5精度达标）**: <n>/<total> (<pct>%)
- 网络导致失败: <count> 个模型

### A3.1 Harbor 镜像发布验证

数据来源：各模型 `traces/08_release.json`、`traces/13_plugin_publish.json`（或 `13_v3_release.json`）、`traces/15_v5_publish.json` 中的 harbor_push action。

| # | 模型 | V2 Harbor | V3 Harbor | V5 Harbor | 异常 |
|---|------|-----------|-----------|-----------|------|
| 1 | <model> | <status> <image_tag> | <status> <image_tag> | <status> <image_tag> | <异常描述或"无"> |

说明：
- V1 无发布步骤（基线测试阶段）
- "incompatible tag" 表示 V3 精度不达标但仍推送了不兼容标记镜像
- 异常列标注"阶段成功但未上传"或"上传失败"的情况

### A4. 耗时汇总

按真实步骤名逐列统计：

| # | 模型 | 容器准备 | 环境检测 | 服务启动 | 精度评测 | 精度调优 | 性能评测 | 性能调优 | 打包发布 | Plugin安装 | Plugin服务 | Plugin精度 | Plugin性能 | Plugin发布 | V4减算子 | V5扩展 | V5发布 | 总耗时 |
|---|------|---------|---------|---------|---------|---------|---------|---------|---------|-----------|-----------|-----------|-----------|-----------|---------|--------|--------|--------|
| 1 | <model> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <time> | <total> |

汇总：
```
批次已完成耗时(wall-clock): <total>
平均每模型: <avg>
最长单模型: <model> <time>
```

### A5. 费用汇总

| # | 模型 | 段1(容器/检查/启动) | 段2(精度/性能) | 段3(打包发布V2) | 段4(Plugin V3) | 段5(V5) | 总计($) |
|---|------|---------------------|----------------|----------------|----------------|---------|---------|
| 1 | <model> | $<cost> | $<cost> | $<cost> | $<cost> | $<cost> | $<total> |

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

每个模型必须列出 V1~V5 全部五个阶段。未执行的阶段标注"跳过"并写明原因。

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

#### V3 Max 阶段 (Plugin)
- **状态**: <成功/失败/跳过>
- **跳过原因**（如跳过）: <如"V2精度不达标，V3被跳过">
- **精度**: <描述>
- **性能**: <描述>
- **流程**: <描述>
- **根因**: <总结>

#### V4 Express 阶段 (减算子)
- **状态**: <成功/失败/跳过>
- **跳过原因**（如跳过）: <如"V3精度不达标，V4被跳过">

#### V5 Royal 阶段 (应开尽开)
- **状态**: <成功/跳过>
- **跳过原因**（如跳过）: <如"V2精度不达标，V5被跳过">
- **配置来源**: <V5=V2(无禁用算子) / V5=V2(扩展0个,N个不兼容) / V5=V2+扩展M个算子>
- **V5精度**: <score>% vs 基线 <baseline>%，rel_drop=<X>%，达标: <Y/N>
  - 数据来源: <gpqa_v5.json / 继承V2(算子集不变) / operator_config_v5.json>
- **流程**: <描述>
- **V5=V2原因**（如适用）: <解释为何V5等于V2配置>

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
