# FlagRelease 通知与模型结果分析设计文档

> 更新时间：2026-08-12  
> 版本：v3.1（V3 Max 为最终交付版本）

## 📋 目录

1. [版本体系变更](#版本体系变更)
2. [达标判定逻辑](#达标判定逻辑)
3. [双 Tag 场景处理](#双-tag-场景处理)
4. [飞书通知设计](#飞书通知设计)
5. [模型结果分析流程](#模型结果分析流程)
6. [关键代码位置](#关键代码位置)

---

## 🔄 版本体系变更

### 旧版本体系（已废弃）

```
V1 基础 → V2 Pro → V3 → V4 → V5 最终交付
```

### 新版本体系（v3.1）

```
V1 基础版（FlagTree only）
  ↓
V2 Pro 版（gems+tree 达标版）
  ↓
V3 Max 版（gems+tree+plugin 达标版）← 最终交付版本
  ↓ [可选]
V4 Express 版（减算子提性能，≥V3，近/超 V1）← 性能优化版
```

### 版本定义

| 版本 | 标签 | 定义 | 镜像 tag 后缀 |
|------|------|------|--------------|
| **V1** | 基础版 | 仅 FlagTree，不开启 FlagGems | `-v1` |
| **V2** | Pro 版 | FlagGems + FlagTree，精度退化≤5%，性能≥80% V1 | `-v2` |
| **V3** | Max 版 | V2 + Plugin，精度退化≤5%，性能≥80% V1 | `-v3` |
| **V4** | Express 版 | V3 基础上减算子提性能，追求性能绝对值最大化，达标基准是超越 V3 | `-v4` |

**关键变更**：
- ❌ **已废弃**：V5 作为最终交付版本
- ✅ **V3 Max** 是最终交付版本
- ✅ **V4 Express** 是可选的性能优化，不改变达标口径

---

## ✅ 达标判定逻辑

### 核心公式

```python
qualified = (version is not None) AND (accuracy_ok is True) AND (uploaded is True)
```

**三要素全部以 V3 为准**：

| 字段 | 判定依据 | 说明 |
|------|---------|------|
| `accuracy_ok` | V3 的精度结论 | 即使产出了 V4，也只看 V3 精度 |
| `uploaded` | V3 的上传结果 | 即使产出了 V4，也只看 V3 是否上传 |
| `version` | 展示用途 | v4 表示额外产出了 V4 Express，但不影响达标 |

### 代码实现

**位置**：`tools/batch_summarize/model_result_analyzer.py:196-199`

```python
# 达标判定一律以 V3 为准：
# - accuracy_ok / uploaded 由 prompt 明确填为 V3 的精度与上传结论
# - version 是否为 v4 不改变达标口径（V4 仅为 V3 之上的性能优化）
# - V3 是最终交付版，无 V3 即无交付
qualified = version is not None and accuracy_ok is True and uploaded is True
```

### 判定示例

| 场景 | version | accuracy_ok | uploaded | qualified | 显示 |
|------|---------|-------------|----------|-----------|------|
| 仅 V3 达标 | v3 | True | True | ✅ True | V3 Max 达标上传 |
| V3+V4 达标 | v4 | True | True | ✅ True | V4 Express 达标上传 |
| V3 精度不达标 | v3 | False | True | ❌ False | V3 Max 未达标 |
| V3 未上传 | v3 | True | False | ❌ False | V3 Max 未上传 |
| 只有 V4 无 V3 | v4 | True | False | ❌ False | 无有效交付（V3 是基准）|

---

## 🏷️ 双 Tag 场景处理

### 触发条件（分支 B）

**场景**：`gems_tree_plugin` 环境下，V1 基线无法获取时

```
V1 三选状态机：
  v1.1：VLLM_PLUGINS='' 纯净基线
  v1.2：厂商 platform plugin
  v1.3：fl plugin 但不开 flaggems
  
触发条件：
  - V1 全部失败 → baseline_selector 返回 none
  - 或仅 v1.3 成功（fl plugin 不开 flaggems 作为基线）
```

### 双 Tag 行为

```bash
# 步骤 8 发布时
publish.py --also-tag v3

# 结果：同一个物理镜像，同时打上两个 tag：
harbor.baai.ac.cn/flagrelease-public/Model-vendor-FlagOS:202608121234-v2
harbor.baai.ac.cn/flagrelease-public/Model-vendor-FlagOS:202608121234-v3
```

### Context 数据结构

```yaml
versions:
  v2:
    harbor_image: "harbor.../Model:202608121234-v2"
    also_tagged_v3: true  # 标记同时打了 v3 tag
  v3:
    harbor_image: ""  # 可能为空（因为在步骤 8 发布，不是步骤 13）
    same_as_v2: true  # 标记 V3=V2 同镜像
```

### 上传判定逻辑

**位置**：`tools/batch_summarize/summarize_model_result.md:18`

```markdown
4. **上传成功一律以 V3 镜像为准**（达标只看 V3）。
   
   判定顺序：
   1. 优先查找 versions.v3.harbor_image 非空
   2. 或 traces/13_plugin_release.json 中 harbor_push status=success
   3. **双 tag 场景**：
      - versions.v2.harbor_image 包含 -v3 tag
      - 或 traces/08_release.json 显示双 tag 上传成功
      - 或 versions.v2.also_tagged_v3 = true
   
   → 以上任一满足则 uploaded=true
   → 证据不足时 uploaded=null
   → 明确失败时 uploaded=false
```

### 实际案例

```json
{
  "versions": {
    "v2": {
      "harbor_image": "harbor.../Model:202608121234-v2",
      "also_tagged_v3": true
    },
    "v3": {
      "harbor_image": "",
      "same_as_v2": true
    }
  }
}
```

**Claude 分析输出**：
```json
{
  "delivery": {
    "version": "v3",
    "accuracy_ok": true,
    "uploaded": true
  },
  "evidence": {
    "upload": [
      "traces/08_release.json (双 tag 上传成功)",
      "versions.v2.harbor_image 包含 -v2 和 -v3 tag"
    ]
  }
}
```

---

## 📢 飞书通知设计

### "达标优先"显示逻辑

**位置**：`tools/notifications/progress_summary.py:568-575`

```python
def status_for_model(model: Dict[str, Any]) -> str:
    # 达标优先：已达标上传即视为 success，即使 pipeline 后段失败/超时
    # （交付事实已成立，异常细节由 model_result_label 的标注体现）
    if model.get("qualified_uploaded") is True:
        return "success"
    # ... 其他情况
```

### 显示逻辑示例

| 时间线 | qualified | outcome | 显示标签 | 状态 |
|--------|-----------|---------|---------|------|
| V3 推送成功 → 后续超时 | True | timeout | ✅ V3 Max 达标上传（流程超时）| success |
| V3 推送成功 → 后续失败 | True | failed | ✅ V3 Max 达标上传（流程异常）| success |
| V3 推送成功 → 正常结束 | True | success | ✅ V3 Max 达标上传 | success |
| V3 精度不达标 | False | success | ⚠️ V3 Max 未达标 | warning |
| V3 未上传 | False | success | ⚠️ V3 Max 未上传 | warning |
| 流程失败，无交付 | False | failed | ❌ 流程失败 | failed |

### 批量成功率计算

**位置**：`tools/notifications/progress_summary.py:297-299`

```python
qualified = [item for item in models if item.get("qualified_uploaded") is True]
success_rate_pct = round(len(qualified) / processed * 100.0, 1) if processed else 0.0
```

**公式**：成功率 = 达标上传模型数 / 已处理模型数

---

## 🔬 模型结果分析流程

### 架构

```
run_pipeline.sh 完成
  ↓
progress_worker.py 异步调用
  ↓
model_result_analyzer.py
  ↓
claude --json-schema (结构化输出)
  ↓
读取 summarize_model_result.md (Prompt)
  ↓
扫描模型目录：
  - config/context*.yaml
  - results/gpqa_*.json
  - results/*_performance.json
  - traces/*.json
  - logs/*.log
  ↓
输出结构化 JSON：
  {
    "cost": {...},
    "delivery": {
      "version": "v3" | "v4" | null,
      "accuracy_ok": bool | null,
      "uploaded": bool | null
    },
    "notification": {...},
    "evidence": {...}
  }
  ↓
验证并保存到 sidecar 结果文件
  ↓
progress_summary.py 生成飞书卡片
```

### Claude 分析的职责边界

**负责**：
- 判定最终交付版本（V3/V4）
- 判定 V3 精度是否达标
- 判定 V3 镜像是否上传成功
- 统计迁移费用
- 提供简短的一句话结论
- 列出支持结论的证据路径

**不负责**（由外层执行器强制覆盖）：
- 流程 outcome（success/failed/timeout/skipped）
- 退出码
- 墙钟时间
- 厂商识别

### 精度判定规则

**位置**：`summarize_model_result.md:17`

```markdown
3. **精度结论一律以 V3 为准**。
   
   判定顺序：
   1. 优先使用 versions.v3.accuracy_ok（明确字段）
   2. 再用 results/gpqa_v3.json 的 score
   3. 或 results/accuracy_compare.json 的 aligned/rel_drop
   4. 根据评测证据和相对退化≤5% 规则判断
   5. 证据不足 → accuracy_ok=null
   
   注意：即使最终交付为 V4，accuracy_ok 也填 V3 精度结论
```

### 上传判定规则

**位置**：`summarize_model_result.md:18`（已更新支持双 tag）

```markdown
4. **上传成功一律以 V3 镜像为准**。
   
   标准场景：
   - versions.v3.harbor_image 非空
   - traces/13_plugin_release.json 中 harbor_push status=success
   
   双 tag 场景（V2=V3）：
   - versions.v2.harbor_image 包含 -v3 tag
   - traces/08_release.json 显示双 tag 上传成功
   - versions.v2.also_tagged_v3 = true
   
   结论：
   - 以上任一满足 → uploaded=true
   - 证据不足 → uploaded=null
   - 明确失败 → uploaded=false
```

---

## 📍 关键代码位置

### 核心文件

| 文件 | 职责 | 关键逻辑 |
|------|------|---------|
| `tools/batch_summarize/model_result_analyzer.py` | 调用 Claude 分析，验证结构 | `qualified` 判定（行 196-199）|
| `tools/batch_summarize/summarize_model_result.md` | Claude 分析 Prompt | 精度/上传判定规则（行 17-18）|
| `tools/notifications/progress_summary.py` | 生成飞书卡片 | 达标优先逻辑（行 568-575）|
| `tools/notifications/progress_worker.py` | 事件消费与后台分析 | 异步调用分析器 |
| `tools/notifications/feishu_notify.py` | 飞书消息发送 | 关键词白名单、卡片格式 |
| `shared/generate_report.py` | 生成详细报告 | 版本定义（行 770-775）|

### 版本定义

**位置**：`shared/generate_report.py:770-775`

```python
VERSION_LABELS = {
    "v1": ("V1", "-", "基础版(FlagTree only)"),
    "v2": ("V2", "Pro", "gems+tree达标版"),
    "v3": ("V3", "Max", "gems+tree+plugin达标版"),
    "v4": ("V4", "Flag-express", "减算子提性能版(≥V3,近/超V1)"),
}
```

### 判定逻辑

**位置**：`tools/batch_summarize/model_result_analyzer.py:189-199`

```python
version = str(raw_version or "").lower() or None
# 新流程 v3.1：交付版本为 V3 Max，或其上的 V4 Express；已无 V5
if version not in {None, "v3", "v4"}:
    raise RuntimeError(f"单模型分析结果 delivery.version 无效")

accuracy_ok = delivery.get("accuracy_ok")  # V3 精度
uploaded = delivery.get("uploaded")        # V3 上传

# 达标判定
qualified = version is not None and accuracy_ok is True and uploaded is True
```

---

## ✅ 验证清单

### Prompt 更新

- [x] `summarize_model_result.md` 已更新双 tag 场景判定逻辑
- [x] 明确 V3 为达标基准
- [x] 明确 V4 不改变达标口径

### 代码实现

- [x] `model_result_analyzer.py` 达标判定逻辑正确（以 V3 为准）
- [x] `progress_summary.py` 达标优先显示逻辑正确
- [x] `generate_report.py` 版本定义已更新（V3 Max / V4 Express）

### 测试场景

需要验证的场景：
- [ ] 标准 V3 达标（独立 V3 镜像）
- [ ] V3 + V4 双版本达标
- [ ] 双 tag 场景（V2=V3 同镜像）
- [ ] V3 精度不达标但上传成功
- [ ] V3 精度达标但未上传
- [ ] V3 达标但后续流程超时
- [ ] 只有 V4 无 V3（应判定为无有效交付）

---

## 🔧 故障排查

### 常见问题

1. **双 tag 场景误判为"未上传"**
   - 原因：只检查 `versions.v3.harbor_image`，忽略了 V2=V3 场景
   - 解决：Claude 分析时同时检查 `versions.v2.harbor_image` 和 `-v3` tag

2. **V4 存在时误判达标口径**
   - 原因：错误地认为 V4 是最终交付版本
   - 解决：明确 V3 是达标基准，V4 只是性能优化

3. **流程超时导致"失败"显示**
   - 原因：未实现"达标优先"逻辑
   - 解决：`qualified_uploaded=True` 时优先显示成功，再附加异常标注

---

## 📚 参考资料

- 工作流定义：`CLAUDE.md` 行 99-137
- 分支 B 双 tag 说明：`CLAUDE.md` 行 150-164
- 飞书通知设计：`tools/notifications/README.md`
- 版本体系说明：`CLAUDE.md` 行 139-148

---

**文档维护者**：FlagRelease 团队  
**最后更新**：2026-08-12  
**版本**：v3.1
