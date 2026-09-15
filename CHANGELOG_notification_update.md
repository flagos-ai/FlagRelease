# 通知与结果分析模块更新日志

## 2026-08-12 - 双 Tag 场景判定逻辑修复

### 🎯 问题背景

在分支 B（`gems_tree_plugin`）环境下，当 V1 基线无法获取时（全部失败或仅 v1.3 成功），会触发 V2=V3 双 tag 同镜像发布：
- 步骤 8 使用 `--also-tag v3` 同时打 `-v2` 和 `-v3` tag
- 同一个物理镜像有两个 tag：`202608121234-v2` 和 `202608121234-v3`
- `versions.v3.harbor_image` 可能为空（因为在步骤 8 发布，不是步骤 13）
- `versions.v2.harbor_image` 包含实际镜像地址

### ⚠️ 原有问题

**模型结果分析 Prompt** 只检查 `versions.v3.harbor_image`：

```markdown
# 旧版 summarize_model_result.md:18
4. 上传成功一律以 V3 镜像为准。
   必须找到 V3 的 Harbor 镜像地址（versions.v3.harbor_image 非空）
   或 traces/13_plugin_release.json 中 harbor_push status=success
```

**导致问题**：
- 双 tag 场景下 `versions.v3.harbor_image` 为空
- Claude 分析判定 `uploaded=null` 或 `uploaded=false`
- 实际上 V3 镜像已成功上传（只是在 V2 位置且带双 tag）
- 最终导致 `qualified=false`，误报"未达标"

### ✅ 修复方案

**更新文件**：`tools/batch_summarize/summarize_model_result.md`

**新增双 tag 判定逻辑**：

```markdown
# 新版 summarize_model_result.md:18
4. **上传成功一律以 V3 镜像为准**（达标只看 V3）。
   
   优先查找 versions.v3.harbor_image 非空
   或 traces/13_plugin_release.json 中 harbor_push status=success；
   
   **特殊场景**：V2=V3 双 tag 同镜像时（分支 B 无独立 V1 或仅 v1.3 成功），
   V3 镜像在步骤 8 发布且 versions.v2.harbor_image 同时带 -v3 后缀，
   此时若 versions.v2.harbor_image 包含 -v3 tag
   或 traces/08_release.json 显示双 tag 上传成功，
   也判定 uploaded=true；
   
   证据不足时 uploaded=null，明确失败时 uploaded=false。
   即使额外产出了 V4 镜像，uploaded 仍只反映 V3 是否上传成功。
```

### 🔍 判定逻辑（Claude 分析时）

```
标准场景（独立 V3）：
  versions.v3.harbor_image 非空 → uploaded=true
  traces/13_plugin_release.json harbor_push success → uploaded=true

双 tag 场景（V2=V3）：
  versions.v2.harbor_image 包含 -v3 tag → uploaded=true
  traces/08_release.json 双 tag 上传成功 → uploaded=true
  versions.v2.also_tagged_v3 = true → uploaded=true

证据不足：
  以上均不满足 → uploaded=null

明确失败：
  traces 中有失败记录 → uploaded=false
```

### 📝 相关文件

**已修改**：
- `tools/batch_summarize/summarize_model_result.md` - 新增双 tag 判定规则

**未修改（无需修改）**：
- `tools/batch_summarize/model_result_analyzer.py` - 只验证结构，不做业务判定
- `tools/notifications/progress_summary.py` - 从分析结果读取 `uploaded` 字段
- `tools/notifications/progress_worker.py` - 调用分析器，不参与判定

### 🧪 测试场景

需要验证的双 tag 场景：

1. **V2=V3 同镜像，精度达标**
   ```json
   {
     "versions": {
       "v2": {
         "harbor_image": "harbor.../Model:202608121234-v2",
         "also_tagged_v3": true
       },
       "v3": {
         "harbor_image": "",
         "same_as_v2": true,
         "accuracy_ok": true
       }
     }
   }
   ```
   **期望输出**：
   ```json
   {
     "delivery": {
       "version": "v3",
       "accuracy_ok": true,
       "uploaded": true  // ← 正确识别双 tag
     }
   }
   ```
   **qualified**: `true` ✅

2. **V2=V3 同镜像，精度不达标**
   ```json
   {
     "versions": {
       "v2": {"harbor_image": "...-v2", "also_tagged_v3": true},
       "v3": {"accuracy_ok": false}
     }
   }
   ```
   **期望输出**：
   ```json
   {
     "delivery": {
       "version": "v3",
       "accuracy_ok": false,  // ← 精度不达标
       "uploaded": true       // ← 但上传成功
     }
   }
   ```
   **qualified**: `false` (精度不达标)

3. **V2=V3 同镜像，上传失败**
   ```json
   {
     "versions": {
       "v2": {"harbor_image": ""},
       "v3": {"harbor_image": "", "accuracy_ok": true}
     },
     "traces": {
       "08_release": {"harbor_push": "failed"}
     }
   }
   ```
   **期望输出**：
   ```json
   {
     "delivery": {
       "version": "v3",
       "accuracy_ok": true,
       "uploaded": false  // ← 明确失败
     }
   }
   ```
   **qualified**: `false` (未上传)

### 📚 相关文档

- **设计文档**：`docs/notification_and_result_analysis_design.md` (新建)
- **工作流定义**：`CLAUDE.md` 行 150-164 (分支 B 双 tag 说明)
- **通知架构**：`tools/notifications/README.md`

### 🔗 相关 Issue/Memory

- Memory: 参见 `.claude/projects/-data-ckxu-FlagRelease/memory/` 中分支 B 相关记录
- 相关模型：多个 `*-seg1.md` / `*-v2-release.md` 记录了双 tag 场景

### ✅ 验证清单

- [x] Prompt 已更新双 tag 判定逻辑
- [x] 设计文档已创建
- [x] 更新日志已记录
- [ ] 需要在实际双 tag 场景下验证 Claude 分析输出
- [ ] 需要验证飞书卡片显示正确

---

## 附：版本体系变更（v3.1）

本次同时梳理了版本体系的语义变更：

**旧版**：V1 → V2 Pro → V3 → V4 → V5（最终交付）  
**新版**：V1 → V2 Pro → V3 Max（最终交付）→ V4 Express（可选性能优化）

**关键变化**：
- ❌ 废弃 V5
- ✅ V3 Max 是最终交付版本
- ✅ V4 Express 是可选的减算子性能优化
- ✅ 达标判定统一以 V3 为准（V4 不改变达标口径）

详见 `docs/notification_and_result_analysis_design.md`。

---

**修改人**：Claude (Opus 4.8)  
**审核人**：待确认  
**生效时间**：2026-08-12
