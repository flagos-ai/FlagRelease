---
name: flagos-release
description: FlagOS 镜像打包发布 + 模型权重上传（Harbor / ModelScope / HuggingFace）
version: 1.0.0
triggers:
  - 发布
  - 镜像上传
  - 镜像打包
  - 模型发布
  - release
  - image upload
  - package image
  - model release
  - upload model
  - publish
depends_on:
  - flagos-performance-testing
provides:
  - image.registry_url
  - image.upload_timestamp
  - release.modelscope_url
  - release.huggingface_url
---

# 发布 Skill

将验证完成的 FlagOS 环境打包为 Docker 镜像并发布到 Harbor，同时将模型权重上传到 ModelScope / HuggingFace。

**工具脚本**（宿主机执行，不部署到容器）：

```
tools/
├── main.py                      # 流水线主入口
├── requirements.txt             # Python 依赖
├── src/
│   ├── config.py                # 配置管理（从 context.yaml 加载 + 自动填充）
│   ├── chip_detector.py         # 芯片检测（9 厂商 SMI 解析 + 镜像 tag 生成）
│   ├── utils.py                 # 工具函数
│   └── stages/
│       ├── base.py              # Stage 基类（命令执行、结果记录）
│       └── publish.py           # 发布阶段（commit→tag→push→README→上传）
└── templates/
    └── README_TEMPLATE.md       # README 模板
```

**支持芯片厂商**（自动检测）：

| 厂商 | 检测命令 | SDK | GPU 编码示例 |
|------|---------|-----|-------------|
| nvidia | nvidia-smi | CUDA | nvidia001 (A100) |
| metax | mx-smi | MXMACA | metax001 (C550) |
| mthreads | mthreads-gmi | MUSA | mthreads001 (S5000) |
| iluvatar | ixsmi | IXRT | iluvatar001 (BI-V150) |
| ascend | npu-smi | CANN | ascend001 (910B) |
| hygon | hy-smi | DTK | hygon001 (BW1000) |
| kunlunxin | xpu-smi | XRE | kunlunxin001 (P800) |
| cambricon | cnmon | CNToolkit | cambricon001 (MLU590) |
| tsingmicro | tsm_smi | TSM | tsingmicro001 (REX1032) |

---

# 上下文集成

## 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
model:
  name: <来自 container-preparation>
service:
  healthy: <来自 service-startup>
gpu:
  vendor: <来自 container-preparation>
workflow:
  service_ok: <来自编排层>
  accuracy_ok: <来自编排层>
  performance_ok: <来自编排层>
  qualified: <来自编排层>
  skip_reason: <来自编排层>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
image:
  registry_url: <推送后的 Harbor 完整地址>
  upload_timestamp: <YYYYMMDDHHMM>
release:
  modelscope_url: <ModelScope 模型 URL>
  huggingface_url: <HuggingFace 模型 URL>
```

---

# 工作流程

## 执行流水线

从工作流共享状态 `context.yaml` 读取所有信息，无需手写配置文件：

```bash
cd skills/flagos-release/tools

# 执行发布
python main.py --from-context /flagos-workspace/shared/context.yaml

# 干运行（只验证配置，不实际执行）
python main.py --from-context /flagos-workspace/shared/context.yaml --dry-run

# 只生成 README
python main.py --from-context /flagos-workspace/shared/context.yaml --only-readme
```

`--from-context` 自动映射的字段：
- `container.name` → 容器名
- `model.name` → 模型来源
- `model.container_path` → 权重目录 + 服务启动命令
- `service.port/max_model_len` + `runtime.tp_size` → 服务启动命令
- `gpu.vendor` → 芯片厂商（自动检测填充）
- `eval.v1_score/v2_score` → 评测结果（填入 README）
- `image.registry_url` → 已发布的 Harbor 镜像地址（有则跳过 commit/tag/push）

## 阶段详情

### 步骤 0 — 发布条件检查

从 `context.yaml` 读取 `workflow.qualified` 状态，用于报告生成。所有发布统一为私有。

```
读取 workflow.qualified (= service_ok AND accuracy_ok AND performance_ok)

publish.private = true   → 统一私有发布
日志: "发布模式: 私有"

qualified 状态记录到报告中:
  - qualified=true: "达标"
  - qualified=false: "未达标"
```

**qualified 判定细节**（仅用于报告展示）：
- `service_ok = true`：V1 和 V2 都能正常启动
- `accuracy_ok = true`：V2精度下降 ≤5%，或经 ≤3 轮优化后达标
- `performance_ok = true`：V2/V1 每个并发级别 ≥80%，或经 elimination 逐删优化后达标
- 提交了 issue 但优化成功 → 仍算合格（qualified=true）
- `skip_reason` 非空时（如 `"service_startup_failed"`）→ 跳过了3/4，qualified=false

### 发布（Publish）

| 步骤 | 操作 | 可跳过 |
|------|------|--------|
| B0 | 容器 commit 为镜像（input_type=container 时） | 有 existing_harbor_image 时跳过 |
| B1 | 镜像打 tag（自动生成标准命名） | 可配置跳过 |
| B2 | 推送到 Harbor（流式输出进度） | 可配置跳过 |
| B3 | 生成 README（含发布可见性标记 + 评测结果） | 可配置跳过 |
| B4 | 发布到 ModelScope（SDK + CLI 降级，`private` 由步骤 0 决定） | 可配置跳过 |
| B5 | 发布到 HuggingFace（SDK + CLI 降级，`private` 由步骤 0 决定） | 可配置跳过 |

---

# 镜像命名规范

## Tag 格式

```
{registry}/flagrelease-{vendor}-release-model_{model}-tree_{tree}-gems_{gems}-cx_{cx}-python_{python}-torch_{backend}-{torch_version}-pcp_{sdk}-gpu_{gpu_code}-arc_{arch}-driver_{driver}:{YYYYMMDDHHMM}
```

## 示例

```
harbor.baai.ac.cn/flagrelease-public/flagrelease-nvidia-release-model_qwen3.5-8b-tree_none-gems_4.2.1rc0-cx_none-python_3.12.3-torch_cuda-2.9.0-pcp_cuda13.1-gpu_nvidia003-arc_amd64-driver_570.158.01:202603301143
```

## 规则

- GPU 型号使用编码（`nvidia001` = A100, `nvidia003` = H20, `metax001` = C550）
- 版本号中 `+` 替换为 `-`
- 模型名小写
- 日期 tag 格式 `YYYYMMDDHHMM`（12 位）
- `_-` / `-_` / `--` 等非法组合自动清理

## 模型命名规范

```
output_name:       {Model}-{vendor}          (nvidia 不加后缀)
flagrelease_name:  {output_name}-FlagOS
仓库 ID:           FlagRelease/{flagrelease_name}
```

示例：`Qwen3-8B-metax` → `Qwen3-8B-metax-FlagOS` → `FlagRelease/Qwen3-8B-metax-FlagOS`

---

# 产出目录结构

```
output/
  {flagrelease_name}/
    README.md              # 自动生成
    *.safetensors          # 权重文件（软链接）
    config.json
    tokenizer.json
    ...
```

---

# 完成条件

**发布条件检查**：
- `workflow.qualified` 已读取并记录到报告中

**镜像发布**：
- 环境信息已自动检测
- Docker 镜像已 commit + tag
- 镜像已推送到 Harbor

**模型发布**：
- README 已生成（含评测结果、环境信息、启动命令）
- 模型已上传到 ModelScope / HuggingFace
- 仓库 URL 已记录

**流程集成**：
- context.yaml 已更新（`image.registry_url`、`image.upload_timestamp`、`release.modelscope_url`、`release.huggingface_url`）
- `traces/08_release.json` 已写入（记录发布条件判定、commit/tag/push 命令、README 路径、ModelScope/HuggingFace URL）
- `results/release_info.json` 已保存（qualified 状态、Harbor URL、ModelScope URL、HuggingFace URL）
- `timing.steps.release` 已更新为本步骤的 `duration_seconds`

**容器产出同步到宿主机**（必须在输出最终报告前完成）：
- 容器内 `/flagos-workspace/{results,traces,logs}/` 已通过 `docker cp` 同步到宿主机 `/data/flagos-workspace/<model>/` 对应目录
- `context.yaml` 已同步到宿主机 `config/context_snapshot.yaml`
- 宿主机文件数量与容器内一致

---

# 故障排查

| 问题 | 解决方案 |
|------|----------|
| 芯片检测失败 | 在配置中手动指定 `chip.vendor` |
| Harbor 推送失败 | 脚本自动通过 `HARBOR_USER` / `HARBOR_PASSWORD` 环境变量登录；若未设置则需手动 `docker login` |
| ModelScope 上传失败 | 检查 `MODELSCOPE_TOKEN` 环境变量 |
| HuggingFace 上传失败 | 检查 `HF_TOKEN` 环境变量 |
| 镜像 tag 生成异常 | 使用 `--dry-run` 检查自动生成的配置 |
| 已有 Harbor 镜像 | 配置 `publish.existing_harbor_image` 跳过 commit/tag/push |
| 权重文件过大 | 上传自动重试（5 次，指数退避） |

---

## 编排层指令（步骤8 自动发布 — 固化决策）

### 调用方式

发布步骤通过宿主机工具统一执行，**禁止手动拼 docker commit/tag/push 命令**：

```bash
# 宿主机执行（不是 docker exec），先同步 context 到宿主机再调用
MOUNT_MODE=$(docker exec <container> cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
    cp /data/flagos-workspace/<model>/shared/context.yaml /data/flagos-workspace/<model>/config/context_snapshot.yaml
else
    docker cp <container>:/flagos-workspace/shared/context.yaml /data/flagos-workspace/<model>/config/context_snapshot.yaml
fi
python3 skills/flagos-release/tools/main.py --from-context /data/flagos-workspace/<model>/config/context_snapshot.yaml
```

**执行路径强制规则**：release 脚本**必须从项目目录执行**（`python3 skills/flagos-release/tools/main.py`），**严禁**复制到 `/tmp` 或其他临时目录后执行。

### 编排层后续操作

工具执行完成后，编排层仍需完成：
- 写入 `traces/08_release.json`（记录工具输出、发布 URL、耗时）
- 更新容器内 context.yaml 的 `image`、`release` 字段和 `workflow_ledger`
- 更新 `timing.steps.release`

### 容器产出同步到宿主机

步骤8完成后，根据 `workspace.mount_mode` 决定同步策略（读取容器内 `/flagos-workspace/.mount_mode`）：

| mount_mode | 同步策略 |
|------------|---------|
| `mounted` / `symlink` | 无需 docker cp，只需同步 context_snapshot |
| `internal` | 必须 docker cp 回传 results/traces/logs |

```bash
CONTAINER=<container_name>
HOST_BASE=/data/flagos-workspace/<model>
MOUNT_MODE=$(docker exec ${CONTAINER} cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")

if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
    cp ${HOST_BASE}/shared/context.yaml ${HOST_BASE}/config/context_snapshot.yaml
else
    for dir in results traces logs; do
        docker cp ${CONTAINER}:/flagos-workspace/${dir}/. ${HOST_BASE}/${dir}/
    done
    docker cp ${CONTAINER}:/flagos-workspace/shared/context.yaml ${HOST_BASE}/config/context_snapshot.yaml
fi
```

**禁止回传到项目源码目录**。`docker cp` 目标必须是 `/data/flagos-workspace/<model>/` 下的子目录。

### 报告生成

每个步骤完成后调用 `generate_report.py` 生成/更新报告：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/generate_report.py \
  --output /flagos-workspace/results/report.md"
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/generate_report.py \
  --json --output /flagos-workspace/results/report.json"
```

### 交付物清单

- `results/` — 性能/精度结果文件
- `results/report.md` — 迁移报告
- `results/report.json` — 迁移报告 JSON 格式
- `traces/` — 全流程执行留痕
- `logs/` — 运行日志（含 `pipeline.log`）
- `config/context_snapshot.yaml` — 流程结束时的完整 context 快照

---

## 编排层指令（步骤13 Plugin 发布 — 固化决策）

### 触发条件

步骤 12（Plugin 性能评测）完成后，检查 `plugin_workflow`：
- `plugin_workflow.accuracy_ok=true AND plugin_workflow.performance_ok=true` → 执行 plugin 发布
- 否则 → 跳过 plugin 发布，设置 `plugin_workflow.qualified=false`

### 调用方式

```bash
# 宿主机执行（同步 context 后调用）
MOUNT_MODE=$(docker exec <container> cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
    cp /data/flagos-workspace/<model>/shared/context.yaml /data/flagos-workspace/<model>/config/context_snapshot.yaml
else
    docker cp <container>:/flagos-workspace/shared/context.yaml /data/flagos-workspace/<model>/config/context_snapshot.yaml
fi
python3 skills/flagos-release/tools/main.py \
    --from-context /data/flagos-workspace/<model>/config/context_snapshot.yaml \
    --plugin-mode
```

### Plugin 模式行为

`--plugin-mode` 标志使发布脚本执行以下差异化逻辑：

| 步骤 | 主流程（步骤8） | Plugin 模式（步骤13） |
|------|---------------|---------------------|
| 镜像 tag | `{date_tag}` | `{date_tag}-plugin` |
| 仓库命名 | `FlagRelease/{Model}-{vendor}-FlagOS` | 复用步骤8仓库（不创建新仓库） |
| commit/tag/push | 正常执行 | 正常执行（plugin 版本是新镜像） |
| README | 生成标准 README | 用 plugin 镜像+评测数据重新生成 README |
| ModelScope/HuggingFace | 创建仓库 + 上传权重 + README | 只更新步骤8仓库的 README（覆盖） |

### README 更新步骤8仓库

Plugin 发布成功后，用 plugin 镜像地址和评测数据重新生成 README，覆盖上传到步骤8已发布的 ModelScope/HuggingFace 仓库。README 中的镜像地址、评测结果均替换为 plugin 版本数据。

更新方式：
1. 生成包含 plugin 镜像地址和评测数据的 README
2. 将 README 复制到容器内，通过容器内 CLI 上传到步骤8的原仓库

### 编排层后续操作

Plugin 发布完成后：
- 更新 `context.yaml` 的 `plugin_workflow` 字段（qualified、plugin_image_url、plugin_modelscope_url、plugin_huggingface_url）
- 写入 `traces/13_plugin_release.json`
- 更新 `workflow_ledger` 步骤 13 状态
- 更新 `timing.steps.plugin_release`
- 同步容器产出到宿主机（同步骤8的同步逻辑）
