---
name: flagos-container-preparation
description: 多入口容器准备，支持已有容器/已有镜像/ModelScope URL，通过 setup_workspace.sh 一次性部署所有工具
version: 5.0.0
triggers:
  - container preparation
  - prepare container
  - 容器准备
  - 环境准备
depends_on: []
next_skill: flagos-pre-service-inspection
provides:
  - container.name
  - container.status
  - model.name
  - model.local_path
  - gpu.vendor
  - gpu.count
  - entry.type
---

# 容器准备 Skill

支持三种入口，自动识别用户输入类型。容器就绪后通过 `setup_workspace.sh` 一次性部署所有工具脚本。

---

# 用户输入

| 入口 | 用户提供 | 系统做什么 |
|------|---------|-----------|
| **已有容器** | 容器名称 + 模型名 | 跳过创建，搜索模型权重（容器内+宿主机），未找到则容器内下载 |
| **已有镜像** | 镜像地址 + 模型名 | 自动搜索宿主机模型路径 → docker run 创建容器（找到则挂载，未找到则容器内下载） |
| **ModelScope URL** | URL | API 解析 → docker pull → docker run |

> **自动识别**：`run_pipeline.sh` 优先检查第一参数是否含 `:` 或 `/`（镜像地址特征），是则走镜像模式；否则通过 `docker inspect --type=container` 判断是否为已有容器。用户无需指定 `--image`。

---

# 工作流程

## 步骤 1 — 模型权重搜索与自动下载

### 已有镜像/URL 入口 — 自动搜索宿主机模型路径

`run_pipeline.sh` 在启动 Claude 之前自动执行 pre-flight 搜索：

```bash
python3 skills/flagos-container-preparation/tools/check_model_local.py \
    --model "<模型名>" --no-download --output-json
```

- 在宿主机搜索路径（/data, /nfs, /share, /models, /home）中查找匹配目录
- 三级匹配：精确匹配 > 包含匹配 > config.json 匹配
- **找到** → 记录 `model.local_path`，docker run 挂载到容器
- **未找到** → 预创建宿主机目录 `/data/models/<model_name>`，docker run 仍挂载此空目录，容器内下载到已挂载路径
- `--download-dir` 可指定下载目录（默认 `/mnt/data/models`）

### 已有容器入口 — 容器内搜索

```bash
python3 skills/flagos-container-preparation/tools/check_model_local.py \
    --model "<用户输入的模型名或URL>" --mode container --container <container_name> --output-json
```

搜索策略：
1. 先在容器内搜索（/data, /models, /root, /home, /workspace, /mnt, /opt）
2. 再在宿主机搜索，通过 `docker inspect` 检查挂载关系
   - 宿主机找到且已挂载 → 计算容器内路径，直接使用
   - 宿主机找到但未挂载 → 警告，进入容器内下载
3. 如容器内未找到 → 在容器内自动从 ModelScope 下载
   - 优先下载到已挂载的宿主机卷路径（/data > /mnt > /nfs > /share），避免写入 overlay 文件系统
   - 无可用挂载卷时 fallback 到 `/data/models/<model_name>`
   - 下载前自动检查 modelscope CLI 是否可用，不可用则先安装
   - 下载成功后重新校验权重完整性

输出字段：
- `final_container_path`：容器内模型路径 → 写入 `model.container_path`
- `final_host_path`：宿主机对应路径（如可推算）→ 写入 `model.local_path`

## 入口 1 — 已有容器

```bash
docker inspect <container_name> --format '{{.State.Status}}'
docker start <container_name>  # 如果停止
```

**模型权重搜索**（步骤 1 的容器模式）：

```bash
python3 skills/flagos-container-preparation/tools/check_model_local.py \
    --model "<model>" --mode container --container <container_name> --output-json
```

自动检测 GPU、记录模型路径（container_path + local_path）。`setup_workspace.sh` 会自动检测 `/flagos-workspace` 挂载状态：

| 情况 | 处理方式 |
|------|---------|
| 已挂载 `/flagos-workspace` | 直接使用，行为不变 |
| 有其他 bind mount（如 `/data`） | 在挂载点下创建 `flagos-workspace/`，软链接 `/flagos-workspace` 指向它，宿主机可直接访问 |
| 无任何挂载 | 容器内创建非持久化目录，给出警告 |

## 入口 2 — 已有镜像

1. 自动检测 GPU 厂商
2. **根据 GPU 厂商选择对应模板**，填充变量后生成 docker run 命令并自动执行
3. 验证容器状态

### docker run 命令模板

| 变量 | 说明 | 来源 |
|------|------|------|
| `${CONTAINER_NAME}` | 容器名称 | 自动生成，含冲突检测（见下方命名规则） |
| `${MODEL_PATH}` | 宿主机模型路径 | `check_model_local.py` 搜索：**找到则使用实际路径**（如 `/home/admin/workspace/models/Qwen3-0.6B`）；**未找到则使用 `/data/models/<model_name>`**（预创建并挂载空目录，容器内下载） |
| `${CONTAINER_MODEL_PATH}` | 容器内模型路径 | 与 `${MODEL_PATH}` 保持一致（宿主机路径原样映射到容器内同路径） |
| `${WORKSPACE_PATH}` | 宿主机工作目录 | `/data/flagos-workspace` |
| `${SHM_SIZE}` | 共享内存 | `64g` |
| `${IMAGE}` | 镜像地址 | 用户提供 |

> **模型路径未找到时**：`run_pipeline.sh` 自动预创建 `/data/models/<model_name>` 目录，docker run 始终包含 `-v ${MODEL_PATH}:${CONTAINER_MODEL_PATH}` 挂载行。容器创建后通过 `check_model_local.py --mode container --container-model-path` 下载到已挂载目录，确保权重落在宿主机。

### 容器命名与冲突处理（镜像模式专用）

容器名生成规则：
1. 基础名称：`<model_short_name>_flagos`（如 `Qwen3-8B_flagos`）
2. 创建前检测：`docker inspect --type=container <基础名称>`
3. 如不存在 → 直接使用基础名称
4. 如已存在 → 追加时间戳：`<model_short_name>_flagos_<MMDD_HHMM>`（如 `Qwen3-8B_flagos_0410_1500`）

**禁止行为**：镜像模式下禁止复用任何已存在的容器，即使该容器是由同一镜像创建的。必须通过 `docker run` 创建全新容器。

#### 模板 A：NVIDIA

```bash
docker run -itd --name=${CONTAINER_NAME} \
    --gpus=all --network=host \
    -v ${MODEL_PATH}:${CONTAINER_MODEL_PATH} \
    -v ${WORKSPACE_PATH:-/data/flagos-workspace/${MODEL_NAME}}:/flagos-workspace \
    ${IMAGE}
```

> **NVIDIA 专属限制**：严禁添加模板之外的任何参数（如 `--privileged`、`--ipc=host`、`--shm-size`、`--ulimit`、`--security-opt`、`--cap-add` 等，会触发 authZ 拒绝）。生成的 docker run 命令必须与上述模板完全一致，仅替换变量值，不增删参数。
>
> **降级策略（仅 NVIDIA）**：
> 1. **首选**：严格按上述模板生成 docker run 命令并执行
> 2. **模板失败时**（如 authZ 拦截）：先检查变量值是否正确（路径拼写、权限白名单匹配），修正后重试一次
> 3. **修正后仍失败**：通过 `docker inspect` 查看同宿主机上已有的**同镜像或同类型**容器的挂载配置，借鉴其 `-v` 参数组合重试一次
> 4. **借鉴重试仍失败**：终止任务，输出错误信息
>
> **禁止行为**：禁止跳过步骤 1-2 直接借鉴已有容器。必须先尝试模板，模板失败且修正无效后才允许借鉴。

#### 模板 B：Ascend（华为昇腾）

```bash
docker run -d --name ${CONTAINER_NAME} \
    --net=host --ipc=host --privileged \
    --shm-size=${SHM_SIZE:-64g} \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v ${MODEL_PATH}:${CONTAINER_MODEL_PATH} \
    -v ${WORKSPACE_PATH:-/data/flagos-workspace/${MODEL_NAME}}:/flagos-workspace \
    -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
    ${IMAGE} sleep infinity
```

#### 模板 C：Moore Threads（摩尔线程）

```bash
docker run -d --name ${CONTAINER_NAME} \
    --net=host --ipc=host --privileged \
    --shm-size=${SHM_SIZE:-16g} \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --tmpfs /tmp:exec \
    -e MTHREADS_VISIBLE_DEVICES=all \
    -e MTHREADS_DRIVER_CAPABILITIES=all \
    -v ${MODEL_PATH}:${CONTAINER_MODEL_PATH} \
    -v ${WORKSPACE_PATH:-/data/flagos-workspace/${MODEL_NAME}}:/flagos-workspace \
    ${IMAGE} sleep infinity
```

#### 模板 D：MetaX（沐曦）

```bash
docker run -d --name ${CONTAINER_NAME} \
    --net=host --ipc=host --privileged \
    --shm-size=${SHM_SIZE:-64g} \
    --group-add video --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v ${MODEL_PATH}:${CONTAINER_MODEL_PATH} \
    -v ${WORKSPACE_PATH:-/data/flagos-workspace/${MODEL_NAME}}:/flagos-workspace \
    ${IMAGE} sleep infinity
```

#### 模板 E：Cambricon（寒武纪）

```bash
docker run -d --name ${CONTAINER_NAME} \
    --net=host --pid=host --ipc=host --privileged \
    -v /usr/bin/cnmon:/usr/bin/cnmon \
    -v ${MODEL_PATH}:${CONTAINER_MODEL_PATH} \
    -v ${WORKSPACE_PATH:-/data/flagos-workspace/${MODEL_NAME}}:/flagos-workspace \
    -v /data:/data \
    ${IMAGE} sleep infinity
```

**模板规则**：
- 业务环境变量（`USE_FLAGGEMS`、`VLLM_USE_V1` 等）不写入模板，由后续 skill 按需添加
- 所有模板统一挂载 `/flagos-workspace`（宿主机路径为 `/data/flagos-workspace/${MODEL_NAME}`，按模型隔离）
- 生成命令后自动执行，无需用户确认

## 入口 3 — ModelScope / HuggingFace URL

1. 从 URL 提取 `<owner>/<model_name>`，调用 API 获取 README：

```bash
curl -s "https://modelscope.cn/api/v1/models/<owner>/<model_name>"
```

2. 从返回的 README 中提取：镜像地址、启动参数、模型路径
3. docker pull + 按入口 2 流程创建容器

**API 访问失败时**，模型名称和路径已从 URL 推导，只需用户补充镜像地址。

## 步骤 — 部署工具脚本

容器就绪后立即执行：

```bash
bash skills/flagos-container-preparation/tools/setup_workspace.sh $CONTAINER $MODEL_NAME
```

第二参数 `$MODEL_NAME` 为模型名称，传入后自动在宿主机创建 `/data/flagos-workspace/<MODEL_NAME>/` 及其子目录（results/traces/logs/config）。

此命令会：
1. 检测容器内 `results/`、`traces/`、`logs/` 是否有上一轮数据，若有则自动归档到 `archive/<YYYYMMDD_HHMMSS>/`
2. 宿主机同步归档（如传了 MODEL_NAME）
3. 创建干净的 `results/`、`traces/`、`logs/`、`config/` 目录
4. 部署所有工具脚本

## 步骤 — 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
entry:
  type: "<existing_container|new_container|url_parse>"
container:
  name: "<容器名称>"
  status: "running"
model:
  name: "<模型名称>"
  local_path: "<宿主机路径>"
  container_path: "<容器内路径>"
gpu:
  vendor: "<nvidia|huawei|mthreads|metax|cambricon>"
  type: "<GPU 型号>"
  count: <数量>
workspace:
  host_path: "<宿主机路径，如 /data/flagos-workspace/Qwen3-8B>"
  container_path: "/flagos-workspace"
  mount_mode: "<mounted|symlink|internal>"
```

---

# 完成条件

- 容器已运行，GPU 可见，模型目录已确认
- 工具脚本已通过 setup_workspace.sh 部署
- 四个子目录已创建（results/、traces/、logs/、config/）
- 容器内 `/flagos-workspace/shared/context.yaml` 已更新
- `traces/01_container_preparation.json` 已写入（记录 docker run 命令、setup_workspace 部署结果）
- `timing.workflow_start` 已写入容器内 context.yaml（ISO 8601，流程起始时间）
- `timing.steps.container_preparation` 已更新为本步骤的 `duration_seconds`

---

# 故障排查

| 问题 | 解决方案 |
|------|----------|
| GPU 未检测到 | 检查驱动安装 |
| 镜像拉取失败 | 检查网络，或 `docker load` 导入 |
| setup_workspace.sh 失败 | 检查容器是否运行，手动 docker cp |

下一步：**flagos-pre-service-inspection**
