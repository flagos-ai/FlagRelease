---
name: flagos-service-startup
description: 在容器内启动推理服务（支持 default/native/flagos 模式切换），使用 toggle_flaggems.py 和 wait_for_service.sh
version: 5.0.0
license: internal
triggers:
  - service startup
  - start service
  - 启动服务
  - health check
  - 健康检查
depends_on:
  - flagos-pre-service-inspection
next_skill: flagos-eval-comprehensive
provides:
  - service.cluster
  - service.external_ip
  - service.host
  - service.port
  - service.healthy
  - service.model_id
  - service.log_path
  - service.gems_txt_path
  - service.initial_operator_list
  - service.max_model_len
  - runtime.gpu_count
  - runtime.tp_size
  - runtime.tp_reason
  - runtime.flaggems_enabled
  - runtime.framework
  - runtime.thinking_model
  - environment.initial_env_verified
---

# 服务启动 Skill

支持 default/native/flagos 三种模式，基于 `flaggems_control` 探测结果动态决定启停方式。

**启动模式**：
- **default** — 不修改任何 FlagGems 状态，以容器现有配置原样启动。用于步骤3验证初始环境可用性。
- **native** — 关闭 FlagGems，纯原生环境。对应 V1 版本。
- **flagos** — 启用全量 FlagGems。对应 V2 版本。

**工具脚本**（已由 setup_workspace.sh 部署到容器）:
- `calc_tp_size.py` — TP 自动推算（根据模型大小和 GPU 显存）
- `toggle_flaggems.py` — FlagGems 开关切换（替代 sed）
- `wait_for_service.sh` — 服务就绪检测（动态超时 + 日志监控 + 早期失败检测）

---

# 统一工作目录

所有服务启动操作在 `/flagos-workspace` 目录下执行。

```
容器内: /flagos-workspace/logs/ ← 服务日志（按模式命名）
  startup_default.log  — 步骤3 初始服务启动
  startup_native.log   — 步骤4/5 中关闭 FlagGems 的 native 模式
  startup_flagos.log   — 步骤4/5 中开启 FlagGems 的 flagos 模式
宿主机: /data/flagos-workspace/<model_name>/logs/ ← 实时同步
```

---

# 上下文集成

## 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
model:
  name: <来自 container-preparation>
  container_path: <来自 container-preparation>
execution:
  cmd_prefix: <来自 pre-service-inspection>
flaggems_control:
  enable_method: <来自 pre-service-inspection>
  disable_method: <来自 pre-service-inspection>
  integration_type: <来自 pre-service-inspection>
environment:
  env_type: <来自 pre-service-inspection>       # native | vllm_flaggems | vllm_plugin_flaggems
  flaggems_txt_path: <来自 pre-service-inspection>  # vllm_flaggems 场景的 txt 路径
  gems_txt_auto_detect: <来自 pre-service-inspection>
```

## 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
service:
  cluster: <集群标识>
  external_ip: <宿主机 IP>
  host: <服务主机>
  port: <服务端口>
  healthy: true|false
  model_id: <模型标识符>
  log_path: <日志路径>
  gems_txt_path: <gems.txt 路径>
  enable_oplist_path: <flaggems_enable_oplist.txt 路径，无则为空>
  enable_oplist_count: <oplist 中算子数量，无则为 0>
  initial_operator_list: [...]
  max_model_len: <服务实际的 max_model_len>
runtime:
  framework: <vllm|sglang>
  gpu_count: <GPU 数量>
  tp_size: <tensor-parallel-size>
  tp_reason: <TP 推算原因>
  flaggems_enabled: true|false        # 根据 oplist 文件是否存在判断，而非启动模式
  thinking_model: true|false            # 是否为 thinking model（传递给后续评测 Skill）
```

---

# 工作流程

## 步骤 1 — 停止现有服务

```bash
# 推荐方式：docker restart 确保资源完全释放（避免僵尸进程占用显存）
docker restart $CONTAINER
sleep 5
```

备选方式（仅当不能重启容器时）：
```bash
docker exec $CONTAINER bash -c "pkill -f 'vllm\|sglang\|flagscale' 2>/dev/null; sleep 3"
```

## 步骤 2 — 切换 FlagGems 状态（按 env_type 分路径）

根据 `environment.env_type` 和启动模式决定 FlagGems 开关方式：

**Default 模式**（不修改任何状态）：
不调用 `toggle_flaggems.py`，直接跳到步骤 3。用于步骤3验证初始环境可用性。所有 env_type 通用。

---

### env_type = native（纯 vllm 原生）

无 FlagGems 可切换，无需调用 `toggle_flaggems.py`。
- Native 模式 / FlagOS 模式均直接启动标准 vllm 命令
- 跳过算子列表记录（步骤 5）
- 不执行 V2/V3 相关步骤

---

### env_type = vllm_flaggems（通过环境变量驱动 FlagGems 开关）

环境检测阶段（步骤2）已自动完成一次性代码注入，将原始 `flag_gems.enable()` 替换为环境变量驱动逻辑。后续所有 FlagGems 开关和算子控制通过环境变量 + 配置文件实现，不再修改源码。

**控制机制**：
- `USE_FLAGGEMS`：控制 FlagGems 开关（`1`=开启，`0`=关闭）
- `FLAGGEMS_CONTROL_MODE`：控制算子分支模式（`only_enable`=白名单，`unused`=黑名单）
- `/root/flaggems_ops_control.json`：算子控制配置文件

**Native 模式**（关闭 FlagGems）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action disable --json"
```
输出 JSON 包含 `env_vars` 和 `env_inline` 字段（`USE_FLAGGEMS=0`），在启动命令中使用 `env_inline` 作为内联前缀。

**Native 模式服务启动命令**（使用 `start_service.sh --mode native`）：
```bash
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=0 bash /flagos-workspace/scripts/start_service.sh --mode native > /flagos-workspace/logs/startup_native.log 2>&1"
```

**FlagOS 模式**（启用 FlagGems）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action enable --json"
```
输出 JSON 包含 `env_vars` 和 `env_inline` 字段（`USE_FLAGGEMS=1`），同时写入控制文件 `/root/flaggems_ops_control.json`（默认全量开启）。

**注意**：如果 `optimization.disabled_ops` 非空（之前已禁用部分算子），应改用 `--action modify-enable --disabled-ops '<list>'`，确保已禁用算子不被重新加载。`--action enable` 会自动检测已有 control file 并继承 `FLAGGEMS_CONTROL_MODE`，但显式使用 `modify-enable` 更可靠。

**服务启动命令**（使用内联环境变量前缀）：
```bash
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=1 <startup_command> > /flagos-workspace/logs/startup_flagos.log 2>&1"
```

**算子列表获取**（启动后）：
- 读取 `environment.flaggems_txt_path`（由 pre-service-inspection 步骤 2.6 写入）
- 如果 `gems_txt_auto_detect: true`（代码解析未找到路径），启动后调用：
  ```bash
  docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action find-gems-txt --json"
  ```
  从输出的 `recommended` 字段获取路径，回写 `context.yaml` 的 `environment.flaggems_txt_path`

**未注入兜底**：如果环境检测阶段注入失败（源码格式异常），toggle_flaggems.py 自动降级为原有的注释/取消注释方式。

---

### env_type = vllm_plugin_flaggems（通过环境变量控制 FlagGems 开关）

**Native 模式**（关闭 FlagGems）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action disable --integration-type plugin --json"
```
输出 JSON 包含 `env_vars` 和 `env_inline` 字段，在启动命令中使用 `env_inline` 作为内联前缀。

**FlagOS 模式**（启用 FlagGems）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action enable --integration-type plugin --json"
```

**算子列表获取**（启动后）：
- 检查 `/tmp/flaggems_enable_oplist.txt`（plugin 架构下的权威算子列表）

## 步骤 2.4 — GPU 空闲检测（强制）

服务启动前检测各 GPU 的显存占用情况，**只使用空闲 GPU，不清理其他进程。**

使用统一检测脚本（自动适配 NVIDIA / 华为昇腾 / 沐曦等厂商）：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/detect_gpu.py --check-free --vendor \$(python3 -c \"import yaml; print(yaml.safe_load(open('/flagos-workspace/shared/context.yaml')).get('gpu',{}).get('vendor',''))\")"
```

输出 JSON 格式：`{vendor, free_gpus: [idx...], busy_gpus: [idx...], total, details: [{index, used_mib, total_mib, free_mib, usage_pct}...], visible_devices_env}`

**处理逻辑**：

| 情况 | 操作 |
|------|------|
| 全部 GPU 空闲 | 正常使用全部 GPU，不设 VISIBLE_DEVICES |
| 部分 GPU 空闲 | 设置对应厂商的 VISIBLE_DEVICES 环境变量（从输出的 `visible_devices_env` 字段获取），TP 按空闲 GPU 数重新推算 |
| 无空闲 GPU | 记录警告，仍尝试启动（小模型可能共享显存），OOM 后报错 |

**部分 GPU 空闲时**：
1. 将空闲 GPU 索引写入 `runtime.cuda_visible_devices`（如 `"2,3,4,5,6,7"`）
2. 更新 `runtime.gpu_count` 为空闲 GPU 数量
3. 步骤 2.5 的 TP 推算基于空闲 GPU 数量
4. `start_service.sh` 会自动从 `gpu.visible_devices_env` 读取正确的环境变量名并设置
5. 输出提示并记录到 trace

```
⚠ GPU 资源检测: 8 张 GPU 中 6 张空闲
  占用中: GPU 0,1（显存占用 45.2%, 38.7%）
  本次使用: GPU 2,3,4,5,6,7（CUDA_VISIBLE_DEVICES=2,3,4,5,6,7）
```

**写入 context.yaml**：
```yaml
runtime:
  gpu_count: 6                          # 实际使用的 GPU 数量
  cuda_visible_devices: "2,3,4,5,6,7"   # 指定卡的索引值（环境变量名由 gpu.visible_devices_env 决定）
  total_gpus: 8                          # 机器总 GPU 数
  gpu_selection_reason: "GPU 0,1 被其他进程占用，使用剩余 6 张空闲 GPU"
```

## 步骤 2.5 — TP 自动推算

在启动服务前，自动推算最小可用 `--tensor-parallel-size`：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/calc_tp_size.py --model-path $MODEL_PATH --json"
```

输出示例：
```json
{
  "recommended_tp": 1,
  "gpu_count": 8,
  "gpu_memory_gb": 80.0,
  "model_size_gb": 15.2,
  "estimated_required_gb": 18.2,
  "reason": "模型 15.2GB，单卡 80GB 显存充足，推荐 TP=1"
}
```

**使用规则**：
- 读取 `recommended_tp` 作为 `${TP_SIZE}` 的值
- 如果脚本执行失败（退出码非 0），fallback 到 GPU 总数
- 如果推荐 TP 启动后 OOM，自动翻倍重试（TP×2），直到 GPU 总数

将推荐值写入 context.yaml 的 `runtime.tp_size` 和 `runtime.tp_reason`。

## 步骤 2.6 — max_model_len 决策

`--max-model-len` 直接决定模型单次请求能处理的最大 token 数。**确定后写入 context.yaml，后续每次启动复用同一值。**

**决策逻辑**：

1. 读取 `service.max_model_len`
   - **已有值（>0）** → 直接复用，不重新计算
   - **为 0（首次）** → 按下方规则计算后写入

2. 首次计算规则：

| 模型类型 | max_model_len | 原因 |
|---------|---------------|------|
| 所有模型 | **32768** | GPQA 等评测 prompt 较长，需要充足上下文窗口保证精度评测不被截断 |

3. **显存约束**：如果启动 OOM，降级 `max_model_len`（最低 16384），并更新 context.yaml

4. **验证**：启动后 `wait_for_service.sh` 输出实际 `max_model_len`，确认与预期一致

## 步骤 2.7 — 端口可用性检测（强制）

服务启动前检测目标端口是否被占用。**如果被占用，自动换端口，不停止占用方。**

由于容器使用 `--net=host` 模式，容器与宿主机共享网络栈，直接在宿主机检测即可：

```bash
ss -tlnp 2>/dev/null | grep ":${PORT} " && echo 'PORT_IN_USE' || echo 'PORT_FREE'
```

**端口被占用时的处理**：
1. 不停止占用进程/容器
2. 从 PORT+1 开始递增探测，找到第一个可用端口（上限 PORT+10）
3. 更新 context.yaml 的 `service.port` 为新端口
4. 后续所有操作（评测、性能测试）使用新端口
5. 记录端口变更到 trace：`"port_changed": {"from": 8000, "to": 8001, "reason": "原端口被其他任务占用"}`

```bash
# 端口探测逻辑
ORIGINAL_PORT=${PORT}
while ss -tlnp 2>/dev/null | grep -q ":${PORT} "; do
    echo "⚠ 端口 ${PORT} 被占用，尝试 $((PORT+1))..."
    PORT=$((PORT+1))
    if [ $((PORT - ORIGINAL_PORT)) -gt 10 ]; then
        echo "✗ 连续 10 个端口均被占用，需人工介入"
        exit 1
    fi
done
echo "✓ 使用端口 ${PORT}"
```

如果端口发生变更，输出提示并写入 context.yaml：
```
⚠ 端口变更: 8000 → 8001（原端口被其他任务占用）
```

```yaml
service:
  port: 8001                    # 实际使用的端口
  original_port: 8000           # 原始默认端口（仅端口变更时记录）
  port_change_reason: "原端口被其他任务占用"
```

## 步骤 3 — 启动服务

**GPU 选择适配**：如果步骤 2.4 检测到部分 GPU 被占用（`runtime.cuda_visible_devices` 非空），启动命令需注入对应厂商的 VISIBLE_DEVICES 环境变量（`start_service.sh` 会自动从 `gpu.visible_devices_env` 读取正确的变量名）。

**非 plugin 场景**：
```bash
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && ${VISIBLE_DEVICES_ENV}=${VISIBLE_DEVICES:-} <startup_command> > /flagos-workspace/logs/startup_<mode>.log 2>&1"
```

**Plugin 场景**（内联环境变量前缀）：
```bash
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && ${VISIBLE_DEVICES_ENV}=${VISIBLE_DEVICES:-} PATH=/opt/conda/bin:\$PATH <env_inline> <startup_command> > /flagos-workspace/logs/startup_<mode>.log 2>&1"
```

其中 `${VISIBLE_DEVICES_ENV}` 从 context.yaml 的 `gpu.visible_devices_env` 获取（如 `CUDA_VISIBLE_DEVICES`、`ASCEND_RT_VISIBLE_DEVICES` 等），`${VISIBLE_DEVICES}` 从 `runtime.cuda_visible_devices` 获取。

### Plugin 场景 vllm 服务启动模板

Plugin 环境下服务启动命令统一使用标准 vllm 格式，FlagGems 控制通过**内联环境变量**注入，与启动命令分离。

```bash
vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --served-model-name ${MODEL_NAME} \
    --tensor-parallel-size ${TP_SIZE} \
    --max-num-batched-tokens ${MAX_BATCHED_TOKENS:-16384} \
    --max-num-seqs ${MAX_NUM_SEQS:-256} \
    --max-model-len ${MAX_MODEL_LEN:-32768} \
    --trust-remote-code
```

**可选参数**（按需添加）：

| 参数 | 场景 | 示例 |
|------|------|------|
| `--pipeline-parallel-size` | 多机或超大模型 | `--pipeline-parallel-size 2` |
| `--gpu-memory-utilization` | 需限制显存占用 | `--gpu-memory-utilization 0.8` |
| `--reasoning-parser` | Thinking model | `--reasoning-parser qwen3` |

**四种模式启动方式**：

```bash
# Default（不修改环境，原样启动）
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH bash /flagos-workspace/scripts/start_service.sh --mode flagos > /flagos-workspace/logs/startup_default.log 2>&1"

# Native（关闭 FlagGems）
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=0 bash /flagos-workspace/scripts/start_service.sh --mode native > /flagos-workspace/logs/startup_native.log 2>&1"

# FlagOS Full（全量 FlagGems）
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=1 bash /flagos-workspace/scripts/start_service.sh --mode flagos > /flagos-workspace/logs/startup_flagos.log 2>&1"

# FlagOS Optimized（自定义算子集）
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=1 bash /flagos-workspace/scripts/start_service.sh --mode flagos_optimized > /flagos-workspace/logs/startup_flagos.log 2>&1"
```

四种模式差异仅在内联环境变量前缀（由 `toggle_flaggems.py` 或 `apply_op_config.py` 的 JSON 输出中的 `env_inline` 提供）。

**模板使用规则**：
- 具体参数值从容器 README / 用户输入 / context.yaml 获取
- `--served-model-name` 默认使用模型目录名
- `--tensor-parallel-size` 默认使用 `calc_tp_size.py` 的推荐值（基于模型大小和单卡显存自动推算），fallback 到 GPU 总数
- 业务环境变量（`VLLM_USE_MODELSCOPE` 等）按需在 docker exec 中追加，不写入模板
- `--max-model-len` 使用 context.yaml 中 `service.max_model_len` 的值（由步骤 2.6 决策）

## 步骤 4 — 等待服务就绪（动态超时）

脚本支持两种模式：
- **动态模式**（传 `--log-path`）：监控启动日志，检测进度信号和失败信号，动态调整等待时间
- **兼容模式**（不传 `--log-path`）：`--timeout` 作为绝对超时，行为与旧版一致

**参数说明**：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--timeout` | 动态模式：无活动超时（日志无新输出多久算卡住）；兼容模式：绝对超时 | 120s |
| `--max-timeout` | 绝对上限（安全兜底，无论如何不超过） | 1800s |
| `--log-path` | 启动日志路径，传入则启用动态模式 | 空 |
| `--mode` | 启动模式提示（`default`/`native`/`flagos`） | `default` |

```bash
# Native 模式 / Default 模式
docker exec $CONTAINER bash -c "/flagos-workspace/scripts/wait_for_service.sh \
    --port $PORT --model-name '$MODEL_NAME' \
    --timeout 120 --max-timeout 900 \
    --log-path /flagos-workspace/logs/startup_<mode>.log \
    --mode <mode>"

# FlagGems 模式（CUDA graph + Triton 编译需要更长耐心）
docker exec $CONTAINER bash -c "/flagos-workspace/scripts/wait_for_service.sh \
    --port $PORT --model-name '$MODEL_NAME' \
    --timeout 180 --max-timeout 1800 \
    --log-path /flagos-workspace/logs/startup_flagos.log \
    --mode flagos"
```

**动态监控行为**：
- 自动轮询（2s→4s→5s 快速收敛，最大间隔 5s）
- 检测启动阶段：权重加载 → CUDA graph 编译 → FlagGems 算子注册 → Triton 内核编译 → 端口绑定
- 日志持续增长 = 服务在正常启动中，自动延长等待（直到 `--max-timeout`）
- 检测到 OOM / CUDA error / 进程崩溃 → 立即退出，不等超时
- 日志停止增长超过 `--timeout` → 判定为停滞，退出

**进度输出示例**：
```
[30s] 阶段: 加载模型权重...
[85s] 阶段: 权重加载完成
[90s] 阶段: CUDA graph 编译中...
[320s] 阶段: CUDA graph 编译中... (35s 无新日志)
[450s] 服务就绪！耗时 450s
```

**早期失败检测**：
```
[12s] ✗ 检测到致命错误: CUDA out of memory
```

**启动后校验**：检查 `wait_for_service.sh` 输出的 `max_model_len`：
- 如果是 thinking model 且 `max_model_len < 32768` → 警告，建议重启并加大 `--max-model-len`
- 如果 `max_model_len < 8192` → 评测可能出问题，必须修正

## 步骤 5 — 服务验证

```bash
curl -s http://localhost:$PORT/v1/models
curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_name>", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}'
```

## 步骤 6 — 探测宿主机 IP 和输出连接信息

```
============================================================
服务连接信息
============================================================
<集群, IP, 服务端口, 模型名称>
评测接口: http://${EXTERNAL_IP}:${PORT}/v1/chat/completions
启动模式: native / flagos
============================================================
```

## 步骤 7 — 检查算子列表（每次启动后，强制）

**每次服务启动后（无论 default/native/flagos 模式），都必须检查 `flaggems_enable_oplist.txt`。**

该文件是 FlagGems 运行时自动生成的**唯一权威算子列表**，路径默认为 `/tmp/flaggems_enable_oplist.txt`。

```bash
# 检查 oplist 文件
docker exec $CONTAINER bash -c "
if [ -f /tmp/flaggems_enable_oplist.txt ]; then
    echo 'OPLIST_FOUND: /tmp/flaggems_enable_oplist.txt'
    echo 'OPLIST_MTIME:' \$(stat -c %Y /tmp/flaggems_enable_oplist.txt)
    echo 'OPLIST_COUNT:' \$(wc -l < /tmp/flaggems_enable_oplist.txt)
    cat /tmp/flaggems_enable_oplist.txt
else
    echo 'OPLIST_NOT_FOUND'
fi
"
```

**判断逻辑**：

| 文件状态 | 含义 | 后续操作 |
|----------|------|----------|
| 存在且有内容 | FlagGems 实际在运行 | 以此文件内容为当前生效算子列表，同步到 `results/ops_list.json` |
| 不存在或为空 | FlagGems 未启用 | 不依赖任何缓存的算子列表 |

**文件存在时**：

```bash
# 以 oplist 文件为准，同步保存到 results/ops_list.json
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/operator_optimizer.py discover \
  --save-ops /flagos-workspace/results/ops_list.json"

# 保存初始全开算子列表的 txt 副本到 results/（供事后查看和对比）
docker exec $CONTAINER cp /tmp/flaggems_enable_oplist.txt /flagos-workspace/results/initial_oplist.txt

# 保存初始控制文件副本（供报告中配置 vs 运行时 txt 对比，仅首次启动时保存）
docker exec $CONTAINER bash -c '[ ! -f /flagos-workspace/results/ops_control_initial.json ] && cp /root/flaggems_ops_control.json /flagos-workspace/results/ops_control_initial.json 2>/dev/null || true'
```

**关键原则**：
- `flaggems_enable_oplist.txt` = 当前实际生效的算子列表，**唯一权威来源**
- `results/ops_list.json` 是此文件的持久化副本，供后续对比和报告使用
- 如果启动模式为 flagos 但文件不存在 → 异常，需排查（服务可能未正确加载 FlagGems）
- 如果启动模式为 native 但文件存在 → 可能是上次 flagos 的残留，不应作为当前算子列表
- 每次 FlagGems 重新启动都会**重新生成**此文件，内容反映最新的算子配置（含 blacklist 生效后的结果）

**Native 模式残留检测**：

native 模式启动后，如果发现 `flaggems_enable_oplist.txt` 仍然存在，执行以下检查：

```bash
# 获取 oplist 文件修改时间和服务启动时间
OPLIST_MTIME=$(${CMD_PREFIX} stat -c %Y /tmp/flaggems_enable_oplist.txt 2>/dev/null || echo 0)
SERVICE_START=$(${CMD_PREFIX} stat -c %Y /proc/1/cmdline 2>/dev/null || echo 999999999)

if [ "$OPLIST_MTIME" -lt "$SERVICE_START" ]; then
    echo "检测到旧 oplist 残留（mtime < 服务启动时间），清理中..."
    ${CMD_PREFIX} rm -f /tmp/flaggems_enable_oplist.txt
    echo "已清理残留 oplist 文件"
else
    echo "WARNING: native 模式下 oplist 文件 mtime 晚于服务启动，可能 FlagGems 未正确关闭"
fi
```

- mtime 早于本次服务启动时间 → 旧残留，清理并记录到 trace
- mtime 晚于启动时间 → 异常，FlagGems 可能未正确关闭，报告给用户

**反馈输出**：
```
# FlagGems 运行时
算子列表验证：
  oplist 文件: /tmp/flaggems_enable_oplist.txt
  当前生效算子: XX 个
  已同步到: /flagos-workspace/results/ops_list.json

# FlagGems 未运行时
算子列表检查: oplist 文件不存在，FlagGems 未启用
```

## 步骤 8 — 写入 context.yaml

写入 `environment` 字段（步骤3 default 模式时）：
```yaml
environment:
  initial_env_verified: true    # 步骤3通过后设为 true
  has_plugin: <from inspection>
```

---

# 失败恢复

如果 flagos 模式启动失败：
1. **备份崩溃日志**（避免重试时被覆盖）：`cp startup_default.log startup_default_crashed.log`
2. **清理 Triton/FlagGems 编译缓存**（确保重试时暴露所有问题算子，不依赖旧缓存侥幸通过）：
   ```bash
   ${CMD_PREFIX} bash -c "rm -rf /root/.triton/cache/ /tmp/triton_cache/ /root/.flaggems/code_cache/ 2>/dev/null && echo 'Triton/FlagGems cache cleared'"
   ```
3. **算子级诊断重试**（不限轮次，每轮暴露新算子就继续）：

   **第一原则：禁用算子是启动崩溃的最高优先解。** 任何形式的崩溃（AICore 异常、Triton 编译错误、graph capture 失败、RuntimeError）都必须首先归因为算子问题并定位禁用。在穷尽所有算子定位手段之前，严禁尝试 enforce-eager、切 native、或判定不可恢复。

   ```bash
   ${CMD_PREFIX} python3 /flagos-workspace/scripts/diagnose_ops.py crash-log \
     --log-path /flagos-workspace/logs/startup_default_crashed.log --json
   ```
   - `crashed_ops` 非空 → 累积禁用问题算子 → 重启服务（仍为 FlagGems 模式）：
     ```bash
     ${CMD_PREFIX} python3 /flagos-workspace/scripts/toggle_flaggems.py \
       --action modify-enable --disabled-ops "<所有已累积的问题算子>" --json
     ```
   - 重启成功（含推理验证通过）→ 记录 `disabled_ops` 到 context.yaml，继续正常流程
   - 重启后再次崩溃（启动阶段或推理阶段均算）→ 备份日志 → 清缓存 → 再次 crash-log 解析 → 累积禁用新算子 → 继续重试
   - **`diagnose_ops.py` 返回空时的算子定位方法**（工具返回空 ≠ 无问题算子，必须人工定位）：
     1. 查看 traceback 中的 `flag_gems/` 路径，提取文件名即为算子名
     2. 查看崩溃前最后编译的 Triton kernel 名（通常在 `Compiling ...` 日志行中）
     3. 查看 `q.xxx_()` 或 `torch.xxx()` 调用栈中紧邻 flag_gems 的函数名
     4. 如果崩溃发生在 graph capture 阶段，查看 capture 前最后注册/编译的算子
     5. 如果以上均无法定位，逐步禁用最近一轮新启用的算子组（二分法排查）
   - **停止条件**：连续 2 轮重试后服务仍崩溃，且上述 5 种定位手段均无法识别新的问题算子，判定为不可恢复
   - 注意：推理阶段崩溃也属于"重试失败"，需要同样走 diagnose → 禁用 → 重启流程，不单独计数
4. 连续 2 轮确认无新可禁用算子（5 种定位手段均无结果）→ 最后尝试添加 `--enforce-eager` 重启一次 → 仍失败 → 切回 Native 验证
5. Native 也失败 → 报告环境问题；Native 成功 → 确认是 FlagGems 问题

### 问题日志写入

服务启动失败时，必须将失败信息追加写入 `logs/issues_startup.log`：

```bash
docker exec $CONTAINER bash -c "cat >> /flagos-workspace/logs/issues_startup.log << 'ISSUE_EOF'
[$(date '+%Y-%m-%d %H:%M:%S')] <启动模式> | <问题摘要>
  详情: <错误信息，如 OOM/端口占用/进程崩溃/超时>
  操作: <恢复措施，如 TP 翻倍/切回 Native/降低 max-model-len>
  结果: <恢复结果>
ISSUE_EOF"
```

记录场景：
- 启动超时（wait_for_service.sh 超时）
- 进程启动后立即退出（OOM、GPU 显存不足）
- FlagGems 模式启动失败，切回 Native
- TP 调整重试
- 端口被占用

---

# 完成条件

- 启动模式已确认（native / flagos）
- 服务进程正在运行
- API /v1/models 可访问
- 推理测试通过
- 已输出服务连接信息
- gems.txt 已检查（flagos 模式）
- context.yaml 已更新（必须包含 `workflow.service_ok=true`，无论是首次启动成功还是崩溃重试后成功）
- 对应 trace 文件已写入：
  - 步骤3初始启动 → `traces/03_service_startup.json`
  - 步骤4/6中的 native/flagos 模式切换 → 记录在 `traces/04_quick_accuracy.json` 或 `traces/06_quick_performance.json` 的 actions 中
- 启动失败时，`logs/issues_startup.log` 已追加写入问题记录
- `timing.steps.service_startup` 已更新为本步骤的 `duration_seconds`
- 更新报告：`docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"`

---

# 故障排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 进程启动后立即退出 | GPU 显存不足 | 自动将 TP 翻倍重试（TP×2），或降低 max-model-len |
| API 无响应 | 端口被占用 | 检查 `lsof -i:$PORT` |
| FlagGems 未生效 | toggle_flaggems.py 未正确切换 | 运行 `--action status` 检查 |
| gems.txt 未生成 | FlagGems 未启用 | 确认 toggle 状态 |
| 服务启动超时（无活动） | 日志停止增长，进程可能卡死 | 检查 GPU 状态（`nvidia-smi`），考虑增大 `--timeout` |
| 服务启动超时（绝对上限） | 大模型加载 + CUDA graph 编译超过 max-timeout | 增大 `--max-timeout`（默认 1800s） |
| 进程崩溃提前检测 | OOM/CUDA error/Segfault，动态模式自动检测 | 查看 `_last_error.json` 中的 `fatal_signal` 和 `fatal_line` |
| 日志活跃但长时间未就绪 | CUDA graph 编译或 Triton 内核编译耗时长 | 正常现象，动态模式会自动延长等待 |
| Thinking model 评测分数异常低 | max_model_len 过小，推理链被截断 | 重启服务，加大 `--max-model-len` 至 32768+ |
| OOM: max_model_len 过大 | KV cache 显存预分配超限 | 降低 max-model-len（thinking model 最低 16384） |
| Triton 缓存隐藏问题算子 | 崩溃重试时旧缓存命中，跳过有 bug 的编译路径，导致问题延迟暴露 | 崩溃后重试前必须清理 `/root/.triton/cache/`、`/root/.flaggems/code_cache/` |

---

## 编排层指令（步骤3 — 固化决策）

**FlagGems 模式启动成功处理**（正常路径）：
- FlagGems 模式服务启动成功且推理验证通过 → **必须设置 `workflow.service_ok = true`**
- 通过 `update_context.py --set workflow.service_ok=true` 更新
- 此设置是段间流转的关键判定字段，遗漏会导致后续步骤被跳过

**FlagGems 模式启动失败处理**（不含超时，超时属于正常等待）：

**第一原则：禁用算子是最高优先解。任何崩溃都必须首先定位并禁用具体算子，穷尽所有定位手段之前严禁走其他路径。**

1. 备份崩溃日志：`cp startup_default.log startup_default_crashed.log`
2. 清理 Triton/FlagGems 编译缓存：`rm -rf /root/.triton/cache/ /tmp/triton_cache/ /root/.flaggems/code_cache/`
3. 调用 `diagnose_ops.py crash-log` 解析崩溃日志
4. `crashed_ops` 非空 → 累积禁用问题算子（`toggle_flaggems.py --action modify-enable --disabled-ops`）→ 重启（每轮重试前均需清理缓存）
5. 重试成功（含推理验证通过）→ 记录 `disabled_ops` 到 context.yaml，`workflow.service_ok = true`，继续正常流程
6. 重试后再次崩溃（启动或推理阶段）→ 备份日志 → 清缓存 → 再次 diagnose → 累积禁用新算子 → 继续重试。**不限轮次，只要每轮能定位到新问题算子就继续**
7. **`diagnose_ops.py` 返回空时的算子定位**（返回空 ≠ 无问题算子，严禁跳过）：
   - 查看 traceback 中 `flag_gems/` 路径，文件名即算子名
   - 查看崩溃前最后编译的 Triton kernel 名（`Compiling ...` 日志行）
   - 查看 `q.xxx_()` / `torch.xxx()` 调用栈中紧邻 flag_gems 的函数名
   - 查看 graph capture 前最后注册/编译的算子
   - 以上均无法定位 → 逐步禁用最近一轮新启用的算子组（二分法排查）
8. **停止条件**：连续 2 轮重试后服务仍崩溃，且上述所有定位手段均无法识别新的问题算子 → 最后尝试 `--enforce-eager` 一次 → 仍失败 → 判定不可恢复 → 调用 `issue_reporter.py full --type operator-crash`
9. 排除操作失误：native 模式也失败 → 环境问题，需人工介入
10. 确认是 FlagGems 问题（非硬件）→ `workflow.service_ok = false` → 提交 issue 后**停止任务**，不继续步骤4/6/7 的精度性能评测（FlagGems 完全不可用时评测无意义）→ 直接到步骤8发布（私有，附带崩溃原因）

**崩溃重试时的服务启动方式**：禁用算子后重启服务时，**必须使用 `start_service.sh`**（而非直接 `docker exec -d` 拼接 vllm 命令），确保 `FLAGGEMS_CONTROL_MODE` 等环境变量被正确加载：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=1 bash /flagos-workspace/scripts/start_service.sh --mode flagos"
```
如果必须使用 `docker exec -d` 直接启动，则需内联传递环境变量：
```bash
docker exec -d $CONTAINER bash -c "source /etc/environment && export USE_FLAGGEMS=1 && cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH vllm serve ..."
```

**关键判定**：`workflow.service_ok` 表示"FlagGems 模式可用"，不是"native 模式可用"。native 能启动但 FlagGems 不能 → `service_ok = false`。只有 FlagGems 模式（含禁用部分算子后）能正常启动 → `service_ok = true`。

**算子列表持久化**：flagos 模式首次启动成功后，执行上方"算子列表检查"章节中的 `initial_oplist.txt` 保存（L506）。
