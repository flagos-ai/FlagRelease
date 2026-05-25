---
name: flagos-eval-comprehensive
description: 统一精度评测 Skill，支持本地快速评测（GPQA Diamond）和远端 flageval 正式评测，含 V1 vs V2 精度对比（5% 阈值）和算子排查闭环
version: 4.0.0
triggers:
  - 精度评测
  - quick 评测
  - 本地评测
  - 综合评测
  - 全面评测
  - comprehensive eval
  - evalscope
  - GPQA
  - fast gpqa
  - 远端评测
  - FlagRelease 评测
  - remote eval
  - eval correctness
  - flageval
depends_on:
  - flagos-service-startup
next_skill: flagos-performance-testing
provides:
  - eval_results
  - eval_report
  - eval.v1_score
  - eval.v2_score
  - eval.accuracy_aligned
  - eval.request_id
---

# FlagOS 精度评测 Skill（统一）

统一精度评测入口，包含四个模块：

| 模块 | 用途 | 场景 |
|------|------|------|
| **A — 本地快速评测** | GPQA Diamond 快速评测（fast_gpqa.py） | 步骤4快速精度评测（V1/V2） |
| **B — 远端 flageval 正式评测** | 远端 FlagRelease 平台，6 个数据集 | 独立调用（不参与主流程） |
| **C — 精度对比（V1 vs V2）** | 对比两版评测结果，5% 阈值判定 | 步骤4中 V2 评测完成后自动执行 |
| **D — 错误处理与算子排查** | 服务端报错 → 算子替换 → 重启 → 重评 | 精度不达标或评测报错时 |

---

## 强制约束

**启动前互斥检查**：精度评测启动前，必须确认没有正在运行的性能测试进程。并发执行会互相抢占 GPU 资源，导致结果不可信。

```bash
docker exec $CONTAINER bash -c "pgrep -f 'benchmark_runner\|vllm.*bench' && echo 'BLOCKED: 性能测试运行中，等待结束' && exit 1 || echo 'OK: 无性能测试进程'"
```

---

## 上下文集成

### 从容器内 /flagos-workspace/shared/context.yaml 读取

```yaml
container:
  name: <来自 container-preparation>
model:
  name: <来自 container-preparation>
  url: <来自 container-preparation>
  container_path: <来自 container-preparation>
service:
  cluster: <来自 service-startup>
  external_ip: <来自 service-startup>
  host: <来自 service-startup>
  port: <来自 service-startup>
  healthy: <来自 service-startup>
  enable_oplist_path: <来自 service-startup>
  enable_oplist_count: <来自 service-startup>
  initial_operator_list: <来自 service-startup>
gpu:
  vendor: <来自 container-preparation>
inspection:
  flaggems_control: <来自 pre-service-inspection>
  flaggems_logic: <来自 pre-service-inspection>
```

### 写入容器内 /flagos-workspace/shared/context.yaml

```yaml
eval:
  request_id: "<远端评测任务 ID>"
  domain: "<NLP|MM>"
  mode: "<评测模式>"
  eval_method: "<remote|local>"
  status: "<S|F|C|OOR|running|local>"
  results: {}
  v1_score: <V1 GPQA Diamond 分数>
  v2_score: <V2 GPQA Diamond 分数>
  v3_score: <V3 调优后 GPQA Diamond 分数, 仅步骤5触发时有值>
  accuracy_diff: <V1 - V2 精度下降, 正值=V2低于V1>
  accuracy_aligned: <true|false>
  accuracy_threshold: 5.0
  excluded_ops_accuracy: [<因精度问题关闭的算子>]
```

---

## 统一工作目录

```
容器内: /flagos-workspace/
  ├── eval/
  │   ├── fast_gpqa.py              ← 本地快速评测脚本
  │   ├── fast_gpqa_config.yaml     ← 快速评测配置
  │   └── gpqa_result.json          ← 评测结果（临时）
  ├── results/
  │   ├── gpqa_native.json          ← V1 精度结果
  │   ├── gpqa_flagos.json          ← V2 精度结果（步骤5触发时会被调优后结果覆盖）
  │   ├── gpqa_flagos_optimized.json ← V3 调优后精度结果（仅步骤5触发时生成）
  │   ├── eval_result.json          ← 远端评测结果
  │   └── ops_list.json             ← 当前算子列表
  ├── logs/
  │   └── eval_gpqa_progress.log    ← 评测进度日志
  └── config/
      └── eval_config.yaml          ← 评测配置快照
```

---

# 模块 A：本地快速评测（GPQA Diamond）

主模式：**GPQA Diamond 快速评测** — 一条命令跑完，自动适配所有模型（thinking/non-thinking），自动探测吞吐选并发。

## 核心特性

- **自动适配所有模型**：自动检测 thinking model（Qwen3/QwQ/DeepSeek-R1/R2），设置对应的 temperature/filters
- **自动 max_tokens**：查询 `/v1/models` 获取 `max_model_len`，thinking 模型 `max_tokens = max(max_model_len - 8192, 8192)` 不设上限 cap，标准模型 clamp 到 [4096, 32768]
- **截断检测**：评测前发样题检查 `finish_reason`，如果为 `length` 自动翻倍 max_tokens（在 max_model_len 允许范围内）
- **自动选并发**：三阶段探测 — 1 直接 API 调用测单条推理延迟（剥离框架开销）2 基于延迟 + thinking 特性估算候选并发 3 快速验证（每档 3 题并发测试，选吞吐最高且无错误的）
- **thinking 模型保守策略**：thinking 模型输出长度波动大，候选并发范围更保守（如延迟 ≤10s → [8,16,32]，而非 standard 的 [16,32,64]）

## 禁止行为

- 禁止因吞吐量低而手动降低 max_tokens。V1 和 V2 必须使用完全相同的 max_tokens 配置
- max_tokens 由 fast_gpqa.py 自动计算，编排层不得覆盖
- 如果 V2 推理慢，应等待完成而非降低参数

## 使用方式

**步骤 1：复制工具到容器**（已由 setup_workspace.sh 部署，仅在需要更新脚本/配置时执行）

```bash
CONTAINER=<container_name>
docker cp skills/flagos-eval-comprehensive/tools/fast_gpqa.py $CONTAINER:/flagos-workspace/scripts/fast_gpqa.py
docker cp skills/flagos-eval-comprehensive/tools/fast_gpqa_config.yaml $CONTAINER:/flagos-workspace/scripts/fast_gpqa_config.yaml
```

**步骤 2：安装依赖**

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH pip install evalscope pyyaml requests"
```

如使用 ModelScope 数据源（默认）：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH pip install modelscope"
```

**步骤 3：配置**

编辑容器内 `/flagos-workspace/scripts/fast_gpqa_config.yaml`：

```yaml
model:
  name: ""                              # 必填，与 /v1/models 返回的 id 一致
  api_base: "http://localhost:8000/v1"  # 必填，OpenAI 兼容 API 地址
  api_key: "EMPTY"                      # 可选

dataset_dir: ""                         # 可选，预下载缓存目录
dataset_hub: "modelscope"               # modelscope 或 huggingface
```

**步骤 4：运行评测**

```bash
# 方式一：使用配置文件
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --config fast_gpqa_config.yaml"

# 方式二：命令行参数（无需改配置文件）
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1"
```

## 输出

终端打印 + `gpqa_result.json`：

```
============================================================
  GPQA Diamond 快速评测结果
============================================================
  模型:     Qwen3-30B-A3B
  模式:     thinking (temperature=0.6, max_tokens=24576)
  并发:     32
  题数:     30
  得分:     61.11%
  耗时:     10m 25s
  报告:     gpqa_result.json
============================================================
```

## 自动决策逻辑

| 决策项 | 逻辑 |
|--------|------|
| max_tokens | 查询 `/v1/models` → thinking: `max(max_model_len - 8192, 8192)` 无上限 cap；standard: `clamp(max_model_len - 8192, 4096, 32768)`；fallback thinking=16384, standard=8192 |
| 截断检测 | 评测前发样题，`finish_reason == "length"` 时自动翻倍 max_tokens |
| temperature | thinking model → 0.6；standard → 0.0 |
| top_p | thinking → 0.95；standard → 1.0 |
| dataset_filters | thinking → `remove_until: </think>`；standard → 无 |
| eval_batch_size | 三阶段探测：API 延迟 → 候选估算 → 并发验证；探测失败 → 16 |
| few_shot | 始终 0-shot |
| stream | 始终开启 |

## Thinking 模型检测规则

模型名（不区分大小写）包含以下关键词即判定为 thinking model：
- `qwen3`、`qwq`、`deepseek-r1`、`deepseek-r2`

## 迁移流程中的用法

步骤4（快速精度评测）使用模块 A + C + D，模块 B（远端 flageval 正式评测）为独立工作，不参与主流程。

> **⚠ 硬性规则：每次启动/重启 vLLM 服务前，必须先 `docker restart $CONTAINER && sleep 5`**
>
> 旧 vLLM 进程占用 GPU 显存和端口，不 restart 容器会导致新服务启动失败。
> 推荐使用 `safe_restart_service.sh --container $CONTAINER --mode flagos` 一条命令完成。

**步骤4 — V1 (Native) 精度**（始终执行）：

1. 停止现有服务，释放 GPU 显存：
```bash
docker restart $CONTAINER
sleep 5
```

2. 关闭 FlagGems，以 native 模式启动服务：
```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/toggle_flaggems.py --action disable"
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH USE_FLAGGEMS=0 bash /flagos-workspace/scripts/start_service.sh --mode native > /flagos-workspace/logs/startup_native.log 2>&1"
```
等待服务就绪：
```bash
docker exec $CONTAINER bash -c "bash /flagos-workspace/scripts/wait_for_service.sh --port $PORT --model-name '$MODEL_NAME' --timeout 120 --max-timeout 900 --log-path /flagos-workspace/logs/startup_native.log --mode native"
```

3. 运行评测（通过 eval_wrapper.py 包装，自动监控服务状态和评测进度）：
```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 eval_wrapper.py \
    --eval-cmd 'python3 fast_gpqa.py --config fast_gpqa_config.yaml --output /flagos-workspace/results/gpqa_native.json' \
    --service-log /flagos-workspace/logs/startup_native.log \
    --stall-timeout 300 --max-timeout 3600"
```
eval_wrapper.py 会阻塞直到评测完成或异常，无需轮询。退出码 0 表示成功（最后一行为结果 JSON），非 0 表示异常（输出 [EVAL_ERROR] 错误摘要）。

4. **V1 评测完成后，必须停止服务释放 GPU**：
```bash
docker restart $CONTAINER
sleep 5
```

**步骤4 — V2 (FlagGems) 精度**（始终执行）：

1. 启用 FlagGems（调用 service-startup skill 以 flagos 模式启动）。**注意**：如果 context.yaml 的 `optimization.disabled_ops` 非空（步骤3崩溃诊断已禁用部分算子），必须先写入控制文件再启动：
   - 从 `results/ops_list.json` 读取全量算子，排除 `disabled_ops`，生成白名单
   - 写入 `/root/flaggems_ops_control.json`（格式：`{"include": [启用算子列表]}`）
   - `start_service.sh` 会自动从控制文件推断 `FLAGGEMS_CONTROL_MODE=only_enable`
   - 通过 `start_service.sh` 以 flagos 模式启动

2. 运行评测：
```bash
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 eval_wrapper.py \
    --eval-cmd 'python3 fast_gpqa.py --config fast_gpqa_config.yaml --output /flagos-workspace/results/gpqa_flagos.json' \
    --service-log /flagos-workspace/logs/startup_flagos.log \
    --stall-timeout 300 --max-timeout 3600"
```

**强制规则**：V1 和 V2 必须使用相同的 GPU 配置（`CUDA_VISIBLE_DEVICES` 和 `TP_SIZE`），复用 context.yaml 中首次启动时写入的值，禁止重新检测 GPU。

**V2 评测完成后，自动进入模块 C（精度对比）。**


## 工具文件

```
tools/
├── eval_wrapper.py            ← 评测包装器（启动+监控，阻塞等待，无需轮询）
├── fast_gpqa.py              ← 快速 GPQA Diamond 评测（主入口）
├── fast_gpqa_config.yaml     ← 快速评测配置模板
├── accuracy_compare.py       ← V1 vs V2 精度对比与阈值判定
├── eval_monitor.py           ← 远端评测监控（提交→轮询→结果获取）
├── eval_orchestrator.py      ← 全量评测编排器（保留，按需使用）
├── evalscope_runner.py       ← EvalScope 执行器（保留）
├── config.yaml               ← 全量评测配置（保留）
├── benchmark_registry.yaml   ← Benchmark 注册表（保留）
└── datasets/evalscope_cache/ ← 数据集缓存
```

### tools/fast_gpqa.py — 快速 GPQA Diamond 评测（核心）

单文件约 470 行，包含完整的 GPQA Diamond 评测流程：

1. 加载配置 / 解析 CLI 参数
2. 验证 API 可达性
3. 检测 thinking model
4. 查询 `/v1/models` → 自动计算 max_tokens
5. 截断检测 → 发样题检查 finish_reason，必要时自动调整 max_tokens
6. 设置 generation_config
7. 三阶段并发探测（API 延迟 → 候选估算 → 并发验证）→ 选并发
8. 正式评测 198 题
9. 解析结果 → 输出 JSON 报告 + 终端打印

| CLI 参数 | 说明 |
|----------|------|
| `--config` | 配置文件路径 |
| `--model-name` | 模型名称（覆盖 config） |
| `--api-base` | API 地址（覆盖 config） |
| `--api-key` | API 密钥（覆盖 config） |
| `--dataset-dir` | 数据集缓存目录（覆盖 config） |

---

# 模块 B：远端 flageval 正式评测

通过远端 FlagRelease 平台进行大模型正确性评测，支持 6 个数据集。

**远端不可达时自动降级到模块 A（本地快速评测）。**

## 远端评测平台 API 参考

### 平台地址

| 环境 | 地址 | 说明 |
|------|------|------|
| **公网（当前使用）** | `http://110.43.160.159:5050` | 线上环境（原 120.92.17.239 维修中） |
| 本地 | `http://127.0.0.1:5051` | 本地测试环境 |

### API 接口一览

| 接口 | 方法 | 路径 | 用途 |
|------|------|------|------|
| 提交评测 | POST | `/evaluation` | 发起评测任务，返回 request_id |
| 查询进度 | POST | `/evaluation_progress` | 查询任务执行进度 |
| 获取结果 | GET | `/evaldiffs` | 获取最终评测结果 |
| 停止评测 | POST | `/stop_evaluation` | 停止正在运行的任务 |
| 恢复评测 | POST | `/resume_evaluation` | 恢复已停止的任务 |
| 对比评测 | GET | `/evaluation_diffs` | 对比多个评测任务的差异 |

### 正式评测数据集（6 个）

| 数据集 | 说明 |
|--------|------|
| `gpqa_generative_cot` | GPQA 生成式 CoT |
| `aime` | AIME 数学竞赛 |
| `livebench_new` | LiveBench 最新版 |
| `musr_generative` | MuSR 生成式推理 |
| `mmlu_pro` | MMLU-Pro 专业知识 |
| `gpqa_diamond_generative_cot` | GPQA Diamond 生成式 CoT |

## 入口判断

| 模式 | 说明 |
|------|------|
| **提交新评测** | 通过远端 FlagRelease API 提交新评测任务（默认） |
| **查询已有任务** | 用户提供 request_id，查询进度或获取结果 |

## 步骤 B1：服务稳定性预检

在提交评测之前，先用一条极简 benchmark 验证服务不会崩溃：

```bash
docker exec $CONTAINER bash -c "vllm bench serve \
  --host <service_host> --port <service_port> \
  --model <model_name> --tokenizer <tokenizer_path> \
  --dataset-name random --random-input-len 1024 --random-output-len 15 \
  --num-prompts 1 --endpoint /v1/completions --ignore-eos --trust-remote-code"
```

- 返回码 `0` → 服务稳定，继续
- 返回码非 `0` → 停止评测，跳转到模块 D（错误处理）

## 步骤 B2：确定评测参数

从 context.yaml 读取服务信息，询问用户确认或补充：

| 参数 | 来源 | 说明 |
|------|------|------|
| `eval_model` | 用户提供 | 评测唯一名称，如 `qwen2.5-7b-nv-flagos` |
| `model` | context.yaml `model.name` | 大模型名称 |
| `eval_url` | **询问用户提供本机 IP** | `http://<IP>:<port>/v1/chat/completions` |
| `tokenizer` | 用户提供 | Tokenizer 路径 |
| `domain` | 用户选择 | `NLP` 或 `MM` |
| `mode` | 用户选择 | NLP: `FlagRelease`/`XLC_train`/`XLC_infer`/`Qnext`/`quickrun`；MM: `FlagRelease`/`XLC`/`EmbodiedVerse`/`RoboTrain`/`quickrun` |
| `chip` | 自动检测 | 芯片名称，如 `Nvidia-H100` |
| `api_key` | 默认 `EMPTY` | API 密钥 |
| `batch_size` | 默认 `1` | |
| `num_concurrent` | 默认 `1` | 并发数 |
| `num_retry` | 默认 `10` | 重试次数 |
| `gen_kwargs` | 可选 | 如 `temperature=0.6,top_k=20,max_gen_toks=16000` |
| `region` | 默认 `bj` | `bj`（大兴）或 `sz`（上庄） |
| `user_id` | 用户提供或默认 `0` | FlagEval 平台用户 ID（**整数类型**） |
| `dry_run` | 默认 `false` | 是否仅做推理验证 |

**eval_model 命名规范**：
- V1 版本（baseline）：`<model>-<vendor>-origin`
- V2 版本（FlagOS）：`<model>-<vendor>-flagos`

## 步骤 B3：参数预检与确认

1. **询问用户提供本机 IP**（禁止自动获取，必须用户确认）
2. **验证服务可达性**：`curl -s --connect-timeout 5 http://<IP>:<port>/v1/models`
3. **组装完整参数展示给用户确认**
4. **参数类型检查**：`user_id` 为整数，`dry_run` 为布尔值

## 步骤 B4：提交评测任务

```bash
curl -X POST http://110.43.160.159:5050/evaluation \
-H "Content-Type: application/json" \
-d '{
    "eval_model": "<eval_model>",
    "model": "<model>",
    "eval_url": "http://<external_ip>:<port>/v1/chat/completions",
    "tokenizer": "<tokenizer>",
    "domain": "<NLP|MM>",
    "mode": "<mode>",
    "chip": "<chip>",
    "api_key": "<api_key>",
    "batch_size": <batch_size>,
    "num_concurrent": <num_concurrent>,
    "num_retry": <num_retry>,
    "gen_kwargs": "<gen_kwargs>",
    "region": "<region>",
    "dry_run": <dry_run>,
    "user_id": <user_id>
}'
```

- `err_code == 0`：提交成功，**记录 `request_id`**
- `err_code == 1`：提交失败
- 网络不可达 → 降级到模块 A

## 步骤 B5：监控评测进度

```bash
curl -X POST http://110.43.160.159:5050/evaluation_progress \
-H "Content-Type: application/json" \
-d '{"request_id": "<request_id>", "domain": "<NLP|MM>"}'
```

**轮询策略**：
- 第 1-5 次：每 30 秒
- 第 6-15 次：每 60 秒
- 第 16-30 次：每 2 分钟
- 进度 > 80%：切换到每 30 秒密集轮询
- 最多 30 次（约 1 小时），超出后停止
- `finished == true` → 获取结果
- 连续 3 次网络不可达 → 停止轮询，输出 request_id

## 步骤 B6：获取评测结果

```bash
curl -X GET http://110.43.160.159:5050/evaldiffs \
-H "Content-Type: application/json" \
-d '{"request_id": "<request_id>"}'
```

**结果状态**：

| status | 含义 | 后续操作 |
|--------|------|----------|
| `S` | 成功 | 输出精度报告 → 完成 |
| `F` | 失败 | 跳转模块 D（错误处理） |
| `C` | 已取消 | 询问用户是否恢复 |
| `OOR` | 超过重试次数 | 跳转模块 D |

## 步骤 B7：查询已有任务（用户提供 request_id）

```bash
# 查询进度
curl -X POST http://110.43.160.159:5050/evaluation_progress \
-H "Content-Type: application/json" \
-d '{"request_id": "<request_id>", "domain": "<NLP|MM>"}'

# 获取结果
curl -X GET http://110.43.160.159:5050/evaldiffs \
-H "Content-Type: application/json" \
-d '{"request_id": "<request_id>"}'

# 停止任务
curl -X POST http://110.43.160.159:5050/stop_evaluation \
-H "Content-Type: application/json" \
-d '{"request_id": "<request_id>"}'

# 恢复任务
curl -X POST http://110.43.160.159:5050/resume_evaluation \
-H "Content-Type: application/json" \
-d '{"request_id": "<request_id>"}'
```

## 步骤 B8：对比多个评测任务（V1 vs V2 远端对比）

```bash
curl -X GET http://110.43.160.159:5050/evaluation_diffs \
-H "Content-Type: application/json" \
-d '{"request_ids": ["<request_id_v1>", "<request_id_v2>"]}'
```

---

# 模块 C：精度对比（V1 vs V2）

## 触发时机

步骤4（快速精度评测）中 V2 精度完成后，如果 V1 精度有结果，**自动执行对比**。

## 对比逻辑

通过 `accuracy_compare.py` 脚本执行对比：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/accuracy_compare.py \
    --v1 /flagos-workspace/results/gpqa_native.json \
    --v2 /flagos-workspace/results/gpqa_flagos.json \
    --output /flagos-workspace/results/accuracy_compare.json \
    --json"
```

脚本逻辑：
1. 读取 `results/gpqa_native.json`（V1）和 `results/gpqa_flagos.json`（V2）
2. 提取 `score` 字段
3. 计算 `accuracy_diff = V1_score - V2_score`（正值表示V2下降）
4. 判定：`accuracy_diff ≤ 5.0%` → 通过（退出码 0）；`> 5.0%` → 不达标（退出码 1）。V2高于V1也算达标

| CLI 参数 | 说明 |
|----------|------|
| `--v1` | V1 (Native) 评测结果 JSON |
| `--v2` | V2 (FlagGems) 评测结果 JSON |
| `--threshold` | 下降阈值百分比（默认 5.0%） |
| `--json` | JSON 格式输出 |
| `--output` | 结果输出文件路径 |

## 对比输出格式

```
精度对比 (V1 vs V2)
========================================
V1 (Native):        62.12%
V2 (Full FlagGems):  59.09%
偏差:                3.03%
阈值:                5.0%
结论:                通过 ✓
当前启用算子:         38 个
========================================
```

## 精度不达标时的处理

V2精度下降 > 5% 时：

1. 输出偏差报告 + 当前启用算子列表（从 `flaggems_enable_oplist.txt` 读取）
2. 触发 `diagnose_ops.py accuracy-groups` 分组定位：
   ```bash
   ${CMD_PREFIX} python3 /flagos-workspace/scripts/diagnose_ops.py accuracy-groups \
     --ops-file /flagos-workspace/results/ops_list.json \
     --plugin-mode --json
   ```
3. 按组逐轮累积禁用测试：第1轮禁用组A，第2轮禁用组A+B，第3轮禁用组A+B+C
4. **每轮算子控制方式**（与性能调优 operator_search.py 一致）：
   - 从 `diagnose_ops.py` 输出的当轮 `cumulative_test_env.control_file` 获取白名单
   - 写入控制文件 `/root/flaggems_ops_control.json`（格式：`{"include": [启用算子列表]}`）
   - 通过 `start_service.sh` 启动服务（`start_service.sh` 会自动从控制文件推断 `FLAGGEMS_CONTROL_MODE=only_enable`，无需手动写 `/etc/environment`）
   - **不使用** `toggle_flaggems.py --action modify-enable --disabled-ops`
5. 某轮累积禁用后精度恢复（下降 ≤5%）→ 达标即停，保留所有已累积禁用的算子
6. **每轮输出算子状态**（见下方格式）

## 每轮算子状态输出（强制）

```
算子状态更新
========================================
本轮关闭: softmax, layer_norm（原因: 精度下降 >5%）
累计关闭: 3 个 (softmax, layer_norm, rms_norm)
当前启用: 35 个
启用列表: [addmm, mm, bmm, ...]
========================================
```

## 写入 context.yaml

对比完成后更新：
```yaml
eval:
  v1_score: 62.12
  v2_score: 59.09
  accuracy_diff: 3.03
  accuracy_aligned: true
  excluded_ops_accuracy: []  # 或 [softmax, layer_norm]
```

## 模块 C 完成后的强制闭环（不可跳过）

精度对比完成后，编排层**必须**按以下逻辑执行：

```
完成 V1 + V2 评测后:

IF drop > threshold (V1 - V2 > 5%):
    # 1. 写 issue log（强制）
    追加写入 logs/issues_accuracy.log:
      "[时间] V2 | 精度下降超阈值"
      "  详情: V1=XX%, V2=XX%, 下降=XX% (阈值 5%)"

    # 2. 设置状态
    workflow.accuracy_ok = false

    # 3. 触发步骤5精度算子调优（不终止流程）
    #    5完成后再继续到步骤6性能评测

ELSE:  # V2下降≤5% 或 V2高于V1，均达标
    workflow.accuracy_ok = true

→ accuracy_ok=false 时触发步骤5精度算子调优，5完成（或跳过）后继续步骤6性能评测
```

**禁止行为**：V2精度下降 > 5% 时终止流程。必须写入 issue log、标记 `accuracy_ok=false`，然后继续后续步骤，最终走到发布（私有发布）。

---

# 模块 D：错误处理与算子排查

## 错误分类

```
评测报错
  │
  ├── 服务端报错（算子问题）
  │     特征: CUDA error, OOM, RuntimeError, operator not supported,
  │           服务进程退出, status=F/OOR（远端结果中）
  │     → D1: 算子替换 → 重启服务 → 重新评测
  │
  └── 评测端/网络问题
        特征: timeout, connection refused, 平台 5xx, DNS 解析失败
        → D2: 降级到模块 A 本地评测
```

## D1 — 服务端报错处理（算子替换闭环）

**此流程可多轮迭代，直到评测通过或用户终止。**

**第 1 步：分析报错原因**

远端评测结果中发现失败：
```bash
# 检查远端返回的结果中 status 为 F 或 OOR 的数据集
```

本地评测日志中发现失败：
```bash
grep -iE "(CUDA|OOM|RuntimeError|operator.*not.*support|process.*exit)" \
    /data/flagos-workspace/<model_name>/output/**/*.log
```

**第 2 步：整理报错报告**

向用户输出：
- 错误类型（CUDA error / OOM / 算子不支持等）
- 涉及的算子名称（从日志中提取）
- 建议关闭的算子列表

**第 3 步：触发算子替换**

调用 `flagos-operator-replacement` skill：
- 根据错误信息确定需要排除的算子
- 执行替换并记录

**第 4 步：重启服务（使用 safe_restart_service.sh）**

```bash
# 推荐：一条命令完成 restart + start + wait（宿主机执行）
bash skills/flagos-service-startup/tools/safe_restart_service.sh \
    --container $CONTAINER --mode flagos \
    --log-name startup_round${ROUND}.log \
    --model-name "$MODEL_NAME" --port $PORT \
    --env "FLAGGEMS_CONTROL_MODE=only_enable"
```

或手动三步（不推荐，容易遗漏）：
```bash
docker restart $CONTAINER && sleep 5
docker exec -d $CONTAINER bash -c "cd /flagos-workspace && PATH=/opt/conda/bin:\$PATH bash /flagos-workspace/scripts/start_service.sh --mode flagos > /flagos-workspace/logs/startup_round${ROUND}.log 2>&1"
docker exec $CONTAINER bash -c "bash /flagos-workspace/scripts/wait_for_service.sh --port 8000 --model-name '$MODEL_NAME' --log-path /flagos-workspace/logs/startup_round${ROUND}.log --mode flagos"
```

> **禁止**：跳过 docker restart 直接启动新服务。

**第 5 步：重新评测**

- 远端评测 → 回到步骤 B4 重新提交
- 本地评测 → 回到模块 A 重新运行

**第 6 步：输出算子状态（每轮强制）**

```
算子状态更新
========================================
本轮关闭: softmax（原因: CUDA error in eval）
累计关闭: 1 个 (softmax)
当前启用: 37 个
启用列表: [addmm, mm, bmm, ...]
========================================
```

**迭代控制**：
- 每轮记录关闭了哪些算子
- 最多迭代 3 轮，超限标记 `workflow.accuracy_ok: false` 进入步骤4
- 每轮输出算子状态报告

### 精度问题日志写入

V2精度下降 >5% 或评测过程中服务端报错时，必须追加写入 `logs/issues_accuracy.log`：

```bash
docker exec $CONTAINER bash -c "cat >> /flagos-workspace/logs/issues_accuracy.log << 'ISSUE_EOF'
[$(date '+%Y-%m-%d %H:%M:%S')] <版本> | <问题摘要>
  详情: <V1/V2 分数、下降值，或错误信息>
  操作: <排查措施，如 diagnose_ops.py 分组测试、禁用算子>
  结果: <措施结果，如下降降至 X%、服务恢复>
ISSUE_EOF"
```

记录场景：
- V1 vs V2 精度下降 >5%（记录双方分数和下降值）
- 评测过程中服务端 CUDA error / OOM / 进程崩溃
- 算子排查每轮结果（禁用了哪些算子、重测后的下降）

## D2 — 评测端/网络问题处理

1. 确认是网络/平台问题而非服务端问题
2. 检查服务是否正常：`curl -s http://localhost:8000/v1/models`
3. 降级到模块 A 进行本地评测

---

# 阶段性反馈格式

每次评测完成后，必须向用户输出：

```
精度评测结果
========================================
版本:   V1 (Native) / V2 (Full FlagGems)
方式:   本地 GPQA Diamond / 远端 flageval
状态:   通过 / 有报错
得分:   XX.XX%
问题算子: softmax, layer_norm（仅有报错时显示）
建议操作: 关闭问题算子 → 重启服务 → 重新评测
已累计剔除: 2 个算子（softmax, layer_norm）
当前启用: 36 个算子
========================================
```

**反馈规则**：
- 状态为"通过"时不显示"问题算子"和"建议操作"
- 每次重新评测后更新"已累计剔除"计数
- 如果 3 轮优化后仍不达标，标记 `workflow.accuracy_ok: false`，进入下一步

---

# 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `evalscope not found` | 未安装 | `pip install evalscope` |
| API 不可达 | 服务未启动或地址错误 | 检查 `--api-base`，确认 `curl <api_base>/models` 正常 |
| 精度异常低 | max_tokens 不够 | 脚本自动计算+截断检测，检查日志中 `truncation_detected` 是否为 true |
| 探测选并发偏保守 | thinking 模型延迟波动大 | 三阶段探测已内置 thinking 保守策略，验证阶段会实测选最优 |
| model_id 含 `/` 报路径错误 | 模型名是路径格式 | 已内置 sanitize 逻辑，自动取最后一段 |
| 远端提交失败 `err_code=1` | 参数错误 | 检查 eval_infos 参数格式 |
| 连接远端平台超时 | 平台不可达 | 降级到模块 A 本地评测 |
| V1 vs V2 精度下降 >5% | FlagGems 算子精度问题 | 触发模块 C 精度排查流程 |

---

# 完成标准

- [ ] 评测完成（本地或远端）
- [ ] 评测结果已保存到 `results/` 目录
- [ ] score 字段有值（非 null）
- [ ] V1 vs V2 精度对比已执行（如两版结果都有）
- [ ] 如有错误，已完成错误分析和算子替换处理
- [ ] 精度异常或评测报错时，`logs/issues_accuracy.log` 已追加写入问题记录
- [ ] **每轮算子变更已输出状态报告**（关闭的算子 + 当前启用算子）
- [ ] context.yaml 已更新评测结果和精度对比字段
- [ ] 评测配置快照已保存到 `config/eval_config.yaml`
- [ ] 对应 trace 文件已写入：
  - V1 评测 → 记录在 `traces/04_quick_accuracy.json` 中
  - V2 评测 → 记录在 `traces/04_quick_accuracy.json` 中
- [ ] `timing.steps.quick_accuracy` 已更新为本步骤的 `duration_seconds`
- [ ] 更新报告：`docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"`

---

## 编排层指令（步骤4 精度评测 — 固化决策）

主流程步骤4使用**模块 A（本地 GPQA Diamond）+ 模块 C（精度对比）+ 模块 D（错误处理）**，不使用模块 B（远端 flageval）。

执行顺序（固定）：
1. 关闭 flaggems → 启动服务 → GPQA Diamond V1 精度基线 → 停服务
2. 开启 flaggems → 启动服务 → GPQA Diamond V2 精度
3. V1 vs V2 精度对比（下降阈值 5%）
4. 结果判定：
   - 下降 ≤5%（含V2高于V1）→ `workflow.accuracy_ok=true`，进入步骤6
   - 下降 >5% → 必须按顺序：① 标记 `accuracy_ok=false` ② issue_reporter.py --type accuracy-degraded ③ 追加 `logs/issues_accuracy.log` → 触发步骤5
   - 服务崩溃 → issue_reporter.py --type operator-crash（同步骤3）

精度评测+精度调优全部完成后才进入性能测试。

---

## 编排层指令（步骤5 精度算子调优 — 固化决策）

**触发条件**：`workflow.accuracy_ok = false` 且 `env_type ≠ native`
**跳过条件**：`accuracy_ok = true`（不触发时显示已完成）

固化选择：
- 使用 `diagnose_ops.py accuracy-groups` 分组排查（不使用逐个排查）
- **最多 3 轮**（3 组），达标即停
- **累积禁用**：第1轮禁用组A，第2轮禁用组A+B，第3轮禁用组A+B+C（每轮在上一轮基础上追加禁用）
- 达标标准：累积禁用后 V2 精度下降 ≤5%（相对 V1）
- **算子控制方式**：每轮使用 `cumulative_test_env.control_file` 白名单写入 `/root/flaggems_ops_control.json`，设置 `FLAGGEMS_CONTROL_MODE=only_enable`（与性能调优 operator_search.py 一致）

执行后必须完成：
- 写入 `context.yaml` 的 `eval.excluded_ops_accuracy` 和 `optimization` 字段
- 写入 `traces/05_accuracy_tuning.json`
- 保存算子列表：`docker exec $CONTAINER cp /tmp/flaggems_enable_oplist.txt /flagos-workspace/results/accuracy_tuned_oplist.txt`
- 更新报告：`docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"`

**注意**：精度调优禁用的算子会传递给后续步骤6，6在此算子集上采集性能基线。
