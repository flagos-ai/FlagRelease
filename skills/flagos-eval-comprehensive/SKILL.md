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

# 模块 A：本地快速评测（GPQA Diamond / MMLU / MATH-500）

主模式：**快速评测** — 一条命令跑完，自动适配所有模型（thinking/non-thinking），自动探测吞吐选并发。数据集通过 `--dataset` 指定（默认 `gpqa_diamond`）：

| 数据集 | --dataset 值 | 默认 few-shot | 默认题数 | 说明 |
|--------|-------------|--------------|---------|------|
| GPQA Diamond | `gpqa_diamond` | 0-shot | 50（--limit 0 全量 198） | 主流程 V1/V2 精度判据 |
| MMLU | `mmlu` | 5-shot | 20/子集 | **per-subset 语义**：57 子集各取 limit 题（默认 20/子集 = 1140 题；--limit 100 = 5700 题，--limit 0 = 全量 14042） |
| MATH-500 | `math_500` | 0-shot | 200（40/等级） | 500 题 test，`\boxed{}` 答案；**同为 per-subset 语义**（实测 1.5.1：5 子集 Level 1-5 各取 limit 题，默认 40/等级 = 200 题，--limit 0 = 全量 500） |
| MMStar | `mm_star` | 0-shot | **全量 1500**（默认；降采样显式传 --limit N/子集） | **多模态（VLM）视觉必答 4 选一 MCQ**，仅对视觉语言模型有意义（纯文本 LLM 收到图片会报错/胡答，且 vLLM 须以支持图像输入方式起服务）；6 子集各 250 题（**per-subset 语义**，reformat_subset=True）；**默认全量**（采样偏最难子集会显著偏低：300 题 44.67% vs 全量 62.6%）；多模态专用 max_tokens 硬上限 4096（MCQ 答案短，防复读死循环吃满窗口）、prompt 预留 16K、并发档位对半下调 |

> 默认题数降采样为 2026-08-14 定稿（迁移成本优化）：mmlu 20/子集 gap 实测 -0.4pt（95% ±2.05pt）、math 200 题 ±5.4pt（用户已确认接受），详见 `eval-sampling-gap-measured` 记忆。
> MMStar 全量 1500 题实测（Qwen3-VL-4B-Instruct，2026-08-28）：max_tokens=4096 + 遵循模型 generation_config.json 采样参数下 62.6%，约 15 分钟（并发 8，~0.6s/题）；采样偏最难子集（300 题）会显著偏低，多模态精度评测建议 `--limit 0` 全量。

**多数据集**：`--dataset` 支持一次指定多个（空格或逗号分隔），依次评测、每数据集独立结果文件；多数据集时 `--output` 视为目录（写 `{dir}/{dataset}_result.json`），单数据集时仍为文件路径：

```bash
# 一次跑多个数据集（mmlu 1140 题 + math_500 200 题，各用默认题数）
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 \
    --dataset mmlu math_500 --output /flagos-workspace/results/multi"

# 逗号分隔等价写法
--dataset mmlu,math_500

# 也可在 config 指定多数据集: dataset: "mmlu, math_500"
```

多数据集时 `--limit` 对所有数据集生效（per-subset 数据集按各子集取数）；不传则各数据集用默认题数。结果汇总表在终端打印，退出码 = 所有数据集得分都解析成功才为 0。

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
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH pip install 'evalscope==1.5.1' pyyaml requests"
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
dataset: "gpqa_diamond"                 # 可选，gpqa_diamond / mmlu / math_500 / mm_star（多模态，仅 VLM）
```

**步骤 4：运行评测**

```bash
# 方式一：使用配置文件
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --config fast_gpqa_config.yaml"

# 方式二：命令行参数（无需改配置文件）
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1"

# 方式三：指定数据集（MMLU 57 子集各 20 题 / MATH-500 200 题，均用默认题数；--limit 100 / --limit 0 可取更大样本）
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 --dataset mmlu --limit 100"
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 --dataset math_500 --limit 0"
```

## 输出

终端打印 + `{dataset}_result.json`（gpqa_diamond → `gpqa_result.json`，mmlu → `mmlu_result.json`，math_500 → `math_500_result.json`，mm_star → `mm_star_result.json`）：

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
| few_shot | 按数据集：gpqa_diamond/math_500 → 0-shot；mmlu → 5-shot（evalscope 内置 few-shot 示例） |
| stream | 始终开启 |

## Thinking 模型检测规则

模型名（不区分大小写）包含以下关键词即判定为 thinking model（与 fast_gpqa.py THINKING_PATTERNS 一致）：
- `qwen3`、`qwq`、`deepseek-r1`、`deepseek-r2`、`mimo`、`hunyuan`
- 或 context.yaml 的 `runtime.thinking_model=true` / `model.thinking_model=true`（优先级最高）

## 评测时间预算（2026-08-10 定稿）

**背景**：thinking 模型正常评测 7-10h（慢速芯片思考链 500-720s/题），原 max-timeout 写死 2h 必然超时 → 评测反复失败 → 自发跳过评测 → 数据缺失。本规则按题数×单题预算×1.25 缓冲计算确定性预算，消除超时根因。

| 模型类型 | --limit | 单题预算 | max_timeout |
|---------|---------|---------|-------------|
| thinking | 30 | 600s | 22500（约 6.3h） |
| 普通 | 50 | 60s | 7200（维持原 2h 语义） |

- **V1 与 V2 必须使用相同参数**：thinking 模型两侧均 `--limit 30`，普通模型两侧均默认 50 题（同样本可对比，禁止一方 30 一方 50）
- **非 gpqa 数据集预算**：mmlu 默认 20/子集 = 1140 题（57 子集 × 20），math_500 默认 40/等级 = 200 题。task_runner `--timeout` 按「题数 × 单题耗时（实测 1.5B 模型约 0.2-3s/题）× 1.25 缓冲」估算，如 mmlu 1140 题建议 ≥21600s，math_500 200 题 ≥7200s（timeout 是防挂死上限而非性能预期，国产慢卡 10×~30× 也绝不误杀；--limit 100/--limit 0 取更大样本时相应上调）
- **mm_star（多模态）预算**：全量 1500 题（6 子集 × 250），max_tokens 硬上限 4096 使单题上界可控。实测 Qwen3-VL-4B-Instruct 并发 8 约 15 分钟（~0.6s/题），慢卡按 10×~30× 上调，`--timeout` 建议 ≥14400s；**多模态精度评测传 `--limit 0` 全量**（采样偏最难子集会显著偏低）。⚠ 前置门控：服务须以支持图像输入方式启动，纯文本 LLM 不适用 mm_star
- **禁止因耗时长跳过评测**：评测耗时长（尤其 thinking 模型 6h+）是预算内预期，严禁因等待时间长主动跳过、放弃或截断评测；预算内完成即为正常，超时按等待策略重试
- **执行方式（2026-08 长任务执行协议）**：Bash 工具前台命令有 10 分钟硬上限，超过自动转后台 + 批次控制器 10 分钟无输出判会话失败。**禁止**用 Bash(timeout=大数) 前台阻塞等待评测（旧 Bash timeout(ms) 列已废弃），**禁止** TaskOutput 轮询。评测命令写入任务文件后经 task_runner.py detached 启动（`--timeout <max_timeout>`），Claude 每 8 分钟短轮询状态文件，见下方「长任务执行协议」三步模板
- 安全网保障：预算调大不会让 runaway 无限拖 —— fast_gpqa max_tokens cap（防线1）锁死复读生成窗口；eval_wrapper 收尾停滞看门狗照杀（防线3），生成中停滞仅提示不杀（慢≠死），给足预算不会误杀正常长推理

### 长任务执行协议（评测等待 — 硬性三步）

```bash
# 1. 写任务命令文件（一条 docker exec，内容自由写无转义问题）：
docker exec $CONTAINER bash -c "mkdir -p /flagos-workspace/logs/tasks && cat > /flagos-workspace/logs/tasks/eval_v2.cmd << 'CMD_EOF'
cd /flagos-workspace/scripts
PATH=/opt/conda/bin:\$PATH python3 eval_wrapper.py --eval-cmd 'python3 fast_gpqa.py --config fast_gpqa_config.yaml --limit 30 --output <输出路径>' --service-log <服务日志路径> --stall-timeout 300 --max-timeout 22500
CMD_EOF"

# 2. detached 启动（一条命令立即返回，不等待）：
docker exec -d $CONTAINER bash -c "cd /flagos-workspace/scripts && PATH=/opt/conda/bin:\$PATH python3 task_runner.py --cmd 'bash /flagos-workspace/logs/tasks/eval_v2.cmd' --state /flagos-workspace/logs/tasks/eval_v2.state --log /flagos-workspace/logs/tasks/eval_v2.log --timeout 22500"

# 3. 短轮询（每 8 分钟一次，单条轮询命令 <10 分钟且每次都有输出，永不触发转后台/空闲判定）：
sleep 480 && docker exec $CONTAINER bash -c "cat /flagos-workspace/logs/tasks/eval_v2.state 2>/dev/null; echo '---'; tail -3 /flagos-workspace/logs/tasks/eval_v2.log"
#   status=running → 继续等待（重复上一条轮询命令）。若 state 长时间停在 running 且日志停止增长（上次 tail 内容无变化），用 pgrep -f 'eval_v2.cmd' 确认任务进程：进程存活=任务仍在跑（task_runner 可能失联，日志 fd 由任务持有仍会增长），继续等待；进程消失=任务已死，读日志诊断
#   status=done    → 评测成功（日志最后一行 [RESULT_JSON] 为结果），读取结果文件继续
#   status=error   → 读日志按 error type 处理：service_crash/service_exited → 重启服务后重试；stall/timeout → 检查日志后重试；eval_failed → 检查错误信息
#   status=timeout → 超过 --max-timeout 总闸，读日志诊断
```

**断点恢复（硬性）**：启动评测前先检查 `/flagos-workspace/logs/tasks/eval_v2.state`——若存在且 `status=running`，说明上一会话已启动评测（会话被杀任务继续跑），**直接接管轮询，禁止重复启动评测**；`status=done/error` 则按终态直接处理。

（普通模型把 `--limit 30 --max-timeout 22500` 换为 `50/7200`；V1 评测用 `eval_v1.cmd/eval_v1.state`，V2 用 `eval_v2.*`，Plugin 步骤11 用 `plugin_eval.*`。）

**Plugin 步骤11（V3 精度）与 V2 步骤4 口径完全一致**：对全部 `--datasets` 数据集**逐个独立评测、独立判定，全部达标才 `plugin_workflow.accuracy_ok=true`**。多数据集时每个数据集一个独立任务 `plugin_eval_{prefix}.*`（`--dataset {ds}`、输出 `{prefix}_flagos_optimized.json`、判定 `accuracy_compare.py --metric {ds} --output accuracy_compare_{prefix}_v3.json`）；不达标时**按不达标数据集逐个**在 plugin 模式关算子调优（`operator_search.py --plugin-mode --dataset {ds}`），已达标数据集不重复评测。V4 减算子（`operator_reduction.py`）同样以 `--accuracy-datasets` 对全部数据集做精度终检，全部达标才 V4 成立。

## 迁移流程中的用法

步骤4（快速精度评测）使用模块 A + C + D，模块 B（远端 flageval 正式评测）为独立工作，不参与主流程。

> **⚠ 硬性规则：每次启动/重启 vLLM 服务前，必须先 `docker restart $CONTAINER && sleep 5`**
>
> 旧 vLLM 进程占用 GPU 显存和端口，不 restart 容器会导致新服务启动失败。
> 推荐使用 `safe_restart_service.sh --container $CONTAINER --mode flagos` 一条命令完成。

**步骤4 — V1 (Native) 精度**（始终执行）：

> **多数据集**：以下模板为默认场景 gpqa_diamond 单数据集。`--datasets` 指定多数据集（或 mmlu/math_500）时，**每个数据集独立评测**：eval-cmd 加 `--dataset {ds}`（如 `--dataset mmlu`）、输出 `{ds}_native.json`、任务文件 `eval_v1_{ds}.cmd`；`--limit` 仅 gpqa_diamond 单数据集时传（30/50），mmlu/math_500 及多数据集不传（fast_gpqa 按数据集默认题数：mmlu 20/子集=1140、math_500 40/等级=200）。判定见模块 C 与下方编排层指令（per-dataset、`--metric {ds}`、输出 `accuracy_compare_{ds}.json`）。

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
等待服务就绪（**长任务执行协议**：写 startup_native.cmd → task_runner detached 启动 → sleep 480 短轮询，见上方协议模板；此处 --max-timeout 900 较短可前台直跑，但服务启动可能超过 10 分钟时一律走协议）：
```bash
docker exec $CONTAINER bash -c "bash /flagos-workspace/scripts/wait_for_service.sh --port $PORT --model-name '$MODEL_NAME' --timeout 120 --max-timeout 900 --log-path /flagos-workspace/logs/startup_native.log --mode native"
```

3. 运行评测（通过 eval_wrapper.py 包装，自动监控服务状态和评测进度；**参数按上方「评测时间预算」表选择**——thinking 模型 `--limit 30 --max-timeout 22500`，普通模型 `--limit 50 --max-timeout 7200`）：
```bash
# 以 thinking 模型为例（普通模型把 --limit 30 --max-timeout 22500 换为 50/7200）
# 评测命令（写入 eval_v1.cmd 经 task_runner detached 启动，禁止 Bash(timeout=大数) 前台阻塞）：
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 eval_wrapper.py \
    --eval-cmd 'python3 fast_gpqa.py --config fast_gpqa_config.yaml --limit 30 --output /flagos-workspace/results/gpqa_native.json' \
    --service-log /flagos-workspace/logs/startup_native.log \
    --stall-timeout 300 --max-timeout 22500"
```
按上方「长任务执行协议」三步执行（eval_v1.cmd/eval_v1.state）。eval_wrapper.py 退出码 0 表示成功（最后一行为结果 JSON），非 0 表示异常（输出 [EVAL_ERROR] 错误摘要）；状态文件 status=done/error 对应处理，断点恢复见协议（禁止重复启动评测）。

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

2. 运行评测（**参数与 V1 完全一致**——同一模型同一套 `--limit`/`--max-timeout`，thinking 模型两侧均 `--limit 30 --max-timeout 22500`，普通模型两侧均 `--limit 50 --max-timeout 7200`；禁止 V1/V2 参数不同）：
```bash
# 以 thinking 模型为例（与 V1 相同）
docker exec $CONTAINER bash -c "cd /flagos-workspace/scripts && \
    PATH=/opt/conda/bin:\$PATH python3 eval_wrapper.py \
    --eval-cmd 'python3 fast_gpqa.py --config fast_gpqa_config.yaml --limit 30 --output /flagos-workspace/results/gpqa_flagos.json' \
    --service-log /flagos-workspace/logs/startup_flagos.log \
    --stall-timeout 300 --max-timeout 22500"
```

**强制规则**：V1 和 V2 必须使用相同的 GPU **卡数**（`TP_SIZE` 一致，卡数锁定见 service-startup SKILL 步骤 2.4）。`CUDA_VISIBLE_DEVICES` 允许换物理卡（优先复用 context 中上次的值），但 `runtime.gpu_count` 与 `TP_SIZE` 复用首次启动锁定的值，不得变更。

**V2 评测完成后，自动进入模块 C（精度对比）。**


## 工具文件

```
tools/
├── task_runner.py            ← 长任务执行器（detached 启动 + 状态文件 + 超时管理 + 断点恢复，长任务执行协议核心）
├── eval_wrapper.py            ← 评测包装器（启动+监控，输出结果 JSON / [EVAL_ERROR]，经 task_runner 协议执行）
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
8. 正式评测（默认 50 题；thinking 模型按时间预算用 `--limit 30`；仅 `--limit 0` 时才跑全量 198 题，常规流程不使用）
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
3. 计算 `rel_drop = (V1_score - V2_score) / V1_score`（相对退化，正值表示 V2 下降）
4. 判定：`rel_drop ≤ 5%` → 通过（退出码 0）；`> 5%` → 不达标（退出码 1）。V2 高于 V1 也算达标
   > 精度判据统一为「相对退化」口径（本地 V1 与 NV 基线一致），阈值单位为比例（0.05 = 5%）

### 小样本噪声容忍（重要：禁止触发 198 题全量复测）

GPQA 快速评测默认 **50 题**（每题 2%），低分基线上 1~2 题抖动就会把 `rel_drop` 放大到超 5% 红线，造成假阳性判负。因此当 `rel_drop` 超阈值、但**绝对差异 ≤ 2 题**（且题数 ≤ 100 的小样本）时，`accuracy_compare.py` 直接**判定达标**：`aligned=true`、退出码 `0`，并在结果 JSON 中带 `noise_zone=true` / `noise_adjusted=true` / `noise_detail`（供报告注明"小样本噪声容忍达标"）。

- **退出码语义**：`0`=达标（含噪声容忍达标）、`1`=不达标、`2`=参数/文件错误、`3`=缺 NV 基线（编排层兜底）。**不再有退出码 4**。
- **强制规则**：噪声区即视同达标，**严禁**因 `noise_zone=true` 或"想提高置信度"而触发 198 题全量复测（`fast_gpqa.py --limit 0`）或任何形式的重跑。单次快速评测（默认 50 题）即为精度终判。历史事故：噪声区旧退出码 4 被 agent 误解为"需复测/复核"，转而跑 198 题全量，既浪费时间又可能因大样本另一处方差反而推翻本应容忍的结果、造成误判。
- 噪声区结果照常驱动下游：`workflow.accuracy_ok=true` 可写入，V2/V3 对外发布门控按达标处理。

| CLI 参数 | 说明 |
|----------|------|
| `--v1` | V1 (Native) 评测结果 JSON |
| `--v2` | V2 (FlagGems) 评测结果 JSON |
| `--threshold` | 本地 V1 模式：相对退化容差，比例值（默认 0.05 = 5%） |
| `--nv-baseline` | 启用 NV 基线模式：按模型名查 nv_baseline.yaml |
| `--nv-tolerance` | NV 基线模式：相对退化容差（默认表内 default_tolerance 或 0.05） |
| `--json` | JSON 格式输出 |
| `--output` | 结果输出文件路径 |

## 对比输出格式

```
GPQA Diamond 精度对比
=================================
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
4. **每轮算子控制方式**（判据是**当前实际控制方式**而非 env_type——vllm_fl 包存在≠plugin 控制生效。判定：`grep -q '^VLLM_FL_PREFER_ENABLED=true' /etc/environment` 命中=plugin_env 路径，未命中=control_file 路径。plugin_env 路径与性能调优 operator_search.py 走相同的 env_inline 路径）：
   - **plugin_env 路径**：worker 只读环境变量、**不读控制文件**。使用 `diagnose_ops.py` 输出的当轮 `cumulative_test_env.env_inline`（已含 `VLLM_FL_FLAGOS_BLACKLIST` / `VLLM_FL_OOT_BLACKLIST`）作为启动命令内联前缀重启服务：
     ```bash
     # env_inline 形如: USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true VLLM_FL_FLAGOS_BLACKLIST=softmax,layer_norm
     <env_inline> nohup vllm serve ... > /flagos-workspace/logs/startup_flagos.log 2>&1 &
     ```
     > ⚠️ **禁止写 `/root/flaggems_ops_control.json`**：plugin 模式 `VLLM_FL_PREFER_ENABLED=true` 会使注入代码 `pass`，控制文件完全无效，写它会导致每轮禁用不生效、调优空转。
   - **control_file 路径**（含分支 B V1=v1.1/v1.2 场景的 V2：baseline_selector 已清除 `VLLM_FL_PREFER_ENABLED`，flaggems 经注入代码+控制文件生效，即使镜像装有 vllm_fl 包）：使用当轮 `cumulative_test_env.control_file` 白名单写入 `/root/flaggems_ops_control.json`（`{"include": [...]}`），通过 `start_service.sh` 启动（自动推断 `FLAGGEMS_CONTROL_MODE=only_enable`）。
   - **不使用** `toggle_flaggems.py --action modify-enable --disabled-ops`
5. 某轮累积禁用后精度恢复（下降 ≤5%）→ 达标即停，保留所有已累积禁用的算子
6. **每轮输出算子状态**（见下方格式）
7. **每轮必须验证生效**：重启后确认 worker 进程真实生效——plugin 场景 `cat /proc/<vllm-pid>/environ | tr '\0' '\n' | grep VLLM_FL_FLAGOS_BLACKLIST` 应含本轮禁用算子；两种场景均可读 `/tmp/flaggems_enable_oplist.txt`（权威来源）确认禁用算子已不在启用列表

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
- **轮次上限 = `diagnose_ops.py` 分组数**（分几组跑几轮，累积禁用覆盖所有组），绝对上限 8 轮防失控；跑满上限仍不达标才标记 `workflow.accuracy_ok: false` 进入步骤4
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
- 如果跑满轮次上限（=分组数，绝对上限 8 轮）后仍不达标，标记 `workflow.accuracy_ok: false`，进入下一步

---

# 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `evalscope not found` | 未安装 | `pip install 'evalscope==1.5.1'` |
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

主流程步骤4使用**模块 A（本地快速评测）+ 模块 C（精度对比）+ 模块 D（错误处理）**，不使用模块 B（远端 flageval）。数据集由 `--datasets` 参数指定（默认 gpqa_diamond，可多数据集：mmlu/math_500，见模块 A），**每个数据集独立评测、独立判定，全部达标才 `accuracy_ok=true`**。

执行顺序（固定）：
1. 关闭 flaggems → 启动服务 → 各数据集 V1 精度基线（评测命令加 `--dataset {ds}`，输出 `/flagos-workspace/results/{ds}_native.json`）→ 停服务
2. 开启 flaggems → 启动服务 → 各数据集 V2 精度（输出 `/flagos-workspace/results/{ds}_flagos.json`）
   - `--limit`：仅 gpqa_diamond 单数据集时显式传（thinking 30 / 普通 50）；mmlu/math_500 及多数据集**不传**（fast_gpqa 按数据集默认题数：mmlu 100/子集=5700、math_500 全量 500）。V1/V2 参数必须完全相同
3. 每数据集独立对比：`accuracy_compare.py --v1 {ds}_native.json --v2 {ds}_flagos.json --metric {ds} --output accuracy_compare_{ds}.json`（下降阈值 5%；本地 V1 基线模式下 `--metric` 不影响判据，仅报告标题）
4. 结果判定（per-dataset）：
   - 该数据集下降 ≤5%（含V2高于V1）→ 该数据集通过；**全部数据集通过** → `workflow.accuracy_ok=true`，进入步骤6（update_context.py 写入 true 前自动校验全部 `accuracy_compare_{ds}.json` 均 aligned，缺失或任一不达标则拒绝）
   - 任一数据集下降 >5% → 必须按顺序：① 标记 `accuracy_ok=false` ② issue_reporter.py --type accuracy-degraded ③ 追加 `logs/issues_accuracy.log` → 触发步骤5（按不达标数据集逐个调优）
   - 服务崩溃 → issue_reporter.py --type operator-crash（同步骤3）

精度评测+精度调优全部完成后才进入性能测试。

---

## 编排层指令（步骤5 精度算子调优 — 固化决策）

**触发条件**：`workflow.accuracy_ok = false` 且 `env_type ≠ native`（多数据集时任一数据集判定不达标即触发）
**跳过条件**：`accuracy_ok = true`（不触发时显示已完成；多数据集时全部数据集达标才为 true）
**多数据集调优语义**：**按不达标数据集逐个进行**——每轮只针对一个数据集（该数据集 V1/V2 判定 rel_drop >5% 才调优，调优评测用 `--dataset <该数据集>` 与判定同参；已达标数据集不重复评测）。每轮 checkpoint 的 SCORE/BASELINE 均为**当前调优数据集**的单分数（persist_tuning_checkpoint.py 单分数接口不变）。全部数据集达标才 `accuracy_ok=true`（update_context.py 校验兜底）。

### 步骤5入口 — 恢复检测（必须先执行）

会话可能在调优中途超时，重试时必须先检查是否已有达标结果：

```bash
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 -c \"
import yaml
with open('/flagos-workspace/shared/context.yaml') as f:
    ctx = yaml.safe_load(f)
tuning = ctx.get('accuracy_tuning', {})
if tuning.get('completed'):
    print('TUNING_COMPLETED')
    print('final_disabled_ops=' + str(tuning.get('final_disabled_ops', '')))
    print('final_round=' + str(tuning.get('final_round', 0)))
else:
    latest = tuning.get('latest_round', 0)
    if latest:
        print(f'TUNING_IN_PROGRESS round={latest}')
    else:
        print('TUNING_NOT_STARTED')
\""
```

- 输出 `TUNING_COMPLETED` → 调优已达标，应用 `final_disabled_ops` 配置，标记步骤5为 success，直接进入步骤6
- 输出 `TUNING_IN_PROGRESS` → 从 `latest_round + 1` 继续调优（跳过已完成的轮次）
- 输出 `TUNING_NOT_STARTED` → 正常开始调优

### 每轮评测后 — checkpoint 持久化（必须执行）

每轮评测完成后，**立即**调用 checkpoint 工具写入结果，确保会话超时不丢失进度：

```bash
# ROUND: 当前轮次 (从 1 起递增，上限 = 分组数，绝对上限 8)
# SCORE: 本轮评测得分 (如 22.0)
# BASELINE: V1 基线分数 (如 22.0)
# DISABLED_OPS: 本轮累积禁用的算子，逗号分隔 (如 "addmm,mm,bmm")
docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/persist_tuning_checkpoint.py $ROUND $SCORE $BASELINE '$DISABLED_OPS'"
```

该工具会：
- 写入本轮结果到 `context.yaml` 的 `accuracy_tuning` 字段
- 如果本轮达标（下降 ≤5%），自动标记 `accuracy_tuning.completed=true` 和 `workflow.accuracy_ok=true`

### 固化选择

- 使用 `diagnose_ops.py accuracy-groups` 分组排查（不使用逐个排查）
- **轮次上限 = 分组数**（分几组跑几轮，绝对上限 8 轮），达标即停
- **累积禁用**：第1轮禁用组A，第2轮禁用组A+B，第3轮禁用组A+B+C……逐组追加，直至覆盖所有组（每轮在上一轮基础上追加禁用）
- 达标标准：累积禁用后 V2 精度下降 ≤5%（相对 V1）
- **算子控制方式**（按当前实际控制方式区分——`grep -q '^VLLM_FL_PREFER_ENABLED=true' /etc/environment` 命中与否，见模块 C 第4步）：plugin_env 路径用 `cumulative_test_env.env_inline`（`VLLM_FL_FLAGOS_BLACKLIST` 等）内联重启，**禁止写控制文件**（该模式下无效）；control_file 路径才用 `cumulative_test_env.control_file` 白名单写 `/root/flaggems_ops_control.json` + `FLAGGEMS_CONTROL_MODE=only_enable`

执行后必须完成：
- 写入 `context.yaml` 的 `eval.excluded_ops_accuracy` 和 `optimization` 字段
- 写入 `traces/05_accuracy_tuning.json`
- 保存算子列表：`docker exec $CONTAINER cp /tmp/flaggems_enable_oplist.txt /flagos-workspace/results/accuracy_tuned_oplist.txt`
- 更新报告：`docker exec $CONTAINER bash -c "PATH=/opt/conda/bin:$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"`

**注意**：精度调优禁用的算子会传递给后续步骤6，6在此算子集上采集性能基线。
