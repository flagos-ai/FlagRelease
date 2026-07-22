# 完全旁路的模型进度汇报

FlagRelease 的迁移流程始终拥有模型执行权。汇报组件只接收一次性事件，不启动、包装、等待或控制 `run_pipeline.sh`。

```text
run_batch.sh / run_pipeline.sh
  → nohup progress_runner.sh emit（stdin/stdout/stderr 全断开，主流程不等待）
  → 独立事件队列
  → progress_worker.py
  → Claude 单模型结构化分析
  → 批次状态与飞书卡片
```

即使删除整个 `tools/notifications/`、Python/Claude/curl 不可用、队列不可写或飞书请求失败，迁移任务仍按原来的执行、超时、清理和退出码继续。最坏结果是事件丢失或通知延迟。

## 组件边界

- `progress_runner.sh`：薄入口，只执行事件队列程序；不包含 pipeline、timeout、PID、signal 或退出码转发逻辑。
- `progress_worker.py`：写入/消费事件、调用后台分析、保存 sidecar 结果、更新状态和触发卡片。
- `progress_summary.py`：读取已经校验的 sidecar 结果和批次状态，计算指标并生成卡片。
- `feishu_notify.py`：飞书发送层，负责关键词、卡片格式、超时、重试和 dry-run。
- `tools/batch_summarize/summarize.sh`：保留原有离线长报告模式，并提供 worker 使用的单模型结构化分析模式。

主流程中的钩子固定为：

```bash
nohup "${PROGRESS_RUNNER}" emit "$@" \
    </dev/null \
    >/dev/null \
    2>&1 &
return 0
```

钩子不检查组件是否存在或可执行，不创建汇报目录，不保存后台 PID，也不调用 `wait`。

## 事件与执行位置

批量流程由 `run_batch.sh` 直接执行原生 timeout：

```text
timeout → run_pipeline.sh → 原有上传 → 原有 GPU 清理 → 原退出码
```

捕获退出码后只异步投递：

```text
batch-start
model-finish / skip-model
batch-end
```

批量调用设置 `FLAGOS_BATCH_MODE=1`，仅用于关闭 `run_pipeline.sh` 的重复单模型事件，不参与任何迁移判断。

独立 `run_pipeline.sh` 不再自包装，也没有汇报组件添加的外层 timeout。它在参数和计时初始化完成后投递 `single-start`，并在原有“平台上传 → GPU 服务清理”之后投递 `single-finish`。

## 独立数据目录

默认根目录：

```text
/data/flagos-workspace/.flagrelease-progress/
├── queue/<batch-id>/
├── processing/<batch-id>/
├── processed/<batch-id>/
├── dead-letter/<batch-id>/
├── results/<batch-id>/<task-index>_<model>.json
├── states/batch_<batch-id>_progress.json
├── metadata/<batch-id>/tasks.txt
├── locks/
└── logs/
```

汇报系统不会在模型工作目录的 `config/`、`results/`、`traces/`、`logs/`、`eval/` 或 `reports/` 中写入、删除或移动文件。Claude 只读这些产物，结构化结果写入上述 `results/` sidecar 路径。

事件使用临时文件加原子重命名落盘，文件名带确定性序号和去重键。同批次 worker 使用文件锁串行消费；worker 被终止后，遗留在 `processing/` 的未确认事件会在下次启动时退回队列。失败达到 `FLAGOS_PROGRESS_MAX_ATTEMPTS` 后进入 `dead-letter/`。

## Worker 模式

```text
FLAGOS_PROGRESS_WORKER_MODE=after-batch  默认；开始卡立即处理，模型事件入队，批次结束后再分析
FLAGOS_PROGRESS_WORKER_MODE=live         每个模型结束后立即在后台分析并通知
FLAGOS_PROGRESS_WORKER_MODE=external     本机只写事件，由外部机器或服务消费
```

默认 `after-batch` 下，正常批次在原有详细报告流程结束、主脚本即将退出时才投递 `batch-end`，因此进度分析不会阻塞下一模型，也不会与原有详细报告争用 Claude。异常退出则由 EXIT trap 投递最终事件。

`live` 在控制流上仍完全异步，但可能与下一模型产生少量 CPU、磁盘或网络资源竞争。需要连资源竞争也隔离时，使用默认 `after-batch` 或 `external`。

## Claude 分析边界

worker 以低优先级运行分析，并清空 GPU 可见环境变量：

```text
CUDA_VISIBLE_DEVICES=""
NVIDIA_VISIBLE_DEVICES="void"
nice 19
ionice -c 3（系统支持时）
```

迁移事实由事件强制覆盖：

```text
model / target / vendor / outcome / exit_code / elapsed_seconds
```

Claude 只负责最终交付版本、精度、上传、迁移费用、证据和一句简短说明。分析超时、CLI 不存在、登录失效、非法 JSON 或 schema 校验失败时，worker 保存 `analysis_status=failed`，卡片显示“无法确认”；这些错误不会反馈给迁移进程。

默认分析超时为 900 秒。Claude 分析自身耗时和费用单独记录，不计入模型迁移耗时、迁移费用或成功模型均值。

## 厂商识别

厂商只从 tasks.txt 第一列的目标/镜像名解析，不读取 context，也不探测宿主机 GPU：

| 标识 | 归一化厂商 |
|---|---|
| `nvidia` | `nvidia` |
| `huawei` / `ascend` | `huawei` |
| `hygon` / `dcu` | `hygon` |
| `metax` | `metax` |
| `cambricon` / `mlu` | `cambricon` |
| `mthreads` / `musa` / `mtt` | `mthreads` |
| `kunlunxin` / `kunlun` / `xpu` | `kunlunxin` |
| `tianshu` | `tianshu` |

同一目标匹配多个厂商时为 `unknown`；批次包含不同厂商，或同时包含已识别和未知任务时为 `mixed`。

## 飞书消息

| 事件 | 卡片关键词 |
|---|---|
| 批次/单模型开始 | `任务开始` |
| 批次中模型分析完成 | `结果汇总` |
| 单模型结束 | `任务结束` |
| 批次结束 | `任务结束｜结果汇总` |

所有消息使用 interactive 卡片。批量卡片包含已处理数/总数、模型耗时、当前批次耗时、费用、成功模型均值、达标上传数、成功率和模型明细表；单模型结束卡不展示成功率。

Webhook 只通过环境变量配置：

```bash
export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/REPLACE_ME'
```

## 配置项

```text
FLAGOS_PROGRESS_REPORT=0                    关闭独立单模型事件钩子
FLAGOS_PROGRESS_ROOT=...                    修改独立事件根目录
FLAGOS_PROGRESS_WORKER_MODE=after-batch     worker 模式
FLAGOS_PROGRESS_NOTIFY=0                    处理事件但不发送飞书
FLAGOS_PROGRESS_DRY_RUN=1                   输出卡片 JSON，不发网络请求
FLAGOS_MODEL_ANALYSIS_TIMEOUT_SECONDS=900   单模型分析超时
FLAGOS_PROGRESS_MAX_ATTEMPTS=3              事件最大处理次数
FLAGOS_PROGRESS_END_SETTLE_SECONDS=2         batch-end 前的队列稳定时间
FLAGOS_PROGRESS_END_MAX_WAIT_SECONDS=30      batch-end 最长后台等待时间
FLAGOS_PROGRESS_CLAUDE_CONFIG_DIR=...        独立 Claude 配置目录
FLAGOS_FEISHU_TABLE_MAX_ROWS=20             卡片表格最大行数
```

## 手动操作

```bash
# 只投递事件
tools/notifications/progress_runner.sh emit batch-start \
  --batch-id demo --task-file tasks.txt --total-models 2 \
  --batch-started-at "$(date +%s)"

tools/notifications/progress_runner.sh emit model-finish \
  --batch-id demo --task-index 1 --total-models 2 \
  --target harbor.example/metax/qwen3:latest --model Qwen3-8B \
  --outcome success --exit-code 0 --elapsed-seconds 3600

tools/notifications/progress_runner.sh emit batch-end \
  --batch-id demo --exit-code 0 --elapsed-seconds 7200 --processed 2

# external 模式下手动消费
tools/notifications/progress_runner.sh worker --batch-id demo

# 重试 dead-letter
tools/notifications/progress_runner.sh retry-dead-letter --batch-id demo
```

## 指标口径

- 已处理模型包括 `success`、`failed`、`timeout` 和 `skipped`。
- 模型耗时和当前批次耗时来自原任务捕获的墙钟时间，不使用 Claude 推测值，也不包含后台分析延迟。
- 迁移费用缺失显示“未知”，不能按 `$0` 处理。
- 成功模型均值只统计流程成功且对应耗时或费用可用的模型。
- 最终交付版只允许 V5 或 V3 Max；V4 Express 不计为最终交付。
- 达标上传必须同时满足有效最终版本、精度达标和 Harbor 上传成功。
- 批量成功率为达标上传模型数除以已处理模型数。
