#!/bin/bash
# FlagOS 批量串行迁移 — 逐个调用 run_pipeline.sh
#
# 用法:
#   bash prompts/run_batch.sh <任务列表文件> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--verbose] [--stop-on-error] [--force] [--proxy proxy1,proxy2,...] [--timeout seconds] [--feishu-webhook URL]
#
# 任务列表文件格式（每行一个任务，| 分隔）:
#   # 注释行和空行自动跳过
#   harbor.baai.ac.cn/flagrelease/qwen3:latest | Qwen3-8B
#   my_existing_container | Qwen2.5-7B-Instruct
#
# 选项:
#   --verbose         透传给 run_pipeline.sh，显示全量终端输出
#   --stop-on-error   某个任务失败后终止整个批次（默认继续下一个）
#   --force           强制重跑已完成的任务（默认跳过 workflow.all_done=true 的任务）
#   --feishu-webhook  飞书自定义机器人 Webhook 地址；不传则不发送飞书通知
#
# 双 pipeline 说明:
#   本脚本仅做批量调度，分支路由（A/B/native）由 run_pipeline.sh 内部按环境检测
#   （inspect_env.py 输出的 entry_image_type）自动完成，任务列表无需指定分支。
#     - gems_tree        → 分支 A（简单：V1裸启动→V2代码注入→V3切plugin→V4减算子→V5）
#     - gems_tree_plugin → 分支 B（复杂：V1三选→V2(2.1/2.2)→V3(3.1/3.2)→V4→V5）
#     - native           → native 简化流程（仅评测，不发多版本）
#   汇总表额外展示每个任务实际走的分支与已产出版本（V1-V5，含 V1 变体/不适配标记），
#   数据来自 /data/flagos-workspace/<model>/config/context_snapshot.yaml。
#
# 断点续跑:
#   默认检查 /data/flagos-workspace/<model>/config/context_snapshot.yaml 中的
#   workflow.all_done 字段，已完成的任务自动跳过。中断后重跑同一个任务列表即可续跑。
#   注：续跑粒度为"整任务"——某任务中途失败（如仅 V4 发布失败）重跑时会重头执行，
#   版本级续跑由 run_pipeline.sh 内部的 workflow_ledger 台账负责，本脚本不做版本级调度。

set -uo pipefail

# ========== 参数解析 ==========
if [ $# -lt 6 ]; then
    echo "用法: $0 <任务列表文件> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--verbose] [--stop-on-error] [--force] [--feishu-webhook URL]"
    echo ""
    echo "任务列表文件格式（每行 | 分隔）:"
    echo "  镜像地址或容器名 | 模型名"
    exit 1
fi

TASK_FILE="$1"; shift
MS_TOKEN="$1"; HF_TOKEN="$2"; GH_TOKEN="$3"; HARBOR_USER="$4"; HARBOR_PASS="$5"; shift 5

STOP_ON_ERROR=false
FORCE=false
VERBOSE_FLAG=""
PROXY_FLAG=""
MODEL_TIMEOUT=86400  # 单模型超时（秒），默认 24 小时
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop-on-error) STOP_ON_ERROR=true; shift ;;
        --force) FORCE=true; shift ;;
        --verbose) VERBOSE_FLAG="--verbose"; shift ;;
        --proxy) PROXY_FLAG="--proxy $2"; shift 2 ;;
        --timeout) MODEL_TIMEOUT="$2"; shift 2 ;;
        # export 后 detached worker 与子进程 run_pipeline.sh 均自动继承此变量。
        --feishu-webhook) export FEISHU_WEBHOOK_URL="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ ! -f "$TASK_FILE" ]; then
    echo "错误: 任务列表文件不存在: $TASK_FILE"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROGRESS_RUNNER="${PROJECT_ROOT}/tools/notifications/progress_runner.sh"

# 汇报默认 live 模式：每个模型跑完立即后台分析并通知（控制流仍完全异步、不阻塞主迁移）。
# 允许调用方显式覆盖（after-batch=批次结束统一分析；external=只写事件不消费）。
export FLAGOS_PROGRESS_WORKER_MODE="${FLAGOS_PROGRESS_WORKER_MODE:-live}"

# 汇报是完全旁路的观察者：不检查组件、不等待、不继承 tee 文件描述符，
# 也不保存后台 PID。命令不存在、无权限或自身失败都只会丢失本次事件。
progress_emit_detached() {
    nohup "${PROGRESS_RUNNER}" emit "$@" \
        </dev/null \
        >/dev/null \
        2>&1 &
    return 0
}

# ========== 断点检查 ==========
is_task_done() {
    local model="$1"
    local ctx="/data/flagos-workspace/${model}/config/context_snapshot.yaml"
    [ -f "$ctx" ] && python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    ctx = yaml.safe_load(f)
sys.exit(0 if ctx.get('workflow',{}).get('all_done') is True else 1)
" "$ctx" 2>/dev/null
}

# ========== 分支/版本信息提取（用于汇总列，读 context_snapshot.yaml）===
# ========== 预扫描任务数 ==========
TOTAL=0
while IFS='|' read -r T M || [ -n "$T" ]; do
    T=$(echo "$T" | xargs)
    [[ -z "$T" || "$T" == \#* ]] && continue
    ((TOTAL++))
done < "$TASK_FILE"

if [ "$TOTAL" -eq 0 ]; then
    echo "错误: 任务列表为空（无有效任务行）"
    exit 1
fi

# ========== 全量日志记录 ==========
BATCH_START_TS=$(date +%s)
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="/data/flagos-workspace/batch_${BATCH_TIMESTAMP}.log"
SUMMARY_FILE=""
REPORT_FILE=""
IDX=0; PASS=0; FAIL=0; SKIP=0
RESULTS=()
BATCH_END_EVENT_EMITTED=false
mkdir -p /data/flagos-workspace 2>/dev/null || true
exec > >(tee -a "$BATCH_LOG") 2>&1

batch_on_exit() {
    local exit_code="$1"
    local ended_at elapsed processed
    [ "${BATCH_END_EVENT_EMITTED}" = "true" ] && return 0
    ended_at=$(date +%s)
    elapsed=$(( ended_at - BATCH_START_TS ))
    processed=$(( PASS + FAIL + SKIP ))
    progress_emit_detached batch-end \
        --batch-id "${BATCH_TIMESTAMP}" \
        --workspace /data/flagos-workspace \
        --exit-code "${exit_code}" \
        --elapsed-seconds "${elapsed}" \
        --run-ended-at "${ended_at}" \
        --processed "${processed}" \
        --passed "${PASS}" \
        --failed "${FAIL}" \
        --skipped "${SKIP}" || :
    return 0
}
trap 'batch_on_exit "$?"' EXIT

progress_emit_detached batch-start \
    --batch-id "${BATCH_TIMESTAMP}" \
    --workspace /data/flagos-workspace \
    --task-file "${TASK_FILE}" \
    --total-models "${TOTAL}" \
    --batch-started-at "${BATCH_START_TS}" || :

# ========== Banner ==========
echo "============================================================"
echo "  FlagOS 批量串行迁移"
echo "============================================================"
echo "  任务列表: ${TASK_FILE}"
echo "  任务总数: ${TOTAL}"
echo "  单模型超时: $(( MODEL_TIMEOUT / 3600 ))h$(( (MODEL_TIMEOUT % 3600) / 60 ))m"
echo "  断点续跑: $( $FORCE && echo '关闭 (--force)' || echo '开启' )"
echo "  失败策略: $( $STOP_ON_ERROR && echo '失败即停' || echo '继续下一个' )"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  批次日志: ${BATCH_LOG}"
echo "============================================================"
echo ""

# ========== 逐个执行 ==========
while IFS='|' read -r TARGET MODEL || [ -n "$TARGET" ]; do
    TARGET=$(echo "$TARGET" | xargs)
    MODEL=$(echo "$MODEL" | xargs)
    [[ -z "$TARGET" || "$TARGET" == \#* ]] && continue
    ((IDX++))

    # 归档旧数据（与 run_pipeline.sh 逻辑一致：先归档再判断断点）
    HOST_BASE="/data/flagos-workspace/${MODEL}"
    if [ -d "${HOST_BASE}" ]; then
        TASK_HAS_HISTORY=0
        for d in results traces logs config reports eval; do
            if [ -d "${HOST_BASE}/${d}" ] && [ "$(ls -A "${HOST_BASE}/${d}" 2>/dev/null)" ]; then
                TASK_HAS_HISTORY=1; break
            fi
        done
        if [ "${TASK_HAS_HISTORY}" = "1" ]; then
            ARCHIVE_TS="$(date +%Y%m%d_%H%M%S)"
            TASK_ARCHIVE="${HOST_BASE}/archive/${ARCHIVE_TS}"
            mkdir -p "${TASK_ARCHIVE}"
            for d in results traces logs config reports eval; do
                if [ -d "${HOST_BASE}/${d}" ] && [ "$(ls -A "${HOST_BASE}/${d}" 2>/dev/null)" ]; then
                    mv "${HOST_BASE}/${d}" "${TASK_ARCHIVE}/${d}"
                fi
            done
            echo "  归档旧数据: ${TASK_ARCHIVE}/"
        fi
    fi

    # 断点续跑（归档后 context_snapshot.yaml 已不存在，仅对未归档的中间状态生效）
    if ! $FORCE && is_task_done "$MODEL"; then
        BV=$(task_branch_versions "$MODEL")
        echo "[${IDX}/${TOTAL}] ${MODEL} — 已完成，跳过"
        ((SKIP++))
        RESULTS+=("⊘ | ${MODEL} | ${TARGET} | ${BV%%|*} | ${BV#*|} | - | skipped")
        progress_emit_detached skip-model \
            --batch-id "${BATCH_TIMESTAMP}" \
            --workspace /data/flagos-workspace \
            --model "${MODEL}" \
            --target "${TARGET}" \
            --task-index "${IDX}" \
            --total-models "${TOTAL}" \
            --batch-elapsed-seconds "$(( $(date +%s) - BATCH_START_TS ))" || :
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  [${IDX}/${TOTAL}] ${MODEL}"
    echo "║  目标: ${TARGET}"
    echo "╚══════════════════════════════════════════════════════════════╝"

    TASK_START_TS=$(date +%s)
    progress_emit_detached model-start \
        --batch-id "${BATCH_TIMESTAMP}" \
        --workspace /data/flagos-workspace \
        --task-index "${IDX}" \
        --total-models "${TOTAL}" \
        --target "${TARGET}" \
        --model "${MODEL}" \
        --run-started-at "${TASK_START_TS}" \
        --batch-elapsed-seconds "$(( TASK_START_TS - BATCH_START_TS ))" || :
    FLAGOS_BATCH_MODE=1 \
    timeout --signal=TERM --kill-after=60 "${MODEL_TIMEOUT}" \
        bash prompts/run_pipeline.sh "$TARGET" "$MODEL" "$MS_TOKEN" "$HF_TOKEN" "$GH_TOKEN" "$HARBOR_USER" "$HARBOR_PASS" $VERBOSE_FLAG $PROXY_FLAG < /dev/null
    EXIT_CODE=$?
    TASK_END_TS=$(date +%s)
    TASK_ELAPSED=$(( TASK_END_TS - TASK_START_TS ))
    TASK_MIN=$(( TASK_ELAPSED / 60 ))
    TASK_SEC=$(( TASK_ELAPSED % 60 ))
    ELAPSED_FMT="${TASK_MIN}m${TASK_SEC}s"

    # 提取分支/版本信息（context_snapshot.yaml 由 run_pipeline.sh 兜底同步写入）
    BV=$(task_branch_versions "$MODEL")
    BRANCH="${BV%%|*}"; VERS="${BV#*|}"

    if [ $EXIT_CODE -eq 124 ]; then
        ((FAIL++))
        RESULTS+=("⏱ | ${MODEL} | ${TARGET} | ${BRANCH} | ${VERS} | ${ELAPSED_FMT} | timeout")
        echo "[${IDX}/${TOTAL}] ${MODEL} — ⏱ 超时 (>${MODEL_TIMEOUT}s, ${ELAPSED_FMT})"
        progress_emit_detached model-finish \
            --batch-id "${BATCH_TIMESTAMP}" \
            --workspace /data/flagos-workspace \
            --task-index "${IDX}" \
            --total-models "${TOTAL}" \
            --target "${TARGET}" \
            --model "${MODEL}" \
            --outcome timeout \
            --exit-code "${EXIT_CODE}" \
            --elapsed-seconds "${TASK_ELAPSED}" \
            --batch-elapsed-seconds "$(( TASK_END_TS - BATCH_START_TS ))" \
            --run-started-at "${TASK_START_TS}" \
            --run-ended-at "${TASK_END_TS}" || :
        if $STOP_ON_ERROR; then
            echo ""
            echo "⚠ --stop-on-error 已启用，终止批量执行"
            break
        fi
    elif [ $EXIT_CODE -eq 0 ]; then
        ((PASS++))
        RESULTS+=("✓ | ${MODEL} | ${TARGET} | ${BRANCH} | ${VERS} | ${ELAPSED_FMT} | exit=0")
        echo "[${IDX}/${TOTAL}] ${MODEL} — ✓ 完成 (分支${BRANCH}, ${VERS}, ${ELAPSED_FMT})"
        progress_emit_detached model-finish \
            --batch-id "${BATCH_TIMESTAMP}" \
            --workspace /data/flagos-workspace \
            --task-index "${IDX}" \
            --total-models "${TOTAL}" \
            --target "${TARGET}" \
            --model "${MODEL}" \
            --outcome success \
            --exit-code "${EXIT_CODE}" \
            --elapsed-seconds "${TASK_ELAPSED}" \
            --batch-elapsed-seconds "$(( TASK_END_TS - BATCH_START_TS ))" \
            --run-started-at "${TASK_START_TS}" \
            --run-ended-at "${TASK_END_TS}" || :
    else
        ((FAIL++))
        RESULTS+=("✗ | ${MODEL} | ${TARGET} | ${BRANCH} | ${VERS} | ${ELAPSED_FMT} | exit=${EXIT_CODE}")
        echo "[${IDX}/${TOTAL}] ${MODEL} — ✗ 失败 (exit=${EXIT_CODE}, ${ELAPSED_FMT})"
        progress_emit_detached model-finish \
            --batch-id "${BATCH_TIMESTAMP}" \
            --workspace /data/flagos-workspace \
            --task-index "${IDX}" \
            --total-models "${TOTAL}" \
            --target "${TARGET}" \
            --model "${MODEL}" \
            --outcome failed \
            --exit-code "${EXIT_CODE}" \
            --elapsed-seconds "${TASK_ELAPSED}" \
            --batch-elapsed-seconds "$(( TASK_END_TS - BATCH_START_TS ))" \
            --run-started-at "${TASK_START_TS}" \
            --run-ended-at "${TASK_END_TS}" || :
        if $STOP_ON_ERROR; then
            echo ""
            echo "⚠ --stop-on-error 已启用，终止批量执行"
            break
        fi
    fi
done < "$TASK_FILE"

# ========== 汇总 ==========
BATCH_ELAPSED=$(( $(date +%s) - BATCH_START_TS ))
BATCH_MIN=$(( BATCH_ELAPSED / 60 ))
BATCH_SEC=$(( BATCH_ELAPSED % 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  批量执行汇总                                                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  ✓ 成功: %-3d  ✗ 失败: %-3d  ⊘ 跳过: %-3d  总计: %-3d       ║\n" "$PASS" "$FAIL" "$SKIP" "$TOTAL"
printf "║  总耗时: %sm%ss                                             ║\n" "$BATCH_MIN" "$BATCH_SEC"
echo "╠══════════════════════════════════════════════════════════════╣"
for r in "${RESULTS[@]}"; do
    printf "║  %s\n" "$r"
done
echo "╚══════════════════════════════════════════════════════════════╝"

# 写入汇总文件
SUMMARY_FILE="/data/flagos-workspace/batch_summary_${BATCH_TIMESTAMP}.txt"
{
    echo "FlagOS 批量迁移汇总 — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "任务列表: ${TASK_FILE}"
    echo "✓ 成功: ${PASS}  ✗ 失败: ${FAIL}  ⊘ 跳过: ${SKIP}  总计: ${TOTAL}"
    echo "总耗时: ${BATCH_MIN}m${BATCH_SEC}s"
    echo ""
    printf "%-4s | %-30s | %-50s | %-6s | %-28s | %-10s | %s\n" "状态" "模型" "目标" "分支" "产出版本" "耗时" "退出码"
    echo "---- | ------------------------------ | -------------------------------------------------- | ------ | ---------------------------- | ---------- | ------"
    for r in "${RESULTS[@]}"; do echo "$r"; done
} > "$SUMMARY_FILE" 2>/dev/null && echo "汇总文件: ${SUMMARY_FILE}" || true

# ========== 批量执行退出码 ===
