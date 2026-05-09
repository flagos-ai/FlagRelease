#!/usr/bin/env bash
# wait_for_service.sh — 统一的服务就绪检测（动态超时 + 日志监控）
#
# 核心改进：
#   - 监控启动日志，检测进度信号（权重加载、CUDA graph 编译等）
#   - 检测失败信号（OOM、CUDA error、进程崩溃），立即退出
#   - --timeout 为无活动超时（日志无新输出多久算卡住）
#   - --max-timeout 为绝对上限（安全兜底）
#   - 不传 --log-path 时退化为旧行为（--timeout 作为绝对超时）
#
# Usage:
#   # 动态模式（推荐）
#   ./wait_for_service.sh --port 8000 --model-name "Qwen3-0.6B" \
#       --timeout 120 --max-timeout 1800 \
#       --log-path /flagos-workspace/logs/startup_flagos.log --mode flagos
#
#   # 兼容旧模式（不传 --log-path）
#   ./wait_for_service.sh --port 8000 --timeout 300

set -euo pipefail

# 默认值
PORT=8000
HOST="127.0.0.1"
MODEL_NAME=""
TIMEOUT=120          # 无活动超时（秒），不传 --log-path 时作为绝对超时
MAX_TIMEOUT=1800     # 绝对上限（秒）
LOG_PATH=""
MODE="default"       # default / native / flagos

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --max-timeout) MAX_TIMEOUT="$2"; shift 2 ;;
        --log-path) LOG_PATH="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: wait_for_service.sh [OPTIONS]"
            echo "  --port PORT          服务端口 (默认 8000)"
            echo "  --host HOST          服务地址 (默认 127.0.0.1)"
            echo "  --model-name NAME    模型名称（用于 /v1/models 验证）"
            echo "  --timeout SECS       无活动超时 (默认 120)"
            echo "  --max-timeout SECS   绝对上限 (默认 1800)"
            echo "  --log-path PATH      启动日志路径（启用动态超时模式）"
            echo "  --mode MODE          启动模式: default/native/flagos (默认 default)"
            exit 0 ;;
        *) echo "未知参数: $1"
           echo "可用参数: --port --host --model-name --timeout --max-timeout --log-path --mode --help"
           exit 1 ;;
    esac
done

BASE_URL="http://${HOST}:${PORT}"
MODELS_URL="${BASE_URL}/v1/models"

# 判断是否启用动态模式
DYNAMIC_MODE=false
if [ -n "$LOG_PATH" ]; then
    DYNAMIC_MODE=true
fi

echo "等待服务就绪..."
echo "  地址: ${BASE_URL}"
if [ "$DYNAMIC_MODE" = true ]; then
    echo "  模式: 动态超时（日志监控）"
    echo "  无活动超时: ${TIMEOUT}s"
    echo "  绝对上限: ${MAX_TIMEOUT}s"
    echo "  日志文件: ${LOG_PATH}"
    echo "  启动模式: ${MODE}"
else
    echo "  模式: 固定超时（兼容）"
    echo "  超时: ${TIMEOUT}s"
fi
if [ -n "$MODEL_NAME" ]; then
    echo "  模型: ${MODEL_NAME}"
fi
echo ""

# 轮询参数
INTERVAL=2
MAX_INTERVAL=5
ELAPSED=0

# 动态模式状态
LAST_LOG_SIZE=0
LAST_ACTIVITY_TIME=$(date +%s)
CURRENT_PHASE="initializing"
PHASES_OBSERVED=""
FATAL_SIGNAL=""
FATAL_LINE=""

# 初始化日志跟踪
if [ "$DYNAMIC_MODE" = true ] && [ -f "$LOG_PATH" ]; then
    LAST_LOG_SIZE=$(wc -c < "$LOG_PATH" 2>/dev/null || echo 0)
    LAST_ACTIVITY_TIME=$(date +%s)
fi

# ============================================================
# 阶段标签映射
# ============================================================
phase_label() {
    case "$1" in
        initializing)        echo "初始化中..." ;;
        gpu_initialized)     echo "GPU 已初始化" ;;
        loading_weights)     echo "加载模型权重..." ;;
        weights_loaded)      echo "权重加载完成" ;;
        flaggems_init)       echo "FlagGems 初始化..." ;;
        flaggems_op_register) echo "注册 FlagGems 算子..." ;;
        triton_compile)      echo "编译 Triton 内核..." ;;
        cuda_graph_capture)  echo "CUDA graph 编译中..." ;;
        cuda_graph_done)     echo "CUDA graph 编译完成" ;;
        torch_compile)       echo "torch.compile 编译中..." ;;
        port_bound)          echo "端口已绑定，最终初始化..." ;;
        service_ready)       echo "服务就绪" ;;
        *)                   echo "$1" ;;
    esac
}

# ============================================================
# 日志分析（内嵌 Python）
# ============================================================
analyze_new_lines() {
    # 从 LOG_PATH 读取 LAST_LOG_SIZE 之后的新内容并分析
    # 输出 JSON: {fatal, fatal_line, latest_phase, progress}
    python3 -c "
import sys, re, json

log_path = sys.argv[1]
offset = int(sys.argv[2])

try:
    with open(log_path, 'r', errors='replace') as f:
        f.seek(offset)
        new_content = f.read()
except Exception:
    print(json.dumps({'fatal': '', 'fatal_line': '', 'latest_phase': '', 'progress': False, 'new_size': offset}))
    sys.exit(0)

new_size = offset + len(new_content.encode('utf-8', errors='replace'))
lines = new_content.splitlines()

# 致命信号 — 检测到立即退出
FATAL = [
    (re.compile(r'(?:CUDA\s+)?out\s+of\s+memory|torch\.cuda\.OutOfMemoryError|\bOOM\b', re.I), 'oom'),
    (re.compile(r'CUDA\s*(?:error|Error|ERROR)\s*:|CUDAError|no kernel image is available', re.I), 'cuda_error'),
    (re.compile(r'Segmentation fault|SIGSEGV|SIGKILL', re.I), 'segfault'),
    (re.compile(r'Killed\s+.*(?:vllm|sglang)|killed by signal', re.I), 'killed'),
    (re.compile(r'Address already in use', re.I), 'port_conflict'),
    (re.compile(r'ModuleNotFoundError|ImportError:\s', re.I), 'import_error'),
    (re.compile(r'OSError.*(?:model|tokenizer).*not found|Cannot load model', re.I), 'model_not_found'),
]

# 进度信号 — 证明服务在正常启动
PROGRESS = [
    (re.compile(r'Loading.*(?:model|safetensors|weights)', re.I), 'loading_weights'),
    (re.compile(r'(?:Model\s+)?weights.*(?:loaded|took)', re.I), 'weights_loaded'),
    (re.compile(r'(?:CUDA|GPU)\s+(?:initialized|available|detected)|Number of GPUs', re.I), 'gpu_initialized'),
    (re.compile(r'Capturing.*CUDA\s*graph|cuda\s*graph\s*captur', re.I), 'cuda_graph_capture'),
    (re.compile(r'Graph capturing finished', re.I), 'cuda_graph_done'),
    (re.compile(r'GEMS\s+\w+', re.I), 'flaggems_op_register'),
    (re.compile(r'flag_gems\.enable|import flag_gems', re.I), 'flaggems_init'),
    (re.compile(r'triton|Compiling\s+\w+', re.I), 'triton_compile'),
    (re.compile(r'torch\.compile|inductor', re.I), 'torch_compile'),
    (re.compile(r'Uvicorn running on|Listening on|Serving on', re.I), 'port_bound'),
    (re.compile(r'Application startup complete|Ready to serve', re.I), 'service_ready'),
]

# Traceback 检测
TRACEBACK_RE = re.compile(r'Traceback \(most recent call last\)', re.I)
ERROR_RE = re.compile(r'^\w*(?:Error|Exception):', re.I)

fatal_signal = ''
fatal_line = ''
latest_phase = ''
progress = False
has_traceback = False

for line in lines:
    s = line.strip()
    if not s:
        continue

    # 致命信号
    for pat, label in FATAL:
        if pat.search(s):
            fatal_signal = label
            fatal_line = s[:200]
            break
    if fatal_signal:
        break

    # Traceback + Error 组合
    if TRACEBACK_RE.search(s):
        has_traceback = True
    if has_traceback and ERROR_RE.search(s):
        if not any(w in s for w in ['FutureWarning', 'DeprecationWarning', 'UserWarning']):
            fatal_signal = 'traceback_error'
            fatal_line = s[:200]
            break

    # 进度信号
    for pat, label in PROGRESS:
        if pat.search(s):
            latest_phase = label
            progress = True
            break

print(json.dumps({
    'fatal': fatal_signal,
    'fatal_line': fatal_line,
    'latest_phase': latest_phase,
    'progress': progress,
    'new_size': new_size,
}))
" "$LOG_PATH" "$LAST_LOG_SIZE" 2>/dev/null || echo '{"fatal":"","fatal_line":"","latest_phase":"","progress":false,"new_size":'"$LAST_LOG_SIZE"'}'
}

# ============================================================
# 致命信号标签
# ============================================================
fatal_label() {
    case "$1" in
        oom)            echo "CUDA out of memory" ;;
        cuda_error)     echo "CUDA 错误" ;;
        segfault)       echo "段错误 (Segmentation fault)" ;;
        killed)         echo "进程被杀" ;;
        port_conflict)  echo "端口被占用" ;;
        import_error)   echo "Python 模块缺失" ;;
        model_not_found) echo "模型文件未找到" ;;
        traceback_error) echo "Python 异常" ;;
        *)              echo "$1" ;;
    esac
}

# ============================================================
# 成功报告
# ============================================================
report_success() {
    local model_id="$1"
    local max_model_len="$2"

    echo ""
    echo "=========================================="
    echo "服务就绪！"
    echo "=========================================="
    echo "  耗时: ${ELAPSED}s"
    echo "  模型: ${model_id}"
    echo "  max_model_len: ${max_model_len}"
    echo "  端点: ${BASE_URL}/v1/chat/completions"
    if [ "$DYNAMIC_MODE" = true ] && [ -n "$PHASES_OBSERVED" ]; then
        echo "  经历阶段: ${PHASES_OBSERVED}"
    fi
    echo "=========================================="

    # JSON 输出
    echo ""
    echo "JSON_RESULT:"
    python3 -c "
import json, sys
phases = [p for p in sys.argv[5].split(',') if p] if sys.argv[5] else []
result = {
    'success': True,
    'elapsed_seconds': int(sys.argv[1]),
    'model_id': sys.argv[2],
    'max_model_len': sys.argv[3],
    'endpoint': sys.argv[4],
    'phases_observed': phases,
}
if int(sys.argv[1]) == 0 and sys.argv[6] == 'false':
    result['warning'] = 'instant_response_no_log_confirmation'
print(json.dumps(result, indent=2))
" "${ELAPSED}" "${model_id}" "${max_model_len}" "${BASE_URL}" "${PHASES_OBSERVED}" "${LOG_CONFIRMED_READY}"
}

# ============================================================
# 写入 _last_error.json
# ============================================================
write_error_json() {
    local error_type="$1"
    local error_message="$2"
    local phase="$3"
    local signal="$4"
    local signal_line="$5"

    python3 -c "
import json, os, sys
from datetime import datetime
log_dir = '/flagos-workspace/logs' if os.path.isdir('/flagos-workspace/logs') else '/tmp'
record = {
    'tool': 'wait_for_service.sh',
    'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'exit_code': 1,
    'error_type': sys.argv[1],
    'error_message': sys.argv[2],
    'context': {
        'port': int(sys.argv[3]),
        'host': sys.argv[4],
        'timeout': int(sys.argv[5]),
        'max_timeout': int(sys.argv[6]),
        'elapsed_seconds': int(sys.argv[7]),
        'phase_at_failure': sys.argv[8],
        'failure_signal': sys.argv[9],
        'failure_line': sys.argv[10],
        'mode': sys.argv[11],
    },
}
with open(os.path.join(log_dir, '_last_error.json'), 'w') as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
with open(os.path.join(log_dir, '_error_history.jsonl'), 'a') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
" "$error_type" "$error_message" "$PORT" "$HOST" "$TIMEOUT" "$MAX_TIMEOUT" "$ELAPSED" "$phase" "$signal" "$signal_line" "$MODE" 2>/dev/null || true
}

# ============================================================
# 失败诊断输出
# ============================================================
print_failure_diagnostics() {
    # 检查进程
    echo ""
    echo "进程状态:"
    ps -ef | grep -E "vllm|sglang|flagscale" | grep -v grep || echo "  无相关进程"

    # 检查端口
    echo ""
    echo "端口状态:"
    if command -v ss &>/dev/null; then
        ss -tlnp | grep ":${PORT}" || echo "  端口 ${PORT} 未监听"
    elif command -v netstat &>/dev/null; then
        netstat -tlnp 2>/dev/null | grep ":${PORT}" || echo "  端口 ${PORT} 未监听"
    else
        # ss/netstat 都没有，用 curl 探测
        if curl -s --connect-timeout 1 "http://127.0.0.1:${PORT}/" &>/dev/null; then
            echo "  端口 ${PORT} 有响应"
        else
            echo "  端口 ${PORT} 未监听（ss/netstat 不可用，curl 探测）"
        fi
    fi

    # 输出日志尾部
    if [ -n "$LOG_PATH" ] && [ -f "$LOG_PATH" ]; then
        echo ""
        echo "最后 20 行日志:"
        tail -20 "$LOG_PATH"
    else
        LOG_FILES=$(find /flagos-workspace/logs -name "*.log" -newer /proc/1/cmdline 2>/dev/null | head -3)
        if [ -n "$LOG_FILES" ]; then
            echo ""
            echo "自动发现的日志文件:"
            for lf in $LOG_FILES; do
                echo "--- $lf (最后 10 行) ---"
                tail -10 "$lf"
            done
        fi
    fi
}

# ============================================================
# 主循环
# ============================================================

# 确定实际超时上限
if [ "$DYNAMIC_MODE" = true ]; then
    EFFECTIVE_MAX=$MAX_TIMEOUT
else
    # 兼容模式：--timeout 作为绝对超时
    EFFECTIVE_MAX=$TIMEOUT
fi

# 标记日志中是否观测到 service_ready 信号
LOG_CONFIRMED_READY=false

while [ "$ELAPSED" -lt "$EFFECTIVE_MAX" ]; do

    # === CHECK 1: 日志监控（优先于端口检测，确保致命信号不被跳过） ===
    if [ "$DYNAMIC_MODE" = true ] && [ -f "$LOG_PATH" ]; then
        CURRENT_LOG_SIZE=$(wc -c < "$LOG_PATH" 2>/dev/null || echo 0)

        if [ "$CURRENT_LOG_SIZE" -gt "$LAST_LOG_SIZE" ]; then
            # 日志有增长 — 分析新内容
            ANALYSIS=$(analyze_new_lines)

            # 解析分析结果
            FATAL_SIGNAL=$(echo "$ANALYSIS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('fatal',''))" 2>/dev/null || echo "")
            FATAL_LINE=$(echo "$ANALYSIS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('fatal_line',''))" 2>/dev/null || echo "")
            LATEST_PHASE=$(echo "$ANALYSIS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('latest_phase',''))" 2>/dev/null || echo "")
            PROGRESS=$(echo "$ANALYSIS" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if d.get('progress') else 'no')" 2>/dev/null || echo "no")
            NEW_SIZE=$(echo "$ANALYSIS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_size',0))" 2>/dev/null || echo "$CURRENT_LOG_SIZE")

            # 2a. 致命信号 → 立即退出
            if [ -n "$FATAL_SIGNAL" ]; then
                LABEL=$(fatal_label "$FATAL_SIGNAL")
                echo ""
                echo "=========================================="
                echo "✗ 检测到致命错误: ${LABEL}"
                echo "=========================================="
                echo "  耗时: ${ELAPSED}s"
                echo "  阶段: $(phase_label "$CURRENT_PHASE")"
                echo "  信号: ${FATAL_SIGNAL}"
                echo "  详情: ${FATAL_LINE}"
                echo "=========================================="

                write_error_json "$FATAL_SIGNAL" "${LABEL}: ${FATAL_LINE}" "$CURRENT_PHASE" "$FATAL_SIGNAL" "$FATAL_LINE"
                print_failure_diagnostics

                echo ""
                echo "JSON_RESULT:"
                python3 -c "
import json, sys
phases = [p for p in sys.argv[5].split(',') if p] if sys.argv[5] else []
print(json.dumps({
    'success': False,
    'elapsed_seconds': int(sys.argv[1]),
    'error': sys.argv[2],
    'error_detail': sys.argv[3],
    'phase_at_failure': sys.argv[4],
    'endpoint': sys.argv[6],
    'phases_observed': phases,
}, indent=2))
" "${ELAPSED}" "${FATAL_SIGNAL}" "${FATAL_LINE}" "${CURRENT_PHASE}" "${PHASES_OBSERVED}" "${BASE_URL}"
                exit 1
            fi

            # 2b. 进度信号 → 重置无活动计时器
            if [ "$PROGRESS" = "yes" ]; then
                LAST_ACTIVITY_TIME=$(date +%s)
                if [ -n "$LATEST_PHASE" ]; then
                    CURRENT_PHASE="$LATEST_PHASE"
                    # 记录经历的阶段（去重）
                    if ! echo ",$PHASES_OBSERVED," | grep -q ",${LATEST_PHASE},"; then
                        if [ -n "$PHASES_OBSERVED" ]; then
                            PHASES_OBSERVED="${PHASES_OBSERVED},${LATEST_PHASE}"
                        else
                            PHASES_OBSERVED="${LATEST_PHASE}"
                        fi
                    fi
                fi
            fi

            # 日志增长本身也算活动（即使没匹配到已知阶段）
            LAST_ACTIVITY_TIME=$(date +%s)
            LAST_LOG_SIZE=$NEW_SIZE

            # 标记日志中是否观测到 service_ready
            if echo ",$PHASES_OBSERVED," | grep -q ",service_ready,"; then
                LOG_CONFIRMED_READY=true
            fi
        fi
    fi

    # === CHECK 2: 端点检查 ===
    RESPONSE=$(curl -s --connect-timeout 3 "${MODELS_URL}" 2>/dev/null || true)

    if [ -n "$RESPONSE" ]; then
        HAS_DATA=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('data', [])
    if models:
        for m in models:
            print(m.get('id', ''))
except:
    pass
" 2>/dev/null || true)

        if [ -n "$HAS_DATA" ]; then
            MODEL_ID=$(echo "$HAS_DATA" | head -1)

            if [ -n "$MODEL_NAME" ] && ! echo "$HAS_DATA" | grep -qi "$MODEL_NAME"; then
                echo "[${ELAPSED}s] 服务已响应，但模型名不匹配: 期望=${MODEL_NAME}, 实际=${MODEL_ID}"
            fi

            # 残留服务检测：日志未确认 service_ready 时，端口响应视为可疑
            if [ "$LOG_CONFIRMED_READY" = false ] && [ "$DYNAMIC_MODE" = true ]; then
                echo "[${ELAPSED}s] ⚠ 端口 ${PORT} 已响应但启动日志未确认 service_ready，疑似残留服务"
                echo "  响应模型: ${MODEL_ID}"
                echo "  等待日志确认或致命信号..."
                sleep "$INTERVAL"
                ELAPSED=$((ELAPSED + INTERVAL))
                INTERVAL=$((INTERVAL * 2))
                if [ "$INTERVAL" -gt "$MAX_INTERVAL" ]; then INTERVAL=$MAX_INTERVAL; fi
                continue
            fi

            # 非动态模式下的瞬时响应警告：端口在首次轮询即响应，可能是残留服务
            if [ "$DYNAMIC_MODE" = false ] && [ "$ELAPSED" -eq 0 ]; then
                echo ""
                echo "⚠ 警告: 端口 ${PORT} 在 0s 内响应，可能是残留服务而非新启动的服务"
                echo "  响应模型: ${MODEL_ID}"
                echo "  建议使用 --log-path 启用动态模式以获得可靠检测"
            fi

            MAX_MODEL_LEN=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('data', []):
        print(m.get('max_model_len', 'unknown'))
        break
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

            report_success "$MODEL_ID" "$MAX_MODEL_LEN"
            exit 0
        fi
    fi

    # === CHECK 3: 进程存活检测（仅动态模式，启动 10s 后） ===
    if [ "$DYNAMIC_MODE" = true ] && [ "$ELAPSED" -gt 10 ]; then
        PROCESS_COUNT=$(ps -ef | grep -E "vllm|sglang|flagscale" | grep -v grep | wc -l)
        if [ "$PROCESS_COUNT" -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "✗ 服务进程已退出"
            echo "=========================================="
            echo "  耗时: ${ELAPSED}s"
            echo "  阶段: $(phase_label "$CURRENT_PHASE")"
            echo "=========================================="

            write_error_json "process_exited" "服务进程已退出，最后阶段: $(phase_label "$CURRENT_PHASE")" "$CURRENT_PHASE" "process_exited" ""
            print_failure_diagnostics

            echo ""
            echo "JSON_RESULT:"
            python3 -c "
import json, sys
phases = [p for p in sys.argv[4].split(',') if p] if sys.argv[4] else []
print(json.dumps({
    'success': False,
    'elapsed_seconds': int(sys.argv[1]),
    'error': 'process_exited',
    'phase_at_failure': sys.argv[2],
    'endpoint': sys.argv[3],
    'phases_observed': phases,
}, indent=2))
" "${ELAPSED}" "${CURRENT_PHASE}" "${BASE_URL}" "${PHASES_OBSERVED}"
            exit 1
        fi
    fi

    # === CHECK 4: 无活动超时（仅动态模式） ===
    if [ "$DYNAMIC_MODE" = true ]; then
        NOW=$(date +%s)
        SINCE_ACTIVITY=$((NOW - LAST_ACTIVITY_TIME))

        # 已知长编译阶段（torch.compile / CUDA graph）允许更长的无活动时间
        # 这些阶段编译期间不输出日志是正常行为，只要进程还活着就继续等
        EFFECTIVE_STALL_TIMEOUT=$TIMEOUT
        case "$CURRENT_PHASE" in
            torch_compile|cuda_graph_capture|triton_compile)
                # 编译阶段：无活动超时至少 300s，取 max(TIMEOUT, 300)
                if [ "$EFFECTIVE_STALL_TIMEOUT" -lt 300 ]; then
                    EFFECTIVE_STALL_TIMEOUT=300
                fi
                ;;
        esac

        if [ "$SINCE_ACTIVITY" -gt "$EFFECTIVE_STALL_TIMEOUT" ]; then
            echo ""
            echo "=========================================="
            echo "✗ 服务启动停滞（${SINCE_ACTIVITY}s 无日志活动）"
            echo "=========================================="
            echo "  总耗时: ${ELAPSED}s"
            echo "  无活动超时: ${TIMEOUT}s"
            echo "  最后阶段: $(phase_label "$CURRENT_PHASE")"
            echo "=========================================="

            write_error_json "service_stall" "服务启动停滞 (${SINCE_ACTIVITY}s 无日志活动), 最后阶段: $(phase_label "$CURRENT_PHASE")" "$CURRENT_PHASE" "stall" ""
            print_failure_diagnostics

            echo ""
            echo "JSON_RESULT:"
            python3 -c "
import json, sys
phases = [p for p in sys.argv[5].split(',') if p] if sys.argv[5] else []
print(json.dumps({
    'success': False,
    'elapsed_seconds': int(sys.argv[1]),
    'error': 'stall',
    'stall_seconds': int(sys.argv[2]),
    'phase_at_failure': sys.argv[3],
    'endpoint': sys.argv[4],
    'phases_observed': phases,
}, indent=2))
" "${ELAPSED}" "${SINCE_ACTIVITY}" "${CURRENT_PHASE}" "${BASE_URL}" "${PHASES_OBSERVED}"
            exit 1
        fi
    fi

    # === 进度输出 ===
    if [ "$DYNAMIC_MODE" = true ]; then
        PHASE_TEXT=$(phase_label "$CURRENT_PHASE")
        NOW=$(date +%s)
        SINCE_ACTIVITY=$((NOW - LAST_ACTIVITY_TIME))
        if [ "$SINCE_ACTIVITY" -gt 30 ]; then
            echo "[${ELAPSED}s] 阶段: ${PHASE_TEXT} (${SINCE_ACTIVITY}s 无新日志)"
        else
            echo "[${ELAPSED}s] 阶段: ${PHASE_TEXT}"
        fi
    else
        echo "[${ELAPSED}s] 服务未就绪，${INTERVAL}s 后重试..."
    fi

    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))

    # 指数退避
    INTERVAL=$((INTERVAL * 2))
    if [ "$INTERVAL" -gt "$MAX_INTERVAL" ]; then
        INTERVAL=$MAX_INTERVAL
    fi
done

# ============================================================
# 绝对超时
# ============================================================
echo ""
echo "=========================================="
if [ "$DYNAMIC_MODE" = true ]; then
    echo "ERROR: 服务启动超时（绝对上限 ${MAX_TIMEOUT}s）"
else
    echo "ERROR: 服务启动超时（${TIMEOUT}s）"
fi
echo "=========================================="
if [ "$DYNAMIC_MODE" = true ]; then
    echo "  最后阶段: $(phase_label "$CURRENT_PHASE")"
fi

write_error_json "service_timeout" "服务启动超时 (${ELAPSED}s), 端口 ${PORT} 无响应" "$CURRENT_PHASE" "timeout" ""
print_failure_diagnostics

# 输出失败 JSON
echo ""
echo "JSON_RESULT:"
python3 -c "
import json, sys
phases = [p for p in sys.argv[4].split(',') if p] if sys.argv[4] else []
print(json.dumps({
    'success': False,
    'elapsed_seconds': int(sys.argv[1]),
    'error': 'timeout',
    'phase_at_failure': sys.argv[2],
    'endpoint': sys.argv[3],
    'phases_observed': phases,
}, indent=2))
" "${ELAPSED}" "${CURRENT_PHASE}" "${BASE_URL}" "${PHASES_OBSERVED}"

exit 1
