#!/bin/bash
# start_service.sh — 从 context.yaml 读取配置并启动 vllm/sglang 服务
#
# 供 operator_search.py 的 --service-startup-cmd 调用。
# 在容器内执行，读取 /flagos-workspace/shared/context.yaml 获取启动参数。
#
# 用法:
#   bash /flagos-workspace/scripts/start_service.sh
#   bash /flagos-workspace/scripts/start_service.sh --mode flagos
#   bash /flagos-workspace/scripts/start_service.sh --mode native

set -euo pipefail

CONTEXT_YAML="/flagos-workspace/shared/context.yaml"
MODE=""

# 解析参数（支持 --mode flagos / --mode=flagos / 裸值）
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode=*) MODE="${1#--mode=}"; shift ;;
        --mode)   MODE="${2:-}"; shift; shift 2>/dev/null || true ;;
        *)        shift ;;
    esac
done

# 如果未传 --mode，从环境变量 USE_FLAGGEMS 推断
if [ -z "$MODE" ]; then
    if [ "${USE_FLAGGEMS:-}" = "0" ]; then
        MODE="native"
    elif [ "${USE_FLAGGEMS:-}" = "1" ]; then
        MODE="flagos"
    else
        MODE="flagos"
    fi
    echo "[start_service.sh] --mode 未指定，从环境推断 mode=${MODE}"
fi

# flagos_optimized 也是 FlagGems 启用模式
case "$MODE" in
    native)       USE_FLAGGEMS_FLAG=0 ;;
    flagos|flagos_optimized|flagos_full)  USE_FLAGGEMS_FLAG=1 ;;
    *)            USE_FLAGGEMS_FLAG=1 ;;
esac

# 从 context.yaml 读取启动参数
read_context() {
    PATH=/opt/conda/bin:$PATH python3 -c "
import yaml, json, sys
with open('${CONTEXT_YAML}') as f:
    ctx = yaml.safe_load(f)

model_path = ctx.get('model', {}).get('container_path', '')
model_name = ctx.get('model', {}).get('name', '').split('/')[-1]
port = ctx.get('service', {}).get('port', 8000)
tp_size = ctx.get('runtime', {}).get('tp_size', 0)
gpu_count = ctx.get('runtime', {}).get('gpu_count', ctx.get('gpu', {}).get('count', 0))
max_model_len = ctx.get('service', {}).get('max_model_len', 32768)
framework = ctx.get('runtime', {}).get('framework', 'vllm')
cuda_visible = ctx.get('runtime', {}).get('cuda_visible_devices', '')
visible_devices_env = ctx.get('gpu', {}).get('visible_devices_env', 'CUDA_VISIBLE_DEVICES')
thinking = ctx.get('runtime', {}).get('thinking_model', False)

# TP fallback: 如果为 0，使用 GPU 数量
if tp_size <= 0:
    tp_size = gpu_count if gpu_count > 0 else 1

print(json.dumps({
    'model_path': model_path,
    'model_name': model_name,
    'port': port,
    'tp_size': tp_size,
    'max_model_len': max_model_len,
    'framework': framework,
    'cuda_visible': cuda_visible,
    'visible_devices_env': visible_devices_env,
    'thinking': thinking,
}))
"
}

CONFIG_JSON=$(read_context)

MODEL_PATH=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['model_path'])")
MODEL_NAME=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['model_name'])")
PORT=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['port'])")
TP_SIZE=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tp_size'])")
MAX_MODEL_LEN=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['max_model_len'])")
FRAMEWORK=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['framework'])")
CUDA_VISIBLE=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['cuda_visible'])")
VISIBLE_ENV=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['visible_devices_env'])")
THINKING=$(echo "$CONFIG_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['thinking'])")

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: model.container_path 为空，无法启动服务" >&2
    exit 1
fi

# 强制清理残留进程和编译缓存（每次启动前无条件执行）
pkill -9 -f 'vllm.entrypoints|sglang.launch_server|vllm serve|vllm.serve' 2>/dev/null || true
for _i in $(seq 1 15); do
    if ! ss -tlnp 2>/dev/null | grep -qE ":${PORT}\b"; then break; fi
    sleep 1
done
rm -rf /root/.triton/cache/ /tmp/triton_cache/ /root/.flaggems/code_cache/ 2>/dev/null || true
echo "[start_service.sh] 已清理残留进程和编译缓存"

# 端口占用检测与自动递增（最多尝试 +10）
ORIGINAL_PORT="$PORT"
for i in $(seq 0 10); do
    CANDIDATE_PORT=$((ORIGINAL_PORT + i))
    if ! ss -tlnp 2>/dev/null | grep -qE ":${CANDIDATE_PORT}\b" && \
       ! netstat -tlnp 2>/dev/null | grep -qE ":${CANDIDATE_PORT}\b"; then
        PORT="$CANDIDATE_PORT"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "ERROR: 端口 ${ORIGINAL_PORT}-${CANDIDATE_PORT} 全部被占用" >&2
        exit 1
    fi
done
if [ "$PORT" != "$ORIGINAL_PORT" ]; then
    echo "[start_service.sh] 端口 ${ORIGINAL_PORT} 被占用，自动递增到 ${PORT}"
fi

# 设置 GPU 可见设备（根据厂商使用对应环境变量名）
if [ -n "$CUDA_VISIBLE" ]; then
    if [[ "$VISIBLE_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        export "${VISIBLE_ENV}=${CUDA_VISIBLE}"
    else
        echo "WARNING: VISIBLE_ENV='${VISIBLE_ENV}' 不是合法变量名，使用 CUDA_VISIBLE_DEVICES" >&2
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"
    fi
fi

# 确保 conda 环境在 PATH 中
export PATH=/opt/conda/bin:$PATH

# 加载持久化的 FlagGems 相关环境变量（只提取相关变量，避免覆盖 PATH 等系统变量）
if [ -f /etc/environment ]; then
    while IFS='=' read -r key val; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        case "$key" in
            USE_FLAGGEMS|FLAGGEMS_*|VLLM_FL_*)
                val="${val%\"}" ; val="${val#\"}"
                val="${val%\'}" ; val="${val#\'}"
                export "$key=$val"
                ;;
        esac
    done < /etc/environment
fi

# 从控制文件推断 FLAGGEMS_CONTROL_MODE（兜底：docker exec 不继承宿主进程环境变量）
if [ -z "${FLAGGEMS_CONTROL_MODE:-}" ] && [ -f /root/flaggems_ops_control.json ]; then
    HAS_INCLUDE=$(PATH=/opt/conda/bin:$PATH python3 -c "
import json
try:
    d = json.load(open('/root/flaggems_ops_control.json'))
    print('only_enable' if d.get('include') else 'unused')
except: print('')
" 2>/dev/null)
    if [ -n "$HAS_INCLUDE" ]; then
        export FLAGGEMS_CONTROL_MODE="$HAS_INCLUDE"
        echo "[start_service.sh] FLAGGEMS_CONTROL_MODE=$HAS_INCLUDE (从控制文件推断)"
    fi
fi

# 根据 mode 强制覆盖 USE_FLAGGEMS（确保 native/flagos 模式正确）
export USE_FLAGGEMS="$USE_FLAGGEMS_FLAG"

# native 模式下清除 FlagGems 控制变量，避免残留配置干扰
if [ "$MODE" = "native" ]; then
    unset FLAGGEMS_CONTROL_MODE 2>/dev/null || true
fi

# plugin 场景：显式指定 VLLM_PLUGINS 避免多 platform plugin 冲突（ascend vs fl）
if [ "$USE_FLAGGEMS_FLAG" = "1" ]; then
    HAS_PLUGIN=$(PATH=/opt/conda/bin:$PATH python3 -c "
import importlib.util
print('yes' if importlib.util.find_spec('vllm_fl') else 'no')
" 2>/dev/null || echo "no")
    if [ "$HAS_PLUGIN" = "yes" ]; then
        export VLLM_PLUGINS="fl"
        echo "[start_service.sh] plugin 场景：设置 VLLM_PLUGINS=fl"
    fi
fi

LOG_FILE="/flagos-workspace/logs/startup_${MODE}.log"

# 创建 startup_default.log 符号链接指向当前 mode 的日志
# 崩溃诊断脚本统一引用 startup_default.log，确保路径一致
if [ "$MODE" != "default" ]; then
    ln -sf "startup_${MODE}.log" /flagos-workspace/logs/startup_default.log
fi

# FlagGems 模式启动前清理 Triton/FlagGems 编译缓存（约束39：避免旧缓存隐藏问题算子）
if [ "$USE_FLAGGEMS_FLAG" = "1" ]; then
    rm -rf /root/.triton/cache/ /tmp/triton_cache/ /root/.flaggems/code_cache/ 2>/dev/null || true
fi

# 构建启动命令
if [ "$FRAMEWORK" = "vllm" ]; then
    CMD="vllm serve '${MODEL_PATH}' \
        --host 0.0.0.0 \
        --port ${PORT} \
        --served-model-name '${MODEL_NAME}' \
        --tensor-parallel-size ${TP_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --trust-remote-code"

    # Thinking model 添加 reasoning parser
    if [ "$THINKING" = "true" ]; then
        # 根据模型名推断 parser
        MODEL_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
        if echo "$MODEL_LOWER" | grep -qE 'qwen3|qwq'; then
            CMD="$CMD --reasoning-parser qwen3"
        elif echo "$MODEL_LOWER" | grep -qE 'deepseek'; then
            CMD="$CMD --reasoning-parser deepseek_r1"
        fi
    fi
else
    # sglang
    CMD="python3 -m sglang.launch_server \
        --model-path '${MODEL_PATH}' \
        --host 0.0.0.0 \
        --port ${PORT} \
        --tp ${TP_SIZE} \
        --context-length ${MAX_MODEL_LEN} \
        --trust-remote-code"
fi

echo "[start_service.sh] mode=${MODE}, framework=${FRAMEWORK}, port=${PORT}, tp=${TP_SIZE}"
echo "[start_service.sh] CMD: ${CMD}"

# 后台启动，日志写入文件
nohup bash -c "cd /flagos-workspace && ${CMD}" > "${LOG_FILE}" 2>&1 &
SVC_PID=$!
echo "${SVC_PID}" > /flagos-workspace/logs/service.pid
echo "${LOG_FILE}" > /flagos-workspace/logs/service_log_path
echo "[start_service.sh] PID=${SVC_PID}, log=${LOG_FILE}"

# 保存控制文件副本到 results/（供报告对比配置 vs 运行时算子，仅首次启动时保存）
if [ "$USE_FLAGGEMS_FLAG" = "1" ] && [ -f /root/flaggems_ops_control.json ] && [ ! -f /flagos-workspace/results/ops_control_initial.json ]; then
    cp /root/flaggems_ops_control.json /flagos-workspace/results/ops_control_initial.json 2>/dev/null || true
fi

# 短暂等待后验证进程是否存活（快速发现启动参数错误导致的立即崩溃）
sleep 2
if ! kill -0 "${SVC_PID}" 2>/dev/null; then
    echo "ERROR: 服务进程 ${SVC_PID} 启动后立即退出，请检查日志: ${LOG_FILE}" >&2
    tail -20 "${LOG_FILE}" 2>/dev/null >&2
    exit 1
fi
