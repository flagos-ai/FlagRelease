#!/bin/bash
# safe_restart_service.sh — 宿主机执行，完整的服务重启流程
# 用法：
#   bash /path/to/safe_restart_service.sh \
#       --container $CTR --mode flagos \
#       [--log-name startup_round1.log] \
#       [--model-name NAME] [--port 8000] \
#       [--env "VAR1=val1 VAR2=val2"]
#
# 流程：
#   1. docker restart $CONTAINER && sleep 5
#   2. docker exec -d ... start_service.sh --mode $MODE > $LOG 2>&1
#   3. docker exec ... wait_for_service.sh --log-path $LOG
#
# 退出码：
#   0 = 服务启动成功
#   1 = wait_for_service 检测到致命错误（日志报错）
#   2 = docker restart 失败
#   3 = 参数错误

set -euo pipefail

CONTAINER=""
MODE="flagos"
LOG_NAME=""
MODEL_NAME=""
PORT=8000
EXTRA_ENV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container) CONTAINER="$2"; shift 2 ;;
        --mode)      MODE="$2"; shift 2 ;;
        --log-name)  LOG_NAME="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --env)       EXTRA_ENV="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 3 ;;
    esac
done

if [ -z "$CONTAINER" ]; then
    echo "错误: --container 必填"
    exit 3
fi

[ -z "$LOG_NAME" ] && LOG_NAME="startup_${MODE}.log"
LOG_PATH="/flagos-workspace/logs/${LOG_NAME}"

echo "═══════════════════════════════════════"
echo "  safe_restart_service.sh"
echo "  容器: ${CONTAINER}"
echo "  模式: ${MODE}"
echo "  日志: ${LOG_PATH}"
echo "═══════════════════════════════════════"

# ① 重启容器
echo "[1/3] docker restart ${CONTAINER}..."
if ! docker restart "$CONTAINER"; then
    echo "✗ docker restart 失败"
    exit 2
fi
sleep 5
echo "  ✓ 容器已重启"

# ② 后台启动服务
echo "[2/3] 启动服务 (mode=${MODE})..."
ENV_PREFIX=""
if [ -n "$EXTRA_ENV" ]; then
    ENV_PREFIX="export ${EXTRA_ENV}; "
fi

docker exec -d "$CONTAINER" bash -c "
    ${ENV_PREFIX}
    cd /flagos-workspace && \
    PATH=/opt/conda/bin:\$PATH \
    bash /flagos-workspace/scripts/start_service.sh --mode ${MODE} \
        > ${LOG_PATH} 2>&1
"
echo "  ✓ 服务启动命令已发出"

# ③ 等待服务就绪
echo "[3/3] 等待服务就绪..."
WAIT_ARGS="--port ${PORT} --max-timeout 1800 --log-path ${LOG_PATH} --mode ${MODE}"
if [ -n "$MODEL_NAME" ]; then
    WAIT_ARGS="${WAIT_ARGS} --model-name '${MODEL_NAME}'"
fi

EXIT_CODE=0
docker exec "$CONTAINER" bash -c "bash /flagos-workspace/scripts/wait_for_service.sh ${WAIT_ARGS}" || EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "═══════════════════════════════════════"
    echo "  ✓ 服务启动成功"
    echo "═══════════════════════════════════════"
else
    echo "═══════════════════════════════════════"
    echo "  ✗ 服务启动失败 (exit ${EXIT_CODE})"
    echo "═══════════════════════════════════════"
fi
exit $EXIT_CODE
