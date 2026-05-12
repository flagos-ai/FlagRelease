#!/usr/bin/env bash
# setup_workspace.sh — 一次性工作区初始化
#
# 在容器准备阶段一次性完成：创建目录、复制脚本、安装依赖。
# 替代每个阶段各自 docker cp 的重复操作。
#
# Usage:
#   bash skills/flagos-container-preparation/tools/setup_workspace.sh <container_name>
#   bash skills/flagos-container-preparation/tools/setup_workspace.sh RoboBrain2.0-7B_flagos Qwen3-8B

set -euo pipefail

SKIP_ARCHIVE=false
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --skip-archive) SKIP_ARCHIVE=true ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
CONTAINER="${POSITIONAL[0]:?用法: $0 <container_name> [model_path] [--skip-archive]}"
MODEL_NAME="${POSITIONAL[1]:-}"

# 项目根目录（此脚本所在位置的上三级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "FlagOS 工作区初始化"
echo "=========================================="
echo "  容器: ${CONTAINER}"
echo "  项目: ${PROJECT_ROOT}"
[ -n "${MODEL_NAME}" ] && echo "  模型: ${MODEL_NAME}"
echo ""

# 0. 检测 /flagos-workspace 挂载状态
WORKSPACE="/flagos-workspace"
HOST_WORKSPACE=""
MOUNT_MODE="unknown"
MOUNT_INFO=$(docker inspect --format '{{json .Mounts}}' "${CONTAINER}" 2>/dev/null || echo "[]")

echo "[0/6] 检测工作目录挂载状态..."

# 检查是否已有 /flagos-workspace 挂载
HAS_WORKSPACE_MOUNT=$(echo "${MOUNT_INFO}" | python3 -c "
import json, sys
mounts = json.load(sys.stdin)
for m in mounts:
    if m.get('Destination','').rstrip('/') == '/flagos-workspace':
        print(m['Source']); break
" 2>/dev/null || true)

if [ -n "${HAS_WORKSPACE_MOUNT}" ]; then
    echo "  已检测到 /flagos-workspace 挂载: ${HAS_WORKSPACE_MOUNT}"
    HOST_WORKSPACE="${HAS_WORKSPACE_MOUNT}"
    MOUNT_MODE="mounted"
else
    # 从已有挂载中选择：优先 /data，其次第一个 rw bind mount
    SELECTED_MOUNT=$(echo "${MOUNT_INFO}" | python3 -c "
import json, sys
mounts = json.load(sys.stdin)
binds = [m for m in mounts if m.get('Type') == 'bind' and m.get('RW', True)]
for m in binds:
    if m['Destination'].rstrip('/') == '/data':
        print(m['Destination']); sys.exit()
if binds:
    print(binds[0]['Destination'])
" 2>/dev/null || true)

    if [ -z "${SELECTED_MOUNT}" ]; then
        echo "  警告: 未检测到可用挂载点，将在容器内创建非持久化工作目录"
        docker exec "${CONTAINER}" mkdir -p "${WORKSPACE}"
        MOUNT_MODE="internal"
    else
        # 获取宿主机对应路径
        HOST_WORKSPACE=$(echo "${MOUNT_INFO}" | python3 -c "
import json, sys
target = '${SELECTED_MOUNT}'.rstrip('/')
mounts = json.load(sys.stdin)
for m in mounts:
    if m.get('Destination','').rstrip('/') == target:
        print(m['Source'] + '/flagos-workspace'); break
" 2>/dev/null || true)

        echo "  在已有挂载点 ${SELECTED_MOUNT} 下创建工作目录..."
        docker exec "${CONTAINER}" mkdir -p "${SELECTED_MOUNT}/flagos-workspace"
        docker exec "${CONTAINER}" ln -sfn "${SELECTED_MOUNT}/flagos-workspace" "${WORKSPACE}"
        echo "  软链接: ${WORKSPACE} → ${SELECTED_MOUNT}/flagos-workspace"
        echo "  宿主机路径: ${HOST_WORKSPACE}"
        MOUNT_MODE="symlink"
    fi
fi

# 1. 归档上一轮数据（容器内）
if $SKIP_ARCHIVE; then
    echo "[1/6] 跳过归档（--skip-archive）"
else
echo "[1/6] 检查并归档历史数据..."
HAS_HISTORY=$(docker exec "${CONTAINER}" bash -c "
    found=0
    for d in /flagos-workspace/results /flagos-workspace/traces /flagos-workspace/logs /flagos-workspace/config /flagos-workspace/output /flagos-workspace/reports /flagos-workspace/eval; do
        if [ -d \"\$d\" ] && [ \"\$(ls -A \"\$d\" 2>/dev/null)\" ]; then
            found=1; break
        fi
    done
    echo \$found
" 2>/dev/null || echo "0")

if [ "${HAS_HISTORY}" = "1" ]; then
    ARCHIVE_TS=$(date +%Y%m%d_%H%M%S)
    echo "  发现历史数据，归档到 archive/${ARCHIVE_TS}/ ..."
    docker exec "${CONTAINER}" bash -c "
        ARCHIVE_DIR=/flagos-workspace/archive/${ARCHIVE_TS}
        mkdir -p \"\${ARCHIVE_DIR}\"
        for d in results traces logs config output reports eval; do
            if [ -d /flagos-workspace/\$d ] && [ \"\$(ls -A /flagos-workspace/\$d 2>/dev/null)\" ]; then
                mv /flagos-workspace/\$d \"\${ARCHIVE_DIR}/\$d\"
            fi
        done
        if [ -f /flagos-workspace/shared/context.yaml ]; then
            cp /flagos-workspace/shared/context.yaml \"\${ARCHIVE_DIR}/context.yaml\"
        fi
    "
    echo "  容器内归档完成: /flagos-workspace/archive/${ARCHIVE_TS}/"

    # 归档后重置：清理残留状态，避免历史数据污染新一轮运行
    echo "  清理残留状态..."
    docker exec "${CONTAINER}" bash -c "
        # 清理残留的算子列表文件
        rm -f /tmp/flaggems_enable_oplist.txt
        rm -f /root/gems.txt

        # 清理残留的评测结果文件（避免被误 cp 为新结果）
        rm -f /workspace/gpqa_result.json
        rm -f /flagos-workspace/scripts/gpqa_result.json

        # 停止可能残留的 vllm/sglang 服务进程
        pkill -f 'vllm.entrypoints' 2>/dev/null || true
        pkill -f 'sglang' 2>/dev/null || true
    "
    # 重置 context.yaml：从项目模板复制，确保与模板字段同步
    docker cp "${PROJECT_ROOT}/shared/context.template.yaml" "${CONTAINER}:/flagos-workspace/shared/context.yaml"
    echo "  ✓ context.yaml 已重置（从模板复制）"
    echo "  ✓ 残留算子列表已清理"
    echo "  ✓ 残留服务进程已清理"
else
    echo "  无历史数据，跳过归档"
fi
fi  # end SKIP_ARCHIVE

# 1.5. 创建宿主机工作目录
if [ -n "${MODEL_NAME}" ]; then
    echo "[1.5/6] 创建宿主机工作目录..."

    HOST_BASE="/data/flagos-workspace/${MODEL_NAME}"

    if ! $SKIP_ARCHIVE; then
    # 归档宿主机历史数据
    if [ -d "${HOST_BASE}" ]; then
        HOST_HAS_HISTORY=0
        for d in results traces logs config; do
            if [ -d "${HOST_BASE}/${d}" ] && [ "$(ls -A "${HOST_BASE}/${d}" 2>/dev/null)" ]; then
                HOST_HAS_HISTORY=1; break
            fi
        done
        if [ "${HOST_HAS_HISTORY}" = "1" ]; then
            HOST_ARCHIVE_TS=${ARCHIVE_TS:-$(date +%Y%m%d_%H%M%S)}
            HOST_ARCHIVE="${HOST_BASE}/archive/${HOST_ARCHIVE_TS}"
            mkdir -p "${HOST_ARCHIVE}"
            for d in results traces config reports eval; do
                if [ -d "${HOST_BASE}/${d}" ] && [ "$(ls -A "${HOST_BASE}/${d}" 2>/dev/null)" ]; then
                    mv "${HOST_BASE}/${d}" "${HOST_ARCHIVE}/${d}"
                fi
            done
            if [ -d "${HOST_BASE}/logs" ] && [ "$(ls -A "${HOST_BASE}/logs" 2>/dev/null)" ]; then
                mkdir -p "${HOST_ARCHIVE}/logs"
                find "${HOST_BASE}/logs" -maxdepth 1 -type f ! -name "*.log" -newer "${HOST_BASE}/logs" -prune -o -type f -print 2>/dev/null | while read -r f; do
                    if ! fuser "$f" >/dev/null 2>&1; then
                        mv "$f" "${HOST_ARCHIVE}/logs/" 2>/dev/null || true
                    fi
                done
            fi
            echo "  宿主机历史数据归档到: ${HOST_ARCHIVE}/"
        fi
    fi
    fi  # end SKIP_ARCHIVE for host

    mkdir -p "${HOST_BASE}/results"
    mkdir -p "${HOST_BASE}/traces"
    mkdir -p "${HOST_BASE}/logs"
    mkdir -p "${HOST_BASE}/config"
    echo "  宿主机目录创建完成: ${HOST_BASE}"
fi

# 2. 创建容器内目录结构
echo "[2/6] 创建目录结构..."
docker exec "${CONTAINER}" bash -c "
    mkdir -p /flagos-workspace/{scripts,logs,results,reports,eval,perf/config,shared,output,traces,config}
"
echo "  目录创建完成"

# 3. 复制所有脚本到容器
echo "[3/6] 复制脚本到容器..."

SCRIPTS_COPIED=0

# 脚本清单：源路径（相对 PROJECT_ROOT）→ 容器目标路径
# 格式：source_relative_path:container_dest_path
SCRIPT_MAP=(
    # 环境检查
    "skills/flagos-pre-service-inspection/tools/inspect_env.py:scripts/inspect_env.py"
    # FlagGems 开关切换
    "skills/flagos-service-startup/tools/toggle_flaggems.py:scripts/toggle_flaggems.py"
    # 服务就绪检测
    "skills/flagos-service-startup/tools/wait_for_service.sh:scripts/wait_for_service.sh"
    # 服务活性监控（评测/性能测试期间）
    "skills/flagos-service-startup/tools/service_monitor.py:scripts/service_monitor.py"
    # 服务启动（供 operator_search.py 调用）
    "skills/flagos-service-startup/tools/start_service.sh:scripts/start_service.sh"
    # TP 推算
    "skills/flagos-service-startup/tools/calc_tp_size.py:scripts/calc_tp_size.py"
    # 性能测试
    "skills/flagos-performance-testing/tools/benchmark_runner.py:scripts/benchmark_runner.py"
    # 性能对比
    "skills/flagos-performance-testing/tools/performance_compare.py:scripts/performance_compare.py"
    # 算子优化
    "skills/flagos-operator-replacement/tools/operator_optimizer.py:scripts/operator_optimizer.py"
    # 算子搜索编排
    "skills/flagos-operator-replacement/tools/operator_search.py:scripts/operator_search.py"
    # 算子配置生成（Plugin 场景）
    "skills/flagos-operator-replacement/tools/apply_op_config.py:scripts/apply_op_config.py"
    # 算子快速诊断
    "skills/flagos-operator-replacement/tools/diagnose_ops.py:scripts/diagnose_ops.py"
    # 算子配置固化
    "skills/flagos-operator-replacement/tools/persist_op_config.py:scripts/persist_op_config.py"
    # 组件安装（统一入口）
    "skills/flagos-component-install/tools/install_component.py:scripts/install_component.py"
    # FlagTree 安装脚本
    "skills/flagos-component-install/tools/install_flagtree.sh:scripts/install_flagtree.sh"
    # GPQA Diamond 快速精度评测（eval/ 兼容 SKILL.md 引用，scripts/ 供编排层统一调用）
    "skills/flagos-eval-comprehensive/tools/fast_gpqa.py:eval/fast_gpqa.py"
    "skills/flagos-eval-comprehensive/tools/fast_gpqa_config.yaml:eval/fast_gpqa_config.yaml"
    # 精度对比
    "skills/flagos-eval-comprehensive/tools/accuracy_compare.py:scripts/accuracy_compare.py"
    "skills/flagos-eval-comprehensive/tools/fast_gpqa.py:scripts/fast_gpqa.py"
    "skills/flagos-eval-comprehensive/tools/fast_gpqa_config.yaml:scripts/fast_gpqa_config.yaml"
    # 远端评测监控
    "skills/flagos-eval-comprehensive/tools/eval_monitor.py:scripts/eval_monitor.py"
    # 评测配置模板
    "skills/flagos-eval-comprehensive/tools/config.yaml:eval/config.yaml"
    # Plugin 安装
    "skills/flagos-plugin-install/tools/install_plugin.py:scripts/install_plugin.py"
    # 问题自动提交
    "skills/flagos-issue-reporter/tools/issue_reporter.py:scripts/issue_reporter.py"
    # 日志分析
    "skills/flagos-log-analyzer/tools/log_analyzer.py:scripts/log_analyzer.py"
    # 共享模块
    "skills/flagos-operator-replacement/tools/ops_constants.py:scripts/ops_constants.py"
    # context.yaml 结构化更新工具
    "shared/update_context.py:scripts/update_context.py"
    # GPU 统一检测
    "shared/detect_gpu.py:scripts/detect_gpu.py"
    # 错误/检查点持久化
    "shared/error_writer.py:scripts/error_writer.py"
    # 故障诊断工具
    "skills/flagos-log-analyzer/tools/diagnose_failure.py:scripts/diagnose_failure.py"
    # 报告生成工具
    "shared/generate_report.py:scripts/generate_report.py"
)

for entry in "${SCRIPT_MAP[@]}"; do
    src="${PROJECT_ROOT}/${entry%%:*}"
    dest="/flagos-workspace/${entry##*:}"
    if [ -f "$src" ]; then
        docker cp "$src" "${CONTAINER}:${dest}"
        SCRIPTS_COPIED=$((SCRIPTS_COPIED + 1))
        echo "  ✓ ${entry##*:}"
    fi
done

# .sh 文件需要 +x 权限
docker exec "${CONTAINER}" bash -c "chmod +x /flagos-workspace/scripts/*.sh 2>/dev/null || true"

# 评测脚本（eval_*.py 批量复制）
for eval_script in "${PROJECT_ROOT}"/skills/flagos-eval-comprehensive/tools/eval_*.py; do
    if [ -f "$eval_script" ]; then
        docker cp "$eval_script" "${CONTAINER}:/flagos-workspace/scripts/"
        echo "  ✓ $(basename "$eval_script")"
        SCRIPTS_COPIED=$((SCRIPTS_COPIED + 1))
    fi
done

# 性能测试配置 → scripts/config/（benchmark_runner.py 默认路径）
if [ -d "${PROJECT_ROOT}/skills/flagos-performance-testing/config" ]; then
    docker exec "${CONTAINER}" mkdir -p /flagos-workspace/scripts/config
    docker cp "${PROJECT_ROOT}/skills/flagos-performance-testing/config/." \
        "${CONTAINER}:/flagos-workspace/scripts/config/"
    echo "  ✓ scripts/config/ (perf_config)"
fi

echo "  共复制 ${SCRIPTS_COPIED} 个脚本"

# 3.5. 从模板初始化容器内 context.yaml（每个容器独立，避免多任务冲突）
# 每次都从模板重新初始化，确保干净的初始状态
TEMPLATE_FILE="${PROJECT_ROOT}/shared/context.template.yaml"
if [ -f "${TEMPLATE_FILE}" ]; then
    docker cp "${TEMPLATE_FILE}" "${CONTAINER}:/flagos-workspace/shared/context.yaml"
    echo "  ✓ shared/context.yaml (从 context.template.yaml 初始化)"
else
    # 兼容旧版：模板文件不存在时尝试旧路径
    if [ -f "${PROJECT_ROOT}/shared/context.yaml" ]; then
        docker cp "${PROJECT_ROOT}/shared/context.yaml" "${CONTAINER}:/flagos-workspace/shared/context.yaml"
        echo "  ⚠ shared/context.yaml (从旧 context.yaml 复制，请迁移到 context.template.yaml)"
    else
        docker exec "${CONTAINER}" bash -c "echo '# FlagOS context' > /flagos-workspace/shared/context.yaml"
        echo "  ⚠ shared/context.yaml (空文件，未找到模板)"
    fi
fi

# 4. 安装脚本依赖（如需要）
echo "[4/6] 检查脚本依赖..."
docker exec "${CONTAINER}" bash -c "
    PATH=/opt/conda/bin:\$PATH python3 -c 'import yaml' 2>/dev/null || PATH=/opt/conda/bin:\$PATH pip install pyyaml -q 2>/dev/null || true
    PATH=/opt/conda/bin:\$PATH python3 -c 'import evalscope' 2>/dev/null || PATH=/opt/conda/bin:\$PATH pip install evalscope pyyaml requests modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -q 2>/dev/null || true
"
echo "  依赖检查完成"

# 4.5. 写入 Token 到容器 .env（供脚本 fallback 读取）
echo "[4.5/6] 写入 Token 到容器 .env..."
ENV_LINES=""
ENV_COUNT=0
for VAR_NAME in GITHUB_TOKEN MODELSCOPE_TOKEN HF_TOKEN HARBOR_USER HARBOR_PASSWORD; do
    VAR_VAL="${!VAR_NAME:-}"
    if [ -n "${VAR_VAL}" ]; then
        ENV_LINES="${ENV_LINES}${VAR_NAME}=${VAR_VAL}
"
        ENV_COUNT=$((ENV_COUNT + 1))
    fi
done
if [ -n "${ENV_LINES}" ]; then
    echo "${ENV_LINES}" | docker exec -i "${CONTAINER}" bash -c "cat > /flagos-workspace/.env && chmod 600 /flagos-workspace/.env"
    echo "  ✓ /flagos-workspace/.env 已写入 (${ENV_COUNT} 个 token)"
else
    echo "  ⚠ 宿主机未设置任何 token 环境变量，跳过 .env 写入"
fi

# 4.6. 预装评测依赖（避免评测阶段首次 pip install 浪费 2-3 分钟）
echo "[4.6/6] 预装评测依赖..."
docker exec "${CONTAINER}" bash -c "PATH=/opt/conda/bin:\$PATH pip install evalscope pyyaml requests -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -q 2>&1 | tail -3" && \
    echo "  ✓ 评测依赖预装完成" || \
    echo "  ⚠ 评测依赖安装失败（非致命，评测阶段会重试）"

# 5. 验证
echo "[5/6] 验证部署..."
SCRIPT_COUNT=$(docker exec "${CONTAINER}" bash -c "ls /flagos-workspace/scripts/*.py /flagos-workspace/scripts/*.sh 2>/dev/null | wc -l")
echo "  容器内脚本数: ${SCRIPT_COUNT}"
docker exec "${CONTAINER}" ls -la /flagos-workspace/scripts/ 2>/dev/null || true

# 6. 记录挂载信息
echo ""
echo "[6/6] 记录挂载信息..."
echo "  挂载模式: ${MOUNT_MODE}"
if [ -n "${HOST_WORKSPACE}" ]; then
    echo "  宿主机工作目录: ${HOST_WORKSPACE}"
fi
# 写入标记文件供后续脚本读取
docker exec "${CONTAINER}" bash -c "echo '${MOUNT_MODE}' > /flagos-workspace/.mount_mode"

# 7. 写入基础 context 字段（确保段间传递不依赖 Claude 后续写入）
# 即使 Claude 会话中途断连，容器名等关键信息也已持久化到 context.yaml
echo ""
echo "[7/7] 写入基础 context 字段..."
CTX_CMD_ARGS=()
CTX_CMD_ARGS+=(--set "container.name=${CONTAINER}")
CTX_CMD_ARGS+=(--set "container.status=running")
[ -n "${HOST_WORKSPACE}" ] && CTX_CMD_ARGS+=(--set "workspace.host_path=${HOST_WORKSPACE}")
[ -n "${MODEL_NAME}" ] && CTX_CMD_ARGS+=(--set "model.name=${MODEL_NAME}")
# 用 printf %q 安全转义每个参数，避免空格/特殊字符注入
CTX_SET_ESCAPED=$(printf ' %q' "${CTX_CMD_ARGS[@]}")
docker exec "${CONTAINER}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py${CTX_SET_ESCAPED} --json" >/dev/null 2>&1 && \
    echo "  ✓ 基础 context 已写入 (container.name, status, workspace, model)" || \
    echo "  ⚠ 基础 context 写入失败（非致命，Claude 会补写）"

echo ""
echo "=========================================="
echo "工作区初始化完成"
echo "=========================================="
echo "  容器: ${CONTAINER}"
echo "  挂载模式: ${MOUNT_MODE}"
echo "  容器内路径: /flagos-workspace"
if [ -n "${HOST_WORKSPACE}" ]; then
echo "  宿主机路径: ${HOST_WORKSPACE}"
fi
echo "  脚本目录: /flagos-workspace/scripts/"
echo "  结果目录: /flagos-workspace/results/"
echo "  报告目录: /flagos-workspace/reports/"
echo "=========================================="
