#!/bin/bash
# FlagOS 全自动迁移流程 — 一键启动脚本（V1+V2+V3 算子调优）
#
# 用法:
#   bash prompts/run_pipeline.sh <容器名或镜像地址> <模型名> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--model-path <路径>] [--verbose] [--proxy proxy1,proxy2,...]
#
# 自动识别：第一参数若为已有容器则走容器模式，否则视为镜像地址
# 模型路径：仅需模型名，自动搜索宿主机路径；未找到则容器内自动下载。也可通过 --model-path 显式指定
#
# 示例:
#   bash prompts/run_pipeline.sh qwen3-8b-test Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass
#   bash prompts/run_pipeline.sh harbor.baai.ac.cn/flagrelease/qwen3:latest Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --proxy http://10.1.12.192:80
#   bash prompts/run_pipeline.sh harbor.baai.ac.cn/flagrelease/qwen3:latest Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --model-path /data/models/Qwen3-8B
#
# 向后兼容（已弃用）:
#   bash prompts/run_pipeline.sh --image <镜像地址> <模型名> [<宿主机模型路径>] <tokens...>
#
# 前置条件:
#   - Claude Code CLI 已安装 (claude 命令可用)
#   - Docker daemon 正在运行
#   - 当前目录为项目根目录 (flagos_skills_V3/)

set -euo pipefail

# ========== Docker 前置检查 ==========
# 用 docker ps 代替 docker info，避免 authZ 插件拦截 docker info 导致误判
if ! docker ps &>/dev/null; then
    echo "错误: Docker daemon 未运行或无权限，请检查 Docker 状态"
    exit 1
fi

# ========== 宿主机 Python 依赖检查 ==========
if ! command -v python3 &>/dev/null; then
    echo "错误: python3 未安装，请先安装 Python 3"
    exit 1
fi

HOST_PY_DEPS=("yaml:pyyaml" "huggingface_hub:huggingface_hub")
MISSING_PKGS=()
for dep in "${HOST_PY_DEPS[@]}"; do
    mod="${dep%%:*}"
    pkg="${dep##*:}"
    if ! python3 -c "import ${mod}" 2>/dev/null; then
        MISSING_PKGS+=("${pkg}")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "[pre-flight] 安装缺失的宿主机 Python 依赖: ${MISSING_PKGS[*]}"
    pip3 install "${MISSING_PKGS[@]}" -q 2>/dev/null || \
    pip3 install "${MISSING_PKGS[@]}" -q -i https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || \
    pip3 install "${MISSING_PKGS[@]}" -q -i https://pypi.tuna.tsinghua.edu.cn/simple/ 2>/dev/null || \
    { echo "错误: 宿主机 Python 依赖安装失败: ${MISSING_PKGS[*]}"; exit 1; }
fi

# ========== 参数解析与自动识别 ==========
IMAGE_MODE=false
MODEL_PATH=""
FILTER_FLAGS=""
PROXY_LIST=""

if [[ "${1:-}" == "--image" ]]; then
    # 向后兼容：旧 --image 格式
    echo "⚠ --image 标志已弃用，直接传镜像地址作为第一参数即可自动识别"
    shift
    IMAGE_MODE=true
    if [ $# -ge 8 ]; then
        # 旧格式: --image <镜像> <模型名> <宿主机模型路径> <5个token> [--verbose]
        IMAGE="$1"
        MODEL="$2"
        MODEL_PATH="$3"
        export MODELSCOPE_TOKEN="$4"
        export HF_TOKEN="$5"
        export GITHUB_TOKEN="$6"
        export HARBOR_USER="$7"
        export HARBOR_PASSWORD="$8"
        if [[ "${9:-}" == "--verbose" ]]; then
            FILTER_FLAGS="--verbose"
        fi
    elif [ $# -ge 7 ]; then
        # 新格式带 --image: --image <镜像> <模型名> <5个token> [--verbose]
        IMAGE="$1"
        MODEL="$2"
        export MODELSCOPE_TOKEN="$3"
        export HF_TOKEN="$4"
        export GITHUB_TOKEN="$5"
        export HARBOR_USER="$6"
        export HARBOR_PASSWORD="$7"
        if [[ "${8:-}" == "--verbose" ]]; then
            FILTER_FLAGS="--verbose"
        fi
    else
        echo "用法: $0 <容器名或镜像地址> <模型名> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--verbose] [--proxy proxy1,proxy2,...]"
        exit 1
    fi
else
    # 统一格式：7 个位置参数
    if [ $# -lt 7 ]; then
        echo "用法: $0 <容器名或镜像地址> <模型名> <MODELSCOPE_TOKEN> <HF_TOKEN> <GITHUB_TOKEN> <HARBOR_USER> <HARBOR_PASSWORD> [--model-path <路径>] [--verbose] [--proxy proxy1,proxy2,...]"
        echo ""
        echo "自动识别：第一参数若为已有容器则走容器模式，否则视为镜像地址"
        echo ""
        echo "示例:"
        echo "  $0 qwen3-8b-test Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass"
        echo "  $0 harbor.baai.ac.cn/flagrelease/qwen3:latest Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --proxy http://10.1.12.192:80"
        echo "  $0 harbor.baai.ac.cn/flagrelease/qwen3:latest Qwen3-8B ms_xxx hf_xxx ghp_xxx harbor_user harbor_pass --model-path /data/models/Qwen3-8B"
        echo "  加 --verbose 显示全量终端输出（调试用）"
        exit 1
    fi

    TARGET="$1"
    MODEL="$2"
    export MODELSCOPE_TOKEN="$3"
    export HF_TOKEN="$4"
    export GITHUB_TOKEN="$5"
    export HARBOR_USER="$6"
    export HARBOR_PASSWORD="$7"
    shift 7
    # 解析可选参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verbose) FILTER_FLAGS="--verbose"; shift ;;
            --proxy) PROXY_LIST="$2"; shift 2 ;;
            --model-path)
                if [ -z "${2:-}" ]; then
                    echo "错误: --model-path 需要指定路径"
                    exit 1
                fi
                MODEL_PATH="$2"
                shift 2
                ;;
            *)
                echo "警告: 未知参数 '$1'，已忽略"
                shift
                ;;
        esac
    done

    # 自动识别：含冒号(:)或斜杠(/)的视为镜像地址，否则尝试 docker inspect 判断
    if [[ "$TARGET" == *":"* ]] || [[ "$TARGET" == *"/"* ]]; then
        # 包含冒号(tag)或斜杠(registry路径)，强制镜像模式
        IMAGE_MODE=true
        IMAGE="$TARGET"
    elif docker inspect --type=container "$TARGET" &>/dev/null; then
        IMAGE_MODE=false
        CONTAINER="$TARGET"
    else
        IMAGE_MODE=true
        IMAGE="$TARGET"
    fi
fi

if [ -z "$MODEL" ]; then
    echo "错误: 模型名为空，请检查输入参数（可能是分隔符使用了全角字符）"
    exit 1
fi

# ========== 镜像模式：自动搜索宿主机模型路径 ==========
if $IMAGE_MODE && [ -z "$MODEL_PATH" ]; then
    echo "[pre-flight] 搜索宿主机模型路径: ${MODEL} ..."
    SEARCH_JSON=$(python3 skills/flagos-container-preparation/tools/check_model_local.py \
        --model "${MODEL}" --no-download --output-json 2>/dev/null) || SEARCH_JSON=""

    MODEL_PATH=$(echo "$SEARCH_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('best_match') or '')
except:
    print('')
" 2>/dev/null) || MODEL_PATH=""

    MODEL_FOUND_ON_HOST=false
    if [ -n "$MODEL_PATH" ]; then
        echo "  ✓ 找到: ${MODEL_PATH}"
        MODEL_FOUND_ON_HOST=true
    else
        MODEL_SHORT=$(echo "${MODEL}" | sed 's|.*/||')
        MODEL_PATH="/data/models/${MODEL_SHORT}"
        mkdir -p "${MODEL_PATH}"
        echo "  ⚠ 宿主机未找到模型，预创建挂载目录: ${MODEL_PATH}"
    fi
    CONTAINER_MODEL_PATH="${MODEL_PATH}"
fi

# 用户通过 --model-path 显式指定时，设置相关变量
if [ -n "$MODEL_PATH" ] && [ -z "${MODEL_FOUND_ON_HOST:-}" ]; then
    if ! $IMAGE_MODE; then
        echo "警告: --model-path 在容器模式下无效（容器已有自己的文件系统），已忽略"
        MODEL_PATH=""
    elif [ ! -d "$MODEL_PATH" ]; then
        echo "错误: 指定的模型路径不存在: ${MODEL_PATH}"
        exit 1
    else
        MODEL_FOUND_ON_HOST=true
        CONTAINER_MODEL_PATH="${MODEL_PATH}"
        USER_SPECIFIED_MODEL_PATH=true
        echo "[pre-flight] 使用指定模型路径: ${MODEL_PATH}"
    fi
fi

# ========== Banner ==========
echo "============================================================"
echo "  FlagOS 全自动迁移流程"
echo "============================================================"
if $IMAGE_MODE; then
    echo "  目标: ${IMAGE} (镜像，自动识别)"
else
    echo "  目标: ${CONTAINER} (容器，自动识别)"
fi
echo "  模型: ${MODEL}"
if $IMAGE_MODE; then
    if $MODEL_FOUND_ON_HOST; then
        if [ -n "${USER_SPECIFIED_MODEL_PATH:-}" ]; then
            echo "  模型路径: ${MODEL_PATH} (用户指定)"
        else
            echo "  模型路径: ${MODEL_PATH} (自动检测)"
        fi
    else
        echo "  模型路径: ${MODEL_PATH} (预创建，容器内下载)"
    fi
fi
echo "  模式: V1 + V2 + V3（不达标时自动算子优化）"
echo "  权限: --permission-mode auto + settings.local.json allowlist (89 rules)"
echo "============================================================"
echo ""

# ========== 网络预检 ==========
# 如果未传 --proxy，从环境变量读取
if [ -z "$PROXY_LIST" ]; then
    CURRENT_PROXY="${https_proxy:-${http_proxy:-}}"
    [ -n "$CURRENT_PROXY" ] && PROXY_LIST="$CURRENT_PROXY"
fi

if [ -n "$PROXY_LIST" ]; then
    echo "[pre-flight] 网络连通性检测..."
    NETWORK_JSON=$(bash prompts/check_network.sh --proxies "${PROXY_LIST}" --json 2>/dev/null) || NETWORK_JSON=""
    if [ -n "$NETWORK_JSON" ]; then
        BEST_PROXY=$(echo "$NETWORK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['recommended_proxy'])" 2>/dev/null) || BEST_PROXY=""
        ALL_FAILED=$(echo "$NETWORK_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['all_failed'])" 2>/dev/null) || ALL_FAILED="false"

        if [ "$ALL_FAILED" = "true" ] || [ "$ALL_FAILED" = "True" ]; then
            echo "  ✗ 所有代理均不可用，流程终止"
            echo "  代理列表: ${PROXY_LIST}"
            exit 1
        fi

        if [ -n "$BEST_PROXY" ] && [ "$BEST_PROXY" != "direct" ]; then
            export http_proxy="$BEST_PROXY"
            export https_proxy="$BEST_PROXY"
            echo "  ✓ 推荐代理: ${BEST_PROXY}"
        elif [ "$BEST_PROXY" = "direct" ]; then
            echo "  ✓ 直连可用，无需代理"
        fi
    else
        echo "  ⚠ 网络检测脚本执行失败，使用默认代理继续"
    fi
    export FLAGOS_PROXY_LIST="${PROXY_LIST}"
    echo ""
fi

# ========== 构造 Prompt ==========
# 公共部分：tokens、执行模式、进度输出要求、步骤2-6
COMMON_TOKENS=$(cat <<TOKENS_EOF

**容器内 Token**（已通过 setup_workspace.sh 写入容器 /flagos-workspace/.env，脚本自动加载；docker exec -e 仍建议保留作为双保险）：
  MODELSCOPE_TOKEN=${MODELSCOPE_TOKEN}
  HF_TOKEN=${HF_TOKEN}
  GITHUB_TOKEN=${GITHUB_TOKEN}
  HARBOR_USER=${HARBOR_USER}
  HARBOR_PASSWORD=${HARBOR_PASSWORD}

**网络代理**（已通过 setup_workspace.sh 写入容器 /flagos-workspace/.proxy）：
  FLAGOS_PROXY_LIST=${FLAGOS_PROXY_LIST:-}
  FLAGOS_ACTIVE_PROXY=${http_proxy:-}
  所有需要外网的 docker exec 命令必须传入代理: docker exec -e http_proxy=${http_proxy:-} -e https_proxy=${https_proxy:-} ...
  网络操作失败时，从 FLAGOS_PROXY_LIST 中逐个切换代理重试
TOKENS_EOF
)

COMMON_PLAN_FIRST=$(cat <<'PLAN_EOF'

**执行模式：计划优先（Plan-First）**

在执行任何操作之前，先完成规划阶段：
1. 只读取本段需要的 SKILL.md 文件，提取关键命令、参数、文件路径：
   - 段1（步骤1/2/3）：skills/flagos-container-preparation/SKILL.md、skills/flagos-pre-service-inspection/SKILL.md、skills/flagos-service-startup/SKILL.md
   - 不要读取本段不执行的步骤对应的 SKILL.md（节省时间）
PLAN_EOF
)

COMMON_PLAN_STEPS=$(cat <<PLAN_STEPS_EOF
2. 每个步骤开始前，通过 docker exec 读取容器内 /flagos-workspace/shared/context.yaml 获取最新状态

**重要：所有 /flagos-workspace 下的文件操作必须通过 docker exec 在容器内执行。Claude Code 的 Bash 沙箱禁止直接操作 /data/ 路径。**

全自动执行，步骤间无需询问。
**进度输出（硬性要求）**：每个步骤的第一条命令执行之前，必须先输出 \`[步骤X] <步骤名> — 开始\` 标记行（步骤1也不例外——规划完成后、执行第一条命令之前，必须先输出 \`[步骤1] 容器准备 — 开始\`）。步骤完成时输出 \`[步骤X] <步骤名> — 完成 (耗时)\`。关键命令后输出 ✓/✗ 结果摘要。按 CLAUDE.md 流水线执行日志规范输出。

**强制规则：每个步骤完成后立即同步 context_snapshot.yaml**
每完成一个步骤（1/2/3/4/5/6/7/8），在写入 trace 和更新容器内 context.yaml 之后，必须立即执行以下同步命令：
  MOUNT_MODE=\$(docker exec \${CONTAINER} cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
  if [ "\$MOUNT_MODE" = "mounted" ] || [ "\$MOUNT_MODE" = "symlink" ]; then
    cp /data/flagos-workspace/${MODEL}/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
  else
    docker cp \${CONTAINER}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
  fi
这是硬性要求，不可省略、不可延迟到段末尾。每个步骤结束时都必须执行一次。

**强制规则：每个步骤完成后生成/更新报告**
台账、trace、timing 更新并同步 context_snapshot.yaml 之后，立即调用：
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/generate_report.py --output /flagos-workspace/results/report.md"
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/generate_report.py --json --output /flagos-workspace/results/report.json"
报告随步骤推进自动丰富，无需手动拼接。

**context.yaml 更新方式（必须使用工具脚本，禁止手写 Python）**：
容器内已部署 update_context.py，支持嵌套字段设置、数组追加、台账更新，避免 sandbox 拦截：
  # 设置字段
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --set container.name=xxx --set gpu.count=8 --json"
  # 设置复杂 JSON 值
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --json-set 'service={\\\"port\\\":8001}' --json"
  # 更新台账
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --ledger-update 01_container_preparation --ledger-status success --ledger-notes '容器就绪' --json"
  # 追加数组
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --append issues.submitted=/path/to/issue.md --json"
  # 设置 timing
  docker exec \${CONTAINER} bash -c "PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --set-timing steps.container_preparation=171 --json"
PLAN_STEPS_EOF
)


# ========== 根据模式构造步骤1 ==========
if $IMAGE_MODE; then
    if $MODEL_FOUND_ON_HOST; then
        MODEL_NOTE="宿主机模型路径已自动检测: ${MODEL_PATH}"
        DOWNLOAD_NOTE=""
    else
        MODEL_NOTE="宿主机未找到模型 ${MODEL}，已预创建挂载目录: ${MODEL_PATH}"
        DOWNLOAD_NOTE="
   - 容器创建后，在容器内搜索+下载模型（下载到已挂载的 ${CONTAINER_MODEL_PATH}）：
     python3 skills/flagos-container-preparation/tools/check_model_local.py --model \"${MODEL}\" --mode container --container \${CONTAINER} --output-json
     从输出 JSON 中提取 final_container_path 和 final_host_path，记录到容器内 /flagos-workspace/shared/context.yaml"
    fi
    STEP1=$(cat <<STEP1_EOF
1. 容器准备（从镜像创建）：
   - ${MODEL_NOTE}
   - 检测 GPU 厂商（nvidia-smi / npu-smi 等），选择 SKILL.md 中对应的 docker run 模板
   - **NVIDIA 模板（严格执行，仅替换变量值，禁止增删参数）**：
     docker run -itd --name=\${CONTAINER_NAME} --gpus=all --network=host -v ${MODEL_PATH}:${CONTAINER_MODEL_PATH} -v /data/flagos-workspace/${MODEL}:/flagos-workspace ${IMAGE}
   - **降级策略**：模板失败 → 检查变量值修正后重试 → 仍失败则 docker inspect 借鉴已有容器挂载配置重试一次 → 仍失败则终止
   - 容器名自动生成为 <model_short_name>_flagos（如 Qwen3-8B_flagos）
   - 如同名容器已存在，追加时间戳：<model_short_name>_flagos_<MMDD_HHMM>
   - 镜像模式下禁止复用已有容器，必须 docker run 新建${DOWNLOAD_NOTE}
   - bash skills/flagos-container-preparation/tools/setup_workspace.sh \${CONTAINER} ${MODEL} --skip-archive 部署工具脚本（宿主机已归档，跳过容器内归档避免移走正在写入的日志）
   - 写入容器内 /flagos-workspace/shared/context.yaml（entry.type=new_container, image.name=${IMAGE}）+ traces/01_container_preparation.json
STEP1_EOF
    )
    PROMPT_SEG1="镜像: ${IMAGE}，模型名: ${MODEL}
${COMMON_TOKENS}
${COMMON_PLAN_FIRST}
${COMMON_PLAN_STEPS}

${STEP1}

步骤2/3 按 CLAUDE.md 工作流定义执行。GITHUB_TOKEN=${GITHUB_TOKEN}（issue 提交时通过 docker exec -e 传入）。

**步骤3 FlagGems 启动崩溃算子诊断**：
- FlagGems 模式启动崩溃时（不含超时），先备份崩溃日志再诊断：
  docker exec \${CONTAINER} bash -c \"cp /flagos-workspace/logs/startup_default.log /flagos-workspace/logs/startup_default_crashed.log 2>/dev/null; true\"
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/diagnose_ops.py crash-log \\
    --log-path /flagos-workspace/logs/startup_default_crashed.log --json\"
- crashed_ops 非空 → 禁用问题算子（toggle_flaggems.py --action modify-enable --disabled-ops \"算子列表\"）→ 重启服务 → 最多重试 2 轮
- 重试成功 → 记录 disabled_ops 到 context.yaml 的 optimization.disabled_ops，正常继续
- 重试全部失败或 crashed_ops 为空 → 走下方 Issue 强制规则

**步骤3 Issue 强制规则**：
- FlagGems 模式启动崩溃（不含超时）→ 必须调用 issue_reporter.py：
  docker exec -e GITHUB_TOKEN=${GITHUB_TOKEN} \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \\
    --type operator-crash --log-path /flagos-workspace/logs/startup_default_crashed.log \\
    --context-yaml /flagos-workspace/shared/context.yaml --repo flagos-ai/FlagGems \\
    --output-dir /flagos-workspace/results/ --json\"
- 生成的 issue 文件路径写入 context.yaml 的 issues.submitted[]
- 追加写入 logs/issues_startup.log（格式见 CLAUDE.md 问题日志规范）

**步骤3 FlagGems 崩溃后流程控制**：
- FlagGems 模式启动成功（推理验证通过）→ **必须设置 workflow.service_ok=true**（通过 update_context.py --set workflow.service_ok=true）
- 算子诊断重试全部失败或 crashed_ops 为空 → 提交 issue 后设置 workflow.service_ok=false
- 非算子原因（非硬件）导致的 FlagGems 崩溃 → 同样设置 workflow.service_ok=false
- service_ok=false 时：用 USE_FLAGGEMS=0 启动 native 服务验证环境可用性，但不影响 service_ok 判定
- service_ok 含义：FlagGems 模式是否可用（native 能启动但 FlagGems 不能 → service_ok=false）

完成步骤3后，通过 docker cp 同步 context 到宿主机：
  docker cp \${CONTAINER}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
（如果 mount_mode=mounted，也可：cp /data/flagos-workspace/${MODEL}/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml）

**⚠ 段边界（硬性约束 — 最高优先级）**：
- 本段**只执行步骤1/2/3**，步骤3完成并同步 context_snapshot.yaml 后**必须立即停止**
- **绝对禁止**执行步骤4（精度评测）、步骤5、步骤6、步骤7 或任何后续步骤
- **绝对禁止**启动 benchmark、eval、算子调优等属于步骤4-7的操作
- **禁止**更新 ledger 中 04_quick_accuracy 及之后步骤的状态。违反此约束将导致后续段重复执行
- 步骤3完成 = V1/V2 服务启动验证通过 + context 同步。验证通过后停止服务、同步 context，然后立即结束
- 步骤4-13 由后续独立会话执行，本段无权触碰
完成标志：输出 \"[段1] 步骤1/2/3全部完成，context 已同步\" 后**立即停止所有操作，不再执行任何命令**。"
else
    STEP1=$(cat <<STEP1_EOF
1. 下载模型+容器准备：
   - 验证容器 ${CONTAINER} 运行状态（docker inspect + docker start）
   - 搜索模型权重：python3 skills/flagos-container-preparation/tools/check_model_local.py --model "${MODEL}" --mode container --container ${CONTAINER} --output-json
     - 先在容器内搜索（/data, /models, /root, /home, /workspace, /mnt, /opt）
     - 再在宿主机搜索，检查是否已通过挂载卷映射到容器
     - 如容器内未找到 → 在容器内自动从 ModelScope 下载（优先下载到已挂载卷路径，避免写入 overlay）
     - 从输出 JSON 中提取 final_container_path 和 final_host_path，记录到容器内 /flagos-workspace/shared/context.yaml 的 model.container_path 和 model.local_path
   - bash skills/flagos-container-preparation/tools/setup_workspace.sh ${CONTAINER} ${MODEL} --skip-archive 部署工具脚本（宿主机已归档，跳过容器内归档避免移走正在写入的日志）
   - 写入容器内 /flagos-workspace/shared/context.yaml + traces/01_container_preparation.json
STEP1_EOF
    )
    PROMPT_SEG1="容器名: ${CONTAINER}，模型名: ${MODEL}
${COMMON_TOKENS}
${COMMON_PLAN_FIRST}
${COMMON_PLAN_STEPS}

${STEP1}

步骤2/3 按 CLAUDE.md 工作流定义执行。GITHUB_TOKEN=${GITHUB_TOKEN}（issue 提交时通过 docker exec -e 传入）。

**步骤3 FlagGems 启动崩溃算子诊断**：
- FlagGems 模式启动崩溃时（不含超时），先备份崩溃日志再诊断：
  docker exec \${CONTAINER} bash -c \"cp /flagos-workspace/logs/startup_default.log /flagos-workspace/logs/startup_default_crashed.log 2>/dev/null; true\"
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/diagnose_ops.py crash-log \\
    --log-path /flagos-workspace/logs/startup_default_crashed.log --json\"
- crashed_ops 非空 → 禁用问题算子（toggle_flaggems.py --action modify-enable --disabled-ops \"算子列表\"）→ 重启服务 → 最多重试 2 轮
- 重试成功 → 记录 disabled_ops 到 context.yaml 的 optimization.disabled_ops，正常继续
- 重试全部失败或 crashed_ops 为空 → 走下方 Issue 强制规则

**步骤3 Issue 强制规则**：
- FlagGems 模式启动崩溃（不含超时）→ 必须调用 issue_reporter.py：
  docker exec -e GITHUB_TOKEN=${GITHUB_TOKEN} ${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \\
    --type operator-crash --log-path /flagos-workspace/logs/startup_default_crashed.log \\
    --context-yaml /flagos-workspace/shared/context.yaml --repo flagos-ai/FlagGems \\
    --output-dir /flagos-workspace/results/ --json\"
- 生成的 issue 文件路径写入 context.yaml 的 issues.submitted[]
- 追加写入 logs/issues_startup.log（格式见 CLAUDE.md 问题日志规范）

**步骤3 FlagGems 崩溃后流程控制**：
- FlagGems 模式启动成功（推理验证通过）→ **必须设置 workflow.service_ok=true**（通过 update_context.py --set workflow.service_ok=true）
- 算子诊断重试全部失败或 crashed_ops 为空 → 提交 issue 后设置 workflow.service_ok=false
- 非算子原因（非硬件）导致的 FlagGems 崩溃 → 同样设置 workflow.service_ok=false
- service_ok=false 时：用 USE_FLAGGEMS=0 启动 native 服务验证环境可用性，但不影响 service_ok 判定
- service_ok 含义：FlagGems 模式是否可用（native 能启动但 FlagGems 不能 → service_ok=false）

完成步骤3后，通过 docker cp 同步 context 到宿主机：
  docker cp ${CONTAINER}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
（如果 mount_mode=mounted，也可：cp /data/flagos-workspace/${MODEL}/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml）

**⚠ 段边界（硬性约束 — 最高优先级）**：
- 本段**只执行步骤1/2/3**，步骤3完成并同步 context_snapshot.yaml 后**必须立即停止**
- **绝对禁止**执行步骤4（精度评测）、步骤5、步骤6、步骤7 或任何后续步骤
- **绝对禁止**启动 benchmark、eval、算子调优等属于步骤4-7的操作
- **禁止**更新 ledger 中 04_quick_accuracy 及之后步骤的状态。违反此约束将导致后续段重复执行
- 步骤3完成 = V1/V2 服务启动验证通过 + context 同步。验证通过后停止服务、同步 context，然后立即结束
- 步骤4-13 由后续独立会话执行，本段无权触碰
完成标志：输出 \"[段1] 步骤1/2/3全部完成，context 已同步\" 后**立即停止所有操作，不再执行任何命令**。"
fi

# ========== 部署权限白名单 ==========
mkdir -p .claude && cp settings.local.json .claude/settings.local.json

# ========== 动态注入模型特定权限 ==========
# auto mode 可能被降级为 default mode（mco-4 等非官方模型 ID），
# default mode 下通配符规则对链式命令不生效，需要精确匹配的模型特定规则
python3 -c "
import json, sys
model = sys.argv[1]
with open('.claude/settings.local.json') as f:
    cfg = json.load(f)
rules = cfg.setdefault('permissions', {}).setdefault('allow', [])
# 模型特定的目录操作权限
for d in ['logs', 'config', 'results', 'traces']:
    rule = f'Bash(mkdir -p /data/flagos-workspace/{model}/{d})'
    if rule not in rules:
        rules.append(rule)
# 模型特定的文件读取权限
for rule in [
    f'Read(//data/flagos-workspace/{model}/**)',
    f'Bash(cat /data/flagos-workspace/{model}/*)',
    f'Bash(find /data/flagos-workspace/{model}/*)',
    f'Bash(tail /data/flagos-workspace/{model}/*)',
]:
    if rule not in rules:
        rules.append(rule)
with open('.claude/settings.local.json', 'w') as f:
    json.dump(cfg, f, indent=2)
" "${MODEL}"
echo "  ✓ 已注入 ${MODEL} 模型特定权限规则"

# ========== 启动 Claude Code ==========
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Claude Code 全自动流程..."
echo ""

# ========== 宿主机历史数据全量归档 ==========
# 在创建任何新文件之前，统一归档上一轮的全部产出（results/traces/logs/config）
# setup_workspace.sh 中的宿主机归档作为独立调用时的兜底
HOST_BASE="/data/flagos-workspace/${MODEL}"
if [ -d "${HOST_BASE}" ]; then
    HOST_HAS_HISTORY=0
    for d in results traces logs config reports eval; do
        if [ -d "${HOST_BASE}/${d}" ] && [ "$(ls -A "${HOST_BASE}/${d}" 2>/dev/null)" ]; then
            HOST_HAS_HISTORY=1; break
        fi
    done
    if [ "${HOST_HAS_HISTORY}" = "1" ]; then
        ARCHIVE_TS="$(date +%Y%m%d_%H%M%S)"
        HOST_ARCHIVE="${HOST_BASE}/archive/${ARCHIVE_TS}"
        mkdir -p "${HOST_ARCHIVE}"
        for d in results traces logs config reports eval; do
            if [ -d "${HOST_BASE}/${d}" ] && [ "$(ls -A "${HOST_BASE}/${d}" 2>/dev/null)" ]; then
                mv "${HOST_BASE}/${d}" "${HOST_ARCHIVE}/${d}"
            fi
        done
        echo "  宿主机历史数据已归档到: ${HOST_ARCHIVE}/"
    fi
fi

for d in logs config results traces; do
    mkdir -p "/data/flagos-workspace/${MODEL}/${d}"
done

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="/data/flagos-workspace/${MODEL}/logs"
LOG_FILE="${LOG_DIR}/claude_pipeline_${TIMESTAMP}.log"
FULL_LOG="${LOG_DIR}/claude_full_${TIMESTAMP}.log"
DEBUG_FILE="${LOG_DIR}/claude_debug_${TIMESTAMP}.log"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
TERMINAL_LOG="${LOG_DIR}/terminal.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "日志文件:"
echo "  原始事件流: ${LOG_FILE}"
echo "  可读执行记录: ${FULL_LOG}"
echo "  流水线日志: ${PIPELINE_LOG}"
echo "  内部 debug: ${DEBUG_FILE}"
echo ""

# 禁用实验性 beta 功能，避免第三方代理不支持 context_management 返回 400
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
# 禁用非核心 haiku 调用（session title 等），第三方代理通常无 haiku 权限会 403
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# ===== 段间状态传递函数 =====
read_context() {
    local MODEL_ARG="$1"
    local CTX="/data/flagos-workspace/${MODEL_ARG}/config/context_snapshot.yaml"
    if [ ! -f "${CTX}" ]; then
        echo "ERROR: context_snapshot.yaml 不存在，前段可能未完成" >&2
        return 1
    fi
    # 输出 CONTAINER_NAME|ENV_TYPE|LAST_COMPLETED_STEP
    python3 -c "
import yaml
with open('${CTX}') as f:
    ctx = yaml.safe_load(f)
ctr = ctx.get('container',{}).get('name','')
env = ctx.get('environment',{}).get('env_type','')
ledger = ctx.get('workflow_ledger',{}).get('steps',{})
last = ''
if isinstance(ledger, dict):
    for key, step_data in ledger.items():
        if isinstance(step_data, dict) and step_data.get('status') == 'success':
            last = step_data.get('step', key)
elif isinstance(ledger, list):
    for s in ledger:
        if isinstance(s, dict) and s.get('status') == 'success':
            last = s.get('step','')
print(f'{ctr}|{env}|{last}')
" 2>/dev/null
}

# ===== GPU 服务清理（脚本退出时自动执行） =====
cleanup_gpu_services() {
    local ctr=""
    # 按优先级查找容器名：DIAG_CONTAINER > SEG_CTR > CONTAINER
    for candidate in "${DIAG_CONTAINER:-}" "${SEG_CTR:-}" "${CONTAINER:-}"; do
        if [ -n "${candidate}" ] && docker inspect --type=container "${candidate}" &>/dev/null; then
            ctr="${candidate}"
            break
        fi
    done
    # fallback: 从宿主机 context_snapshot.yaml 读取
    if [ -z "${ctr}" ]; then
        local ctx_file="/data/flagos-workspace/${MODEL:-unknown}/config/context_snapshot.yaml"
        if [ -f "${ctx_file}" ]; then
            ctr=$(python3 -c "
import yaml
try:
    with open('${ctx_file}') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('container',{}).get('name',''))
except: pass
" 2>/dev/null) || ctr=""
        fi
    fi
    # 执行清理
    if [ -n "${ctr}" ] && docker inspect --type=container "${ctr}" &>/dev/null; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理 GPU 资源：停止容器 ${ctr} 内的推理服务..."
        docker exec "${ctr}" bash -c "pkill -f 'vllm\|sglang\|flagscale' 2>/dev/null; sleep 2" 2>/dev/null && \
            echo "  ✓ 推理服务已停止，GPU 显存已释放" || \
            echo "  ⚠ 未发现运行中的推理服务（可能已停止）"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 未找到有效容器，跳过 GPU 服务清理"
    fi
}
trap 'cleanup_gpu_services' EXIT

# ===== 段间完成性校验 =====
# 检查 context_snapshot.yaml 中 workflow_ledger 的步骤完成状态
# 用法: check_seg_complete <model> <keyword1> [keyword2...]
# 返回: "complete" 或 "incomplete"
check_seg_complete() {
    local model="$1"
    shift
    local keywords_csv=""
    for kw in "$@"; do
        [ -n "$keywords_csv" ] && keywords_csv="${keywords_csv},"
        keywords_csv="${keywords_csv}${kw}"
    done
    python3 - "$model" "$keywords_csv" <<'PYEOF'
import yaml, sys
model = sys.argv[1]
keywords = [k.lower() for k in sys.argv[2].split(",") if k]
try:
    with open(f"/data/flagos-workspace/{model}/config/context_snapshot.yaml") as f:
        ctx = yaml.safe_load(f)
except:
    print("incomplete")
    sys.exit(0)
ledger = ctx.get("workflow_ledger", {}).get("steps", {})
found = set()
items = ledger.items() if isinstance(ledger, dict) else [(i, s) for i, s in enumerate(ledger)] if isinstance(ledger, list) else []
for key, step_data in items:
    if not isinstance(step_data, dict):
        continue
    step_name = str(step_data.get("step", key)).lower()
    status = step_data.get("status", "")
    if status in ("success", "failed", "skipped"):
        for kw in keywords:
            if kw in step_name:
                found.add(kw)
if all(kw in found for kw in keywords):
    print("complete")
else:
    print("incomplete")
PYEOF
}

# 输出 ledger 状态摘要
print_ledger_summary() {
    local ctx_file="$1"
    python3 - "$ctx_file" <<'PYEOF'
import yaml, sys
try:
    with open(sys.argv[1]) as f:
        ctx = yaml.safe_load(f)
    ledger = ctx.get("workflow_ledger", {}).get("steps", {})
    if isinstance(ledger, dict):
        for k, v in ledger.items():
            if isinstance(v, dict):
                print(f'    {v.get("step",k)}: {v.get("status","unknown")}')
    elif isinstance(ledger, list):
        for s in ledger:
            if isinstance(s, dict):
                print(f'    {s.get("step","?")}: {s.get("status","unknown")}')
except Exception as e:
    print(f"    (ledger 读取失败: {e})")
PYEOF
}

# ===== 全流程计时 =====
PIPELINE_START_TS=$(date +%s)

# ===== 段1: 1/2/3 (容器准备 + 环境检测 + 服务启动) =====
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  段1/3  容器准备 + 环境检测 + 服务启动  (步骤 1→2→3)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
SEG1_START_TS=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 段1 开始"
claude -p "${PROMPT_SEG1}" \
    --permission-mode auto \
    --output-format stream-json \
    --verbose \
    --debug-file "${DEBUG_FILE}.seg1" \
    --max-turns 100 \
    2>&1 | tee "${LOG_FILE}" \
         | tee >(python3 "${SCRIPT_DIR}/stream_to_debug_log.py" > "${FULL_LOG}") \
         | python3 "${SCRIPT_DIR}/stream_filter.py" --pipeline-log "${PIPELINE_LOG}" --terminal-log "${TERMINAL_LOG}" --cost-file "${LOG_DIR}/seg1_cost.txt" --durations-file "${LOG_DIR}/seg1_durations.json" ${FILTER_FLAGS} || true

# 段间检查
SEG1_END_TS=$(date +%s)
SEG1_ELAPSED=$(( SEG1_END_TS - SEG1_START_TS ))
SEG1_MIN=$(( SEG1_ELAPSED / 60 ))
SEG1_SEC=$(( SEG1_ELAPSED % 60 ))
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  段1 完成 — 耗时 ${SEG1_MIN}m ${SEG1_SEC}s                                     │"
echo "└──────────────────────────────────────────────────────────────┘"

# 强制同步 context（不依赖 Claude 会话是否已同步）
CTX_FILE="/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml"
SHARED_CTX="/data/flagos-workspace/${MODEL}/shared/context.yaml"
mkdir -p "$(dirname "${CTX_FILE}")"
if [ -f "${SHARED_CTX}" ]; then
    cp "${SHARED_CTX}" "${CTX_FILE}"
else
    FALLBACK_CTR=$(grep -oP '(?<=容器 )\S+(?= 就绪)' "${PIPELINE_LOG}" 2>/dev/null | tail -1)
    if [ -n "${FALLBACK_CTR}" ] && docker inspect --type=container "${FALLBACK_CTR}" &>/dev/null; then
        docker cp "${FALLBACK_CTR}:/flagos-workspace/shared/context.yaml" "${CTX_FILE}" 2>/dev/null || true
    fi
fi
CTX_INFO=$(read_context "${MODEL}" 2>/dev/null) || { echo "错误：段1未产出 context_snapshot.yaml 或解析失败，终止"; exit 1; }
SEG_CTR=$(echo "$CTX_INFO" | cut -d'|' -f1)
SEG_ENV=$(echo "$CTX_INFO" | cut -d'|' -f2)
SEG_LAST=$(echo "$CTX_INFO" | cut -d'|' -f3)

if [ -z "$SEG_CTR" ]; then
    echo "  ⚠ context 中容器名为空，尝试从日志兜底提取..."
    # 兜底1：从 setup_workspace 输出中提取（格式：容器: xxx）
    SEG_CTR=$(grep -oP '容器: \K\S+' "${FULL_LOG}" 2>/dev/null | tail -1)
    # 兜底2：从 docker run 命令中提取 --name= 参数
    [ -z "$SEG_CTR" ] && SEG_CTR=$(grep -oP -- '--name[= ]\K[^ ]+' "${FULL_LOG}" 2>/dev/null | tail -1)
    if [ -z "$SEG_CTR" ]; then
        echo "错误：段1未产出容器名且日志兜底失败，终止"
        exit 1
    fi
    echo "  ✓ 从日志兜底提取容器名: ${SEG_CTR}"
    # 验证容器存在
    if ! docker inspect --type=container "${SEG_CTR}" &>/dev/null; then
        echo "错误：兜底容器 ${SEG_CTR} 不存在，终止"
        exit 1
    fi
    # 补写 context 并重新同步 snapshot
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --set container.name=${SEG_CTR} --set container.status=running --json" 2>/dev/null || true
    MOUNT_MODE=$(docker exec "${SEG_CTR}" cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
    if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
        cp "/data/flagos-workspace/${MODEL}/shared/context.yaml" "${CTX_FILE}" 2>/dev/null
    else
        docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "${CTX_FILE}" 2>/dev/null
    fi
    CTX_INFO=$(read_context "${MODEL}" 2>/dev/null) || true
    SEG_ENV=$(echo "$CTX_INFO" | cut -d'|' -f2)
    SEG_LAST=$(echo "$CTX_INFO" | cut -d'|' -f3)
fi

echo "  容器名: ${SEG_CTR}"
echo "  环境类型: ${SEG_ENV}"
echo "  最后完成步骤: ${SEG_LAST}"
echo ""

# ===== 段1越界检测：如果段1执行了步骤4+的操作，回滚 context 中的越界状态 =====
SEG1_OVERFLOW=$(python3 -c "
import yaml, re
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
ledger = ctx.get('workflow_ledger', {}).get('steps', [])
overflow_steps = []
def step_num(s):
    m = re.match(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0
if isinstance(ledger, list):
    for s in ledger:
        if not isinstance(s, dict):
            continue
        step = s.get('step', '')
        status = s.get('status', '')
        if step_num(step) >= 4 and status not in ('pending', ''):
            overflow_steps.append(f'{step}={status}')
elif isinstance(ledger, dict):
    for key, s in ledger.items():
        if not isinstance(s, dict):
            continue
        step = str(s.get('step', key))
        status = s.get('status', '')
        if step_num(step) >= 4 and status not in ('pending', ''):
            overflow_steps.append(f'{step}={status}')
print(','.join(overflow_steps) if overflow_steps else '')
" 2>/dev/null) || SEG1_OVERFLOW=""

if [ -n "${SEG1_OVERFLOW}" ]; then
    echo "  ⚠ 段1越界检测：以下步骤被段1提前执行，将回滚为 pending 状态: ${SEG1_OVERFLOW}"
    # 通过容器内 update_context.py 回滚越界步骤
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --ledger-update 04_quick_accuracy --ledger-status pending --ledger-notes '段1越界回滚' \
        --json" >/dev/null 2>&1 || true
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --ledger-update 05_accuracy_tuning --ledger-status pending --ledger-notes '段1越界回滚' \
        --json" >/dev/null 2>&1 || true
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --ledger-update 06_quick_performance --ledger-status pending --ledger-notes '段1越界回滚' \
        --json" >/dev/null 2>&1 || true
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --ledger-update 07_performance_tuning --ledger-status pending --ledger-notes '段1越界回滚' \
        --json" >/dev/null 2>&1 || true
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --ledger-update 08_release --ledger-status pending --ledger-notes '段1越界回滚' \
        --json" >/dev/null 2>&1 || true
    for STEP_KEY in 09_plugin_install 10_plugin_service_startup 11_plugin_accuracy 12_plugin_performance 13_plugin_release; do
        docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
            --ledger-update ${STEP_KEY} --ledger-status pending --ledger-notes '段1越界回滚' \
            --json" >/dev/null 2>&1 || true
    done
    # 回滚 workflow 状态字段
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --set workflow.accuracy_ok=false --set workflow.performance_ok=false \
        --set workflow.qualified=false --set workflow.all_done=false \
        --json" >/dev/null 2>&1 || true
    # 回滚 timing 中越界步骤的计时
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
        --set-timing steps.quick_accuracy=0 --set-timing steps.accuracy_tuning=0 \
        --set-timing steps.quick_performance=0 --set-timing steps.performance_tuning=0 \
        --set-timing steps.release=0 \
        --json" >/dev/null 2>&1 || true
    # 清理越位执行产生的实际数据（算子配置、评测结果等）
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/rollback_overflow.py \
        --overflow-from 4 --preserve-step3-disabled-ops" 2>/dev/null || {
        echo "  ⚠ rollback_overflow.py 执行失败，手动清理关键文件"
        docker exec "${SEG_CTR}" bash -c "rm -f /root/flaggems_ops_control.json /flagos-workspace/results/operator_config.json" 2>/dev/null || true
    }
    echo "  ✓ 越位执行产生的数据已清理"
    # 同步回滚后的 context 到宿主机快照
    MOUNT_MODE=$(docker exec "${SEG_CTR}" cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
    if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
        cp "/data/flagos-workspace/${MODEL}/shared/context.yaml" "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml" 2>/dev/null || \
            echo "  ⚠ 回滚后 context 同步失败（shared/context.yaml 可能不存在），继续执行"
    else
        docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml" 2>/dev/null || \
            echo "  ⚠ 回滚后 context 同步失败（容器可能已停止），继续执行"
    fi
    echo "  ✓ 越界状态已回滚，段2 将从步骤4重新开始"
    SEG_LAST="03_service_startup"
fi

# 从 context_snapshot 提取关键参数，注入段2 prompt 减少 Claude 重读文件时间
SEG2_CTX_SUMMARY=$(python3 -c "
import yaml
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
svc = ctx.get('service', {})
rt = ctx.get('runtime', {})
gpu = ctx.get('gpu', {})
env = ctx.get('environment', {})
mdl = ctx.get('model', {})
fc = ctx.get('flaggems_control', {})
print(f'''- 模型路径(容器内): {mdl.get('container_path','')}
- GPU: {gpu.get('count','')}x {gpu.get('type','')}, {gpu.get('visible_devices_env','CUDA_VISIBLE_DEVICES')}={rt.get('cuda_visible_devices', gpu.get('cuda_visible_devices',''))}
- TP: {rt.get('tp_size','')}
- 端口: {svc.get('port','')}
- FlagGems 算子数: {svc.get('enable_oplist_count','')}
- gems_txt_path: {env.get('flaggems_txt_path', svc.get('gems_txt_path',''))}
- max_model_len: {svc.get('max_model_len','')}
- thinking_model: {rt.get('thinking_model','')}
- integration_type: {fc.get('integration_type','')}
- disabled_ops: {ctx.get('optimization',{}).get('disabled_ops',[])}
- service_ok: {ctx.get('workflow',{}).get('service_ok','')}''')
" 2>/dev/null || echo "  (context 摘要提取失败)")

# 检查 service_ok：FlagGems 崩溃且无法恢复时跳过精度/性能评测，直接到发布
SERVICE_OK=$(python3 -c "
import yaml
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
print(ctx.get('workflow',{}).get('service_ok', True))
" 2>/dev/null) || SERVICE_OK="True"

# 兜底：如果 service.inference_verified=true 且 service.mode=flaggems，但 workflow.service_ok 未设置，自动修正
if [ "${SERVICE_OK}" = "False" ]; then
    FALLBACK_CHECK=$(python3 -c "
import yaml
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
svc = ctx.get('service', {})
if svc.get('inference_verified') == True and svc.get('mode') == 'flaggems':
    ctx.setdefault('workflow', {})['service_ok'] = True
    with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml', 'w') as f:
        yaml.dump(ctx, f, default_flow_style=False, allow_unicode=True)
    print('True')
else:
    print('False')
" 2>/dev/null) || FALLBACK_CHECK="False"
    if [ "${FALLBACK_CHECK}" = "True" ]; then
        echo "⚠ 兜底修正：service.inference_verified=true + mode=flaggems，但 workflow.service_ok=false，自动修正为 true"
        SERVICE_OK="True"
    fi
fi

SKIP_SEG2=false
IS_NATIVE=false
if [ "${SERVICE_OK}" = "False" ]; then
    echo "⚠ service_ok=false（FlagGems 不可用），跳过段2（步骤4-7），直接进入段3发布"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 段2 跳过：service_ok=false"
    SKIP_SEG2=true
fi
if [ "${SEG_ENV}" = "native" ]; then
    IS_NATIVE=true
    echo "ℹ env_type=native（纯原生环境），段2仅执行步骤4/6，跳过步骤5/7（无 FlagGems 算子调优）"
fi

if [ "${SKIP_SEG2}" = "false" ]; then
# ===== 段2: 4/5/6/7 (精度评测 + 精度调优 + 性能评测 + 性能调优) =====
PROMPT_SEG2="容器名: ${SEG_CTR}，模型名: ${MODEL}，env_type: ${SEG_ENV}

**变量定义（后续命令中直接使用）**：CONTAINER=${SEG_CTR}
${COMMON_TOKENS}

按 CLAUDE.md 工作流定义执行步骤4精度评测、步骤5精度算子调优（如需）、步骤6性能评测、步骤7性能算子调优（如需）。

**前段状态（段1已完成，无需验证）**：
- 步骤1/2/3 已在上一段全部完成，容器 ${SEG_CTR} 已就绪，工具脚本已部署
- env_type=${SEG_ENV}，最后完成步骤: ${SEG_LAST}
- context.yaml 中 workflow_ledger 的步骤3状态可能未更新（段1的已知问题），但步骤3确实已完成，直接从步骤4开始
- **禁止**回头检查或重做步骤1/2/3，**禁止**查找 execution_plan.md（不存在）

**关键参数（从 context.yaml 提取，无需重新读取文件）**：
${SEG2_CTX_SUMMARY}

**context.yaml 更新方式**：使用容器内 update_context.py 工具（已部署到 /flagos-workspace/scripts/），避免手写 Python 脚本：
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --set key.path=value --json\"
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --ledger-update 04_quick_accuracy --ledger-status success --ledger-notes '...'\"

**步骤编号（严格遵守，输出 [步骤X] 时必须使用以下编号）**：
- [步骤4] 精度评测（GPQA Diamond）
- [步骤5] 精度算子调优（条件触发：accuracy_ok=false 时执行）
- [步骤6] 性能评测（benchmark）
- [步骤7] 性能算子调优（条件触发：performance_ok=false 时执行）

**执行前**：
0. 检查 workflow.service_ok：如果为 false，输出 \"[段2] service_ok=false，FlagGems 不可用，跳过步骤4-7\" 后立即停止
1. 读取容器内 /flagos-workspace/shared/context.yaml 获取模型路径、GPU 配置、FlagGems 状态等关键参数
2. 读取 skills/flagos-eval-comprehensive/SKILL.md 了解精度评测工具用法
3. 读取 skills/flagos-performance-testing/SKILL.md 了解性能评测工具用法
4. 读取 skills/flagos-operator-replacement/SKILL.md 了解算子调优工具用法（仅在步骤5/7需要时读取）

**算子调优**：
- 步骤4完成后如 accuracy_ok=false → 立即执行步骤5（5完成后再进入6）
- 步骤5精度算子调优采用**累积禁用**策略：第1轮禁用组A，第2轮禁用组A+B，第3轮禁用组A+B+C（每轮在上一轮基础上追加禁用，而非独立禁用单组）
- 步骤5每轮使用 diagnose_ops.py 输出的 cumulative_test_env.control_file 白名单写入 /root/flaggems_ops_control.json + FLAGGEMS_CONTROL_MODE=only_enable，通过 start_service.sh 启动服务（与步骤7性能调优的 operator_search.py 使用相同的白名单控制路径，**不使用** toggle_flaggems.py --action modify-enable --disabled-ops）
- 步骤6完成后如 performance_ok=false → 执行步骤7（elimination 逐删策略）
- 调优后产出 V3 结果（flagos_optimized.json），更新 context.yaml

**V2 服务启动时保留已禁用算子**：
- 如果 context.yaml 中 optimization.disabled_ops 非空（步骤3算子诊断已禁用部分算子），V2 启动前必须写入白名单控制文件：将启用算子写入 /root/flaggems_ops_control.json（{"include": [启用算子]}），start_service.sh 会自动从控制文件推断 FLAGGEMS_CONTROL_MODE=only_enable
- 禁止使用 --action enable（会重置为全量开启）
- 步骤5/7 的算子调优在此基础上累加禁用

**算子调优硬性约束**：
- 步骤7性能算子调优**必须**通过容器内 operator_search.py run 执行完整自动化循环
- **禁止**手动拼 toggle_flaggems.py + benchmark_runner.py 循环，**禁止**手动分组禁用算子
- 调用方式：
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/operator_search.py run \\
    --state-path /flagos-workspace/results/operator_config.json \\
    --perf-config /flagos-workspace/scripts/config/perf_config.yaml \\
    --service-startup-cmd 'bash /flagos-workspace/scripts/start_service.sh' \\
    --max-rounds 50\"
- operator_search.py 已封装 next→配置→重启→benchmark→update 全流程，含 GPU 显存释放验证和可用性前置检查
- 步骤5精度算子调优：通过 diagnose_ops.py accuracy-groups 获取分组和累积白名单，每轮写控制文件 + 重启服务 + fast_gpqa.py 评测

**进度输出**：步骤开始/完成时输出 [步骤X] 标记，关键命令后输出 ✓/✗ 结果摘要。

**Issue 强制规则**（达到条件必须生成 issue 文件）：
GITHUB_TOKEN=${GITHUB_TOKEN}（issue 提交时通过 docker exec -e 传入）。
1. 步骤4/6 评测中服务崩溃 → 调用 issue_reporter.py full --type operator-crash + 追加 logs/issues_startup.log
2. 步骤4 精度偏差 >5% → 必须按顺序完成三步：
   ① 标记 workflow.accuracy_ok=false
   ② 调用 issue_reporter.py full --type accuracy-degraded（写入 results/issue_accuracy-degraded_*.md）
   ③ 追加写入 logs/issues_accuracy.log
3. 步骤5 精度调优禁用了算子 → 调用 issue_reporter.py full --type accuracy-degraded（含禁用算子列表和原因）
4. 步骤6 任一并发级别 V2/V1 < 80% → 必须按顺序完成三步：
   ① 标记 workflow.performance_ok=false
   ② 调用 issue_reporter.py full --type performance-degraded（写入 results/issue_performance-degraded_*.md）
   ③ 追加写入 logs/issues_performance.log
5. 步骤7 性能调优禁用了算子 → 调用 issue_reporter.py full --type performance-degraded（含禁用算子列表和原因）
- 所有 issue 文件路径写入 context.yaml 的 issues.submitted[]
- issue_reporter.py 调用模板：
  docker exec -e GITHUB_TOKEN=${GITHUB_TOKEN} \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \\
    --type <issue类型> --context-yaml /flagos-workspace/shared/context.yaml \\
    --repo flagos-ai/FlagGems --output-dir /flagos-workspace/results/ --json\"

完成后通过 docker cp 同步 context 到宿主机：
  docker cp ${SEG_CTR}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
（如果 mount_mode=mounted，也可：cp /data/flagos-workspace/${MODEL}/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml）

**⚠ 段边界（硬性约束）**：本段只执行步骤4/5/6/7，最后一个步骤完成并同步 context_snapshot.yaml 后必须立即停止。
**绝对禁止**进入步骤8（发布）。步骤8由下一段独立会话执行。即使所有步骤都已完成，也不得执行任何发布操作（Harbor/ModelScope/HuggingFace）。
禁止更新 ledger 中 08_release 及之后步骤的状态。违反此约束将导致重复发布。
完成标志：输出 \"[段2] 步骤4/5/6/7全部完成，context 已同步\" 后停止所有操作。

**⚠ 单模型约束（硬性）**：本会话只处理一个模型：${MODEL}，容器：${SEG_CTR}。
- **绝对禁止**读取 task.txt、tasks.txt、tasks1.txt 或任何批量任务文件
- **绝对禁止**操作其他模型的容器（只允许操作 ${SEG_CTR}）
- 步骤4精度评测包含 V1（native）和 V2（FlagGems）两轮，V1 完成后必须继续 V2，不得中断去做其他事情
- 步骤4完成的标志是：V1 和 V2 结果都已写入 results/，且 ledger 04_quick_accuracy 状态更新为 success/failed
- 步骤4完成后必须继续执行步骤5（如需）→步骤6→步骤7（如需），不得提前结束"

# native 场景追加硬性约束：只执行步骤4/6，跳过5/7
if [ "${IS_NATIVE}" = "true" ]; then
    PROMPT_SEG2="${PROMPT_SEG2}

**⚠ native 场景硬性约束（最高优先级）**：
env_type=native 表示纯原生环境，无 FlagGems。本段**只执行步骤4（精度评测）和步骤6（性能评测）**。
- **绝对禁止**执行步骤5（精度算子调优）和步骤7（性能算子调优）
- 即使 accuracy_ok=false 或 performance_ok=false，也**不触发**算子调优
- 步骤5/7 直接标记为 skipped（skip_reason='native 场景无 FlagGems'）
- 精度评测只有 V1（native 模式），无 V2 对比
- 性能评测只有 V1（native 模式），无 V2 对比"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  段2/3  精度评测 + 精度调优 + 性能评测 + 性能调优           ║"
echo "║         (步骤 4→[5]→6→[7])                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
SEG2_START_TS=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 段2 开始"
mkdir -p "${LOG_DIR}"
# 隐藏批量任务文件，防止 Claude 读取后误操作其他模型
for tf in task.txt tasks.txt tasks1.txt; do
    [ -f "${SCRIPT_DIR}/../${tf}" ] && mv "${SCRIPT_DIR}/../${tf}" "${SCRIPT_DIR}/../.${tf}.hidden" 2>/dev/null || true
done
claude -p "${PROMPT_SEG2}" \
    --permission-mode auto \
    --output-format stream-json \
    --verbose \
    --debug-file "${DEBUG_FILE}.seg2" \
    --max-turns 250 \
    2>&1 | tee -a "${LOG_FILE}" \
         | tee >(python3 "${SCRIPT_DIR}/stream_to_debug_log.py" >> "${FULL_LOG}") \
         | python3 "${SCRIPT_DIR}/stream_filter.py" --pipeline-log "${PIPELINE_LOG}" --terminal-log "${TERMINAL_LOG}" --start-step 4 --cost-file "${LOG_DIR}/seg2_cost.txt" --load-durations "${LOG_DIR}/seg1_durations.json" --durations-file "${LOG_DIR}/seg2_durations.json" ${FILTER_FLAGS} || true

# 段间检查
SEG2_END_TS=$(date +%s)
SEG2_ELAPSED=$(( SEG2_END_TS - SEG2_START_TS ))
SEG2_MIN=$(( SEG2_ELAPSED / 60 ))
SEG2_SEC=$(( SEG2_ELAPSED % 60 ))
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  段2 完成 — 耗时 ${SEG2_MIN}m ${SEG2_SEC}s                                     │"
echo "└──────────────────────────────────────────────────────────────┘"

# 强制同步 context（不依赖 Claude 会话是否已同步，防止读到段1的旧数据）
CTX_FILE="/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml"
SHARED_CTX="/data/flagos-workspace/${MODEL}/shared/context.yaml"
if [ -f "${SHARED_CTX}" ]; then
    cp "${SHARED_CTX}" "${CTX_FILE}"
elif [ -n "${SEG_CTR}" ] && docker inspect --type=container "${SEG_CTR}" &>/dev/null; then
    docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "${CTX_FILE}" 2>/dev/null || true
fi
CTX_INFO=$(read_context "${MODEL}" 2>/dev/null) || {
    echo "  ⚠ context_snapshot.yaml 同步后仍无法解析，终止"
    exit 1
}
SEG_CTR=$(echo "$CTX_INFO" | cut -d'|' -f1)
echo "  容器名: ${SEG_CTR}"

# 段2 完成性校验：步骤4（精度评测）必须有明确结果
echo "  段2 ledger 状态:"
print_ledger_summary "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml"

SEG2_STATUS=$(check_seg_complete "${MODEL}" "04" "accuracy")
SEG2_PERF_STATUS=$(check_seg_complete "${MODEL}" "06" "performance")
if [ "$SEG2_STATUS" != "complete" ] || [ "$SEG2_PERF_STATUS" != "complete" ]; then
    echo ""
    if [ "$SEG2_STATUS" != "complete" ]; then
        echo "  ⚠ 段2 未正常完成（步骤4精度评测无结果），重新执行段2..."
    else
        echo "  ⚠ 段2 未正常完成（步骤6性能评测无结果），重新执行段2..."
    fi
    SEG2_START_TS=$(date +%s)
    claude -p "${PROMPT_SEG2}" \
        --permission-mode auto \
        --output-format stream-json \
        --verbose \
        --debug-file "${DEBUG_FILE}.seg2_retry" \
        --max-turns 250 \
        2>&1 | tee -a "${LOG_FILE}" \
             | tee >(python3 "${SCRIPT_DIR}/stream_to_debug_log.py" >> "${FULL_LOG}") \
             | python3 "${SCRIPT_DIR}/stream_filter.py" --pipeline-log "${PIPELINE_LOG}" --terminal-log "${TERMINAL_LOG}" --start-step 4 --cost-file "${LOG_DIR}/seg2_retry_cost.txt" --load-durations "${LOG_DIR}/seg1_durations.json" --durations-file "${LOG_DIR}/seg2_durations.json" ${FILTER_FLAGS} || true
    # 重试后再次同步 context
    CTX_FILE="/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml"
    SHARED_CTX="/data/flagos-workspace/${MODEL}/shared/context.yaml"
    if [ -f "${SHARED_CTX}" ]; then
        cp "${SHARED_CTX}" "${CTX_FILE}"
    elif [ -n "${SEG_CTR}" ] && docker inspect --type=container "${SEG_CTR}" &>/dev/null; then
        docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "${CTX_FILE}" 2>/dev/null || true
    fi
    echo "  段2 重试后 ledger 状态:"
    print_ledger_summary "${CTX_FILE}"
    # 重试后更新 SEG_CTR（防止容器名变更导致后续段使用旧值）
    CTX_INFO=$(read_context "${MODEL}" 2>/dev/null) || true
    [ -n "$(echo "$CTX_INFO" | cut -d'|' -f1)" ] && SEG_CTR=$(echo "$CTX_INFO" | cut -d'|' -f1)

    # 二次校验：检查步骤6（性能评测）是否也完成
    SEG2_STATUS2=$(check_seg_complete "${MODEL}" "06" "performance")
    if [ "$SEG2_STATUS2" != "complete" ]; then
        echo "  ⚠ 段2 重试后步骤6（性能评测）仍未完成，再次重试..."
        claude -p "${PROMPT_SEG2}" \
            --permission-mode auto \
            --output-format stream-json \
            --verbose \
            --debug-file "${DEBUG_FILE}.seg2_retry2" \
            --max-turns 250 \
            2>&1 | tee -a "${LOG_FILE}" \
                 | tee >(python3 "${SCRIPT_DIR}/stream_to_debug_log.py" >> "${FULL_LOG}") \
                 | python3 "${SCRIPT_DIR}/stream_filter.py" --pipeline-log "${PIPELINE_LOG}" --terminal-log "${TERMINAL_LOG}" --start-step 4 --cost-file "${LOG_DIR}/seg2_retry2_cost.txt" --load-durations "${LOG_DIR}/seg1_durations.json" --durations-file "${LOG_DIR}/seg2_durations.json" ${FILTER_FLAGS} || true
        # 再次同步 context
        if [ -f "${SHARED_CTX}" ]; then
            cp "${SHARED_CTX}" "${CTX_FILE}"
        elif [ -n "${SEG_CTR}" ] && docker inspect --type=container "${SEG_CTR}" &>/dev/null; then
            docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "${CTX_FILE}" 2>/dev/null || true
        fi
        echo "  段2 二次重试后 ledger 状态:"
        print_ledger_summary "${CTX_FILE}"
        CTX_INFO=$(read_context "${MODEL}" 2>/dev/null) || true
        [ -n "$(echo "$CTX_INFO" | cut -d'|' -f1)" ] && SEG_CTR=$(echo "$CTX_INFO" | cut -d'|' -f1)
    fi
fi

# ===== 段2越界检测：如果段2执行了步骤8+的操作，回滚 context 中的越界状态 =====
SEG2_OVERFLOW=$(python3 -c "
import yaml, re
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
ledger = ctx.get('workflow_ledger', {}).get('steps', [])
overflow_steps = []
def step_num(s):
    m = re.match(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0
if isinstance(ledger, list):
    for s in ledger:
        if not isinstance(s, dict):
            continue
        step = s.get('step', '')
        status = s.get('status', '')
        if step_num(step) >= 8 and status not in ('pending', ''):
            overflow_steps.append(f'{step}={status}')
elif isinstance(ledger, dict):
    for key, s in ledger.items():
        if not isinstance(s, dict):
            continue
        step = str(s.get('step', key))
        status = s.get('status', '')
        if step_num(step) >= 8 and status not in ('pending', ''):
            overflow_steps.append(f'{step}={status}')
print(','.join(overflow_steps) if overflow_steps else '')
" 2>/dev/null) || SEG2_OVERFLOW=""

if [ -n "${SEG2_OVERFLOW}" ]; then
    echo "  ⚠ 段2越界检测：以下步骤被段2提前执行，将回滚为 pending 状态: ${SEG2_OVERFLOW}"
    for STEP_KEY in 08_release 09_plugin_install 10_plugin_service_startup 11_plugin_accuracy 12_plugin_performance 13_plugin_release; do
        docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
            --ledger-update ${STEP_KEY} --ledger-status pending --ledger-notes '段2越界回滚' \
            --json" >/dev/null 2>&1 || true
    done
    # 清理越位执行产生的实际数据（发布状态、Plugin 配置等）
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/rollback_overflow.py \
        --overflow-from 8" 2>/dev/null || \
        echo "  ⚠ rollback_overflow.py 执行失败，继续执行"
    echo "  ✓ 越位执行产生的数据已清理"
    # 同步回滚后的 context 到宿主机
    MOUNT_MODE=$(docker exec "${SEG_CTR}" cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
    if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
        cp "/data/flagos-workspace/${MODEL}/shared/context.yaml" "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml" 2>/dev/null || \
            echo "  ⚠ 回滚后 context 同步失败（shared/context.yaml 可能不存在），继续执行"
    else
        docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml" 2>/dev/null || \
            echo "  ⚠ 回滚后 context 同步失败（容器可能已停止），继续执行"
    fi
    echo "  ✓ 越界状态已回滚，段3 将从步骤8重新开始"
fi
echo ""

# 恢复隐藏的批量任务文件
for tf in task.txt tasks.txt tasks1.txt; do
    [ -f "${SCRIPT_DIR}/../.${tf}.hidden" ] && mv "${SCRIPT_DIR}/../.${tf}.hidden" "${SCRIPT_DIR}/../${tf}" 2>/dev/null || true
done

fi  # end SKIP_SEG2 check

# ===== 段2→段3 过渡：兜底计算 qualified 状态 =====
# 如果段2完成后 service_ok/accuracy_ok/performance_ok 都为 true 但 qualified 未设置，自动修正
python3 -c "
import yaml
ctx_path = '/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml'
try:
    with open(ctx_path) as f:
        ctx = yaml.safe_load(f)
    wf = ctx.get('workflow', {})
    if 'qualified' not in wf and wf.get('service_ok') and wf.get('accuracy_ok') and wf.get('performance_ok'):
        wf['qualified'] = True
        ctx['workflow'] = wf
        with open(ctx_path, 'w') as f:
            yaml.dump(ctx, f, default_flow_style=False, allow_unicode=True)
        print('  ✓ 兜底设置 workflow.qualified=true（service_ok/accuracy_ok/performance_ok 均为 true）')
except: pass
" 2>/dev/null || true

# 从 context_snapshot 提取段3所需的关键参数
SEG3_CTX_SUMMARY=$(python3 -c "
import yaml
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
wf = ctx.get('workflow', {})
ev = ctx.get('eval', {})
perf = ctx.get('performance', {})
svc = ctx.get('service', {})
gpu = ctx.get('gpu', {})
mdl = ctx.get('model', {})
ws = ctx.get('workspace', {})
print(f'''- 模型路径(容器内): {mdl.get('container_path','')}
- GPU: {gpu.get('count','')}x {gpu.get('type','')}
- mount_mode: {ws.get('mount_mode','')}
- service_ok: {wf.get('service_ok','')}
- accuracy_ok: {wf.get('accuracy_ok','')}
- performance_ok: {wf.get('performance_ok','')}
- qualified: {wf.get('qualified','')}
- V1 精度: {ev.get('v1_score','')}%, V2 精度: {ev.get('v2_score','')}%, 偏差: {ev.get('accuracy_diff','')}%
- 性能 ratio: {perf.get('min_ratio','')}%
- 宿主机路径: {ws.get('host_path','')}''')
" 2>/dev/null || echo "  (context 摘要提取失败)")

# ===== 段3: 8 (打包发布) =====
PROMPT_SEG3="容器名: ${SEG_CTR}，模型名: ${MODEL}

**变量定义（后续命令中直接使用）**：CONTAINER=${SEG_CTR}
${COMMON_TOKENS}

按 CLAUDE.md 工作流定义执行步骤8自动发布。

**前段状态（段1+段2已完成，无需验证）**：
- 步骤1/2/3/4/5/6/7 已在前两段全部完成（5/7为条件触发，可能跳过）
- 容器 ${SEG_CTR} 已就绪，评测结果已写入 results/ 目录
- context.yaml 中 workflow_ledger 的部分步骤状态可能未更新（已知问题），但前段步骤确实已完成
- **禁止**回头检查或重做步骤1-7，直接执行步骤8
- **禁止**修改 workflow_ledger 中步骤1-7和步骤9-13的状态。只允许更新步骤8（08_release）的 ledger 条目
- **禁止**使用 --set 'workflow_ledger.steps...' 的 dot notation 写入 ledger，必须使用 --ledger-update API

**关键参数（从 context.yaml 提取，无需重新读取文件）**：
${SEG3_CTX_SUMMARY}

**context.yaml 更新方式**：使用容器内 update_context.py 工具：
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --set key.path=value --json\"

**步骤编号（严格遵守）**：本段只有 [步骤8] 自动发布，输出进度标记时必须使用 [步骤8]，不要使用其他编号。

**执行前**：读取容器内 /flagos-workspace/shared/context.yaml 获取 workflow 状态（accuracy_ok、performance_ok、service_ok）和模型/GPU 信息。
如果 accuracy_ok 或 performance_ok 为 false，发布为私有镜像（qualified=false）。
**进度输出**：步骤开始/完成时输出 [步骤8] 标记，关键命令后输出 ✓/✗ 结果摘要。

**发布前同步 context 到宿主机**（发布工具从宿主机路径读取）：
  docker cp ${SEG_CTR}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
（如果 mount_mode=mounted，也可：cp /data/flagos-workspace/${MODEL}/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml）
发布工具: python3 skills/flagos-release/tools/main.py --from-context /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
完成后通过 docker cp 回传最终 context：
  docker cp ${SEG_CTR}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_final.yaml

全流程结束后输出完整的 FlagOS 迁移报告（含精度、性能、发布信息、耗时统计、问题记录摘要）。

**完成标志**：输出最终迁移报告后，输出 \"[段3] 步骤8完成，流程结束\" 后停止所有操作。
**绝对禁止**进入步骤9-13（Plugin 流程）。Plugin 流程由下一段独立会话执行。
禁止更新 ledger 中 09_plugin_install 及之后步骤的状态。违反此约束将导致重复执行。"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  段3/3  打包发布  (步骤 8)                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
SEG3_START_TS=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 段3 开始"

# ===== 幂等检查：步骤8 是否已在段2中被越界完成 =====
SEG3_RELEASE_DONE=$(python3 -c "
import yaml
try:
    with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
        ctx = yaml.safe_load(f)
    ledger = ctx.get('workflow_ledger', {}).get('steps', [])
    if isinstance(ledger, list):
        for s in ledger:
            if isinstance(s, dict) and s.get('step','').startswith('08') and s.get('status') == 'success':
                print('yes'); exit()
    elif isinstance(ledger, dict):
        for k, s in ledger.items():
            if isinstance(s, dict) and str(s.get('step', k)).startswith('08') and s.get('status') == 'success':
                print('yes'); exit()
except: pass
print('no')
" 2>/dev/null) || SEG3_RELEASE_DONE="no"

if [ "${SEG3_RELEASE_DONE}" = "yes" ]; then
    echo "  ✓ 步骤8 已在前段完成（ledger 08_release=success），跳过段3 Claude 调用"
    SEG3_ELAPSED=0
    SEG3_MIN=0
    SEG3_SEC=0
else
mkdir -p "${LOG_DIR}"
claude -p "${PROMPT_SEG3}" \
    --permission-mode auto \
    --output-format stream-json \
    --verbose \
    --debug-file "${DEBUG_FILE}.seg3" \
    --max-turns 100 \
    2>&1 | tee -a "${LOG_FILE}" \
         | tee >(python3 "${SCRIPT_DIR}/stream_to_debug_log.py" >> "${FULL_LOG}") \
         | python3 "${SCRIPT_DIR}/stream_filter.py" --pipeline-log "${PIPELINE_LOG}" --terminal-log "${TERMINAL_LOG}" --start-step 8 --cost-file "${LOG_DIR}/seg3_cost.txt" --load-durations "${LOG_DIR}/seg2_durations.json" --durations-file "${LOG_DIR}/seg3_durations.json" ${FILTER_FLAGS} || true

SEG3_END_TS=$(date +%s)
SEG3_ELAPSED=$(( SEG3_END_TS - SEG3_START_TS ))
SEG3_MIN=$(( SEG3_ELAPSED / 60 ))
SEG3_SEC=$(( SEG3_ELAPSED % 60 ))
fi

# ===== 段3越界检测：如果段3执行了步骤9+的操作，回滚 context 中的越界状态 =====
SEG3_OVERFLOW=$(python3 -c "
import yaml, re
with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
    ctx = yaml.safe_load(f)
ledger = ctx.get('workflow_ledger', {}).get('steps', [])
overflow_steps = []
def step_num(s):
    m = re.match(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0
if isinstance(ledger, list):
    for s in ledger:
        if not isinstance(s, dict):
            continue
        step = s.get('step', '')
        status = s.get('status', '')
        if step_num(step) >= 9 and status not in ('pending', ''):
            overflow_steps.append(f'{step}={status}')
elif isinstance(ledger, dict):
    for key, s in ledger.items():
        if not isinstance(s, dict):
            continue
        step = str(s.get('step', key))
        status = s.get('status', '')
        if step_num(step) >= 9 and status not in ('pending', ''):
            overflow_steps.append(f'{step}={status}')
print(','.join(overflow_steps) if overflow_steps else '')
" 2>/dev/null) || SEG3_OVERFLOW=""

if [ -n "${SEG3_OVERFLOW}" ]; then
    echo "  ⚠ 段3越界检测：以下步骤被段3提前执行，将回滚为 pending 状态: ${SEG3_OVERFLOW}"
    for STEP_KEY in 09_plugin_install 10_plugin_service_startup 11_plugin_accuracy 12_plugin_performance 13_plugin_release; do
        docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/update_context.py \
            --ledger-update ${STEP_KEY} --ledger-status pending --ledger-notes '段3越界回滚' \
            --json" >/dev/null 2>&1 || true
    done
    # 清理越位执行产生的实际数据（Plugin 配置等）
    docker exec "${SEG_CTR}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/rollback_overflow.py \
        --overflow-from 9" 2>/dev/null || \
        echo "  ⚠ rollback_overflow.py 执行失败，继续执行"
    echo "  ✓ 越位执行产生的数据已清理"
    MOUNT_MODE=$(docker exec "${SEG_CTR}" cat /flagos-workspace/.mount_mode 2>/dev/null || echo "internal")
    if [ "$MOUNT_MODE" = "mounted" ] || [ "$MOUNT_MODE" = "symlink" ]; then
        cp "/data/flagos-workspace/${MODEL}/shared/context.yaml" "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml" 2>/dev/null || \
            echo "  ⚠ 回滚后 context 同步失败（shared/context.yaml 可能不存在），继续执行"
    else
        docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml" 2>/dev/null || \
            echo "  ⚠ 回滚后 context 同步失败（容器可能已停止），继续执行"
    fi
    echo "  ✓ 越界状态已回滚，段4 将从步骤9重新开始"
fi

# ===== 段间检查：是否触发 plugin 流程（步骤 9-13） =====
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  段3 完成 — 检查是否触发 Plugin 流程                         │"
echo "└──────────────────────────────────────────────────────────────┘"

# 强制同步 context（不依赖 Claude 会话是否已同步）
CTX_FILE="/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml"
SHARED_CTX="/data/flagos-workspace/${MODEL}/shared/context.yaml"
if [ -f "${SHARED_CTX}" ]; then
    cp "${SHARED_CTX}" "${CTX_FILE}"
elif [ -n "${SEG_CTR}" ] && docker inspect --type=container "${SEG_CTR}" &>/dev/null; then
    docker cp "${SEG_CTR}:/flagos-workspace/shared/context.yaml" "${CTX_FILE}" 2>/dev/null || true
fi
CTX_INFO=$(read_context "${MODEL}" 2>/dev/null) || { echo "错误：段3未更新 context_snapshot.yaml 或解析失败，终止"; exit 1; }
SEG_CTR=$(echo "$CTX_INFO" | cut -d'|' -f1)
echo "  容器名: ${SEG_CTR}"

# 读取 qualified 状态（含兜底计算：如果 qualified 字段未设置但三个条件都满足，自动判定为 True）
QUALIFIED=$(python3 -c "
import yaml
try:
    with open('${CTX_FILE}') as f:
        ctx = yaml.safe_load(f)
    wf = ctx.get('workflow', {})
    qualified = wf.get('qualified', False)
    if not qualified:
        # 兜底：独立计算 qualified = service_ok AND accuracy_ok AND performance_ok
        if wf.get('service_ok') and wf.get('accuracy_ok') and wf.get('performance_ok'):
            qualified = True
    print(qualified)
except: print('False')
" 2>/dev/null || echo "False")

SEG4_ELAPSED=0
SEG4_MIN=0
SEG4_SEC=0

if [ "${QUALIFIED}" = "True" ]; then
    # 提取段4所需参数
    SEG4_CTX_SUMMARY=$(python3 -c "
import yaml
with open('${CTX_FILE}') as f:
    ctx = yaml.safe_load(f)
ctr = ctx.get('container', {})
model = ctx.get('model', {})
gpu = ctx.get('gpu', {})
ws = ctx.get('workspace', {})
wf = ctx.get('workflow', {})
ev = ctx.get('eval', {})
perf = ctx.get('performance', {})
rt = ctx.get('runtime', {})
env = ctx.get('environment', {})
image = ctx.get('image', {})
opt = ctx.get('optimization', {})
print(f'''- container_name: {ctr.get('name','')}
- model_name: {model.get('name','')}
- model_container_path: {model.get('container_path','')}
- env_type: {env.get('env_type','')}
- gpu_vendor: {gpu.get('vendor','')}
- gpu_count: {gpu.get('count','')}
- gpu_type: {gpu.get('type','')}
- tp_size: {rt.get('tp_size','')}
- service_port: {ctx.get('service',{}).get('port', 8000)}
- v1_score: {ev.get('v1_score','')}
- v2_score: {ev.get('v2_score','')}
- accuracy_ok: {wf.get('accuracy_ok','')}
- performance_ok: {wf.get('performance_ok','')}
- config_persisted: {wf.get('config_persisted', False)}
- flagos_blacklist: {opt.get('flagos_blacklist', [])}
- disabled_ops: {opt.get('disabled_ops', [])}
- image_registry_url: {image.get('registry_url','')}
- host_path: {ws.get('host_path','')}''')
" 2>/dev/null || echo "  (context 摘要提取失败)")

    # ===== 段4: 9-13 (Plugin 验证 + 发布) =====
    PROMPT_SEG4="容器名: ${SEG_CTR}，模型名: ${MODEL}

**变量定义（后续命令中直接使用）**：CONTAINER=${SEG_CTR}
${COMMON_TOKENS}

按 CLAUDE.md 工作流定义执行步骤 9-13 Plugin 验证流程。

**前段状态（段1+段2+段3已完成，无需验证）**：
- 步骤 1-8 已在前三段全部完成
- 容器 ${SEG_CTR} 已就绪，主流程发布已完成
- workflow.qualified=true（已由 shell 层验证）
- **禁止**回头检查或重做步骤 1-8，直接执行步骤 9

**关键参数（从 context.yaml 提取）**：
${SEG4_CTX_SUMMARY}

**context.yaml 更新方式**：使用容器内 update_context.py 工具：
  docker exec \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/update_context.py --set key.path=value --json\"

**步骤编号（严格遵守）**：
- [步骤9] Plugin 安装
- [步骤10] Plugin 服务启动
- [步骤11] Plugin 精度评测
- [步骤12] Plugin 性能评测
- [步骤13] Plugin 发布

**执行前**：
1. 读取容器内 context.yaml 获取已达标算子集、GPU 配置、V1 基线结果
2. 读取 skills/flagos-plugin-install/SKILL.md 了解 plugin 安装流程
3. 读取 skills/flagos-service-startup/SKILL.md（步骤10复用）
4. 读取 skills/flagos-eval-comprehensive/SKILL.md（步骤11复用）
5. 读取 skills/flagos-performance-testing/SKILL.md（步骤12复用）

**Plugin 流程特殊规则**：
- 步骤 9 安装失败 → issue_reporter.py --type plugin-error --repo flagos-ai/vllm-plugin-FL → 停止任务
- 步骤 10 服务崩溃 → issue_reporter.py --type plugin-error --repo flagos-ai/vllm-plugin-FL → 停止任务
- 步骤 11/12 不达标 → 写 issue 到 flagos-ai/vllm-plugin-FL，继续（不调优）
- 步骤 13 触发条件：plugin_workflow.accuracy_ok=true AND performance_ok=true
- 所有 issue 提交到 flagos-ai/vllm-plugin-FL（非 FlagGems）
- 算子集复用主流程已达标版本，不重新调优
- 启动环境变量：USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true + 已有 blacklist
- 如果 disabled_ops 非空，启动前必须写入白名单控制文件 /root/flaggems_ops_control.json（{"include": [启用算子]}），start_service.sh 自动推断 FLAGGEMS_CONTROL_MODE=only_enable

**进度输出**：步骤开始/完成时输出 [步骤N] 标记，关键命令后输出 ✓/✗ 结果摘要。

- Issue 模板：
  docker exec -e GITHUB_TOKEN=${GITHUB_TOKEN} \${CONTAINER} bash -c \"PATH=/opt/conda/bin:\\\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \\
    --type plugin-error --context-yaml /flagos-workspace/shared/context.yaml \\
    --repo flagos-ai/vllm-plugin-FL --output-dir /flagos-workspace/results/ --json\"

**步骤 13 发布**：
发布前同步 context 到宿主机：
  docker cp ${SEG_CTR}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml
发布工具: python3 skills/flagos-release/tools/main.py --from-context /data/flagos-workspace/${MODEL}/config/context_snapshot.yaml --plugin-mode
完成后通过 docker cp 回传最终 context：
  docker cp ${SEG_CTR}:/flagos-workspace/shared/context.yaml /data/flagos-workspace/${MODEL}/config/context_final.yaml

**⚠ 段边界**：本段执行步骤 9-13，最后一个步骤完成后停止。
完成标志：输出 \"[段4] Plugin 流程完成\" 后停止所有操作。"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  段4  Plugin 验证 + 发布  (步骤 9→10→11→12→13)             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    SEG4_START_TS=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 段4 开始 (qualified=true, 触发 Plugin 流程)"

    # ===== 幂等检查：步骤9 是否已在段3中被越界完成 =====
    SEG4_PLUGIN_DONE=$(python3 -c "
import yaml
try:
    with open('/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml') as f:
        ctx = yaml.safe_load(f)
    ledger = ctx.get('workflow_ledger', {}).get('steps', [])
    if isinstance(ledger, list):
        for s in ledger:
            if isinstance(s, dict) and s.get('step','').startswith('09') and s.get('status') in ('success','failed','skipped'):
                print('yes'); exit()
    elif isinstance(ledger, dict):
        for k, s in ledger.items():
            if isinstance(s, dict) and str(s.get('step', k)).startswith('09') and s.get('status') in ('success','failed','skipped'):
                print('yes'); exit()
except: pass
print('no')
" 2>/dev/null) || SEG4_PLUGIN_DONE="no"

    if [ "${SEG4_PLUGIN_DONE}" = "yes" ]; then
        echo "  ✓ 步骤9 已在前段完成（ledger 09_plugin_install 非 pending），跳过段4 Claude 调用"
        SEG4_ELAPSED=0
        SEG4_MIN=0
        SEG4_SEC=0
    else
    mkdir -p "${LOG_DIR}"
    claude -p "${PROMPT_SEG4}" \
        --permission-mode auto \
        --output-format stream-json \
        --verbose \
        --debug-file "${DEBUG_FILE}.seg4" \
        --max-turns 200 \
        2>&1 | tee -a "${LOG_FILE}" \
             | tee >(python3 "${SCRIPT_DIR}/stream_to_debug_log.py" >> "${FULL_LOG}") \
             | python3 "${SCRIPT_DIR}/stream_filter.py" --pipeline-log "${PIPELINE_LOG}" --terminal-log "${TERMINAL_LOG}" --start-step 9 --cost-file "${LOG_DIR}/seg4_cost.txt" --load-durations "${LOG_DIR}/seg3_durations.json" --durations-file "${LOG_DIR}/seg4_durations.json" ${FILTER_FLAGS} || true

    SEG4_END_TS=$(date +%s)
    SEG4_ELAPSED=$(( SEG4_END_TS - SEG4_START_TS ))
    SEG4_MIN=$(( SEG4_ELAPSED / 60 ))
    SEG4_SEC=$(( SEG4_ELAPSED % 60 ))
    fi
else
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] qualified=${QUALIFIED}，跳过 Plugin 流程（步骤 9-13）"
fi

# ===== Claude 退出后自动故障诊断 =====
# 从宿主机 context_snapshot 读取容器名（镜像模式下容器名由 Claude 动态创建）
# 不再 fallback 到项目级 shared/context.yaml，避免多任务冲突
DIAG_CONTAINER=""
CONTEXT_FILE="/data/flagos-workspace/${MODEL}/config/context_snapshot.yaml"
if [ -f "${CONTEXT_FILE}" ]; then
    DIAG_CONTAINER=$(python3 -c "
import yaml, sys
try:
    with open('${CONTEXT_FILE}') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('container',{}).get('name',''))
except: pass
" 2>/dev/null) || DIAG_CONTAINER=""
fi
# fallback: 容器模式下直接用脚本变量；镜像模式下用段间传递的 SEG_CTR
[ -z "${DIAG_CONTAINER}" ] && DIAG_CONTAINER="${CONTAINER:-${SEG_CTR:-}}"

if [ -n "${DIAG_CONTAINER}" ] && docker inspect --type=container "${DIAG_CONTAINER}" &>/dev/null; then
    ALL_DONE=$(docker exec "${DIAG_CONTAINER}" bash -c "
        PATH=/opt/conda/bin:\$PATH python3 -c \"
import yaml
try:
    with open('/flagos-workspace/shared/context.yaml') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('workflow',{}).get('all_done', False))
except: print('False')
\"" 2>/dev/null || echo "False")

    if [ "${ALL_DONE}" != "True" ]; then
        echo ""
        echo "============================================"
        echo "⚠  Claude 进程已退出但流程未完成，自动诊断中..."
        echo "============================================"
        echo ""
        # tee: 终端打印 + 写入文件
        docker exec "${DIAG_CONTAINER}" bash -c \
          "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/diagnose_failure.py" \
          2>&1 | tee "${LOG_DIR}/failure_diagnosis.txt"
        # JSON 版本供新 Claude 会话读取
        docker exec "${DIAG_CONTAINER}" bash -c \
          "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/diagnose_failure.py --json" \
          > "${LOG_DIR}/failure_diagnosis.json" 2>/dev/null || true
        echo ""
        echo "诊断报告已保存:"
        echo "  可读版: ${LOG_DIR}/failure_diagnosis.txt"
        echo "  JSON版: ${LOG_DIR}/failure_diagnosis.json"
    fi
fi

# ========== 兜底：同步容器产出到宿主机 ==========
# 无论 Claude 是否在步骤8中调用了 main.py，都确保容器内产出同步到宿主机。
# setup_workspace.sh --skip-archive 跳过归档，但 run_pipeline.sh 启动前已归档清空宿主机 results/traces，
# 如果 Claude 跳过了 main.py 的 _sync_to_host()，宿主机将无数据。
if [ -n "${DIAG_CONTAINER}" ] && docker inspect --type=container "${DIAG_CONTAINER}" &>/dev/null; then
    HOST_BASE="/data/flagos-workspace/${MODEL}"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底同步：容器产出 → 宿主机..."
    for dir in results traces; do
        mkdir -p "${HOST_BASE}/${dir}"
        docker cp "${DIAG_CONTAINER}:/flagos-workspace/${dir}/." "${HOST_BASE}/${dir}/" 2>/dev/null && \
            echo "  ✓ ${dir}/ 已同步" || echo "  ⚠ ${dir}/ 同步失败或为空"
    done
    # context snapshot
    mkdir -p "${HOST_BASE}/config"
    docker cp "${DIAG_CONTAINER}:/flagos-workspace/shared/context.yaml" "${HOST_BASE}/config/context_snapshot.yaml" 2>/dev/null && \
        echo "  ✓ context_snapshot.yaml 已同步" || echo "  ⚠ context_snapshot.yaml 同步失败"
    # context_final（全流程结束时的最终状态）
    if [ ! -f "${HOST_BASE}/config/context_final.yaml" ]; then
        docker cp "${DIAG_CONTAINER}:/flagos-workspace/shared/context.yaml" "${HOST_BASE}/config/context_final.yaml" 2>/dev/null && \
            echo "  ✓ context_final.yaml 已同步（兜底）" || true
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底同步完成"

    # ========== 兜底：Harbor 发布（如 Claude 未执行） ==========
    CONTEXT_SNAP="${HOST_BASE}/config/context_snapshot.yaml"
    if [ -f "${CONTEXT_SNAP}" ]; then
        HARBOR_DONE=$(python3 -c "
import yaml, sys
try:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f)
    # plugin 发布完成时 plugin_image_url 非空，直接视为 done
    plugin_url = ctx.get('plugin_workflow', {}).get('plugin_image_url', '')
    if plugin_url:
        print('done')
        sys.exit(0)
    tag = ctx.get('image', {}).get('registry_url', '') or ctx.get('image', {}).get('harbor_tag', '')
    model_name = ctx.get('model', {}).get('name', '')
    # 简单校验：registry_url 非空，且包含当前模型关键词（排除其他模型的残留 tag）
    if tag and model_name:
        # 'tiiuae/Falcon-H1-0.5B-Base' -> 'falcon-h1-0.5b-base'
        key = model_name.split('/')[-1].lower().replace('-', '').replace('_', '').replace('.', '')
        tag_norm = tag.lower().replace('-', '').replace('_', '').replace('.', '')
        if key in tag_norm:
            print('done')
            sys.exit(0)
    print('needed')
except Exception as e:
    print('needed', file=sys.stderr)
    print('needed')
" 2>/dev/null) || HARBOR_DONE="needed"

        if [ "${HARBOR_DONE}" = "needed" ]; then
            # D1: 兜底发布前检查 service_ok，服务未启动成功则跳过
            FALLBACK_SERVICE_OK=$(python3 -c "
import yaml
try:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('workflow',{}).get('service_ok', False))
except:
    print('False')
" 2>/dev/null) || FALLBACK_SERVICE_OK="False"
            if [ "${FALLBACK_SERVICE_OK}" != "True" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底跳过：服务未启动成功，不具备发布条件"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底发布：Claude 未完成 Harbor push，自动补执行..."
                HARBOR_OUTPUT=$(python3 skills/flagos-release/tools/main.py --from-context "${CONTEXT_SNAP}" --only-harbor 2>&1) && \
                    echo "  ✓ 兜底 Harbor 发布成功" || echo "  ✗ 兜底 Harbor 发布失败"
                echo "${HARBOR_OUTPUT}"
                # 从输出中提取 harbor_image 并写入 context（供后续 MS/HF 兜底使用）
                HARBOR_IMAGE=$(echo "${HARBOR_OUTPUT}" | python3 -c "
import sys, json, re
for line in sys.stdin:
    m = re.search(r'\[RELEASE_SUMMARY\](.*?)\[/RELEASE_SUMMARY\]', line)
    if m:
        data = json.loads(m.group(1))
        print(data.get('harbor_image', ''))
        break
" 2>/dev/null) || HARBOR_IMAGE=""
                # 重新同步 context 和 traces（main.py 可能更新了）
                docker cp "${DIAG_CONTAINER}:/flagos-workspace/shared/context.yaml" "${CONTEXT_SNAP}" 2>/dev/null
                docker cp "${DIAG_CONTAINER}:/flagos-workspace/traces/." "${HOST_BASE}/traces/" 2>/dev/null
                # 写入 registry_url 到 context（docker cp 之后，确保不被覆盖）
                if [ -n "${HARBOR_IMAGE}" ]; then
                    python3 -c "
import yaml
with open('${CONTEXT_SNAP}') as f:
    ctx = yaml.safe_load(f) or {}
ctx.setdefault('image', {})['registry_url'] = '${HARBOR_IMAGE}'
with open('${CONTEXT_SNAP}', 'w') as f:
    yaml.dump(ctx, f, default_flow_style=False, allow_unicode=True)
" 2>/dev/null && echo "  ✓ context 已更新 image.registry_url"
                fi
            fi
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Harbor 发布已完成，跳过兜底"
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ context_snapshot.yaml 不存在，无法判断 Harbor 发布状态"
    fi

    # ========== 兜底：ModelScope/HuggingFace 上传（如 Claude 未完成） ==========
    CONTEXT_SNAP="${HOST_BASE}/config/context_snapshot.yaml"
    if [ -f "${CONTEXT_SNAP}" ] && [ -n "${MODELSCOPE_TOKEN}" ] && [ -n "${HF_TOKEN}" ]; then
        UPLOAD_STATUS=$(python3 -c "
import yaml, sys
try:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f)
    release = ctx.get('release', {})
    ms_done = bool(release.get('modelscope_url', ''))
    hf_done = bool(release.get('huggingface_url', ''))
    svc_ok = ctx.get('workflow', {}).get('service_ok', False)
    harbor_done = bool(ctx.get('image', {}).get('registry_url', ''))
    if not svc_ok:
        print('skip_svc')
    elif not harbor_done:
        print('skip_harbor')
    elif ms_done and hf_done:
        print('done')
    else:
        print('needed')
except:
    print('skip_err')
" 2>/dev/null) || UPLOAD_STATUS="skip_err"

        if [ "${UPLOAD_STATUS}" = "needed" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底上传：ModelScope/HuggingFace 未完成，自动补执行..."
            UPLOAD_OUTPUT=$(python3 skills/flagos-release/tools/main.py \
                --from-context "${CONTEXT_SNAP}" \
                --container-name "${DIAG_CONTAINER}" \
                2>&1) && \
                echo "  ✓ 兜底 ModelScope/HuggingFace 上传成功" || echo "  ⚠ 兜底 ModelScope/HuggingFace 上传失败（部分或全部）"
            echo "${UPLOAD_OUTPUT}"
            # 从输出中提取上传结果并写入 context
            echo "${UPLOAD_OUTPUT}" | python3 -c "
import sys, json, re, yaml
summary = None
for line in sys.stdin:
    m = re.search(r'\[RELEASE_SUMMARY\](.*?)\[/RELEASE_SUMMARY\]', line)
    if m:
        summary = json.loads(m.group(1))
        break
if summary:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f) or {}
    release = ctx.setdefault('release', {})
    if summary.get('modelscope_url'):
        release['modelscope_url'] = summary['modelscope_url']
    if summary.get('huggingface_url'):
        release['huggingface_url'] = summary['huggingface_url']
    with open('${CONTEXT_SNAP}', 'w') as f:
        yaml.dump(ctx, f, default_flow_style=False, allow_unicode=True)
    print('  ✓ context release 字段已更新')
" 2>/dev/null || true
        elif [ "${UPLOAD_STATUS}" = "done" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ModelScope/HuggingFace 上传已完成，跳过兜底"
        elif [ "${UPLOAD_STATUS}" = "skip_harbor" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Harbor 镜像未就绪（image.registry_url 为空），跳过 MS/HF 兜底上传"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务未启动成功，跳过 ModelScope/HuggingFace 兜底上传"
        fi
    fi

    # ========== 兜底：生成缺失的 issue 文件（精度/性能不达标时） ==========
    CONTEXT_SNAP="${HOST_BASE}/config/context_snapshot.yaml"
    if [ -f "${CONTEXT_SNAP}" ]; then
        # 读取 workflow 状态
        SERVICE_OK=$(python3 -c "
import yaml
try:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('workflow',{}).get('service_ok', True))
except: print('True')
" 2>/dev/null) || SERVICE_OK="True"

        PERF_OK=$(python3 -c "
import yaml
try:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('workflow',{}).get('performance_ok', True))
except: print('True')
" 2>/dev/null) || PERF_OK="True"

        ACC_OK=$(python3 -c "
import yaml
try:
    with open('${CONTEXT_SNAP}') as f:
        ctx = yaml.safe_load(f)
    print(ctx.get('workflow',{}).get('accuracy_ok', True))
except: print('True')
" 2>/dev/null) || ACC_OK="True"

        # 服务启动失败但 issue 文件缺失 → 兜底调用 issue_reporter.py
        if [ "${SERVICE_OK}" = "False" ]; then
            CRASH_ISSUE_EXISTS=$(docker exec "${DIAG_CONTAINER}" bash -c "ls /flagos-workspace/results/issue_operator-crash_*.md 2>/dev/null | head -1" 2>/dev/null || echo "")
            if [ -z "${CRASH_ISSUE_EXISTS}" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底生成启动崩溃 issue（Claude 遗漏）..."
                docker exec "${DIAG_CONTAINER}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \
                    --type operator-crash \
                    --log-path /flagos-workspace/logs/startup_flagos.log \
                    --context-yaml /flagos-workspace/shared/context.yaml \
                    --repo flagos-ai/FlagGems \
                    --output-dir /flagos-workspace/results/ \
                    --json" 2>&1 && \
                    echo "  ✓ 启动崩溃 issue 文件已生成" || echo "  ⚠ 启动崩溃 issue 文件生成失败"
                docker cp "${DIAG_CONTAINER}:/flagos-workspace/results/." "${HOST_BASE}/results/" 2>/dev/null
            fi
        fi

        # 性能不达标但 issue 文件缺失 → 兜底调用 issue_reporter.py
        if [ "${PERF_OK}" = "False" ]; then
            PERF_ISSUE_EXISTS=$(docker exec "${DIAG_CONTAINER}" bash -c "ls /flagos-workspace/results/issue_performance-degraded_*.md 2>/dev/null | head -1" 2>/dev/null || echo "")
            PERF_ISSUE_LOG=$(docker exec "${DIAG_CONTAINER}" bash -c "[ -s /flagos-workspace/logs/issues_performance.log ] && echo 'exists' || echo ''" 2>/dev/null || echo "")
            if [ -z "${PERF_ISSUE_EXISTS}" ] || [ -z "${PERF_ISSUE_LOG}" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底生成性能不达标 issue（Claude 遗漏）..."
                docker exec "${DIAG_CONTAINER}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \
                    --type performance-degraded \
                    --context-yaml /flagos-workspace/shared/context.yaml \
                    --repo flagos-ai/FlagGems \
                    --output-dir /flagos-workspace/results/ \
                    --json" 2>&1 && \
                    echo "  ✓ 性能 issue 文件已生成" || echo "  ⚠ 性能 issue 文件生成失败"
                # 同步到宿主机
                docker cp "${DIAG_CONTAINER}:/flagos-workspace/results/." "${HOST_BASE}/results/" 2>/dev/null
            fi
        fi

        # 精度不达标但 issue 文件缺失 → 兜底调用 issue_reporter.py
        if [ "${ACC_OK}" = "False" ]; then
            ACC_ISSUE_EXISTS=$(docker exec "${DIAG_CONTAINER}" bash -c "ls /flagos-workspace/results/issue_accuracy-degraded_*.md 2>/dev/null | head -1" 2>/dev/null || echo "")
            ACC_ISSUE_LOG=$(docker exec "${DIAG_CONTAINER}" bash -c "[ -s /flagos-workspace/logs/issues_accuracy.log ] && echo 'exists' || echo ''" 2>/dev/null || echo "")
            if [ -z "${ACC_ISSUE_EXISTS}" ] || [ -z "${ACC_ISSUE_LOG}" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底生成精度不达标 issue（Claude 遗漏）..."
                docker exec "${DIAG_CONTAINER}" bash -c "PATH=/opt/conda/bin:\$PATH python3 /flagos-workspace/scripts/issue_reporter.py full \
                    --type accuracy-degraded \
                    --context-yaml /flagos-workspace/shared/context.yaml \
                    --repo flagos-ai/FlagGems \
                    --output-dir /flagos-workspace/results/ \
                    --json" 2>&1 && \
                    echo "  ✓ 精度 issue 文件已生成" || echo "  ⚠ 精度 issue 文件生成失败"
                docker cp "${DIAG_CONTAINER}:/flagos-workspace/results/." "${HOST_BASE}/results/" 2>/dev/null
            fi
        fi
    fi

    # ========== 兜底：生成性能对比文件（如缺失） ==========
    NATIVE_PERF="${HOST_BASE}/results/native_performance.json"
    FLAGOS_PERF="${HOST_BASE}/results/flagos_performance.json"
    COMPARE_CSV="${HOST_BASE}/results/performance_compare.csv"
    if [ -f "${NATIVE_PERF}" ] && [ -f "${FLAGOS_PERF}" ] && [ ! -f "${COMPARE_CSV}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底生成性能对比文件..."
        python3 skills/flagos-performance-testing/tools/performance_compare.py \
            --native "${NATIVE_PERF}" \
            --flagos-initial "${FLAGOS_PERF}" \
            --output "${COMPARE_CSV}" 2>&1 && \
            echo "  ✓ performance_compare.csv 已生成" || echo "  ⚠ 性能对比文件生成失败"
    fi

    # ========== 兜底：V3 性能对比（如有 flagos_optimized.json） ==========
    OPTIMIZED_PERF="${HOST_BASE}/results/flagos_optimized.json"
    COMPARE_V3_CSV="${HOST_BASE}/results/performance_compare_v3.csv"
    if [ -f "${NATIVE_PERF}" ] && [ -f "${OPTIMIZED_PERF}" ] && [ ! -f "${COMPARE_V3_CSV}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 兜底生成 V3 性能对比文件..."
        python3 skills/flagos-performance-testing/tools/performance_compare.py \
            --native "${NATIVE_PERF}" \
            --flagos-initial "${FLAGOS_PERF}" \
            --flagos-optimized "${OPTIMIZED_PERF}" \
            --output "${COMPARE_V3_CSV}" 2>&1 && \
            echo "  ✓ performance_compare_v3.csv 已生成" || echo "  ⚠ V3 性能对比文件生成失败"
    fi
fi

# 流程结束前清理推理服务
cleanup_gpu_services

echo ""
PIPELINE_END_TS=$(date +%s)
PIPELINE_ELAPSED=$(( PIPELINE_END_TS - PIPELINE_START_TS ))
PIPELINE_MIN=$(( PIPELINE_ELAPSED / 60 ))
PIPELINE_SEC=$(( PIPELINE_ELAPSED % 60 ))

# 读取各段费用
SEG1_COST=$(cat "${LOG_DIR}/seg1_cost.txt" 2>/dev/null || echo "N/A")
SEG2_COST=$(cat "${LOG_DIR}/seg2_cost.txt" 2>/dev/null || echo "N/A")
SEG3_COST=$(cat "${LOG_DIR}/seg3_cost.txt" 2>/dev/null || echo "N/A")
SEG4_COST=$(cat "${LOG_DIR}/seg4_cost.txt" 2>/dev/null || echo "N/A")
# 计算总费用（通过环境变量传递，避免 shell 注入）
TOTAL_COST=$(SEG1_COST="${SEG1_COST}" SEG2_COST="${SEG2_COST}" SEG3_COST="${SEG3_COST}" SEG4_COST="${SEG4_COST}" python3 -c "
import os
costs = []
for k in ['SEG1_COST', 'SEG2_COST', 'SEG3_COST', 'SEG4_COST']:
    try: costs.append(float(os.environ.get(k, '').strip()))
    except: pass
print(f'{sum(costs):.2f}' if costs else 'N/A')
" 2>/dev/null || echo "N/A")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全流程完成 — 耗时 & 费用汇总                                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  段1  容器准备+环境检测+服务启动   %6s   \$%-8s║\n" "${SEG1_MIN}m${SEG1_SEC}s" "${SEG1_COST}"
printf "║  段2  精度评测+调优+性能评测+调优  %6s   \$%-8s║\n" "${SEG2_MIN}m${SEG2_SEC}s" "${SEG2_COST}"
printf "║  段3  打包发布                     %6s   \$%-8s║\n" "${SEG3_MIN}m${SEG3_SEC}s" "${SEG3_COST}"
if [ "${QUALIFIED}" = "True" ]; then
printf "║  段4  Plugin验证+发布              %6s   \$%-8s║\n" "${SEG4_MIN}m${SEG4_SEC}s" "${SEG4_COST}"
fi
echo "║──────────────────────────────────────────────────────────────║"
printf "║  总计                              %6s   \$%-8s║\n" "${PIPELINE_MIN}m${PIPELINE_SEC}s" "${TOTAL_COST}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Claude Code 流程结束"
echo "日志文件:"
echo "  原始事件流: ${LOG_FILE}"
echo "  可读执行记录: ${FULL_LOG}"
echo "  流水线日志: ${PIPELINE_LOG}"
echo "  内部 debug: ${DEBUG_FILE}"
