#!/bin/bash
# 统一镜像名称生成脚本（压缩版）
# 用法: bash get_image_name.sh <厂商> <容器名/模型名> [模型名]
# 厂商: ascend, hygon, iluvatar, kunlunxin, metax, mthreads, nvidia, tsingmicro, zhenwu

set -euo pipefail

VALID_VENDORS="ascend hygon iluvatar kunlunxin metax mthreads nvidia tsingmicro zhenwu"

VENDOR="${1:-}"
if [ -z "$VENDOR" ] || ! echo "$VALID_VENDORS" | grep -qw "$VENDOR"; then
    echo "错误: 请输入有效的厂商名称"
    echo "支持的厂商: $VALID_VENDORS"
    exit 1
fi

shift

# ========== 公共函数 ==========
parse_ver() {
    echo "$1" | awk -v pkg="$2" '
    BEGIN { gsub(/[-_]/, "-", pkg); pkg = tolower(pkg) }
    { name = tolower($1); gsub(/[-_]/, "-", name); if (name == pkg) print $2 }
    '
}

# 版本压缩: "3.13.2" → "313", "2.8.4+metax0425" → "28", "none" → "none"
cver() {
    local v="$1"
    [ "$v" = "none" ] || [ -z "$v" ] && { echo "none"; return; }
    v="${v%%+*}"
    echo "$v" | grep -oP '^\d+\.\d+' | tr -d '.'
}

# 保留x.y.z粒度，去掉后缀: "0.3.0+metax0425" → "0.3.0", "0.8.5.dev123" → "0.8.5", "none" → "none"
semver() {
    local v="$1"
    [ "$v" = "none" ] || [ -z "$v" ] && { echo "none"; return; }
    echo "$v" | grep -oP '^\d+\.\d+(\.\d+)?' | head -1
}

# 架构压缩
carc() {
    case "$1" in
        x86_64|amd64) echo "x64" ;;
        aarch64|arm64) echo "a64" ;;
        *) echo "$1" ;;
    esac
}

# ========== zhenwu 特殊处理（不需要容器） ==========
if [ "$VENDOR" = "zhenwu" ]; then
    MODEL="${1:-unknown}"
    GPU="pp001"

    PIP_LIST=$(pip list 2>/dev/null || echo "")
    TREE_VER=$(parse_ver "$PIP_LIST" flagtree)
    GEMS_VER=$(parse_ver "$PIP_LIST" flag_gems)
    VLLM_VER=$(parse_ver "$PIP_LIST" vllm)
    PLUGIN_VER=$(parse_ver "$PIP_LIST" vllm-plugin-fl)
    CX_VER=$(parse_ver "$PIP_LIST" flagcx)
    TREE_VER="${TREE_VER:-none}"
    GEMS_VER="${GEMS_VER:-none}"
    VLLM_VER="${VLLM_VER:-none}"
    PLUGIN_VER="${PLUGIN_VER:-none}"
    CX_VER="${CX_VER:-none}"

    PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
    TORCH_VER=$(parse_ver "$PIP_LIST" torch)
    TORCH_VER="${TORCH_VER:-none}"

    SMI_OUTPUT=$(nvidia-smi 2>/dev/null | head -3 || echo "")
    HGGC_VER=$(echo "$SMI_OUTPUT" | grep -oP 'HGGC Version:\s*\K[0-9.]+' || echo "none")

    ARCH=$(carc "$(uname -m)")
    DRIVER=$(echo "$SMI_OUTPUT" | grep -oP 'Driver Version:\s*\K[^\s]+' || echo "none")
    TIMESTAMP=$(date +"%Y%m%d%H%M")

    G=$(semver "$GEMS_VER"); T=$(semver "$TREE_VER")
    C=$(semver "$CX_VER"); P=$(semver "$PLUGIN_VER")
    V=$(semver "$VLLM_VER"); PY=$(cver "$PYTHON_VER"); PT=$(cver "$TORCH_VER")
    HV=$(cver "$HGGC_VER")

    echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-hggc${HV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
    exit 0
fi

# ========== 其他厂商需要容器参数 ==========
CONTAINER="${1:?用法: $0 <厂商> <容器名/ID> [模型名]}"
if [ -n "${2:-}" ]; then
    MODEL="$2"
else
    CNAME=$(docker inspect --format '{{.Name}}' "$CONTAINER" 2>/dev/null | sed 's|^/||')
    MODEL=$(echo "$CNAME" | sed 's/^[^_]*_//')
fi

# ========== 公共采集 ==========
case "$VENDOR" in
    ascend|tsingmicro|nvidia)
        run() { docker exec "$CONTAINER" bash -c "$1" 2>/dev/null || echo "none"; }
        ;;
    kunlunxin)
        SHELL_INIT='export PS1=x; source ~/.bashrc 2>/dev/null'
        run() { timeout 15 docker exec "$CONTAINER" bash -c "${SHELL_INIT}; $1" 2>/dev/null || echo "none"; }
        ;;
    *)
        run() { docker exec "$CONTAINER" bash -lc "$1" 2>/dev/null || echo "none"; }
        ;;
esac

PIP_LIST=$(run "pip list 2>/dev/null")
TREE_VER=$(parse_ver "$PIP_LIST" flagtree)
GEMS_VER=$(parse_ver "$PIP_LIST" flag_gems)
VLLM_VER=$(parse_ver "$PIP_LIST" vllm)
PLUGIN_VER=$(parse_ver "$PIP_LIST" vllm-plugin-fl)
[ -z "$PLUGIN_VER" ] && PLUGIN_VER=$(parse_ver "$PIP_LIST" vllm_fl)
CX_VER=$(parse_ver "$PIP_LIST" flagcx)
TREE_VER="${TREE_VER:-none}"
GEMS_VER="${GEMS_VER:-none}"
VLLM_VER="${VLLM_VER:-none}"
PLUGIN_VER="${PLUGIN_VER:-none}"
CX_VER="${CX_VER:-none}"

PYTHON_VER=$(run "python3 --version 2>&1 | awk '{print \$2}'")
TORCH_VER=$(parse_ver "$PIP_LIST" torch)
TORCH_VER="${TORCH_VER:-none}"

ARCH=$(carc "$(run "uname -m")")
TIMESTAMP=$(date +"%Y%m%d%H%M")

# 公共版本压缩 (顺序: gems, tree, cx, plugin | vllm, python, torch)
G=$(semver "$GEMS_VER"); T=$(semver "$TREE_VER")
C=$(semver "$CX_VER"); P=$(semver "$PLUGIN_VER")
V=$(semver "$VLLM_VER"); PY=$(cver "$PYTHON_VER"); PT=$(cver "$TORCH_VER")

# ========== 厂商特有逻辑 ==========
case "$VENDOR" in
    ascend)
        GPU="ascend001"
        VLLM_ASCEND_VER=$(parse_ver "$PIP_LIST" vllm-ascend)
        if [ -n "$VLLM_ASCEND_VER" ]; then
            V=$(semver "$VLLM_ASCEND_VER")
            VL="vllm-ascend"
        else
            VL="vllm"
        fi
        TORCH_NPU_VER=$(parse_ver "$PIP_LIST" torch_npu)
        TORCH_NPU_VER="${TORCH_NPU_VER:-none}"
        PT=$(cver "$TORCH_NPU_VER")
        CANN_VER=$(run "find /usr/local/Ascend/ -name 'version.cfg' -o -name 'version.info' 2>/dev/null | head -1 | xargs grep -i 'version' 2>/dev/null | head -1 | sed 's/.*=//;s/.*: *//' | xargs")
        [ -z "$CANN_VER" ] && CANN_VER="none"
        CV=$(cver "$CANN_VER")
        DRIVER=$(run "npu-smi info 2>/dev/null | grep -i 'Version' | head -1 | grep -oP 'Version:\s*\K[0-9.]+'" )
        [ -z "$DRIVER" ] || [ "$DRIVER" = "none" ] && DRIVER=$(npu-smi info 2>/dev/null | grep -i 'Version' | head -1 | grep -oP 'Version:\s*\K[0-9.]+' || echo "none")
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-${VL}${V}-cp${PY}-ptnpu${PT}-cann${CV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    hygon)
        GPU="hygon001"
        DTK_VER=$(run "hipcc --version 2>&1 | grep -oP '/opt/dtk-\K[0-9.]+'" )
        [ -z "$DTK_VER" ] || [ "$DTK_VER" = "none" ] && DTK_VER=$(hipcc --version 2>&1 | grep -oP '/opt/dtk-\K[0-9.]+' || echo "none")
        DTK_VER="${DTK_VER:-none}"
        DV=$(cver "$DTK_VER")
        DRIVER=$(run "hy-smi --showdriverversion 2>/dev/null | grep -oP 'Driver Version:\s*\K[^\s]+'" )
        [ -z "$DRIVER" ] || [ "$DRIVER" = "none" ] && DRIVER=$(hy-smi --showdriverversion 2>/dev/null | grep -oP 'Driver Version:\s*\K[^\s]+' || echo "none")
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-dtk${DV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    iluvatar)
        GPU="iluvatar001"
        IXML_VER=$(run "ixsmi 2>/dev/null | grep -oP 'IX-ML:\s*\K[0-9.]+'" )
        [ -z "$IXML_VER" ] || [ "$IXML_VER" = "none" ] && IXML_VER=$(ixsmi 2>/dev/null | grep -oP 'IX-ML:\s*\K[0-9.]+' || echo "none")
        IXML_VER="${IXML_VER:-none}"
        IV=$(cver "$IXML_VER")
        DRIVER=$(run "ixsmi 2>/dev/null | grep -oP 'Driver Version:\s*\K[0-9.]+'" )
        [ -z "$DRIVER" ] || [ "$DRIVER" = "none" ] && DRIVER=$(ixsmi 2>/dev/null | grep -oP 'Driver Version:\s*\K[0-9.]+' || echo "none")
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-ixml${IV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    kunlunxin)
        GPU="kunlunxin001"
        XPU_SMI_OUTPUT=$(timeout 5 docker exec "$CONTAINER" bash -lc "xpu-smi" 2>/dev/null || true)
        XRE_VER=$(echo "$XPU_SMI_OUTPUT" | awk '/XPU-RT Version/{match($0, /XPU-RT Version:[[:space:]]*([0-9.]+)/, m); print m[1]}' | head -1)
        XRE_VER="${XRE_VER:-none}"
        XV=$(cver "$XRE_VER")
        DRIVER=$(echo "$XPU_SMI_OUTPUT" | awk '/Driver Version/{match($0, /Driver Version:[[:space:]]*([0-9.]+)/, m); print m[1]}' | head -1)
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-xrt${XV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    metax)
        GPU="metax001"
        MACA_VER=$(run "mx-smi 2>/dev/null | grep -oP 'MACA Version:\s*\K[0-9.]+'" )
        [ -z "$MACA_VER" ] || [ "$MACA_VER" = "none" ] && MACA_VER=$(mx-smi 2>/dev/null | grep -oP 'MACA Version:\s*\K[0-9.]+' || echo "none")
        MACA_VER="${MACA_VER:-none}"
        MV=$(cver "$MACA_VER")
        DRIVER=$(run "mx-smi 2>/dev/null | grep -oP 'Kernel Mode Driver Version:\s*\K[0-9.]+'" )
        [ -z "$DRIVER" ] || [ "$DRIVER" = "none" ] && DRIVER=$(mx-smi 2>/dev/null | grep -oP 'Kernel Mode Driver Version:\s*\K[0-9.]+' || echo "none")
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-maca${MV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    mthreads)
        GPU="mthreads001"
        MUSA_VER=$(run "cat /usr/local/musa/version.json 2>/dev/null | grep -A1 'musa_toolkits' | grep -oP '\"version\":\s*\"\K[^\"]+'" )
        [ -z "$MUSA_VER" ] || [ "$MUSA_VER" = "none" ] && MUSA_VER=$(run "cat /usr/local/musa-*/version.json 2>/dev/null | grep -A1 'musa_toolkits' | grep -oP '\"version\":\s*\"\K[^\"]+'" )
        [ -z "$MUSA_VER" ] || [ "$MUSA_VER" = "none" ] && MUSA_VER=$(cat /usr/local/musa/version.json 2>/dev/null | grep -A1 'musa_toolkits' | grep -oP '"version":\s*"\K[^"]+' || echo "none")
        [ -z "$MUSA_VER" ] || [ "$MUSA_VER" = "none" ] && MUSA_VER=$(cat /usr/local/musa-*/version.json 2>/dev/null | grep -A1 'musa_toolkits' | grep -oP '"version":\s*"\K[^"]+' || echo "none")
        MUSA_VER="${MUSA_VER:-none}"
        MV=$(cver "$MUSA_VER")
        DRIVER=$(run "mthreads-gmi 2>/dev/null | grep -oP 'Driver Version:\K[^\s]+'" )
        [ -z "$DRIVER" ] || [ "$DRIVER" = "none" ] && DRIVER=$(mthreads-gmi 2>/dev/null | grep -oP 'Driver Version:\K[^\s]+' || echo "none")
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-musa${MV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    nvidia)
        GPU="nvidia003"
        CUDA_VER=$(run "nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+'")
        CV=$(cver "$CUDA_VER")
        DRIVER=$(run "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1" | xargs)
        DRIVER="${DRIVER:-none}"
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-cu${CV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
    tsingmicro)
        GPU="tsingmicro001"
        TSM_INFO=$(run "tsm_smi --version 2>/dev/null")
        DRIVER=$(echo "$TSM_INFO" | grep -i 'Driver Version' | awk -F'|' '{print $3}' | xargs)
        DRIVER="${DRIVER:-none}"
        DV=$(cver "$DRIVER")
        echo "${MODEL}-${GPU}-gems${G}-tree${T}-cx${C}-plugin${P}-vllm${V}-cp${PY}-pt${PT}-raisa${DV}-${ARCH}-${DRIVER}:${TIMESTAMP}"
        ;;
esac
