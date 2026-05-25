#!/usr/bin/env bash
# install_flagtree.sh — FlagTree 安装/卸载/验证
#
# FlagTree 是统一 Triton 编译器，替换 triton 包（import triton 仍然生效）
# 参考: https://github.com/flagos-ai/FlagTree
#
# 支持两种安装方式：
#   1. 免源码安装（pip wheel）— 对已有预编译包的后端
#   2. 源码编译 — 对无预编译包或需要自定义版本的后端
#
# Usage:
#   install_flagtree.sh install --vendor nvidia [--version 0.5.0rc1]
#   install_flagtree.sh install --vendor iluvatar [--version 0.4.0+iluvatar3.1]
#   install_flagtree.sh install --vendor ascend --source [--branch triton_v3.2.x]
#   install_flagtree.sh uninstall
#   install_flagtree.sh verify
#   install_flagtree.sh list-vendors

set -euo pipefail

ACTION="${1:?Usage: $0 install|uninstall|verify|list-vendors}"
shift || true

# 默认值
VENDOR="${VENDOR:-nvidia}"
VERSION="${VERSION:-}"
SOURCE_BUILD=false
BRANCH=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vendor) VENDOR="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --source) SOURCE_BUILD=true; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BACKUP_DIR="/tmp/flagtree_backup"
PIP_INDEX="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=resource.flagos.net"

# =============================================================================
# 免源码安装：后端 → 默认 pip 包名映射
# 参考 FlagTree README 免源码安装表格
# https://github.com/flagos-ai/FlagTree#免源码安装
# =============================================================================
get_default_wheel_version() {
    local vendor="$1"
    case "$vendor" in
        nvidia)     echo "0.5.0rc1" ;;         # Triton 3.6, Python 3.12
        nvidia-3.5) echo "0.5.0rc1+3.5" ;;     # Triton 3.5, Python 3.12
        nvidia-3.3) echo "0.4.0+3.3" ;;        # Triton 3.3, Python 3.10/3.11/3.12
        iluvatar)   echo "0.4.0+iluvatar3.1" ;; # Triton 3.1, Python 3.10
        mthreads)   echo "0.4.0+mthreads3.1" ;; # Triton 3.1, Python 3.10
        metax)      echo "0.4.0rc1+metax3.1" ;; # Triton 3.1, Python 3.10
        ascend)     echo "0.5.0+ascend3.2" ;;   # Triton 3.2, Python 3.11
        tsingmicro) echo "0.5.0rc1+tsingmicro3.3" ;; # Triton 3.3, Python 3.10
        hcu)        echo "0.4.0+hcu3.0" ;;      # Triton 3.0, Python 3.10
        enflame)    echo "0.4.0+enflame3.3" ;;  # Triton 3.3, Python 3.10
        sunrise)    echo "0.4.0+sunrise3.4" ;;  # Triton 3.4, Python 3.10
        *)          echo "" ;;                   # 无预编译包
    esac
}

# =============================================================================
# 源码构建：后端 → 默认分支映射
# =============================================================================
get_default_branch() {
    local vendor="$1"
    case "$vendor" in
        nvidia|amd|iluvatar|mthreads|xpu|metax|hcu) echo "main" ;;       # Triton 3.0/3.1
        ascend|cambricon)                            echo "triton_v3.2.x" ;; # Triton 3.2
        tsingmicro|aipu|enflame-3.3)                 echo "triton_v3.3.x" ;; # Triton 3.3
        sunrise)                                     echo "triton_v3.4.x" ;; # Triton 3.4
        enflame)                                     echo "triton_v3.5.x" ;; # Triton 3.5 (GCU400)
        *)                                           echo "main" ;;
    esac
}

# =============================================================================
# 源码构建：后端 → FLAGTREE_BACKEND 环境变量
# nvidia/amd/triton-shared 不需要设置此变量
# =============================================================================
get_backend_env() {
    local vendor="$1"
    case "$vendor" in
        nvidia|amd|cpu) echo "" ;;  # 不设置 FLAGTREE_BACKEND
        iluvatar)   echo "iluvatar" ;;
        mthreads)   echo "mthreads" ;;
        xpu)        echo "xpu" ;;
        metax)      echo "metax" ;;
        hcu)        echo "hcu" ;;
        ascend)     echo "ascend" ;;
        cambricon)  echo "cambricon" ;;
        tsingmicro) echo "tsingmicro" ;;
        aipu)       echo "aipu" ;;
        enflame*)   echo "enflame" ;;
        sunrise)    echo "sunrise" ;;
        *)          echo "$vendor" ;;
    esac
}

# =============================================================================
# 辅助函数
# =============================================================================
check_wheel_available() {
    local ver="$1"
    [ -n "$ver" ]
}

detect_python_version() {
    python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
}

case "$ACTION" in
    list-vendors)
        echo "=========================================="
        echo "FlagTree 支持的后端"
        echo "=========================================="
        echo ""
        echo "免源码安装（pip wheel）:"
        echo "  ┌──────────────┬───────────────────────────────┬──────────┬────────┐"
        echo "  │ 后端         │ 默认版本                      │ Triton   │ Python │"
        echo "  ├──────────────┼───────────────────────────────┼──────────┼────────┤"
        echo "  │ nvidia       │ 0.5.0rc1                      │ 3.6      │ 3.12   │"
        echo "  │ nvidia-3.5   │ 0.5.0rc1+3.5                  │ 3.5      │ 3.12   │"
        echo "  │ nvidia-3.3   │ 0.4.0+3.3                     │ 3.3      │ 3.10+  │"
        echo "  │ iluvatar     │ 0.4.0+iluvatar3.1              │ 3.1      │ 3.10   │"
        echo "  │ mthreads     │ 0.4.0+mthreads3.1              │ 3.1      │ 3.10   │"
        echo "  │ metax        │ 0.4.0rc1+metax3.1              │ 3.1      │ 3.10   │"
        echo "  │ ascend       │ 0.5.0+ascend3.2                │ 3.2      │ 3.11   │"
        echo "  │ tsingmicro   │ 0.5.0rc1+tsingmicro3.3         │ 3.3      │ 3.10   │"
        echo "  │ hcu          │ 0.4.0+hcu3.0                   │ 3.0      │ 3.10   │"
        echo "  │ enflame      │ 0.4.0+enflame3.3               │ 3.3      │ 3.10   │"
        echo "  │ sunrise      │ 0.4.0+sunrise3.4               │ 3.4      │ 3.10   │"
        echo "  └──────────────┴───────────────────────────────┴──────────┴────────┘"
        echo ""
        echo "仅源码构建:"
        echo "  amd, cpu, xpu(KLX), cambricon, aipu(ARM China)"
        echo ""
        echo "用法:"
        echo "  $0 install --vendor nvidia              # 免源码安装（最新 wheel）"
        echo "  $0 install --vendor ascend              # 免源码安装"
        echo "  $0 install --vendor amd --source        # 源码构建"
        echo "  $0 install --vendor nvidia --version 0.4.0  # 指定版本"
        ;;

    install)
        echo "=========================================="
        echo "FlagTree 安装"
        echo "=========================================="
        echo "  厂商: ${VENDOR}"
        echo "  安装方式: $([ "$SOURCE_BUILD" = true ] && echo '源码构建' || echo '免源码安装')"

        # 检测 Python 版本
        PY_VER=$(detect_python_version)
        echo "  Python: ${PY_VER}"
        echo ""

        # 备份原始 triton 信息
        mkdir -p "$BACKUP_DIR"
        python3 -c "import triton; print(triton.__version__)" > "${BACKUP_DIR}/triton_version" 2>/dev/null || echo "" > "${BACKUP_DIR}/triton_version"
        pip show triton 2>/dev/null > "${BACKUP_DIR}/triton_pip_info" || true
        echo "  原始 triton 版本: $(cat "${BACKUP_DIR}/triton_version")"

        if [ "$SOURCE_BUILD" = true ]; then
            # =========================================================
            # 源码构建
            # =========================================================
            BRANCH="${BRANCH:-$(get_default_branch "$VENDOR")}"
            BACKEND_ENV=$(get_backend_env "$VENDOR")

            echo "  分支: ${BRANCH}"
            [ -n "$BACKEND_ENV" ] && echo "  FLAGTREE_BACKEND: ${BACKEND_ENV}" || echo "  FLAGTREE_BACKEND: (不设置)"
            echo ""

            echo "[1/4] 卸载原始 triton..."
            pip uninstall -y triton 2>/dev/null || true
            pip uninstall -y triton 2>/dev/null || true  # 重复确保完全卸载

            echo "[2/4] 安装构建依赖..."
            apt-get update -qq && apt-get install -y -qq zlib1g-dev libxml2-dev 2>/dev/null || true
            pip install -r /tmp/FlagTree/python/requirements.txt 2>/dev/null || true

            echo "[3/4] 克隆 FlagTree (branch: ${BRANCH})..."
            if [ -d /tmp/FlagTree ]; then
                rm -rf /tmp/FlagTree
            fi
            git clone --depth 1 --branch "$BRANCH" https://github.com/flagos-ai/FlagTree.git /tmp/FlagTree

            echo "[4/4] 编译安装..."
            cd /tmp/FlagTree
            pip install -r python/requirements.txt 2>/dev/null || true
            if [ -n "$BACKEND_ENV" ]; then
                export FLAGTREE_BACKEND="${BACKEND_ENV}"
            fi
            cd python && pip install . --no-build-isolation -v
        else
            # =========================================================
            # 免源码安装（pip wheel）
            # =========================================================
            DEFAULT_VER=$(get_default_wheel_version "$VENDOR")

            if [ -z "$DEFAULT_VER" ] && [ -z "$VERSION" ]; then
                echo ""
                echo "ERROR: 后端 '${VENDOR}' 没有预编译 wheel 包。"
                echo "请使用 --source 参数进行源码构建："
                echo "  $0 install --vendor ${VENDOR} --source"
                echo ""
                echo "或使用 'list-vendors' 查看支持的后端："
                echo "  $0 list-vendors"
                exit 1
            fi

            INSTALL_VER="${VERSION:-$DEFAULT_VER}"
            echo "  安装版本: flagtree==${INSTALL_VER}"
            echo ""

            echo "[1/2] 卸载原始 triton..."
            pip uninstall -y triton 2>/dev/null || true
            pip uninstall -y triton 2>/dev/null || true  # 重复确保完全卸载

            echo "[2/2] 安装 FlagTree..."
            pip install "flagtree==${INSTALL_VER}" ${PIP_INDEX}
        fi

        echo ""
        echo "安装完成，执行验证..."
        "$0" verify
        ;;

    uninstall)
        echo "=========================================="
        echo "FlagTree 卸载"
        echo "=========================================="

        echo "[1/2] 卸载 flagtree..."
        pip uninstall -y flagtree 2>/dev/null || true

        echo "[2/2] 恢复原始 triton..."
        ORIG_VER=""
        if [ -f "${BACKUP_DIR}/triton_version" ]; then
            ORIG_VER=$(cat "${BACKUP_DIR}/triton_version")
        fi
        if [ -n "$ORIG_VER" ]; then
            echo "  恢复 triton==${ORIG_VER}"
            pip install "triton==${ORIG_VER}"
        else
            echo "  无备份版本信息，安装最新 triton"
            pip install triton
        fi

        echo ""
        echo "卸载完成，验证状态..."
        "$0" verify
        ;;

    verify)
        echo "=========================================="
        echo "FlagTree 状态检查"
        echo "=========================================="
        python3 -c "
import json

result = {}

# 检查 triton
try:
    import triton
    result['triton_version'] = getattr(triton, '__version__', 'unknown')
    result['triton_installed'] = True
except ImportError:
    result['triton_installed'] = False
    result['triton_version'] = ''

# 检查 flagtree
try:
    import flagtree
    result['flagtree_installed'] = True
    result['flagtree_version'] = getattr(flagtree, '__version__', 'unknown')
    result['backend'] = getattr(flagtree, 'backend', '')
except ImportError:
    result['flagtree_installed'] = False
    result['flagtree_version'] = ''
    result['backend'] = ''

# Python 版本
import sys
result['python_version'] = f'{sys.version_info.major}.{sys.version_info.minor}'

print(json.dumps(result, indent=2))

# 人类可读输出
print()
print(f\"  triton: {'v' + result['triton_version'] if result['triton_installed'] else 'NOT INSTALLED'}\")
print(f\"  flagtree: {'v' + result['flagtree_version'] if result['flagtree_installed'] else 'NOT INSTALLED'}\")
if result.get('backend'):
    print(f\"  backend: {result['backend']}\")
print(f\"  python: {result['python_version']}\")
"
        ;;

    *)
        echo "Unknown action: $ACTION"
        echo "Usage: $0 install|uninstall|verify|list-vendors"
        exit 1
        ;;
esac
