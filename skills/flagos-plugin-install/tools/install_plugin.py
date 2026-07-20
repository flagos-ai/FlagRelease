#!/usr/bin/env python3
"""
install_plugin.py — vllm-plugin-FL 安装/验证/卸载

Usage:
    python3 install_plugin.py --action install --json
    python3 install_plugin.py --action install --branch dev --editable --json
    python3 install_plugin.py --action verify --json
    python3 install_plugin.py --action uninstall --json

退出码: 0=成功, 1=失败, 2=参数错误
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# error_writer 集成
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from error_writer import write_last_error, write_checkpoint
except ImportError:
    def write_last_error(*a, **kw): pass
    def write_checkpoint(*a, **kw): pass

DEFAULT_REPO = "https://github.com/flagos-ai/vllm-plugin-FL"
PACKAGE_NAME = "vllm-plugin-FL"
# 真实导入名候选：镜像自带的 plugin 实际模块名是 vllm_fl（entry point fl），
# 旧版硬编码 vllm_plugin_fl 已过期，会导致 verify 误报 importable=false。
# 按顺序探测，任一可导入即视为可用。
IMPORT_NAME = "vllm_plugin_fl"          # 兼容保留（历史包名）
IMPORT_NAME_CANDIDATES = ["vllm_fl", "vllm_plugin_fl"]
CLONE_DIR = "/tmp/vllm-plugin-FL"


def run_cmd(cmd, timeout=600, env=None):
    """运行命令，返回 (returncode, stdout, stderr)"""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, env=merged_env
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "command timed out"
    except Exception as e:
        return -1, "", str(e)


def get_current_version():
    """获取当前安装版本"""
    code, out, _ = run_cmd(f"pip show {PACKAGE_NAME} 2>/dev/null")
    if code == 0:
        for line in out.split("\n"):
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    return None


def _load_proxy_list():
    """从 .proxy 文件或环境变量加载代理列表"""
    proxy_file = "/flagos-workspace/.proxy"
    if os.path.exists(proxy_file):
        with open(proxy_file) as f:
            return [line.strip() for line in f if line.strip()]
    proxy_str = os.environ.get("FLAGOS_PROXY_LIST", "")
    if proxy_str:
        return [p.strip() for p in proxy_str.split(",") if p.strip()]
    current = os.environ.get("https_proxy") or os.environ.get("http_proxy", "")
    return [current] if current else []


DEFAULT_CONTEXT_PATH = "/flagos-workspace/shared/context.yaml"


def _is_branch_b(context_path=DEFAULT_CONTEXT_PATH):
    """判断当前是否为分支 B（准入镜像自带 plugin）。

    分支 B 的物理本质是"准入镜像本就带 plugin"（inspect_env.classify_entry_image_type：
    has_plugin=True → gems_tree_plugin）。此处用双信号做"或"判断，任一命中即视为分支 B：
      1. workflow.entry_image_type == "gems_tree_plugin"（段1路由的确定性分类）
      2. workflow.pipeline_branch == "B"

    读不到 context 或字段缺失 → 返回 None（表示"无法判定"，由调用方保守放行，
    不误伤分支 A）。
    """
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(context_path) as f:
            ctx = yaml.safe_load(f) or {}
    except Exception:
        return None
    wf = ctx.get("workflow", {}) or {}
    entry = str(wf.get("entry_image_type", "") or "")
    branch = str(wf.get("pipeline_branch", "") or "")
    if entry == "gems_tree_plugin" or branch == "B":
        return True
    if entry or branch:
        # 有明确的分支信息但不是 B → 分支 A / native
        return False
    # 两个字段都缺失 → 无法判定
    return None


def install_plugin(repo_url=DEFAULT_REPO, branch="main", proxy=None, editable=False,
                   force=False, context_path=DEFAULT_CONTEXT_PATH):
    """git clone + pip install --no-build-isolation

    分支 B 硬闸门：准入镜像自带 plugin，禁止重装（重装会 rm -rf + 重新 clone/pip，
    覆盖镜像里厂商适配好的 plugin）。命中"(分支 B) 且 (plugin 已安装)"时拒绝重装、
    返回 skipped，不执行任何安装动作。--force 可绕过（应急逃生口）。
    """
    if not force:
        is_b = _is_branch_b(context_path)
        existing_version = get_current_version()
        if is_b is True and existing_version:
            return {
                "success": True,
                "action": "install",
                "skipped": True,
                "version": existing_version,
                "reason": "分支 B（准入镜像自带 plugin），plugin 已安装，禁止重装（避免覆盖镜像厂商适配的 plugin）。如确需重装请加 --force。",
                "timestamp": datetime.now().isoformat(),
            }

    env = {}
    if proxy:
        env["http_proxy"] = proxy
        env["https_proxy"] = proxy

    # 清理旧目录
    if os.path.exists(CLONE_DIR):
        run_cmd(f"rm -rf {CLONE_DIR}")

    # clone（支持代理切换重试）
    clone_cmd = f"git clone --depth 1 -b {branch} {repo_url} {CLONE_DIR}"
    code, out, err = run_cmd(clone_cmd, timeout=120, env=env)
    if code != 0:
        # 尝试从代理列表逐个切换重试
        proxy_list = _load_proxy_list()
        clone_success = False
        for fallback_proxy in proxy_list:
            if fallback_proxy == proxy:
                continue
            print(f"  git clone 失败，切换代理重试: {fallback_proxy}")
            env["http_proxy"] = fallback_proxy
            env["https_proxy"] = fallback_proxy
            if os.path.exists(CLONE_DIR):
                run_cmd(f"rm -rf {CLONE_DIR}")
            code, out, err = run_cmd(clone_cmd, timeout=120, env=env)
            if code == 0:
                clone_success = True
                break
        if not clone_success:
            return {
                "success": False,
                "action": "install",
                "error": f"git clone 失败 (所有代理均尝试): {err}",
                "command": clone_cmd,
            }

    # install (with pip mirror fallback)
    edit_flag = "-e " if editable else ""
    pip_mirrors = [
        None,
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://mirrors.cloud.tencent.com/pypi/simple/",
    ]
    last_err = ""
    for mirror in pip_mirrors:
        index_flag = f" -i {mirror}" if mirror else ""
        install_cmd = f"pip install --no-build-isolation{index_flag} {edit_flag}{CLONE_DIR}"
        code, out, err = run_cmd(install_cmd, timeout=600, env=env)
        if code == 0:
            break
        last_err = err
        if mirror:
            print(f"  pip install 失败 (mirror={mirror})，尝试下一个镜像源...")
    else:
        return {
            "success": False,
            "action": "install",
            "error": f"pip install 失败 (所有镜像源均失败): {last_err}",
            "command": install_cmd,
        }

    version = get_current_version()
    return {
        "success": True,
        "action": "install",
        "version": version or "unknown",
        "install_method": "editable" if editable else "source",
        "editable": editable,
        "repo_url": repo_url,
        "branch": branch,
        "timestamp": datetime.now().isoformat(),
    }


def verify_plugin():
    """验证 plugin 安装状态"""
    version = get_current_version()
    if not version:
        return {
            "success": False,
            "action": "verify",
            "installed": False,
            "error": f"{PACKAGE_NAME} 未安装",
        }

    # 尝试 import：按候选模块名依次探测，任一可导入即视为可用。
    # （镜像自带 plugin 的真实模块名是 vllm_fl，旧硬编码 vllm_plugin_fl 会误报未安装）
    importable = False
    imported_name = ""
    last_err = ""
    for cand in IMPORT_NAME_CANDIDATES:
        code, out, err = run_cmd(
            f'python3 -c "import {cand}; print(getattr({cand}, \'__version__\', \'\'))"'
        )
        if code == 0:
            importable = True
            imported_name = cand
            break
        last_err = err

    return {
        "success": importable,
        "action": "verify",
        "installed": True,
        "version": version,
        "importable": importable,
        "import_name": imported_name,
        "import_error": last_err if not importable else "",
        "timestamp": datetime.now().isoformat(),
    }


def uninstall_plugin():
    """卸载 plugin"""
    version = get_current_version()
    if not version:
        return {
            "success": True,
            "action": "uninstall",
            "message": f"{PACKAGE_NAME} 未安装，无需卸载",
        }

    code, out, err = run_cmd(f"pip uninstall -y {PACKAGE_NAME}")
    if code != 0:
        return {
            "success": False,
            "action": "uninstall",
            "error": f"卸载失败: {err}",
        }

    # 清理 clone 目录
    if os.path.exists(CLONE_DIR):
        run_cmd(f"rm -rf {CLONE_DIR}")

    return {
        "success": True,
        "action": "uninstall",
        "previous_version": version,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="vllm-plugin-FL 安装管理")
    parser.add_argument("--action", required=True, choices=["install", "verify", "uninstall"])
    parser.add_argument("--repo-url", default=DEFAULT_REPO, help="仓库地址")
    parser.add_argument("--branch", default="main", help="分支")
    parser.add_argument("--proxy", default=None, help="代理地址")
    parser.add_argument("--editable", action="store_true", help="editable install (-e)")
    parser.add_argument("--force", action="store_true",
                        help="绕过分支 B 禁重装闸门（应急逃生口，会覆盖镜像自带 plugin）")
    parser.add_argument("--context-path", default=DEFAULT_CONTEXT_PATH,
                        help="context.yaml 路径（分支判定用）")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    try:
        if args.action == "install":
            result = install_plugin(
                repo_url=args.repo_url,
                branch=args.branch,
                proxy=args.proxy,
                editable=args.editable,
                force=args.force,
                context_path=args.context_path,
            )
        elif args.action == "verify":
            result = verify_plugin()
        elif args.action == "uninstall":
            result = uninstall_plugin()
        else:
            result = {"success": False, "error": f"未知操作: {args.action}"}
    except Exception as e:
        result = {"success": False, "action": args.action, "error": str(e)}
        write_last_error("install_plugin", str(e), step=f"plugin_{args.action}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result.get("skipped"):
            print(f"⊘ {args.action} 跳过: {result.get('reason', '')}")
            if result.get("version"):
                print(f"  当前版本: {result['version']}")
        elif result.get("success"):
            print(f"✓ {args.action} 成功")
            if result.get("version"):
                print(f"  版本: {result['version']}")
        else:
            print(f"✗ {args.action} 失败: {result.get('error', 'unknown')}")

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
