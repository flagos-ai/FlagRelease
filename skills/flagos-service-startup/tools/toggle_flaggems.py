#!/usr/bin/env python3
"""
toggle_flaggems.py — 可靠的 FlagGems 开关切换

替代脆弱的 sed 行号操作，使用正则匹配 + 自动备份。

Usage:
    python3 toggle_flaggems.py --action enable    # 启用 FlagGems
    python3 toggle_flaggems.py --action disable   # 关闭 FlagGems
    python3 toggle_flaggems.py --action status    # 查看当前状态
    python3 toggle_flaggems.py --action rollback  # 回滚到备份版本
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 共享模块导入（兼容本地开发和容器内扁平部署）
sys.path.insert(0, str(Path(__file__).resolve().parent))


def env_to_inline(env_dict):
    """将 env dict 转为内联前缀字符串: VAR1=val1 VAR2=val2"""
    import shlex
    parts = []
    for k, v in env_dict.items():
        parts.append(f"{k}={shlex.quote(str(v))}")
    return " ".join(parts)


# FlagGems 相关的代码模式
FLAGGEMS_PATTERNS = [
    re.compile(r"^(\s*)(import flag_gems.*)$"),
    re.compile(r"^(\s*)(from flag_gems.*)$"),
    re.compile(r"^(\s*)(flag_gems\.\w+.*)$"),
]

COMMENTED_PATTERNS = [
    re.compile(r"^(\s*)#\s*(import flag_gems.*)$"),
    re.compile(r"^(\s*)#\s*(from flag_gems.*)$"),
    re.compile(r"^(\s*)#\s*(flag_gems\.\w+.*)$"),
]

BACKUP_SUFFIX = ".flaggems_backup"


def detect_plugin_mode():
    """检测是否为 plugin 场景"""
    try:
        import vllm_fl
        return True
    except ImportError:
        return False


def generate_env_vars(action):
    """Plugin 场景：生成环境变量字典（不再写文件）"""
    env = {}
    if action == "enable":
        env["USE_FLAGGEMS"] = "1"
        env["VLLM_FL_PREFER_ENABLED"] = "true"
    elif action == "disable":
        env["USE_FLAGGEMS"] = "0"
        env["VLLM_FL_PREFER_ENABLED"] = "false"
    return env


def find_model_runner_files():
    """自动扫描所有 model_runner.py 文件"""
    candidates = []
    search_dirs = [
        "/usr/local/lib",
        "/usr/lib",
        "/opt",
    ]
    # 也通过 Python 路径查找
    try:
        import vllm
        vllm_path = Path(vllm.__path__[0])
        search_dirs.append(str(vllm_path.parent))
    except ImportError:
        pass
    try:
        import sglang
        sgl_path = Path(sglang.__path__[0])
        search_dirs.append(str(sgl_path.parent))
    except ImportError:
        pass

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        for py_file in search_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if "flag_gems" in content:
                    candidates.append(str(py_file))
            except (PermissionError, OSError):
                continue

    return sorted(set(candidates))


def get_file_status(filepath):
    """检查单个文件的 FlagGems 状态"""
    try:
        content = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        return {"file": filepath, "error": str(e)}

    lines = content.split("\n")
    active_lines = []
    commented_lines = []

    for i, line in enumerate(lines, 1):
        matched = False
        for pat in COMMENTED_PATTERNS:
            if pat.match(line):
                commented_lines.append({"line": i, "content": line.strip()})
                matched = True
                break
        if not matched:
            for pat in FLAGGEMS_PATTERNS:
                if pat.match(line):
                    active_lines.append({"line": i, "content": line.strip()})
                    break

    status = "unknown"
    if active_lines and not commented_lines:
        status = "enabled"
    elif commented_lines and not active_lines:
        status = "disabled"
    elif active_lines and commented_lines:
        status = "mixed"
    elif not active_lines and not commented_lines:
        status = "not_found"

    has_backup = Path(filepath + BACKUP_SUFFIX).exists()

    return {
        "file": filepath,
        "status": status,
        "active_lines": active_lines,
        "commented_lines": commented_lines,
        "has_backup": has_backup,
    }


def backup_file(filepath):
    """备份文件"""
    backup_path = filepath + BACKUP_SUFFIX
    shutil.copy2(filepath, backup_path)
    return backup_path


def disable_flaggems(filepath):
    """注释掉 FlagGems 相关代码"""
    content = Path(filepath).read_text(encoding="utf-8")
    lines = content.split("\n")
    modified = False

    new_lines = []
    for line in lines:
        commented = False
        for pat in FLAGGEMS_PATTERNS:
            match = pat.match(line)
            if match:
                indent = match.group(1)
                code = match.group(2)
                new_lines.append(f"{indent}# {code}")
                commented = True
                modified = True
                break
        if not commented:
            new_lines.append(line)

    if modified:
        backup_file(filepath)
        Path(filepath).write_text("\n".join(new_lines), encoding="utf-8")

    return modified


def enable_flaggems(filepath):
    """取消注释 FlagGems 相关代码"""
    content = Path(filepath).read_text(encoding="utf-8")
    lines = content.split("\n")
    modified = False

    new_lines = []
    for line in lines:
        uncommented = False
        for pat in COMMENTED_PATTERNS:
            match = pat.match(line)
            if match:
                indent = match.group(1)
                code = match.group(2)
                new_lines.append(f"{indent}{code}")
                uncommented = True
                modified = True
                break
        if not uncommented:
            new_lines.append(line)

    if modified:
        backup_file(filepath)
        Path(filepath).write_text("\n".join(new_lines), encoding="utf-8")

    return modified


def rollback_file(filepath):
    """从备份恢复文件"""
    backup_path = filepath + BACKUP_SUFFIX
    if not Path(backup_path).exists():
        return False
    shutil.copy2(backup_path, filepath)
    return True


def verify_change(filepath, expected_status):
    """验证修改后状态是否正确"""
    status = get_file_status(filepath)
    return status["status"] == expected_status


def analyze_flaggems_code():
    """分析所有含 flag_gems 的文件，提取 enable() 调用和 txt 路径"""
    files = find_model_runner_files()
    result = {
        "files": files,
        "enable_calls": [],
        "gems_txt_path": None,
        "auto_detect_needed": False,
    }

    if not files:
        result["auto_detect_needed"] = True
        return result

    enable_pattern = re.compile(r"flag_gems\.\w*enable\w*\s*\(")

    for filepath in files:
        try:
            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if enable_pattern.search(line):
                # 拼接多行调用（到闭合括号）
                call_text = line
                paren_depth = line.count("(") - line.count(")")
                for j in range(i, min(i + 10, len(lines))):
                    if paren_depth <= 0:
                        break
                    call_text += "\n" + lines[j]
                    paren_depth += lines[j].count("(") - lines[j].count(")")

                call_stripped = call_text.strip()
                txt_path = _extract_txt_path(call_stripped)

                entry = {
                    "file": filepath,
                    "line": i,
                    "call": call_stripped,
                    "txt_path": txt_path,
                }
                result["enable_calls"].append(entry)

                if txt_path and not result["gems_txt_path"]:
                    result["gems_txt_path"] = txt_path

    if not result["gems_txt_path"]:
        result["auto_detect_needed"] = True

    return result


def _extract_txt_path(call_content):
    """从 flag_gems.enable() 调用中提取 txt 文件路径"""
    patterns = [
        # 关键字参数: unused="/root/gems.txt", record_log="/tmp/gems.txt"
        r"""(?:unused|record_log|log_file|output)\s*=\s*["']([^"']*\.txt)["']""",
        # 位置参数中的 .txt 路径
        r"""["'](/[^"']*\.txt)["']""",
    ]
    for pattern in patterns:
        m = re.search(pattern, call_content)
        if m:
            return m.group(1)
    return None


def find_gems_txt_files():
    """在容器内搜索 FlagGems 生成的算子 txt 文件"""
    import subprocess as sp

    search_dirs = ["/root", "/tmp", "/opt", "/var/tmp"]
    # 也搜索 flag_gems 包目录
    try:
        import flag_gems
        search_dirs.append(os.path.dirname(flag_gems.__file__))
    except ImportError:
        pass

    # 常见算子名模式（用于匹配 txt 文件内容）
    op_keywords = ["aten::", "torch.", "addmm", "softmax", "layer_norm", "rms_norm",
                   "mm", "bmm", "cross_entropy", "gelu", "silu", "relu"]

    found_files = []
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        try:
            # 搜索 .txt 文件
            proc = sp.run(
                f"find {search_dir} -maxdepth 3 -name '*.txt' -size +0c 2>/dev/null",
                shell=True, capture_output=True, text=True, timeout=10
            )
            for fpath in proc.stdout.strip().split("\n"):
                fpath = fpath.strip()
                if not fpath or not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(4096)
                    # 检查是否包含算子名模式
                    matches = sum(1 for kw in op_keywords if kw in content)
                    if matches >= 2:
                        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
                        found_files.append({
                            "path": fpath,
                            "line_count": len(lines),
                            "sample_lines": lines[:5],
                            "keyword_matches": matches,
                        })
                except Exception:
                    continue
        except Exception:
            continue

    # 按匹配度排序
    found_files.sort(key=lambda x: x["keyword_matches"], reverse=True)

    recommended = found_files[0]["path"] if found_files else None

    return {
        "found_files": found_files,
        "recommended": recommended,
    }


OPS_CONTROL_FILE = "/root/flaggems_ops_control.json"
FLAGGEMS_INJECT_MARKER = "FLAGGEMS_CONTROL_MODE"


def _write_ops_control_file(enabled_ops=None, disabled_ops=None):
    """写入算子控制文件，配合注入的环境变量分支代码使用"""
    data = {}
    if enabled_ops is not None:
        data["include"] = sorted(enabled_ops)
    if disabled_ops is not None:
        data["unused"] = sorted(disabled_ops)
    with open(OPS_CONTROL_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 算子控制文件已写入: {OPS_CONTROL_FILE}")
    return True


ETC_ENVIRONMENT = "/etc/environment"
BASHRC = "/root/.bashrc"
FLAGGEMS_BASHRC_MARKER = "# === FlagGems 算子配置（自动生成，勿手动修改）==="
FLAGGEMS_BASHRC_END = "# === FlagGems 算子配置结束 ==="


def _persist_env_vars(env_vars):
    """持久化环境变量到 /etc/environment + /root/.bashrc"""
    # 同步到当前进程
    for k, v in env_vars.items():
        os.environ[k] = v

    # /etc/environment
    try:
        existing = ""
        if os.path.isfile(ETC_ENVIRONMENT):
            with open(ETC_ENVIRONMENT, 'r', encoding='utf-8') as f:
                existing = f.read()
        lines = [l for l in existing.split('\n')
                 if not any(l.startswith(k + "=") for k in env_vars)]
        for k, v in env_vars.items():
            lines.append(f"{k}={v}")
        with open(ETC_ENVIRONMENT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l for l in lines if l is not None) + '\n')
        print(f"  ✓ {ETC_ENVIRONMENT} 已更新")
    except Exception as e:
        print(f"  WARN: 写入 {ETC_ENVIRONMENT} 失败: {e}")

    # /root/.bashrc
    try:
        existing = ""
        if os.path.isfile(BASHRC):
            with open(BASHRC, 'r', encoding='utf-8') as f:
                existing = f.read()
        if FLAGGEMS_BASHRC_MARKER in existing:
            pattern = re.compile(
                re.escape(FLAGGEMS_BASHRC_MARKER) + r".*?" + re.escape(FLAGGEMS_BASHRC_END),
                re.DOTALL
            )
            existing = pattern.sub("", existing).strip()
        block = f"\n{FLAGGEMS_BASHRC_MARKER}\n"
        for k, v in env_vars.items():
            block += f"export {k}={v}\n"
        block += f"{FLAGGEMS_BASHRC_END}\n"
        with open(BASHRC, 'w', encoding='utf-8') as f:
            f.write(existing + block)
        print(f"  ✓ {BASHRC} 已更新")
    except Exception as e:
        print(f"  WARN: 写入 {BASHRC} 失败: {e}")


def _compute_enabled_from_disabled(disabled_ops):
    """从 disabled_ops 反推 enabled_ops（需要全量算子列表）"""
    all_ops = None
    candidates = ["/tmp/flaggems_enable_oplist.txt", "/root/gems.txt", "/tmp/gems.txt"]
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    ops = [l.strip() for l in f if l.strip()]
                if ops:
                    all_ops = ops
                    print(f"  ✓ 全量算子列表来源: {path} ({len(ops)} 个)")
                    break
            except Exception:
                continue
    if not all_ops:
        try:
            import flag_gems
            if hasattr(flag_gems, "all_registered_ops"):
                all_ops = list(flag_gems.all_registered_ops())
            elif hasattr(flag_gems, "all_ops"):
                all_ops = list(flag_gems.all_ops())
            if all_ops:
                print(f"  ✓ 全量算子列表来源: flag_gems API ({len(all_ops)} 个)")
        except Exception:
            pass
    # 静态 fallback：从 flag_gems 包的 OPS_REGISTRY 或 __init__ 中提取
    if not all_ops:
        try:
            import flag_gems
            # 尝试从 flag_gems 的内部注册表获取
            for attr in ("OPS_REGISTRY", "_ops_registry", "REGISTERED_OPS", "_registered_ops"):
                reg = getattr(flag_gems, attr, None)
                if reg and hasattr(reg, '__iter__'):
                    all_ops = sorted(set(str(k) for k in reg))
                    if all_ops:
                        print(f"  ✓ 全量算子列表来源: flag_gems.{attr} ({len(all_ops)} 个)")
                        break
            # 尝试从 enable 函数的默认参数或文档中提取
            if not all_ops and hasattr(flag_gems, 'enable'):
                import inspect as _insp
                src = _insp.getsource(flag_gems.enable)
                # 搜索源码中的算子列表定义
                op_match = re.findall(r'"(\w+)"', src)
                if len(op_match) > 5:
                    all_ops = sorted(set(op_match))
                    print(f"  ✓ 全量算子列表来源: flag_gems.enable() 源码解析 ({len(all_ops)} 个)")
        except Exception:
            pass
    if not all_ops:
        print("  ✗ 无法获取全量算子列表（运行时文件为空/不存在，flag_gems API 无可用接口）")
        return None
    return sorted(set(all_ops) - set(disabled_ops))


def _replace_enable_call_balanced(content, replacement):
    """使用括号平衡匹配替换 flag_gems.enable() 调用

    处理嵌套括号场景，如 flag_gems.enable(unused=get_list())
    """
    pattern = r"flag_gems\.(?:only_)?enable\s*\("
    result = []
    pos = 0
    count = 0

    for match in re.finditer(pattern, content):
        result.append(content[pos:match.start()])

        # 从左括号开始计数，找到匹配的右括号
        start = match.end()
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
            i += 1

        if depth == 0:
            # 找到匹配的右括号，替换整个调用
            result.append(replacement)
            pos = i
            count += 1
        else:
            # 未找到匹配（不应该发生），保留原文
            result.append(content[match.start():start])
            pos = start

    result.append(content[pos:])
    return ''.join(result), count


def modify_enable_call(files, enabled_ops=None, disabled_ops=None):
    """修改 flag_gems.enable() 调用以控制算子子集（算子优化用）

    如果源码已注入环境变量驱动代码（含 FLAGGEMS_CONTROL_MODE），
    只写控制文件 + 设置环境变量，不再修改源码。

    未注入时保留原有正则替换逻辑（兼容）。
    """
    # 探测 capabilities
    caps = []
    try:
        import flag_gems
        if hasattr(flag_gems, "only_enable"):
            caps.append("only_enable")
        if hasattr(flag_gems, "enable"):
            import inspect as insp_mod
            sig = insp_mod.signature(flag_gems.enable)
            if "unused" in list(sig.parameters.keys()):
                caps.append("enable_unused")
    except ImportError:
        pass

    if not files:
        files = find_model_runner_files()

    results = []
    for filepath in files:
        try:
            content = Path(filepath).read_text(encoding="utf-8")
        except Exception as e:
            results.append({"file": filepath, "success": False, "error": str(e)})
            continue

        # 已注入环境变量驱动代码 → 只写控制文件 + 设环境变量，不改源码
        if FLAGGEMS_INJECT_MARKER in content:
            if disabled_ops:
                # 有禁用算子 → only_enable 白名单模式（从 txt 全量列表减去禁用）
                if not enabled_ops:
                    enabled_ops = _compute_enabled_from_disabled(disabled_ops)
                if enabled_ops:
                    control_mode = "only_enable"
                    _write_ops_control_file(enabled_ops=enabled_ops, disabled_ops=None)
                elif "enable_unused" in caps:
                    # fallback: 全量列表不可用但 enable(unused=) 可用
                    control_mode = "unused"
                    _write_ops_control_file(enabled_ops=None, disabled_ops=disabled_ops)
                    print(f"  ⚠ 全量算子列表不可用，降级到 unused 模式（禁用 {disabled_ops}）")
                else:
                    # 最终 fallback: 既无全量列表也无 unused 能力，写空控制文件让 enable() 全量启动
                    control_mode = "unused"
                    _write_ops_control_file(enabled_ops=None, disabled_ops=disabled_ops)
                    print(f"  ⚠ 全量算子列表不可用且无 unused 能力，算子控制可能不生效")
            else:
                # 全量开启 → unused 模式（无禁用）
                control_mode = "unused"
                _write_ops_control_file(enabled_ops=None, disabled_ops=[])
            os.environ["FLAGGEMS_CONTROL_MODE"] = control_mode
            # modify-enable 隐含 FlagGems 应开启，确保 USE_FLAGGEMS=1
            _persist_env_vars({"FLAGGEMS_CONTROL_MODE": control_mode, "USE_FLAGGEMS": "1"})
            results.append({
                "file": filepath,
                "method": f"env_control({control_mode})",
                "success": True,
            })
            continue

        # 未注入 → 保留正则替换逻辑（兼容）
        original = content
        modified = False
        method = "unknown"

        # 读取环境变量决定控制模式
        # 规则：有禁用算子 → only_enable（白名单）；全开 → unused
        control_mode = os.environ.get("FLAGGEMS_CONTROL_MODE", "")
        if not control_mode:
            if disabled_ops and "only_enable" in caps:
                control_mode = "only_enable"
            elif not disabled_ops and "enable_unused" in caps:
                control_mode = "unused"
            elif "only_enable" in caps:
                control_mode = "only_enable"
            elif "enable_unused" in caps:
                control_mode = "unused"

        # 分支1: unused 模式
        if control_mode == "unused" and "enable_unused" in caps:
            if disabled_ops is not None:
                ops_str = ", ".join(f'"{op}"' for op in sorted(disabled_ops))
            else:
                ops_str = ""
            content, count = re.subn(
                r"flag_gems\.(?:only_)?enable\s*\([^)]*\)",
                f"flag_gems.enable(unused=[{ops_str}])",
                content
            )
            if count == 0:
                content, count = _replace_enable_call_balanced(
                    content, f"flag_gems.enable(unused=[{ops_str}])")
            modified = count > 0
            method = "enable_unused"

        # 分支2: only_enable 模式
        elif control_mode == "only_enable" and "only_enable" in caps:
            if enabled_ops is None and disabled_ops is not None:
                enabled_ops = _compute_enabled_from_disabled(disabled_ops)
            if enabled_ops is not None and len(enabled_ops) > 0:
                ops_str = ", ".join(f'"{op}"' for op in sorted(enabled_ops))
                content, count = re.subn(
                    r"flag_gems\.(?:only_)?enable\s*\([^)]*\)",
                    f"flag_gems.only_enable(include=[{ops_str}])",
                    content
                )
                if count == 0:
                    content, count = _replace_enable_call_balanced(
                        content, f"flag_gems.only_enable(include=[{ops_str}])")
                modified = count > 0
                method = "only_enable"
            elif "enable_unused" in caps:
                if disabled_ops is not None:
                    ops_str = ", ".join(f'"{op}"' for op in sorted(disabled_ops))
                else:
                    ops_str = ""
                content, count = re.subn(
                    r"flag_gems\.(?:only_)?enable\s*\([^)]*\)",
                    f"flag_gems.enable(unused=[{ops_str}])",
                    content
                )
                if count == 0:
                    content, count = _replace_enable_call_balanced(
                        content, f"flag_gems.enable(unused=[{ops_str}])")
                modified = count > 0
                method = "enable_unused_fallback"

        if modified and content != original:
            backup_path = backup_file(filepath)
            Path(filepath).write_text(content, encoding="utf-8")
            results.append({
                "file": filepath,
                "method": method,
                "backup": backup_path,
                "success": True,
            })
        elif not modified:
            method = "txt_fallback"
            analysis = analyze_flaggems_code()
            txt_path = analysis.get("gems_txt_path")
            if txt_path and enabled_ops is not None:
                with open(txt_path, "w", encoding="utf-8") as f:
                    for op in sorted(enabled_ops):
                        f.write(f"{op}\n")
                results.append({
                    "file": txt_path,
                    "method": method,
                    "success": True,
                })
            else:
                results.append({
                    "file": filepath,
                    "method": method,
                    "success": False,
                    "error": "无法确定修改方式或 txt 路径",
                })

    return {
        "action": "modify-enable",
        "capabilities": caps,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="FlagGems 开关切换工具")
    parser.add_argument(
        "--action",
        required=True,
        choices=["enable", "disable", "status", "rollback", "analyze", "find-gems-txt", "modify-enable"],
        help="操作类型",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="指定文件列表（不指定则自动扫描）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式",
    )
    parser.add_argument(
        "--integration-type",
        choices=["auto", "plugin", "code_import"],
        default="auto",
        help="集成方式（auto=自动检测，plugin=环境变量模式）",
    )
    parser.add_argument(
        "--enabled-ops",
        help="modify-enable: 启用的算子列表（逗号分隔）",
    )
    parser.add_argument(
        "--disabled-ops",
        help="modify-enable: 禁用的算子列表（逗号分隔）",
    )
    args = parser.parse_args()

    # 新增 action: analyze
    if args.action == "analyze":
        result = analyze_flaggems_code()
        result["timestamp"] = datetime.now().isoformat()
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFlagGems Analyze")
            print("=" * 50)
            print(f"  扫描文件数: {len(result['files'])}")
            for ec in result["enable_calls"]:
                print(f"  {ec['file']}:L{ec['line']}: {ec['call'][:80]}")
                if ec["txt_path"]:
                    print(f"    → txt 路径: {ec['txt_path']}")
            if result["gems_txt_path"]:
                print(f"\n  推荐 gems_txt_path: {result['gems_txt_path']}")
            elif result["auto_detect_needed"]:
                print(f"\n  未找到 txt 路径，需启动服务后调用 find-gems-txt")
        return

    # 新增 action: find-gems-txt
    if args.action == "find-gems-txt":
        result = find_gems_txt_files()
        result["timestamp"] = datetime.now().isoformat()
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFlagGems TXT 文件搜索")
            print("=" * 50)
            for ff in result["found_files"]:
                print(f"  {ff['path']} ({ff['line_count']} 行, {ff['keyword_matches']} 关键词匹配)")
                for sl in ff["sample_lines"][:3]:
                    print(f"    | {sl}")
            if result["recommended"]:
                print(f"\n  推荐: {result['recommended']}")
            else:
                print(f"\n  未找到匹配的算子 txt 文件")
        return

    # 新增 action: modify-enable
    if args.action == "modify-enable":
        enabled = args.enabled_ops.split(",") if args.enabled_ops else None
        disabled = args.disabled_ops.split(",") if args.disabled_ops else None
        result = modify_enable_call(args.files or [], enabled_ops=enabled, disabled_ops=disabled)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFlagGems Modify Enable")
            print("=" * 50)
            for r in result["results"]:
                status = "OK" if r.get("success") else "FAILED"
                print(f"  {r['file']} → {r.get('method', '?')} [{status}]")
        return

    # 确定集成方式
    is_plugin = False
    if args.integration_type == "plugin":
        is_plugin = True
    elif args.integration_type == "auto":
        is_plugin = detect_plugin_mode()

    # Plugin 模式：生成环境变量字典（不再写文件）
    if is_plugin and args.action in ("enable", "disable"):
        env_vars = generate_env_vars(args.action)
        inline = env_to_inline(env_vars)
        result = {
            "action": args.action,
            "mode": "plugin",
            "env_vars": env_vars,
            "env_inline": inline,
            "success": True,
            "message": f"Plugin 模式: 内联环境变量已生成 ({args.action})",
            "timestamp": datetime.now().isoformat(),
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFlagGems Toggle — {args.action} (Plugin 模式)")
            print("=" * 50)
            print(f"  env_vars: {env_vars}")
            print(f"  env_inline: {inline}")
            print(f"  提示: 在启动命令前添加内联环境变量")
        return

    if is_plugin and args.action == "status":
        # Plugin 模式下检查环境变量
        prefer = os.environ.get("VLLM_FL_PREFER_ENABLED", "not_set")
        use_flaggems = os.environ.get("USE_FLAGGEMS", "not_set")
        result = {
            "mode": "plugin",
            "USE_FLAGGEMS": use_flaggems,
            "VLLM_FL_PREFER_ENABLED": prefer,
            "status": "enabled" if prefer == "true" else ("disabled" if prefer == "false" else "unknown"),
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFlagGems Toggle — status (Plugin 模式)")
            print("=" * 50)
            print(f"  USE_FLAGGEMS: {use_flaggems}")
            print(f"  VLLM_FL_PREFER_ENABLED: {prefer}")
        return

    # 非 plugin 模式

    # 查找文件
    if args.files:
        files = args.files
    else:
        files = find_model_runner_files()

    if not files:
        result = {"success": False, "error": "未找到包含 flag_gems 的文件"}
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("ERROR: 未找到包含 flag_gems 的文件")
        sys.exit(1)

    # 检测是否已注入环境变量驱动代码
    injected = any(
        FLAGGEMS_INJECT_MARKER in Path(f).read_text(encoding='utf-8', errors='ignore')
        for f in files
        if os.path.isfile(f)
    )

    # 已注入场景：通过环境变量 + 控制文件控制，不再修改源码
    if injected and args.action in ("enable", "disable"):
        env_vars = {}
        if args.action == "enable":
            env_vars["USE_FLAGGEMS"] = "1"
            if not os.path.isfile(OPS_CONTROL_FILE):
                _write_ops_control_file(enabled_ops=None, disabled_ops=[])
            # 自动继承已有 control file 的模式，避免丢失 disabled_ops 配置
            if os.path.isfile(OPS_CONTROL_FILE):
                try:
                    with open(OPS_CONTROL_FILE, 'r', encoding='utf-8') as cf:
                        ctrl = json.load(cf)
                    if ctrl.get("include"):
                        env_vars["FLAGGEMS_CONTROL_MODE"] = "only_enable"
                    elif ctrl.get("unused"):
                        env_vars["FLAGGEMS_CONTROL_MODE"] = "unused"
                except (json.JSONDecodeError, OSError):
                    pass
        else:
            env_vars["USE_FLAGGEMS"] = "0"

        # 持久化环境变量到 /etc/environment + /root/.bashrc
        _persist_env_vars(env_vars)

        inline = env_to_inline(env_vars)
        result = {
            "action": args.action,
            "mode": "env_control",
            "env_vars": env_vars,
            "env_inline": inline,
            "control_file": OPS_CONTROL_FILE if args.action == "enable" else None,
            "success": True,
            "message": f"环境变量驱动模式: {args.action} (USE_FLAGGEMS={env_vars['USE_FLAGGEMS']})",
            "timestamp": datetime.now().isoformat(),
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\nFlagGems Toggle — {args.action} (环境变量驱动模式)")
            print("=" * 50)
            print(f"  env_vars: {env_vars}")
            print(f"  env_inline: {inline}")
            if args.action == "enable":
                print(f"  控制文件: {OPS_CONTROL_FILE}")
            print(f"  提示: 在启动命令前添加内联环境变量")
        return

    results = []

    if args.action == "status":
        # status 增加注入状态信息
        for f in files:
            status = get_file_status(f)
            if injected:
                status["injected"] = True
                status["use_flaggems"] = os.environ.get("USE_FLAGGEMS", "not_set")
                status["control_mode"] = os.environ.get("FLAGGEMS_CONTROL_MODE", "not_set")
                ctrl_data = {}
                if os.path.isfile(OPS_CONTROL_FILE):
                    try:
                        with open(OPS_CONTROL_FILE, 'r', encoding='utf-8') as cf:
                            ctrl_data = json.load(cf)
                    except Exception:
                        pass
                status["control_file"] = ctrl_data
            results.append(status)

    elif args.action == "disable":
        # 未注入场景：保留原有源码注释逻辑
        for f in files:
            before = get_file_status(f)
            if before.get("status") == "disabled":
                results.append({"file": f, "action": "skip", "reason": "already disabled"})
                continue
            modified = disable_flaggems(f)
            if modified and verify_change(f, "disabled"):
                results.append({"file": f, "action": "disabled", "success": True})
            elif not modified:
                results.append({"file": f, "action": "skip", "reason": "no active lines found"})
            else:
                results.append({"file": f, "action": "disabled", "success": False, "warning": "verification failed"})

    elif args.action == "enable":
        # 未注入场景：保留原有源码取消注释逻辑
        for f in files:
            before = get_file_status(f)
            if before.get("status") == "enabled":
                results.append({"file": f, "action": "skip", "reason": "already enabled"})
                continue
            modified = enable_flaggems(f)
            if modified and verify_change(f, "enabled"):
                results.append({"file": f, "action": "enabled", "success": True})
            elif not modified:
                results.append({"file": f, "action": "skip", "reason": "no commented lines found"})
            else:
                results.append({"file": f, "action": "enabled", "success": False, "warning": "verification failed"})

    elif args.action == "rollback":
        for f in files:
            if rollback_file(f):
                results.append({"file": f, "action": "rollback", "success": True})
            else:
                results.append({"file": f, "action": "rollback", "success": False, "reason": "no backup found"})

    # 输出
    output = {
        "action": args.action,
        "files_processed": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"\nFlagGems Toggle — {args.action}")
        print("=" * 50)
        for r in results:
            action = r.get("action", r.get("status", "?"))
            success = r.get("success", "")
            reason = r.get("reason", "")
            warning = r.get("warning", "")
            extra = ""
            if reason:
                extra = f" ({reason})"
            if warning:
                extra = f" [WARNING: {warning}]"
            if success is True:
                extra = " [OK]"
            elif success is False:
                extra = f" [FAILED]{extra}"

            # status action has different format
            if args.action == "status":
                status = r.get("status", "?")
                active = len(r.get("active_lines", []))
                commented = len(r.get("commented_lines", []))
                backup = "有备份" if r.get("has_backup") else "无备份"
                print(f"  {r['file']}")
                print(f"    状态: {status}  活跃行: {active}  注释行: {commented}  {backup}")
                for al in r.get("active_lines", []):
                    print(f"    L{al['line']}: {al['content']}")
                for cl in r.get("commented_lines", []):
                    print(f"    L{cl['line']}: {cl['content']}")
            else:
                print(f"  {r['file']} → {action}{extra}")

        print(f"\n处理文件数: {len(results)}")


if __name__ == "__main__":
    main()
