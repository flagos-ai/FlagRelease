#!/usr/bin/env python3
"""
inspect_env.py — 合并环境检查脚本

一次运行完成全部环境检查，替代原来 10+ 次 docker exec 串行执行。
输出结构化 JSON，可直接写入 context.yaml。

Usage:
    python3 inspect_env.py --output-json    # 输出 JSON（供程序读取）
    python3 inspect_env.py --report         # 输出人类可读报告
    python3 inspect_env.py                  # 同时输出 JSON 和报告
"""

import argparse
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_best_python():
    """探测最佳 Python 解释器（优先 conda/venv）"""
    candidates = [
        "/opt/conda/bin/python3",
        os.path.expanduser("~/miniconda3/bin/python3"),
        os.path.expanduser("~/anaconda3/bin/python3"),
    ]
    # 检查 PATH 中是否有更高优先级的 python3
    for c in candidates:
        if os.path.isfile(c):
            return c
    return sys.executable


# 如果当前解释器不是最佳的，用最佳解释器重新执行自身
if __name__ == '__main__' and not os.environ.get('_INSPECT_ENV_REEXEC'):
    best = find_best_python()
    if best != sys.executable and os.path.isfile(best):
        os.environ['_INSPECT_ENV_REEXEC'] = '1'
        try:
            os.execv(best, [best] + sys.argv)
        except OSError as e:
            print(f"[WARN] execv({best}) 失败: {e}，使用当前解释器继续", file=sys.stderr)


def run_cmd(cmd, timeout=30):
    """运行 shell 命令并返回 stdout"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def check_execution_mode():
    """检测是否在容器内运行"""
    if os.path.exists("/.dockerenv"):
        return "container"
    try:
        with open("/proc/1/cgroup", "r") as f:
            if "docker" in f.read():
                return "container"
    except Exception:
        pass
    return "host"


def check_core_packages():
    """检查核心组件版本"""
    packages = {}
    for pkg_name, import_name in [("torch", "torch"), ("vllm", "vllm"), ("sglang", "sglang")]:
        try:
            mod = importlib.import_module(import_name)
            packages[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg_name] = None
    # torch CUDA version
    try:
        import torch
        packages["torch_cuda"] = torch.version.cuda if hasattr(torch.version, "cuda") else None
    except Exception:
        packages["torch_cuda"] = None
    return packages


def check_flag_packages():
    """检查 flag 生态组件版本"""
    packages = {}
    for pkg_name, import_name in [
        ("flaggems", "flag_gems"),
        ("flagscale", "flag_scale"),
        ("flagcx", "flagcx"),
        ("vllm_plugin", "vllm_fl"),
    ]:
        try:
            mod = importlib.import_module(import_name)
            packages[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg_name] = None
    return packages


def probe_flaggems_capabilities():
    """探测 FlagGems 运行时能力"""
    result = {
        "flaggems_installed": False,
        "capabilities": [],
        "enable_signature": "",
        "enable_params": [],
        "vendor_config_path": "",
        "vllm_plugin_installed": False,
        "plugin_has_dispatch": False,
        "probe_error": "",
        "gpu_compute_capability": "",
        "gpu_arch": "",
        "plugin_env_vars": {},
        "plugin_control": {},
        "oot_ops": [],
    }

    # GPU compute capability 探测
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            result["gpu_compute_capability"] = f"{major}.{minor}"
            result["gpu_arch"] = f"sm_{major}{minor}"
    except Exception:
        pass

    # Plugin dispatch 环境变量探测
    for var in ["VLLM_FL_FLAGOS_WHITELIST", "VLLM_FL_FLAGOS_BLACKLIST",
                "VLLM_FL_OOT_WHITELIST", "VLLM_FL_OOT_BLACKLIST",
                "VLLM_FL_PREFER_ENABLED", "VLLM_FL_OOT_ENABLED",
                "VLLM_FL_PER_OP", "VLLM_FL_DISPATCH_MODE",
                "VLLM_FL_DISPATCH_DEBUG",
                "VLLM_USE_DEEP_GEMM"]:
        val = os.environ.get(var)
        if val is not None:
            result["plugin_env_vars"][var] = val

    # 探测 FlagGems
    try:
        import flag_gems

        result["flaggems_installed"] = True

        # enable() 签名
        if hasattr(flag_gems, "enable"):
            sig = inspect.signature(flag_gems.enable)
            result["enable_signature"] = str(sig)
            params = list(sig.parameters.keys())
            result["enable_params"] = params
            if "unused" in params:
                result["capabilities"].append("enable_unused")

        # only_enable()
        if hasattr(flag_gems, "only_enable"):
            result["capabilities"].append("only_enable")

        # use_gems 上下文管理器
        if hasattr(flag_gems, "use_gems"):
            result["capabilities"].append("use_gems")
            try:
                sig = inspect.signature(flag_gems.use_gems)
                params = list(sig.parameters.keys())
                if "include" in params or "exclude" in params:
                    result["capabilities"].append("use_gems_filter")
            except Exception:
                pass

        # YAML 配置支持
        if hasattr(flag_gems, "config"):
            cfg = flag_gems.config
            if hasattr(cfg, "resolve_user_setting"):
                result["capabilities"].append("yaml_config")
            if hasattr(cfg, "get_default_enable_config"):
                result["capabilities"].append("vendor_default")
                try:
                    path = cfg.get_default_enable_config()
                    result["vendor_config_path"] = str(path) if path else ""
                except Exception:
                    pass

        # 算子查询接口
        if hasattr(flag_gems, "all_registered_ops"):
            result["capabilities"].append("query_ops")
        elif hasattr(flag_gems, "all_ops"):
            result["capabilities"].append("query_ops_legacy")

    except ImportError:
        pass
    except Exception as e:
        result["probe_error"] = str(e)

    # 探测 vllm-plugin-FL
    try:
        import vllm_fl

        result["vllm_plugin_installed"] = True
        try:
            from vllm_fl.dispatch import OpManager
            result["plugin_has_dispatch"] = True
        except ImportError:
            pass

        # 探测 OOT 算子列表
        try:
            from vllm_fl.ops import oot as oot_module
            oot_ops = [name for name in dir(oot_module)
                       if not name.startswith('_') and callable(getattr(oot_module, name, None))]
            result["oot_ops"] = oot_ops
        except (ImportError, AttributeError):
            # 兜底：使用已知的 OOT 算子列表
            result["oot_ops"] = [
                "silu_and_mul", "rms_norm", "rotary_embedding",
                "fused_moe", "attention_backend",
            ]

        # 构建 plugin_control 信息
        result["plugin_control"] = {
            "prefer_enabled": os.environ.get("VLLM_FL_PREFER_ENABLED", "not_set"),
            "oot_enabled": os.environ.get("VLLM_FL_OOT_ENABLED", "not_set"),
            "oot_ops": result["oot_ops"],
            "flagos_whitelist": os.environ.get("VLLM_FL_FLAGOS_WHITELIST", ""),
            "flagos_blacklist": os.environ.get("VLLM_FL_FLAGOS_BLACKLIST", ""),
            "oot_blacklist": os.environ.get("VLLM_FL_OOT_BLACKLIST", ""),
            "dispatch_mode": os.environ.get("VLLM_FL_DISPATCH_MODE", ""),
        }
    except ImportError:
        pass

    return result


def scan_flaggems_integration():
    """多维度扫描 FlagGems 集成方式"""
    integration = {
        "env_vars": {},
        "code_locations": [],
        "entry_points": [],
        "startup_scripts": [],
        "integration_type": "unknown",
        "enable_method": "",
        "disable_method": "",
    }

    # 维度1：环境变量检查
    for var in ["USE_FLAGGEMS", "USE_FLAGOS", "FLAGGEMS_LOG_LEVEL", "ENABLE_FLAGGEMS"]:
        val = os.environ.get(var)
        if val is not None:
            integration["env_vars"][var] = val

    # 维度2：vllm/sglang 代码扫描
    for framework in ["vllm", "sglang"]:
        try:
            mod = importlib.import_module(framework)
            fw_path = mod.__path__[0]
            output = run_cmd(
                f"grep -rn 'flag_gems\\|flaggems\\|use_gems\\|enable.*gems\\|import.*gems' {fw_path}/ 2>/dev/null"
            )
            if output:
                for line in output.strip().split("\n"):
                    if line:
                        integration["code_locations"].append(line)
        except (ImportError, Exception):
            pass

    # 维度3：入口点扫描
    try:
        import pkg_resources
        for group in ["vllm.general_plugins", "vllm.platform_plugins"]:
            for ep in pkg_resources.iter_entry_points(group):
                integration["entry_points"].append(f"{group}: {ep.name} = {ep}")
    except Exception:
        pass

    # 维度4：启动脚本扫描
    output = run_cmd(
        "find /usr/local/bin /opt /root -name '*.sh' -exec grep -l 'gems\\|flagos\\|flag_gems' {} \\; 2>/dev/null"
    )
    if output:
        integration["startup_scripts"] = [s for s in output.strip().split("\n") if s]

    # 推导集成方式
    _derive_integration_methods(integration)

    return integration


def _derive_integration_methods(integration):
    """根据扫描结果推导 FlagGems 启用/关闭方法"""
    code_locs = integration["code_locations"]
    env_vars = integration["env_vars"]
    entry_points = integration["entry_points"]

    # 优先级1：环境变量控制
    for var in ["USE_FLAGGEMS", "USE_FLAGOS"]:
        if var in env_vars:
            integration["integration_type"] = "env_var"
            integration["enable_method"] = f"env:{var}=1"
            integration["disable_method"] = f"env:{var}=0"
            return
    # 检查代码中是否引用了这些环境变量
    for loc in code_locs:
        for var in ["USE_FLAGGEMS", "USE_FLAGOS"]:
            if var in loc:
                integration["integration_type"] = "env_var"
                integration["enable_method"] = f"env:{var}=1"
                integration["disable_method"] = f"env:{var}=0"
                return

    # 优先级2：插件入口点
    if entry_points:
        integration["integration_type"] = "plugin"
        integration["enable_method"] = "auto"
        integration["disable_method"] = "env:USE_FLAGGEMS=0"
        return

    # 优先级3：代码中直接 import
    if code_locs:
        # 解析具体的代码位置
        import_locs = []
        for loc in code_locs:
            match = re.match(r"^(.+):(\d+):(.+)$", loc)
            if match:
                filepath, lineno, content = match.groups()
                if "import flag_gems" in content or "flag_gems.enable" in content:
                    import_locs.append({"file": filepath, "line": int(lineno), "content": content.strip()})

        if import_locs:
            integration["integration_type"] = "code_import"
            # 提供代码文件列表供 toggle_flaggems.py 使用
            files = list(set(loc["file"] for loc in import_locs))
            integration["enable_method"] = f"code:uncomment:{json.dumps(files)}"
            integration["disable_method"] = f"code:comment:{json.dumps(files)}"
            integration["code_import_details"] = import_locs
            return

    # 优先级4：启动脚本
    if integration["startup_scripts"]:
        integration["integration_type"] = "script"
        integration["enable_method"] = f"script:{integration['startup_scripts'][0]}"
        integration["disable_method"] = f"script:{integration['startup_scripts'][0]}"
        return

    # 无法确定
    integration["integration_type"] = "unknown"
    integration["enable_method"] = "unknown"
    integration["disable_method"] = "unknown"


def check_env_vars():
    """列出所有 flag 相关环境变量"""
    result = {}
    for key, val in os.environ.items():
        if re.search(r"flag|gems|flagos", key, re.IGNORECASE):
            result[key] = val
    return result


def classify_env_type(capabilities, integration):
    """根据 flaggems 和 plugin 安装情况分类环境场景

    Returns:
        str: native | vllm_flaggems | vllm_plugin_flaggems
    """
    flaggems_installed = capabilities.get("flaggems_installed", False)
    plugin_installed = capabilities.get("vllm_plugin_installed", False)

    if not flaggems_installed:
        return "native"
    elif plugin_installed:
        return "vllm_plugin_flaggems"
    else:
        return "vllm_flaggems"


def extract_flaggems_code_details(integration):
    """从代码扫描结果中提取 flag_gems.enable() 调用详情（仅 vllm_flaggems 场景）

    解析 code_locations 中的 flag_gems.enable(...) 调用，提取：
    - 所有包含 enable()/only_enable() 的文件路径列表
    - 每个文件的 enable() 完整调用
    - txt 文件路径参数（如 unused="/root/gems.txt"）

    Returns:
        dict: {
            code_paths: [{"file": str, "enable_call": str, "priority": int}],
            txt_path: str,
            auto_detect: bool
        }
    """
    result = {
        "code_paths": [],
        "txt_path": "",
        "auto_detect": False,
    }

    code_locs = integration.get("code_locations", [])
    if not code_locs:
        result["auto_detect"] = True
        return result

    # 从 code_locations 中找 flag_gems.enable/only_enable 调用
    enable_locs = []
    import_locs = []
    for loc in code_locs:
        match = re.match(r"^(.+):(\d+):(.+)$", loc)
        if not match:
            continue
        filepath, lineno, content = match.groups()
        content_stripped = content.strip()

        if ("flag_gems.enable" in content_stripped or "flag_gems.only_enable" in content_stripped) and "import" not in content_stripped:
            enable_locs.append({
                "file": filepath,
                "line": int(lineno),
                "content": content_stripped,
            })
        elif "import flag_gems" in content_stripped or "from flag_gems" in content_stripped:
            import_locs.append({
                "file": filepath,
                "line": int(lineno),
                "content": content_stripped,
            })

    if not enable_locs:
        result["auto_detect"] = True
        return result

    # 按文件分组 enable 调用
    files_with_enable = {}
    for loc in enable_locs:
        filepath = loc["file"]
        if filepath not in files_with_enable:
            files_with_enable[filepath] = []
        files_with_enable[filepath].append(loc)

    # 为每个文件确定优先级和提取调用详情
    for filepath, locs in files_with_enable.items():
        # 优先级判断：包含 .enable( 且不在条件块内的文件优先级最高
        priority = 0
        enable_call = locs[0]["content"]

        # 读取文件检查是否在条件块内
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # 检查第一个 enable 调用的上下文
            loc = locs[0]
            start_idx = max(0, loc["line"] - 10)
            context_lines = lines[start_idx:loc["line"]]

            # 如果前面有 if/try 等条件语句，降低优先级
            has_condition = any(
                re.match(r'^\s*(if|try|with|for|while)\s+', line)
                for line in context_lines
            )

            # 提取完整的 enable 调用（可能跨多行）
            call_start = loc["line"] - 1
            call_text = ""
            paren_depth = 0
            for i in range(call_start, min(call_start + 10, len(lines))):
                call_text += lines[i]
                paren_depth += lines[i].count("(") - lines[i].count(")")
                if paren_depth <= 0 and "(" in call_text:
                    break
            if call_text.strip():
                enable_call = call_text.strip()

            # 优先级：无条件 enable() > 条件 enable() > only_enable()
            if not has_condition and ".enable(" in enable_call:
                priority = 3
            elif ".enable(" in enable_call:
                priority = 2
            else:
                priority = 1

        except Exception:
            priority = 1

        result["code_paths"].append({
            "file": filepath,
            "enable_call": enable_call,
            "priority": priority
        })

    # 按优先级排序（高优先级在前）
    result["code_paths"].sort(key=lambda x: x["priority"], reverse=True)

    # 从第一个（最高优先级）调用中提取 txt 路径
    if result["code_paths"]:
        enable_call = result["code_paths"][0]["enable_call"]

    # 从所有 enable 调用中提取 txt 路径
    call_content = enable_call if result["code_paths"] else ""

    # 提取字符串参数中的文件路径
    # 匹配引号内的路径（包含 / 的字符串，以 .txt 结尾）
    txt_patterns = [
        # 关键字参数: unused="/root/gems.txt" 或 record_log="/tmp/gems.txt"
        r"""(?:unused|record_log|log_file|output)\s*=\s*["']([^"']*\.txt)["']""",
        # 位置参数: "/root/gems.txt"
        r"""["'](/[^"']*\.txt)["']""",
        # 任何包含路径的字符串参数
        r"""["'](/[^"']+)["']""",
    ]

    for pattern in txt_patterns:
        m = re.search(pattern, call_content)
        if m:
            result["txt_path"] = m.group(1)
            break

    if not result["txt_path"]:
        # 尝试从其他 enable 调用中提取 txt 路径
        for cp in result["code_paths"]:
            for pattern in txt_patterns:
                m = re.search(pattern, cp["enable_call"])
                if m:
                    result["txt_path"] = m.group(1)
                    break
            if result["txt_path"]:
                break

    if not result["txt_path"]:
        result["auto_detect"] = True

    # 向后兼容：code_path 取最高优先级文件
    result["code_path"] = result["code_paths"][0]["file"] if result["code_paths"] else ""
    result["enable_call"] = result["code_paths"][0]["enable_call"] if result["code_paths"] else ""

    return result


# =========================================================================
# 环境变量驱动算子控制：一次性注入 + 环境变量写入
# =========================================================================

OPS_CONTROL_FILE = "/root/flaggems_ops_control.json"
FLAGGEMS_INJECT_MARKER = "FLAGGEMS_CONTROL_MODE"
FLAGGEMS_INJECT_COMMENT = "# FlagGems 环境变量驱动算子控制（由 FlagOS inspect_env 自动注入）"


def _extract_extra_kwargs(call_text):
    """从原始 flag_gems.enable/only_enable 调用中提取 record/once/path 参数"""
    extras = {}
    m = re.search(r'record\s*=\s*(True|False)', call_text)
    if m:
        extras['record'] = m.group(1)
    m = re.search(r'once\s*=\s*(True|False)', call_text)
    if m:
        extras['once'] = m.group(1)
    m = re.search(r'path\s*=\s*["\']([^"\']+)["\']', call_text)
    if m:
        extras['path'] = m.group(1)
    return extras


def _build_inject_block(caps, indent="", extra_kwargs=None):
    """构建注入代码块，保留原始调用中的 record/once/path 参数

    添加 plugin 场景检测：如果存在 VLLM_FL_PREFER_ENABLED 环境变量，
    跳过注入逻辑，让 plugin 的 dispatch 机制接管算子控制。
    """
    has_only_enable = "only_enable" in caps
    extra_kwargs = extra_kwargs or {}

    extra_parts = []
    for k in ('record', 'once'):
        if k in extra_kwargs:
            extra_parts.append(f"{k}={extra_kwargs[k]}")
    if 'path' in extra_kwargs:
        extra_parts.append(f'path="{extra_kwargs["path"]}"')
    extra_str = ", " + ", ".join(extra_parts) if extra_parts else ""

    lines = [
        FLAGGEMS_INJECT_COMMENT,
        "import os as _fgos",
        "# Plugin 场景检测：如果存在 plugin 环境变量，跳过注入逻辑，让 plugin dispatch 接管",
        'if _fgos.environ.get("VLLM_FL_PREFER_ENABLED") is not None:',
        "    pass  # Plugin 场景，跳过注入逻辑",
        'elif _fgos.environ.get("USE_FLAGGEMS", "1") == "1":',
        "    import flag_gems as _fg",
        "    _fg_ops = {}",
        "    try:",
        "        import json as _fgjson",
        f'        with open("{OPS_CONTROL_FILE}", "r") as _fgf:',
        "            _fg_ops = _fgjson.load(_fgf)",
        "    except (FileNotFoundError, Exception):",
        "        pass",
        '    _fg_mode = _fgos.environ.get("FLAGGEMS_CONTROL_MODE", "")',
        "    if not _fg_mode:",
        '        _fg_mode = "only_enable" if _fg_ops.get("include") else "unused"',
    ]
    if has_only_enable:
        lines += [
            '    if _fg_mode == "only_enable" and hasattr(_fg, "only_enable"):',
            '        _fg_inc = _fg_ops.get("include", [])',
            '        if _fg_inc and any(c != c.lower() or " " in c for c in _fg_inc):',
            '            import re as _fgre',
            '            _fg_keys = set(_fg.FULL_CONFIG_BY_FUNC.keys()) if hasattr(_fg, "FULL_CONFIG_BY_FUNC") else set()',
            '            _fg_norm = []',
            '            for _op in _fg_inc:',
            r'                _s = _fgre.sub(r"\s*\(.*?\)", "", _op)',
            '                _s = _s.split(",")[0].strip()',
            r'                _s = _fgre.sub(r"-hopper$", "", _s, flags=_fgre.IGNORECASE)',
            '                _s = _s.replace(".STABLE", "_stable")',
            r'                _s = _fgre.sub(r"\s+FORWARD$", "", _s, flags=_fgre.IGNORECASE)',
            r'                _s = _fgre.sub(r"\s+BACKWARD$", "", _s, flags=_fgre.IGNORECASE)',
            '                _s = _s.lower().replace(" ", "_").lstrip("_")',
            '                if _s in _fg_keys:',
            '                    _fg_norm.append(_s)',
            '                elif _s + "_" in _fg_keys:',
            '                    _fg_norm.append(_s + "_")',
            '                elif _fg_keys:',
            '                    _fg_pfx = [k for k in _fg_keys if k.startswith(_s + "_")]',
            '                    _fg_norm.extend(_fg_pfx) if _fg_pfx else _fg_norm.append(_s)',
            '                else:',
            '                    _fg_norm.append(_s)',
            '            _fg_inc = list(set(_fg_norm))',
            f'        _fg.only_enable(include=_fg_inc{extra_str})',
            "    else:",
            f'        _fg.enable(unused=_fg_ops.get("unused", []){extra_str})',
        ]
    else:
        lines += [
            f'    _fg.enable(unused=_fg_ops.get("unused", []){extra_str})',
        ]

    return "\n".join(indent + line for line in lines)


def _inject_single_file(code_path, caps):
    """对单个文件注入环境变量驱动代码，替换 flag_gems.enable()/only_enable() 调用"""
    if not code_path or not os.path.isfile(code_path):
        return {"injected": False, "file": code_path, "error": "file not found"}

    content = Path(code_path).read_text(encoding="utf-8", errors="ignore")

    if FLAGGEMS_INJECT_MARKER in content:
        return {"injected": True, "already": True, "file": code_path}

    # 匹配 flag_gems.enable(...) 或 flag_gems.only_enable(...)（含多行）
    pattern = re.compile(
        r"^([ \t]*)(flag_gems\.(?:only_)?enable\s*\(.*?\))",
        re.MULTILINE | re.DOTALL
    )
    match = pattern.search(content)
    if not match:
        return {"injected": False, "file": code_path, "error": "flag_gems.enable() pattern not found"}

    indent = match.group(1)
    original_call = match.group(2)
    extra_kwargs = _extract_extra_kwargs(original_call)
    inject_block = _build_inject_block(caps, indent, extra_kwargs=extra_kwargs)

    backup_path = code_path + ".flagos_backup"
    Path(backup_path).write_text(content, encoding="utf-8")

    new_content = content[:match.start()] + inject_block + content[match.end():]
    Path(code_path).write_text(new_content, encoding="utf-8")

    print(f"  ✓ 已注入环境变量驱动代码到 {code_path}")
    print(f"    备份: {backup_path}")
    return {
        "injected": True,
        "file": code_path,
        "backup": backup_path,
    }


def _inject_control_code(code_details, caps):
    """注入环境变量驱动的算子控制代码到所有包含 flag_gems.enable() 的文件"""
    code_paths = code_details.get("code_paths", [])

    # 向后兼容：如果没有 code_paths，使用旧的 code_path
    if not code_paths:
        code_path = code_details.get("code_path", "")
        if code_path:
            code_paths = [{"file": code_path, "enable_call": "", "priority": 1}]

    if not code_paths:
        return {"injected": False, "error": "no code_paths found"}

    results = []
    injected_count = 0
    already_count = 0
    errors = []

    for cp in code_paths:
        filepath = cp["file"]
        r = _inject_single_file(filepath, caps)
        results.append(r)
        if r.get("injected"):
            if r.get("already"):
                already_count += 1
            else:
                injected_count += 1
        elif r.get("error"):
            errors.append(f"{filepath}: {r['error']}")

    # 创建初始控制文件（只需一次）
    if injected_count > 0:
        try:
            import json as _json_init
            with open(OPS_CONTROL_FILE, 'w', encoding='utf-8') as f:
                _json_init.dump({"unused": [], "include": []}, f, indent=2)
            print(f"  ✓ 初始控制文件已创建: {OPS_CONTROL_FILE}")
        except Exception as e:
            print(f"  WARN: 创建控制文件失败: {e}")

    total = injected_count + already_count
    if total == 0:
        return {"injected": False, "error": "; ".join(errors) if errors else "no files injected"}

    # 向后兼容：file 字段取第一个成功注入的文件
    first_file = next((r["file"] for r in results if r.get("injected")), "")
    first_backup = next((r.get("backup", "") for r in results if r.get("injected") and not r.get("already")), "")

    print(f"  ✓ 共处理 {len(code_paths)} 个文件: {injected_count} 新注入, {already_count} 已注入")
    return {
        "injected": True,
        "file": first_file,
        "backup": first_backup,
        "files": [r["file"] for r in results if r.get("injected")],
        "has_only_enable": "only_enable" in caps,
        "control_file": OPS_CONTROL_FILE,
    }


def _write_control_env_vars(env_type, caps):
    """根据环境检测结果写入 FlagGems 控制环境变量到 /etc/environment"""
    if env_type == "native":
        use_flaggems = "0"
        control_mode = ""
    else:
        use_flaggems = "1"
        if "only_enable" in caps and "enable_unused" in caps:
            control_mode = "unused"
        elif "only_enable" in caps:
            control_mode = "only_enable"
        elif "enable_unused" in caps:
            control_mode = "unused"
        else:
            control_mode = ""

    os.environ["USE_FLAGGEMS"] = use_flaggems
    if control_mode:
        os.environ["FLAGGEMS_CONTROL_MODE"] = control_mode

    env_path = "/etc/environment"
    try:
        existing = ""
        if os.path.isfile(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                existing = f.read()
        lines = [l for l in existing.split('\n')
                 if not l.startswith("USE_FLAGGEMS=") and not l.startswith("FLAGGEMS_CONTROL_MODE=")]
        lines.append(f"USE_FLAGGEMS={use_flaggems}")
        if control_mode:
            lines.append(f"FLAGGEMS_CONTROL_MODE={control_mode}")
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l for l in lines if l is not None) + '\n')
        print(f"  ✓ 环境变量已写入 {env_path}: USE_FLAGGEMS={use_flaggems}, FLAGGEMS_CONTROL_MODE={control_mode}")
    except Exception as e:
        print(f"  WARN: 写入 {env_path} 失败: {e}")

    return {"USE_FLAGGEMS": use_flaggems, "FLAGGEMS_CONTROL_MODE": control_mode}


def check_flagtree():
    """检测 FlagTree 安装状态"""
    result = {
        "installed": False,
        "version": "",
        "triton_version": "",
        "backend": "",
    }
    try:
        import triton
        result["triton_version"] = getattr(triton, "__version__", "unknown")
    except ImportError:
        return result

    try:
        import flagtree
        result["installed"] = True
        result["version"] = getattr(flagtree, "__version__", "unknown")
        result["backend"] = getattr(flagtree, "backend", "")
    except ImportError:
        # triton 存在但非 FlagTree
        pass

    return result


def collect_all():
    """收集全部检查结果"""
    exec_mode = check_execution_mode()
    core = check_core_packages()
    flag = check_flag_packages()
    capabilities = probe_flaggems_capabilities()
    integration = scan_flaggems_integration()
    flagtree = check_flagtree()
    env_vars = check_env_vars()

    env_type = classify_env_type(capabilities, integration)
    code_details = extract_flaggems_code_details(integration)
    caps = capabilities["capabilities"]

    # 非 plugin 环境：一次性注入环境变量驱动代码 + 写入控制环境变量
    inject_result = {}
    control_env = {}
    if env_type == "vllm_flaggems":
        inject_result = _inject_control_code(code_details, caps)
        control_env = _write_control_env_vars(env_type, caps)
    elif env_type == "native":
        control_env = _write_control_env_vars(env_type, caps)

    return {
        "execution": {
            "mode": exec_mode,
        },
        "inspection": {
            "core_packages": core,
            "flag_packages": flag,
            "flaggems_capabilities": caps,
            "flaggems_enable_signature": capabilities["enable_signature"],
            "flaggems_enable_params": capabilities["enable_params"],
            "vendor_config_path": capabilities["vendor_config_path"],
            "vllm_plugin_installed": capabilities["vllm_plugin_installed"],
            "plugin_has_dispatch": capabilities["plugin_has_dispatch"],
            "probe_error": capabilities["probe_error"],
            "gpu_compute_capability": capabilities["gpu_compute_capability"],
            "gpu_arch": capabilities["gpu_arch"],
            "plugin_env_vars": capabilities["plugin_env_vars"],
            "plugin_control": capabilities.get("plugin_control", {}),
            "oot_ops": capabilities.get("oot_ops", []),
            "env_vars": env_vars,
        },
        "flagtree": flagtree,
        "flaggems_control": {
            "integration_type": integration["integration_type"],
            "enable_method": integration["enable_method"],
            "disable_method": integration["disable_method"],
            "code_locations": integration["code_locations"],
            "entry_points": integration["entry_points"],
            "startup_scripts": integration["startup_scripts"],
        },
        "env_classification": {
            "env_type": env_type,
            "has_flagtree": flagtree["installed"],
            **code_details,
        },
        "control_env": control_env,
        "inject_result": inject_result,
    }


def output_json(data):
    """输出 JSON 格式"""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def output_report(data):
    """输出人类可读报告"""
    insp = data["inspection"]
    ctrl = data["flaggems_control"]

    report = []
    report.append("=" * 60)
    report.append("环境检测报告")
    report.append("=" * 60)

    report.append(f"\n## 执行模式: {data['execution']['mode']}")

    # 环境场景分类
    env_cls = data.get("env_classification", {})
    env_type = env_cls.get("env_type", "unknown")
    env_type_labels = {
        "native": "纯 vllm 原生（无 FlagGems）",
        "vllm_flaggems": "vllm + flaggems（代码直接集成）",
        "vllm_plugin_flaggems": "vllm + plugin + flaggems（环境变量控制）",
    }
    report.append(f"\n## 环境场景: {env_type} — {env_type_labels.get(env_type, '未知')}")
    if env_cls.get("has_flagtree"):
        report.append(f"  FlagTree:     已安装")
    if env_type == "vllm_flaggems":
        code_paths = env_cls.get('code_paths', [])
        if code_paths:
            report.append(f"  代码路径 ({len(code_paths)} 个文件):")
            for cp in code_paths:
                pri_label = {3: "无条件enable", 2: "条件enable", 1: "only_enable"}.get(cp.get("priority", 0), "")
                report.append(f"    - {cp['file']} [{pri_label}]")
        else:
            report.append(f"  代码路径:     {env_cls.get('code_path', '-')}")
        report.append(f"  enable() 调用: {env_cls.get('enable_call', '-')}")
        txt_path = env_cls.get("txt_path", "")
        if txt_path:
            report.append(f"  算子 txt 路径: {txt_path}")
        elif env_cls.get("auto_detect"):
            report.append(f"  算子 txt 路径: 未解析到，需启动服务后自动搜索")

    # 控制环境变量 & 注入结果
    ctrl_env = data.get("control_env", {})
    inject_res = data.get("inject_result", {})
    if ctrl_env:
        report.append(f"\n## FlagGems 控制环境变量")
        report.append(f"  USE_FLAGGEMS:          {ctrl_env.get('USE_FLAGGEMS', '-')}")
        report.append(f"  FLAGGEMS_CONTROL_MODE: {ctrl_env.get('FLAGGEMS_CONTROL_MODE', '-')}")
    if inject_res:
        if inject_res.get("already"):
            report.append(f"  代码注入:  已存在（跳过）")
        elif inject_res.get("injected"):
            report.append(f"  代码注入:  ✓ 已注入到 {inject_res.get('file', '-')}")
        elif inject_res.get("error"):
            report.append(f"  代码注入:  ✗ {inject_res.get('error', '-')}")

    report.append("\n## 核心组件")
    report.append(f"  {'组件':<15} {'版本':<20} {'状态'}")
    report.append(f"  {'-'*15} {'-'*20} {'-'*10}")
    for pkg, ver in insp["core_packages"].items():
        if pkg == "torch_cuda":
            continue
        status = "已安装" if ver else "未安装"
        report.append(f"  {pkg:<15} {str(ver or '-'):<20} {status}")
    cuda_ver = insp["core_packages"].get("torch_cuda")
    if cuda_ver:
        report.append(f"  {'CUDA':<15} {cuda_ver:<20} {'已安装'}")

    report.append("\n## Flag 生态组件")
    report.append(f"  {'组件':<15} {'版本':<20} {'状态'}")
    report.append(f"  {'-'*15} {'-'*20} {'-'*10}")
    for pkg, ver in insp["flag_packages"].items():
        status = "已安装" if ver else "未安装"
        report.append(f"  {pkg:<15} {str(ver or '-'):<20} {status}")

    report.append("\n## FlagGems 集成分析")
    report.append(f"  集成方式:    {ctrl['integration_type']}")
    report.append(f"  启用方法:    {ctrl['enable_method']}")
    report.append(f"  关闭方法:    {ctrl['disable_method']}")
    report.append(f"  运行时能力:  {', '.join(insp['flaggems_capabilities']) or '无'}")
    if insp["flaggems_enable_signature"]:
        report.append(f"  enable() 签名: {insp['flaggems_enable_signature']}")

    if insp.get("gpu_compute_capability"):
        report.append(f"  GPU Compute:    {insp['gpu_compute_capability']} ({insp.get('gpu_arch', '')})")

    if insp.get("plugin_env_vars"):
        report.append(f"  Plugin 环境变量:")
        for k, v in insp["plugin_env_vars"].items():
            report.append(f"    {k}={v}")

    if insp.get("plugin_control"):
        pc = insp["plugin_control"]
        report.append(f"\n  Plugin 控制信息:")
        report.append(f"    prefer_enabled: {pc.get('prefer_enabled', 'not_set')}")
        report.append(f"    oot_enabled:    {pc.get('oot_enabled', 'not_set')}")
        if pc.get("oot_ops"):
            report.append(f"    OOT 算子:       {', '.join(pc['oot_ops'])}")
        if pc.get("dispatch_mode"):
            report.append(f"    dispatch_mode:  {pc['dispatch_mode']}")

    if ctrl["code_locations"]:
        report.append("\n  代码级扫描结果:")
        for loc in ctrl["code_locations"][:10]:
            report.append(f"    {loc}")

    if insp["env_vars"]:
        report.append("\n## 环境变量")
        for k, v in insp["env_vars"].items():
            report.append(f"  {k}={v}")
    else:
        report.append("\n## 环境变量: 无 flag 相关环境变量")

    # FlagTree
    ft = data.get("flagtree", {})
    report.append("\n## FlagTree")
    if ft.get("installed"):
        report.append(f"  状态:        已安装")
        report.append(f"  版本:        {ft.get('version', 'unknown')}")
        report.append(f"  Triton 版本: {ft.get('triton_version', 'unknown')}")
        if ft.get("backend"):
            report.append(f"  Backend:     {ft['backend']}")
    else:
        triton_ver = ft.get("triton_version", "")
        if triton_ver:
            report.append(f"  状态:        未安装（triton {triton_ver} 为原版）")
        else:
            report.append(f"  状态:        未安装（triton 也未安装）")

    if insp["probe_error"]:
        report.append(f"\n## 探测错误: {insp['probe_error']}")

    report.append("\n" + "=" * 60)
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="FlagOS 环境检查合并脚本")
    parser.add_argument("--output-json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--report", action="store_true", help="输出人类可读报告")
    args = parser.parse_args()

    data = collect_all()

    if args.output_json:
        output_json(data)
    elif args.report:
        output_report(data)
    else:
        # 默认都输出
        output_json(data)
        print("\n---\n")
        output_report(data)


if __name__ == "__main__":
    main()
