#!/usr/bin/env python3
"""
issue_reporter.py — FlagGems/FlagTree 问题自动收集、格式化与提交

子命令:
    collect   从日志/结果文件收集问题信息 + 环境版本
    format    将 issue_data.json 格式化为 Bug Report markdown
    submit    保存 markdown 到本地，有 GITHUB_TOKEN 时自动提交
    full      collect → format → submit 一步完成

提交策略:
    默认只生成带类型标注的 markdown 文件保存到本地
    显式传入 --submit 且有 GITHUB_TOKEN 环境变量时才提交到 GitHub
    文件命名: issue_{type}_{repo}_{timestamp}.md

Issue 类型:
    operator-crash       算子导致服务崩溃
    accuracy-zero        精度结果为零或严重异常
    accuracy-degraded    精度调优筛出的问题算子
    performance-degraded 性能调优筛出的问题算子
    flagtree-error       FlagTree/Triton 框架报错
    plugin-error         vllm-plugin-FL 框架报错

Usage:
    python3 issue_reporter.py collect --type operator-crash --log-path crash.log --env-file env.json --json
    python3 issue_reporter.py format --collected-file issue_data.json --json
    python3 issue_reporter.py submit --issue-file issue_report.md --repo flagos-ai/FlagGems --json
    python3 issue_reporter.py full --type operator-crash --log-path crash.log --env-file env.json --repo flagos-ai/FlagGems --json

退出码: 0=成功（含 markdown 已保存）, 1=收集无结果或 API 提交失败, 2=参数错误
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_ENV_FILE = "/flagos-workspace/.env"


def _load_env_file():
    """从 /flagos-workspace/.env 加载 token，环境变量已有值的不覆盖"""
    if not os.path.isfile(_ENV_FILE):
        return
    with open(_ENV_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key and not os.environ.get(key):
                os.environ[key] = val


_load_env_file()

# 共享模块导入（兼容本地开发和容器内扁平部署）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "flagos-operator-replacement" / "tools"))

try:
    from ops_constants import OPERATOR_GROUPS, OOT_OPERATORS
except ImportError:
    OPERATOR_GROUPS = {}
    OOT_OPERATORS = []

# 复用 diagnose_ops 的日志解析逻辑
try:
    from diagnose_ops import CRASH_PATTERNS, OP_EXTRACT_PATTERNS, extract_ops_from_text
except ImportError:
    # Fallback: 内联精简版
    CRASH_PATTERNS = [
        {"pattern": re.compile(r"(?:CUDA error|CUDAError).*?no kernel image", re.IGNORECASE | re.DOTALL),
         "type": "sm_unsupported", "description": "Triton kernel 未编译对应 SM 架构"},
        {"pattern": re.compile(r"(?:RuntimeError|TypeError).*?(?:got an unexpected keyword argument|missing \d+ required)", re.IGNORECASE),
         "type": "signature_mismatch", "description": "算子签名不匹配"},
        {"pattern": re.compile(r"(?:triton|CompilationError).*?(?:compile|compilation).*?(?:fail|error)", re.IGNORECASE),
         "type": "triton_compile", "description": "Triton kernel 编译失败"},
        {"pattern": re.compile(r"(?:CUDA|cuda)(?:\s+)?(?:error|Error|ERROR)(?:\s*:)?\s*(.+)"),
         "type": "cuda_error", "description": "CUDA 运行时错误"},
        {"pattern": re.compile(r"RuntimeError:\s*(.+)"),
         "type": "runtime_error", "description": "Python 运行时错误"},
    ]
    OP_EXTRACT_PATTERNS = [
        re.compile(r"flag_gems[./]ops[./](\w+)"),
        re.compile(r"vllm_fl[./]ops[./](?:oot[./])?(\w+)"),
        re.compile(r'File\s+"[^"]*flag_gems[^"]*?/(\w+)\.py"'),
    ]

    def extract_ops_from_text(text, known_ops):
        found = set()
        for pattern in OP_EXTRACT_PATTERNS:
            for match in pattern.finditer(text):
                op = match.group(1).lower().strip("_")
                if op in known_ops:
                    found.add(op)
        return sorted(found)


# =============================================================================
# 常量
# =============================================================================

ISSUE_TYPES = {
    "operator-crash": {
        "title_prefix": "Bug: Operator crash",
        "labels": ["bug"],
        "description": "算子导致服务崩溃",
    },
    "accuracy-zero": {
        "title_prefix": "Bug: Accuracy drops to zero",
        "labels": ["bug"],
        "description": "精度结果为零或严重异常",
    },
    "accuracy-degraded": {
        "title_prefix": "Bug: Operator accuracy degradation",
        "labels": ["bug"],
        "description": "算子导致精度不达标",
    },
    "performance-degraded": {
        "title_prefix": "Bug: Operator performance degradation",
        "labels": ["bug"],
        "description": "算子导致性能不达标",
    },
    "flagtree-error": {
        "title_prefix": "Bug: FlagTree/Triton error",
        "labels": ["bug"],
        "description": "FlagTree/Triton 框架报错",
    },
    "plugin-error": {
        "title_prefix": "Bug: vllm-plugin-FL error",
        "labels": ["bug"],
        "description": "vllm-plugin-FL 框架报错",
    },
}

MAX_LOG_CHARS = 2000


# =============================================================================
# collect — 收集问题数据
# =============================================================================

def collect_issue_data(
    issue_type: str,
    log_path: Optional[str] = None,
    env_file: Optional[str] = None,
    result_file: Optional[str] = None,
    ops_file: Optional[str] = None,
    context_yaml: Optional[str] = None,
    disabled_ops: Optional[List[str]] = None,
    disabled_reasons: Optional[Dict[str, str]] = None,
    model_name: Optional[str] = None,
    flaggems_code_path: Optional[str] = None,
    flaggems_code: Optional[str] = None,
    gems_txt_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """收集问题信息，组装 issue_data"""

    type_info = ISSUE_TYPES.get(issue_type, ISSUE_TYPES["operator-crash"])

    # 1. 加载环境信息
    env = _load_environment(env_file, context_yaml)

    # 2. 加载已知算子集
    known_ops = _load_known_ops(ops_file)

    # 3. 根据类型收集问题详情
    affected_ops = []
    error_type = ""
    error_messages = []
    error_logs = ""
    op_details = []

    if issue_type == "operator-crash" and log_path:
        crash = _parse_crash_log(log_path, known_ops)
        affected_ops = crash.get("crashed_ops", [])
        error_type = crash.get("primary_error_type", "unknown")
        error_messages = crash.get("error_messages", [])
        error_logs = crash.get("error_logs", "")

    elif issue_type == "accuracy-zero" and result_file:
        acc = _parse_accuracy_result(result_file)
        affected_ops = acc.get("suspected_ops", [])
        error_type = "accuracy_zero"
        error_messages = [acc.get("summary", "")]
        if log_path:
            error_logs = _read_log_tail(log_path, MAX_LOG_CHARS)

    elif issue_type in ("accuracy-degraded", "performance-degraded"):
        affected_ops = disabled_ops or []
        error_type = issue_type.replace("-", "_")
        if disabled_reasons:
            for op, reason in disabled_reasons.items():
                op_details.append({"op": op, "reason": reason})
                if reason not in error_messages:
                    error_messages.append(f"{op}: {reason}")
        if log_path:
            error_logs = _read_log_tail(log_path, MAX_LOG_CHARS)

    elif issue_type == "flagtree-error" and log_path:
        ft = _parse_flagtree_error(log_path)
        affected_ops = ft.get("related_components", [])
        error_type = ft.get("error_type", "triton_error")
        error_messages = ft.get("error_messages", [])
        error_logs = ft.get("error_logs", "")

    elif issue_type == "plugin-error" and log_path:
        pe = _parse_plugin_error(log_path)
        affected_ops = pe.get("related_components", [])
        error_type = pe.get("error_type", "plugin_error")
        error_messages = pe.get("error_messages", [])
        error_logs = pe.get("error_logs", "")

    # 4. 加载 FlagGems 代码上下文
    flaggems_ctx = _load_flaggems_context(
        flaggems_code_path, flaggems_code, context_yaml, gems_txt_path
    )

    # 5. 生成标题
    model = model_name or env.get("model", "")
    hardware = env.get("hardware", "unknown platform")
    if affected_ops:
        ops_str = ", ".join(affected_ops[:5])
        if len(affected_ops) > 5:
            ops_str += f" (+{len(affected_ops)-5} more)"
        title = f"{type_info['title_prefix']}: {ops_str} on {hardware}"
    else:
        title = f"{type_info['title_prefix']} on {hardware}"
    if model:
        title += f" ({model})"

    # 5. 组装结果
    data = {
        "type": issue_type,
        "title": title,
        "affected_ops": affected_ops,
        "op_details": op_details,
        "error_type": error_type,
        "error_messages": error_messages,
        "error_logs": error_logs[:MAX_LOG_CHARS],
        "environment": env,
        "flaggems_context": flaggems_ctx,
        "model": model,
        "labels": type_info["labels"],
        "timestamp": datetime.now().isoformat(),
    }

    # 6. 保存
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def _load_environment(env_file: Optional[str], context_yaml: Optional[str]) -> Dict[str, str]:
    """从环境文件或 context.yaml 加载环境信息"""
    env = {}

    # 优先从 env_file 加载（inspect_env.py 的 JSON 输出）
    if env_file and os.path.isfile(env_file):
        try:
            with open(env_file, "r") as f:
                data = json.load(f)
            # inspect_env.py 输出格式适配
            env["hardware"] = data.get("gpu_vendor", data.get("hardware", ""))
            env["gpu_type"] = data.get("gpu_type", data.get("gpu_model", ""))
            env["os"] = data.get("os_version", "")
            env["python"] = data.get("python_version", "")
            pkgs = data.get("core_packages", {})
            env["pytorch"] = pkgs.get("torch", data.get("torch_version", ""))
            env["triton"] = pkgs.get("triton", data.get("triton_version", ""))
            env["vllm"] = pkgs.get("vllm", data.get("vllm_version", ""))
            flag_pkgs = data.get("flag_packages", {})
            env["flaggems"] = flag_pkgs.get("flaggems", data.get("flaggems_version", ""))
            env["flagtree"] = flag_pkgs.get("flagtree", data.get("flagtree_version", ""))
            env["driver"] = data.get("driver_version", "")
            env["model"] = data.get("model_name", "")
            return {k: v for k, v in env.items() if v}
        except (json.JSONDecodeError, OSError):
            pass

    # 从 context.yaml 加载
    if context_yaml and os.path.isfile(context_yaml):
        try:
            import yaml
            with open(context_yaml, "r") as f:
                ctx = yaml.safe_load(f) or {}
            gpu = ctx.get("gpu", {})
            env["hardware"] = gpu.get("vendor", "")
            env["gpu_type"] = gpu.get("type", "")
            insp = ctx.get("inspection", {})
            core = insp.get("core_packages", {})
            env["pytorch"] = core.get("torch", "")
            env["triton"] = core.get("triton", "")
            env["vllm"] = core.get("vllm", "")
            flag = insp.get("flag_packages", {})
            env["flaggems"] = flag.get("flaggems", "")
            env["flagtree"] = flag.get("flagtree", "")
            model = ctx.get("model", {})
            env["model"] = model.get("name", "")
            return {k: v for k, v in env.items() if v}
        except (ImportError, OSError):
            pass

    return env


def _load_known_ops(ops_file: Optional[str]) -> set:
    """加载已知算子集合"""
    known = set()
    if ops_file and os.path.isfile(ops_file):
        try:
            with open(ops_file, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                known = set(data)
            elif isinstance(data, dict) and "ops" in data:
                known = set(data["ops"])
        except (json.JSONDecodeError, OSError):
            pass

    for group_ops in OPERATOR_GROUPS.values():
        known.update(group_ops)
    known.update(OOT_OPERATORS)
    return {op.lower() for op in known} | known


def _load_flaggems_context(
    code_path: Optional[str],
    code_snippet: Optional[str],
    context_yaml: Optional[str],
    gems_txt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """加载 FlagGems 代码上下文：enable 调用逻辑 + gems.txt 内容"""
    ctx: Dict[str, Any] = {
        "code_path": "",
        "enable_call": "",
        "code_snippet": "",
        "gems_txt_path": "",
        "gems_txt_content": [],
        "gems_txt_count": 0,
    }

    # 1. 代码片段来源
    if code_snippet:
        ctx["code_snippet"] = code_snippet
        # 提取 enable 调用
        for line in code_snippet.splitlines():
            if "flag_gems.enable" in line or "flag_gems.enable(" in line:
                ctx["enable_call"] = line.strip()
                break

    elif code_path and os.path.isfile(code_path):
        ctx["code_path"] = code_path
        try:
            lines = Path(code_path).read_text(encoding="utf-8", errors="replace").splitlines()
            # 找到 flag_gems 相关行，提取上下文 ±5 行
            relevant_ranges = []
            for i, line in enumerate(lines):
                if "flag_gems" in line or "import flag_gems" in line:
                    start = max(0, i - 5)
                    end = min(len(lines), i + 6)
                    relevant_ranges.append((start, end))
                    if "flag_gems.enable" in line:
                        ctx["enable_call"] = line.strip()
            # 合并重叠范围
            merged = _merge_ranges(relevant_ranges)
            snippets = []
            for start, end in merged:
                header = f"# Lines {start+1}-{end} of {code_path}"
                snippets.append(header)
                snippets.extend(lines[start:end])
                snippets.append("")
            ctx["code_snippet"] = "\n".join(snippets)
        except OSError:
            pass

    # 3. 从 context.yaml 补充
    if context_yaml and os.path.isfile(context_yaml):
        try:
            import yaml
            with open(context_yaml, "r") as f:
                yctx = yaml.safe_load(f) or {}
            env_section = yctx.get("environment", {})
            if not ctx["code_path"]:
                ctx["code_path"] = env_section.get("flaggems_code_path", "")
            if not ctx["enable_call"]:
                ctx["enable_call"] = env_section.get("flaggems_enable_call", "")
            if not gems_txt_path:
                gems_txt_path = env_section.get("flaggems_txt_path", "")
        except (ImportError, OSError):
            pass

    # 4. 读取 gems.txt 内容
    if not gems_txt_path:
        gems_txt_path = "/root/gems.txt"
    ctx["gems_txt_path"] = gems_txt_path
    if os.path.isfile(gems_txt_path):
        try:
            content = Path(gems_txt_path).read_text(encoding="utf-8", errors="replace")
            ops = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
            ctx["gems_txt_content"] = ops
            ctx["gems_txt_count"] = len(ops)
        except OSError:
            pass

    return ctx


def _merge_ranges(ranges: List[tuple]) -> List[tuple]:
    """合并重叠的行号范围"""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse_crash_log(log_path: str, known_ops: set) -> Dict[str, Any]:
    """解析崩溃日志提取问题算子和错误信息"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return {"crashed_ops": [], "error_messages": [], "error_logs": ""}

    lines = content.split("\n")

    # 提取 traceback 块
    tracebacks = []
    current_tb = []
    in_tb = False
    for line in lines:
        if "Traceback" in line and "most recent call" in line.lower():
            in_tb = True
            current_tb = [line]
        elif in_tb:
            current_tb.append(line)
            if line and not line.startswith(" ") and not line.startswith("\t") and "Error" in line:
                tracebacks.append("\n".join(current_tb))
                current_tb = []
                in_tb = False
            elif len(current_tb) > 100:
                tracebacks.append("\n".join(current_tb))
                current_tb = []
                in_tb = False

    crashed_ops = set()
    error_messages = []
    primary_error_type = "unknown"

    for tb in tracebacks:
        # 匹配错误模式
        for cp in CRASH_PATTERNS:
            if cp["pattern"].search(tb):
                primary_error_type = cp["type"]
                break
        # 提取算子名
        ops = extract_ops_from_text(tb, known_ops)
        crashed_ops.update(ops)
        # 提取最后一行错误信息
        last_line = tb.strip().split("\n")[-1].strip()
        if last_line and last_line not in error_messages:
            error_messages.append(last_line[:300])

    # 截取日志尾部作为 error_logs
    error_logs = "\n".join(tracebacks[-2:]) if tracebacks else _read_log_tail(log_path, MAX_LOG_CHARS)

    return {
        "crashed_ops": sorted(crashed_ops),
        "primary_error_type": primary_error_type,
        "error_messages": error_messages[:5],
        "error_logs": error_logs[:MAX_LOG_CHARS],
    }


def _parse_accuracy_result(result_file: str) -> Dict[str, Any]:
    """解析精度结果文件，检测精度为零"""
    try:
        with open(result_file, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"suspected_ops": [], "summary": "无法读取结果文件"}

    score = data.get("score", data.get("accuracy", None))
    if score is None:
        return {"suspected_ops": [], "summary": "结果文件中无 score 字段"}

    if isinstance(score, str):
        try:
            score = float(score.replace("%", ""))
        except ValueError:
            return {"suspected_ops": [], "summary": f"无法解析 score: {score}"}

    summary = f"精度评测结果: {score}%"
    if score == 0:
        summary += " (得分为零，可能所有 FlagGems 算子存在严重兼容性问题)"
    elif score < 5:
        summary += f" (得分极低，远低于随机基线 ~25%)"

    return {
        "suspected_ops": [],  # 精度为零时无法定位具体算子
        "summary": summary,
        "score": score,
    }


def _parse_flagtree_error(log_path: str) -> Dict[str, Any]:
    """解析 FlagTree/Triton 框架错误"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return {"related_components": [], "error_messages": [], "error_logs": ""}

    flagtree_patterns = [
        re.compile(r"(?:triton|flagtree).*?(?:compile|compilation).*?(?:fail|error)", re.IGNORECASE),
        re.compile(r"(?:triton|flagtree).*?(?:import|load).*?(?:fail|error)", re.IGNORECASE),
        re.compile(r"(?:CompilationError|JITError).*", re.IGNORECASE),
        re.compile(r"(?:LLVM|MLIR|PTX).*?(?:error|fail)", re.IGNORECASE),
    ]

    components = set()
    error_messages = []
    error_type = "triton_error"

    for pattern in flagtree_patterns:
        for match in pattern.finditer(content):
            msg = match.group(0).strip()[:300]
            if msg not in error_messages:
                error_messages.append(msg)
            if "triton" in msg.lower():
                components.add("triton")
            if "flagtree" in msg.lower():
                components.add("flagtree")
            if "llvm" in msg.lower() or "mlir" in msg.lower():
                components.add("compiler_backend")
                error_type = "compiler_backend_error"

    if not components:
        components.add("triton")

    return {
        "related_components": sorted(components),
        "error_type": error_type,
        "error_messages": error_messages[:5],
        "error_logs": _read_log_tail(log_path, MAX_LOG_CHARS),
    }


def _parse_plugin_error(log_path: str) -> Dict[str, Any]:
    """解析 vllm-plugin-FL 相关错误"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return {"related_components": [], "error_messages": [], "error_logs": ""}

    plugin_patterns = [
        re.compile(r"(?:vllm_plugin_fl|vllm-plugin-FL).*?(?:error|fail|exception)", re.IGNORECASE),
        re.compile(r"(?:OpManager|op_manager).*?(?:error|fail|dispatch)", re.IGNORECASE),
        re.compile(r"(?:ImportError|ModuleNotFoundError).*?(?:vllm_plugin|vllm_fl)", re.IGNORECASE),
        re.compile(r"(?:vllm_fl)\.(?:ops|dispatch|register).*?(?:error|fail)", re.IGNORECASE),
    ]

    components = set()
    error_messages = []
    error_type = "plugin_error"

    for pattern in plugin_patterns:
        for match in pattern.finditer(content):
            msg = match.group(0).strip()[:300]
            if msg not in error_messages:
                error_messages.append(msg)
            if "opmanager" in msg.lower() or "dispatch" in msg.lower():
                components.add("dispatch")
                error_type = "dispatch_error"
            if "import" in msg.lower():
                components.add("import")
                error_type = "import_error"
            if "vllm_fl" in msg.lower():
                components.add("vllm_fl")

    if not components:
        components.add("vllm-plugin-FL")

    return {
        "related_components": sorted(components),
        "error_type": error_type,
        "error_messages": error_messages[:5],
        "error_logs": _read_log_tail(log_path, MAX_LOG_CHARS),
    }


def _read_log_tail(log_path: str, max_chars: int) -> str:
    """读取日志文件尾部"""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > max_chars:
            return "... (truncated) ...\n" + content[-max_chars:]
        return content
    except OSError:
        return ""


# =============================================================================
# format — 格式化为 issue markdown
# =============================================================================

def format_issue(
    issue_data: Dict[str, Any],
    output_path: Optional[str] = None,
    repo: str = "flagos-ai/FlagGems",
) -> str:
    """将 issue_data 格式化为 Bug Report markdown"""

    issue_type = issue_data.get("type", "operator-crash")
    title = issue_data.get("title", "Bug Report")
    affected_ops = issue_data.get("affected_ops", [])
    op_details = issue_data.get("op_details", [])
    error_messages = issue_data.get("error_messages", [])
    error_logs = issue_data.get("error_logs", "")
    env = issue_data.get("environment", {})
    model = issue_data.get("model", "")
    flaggems_ctx = issue_data.get("flaggems_context", {})

    # Description
    description = _generate_description(issue_type, affected_ops, op_details, error_messages, model)

    # Environment table
    env_table = _generate_env_table(env)

    # Steps to Reproduce
    steps = _generate_steps(issue_type, model, env)

    # Expected Behavior
    expected = _generate_expected(issue_type)

    # Actual Behavior
    actual = _generate_actual(issue_type, affected_ops, op_details, error_messages)

    # Additional Context
    context = _generate_context(issue_type, env, affected_ops)

    # Possible Directions
    directions = _generate_directions(issue_type, env, affected_ops)

    # Assemble markdown
    header = (
        f"<!-- Issue Target: https://github.com/{repo}/issues/new -->\n"
        f"<!-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"
        f"<!-- Type: {issue_type} -->\n\n"
    )
    md = header + f"""## Bug Report: {title}

### Description

{description}

### Environment

{env_table}

### Steps to Reproduce

{steps}

### Expected Behavior

{expected}

### Actual Behavior

{actual}
"""

    if error_logs.strip():
        md += f"""
### Error Logs

```
{error_logs.strip()}
```
"""

    # FlagGems 代码上下文（所有 issue 必须包含）
    flaggems_section = _generate_flaggems_section(flaggems_ctx)
    if flaggems_section:
        md += f"""
### FlagGems Integration Code

{flaggems_section}
"""

    md += f"""
### Additional Context

{context}

### Possible Directions

{directions}
"""

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

    return md


def _generate_description(issue_type, affected_ops, op_details, error_messages, model):
    model_str = f" on model **{model}**" if model else ""

    if issue_type == "operator-crash":
        ops_list = "\n".join(f"- `{op}`" for op in affected_ops) if affected_ops else "- (unable to identify specific operators)"
        return f"When running model inference{model_str}, several operators fail at runtime, causing the inference process to crash.\n\nThe affected operators include:\n{ops_list}"

    elif issue_type == "accuracy-zero":
        return f"When running accuracy evaluation (GPQA Diamond){model_str} with FlagGems enabled, the evaluation score drops to zero or near-zero, indicating severe compatibility issues with FlagGems operators.\n\n{error_messages[0] if error_messages else ''}"

    elif issue_type == "accuracy-degraded":
        ops_list = _format_op_details(affected_ops, op_details, "accuracy")
        return f"During accuracy evaluation{model_str}, the following operators were identified as causing accuracy degradation (>5% deviation from baseline):\n\n{ops_list}"

    elif issue_type == "performance-degraded":
        ops_list = _format_op_details(affected_ops, op_details, "performance")
        return f"During performance benchmarking{model_str}, the following operators were identified as causing significant performance degradation (<80% of native baseline):\n\n{ops_list}"

    elif issue_type == "flagtree-error":
        components = ", ".join(f"`{op}`" for op in affected_ops) if affected_ops else "triton/flagtree"
        return f"FlagTree/Triton framework error encountered{model_str}. Related components: {components}.\n\n{error_messages[0] if error_messages else ''}"

    elif issue_type == "plugin-error":
        components = ", ".join(f"`{op}`" for op in affected_ops) if affected_ops else "vllm-plugin-FL"
        return f"vllm-plugin-FL error encountered{model_str}. Related components: {components}.\n\n{error_messages[0] if error_messages else ''}"

    return "Bug report."


def _format_op_details(affected_ops, op_details, category):
    if op_details:
        lines = []
        for d in op_details:
            lines.append(f"- `{d['op']}` — {d.get('reason', 'N/A')}")
        return "\n".join(lines)
    if affected_ops:
        return "\n".join(f"- `{op}`" for op in affected_ops)
    return "- (no specific operators identified)"


def _generate_env_table(env):
    rows = [
        ("Hardware", env.get("hardware", "N/A")),
        ("GPU", env.get("gpu_type", "N/A")),
        ("OS", env.get("os", "N/A")),
        ("Python", env.get("python", "N/A")),
        ("PyTorch", env.get("pytorch", "N/A")),
        ("Triton", env.get("triton", "N/A")),
        ("FlagGems", env.get("flaggems", "N/A")),
        ("FlagTree", env.get("flagtree", "N/A")),
        ("vLLM", env.get("vllm", "N/A")),
        ("Driver", env.get("driver", "N/A")),
    ]
    table = "| Item | Value |\n|------|-------|\n"
    for item, value in rows:
        if value and value != "N/A":
            table += f"| {item} | {value} |\n"
    return table.strip()


def _generate_steps(issue_type, model, env):
    hardware = env.get("hardware", "target platform")
    steps = [f"1. Set up environment on {hardware}"]
    steps.append("2. Install dependencies (PyTorch + Triton + FlagGems + vLLM)")

    if issue_type in ("operator-crash", "flagtree-error"):
        steps.append("3. Start vLLM inference service with FlagGems enabled")
        steps.append("4. Observe operator runtime failures / framework errors")
    elif issue_type == "plugin-error":
        steps.append("3. Install vllm-plugin-FL (git clone + pip install --no-build-isolation)")
        steps.append("4. Start vLLM inference service with plugin enabled")
        steps.append("5. Observe plugin-related errors")
    elif issue_type == "accuracy-zero":
        steps.append("3. Start vLLM inference service with FlagGems enabled")
        steps.append("4. Run GPQA Diamond evaluation (198 questions)")
        steps.append("5. Observe evaluation score drops to zero")
    elif issue_type == "accuracy-degraded":
        steps.append("3. Run accuracy evaluation with FlagGems V1 (native) and V2 (FlagGems enabled)")
        steps.append("4. Compare V1 vs V2 scores, deviation exceeds 5% threshold")
        steps.append("5. Run operator-level accuracy diagnosis to identify problematic operators")
    elif issue_type == "performance-degraded":
        steps.append("3. Run performance benchmark with V1 (native) and V2 (FlagGems enabled)")
        steps.append("4. Compare V1 vs V2 throughput, ratio below 80% at specific concurrency levels")
        steps.append("5. Run operator search optimization to identify problematic operators")

    return "\n".join(steps)


def _generate_expected(issue_type):
    if issue_type == "operator-crash":
        return "All operators should execute correctly and inference should complete without errors."
    elif issue_type == "accuracy-zero":
        return "Evaluation score should be comparable to native (non-FlagGems) baseline, within 5% deviation."
    elif issue_type == "accuracy-degraded":
        return "All FlagGems operators should maintain accuracy within 5% of native baseline."
    elif issue_type == "performance-degraded":
        return "All FlagGems operators should maintain performance at >=80% of native baseline at each concurrency level."
    elif issue_type == "flagtree-error":
        return "FlagTree/Triton should compile and execute kernels without errors."
    elif issue_type == "plugin-error":
        return "vllm-plugin-FL should load and dispatch operators without errors."
    return "Expected behavior."


def _generate_actual(issue_type, affected_ops, op_details, error_messages):
    if issue_type == "operator-crash":
        ops_list = "\n".join(f"- `{op}`" for op in affected_ops) if affected_ops else "- (see error logs)"
        return f"The following operators fail during execution:\n{ops_list}\n\nThese failures cause the model inference process to crash."

    elif issue_type == "accuracy-zero":
        return "Evaluation score drops to zero or near-zero when FlagGems is enabled, while native mode produces normal scores."

    elif issue_type in ("accuracy-degraded", "performance-degraded"):
        label = "accuracy" if "accuracy" in issue_type else "performance"
        lines = []
        if op_details:
            for d in op_details:
                lines.append(f"- `{d['op']}`: {d.get('reason', 'degraded')}")
        elif affected_ops:
            for op in affected_ops:
                lines.append(f"- `{op}`: {label} degradation detected")
        ops_str = "\n".join(lines) if lines else "- (see error logs)"
        return f"The following operators cause {label} degradation:\n{ops_str}\n\nThese operators have been disabled to achieve acceptable {label}."

    elif issue_type == "flagtree-error":
        msgs = "\n".join(f"- {m}" for m in error_messages[:3]) if error_messages else "- (see error logs)"
        return f"FlagTree/Triton framework errors:\n{msgs}"

    elif issue_type == "plugin-error":
        msgs = "\n".join(f"- {m}" for m in error_messages[:3]) if error_messages else "- (see error logs)"
        return f"vllm-plugin-FL errors:\n{msgs}"

    return "Unexpected behavior."


def _generate_context(issue_type, env, affected_ops):
    hardware = env.get("hardware", "")
    lines = []

    if hardware and hardware.lower() not in ("nvidia", ""):
        lines.append(f"- The issue appears to be specific to **{hardware}** platform and may not reproduce on standard CUDA (NVIDIA) environments.")

    pytorch = env.get("pytorch", "")
    if "hip" in pytorch.lower() or "rocm" in pytorch.lower():
        lines.append("- PyTorch is using HIP backend, suggesting ROCm compatibility layer involvement.")
    elif "npu" in pytorch.lower():
        lines.append("- PyTorch is using NPU backend (Ascend).")

    if issue_type in ("accuracy-degraded", "performance-degraded") and affected_ops:
        lines.append(f"- {len(affected_ops)} operator(s) were identified through automated binary search / group testing.")
        lines.append("- Disabling these operators restores acceptable accuracy/performance levels.")

    if issue_type == "accuracy-zero":
        lines.append("- Score of zero suggests fundamental incompatibility rather than marginal precision issues.")

    if not lines:
        lines.append("- No additional context available.")

    return "\n".join(lines)


def _generate_directions(issue_type, env, affected_ops):
    directions = []

    if issue_type == "operator-crash":
        directions.append("- Verify kernel support for the failing operators on this backend")
        directions.append("- Check dtype compatibility (e.g., BF16 / FP16)")
        directions.append("- Investigate Triton code generation for this platform")
        directions.append("- Confirm whether fallback kernels are correctly triggered")

    elif issue_type == "accuracy-zero":
        directions.append("- Check if FlagGems dispatch correctly intercepts operators on this platform")
        directions.append("- Verify numerical precision of core operators (softmax, layer_norm, etc.)")
        directions.append("- Test with a minimal model to isolate the issue")

    elif issue_type == "accuracy-degraded":
        directions.append("- Investigate precision of the identified operators with unit tests")
        directions.append("- Check numerical accumulation differences (FP32 vs FP16 intermediate)")
        directions.append("- Verify operator implementations match PyTorch reference semantics")

    elif issue_type == "performance-degraded":
        directions.append("- Profile the identified operators to find performance bottlenecks")
        directions.append("- Check Triton autotuning configurations for this platform")
        directions.append("- Compare kernel launch overhead with native implementations")

    elif issue_type == "flagtree-error":
        directions.append("- Verify FlagTree version compatibility with Triton and PyTorch")
        directions.append("- Check LLVM/MLIR backend support for this platform")
        directions.append("- Try rebuilding FlagTree from source with platform-specific flags")

    elif issue_type == "plugin-error":
        directions.append("- Verify vllm-plugin-FL version compatibility with vLLM and FlagGems")
        directions.append("- Check OpManager dispatch configuration and operator registration")
        directions.append("- Try reinstalling plugin with --no-build-isolation flag")

    return "\n".join(directions)


def _generate_flaggems_section(flaggems_ctx: Dict[str, Any]) -> str:
    """生成 FlagGems Integration Code section"""
    if not flaggems_ctx:
        return ""

    parts = []

    code_path = flaggems_ctx.get("code_path", "")
    enable_call = flaggems_ctx.get("enable_call", "")
    code_snippet = flaggems_ctx.get("code_snippet", "")
    gems_txt_path = flaggems_ctx.get("gems_txt_path", "")
    gems_txt_content = flaggems_ctx.get("gems_txt_content", [])
    gems_txt_count = flaggems_ctx.get("gems_txt_count", 0)

    if code_path:
        parts.append(f"**File**: `{code_path}`")
    if enable_call:
        parts.append(f"**Enable call**: `{enable_call}`")

    if code_snippet:
        parts.append(f"\n```python\n{code_snippet.strip()}\n```")

    if gems_txt_content:
        ops_str = "\n".join(gems_txt_content)
        parts.append(f"\n**Current gems.txt** ({gems_txt_count} operators, path: `{gems_txt_path}`):\n```\n{ops_str}\n```")
    elif gems_txt_path:
        parts.append(f"\n**gems.txt path**: `{gems_txt_path}` (file not found or empty)")

    return "\n".join(parts) if parts else ""


# =============================================================================
# submit — 提交 issue
# =============================================================================

def submit_issue(
    issue_file: str,
    repo: str = "flagos-ai/FlagGems",
    title: Optional[str] = None,
    labels: Optional[List[str]] = None,
    dry_run: bool = False,
    output_dir: Optional[str] = None,
    issue_type: Optional[str] = None,
    auto_submit: bool = False,
) -> Dict[str, Any]:
    """保存 issue markdown 到本地，仅在 auto_submit=True 且有 GITHUB_TOKEN 时提交"""

    # 读取 issue 内容
    try:
        with open(issue_file, "r", encoding="utf-8") as f:
            body = f.read()
    except OSError as e:
        return {"submitted": False, "error": f"无法读取 issue 文件: {e}"}

    # 从 body 提取标题（如果未提供）
    if not title:
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("## ") and "Bug Report" in line:
                title = line.lstrip("# ").strip()
                break
        if not title:
            title = "Bug Report"

    # 保存带类型+仓库名+时间戳的 markdown 文件
    repo_short = repo.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    type_tag = f"{issue_type}_" if issue_type else ""
    filename = f"issue_{type_tag}{repo_short}_{timestamp}.md"
    save_dir = output_dir or os.path.dirname(os.path.abspath(issue_file))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 写入带元信息头的 markdown
    header = (
        f"<!-- Issue Target: https://github.com/{repo}/issues/new -->\n"
        f"<!-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"
        f"<!-- Title: {title} -->\n"
        f"<!-- Type: {issue_type or 'unknown'} -->\n\n"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(header + body)

    if dry_run:
        return {
            "submitted": False,
            "method": "dry_run",
            "title": title,
            "repo": repo,
            "labels": labels or [],
            "body_length": len(body),
            "report_path": save_path,
        }

    # 仅在显式 --submit 时尝试提交 GitHub
    if not auto_submit:
        return {
            "submitted": False,
            "method": "local_only",
            "report_path": save_path,
            "message": (
                f"Issue 报告已保存到 {save_path}\n"
                f"如需提交到 GitHub，请加 --submit 参数\n"
                f"或手动提交到 https://github.com/{repo}/issues/new"
            ),
        }

    # auto_submit=True: 有 GITHUB_TOKEN 时提交
    token = os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))
    if token:
        api_result = _submit_via_api(body, repo, title, labels, token)
        if api_result.get("submitted"):
            api_result["report_path"] = save_path
            return api_result
        # API 失败，markdown 已保存
        return {
            "submitted": False,
            "method": "api_failed",
            "error": api_result.get("error", ""),
            "report_path": save_path,
            "message": f"API 提交失败，Issue 报告已保存到 {save_path}",
        }

    # auto_submit=True 但无 token
    return {
        "submitted": False,
        "method": "no_token",
        "report_path": save_path,
        "message": (
            f"Issue 报告已保存到 {save_path}\n"
            f"如需自动提交，请设置环境变量: export GITHUB_TOKEN=ghp_xxx\n"
            f"或手动提交到 https://github.com/{repo}/issues/new"
        ),
    }


def _submit_via_api(body: str, repo: str, title: str, labels: Optional[List[str]], token: str) -> Dict[str, Any]:
    """通过 GitHub API 提交"""
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = {
        "title": title,
        "body": body,
    }
    if labels:
        payload["labels"] = labels

    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return {
                "submitted": True,
                "method": "api_token",
                "issue_url": result.get("html_url", ""),
                "issue_number": result.get("number", 0),
            }
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        return {"submitted": False, "error": f"GitHub API 提交失败: {e}"}


def _extract_issue_number(url: str) -> int:
    """从 URL 提取 issue 编号"""
    m = re.search(r"/issues/(\d+)", url)
    return int(m.group(1)) if m else 0


# =============================================================================
# full — 一步完成
# =============================================================================

def full_pipeline(args) -> Dict[str, Any]:
    """collect → format → submit"""

    # collect
    disabled_ops = args.disabled_ops.split(",") if args.disabled_ops else None
    disabled_reasons = None
    if args.disabled_reasons:
        try:
            disabled_reasons = json.loads(args.disabled_reasons)
        except json.JSONDecodeError:
            pass

    output_dir = args.output_dir or "/flagos-workspace/results"
    os.makedirs(output_dir, exist_ok=True)

    type_tag = args.type.replace("-", "_")
    data_path = os.path.join(output_dir, f"issue_data_{type_tag}.json")
    data = collect_issue_data(
        issue_type=args.type,
        log_path=args.log_path,
        env_file=args.env_file,
        result_file=args.result_file,
        ops_file=args.ops_file,
        context_yaml=args.context_yaml,
        disabled_ops=disabled_ops,
        disabled_reasons=disabled_reasons,
        model_name=args.model_name,
        flaggems_code_path=getattr(args, "flaggems_code_path", None),
        flaggems_code=getattr(args, "flaggems_code", None),
        gems_txt_path=getattr(args, "gems_txt_path", None),
        output_path=data_path,
    )

    # format
    report_path = os.path.join(output_dir, f"issue_report_{type_tag}.md")
    format_issue(data, output_path=report_path, repo=args.repo)

    # submit
    result = submit_issue(
        issue_file=report_path,
        repo=args.repo,
        title=data.get("title"),
        labels=data.get("labels"),
        dry_run=args.dry_run,
        output_dir=output_dir,
        issue_type=args.type,
        auto_submit=getattr(args, "submit", False),
    )

    result["collected_data"] = data_path
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FlagGems/FlagTree 问题自动收集与提交",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="操作类型")

    # collect
    col = subparsers.add_parser("collect", help="收集问题数据")
    col.add_argument("--type", required=True, choices=list(ISSUE_TYPES.keys()), help="问题类型")
    col.add_argument("--log-path", help="日志文件路径")
    col.add_argument("--env-file", help="环境信息 JSON 文件")
    col.add_argument("--result-file", help="评测结果 JSON 文件")
    col.add_argument("--ops-file", help="算子列表 JSON 文件")
    col.add_argument("--context-yaml", help="context.yaml 路径")
    col.add_argument("--disabled-ops", help="逗号分隔的问题算子列表")
    col.add_argument("--disabled-reasons", help="JSON 格式的算子禁用原因")
    col.add_argument("--model-name", help="模型名称")
    col.add_argument("--flaggems-code-path", help="flaggems 代码文件路径")
    col.add_argument("--flaggems-code", help="直接传入 flaggems 代码片段")
    col.add_argument("--gems-txt-path", help="gems.txt 文件路径（默认 /root/gems.txt）")
    col.add_argument("--output", default=None, help="输出 JSON 路径")
    col.add_argument("--json", action="store_true", help="JSON 输出")

    # format
    fmt = subparsers.add_parser("format", help="格式化为 issue markdown")
    fmt.add_argument("--collected-file", required=True, help="issue_data.json 路径")
    fmt.add_argument("--output", default=None, help="输出 markdown 路径")
    fmt.add_argument("--repo", default="flagos-ai/FlagGems", help="目标 GitHub 仓库（用于 markdown 元信息）")
    fmt.add_argument("--json", action="store_true", help="JSON 输出")

    # submit
    sub = subparsers.add_parser("submit", help="提交 issue")
    sub.add_argument("--issue-file", required=True, help="issue markdown 文件路径")
    sub.add_argument("--repo", default="flagos-ai/FlagGems", help="目标 GitHub 仓库")
    sub.add_argument("--title", help="issue 标题（默认从 markdown 提取）")
    sub.add_argument("--labels", help="逗号分隔的标签")
    sub.add_argument("--output-dir", default=None, help="markdown 保存目录")
    sub.add_argument("--dry-run", action="store_true", help="不实际提交")
    sub.add_argument("--submit", action="store_true", help="显式提交到 GitHub（默认只生成 markdown）")
    sub.add_argument("--issue-type", default=None, help="issue 类型（用于文件名标注）")
    sub.add_argument("--json", action="store_true", help="JSON 输出")

    # full
    fl = subparsers.add_parser("full", help="collect → format → submit")
    fl.add_argument("--type", required=True, choices=list(ISSUE_TYPES.keys()), help="问题类型")
    fl.add_argument("--log-path", help="日志文件路径")
    fl.add_argument("--env-file", help="环境信息 JSON 文件")
    fl.add_argument("--result-file", help="评测结果 JSON 文件")
    fl.add_argument("--ops-file", help="算子列表 JSON 文件")
    fl.add_argument("--context-yaml", help="context.yaml 路径")
    fl.add_argument("--disabled-ops", help="逗号分隔的问题算子列表")
    fl.add_argument("--disabled-reasons", help="JSON 格式的算子禁用原因")
    fl.add_argument("--model-name", help="模型名称")
    fl.add_argument("--flaggems-code-path", help="flaggems 代码文件路径")
    fl.add_argument("--flaggems-code", help="直接传入 flaggems 代码片段")
    fl.add_argument("--gems-txt-path", help="gems.txt 文件路径（默认 /root/gems.txt）")
    fl.add_argument("--repo", default="flagos-ai/FlagGems", help="目标 GitHub 仓库")
    fl.add_argument("--output-dir", default=None, help="输出目录")
    fl.add_argument("--dry-run", action="store_true", help="不实际提交")
    fl.add_argument("--submit", action="store_true", help="显式提交到 GitHub（默认只生成 markdown）")
    fl.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(2)

    if args.command == "collect":
        disabled_ops = args.disabled_ops.split(",") if args.disabled_ops else None
        disabled_reasons = None
        if args.disabled_reasons:
            try:
                disabled_reasons = json.loads(args.disabled_reasons)
            except json.JSONDecodeError:
                print("错误: --disabled-reasons 必须是有效 JSON", file=sys.stderr)
                sys.exit(2)

        result = collect_issue_data(
            issue_type=args.type,
            log_path=args.log_path,
            env_file=args.env_file,
            result_file=args.result_file,
            ops_file=args.ops_file,
            context_yaml=args.context_yaml,
            disabled_ops=disabled_ops,
            disabled_reasons=disabled_reasons,
            model_name=args.model_name,
            flaggems_code_path=args.flaggems_code_path,
            flaggems_code=args.flaggems_code,
            gems_txt_path=args.gems_txt_path,
            output_path=args.output,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_collect_report(result)
        sys.exit(0 if result.get("affected_ops") or result.get("error_messages") else 1)

    elif args.command == "format":
        with open(args.collected_file, "r") as f:
            data = json.load(f)
        md = format_issue(data, output_path=args.output, repo=args.repo)
        if args.json:
            print(json.dumps({"markdown": md, "output_path": args.output}, ensure_ascii=False, indent=2))
        else:
            print(md)

    elif args.command == "submit":
        labels = args.labels.split(",") if args.labels else None
        result = submit_issue(
            issue_file=args.issue_file,
            repo=args.repo,
            title=args.title,
            labels=labels,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            issue_type=getattr(args, "issue_type", None),
            auto_submit=getattr(args, "submit", False),
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_submit_report(result)
        # local_only / dry_run 也算成功（markdown 已保存），仅 api_failed 算失败
        sys.exit(0 if result.get("submitted") or result.get("method") in ("local_only", "dry_run") else 1)

    elif args.command == "full":
        result = full_pipeline(args)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_submit_report(result)
        sys.exit(0 if result.get("submitted") or result.get("method") in ("local_only", "dry_run") else 1)


def _print_collect_report(data):
    print("=" * 60)
    print("Issue 数据收集报告")
    print("=" * 60)
    print(f"类型: {data.get('type', 'N/A')}")
    print(f"标题: {data.get('title', 'N/A')}")
    ops = data.get("affected_ops", [])
    if ops:
        print(f"问题算子: {', '.join(ops)}")
    details = data.get("op_details", [])
    if details:
        for d in details:
            print(f"  - {d['op']}: {d.get('reason', 'N/A')}")
    msgs = data.get("error_messages", [])
    if msgs:
        print(f"错误信息: {msgs[0][:100]}")
    print("=" * 60)


def _print_submit_report(result):
    print("=" * 60)
    print("Issue 提交结果")
    print("=" * 60)
    if result.get("submitted"):
        print(f"状态: 已提交")
        print(f"方式: {result.get('method', 'N/A')}")
        print(f"URL: {result.get('issue_url', 'N/A')}")
        print(f"编号: #{result.get('issue_number', 'N/A')}")
    else:
        print(f"状态: 未提交")
        print(f"方式: {result.get('method', 'N/A')}")
        if result.get("message"):
            print(f"\n{result['message']}")
        if result.get("error"):
            print(f"错误: {result['error']}")
    if result.get("report_path"):
        print(f"报告: {result['report_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
