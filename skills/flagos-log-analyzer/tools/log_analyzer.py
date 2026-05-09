#!/usr/bin/env python3
"""
log_analyzer.py — 日志分析与诊断工具

子命令:
    analyze   分析单个日志文件，分类错误，给出诊断建议
    scan      扫描整个日志目录，汇总所有发现

Usage:
    python3 log_analyzer.py analyze --log-path /flagos-workspace/logs/startup_flagos.log --json
    python3 log_analyzer.py scan --log-dir /flagos-workspace/logs/ --json

退出码: 0=成功, 1=无问题发现, 2=参数错误
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 错误分类模式
# =============================================================================

ERROR_PATTERNS = [
    # CUDA 错误（严重）
    {
        "category": "cuda_error",
        "severity": "critical",
        "patterns": [
            re.compile(r"CUDA\s*(?:error|Error|ERROR)\s*:?\s*(.+)", re.IGNORECASE),
            re.compile(r"CUDAError\s*:?\s*(.+)", re.IGNORECASE),
            re.compile(r"cudaError(?:_t)?\s*=\s*(\d+)", re.IGNORECASE),
            re.compile(r"no kernel image is available", re.IGNORECASE),
        ],
        "suggestion": "检查 GPU 驱动版本与 CUDA 版本兼容性，或禁用问题算子",
    },
    # OOM（严重）
    {
        "category": "oom",
        "severity": "critical",
        "patterns": [
            re.compile(r"(?:CUDA\s+)?out\s+of\s+memory", re.IGNORECASE),
            re.compile(r"OOM\b", re.IGNORECASE),
            re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
            re.compile(r"Allocation on device.*exceeds", re.IGNORECASE),
        ],
        "suggestion": "减小 tensor-parallel-size、max-model-len 或 batch size，或增加 GPU 数量",
    },
    # Triton 编译失败（严重）
    {
        "category": "triton_compile",
        "severity": "critical",
        "patterns": [
            re.compile(r"triton.*?compil(?:e|ation).*?(?:fail|error)", re.IGNORECASE),
            re.compile(r"CompilationError.*?triton", re.IGNORECASE),
            re.compile(r"triton\.compiler.*?(?:Error|Exception)", re.IGNORECASE),
            re.compile(r"flagtree.*?compil(?:e|ation).*?(?:fail|error)", re.IGNORECASE),
        ],
        "suggestion": "检查 FlagTree/Triton 版本兼容性，可能需要升级或重装",
    },
    # 算子错误（高）
    {
        "category": "operator_error",
        "severity": "high",
        "patterns": [
            re.compile(r"flag_gems.*?(?:Error|Exception|error)", re.IGNORECASE),
            re.compile(r"operator\s+not\s+supported", re.IGNORECASE),
            re.compile(r"Unsupported\s+(?:operator|op)\b", re.IGNORECASE),
            re.compile(r"GEMS\s+\w+.*?(?:Error|Failed)", re.IGNORECASE),
        ],
        "suggestion": "禁用问题算子后重启服务",
    },
    # 模型加载（高）
    {
        "category": "model_load",
        "severity": "high",
        "patterns": [
            re.compile(r"(?:model|weights?).*?(?:not\s+found|does\s+not\s+exist)", re.IGNORECASE),
            re.compile(r"tokenizer.*?(?:Error|not\s+found|failed)", re.IGNORECASE),
            re.compile(r"OSError.*?(?:model|tokenizer|config)", re.IGNORECASE),
            re.compile(r"Cannot\s+load\s+(?:model|weights|tokenizer)", re.IGNORECASE),
        ],
        "suggestion": "检查模型路径是否正确，权重文件是否完整",
    },
    # 端口冲突（中）
    {
        "category": "port_conflict",
        "severity": "medium",
        "patterns": [
            re.compile(r"[Aa]ddress\s+already\s+in\s+use", re.IGNORECASE),
            re.compile(r"bind.*?failed", re.IGNORECASE),
            re.compile(r"port\s+\d+.*?(?:in use|occupied|busy)", re.IGNORECASE),
        ],
        "suggestion": "更换端口或 kill 占用进程 (lsof -i :<port>)",
    },
    # 依赖缺失（中）
    {
        "category": "dependency",
        "severity": "medium",
        "patterns": [
            re.compile(r"ModuleNotFoundError:\s*No module named\s+'([^']+)'"),
            re.compile(r"ImportError:\s*(.+)"),
            re.compile(r"No module named\s+'([^']+)'"),
        ],
        "suggestion": "安装缺失的 Python 包",
    },
    # 超时/连接（低）
    {
        "category": "timeout",
        "severity": "low",
        "patterns": [
            re.compile(r"(?:connect(?:ion)?\s+)?timed?\s*out", re.IGNORECASE),
            re.compile(r"connection\s+refused", re.IGNORECASE),
            re.compile(r"ConnectionResetError", re.IGNORECASE),
        ],
        "suggestion": "检查网络连接或等待服务完全启动后重试",
    },
    # 一般警告（信息）
    {
        "category": "warning",
        "severity": "info",
        "patterns": [
            re.compile(r"\bWARNING\b:?\s*(.+)"),
            re.compile(r"DeprecationWarning:\s*(.+)"),
            re.compile(r"FutureWarning:\s*(.+)"),
        ],
        "suggestion": "记录但通常不影响运行",
    },
]

# 严重性排序（用于优先展示）
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


# =============================================================================
# FlagGems 检测模式
# =============================================================================

FLAGGEMS_PATTERNS = {
    "loaded": re.compile(r"flag_gems[._]ops?\s+loaded", re.IGNORECASE),
    "gems_op": re.compile(r"GEMS\s+(\w+)", re.IGNORECASE),
    "enable": re.compile(r"flag_gems\.enable\(", re.IGNORECASE),
    "import": re.compile(r"import\s+flag_gems|from\s+flag_gems", re.IGNORECASE),
    "oplist": re.compile(
        r"(?:flaggems_enable_oplist|gems)\.txt.*?(\d+)\s*(?:ops|operators|算子)",
        re.IGNORECASE,
    ),
}


# =============================================================================
# 启动序列检测
# =============================================================================

STARTUP_MARKERS = {
    "model_loaded": [
        re.compile(r"(?:Loading|Loaded)\s+model\b", re.IGNORECASE),
        re.compile(r"Model\s+weights\s+loaded", re.IGNORECASE),
        re.compile(r"Loading\s+model\s+weights", re.IGNORECASE),
    ],
    "gpu_initialized": [
        re.compile(r"(?:CUDA|GPU)\s+(?:initialized|available|detected)", re.IGNORECASE),
        re.compile(r"Using\s+(?:device|GPU)", re.IGNORECASE),
        re.compile(r"torch\.cuda\.is_available.*True", re.IGNORECASE),
        re.compile(r"Number\s+of\s+GPUs:\s*\d+", re.IGNORECASE),
    ],
    "port_bound": [
        re.compile(r"(?:Uvicorn|Server)\s+running\s+on\s+\S+:\d+", re.IGNORECASE),
        re.compile(r"(?:Listening|Serving)\s+(?:on|at)\s+\S+:\d+", re.IGNORECASE),
        re.compile(r"bind.*?(?:0\.0\.0\.0|localhost):\d+", re.IGNORECASE),
    ],
    "service_ready": [
        re.compile(r"(?:Application|Server)\s+startup\s+complete", re.IGNORECASE),
        re.compile(r"(?:Ready|Started)\s+(?:to\s+)?serv(?:e|ing)", re.IGNORECASE),
        re.compile(r"health\s*check.*?(?:pass|ok|200)", re.IGNORECASE),
    ],
}


# =============================================================================
# Traceback 提取
# =============================================================================

TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\):\n(?:.*\n)*?(?:\w+(?:Error|Exception).*)",
    re.MULTILINE,
)


# =============================================================================
# 核心分析逻辑
# =============================================================================


def analyze_log(
    log_path: str,
    max_errors: int = 50,
    max_warnings: int = 20,
) -> Dict[str, Any]:
    """分析单个日志文件，返回结构化诊断结果"""

    path = Path(log_path)
    if not path.is_file():
        return {
            "log_path": log_path,
            "status": "error",
            "message": f"文件不存在: {log_path}",
        }

    stat = path.stat()
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "log_path": log_path,
            "status": "error",
            "message": f"读取失败: {e}",
        }

    lines = content.splitlines()

    result: Dict[str, Any] = {
        "log_path": str(path),
        "log_size_bytes": stat.st_size,
        "log_lines": len(lines),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "status": "ok",
        "errors": [],
        "warnings": [],
        "tracebacks": [],
        "service_status": "unknown",
        "flaggems_detected": False,
        "flaggems_ops_loaded": [],
        "startup_sequence": {
            "model_loaded": False,
            "gpu_initialized": False,
            "port_bound": False,
            "service_ready": False,
        },
        "diagnosis": "",
        "suggestions": [],
    }

    # --- 提取 tracebacks ---
    tracebacks = TRACEBACK_RE.findall(content)
    result["tracebacks"] = tracebacks[:10]  # 最多保留 10 个

    # --- 逐行扫描 ---
    seen_errors: Dict[str, int] = {}  # category -> count
    error_entries = []
    warning_entries = []
    flaggems_ops = set()

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue

        # FlagGems 检测
        if FLAGGEMS_PATTERNS["loaded"].search(stripped):
            result["flaggems_detected"] = True
        if FLAGGEMS_PATTERNS["enable"].search(stripped):
            result["flaggems_detected"] = True
        if FLAGGEMS_PATTERNS["import"].search(stripped):
            result["flaggems_detected"] = True
        gems_match = FLAGGEMS_PATTERNS["gems_op"].search(stripped)
        if gems_match:
            result["flaggems_detected"] = True
            flaggems_ops.add(gems_match.group(1))

        # 启动序列检测
        for stage, patterns in STARTUP_MARKERS.items():
            if not result["startup_sequence"][stage]:
                for pat in patterns:
                    if pat.search(stripped):
                        result["startup_sequence"][stage] = True
                        break

        # 错误分类
        for error_def in ERROR_PATTERNS:
            cat = error_def["category"]
            sev = error_def["severity"]
            for pat in error_def["patterns"]:
                match = pat.search(stripped)
                if match:
                    count = seen_errors.get(cat, 0)
                    if sev == "info":
                        if count < max_warnings:
                            warning_entries.append({
                                "category": cat,
                                "severity": sev,
                                "line_number": line_num,
                                "message": stripped[:500],
                                "suggestion": error_def["suggestion"],
                            })
                    else:
                        if count < max_errors:
                            error_entries.append({
                                "category": cat,
                                "severity": sev,
                                "line_number": line_num,
                                "message": stripped[:500],
                                "suggestion": error_def["suggestion"],
                            })
                    seen_errors[cat] = count + 1
                    break  # 一行只匹配一个分类

    result["flaggems_ops_loaded"] = sorted(flaggems_ops)

    # 按严重性排序
    error_entries.sort(key=lambda e: SEVERITY_ORDER.get(e["severity"], 99))
    result["errors"] = error_entries
    result["warnings"] = warning_entries

    # --- 推断服务状态 ---
    result["service_status"] = _infer_service_status(result)

    # --- 生成诊断和建议 ---
    result["diagnosis"], result["suggestions"] = _generate_diagnosis(result)

    # 有错误则标记状态
    if error_entries:
        result["status"] = "error"
    elif warning_entries:
        result["status"] = "warning"

    return result


def _infer_service_status(result: Dict[str, Any]) -> str:
    """根据启动序列和错误推断服务状态"""
    seq = result["startup_sequence"]
    errors = result["errors"]

    has_critical = any(e["severity"] == "critical" for e in errors)
    has_oom = any(e["category"] == "oom" for e in errors)

    if seq["service_ready"]:
        if has_critical:
            return "running_with_errors"
        return "running"

    if has_oom:
        return "oom_killed"

    if has_critical:
        if not seq["model_loaded"]:
            return "model_load_failed"
        if not seq["gpu_initialized"]:
            return "gpu_init_failed"
        return "crashed"

    if seq["port_bound"] and not seq["service_ready"]:
        return "starting"

    if seq["model_loaded"] and not seq["port_bound"]:
        return "bind_failed"

    if not seq["model_loaded"] and not seq["gpu_initialized"]:
        return "not_started"

    return "unknown"


def _generate_diagnosis(result: Dict[str, Any]) -> Tuple[str, List[str]]:
    """根据分析结果生成诊断摘要和建议列表"""
    status = result["service_status"]
    errors = result["errors"]
    suggestions = []

    # 按类别统计
    cat_counts: Dict[str, int] = {}
    for e in errors:
        cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1

    # 诊断描述
    status_desc = {
        "running": "服务正常运行",
        "running_with_errors": "服务运行中但存在错误",
        "crashed": "服务启动后崩溃",
        "oom_killed": "GPU 显存不足导致服务终止",
        "model_load_failed": "模型加载失败",
        "gpu_init_failed": "GPU 初始化失败",
        "bind_failed": "端口绑定失败",
        "starting": "服务正在启动中",
        "not_started": "服务未启动",
        "unknown": "服务状态未知",
    }
    diagnosis = status_desc.get(status, f"服务状态: {status}")

    # 细化诊断
    if cat_counts:
        top_cats = sorted(cat_counts.items(), key=lambda x: SEVERITY_ORDER.get(
            next((e["severity"] for e in errors if e["category"] == x[0]), "info"), 99
        ))
        details = []
        for cat, cnt in top_cats[:3]:
            details.append(f"{cat}({cnt}次)")
        diagnosis += f"，主要问题: {', '.join(details)}"

    # 收集去重建议
    seen_suggestions = set()
    for e in errors:
        sug = e["suggestion"]
        if sug not in seen_suggestions:
            suggestions.append(sug)
            seen_suggestions.add(sug)

    # 状态特定建议
    if status == "oom_killed":
        suggestions.insert(0, "减小 tensor-parallel-size 或 max-model-len")
    elif status == "model_load_failed":
        suggestions.insert(0, "检查模型路径和权重文件完整性")
    elif status == "gpu_init_failed":
        suggestions.insert(0, "检查 GPU 驱动版本和 CUDA 工具链")
    elif status == "bind_failed":
        suggestions.insert(0, "检查端口是否被占用 (lsof -i :<port>)")
    elif status == "crashed" and result["flaggems_detected"]:
        suggestions.insert(0, "尝试关闭 FlagGems 启动 Native 模式验证是否为算子问题")

    return diagnosis, suggestions


# =============================================================================
# scan 子命令 — 扫描日志目录
# =============================================================================


def scan_logs(
    log_dir: str,
    pattern: str = "*.log",
    max_files: int = 50,
) -> Dict[str, Any]:
    """扫描日志目录，汇总所有发现"""

    dir_path = Path(log_dir)
    if not dir_path.is_dir():
        return {
            "log_dir": log_dir,
            "status": "error",
            "message": f"目录不存在: {log_dir}",
        }

    # 查找日志文件，按修改时间倒序
    log_files = sorted(
        dir_path.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:max_files]

    if not log_files:
        return {
            "log_dir": log_dir,
            "status": "ok",
            "message": "未找到日志文件",
            "files_scanned": 0,
            "files": [],
            "summary": {"total_errors": 0, "total_warnings": 0},
        }

    scan_result: Dict[str, Any] = {
        "log_dir": str(dir_path),
        "status": "ok",
        "files_scanned": len(log_files),
        "files": [],
        "summary": {
            "total_errors": 0,
            "total_warnings": 0,
            "error_categories": {},
            "flaggems_detected": False,
            "overall_service_status": "unknown",
        },
        "diagnosis": "",
        "suggestions": [],
    }

    all_categories: Dict[str, int] = {}
    all_suggestions: List[str] = []
    seen_suggestions = set()
    latest_status = "unknown"

    for log_file in log_files:
        analysis = analyze_log(str(log_file))
        file_summary = {
            "path": str(log_file),
            "size_bytes": analysis.get("log_size_bytes", 0),
            "lines": analysis.get("log_lines", 0),
            "last_modified": analysis.get("last_modified", ""),
            "status": analysis.get("status", "ok"),
            "error_count": len(analysis.get("errors", [])),
            "warning_count": len(analysis.get("warnings", [])),
            "service_status": analysis.get("service_status", "unknown"),
            "flaggems_detected": analysis.get("flaggems_detected", False),
        }
        scan_result["files"].append(file_summary)

        # 汇总
        scan_result["summary"]["total_errors"] += file_summary["error_count"]
        scan_result["summary"]["total_warnings"] += file_summary["warning_count"]

        if analysis.get("flaggems_detected"):
            scan_result["summary"]["flaggems_detected"] = True

        for err in analysis.get("errors", []):
            cat = err["category"]
            all_categories[cat] = all_categories.get(cat, 0) + 1
            sug = err["suggestion"]
            if sug not in seen_suggestions:
                all_suggestions.append(sug)
                seen_suggestions.add(sug)

        # 最新文件的服务状态
        if file_summary == scan_result["files"][0]:
            latest_status = file_summary["service_status"]

    scan_result["summary"]["error_categories"] = all_categories
    scan_result["summary"]["overall_service_status"] = latest_status
    scan_result["suggestions"] = all_suggestions

    # 生成整体诊断
    total_errors = scan_result["summary"]["total_errors"]
    total_warnings = scan_result["summary"]["total_warnings"]
    if total_errors == 0 and total_warnings == 0:
        scan_result["diagnosis"] = f"扫描 {len(log_files)} 个日志文件，未发现问题"
    else:
        top_cats = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)[:3]
        cat_desc = ", ".join(f"{c}({n}次)" for c, n in top_cats)
        scan_result["diagnosis"] = (
            f"扫描 {len(log_files)} 个日志文件，"
            f"发现 {total_errors} 个错误、{total_warnings} 个警告。"
            f"主要问题: {cat_desc}"
        )

    if total_errors > 0:
        scan_result["status"] = "error"
    elif total_warnings > 0:
        scan_result["status"] = "warning"

    return scan_result


# =============================================================================
# CLI 入口
# =============================================================================


def _output(data: Dict[str, Any], as_json: bool) -> None:
    """统一输出"""
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        _print_human(data)


def _print_human(data: Dict[str, Any]) -> None:
    """人类友好的输出格式"""
    if "files_scanned" in data:
        # scan 结果
        print(f"=== 日志扫描: {data.get('log_dir', '?')} ===")
        print(f"扫描文件数: {data.get('files_scanned', 0)}")
        summary = data.get("summary", {})
        print(f"错误总数: {summary.get('total_errors', 0)}")
        print(f"警告总数: {summary.get('total_warnings', 0)}")
        print(f"FlagGems 检测: {'是' if summary.get('flaggems_detected') else '否'}")
        print(f"服务状态: {summary.get('overall_service_status', 'unknown')}")
        print()
        for f in data.get("files", []):
            marker = "!" if f["error_count"] > 0 else " "
            print(f"  [{marker}] {f['path']} ({f['lines']} lines, {f['error_count']} errors)")
    else:
        # analyze 结果
        print(f"=== 日志分析: {data.get('log_path', '?')} ===")
        print(f"大小: {data.get('log_size_bytes', 0)} bytes, {data.get('log_lines', 0)} lines")
        print(f"服务状态: {data.get('service_status', 'unknown')}")
        print(f"FlagGems 检测: {'是' if data.get('flaggems_detected') else '否'}")
        if data.get("flaggems_ops_loaded"):
            print(f"FlagGems 算子: {', '.join(data['flaggems_ops_loaded'])}")
        seq = data.get("startup_sequence", {})
        print(f"启动序列: model={'OK' if seq.get('model_loaded') else '--'} "
              f"gpu={'OK' if seq.get('gpu_initialized') else '--'} "
              f"port={'OK' if seq.get('port_bound') else '--'} "
              f"ready={'OK' if seq.get('service_ready') else '--'}")
        print()
        for e in data.get("errors", [])[:20]:
            print(f"  [{e['severity'].upper():8s}] L{e['line_number']}: {e['category']} — {e['message'][:120]}")

    print()
    if data.get("diagnosis"):
        print(f"诊断: {data['diagnosis']}")
    if data.get("suggestions"):
        print("建议:")
        for i, s in enumerate(data["suggestions"][:10], 1):
            print(f"  {i}. {s}")


def main():
    parser = argparse.ArgumentParser(
        description="日志分析与诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # analyze
    p_analyze = sub.add_parser("analyze", help="分析单个日志文件")
    p_analyze.add_argument("--log-path", required=True, help="日志文件路径")
    p_analyze.add_argument("--max-errors", type=int, default=50, help="最多报告的错误数")
    p_analyze.add_argument("--max-warnings", type=int, default=20, help="最多报告的警告数")
    p_analyze.add_argument("--json", action="store_true", help="JSON 输出")

    # scan
    p_scan = sub.add_parser("scan", help="扫描日志目录")
    p_scan.add_argument("--log-dir", required=True, help="日志目录路径")
    p_scan.add_argument("--pattern", default="*.log", help="文件匹配模式 (default: *.log)")
    p_scan.add_argument("--max-files", type=int, default=50, help="最多扫描的文件数")
    p_scan.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(2)

    if args.command == "analyze":
        result = analyze_log(
            log_path=args.log_path,
            max_errors=args.max_errors,
            max_warnings=args.max_warnings,
        )
        _output(result, args.json)
        sys.exit(0 if result.get("errors") or result.get("warnings") else 1)

    elif args.command == "scan":
        result = scan_logs(
            log_dir=args.log_dir,
            pattern=args.pattern,
            max_files=args.max_files,
        )
        _output(result, args.json)
        sys.exit(0 if result["summary"]["total_errors"] > 0 or result["summary"]["total_warnings"] > 0 else 1)


if __name__ == "__main__":
    main()
