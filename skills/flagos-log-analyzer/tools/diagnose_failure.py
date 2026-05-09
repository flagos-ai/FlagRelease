#!/usr/bin/env python3
"""
diagnose_failure.py — FlagOS 流程中断自动诊断工具

Claude 中断后由 run_pipeline.sh 自动调用，也可手动执行。
完全独立运行，不依赖 Claude。

Usage:
    python3 diagnose_failure.py              # 人可读输出
    python3 diagnose_failure.py --json       # JSON 输出（供程序/新会话读取）
    python3 diagnose_failure.py --workspace /flagos-workspace  # 指定工作目录

退出码: 0=诊断完成, 1=无异常（流程已正常完成）, 2=参数错误
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 日志错误模式（从 log_analyzer.py 精简复用）
# =============================================================================

ERROR_PATTERNS = [
    {"category": "oom", "severity": "critical",
     "pattern": re.compile(r"(?:CUDA\s+)?out\s+of\s+memory|torch\.cuda\.OutOfMemoryError|OOM\b", re.IGNORECASE),
     "suggestion": "降低 max-model-len 或增大 TP"},
    {"category": "cuda_error", "severity": "critical",
     "pattern": re.compile(r"CUDA\s*(?:error|Error|ERROR)\s*:?\s*(.+)|no kernel image is available", re.IGNORECASE),
     "suggestion": "检查 GPU 驱动与 CUDA 版本兼容性，或禁用问题算子"},
    {"category": "triton_compile", "severity": "critical",
     "pattern": re.compile(r"triton.*?compil(?:e|ation).*?(?:fail|error)|CompilationError.*?triton", re.IGNORECASE),
     "suggestion": "检查 FlagTree/Triton 版本兼容性"},
    {"category": "operator_error", "severity": "high",
     "pattern": re.compile(r"flag_gems.*?(?:Error|Exception)|GEMS\s+\w+.*?(?:Error|Failed)", re.IGNORECASE),
     "suggestion": "禁用问题算子后重启服务"},
    {"category": "model_load", "severity": "high",
     "pattern": re.compile(r"(?:model|weights?).*?(?:not\s+found|does\s+not\s+exist)|Cannot\s+load", re.IGNORECASE),
     "suggestion": "检查模型路径和权重完整性"},
    {"category": "port_conflict", "severity": "medium",
     "pattern": re.compile(r"[Aa]ddress\s+already\s+in\s+use|port.*?(?:in use|occupied)", re.IGNORECASE),
     "suggestion": "更换端口或 kill 占用进程"},
    {"category": "dependency", "severity": "medium",
     "pattern": re.compile(r"ModuleNotFoundError|ImportError:\s*(.+)", re.IGNORECASE),
     "suggestion": "安装缺失的 Python 包"},
    {"category": "timeout", "severity": "low",
     "pattern": re.compile(r"timed?\s*out|connection\s+refused|ConnectionResetError", re.IGNORECASE),
     "suggestion": "检查服务是否存活或网络连接"},
]

STEP_NAMES = {
    "01_container_preparation": ("1", "容器准备"),
    "02_environment_inspection": ("2", "环境检测"),
    "03_service_startup": ("3", "启服务"),
    "04_quick_accuracy": ("4", "精度评测"),
    "05_accuracy_tuning": ("5", "精度算子调优"),
    "06_quick_performance": ("6", "性能评测"),
    "07_performance_tuning": ("7", "性能算子调优"),
    "08_release": ("8", "自动发布"),
    "09_plugin_install": ("9", "Plugin安装"),
    "10_plugin_service_startup": ("10", "Plugin启服务"),
    "11_plugin_accuracy": ("11", "Plugin精度评测"),
    "12_plugin_performance": ("12", "Plugin性能评测"),
    "13_plugin_release": ("13", "Plugin发布"),
}


# =============================================================================
# 数据收集
# =============================================================================

def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_yaml(path: str) -> Optional[dict]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def read_checkpoint(workspace: str) -> Optional[dict]:
    return read_json(os.path.join(workspace, "logs", "checkpoint.json"))


def read_last_error(workspace: str) -> Optional[dict]:
    return read_json(os.path.join(workspace, "logs", "_last_error.json"))


def read_context(workspace: str) -> Optional[dict]:
    for p in [
        os.path.join(workspace, "shared", "context.yaml"),
        os.path.join(workspace, "config", "context_snapshot.yaml"),
    ]:
        ctx = read_yaml(p)
        if ctx:
            return ctx
    return None


def check_processes() -> Dict[str, Any]:
    """检查关键进程是否在运行。"""
    result = {"vllm": False, "sglang": False, "eval": False, "benchmark": False, "details": []}
    try:
        ps = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True, timeout=5)
        for line in ps.stdout.splitlines():
            lower = line.lower()
            if "vllm" in lower and "serve" in lower:
                result["vllm"] = True
                result["details"].append(line.strip())
            elif "sglang" in lower:
                result["sglang"] = True
                result["details"].append(line.strip())
            elif any(k in lower for k in ["fast_gpqa", "eval_monitor", "eval_aime", "eval_erqa"]):
                result["eval"] = True
                result["details"].append(line.strip())
            elif "benchmark_runner" in lower:
                result["benchmark"] = True
                result["details"].append(line.strip())
    except Exception:
        pass
    return result


def check_gpu() -> Dict[str, Any]:
    """检查 GPU 状态（多厂商统一，通过 detect_gpu.py）。"""
    result = {"available": False, "count": 0, "memory_used_pct": 0, "oom_likely": False, "details": ""}
    try:
        from detect_gpu import check_gpu_free
        free_info = check_gpu_free()
        if free_info and free_info.get("total", 0) > 0:
            result["available"] = True
            result["count"] = free_info["total"]
            details = free_info.get("details", [])
            total_used = sum(d.get("used_mib", 0) for d in details)
            total_mem = sum(d.get("total_mib", 0) for d in details)
            if total_mem > 0:
                result["memory_used_pct"] = round(total_used / total_mem * 100, 1)
            result["details"] = json.dumps(details)
    except Exception:
        pass
    return result


def check_service(port: int = 8000) -> Dict[str, Any]:
    """检查推理服务是否存活。"""
    result = {"running": False, "port": port, "model_id": ""}
    try:
        import urllib.request
        url = f"http://127.0.0.1:{port}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = data.get("data", [])
            if models:
                result["running"] = True
                result["model_id"] = models[0].get("id", "")
    except Exception:
        pass
    return result


def scan_logs(workspace: str, max_lines: int = 500) -> List[Dict[str, Any]]:
    """扫描日志文件提取错误。"""
    log_dir = os.path.join(workspace, "logs")
    errors = []
    if not os.path.isdir(log_dir):
        return errors

    log_files = sorted(Path(log_dir).glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log_file in log_files[:5]:  # 只看最近 5 个日志
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            # 只看最后 max_lines 行
            for i, line in enumerate(lines[-max_lines:]):
                for ep in ERROR_PATTERNS:
                    if ep["pattern"].search(line):
                        errors.append({
                            "file": log_file.name,
                            "line_num": len(lines) - max_lines + i + 1 if len(lines) > max_lines else i + 1,
                            "category": ep["category"],
                            "severity": ep["severity"],
                            "text": line.strip()[:200],
                            "suggestion": ep["suggestion"],
                        })
                        break  # 一行只匹配一个模式
        except Exception:
            pass

    # 去重：同文件同 category 只保留最后一条
    seen = {}
    for e in errors:
        key = (e["file"], e["category"])
        seen[key] = e
    return sorted(seen.values(), key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 4))


# =============================================================================
# 诊断推理
# =============================================================================

def infer_root_cause(
    checkpoint: Optional[dict],
    last_error: Optional[dict],
    context: Optional[dict],
    processes: dict,
    gpu: dict,
    service: dict,
    log_errors: list,
) -> Tuple[str, str]:
    """综合推理根因和恢复建议。"""
    causes = []
    suggestions = []

    # 从 last_error 推理
    if last_error:
        et = last_error.get("error_type", "")
        em = last_error.get("error_message", "")
        tool = last_error.get("tool", "")
        if "unreachable" in et or "timeout" in et.lower() or "连接" in em:
            causes.append(f"工具 {tool} 报告服务不可达: {em}")
        elif "oom" in et.lower():
            causes.append(f"工具 {tool} 报告 OOM: {em}")
        else:
            causes.append(f"工具 {tool} 报错 ({et}): {em}")

    # 从进程状态推理
    service_expected = True
    if checkpoint:
        action = checkpoint.get("action", "")
        if any(k in action for k in ["gpqa", "eval", "benchmark", "perf"]):
            service_expected = True
    if service_expected and not service["running"] and not processes["vllm"] and not processes["sglang"]:
        causes.append("推理服务未运行（进程不存在，端口无监听）")
        suggestions.append("重启推理服务")

    # 从 GPU 推理
    if gpu["available"] and gpu["memory_used_pct"] > 95:
        causes.append(f"GPU 显存占用 {gpu['memory_used_pct']}%，可能 OOM")
        suggestions.append("降低 max-model-len 或增大 TP")

    # 从日志推理
    for le in log_errors[:3]:
        if le["severity"] in ("critical", "high"):
            causes.append(f"日志 {le['file']}: {le['category']} — {le['text'][:100]}")
            suggestions.append(le["suggestion"])

    # 兜底
    if not causes:
        if checkpoint:
            causes.append(f"流程在 {checkpoint.get('step_name', checkpoint.get('step', '未知'))} 阶段中断，无明确错误信息")
            suggestions.append("检查 Claude 日志确认中断原因（可能是 API 超时或 context 溢出）")
        else:
            causes.append("无检查点和错误信息，可能是流程尚未开始或 Claude 启动即失败")
            suggestions.append("检查 Claude Code 日志和网络连接")

    root_cause = "; ".join(causes)
    suggested_action = "; ".join(dict.fromkeys(suggestions)) if suggestions else "从中断步骤恢复执行"

    return root_cause, suggested_action


def get_step_status(context: Optional[dict]) -> Tuple[List[str], List[str]]:
    """从 context 提取已完成和待恢复步骤。"""
    completed = []
    pending = []
    if not context:
        return completed, pending

    ledger = context.get("workflow_ledger", {}).get("steps", [])
    for step in ledger:
        sid = step.get("id", "")
        status = step.get("status", "")
        if status == "success":
            completed.append(sid)
        elif status in ("pending", "in_progress", "failed"):
            pending.append(sid)
    return completed, pending


# =============================================================================
# 输出格式化
# =============================================================================

def format_human(diag: dict) -> str:
    """格式化为人可读输出。"""
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append("  FlagOS 故障诊断")
    lines.append("=" * 50)

    # 中断位置
    ia = diag.get("interrupted_at", {})
    if ia:
        step = ia.get("step", "未知")
        num, name = STEP_NAMES.get(step, ("?", step))
        action = ia.get("action", "")
        pid_info = ""
        if ia.get("pid"):
            pid_info = f" (PID {ia['pid']}, {'运行中' if ia.get('pid_alive') else '已退出'})"
        lines.append(f"中断位置: [步骤{num}] {name} — {action}{pid_info}")
    else:
        lines.append("中断位置: 未知（无检查点）")

    # 最后错误
    le = diag.get("last_error")
    if le:
        lines.append(f"最后错误: {le.get('error_message', '未知')}")
        lines.append(f"         工具: {le.get('tool', '?')}, 退出码: {le.get('exit_code', '?')}")
    lines.append("")

    # 环境状态
    lines.append("环境状态:")
    lines.append(f"  容器: 运行中")
    gpu = diag.get("gpu_status", {})
    if gpu.get("available"):
        lines.append(f"  GPU: {gpu.get('count', '?')}卡, 显存占用 {gpu.get('memory_used_pct', '?')}%")
    else:
        lines.append("  GPU: 无法检测")
    svc = diag.get("service_status", {})
    if svc.get("running"):
        lines.append(f"  服务: 运行中 (端口 {svc.get('port')}, 模型 {svc.get('model_id', '?')})")
    else:
        lines.append(f"  服务: 未运行 (端口 {svc.get('port', '?')} 无监听)")

    procs = diag.get("process_status", {})
    running = [k for k in ["vllm", "sglang", "eval", "benchmark"] if procs.get(k)]
    if running:
        lines.append(f"  活跃进程: {', '.join(running)}")
    lines.append("")

    # 日志发现
    log_errors = diag.get("log_errors", [])
    if log_errors:
        lines.append("日志发现:")
        for le in log_errors[:5]:
            lines.append(f"  {le['file']}: [{le['category']}] {le['text'][:120]}")
        lines.append("")

    # 根因和建议
    lines.append(f"根因: {diag.get('root_cause', '未知')}")
    lines.append(f"建议: {diag.get('suggested_action', '从中断步骤恢复')}")
    lines.append("")

    # 步骤进度
    completed = diag.get("completed_steps", [])
    pending = diag.get("pending_steps", [])
    if completed or pending:
        parts = []
        for sid in completed:
            num, name = STEP_NAMES.get(sid, ("?", sid))
            parts.append(f"{num}{name} ✓")
        lines.append(f"已完成: {' → '.join(parts) if parts else '无'}")
        parts = []
        for sid in pending:
            num, name = STEP_NAMES.get(sid, ("?", sid))
            parts.append(f"{num}{name}")
        lines.append(f"待恢复: {' → '.join(parts) if parts else '无'}")

    lines.append("=" * 50)
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# 主诊断流程
# =============================================================================

def diagnose(workspace: str) -> dict:
    """执行完整诊断，返回结构化结果。"""
    checkpoint = read_checkpoint(workspace)
    last_error = read_last_error(workspace)
    context = read_context(workspace)
    processes = check_processes()
    gpu = check_gpu()

    # 从 context 获取服务端口
    port = 8000
    if context:
        port = context.get("service", {}).get("port", 8000) or 8000
    service = check_service(port)

    log_errors = scan_logs(workspace)
    completed, pending = get_step_status(context)

    root_cause, suggested_action = infer_root_cause(
        checkpoint, last_error, context, processes, gpu, service, log_errors,
    )

    # 检查 checkpoint PID 是否存活
    pid_alive = False
    if checkpoint and checkpoint.get("pid"):
        try:
            os.kill(checkpoint["pid"], 0)
            pid_alive = True
        except (OSError, TypeError):
            pass

    # 构建诊断结果
    diag = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "interrupted_at": {
            "step": checkpoint.get("step", "") if checkpoint else "",
            "step_name": checkpoint.get("step_name", "") if checkpoint else "",
            "action": checkpoint.get("action", "") if checkpoint else "",
            "pid": checkpoint.get("pid") if checkpoint else None,
            "pid_alive": pid_alive,
            "updated_at": checkpoint.get("updated_at", "") if checkpoint else "",
        } if checkpoint else {},
        "last_error": last_error,
        "process_status": processes,
        "gpu_status": {
            "available": gpu["available"],
            "count": gpu["count"],
            "memory_used_pct": gpu["memory_used_pct"],
        },
        "service_status": {
            "running": service["running"],
            "port": port,
            "model_id": service.get("model_id", ""),
        },
        "log_errors": log_errors[:10],
        "root_cause": root_cause,
        "suggested_action": suggested_action,
        "completed_steps": completed,
        "pending_steps": pending,
    }

    # 写入诊断结果到 workspace
    diag_path = os.path.join(workspace, "logs", "failure_diagnosis.json")
    try:
        os.makedirs(os.path.dirname(diag_path), exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return diag


def main():
    parser = argparse.ArgumentParser(description="FlagOS 流程中断自动诊断")
    parser.add_argument("--workspace", default="/flagos-workspace", help="工作目录路径")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    # 检查流程是否已正常完成
    context = read_context(args.workspace)
    if context and context.get("workflow", {}).get("all_done") is True:
        checkpoint = read_checkpoint(args.workspace)
        if not checkpoint:
            if args.json:
                print(json.dumps({"status": "completed", "message": "流程已正常完成"}, ensure_ascii=False))
            else:
                print("流程已正常完成，无需诊断。")
            sys.exit(1)

    diag = diagnose(args.workspace)

    if args.json:
        print(json.dumps(diag, ensure_ascii=False, indent=2))
    else:
        print(format_human(diag))

    sys.exit(0)


if __name__ == "__main__":
    main()
