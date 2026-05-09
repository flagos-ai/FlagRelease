#!/usr/bin/env python3
"""
error_writer.py — 统一错误/检查点持久化模块

工具脚本 import 后使用，失败时自动写文件，不依赖 Claude。
部署到容器 /flagos-workspace/scripts/error_writer.py

Usage:
    from error_writer import write_last_error, write_checkpoint

    # 工具脚本入口
    write_checkpoint("04_accuracy_eval", "精度评测", "running_fast_gpqa_v2",
                     action_detail="fast_gpqa.py --version V2")

    # 异常处理
    try:
        ...
    except Exception as e:
        write_last_error("fast_gpqa.py", "service_unreachable",
                         str(e), traceback_str=traceback.format_exc())
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# 日志目录优先级
_LOG_DIRS = ["/flagos-workspace/logs", "/tmp"]


def _find_log_dir() -> str:
    for d in _LOG_DIRS:
        if os.path.isdir(d):
            return d
    os.makedirs(_LOG_DIRS[0], exist_ok=True)
    return _LOG_DIRS[0]


def _atomic_write(path: str, data: dict):
    """原子写入：先写 tmp 再 rename，避免读到半成品。"""
    dir_path = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_last_error(
    tool: str,
    error_type: str,
    error_message: str,
    traceback_str: Optional[str] = None,
    partial_result: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    exit_code: int = 1,
):
    """写入 _last_error.json（覆盖）+ _error_history.jsonl（追加）。"""
    log_dir = _find_log_dir()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    record = {
        "tool": tool,
        "timestamp": now,
        "exit_code": exit_code,
        "error_type": error_type,
        "error_message": error_message,
    }
    if traceback_str:
        record["traceback"] = traceback_str
    if partial_result:
        record["partial_result"] = partial_result
    if context:
        record["context"] = context
    record["_meta"] = {
        "tool": "产生错误的工具脚本名",
        "error_type": "错误分类（如 KeyError / service_unreachable / timeout）",
        "error_message": "错误详情",
        "exit_code": "进程退出码（非零表示异常）",
        "traceback": "Python 完整堆栈（可选）",
        "partial_result": "错误发生前已产出的部分结果（可选）",
        "context": "错误发生时的上下文信息（可选）",
    }

    # 覆盖写最新错误
    error_path = os.path.join(log_dir, "_last_error.json")
    try:
        _atomic_write(error_path, record)
    except Exception as e:
        print(f"[error_writer] 写入 _last_error.json 失败: {e}")

    # 追加历史
    history_path = os.path.join(log_dir, "_error_history.jsonl")
    try:
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # best-effort 同步到 context.yaml
    _sync_error_to_context(record, log_dir)


def write_checkpoint(
    step: str,
    step_name: str,
    action: str,
    action_detail: Optional[str] = None,
    last_success: Optional[Dict[str, Any]] = None,
):
    """写入 checkpoint.json（覆盖），记录当前正在执行的操作。"""
    log_dir = _find_log_dir()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    record = {
        "step": step,
        "step_name": step_name,
        "action": action,
        "started_at": now,
        "updated_at": now,
        "pid": os.getpid(),
    }
    if action_detail:
        record["action_detail"] = action_detail
    if last_success:
        record["last_success"] = last_success
    record["_meta"] = {
        "step": "当前步骤编号（如 04_accuracy_eval）",
        "step_name": "步骤中文名",
        "action": "当前正在执行的操作标识",
        "action_detail": "操作详情（如完整命令，可选）",
        "last_success": "上一个成功完成的操作信息（可选）",
        "started_at": "检查点创建时间 (ISO 8601)",
        "updated_at": "检查点最后更新时间 (ISO 8601)",
        "pid": "执行进程 ID",
    }

    ckpt_path = os.path.join(log_dir, "checkpoint.json")
    try:
        _atomic_write(ckpt_path, record)
    except Exception as e:
        print(f"[error_writer] 写入 checkpoint.json 失败: {e}")


def clear_checkpoint():
    """流程正常完成时清除 checkpoint。"""
    log_dir = _find_log_dir()
    ckpt_path = os.path.join(log_dir, "checkpoint.json")
    try:
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
    except Exception:
        pass


def _sync_error_to_context(record: dict, log_dir: str):
    """best-effort 将错误信息同步到 context.yaml（通过 update_context.py 避免破坏 YAML 格式）。"""
    import subprocess

    update_script = None
    candidates = [
        os.path.join(log_dir, "..", "scripts", "update_context.py"),
        "/flagos-workspace/scripts/update_context.py",
    ]
    for p in candidates:
        p = os.path.realpath(p)
        if os.path.isfile(p):
            update_script = p
            break

    if not update_script:
        return

    error_json = json.dumps({
        "step": record.get("tool", ""),
        "action": "",
        "error_type": record.get("error_type", ""),
        "error_message": record.get("error_message", ""),
        "timestamp": record.get("timestamp", ""),
        "recoverable": True,
    }, ensure_ascii=False)

    python_bin = "/opt/conda/bin/python3" if os.path.isfile("/opt/conda/bin/python3") else "python3"

    try:
        subprocess.run(
            [python_bin, update_script,
             "--json-set", f"workflow.last_error={error_json}",
             "--json"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass
