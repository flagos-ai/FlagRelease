#!/usr/bin/env python3
"""
服务活性监控 — 后台线程，评测/性能测试期间持续检查服务进程和日志。

用法:
    monitor = ServiceMonitor(log_path="/flagos-workspace/logs/startup_flagos.log")
    monitor.start()
    # ... 执行评测 ...
    if monitor.is_dead():
        info = monitor.death_reason()
        print(f"服务崩溃: {info['type']} — {info['detail']}")
    monitor.stop()
"""

import os
import re
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

FATAL_PATTERNS = [
    (re.compile(r"(?:CUDA\s+)?out\s+of\s+memory|torch\.cuda\.OutOfMemoryError|\bOOM\b", re.I), "oom", "CUDA out of memory"),
    (re.compile(r"CUDA\s*(?:error|Error|ERROR)\s*:|CUDAError|no kernel image is available", re.I), "cuda_error", "CUDA 错误"),
    (re.compile(r"Segmentation fault|SIGSEGV|SIGKILL", re.I), "segfault", "段错误 (Segmentation fault)"),
    (re.compile(r"Killed\s+.*(?:vllm|sglang)|killed by signal", re.I), "killed", "进程被杀"),
    (re.compile(r"Address already in use", re.I), "port_conflict", "端口被占用"),
]

TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)", re.I)
ERROR_RE = re.compile(r"^\w*(?:Error|Exception):", re.I)
WARN_SKIP = ("FutureWarning", "DeprecationWarning", "UserWarning")


class ServiceMonitor:
    """后台监控服务进程存活 + 日志致命信号。"""

    def __init__(
        self,
        log_path: Optional[str] = None,
        check_interval: int = 10,
        process_patterns: tuple = ("vllm", "sglang", "flagscale"),
        grace_period: int = 30,
    ):
        self._log_path = log_path
        self._interval = check_interval
        self._proc_patterns = process_patterns
        self._grace_period = grace_period

        self._dead = threading.Event()
        self._stop = threading.Event()
        self._reason: Dict = {}
        self._thread: Optional[threading.Thread] = None
        self._log_offset = 0

        if log_path and os.path.isfile(log_path):
            self._log_offset = os.path.getsize(log_path)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._dead.clear()
        self._reason = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_dead(self) -> bool:
        return self._dead.is_set()

    def death_reason(self) -> Dict:
        return dict(self._reason) if self._reason else {}

    def _set_dead(self, reason_type: str, detail: str, log_line: str = "") -> None:
        if self._dead.is_set():
            return
        self._reason = {
            "type": reason_type,
            "detail": detail,
            "log_line": log_line,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._dead.set()

    def _loop(self) -> None:
        start_time = time.monotonic()
        no_process_since: Optional[float] = None

        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break

            elapsed = time.monotonic() - start_time
            if elapsed < self._grace_period:
                continue

            # 1. 日志致命信号检测
            fatal = self._check_log_fatal()
            if fatal:
                self._set_dead(fatal["type"], fatal["detail"], fatal.get("line", ""))
                break

            # 2. 进程存活检测
            alive = self._check_process_alive()
            if not alive:
                if no_process_since is None:
                    no_process_since = time.monotonic()
                elif time.monotonic() - no_process_since > 10:
                    self._set_dead("process_exited", "服务进程已退出（连续 10s 未检测到）")
                    break
            else:
                no_process_since = None

    def _check_process_alive(self) -> bool:
        try:
            result = subprocess.run(
                ["ps", "-eo", "args"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                for pat in self._proc_patterns:
                    if pat in line and "grep" not in line and "service_monitor" not in line:
                        return True
        except Exception:
            return True
        return False

    def _check_log_fatal(self) -> Optional[Dict]:
        if not self._log_path or not os.path.isfile(self._log_path):
            return None
        try:
            size = os.path.getsize(self._log_path)
            if size <= self._log_offset:
                return None
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_offset)
                new_content = f.read()
            self._log_offset = size
        except Exception:
            return None

        has_traceback = False
        for line in new_content.splitlines():
            s = line.strip()
            if not s:
                continue
            for pat, sig_type, sig_desc in FATAL_PATTERNS:
                if pat.search(s):
                    return {"type": sig_type, "detail": sig_desc, "line": s[:200]}
            if TRACEBACK_RE.search(s):
                has_traceback = True
            if has_traceback and ERROR_RE.search(s):
                if not any(w in s for w in WARN_SKIP):
                    return {"type": "traceback_error", "detail": "Python 异常", "line": s[:200]}

        return None


def find_latest_startup_log(log_dir: str = "/flagos-workspace/logs") -> Optional[str]:
    """查找最新的 startup_*.log 文件。"""
    if not os.path.isdir(log_dir):
        return None
    candidates = []
    for f in os.listdir(log_dir):
        if f.startswith("startup_") and f.endswith(".log"):
            path = os.path.join(log_dir, f)
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
