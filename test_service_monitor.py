#!/usr/bin/env python3
"""
service_monitor.py 单元测试
验证：线程生命周期、日志致命信号检测、进程检测、边界条件
"""
import os
import sys
import tempfile
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
    "skills/flagos-service-startup/tools"))

from service_monitor import ServiceMonitor, find_latest_startup_log, FATAL_PATTERNS

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")


# ============================================================
print("\n=== Test 1: 基本生命周期 ===")
m = ServiceMonitor(log_path=None, check_interval=1, grace_period=0)
check("初始状态 is_dead=False", not m.is_dead())
check("初始状态 death_reason={}", m.death_reason() == {})
m.start()
check("start 后线程存活", m._thread is not None and m._thread.is_alive())
m.stop()
check("stop 后线程结束", not m._thread.is_alive())
check("stop 后 is_dead 仍为 False（正常停止）", not m.is_dead())


# ============================================================
print("\n=== Test 2: 重复 start/stop ===")
m = ServiceMonitor(log_path=None, check_interval=1, grace_period=0)
m.start()
m.start()  # 重复 start 不应崩溃
check("重复 start 不崩溃", m._thread.is_alive())
m.stop()
m.stop()  # 重复 stop 不应崩溃
check("重复 stop 不崩溃", True)


# ============================================================
print("\n=== Test 3: 日志致命信号检测 — OOM ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("INFO: Loading model weights...\n")
    f.write("INFO: Weights loaded.\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
m.start()
time.sleep(0.5)
check("正常日志不触发 is_dead", not m.is_dead())

# 追加 OOM
with open(log_path, 'a') as f:
    f.write("torch.cuda.OutOfMemoryError: CUDA out of memory\n")

time.sleep(2.5)
check("OOM 触发 is_dead", m.is_dead())
reason = m.death_reason()
check("reason.type == 'oom'", reason.get("type") == "oom", f"got {reason}")
check("reason.log_line 包含 OOM 信息", "OutOfMemoryError" in reason.get("log_line", ""))
check("reason.timestamp 非空", bool(reason.get("timestamp")))
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 4: 日志致命信号检测 — CUDA error ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("Starting server...\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
m.start()
time.sleep(0.5)

with open(log_path, 'a') as f:
    f.write("CUDA Error: no kernel image is available for execution\n")

time.sleep(2.5)
check("CUDA error 触发 is_dead", m.is_dead())
check("reason.type == 'cuda_error'", m.death_reason().get("type") == "cuda_error")
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 5: 日志致命信号检测 — Traceback + Error 组合 ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("Normal log line\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
m.start()
time.sleep(0.5)

with open(log_path, 'a') as f:
    f.write("Traceback (most recent call last):\n")
    f.write("  File 'foo.py', line 10\n")
    f.write("RuntimeError: something went wrong\n")

time.sleep(2.5)
check("Traceback+Error 触发 is_dead", m.is_dead())
check("reason.type == 'traceback_error'", m.death_reason().get("type") == "traceback_error")
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 6: Warning 不触发误报 ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("Normal log line\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
m.start()
time.sleep(0.5)

with open(log_path, 'a') as f:
    f.write("Traceback (most recent call last):\n")
    f.write("  File 'foo.py', line 10\n")
    f.write("FutureWarning: something deprecated\n")

time.sleep(2.5)
check("FutureWarning 不触发 is_dead", not m.is_dead())
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 7: 日志致命信号检测 — Segfault ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("Starting...\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
m.start()
time.sleep(0.5)

with open(log_path, 'a') as f:
    f.write("Segmentation fault (core dumped)\n")

time.sleep(2.5)
check("Segfault 触发 is_dead", m.is_dead())
check("reason.type == 'segfault'", m.death_reason().get("type") == "segfault")
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 8: 日志致命信号检测 — port conflict ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("Binding...\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
m.start()
time.sleep(0.5)

with open(log_path, 'a') as f:
    f.write("OSError: [Errno 98] Address already in use\n")

time.sleep(2.5)
check("port_conflict 触发 is_dead", m.is_dead())
check("reason.type == 'port_conflict'", m.death_reason().get("type") == "port_conflict")
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 9: grace_period 内不检测 ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    f.write("torch.cuda.OutOfMemoryError: CUDA out of memory\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=60)
m.start()
time.sleep(2.5)
check("grace_period=60 内不触发 is_dead", not m.is_dead())
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 10: 无日志路径时只做进程检测 ===")
m = ServiceMonitor(log_path=None, check_interval=1, grace_period=0)
m.start()
time.sleep(2.5)
# 当前机器上应该没有 vllm/sglang 进程，但进程检测需要连续 10s 无进程才判死
# 所以 2.5s 内不应该触发
# 实际上：check_interval=1, grace_period=0, 第一次检测在 1s 后
# 如果无进程，no_process_since 设置，再过 10s 才判死
# 2.5s 内不会判死
check("无日志路径 2.5s 内不误判", not m.is_dead())
m.stop()


# ============================================================
print("\n=== Test 11: 进程退出检测（需要连续 10s 无进程） ===")
m = ServiceMonitor(log_path=None, check_interval=1, grace_period=0,
                   process_patterns=("__nonexistent_process_xyz__",))
m.start()
# 需要等 grace_period(0) + 第一次检测(1s) + 连续无进程(10s) + 余量
time.sleep(14)
check("连续无进程 >10s 触发 is_dead", m.is_dead())
reason = m.death_reason()
check("reason.type == 'process_exited'", reason.get("type") == "process_exited")
m.stop()


# ============================================================
print("\n=== Test 12: _set_dead 幂等性（只记录第一次） ===")
m = ServiceMonitor(log_path=None, check_interval=100, grace_period=0)
m._set_dead("oom", "first crash", "line1")
m._set_dead("cuda_error", "second crash", "line2")
check("只记录第一次崩溃", m.death_reason()["type"] == "oom")
check("detail 是第一次的", m.death_reason()["detail"] == "first crash")


# ============================================================
print("\n=== Test 13: find_latest_startup_log ===")
with tempfile.TemporaryDirectory() as tmpdir:
    # 空目录
    result = find_latest_startup_log(tmpdir)
    check("空目录返回 None", result is None)

    # 创建文件
    p1 = os.path.join(tmpdir, "startup_native.log")
    with open(p1, 'w') as f:
        f.write("old")
    time.sleep(0.1)
    p2 = os.path.join(tmpdir, "startup_flagos.log")
    with open(p2, 'w') as f:
        f.write("new")

    result = find_latest_startup_log(tmpdir)
    check("返回最新的 startup_*.log", result == p2, f"got {result}")

    # 非 startup_ 文件不匹配
    p3 = os.path.join(tmpdir, "other.log")
    with open(p3, 'w') as f:
        f.write("irrelevant")
    result = find_latest_startup_log(tmpdir)
    check("非 startup_ 文件不匹配", result == p2)

# 不存在的目录
result = find_latest_startup_log("/nonexistent_dir_xyz")
check("不存在的目录返回 None", result is None)


# ============================================================
print("\n=== Test 14: 日志文件不存在时不崩溃 ===")
m = ServiceMonitor(log_path="/nonexistent/path/to/log.log", check_interval=1, grace_period=0)
m.start()
time.sleep(2.5)
check("日志文件不存在不崩溃", not m.is_dead())
m.stop()


# ============================================================
print("\n=== Test 15: 日志 offset 正确（只读增量） ===")
with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
    log_path = f.name
    # 初始内容包含 OOM（但 monitor 创建时会记录 offset，跳过已有内容）
    f.write("torch.cuda.OutOfMemoryError: old crash\n")

m = ServiceMonitor(log_path=log_path, check_interval=1, grace_period=0)
check("初始 offset 跳过已有内容", m._log_offset > 0)
m.start()
time.sleep(2.5)
check("已有的 OOM 不触发 is_dead", not m.is_dead())

# 追加新的正常日志
with open(log_path, 'a') as f:
    f.write("INFO: All good\n")
time.sleep(2.5)
check("正常追加不触发 is_dead", not m.is_dead())
m.stop()
os.unlink(log_path)


# ============================================================
print("\n=== Test 16: FATAL_PATTERNS 正则覆盖测试 ===")
test_lines = [
    ("CUDA out of memory. Tried to allocate 2.00 GiB", "oom"),
    ("torch.cuda.OutOfMemoryError: CUDA out of memory", "oom"),
    ("OOM when allocating tensor", "oom"),
    ("CUDA Error: no kernel image is available", "cuda_error"),
    ("CUDAError: invalid device ordinal", "cuda_error"),
    ("CUDA error: device-side assert triggered", "cuda_error"),
    ("Segmentation fault", "segfault"),
    ("SIGSEGV received", "segfault"),
    ("SIGKILL received", "segfault"),
    ("Killed vllm serve process", "killed"),
    ("killed by signal 9", "killed"),
    ("Address already in use", "port_conflict"),
]
for line, expected_type in test_lines:
    matched = None
    for pat, sig_type, _ in FATAL_PATTERNS:
        if pat.search(line):
            matched = sig_type
            break
    check(f"'{line[:50]}...' → {expected_type}", matched == expected_type,
          f"got {matched}")


# ============================================================
print("\n=== Test 17: 非致命日志不匹配 ===")
safe_lines = [
    "INFO: Loading model weights...",
    "INFO: Model loaded successfully",
    "INFO: Uvicorn running on 0.0.0.0:8000",
    "WARNING: Some deprecation notice",
    "DEBUG: Processing request",
    "INFO: GEMS softmax registered",
]
for line in safe_lines:
    matched = None
    for pat, sig_type, _ in FATAL_PATTERNS:
        if pat.search(line):
            matched = sig_type
            break
    check(f"'{line[:50]}' 不匹配致命模式", matched is None, f"误匹配为 {matched}")


# ============================================================
print("\n=== Test 18: 集成路径验证 — fast_gpqa.py import 路径 ===")
from pathlib import Path
fast_gpqa_dir = os.path.join(os.path.dirname(__file__),
    "skills/flagos-eval-comprehensive/tools")
monitor_dir = os.path.join(os.path.dirname(__file__),
    "skills/flagos-service-startup/tools")
# 模拟 fast_gpqa.py 的 sys.path 逻辑
actual_path = str(Path(fast_gpqa_dir).resolve().parent.parent / "flagos-service-startup" / "tools")
check("fast_gpqa.py import 路径正确", os.path.isfile(os.path.join(actual_path, "service_monitor.py")),
      f"path={actual_path}")


# ============================================================
print("\n=== Test 19: 集成路径验证 — benchmark_runner.py import 路径 ===")
bench_dir = os.path.join(os.path.dirname(__file__),
    "skills/flagos-performance-testing/tools")
expected_path = str(Path(bench_dir).resolve().parent.parent / "flagos-service-startup" / "tools")
check("benchmark_runner.py import 路径正确", os.path.isfile(os.path.join(expected_path, "service_monitor.py")),
      f"path={expected_path}")


# ============================================================
print(f"\n{'='*60}")
print(f"  结果: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
sys.exit(1 if FAIL > 0 else 0)
