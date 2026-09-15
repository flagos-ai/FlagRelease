#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
task_runner.py — 长任务执行器（长任务执行协议核心组件）

背景（2026-08 V3 精度评测会话被误杀事故根因）：
  Claude Code Bash 工具前台命令有 10 分钟硬性上限（600000ms），超过自动转后台
  并静默等待完成通知；部署端批次控制器按"10 分钟无输出 = 会话失败"终止会话。
  评测/起服务/性能测试/算子调优/发布推送等长任务若让 Claude 前台阻塞等待，
  会话会被误杀、任务连带丢失。

协议（本脚本 + 编排层指令配合）：
  1. detached 启动：docker exec -d 容器内启动本脚本（或宿主机 python3 ... &），立即返回
  2. 本脚本以新会话（setsid 语义）Popen 启动 --cmd 任务，stdout/stderr 全部进 --log
  3. 立即写状态文件 {status:running, pid, started_at} —— Claude 轮询的事实来源
  4. 15s 间隔监控子进程；--timeout 到达 → SIGTERM 整进程组 → 15s 宽限 → SIGKILL
  5. 任务退出写终态 {status:done|error|timeout, exit_code, finished_at}
  6. 任务进程与 Claude 会话彻底脱离：会话被杀任务继续跑，恢复会话读状态文件
     直接接管（断点恢复）

用法:
  python3 task_runner.py --cmd 'python3 eval_wrapper.py --eval-cmd ...' \
      --state /flagos-workspace/logs/tasks/eval_v2.state \
      --log /flagos-workspace/logs/tasks/eval_v2.log \
      [--timeout 22500]

Claude 轮询约定（每条轮询指令一次调用，单次 <10 分钟，每次都有输出）:
  sleep 480 && docker exec <C> cat <state> && echo --- && docker exec <C> tail -3 <log>
  status=running  → 继续轮询（sleep 480 后重复同一条）
  status=done     → 任务成功，读取结果文件继续后续步骤
  status=error    → 读日志按原错误处理规则处理（服务崩溃/评测失败等）
  status=timeout  → 超过 --timeout 总闸，读日志诊断

退出码: 0 = 任务成功(done)  1 = 任务失败(error)  2 = 超时(timeout)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

POLL_INTERVAL = 15          # 子进程监控间隔（秒）
SIGTERM_GRACE = 15          # SIGTERM 后宽限（秒），超时转 SIGKILL


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def write_state(path, **fields):
    """原子写状态文件（临时文件 + rename，避免轮询读到半截 JSON）"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(fields, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="长任务执行器（detached 启动 + 状态文件 + 超时管理）")
    parser.add_argument("--cmd", required=True, help="要执行的 shell 命令（原样传给 sh -c）")
    parser.add_argument("--state", required=True, help="状态文件路径（JSON，Claude 轮询的事实来源）")
    parser.add_argument("--log", required=True, help="日志文件路径（任务 stdout/stderr 全部落此文件）")
    parser.add_argument("--timeout", type=int, default=0, help="任务总超时（秒），0 = 不限")
    args = parser.parse_args()

    for path in (args.state, args.log):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    started = now_iso()
    write_state(args.state, status="running", pid=os.getpid(), started_at=started)

    # start_new_session=True：任务进程独立会话 + 独立进程组。
    # 好处 1：任务与 task_runner / Claude 会话彻底脱离，会话被杀任务不受影响；
    # 好处 2：超时/终止时 killpg 可对整进程组（含任务派生的子进程）发信号。
    try:
        with open(args.log, "a") as logf:
            proc = subprocess.Popen(["sh", "-c", args.cmd],
                                    stdout=logf, stderr=logf,
                                    start_new_session=True)
    except OSError as e:
        write_state(args.state, status="error", error=f"启动失败: {e}",
                    started_at=started, finished_at=now_iso())
        print(f"[task_runner] 启动失败: {e}", file=sys.stderr)
        return 1

    t0 = time.monotonic()
    while True:
        rc = proc.poll()
        if rc is not None:
            status = "done" if rc == 0 else "error"
            write_state(args.state, status=status, exit_code=rc,
                        started_at=started, finished_at=now_iso())
            return 0 if rc == 0 else 1
        if args.timeout and (time.monotonic() - t0) >= args.timeout:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=SIGTERM_GRACE)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            except ProcessLookupError:
                pass
            write_state(args.state, status="timeout", exit_code=None,
                        started_at=started, finished_at=now_iso(),
                        timeout_seconds=args.timeout)
            return 2
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
