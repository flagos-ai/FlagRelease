#!/usr/bin/env python3
"""
评测包装器 — 启动 fast_gpqa.py 并监控其运行状态。

设计目标：让 Claude 只需一条 docker exec 命令，阻塞等待结果，无需轮询。
脚本内部负责：
  1. 后台启动 fast_gpqa.py
  2. 实时监控：评测进程存活、服务进程存活、日志异常、输出停滞
  3. 正常完成 → 输出结果 JSON，退出码 0
  4. 异常 → 输出结构化错误摘要，退出码 1

用法:
  python3 eval_wrapper.py --eval-cmd "python3 fast_gpqa.py --config fast_gpqa_config.yaml --output /flagos-workspace/results/gpqa_native.json" \
      --service-log /flagos-workspace/logs/startup_native.log \
      --stall-timeout 300 \
      --max-timeout 3600

输出约定:
  正常: 最后一行为 JSON (结果文件内容)，退出码 0
  异常: [EVAL_ERROR] 开头的结构化错误，退出码 1
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

FATAL_LOG_PATTERNS = [
    (re.compile(r"(?:CUDA\s+)?out\s+of\s+memory|torch\.cuda\.OutOfMemoryError|\bOOM\b", re.I), "oom"),
    (re.compile(r"CUDA\s*(?:error|Error|ERROR)\s*:|CUDAError|no kernel image", re.I), "cuda_error"),
    (re.compile(r"Segmentation fault|SIGSEGV|SIGKILL", re.I), "segfault"),
    (re.compile(r"Killed\s+.*(?:vllm|sglang)|killed by signal", re.I), "process_killed"),
    (re.compile(r"Address already in use", re.I), "port_conflict"),
    (re.compile(r"Connection refused", re.I), "connection_refused"),
]

SERVICE_PROCESS_PATTERNS = ("vllm", "sglang", "flagscale")


def check_service_alive() -> bool:
    try:
        result = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            for pat in SERVICE_PROCESS_PATTERNS:
                if pat in line and "grep" not in line and "eval_wrapper" not in line:
                    return True
    except Exception:
        return True
    return False


def check_service_healthy(api_base: str) -> bool:
    """检查推理服务是否仍在正常响应（进程存活 + API 可达）"""
    if not check_service_alive():
        return False
    import urllib.request
    import urllib.error
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    try:
        req = urllib.request.Request(f"{base}/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def scan_log_fatal(log_path: str, offset: int) -> tuple:
    """扫描日志文件新增内容，返回 (new_offset, fatal_info_or_None)"""
    if not log_path or not os.path.isfile(log_path):
        return offset, None
    try:
        size = os.path.getsize(log_path)
        if size <= offset:
            return offset, None
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            new_content = f.read()
        new_offset = size
    except Exception:
        return offset, None

    for line in new_content.splitlines():
        s = line.strip()
        if not s:
            continue
        for pat, sig_type in FATAL_LOG_PATTERNS:
            if pat.search(s):
                return new_offset, {"type": sig_type, "line": s[:300]}
    return new_offset, None


def get_eval_output_file(eval_cmd: str) -> Optional[str]:
    """从命令行中提取 --output 参数值"""
    parts = eval_cmd.split()
    for i, p in enumerate(parts):
        if p == "--output" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def read_context_yaml(path: str) -> dict:
    """从 context.yaml 读取 service.port 等关键信息"""
    try:
        import yaml
        with open(path) as f:
            ctx = yaml.safe_load(f)
        return {
            'port': ctx.get('service', {}).get('port', 8000),
            'model_name': ctx.get('model', {}).get('name', ''),
        }
    except Exception:
        return {'port': 8000, 'model_name': ''}


def auto_detect_model_name(api_base: str = "http://localhost:8000/v1") -> Optional[str]:
    """查询 /v1/models 获取实际 served model name"""
    import urllib.request
    import urllib.error
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    try:
        req = urllib.request.Request(f"{base}/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            return models[0].get("id", "")
    except Exception:
        pass
    return None


def inject_model_name(eval_cmd: str, api_base: str = "http://localhost:8000/v1") -> str:
    """如果 eval_cmd 中没有 --model-name，自动从 /v1/models 探测并注入"""
    if "--model-name" in eval_cmd:
        return eval_cmd
    model_name = auto_detect_model_name(api_base)
    if not model_name:
        return eval_cmd
    if "fast_gpqa.py" in eval_cmd:
        eval_cmd = eval_cmd.replace("fast_gpqa.py", f"fast_gpqa.py --model-name '{model_name}'", 1)
        print(f"[WRAPPER] 自动探测模型名: {model_name}")
    return eval_cmd


def main():
    parser = argparse.ArgumentParser(description="评测包装器")
    parser.add_argument("--eval-cmd", required=True, help="评测命令（在当前目录执行）")
    parser.add_argument("--service-log", default="", help="服务日志路径（用于检测服务崩溃）")
    parser.add_argument("--context-yaml", default="/flagos-workspace/shared/context.yaml",
                        help="context.yaml 路径（自动读取 port 等信息）")
    parser.add_argument("--api-base", default="",
                        help="API 地址（显式指定时优先级最高，否则从 context-yaml 读取 port）")
    parser.add_argument("--stall-timeout", type=int, default=300,
                        help="评测进程无新输出超过此秒数视为卡死 (默认 300s)")
    parser.add_argument("--max-timeout", type=int, default=3600,
                        help="评测最大允许时间 (默认 3600s)")
    parser.add_argument("--check-interval", type=int, default=15,
                        help="监控检查间隔 (默认 15s)")
    args = parser.parse_args()

    eval_cmd = args.eval_cmd
    service_log = args.service_log
    stall_timeout = args.stall_timeout
    max_timeout = args.max_timeout
    check_interval = args.check_interval

    # 确定 api_base：--api-base > context.yaml port > 默认 8000
    if args.api_base:
        api_base = args.api_base
    else:
        ctx_info = read_context_yaml(args.context_yaml)
        port = ctx_info['port']
        api_base = f"http://localhost:{port}/v1"
        if port != 8000:
            print(f"[WRAPPER] 从 context.yaml 读取端口: {port}")

    # 自动注入模型名（防止使用模板默认值或遗漏）
    eval_cmd = inject_model_name(eval_cmd, api_base)

    # 自动注入 --api-base（防止 fast_gpqa.py 使用默认端口而非实际端口）
    if "--api-base" not in eval_cmd and "fast_gpqa.py" in eval_cmd:
        eval_cmd = eval_cmd.replace("fast_gpqa.py", f"fast_gpqa.py --api-base '{api_base}'", 1)
        print(f"[WRAPPER] 自动注入 api_base: {api_base}")

    output_file = get_eval_output_file(eval_cmd)

    # 记录服务日志初始偏移
    log_offset = 0
    if service_log and os.path.isfile(service_log):
        log_offset = os.path.getsize(service_log)

    # 启动评测进程，捕获 stdout/stderr 到临时文件
    eval_log = "/tmp/eval_wrapper_output.log"
    print(f"[WRAPPER] 启动评测: {eval_cmd}")
    print(f"[WRAPPER] 监控参数: stall_timeout={stall_timeout}s, max_timeout={max_timeout}s")
    if service_log:
        print(f"[WRAPPER] 服务日志: {service_log}")
    sys.stdout.flush()

    with open(eval_log, "w") as log_f:
        proc = subprocess.Popen(
            eval_cmd,
            shell=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    start_time = time.time()
    last_output_size = 0
    last_output_time = start_time
    stall_extensions = 0
    service_dead_since = None
    grace_period = 60  # 前 60s 不检查服务（可能还在初始化）

    try:
        while True:
            time.sleep(check_interval)
            elapsed = time.time() - start_time

            # 1. 评测进程是否已退出
            ret = proc.poll()
            if ret is not None:
                break

            # 2. 最大超时
            if elapsed > max_timeout:
                print(f"[WRAPPER] 评测超时 ({max_timeout}s)，终止进程")
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(3)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                emit_error("timeout", f"评测超时 ({int(elapsed)}s > {max_timeout}s)", get_tail(eval_log))
                return 1

            # 3. 输出停滞检测
            try:
                current_size = os.path.getsize(eval_log)
            except OSError:
                current_size = last_output_size
            if current_size > last_output_size:
                last_output_size = current_size
                last_output_time = time.time()
                stall_extensions = 0
            else:
                stall_duration = time.time() - last_output_time
                if stall_duration > stall_timeout:
                    # 先检查服务是否仍在正常运行
                    if check_service_healthy(api_base):
                        stall_extensions += 1
                        print(f"[WRAPPER] 输出停滞 {int(stall_duration)}s，但服务仍正常响应，延长等待 600s（第 {stall_extensions} 次延长）")
                        last_output_time = time.time()
                    else:
                        print(f"[WRAPPER] 评测输出停滞 ({int(stall_duration)}s)，服务无响应，终止进程")
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        time.sleep(3)
                        if proc.poll() is None:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait()
                        emit_error("stall", f"评测输出停滞 {int(stall_duration)}s，服务无响应，进程已终止", get_tail(eval_log))
                        return 1

            # 4. 服务日志致命信号
            if service_log and elapsed > grace_period:
                log_offset, fatal = scan_log_fatal(service_log, log_offset)
                if fatal:
                    print(f"[WRAPPER] 服务崩溃检测: {fatal['type']}")
                    # 等待评测进程自行退出（它内部也有 ServiceMonitor）
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=5)
                    emit_error("service_crash", f"服务崩溃 ({fatal['type']}): {fatal['line']}", get_tail(eval_log))
                    return 1

            # 5. 服务进程存活检测
            if elapsed > grace_period:
                if not check_service_alive():
                    if service_dead_since is None:
                        service_dead_since = time.time()
                    elif time.time() - service_dead_since > 20:
                        print("[WRAPPER] 服务进程已退出超过 20s")
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            proc.wait(timeout=5)
                        emit_error("service_exited", "服务进程已退出，评测无法继续", get_tail(eval_log))
                        return 1
                else:
                    service_dead_since = None

    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        emit_error("interrupted", "用户中断", "")
        return 1

    # 评测进程已退出
    ret = proc.returncode
    eval_output = get_tail(eval_log, lines=50)

    if ret != 0:
        print(f"[WRAPPER] 评测进程退出码: {ret}")
        emit_error("eval_failed", f"fast_gpqa.py 退出码 {ret}", eval_output)
        return 1

    # 成功：输出结果
    if output_file and os.path.isfile(output_file):
        try:
            with open(output_file) as f:
                result = json.load(f)
            print("\n" + "=" * 60)
            print("  评测完成")
            print("=" * 60)
            score = result.get("score")
            model = result.get("model", "unknown")
            duration = result.get("total_duration_seconds", 0)
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"  模型: {model}")
            print(f"  得分: {score}%")
            print(f"  耗时: {minutes}m {seconds}s")
            print(f"  报告: {output_file}")
            print("=" * 60)
            # 最后一行输出 JSON 供 Claude 解析
            print(f"\n[RESULT_JSON] {json.dumps(result, ensure_ascii=False)}")
            return 0
        except Exception as e:
            emit_error("result_parse", f"结果文件解析失败: {e}", eval_output)
            return 1
    else:
        # 没有 --output 参数或文件不存在，从 stdout 中提取
        if "GPQA Diamond 快速评测结果" in eval_output:
            print(eval_output)
            return 0
        else:
            emit_error("no_result", "评测完成但未找到结果文件", eval_output)
            return 1


def emit_error(error_type: str, message: str, context: str):
    """输出结构化错误"""
    print(f"\n[EVAL_ERROR] type={error_type}")
    print(f"[EVAL_ERROR] message={message}")
    if context:
        print(f"[EVAL_ERROR] last_output:")
        for line in context.strip().splitlines()[-20:]:
            print(f"  | {line}")
    error_json = json.dumps({
        "error": True,
        "type": error_type,
        "message": message,
    }, ensure_ascii=False)
    print(f"\n[RESULT_JSON] {error_json}")


def get_tail(filepath: str, lines: int = 30) -> str:
    """获取文件最后 N 行"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main() or 0)
