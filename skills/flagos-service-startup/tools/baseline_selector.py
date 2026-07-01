#!/usr/bin/env python3
"""V1 基线三选状态机 — 分支 B (gems+tree+plugin) 专用

分支 B 的准入镜像同时预装 flaggems + plugin，V1 基础版需要确定"在不开启 flaggems 算子
替换的前提下，模型靠什么依赖跑起来"。按优先级依次尝试并冒烟验证：

  V1.1  VLLM_PLUGINS=''          纯净基线，不依赖任何 plugin
  V1.2  VLLM_PLUGINS=<厂商插件>   依赖厂商 platform plugin（如 metax），仍不开 flaggems
  V1.3  VLLM_PLUGINS='fl' 但 USE_FLAGGEMS=0   依赖 fl plugin 注册的 platform，但不启用算子替换
  none  三者均失败                无独立 V1（强依赖 flaggems），精度基线回退 NV

每种变体：设置环境 → 启动服务 → 冒烟测例（"中国的首都是哪里"）→ 通过则定为 V1 变体。

本脚本是**确定性判定**：输出唯一的 v1_variant，编排层据此直接选择 V2 分支（2.1/2.2），
无需 Claude 判断。

用法:
    python3 baseline_selector.py \\
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh" \\
        --vendor-plugin metax \\
        --output /flagos-workspace/results/v1_baseline_selection.json \\
        --json

退出码: 0=选出某个 V1 变体, 2=无 V1（强依赖，需 NV 兜底）, 1=脚本错误
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Any


DEFAULT_WAIT_SCRIPT = "/flagos-workspace/scripts/wait_for_service.sh"
DEFAULT_LOG_DIR = "/flagos-workspace/logs"
SMOKE_PROMPT = "中国的首都是哪里？"
# 冒烟判定：回答中包含以下任一关键词即视为模型语义正常
SMOKE_KEYWORDS = ["北京", "Beijing", "beijing"]

# 退出码
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_V1 = 2


def run_cmd(cmd: str, timeout: int = 300) -> tuple:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"超时 ({timeout}s)"
    except Exception as e:
        return -1, "", str(e)


def stop_service():
    """停止残留服务，释放 GPU"""
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)


def clear_caches():
    for d in ["/root/.triton/cache/", "/tmp/triton_cache/", "/root/.flaggems/code_cache/"]:
        if os.path.exists(d):
            subprocess.run(f"rm -rf {d}", shell=True, capture_output=True)


def start_variant(service_cmd: str, vllm_plugins: str, use_flaggems: bool,
                  wait_script: str, port: int, model_name: str,
                  log_path: str, max_timeout: int) -> bool:
    """按指定 VLLM_PLUGINS / USE_FLAGGEMS 启动服务并等待就绪。"""
    stop_service()
    clear_caches()

    # 组装启动命令：显式传 --vllm-plugins（含空串），USE_FLAGGEMS 通过 mode 控制
    mode = "flagos" if use_flaggems else "native"
    # 用 shell 引号安全传递空串
    cmd = f"{service_cmd} --mode {mode} --vllm-plugins '{vllm_plugins}'"
    env_prefix = f"USE_FLAGGEMS={'1' if use_flaggems else '0'} "

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    subprocess.Popen(
        env_prefix + cmd, shell=True,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT
    )

    wait_cmd = (
        f"{wait_script} --port {port} --timeout 300 --max-timeout {max_timeout}"
        f" --log-path {log_path} --mode {mode}"
    )
    if model_name:
        wait_cmd += f" --model-name '{model_name}'"
    rc, _, _ = run_cmd(wait_cmd, timeout=max_timeout + 60)
    return rc == 0


def smoke_test(port: int, model_name: str) -> tuple:
    """冒烟测例：问"中国的首都"，检查回答含关键词。返回 (passed, answer)。"""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name or "default",
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        passed = any(kw in answer for kw in SMOKE_KEYWORDS)
        return passed, answer.strip()[:200]
    except Exception as e:
        return False, f"请求失败: {e}"


def try_variant(variant: str, vllm_plugins: str, use_flaggems: bool,
                service_cmd: str, wait_script: str, port: int, model_name: str,
                max_timeout: int) -> Dict[str, Any]:
    """尝试一个 V1 变体：启动 + 冒烟。返回 attempt 记录。"""
    log_path = os.path.join(DEFAULT_LOG_DIR, f"startup_{variant}.log")
    print(f"\n{'=' * 56}")
    print(f"  尝试 {variant}: VLLM_PLUGINS='{vllm_plugins}' USE_FLAGGEMS={int(use_flaggems)}")
    print(f"{'=' * 56}")

    attempt = {
        "variant": variant,
        "vllm_plugins": vllm_plugins,
        "use_flaggems": use_flaggems,
        "service_ok": False,
        "smoke_passed": False,
        "smoke_answer": "",
        "reason": "",
    }

    service_ok = start_variant(service_cmd, vllm_plugins, use_flaggems,
                               wait_script, port, model_name, log_path, max_timeout)
    attempt["service_ok"] = service_ok
    if not service_ok:
        attempt["reason"] = "服务启动失败"
        print(f"  ✗ 服务启动失败")
        return attempt

    print(f"  ✓ 服务已就绪，运行冒烟测例...")
    passed, answer = smoke_test(port, model_name)
    attempt["smoke_passed"] = passed
    attempt["smoke_answer"] = answer
    if passed:
        attempt["reason"] = "冒烟通过"
        print(f"  ✓ 冒烟通过：{answer[:80]}")
    else:
        attempt["reason"] = "冒烟未通过（回答不含预期关键词）"
        print(f"  ✗ 冒烟未通过：{answer[:80]}")
    return attempt


def select_v1(service_cmd: str, vendor_plugin: str, wait_script: str,
              port: int, model_name: str, max_timeout: int) -> Dict[str, Any]:
    """按 V1.1 → V1.2 → V1.3 优先级依次尝试，返回选择结果。"""
    # 构建候选列表（vendor_plugin 为空则跳过 V1.2）
    candidates = [
        ("v1.1", "", False),          # 纯净，无 plugin，不开 flaggems
    ]
    if vendor_plugin:
        candidates.append(("v1.2", vendor_plugin, False))  # 厂商插件，不开 flaggems
    candidates.append(("v1.3", "fl", False))  # fl plugin 注册 platform，但 USE_FLAGGEMS=0

    attempts: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None

    for variant, plugins, use_gems in candidates:
        attempt = try_variant(variant, plugins, use_gems, service_cmd,
                              wait_script, port, model_name, max_timeout)
        attempts.append(attempt)
        if attempt["smoke_passed"]:
            selected = attempt
            break

    stop_service()

    if selected:
        result = {
            "v1_variant": selected["variant"],
            "vllm_plugins": selected["vllm_plugins"],
            "vendor_plugin": vendor_plugin if selected["variant"] == "v1.2" else "",
            "v1_available": True,
            "smoke_passed": True,
            "nv_baseline_used": False,
            "attempts": attempts,
            "message": f"选定 V1 变体: {selected['variant']} (VLLM_PLUGINS='{selected['vllm_plugins']}')",
        }
    else:
        # 三选均失败 → 无独立 V1，强依赖 flaggems，精度基线回退 NV
        result = {
            "v1_variant": "none",
            "vllm_plugins": "",
            "vendor_plugin": "",
            "v1_available": False,
            "smoke_passed": False,
            "nv_baseline_used": True,
            "attempts": attempts,
            "message": "三选均失败 → 无独立 V1（强依赖 flaggems），精度基线回退 NV",
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="V1 基线三选状态机（分支 B）")
    parser.add_argument("--service-startup-cmd", required=True,
                        help="服务启动命令（不含 --mode/--vllm-plugins，本脚本自动追加）")
    parser.add_argument("--vendor-plugin", default="",
                        help="厂商 platform plugin 名（如 metax），为空则跳过 V1.2")
    parser.add_argument("--wait-script", default=DEFAULT_WAIT_SCRIPT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-timeout", type=int, default=1800)
    parser.add_argument("--output", help="结果 JSON 输出路径")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = select_v1(
        service_cmd=args.service_startup_cmd,
        vendor_plugin=args.vendor_plugin,
        wait_script=args.wait_script,
        port=args.port,
        model_name=args.model_name,
        max_timeout=args.max_timeout,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'#' * 56}")
        print(f"# {result['message']}")
        print(f"{'#' * 56}")

    # 供编排层解析的机器可读标记
    print(f"[V1_SELECTION]{json.dumps(result, ensure_ascii=False)}[/V1_SELECTION]")

    sys.exit(EXIT_OK if result["v1_available"] else EXIT_NO_V1)


if __name__ == "__main__":
    main()
