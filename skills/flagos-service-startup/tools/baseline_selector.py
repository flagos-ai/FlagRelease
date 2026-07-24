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
    subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)


def clear_caches():
    for d in ["/root/.triton/cache/", "/tmp/triton_cache/", "/root/.flaggems/code_cache/"]:
        if os.path.exists(d):
            subprocess.run(f"rm -rf {d}", shell=True, capture_output=True)


def detect_vendor_plugin() -> str:
    """从 vllm.platform_plugins 入口点自动推导厂商插件名（排除 fl）。

    确定性逻辑：枚举已注册的 platform plugin，去掉 fl 后剩下的即厂商插件；
    多个厂商插件时取字典序第一个（实际镜像中厂商 platform plugin 只有一个）。
    """
    names = []
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group_eps = eps.select(group="vllm.platform_plugins") if hasattr(eps, "select") \
            else eps.get("vllm.platform_plugins", [])
        names = [ep.name for ep in group_eps]
    except Exception:
        try:
            import pkg_resources
            names = [ep.name for ep in pkg_resources.iter_entry_points("vllm.platform_plugins")]
        except Exception:
            return ""
    vendors = sorted(n for n in names if n != "fl")
    return vendors[0] if vendors else ""


def _persist_plugin_state(result: Dict[str, Any]) -> Dict[str, Any]:
    """选定后确定性落盘（不靠编排层转记）：

    1. VLLM_PLUGINS=<选中值>（含空串）持久化到 /etc/environment
       → start_service.sh 后续启动（V2 等）未显式传参时继承，废除 auto-fl 覆盖
    2. v1.1/v1.2 场景清除 VLLM_FL_PREFER_ENABLED
       → V2.1 冷注入块的 plugin 门控放行 + 调优工具链走控制文件路径
    3. baseline.* 写入 context.yaml
    """
    persisted = {"vllm_plugins": False, "prefer_enabled_cleared": False, "context": False}

    try:
        from flagos_op_config import persist_env, clear_env
    except ImportError:
        # 宿主机/未部署共享模块时的最小实现（容器内 scripts/ 平铺目录可直接 import）
        ETC = "/etc/environment"

        def persist_env(key, value):
            lines = []
            if os.path.exists(ETC):
                with open(ETC) as f:
                    lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
            lines.append(f"{key}={value}\n")
            with open(ETC, "w") as f:
                f.writelines(lines)
            os.environ[key] = value

        def clear_env(key):
            if os.path.exists(ETC):
                with open(ETC) as f:
                    lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
                with open(ETC, "w") as f:
                    f.writelines(lines)
            os.environ.pop(key, None)

    try:
        if result["v1_variant"] == "none":
            # 三选均失败（强依赖 flaggems）：不持久化空 VLLM_PLUGINS，
            # 保留 start_service.sh 的 auto-fl 兜底（V2 大概率需要 fl+flaggems 才能起）
            print("  - V1=none，跳过 VLLM_PLUGINS 持久化（保留 auto-fl 兜底）")
        else:
            persist_env("VLLM_PLUGINS", result["vllm_plugins"])
            persisted["vllm_plugins"] = True
            print(f"  ✓ VLLM_PLUGINS='{result['vllm_plugins']}' 已持久化到 /etc/environment")
        if result["v1_variant"] in ("v1.1", "v1.2"):
            clear_env("VLLM_FL_PREFER_ENABLED")
            persisted["prefer_enabled_cleared"] = True
            print("  ✓ VLLM_FL_PREFER_ENABLED 已清除（V2.1 代码注入路径生效前提）")
    except Exception as e:
        print(f"  WARN: plugin 状态持久化失败: {e}")

    # context.yaml 落盘（update_context.py 与本脚本在容器内同目录）
    update_ctx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_context.py")
    if os.path.isfile(update_ctx):
        rc, _, err = run_cmd(
            f"{sys.executable} {update_ctx}"
            f" --set 'baseline.v1_variant={result['v1_variant']}'"
            f" --set 'baseline.vllm_plugins={result['vllm_plugins']}'"
            f" --set 'baseline.vendor_plugin={result['vendor_plugin']}'"
            f" --set 'baseline.v1_available={str(result['v1_available']).lower()}'",
            timeout=60,
        )
        persisted["context"] = rc == 0
        if rc != 0:
            print(f"  WARN: context 写入失败: {err.strip()[:200]}")
    else:
        print(f"  WARN: 未找到 update_context.py（{update_ctx}），跳过 context 写入")

    return persisted


def start_variant(service_cmd: str, vllm_plugins: str, use_flaggems: bool,
                  wait_script: str, port: int, model_name: str,
                  log_path: str, max_timeout: int) -> bool:
    """按指定 VLLM_PLUGINS / USE_FLAGGEMS 启动服务并等待就绪。"""
    stop_service()
    clear_caches()

    # 清除上一个 variant 的端口回写文件，避免本次启动尚未写入时读到残留端口，
    # 导致 wait/冒烟连到旧端口。start_service.sh 启动后会重新写入实际端口。
    try:
        os.remove(os.path.join(DEFAULT_LOG_DIR, "service_port"))
    except OSError:
        pass

    # 组装启动命令：显式传 --vllm-plugins（含空串），USE_FLAGGEMS 通过 mode 控制
    mode = "flagos" if use_flaggems else "native"
    # --log-file 让 start_service.sh 把 vLLM 服务日志写入本 variant 的独立文件，
    # 与下方 wait_for_service --log-path 监控同一文件（否则 start_service.sh 默认
    # 写 startup_${mode}.log，三个 variant 互相覆盖且监控端抓不到真实日志 → 恒判失败）。
    # 用 shell 引号安全传递空串
    cmd = (f"{service_cmd} --mode {mode} --vllm-plugins '{vllm_plugins}'"
           f" --log-file '{log_path}'")
    env_prefix = f"USE_FLAGGEMS={'1' if use_flaggems else '0'} "

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # 服务日志由 start_service.sh 经 --log-file 写入 log_path；此处 start_service.sh
    # 前台部分的少量 echo 无需保留，丢弃以免与其内部 nohup 重定向混淆。
    subprocess.Popen(
        env_prefix + cmd, shell=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )

    # start_service.sh 在后台启动、走完端口探测后才写 service_port（端口可能被自动
    # 递增）。短轮询等待该文件出现，拿到服务实际端口后同时用于 wait 与冒烟，确保三者
    # 端口严格一致，不会连到占用原端口的其他服务。文件迟迟不出现则回退请求端口。
    actual_port = port
    for _ in range(30):  # 最多等 ~15s
        resolved = resolve_service_port(port)
        if os.path.exists(os.path.join(DEFAULT_LOG_DIR, "service_port")):
            actual_port = resolved
            break
        time.sleep(0.5)

    # --from-start：本 variant 是全新启动，start_service.sh 的 nohup 以 truncate 方式
    # 重写 log_path，wait 必须从 offset 0 读起，才能捕获本次启动的进度信号
    # （loading_weights/service_ready 等）。否则沿用文件残留大小作 offset → 读不到
    # 进度信号 → 端口虽已响应仍被误判为"残留服务"(stale_service) → start_variant 误返回失败。
    wait_cmd = (
        f"{wait_script} --port {actual_port} --timeout 300 --max-timeout {max_timeout}"
        f" --log-path {log_path} --mode {mode} --from-start"
    )
    if model_name:
        wait_cmd += f" --model-name '{model_name}'"
    rc, _, _ = run_cmd(wait_cmd, timeout=max_timeout + 60)
    return rc == 0


def resolve_service_port(default_port: int) -> int:
    """读取服务实际监听端口。

    start_service.sh 的端口来自 context.yaml 且**会因端口占用自动递增**，最终端口
    回写到 logs/service_port。冒烟/查找必须用这个实际端口，不能假设 --port（默认
    8000），否则可能连不上（误判失败）或连到占用同端口的其他服务（误判成功/答非所问）。
    读不到文件时回退到传入的 default_port，保证不比原逻辑差。
    """
    port_file = os.path.join(DEFAULT_LOG_DIR, "service_port")
    try:
        with open(port_file, "r", encoding="utf-8") as f:
            actual = int(f.read().strip())
        if actual != default_port:
            print(f"  [port] 服务实际端口 {actual}（≠ 请求端口 {default_port}），冒烟改用实际端口")
        return actual
    except (OSError, ValueError):
        return default_port


def resolve_served_model_id(port: int, model_name: str) -> str:
    """动态解析 vLLM 实际注册的模型 id。

    vLLM 以 served_model_name 注册模型（start_service.sh 用 name.split('/')[-1]
    去掉了 org 前缀，如 upstage/）。冒烟请求若用带前缀的全名会命中不存在的 model
    触发 404，被误判为冒烟/启动失败。这里先查 /v1/models 取服务实际注册的 id：
      1. 若返回的 id 列表里能匹配到传入名（全名或去前缀短名），用匹配到的那个；
      2. 单模型服务则直接用列表里唯一的 id；
      3. 查询失败时回退到静态去前缀名，保证不比原逻辑更差。
    """
    fallback = (model_name or "default").split("/")[-1]
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/models", method="GET"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        served_ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return fallback

    if not served_ids:
        return fallback
    # 传入名（全名或去前缀）能精确命中已注册 id 则优先用之
    for candidate in (model_name, fallback):
        if candidate and candidate in served_ids:
            return candidate
    # 单模型服务：直接用唯一注册 id
    if len(served_ids) == 1:
        return served_ids[0]
    return fallback


def smoke_test(port: int, model_name: str) -> tuple:
    """冒烟测例：问"中国的首都"，检查回答含关键词。返回 (passed, answer)。"""
    url = f"http://localhost:{port}/v1/chat/completions"
    served_name = resolve_served_model_id(port, model_name)
    payload = {
        "model": served_name,
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
    # 服务已就绪，service_port 必已写入 → 用实际监听端口冒烟，与启动/wait 端口一致
    smoke_port = resolve_service_port(port)
    passed, answer = smoke_test(smoke_port, model_name)
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
    parser.add_argument("--vendor-plugin", default="auto",
                        help="厂商 platform plugin 名（如 metax）。默认 auto=从 "
                             "vllm.platform_plugins 入口点自动推导（排除 fl）；"
                             "显式传空串则跳过 V1.2")
    parser.add_argument("--wait-script", default=DEFAULT_WAIT_SCRIPT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-timeout", type=int, default=1800)
    parser.add_argument("--output", help="结果 JSON 输出路径")
    parser.add_argument("--no-persist", action="store_true",
                        help="跳过 VLLM_PLUGINS 持久化与 context 写入（调试用）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    vendor_plugin = args.vendor_plugin
    if vendor_plugin == "auto":
        vendor_plugin = detect_vendor_plugin()
        print(f"[baseline_selector] 厂商插件自动推导: '{vendor_plugin or '(未发现，跳过 V1.2)'}'")

    result = select_v1(
        service_cmd=args.service_startup_cmd,
        vendor_plugin=vendor_plugin,
        wait_script=args.wait_script,
        port=args.port,
        model_name=args.model_name,
        max_timeout=args.max_timeout,
    )

    # 选定后确定性落盘：VLLM_PLUGINS 持久化 + v1.1/v1.2 清 PREFER_ENABLED + context 写入
    if not args.no_persist:
        result["persisted"] = _persist_plugin_state(result)

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
