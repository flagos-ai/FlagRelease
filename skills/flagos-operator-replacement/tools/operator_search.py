#!/usr/bin/env python3
"""
算子搜索编排脚本 — 自动化完整搜索循环

将 算子优化器(next) → toggle FlagGems → 重启服务 → benchmark → 更新结果(update) 的完整循环
封装为一次脚本调用，避免 Claude Code 在搜索循环中消耗思考 token。

此脚本在**容器内**运行，直接调用各工具脚本。

Usage:
    # 运行完整搜索循环（直到搜索完成或达到最大轮次）
    python operator_search.py run \
        --state-path /flagos-workspace/results/operator_config.json \
        --perf-config /flagos-workspace/scripts/config/perf_config.yaml \
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh" \
        --max-rounds 20

    # 只运行一轮搜索
    python operator_search.py step \
        --state-path /flagos-workspace/results/operator_config.json \
        --perf-config /flagos-workspace/scripts/config/perf_config.yaml \
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh"

    # 查看当前状态
    python operator_search.py status --state-path /flagos-workspace/results/operator_config.json
"""

import sys

# IO 缓冲修复
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    import functools
    print = functools.partial(print, flush=True)

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 共享模块导入（容器内所有脚本在同一 scripts/ 目录）
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from error_writer import write_last_error, write_checkpoint
except ImportError:
    def write_last_error(*a, **kw): pass
    def write_checkpoint(*a, **kw): pass

# =============================================================================
# 配置
# =============================================================================

DEFAULT_STATE_PATH = "/flagos-workspace/results/operator_config.json"
DEFAULT_PERF_CONFIG = "/flagos-workspace/scripts/config/perf_config.yaml"
DEFAULT_TOGGLE_SCRIPT = "/flagos-workspace/scripts/toggle_flaggems.py"
DEFAULT_BENCHMARK_SCRIPT = "/flagos-workspace/scripts/benchmark_runner.py"
DEFAULT_OPTIMIZER_SCRIPT = "/flagos-workspace/scripts/operator_optimizer.py"
DEFAULT_WAIT_SCRIPT = "/flagos-workspace/scripts/wait_for_service.sh"
DEFAULT_APPLY_CONFIG_SCRIPT = "/flagos-workspace/scripts/apply_op_config.py"

SERVICE_STOP_CMD = "pkill -f 'vllm.entrypoints|sglang.launch_server'"
SERVICE_WAIT_TIMEOUT = 300  # 秒
GPU_MEM_FREE_THRESHOLD = 0.95  # GPU 显存空闲比例阈值（>95% 视为已释放）
GPU_RELEASE_TIMEOUT = 60       # GPU 显存释放等待超时（秒）
GPU_RELEASE_POLL_INTERVAL = 2  # 轮询间隔（秒）


# =============================================================================
# GPU 资源管理
# =============================================================================

def _parse_gpu_memory() -> List[Dict[str, float]]:
    """解析 GPU 显存信息，优先通过 detect_gpu.py 统一接口"""
    # 优先使用 detect_gpu.py 统一接口
    try:
        from detect_gpu import check_gpu_free
        vendor = None
        try:
            import yaml
            with open("/flagos-workspace/shared/context.yaml") as f:
                vendor = yaml.safe_load(f).get("gpu", {}).get("vendor")
        except Exception:
            pass
        info = check_gpu_free(vendor=vendor)
        if info and info.get("details"):
            return [{
                "index": d["index"],
                "used_mib": d["used_mib"],
                "total_mib": d["total_mib"],
                "free_ratio": 1.0 - d["used_mib"] / d["total_mib"] if d["total_mib"] > 0 else 0,
            } for d in info["details"]]
    except ImportError:
        pass
    # fallback: detect_gpu.py 不可用时使用本地实现
    return _parse_gpu_memory_fallback()


def _parse_gpu_memory_fallback() -> List[Dict[str, float]]:
    """fallback: 直接调用各厂商 SMI 工具"""
    smi_commands = [
        ("nvidia-smi", "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"),
        ("mx-smi", "mx-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"),
        ("npu-smi", None),
    ]

    # 从 context.yaml 读取厂商（替代未设置的 GPU_VENDOR 环境变量）
    gpu_vendor = ""
    try:
        import yaml
        with open("/flagos-workspace/shared/context.yaml") as f:
            gpu_vendor = yaml.safe_load(f).get("gpu", {}).get("vendor", "")
    except Exception:
        pass

    if gpu_vendor == 'metax':
        smi_commands = [smi_commands[1], smi_commands[0]]
    elif gpu_vendor == 'nvidia':
        smi_commands = [smi_commands[0]]
    elif gpu_vendor == 'huawei':
        smi_commands = [smi_commands[2]]

    for name, cmd in smi_commands:
        try:
            if name == "npu-smi":
                gpus = _parse_npu_smi()
                if gpus:
                    return gpus
                continue
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                continue
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    try:
                        gpus.append({
                            "index": int(parts[0]),
                            "used_mib": float(parts[1]),
                            "total_mib": float(parts[2]),
                            "free_ratio": 1.0 - float(parts[1]) / float(parts[2]) if float(parts[2]) > 0 else 0,
                        })
                    except (ValueError, ZeroDivisionError):
                        continue
            if gpus:
                return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception:
            continue
    return []


def _parse_npu_smi() -> List[Dict[str, float]]:
    """解析华为 npu-smi info 输出的显存信息"""
    try:
        result = subprocess.run(
            "npu-smi info", shell=True, capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        import re
        gpus = []
        for match in re.finditer(r'(\d+)\s+\d+\s+\w+\s+\w+\s+(\d+)\s*/\s*(\d+)', result.stdout):
            idx, used, total = int(match.group(1)), float(match.group(2)), float(match.group(3))
            gpus.append({
                "index": idx,
                "used_mib": used,
                "total_mib": total,
                "free_ratio": 1.0 - used / total if total > 0 else 0,
            })
        return gpus
    except Exception:
        return []


def wait_gpu_memory_release(timeout: int = GPU_RELEASE_TIMEOUT,
                            threshold: float = GPU_MEM_FREE_THRESHOLD,
                            required_free: int = 1) -> bool:
    """
    等待 GPU 显存释放。pkill 后进程退出需要时间释放显存。
    返回 True 表示至少有 required_free 张 GPU 显存已释放，False 表示超时。

    Args:
        timeout: 最大等待时间（秒）
        threshold: 空闲比例阈值（默认 0.95）
        required_free: 需要的空闲 GPU 数量（默认 1）
    """
    print(f"  等待 GPU 显存释放 (最多 {timeout}s, 需要 {required_free} 张)...")
    start = time.time()
    while (time.time() - start) < timeout:
        gpus = _parse_gpu_memory()
        if not gpus:
            print("  WARNING: 无法读取 GPU 信息，跳过显存检查")
            return True
        free_gpus = [g for g in gpus if g["free_ratio"] >= threshold]
        if len(free_gpus) >= required_free:
            print(f"  ✓ {len(free_gpus)} 张 GPU 显存已释放 (需要 {required_free} 张) "
                  f"(如 GPU {free_gpus[0]['index']}: {free_gpus[0]['used_mib']:.0f}/{free_gpus[0]['total_mib']:.0f} MiB)")
            return True
        time.sleep(GPU_RELEASE_POLL_INTERVAL)
    # 超时：打印当前状态
    gpus = _parse_gpu_memory()
    for g in gpus:
        print(f"  GPU {g['index']}: {g['used_mib']:.0f}/{g['total_mib']:.0f} MiB ({g['free_ratio']*100:.1f}% free)")
    return False


def check_gpu_availability(required_gpus: int = 1,
                           threshold: float = GPU_MEM_FREE_THRESHOLD) -> Dict[str, Any]:
    """
    检查是否有足够的空闲 GPU。
    返回 {"available": bool, "free_gpus": [...], "message": str}
    """
    gpus = _parse_gpu_memory()
    if not gpus:
        return {"available": True, "free_gpus": [], "message": "无法读取 GPU 信息，假设可用"}
    free_gpus = [g["index"] for g in gpus if g["free_ratio"] >= threshold]
    available = len(free_gpus) >= required_gpus
    return {
        "available": available,
        "free_gpus": free_gpus,
        "total_gpus": len(gpus),
        "message": f"{len(free_gpus)}/{len(gpus)} GPU 空闲" + ("" if available else f"，需要 {required_gpus} 张"),
    }


# =============================================================================
# 工具函数
# =============================================================================

def run_cmd(cmd: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
    """执行命令并实时输出，超时后强制终止"""
    import threading
    print(f"  $ {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = []
    timed_out = [False]

    def _kill_on_timeout():
        timed_out[0] = True
        print(f"  TIMEOUT: 命令超过 {timeout}s，强制终止")
        proc.kill()

    timer = threading.Timer(timeout, _kill_on_timeout)
    timer.start()
    try:
        for line in proc.stdout:
            output_lines.append(line)
            print(f"    | {line.rstrip()}")
        proc.wait()
    finally:
        timer.cancel()
    returncode = -9 if timed_out[0] else proc.returncode
    result = subprocess.CompletedProcess(
        cmd, returncode,
        stdout="".join(output_lines),
        stderr=""
    )
    if check and returncode != 0:
        print(f"  WARN: 命令返回码 {returncode}")
    return result


def load_json(path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str):
    """保存 JSON 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# 搜索步骤
# =============================================================================

def get_next_action(state_path: str, optimizer_script: str) -> Dict[str, Any]:
    """调用 operator_optimizer.py next 获取下一步操作"""
    result = run_cmd(
        f"python {optimizer_script} next --state-path {state_path}",
        check=False
    )
    try:
        # 从输出中提取 JSON
        output = result.stdout.strip()
        # 找到第一个 { 和最后一个 }
        start = output.index('{')
        end = output.rindex('}') + 1
        return json.loads(output[start:end])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: 解析 next 输出失败: {e}")
        return {"action": "error", "message": str(e)}


def apply_operator_config(action: Dict[str, Any],
                          apply_config_script: str = DEFAULT_APPLY_CONFIG_SCRIPT,
                          plugin_mode: bool = False,
                          capabilities: Optional[List[str]] = None,
                          gems_txt_path: Optional[str] = None,
                          all_ops: Optional[List[str]] = None,
                          registered_ops: Optional[List[str]] = None) -> Any:
    """
    应用算子配置。

    Plugin 场景：从 action 的 env_vars 构建内联环境变量字符串，返回 env_inline 字符串
    非 plugin 场景：通过 Layer 1-4 策略控制，返回 True/False
    all_ops: 全量算子列表，用于 Layer 1/3 黑名单模式补全 unsearched 算子
    registered_ops: FlagGems 完整注册算子列表，用于黑名单模式确保覆盖所有算子
    """
    if plugin_mode:
        return _apply_plugin_config(action, apply_config_script)
    else:
        return _apply_non_plugin_config(action, capabilities, gems_txt_path,
                                        all_ops=all_ops, registered_ops=registered_ops)


def _apply_plugin_config(action: Dict[str, Any],
                         apply_config_script: str) -> Optional[str]:
    """Plugin 场景：从 action 的 env_vars 构建内联环境变量字符串"""
    env_vars = action.get("env_vars", {})

    if not env_vars:
        print("  WARN: action 无 env_vars 信息，使用 full 模式默认值")
        env_vars = {"USE_FLAGGEMS": "1", "VLLM_FL_PREFER_ENABLED": "true"}

    # 构建内联字符串
    env_inline = action.get("env_inline", "")
    if not env_inline:
        parts = []
        for k, v in env_vars.items():
            if " " in v or "'" in v:
                parts.append(f"{k}='{v}'")
            else:
                parts.append(f"{k}={v}")
        env_inline = " ".join(parts)

    print(f"  [Plugin] 内联环境变量: {env_inline}")
    return env_inline


def _detect_flaggems_capabilities() -> List[str]:
    """自动探测当前环境的 FlagGems capabilities"""
    caps = []
    try:
        import flag_gems
        # Layer 1: yaml_config — 检查 vendor 配置目录是否存在
        gems_path = os.path.dirname(flag_gems.__file__)
        for root, dirs, files in os.walk(gems_path):
            if "runtime" in root and "backend" in root:
                caps.append("yaml_config")
                break
        # Layer 2: only_enable
        if hasattr(flag_gems, "only_enable"):
            caps.append("only_enable")
        # Layer 3: enable_unused
        if hasattr(flag_gems, "enable"):
            import inspect as insp_mod
            sig = insp_mod.signature(flag_gems.enable)
            if "unused" in list(sig.parameters.keys()):
                caps.append("enable_unused")
    except ImportError:
        pass
    return caps


OPS_CONTROL_FILE = "/root/flaggems_ops_control.json"
FLAGGEMS_INJECT_MARKER = "FLAGGEMS_CONTROL_MODE"
ETC_ENVIRONMENT = "/etc/environment"


def _persist_control_mode(control_mode: str):
    """持久化 FLAGGEMS_CONTROL_MODE + USE_FLAGGEMS 到 /etc/environment

    _apply_via_control_file() 只在 FlagGems 启用路径调用，所以 USE_FLAGGEMS=1 始终正确。
    同时写入避免中断恢复场景下 /etc/environment 残留 USE_FLAGGEMS=0。
    """
    try:
        existing = ""
        if os.path.isfile(ETC_ENVIRONMENT):
            with open(ETC_ENVIRONMENT, 'r', encoding='utf-8') as f:
                existing = f.read()
        lines = [l for l in existing.split('\n')
                 if not l.startswith("FLAGGEMS_CONTROL_MODE=") and not l.startswith("USE_FLAGGEMS=")]
        lines.append(f"USE_FLAGGEMS=1")
        lines.append(f"FLAGGEMS_CONTROL_MODE={control_mode}")
        with open(ETC_ENVIRONMENT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l for l in lines if l is not None) + '\n')
    except Exception as e:
        print(f"  WARN: 持久化环境变量失败: {e}")


def _normalize_ops_for_control_file(ops: List[str]) -> List[str]:
    """将算子名从大写显示名转换为小写函数名（供控制文件使用）"""
    if not ops:
        return ops
    if all(op == op.lower() and ' ' not in op for op in ops):
        return ops
    try:
        from toggle_flaggems import normalize_ops_to_func_names
        return normalize_ops_to_func_names(ops)
    except ImportError:
        import re
        result = []
        for op in ops:
            s = re.sub(r'\s*\(.*?\)', '', op)
            s = s.split(',')[0].strip()
            s = re.sub(r'-hopper$', '', s, flags=re.IGNORECASE)
            s = s.replace('.STABLE', '_stable')
            s = re.sub(r'\s+FORWARD$', '', s, flags=re.IGNORECASE)
            s = re.sub(r'\s+BACKWARD$', '', s, flags=re.IGNORECASE)
            s = s.lower().replace(' ', '_')
            if s.startswith('_'):
                s = s[1:]
            result.append(s)
        return result


def _apply_via_control_file(test_enabled: List[str], test_disabled: List[str],
                             control_mode: str) -> bool:
    """已注入场景：写控制文件 + 设置环境变量，不改源码"""
    # 将大写显示名转换为小写函数名
    test_enabled = _normalize_ops_for_control_file(test_enabled)
    test_disabled = _normalize_ops_for_control_file(test_disabled)

    data = {}
    if control_mode == "only_enable":
        data["include"] = sorted(test_enabled)
        os.environ["FLAGGEMS_CONTROL_MODE"] = "only_enable"
    else:
        data["unused"] = sorted(test_disabled)
        os.environ["FLAGGEMS_CONTROL_MODE"] = "unused"

    # 持久化到 /etc/environment，确保 start_service.sh 启动的新进程能读到
    _persist_control_mode(control_mode)

    save_json(data, OPS_CONTROL_FILE)
    print(f"  [env_control] mode={control_mode}, 控制文件: {OPS_CONTROL_FILE}")
    if control_mode == "only_enable":
        print(f"    include: {len(test_enabled)} 个算子")
    else:
        print(f"    unused: {len(test_disabled)} 个算子")
    return True


def _is_code_injected() -> bool:
    """检查源码是否已注入环境变量驱动代码"""
    try:
        from toggle_flaggems import find_model_runner_files
        files = find_model_runner_files()
        for f in files:
            try:
                content = Path(f).read_text(encoding="utf-8", errors="ignore")
                if FLAGGEMS_INJECT_MARKER in content:
                    return True
            except Exception:
                continue
    except ImportError:
        pass
    return False


def _apply_non_plugin_config(action: Dict[str, Any],
                              capabilities: Optional[List[str]],
                              gems_txt_path: Optional[str],
                              all_ops: Optional[List[str]] = None,
                              registered_ops: Optional[List[str]] = None) -> bool:
    """非 plugin 场景：优先使用环境变量驱动（已注入），否则 Layer 1-4 分层降级"""
    test_enabled = action.get("test_enabled_ops", [])
    test_disabled = action.get("test_disabled_ops", [])

    # 已注入环境变量驱动代码 → 只写控制文件
    if _is_code_injected():
        # 有禁用算子 → only_enable（白名单）；全开 → unused
        if test_disabled:
            return _apply_via_control_file(test_enabled, test_disabled, "only_enable")
        return _apply_via_control_file(test_enabled, test_disabled, "unused")

    # 未注入 → 原有 Layer 1-4 降级逻辑
    caps = _detect_flaggems_capabilities()
    if caps:
        print(f"  [auto-detect] FlagGems capabilities: {caps}")
    if capabilities and set(capabilities) != set(caps):
        print(f"  [auto-detect] 忽略外部传入 capabilities={capabilities}，以自动探测为准")

    base_ops = registered_ops or all_ops
    if base_ops:
        full_disabled = sorted(set(base_ops) - set(test_enabled))
    else:
        full_disabled = test_disabled

    # 读取环境变量决定控制模式
    # 规则：有禁用算子 → only_enable（白名单）；全开 → unused
    control_mode = os.environ.get("FLAGGEMS_CONTROL_MODE", "")
    if not control_mode:
        if test_disabled and "only_enable" in caps:
            control_mode = "only_enable"
        elif not test_disabled and "enable_unused" in caps:
            control_mode = "unused"
        elif "only_enable" in caps:
            control_mode = "only_enable"
        elif "enable_unused" in caps:
            control_mode = "unused"

    # only_enable 模式
    if control_mode == "only_enable" and "only_enable" in caps:
        if test_enabled:
            if "yaml_config" in caps:
                _apply_yaml_exclude(full_disabled)
            return _apply_only_enable(test_enabled)
        elif "enable_unused" in caps:
            return _apply_enable_unused(full_disabled)

    # unused 模式 / 降级链
    if "yaml_config" in caps:
        return _apply_yaml_exclude(full_disabled)
    elif "enable_unused" in caps:
        return _apply_enable_unused(full_disabled)
    else:
        return _apply_txt_fallback(test_enabled, gems_txt_path)


def _apply_yaml_exclude(disabled_ops: List[str]) -> bool:
    """Layer 1: YAML exclude 配置"""
    try:
        import flag_gems
        import os
        gems_path = os.path.dirname(flag_gems.__file__)
    except ImportError:
        print("  ERROR: flag_gems not installed")
        return False

    # 查找 vendor 配置目录
    config_dirs = []
    for root, dirs, files in os.walk(gems_path):
        if "runtime" in root and "backend" in root:
            config_dirs.append(root)

    if not config_dirs:
        print("  WARN: yaml_config 目录未找到，降级到 txt 兜底")
        return False

    config_dir = config_dirs[0]
    config_path = os.path.join(config_dir, "enable_configs.yaml")
    content = "exclude:\n"
    for op in sorted(disabled_ops):
        content += f"  - {op}\n"

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [Layer 1] YAML exclude 写入 {config_path}: {len(disabled_ops)} 个算子")
    return True


def _apply_only_enable(enabled_ops: List[str]) -> bool:
    """Layer 2: only_enable API — 记录到临时文件，由 Claude Code 修改启动入口"""
    ops_path = "/tmp/only_enable_ops.json"
    save_json(sorted(enabled_ops), ops_path)
    print(f"  [Layer 2] only_enable 算子列表保存到 {ops_path}: {len(enabled_ops)} 个算子")
    print(f"  注意: 需要修改启动入口调用 flag_gems.only_enable(include=[...])")
    return True


def _apply_enable_unused(disabled_ops: List[str]) -> bool:
    """Layer 3: enable(unused=) — 记录到临时文件，由 Claude Code 修改启动入口"""
    ops_path = "/tmp/unused_ops.json"
    save_json(sorted(disabled_ops), ops_path)
    print(f"  [Layer 3] enable_unused 算子列表保存到 {ops_path}: {len(disabled_ops)} 个算子")
    print(f"  注意: 需要修改启动入口调用 flag_gems.enable(unused=[...])")
    return True


def _apply_txt_fallback(enabled_ops: List[str], gems_txt_path: Optional[str]) -> bool:
    """Layer 4 兜底: 已废弃 — gems.txt 是 flag_gems.enable() 的输出记录文件，不是输入控制文件。
    写入的内容会在服务启动后被 FlagGems 覆盖，无法控制算子替换。
    正常流程应通过 Layer 1-3（yaml_config / only_enable / enable_unused）控制算子。
    """
    print("  ERROR: Layer 1-3 均不可用，无法控制算子替换。")
    print("  原因: gems.txt 是 flag_gems.enable() 的输出记录文件，写入后会被服务启动覆盖。")
    print("  请确认 FlagGems 已安装且支持 yaml_config / only_enable / enable_unused 之一。")
    return False


def restart_service(stop_cmd: str, startup_cmd: str,
                    wait_script: str, wait_timeout: int = SERVICE_WAIT_TIMEOUT,
                    env_inline: Optional[str] = None,
                    port: Optional[int] = None,
                    model_name: Optional[str] = None,
                    max_timeout: Optional[int] = None,
                    service_log_path: Optional[str] = None) -> bool:
    """重启服务：停止 → 启动 → 等待就绪"""
    print("\n[重启服务]")

    # 清除 Triton/FlagGems 编译缓存（约束39：避免旧缓存隐藏问题算子）
    print("  清除 Triton/FlagGems cache...")
    run_cmd("rm -rf ~/.triton/cache/ 2>/dev/null", check=False)
    run_cmd("rm -rf /tmp/triton_cache/ 2>/dev/null", check=False)
    run_cmd("rm -rf ~/.flaggems/code_cache/ 2>/dev/null", check=False)

    # 停止
    print("  停止服务...")
    run_cmd(stop_cmd, check=False)
    time.sleep(3)

    # 等待 GPU 显存释放（需要所有服务使用的 GPU 都释放）
    required_gpus = _read_gpu_count()
    if not wait_gpu_memory_release(required_free=required_gpus):
        print("  WARNING: GPU 显存未完全释放，尝试强制清理...")
        run_cmd("pkill -9 -f 'vllm\\|sglang' 2>/dev/null", check=False)
        time.sleep(5)
        if not wait_gpu_memory_release(timeout=15, required_free=required_gpus):
            print("  WARNING: 强制清理后 GPU 显存仍未释放，继续启动（可能使用其他空闲 GPU）")

    # 启动（后台执行，避免 vllm 等服务进程阻塞脚本）
    nohup_log = "/flagos-workspace/logs/startup_search.log"
    if env_inline:
        # env_inline 必须在 nohup 前面，否则 nohup 会把 VAR=val 当命令名
        bg_cmd = f"{env_inline} nohup {startup_cmd} > {nohup_log} 2>&1 &"
        print(f"  启动服务（内联 env vars，后台）...")
    else:
        bg_cmd = f"nohup {startup_cmd} > {nohup_log} 2>&1 &"
        print("  启动服务（后台）...")
    run_cmd(bg_cmd, check=False)

    # 确定 wait_for_service 应监控的日志路径：
    # start_service.sh 内部会将 vllm 输出重定向到 startup_<mode>.log，
    # 导致 nohup_log 只有 shell echo，无 vllm 启动进度。
    # 优先使用显式指定的 service_log_path，否则从 nohup_log 输出中自动探测。
    monitor_log = service_log_path
    if not monitor_log:
        time.sleep(1)
        try:
            with open(nohup_log, "r") as f:
                for line in f:
                    if "log=" in line:
                        monitor_log = line.split("log=")[-1].strip()
                        break
        except (FileNotFoundError, IOError):
            pass
    if not monitor_log:
        monitor_log = nohup_log
    if monitor_log != nohup_log:
        print(f"  监控实际服务日志: {monitor_log}")

    # 传递 --log-path 启用动态超时模式（监控日志活动而非固定超时）
    wait_cmd = f"bash {wait_script} --timeout {wait_timeout}"
    wait_cmd += f" --log-path {monitor_log}"
    if max_timeout:
        wait_cmd += f" --max-timeout {max_timeout}"
    else:
        wait_cmd += f" --max-timeout 1800"
    if port:
        wait_cmd += f" --port {port}"
    if model_name:
        wait_cmd += f" --model-name '{model_name}'"
    effective_max = max_timeout or 1800
    print(f"  等待服务就绪 (动态模式, 无活动超时={wait_timeout}s, 绝对上限={effective_max}s, port={port or 8000})...")
    result = run_cmd(
        wait_cmd,
        timeout=effective_max + 30,
        check=False
    )
    if result.returncode != 0:
        print("  ERROR: 服务启动失败")
        return False

    print("  服务就绪")
    return True


def verify_ops_via_txt() -> Optional[List[str]]:
    """重启后读取运行时 txt 文件验证算子变化"""
    try:
        from operator_optimizer import find_ops_list_file
        result = find_ops_list_file()
        if result.get("found"):
            ops = result["ops"]
            print(f"  [验证] 运行时 txt: {result['path']} ({len(ops)} 个算子)")
            return ops
        else:
            print(f"  [验证] 未找到运行时 txt 文件")
    except ImportError:
        print(f"  [验证] operator_optimizer 模块不可用")
    except Exception as e:
        print(f"  [验证] 读取 txt 失败: {e}")
    return None


def run_benchmark_quick(perf_config: str, benchmark_script: str,
                        output_name: str = "search_benchmark") -> Dict[str, Any]:
    """运行快速 benchmark（搜索阶段始终用 quick，只需快速判断算子影响）"""
    print("\n[运行 Benchmark] strategy=quick")

    output_dir = "/flagos-workspace/results"
    result = run_cmd(
        f"python {benchmark_script} "
        f"--config {perf_config} "
        f"--quick "
        f"--output-name {output_name} "
        f"--output-dir {output_dir} "
        f"--mode search",
        timeout=600,
        check=False
    )

    # 解析结果
    output_path = f"{output_dir}/{output_name}.json"
    try:
        data = load_json(output_path)

        # 兼容新旧格式：旧格式有 results 包装，新格式直接是扁平结构
        results = data.get("results", data) if isinstance(data, dict) else data

        # 提取吞吐量：每个 test_case × 每个 concurrency 的 output + total 双指标
        throughputs = {}
        for tc_name, tc_results in results.items():
            if not isinstance(tc_results, dict):
                continue
            for key, metrics in tc_results.items():
                if key.startswith("_") or not isinstance(metrics, dict) or "error" in metrics:
                    continue
                output_tp = metrics.get('Output token throughput (tok/s)', 0) or 0
                total_tp = metrics.get('Total token throughput (tok/s)', 0) or 0
                if output_tp > 0 or total_tp > 0:
                    throughputs[f"{tc_name}|{key}"] = {
                        "output": output_tp,
                        "total": total_tp,
                    }

        return {"success": True, "throughputs": throughputs, "results": results}
    except Exception as e:
        print(f"  ERROR: 解析 benchmark 结果失败: {e}")
        return {"success": False, "error": str(e)}


def update_optimizer_result(state_path: str, optimizer_script: str,
                            op_name: str, throughputs: Dict[str, float],
                            native_throughput: float) -> Dict[str, Any]:
    """调用 operator_optimizer.py update 更新结果"""
    tp_json = json.dumps(throughputs)
    result = run_cmd(
        f"python {optimizer_script} update "
        f"--op-name {op_name} "
        f"--throughputs '{tp_json}' "
        f"--native-throughput {native_throughput} "
        f"--state-path {state_path}",
        check=False
    )
    try:
        output = result.stdout.strip()
        start = output.index('{')
        end = output.rindex('}') + 1
        return json.loads(output[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "parse failed"}


# =============================================================================
# 框架开销预检
# =============================================================================

def preflight_framework_check(service_startup_cmd: str,
                               perf_config: str,
                               native_throughput: float,
                               wait_script: str = DEFAULT_WAIT_SCRIPT,
                               benchmark_script: str = DEFAULT_BENCHMARK_SCRIPT,
                               model_name: Optional[str] = None,
                               max_timeout: Optional[int] = None,
                               service_log_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Plugin 模式搜索前预检：验证 plugin 框架本身是否有性能开销。

    以 USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false 启动服务（禁用所有 FlagGems 算子），
    运行 quick benchmark 与 native_throughput 对比。

    返回:
        {"pass": bool, "ratio": float, "throughput": float, "message": str}
    """
    print("\n" + "=" * 60)
    print("[Preflight] 验证 plugin 框架开销")
    print("=" * 60)

    # 以 USE_FLAGGEMS=0 启动（完全禁用 FlagGems，仅保留 plugin 框架）
    env_inline = "USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false"
    svc_port = _read_service_port()
    if not restart_service(SERVICE_STOP_CMD, service_startup_cmd, wait_script,
                           env_inline=env_inline, port=svc_port,
                           model_name=model_name, max_timeout=max_timeout,
                           service_log_path=service_log_path):
        return {"pass": False, "ratio": 0, "throughput": 0,
                "message": "ERROR: 框架预检服务启动失败"}

    # 运行 quick benchmark
    bench = run_benchmark_quick(perf_config, benchmark_script, "preflight_framework")
    if not bench.get("success"):
        return {"pass": False, "ratio": 0, "throughput": 0,
                "message": f"ERROR: 框架预检 benchmark 失败: {bench.get('error', '?')}"}

    # 取最大 output 吞吐量与 native 对比（preflight 只做粗略判断）
    throughputs = bench.get("throughputs", {})
    if not throughputs:
        return {"pass": False, "ratio": 0, "throughput": 0,
                "message": "ERROR: 框架预检无有效吞吐量数据"}

    # 兼容新格式 {"case|conc": {"output": x, "total": y}} 和旧格式 {"case": float}
    output_vals = []
    for v in throughputs.values():
        if isinstance(v, dict):
            output_vals.append(v.get("output", 0) or 0)
        else:
            output_vals.append(v)
    max_tp = max(output_vals) if output_vals else 0
    ratio = max_tp / native_throughput if native_throughput > 0 else 0

    result = {
        "throughput": round(max_tp, 2),
        "native_throughput": round(native_throughput, 2),
        "ratio": round(ratio, 4),
        "throughputs": throughputs,
    }

    if ratio >= 0.95:
        result["pass"] = True
        result["message"] = f"PASS: 框架零开销 (ratio={ratio*100:.1f}%)"
        print(f"\n  [Preflight] {result['message']}")
    elif ratio >= 0.80:
        result["pass"] = True
        result["message"] = f"WARNING: 框架有轻微开销 (ratio={ratio*100:.1f}%)，继续搜索但需关注"
        print(f"\n  [Preflight] {result['message']}")
    else:
        result["pass"] = False
        result["message"] = f"ERROR: 框架本身性能 <80% (ratio={ratio*100:.1f}%)，建议先排查 plugin 问题再搜索算子"
        print(f"\n  [Preflight] {result['message']}")

    # 保存预检结果
    pf_path = str(Path(perf_config).parent.parent / "results" / "preflight_framework.json")
    try:
        save_json(result, pf_path)
        print(f"  预检结果已保存: {pf_path}")
    except Exception:
        pass

    return result


# =============================================================================
# 主搜索循环
# =============================================================================

def _read_service_port() -> Optional[int]:
    """从 context.yaml 读取服务端口"""
    try:
        import yaml
        with open("/flagos-workspace/shared/context.yaml", "r") as f:
            ctx = yaml.safe_load(f) or {}
        port = ctx.get("service", {}).get("port")
        return int(port) if port else None
    except Exception:
        return None


def _read_gpu_count() -> int:
    """从 context.yaml 读取 GPU 数量，fallback 到 SMI 工具探测"""
    try:
        import yaml
        with open("/flagos-workspace/shared/context.yaml", "r") as f:
            ctx = yaml.safe_load(f) or {}
        gpu_count = ctx.get("gpu", {}).get("count")
        if gpu_count:
            return int(gpu_count)
    except Exception:
        pass
    try:
        gpus = _parse_gpu_memory()
        if gpus:
            return len(gpus)
    except Exception:
        pass
    return 1


def run_search_step(state_path: str, perf_config: str,
                    service_startup_cmd: str,
                    gems_txt_path: Optional[str] = None,
                    plugin_mode: bool = False,
                    capabilities: Optional[List[str]] = None,
                    optimizer_script: str = DEFAULT_OPTIMIZER_SCRIPT,
                    benchmark_script: str = DEFAULT_BENCHMARK_SCRIPT,
                    toggle_script: str = DEFAULT_TOGGLE_SCRIPT,
                    wait_script: str = DEFAULT_WAIT_SCRIPT,
                    apply_config_script: str = DEFAULT_APPLY_CONFIG_SCRIPT,
                    model_name: Optional[str] = None,
                    max_timeout: Optional[int] = None,
                    service_log_path: Optional[str] = None) -> Dict[str, Any]:
    """执行单轮搜索步骤"""

    step_timing = {}

    # 1. 获取下一步操作
    print("\n" + "=" * 60)
    print(f"[搜索步骤] {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    action = get_next_action(state_path, optimizer_script)
    action_type = action.get("action", "error")

    if action_type in ("completed", "failed", "error"):
        print(f"\n搜索结束: {action.get('message', action_type)}")
        return action

    print(f"\n操作: {action.get('message', action_type)}")

    # 2. 应用算子配置
    t0 = time.time()
    state = load_json(state_path)
    config_result = apply_operator_config(
        action,
        apply_config_script=apply_config_script,
        plugin_mode=plugin_mode,
        capabilities=capabilities,
        gems_txt_path=gems_txt_path,
        all_ops=state.get("all_ops"),
        registered_ops=state.get("registered_ops"),
    )
    step_timing["config_seconds"] = round(time.time() - t0, 1)
    # plugin 模式返回 env_inline 字符串或 None，非 plugin 返回 True/False
    if plugin_mode:
        env_inline = config_result if isinstance(config_result, str) else None
        if not env_inline:
            return {"action": "error", "message": "算子配置应用失败（无 env_inline）"}
    else:
        if not config_result:
            return {"action": "error", "message": "算子配置应用失败"}
        env_inline = None

    # 3. 重启服务
    t0 = time.time()
    svc_port = _read_service_port()
    if not restart_service(SERVICE_STOP_CMD, service_startup_cmd, wait_script,
                           env_inline=env_inline, port=svc_port,
                           model_name=model_name, max_timeout=max_timeout,
                           service_log_path=service_log_path):
        return {"action": "error", "message": "服务重启失败"}
    step_timing["restart_seconds"] = round(time.time() - t0, 1)

    # 3.5. 重启后验证算子变化 — 运行时 txt 是实际生效算子的权威来源
    runtime_ops = verify_ops_via_txt()
    if runtime_ops is not None:
        test_disabled = action.get("test_disabled_ops", [])
        unexpected = [op for op in test_disabled if op in runtime_ops]
        if unexpected:
            print(f"  [验证] 警告: {len(unexpected)} 个应禁用的算子仍在运行时 txt 中: {unexpected[:5]}")
        # 将运行时实际算子列表回写到 optimizer 状态，作为后续判定的权威依据
        state = load_json(state_path)
        state["runtime_enabled_ops"] = sorted(runtime_ops)
        state["runtime_enabled_count"] = len(runtime_ops)
        save_json(state, state_path)
        print(f"  [验证] 运行时实际启用 {len(runtime_ops)} 个算子（已回写 state）")

    # 4. 运行 benchmark
    t0 = time.time()
    bench_result = run_benchmark_quick(perf_config, benchmark_script,
                                       f"search_step_{action.get('step', 0)}")
    step_timing["benchmark_seconds"] = round(time.time() - t0, 1)

    if not bench_result.get("success"):
        return {"action": "error", "message": f"Benchmark 失败: {bench_result.get('error', '?')}"}

    # 5. 更新结果
    state = load_json(state_path)
    native_tp = state.get("native_throughput", 0)
    throughputs = bench_result.get("throughputs", {})

    op_name = action.get("group", action.get("round", action.get("op", "unknown")))
    update = update_optimizer_result(
        state_path, optimizer_script,
        op_name, throughputs, native_tp
    )

    # 构建返回结果，附带运行时实际算子数
    runtime_count = len(runtime_ops) if runtime_ops is not None else None
    expected_count = len(action.get("test_enabled_ops", []))
    if runtime_count is not None and runtime_count != expected_count:
        print(f"  [注意] 预期启用 {expected_count} 个算子，运行时实际 {runtime_count} 个")

    print(f"\n[步骤完成] decision={update.get('decision', '?')}, "
          f"ratio={update.get('ratio', 0)*100:.1f}%"
          f" (config={step_timing['config_seconds']}s"
          f" restart={step_timing['restart_seconds']}s"
          f" bench={step_timing['benchmark_seconds']}s)")

    return {
        "action": action_type,
        "step": action.get("step", 0),
        "op_name": op_name,
        "decision": update.get("decision", "?"),
        "ratio": update.get("ratio", 0),
        "status": update.get("status", "?"),
        "timing": step_timing,
        "runtime_enabled_count": runtime_count,
        "expected_enabled_count": expected_count,
    }


def run_full_search(state_path: str, perf_config: str,
                    service_startup_cmd: str,
                    max_rounds: int = 20,
                    gems_txt_path: Optional[str] = None,
                    plugin_mode: bool = False,
                    capabilities: Optional[List[str]] = None,
                    model_name: Optional[str] = None,
                    max_timeout: Optional[int] = None,
                    **kwargs) -> Dict[str, Any]:
    """运行完整搜索循环"""
    # 读取状态并检查残留 completed 状态
    _state = {}
    try:
        _state = load_json(state_path)
        search_direction = _state.get("search_direction", "forward")
        # 防御：上次失败后残留 status=completed 但无有效搜索结果，自动重置
        # 额外检查 disabled_ops 为空，避免误重置正常完成但 completed_at 写入失败的情况
        if (_state.get("status") == "completed"
                and not _state.get("completed_at")
                and not _state.get("disabled_ops")):
            print("  ⚠ 检测到残留 status=completed（无 completed_at 且无 disabled_ops），自动重置为 in_progress")
            _state["status"] = "in_progress"
            _state["current_step"] = 0
            save_json(_state, state_path)
    except Exception:
        search_direction = "forward"

    print(f"\n{'#' * 60}")
    print(f"# 算子搜索开始 (最多 {max_rounds} 轮)")
    if plugin_mode:
        _search_mode = _state.get("search_mode", "progressive")
        print(f"# 模式: Plugin (OOT → {_search_mode} 两阶段)")
    print(f"# 搜索方向: {search_direction}")
    print(f"{'#' * 60}\n")

    search_log = []
    start_time = time.time()
    framework_check = None
    preflight_elapsed = 0

    # Plugin 模式：搜索前验证框架开销
    if plugin_mode:
        try:
            _state = load_json(state_path)
            native_tp = _state.get("native_throughput", 0)
        except Exception:
            native_tp = 0

        if native_tp > 0:
            t_pf = time.time()
            framework_check = preflight_framework_check(
                service_startup_cmd, perf_config, native_tp,
                wait_script=kwargs.get("wait_script", DEFAULT_WAIT_SCRIPT),
                benchmark_script=kwargs.get("benchmark_script", DEFAULT_BENCHMARK_SCRIPT),
                model_name=model_name,
                max_timeout=max_timeout,
                service_log_path=kwargs.get("service_log_path"),
            )
            preflight_elapsed = round(time.time() - t_pf, 1)
            if not framework_check.get("pass") and framework_check.get("ratio", 1.0) < 0.80:
                print("\nERROR: 框架本身性能 <80%，建议先排查 plugin 问题")
                print("搜索仍将继续，但结果可能不可靠\n")

    search_start = time.time()
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"第 {round_num}/{max_rounds} 轮")
        print(f"{'=' * 60}")

        # 每轮开始前检查 GPU 可用性
        required_gpus = _read_gpu_count()
        gpu_check = check_gpu_availability(required_gpus=required_gpus)
        if not gpu_check["available"]:
            print(f"  ⚠ GPU 不足: {gpu_check['message']}，等待 10s 后重试...")
            time.sleep(10)
            gpu_check = check_gpu_availability(required_gpus=required_gpus)
            if not gpu_check["available"]:
                # 尝试清理残留进程
                print("  尝试清理残留推理进程...")
                run_cmd("pkill -9 -f 'vllm\\|sglang' 2>/dev/null", check=False)
                time.sleep(10)
                gpu_check = check_gpu_availability(required_gpus=required_gpus)
                if not gpu_check["available"]:
                    print(f"  FATAL: GPU 资源不可用 ({gpu_check['message']})，搜索中止")
                    search_log.append({
                        "action": "error",
                        "message": f"GPU 资源不可用: {gpu_check['message']}",
                        "round": round_num,
                    })
                    break
            print(f"  ✓ GPU 恢复可用: {gpu_check['message']}")

        result = run_search_step(
            state_path, perf_config, service_startup_cmd,
            gems_txt_path=gems_txt_path,
            plugin_mode=plugin_mode,
            capabilities=capabilities,
            model_name=model_name,
            max_timeout=max_timeout,
            **kwargs
        )

        search_log.append(result)

        if result.get("action") in ("completed", "failed", "error"):
            break

    elapsed = time.time() - start_time
    search_elapsed = round(time.time() - search_start, 1)
    total_rounds = len(search_log)

    # 汇总子阶段累计耗时
    benchmark_total = sum(r.get("timing", {}).get("benchmark_seconds", 0) for r in search_log)
    restart_total = sum(r.get("timing", {}).get("restart_seconds", 0) for r in search_log)
    config_total = sum(r.get("timing", {}).get("config_seconds", 0) for r in search_log)

    # 最终状态
    try:
        state = load_json(state_path)
    except Exception:
        state = {}

    # 运行时实际算子数以最后一轮的 runtime_enabled_count 为准
    last_runtime_count = None
    for r in reversed(search_log):
        if r.get("runtime_enabled_count") is not None:
            last_runtime_count = r["runtime_enabled_count"]
            break

    summary = {
        "total_rounds": total_rounds,
        "elapsed_seconds": round(elapsed),
        "elapsed_display": f"{int(elapsed // 60)}m{int(elapsed % 60)}s",
        "final_status": state.get("status", "unknown"),
        "search_direction": state.get("search_direction", "forward"),
        "enabled_ops": len(state.get("enabled_ops", [])),
        "disabled_ops": len(state.get("disabled_ops", [])),
        "disabled_list": state.get("disabled_ops", []),
        "runtime_enabled_ops": state.get("runtime_enabled_ops"),
        "runtime_enabled_count": last_runtime_count,
        "framework_check": framework_check,
        "search_log": search_log,
        "timing": {
            "total_seconds": round(elapsed),
            "preflight_seconds": preflight_elapsed,
            "search_seconds": search_elapsed,
            "rounds": total_rounds,
            "benchmark_total_seconds": round(benchmark_total, 1),
            "restart_total_seconds": round(restart_total, 1),
            "config_total_seconds": round(config_total, 1),
        },
    }

    print(f"\n{'#' * 60}")
    print(f"# 搜索完成: {total_rounds} 轮, 耗时 {summary['elapsed_display']}")
    print(f"# 状态: {summary['final_status']}")
    print(f"# 启用: {summary['enabled_ops']}, 禁用: {summary['disabled_ops']}")
    if last_runtime_count is not None:
        print(f"# 运行时实际启用: {last_runtime_count} 个算子（以运行时 txt 为准）")
    if summary["disabled_list"]:
        print(f"# 禁用列表: {', '.join(summary['disabled_list'])}")
    print(f"{'#' * 60}\n")

    # 保存摘要
    summary_path = str(Path(state_path).parent / "search_summary.json")
    save_json(summary, summary_path)
    print(f"搜索摘要已保存: {summary_path}")

    return summary


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="算子搜索编排 — 自动化完整搜索循环")

    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    # 公共参数
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-path", default=DEFAULT_STATE_PATH, help="优化器状态文件")
    common.add_argument("--perf-config", default=DEFAULT_PERF_CONFIG, help="性能测试配置")
    common.add_argument("--service-startup-cmd", required=True, help="服务启动命令")
    common.add_argument("--gems-txt-path", help="gems.txt 路径（非 plugin 兜底写入）")
    common.add_argument("--plugin-mode", action="store_true", help="Plugin 模式（环境变量控制）")
    common.add_argument("--capabilities", help="已废弃：capabilities 始终自动探测，此参数被忽略")
    common.add_argument("--optimizer-script", default=DEFAULT_OPTIMIZER_SCRIPT)
    common.add_argument("--benchmark-script", default=DEFAULT_BENCHMARK_SCRIPT)
    common.add_argument("--toggle-script", default=DEFAULT_TOGGLE_SCRIPT)
    common.add_argument("--wait-script", default=DEFAULT_WAIT_SCRIPT)
    common.add_argument("--apply-config-script", default=DEFAULT_APPLY_CONFIG_SCRIPT)
    common.add_argument("--model-name", help="模型名称（传递给 wait_for_service.sh 精确验证）")
    common.add_argument("--max-timeout", type=int, default=1800, help="服务启动绝对超时上限（秒）")
    common.add_argument("--service-log-path", help="服务实际日志路径（start_service.sh 内部重定向的目标文件，用于 wait_for_service 动态超时监控）")

    # run — 完整搜索
    run_parser = subparsers.add_parser("run", parents=[common], help="运行完整搜索循环")
    run_parser.add_argument("--max-rounds", type=int, default=20, help="最大搜索轮次")

    # step — 单步搜索
    step_parser = subparsers.add_parser("step", parents=[common], help="运行单轮搜索")

    # status — 查看状态
    status_parser = subparsers.add_parser("status", help="查看搜索状态")
    status_parser.add_argument("--state-path", default=DEFAULT_STATE_PATH)

    args = parser.parse_args()

    if args.command == "run":
        caps = args.capabilities.split(",") if args.capabilities else None
        result = run_full_search(
            args.state_path, args.perf_config,
            args.service_startup_cmd,
            max_rounds=args.max_rounds,
            gems_txt_path=args.gems_txt_path,
            plugin_mode=args.plugin_mode,
            capabilities=caps,
            model_name=args.model_name,
            max_timeout=args.max_timeout,
            optimizer_script=args.optimizer_script,
            benchmark_script=args.benchmark_script,
            toggle_script=args.toggle_script,
            wait_script=args.wait_script,
            apply_config_script=args.apply_config_script,
            service_log_path=args.service_log_path,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "step":
        caps = args.capabilities.split(",") if args.capabilities else None
        result = run_search_step(
            args.state_path, args.perf_config,
            args.service_startup_cmd,
            gems_txt_path=args.gems_txt_path,
            plugin_mode=args.plugin_mode,
            capabilities=caps,
            model_name=args.model_name,
            max_timeout=args.max_timeout,
            optimizer_script=args.optimizer_script,
            benchmark_script=args.benchmark_script,
            toggle_script=args.toggle_script,
            wait_script=args.wait_script,
            apply_config_script=args.apply_config_script,
            service_log_path=args.service_log_path,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "status":
        try:
            state = load_json(args.state_path)
            info = {
                "status": state.get("status"),
                "search_mode": state.get("search_mode"),
                "search_direction": state.get("search_direction", "forward"),
                "current_step": state.get("current_step"),
                "enabled": len(state.get("enabled_ops", [])),
                "disabled": len(state.get("disabled_ops", [])),
                "disabled_list": state.get("disabled_ops", []),
            }
            gs = state.get("group_state", {})
            if gs:
                idx = gs.get("current_group_idx", 0)
                order = gs.get("group_order", [])
                info["current_group"] = order[idx] if idx < len(order) else "done"
                info["group_results"] = gs.get("group_results", {})
            print(json.dumps(info, indent=2, ensure_ascii=False))
        except FileNotFoundError:
            print(json.dumps({"error": f"状态文件不存在: {args.state_path}"}))

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        step_id = os.environ.get("FLAGOS_STEP_ID", "07_perf_tuning")
        step_title = os.environ.get("FLAGOS_STEP_TITLE", "算子优化")
        write_checkpoint(step_id, step_title, "running_operator_search",
                         action_detail=" ".join(sys.argv))
        main()
    except Exception as e:
        write_last_error(
            tool="operator_search.py",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=traceback.format_exc(),
        )
        print(f"[FATAL] operator_search.py 异常退出: {e}")
        traceback.print_exc()
        sys.exit(1)
