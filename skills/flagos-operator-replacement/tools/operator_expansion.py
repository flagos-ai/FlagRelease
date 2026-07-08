#!/usr/bin/env python3
"""V5 算子扩展脚本 — 从 V3 基础上最大化开启算子 (Royal Megamaster)

从当前禁用算子列表出发，逐个尝试重新开启：
  - 启用算子 → 重启服务 → 精度评测
  - 服务启动成功 + 精度误差 ≤ threshold → 保留该算子
  - 否则回退

无性能要求，仅需服务可启动 + 精度达标。

用法:
    python3 operator_expansion.py \\
        --context-yaml /flagos-workspace/shared/context.yaml \\
        --v1-result /flagos-workspace/results/gpqa_v1.json \\
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh --mode flagos" \\
        --accuracy-threshold 5.0 \\
        --output-dir /flagos-workspace/results/ \\
        --state-path /flagos-workspace/results/operator_config_v5.json

支持断点续跑：中断后重跑自动跳过已验证的算子（通过 state-path 记录进度）。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any


# =============================================================================
# 常量
# =============================================================================

DEFAULT_WAIT_SCRIPT = "/flagos-workspace/scripts/wait_for_service.sh"
DEFAULT_BENCHMARK_SCRIPT = "/flagos-workspace/scripts/benchmark_runner.py"
DEFAULT_FAST_GPQA_SCRIPT = "/flagos-workspace/scripts/fast_gpqa.py"
DEFAULT_GPQA_CONFIG = "/flagos-workspace/scripts/fast_gpqa_config.yaml"
DEFAULT_CONTROL_FILE = "/root/flaggems_ops_control.json"


# =============================================================================
# 工具函数
# =============================================================================

def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_cmd(cmd: str, timeout: int = 1800) -> tuple:
    """运行命令，返回 (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"命令超时 ({timeout}s): {cmd}"
    except Exception as e:
        return -1, "", str(e)


def load_context(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_disabled_ops(ctx: dict) -> List[str]:
    """从 context 中获取当前被禁用的算子列表"""
    opt = ctx.get('optimization', {})
    disabled = opt.get('disabled_ops', [])
    if isinstance(disabled, list):
        return disabled
    return []


def get_enabled_ops(ctx: dict) -> List[str]:
    """从 context 中获取当前启用的算子列表"""
    opt = ctx.get('optimization', {})
    enabled = opt.get('enabled_ops', [])
    if isinstance(enabled, list):
        return enabled
    # fallback: 从 service.initial_operator_list 获取全量，减去 disabled
    initial = ctx.get('service', {}).get('initial_operator_list', [])
    disabled = set(get_disabled_ops(ctx))
    return [op for op in initial if op not in disabled]


def get_v1_score(v1_result_path: str) -> Optional[float]:
    """读取 V1 精度分数"""
    try:
        with open(v1_result_path, 'r') as f:
            data = json.load(f)
        return data.get('score')
    except Exception:
        return None


# =============================================================================
# 核心逻辑
# =============================================================================

def _is_plugin_env() -> bool:
    """判断当前是否为 plugin 控制环境"""
    return os.environ.get("VLLM_FL_PREFER_ENABLED") == "true" or _env_has("VLLM_FL_PREFER_ENABLED")


def _env_has(key: str) -> bool:
    """Check if key exists in /etc/environment"""
    etc_env = "/etc/environment"
    if not os.path.exists(etc_env):
        return False
    with open(etc_env) as f:
        return any(l.startswith(f"{key}=") for l in f)


def _clear_env(key: str):
    """从 /etc/environment 中移除某个变量"""
    etc_env = "/etc/environment"
    if not os.path.exists(etc_env):
        return
    with open(etc_env, 'r') as f:
        lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
    with open(etc_env, 'w') as f:
        f.writelines(lines)
    os.environ.pop(key, None)


def write_control_file(enabled_ops: List[str], control_file: str = DEFAULT_CONTROL_FILE):
    """根据环境类型选择算子控制方式：plugin 环境用 WHITELIST env，非 plugin 用控制文件。

    ⚠ plugin 场景（VLLM_FL_PREFER_ENABLED=true）下控制文件完全失效（注入代码 pass），
    必须走 VLLM_FL_FLAGOS_WHITELIST env。此前本函数缺 plugin 分支 → V5 在 plugin 下扩算子
    静默不生效（见 v5-operator-expansion-whitelist-bug）。现与 operator_reduction.write_control_file 对齐。
    """
    if _is_plugin_env():
        # plugin 模式：通过环境变量控制，清除可能冲突的 BLACKLIST
        _clear_env("VLLM_FL_FLAGOS_BLACKLIST")
        if enabled_ops:
            _persist_env("USE_FLAGGEMS", "1")
            whitelist = ",".join(sorted(enabled_ops))
            _persist_env("VLLM_FL_FLAGOS_WHITELIST", whitelist)
        else:
            _persist_env("USE_FLAGGEMS", "0")
            _persist_env("VLLM_FL_FLAGOS_WHITELIST", "")
        _persist_env("VLLM_FL_PREFER_ENABLED", "true")
    else:
        # 非 plugin：通过控制文件
        os.makedirs(os.path.dirname(control_file), exist_ok=True)
        control = {"include": sorted(enabled_ops)}
        with open(control_file, 'w') as f:
            json.dump(control, f, indent=2, ensure_ascii=False)
        _persist_env("FLAGGEMS_CONTROL_MODE", "only_enable")
        _persist_env("USE_FLAGGEMS", "1")


def _persist_env(key: str, value: str):
    """将环境变量写入 /etc/environment（持久化）"""
    etc_env = "/etc/environment"
    lines = []
    if os.path.exists(etc_env):
        with open(etc_env, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}\n")
    with open(etc_env, 'w') as f:
        f.writelines(lines)
    os.environ[key] = value


def restart_and_wait(service_cmd: str, wait_script: str, port: int = 8000,
                     model_name: str = "", timeout: int = 300,
                     max_timeout: int = 1800) -> bool:
    """重启容器并等待服务就绪，返回是否成功"""
    # 清理 triton/flaggems 缓存
    for cache_dir in ["/root/.triton/cache/", "/tmp/triton_cache/", "/root/.flaggems/code_cache/"]:
        if os.path.exists(cache_dir):
            subprocess.run(f"rm -rf {cache_dir}", shell=True, capture_output=True)

    # 杀死残留进程
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null; sleep 3", shell=True, capture_output=True)

    # 释放 GPU（等待显存释放）
    time.sleep(5)

    # 启动服务（后台）
    subprocess.Popen(
        service_cmd, shell=True,
        stdout=open("/flagos-workspace/logs/startup_v5.log", "w"),
        stderr=subprocess.STDOUT
    )

    # 等待服务就绪
    wait_cmd = (
        f"{wait_script} --port {port}"
        f" --timeout {timeout} --max-timeout {max_timeout}"
        f" --log-path /flagos-workspace/logs/startup_v5.log"
        f" --mode flagos"
    )
    if model_name:
        wait_cmd += f" --model-name '{model_name}'"

    rc, stdout, stderr = run_cmd(wait_cmd, timeout=max_timeout + 60)
    return rc == 0


def run_accuracy_check(gpqa_script: str, gpqa_config: str,
                       output_path: str, v1_score: float,
                       threshold: float) -> tuple:
    """运行精度评测，返回 (passed: bool, score: float)"""
    cmd = f"python3 {gpqa_script} --config {gpqa_config} --output {output_path}"
    rc, stdout, stderr = run_cmd(cmd, timeout=7200)  # 精度评测可能较慢

    if rc != 0:
        print(f"  ⚠ 精度评测失败: {stderr[:200]}")
        return False, 0.0

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        score = data.get('score', 0.0)
        diff = v1_score - score
        passed = diff <= threshold
        return passed, score
    except Exception as e:
        print(f"  ⚠ 读取精度结果失败: {e}")
        return False, 0.0


def run_expansion(
    disabled_ops: List[str],
    current_enabled_ops: List[str],
    v1_score: float,
    service_cmd: str,
    state_path: str,
    output_dir: str,
    threshold: float = 5.0,
    wait_script: str = DEFAULT_WAIT_SCRIPT,
    gpqa_script: str = DEFAULT_FAST_GPQA_SCRIPT,
    gpqa_config: str = DEFAULT_GPQA_CONFIG,
    benchmark_script: str = DEFAULT_BENCHMARK_SCRIPT,
    port: int = 8000,
    model_name: str = "",
    max_timeout: int = 1800,
) -> Dict[str, Any]:
    """执行 V5 算子扩展循环

    Returns:
        扩展结果字典，包含 expanded_ops, incompatible_ops, accuracy_harmful_ops, final_enabled_ops
    """
    # 加载或初始化状态
    state = load_json(state_path)
    if not state:
        state = {
            "mode": "expansion",
            "version": "v5",
            "disabled_ops_to_probe": disabled_ops.copy(),
            "probed_ops": {},  # {op: "expanded" | "incompatible" | "accuracy_harmful"}
            "current_enabled_ops": current_enabled_ops.copy(),
            "v1_score": v1_score,
            "threshold": threshold,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": False,
        }
        save_json(state, state_path)

    probed = state.get("probed_ops", {})
    enabled_ops = state.get("current_enabled_ops", current_enabled_ops.copy())
    ops_to_probe = [op for op in state["disabled_ops_to_probe"] if op not in probed]

    total = len(state["disabled_ops_to_probe"])
    done_count = len(probed)

    print(f"\n{'=' * 60}")
    print(f"  V5 算子扩展 — 共 {total} 个算子待探测，已完成 {done_count}")
    print(f"  V1 基线精度: {v1_score}%, 阈值: ±{threshold}%")
    print(f"{'=' * 60}\n")

    for i, op in enumerate(ops_to_probe):
        round_num = done_count + i + 1
        print(f"\n[轮次 {round_num}/{total}] 尝试重新开启算子: {op}")
        print("-" * 40)

        # 1. 临时将算子加入 enabled 列表
        test_enabled = enabled_ops + [op]

        # 2. 写控制文件
        write_control_file(test_enabled)
        print(f"  ✓ 控制文件已更新 ({len(test_enabled)} 个算子)")

        # 3. 重启服务
        print(f"  ▶ 重启服务...")
        service_ok = restart_and_wait(
            service_cmd, wait_script, port=port,
            model_name=model_name, max_timeout=max_timeout
        )

        if not service_ok:
            print(f"  ✗ 服务启动失败 → 标记 {op} 为 incompatible")
            probed[op] = "incompatible"
            # 回退控制文件
            write_control_file(enabled_ops)
            state["probed_ops"] = probed
            save_json(state, state_path)
            continue

        # 4. 精度评测
        print(f"  ▶ 运行精度评测...")
        temp_output = os.path.join(output_dir, f"gpqa_v5_probe_{op}.json")
        passed, score = run_accuracy_check(
            gpqa_script, gpqa_config, temp_output, v1_score, threshold
        )

        if not passed:
            diff = v1_score - score
            print(f"  ✗ 精度不达标 (score={score:.1f}%, diff={diff:.1f}%) → 标记 {op} 为 accuracy_harmful")
            probed[op] = "accuracy_harmful"
            # 回退
            write_control_file(enabled_ops)
        else:
            print(f"  ✓ 精度达标 (score={score:.1f}%) → 保留算子 {op}")
            probed[op] = "expanded"
            enabled_ops = test_enabled  # 正式加入

        # 保存进度
        state["probed_ops"] = probed
        state["current_enabled_ops"] = enabled_ops
        save_json(state, state_path)

    # 停止服务，释放 GPU
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)

    # 汇总结果
    expanded_ops = [op for op, status in probed.items() if status == "expanded"]
    incompatible_ops = [op for op, status in probed.items() if status == "incompatible"]
    accuracy_harmful_ops = [op for op, status in probed.items() if status == "accuracy_harmful"]

    # 写最终控制文件
    write_control_file(enabled_ops)

    # 写最终算子列表
    oplist_path = os.path.join(output_dir, "v5_oplist.txt")
    with open(oplist_path, 'w') as f:
        for op in sorted(enabled_ops):
            f.write(f"{op}\n")

    # 运行最终精度评测
    print(f"\n{'=' * 60}")
    print(f"  V5 扩展完成 — 运行最终评测")
    print(f"{'=' * 60}")

    final_gpqa_path = os.path.join(output_dir, "gpqa_v5.json")
    # 需要先启动服务
    write_control_file(enabled_ops)
    service_ok = restart_and_wait(
        service_cmd, wait_script, port=port,
        model_name=model_name, max_timeout=max_timeout
    )
    final_score = 0.0
    if service_ok:
        _, final_score = run_accuracy_check(
            gpqa_script, gpqa_config, final_gpqa_path, v1_score, threshold
        )
        print(f"  V5 最终精度: {final_score:.1f}%")

        # 运行一次性能测试（信息性，不判定）
        print(f"  ▶ 运行最终性能测试（信息性）...")
        perf_cmd = (
            f"python3 {benchmark_script} --quick"
            f" --output-name v5_performance"
            f" --output-dir {output_dir}"
        )
        run_cmd(perf_cmd, timeout=600)
    else:
        print(f"  ⚠ 最终服务启动失败，跳过评测")

    # 停止服务
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null", shell=True, capture_output=True)

    # 标记完成
    state["completed"] = True
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["current_enabled_ops"] = enabled_ops
    state["probed_ops"] = probed
    save_json(state, state_path)

    result = {
        "success": True,
        "expanded_ops": expanded_ops,
        "incompatible_ops": incompatible_ops,
        "accuracy_harmful_ops": accuracy_harmful_ops,
        "final_enabled_count": len(enabled_ops),
        "final_disabled_count": len(incompatible_ops) + len(accuracy_harmful_ops),
        "original_disabled_count": total,
        "v5_score": final_score,
        "v1_score": v1_score,
        "expansion_rounds": len(probed),
    }

    print(f"\n{'#' * 60}")
    print(f"# V5 算子扩展结果")
    print(f"#   原始禁用: {total} 个")
    print(f"#   成功重新开启: {len(expanded_ops)} 个")
    print(f"#   服务不兼容: {len(incompatible_ops)} 个")
    print(f"#   精度不达标: {len(accuracy_harmful_ops)} 个")
    print(f"#   最终启用总数: {len(enabled_ops)} 个")
    print(f"#   V5 精度: {final_score:.1f}%")
    print(f"{'#' * 60}\n")

    # 输出 JSON 结果（供编排层读取）
    print(f"[EXPANSION_RESULT]{json.dumps(result, ensure_ascii=False)}[/EXPANSION_RESULT]")

    return result


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V5 算子扩展 — 从 V3 基础上最大化开启算子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--context-yaml", required=True,
                        help="context.yaml 路径，读取当前算子状态")
    parser.add_argument("--v1-result", required=True,
                        help="V1 精度结果文件路径 (gpqa_native.json / gpqa_v1.json)")
    parser.add_argument("--service-startup-cmd", required=True,
                        help="服务启动命令 (如 'bash /flagos-workspace/scripts/start_service.sh --mode flagos')")
    parser.add_argument("--accuracy-threshold", type=float, default=5.0,
                        help="精度误差阈值 (%%, 默认 5.0)")
    parser.add_argument("--output-dir", default="/flagos-workspace/results/",
                        help="输出目录")
    parser.add_argument("--state-path", default="/flagos-workspace/results/operator_config_v5.json",
                        help="状态文件路径（支持断点续跑）")
    parser.add_argument("--wait-script", default=DEFAULT_WAIT_SCRIPT,
                        help="服务等待脚本路径")
    parser.add_argument("--gpqa-script", default=DEFAULT_FAST_GPQA_SCRIPT,
                        help="精度评测脚本路径")
    parser.add_argument("--gpqa-config", default=DEFAULT_GPQA_CONFIG,
                        help="精度评测配置文件路径")
    parser.add_argument("--benchmark-script", default=DEFAULT_BENCHMARK_SCRIPT,
                        help="性能测试脚本路径")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口")
    parser.add_argument("--model-name", default="",
                        help="模型名称（传递给 wait_for_service.sh）")
    parser.add_argument("--max-timeout", type=int, default=1800,
                        help="服务启动最大超时（秒）")
    parser.add_argument("--json", action="store_true",
                        help="最终输出 JSON 格式")

    args = parser.parse_args()

    # 读取 context
    ctx = load_context(args.context_yaml)

    # 获取禁用算子列表
    disabled_ops = get_disabled_ops(ctx)
    if not disabled_ops:
        print("✓ 无禁用算子，V5 = V3（无需扩展）")
        result = {
            "success": True,
            "expanded_ops": [],
            "incompatible_ops": [],
            "accuracy_harmful_ops": [],
            "final_enabled_count": len(get_enabled_ops(ctx)),
            "final_disabled_count": 0,
            "original_disabled_count": 0,
            "v5_score": 0,
            "v1_score": 0,
            "expansion_rounds": 0,
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    # 获取 V1 精度
    v1_score = get_v1_score(args.v1_result)
    if v1_score is None:
        print(f"错误: 无法读取 V1 精度结果: {args.v1_result}")
        sys.exit(1)

    # 获取当前启用算子
    enabled_ops = get_enabled_ops(ctx)

    # 获取模型名和端口
    port = args.port or ctx.get('service', {}).get('port', 8000)
    model_name = args.model_name or ctx.get('service', {}).get('model_id', '')

    # 执行扩展
    result = run_expansion(
        disabled_ops=disabled_ops,
        current_enabled_ops=enabled_ops,
        v1_score=v1_score,
        service_cmd=args.service_startup_cmd,
        state_path=args.state_path,
        output_dir=args.output_dir,
        threshold=args.accuracy_threshold,
        wait_script=args.wait_script,
        gpqa_script=args.gpqa_script,
        gpqa_config=args.gpqa_config,
        benchmark_script=args.benchmark_script,
        port=port,
        model_name=model_name,
        max_timeout=args.max_timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
