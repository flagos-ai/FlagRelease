#!/usr/bin/env python3
"""V5 算子扩展脚本 — 从 V3 基础上最大化开启算子 (Royal Megamaster)

策略：先试顶，失败降级（最少轮次实现最多算子）：
  Tier 0（试顶）  ：一次性开启 V3 + 全部被禁算子。启动成功 + 精度达标 → 直接完成（1轮）。
  Tier 1（次顶）  ：排除已知精度问题算子（eval.excluded_ops_accuracy），
                    开启 V3 + 其余被禁算子（性能原因/崩溃来源为主，精度风险最低）。
                    通过 → 该批算子批量保留，仅剩精度问题算子逐个探测。
  Tier 2（兜底）  ：现状逐个增量探测（启用 → 重启 → 精度评测，达标保留否则回退）。

无性能要求，仅需服务可启动 + 精度达标（相对退化 ≤ threshold）。
任一 tier 失败只损失一轮启动（+精度评测），正确性与纯逐个探测一致。

用法:
    python3 operator_expansion.py \\
        --context-yaml /flagos-workspace/shared/context.yaml \\
        --v1-result /flagos-workspace/results/gpqa_v1.json \\
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh --mode flagos" \\
        --accuracy-threshold 5.0 \\
        --output-dir /flagos-workspace/results/ \\
        --state-path /flagos-workspace/results/operator_config_v5.json

支持断点续跑：中断后重跑自动跳过已验证的算子与已失败的 tier（通过 state-path 记录进度）。
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

# 算子配置应用走统一共享模块（唯一权威实现，见 flagos_op_config.py）
from flagos_op_config import (
    is_plugin_env as _is_plugin_env,
    persist_env as _persist_env,
    write_op_config as write_control_file,
    DEFAULT_CONTROL_FILE,
)


# =============================================================================
# 常量
# =============================================================================

DEFAULT_WAIT_SCRIPT = "/flagos-workspace/scripts/wait_for_service.sh"
DEFAULT_BENCHMARK_SCRIPT = "/flagos-workspace/scripts/benchmark_runner.py"
DEFAULT_FAST_GPQA_SCRIPT = "/flagos-workspace/scripts/fast_gpqa.py"
DEFAULT_GPQA_CONFIG = "/flagos-workspace/scripts/fast_gpqa_config.yaml"
# DEFAULT_CONTROL_FILE 由 flagos_op_config 提供（统一共享模块）


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


def get_accuracy_excluded_ops(ctx: dict) -> List[str]:
    """从 context 中获取因精度问题被禁用的算子（步骤5/11 记录）。

    用于 tier 分级：这批算子精度风险最高，Tier 1 试顶时排除它们。
    崩溃/性能来源的禁用没有单独标签，归入"其余"（精度风险最低）。
    """
    ev = ctx.get('eval', {})
    excluded = ev.get('excluded_ops_accuracy', [])
    if isinstance(excluded, list):
        return excluded
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
# 算子配置应用（_is_plugin_env / write_control_file / _persist_env）已收敛到
# 统一共享模块 flagos_op_config（见文件头 import），此处不再保留本地副本。
# =============================================================================

def restart_and_wait(service_cmd: str, wait_script: str, port: int = 8000,
                     model_name: str = "", timeout: int = 300,
                     max_timeout: int = 1800) -> bool:
    """重启容器并等待服务就绪，返回是否成功。

    杀进程/等端口逻辑与 operator_reduction.restart_and_wait 对齐：
    pkill -9 强杀（含 worker 的 multiprocessing.spawn）+ 显式等端口释放，
    避免旧进程未死透/端口未释放导致新服务启动失败。
    """
    # 用 pkill -9 强杀所有 vllm 及其 worker 进程
    subprocess.run(
        "pkill -9 -f 'vllm|multiprocessing.spawn' 2>/dev/null; sleep 3",
        shell=True, capture_output=True
    )
    # 等待端口释放
    for _ in range(15):
        rc = subprocess.run(
            f"ss -tlnp 2>/dev/null | grep -q ':{port}\\b'",
            shell=True, capture_output=True
        )
        if rc.returncode != 0:
            break
        time.sleep(1)
    time.sleep(2)

    # 清理 triton/flaggems 缓存
    for cache_dir in ["/root/.triton/cache/", "/tmp/triton_cache/", "/root/.flaggems/code_cache/"]:
        if os.path.exists(cache_dir):
            subprocess.run(f"rm -rf {cache_dir}", shell=True, capture_output=True)

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
    accuracy_excluded_ops: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """执行 V5 算子扩展（先试顶，失败降级）

    Returns:
        扩展结果字典，包含 expanded_ops, incompatible_ops, accuracy_harmful_ops, final_enabled_ops
    """
    accuracy_excluded_ops = accuracy_excluded_ops or []

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
            "tier_results": {},  # {"tier0": "passed"|"failed", "tier1": ...}
            "actual_rounds": 0,  # 真实启动轮次（tier 尝试 + 逐个探测）
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": False,
        }
        save_json(state, state_path)
    # 旧版本 state 兼容：缺 tier_results 且已有探测进度 → 视为 Tier 2 进行中，不再试顶
    if "tier_results" not in state:
        state["tier_results"] = {} if not state.get("probed_ops") else {"tier0": "skipped_legacy", "tier1": "skipped_legacy"}
    if "actual_rounds" not in state:
        state["actual_rounds"] = len(state.get("probed_ops", {}))

    probed = state.get("probed_ops", {})
    enabled_ops = state.get("current_enabled_ops", current_enabled_ops.copy())
    tier_results = state["tier_results"]

    def _bulk_try(candidate_ops: List[str], label: str) -> tuple:
        """一次性开启 candidate_ops，返回 (service_ok, accuracy_ok, score)"""
        state["actual_rounds"] += 1
        write_control_file(candidate_ops)
        print(f"  ✓ 控制文件已更新 ({len(candidate_ops)} 个算子)")
        print(f"  ▶ 重启服务...")
        service_ok = restart_and_wait(
            service_cmd, wait_script, port=port,
            model_name=model_name, max_timeout=max_timeout
        )
        if not service_ok:
            print(f"  ✗ [{label}] 服务启动失败")
            return False, False, 0.0
        print(f"  ▶ 运行精度评测...")
        temp_output = os.path.join(output_dir, f"gpqa_v5_{label}.json")
        passed, score = run_accuracy_check(
            gpqa_script, gpqa_config, temp_output, v1_score, threshold
        )
        return True, passed, score

    total = len(state["disabled_ops_to_probe"])

    # =========================================================================
    # Tier 0（试顶）：V3 + 全部被禁算子，一把开满
    # =========================================================================
    if "tier0" not in tier_results and not probed:
        t0_candidate = enabled_ops + [op for op in state["disabled_ops_to_probe"] if op not in enabled_ops]
        print(f"\n{'=' * 60}")
        print(f"  [Tier 0 试顶] 一次性开启全部 {total} 个被禁算子（共 {len(t0_candidate)} 个）")
        print(f"{'=' * 60}")
        svc_ok, acc_ok, score = _bulk_try(t0_candidate, "tier0")
        if svc_ok and acc_ok:
            print(f"  ✓ [Tier 0] 直接达标 (score={score:.1f}%) — 全部算子保留，扩展完成（1轮）")
            for op in state["disabled_ops_to_probe"]:
                probed[op] = "expanded"
            enabled_ops = t0_candidate
            tier_results["tier0"] = "passed"
        else:
            reason = "精度不达标" if svc_ok else "服务启动失败"
            print(f"  ✗ [Tier 0] {reason} → 降级")
            tier_results["tier0"] = "failed"
        state.update({"probed_ops": probed, "current_enabled_ops": enabled_ops,
                      "tier_results": tier_results})
        save_json(state, state_path)

    # =========================================================================
    # Tier 1（次顶）：排除已知精度问题算子，批量开启其余（性能/崩溃来源，精度风险最低）
    # =========================================================================
    if (tier_results.get("tier0") == "failed" and "tier1" not in tier_results):
        acc_set = set(accuracy_excluded_ops)
        t1_batch = [op for op in state["disabled_ops_to_probe"]
                    if op not in acc_set and op not in probed]
        if t1_batch and len(t1_batch) < len([op for op in state["disabled_ops_to_probe"] if op not in probed]):
            t1_candidate = enabled_ops + [op for op in t1_batch if op not in enabled_ops]
            print(f"\n{'=' * 60}")
            print(f"  [Tier 1 次顶] 排除 {len(acc_set & set(state['disabled_ops_to_probe']))} 个精度问题算子，"
                  f"批量开启其余 {len(t1_batch)} 个")
            print(f"{'=' * 60}")
            svc_ok, acc_ok, score = _bulk_try(t1_candidate, "tier1")
            if svc_ok and acc_ok:
                print(f"  ✓ [Tier 1] 达标 (score={score:.1f}%) — 该批 {len(t1_batch)} 个算子批量保留，"
                      f"剩余精度问题算子转逐个探测")
                for op in t1_batch:
                    probed[op] = "expanded"
                enabled_ops = t1_candidate
                tier_results["tier1"] = "passed"
            else:
                reason = "精度不达标" if svc_ok else "服务启动失败"
                print(f"  ✗ [Tier 1] {reason} → 降级到逐个探测")
                tier_results["tier1"] = "failed"
        else:
            # 无精度禁用记录（T1 集合=T0 集合）或该批为空 → 跳过
            tier_results["tier1"] = "skipped"
            print(f"\n  [Tier 1] 跳过（无精度禁用分类记录或候选为空），直接进入逐个探测")
        state.update({"probed_ops": probed, "current_enabled_ops": enabled_ops,
                      "tier_results": tier_results})
        save_json(state, state_path)

    # =========================================================================
    # Tier 2（兜底）：逐个增量探测剩余算子
    # 排序：非精度禁用的在前（成功率高，先扩大安全基），精度问题算子最后
    # =========================================================================
    acc_set = set(accuracy_excluded_ops)
    ops_to_probe = [op for op in state["disabled_ops_to_probe"] if op not in probed]
    ops_to_probe.sort(key=lambda op: (op in acc_set, op))

    done_count = len(probed)

    print(f"\n{'=' * 60}")
    print(f"  V5 算子扩展（逐个探测） — 共 {total} 个算子，已完成 {done_count}，待探测 {len(ops_to_probe)}")
    print(f"  V1 基线精度: {v1_score}%, 阈值: ±{threshold}%")
    print(f"{'=' * 60}\n")

    for i, op in enumerate(ops_to_probe):
        round_num = done_count + i + 1
        print(f"\n[轮次 {round_num}/{total}] 尝试重新开启算子: {op}")
        print("-" * 40)
        state["actual_rounds"] += 1

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
    subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
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
    subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)

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
        "expansion_rounds": state.get("actual_rounds", len(probed)),
        "tier_results": state.get("tier_results", {}),
        "strategy": (
            "tier0_all_at_once" if state.get("tier_results", {}).get("tier0") == "passed"
            else "tier1_bulk_plus_incremental" if state.get("tier_results", {}).get("tier1") == "passed"
            else "incremental"
        ),
    }

    print(f"\n{'#' * 60}")
    print(f"# V5 算子扩展结果")
    print(f"#   策略: {result['strategy']} (tier0={result['tier_results'].get('tier0','-')}, tier1={result['tier_results'].get('tier1','-')})")
    print(f"#   原始禁用: {total} 个")
    print(f"#   成功重新开启: {len(expanded_ops)} 个")
    print(f"#   服务不兼容: {len(incompatible_ops)} 个")
    print(f"#   精度不达标: {len(accuracy_harmful_ops)} 个")
    print(f"#   最终启用总数: {len(enabled_ops)} 个")
    print(f"#   实际启动轮次: {result['expansion_rounds']} (纯逐个需 {total} 轮)")
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

    # 获取精度问题算子分类（tier 分级依据）
    accuracy_excluded = get_accuracy_excluded_ops(ctx)

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
        accuracy_excluded_ops=accuracy_excluded,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
