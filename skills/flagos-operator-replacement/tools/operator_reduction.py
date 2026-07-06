#!/usr/bin/env python3
"""V4 算子精简脚本 — 从 V3 基础上减少替换算子以提升性能 (Flag-express)

与 operator_expansion.py（V5，逐个"加"算子最大化精度）相反：
V4 从 V3 的启用算子集出发，逐个尝试"移除"对性能有拖累的算子：
  - 移除算子 → 重启服务 → benchmark（+ 轻量精度校验）
  - 移除后性能提升（且服务可启动、精度不崩）→ 确认移除
  - 否则回退保留

目标：性能 ≥ V3，接近/超过 V1。
硬约束：
  1. **非 plugin 环境至少保留 1 个算子**；plugin 环境可减至 0（USE_FLAGGEMS=0 时 plugin 仍可运行）
  2. **精度必须合格**：精度是所有版本（含 V4）成立的前提。逐轮移除算子时做精度护栏，
     收尾再做一次最终精度终检，相对基线退化超护栏则 V4 不成立（success=False）。
     缺精度基线时标记 accuracy_verified=false，编排层须兜底重判，不得直接判达标。
     精度判据统一为「相对退化」口径（与 accuracy_compare.py 一致）。

用法:
    python3 operator_reduction.py \\
        --context-yaml /flagos-workspace/shared/context.yaml \\
        --v1-perf /flagos-workspace/results/native_performance.json \\
        --v3-perf /flagos-workspace/results/flagos_optimized.json \\
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh --mode flagos" \\
        --output-dir /flagos-workspace/results/ \\
        --state-path /flagos-workspace/results/operator_config_v4.json

支持断点续跑：中断后重跑自动跳过已探测的算子（通过 state-path 记录进度）。

退出码: 0=成功, 1=失败
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


# =============================================================================
# 常量
# =============================================================================

DEFAULT_WAIT_SCRIPT = "/flagos-workspace/scripts/wait_for_service.sh"
DEFAULT_BENCHMARK_SCRIPT = "/flagos-workspace/scripts/benchmark_runner.py"
DEFAULT_FAST_GPQA_SCRIPT = "/flagos-workspace/scripts/fast_gpqa.py"
DEFAULT_GPQA_CONFIG = "/flagos-workspace/scripts/fast_gpqa_config.yaml"
DEFAULT_CONTROL_FILE = "/root/flaggems_ops_control.json"
DEFAULT_LOG = "/flagos-workspace/logs/startup_v4.log"

# 性能增益判定：移除算子后综合吞吐提升超过此比例才确认移除（默认 1%）
DEFAULT_GAIN_THRESHOLD = 0.01
# 精度护栏：移除算子后相对精度退化不得超过此值（防止为性能牺牲精度）
DEFAULT_ACCURACY_GUARD = 5.0


# =============================================================================
# 工具函数（与 operator_expansion.py 保持一致的 I/O 约定）
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


def get_enabled_ops(ctx: dict) -> List[str]:
    """从 context 获取 V3 的启用算子列表（V4 的起点）。

    优先级：versions.v3.enabled_ops > optimization.enabled_ops > initial - disabled
    """
    v3 = ctx.get('versions', {}).get('v3', {})
    v3_ops = v3.get('enabled_ops', [])
    if isinstance(v3_ops, list) and v3_ops:
        return list(v3_ops)

    opt = ctx.get('optimization', {})
    enabled = opt.get('enabled_ops', [])
    if isinstance(enabled, list) and enabled:
        return list(enabled)

    initial = ctx.get('service', {}).get('initial_operator_list', [])
    disabled = set(opt.get('disabled_ops', []) or [])
    return [op for op in initial if op not in disabled]


# =============================================================================
# 性能指标提取（复用 benchmark_runner 的扁平输出格式）
# =============================================================================

def extract_throughput(perf_path: str) -> Tuple[float, float]:
    """从 benchmark 结果 JSON 提取综合 (output_tp, total_tp)。

    benchmark_runner 输出格式：{test_case: {concurrency: {metric: value}}}
    取所有 test_case/concurrency 中 Output/Total throughput 的最大值作为代表
    （quick 模式通常只有单一 test_case + 单一 concurrency）。
    """
    data = load_json(perf_path)
    if not data:
        return 0.0, 0.0
    max_out, max_total = 0.0, 0.0
    for tc_name, tc_results in data.items():
        if tc_name.startswith("_") or not isinstance(tc_results, dict):
            continue
        for conc, metrics in tc_results.items():
            if conc.startswith("_") or not isinstance(metrics, dict):
                continue
            out = metrics.get('Output token throughput (tok/s)', 0) or 0
            tot = metrics.get('Total token throughput (tok/s)', 0) or 0
            try:
                max_out = max(max_out, float(out))
                max_total = max(max_total, float(tot))
            except (TypeError, ValueError):
                continue
    return max_out, max_total


def composite_throughput(out_tp: float, total_tp: float) -> float:
    """综合吞吐指标：优先 output，回退 total。用于轮次间性能比较。"""
    return out_tp if out_tp > 0 else total_tp


# =============================================================================
# 服务控制（与 operator_expansion.py 一致）
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
    """根据环境类型选择算子控制方式：plugin 环境用 WHITELIST，非 plugin 用控制文件。

    plugin 环境下允许 enabled_ops 为空（USE_FLAGGEMS=0），此时 plugin 仍可独立运行。
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
        _persist_env("USE_FLAGGEMS", "1")
        os.makedirs(os.path.dirname(control_file), exist_ok=True)
        with open(control_file, 'w') as f:
            json.dump({"include": sorted(enabled_ops)}, f, indent=2, ensure_ascii=False)
        _persist_env("FLAGGEMS_CONTROL_MODE", "only_enable")


def _persist_env(key: str, value: str):
    etc_env = "/etc/environment"
    lines = []
    if os.path.exists(etc_env):
        with open(etc_env, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}\n")
    with open(etc_env, 'w') as f:
        f.writelines(lines)
    os.environ[key] = value


def _load_etc_environment():
    """加载 /etc/environment 中 FlagGems 相关变量到 os.environ"""
    etc_env = "/etc/environment"
    if not os.path.exists(etc_env):
        return
    with open(etc_env, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.startswith(('USE_FLAGGEMS', 'FLAGGEMS_', 'VLLM_FL_')):
                os.environ[key] = val


def restart_and_wait(service_cmd: str, wait_script: str, port: int = 8000,
                     model_name: str = "", timeout: int = 300,
                     max_timeout: int = 1800) -> bool:
    """重启服务并等待就绪，返回是否成功"""
    # 用 pkill -9 强杀所有 vllm/sglang 及其 worker 进程
    subprocess.run(
        "pkill -9 -f 'vllm|sglang|multiprocessing.spawn' 2>/dev/null; sleep 3",
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

    for cache_dir in ["/root/.triton/cache/", "/tmp/triton_cache/", "/root/.flaggems/code_cache/"]:
        if os.path.exists(cache_dir):
            subprocess.run(f"rm -rf {cache_dir}", shell=True, capture_output=True)

    subprocess.Popen(
        service_cmd, shell=True,
        stdout=open(DEFAULT_LOG, "w"),
        stderr=subprocess.STDOUT
    )

    # 简单轮询 health endpoint，不依赖日志监控
    start_t = time.time()
    while time.time() - start_t < max_timeout:
        try:
            rc = subprocess.run(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health",
                shell=True, capture_output=True, text=True, timeout=10
            )
            if rc.stdout.strip() == "200":
                return True
        except Exception:
            pass
        time.sleep(10)
    return False


def run_benchmark(benchmark_script: str, output_dir: str, output_name: str) -> Tuple[float, float]:
    """运行 quick benchmark，返回 (output_tp, total_tp)"""
    cmd = (
        f"python3 {benchmark_script} --quick"
        f" --output-name {output_name}"
        f" --output-dir {output_dir}"
    )
    rc, stdout, stderr = run_cmd(cmd, timeout=1200)
    if rc != 0:
        print(f"  ⚠ benchmark 失败: {stderr[:200]}")
        return 0.0, 0.0
    return extract_throughput(os.path.join(output_dir, f"{output_name}.json"))


def run_accuracy_guard(gpqa_script: str, gpqa_config: str, output_path: str,
                       nv_or_v1_score: float, guard: float) -> Tuple[bool, float]:
    """轻量精度护栏：移除算子后精度相对基线退化不超过 guard（相对退化 %）。

    返回 (passed, score)。基线分数 <=0 时返回 (True, 0.0) 表示「无法校验」——
    调用方须结合 accuracy_baseline 是否 >0 区分「校验通过」与「未校验」，
    不得把未校验当作达标（V4 精度合格是版本成立前提）。
    """
    if nv_or_v1_score <= 0:
        return True, 0.0
    cmd = f"python3 {gpqa_script} --config {gpqa_config} --output {output_path}"
    rc, stdout, stderr = run_cmd(cmd, timeout=7200)
    if rc != 0:
        print(f"  ⚠ 精度护栏评测失败: {stderr[:200]}")
        return False, 0.0
    try:
        score = load_json(output_path).get('score', 0.0)
        rel_drop = (nv_or_v1_score - score) / nv_or_v1_score * 100
        return rel_drop <= guard, score
    except Exception as e:
        print(f"  ⚠ 读取精度结果失败: {e}")
        return False, 0.0


# =============================================================================
# 核心：贪心减算子循环
# =============================================================================

def measure_config(enabled_ops: List[str], service_cmd: str, wait_script: str,
                   benchmark_script: str, output_dir: str, tag: str,
                   port: int, model_name: str, max_timeout: int) -> Tuple[bool, float, float]:
    """应用算子集 → 重启 → benchmark。返回 (service_ok, output_tp, total_tp)。"""
    write_control_file(enabled_ops)
    ok = restart_and_wait(service_cmd, wait_script, port=port,
                          model_name=model_name, max_timeout=max_timeout)
    if not ok:
        return False, 0.0, 0.0
    out_tp, total_tp = run_benchmark(benchmark_script, output_dir, f"v4_probe_{tag}")
    return True, out_tp, total_tp


def run_reduction(
    enabled_ops: List[str],
    v1_composite: float,
    v3_composite: float,
    service_cmd: str,
    state_path: str,
    output_dir: str,
    wait_script: str = DEFAULT_WAIT_SCRIPT,
    benchmark_script: str = DEFAULT_BENCHMARK_SCRIPT,
    gpqa_script: str = DEFAULT_FAST_GPQA_SCRIPT,
    gpqa_config: str = DEFAULT_GPQA_CONFIG,
    gain_threshold: float = DEFAULT_GAIN_THRESHOLD,
    accuracy_guard: float = DEFAULT_ACCURACY_GUARD,
    accuracy_baseline: float = 0.0,
    max_rounds: int = 0,
    port: int = 8000,
    model_name: str = "",
    max_timeout: int = 1800,
) -> Dict[str, Any]:
    """贪心减算子：每轮测量移除各候选算子后的性能，移除增益最大且为正者。

    硬约束：非 plugin 环境至少保留 1 个算子；plugin 环境可减至 0。
    """
    state = load_json(state_path)
    if not state:
        state = {
            "mode": "reduction",
            "version": "v4",
            "initial_enabled_ops": enabled_ops.copy(),
            "current_enabled_ops": enabled_ops.copy(),
            "reduced_ops": [],          # 已确认移除的算子（有序）
            "probed_rounds": [],        # 每轮记录
            "v1_composite": v1_composite,
            "v3_composite": v3_composite,
            "gain_threshold": gain_threshold,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": False,
        }
        save_json(state, state_path)

    current_ops = state.get("current_enabled_ops", enabled_ops.copy())
    reduced_ops = state.get("reduced_ops", [])
    # plugin 环境允许减至 0 算子（USE_FLAGGEMS=0 时 plugin 仍可独立运行）
    min_ops = 0 if _is_plugin_env() else 1

    # 建立当前基线性能（V3 起点）
    print(f"\n{'=' * 60}")
    print(f"  V4 算子精简 (Flag-express) — 起点 {len(current_ops)} 个算子")
    print(f"  V3 综合吞吐: {v3_composite:.1f} tok/s | V1 综合吞吐: {v1_composite:.1f} tok/s")
    print(f"  目标: 性能 ≥ V3，接近/超过 V1 | 保底至少 {min_ops} 个算子")
    print(f"{'=' * 60}\n")

    print("  ▶ 测量当前配置基线性能...")
    ok, out_tp, total_tp = measure_config(
        current_ops, service_cmd, wait_script, benchmark_script,
        output_dir, "baseline", port, model_name, max_timeout
    )
    if not ok:
        print("  ✗ 起点配置服务无法启动，V4 无法进行")
        return {"success": False, "reason": "起点服务启动失败"}
    best_composite = composite_throughput(out_tp, total_tp)
    print(f"  ✓ 当前基线综合吞吐: {best_composite:.1f} tok/s\n")

    round_num = len(state.get("probed_rounds", []))
    round_limit = max_rounds if max_rounds > 0 else len(current_ops)

    while len(current_ops) > min_ops and round_num < round_limit:
        round_num += 1
        print(f"\n[轮次 {round_num}] 当前 {len(current_ops)} 个算子，逐个探测移除增益")
        print("-" * 40)

        best_gain = 0.0
        best_op = None
        best_op_composite = best_composite
        probe_detail = []

        for op in list(current_ops):
            if len(current_ops) - 1 < min_ops:
                break  # 保底约束
            trial_ops = [o for o in current_ops if o != op]
            print(f"  ▶ 尝试移除 {op} (剩 {len(trial_ops)} 个)...")
            ok, out_tp, total_tp = measure_config(
                trial_ops, service_cmd, wait_script, benchmark_script,
                output_dir, f"r{round_num}_{op}", port, model_name, max_timeout
            )
            if not ok:
                print(f"    ✗ 移除 {op} 后服务无法启动 → 保留")
                probe_detail.append({"op": op, "service_ok": False})
                continue
            trial_composite = composite_throughput(out_tp, total_tp)
            gain = (trial_composite - best_composite) / best_composite if best_composite > 0 else 0
            print(f"    吞吐 {trial_composite:.1f} tok/s (增益 {gain * 100:+.2f}%)")
            probe_detail.append({"op": op, "service_ok": True,
                                 "composite": round(trial_composite, 2),
                                 "gain_pct": round(gain * 100, 2)})
            if gain > best_gain:
                best_gain = gain
                best_op = op
                best_op_composite = trial_composite

        state.setdefault("probed_rounds", []).append({
            "round": round_num,
            "probes": probe_detail,
            "selected": best_op,
            "gain_pct": round(best_gain * 100, 2),
        })

        if best_op is None or best_gain < gain_threshold:
            print(f"\n  ⤷ 本轮无正增益（最佳 {best_gain * 100:+.2f}% < 阈值 {gain_threshold * 100:.1f}%），停止精简")
            save_json(state, state_path)
            break

        # 精度护栏：确认移除前校验精度不崩
        trial_ops = [o for o in current_ops if o != best_op]
        guard_ok = True
        if accuracy_baseline > 0:
            print(f"  ▶ 精度护栏校验（移除 {best_op}）...")
            write_control_file(trial_ops)
            restart_and_wait(service_cmd, wait_script, port=port,
                             model_name=model_name, max_timeout=max_timeout)
            guard_ok, gscore = run_accuracy_guard(
                gpqa_script, gpqa_config,
                os.path.join(output_dir, f"gpqa_v4_guard_r{round_num}.json"),
                accuracy_baseline, accuracy_guard
            )
            if not guard_ok:
                print(f"    ✗ 移除 {best_op} 导致精度退化超护栏 (score={gscore:.1f}%) → 保留并停止")
                state["probed_rounds"][-1]["accuracy_rejected"] = best_op
                save_json(state, state_path)
                break

        # 确认移除
        current_ops = trial_ops
        reduced_ops.append(best_op)
        best_composite = best_op_composite
        state["current_enabled_ops"] = current_ops
        state["reduced_ops"] = reduced_ops
        save_json(state, state_path)
        print(f"\n  ✓ 确认移除 {best_op} → 综合吞吐 {best_composite:.1f} tok/s，剩 {len(current_ops)} 个算子")

    # 停止服务
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)

    # 最终配置固化 + 收尾评测
    write_control_file(current_ops)
    oplist_path = os.path.join(output_dir, "v4_oplist.txt")
    with open(oplist_path, 'w') as f:
        for op in sorted(current_ops):
            f.write(f"{op}\n")

    print(f"\n{'=' * 60}")
    print(f"  V4 精简完成 — 运行最终性能 + 精度终检")
    print(f"{'=' * 60}")
    final_composite = best_composite
    final_out, final_total = 0.0, 0.0
    final_score = 0.0
    accuracy_ok: Optional[bool] = None
    accuracy_rel_drop_pct: Optional[float] = None
    ok = restart_and_wait(service_cmd, wait_script, port=port,
                          model_name=model_name, max_timeout=max_timeout)
    if ok:
        final_out, final_total = run_benchmark(benchmark_script, output_dir, "v4_performance")
        final_composite = composite_throughput(final_out, final_total)
        # 精度终检：V4 最终配置必须精度合格（所有版本成立的前提都是精度合格）
        if accuracy_baseline > 0:
            print("  ▶ V4 最终精度终检...")
            accuracy_ok, final_score = run_accuracy_guard(
                gpqa_script, gpqa_config,
                os.path.join(output_dir, "gpqa_v4.json"),
                accuracy_baseline, accuracy_guard
            )
            accuracy_rel_drop_pct = round(
                (accuracy_baseline - final_score) / accuracy_baseline * 100, 2
            )
            if accuracy_ok:
                print(f"    ✓ V4 精度终检达标: score={final_score:.1f}%, "
                      f"相对退化={accuracy_rel_drop_pct:.2f}% (护栏 {accuracy_guard:.0f}%)")
            else:
                print(f"    ✗ V4 精度终检不达标: score={final_score:.1f}%, "
                      f"相对退化={accuracy_rel_drop_pct:.2f}% > 护栏 {accuracy_guard:.0f}%")
        else:
            print("  ⚠ 缺精度基线（accuracy_baseline<=0），无法进行 V4 精度终检 "
                  "→ 精度状态标记为未验证，编排层需以 V1/NV 基线兜底后重判")
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null", shell=True, capture_output=True)

    v4_ratio_v1 = (final_composite / v1_composite) if v1_composite > 0 else 0
    beats_v3 = final_composite >= v3_composite
    beats_v1 = v1_composite > 0 and final_composite >= v1_composite

    state["completed"] = True
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["current_enabled_ops"] = current_ops
    state["final_composite"] = final_composite
    state["v4_score"] = final_score
    state["accuracy_ok"] = accuracy_ok
    save_json(state, state_path)

    # 精度是所有版本成立的前提：终检明确不达标 → V4 不成立（success=False）
    # 缺基线未验证（accuracy_ok is None）→ success 保持 True，但标记 accuracy_verified=False
    accuracy_verified = accuracy_ok is not None
    success = (accuracy_ok is not False)

    result = {
        "success": success,
        "reduced_ops": reduced_ops,
        "kept_ops": sorted(current_ops),
        "final_enabled_count": len(current_ops),
        "v1_composite": round(v1_composite, 2),
        "v3_composite": round(v3_composite, 2),
        "v4_composite": round(final_composite, 2),
        "v4_ratio_v1_pct": round(v4_ratio_v1 * 100, 2),
        "beats_v3": beats_v3,
        "beats_v1": beats_v1,
        "reduction_rounds": round_num,
        "final_output_tp": round(final_out, 2),
        "final_total_tp": round(final_total, 2),
        # ---- 精度终检（V4 成立前提）----
        "v4_score": round(final_score, 2),
        "accuracy_ok": accuracy_ok,
        "accuracy_verified": accuracy_verified,
        "accuracy_baseline": round(accuracy_baseline, 2) if accuracy_baseline > 0 else None,
        "accuracy_rel_drop_pct": accuracy_rel_drop_pct,
        "accuracy_guard_pct": accuracy_guard,
    }
    if accuracy_ok is False:
        result["reason"] = "V4 精度终检不达标（精度是版本成立前提）"

    print(f"\n{'#' * 60}")
    print(f"# V4 算子精简结果 (Flag-express)")
    print(f"#   起点算子: {len(state['initial_enabled_ops'])} 个")
    print(f"#   移除算子: {len(reduced_ops)} 个 → {reduced_ops}")
    print(f"#   最终保留: {len(current_ops)} 个")
    print(f"#   V4 综合吞吐: {final_composite:.1f} tok/s")
    print(f"#   相对 V1 性能比: {v4_ratio_v1 * 100:.1f}% | 超过 V3: {beats_v3} | 超过 V1: {beats_v1}")
    if accuracy_ok is None:
        print(f"#   精度终检: 未验证（缺基线）")
    else:
        print(f"#   精度终检: {'达标' if accuracy_ok else '不达标'} "
              f"(score={final_score:.1f}%, 退化={accuracy_rel_drop_pct:.2f}%)")
    print(f"{'#' * 60}\n")

    print(f"[REDUCTION_RESULT]{json.dumps(result, ensure_ascii=False)}[/REDUCTION_RESULT]")
    return result


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V4 算子精简 — 从 V3 减少算子以提升性能 (Flag-express)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--context-yaml", required=True,
                        help="context.yaml 路径，读取 V3 启用算子集")
    parser.add_argument("--v1-perf", required=True,
                        help="V1 性能结果 JSON (native_performance.json)")
    parser.add_argument("--v3-perf", required=True,
                        help="V3 性能结果 JSON (flagos_optimized.json)")
    parser.add_argument("--service-startup-cmd", required=True,
                        help="服务启动命令")
    parser.add_argument("--output-dir", default="/flagos-workspace/results/",
                        help="输出目录")
    parser.add_argument("--state-path", default="/flagos-workspace/results/operator_config_v4.json",
                        help="状态文件路径（支持断点续跑）")
    parser.add_argument("--wait-script", default=DEFAULT_WAIT_SCRIPT)
    parser.add_argument("--benchmark-script", default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument("--gpqa-script", default=DEFAULT_FAST_GPQA_SCRIPT)
    parser.add_argument("--gpqa-config", default=DEFAULT_GPQA_CONFIG)
    parser.add_argument("--gain-threshold", type=float, default=DEFAULT_GAIN_THRESHOLD,
                        help=f"性能增益阈值 (比例，默认 {DEFAULT_GAIN_THRESHOLD})")
    parser.add_argument("--accuracy-guard", type=float, default=DEFAULT_ACCURACY_GUARD,
                        help=f"精度护栏：相对退化上限 (%%, 默认 {DEFAULT_ACCURACY_GUARD})")
    parser.add_argument("--accuracy-baseline", type=float, default=0.0,
                        help="精度护栏基线分数 (NV 或 V1)。V4 精度合格是版本成立前提，"
                             "应始终传入；0=无基线，精度状态标记为未验证（accuracy_verified=false），"
                             "编排层须以 V1/NV 基线兜底后重判，不得直接判 V4 达标")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="最大精简轮数 (0=不限，最多算子数-1)")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-timeout", type=int, default=1800)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # 加载 /etc/environment 中的 FlagGems 相关变量（docker exec 不继承）
    _load_etc_environment()

    ctx = load_context(args.context_yaml)
    enabled_ops = get_enabled_ops(ctx)
    if not enabled_ops:
        print("错误: 无法从 context 获取 V3 启用算子列表")
        sys.exit(1)
    min_ops = 0 if _is_plugin_env() else 1
    if len(enabled_ops) <= min_ops:
        print(f"⚠ V3 仅 {len(enabled_ops)} 个算子，已满足保底约束（min={min_ops}），无需精简")
        print(f"[REDUCTION_RESULT]{json.dumps({'success': True, 'reduced_ops': [], 'kept_ops': enabled_ops, 'final_enabled_count': len(enabled_ops), 'reduction_rounds': 0, 'skipped': True}, ensure_ascii=False)}[/REDUCTION_RESULT]")
        sys.exit(0)

    v1_out, v1_total = extract_throughput(args.v1_perf)
    v3_out, v3_total = extract_throughput(args.v3_perf)
    v1_composite = composite_throughput(v1_out, v1_total)
    v3_composite = composite_throughput(v3_out, v3_total)

    port = args.port or ctx.get('service', {}).get('port', 8000)
    model_name = args.model_name or ctx.get('service', {}).get('model_id', '')

    result = run_reduction(
        enabled_ops=enabled_ops,
        v1_composite=v1_composite,
        v3_composite=v3_composite,
        service_cmd=args.service_startup_cmd,
        state_path=args.state_path,
        output_dir=args.output_dir,
        wait_script=args.wait_script,
        benchmark_script=args.benchmark_script,
        gpqa_script=args.gpqa_script,
        gpqa_config=args.gpqa_config,
        gain_threshold=args.gain_threshold,
        accuracy_guard=args.accuracy_guard,
        accuracy_baseline=args.accuracy_baseline,
        max_rounds=args.max_rounds,
        port=port,
        model_name=model_name,
        max_timeout=args.max_timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
