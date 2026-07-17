#!/usr/bin/env python3
"""V4 算子精简脚本 — 从 V3 算子集里选极少算子以提升性能 (Flag-express)

新流程 v3.1 定稿算法（只要相对基线有提升即采纳，不追求最优组合）：
  - 起点 / 回退版（精度已合格）：
      · V2.1 路径（有非 plugin V2 对照）：起点 = V2.1 调优后算子列表 ∩ V3 算子列表，
        以该起点实测性能作为"优化基线"；起点本身精度已合格，作 2 轮拿不到时的回退版。
      · V2.2 / 无 V1 路径（无非 plugin V2 对照）：起点 = V3 算子集，
        "优化基线" = V2 首测性能 × 1.05（无实测交集起点）。
  - 每轮从 V3 算子列表里随机选 1~3 个算子，**只开启这几个**（其余全部关闭）启动服务：
      · 性能优于优化基线 → 进一步测精度（对比 NV 基线，rel_drop ≤ 护栏）；
        精度达标 → 采纳为 V4，结束；精度不达标 → 回步骤 2 重新选取算子。
      · 性能不优于优化基线 → 该组合不采纳，回步骤 2 重选。
  - 步骤循环 **不超过 2 轮**。2 轮内拿不到"性能提升 + 精度达标" → 回退到起点
    （V2.1 交集 / V2.2 的 V3 集），保证精度硬闸门不被破坏。
硬约束：
  1. **V4 至少保留 1 个算子**（plugin 也不例外）——随机选取下限即为 1。
  2. **精度必须合格**：精度是所有版本（含 V4）成立的前提，判据统一为
     rel_drop = (NV基线 − 当前) / NV基线 ≤ 护栏（与 accuracy_compare.py 一致）。
     缺精度基线时标记 accuracy_verified=false，编排层须兜底重判，不得直接判达标。

用法:
    python3 operator_reduction.py \\
        --context-yaml /flagos-workspace/shared/context.yaml \\
        --v1-perf /flagos-workspace/results/native_performance.json \\
        --v3-perf /flagos-workspace/results/flagos_optimized.json \\
        --service-startup-cmd "bash /flagos-workspace/scripts/start_service.sh --mode flagos" \\
        --v2-path 2.1 \\
        --accuracy-baseline <NV_score> \\
        --output-dir /flagos-workspace/results/ \\
        --state-path /flagos-workspace/results/operator_config_v4.json

支持断点续跑：中断后重跑自动跳过已完成的轮次（通过 state-path 记录进度）。

退出码: 0=成功, 1=失败
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# 算子配置应用走统一共享模块（唯一权威实现，见 flagos_op_config.py）
from flagos_op_config import (
    is_plugin_env as _is_plugin_env,
    persist_env as _persist_env,
    write_op_config as write_control_file,
    load_etc_environment as _load_etc_environment,
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
DEFAULT_LOG = "/flagos-workspace/logs/startup_v4.log"

# 精度护栏：随机组合相对 NV 基线退化不得超过此值（rel_drop ≤ 护栏，防止为性能牺牲精度）
DEFAULT_ACCURACY_GUARD = 5.0


# =============================================================================
# 工具函数（统一的 JSON I/O 约定）
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
# 服务控制
# 算子配置应用（_is_plugin_env / write_control_file / _persist_env / _load_etc_environment）
# 已收敛到统一共享模块 flagos_op_config（见文件头 import），此处不再保留本地副本。
# =============================================================================

def restart_and_wait(service_cmd: str, wait_script: str, port: int = 8000,
                     model_name: str = "", timeout: int = 300,
                     max_timeout: int = 5760) -> bool:
    """重启服务并等待就绪，返回是否成功"""
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


def _pick_random_subset(pool: List[str], rng: "random.Random", lo: int = 1, hi: int = 3) -> List[str]:
    """从 pool 里随机选 lo~hi 个算子（只开这几个）。pool 少于 lo 个时全取。"""
    k = min(len(pool), rng.randint(lo, min(hi, len(pool))))
    k = max(k, min(1, len(pool)))
    return sorted(rng.sample(pool, k))


def run_reduction(
    v3_ops: List[str],
    start_ops: List[str],
    perf_baseline: float,
    v1_composite: float,
    v3_composite: float,
    baseline_source: str,
    service_cmd: str,
    state_path: str,
    output_dir: str,
    wait_script: str = DEFAULT_WAIT_SCRIPT,
    benchmark_script: str = DEFAULT_BENCHMARK_SCRIPT,
    gpqa_script: str = DEFAULT_FAST_GPQA_SCRIPT,
    gpqa_config: str = DEFAULT_GPQA_CONFIG,
    accuracy_guard: float = DEFAULT_ACCURACY_GUARD,
    accuracy_baseline: float = 0.0,
    max_rounds: int = 2,
    seed: int = 0,
    port: int = 8000,
    model_name: str = "",
    max_timeout: int = 1800,
) -> Dict[str, Any]:
    """V4 精简（Flag-express）：从 V3 算子集里随机选 1~3 个算子只开这几个。

    算法（新流程 v3.1）：
      - 每轮从 v3_ops 里随机选 1~3 个算子，**只开启这几个**（其余全关）启动服务：
          · 性能 > perf_baseline（优化基线）→ 测精度（对比 NV 基线 accuracy_baseline，
            rel_drop ≤ accuracy_guard）；精度达标 → 采纳为 V4，结束。
          · 性能不优 或 精度不达标 → 该组合不采纳，回步骤 2 重选。
      - 步骤循环不超过 max_rounds（默认 2）轮。
      - 2 轮拿不到"性能提升 + 精度达标" → 回退到 start_ops（精度已合格的起点：
        V2.1 交集 / V2.2 的 V3 集），保证精度硬闸门不被破坏。
      - 硬约束：至少保留 1 个算子（随机选取下限即为 1）。

    perf_baseline / baseline_source：
      - V2.1 路径：baseline_source='intersection'，perf_baseline = 起点交集实测吞吐；
      - V2.2/无V1 路径：baseline_source='v2x1.05'，perf_baseline = V2 首测 × 1.05。
    """
    state = load_json(state_path)
    if not state:
        state = {
            "mode": "reduction",
            "version": "v4",
            "v3_ops": list(v3_ops),
            "start_ops": list(start_ops),          # 精度已合格的回退版
            "perf_baseline": round(perf_baseline, 2),
            "baseline_source": baseline_source,
            "v1_composite": v1_composite,
            "v3_composite": v3_composite,
            "probed_rounds": [],                    # 每轮记录（含随机选取、性能、精度）
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": False,
        }
        save_json(state, state_path)

    min_ops = 1
    rng = random.Random(seed)
    probed_rounds = state.get("probed_rounds", [])
    # 断点续跑：重放已消费的随机步，保持随机序列与状态一致
    for _ in probed_rounds:
        _pick_random_subset(v3_ops, rng)

    print(f"\n{'=' * 60}")
    print(f"  V4 算子精简 (Flag-express) — 从 V3 的 {len(v3_ops)} 个算子里随机选 1~3 个只开")
    print(f"  优化基线: {perf_baseline:.1f} tok/s (来源: {baseline_source})")
    print(f"  V3 综合吞吐: {v3_composite:.1f} tok/s | 精度基线(NV): "
          f"{accuracy_baseline if accuracy_baseline > 0 else '缺失'}")
    print(f"  规则: 性能>优化基线 且 精度 rel_drop≤{accuracy_guard:.0f}% 即采纳 | 循环≤{max_rounds}轮 | 保底≥{min_ops}算子")
    print(f"{'=' * 60}\n")

    selected: Optional[Dict[str, Any]] = None      # 采纳的组合 {"enabled","composite","score"}
    round_num = len(probed_rounds)

    while round_num < max_rounds and selected is None:
        round_num += 1
        trial_enabled = _pick_random_subset(v3_ops, rng, 1, 3)
        print(f"\n[第 {round_num}/{max_rounds} 轮] 随机只开 {len(trial_enabled)} 个算子: {trial_enabled}")

        # 步骤2：只开这几个算子，测性能
        ok, out_tp, total_tp = measure_config(
            trial_enabled, service_cmd, wait_script, benchmark_script,
            output_dir, f"round{round_num}", port, model_name, max_timeout
        )
        rec: Dict[str, Any] = {
            "round": round_num,
            "enabled": list(trial_enabled),
            "service_ok": ok,
        }
        if not ok:
            print(f"    ✗ 服务无法启动 → 本轮作废，回步骤2重选")
            rec["outcome"] = "service_fail"
            probed_rounds.append(rec)
            state["probed_rounds"] = probed_rounds
            save_json(state, state_path)
            continue

        trial_composite = composite_throughput(out_tp, total_tp)
        gain_pct = (trial_composite - perf_baseline) / perf_baseline * 100 if perf_baseline > 0 else 0.0
        rec["composite"] = round(trial_composite, 2)
        rec["gain_pct"] = round(gain_pct, 2)
        improved = trial_composite > perf_baseline
        print(f"    吞吐 {trial_composite:.1f} tok/s (相对优化基线 {gain_pct:+.2f}%) → "
              f"{'性能提升，进精度校验' if improved else '未超基线，回步骤2重选'}")

        if not improved:
            rec["outcome"] = "no_gain"
            probed_rounds.append(rec)
            state["probed_rounds"] = probed_rounds
            save_json(state, state_path)
            subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
            time.sleep(5)
            continue

        # 步骤3：性能有提升 → 测精度（对比 NV 基线）
        if accuracy_baseline <= 0:
            # 缺精度基线无法校验 → 保守起见不采纳减算子组合（避免破坏精度硬闸门），
            # 记录后继续；最终会回退到精度已合格的 start_ops。
            print(f"    ⚠ 缺精度基线，无法校验减算子后精度 → 本轮不采纳（守精度红线）")
            rec["outcome"] = "no_accuracy_baseline"
            probed_rounds.append(rec)
            state["probed_rounds"] = probed_rounds
            save_json(state, state_path)
            subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
            time.sleep(5)
            continue

        print(f"    ▶ 精度校验（对比 NV 基线 {accuracy_baseline:.1f}）...")
        guard_ok, gscore = run_accuracy_guard(
            gpqa_script, gpqa_config,
            os.path.join(output_dir, f"gpqa_v4_round{round_num}.json"),
            accuracy_baseline, accuracy_guard
        )
        rec["score"] = round(gscore, 2)
        rec["accuracy_ok"] = guard_ok
        subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
        time.sleep(5)
        if guard_ok:
            print(f"    ✓ 精度达标 (score={gscore:.1f}%) → 采纳该组合为 V4")
            rec["outcome"] = "accepted"
            selected = {"enabled": list(trial_enabled), "composite": trial_composite, "score": gscore}
        else:
            print(f"    ✗ 精度不达标 (score={gscore:.1f}%) → 回步骤2重选")
            rec["outcome"] = "accuracy_fail"
        probed_rounds.append(rec)
        state["probed_rounds"] = probed_rounds
        save_json(state, state_path)

    # 收尾判定
    fell_back = selected is None
    if fell_back:
        # 2 轮拿不到 → 回退到起点（精度已合格）
        print(f"\n  ⤷ {max_rounds} 轮内未拿到'性能提升+精度达标' → 回退到起点（精度已合格版）: "
              f"{len(start_ops)} 个算子")
        final_ops = list(start_ops)
    else:
        final_ops = list(selected["enabled"])

    if len(final_ops) < min_ops:
        # 理论不触发（随机下限即 1、起点非空），保险：起点兜底
        final_ops = list(start_ops) if start_ops else list(v3_ops)

    # 最终配置固化 + 收尾评测
    subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
    time.sleep(5)
    write_control_file(final_ops)
    oplist_path = os.path.join(output_dir, "v4_oplist.txt")
    with open(oplist_path, 'w') as f:
        for op in sorted(final_ops):
            f.write(f"{op}\n")

    print(f"\n{'=' * 60}")
    print(f"  V4 精简完成 — 运行最终性能 + 精度终检（最终配置 {len(final_ops)} 个算子）")
    print(f"{'=' * 60}")
    final_out, final_total = 0.0, 0.0
    final_composite = 0.0
    final_score = 0.0
    accuracy_ok: Optional[bool] = None
    accuracy_rel_drop_pct: Optional[float] = None
    ok = restart_and_wait(service_cmd, wait_script, port=port,
                          model_name=model_name, max_timeout=max_timeout)
    if ok:
        final_out, final_total = run_benchmark(benchmark_script, output_dir, "v4_performance")
        final_composite = composite_throughput(final_out, final_total)
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
                  "→ 精度状态标记为未验证，编排层需以 NV 基线兜底后重判")
    subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)

    # 达标基准：只要相对优化基线有性能提升即可（不追求超越 V3 绝对值）
    beats_baseline = final_composite > perf_baseline
    kept_at_least_one = len(final_ops) >= 1

    state["completed"] = True
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["current_enabled_ops"] = final_ops
    state["fell_back_to_start"] = fell_back
    state["final_composite"] = final_composite
    state["v4_score"] = final_score
    state["accuracy_ok"] = accuracy_ok
    save_json(state, state_path)

    # V4 成立准则：精度终检达标（accuracy_ok is not False）+ 至少保留 1 个算子。
    #   - 采纳了随机组合（未回退）：额外要求性能相对优化基线有提升（beats_baseline）。
    #   - 回退到起点（fell_back）：起点精度已合格、性能等价基线，V4 等价起点仍成立
    #     （只要精度达标 + ≥1 算子），性能提升非硬要求（本轮没找到更优组合而已）。
    accuracy_verified = accuracy_ok is not None
    if fell_back:
        success = kept_at_least_one and (accuracy_ok is not False)
    else:
        success = beats_baseline and kept_at_least_one and (accuracy_ok is not False)

    result = {
        "success": success,
        "fell_back_to_start": fell_back,
        "kept_ops": sorted(final_ops),
        "final_enabled_count": len(final_ops),
        "baseline_source": baseline_source,
        "perf_baseline": round(perf_baseline, 2),
        "v1_composite": round(v1_composite, 2),   # 仅报告参考
        "v3_composite": round(v3_composite, 2),   # 仅报告参考
        "v4_composite": round(final_composite, 2),
        "v4_ratio_v1_pct": round((final_composite / v1_composite) * 100, 2) if v1_composite > 0 else None,
        "beats_baseline": beats_baseline,
        "beats_v3": (final_composite >= v3_composite) if v3_composite > 0 else None,  # 仅报告参考
        "kept_at_least_one": kept_at_least_one,
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
    if not success:
        reasons = []
        if accuracy_ok is False:
            reasons.append("精度终检不达标（精度是版本成立前提）")
        if not kept_at_least_one:
            reasons.append("未保留任何算子（V4 要求至少 1 个）")
        if not fell_back and not beats_baseline:
            reasons.append(f"性能未超优化基线（V4={final_composite:.1f} ≤ 基线={perf_baseline:.1f} tok/s）")
        result["reason"] = "V4 不成立：" + "；".join(reasons)

    print(f"\n{'#' * 60}")
    print(f"# V4 算子精简结果 (Flag-express)")
    print(f"#   V3 算子池: {len(v3_ops)} 个 | 起点(回退版): {len(start_ops)} 个")
    print(f"#   最终配置: {len(final_ops)} 个算子 {'(回退到起点)' if fell_back else '(采纳随机组合)'}")
    print(f"#   V4 综合吞吐: {final_composite:.1f} tok/s (优化基线={perf_baseline:.1f}) | 超过基线: {beats_baseline}")
    _v1r = f"{final_composite / v1_composite * 100:.1f}%" if v1_composite > 0 else "N/A"
    print(f"#   相对 V1 性能比（仅参考）: {_v1r}")
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
    parser.add_argument("--accuracy-guard", type=float, default=DEFAULT_ACCURACY_GUARD,
                        help=f"精度护栏：相对退化上限 (%%, 默认 {DEFAULT_ACCURACY_GUARD})")
    parser.add_argument("--accuracy-baseline", type=float, default=0.0,
                        help="精度护栏基线分数 (NV 基线)。V4 精度合格是版本成立前提，"
                             "应始终传入；0=无基线，精度状态标记为未验证（accuracy_verified=false），"
                             "编排层须以 NV 基线兜底后重判，不得直接判 V4 达标")
    parser.add_argument("--v2-path", choices=["2.1", "2.2"], default="2.1",
                        help="V2 路径：2.1=有非plugin V2对照(起点=V2调优后∩V3, 基线=交集实测)；"
                             "2.2=无V1/无非plugin V2(起点=V3, 基线=V2首测×1.05)")
    parser.add_argument("--v2-final-ops", default="",
                        help="V2.1 调优后算子列表（逗号分隔）。用于与 V3 求交集作为起点/回退版。"
                             "空则回退到 V3 全集作起点。")
    parser.add_argument("--v2-first-perf", default="",
                        help="V2.2 路径：V2 首测性能结果 JSON，优化基线=该吞吐×1.05")
    parser.add_argument("--max-rounds", type=int, default=2,
                        help="随机选取算子的循环轮次上限（新流程定稿=2）")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机选取算子的随机种子（断点续跑复现同一序列）")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-timeout", type=int, default=1800)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # 加载 /etc/environment 中的 FlagGems 相关变量（docker exec 不继承）
    _load_etc_environment()

    # 完成标记文件：正常跑完（无论成功/失败判定）时写入，供 pipeline 段结束轮询校验。
    # 中断/崩溃则不会写 → pipeline 据此判断"任务未真正完成"，不会静默跳过。
    done_marker = os.path.join(args.output_dir, "v4_reduction.done")
    # 起始先清除可能残留的旧标记（断点续跑时上一轮的），避免误判
    try:
        if os.path.exists(done_marker):
            os.remove(done_marker)
    except OSError:
        pass

    def _write_done(exit_code: int, res: dict):
        try:
            with open(done_marker, "w") as f:
                json.dump({
                    "exit_code": exit_code,
                    "success": res.get("success"),
                    "final_composite": res.get("v4_composite"),
                    "fell_back_to_start": res.get("fell_back_to_start"),
                    "kept_count": res.get("final_enabled_count"),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"⚠ 写完成标记失败（不影响结果）: {e}")

    ctx = load_context(args.context_yaml)
    # v3_ops = V3 全量算子集（context 里的 V3 启用算子），作 V4 随机选取的算子池
    v3_ops = get_enabled_ops(ctx)
    if not v3_ops:
        print("错误: 无法从 context 获取 V3 启用算子列表")
        _write_done(1, {"success": False, "reason": "无法获取 V3 启用算子列表"})
        sys.exit(1)

    v1_out, v1_total = extract_throughput(args.v1_perf)
    v3_out, v3_total = extract_throughput(args.v3_perf)
    v1_composite = composite_throughput(v1_out, v1_total)
    v3_composite = composite_throughput(v3_out, v3_total)

    # ---- 起点(回退版) + 优化基线，按 V2 路径确定 ----
    if args.v2_path == "2.1":
        # V2.1：起点 = V2.1 调优后算子列表 ∩ V3；优化基线 = 起点实测吞吐（下方 run_reduction 内不测，
        # 这里直接测一次起点性能作基线；起点精度已合格，作回退版）。
        v2_final = [o.strip() for o in args.v2_final_ops.split(",") if o.strip()]
        start_ops = [o for o in v3_ops if o in set(v2_final)] if v2_final else list(v3_ops)
        if not start_ops:
            start_ops = list(v3_ops)   # 交集为空 → 退回 V3 全集作起点
        baseline_source = "intersection"
        port = args.port or ctx.get('service', {}).get('port', 8000)
        model_name = args.model_name or ctx.get('service', {}).get('model_id', '')
        print(f"[V2.1 路径] 起点交集 = {len(start_ops)} 个算子，测起点性能作优化基线...")
        ok0, o0, t0 = measure_config(start_ops, args.service_startup_cmd, args.wait_script,
                                     args.benchmark_script, args.output_dir, "start_intersection",
                                     port, model_name, args.max_timeout)
        subprocess.run("pkill -f 'vllm' 2>/dev/null", shell=True, capture_output=True)
        time.sleep(5)
        if not ok0:
            print("错误: 起点交集配置服务无法启动，V4 无法确立优化基线")
            _write_done(1, {"success": False, "reason": "起点交集服务启动失败"})
            sys.exit(1)
        perf_baseline = composite_throughput(o0, t0)
    else:
        # V2.2 / 无 V1：起点 = V3 全集；优化基线 = V2 首测 × 1.05
        start_ops = list(v3_ops)
        baseline_source = "v2x1.05"
        v2_out, v2_total = extract_throughput(args.v2_first_perf) if args.v2_first_perf else (0.0, 0.0)
        v2_composite = composite_throughput(v2_out, v2_total)
        perf_baseline = v2_composite * 1.05
        port = args.port or ctx.get('service', {}).get('port', 8000)
        model_name = args.model_name or ctx.get('service', {}).get('model_id', '')
        print(f"[V2.2/无V1 路径] 起点 = V3 全集 {len(start_ops)} 个；优化基线 = V2首测 {v2_composite:.1f} ×1.05 = {perf_baseline:.1f} tok/s")

    result = run_reduction(
        v3_ops=v3_ops,
        start_ops=start_ops,
        perf_baseline=perf_baseline,
        v1_composite=v1_composite,
        v3_composite=v3_composite,
        baseline_source=baseline_source,
        service_cmd=args.service_startup_cmd,
        state_path=args.state_path,
        output_dir=args.output_dir,
        wait_script=args.wait_script,
        benchmark_script=args.benchmark_script,
        gpqa_script=args.gpqa_script,
        gpqa_config=args.gpqa_config,
        accuracy_guard=args.accuracy_guard,
        accuracy_baseline=args.accuracy_baseline,
        max_rounds=args.max_rounds,
        seed=args.seed,
        port=port,
        model_name=model_name,
        max_timeout=args.max_timeout,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    exit_code = 0 if result.get("success") else 1
    _write_done(exit_code, result)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
