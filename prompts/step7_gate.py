#!/usr/bin/env python3
"""
step7_gate.py — 步骤7（性能算子调优）强制闸门判定（编排层专用）

设计动机：
  prompt 层的"必须执行步骤7"不可靠——agent 可以用无数据支撑的臆断
  （如"关算子不可能提够性能"）跳过 operator_search，并把 workflow.performance_ok
  / ledger 步骤7 状态写成任意值绕过原有补跑检查（run_pipeline.sh 旧逻辑信任这两个
  agent 可写的字段，且把 ledger 'skipped' 当成已完成）。

  本闸门**完全不信任 agent 写的判据**，只看两类 agent 无法用嘴伪造的事实：
    1. 达标率：编排层自算 min-ratio（实测吞吐 JSON ÷ 基线 JSON），不读 performance_ok
    2. 是否真跑过 operator_search：看 operator_config.json 的运行痕迹
       （search_log / total_rounds / elimination_state），agent 声明"跳过"不产生这些

判定输出（stdout 单行，供 shell case 分派）：
  ok          —— 已达标（min-ratio >= target），步骤7 无需执行
  needed      —— 未达标 且 无真实搜索痕迹 → 必须补跑 operator_search
  done        —— 未达标 但 已有真实搜索痕迹（真跑过仍没救回来）→ 尊重实测结果，放行
  no_data     —— 缺基线或实测结果，无法判定（交由上层按缺数据兜底，不误判为 ok）

达标率算法与 operator_optimizer.compute_min_ratio 语义一致：
  所有 test_case × concurrency × {output, total} 的最小 ratio。

Usage:
  python3 step7_gate.py \
      --baseline   /path/native_performance.json \
      --flagos     /path/flagos_performance.json \
      [--optimized /path/flagos_optimized.json] \
      --state      /path/operator_config.json \
      [--target 0.8]
"""

import argparse
import json
import os
import sys

_OUTPUT_KEY = "Output token throughput (tok/s)"
_TOTAL_KEY = "Total token throughput (tok/s)"


def _load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    # 兼容 {"results": {...}} 包装
    if isinstance(data, dict) and isinstance(data.get("results"), dict):
        return data["results"]
    return data


def _iter_points(perf):
    """产出 (test_case, concurrency, output_tp, total_tp)，跳过 _meta/error 点。"""
    if not isinstance(perf, dict):
        return
    for tc_name, tc in perf.items():
        if tc_name.startswith("_") or not isinstance(tc, dict):
            continue
        for conc, metrics in tc.items():
            if conc.startswith("_") or not isinstance(metrics, dict):
                continue
            if "error" in metrics:
                continue
            out_tp = metrics.get(_OUTPUT_KEY, 0) or 0
            tot_tp = metrics.get(_TOTAL_KEY, 0) or 0
            yield tc_name, conc, out_tp, tot_tp


def compute_min_ratio(flagos, baseline):
    """所有 test_case×concurrency×{output,total} 的最小 ratio；无有效点返回 None。"""
    base_map = {}
    for tc, conc, out_tp, tot_tp in _iter_points(baseline):
        base_map[(tc, conc)] = (out_tp, tot_tp)

    ratios = []
    for tc, conc, out_tp, tot_tp in _iter_points(flagos):
        base = base_map.get((tc, conc))
        if not base:
            continue
        base_out, base_tot = base
        if base_out > 0 and out_tp > 0:
            ratios.append(out_tp / base_out)
        if base_tot > 0 and tot_tp > 0:
            ratios.append(tot_tp / base_tot)

    return min(ratios) if ratios else None


def has_real_search(state_path):
    """operator_config.json 是否含 operator_search.py run 真实运行痕迹。

    agent 声明"跳过"不会写这些字段；只有 run_full_search 真跑过才会产生。
    """
    state = _load(state_path)
    if not isinstance(state, dict):
        return False
    # 痕迹1：search_log 非空（每轮 append 一条）
    log = state.get("search_log")
    if isinstance(log, list) and len(log) > 0:
        return True
    # 痕迹2：elimination_state 存在且已推进过（current_idx > 0 或有 cumulative_disabled）
    es = state.get("elimination_state")
    if isinstance(es, dict):
        if es.get("current_idx", 0) > 0:
            return True
        if es.get("cumulative_disabled"):
            return True
    # 痕迹3：current_step > 0（搜索循环推进过）
    if isinstance(state.get("current_step"), int) and state["current_step"] > 0:
        return True
    # 痕迹4：已产生禁用算子（搜索的实质结果）
    if state.get("disabled_ops"):
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="步骤7 强制闸门判定")
    ap.add_argument("--baseline", required=True, help="基线 JSON（native_performance.json，可为合成基线）")
    ap.add_argument("--flagos", required=True, help="V2 初始实测 JSON（flagos_performance.json）")
    ap.add_argument("--optimized", default="", help="调优后 JSON（flagos_optimized.json，存在则优先用它算达标）")
    ap.add_argument("--state", default="", help="operator_config.json（搜索状态/痕迹）")
    ap.add_argument("--target", type=float, default=0.8, help="达标 ratio 阈值（默认 0.8）")
    args = ap.parse_args()

    baseline = _load(args.baseline)
    # 优先用调优后结果判达标（若已存在真实调优产出）；否则用 V2 初始结果
    optimized = _load(args.optimized) if args.optimized else None
    flagos = optimized if optimized is not None else _load(args.flagos)

    if baseline is None or flagos is None:
        print("no_data")
        return

    ratio = compute_min_ratio(flagos, baseline)
    if ratio is None:
        print("no_data")
        return

    if ratio >= args.target:
        print("ok")
        return

    # 未达标：区分"没跑过搜索"（必须补跑）与"真跑过仍没救回来"（尊重实测放行）
    if has_real_search(args.state):
        print("done")
    else:
        print("needed")


if __name__ == "__main__":
    main()
