#!/usr/bin/env python3
"""
synthesize_perf_baseline.py — 无 V1 场景的性能基线合成

当 V1 基线性能完全缺失时（分支 B 三选=none 强依赖 flaggems、或 V1 启动/测试失败），
以 V2 使能 flaggems 后首次可正常启动状态的性能测试结果 ×1.5 合成性能基线。

设计要点：
- 输出文件与 benchmark_runner.py 的扁平格式完全一致
  （{tc_name: {concurrency: {metric: value}}, "_meta": {...}}），
  按 native_performance.json 命名落盘 → performance_compare.py /
  operator_optimizer.py init / operator_search.py 全链路零改动即可消费
- 吞吐类指标（tok/s / throughput）×1.5；延迟类指标（ms：TTFT/TPOT/ITL/latency）÷1.5；
  其余字段原样保留
- _meta 写入 baseline_source / synthetic 标记，generate_report.py 据此在报告中
  注明"性能基线为合成值（V2 初始 ×1.5），非实测 V1"
- ×1.5 为全芯片统一标准（用户 2026-07-09 确定），语义=经验推定的 V1/V2全开 性能比，
  80% 判据下等价于要求调优后性能 ≥ V2 初始的 1.2 倍

Usage:
    python synthesize_perf_baseline.py \
        --v2-initial /flagos-workspace/results/v2_initial_performance.json \
        --output /flagos-workspace/results/native_performance.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

FACTOR = 1.5  # 全芯片统一标准

# 指标方向判定（键名子串，大小写不敏感）
_THROUGHPUT_HINTS = ("throughput", "tok/s")
_LATENCY_HINTS = ("(ms)", "ttft", "tpot", "itl", "latency")


def _scale_metric(key: str, value):
    """按指标方向缩放：吞吐 ×FACTOR，延迟 ÷FACTOR，其余原样"""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    k = key.lower()
    if any(h in k for h in _THROUGHPUT_HINTS):
        return round(value * FACTOR, 2)
    if any(h in k for h in _LATENCY_HINTS):
        return round(value / FACTOR, 2)
    return value


def synthesize(v2_initial_path: str, output_path: str) -> dict:
    with open(v2_initial_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    # 兼容旧格式 results 包装
    data = src.get("results") if isinstance(src.get("results"), dict) else src

    out = {}
    scaled_points = 0
    for tc_name, tc_results in data.items():
        if tc_name.startswith("_"):
            continue
        if not isinstance(tc_results, dict):
            out[tc_name] = tc_results
            continue
        out_tc = {}
        for conc, metrics in tc_results.items():
            if conc.startswith("_") or not isinstance(metrics, dict):
                out_tc[conc] = metrics
                continue
            if "error" in metrics:
                out_tc[conc] = metrics
                continue
            out_tc[conc] = {k: _scale_metric(k, v) for k, v in metrics.items()}
            scaled_points += 1
        out[tc_name] = out_tc

    if scaled_points == 0:
        print("ERROR: V2 初始结果中没有任何有效数据点，无法合成基线", file=sys.stderr)
        sys.exit(1)

    out["_meta"] = {
        "说明": "合成性能基线（非实测 V1）：V2 初始性能 ×1.5",
        "格式": "{test_case: {concurrency: {metric: value}}}",
        "关键指标": "Output token throughput (tok/s) 和 Total token throughput (tok/s)",
        "baseline_source": "v2_initial_x1.5",
        "synthetic": True,
        "factor": FACTOR,
        "source_file": os.path.abspath(v2_initial_path),
        "scaling": "吞吐类 ×1.5，延迟类 ÷1.5，其余原样",
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {
        "success": True,
        "output": output_path,
        "scaled_points": scaled_points,
        "factor": FACTOR,
        "baseline_source": "v2_initial_x1.5",
    }


def main():
    parser = argparse.ArgumentParser(description="无 V1 场景：以 V2 初始性能 ×1.5 合成性能基线")
    parser.add_argument("--v2-initial", required=True,
                        help="V2 初始性能测试结果 JSON（benchmark_runner 输出）")
    parser.add_argument("--output", required=True,
                        help="合成基线输出路径（建议 /flagos-workspace/results/native_performance.json）")
    parser.add_argument("--force", action="store_true",
                        help="输出文件已存在时覆盖（默认拒绝，防止覆盖实测 V1 基线）")
    args = parser.parse_args()

    # 安全护栏：绝不覆盖已存在的实测 V1 基线
    if os.path.exists(args.output) and not args.force:
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not existing.get("_meta", {}).get("synthetic"):
                print(f"ERROR: {args.output} 已存在且非合成基线（可能是实测 V1），拒绝覆盖。"
                      f"确需覆盖请加 --force", file=sys.stderr)
                sys.exit(2)
        except (json.JSONDecodeError, OSError):
            print(f"ERROR: {args.output} 已存在但不可解析，拒绝覆盖。确需覆盖请加 --force",
                  file=sys.stderr)
            sys.exit(2)

    result = synthesize(args.v2_initial, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
