#!/usr/bin/env python3
"""
性能对比工具

对比多个 benchmark JSON 结果文件，生成 performance_compare.csv 和摘要报告。
逐并发级别对比，确保每个并发级别都达标。

Usage:
    python performance_compare.py --native results/native_performance.json \
                                   --flagos-initial results/flagos_initial.json
    python performance_compare.py --native results/native_performance.json \
                                   --flagos-initial results/flagos_initial.json \
                                   --flagos-optimized results/flagos_optimized.json
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_benchmark(path: str) -> Dict[str, Any]:
    """加载 benchmark JSON 文件"""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: 文件不存在: {path}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def concurrency_sort_key(key: str) -> int:
    """'64' → 64, 'concurrency_64' → 64, 'max' → 999999"""
    if key.isdigit():
        return int(key)
    m = re.search(r'(\d+)$', key)
    if m:
        return int(m.group(1))
    if 'max' in key.lower():
        return 999999
    return 0


def get_results_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """兼容新旧两种 JSON 格式：旧格式有 results 包装，新格式直接是扁平结构"""
    if "results" in data and isinstance(data["results"], dict):
        return data["results"]
    # 新格式：文件本身就是 {tc_name: {concurrency: metrics}}
    return data



def extract_all_concurrency_throughputs(tc_results: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """
    从测试用例结果中提取每个并发级别的吞吐量。

    Returns:
        {"1": (output_tp, total_tp), "4": (output_tp, total_tp), ...}
    """
    result = {}
    for key, metrics in tc_results.items():
        if key.startswith("_"):
            continue
        if not isinstance(metrics, dict) or "error" in metrics:
            continue

        output_tp = metrics.get('Output token throughput (tok/s)', 0) or 0
        total_tp = metrics.get('Total token throughput (tok/s)', 0) or 0
        result[key] = (output_tp, total_tp)

    return result


def compare_results(benchmarks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    逐并发级别对比多个 benchmark 结果。

    每个 test_case × 每个 concurrency_level 生成一行。

    Args:
        benchmarks: {"native": data, "flagos_initial": data, ...}

    Returns:
        对比行列表。
    """
    # 收集所有测试用例名称
    all_test_cases = set()
    for data in benchmarks.values():
        results = get_results_data(data)
        all_test_cases.update(results.keys())

    rows = []
    for tc in sorted(all_test_cases):
        # 从所有 benchmark 中收集该 test_case 的并发 key 并集
        all_conc_keys = set()
        bm_conc_data = {}  # {bm_name: {conc_key: (output_tp, total_tp)}}

        for bm_name, data in benchmarks.items():
            tc_results = get_results_data(data).get(tc, {})
            conc_data = extract_all_concurrency_throughputs(tc_results)
            bm_conc_data[bm_name] = conc_data
            all_conc_keys.update(conc_data.keys())

        if not all_conc_keys:
            continue

        # 按并发数排序
        sorted_conc_keys = sorted(all_conc_keys, key=concurrency_sort_key)

        # 生成每个并发级别的行
        for conc_key in sorted_conc_keys:
            row = {
                "test_case": tc,
                "concurrency": conc_key,
                "concurrency_num": concurrency_sort_key(conc_key),
            }

            native_output_tp = 0.0
            native_total_tp = 0.0

            for bm_name in benchmarks:
                conc_data = bm_conc_data.get(bm_name, {})
                if conc_key in conc_data:
                    output_tp, total_tp = conc_data[conc_key]
                    row[f"{bm_name}_output_throughput"] = output_tp
                    row[f"{bm_name}_total_throughput"] = total_tp
                    if bm_name == "native":
                        native_output_tp = output_tp
                        native_total_tp = total_tp
                else:
                    row[f"{bm_name}_output_throughput"] = None
                    row[f"{bm_name}_total_throughput"] = None

            # 计算 ratio（output 和 total 分别计算，取较小值作为综合 ratio）
            for bm_name in benchmarks:
                if bm_name == "native":
                    continue
                flagos_output_tp = row.get(f"{bm_name}_output_throughput")
                flagos_total_tp = row.get(f"{bm_name}_total_throughput")

                if native_output_tp > 0 and flagos_output_tp is not None:
                    output_ratio = flagos_output_tp / native_output_tp
                    row[f"{bm_name}_output_ratio"] = output_ratio
                else:
                    row[f"{bm_name}_output_ratio"] = None

                if native_total_tp > 0 and flagos_total_tp is not None:
                    total_ratio = flagos_total_tp / native_total_tp
                    row[f"{bm_name}_total_ratio"] = total_ratio
                else:
                    row[f"{bm_name}_total_ratio"] = None

                # 综合 ratio = min(output_ratio, total_ratio)，兼容下游消费
                or_val = row[f"{bm_name}_output_ratio"]
                tr_val = row[f"{bm_name}_total_ratio"]
                if or_val is not None and tr_val is not None:
                    row[f"{bm_name}_ratio"] = min(or_val, tr_val)
                elif or_val is not None:
                    row[f"{bm_name}_ratio"] = or_val
                elif tr_val is not None:
                    row[f"{bm_name}_ratio"] = tr_val
                else:
                    row[f"{bm_name}_ratio"] = None

            rows.append(row)

    return rows


def check_target(rows: List[Dict[str, Any]], benchmark_names: List[str],
                 target_ratio: float = 0.8) -> Dict[str, bool]:
    """检查各 flagos 版本是否达标（所有并发级别的 output_ratio 和 total_ratio 都必须达标）"""
    result = {}
    for name in benchmark_names:
        if name == "native":
            continue
        # 收集 output_ratio 和 total_ratio，综合判定
        all_ratios = []
        for row in rows:
            for metric in ("output_ratio", "total_ratio"):
                val = row.get(f"{name}_{metric}")
                if val is not None:
                    all_ratios.append(val)
        if all_ratios:
            min_ratio = min(all_ratios)
            result[name] = min_ratio >= target_ratio
        else:
            result[name] = False
    return result


def shorten_test_case(name: str) -> str:
    """将测试用例名转为简写格式: 1k_input_1k_output → 1k→1k"""
    m = re.match(r'(\d+k?)_input_(\d+k?)_output', name)
    if m:
        return f"{m.group(1)}\u2192{m.group(2)}"
    return name


def print_markdown_table(rows: List[Dict[str, Any]], benchmark_names: List[str],
                         target_ratio: float = 0.8):
    """打印标准 markdown 格式的逐并发级别对比表格"""

    display_names = {
        "native": "Native TPS",
        "flagos_initial": "FlagOS Initial TPS",
        "flagos_optimized": "FlagOS Optimized TPS",
        "flagos_full": "Full FlagGems TPS",
    }

    # 构建表头
    headers = ["Test Case", "Concurrency"]
    for name in benchmark_names:
        headers.append(display_names.get(name, name + " TPS"))
        if name != "native":
            headers.append("Ratio")

    # 打印表头
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["-" * max(len(h), 5) for h in headers]) + " |")

    # 数据行
    prev_tc = None
    for row in rows:
        tc = shorten_test_case(row["test_case"])
        conc_num = row["concurrency_num"]

        # 同一用例连续行只在第一行显示名称
        if tc == prev_tc:
            tc_display = ""
        else:
            tc_display = tc
            prev_tc = tc

        cells = [tc_display, str(conc_num)]

        for name in benchmark_names:
            total_tp = row.get(f"{name}_total_throughput")
            if total_tp is not None:
                cells.append(str(int(round(total_tp))))
            else:
                cells.append("")

            if name != "native":
                ratio = row.get(f"{name}_ratio")
                if ratio is not None:
                    cells.append(f"**{ratio*100:.1f}%**")
                else:
                    cells.append("")

        line = "| " + " | ".join(cells) + " |"
        print(line)

    # 打印汇总
    print("")
    for name in benchmark_names:
        if name == "native":
            continue
        all_ratios = []
        for row in rows:
            for metric in ("output_ratio", "total_ratio"):
                val = row.get(f"{name}_{metric}")
                if val is not None:
                    all_ratios.append(val)
        if all_ratios:
            avg_ratio = sum(all_ratios) / len(all_ratios)
            min_ratio = min(all_ratios)
            status = "PASS" if min_ratio >= target_ratio else "FAIL"
            dn = display_names.get(name, name)
            print(f"{dn}: avg_ratio={avg_ratio*100:.1f}%, min_ratio={min_ratio*100:.1f}% [{status}]")


def print_comparison(rows: List[Dict[str, Any]], benchmark_names: List[str],
                     target_ratio: float = 0.8):
    """打印文本格式的逐并发级别对比"""
    print(f"\n{'='*80}")
    print("性能对比摘要（逐并发级别）")
    print(f"{'='*80}")

    # 表头
    header = f"{'Test Case':<20} {'Conc':>6}"
    for name in benchmark_names:
        header += f" {name:>15}"
    for name in benchmark_names:
        if name != "native":
            header += f" {name+'_ratio':>15}"
    print(header)
    print("-" * len(header))

    # 数据行
    prev_tc = None
    for row in rows:
        tc = row['test_case']
        conc_num = row['concurrency_num']

        if tc == prev_tc:
            tc_display = ""
        else:
            tc_display = tc
            prev_tc = tc

        line = f"{tc_display:<20} {conc_num:>6}"
        for name in benchmark_names:
            tp = row.get(f"{name}_output_throughput")
            if tp is not None:
                line += f" {tp:>15.2f}"
            else:
                line += f" {'':>15}"
        for name in benchmark_names:
            if name != "native":
                ratio = row.get(f"{name}_ratio")
                if ratio is not None:
                    line += f" {ratio*100:>14.1f}%"
                else:
                    line += f" {'':>15}"

        print(line)

    # 总体判断
    print(f"\n{'='*80}")
    for name in benchmark_names:
        if name == "native":
            continue
        all_ratios = []
        for row in rows:
            for metric in ("output_ratio", "total_ratio"):
                val = row.get(f"{name}_{metric}")
                if val is not None:
                    all_ratios.append(val)
        if all_ratios:
            avg_ratio = sum(all_ratios) / len(all_ratios)
            min_ratio = min(all_ratios)
            status = "PASS" if min_ratio >= target_ratio else "FAIL"
            print(f"{name}: avg_ratio={avg_ratio*100:.1f}%, min_ratio={min_ratio*100:.1f}% [{status}]")


def save_csv(rows: List[Dict[str, Any]], output_path: str, benchmark_names: List[str]):
    """保存逐并发级别对比结果到 CSV"""
    if not rows:
        print("WARNING: 无数据可保存")
        return

    # 构建列头
    headers = ["test_case", "concurrency", "concurrency_num"]
    for name in benchmark_names:
        headers.append(f"{name}_output_throughput")
        headers.append(f"{name}_total_throughput")
        if name != "native":
            headers.append(f"{name}_output_ratio")
            headers.append(f"{name}_total_ratio")
            headers.append(f"{name}_ratio")

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            formatted_row = {}
            for k, v in row.items():
                if k not in headers:
                    continue
                if v is None:
                    formatted_row[k] = ""
                elif k.endswith("_ratio") and isinstance(v, float):
                    formatted_row[k] = f"{v*100:.1f}%"
                elif isinstance(v, float):
                    formatted_row[k] = f"{v:.2f}"
                elif isinstance(v, bool):
                    formatted_row[k] = str(v)
                else:
                    formatted_row[k] = v
            writer.writerow(formatted_row)

    print(f"CSV 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="性能对比工具")
    parser.add_argument("--native", required=True, help="原生性能结果 JSON 路径")
    parser.add_argument("--flagos-initial", help="FlagOS 初始性能结果 JSON 路径")
    parser.add_argument("--flagos-optimized", help="FlagOS 优化后性能结果 JSON 路径")
    parser.add_argument("--flagos-full", help="FlagOS 全量算子性能结果 JSON 路径")
    parser.add_argument("--output", default="./performance_compare.csv", help="CSV 输出路径")
    parser.add_argument("--target-ratio", type=float, default=0.8, help="性能目标比率 (默认 0.8)")
    parser.add_argument("--format", choices=["text", "markdown"], default="text",
                        help="输出格式: text(默认) 或 markdown")
    args = parser.parse_args()

    # 加载 benchmark 数据
    benchmarks = {}
    benchmark_names = []

    benchmarks["native"] = load_benchmark(args.native)
    benchmark_names.append("native")

    if args.flagos_initial:
        benchmarks["flagos_initial"] = load_benchmark(args.flagos_initial)
        benchmark_names.append("flagos_initial")

    if args.flagos_optimized:
        benchmarks["flagos_optimized"] = load_benchmark(args.flagos_optimized)
        benchmark_names.append("flagos_optimized")

    if args.flagos_full:
        benchmarks["flagos_full"] = load_benchmark(args.flagos_full)
        benchmark_names.append("flagos_full")

    if len(benchmarks) < 2:
        print("ERROR: 至少需要 native + 一个 flagos 结果文件")
        sys.exit(1)

    # 对比
    rows = compare_results(benchmarks)

    # 打印摘要
    if args.format == "markdown":
        print_markdown_table(rows, benchmark_names, target_ratio=args.target_ratio)
    else:
        print_comparison(rows, benchmark_names, target_ratio=args.target_ratio)

    # 保存 CSV
    save_csv(rows, args.output, benchmark_names)

    # 检查是否达标
    target_check = check_target(rows, benchmark_names, args.target_ratio)
    print(f"\n目标检查 (target >= {args.target_ratio*100:.0f}% native):")
    for name, passed in target_check.items():
        status = "PASS" if passed else "NEED OPTIMIZATION"
        print(f"  {name}: {status}")

    # 返回码：任一未达标返回 1
    if any(not passed for passed in target_check.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
