#!/usr/bin/env python3
"""
rollback_overflow.py — 越位执行数据清理工具

当某段 Claude 会话越界执行了不属于本段的步骤（例如段1提前跑了步骤4精度评测），
编排层(run_pipeline.sh)会：
  1. 把越界步骤的 ledger 状态回滚为 pending
  2. 重置 workflow 状态字段
  3. 调用本脚本清理越界步骤 **实际产生的数据文件**

本脚本按 "步骤起点 -> 该步骤及之后产出文件" 的映射，只删除越界部分，
不触碰越界起点之前（已合法完成）的产出。这样断点续跑/正常回滚都不会误删有效产出。

关键约束：
  - 只删明确映射到 "越界起点及之后" 的文件，绝不做无差别 rm。
  - 文件不存在时安全跳过，退出码始终 0（清理是尽力而为，不应阻断主流程）。
  - --preserve-step3-disabled-ops：保留步骤3确定的初始算子集文件
    (ops_control_initial.json / operator_config.json 的 disabled_ops)，
    仅清理越界"起点及之后"新引入的调优/评测/发布产物。

用法：
  python3 rollback_overflow.py --overflow-from 4 --preserve-step3-disabled-ops
  python3 rollback_overflow.py --overflow-from 8
  python3 rollback_overflow.py --overflow-from 9 --results-dir /flagos-workspace/results
"""
import argparse
import json
import os
import sys

# 默认产出目录（容器内）
DEFAULT_RESULTS_DIR = "/flagos-workspace/results"
# 步骤3落盘的算子控制文件（容器内绝对路径，不在 results 下）
FLAGGEMS_CONTROL_FILE = "/root/flaggems_ops_control.json"

# 步骤起点 -> 该步骤(含)之后产生的数据文件（相对 results-dir 的文件名）
# key 为"步骤号起点"：清理 --overflow-from N 时，删除所有 key >= N 的文件集合。
# 说明：
#   步骤4  精度评测产物
#   步骤5  精度调优产物（算子配置/控制文件）
#   步骤6  性能评测产物
#   步骤7  性能调优产物
#   步骤8  发布产物
#   步骤9+ plugin 流程产物
STEP_OUTPUT_MAP = {
    4: [
        "gpqa_native.json",
        "gpqa_flagos.json",
        "gpqa_result.json",
        "accuracy_compare.json",
    ],
    5: [
        # 精度调优：算子配置 + 控制文件。operator_config.json 在 preserve 模式下受保护。
        "operator_config.json",
        "gpqa_flagos_optimized.json",
    ],
    6: [
        "native_performance.json",
        "flagos_initial.json",
        "flagos_performance.json",
    ],
    7: [
        "flagos_optimized.json",
    ],
    8: [
        "release_info.json",
        "report.json",
    ],
    9: [
        # plugin 流程产物
        "operator_config_v3.json",
        "gpqa_plugin.json",
    ],
}

# 步骤3的初始算子集产物：--preserve-step3-disabled-ops 时始终保留。
STEP3_PRESERVE_FILES = [
    "ops_control_initial.json",
    "ops_list.json",
    "op_mapping.json",
    "runtime_ops.json",
]


def _remove_file(path):
    """删除单个文件，返回 True 表示实际删除；不存在或失败均安全返回 False。"""
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"  ✓ 已清理: {path}")
            return True
    except OSError as e:
        print(f"  ⚠ 清理跳过(无法删除): {path} ({e})")
    return False


def _prune_operator_config_preserve_step3(path):
    """
    preserve 模式下不整体删除 operator_config.json，
    而是把它回退为"仅保留步骤3已确定的 disabled_ops"的初始态：
    清空调优过程中新增的搜索状态，保留 disabled_ops 基线。
    找不到/解析失败时安全跳过。
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  ⚠ operator_config.json 解析失败，保留原文件不动: {e}")
        return

    disabled = state.get("disabled_ops", [])
    # 只保留 disabled_ops 基线，丢弃越界调优产生的搜索中间态。
    pruned = {"disabled_ops": disabled, "_meta": {"pruned_by": "rollback_overflow", "preserve_step3": True}}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pruned, f, ensure_ascii=False, indent=2)
        print(f"  ✓ operator_config.json 已回退为步骤3基线(保留 {len(disabled)} 个 disabled_ops)")
    except OSError as e:
        print(f"  ⚠ operator_config.json 回退写入失败,保留原文件: {e}")


def main():
    parser = argparse.ArgumentParser(description="越位执行数据清理工具")
    parser.add_argument("--overflow-from", type=int, required=True,
                        help="越界起点步骤号；清理该步骤及之后产生的数据文件")
    parser.add_argument("--preserve-step3-disabled-ops", action="store_true",
                        help="保留步骤3确定的初始算子集；operator_config.json 仅回退为 disabled_ops 基线而非删除")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help=f"产出文件目录，默认 {DEFAULT_RESULTS_DIR}")
    args = parser.parse_args()

    results_dir = args.results_dir
    start = args.overflow_from
    preserve = args.preserve_step3_disabled_ops

    print(f"[rollback_overflow] 越界起点=步骤{start}, results_dir={results_dir}, "
          f"preserve_step3={preserve}")

    removed = 0
    for step, files in sorted(STEP_OUTPUT_MAP.items()):
        if step < start:
            continue
        for fname in files:
            # preserve 模式：operator_config.json 不删除，改为回退基线
            if preserve and fname == "operator_config.json":
                _prune_operator_config_preserve_step3(os.path.join(results_dir, fname))
                continue
            if _remove_file(os.path.join(results_dir, fname)):
                removed += 1

    # 步骤5+ 越界时，控制文件 /root/flaggems_ops_control.json 也属越界产物。
    # 但 preserve 模式下它承载步骤3已确定的启用集，保留不动。
    if start <= 5 and not preserve:
        if _remove_file(FLAGGEMS_CONTROL_FILE):
            removed += 1
    elif start <= 5 and preserve:
        print(f"  ℹ preserve 模式：保留 {FLAGGEMS_CONTROL_FILE}(步骤3算子控制基线)")

    # preserve 模式：始终保留步骤3初始算子集产物（即使 overflow-from<=其对应步骤）
    if preserve:
        for fname in STEP3_PRESERVE_FILES:
            p = os.path.join(results_dir, fname)
            if os.path.isfile(p):
                print(f"  ℹ preserve 模式：保留步骤3产物 {p}")

    print(f"[rollback_overflow] 清理完成，共删除 {removed} 个越界产物文件")
    # 清理是尽力而为，永远以 0 退出，不阻断主流程
    sys.exit(0)


if __name__ == "__main__":
    main()
