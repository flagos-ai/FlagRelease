#!/usr/bin/env python3
"""
精度对比工具 — GPQA Diamond 精度达标判定

支持两种基线模式：
  1. 本地 V1 基线（向后兼容）：--v1 <json> --v2 <json>
     判据：rel_drop = (v1_score - v2_score) / v1_score <= threshold
     即当前精度相对 V1 的退化不超过 threshold（默认 5%，相对口径）

  2. NV 参考基线（新流程默认）：--v2 <json> --nv-baseline <模型名>
     判据：(v2_score - nv_score) / nv_score >= -tolerance
     即当前精度相对 NV 的退化不超过 tolerance（默认 5%）
     NV 分数从 shared/nv_baseline.yaml 查表获得

  两种模式均为「相对退化」口径，阈值单位统一为比例（0.05 = 5%）。

Usage:
    # 本地 V1 基线（旧）
    python accuracy_compare.py --v1 results/gpqa_native.json --v2 results/gpqa_flagos.json

    # NV 基线（新）
    python accuracy_compare.py --v2 results/gpqa_flagos.json --nv-baseline Qwen3-8B --json
    python accuracy_compare.py --v2 results/gpqa_flagos.json --nv-baseline Qwen3-8B \
        --nv-baseline-file /flagos-workspace/shared/nv_baseline.yaml --output results/accuracy_compare.json

退出码: 0=达标, 1=不达标, 2=参数/文件错误, 3=缺 NV 基线（需编排层兜底）,
        4=判负但落在小样本噪声区（绝对差异≤2题，疑似评测方差假阳性，需复测/复核）
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_THRESHOLD = 0.05       # 本地 V1 基线模式：相对退化容差（5%，与 NV 模式统一）
DEFAULT_NV_TOLERANCE = 0.05    # NV 基线模式：相对退化容差（5%）

# 缺 NV 基线的专用退出码 / 信号
EXIT_MISSING_NV = 3
# 小样本噪声区专用退出码：rel_drop 超阈值，但绝对差异落在评测方差范围内 → 需复核/复测而非直接判负
EXIT_NOISE_ZONE = 4

# 小样本噪声防护：GPQA 快速评测题数少（如 50 题，每题 2%），低分基线上 1-2 题抖动
# 就会把 rel_drop 放大到超 5% 红线，造成假阳性判负（历史事故：granite-4.0-micro
# V3 28% vs NV 30% 仅差 1 题却被判"框架不适配"）。当 rel_drop 超阈值但绝对差异
# 落在 NOISE_ABS_QUESTIONS 题以内时，标记为 noise_zone（需复核/复测），不直接判负。
NOISE_ABS_QUESTIONS = 2.0      # 绝对差异 ≤ 该题数视为统计噪声（默认 2 题；1 题过于苛刻）
NOISE_MAX_TOTAL = 100          # 仅对题数 ≤ 该值的小样本评测启用噪声防护（大样本方差已足够小）


def _noise_zone_check(current_score: Optional[float], baseline_score: Optional[float],
                      total_questions: Optional[int], aligned: bool) -> Dict[str, Any]:
    """判定是否落在小样本噪声区。

    仅在「按 rel_drop 判为不达标(aligned=False)」时才有意义：若绝对差异 ≤ NOISE_ABS_QUESTIONS
    题（且为小样本评测），则该判负很可能是评测方差假阳性，应复核/复测而非直接判负。
    返回 {noise_zone: bool, ...诊断字段}。
    """
    info: Dict[str, Any] = {"noise_zone": False}
    if aligned or current_score is None or baseline_score is None:
        return info
    if not total_questions or total_questions <= 0 or total_questions > NOISE_MAX_TOTAL:
        return info
    per_q = 100.0 / total_questions           # 每题精度粒度（%）
    abs_diff = abs(baseline_score - current_score)
    diff_questions = abs_diff / per_q         # 折合差几题
    if diff_questions <= NOISE_ABS_QUESTIONS + 1e-9:
        info["noise_zone"] = True
        info["noise_detail"] = (
            f"绝对差异 {abs_diff:.2f}% = {diff_questions:.2f} 题 "
            f"(每题 {per_q:.2f}%, 共 {total_questions} 题), "
            f"≤ {NOISE_ABS_QUESTIONS:.0f} 题噪声阈值 → 疑似小样本方差假阳性，建议复测/复核"
        )
        info["diff_questions"] = round(diff_questions, 2)
        info["total_questions"] = total_questions
    return info


def load_result(path: str) -> Dict[str, Any]:
    """加载 gpqa_result.json"""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: 文件不存在: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON 解析失败 ({path}): {e}", file=sys.stderr)
        sys.exit(2)


def extract_score(data: Dict[str, Any]) -> Optional[float]:
    """从评测结果中提取 score（百分比）"""
    score = data.get("score")
    if score is not None:
        return float(score)
    return None


# ==================== NV 基线查表 =============
    v1_data = load_result(v1_path)
    v2_data = load_result(v2_path)

    v1_score = extract_score(v1_data)
    v2_score = extract_score(v2_data)

    result = {
        "baseline_mode": "local_v1",
        "v1": {
            "path": v1_path,
            "model": v1_data.get("model", "unknown"),
            "score": v1_score,
            "mode": v1_data.get("mode", "unknown"),
        },
        "v2": {
            "path": v2_path,
            "model": v2_data.get("model", "unknown"),
            "score": v2_score,
            "mode": v2_data.get("mode", "unknown"),
        },
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
    }

    if v1_score is None or v2_score is None:
        result["aligned"] = False
        result["diff"] = None
        result["message"] = "分数缺失，无法对比"
        if v1_score is None:
            result["message"] = f"V1 分数缺失 ({v1_path})"
        if v2_score is None:
            result["message"] = f"V2 分数缺失 ({v2_path})"
        return result

    if v1_score <= 0:
        result["aligned"] = False
        result["diff"] = None
        result["message"] = f"V1 分数非法（<=0: {v1_score}），无法计算相对退化"
        return result

    drop = v1_score - v2_score
    rel_drop = drop / v1_score
    diff = round(abs(v2_score - v1_score), 2)
    aligned = rel_drop <= threshold

    result["diff"] = diff
    result["drop"] = round(drop, 2)
    result["rel_drop"] = round(rel_drop, 4)
    result["aligned"] = aligned
    result["v2_vs_v1"] = round(v2_score - v1_score, 2)

    # 小样本噪声防护：判负时检查绝对差异是否落在评测方差内（题数取 V2 结果）
    noise = _noise_zone_check(v2_score, v1_score, v2_data.get("total_questions"), aligned)
    result.update(noise)

    if aligned:
        result["message"] = (
            f"精度达标: V1={v1_score:.2f}%, V2={v2_score:.2f}%, 相对退化={rel_drop*100:.2f}% (阈值 {threshold*100:.0f}%)"
        )
    elif noise.get("noise_zone"):
        result["message"] = (
            f"精度落在噪声区(需复核/复测): V1={v1_score:.2f}%, V2={v2_score:.2f}%, "
            f"相对退化={rel_drop*100:.2f}% > 阈值 {threshold*100:.0f}%，但 {noise['noise_detail']}"
        )
    else:
        result["message"] = (
            f"精度不达标: V1={v1_score:.2f}%, V2={v2_score:.2f}%, 相对退化={rel_drop*100:.2f}% > 阈值 {threshold*100:.0f}%"
        )
    return result


def compare_nv(v2_path: str, model_name: str, metric: str,
               baseline_file: Optional[str], tolerance_override: Optional[float]) -> Dict[str, Any]:
    """NV 参考基线对比（新流程）。

    判据：rel_drop = (nv_score - v2_score) / nv_score <= tolerance
    等价于 (v2_score - nv_score) / nv_score >= -tolerance
    """
    v2_data = load_result(v2_path)
    v2_score = extract_score(v2_data)

    nv_score, tol_from_table, source = lookup_nv_score(model_name, metric, baseline_file)
    tolerance = tolerance_override if tolerance_override is not None else (
        tol_from_table if tol_from_table is not None else DEFAULT_NV_TOLERANCE
    )

    result = {
        "baseline_mode": "nv_reference",
        "model": model_name,
        "metric": metric,
        "nv": {
            "score": nv_score,
            "source": source,
        },
        "current": {
            "path": v2_path,
            "model": v2_data.get("model", "unknown"),
            "score": v2_score,
            "mode": v2_data.get("mode", "unknown"),
        },
        "tolerance": tolerance,
        "timestamp": datetime.now().isoformat(),
    }

    # 缺 NV 基线 → 专用信号，让编排层决定兜底
    if nv_score is None:
        result["aligned"] = None
        result["missing_nv"] = True
        result["message"] = f"缺 NV 基线: {source}"
        return result

    if v2_score is None:
        result["aligned"] = False
        result["missing_nv"] = False
        result["message"] = f"当前精度分数缺失 ({v2_path})"
        return result

    rel_drop = (nv_score - v2_score) / nv_score
    aligned = rel_drop <= tolerance

    result["missing_nv"] = False
    result["rel_drop"] = round(rel_drop, 4)
    result["rel_drop_pct"] = round(rel_drop * 100, 2)
    result["abs_diff"] = round(v2_score - nv_score, 2)
    result["aligned"] = aligned

    # 小样本噪声防护：判负时检查绝对差异是否落在评测方差内
    noise = _noise_zone_check(v2_score, nv_score, v2_data.get("total_questions"), aligned)
    result.update(noise)

    if aligned:
        result["message"] = (
            f"精度达标: 当前={v2_score:.2f}%, NV={nv_score:.2f}%, "
            f"相对退化={rel_drop * 100:.2f}% (容差 {tolerance * 100:.1f}%)"
        )
    elif noise.get("noise_zone"):
        result["message"] = (
            f"精度落在噪声区(需复核/复测): 当前={v2_score:.2f}%, NV={nv_score:.2f}%, "
            f"相对退化={rel_drop * 100:.2f}% > 容差 {tolerance * 100:.1f}%，但 {noise['noise_detail']}"
        )
    else:
        result["message"] = (
            f"精度不达标: 当前={v2_score:.2f}%, NV={nv_score:.2f}%, "
            f"相对退化={rel_drop * 100:.2f}% > 容差 {tolerance * 100:.1f}%"
        )
    return result


# ==================== 输出 =============
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--output", help="结果输出文件路径（JSON）")
    args = parser.parse_args()

    # 模式选择：--nv-baseline 优先（新流程默认）
    if args.nv_baseline:
        result = compare_nv(args.v2, args.nv_baseline, args.metric,
                            args.nv_baseline_file, args.nv_tolerance)
    else:
        if not args.v1:
            print("ERROR: 未指定 --nv-baseline 时必须提供 --v1（本地 V1 基线模式）",
                  file=sys.stderr)
            sys.exit(2)
        result = compare_v1(args.v1, args.v2, args.threshold)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_human(result)

    # 退出码：缺 NV 基线用专用码 3，让编排层区分"不达标"和"无法判定"
    if result.get("missing_nv"):
        sys.exit(EXIT_MISSING_NV)
    if result.get("aligned"):
        sys.exit(0)
    # 判负但落在小样本噪声区 → 专用码 4，让编排层触发复测/复核而非直接判负
    if result.get("noise_zone"):
        sys.exit(EXIT_NOISE_ZONE)
    sys.exit(1)


if __name__ == "__main__":
    main()
