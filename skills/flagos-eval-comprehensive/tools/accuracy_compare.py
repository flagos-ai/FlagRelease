#!/usr/bin/env python3
"""
精度对比工具 — V1 vs V2 GPQA Diamond 精度对比与阈值判定

读取两份 gpqa_result.json，计算偏差，判定是否达标（默认阈值 5%）。

Usage:
    python accuracy_compare.py --v1 results/gpqa_native.json --v2 results/gpqa_flagos.json
    python accuracy_compare.py --v1 results/gpqa_native.json --v2 results/gpqa_flagos.json --threshold 3.0 --json
    python accuracy_compare.py --v1 results/gpqa_native.json --v2 results/gpqa_flagos.json --output results/accuracy_compare.json

退出码: 0=达标, 1=不达标, 2=参数/文件错误
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_THRESHOLD = 5.0


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


def compare(v1_path: str, v2_path: str, threshold: float) -> Dict[str, Any]:
    """对比 V1 和 V2 精度结果"""
    v1_data = load_result(v1_path)
    v2_data = load_result(v2_path)

    v1_score = extract_score(v1_data)
    v2_score = extract_score(v2_data)

    result = {
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

    # 分数缺失
    if v1_score is None or v2_score is None:
        result["aligned"] = False
        result["diff"] = None
        result["message"] = "分数缺失，无法对比"
        if v1_score is None:
            result["message"] = f"V1 分数缺失 ({v1_path})"
        if v2_score is None:
            result["message"] = f"V2 分数缺失 ({v2_path})"
        return result

    # 计算精度下降（正值=V2低于V1，仅下降超阈值时不达标）
    drop = v1_score - v2_score
    diff = round(abs(v2_score - v1_score), 2)
    aligned = drop <= threshold

    result["diff"] = diff
    result["drop"] = round(drop, 2)
    result["aligned"] = aligned
    result["v2_vs_v1"] = round(v2_score - v1_score, 2)
    result["message"] = (
        f"精度达标: V1={v1_score:.2f}%, V2={v2_score:.2f}%, 下降={drop:.2f}% (阈值 {threshold}%)"
        if aligned else
        f"精度不达标: V1={v1_score:.2f}%, V2={v2_score:.2f}%, 下降={drop:.2f}% > 阈值 {threshold}%"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="V1 vs V2 精度对比")
    parser.add_argument("--v1", required=True, help="V1 (Native) 评测结果 JSON")
    parser.add_argument("--v2", required=True, help="V2 (FlagGems) 评测结果 JSON")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"偏差阈值百分比（默认 {DEFAULT_THRESHOLD}%%）")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--output", help="结果输出文件路径（JSON）")
    args = parser.parse_args()

    result = compare(args.v1, args.v2, args.threshold)

    # 写文件
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 输出
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print()
        print("=" * 60)
        print("  GPQA Diamond 精度对比")
        print("=" * 60)
        v1 = result["v1"]
        v2 = result["v2"]
        print(f"  V1 (Native):  {v1['score']:.2f}%" if v1["score"] is not None else "  V1 (Native):  N/A")
        print(f"  V2 (FlagOS):  {v2['score']:.2f}%" if v2["score"] is not None else "  V2 (FlagOS):  N/A")
        if result["diff"] is not None:
            print(f"  偏差:         {result['diff']:.2f}%")
            if result.get("drop") is not None and result["drop"] < 0:
                print(f"  方向:         V2 高于 V1（不触发调优）")
            print(f"  阈值:         {args.threshold}%（仅下降超阈值时不达标）")
            status = "✓ 达标" if result["aligned"] else "✗ 不达标"
            print(f"  结论:         {status}")
        else:
            print(f"  结论:         无法对比 — {result['message']}")
        print("=" * 60)

    sys.exit(0 if result.get("aligned") else 1)


if __name__ == "__main__":
    main()
