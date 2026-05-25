#!/usr/bin/env python3
"""
calc_tp_size.py — 根据模型体量和 GPU 显存自动推算最小 tensor-parallel-size

在容器内运行，读取模型权重大小和 GPU 显存，输出推荐的 TP 值。

Usage:
    python3 calc_tp_size.py --model-path /path/to/model --json
    python3 calc_tp_size.py --model-path /path/to/model

退出码: 0=成功, 1=无法推算, 2=参数错误
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# 权重文件后缀
WEIGHT_EXTENSIONS = (".safetensors", ".bin")
# 排除的非权重 .bin 文件
EXCLUDE_BIN = re.compile(r"^(optimizer|training_args|scheduler)", re.IGNORECASE)
# 显存安全系数：权重大小 × OVERHEAD_FACTOR = 预估所需显存
# 包含 KV cache 基础分配、activation、CUDA context 等
OVERHEAD_FACTOR = 1.2


def get_model_weight_size_gb(model_path):
    """计算模型权重文件总大小（GB）"""
    total_bytes = 0
    try:
        for entry in os.listdir(model_path):
            entry_lower = entry.lower()
            full_path = os.path.join(model_path, entry)
            if not os.path.isfile(full_path):
                continue
            if entry_lower.endswith(".safetensors"):
                if not entry_lower.startswith("training_args"):
                    total_bytes += os.path.getsize(full_path)
            elif entry_lower.endswith(".bin"):
                if not EXCLUDE_BIN.match(entry):
                    total_bytes += os.path.getsize(full_path)
    except (PermissionError, OSError) as e:
        print(f"Warning: 无法读取模型目录 {model_path}: {e}", file=sys.stderr)
    return total_bytes / (1024 ** 3)


def get_gpu_info():
    """获取 GPU 数量和单卡显存（GB）

    优先读取已有检测结果（环境检测阶段生成），fallback 到实时检测。
    """
    # 优先读已有检测结果
    for path in ["/flagos-workspace/results/gpu_info.json", "gpu_info.json"]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                if info.get("count") and info.get("count") > 0:
                    return info
            except (json.JSONDecodeError, Exception):
                pass

    # fallback: 实时检测（搜索多个可能路径）
    script_dir = str(Path(__file__).resolve().parent)
    for search_path in [script_dir, "/flagos-workspace/scripts"]:
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
    try:
        from detect_gpu import detect_gpu
        return detect_gpu()
    except (ImportError, Exception):
        pass

    return None


def next_power_of_2(n):
    """向上取最近的 2 的幂（1, 2, 4, 8, 16...）"""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def calc_tp(model_size_gb, gpu_memory_gb, gpu_count):
    """计算推荐的 tensor-parallel-size

    返回 (tp, reason)
    """
    estimated_required_gb = model_size_gb * OVERHEAD_FACTOR

    if estimated_required_gb <= gpu_memory_gb:
        tp = 1
        reason = (
            f"模型 {model_size_gb:.1f}GB，预估需 {estimated_required_gb:.1f}GB，"
            f"单卡 {gpu_memory_gb}GB 显存充足，推荐 TP=1"
        )
    else:
        raw_tp = math.ceil(estimated_required_gb / gpu_memory_gb)
        tp = next_power_of_2(raw_tp)
        reason = (
            f"模型 {model_size_gb:.1f}GB，预估需 {estimated_required_gb:.1f}GB，"
            f"单卡 {gpu_memory_gb}GB，需至少 {raw_tp} 卡，取 2 的幂 TP={tp}"
        )

    # 不超过 GPU 总数
    if tp > gpu_count:
        tp = gpu_count
        reason += f"（受限于 GPU 总数 {gpu_count}）"

    return tp, reason


def main():
    global OVERHEAD_FACTOR

    parser = argparse.ArgumentParser(description="自动推算 tensor-parallel-size")
    parser.add_argument("--model-path", required=True, help="模型权重目录路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--overhead", type=float, default=OVERHEAD_FACTOR,
                        help=f"显存安全系数 (默认 {OVERHEAD_FACTOR})")
    args = parser.parse_args()

    OVERHEAD_FACTOR = args.overhead

    model_path = args.model_path
    if not os.path.isdir(model_path):
        print(f"Error: 模型路径不存在: {model_path}", file=sys.stderr)
        sys.exit(2)

    # 1. 获取模型大小
    model_size_gb = get_model_weight_size_gb(model_path)
    if model_size_gb < 0.01:
        print(f"Error: 未在 {model_path} 找到有效权重文件", file=sys.stderr)
        sys.exit(1)

    # 2. 获取 GPU 信息
    gpu_info = get_gpu_info()
    if gpu_info is None:
        print("Error: 无法检测 GPU 信息", file=sys.stderr)
        sys.exit(1)

    # 3. 计算推荐 TP
    tp, reason = calc_tp(model_size_gb, gpu_info["memory_gb"], gpu_info["count"])

    estimated_required_gb = round(model_size_gb * OVERHEAD_FACTOR, 1)

    output = {
        "recommended_tp": tp,
        "gpu_count": gpu_info["count"],
        "gpu_name": gpu_info["name"],
        "gpu_memory_gb": gpu_info["memory_gb"],
        "gpu_source": gpu_info["source"],
        "model_size_gb": round(model_size_gb, 1),
        "estimated_required_gb": estimated_required_gb,
        "overhead_factor": OVERHEAD_FACTOR,
        "reason": reason,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"模型权重: {model_size_gb:.1f} GB")
        print(f"预估显存: {estimated_required_gb} GB (×{OVERHEAD_FACTOR})")
        print(f"GPU: {gpu_info['count']}× {gpu_info['name']} ({gpu_info['memory_gb']} GB/卡)")
        print(f"推荐 TP: {tp}")
        print(f"原因: {reason}")

    sys.exit(0)


if __name__ == "__main__":
    main()
