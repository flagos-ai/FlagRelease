#!/usr/bin/env python3
"""
多厂商 GPU 统一检测

通过 Python API (torch) 或逐个尝试各厂商 CLI 工具自动识别 GPU 厂商和硬件信息。

Usage:
    python detect_gpu.py                                    # 输出 JSON 到 stdout
    python detect_gpu.py --check-free                       # 输出 per-GPU 空闲/占用 JSON
    python detect_gpu.py --check-free --vendor nvidia       # 指定厂商，跳过探测
    python detect_gpu.py --output /path/to/gpu_info.json    # 同时写入文件

作为模块导入:
    from detect_gpu import detect_gpu, check_gpu_free
    info = detect_gpu()
    free_info = check_gpu_free(vendor="nvidia")
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 厂商探测表
# =============================================================================
# (vendor, cli_cmd, query_cmd, visible_devices_env)
# 按市场占有率排序，常见厂商优先匹配

GPU_VENDORS: List[Tuple[str, str, str, str]] = [
    ("nvidia",    "nvidia-smi",   "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", "CUDA_VISIBLE_DEVICES"),
    ("huawei",    "npu-smi",      "npu-smi info",                           "ASCEND_RT_VISIBLE_DEVICES"),
    ("hygon",     "rocm-smi",     "rocm-smi --showmeminfo vram --csv",      "HIP_VISIBLE_DEVICES"),
    ("cambricon", "cnmon",        "cnmon info",                              "MLU_VISIBLE_DEVICES"),
    ("mthreads",  "mthreads-gmi", "mthreads-gmi -q",                        "MUSA_VISIBLE_DEVICES"),
    ("kunlunxin", "xpu_smi",     "xpu_smi",                                 "XPU_VISIBLE_DEVICES"),
    ("tianshu",   "ixsmi",       "ixsmi -q",                                "CUDA_VISIBLE_DEVICES"),
    ("metax",     "mx-smi",      "mx-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits", "CUDA_VISIBLE_DEVICES"),
]

# GPU name → vendor 推断（用于 torch 检测到 GPU 但无法确定厂商时）
VENDOR_KEYWORDS = {
    "nvidia": ["nvidia", "geforce", "tesla", "quadro", "rtx", "a100", "a800", "h100", "h800", "l40", "v100"],
    "huawei": ["ascend", "atlas"],
    "hygon": ["hygon", "dcu", "bw200", "bw3000", "bw100"],
    "cambricon": ["mlu"],
    "mthreads": ["mtt", "musa"],
    "kunlunxin": ["kunlun", "xpu"],
    "metax": ["metax", "c500", "c550", "n100"],
}


def _infer_vendor_from_name(name: str) -> str:
    """从 GPU 名称推断厂商"""
    name_lower = name.lower()
    for vendor, keywords in VENDOR_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return vendor
    return "unknown"


def _get_visible_devices_env(vendor: str) -> str:
    """根据厂商返回对应的 VISIBLE_DEVICES 环境变量名"""
    for v, _, _, env_var in GPU_VENDORS:
        if v == vendor:
            return env_var
    return "CUDA_VISIBLE_DEVICES"


# =============================================================================
# Python API 检测（最准确，能感知 VISIBLE_DEVICES 限制）
# =============================================================================

def _detect_via_torch() -> Optional[Dict[str, Any]]:
    """通过 torch 检测 GPU"""
    # torch.cuda (NVIDIA / 部分国产兼容)
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem_bytes = getattr(props, 'total_memory', 0)
            vendor = _infer_vendor_from_name(name)
            return {
                "vendor": vendor,
                "name": name,
                "count": count,
                "memory_gb": round(mem_bytes / (1024 ** 3), 1),
                "source": "torch.cuda",
                "visible_devices_env": _get_visible_devices_env(vendor),
            }
    except (ImportError, Exception):
        pass

    # torch.npu (华为昇腾)
    try:
        import torch
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available() and torch.npu.device_count() > 0:
            count = torch.npu.device_count()
            name = torch.npu.get_device_name(0)
            mem_bytes = torch.npu.get_device_properties(0).total_memory
            return {
                "vendor": "huawei",
                "name": name,
                "count": count,
                "memory_gb": round(mem_bytes / (1024 ** 3), 1),
                "source": "torch.npu",
                "visible_devices_env": "ASCEND_RT_VISIBLE_DEVICES",
            }
    except (ImportError, Exception):
        pass

    # torch.mlu (寒武纪)
    try:
        import torch
        if hasattr(torch, "mlu") and torch.mlu.is_available() and torch.mlu.device_count() > 0:
            count = torch.mlu.device_count()
            name = torch.mlu.get_device_name(0)
            mem_bytes = torch.mlu.get_device_properties(0).total_memory
            return {
                "vendor": "cambricon",
                "name": name,
                "count": count,
                "memory_gb": round(mem_bytes / (1024 ** 3), 1),
                "source": "torch.mlu",
                "visible_devices_env": "MLU_VISIBLE_DEVICES",
            }
    except (ImportError, Exception):
        pass

    return None


# =============================================================================
# CLI 工具检测（fallback）
# =============================================================================

def _run_cmd(cmd: str, timeout: int = 10) -> Optional[str]:
    """执行命令，返回 stdout 或 None"""
    try:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def _cli_exists(cli_cmd: str) -> bool:
    """检查 CLI 工具是否存在"""
    try:
        result = subprocess.run(
            ["which", cli_cmd], capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _parse_nvidia_smi(output: str) -> Dict[str, Any]:
    """解析 nvidia-smi --query-gpu=name,memory.total 输出"""
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    if not lines:
        return {}
    first = lines[0].split(",")
    name = first[0].strip() if len(first) > 0 else "unknown"
    mem_mb = float(first[1].strip()) if len(first) > 1 else 0
    return {
        "name": name,
        "count": len(lines),
        "memory_gb": round(mem_mb / 1024, 1),
    }


def _parse_generic_cli(vendor: str, output: str) -> Dict[str, Any]:
    """通用 CLI 输出解析 — 尽力提取，不强求格式"""
    # 对于非 nvidia 厂商，CLI 输出格式各异
    # 至少能确定 vendor，count 和 memory 尽力提取
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    return {
        "name": f"{vendor} GPU",
        "count": max(len(lines), 1),
        "memory_gb": 0,
        "raw_output": output[:500],
    }


def _detect_via_cli() -> Optional[Dict[str, Any]]:
    """遍历厂商 CLI 工具，第一个能执行的就是当前厂商"""
    for vendor, cli_cmd, query_cmd, env_var in GPU_VENDORS:
        if not _cli_exists(cli_cmd):
            continue

        output = _run_cmd(query_cmd)
        if output is None:
            continue

        # Hygon/MetaX 歧义消解：rocm-smi 存在时需进一步判断
        if vendor == "hygon" and cli_cmd == "rocm-smi":
            actual_vendor = _disambiguate_rocm_vendor()
            if actual_vendor and actual_vendor != "hygon":
                continue  # 不是 Hygon，跳过让后续厂商匹配

        # 解析输出
        if vendor == "nvidia":
            info = _parse_nvidia_smi(output)
        else:
            info = _parse_generic_cli(vendor, output)

        if info:
            return {
                "vendor": vendor,
                "name": info.get("name", f"{vendor} GPU"),
                "count": info.get("count", 1),
                "memory_gb": info.get("memory_gb", 0),
                "source": cli_cmd,
                "visible_devices_env": env_var,
                "raw_output": info.get("raw_output"),
            }

    return None


def _disambiguate_rocm_vendor() -> str:
    """当 rocm-smi 存在时，区分 Hygon DCU 和 AMD GPU。

    Hygon 特征：/opt/hyhal 或 /opt/dtk 目录存在，或 hy-smi 命令存在，
    或 torch.cuda.get_device_name() 含 BW 前缀（BW200/BW3000）。
    """
    # 检查 Hygon 特有目录
    if os.path.isdir("/opt/hyhal") or os.path.isdir("/opt/dtk"):
        return "hygon"
    # 检查 hy-smi 命令
    if _cli_exists("hy-smi"):
        return "hygon"
    # 检查 /dev/mkfd（Hygon DCU 特有设备）
    if os.path.exists("/dev/mkfd"):
        return "hygon"
    # 通过 torch 获取设备名
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0).upper()
            if any(kw in dev_name for kw in ["BW", "HYGON", "DCU"]):
                return "hygon"
    except Exception:
        pass
    # 默认视为 AMD（非 Hygon）
    return "amd"


# =============================================================================
# 主检测函数
# =============================================================================

def detect_gpu() -> Optional[Dict[str, Any]]:
    """
    自动检测 GPU 厂商和硬件信息。

    检测顺序：
    1. torch Python API（最准确）
    2. 各厂商 CLI 工具（fallback）

    Returns:
        dict with keys: vendor, name, count, memory_gb, source, visible_devices_env
        None if no GPU detected
    """
    # 1. torch 检测
    result = _detect_via_torch()
    if result:
        return result

    # 2. CLI 检测
    result = _detect_via_cli()
    if result:
        return result

    return None


# =============================================================================
# GPU 空闲检测（per-GPU used/free）
# =============================================================================

# 各厂商的 per-GPU 显存查询命令
_FREE_QUERY_CMDS = {
    "nvidia":    "nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv,noheader,nounits",
    "metax":     "mx-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits",
}

FREE_THRESHOLD_PCT = 5.0  # 显存占用低于此百分比视为空闲


def _parse_csv_gpu_memory(output: str, has_free_col: bool = True) -> List[Dict[str, Any]]:
    """解析 CSV 格式的 per-GPU 显存输出（nvidia-smi / mx-smi 等）"""
    gpus = []
    for line in output.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            used = float(parts[1])
            total = float(parts[2])
            free = float(parts[3]) if has_free_col and len(parts) >= 4 else total - used
            usage_pct = round(used / total * 100, 1) if total > 0 else 0
            gpus.append({
                "index": idx, "used_mib": used, "total_mib": total,
                "free_mib": round(free, 1), "usage_pct": usage_pct,
            })
        except (ValueError, ZeroDivisionError):
            continue
    return gpus


def _parse_npu_smi_free(output: str) -> List[Dict[str, Any]]:
    """解析华为 npu-smi info 输出的 per-GPU 显存信息"""
    import re
    gpus = []
    for match in re.finditer(r'(\d+)\s+\d+\s+\w+\s+\w+\s+(\d+)\s*/\s*(\d+)', output):
        idx, used, total = int(match.group(1)), float(match.group(2)), float(match.group(3))
        free = total - used
        usage_pct = round(used / total * 100, 1) if total > 0 else 0
        gpus.append({
            "index": idx, "used_mib": used, "total_mib": total,
            "free_mib": round(free, 1), "usage_pct": usage_pct,
        })
    return gpus


def _query_gpu_free_for_vendor(vendor: str) -> List[Dict[str, Any]]:
    """查询指定厂商的 per-GPU 显存信息"""
    # CSV 格式厂商
    if vendor in _FREE_QUERY_CMDS:
        output = _run_cmd(_FREE_QUERY_CMDS[vendor])
        if output:
            has_free = vendor == "nvidia"
            return _parse_csv_gpu_memory(output, has_free_col=has_free)

    # 华为昇腾
    if vendor == "huawei":
        output = _run_cmd("npu-smi info")
        if output:
            return _parse_npu_smi_free(output)

    return []


def check_gpu_free(vendor: str = None) -> Dict[str, Any]:
    """检测各 GPU 的显存占用情况（多厂商统一接口）

    Args:
        vendor: 指定厂商（如 "nvidia"），跳过自动探测。
                传 None 则自动探测（先 detect_gpu 确定厂商，再查询）。

    Returns:
        {
            "vendor": str,
            "free_gpus": [int, ...],
            "busy_gpus": [int, ...],
            "total": int,
            "details": [{index, used_mib, total_mib, free_mib, usage_pct}, ...],
            "visible_devices_env": str,
        }
        details 为空列表表示该厂商暂不支持 per-GPU 检测。
    """
    # 确定厂商
    if not vendor:
        info = detect_gpu()
        vendor = info["vendor"] if info else "unknown"

    # 查询 per-GPU 显存
    details = _query_gpu_free_for_vendor(vendor)

    # 如果指定厂商的 SMI 不支持 per-GPU 查询，尝试遍历所有厂商
    if not details and not vendor:
        for v, cli_cmd, _, _ in GPU_VENDORS:
            if _cli_exists(cli_cmd):
                details = _query_gpu_free_for_vendor(v)
                if details:
                    vendor = v
                    break

    free_gpus = [d["index"] for d in details if d["usage_pct"] < FREE_THRESHOLD_PCT]
    busy_gpus = [d["index"] for d in details if d["usage_pct"] >= FREE_THRESHOLD_PCT]

    return {
        "vendor": vendor,
        "free_gpus": free_gpus,
        "busy_gpus": busy_gpus,
        "total": len(details),
        "details": details,
        "visible_devices_env": _get_visible_devices_env(vendor),
    }


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="多厂商 GPU 统一检测")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    parser.add_argument("--check-free", action="store_true", help="检测 per-GPU 显存占用")
    parser.add_argument("--vendor", help="指定厂商（跳过自动探测）")
    args = parser.parse_args()

    if args.check_free:
        result = check_gpu_free(vendor=args.vendor or None)
        print(f"GPU 空闲检测: {result['vendor']} | "
              f"{len(result['free_gpus'])} free / {result['total']} total",
              file=sys.stderr)
    else:
        info = detect_gpu()
        if info is None:
            print("ERROR: 未检测到 GPU", file=sys.stderr)
            result = {"error": "no GPU detected"}
        else:
            result = {k: v for k, v in info.items() if k != "raw_output"}
            print(f"GPU: {result['vendor']} | {result['name']} | "
                  f"{result['count']}x {result['memory_gb']}GB | "
                  f"source={result['source']}", file=sys.stderr)

    # 输出 JSON
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_str)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"已保存: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
