#!/usr/bin/env python3
"""
v1_gate.py — 分支 B V1 三选状态机强制闸门判定（编排层专用）

设计动机（与 step7_gate.py 同源）：
  分支 B 的 V1 基线必须由 baseline_selector.py 三选状态机确定性产出
  （v1.1/v1.2/v1.3/none），但 run_pipeline.sh 全程没有 shell 强制调用点——
  完全靠 Claude "记得"按 CLAUDE.md 调用。Claude 可能自起服务测一下就断言
  "V1 可跑/不可跑"，把 context 的 baseline.v1_variant/v1_available 写成臆断值，
  导致下游（V2 分支选择 2.1/2.2、合成基线触发、精度基线回退 NV）全部建立在
  未经三选验证的判据上。

  本闸门**只信任 baseline_selector.py 的产物事实**，不读 Claude 写的任何 ok 字段：
    - v1_baseline_selection.json 存在，且含 attempts[] 数组
    - attempts[] 每项具备真实尝试痕迹（variant + service_ok + smoke_passed 字段），
      且至少尝试过 v1.1（三选状态机必然从 v1.1 开始）
  Claude 手写 context 或伪造一个简单 json 都不会产生这个结构完整、带每变体
  service_ok/smoke_passed/smoke_answer 记录的 attempts 数组。

判定输出（stdout 单行，供 shell case 分派）：
  ok      —— 已有 baseline_selector 真实产物，V1 三选已确定性执行过，放行
  needed  —— 缺产物或产物不含真实 attempts 痕迹 → 必须 shell 兜底直调 baseline_selector.py

Usage:
  python3 v1_gate.py --selection /path/v1_baseline_selection.json
"""

import argparse
import json
import os
import sys

_VALID_VARIANTS = {"v1.1", "v1.2", "v1.3", "none"}


def _load(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def has_real_selection(data):
    """v1_baseline_selection.json 是否含 baseline_selector.py 的真实三选痕迹。

    Claude 声明式写入不会产生结构完整的 attempts[]（每项带 variant/service_ok/
    smoke_passed）。这些字段只有 try_variant() 实际启动+冒烟才会逐个写出。
    """
    if not isinstance(data, dict):
        return False

    # v1_variant 必须是合法三选结果
    if data.get("v1_variant") not in _VALID_VARIANTS:
        return False

    # attempts[] 必须是非空列表，且每项具备真实尝试痕迹
    attempts = data.get("attempts")
    if not isinstance(attempts, list) or len(attempts) == 0:
        return False

    variants_seen = set()
    for a in attempts:
        if not isinstance(a, dict):
            return False
        # 每个 attempt 必须有 variant + 两道关的判定字段（try_variant 的产物结构）
        if "variant" not in a or "service_ok" not in a or "smoke_passed" not in a:
            return False
        variants_seen.add(a.get("variant"))

    # 三选状态机必然从 v1.1 开始尝试；缺 v1.1 说明不是真实三选产物
    if "v1.1" not in variants_seen:
        return False

    return True


def main():
    ap = argparse.ArgumentParser(description="分支 B V1 三选强制闸门判定")
    ap.add_argument("--selection", required=True,
                    help="baseline_selector.py 产出的 v1_baseline_selection.json 路径")
    args = ap.parse_args()

    data = _load(args.selection)
    if has_real_selection(data):
        print("ok")
    else:
        print("needed")


if __name__ == "__main__":
    main()
