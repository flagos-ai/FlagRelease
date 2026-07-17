#!/usr/bin/env python3
"""
v5_gate.py — 段5（V5 算子扩展 / 精度探测）强制闸门判定（编排层专用）

设计动机（与 step7_gate.py 同源思路）：
  prompt 层的"必须跑 operator_expansion.py 做精度探测"不可靠——agent 可能不信任
  prompt 显式规则（如从 git commit message 推断错误逻辑），跳过精度调优阶段直接发布 V5，
  导致带着精度不达标的算子组合就发出去了（真实事故：摩尔迁移 LFM2.5-1.2B-Thinking，
  开启最多算子精度不达标却未做精度探测，强行产出）。

  本闸门**完全不信任 agent 的判断/自我声明**，只看 agent 无法用嘴伪造的事实。

用户红线（2026-07）：不在乎产出镜像的标签，只要"三组件齐全 + 精度达标"就能交付；
  但"明明可以调优却没调优、精度不达标还强行产出"绝不容忍。因此本闸门做两件事：
    (A) 判断精度探测该不该跑、跑没跑（防止臆断跳过调优）
    (B) 发布前精度终检（守住红线：不管前面怎样，最终精度不达标就拒绝发布）

────────────────────────────────────────────────────────────
子命令 gate —— 精度探测执行闸门（判断该不该补跑扩展）
  两类 agent 无法伪造的事实：
    1. 是否有算子可扩展——四源兜底判断（对齐 run_pipeline.sh HAS_DISABLED_OPS，
       不因单一字段没写就误判"无需扩展"）：
         来源1 optimization.disabled_ops
         来源2 eval.excluded_ops_accuracy
         来源3 别名字段 workflow/optimization.{v2_disabled_ops,disabled_ops,excluded_ops}
         来源4 service.initial_operator_list 与 optimization.enabled_ops 的差集（不依赖字段名，最难写偏）
    2. 精度探测是否真跑过——operator_config_v5.json 的 actual_rounds/probed_ops/tier_results/completed
  输出：
    no_expansion —— 四源均无算子可扩展 → V5=当前最优版本，合法放行（走发布，交精度终检把关）
    done         —— 有算子可扩展 且 已有真实探测痕迹 → 放行
    needed       —— 有算子可扩展 且 无真实探测痕迹 → agent 疑似臆断跳过，必须补跑
    no_data      —— 无法读取 context → 交由上层兜底

子命令 accuracy —— 发布前精度终检（守住红线，最后一道关）
  独立读最终精度，不看 agent 任何声明：
    diff = v1_score - v5_score，pass 当且仅当 diff <= threshold（默认 5.0，与脚本 run_accuracy_check 同义）
  输出：
    pass       —— 精度达标 或 无需扩展(no_expansion，精度由 V3 阶段已保证) → 允许发布
    fail       —— 最终精度不达标（diff > threshold）→ 拒绝发布（守住红线）
    no_result  —— 做过扩展但缺 gpqa_v5.json（没做最终评测）→ 拒绝发布（不确定即不发）
    no_data    —— 缺 v1 基线，无法判定 → 交由上层兜底
────────────────────────────────────────────────────────────

Usage:
  python3 v5_gate.py gate     --context <ctx.yaml> [--state <operator_config_v5.json>]
  python3 v5_gate.py accuracy --context <ctx.yaml> --v1-result <gpqa_native.json> \
                              --v5-result <gpqa_v5.json> [--threshold 5.0]
"""

import argparse
import json
import os
import sys


def _load_yaml(path):
    if not path or not os.path.exists(path):
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _load_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _as_ops(v):
    """把算子列表字段规范化为 set，容忍真实数据里的多种写法：
      - list：       ['a', 'b']
      - 带括号字符串：'[a,b,c]' / '[a, b, c]'（真实 context 里 disabled_ops 常是这种）
      - 裸字符串：    'a,b,c'
      - 空/None：     -> set()
    真实事故：ERNIE 的 optimization.disabled_ops = '[zeros,ones,...]'（字符串），
    旧版只认 list 会把它当空 → 源1 失效 → 误判 no_expansion → 跳过精度探测。
    """
    if isinstance(v, list):
        return {str(x).strip() for x in v if str(x).strip()}
    if isinstance(v, str):
        s = v.strip().lstrip("[").rstrip("]")
        return {p.strip().strip("'\"") for p in s.split(",") if p.strip()}
    return set()


def has_expandable_ops(ctx):
    """四源兜底判断是否有算子可扩展（对齐 run_pipeline.sh HAS_DISABLED_OPS）。

    只要任一来源显示有算子被禁用，就认为"有算子可扩展、精度探测必须跑"——
    不因单一字段（如 optimization.disabled_ops）没被正确写入/写成字符串就误判为无需扩展。
    返回 True/False；ctx 非法返回 None。
    """
    if not isinstance(ctx, dict):
        return None
    opt = ctx.get("optimization", {}) or {}
    wf = ctx.get("workflow", {}) or {}
    ev = ctx.get("eval", {}) or {}
    svc = ctx.get("service", {}) or {}

    disabled = _as_ops(opt.get("disabled_ops"))
    disabled |= _as_ops(ev.get("excluded_ops_accuracy"))
    for k in ("v2_disabled_ops", "disabled_ops", "excluded_ops"):
        disabled |= _as_ops(wf.get(k))
        disabled |= _as_ops(opt.get(k))
    # 来源4：初始全量算子集 vs 当前启用集的差集（不依赖字段名）。
    # 注意：enabled 为空时不代表"0个启用"（可能只是没记录），故差集仅在 enabled 非空时可信；
    # 但 initial 非空且 enabled 为空本身也是"有算子未记为启用"的可疑信号——保守起见，
    # 只要 initial 非空且 enabled 未覆盖全部 initial，就视为有可扩展空间。
    initial = _as_ops(svc.get("initial_operator_list"))
    enabled = _as_ops(opt.get("enabled_ops"))
    diff_positive = bool(initial and (initial - enabled))

    return bool(disabled or diff_positive)


def has_real_expansion(state_path):
    """operator_config_v5.json 是否含 operator_expansion.py 真实运行痕迹。

    agent 声明"跳过"不会写这些字段；只有 run_expansion 真跑过才会产生。
    """
    state = _load_json(state_path)
    if not isinstance(state, dict):
        return False
    if isinstance(state.get("actual_rounds"), int) and state["actual_rounds"] > 0:
        return True
    if state.get("probed_ops"):
        return True
    if state.get("tier_results"):
        return True
    if state.get("completed") is True:
        return True
    return False


def cmd_gate(args):
    ctx = _load_yaml(args.context)
    if ctx is None:
        print("no_data")
        return

    expandable = has_expandable_ops(ctx)
    if expandable is None:
        print("no_data")
        return

    # 四源均无算子可扩展 → V5=当前最优版本，合法放行（脚本会走"无需扩展→success"分支）
    if not expandable:
        print("no_expansion")
        return

    # 有算子可扩展：区分"真跑过精度探测"与"没跑（疑似臆断跳过）"
    if has_real_expansion(args.state):
        print("done")
    else:
        print("needed")


def cmd_accuracy(args):
    """发布前精度终检——守住红线：最终精度不达标绝不发布。"""
    ctx = _load_yaml(args.context) if args.context else None
    # 无算子可扩展场景：V5=当前 V3 状态，精度已在 V3 阶段(QUALIFIED_CORE_V3)保证，放行
    if ctx is not None and has_expandable_ops(ctx) is False:
        print("pass")
        return

    v1_data = _load_json(args.v1_result)
    if not isinstance(v1_data, dict) or v1_data.get("score") is None:
        print("no_data")
        return
    v1_score = v1_data.get("score")

    v5_data = _load_json(args.v5_result)
    if not isinstance(v5_data, dict) or v5_data.get("score") is None:
        # 做过扩展却没有最终评测结果 → 不确定即不发
        print("no_result")
        return
    v5_score = v5_data.get("score")

    try:
        diff = float(v1_score) - float(v5_score)
    except (TypeError, ValueError):
        print("no_data")
        return

    print("pass" if diff <= args.threshold else "fail")


def main():
    ap = argparse.ArgumentParser(description="段5 V5 精度探测/终检强制闸门")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="精度探测执行闸门（该不该补跑扩展）")
    g.add_argument("--context", required=True, help="context.yaml / snapshot")
    g.add_argument("--state", default="", help="operator_config_v5.json")
    g.set_defaults(func=cmd_gate)

    a = sub.add_parser("accuracy", help="发布前精度终检（守住红线）")
    a.add_argument("--context", default="", help="context.yaml（判断是否 no_expansion）")
    a.add_argument("--v1-result", required=True, help="V1 基线精度 JSON（gpqa_native.json）")
    a.add_argument("--v5-result", required=True, help="V5 最终精度 JSON（gpqa_v5.json）")
    a.add_argument("--threshold", type=float, default=5.0, help="精度退化阈值（默认 5.0）")
    a.set_defaults(func=cmd_accuracy)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
