#!/usr/bin/env python3
"""
算子优化器 — 渐进排除搜索最优算子集

通过渐进排除搜索 FlagGems 算子，找到使 FlagOS 性能 >= 目标比率（默认 80% native）的最优算子组合。

核心算法（渐进排除搜索，默认策略 progressive）：
1. 按性能影响力将算子分为 high/medium/low 三级
2. 逐轮排除：Round 1 排除 high → Round 2 追加排除 medium → Round 3 追加排除 low
3. 达标即停，保留尽可能多的算子
4. 最多 3 轮（vs 旧版分组二分 ~22 轮）

备选策略：
- --search-strategy group: 分组二分搜索（按功能分 5 组，组内二分定位）
- --search-strategy linear: 线性逐个搜索（独立测试每个算子的影响）
- --search-strategy elimination: 逐删策略（累积禁用算子直到达标）

其他功能：
- --runtime-ops: 只搜索运行时实际调用的算子
- --multi-throughput: 接受多并发级别吞吐量，用最小值判定
- mapping 子命令: 输出运行时算子名 <-> aten 算子名映射

注意：此脚本设计为被 Claude Code 调用，不直接执行 benchmark。
它生成配置文件和操作指令，由 Claude Code 执行实际的服务重启和 benchmark。

Usage:
    # 初始化（基本）
    python operator_optimizer.py init --ops-file ops.json --native-throughput 1000.0

    # 初始化（仅搜索运行时算子）
    python operator_optimizer.py init --ops-file ops.json --runtime-ops runtime.json --native-throughput 1000.0

    # 获取下一步操作（分组二分搜索）
    python operator_optimizer.py next --state-path state.json

    # 更新结果（多并发吞吐量）
    python operator_optimizer.py update --op-name softmax --throughputs '{"1":800,"64":900,"256":850}'

    # 生成算子名映射
    python operator_optimizer.py mapping --gems-path /path/to/flag_gems
"""

import sys

# IO 缓冲修复：确保容器内实时输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    import functools
    print = functools.partial(print, flush=True)

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 共享模块导入（兼容本地开发和容器内扁平部署）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from ops_constants import (
    OOT_OPERATORS, OPERATOR_GROUPS,
    RUNTIME_TO_ATEN_MAP, ATEN_TO_RUNTIME_MAP,
    OP_RISK_LEVELS,
)


def env_to_inline(env_dict):
    """将 env dict 转为内联前缀字符串: VAR1=val1 VAR2=val2"""
    import shlex
    parts = []
    for k, v in env_dict.items():
        parts.append(f"{k}={shlex.quote(str(v))}")
    return " ".join(parts)


# =============================================================================
# 算子列表自动发现
# =============================================================================

# 运行时生成的算子列表文件（Plugin 架构下优先搜索）
RUNTIME_OPLIST_PATHS = [
    "/tmp/flaggems_enable_oplist.txt",
    "/tmp/flaggems_oplist.txt",
]


def find_ops_list_file(gems_path: Optional[str] = None) -> Dict[str, Any]:
    """
    自动搜索 flaggems 算子列表文件。

    搜索优先级：
    1. 运行时生成的文件（/tmp/flaggems_enable_oplist.txt 等）
    2. flag_gems 安装目录下的 .txt 文件（通过内容特征识别）

    Returns:
        {
            "found": bool,
            "path": str,         # 找到的文件路径
            "ops": list,         # 解析出的算子列表
            "count": int,
            "source": str,       # "runtime" | "source_code"
            "search_paths": list # 搜索过的路径
        }
    """
    result = {
        "found": False,
        "path": "",
        "ops": [],
        "count": 0,
        "source": "",
        "search_paths": [],
    }

    # 第一优先级：运行时生成的算子列表文件
    for rpath in RUNTIME_OPLIST_PATHS:
        result["search_paths"].append(rpath)
        if os.path.isfile(rpath):
            try:
                with open(rpath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    lines = [l.strip() for l in content.split('\n')
                             if l.strip() and not l.strip().startswith('#')]
                    if lines:
                        result["found"] = True
                        result["path"] = rpath
                        result["ops"] = lines
                        result["count"] = len(lines)
                        result["source"] = "runtime"
                        return result
            except Exception:
                continue

    # 第二优先级：源码目录搜索（原有逻辑）

    # 确定搜索起点
    search_root = gems_path
    if not search_root:
        try:
            import flag_gems
            search_root = os.path.dirname(flag_gems.__file__)
        except ImportError:
            result["error"] = "flag_gems not installed"
            return result

    if not os.path.isdir(search_root):
        result["error"] = f"path not found: {search_root}"
        return result

    # 搜索所有 .txt 文件
    candidates = []
    for root, dirs, files in os.walk(search_root):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(root, fname)
            result["search_paths"].append(fpath)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if not content:
                    continue
                lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
                # 特征判断：每行是一个短标识符（算子名通常 < 40 字符，无空格）
                if len(lines) >= 5 and all(len(l) < 40 and ' ' not in l for l in lines):
                    # 进一步验证：至少有一些已知的算子名
                    known_ops = {"addmm", "mm", "bmm", "softmax", "cos", "sin", "exp",
                                 "relu", "gelu", "silu", "mul", "add", "sub", "div",
                                 "layer_norm", "rms_norm", "embedding", "zeros", "ones"}
                    overlap = set(lines) & known_ops
                    score = len(overlap)
                    candidates.append((score, len(lines), fpath, lines))
            except Exception:
                continue

    if candidates:
        # 选择匹配已知算子数最多的文件
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        best = candidates[0]
        result["found"] = True
        result["path"] = best[2]
        result["ops"] = best[3]
        result["count"] = len(best[3])
        result["source"] = "source_code"
        if len(candidates) > 1:
            result["other_candidates"] = [c[2] for c in candidates[1:]]

    return result


# =============================================================================
# 状态管理
# =============================================================================

DEFAULT_STATE_PATH = Path("/flagos-workspace/results/operator_config.json")


def load_state(state_path: Optional[str] = None) -> Dict[str, Any]:
    """加载优化状态"""
    p = Path(state_path) if state_path else DEFAULT_STATE_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "all_ops": [],
        "search_ops": [],
        "enabled_ops": [],
        "disabled_ops": [],
        "native_throughput": 0.0,
        "target_ratio": 0.8,
        "current_ratio": 0.0,
        "search_log": [],
        "status": "not_started",  # not_started | in_progress | completed | failed
        "search_mode": "group",  # group | linear
        "group_state": {},
        "current_step": 0,
        "current_op": "",
    }


def save_state(state: Dict[str, Any], state_path: Optional[str] = None):
    """保存优化状态"""
    p = Path(state_path) if state_path else DEFAULT_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"状态已保存: {p}")


# =============================================================================
# 算子分组工具
# =============================================================================

def classify_ops(ops: List[str]) -> Dict[str, List[str]]:
    """将算子按功能分组，未归类的放入 'other'"""
    classified = {group: [] for group in OPERATOR_GROUPS}
    classified["other"] = []

    known_ops: Set[str] = set()
    for group_ops in OPERATOR_GROUPS.values():
        known_ops.update(group_ops)

    for op in ops:
        placed = False
        for group_name, group_ops in OPERATOR_GROUPS.items():
            if op in group_ops:
                classified[group_name].append(op)
                placed = True
                break
        if not placed:
            classified["other"].append(op)

    # 移除空组
    return {k: v for k, v in classified.items() if v}


def filter_runtime_ops(all_ops: List[str], runtime_ops: List[str]) -> List[str]:
    """过滤出运行时实际调用的算子（交集），算子名完全按 txt 列表原始名称，不做映射"""
    all_set = set(all_ops)
    result = []
    for op in runtime_ops:
        if op in all_set:
            result.append(op)
    return sorted(set(result))


# =============================================================================
# 初始化
# =============================================================================

def _extract_native_throughputs(benchmark_path: str) -> Dict[str, Dict[str, float]]:
    """
    从 native benchmark JSON 提取每个 test_case × concurrency 的 output + total 双指标。

    返回: {"case|conc": {"output": x, "total": y}, ...}
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容新旧格式
    results = data.get("results", data) if isinstance(data, dict) else data

    throughputs = {}
    for tc_name, tc_results in results.items():
        if not isinstance(tc_results, dict):
            continue
        for key, metrics in tc_results.items():
            if key.startswith("_") or not isinstance(metrics, dict) or "error" in metrics:
                continue
            output_tp = metrics.get('Output token throughput (tok/s)', 0) or 0
            total_tp = metrics.get('Total token throughput (tok/s)', 0) or 0
            if output_tp > 0 or total_tp > 0:
                throughputs[f"{tc_name}|{key}"] = {
                    "output": output_tp,
                    "total": total_tp,
                }
    return throughputs


def init_optimization(ops_file: str, native_throughput: float,
                      target_ratio: float = 0.8,
                      runtime_ops_file: Optional[str] = None,
                      group_search: bool = True,
                      plugin_mode: bool = False,
                      oot_ops: Optional[List[str]] = None,
                      reverse: bool = False,
                      search_strategy: str = "progressive",
                      native_benchmark: Optional[str] = None,
                      state_path: Optional[str] = None,
                      registered_ops_file: Optional[str] = None) -> Dict[str, Any]:
    """
    初始化优化状态。

    Args:
        ops_file: 算子列表 JSON 文件路径（全量注册算子）
        native_throughput: 原生性能基线吞吐量 (tok/s)
        target_ratio: 性能目标比率
        runtime_ops_file: 运行时实际调用的算子列表 JSON（可选）
        group_search: 启用分组二分搜索（仅 group 策略使用）
        plugin_mode: 是否为 plugin 场景（使用环境变量控制）
        oot_ops: 自定义 OOT 算子列表（默认使用 OOT_OPERATORS）
        reverse: 反向搜索（从全禁用逐步启用），适合大量注册算子但少量运行时调用的场景
        search_strategy: 搜索策略 "progressive"（先验预筛+渐进排除）| "group"（分组二分）| "linear" | "elimination"（逐删）
        native_benchmark: native benchmark JSON 文件路径（可选，用于提取双指标基线）
        registered_ops_file: FlagGems 完整注册算子列表 JSON（可选，用于黑名单模式确保覆盖所有算子）
    """
    with open(ops_file, "r", encoding="utf-8") as f:
        ops_data = json.load(f)

    if isinstance(ops_data, list):
        all_ops = ops_data
    elif isinstance(ops_data, dict):
        all_ops = ops_data.get("registered_ops", ops_data.get("ops", []))
    else:
        print("ERROR: 无法解析算子列表文件")
        sys.exit(1)

    all_ops = sorted(all_ops)

    # 收集完整注册表（用于黑名单模式确保不在 oplist 中的算子也被显式关闭）
    registered_ops = all_ops  # 默认与 oplist 相同
    if registered_ops_file:
        try:
            with open(registered_ops_file, "r", encoding="utf-8") as f:
                reg_data = json.load(f)
            if isinstance(reg_data, list):
                registered_ops = sorted(reg_data)
            elif isinstance(reg_data, dict):
                registered_ops = sorted(reg_data.get("ops", reg_data.get("registered_ops", [])))
            extra = set(registered_ops) - set(all_ops)
            if extra:
                print(f"完整注册表: {len(registered_ops)} 个算子（oplist: {len(all_ops)} 个，额外 {len(extra)} 个将被显式禁用）")
            else:
                print(f"完整注册表: {len(registered_ops)} 个算子（与 oplist 一致）")
        except Exception as e:
            print(f"WARNING: 无法加载注册表文件 {registered_ops_file}: {e}，使用 oplist 作为注册表")

    # 确定搜索范围
    search_ops = all_ops
    if runtime_ops_file:
        with open(runtime_ops_file, "r", encoding="utf-8") as f:
            runtime_data = json.load(f)
        if isinstance(runtime_data, list):
            runtime_list = runtime_data
        elif isinstance(runtime_data, dict):
            runtime_list = runtime_data.get("ops", runtime_data.get("runtime_ops", []))
        else:
            runtime_list = []
        search_ops = filter_runtime_ops(all_ops, runtime_list)
        print(f"运行时算子过滤: {len(all_ops)} 全量 -> {len(search_ops)} 运行时")

    # 分组信息（group 和 progressive 策略都需要）
    groups = classify_ops(search_ops)
    group_state = {}
    if search_strategy == "group" and group_search:
        group_order = ["compute", "memory", "math", "index", "reduce", "other"]
        group_state = {
            "group_order": [g for g in group_order if g in groups],
            "current_group_idx": 0,
            "phase": "group_test",  # group_test | binary_search | done
            "binary_state": None,  # {low, high, ops, mid_ops}
            "group_results": {},  # group_name -> "all_disabled" | "binary_searched"
        }

    # Progressive 策略：先验预筛 + 渐进排除
    progressive_state = {}
    if search_strategy == "progressive":
        # 将搜索范围内的算子按风险分级
        risk_high = set(OP_RISK_LEVELS.get("high", []))
        risk_medium = set(OP_RISK_LEVELS.get("medium", []))
        risk_low = set(OP_RISK_LEVELS.get("low", []))
        search_set = set(search_ops)

        excluded_high = sorted(search_set & risk_high)
        excluded_medium = sorted(search_set & risk_medium)
        excluded_low = sorted(search_set - risk_high - risk_medium - risk_low)  # 未知算子归 medium
        excluded_low_known = sorted(search_set & risk_low)

        progressive_state = {
            "phase": "round_test",  # round_test | done
            "current_round": 0,     # 0=high, 1=medium, 2=low
            "rounds": ["high", "medium", "low"],
            "excluded_by_round": {
                "high": excluded_high,
                "medium": excluded_medium + excluded_low,  # 未知算子与 medium 一起排除
                "low": excluded_low_known,
            },
            "cumulative_excluded": [],  # 当前累积排除的算子
        }

    # Elimination 策略：逐个累积禁用直到达标
    elimination_state = {}
    if search_strategy == "elimination":
        if reverse:
            print("WARNING: elimination 策略不支持反向搜索，忽略 --reverse")
            reverse = False
        elimination_state = {
            "phase": "testing",       # testing | done
            "current_idx": 0,         # 当前测试到第几个算子
            "cumulative_disabled": [], # 已累积禁用的算子
        }

    # 确定 search_mode
    if search_strategy == "progressive":
        search_mode = "progressive"
    elif search_strategy == "elimination":
        search_mode = "elimination"
    elif search_strategy == "group" and group_search:
        search_mode = "group"
    else:
        search_mode = "linear"

    # Plugin 场景：OOT 搜索阶段
    effective_oot_ops = oot_ops if oot_ops else list(OOT_OPERATORS)
    oot_state = {}
    search_phase = search_mode  # 默认直接进入对应搜索模式

    if plugin_mode:
        search_phase = "oot"  # plugin 场景先搜索 OOT 层
        oot_state = {
            "tested": [],
            "blacklist": [],
            "current_idx": 0,
            "ops": effective_oot_ops,
        }

    # 反向模式：初始全部禁用（= Native 基线），逐步启用定位问题算子
    if reverse:
        init_enabled = []
        init_disabled = sorted(all_ops)
    else:
        init_enabled = sorted(all_ops)
        init_disabled = []

    state = {
        "all_ops": all_ops,
        "registered_ops": registered_ops,
        "search_ops": search_ops,
        "enabled_ops": init_enabled,
        "disabled_ops": init_disabled,
        "native_throughput": native_throughput,
        "native_throughputs": _extract_native_throughputs(native_benchmark) if native_benchmark else {},
        "target_ratio": target_ratio,
        "current_ratio": 0.0,
        "search_log": [],
        "status": "in_progress",
        "search_mode": search_mode,
        "search_direction": "reverse" if reverse else "forward",
        "search_phase": search_phase,  # oot -> progressive/group -> done
        "plugin_mode": plugin_mode,
        "oot_state": oot_state,
        "oot_blacklist": [],
        "flagos_blacklist": [],
        "flagos_whitelist": [],
        "use_whitelist": _supports_whitelist(),
        "group_state": group_state,
        "progressive_state": progressive_state,
        "elimination_state": elimination_state,
        "groups": groups,
        "current_step": 0,
        "current_op": "",
        "created_at": datetime.now().isoformat(),
    }

    save_state(state, state_path)

    mode_labels = {
        "progressive": "先验预筛+渐进排除",
        "group": "分组二分",
        "linear": "线性",
        "elimination": "逐删",
    }
    direction_label = "反向（全禁用→逐步启用）" if reverse else "正向（全启用→逐步禁用）"
    print(f"优化已初始化: {len(all_ops)} 个算子, 搜索范围 {len(search_ops)} 个")
    print(f"搜索模式: {mode_labels.get(search_mode, search_mode)}, 方向: {direction_label}")
    print(f"算子控制: {'白名单 (WHITELIST)' if state['use_whitelist'] else '黑名单 (BLACKLIST)'}")
    if plugin_mode:
        print(f"Plugin 模式: OOT 算子 {len(effective_oot_ops)} 个 ({', '.join(effective_oot_ops)})")
        print(f"搜索阶段: OOT -> {search_mode} -> done")
    if search_mode == "progressive":
        ps = progressive_state
        for rnd in ps["rounds"]:
            ops = ps["excluded_by_round"].get(rnd, [])
            print(f"  {rnd} 风险: {len(ops)} 个算子 ({', '.join(ops[:5])}{'...' if len(ops) > 5 else ''})")
    elif search_mode == "group":
        for gname, gops in groups.items():
            print(f"  {gname}: {len(gops)} 个算子 ({', '.join(gops[:5])}{'...' if len(gops) > 5 else ''})")
    print(f"原生吞吐量: {native_throughput:.2f} tok/s")
    print(f"目标吞吐量: {native_throughput * target_ratio:.2f} tok/s")

    return state


# =============================================================================
# OOT 层搜索（plugin 场景专用）
# =============================================================================

def get_next_action_oot(state: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
    """OOT 层逐个排查的下一步操作"""
    oot = state.get("oot_state", {})
    oot_ops = oot.get("ops", OOT_OPERATORS)
    idx = oot.get("current_idx", 0)

    if idx >= len(oot_ops):
        # OOT 搜索完成，生成累积 blacklist 后的验证动作
        oot_blacklist = oot.get("blacklist", [])
        if oot_blacklist:
            # 需要用累积的 OOT blacklist 做一次验证
            state["oot_blacklist"] = oot_blacklist
            state["search_phase"] = "oot_verify"
            state["current_step"] += 1
            save_state(state, state_path)

            env_vars = {
                "USE_FLAGGEMS": "1",
                "VLLM_FL_PREFER_ENABLED": "true",
                "VLLM_FL_OOT_BLACKLIST": ",".join(oot_blacklist),
            }
            return {
                "action": "test_oot_cumulative",
                "oot_blacklist": oot_blacklist,
                "step": state["current_step"],
                "env_vars": env_vars,
                "env_inline": env_to_inline(env_vars),
                "message": f"OOT 累积验证: 禁用 {len(oot_blacklist)} 个 OOT 算子 ({', '.join(oot_blacklist)})",
            }
        else:
            # 无需禁用任何 OOT 算子，直接进入下一阶段
            search_mode = state.get("search_mode", "group")
            state["search_phase"] = search_mode
            save_state(state, state_path)
            if search_mode == "progressive":
                return get_next_action_progressive(state, state_path)
            return get_next_action_group(state, state_path)

    current_op = oot_ops[idx]
    state["current_step"] += 1
    state["current_op"] = current_op
    save_state(state, state_path)

    env_vars = {
        "USE_FLAGGEMS": "1",
        "VLLM_FL_PREFER_ENABLED": "true",
        "VLLM_FL_OOT_BLACKLIST": current_op,
    }

    return {
        "action": "test_oot_disable",
        "op": current_op,
        "step": state["current_step"],
        "total_oot_ops": len(oot_ops),
        "oot_progress": f"{idx + 1}/{len(oot_ops)}",
        "env_vars": env_vars,
        "env_inline": env_to_inline(env_vars),
        "message": f"测试禁用 OOT 算子 {current_op} ({idx + 1}/{len(oot_ops)})",
    }


# =============================================================================
# 分组二分搜索
# =============================================================================

def get_next_action_group(state: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
    """分组二分搜索的下一步操作"""
    gs = state["group_state"]
    groups = state.get("groups", {})

    if gs["phase"] == "done" or gs["current_group_idx"] >= len(gs["group_order"]):
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))
        save_state(state, state_path)
        return {"action": "completed", "message": "所有组搜索完成"}

    current_group = gs["group_order"][gs["current_group_idx"]]
    group_ops = groups.get(current_group, [])

    if gs["phase"] == "group_test":
        # 阶段 1：整组禁用测试
        test_enabled = [op for op in state["enabled_ops"] if op not in group_ops]
        test_disabled = sorted(set(state["disabled_ops"] + group_ops))

        state["current_step"] += 1
        save_state(state, state_path)

        return {
            "action": "test_disable_group",
            "group": current_group,
            "group_ops": group_ops,
            "step": state["current_step"],
            "test_enabled_ops": test_enabled,
            "test_disabled_ops": test_disabled,
            "message": f"测试整组禁用 '{current_group}' ({len(group_ops)} 个算子)",
        }

    elif gs["phase"] == "binary_search":
        # 阶段 2：组内二分定位
        bs = gs["binary_state"]
        if not bs or bs["low"] > bs["high"]:
            # 二分搜索完成，进入下一组
            gs["group_results"][current_group] = "binary_searched"
            gs["current_group_idx"] += 1
            gs["phase"] = "group_test"
            gs["binary_state"] = None
            save_state(state, state_path)
            return get_next_action_group(state, state_path)

        mid = (bs["low"] + bs["high"]) // 2
        # 禁用前半部分 [low, mid]
        mid_ops = bs["ops"][bs["low"]:mid + 1]
        bs["mid"] = mid
        bs["mid_ops"] = mid_ops

        test_enabled = [op for op in state["enabled_ops"] if op not in mid_ops]
        test_disabled = sorted(set(state["disabled_ops"] + mid_ops))

        state["current_step"] += 1
        save_state(state, state_path)

        return {
            "action": "test_disable_binary",
            "group": current_group,
            "binary_range": f"[{bs['low']}, {mid}] of {len(bs['ops'])}",
            "test_ops": mid_ops,
            "step": state["current_step"],
            "test_enabled_ops": test_enabled,
            "test_disabled_ops": test_disabled,
            "message": f"二分搜索 '{current_group}': 测试禁用 {len(mid_ops)} 个算子 [{bs['low']}:{mid}]",
        }

    return {"action": "error", "message": f"未知阶段: {gs['phase']}"}


def _validate_ops_consistency(group_ops: List[str], state_groups: Dict[str, List[str]],
                               group_name: str):
    """校验组内算子排序与初始化时一致，防止二分搜索中途排序变化导致废轮"""
    expected = state_groups.get(group_name, [])
    if group_ops != expected:
        raise ValueError(
            f"排序不一致: group '{group_name}' 当前 {group_ops[:3]}... "
            f"vs 初始化时 {expected[:3]}... — 二分搜索全程必须使用同一排序列表"
        )


def get_next_action_group_reverse(state: Dict[str, Any],
                                   state_path: Optional[str] = None) -> Dict[str, Any]:
    """反向分组搜索：从全禁用出发，逐组启用定位问题算子"""
    gs = state["group_state"]
    groups = state.get("groups", {})

    if gs["phase"] == "done" or gs["current_group_idx"] >= len(gs["group_order"]):
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))
        save_state(state, state_path)
        return {"action": "completed", "message": "反向搜索完成：所有组已测试"}

    current_group = gs["group_order"][gs["current_group_idx"]]
    group_ops = groups.get(current_group, [])
    _validate_ops_consistency(group_ops, groups, current_group)

    if gs["phase"] == "group_test":
        # 反向阶段 1：整组启用测试（从 disabled 移到 enabled）
        test_enabled = sorted(set(state["enabled_ops"] + group_ops))
        test_disabled = [op for op in state["disabled_ops"] if op not in group_ops]

        state["current_step"] += 1
        save_state(state, state_path)

        return {
            "action": "test_enable_group",
            "group": current_group,
            "group_ops": group_ops,
            "step": state["current_step"],
            "test_enabled_ops": test_enabled,
            "test_disabled_ops": test_disabled,
            "message": f"[反向] 测试启用 '{current_group}' ({len(group_ops)} 个算子)",
        }

    elif gs["phase"] == "binary_search":
        # 反向阶段 2：组内二分，逐步启用子集定位问题算子
        bs = gs["binary_state"]
        if not bs or bs["low"] > bs["high"]:
            gs["group_results"][current_group] = "binary_searched"
            gs["current_group_idx"] += 1
            gs["phase"] = "group_test"
            gs["binary_state"] = None
            save_state(state, state_path)
            return get_next_action_group_reverse(state, state_path)

        # 校验二分状态的算子列表未被修改
        if bs["ops"] != groups.get(current_group, []):
            raise ValueError("binary_state.ops 与初始分组不一致，搜索结果不可比较")

        mid = (bs["low"] + bs["high"]) // 2
        # 启用前半部分 [low, mid]
        mid_ops = bs["ops"][bs["low"]:mid + 1]
        bs["mid"] = mid
        bs["mid_ops"] = mid_ops

        test_enabled = sorted(set(state["enabled_ops"] + mid_ops))
        test_disabled = [op for op in state["disabled_ops"] if op not in mid_ops]

        state["current_step"] += 1
        save_state(state, state_path)

        return {
            "action": "test_enable_binary",
            "group": current_group,
            "binary_range": f"[{bs['low']}, {mid}] of {len(bs['ops'])}",
            "test_ops": mid_ops,
            "step": state["current_step"],
            "test_enabled_ops": test_enabled,
            "test_disabled_ops": test_disabled,
            "message": f"[反向] 二分 '{current_group}': 启用 {len(mid_ops)} 个算子 [{bs['low']}:{mid}]",
        }

    return {"action": "error", "message": f"未知阶段: {gs['phase']}"}


# =============================================================================
# 先验预筛 + 渐进排除（progressive 策略）
# =============================================================================

def get_next_action_progressive(state: Dict[str, Any],
                                 state_path: Optional[str] = None) -> Dict[str, Any]:
    """渐进排除：按风险等级逐轮排除，达标即停"""
    ps = state.get("progressive_state", {})
    if not ps or ps.get("phase") == "done":
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))
        save_state(state, state_path)
        return {"action": "completed", "message": "渐进排除搜索完成"}

    current_round = ps.get("current_round", 0)
    rounds = ps.get("rounds", ["high", "medium", "low"])

    if current_round >= len(rounds):
        # 所有轮次都不达标
        state["status"] = "failed"
        state["completed_at"] = datetime.now().isoformat()
        save_state(state, state_path)
        return {"action": "failed", "message": "所有风险等级均已排除仍不达标，问题可能不在算子层面"}

    round_name = rounds[current_round]
    round_ops = ps["excluded_by_round"].get(round_name, [])

    # 累积排除：当前轮的排除 = 之前所有轮的排除 + 本轮
    prev_excluded = list(ps.get("cumulative_excluded", []))
    new_excluded = sorted(set(prev_excluded + round_ops))

    # 计算本轮的 enabled/disabled
    all_search = set(state.get("search_ops", state["all_ops"]))
    test_disabled = sorted(set(state.get("disabled_ops", [])) | set(new_excluded))
    test_enabled = sorted(all_search - set(new_excluded))

    state["current_step"] += 1
    save_state(state, state_path)

    return {
        "action": "test_progressive_round",
        "round": round_name,
        "round_index": current_round,
        "excluded_this_round": round_ops,
        "cumulative_excluded": new_excluded,
        "step": state["current_step"],
        "test_enabled_ops": test_enabled,
        "test_disabled_ops": test_disabled,
        "message": (
            f"渐进排除 Round {current_round + 1}/{len(rounds)}: "
            f"排除 {round_name} 风险算子 {len(round_ops)} 个 "
            f"(累计排除 {len(new_excluded)} 个, 保留 {len(test_enabled)} 个)"
        ),
    }


def _update_progressive_result(state: Dict[str, Any], op_name: str,
                                ratio: float, target_ratio: float,
                                log_entry: Dict[str, Any]):
    """处理渐进排除的结果更新：达标即停"""
    ps = state["progressive_state"]
    current_round = ps.get("current_round", 0)
    rounds = ps.get("rounds", ["high", "medium", "low"])
    round_name = rounds[current_round] if current_round < len(rounds) else "?"

    round_ops = ps["excluded_by_round"].get(round_name, [])
    prev_excluded = list(ps.get("cumulative_excluded", []))
    new_excluded = sorted(set(prev_excluded + round_ops))

    if ratio >= target_ratio:
        # 达标 → 结束
        log_entry["decision"] = "progressive_pass"
        log_entry["reason"] = (
            f"Round {current_round + 1} ({round_name}) 达标: "
            f"ratio {ratio*100:.1f}% >= {target_ratio*100:.0f}%"
        )
        ps["cumulative_excluded"] = new_excluded
        ps["phase"] = "done"
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()

        # 更新 enabled/disabled
        all_search = set(state.get("search_ops", state["all_ops"]))
        state["disabled_ops"] = sorted(set(state.get("disabled_ops", [])) | set(new_excluded))
        state["enabled_ops"] = sorted(all_search - set(new_excluded))
        _compute_final_lists(state, new_excluded)

        print(f"  [Progressive Round {current_round + 1}] 达标 {ratio*100:.1f}% — 搜索完成")
        print(f"  排除 {len(new_excluded)} 个算子, 保留 {len(state['enabled_ops'])} 个")
    else:
        # 不达标 → 进入下一轮
        log_entry["decision"] = "progressive_next"
        log_entry["reason"] = (
            f"Round {current_round + 1} ({round_name}) 未达标: "
            f"ratio {ratio*100:.1f}% < {target_ratio*100:.0f}%, 进入下一轮"
        )
        ps["cumulative_excluded"] = new_excluded
        ps["current_round"] = current_round + 1

        if current_round + 1 >= len(rounds):
            state["status"] = "failed"
            state["completed_at"] = datetime.now().isoformat()
            print(f"  [Progressive] 所有轮次均不达标 — 搜索失败")
        else:
            next_round = rounds[current_round + 1]
            next_ops = ps["excluded_by_round"].get(next_round, [])
            print(f"  [Progressive Round {current_round + 1}] 未达标 {ratio*100:.1f}% — "
                  f"进入 Round {current_round + 2} ({next_round}, +{len(next_ops)} 个算子)")


def get_next_action_linear(state: Dict[str, Any], state_path: Optional[str] = None) -> Dict[str, Any]:
    """线性逐个搜索的下一步操作（兼容旧模式）"""
    step = state["current_step"]
    search_ops = state.get("search_ops", state["all_ops"])

    if step >= len(search_ops):
        state["status"] = "completed"
        _compute_final_lists(state, state.get("disabled_ops", []))
        save_state(state, state_path)
        return {"action": "completed", "message": "所有算子已测试"}

    current_op = search_ops[step]
    state["current_op"] = current_op
    save_state(state, state_path)

    test_enabled = [op for op in state["enabled_ops"] if op != current_op]

    return {
        "action": "test_disable",
        "op": current_op,
        "step": step + 1,
        "total_steps": len(search_ops),
        "test_enabled_ops": test_enabled,
        "test_disabled_ops": state["disabled_ops"] + [current_op],
        "message": f"测试禁用算子 '{current_op}' (步骤 {step+1}/{len(search_ops)})",
    }


def get_next_action_elimination(state, state_path=None):
    """逐删策略：累积禁用算子直到达标"""
    es = state.get("elimination_state", {})
    search_ops = state.get("search_ops", state["all_ops"])
    idx = es.get("current_idx", 0)
    cumulative = list(es.get("cumulative_disabled", []))

    if idx >= len(search_ops):
        state["status"] = "failed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))
        save_state(state, state_path)
        return {
            "action": "failed",
            "message": f"所有 {len(search_ops)} 个算子均已禁用，仍未达标",
        }

    current_op = search_ops[idx]
    new_cumulative = cumulative + [current_op]

    # 已有的 disabled_ops + 本轮累积禁用
    test_disabled = sorted(set(state["disabled_ops"]) | set(new_cumulative))
    test_enabled = sorted(set(search_ops) - set(new_cumulative))

    step = state.get("current_step", 0)

    return {
        "action": "test_elimination",
        "op": current_op,
        "step": step + 1,
        "total_steps": len(search_ops),
        "cumulative_disabled": new_cumulative,
        "test_enabled_ops": test_enabled,
        "test_disabled_ops": test_disabled,
        "message": f"累积禁用算子 '{current_op}' ({idx+1}/{len(search_ops)}, 已禁用 {len(cumulative)} 个)",
    }


def get_next_action(state_path: Optional[str] = None) -> Dict[str, Any]:
    """获取下一步操作指令（自动选择搜索模式和阶段）"""
    state = load_state(state_path)

    if state["status"] == "completed":
        return {"action": "completed", "message": "优化已完成"}
    if state["status"] == "failed":
        return {"action": "failed", "message": "优化失败"}
    if state["status"] == "not_started":
        return {"action": "error", "message": "请先执行 init"}

    search_phase = state.get("search_phase", "group")

    # OOT 阶段（plugin 场景）
    if search_phase == "oot":
        return get_next_action_oot(state, state_path)
    elif search_phase == "oot_verify":
        # 累积验证已完成，根据结果决定下一步（由 update 处理）
        # 如果还在 oot_verify 说明 update 后没切换，进入下一阶段
        search_mode = state.get("search_mode", "group")
        state["search_phase"] = search_mode
        save_state(state, state_path)
        if search_mode == "progressive":
            return get_next_action_progressive(state, state_path)
        elif search_mode == "elimination":
            return get_next_action_elimination(state, state_path)
        return get_next_action_group(state, state_path)

    # progressive / group / linear 阶段
    search_mode = state.get("search_mode", "group")
    if search_mode == "progressive":
        action = get_next_action_progressive(state, state_path)
    elif search_mode == "group":
        is_reverse = state.get("search_direction") == "reverse"
        if is_reverse:
            action = get_next_action_group_reverse(state, state_path)
        else:
            action = get_next_action_group(state, state_path)
    elif search_mode == "elimination":
        action = get_next_action_elimination(state, state_path)
    else:
        action = get_next_action_linear(state, state_path)

    # Plugin 模式：在 action 中附加环境变量信息
    if state.get("plugin_mode") and action.get("action") not in ("completed", "failed", "error"):
        oot_bl = state.get("oot_blacklist", [])
        env_vars = {"USE_FLAGGEMS": "1", "VLLM_FL_PREFER_ENABLED": "true"}
        if oot_bl:
            env_vars["VLLM_FL_OOT_BLACKLIST"] = ",".join(oot_bl)

        test_disabled = action.get("test_disabled_ops", [])
        if state.get("use_whitelist"):
            whitelist = _compute_enabled_whitelist(state, test_disabled)
            if whitelist:
                env_vars["VLLM_FL_FLAGOS_WHITELIST"] = ",".join(whitelist)
        else:
            blacklist = _compute_full_blacklist(state, test_disabled)
            if blacklist:
                env_vars["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(blacklist)

        action["env_vars"] = env_vars
        action["env_inline"] = env_to_inline(env_vars)

    return action


# =============================================================================
# 结果更新
# =============================================================================

def _compute_full_blacklist(state: Dict[str, Any], search_disabled: List[str]) -> List[str]:
    """计算完整 flagos 黑名单 = 搜索排除 + 注册表中不在 search_ops 的算子"""
    registered_ops = set(state.get("registered_ops", state.get("all_ops", [])))
    if not registered_ops:
        try:
            import flag_gems
            if hasattr(flag_gems, "all_registered_ops"):
                registered_ops = set(flag_gems.all_registered_ops())
            elif hasattr(flag_gems, "all_ops"):
                registered_ops = set(flag_gems.all_ops())
            if registered_ops:
                print(f"  WARNING: registered_ops 为空，已从 flag_gems 自动收集 {len(registered_ops)} 个算子")
        except Exception:
            print("  WARNING: registered_ops 为空且无法自动收集，黑名单可能不完整")
    search_ops = set(state.get("search_ops", state.get("all_ops", [])))
    unsearched = registered_ops - search_ops
    return sorted(set(search_disabled) | unsearched)


def _compute_enabled_whitelist(state: Dict[str, Any], search_disabled: List[str]) -> List[str]:
    """计算白名单 = search_ops - search_disabled（只列出要启用的算子）"""
    search_ops = set(state.get("search_ops", state.get("all_ops", [])))
    return sorted(search_ops - set(search_disabled))


def _supports_whitelist() -> bool:
    """判断当前环境 FlagGems 是否支持白名单（>= 4.2.1rc0）"""
    try:
        import flag_gems
        ver = getattr(flag_gems, "__version__", "")
        if not ver or ver == "installed":
            return False
        try:
            from packaging.version import Version
            return Version(ver) >= Version("4.2.1rc0")
        except ImportError:
            base = re.match(r"(\d+)\.(\d+)\.(\d+)", ver)
            if not base:
                return False
            parts = [int(base.group(i)) for i in (1, 2, 3)]
            return parts >= [4, 2, 1]
    except Exception:
        return False


def _compute_final_lists(state: Dict[str, Any], search_disabled: List[str]):
    """搜索完成时，根据 use_whitelist 计算并存储最终的 whitelist/blacklist"""
    if state.get("use_whitelist"):
        state["flagos_whitelist"] = _compute_enabled_whitelist(state, search_disabled)
        state["flagos_blacklist"] = []
    else:
        state["flagos_blacklist"] = _compute_full_blacklist(state, search_disabled)
        state["flagos_whitelist"] = []


def compute_min_ratio(throughputs: Dict[str, Any], native_throughput: float = 0,
                      native_throughputs: Optional[Dict[str, Any]] = None) -> float:
    """
    计算所有 test_case × concurrency × {output, total} 的最小 ratio。

    throughputs 格式:
      新格式: {"case|conc": {"output": x, "total": y}, ...}
      旧格式兼容: {"case": float}  (仅 output，用 native_throughput 单值对比)
    native_throughputs: 同格式的 native 基线（新格式时必须提供）
    """
    if not throughputs:
        return 0.0

    ratios = []
    for key, val in throughputs.items():
        if isinstance(val, dict):
            # 新格式: {"output": x, "total": y}
            native_val = (native_throughputs or {}).get(key, {})
            if not native_val and native_throughput > 0:
                # fallback: native_throughputs 为空时用单值对比
                native_val = {"output": native_throughput, "total": native_throughput}
            if isinstance(native_val, (int, float)):
                # 兼容: native 还是旧格式单值
                native_val = {"output": native_val, "total": native_val}
            for metric in ("output", "total"):
                tp = val.get(metric, 0) or 0
                native_tp = native_val.get(metric, 0) if isinstance(native_val, dict) else 0
                if native_tp > 0 and tp > 0:
                    ratios.append(tp / native_tp)
        else:
            # 旧格式兼容: 单一 float 值
            if native_throughput > 0 and val > 0:
                ratios.append(val / native_throughput)

    return min(ratios) if ratios else 0.0


def update_result(op_name: str, throughput: Optional[float] = None,
                  native_throughput: Optional[float] = None,
                  throughputs: Optional[str] = None,
                  state_path: Optional[str] = None) -> Dict[str, Any]:
    """
    更新某个算子/组禁用测试的结果。

    支持两种输入方式：
    1. 单一吞吐量: --throughput + --native-throughput
    2. 多并发吞吐量: --throughputs '{"case|conc": {"output": x, "total": y}, ...}'
       判定使用所有 test_case × concurrency × {output, total} 的最小 ratio
       兼容旧格式: --throughputs '{"case": float}' + --native-throughput
    """
    state = load_state(state_path)
    native_tp = native_throughput or state.get("native_throughput", 0)
    native_tps = state.get("native_throughputs")  # 新格式 native 基线
    target_ratio = state["target_ratio"]

    # 计算 ratio
    if throughputs:
        tp_dict = json.loads(throughputs) if isinstance(throughputs, str) else throughputs
        ratio = compute_min_ratio(tp_dict, native_tp, native_tps)
        # throughput_val: 取最小的 output throughput 用于日志显示
        min_vals = []
        for v in tp_dict.values():
            if isinstance(v, dict):
                min_vals.append(v.get("output", 0) or 0)
            else:
                min_vals.append(v)
        throughput_val = min(min_vals) if min_vals else 0
    elif throughput is not None:
        ratio = throughput / native_tp if native_tp > 0 else 0
        throughput_val = throughput
    else:
        print("ERROR: 必须提供 --throughput 或 --throughputs")
        sys.exit(1)

    log_entry = {
        "op": op_name,
        "throughput": throughput_val,
        "ratio": ratio,
        "timestamp": datetime.now().isoformat(),
    }
    if throughputs:
        log_entry["throughputs"] = json.loads(throughputs) if isinstance(throughputs, str) else throughputs

    search_mode = state.get("search_mode", "linear")
    search_phase = state.get("search_phase", "group")

    if search_phase in ("oot", "oot_verify"):
        _update_oot_result(state, op_name, ratio, target_ratio, log_entry)
    elif search_mode == "progressive":
        _update_progressive_result(state, op_name, ratio, target_ratio, log_entry)
    elif search_mode == "group":
        is_reverse = state.get("search_direction") == "reverse"
        if is_reverse:
            _update_group_result_reverse(state, op_name, ratio, target_ratio, log_entry)
        else:
            _update_group_result(state, op_name, ratio, target_ratio, log_entry)
    elif search_mode == "elimination":
        _update_elimination_result(state, op_name, ratio, target_ratio, log_entry)
    else:
        _update_linear_result(state, op_name, ratio, target_ratio, log_entry)

    state["search_log"].append(log_entry)
    state["current_ratio"] = ratio

    # 达标即停：无论哪种搜索模式，ratio 达标且状态未被子函数标记为 completed 时，强制完成
    if ratio >= target_ratio and state["status"] != "completed":
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))
        print(f"  [达标即停] ratio {ratio*100:.1f}% >= {target_ratio*100:.0f}% — 停止优化")

    # 检查线性模式是否完成
    search_ops = state.get("search_ops", state["all_ops"])
    if search_mode == "linear" and state["current_step"] >= len(search_ops) and state["status"] != "completed":
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))

    save_state(state, state_path)

    return {
        "decision": log_entry.get("decision", "unknown"),
        "ratio": ratio,
        "enabled_ops": state["enabled_ops"],
        "disabled_ops": state["disabled_ops"],
        "progress": f"step {state['current_step']}",
        "status": state["status"],
    }


def _update_oot_result(state: Dict[str, Any], op_name: str,
                       ratio: float, target_ratio: float,
                       log_entry: Dict[str, Any]):
    """处理 OOT 搜索阶段的结果更新"""
    search_phase = state.get("search_phase", "oot")
    oot = state.get("oot_state", {})

    if search_phase == "oot_verify":
        # 累积验证结果
        if ratio >= target_ratio:
            log_entry["decision"] = "oot_sufficient"
            log_entry["reason"] = f"OOT 禁用后 ratio {ratio*100:.1f}% >= {target_ratio*100:.0f}%, 无需搜索 torch 底层"
            state["status"] = "completed"
            state["search_phase"] = "done"
            state["completed_at"] = datetime.now().isoformat()
            _compute_final_lists(state, [])  # OOT 达标，flagos 层无需排除
            print(f"  [OOT 累积] 达标 {ratio*100:.1f}% - 搜索完成")
        else:
            search_mode = state.get("search_mode", "group")
            log_entry["decision"] = "oot_insufficient"
            log_entry["reason"] = f"OOT 禁用后 ratio {ratio*100:.1f}% < {target_ratio*100:.0f}%, 进入 {search_mode} 搜索"
            state["search_phase"] = search_mode
            print(f"  [OOT 累积] 未达标 {ratio*100:.1f}% - 进入 {search_mode} 阶段")
        return

    # 单个 OOT 算子测试
    # 基线：全量 FlagGems（含该 OOT 算子）的性能
    # 测试：禁用该 OOT 算子后的性能
    # 如果禁用后性能显著提升 → 该算子拖慢性能 → 加入 blacklist
    baseline_ratio = state.get("current_ratio", 0)

    if ratio > baseline_ratio + 0.02:  # 禁用后性能提升 > 2%
        log_entry["decision"] = "oot_blacklisted"
        log_entry["reason"] = f"禁用 {op_name} 后 ratio 从 {baseline_ratio*100:.1f}% 提升到 {ratio*100:.1f}%"
        oot["blacklist"].append(op_name)
        print(f"  [OOT {op_name}] BLACKLISTED - 性能提升 {(ratio-baseline_ratio)*100:.1f}%")
    else:
        log_entry["decision"] = "oot_kept"
        log_entry["reason"] = f"禁用 {op_name} 后 ratio {ratio*100:.1f}% 无显著提升"
        print(f"  [OOT {op_name}] KEPT - 无显著提升")

    oot["tested"].append(op_name)
    oot["current_idx"] = len(oot["tested"])


def _update_group_result(state: Dict[str, Any], op_name: str,
                         ratio: float, target_ratio: float,
                         log_entry: Dict[str, Any]):
    """处理分组搜索模式的结果更新"""
    gs = state["group_state"]
    groups = state.get("groups", {})

    if gs["current_group_idx"] >= len(gs["group_order"]):
        return

    current_group = gs["group_order"][gs["current_group_idx"]]
    group_ops = groups.get(current_group, [])

    if gs["phase"] == "group_test":
        if ratio >= target_ratio:
            # 整组禁用仍达标 → 全部禁用
            log_entry["decision"] = "group_disabled"
            log_entry["reason"] = f"整组 {current_group} 禁用后 ratio {ratio*100:.1f}% >= {target_ratio*100:.0f}%"
            for op in group_ops:
                if op in state["enabled_ops"]:
                    state["enabled_ops"].remove(op)
                if op not in state["disabled_ops"]:
                    state["disabled_ops"].append(op)
            gs["group_results"][current_group] = "all_disabled"
            gs["current_group_idx"] += 1
            print(f"  [{current_group}] 整组禁用 - {ratio*100:.1f}% >= {target_ratio*100:.0f}%")
        else:
            # 不达标 → 进入二分搜索
            log_entry["decision"] = "need_binary_search"
            log_entry["reason"] = f"整组 {current_group} 禁用后 ratio {ratio*100:.1f}% < {target_ratio*100:.0f}%"
            gs["phase"] = "binary_search"
            gs["binary_state"] = {
                "ops": group_ops,
                "low": 0,
                "high": len(group_ops) - 1,
                "mid": 0,
                "mid_ops": [],
            }
            print(f"  [{current_group}] 需要二分搜索 - {ratio*100:.1f}% < {target_ratio*100:.0f}%")

    elif gs["phase"] == "binary_search":
        bs = gs["binary_state"]
        mid = bs["mid"]
        mid_ops = bs["mid_ops"]

        if ratio >= target_ratio:
            # 禁用前半部分仍达标 → 前半部分可以禁用，继续搜索后半部分
            log_entry["decision"] = "binary_disabled_half"
            for op in mid_ops:
                if op in state["enabled_ops"]:
                    state["enabled_ops"].remove(op)
                if op not in state["disabled_ops"]:
                    state["disabled_ops"].append(op)
            bs["low"] = mid + 1
            print(f"  [{current_group}] 二分: 前半可禁用 [{bs['low']-len(mid_ops)},{mid}], "
                  f"继续搜索 [{bs['low']},{bs['high']}]")
        else:
            # 不达标 → 前半部分有关键算子，缩小搜索到前半部分
            log_entry["decision"] = "binary_kept_half"
            if mid_ops and len(mid_ops) == 1:
                # 单个算子已定位 → 保留它，继续下一段
                print(f"  [{current_group}] 二分: 定位关键算子 '{mid_ops[0]}', 保留")
                bs["low"] = mid + 1
            else:
                bs["high"] = mid
                print(f"  [{current_group}] 二分: 前半有关键算子, "
                      f"缩小到 [{bs['low']},{bs['high']}]")

        # 检查二分是否完成
        if bs["low"] > bs["high"]:
            gs["group_results"][current_group] = "binary_searched"
            gs["current_group_idx"] += 1
            gs["phase"] = "group_test"
            gs["binary_state"] = None
            print(f"  [{current_group}] 二分搜索完成")


def _update_group_result_reverse(state: Dict[str, Any], op_name: str,
                                  ratio: float, target_ratio: float,
                                  log_entry: Dict[str, Any]):
    """处理反向分组搜索的结果更新（启用算子后判断性能）"""
    gs = state["group_state"]
    groups = state.get("groups", {})

    if gs["current_group_idx"] >= len(gs["group_order"]):
        return

    current_group = gs["group_order"][gs["current_group_idx"]]
    group_ops = groups.get(current_group, [])

    if gs["phase"] == "group_test":
        if ratio >= target_ratio:
            # 整组启用后仍达标 → 该组安全，保留启用
            log_entry["decision"] = "group_enabled"
            log_entry["reason"] = f"[反向] 启用 {current_group} 后 ratio {ratio*100:.1f}% >= {target_ratio*100:.0f}%"
            for op in group_ops:
                if op not in state["enabled_ops"]:
                    state["enabled_ops"].append(op)
                    state["enabled_ops"].sort()
                if op in state["disabled_ops"]:
                    state["disabled_ops"].remove(op)
            gs["group_results"][current_group] = "all_enabled"
            gs["current_group_idx"] += 1
            print(f"  [{current_group}] 整组启用安全 - {ratio*100:.1f}% >= {target_ratio*100:.0f}%")
        else:
            # 不达标 → 该组有问题算子 → 回退到 disabled → 进入二分
            log_entry["decision"] = "need_binary_search"
            log_entry["reason"] = f"[反向] 启用 {current_group} 后 ratio {ratio*100:.1f}% < {target_ratio*100:.0f}%"
            # 确保组内算子全部在 disabled（回退）
            for op in group_ops:
                if op in state["enabled_ops"]:
                    state["enabled_ops"].remove(op)
                if op not in state["disabled_ops"]:
                    state["disabled_ops"].append(op)
            gs["phase"] = "binary_search"
            gs["binary_state"] = {
                "ops": group_ops,
                "low": 0,
                "high": len(group_ops) - 1,
                "mid": 0,
                "mid_ops": [],
            }
            print(f"  [{current_group}] 有问题算子，需二分 - {ratio*100:.1f}% < {target_ratio*100:.0f}%")

    elif gs["phase"] == "binary_search":
        bs = gs["binary_state"]
        mid = bs["mid"]
        mid_ops = bs["mid_ops"]

        if ratio >= target_ratio:
            # 启用前半仍达标 → 前半安全，保留启用，继续测试后半
            log_entry["decision"] = "binary_enabled_half"
            for op in mid_ops:
                if op not in state["enabled_ops"]:
                    state["enabled_ops"].append(op)
                    state["enabled_ops"].sort()
                if op in state["disabled_ops"]:
                    state["disabled_ops"].remove(op)
            bs["low"] = mid + 1
            print(f"  [{current_group}] 二分: 前半安全启用, 继续 [{bs['low']},{bs['high']}]")
        else:
            # 启用前半不达标 → 前半有问题 → 回退前半到 disabled
            log_entry["decision"] = "binary_reverted_half"
            for op in mid_ops:
                if op in state["enabled_ops"]:
                    state["enabled_ops"].remove(op)
                if op not in state["disabled_ops"]:
                    state["disabled_ops"].append(op)
            if mid_ops and len(mid_ops) == 1:
                # 单个算子定位 → 保持禁用，继续下一段
                print(f"  [{current_group}] 二分: 定位问题算子 '{mid_ops[0]}', 保持禁用")
                bs["low"] = mid + 1
            else:
                bs["high"] = mid
                print(f"  [{current_group}] 二分: 前半有问题, 缩小到 [{bs['low']},{bs['high']}]")

        # 检查二分是否完成
        if bs["low"] > bs["high"]:
            gs["group_results"][current_group] = "binary_searched"
            gs["current_group_idx"] += 1
            gs["phase"] = "group_test"
            gs["binary_state"] = None
            print(f"  [{current_group}] 反向二分完成")


def _update_linear_result(state: Dict[str, Any], op_name: str,
                          ratio: float, target_ratio: float,
                          log_entry: Dict[str, Any]):
    """处理线性搜索模式的结果更新"""
    if ratio >= target_ratio:
        log_entry["decision"] = "disabled"
        log_entry["reason"] = f"ratio {ratio*100:.1f}% >= target {target_ratio*100:.0f}%"
        if op_name in state["enabled_ops"]:
            state["enabled_ops"].remove(op_name)
        if op_name not in state["disabled_ops"]:
            state["disabled_ops"].append(op_name)
        print(f"  [{op_name}] DISABLED - {ratio*100:.1f}% >= {target_ratio*100:.0f}%")
    else:
        log_entry["decision"] = "kept"
        log_entry["reason"] = f"ratio {ratio*100:.1f}% < target {target_ratio*100:.0f}%"
        if op_name not in state["enabled_ops"]:
            state["enabled_ops"].append(op_name)
            state["enabled_ops"].sort()
        if op_name in state["disabled_ops"]:
            state["disabled_ops"].remove(op_name)
        print(f"  [{op_name}] KEPT - {ratio*100:.1f}% < {target_ratio*100:.0f}%")

    state["current_step"] += 1


def _update_elimination_result(state, op_name, ratio, target_ratio, log_entry):
    """逐删策略结果更新：累积禁用，达标即停"""
    es = state.get("elimination_state", {})
    cumulative = list(es.get("cumulative_disabled", []))

    # 无论达标与否，当前算子都加入累积禁用列表（逐删语义）
    if op_name not in cumulative:
        cumulative.append(op_name)
    es["cumulative_disabled"] = cumulative

    # 更新 enabled_ops / disabled_ops 反映累积禁用
    if op_name in state["enabled_ops"]:
        state["enabled_ops"].remove(op_name)
    if op_name not in state["disabled_ops"]:
        state["disabled_ops"].append(op_name)

    if ratio >= target_ratio:
        # 达标：停止搜索
        log_entry["decision"] = "elimination_done"
        log_entry["reason"] = f"累积禁用 {len(cumulative)} 个算子后达标 {ratio*100:.1f}% >= {target_ratio*100:.0f}%"
        es["phase"] = "done"
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        _compute_final_lists(state, state.get("disabled_ops", []))
        print(f"  [elimination] 达标! 累积禁用 {len(cumulative)} 个算子, ratio {ratio*100:.1f}% >= {target_ratio*100:.0f}%")
    else:
        # 不达标：继续禁用下一个
        log_entry["decision"] = "elimination_continue"
        log_entry["reason"] = f"累积禁用 {len(cumulative)} 个仍不达标 {ratio*100:.1f}% < {target_ratio*100:.0f}%"
        es["current_idx"] = es.get("current_idx", 0) + 1
        search_ops = state.get("search_ops", state["all_ops"])
        if es["current_idx"] >= len(search_ops):
            state["status"] = "failed"
            state["completed_at"] = datetime.now().isoformat()
            _compute_final_lists(state, state.get("disabled_ops", []))
            print(f"  [elimination] 失败: 所有 {len(search_ops)} 个算子均已禁用，仍未达标")
        else:
            print(f"  [elimination] 继续: 已禁用 {len(cumulative)} 个, ratio {ratio*100:.1f}% < {target_ratio*100:.0f}%")

    state["elimination_state"] = es
    state["current_step"] += 1


# =============================================================================
# 算子名映射
# =============================================================================

def generate_mapping(gems_path: Optional[str] = None) -> Dict[str, Any]:
    """
    生成运行时算子名 <-> aten 算子名映射。

    优先从 flag_gems 源码的 @register 装饰器提取，
    回退到内置的静态映射表。
    """
    mapping = {
        "runtime_to_aten": dict(RUNTIME_TO_ATEN_MAP),
        "aten_to_runtime": dict(ATEN_TO_RUNTIME_MAP),
        "source": "builtin",
        "dynamic_entries": [],
    }

    # 尝试从源码提取
    search_path = gems_path
    if not search_path:
        try:
            import flag_gems
            search_path = os.path.dirname(flag_gems.__file__)
        except ImportError:
            pass

    if search_path and os.path.isdir(search_path):
        dynamic = _extract_register_decorators(search_path)
        if dynamic:
            mapping["source"] = "source_code"
            mapping["dynamic_entries"] = dynamic
            for entry in dynamic:
                rt_name = entry.get("func_name", "")
                aten_name = entry.get("aten_name", "")
                if rt_name and aten_name and rt_name != aten_name:
                    mapping["runtime_to_aten"][rt_name] = aten_name
                    mapping["aten_to_runtime"][aten_name] = rt_name

    return mapping


def _extract_register_decorators(gems_path: str) -> List[Dict[str, str]]:
    """从 flag_gems 源码中提取 @register 装饰器的映射信息"""
    entries = []
    pattern = re.compile(
        r'@(?:flag_gems\.)?register\s*\(\s*["\']([^"\']+)["\']\s*\)'
    )

    for root, dirs, files in os.walk(gems_path):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                for match in pattern.finditer(content):
                    aten_name = match.group(1)
                    # 找到紧接的 def 行
                    rest = content[match.end():]
                    def_match = re.search(r'\ndef\s+(\w+)\s*\(', rest)
                    if def_match:
                        func_name = def_match.group(1)
                        entries.append({
                            "aten_name": aten_name,
                            "func_name": func_name,
                            "file": filepath,
                        })
            except Exception:
                continue

    return entries


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(state_path: Optional[str] = None) -> str:
    """生成优化报告"""
    state = load_state(state_path)

    report = []
    report.append("=" * 60)
    report.append("算子优化报告")
    report.append("=" * 60)
    report.append(f"状态: {state['status']}")
    report.append(f"搜索模式: {state.get('search_mode', 'linear')}")
    report.append(f"搜索方向: {state.get('search_direction', 'forward')}")
    report.append(f"原生吞吐量: {state['native_throughput']:.2f} tok/s")
    report.append(f"目标比率: {state['target_ratio']*100:.0f}%")
    report.append(f"目标吞吐量: {state['native_throughput'] * state['target_ratio']:.2f} tok/s")
    report.append(f"")
    report.append(f"总算子数: {len(state['all_ops'])}")
    report.append(f"搜索范围: {len(state.get('search_ops', state['all_ops']))}")
    report.append(f"启用算子: {len(state['enabled_ops'])}")
    report.append(f"禁用算子: {len(state['disabled_ops'])}")
    report.append(f"总搜索步数: {state['current_step']}")
    report.append(f"")

    # 分组搜索结果
    gs = state.get("group_state", {})
    if gs.get("group_results"):
        report.append("分组搜索结果:")
        for gname, gresult in gs["group_results"].items():
            group_ops = state.get("groups", {}).get(gname, [])
            disabled_in_group = [op for op in group_ops if op in state["disabled_ops"]]
            report.append(f"  {gname}: {gresult} ({len(disabled_in_group)}/{len(group_ops)} 禁用)")
        report.append("")

    # 渐进排除结果
    ps = state.get("progressive_state", {})
    if ps and ps.get("phase") == "done":
        report.append("渐进排除结果:")
        rounds = ps.get("rounds", [])
        current_round = ps.get("current_round", 0)
        for i, rnd in enumerate(rounds):
            ops = ps["excluded_by_round"].get(rnd, [])
            if i <= current_round:
                status = "已排除" if i < current_round or ps["phase"] == "done" else "当前轮"
                report.append(f"  {rnd} 风险 ({len(ops)} 个): {status}")
            else:
                report.append(f"  {rnd} 风险 ({len(ops)} 个): 未测试（已提前达标）")
        report.append(f"  累计排除: {len(ps.get('cumulative_excluded', []))} 个算子")
        report.append("")

    # 逐删策略结果
    es = state.get("elimination_state", {})
    if es and es.get("cumulative_disabled"):
        report.append("逐删策略结果:")
        cumulative = es["cumulative_disabled"]
        report.append(f"  累积禁用: {len(cumulative)} 个算子")
        report.append(f"  结果: {'达标' if es.get('phase') == 'done' else '未达标'}")
        report.append(f"  禁用顺序:")
        for i, op in enumerate(cumulative):
            report.append(f"    {i+1}. {op}")
        report.append("")

    if state["disabled_ops"]:
        report.append("禁用的算子:")
        for op in sorted(state["disabled_ops"]):
            reason = ""
            for log in state["search_log"]:
                if log.get("op") == op and "disabled" in log.get("decision", ""):
                    reason = f"({log.get('throughput', 0):.2f} tok/s, {log.get('ratio', 0)*100:.1f}%)"
                    break
            report.append(f"  - {op} {reason}")
    else:
        report.append("无需禁用任何算子")

    report.append("")
    report.append("搜索日志:")
    for i, log in enumerate(state["search_log"]):
        tp = log.get('throughput', 0)
        r = log.get('ratio', 0)
        report.append(f"  {i+1}. {log.get('op', '?')}: {log.get('decision', '?')} - "
                       f"{tp:.2f} tok/s ({r*100:.1f}%)")

    result = "\n".join(report)
    print(result)
    return result


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="算子优化器 - 分组二分搜索最优算子集")
    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    # init 子命令
    init_parser = subparsers.add_parser("init", help="初始化优化")
    init_parser.add_argument("--ops-file", required=True, help="算子列表 JSON 文件")
    init_parser.add_argument("--native-throughput", type=float, required=True, help="原生吞吐量 (tok/s)")
    init_parser.add_argument("--target-ratio", type=float, default=0.8, help="性能目标比率")
    init_parser.add_argument("--runtime-ops", help="运行时实际调用的算子列表 JSON（可选，只搜索这些算子）")
    init_parser.add_argument("--no-group-search", action="store_true", help="禁用分组二分，使用线性搜索")
    init_parser.add_argument("--plugin-mode", action="store_true", help="Plugin 模式（使用环境变量控制，含 OOT 搜索）")
    init_parser.add_argument("--oot-ops", help="自定义 OOT 算子列表（逗号分隔，默认使用内置列表）")
    init_parser.add_argument("--reverse", action="store_true", help="反向搜索（从全禁用逐步启用，适合大量算子注册干扰 dispatch 的场景）")
    init_parser.add_argument("--search-strategy", choices=["progressive", "group", "linear", "elimination"],
                             default="progressive", help="搜索策略: progressive(先验预筛+渐进排除) | group(分组二分) | linear(线性) | elimination(逐删)")
    init_parser.add_argument("--native-benchmark", help="native benchmark JSON 文件路径（用于提取双指标基线）")
    init_parser.add_argument("--registered-ops", help="FlagGems 完整注册算子列表 JSON（用于黑名单模式确保覆盖所有算子）")
    init_parser.add_argument("--state-path", help="状态文件路径")

    # next 子命令
    next_parser = subparsers.add_parser("next", help="获取下一步操作")
    next_parser.add_argument("--state-path", help="状态文件路径")

    # update 子命令
    update_parser = subparsers.add_parser("update", help="更新测试结果")
    update_parser.add_argument("--op-name", required=True, help="被测试的算子/组名")
    update_parser.add_argument("--throughput", type=float, help="禁用后的吞吐量（单一值）")
    update_parser.add_argument("--native-throughput", type=float, help="原生基线吞吐量")
    update_parser.add_argument("--throughputs", help='多并发吞吐量 JSON: {"1":800,"64":900}')
    update_parser.add_argument("--state-path", help="状态文件路径")

    # report 子命令
    report_parser = subparsers.add_parser("report", help="生成优化报告")
    report_parser.add_argument("--state-path", help="状态文件路径")

    # status 子命令
    status_parser = subparsers.add_parser("status", help="查看当前状态")
    status_parser.add_argument("--state-path", help="状态文件路径")

    # mapping 子命令
    mapping_parser = subparsers.add_parser("mapping", help="生成算子名映射表")
    mapping_parser.add_argument("--gems-path", help="flag_gems 源码路径（可选，自动探测）")
    mapping_parser.add_argument("--output", help="输出 JSON 文件路径")

    # discover 子命令
    discover_parser = subparsers.add_parser("discover", help="自动搜索算子列表文件")
    discover_parser.add_argument("--gems-path", help="flag_gems 源码路径（可选，自动探测）")
    discover_parser.add_argument("--save-ops", help="将发现的算子列表保存为 JSON 文件")

    args = parser.parse_args()

    if args.command == "init":
        oot_ops_list = args.oot_ops.split(",") if args.oot_ops else None
        init_optimization(
            args.ops_file, args.native_throughput,
            args.target_ratio,
            runtime_ops_file=args.runtime_ops,
            group_search=not args.no_group_search,
            plugin_mode=args.plugin_mode,
            oot_ops=oot_ops_list,
            reverse=args.reverse,
            search_strategy=args.search_strategy,
            native_benchmark=args.native_benchmark,
            state_path=args.state_path,
            registered_ops_file=args.registered_ops,
        )

    elif args.command == "next":
        action = get_next_action(args.state_path)
        print(json.dumps(action, indent=2, ensure_ascii=False))

    elif args.command == "update":
        result = update_result(
            args.op_name,
            throughput=args.throughput,
            native_throughput=args.native_throughput,
            throughputs=args.throughputs,
            state_path=args.state_path,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "report":
        generate_report(args.state_path)

    elif args.command == "status":
        state = load_state(args.state_path)
        status_info = {
            "status": state["status"],
            "search_mode": state.get("search_mode", "linear"),
            "search_direction": state.get("search_direction", "forward"),
            "search_phase": state.get("search_phase", "group"),
            "plugin_mode": state.get("plugin_mode", False),
            "progress": f"step {state['current_step']}",
            "total_ops": len(state["all_ops"]),
            "search_ops": len(state.get("search_ops", state["all_ops"])),
            "enabled": len(state["enabled_ops"]),
            "disabled": len(state["disabled_ops"]),
            "current_op": state.get("current_op", ""),
            "oot_blacklist": state.get("oot_blacklist", []),
            "flagos_blacklist": state.get("flagos_blacklist", []),
            "flagos_whitelist": state.get("flagos_whitelist", []),
            "use_whitelist": state.get("use_whitelist", False),
        }
        gs = state.get("group_state", {})
        if gs:
            status_info["group_progress"] = gs.get("group_results", {})
            idx = gs.get("current_group_idx", 0)
            order = gs.get("group_order", [])
            status_info["current_group"] = order[idx] if idx < len(order) else "done"
        ps = state.get("progressive_state", {})
        if ps:
            rounds = ps.get("rounds", [])
            current_round = ps.get("current_round", 0)
            status_info["progressive_phase"] = ps.get("phase", "?")
            status_info["progressive_round"] = f"{current_round + 1}/{len(rounds)}" if current_round < len(rounds) else "done"
            status_info["cumulative_excluded"] = len(ps.get("cumulative_excluded", []))
        print(json.dumps(status_info, indent=2, ensure_ascii=False))

    elif args.command == "mapping":
        mapping = generate_mapping(getattr(args, 'gems_path', None))
        output = json.dumps(mapping, indent=2, ensure_ascii=False)
        print(output)
        if hasattr(args, 'output') and args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"\n映射表已保存: {args.output}")

    elif args.command == "discover":
        result = find_ops_list_file(getattr(args, 'gems_path', None))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["found"] and args.save_ops:
            save_path = Path(args.save_ops)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(sorted(result["ops"]), f, indent=2, ensure_ascii=False)
            print(f"\n算子列表已保存: {save_path} ({result['count']} 个算子)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
