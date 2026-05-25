#!/usr/bin/env python3
"""
apply_op_config.py — 生成算子替换环境变量配置（仅 plugin 场景）

输出 JSON 格式的环境变量字典和内联前缀字符串，
服务启动时使用内联方式 `VAR=val cmd` 注入环境变量。

注意：此脚本不写入算子列表 txt 文件。txt 文件由 FlagGems 运行时自动生成。
重启服务后可通过 find_ops_list_file() 读取 txt 文件验证环境变量是否生效。

Usage:
    python apply_op_config.py --mode native
    python apply_op_config.py --mode full
    python apply_op_config.py --mode custom \
        --oot-blacklist "fused_moe" \
        --flagos-blacklist "softmax,layer_norm"
    python apply_op_config.py --mode custom \
        --flagos-whitelist "addmm,mm,bmm"
    python apply_op_config.py --from-state /path/to/operator_config.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def env_to_inline(env_dict):
    """将 env dict 转为内联前缀字符串: VAR1=val1 VAR2=val2"""
    parts = []
    for k, v in env_dict.items():
        if " " in v or "'" in v:
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def generate(mode, oot_blacklist=None, flagos_blacklist=None, flagos_whitelist=None, per_op=None):
    """
    生成环境变量字典（仅 plugin 场景使用）。

    mode:
      "native"  -> USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false
      "full"    -> USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true
      "custom"  -> 按 whitelist/blacklist 自定义（白名单优先）
    """
    env = {}

    if mode == "native":
        env["USE_FLAGGEMS"] = "0"
        env["VLLM_FL_PREFER_ENABLED"] = "false"
    elif mode == "full":
        env["USE_FLAGGEMS"] = "1"
        env["VLLM_FL_PREFER_ENABLED"] = "true"
    elif mode == "custom":
        env["USE_FLAGGEMS"] = "1"
        env["VLLM_FL_PREFER_ENABLED"] = "true"
        if oot_blacklist:
            bl = ",".join(oot_blacklist) if isinstance(oot_blacklist, list) else oot_blacklist
            env["VLLM_FL_OOT_BLACKLIST"] = bl
        # 白名单优先，黑名单兜底
        if flagos_whitelist:
            wl = ",".join(flagos_whitelist) if isinstance(flagos_whitelist, list) else flagos_whitelist
            env["VLLM_FL_FLAGOS_WHITELIST"] = wl
        elif flagos_blacklist:
            bl = ",".join(flagos_blacklist) if isinstance(flagos_blacklist, list) else flagos_blacklist
            env["VLLM_FL_FLAGOS_BLACKLIST"] = bl
        if per_op:
            env["VLLM_FL_PER_OP"] = per_op
    else:
        print(f"ERROR: unknown mode '{mode}'", file=sys.stderr)
        sys.exit(1)

    result = {
        "success": True,
        "mode": mode,
        "env_vars": env,
        "env_inline": env_to_inline(env),
        "oot_blacklist": oot_blacklist or [],
        "flagos_blacklist": flagos_blacklist or [],
        "flagos_whitelist": flagos_whitelist or [],
        "timestamp": datetime.now().isoformat(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return env


def from_state(state_path):
    """从 operator_config.json 状态文件生成配置。"""
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    oot_blacklist = state.get("oot_blacklist", [])
    flagos_blacklist = state.get("flagos_blacklist", [])
    flagos_whitelist = state.get("flagos_whitelist", [])

    if not oot_blacklist and not flagos_blacklist and not flagos_whitelist:
        # 无 blacklist/whitelist = full 模式
        generate("full")
    else:
        generate("custom", oot_blacklist=oot_blacklist,
                 flagos_blacklist=flagos_blacklist,
                 flagos_whitelist=flagos_whitelist)


def main():
    parser = argparse.ArgumentParser(
        description="生成算子替换环境变量配置（仅 plugin 场景）")

    parser.add_argument("--mode", choices=["native", "full", "custom"],
                        help="配置模式")
    parser.add_argument("--oot-blacklist",
                        help="OOT 层 blacklist（逗号分隔）")
    parser.add_argument("--flagos-blacklist",
                        help="FlagOS 层 blacklist（逗号分隔）")
    parser.add_argument("--flagos-whitelist",
                        help="FlagOS 层 whitelist（逗号分隔，优先于 blacklist）")
    parser.add_argument("--per-op",
                        help="逐算子控制（如 rms_norm=vendor;attention_backend=vendor）")
    parser.add_argument("--from-state",
                        help="从 operator_config.json 生成")

    args = parser.parse_args()

    if args.from_state:
        from_state(args.from_state)
    elif args.mode:
        oot_bl = args.oot_blacklist.split(",") if args.oot_blacklist else None
        flagos_bl = args.flagos_blacklist.split(",") if args.flagos_blacklist else None
        flagos_wl = args.flagos_whitelist.split(",") if args.flagos_whitelist else None
        generate(args.mode, oot_blacklist=oot_bl, flagos_blacklist=flagos_bl,
                 flagos_whitelist=flagos_wl, per_op=args.per_op)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
