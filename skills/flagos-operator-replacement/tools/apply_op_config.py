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

# env 构建走统一共享模块（唯一权威实现）
from flagos_op_config import build_op_env, env_to_inline


def generate(mode, oot_blacklist=None, flagos_blacklist=None, flagos_whitelist=None, per_op=None):
    """
    生成环境变量字典（仅 plugin 场景使用）。

    mode:
      "native"  -> USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false
      "full"    -> USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true
      "custom"  -> 按 whitelist/blacklist 自定义（白名单优先）
    """
    if mode not in ("native", "full", "custom"):
        print(f"ERROR: unknown mode '{mode}'", file=sys.stderr)
        sys.exit(1)

    def as_list(v):
        if v is None:
            return None
        return v if isinstance(v, list) else [s for s in str(v).split(",") if s]

    env = build_op_env(
        mode=mode,
        enabled_ops=as_list(flagos_whitelist),
        disabled_ops=as_list(flagos_blacklist),
        oot_blacklist=as_list(oot_blacklist),
        per_op=per_op,
    )

    result = {
        "success": True,
        "mode": mode,
        "env_vars": env,
        "env_inline": env_to_inline(env),
        "oot_blacklist": as_list(oot_blacklist) or [],
        "flagos_blacklist": as_list(flagos_blacklist) or [],
        "flagos_whitelist": as_list(flagos_whitelist) or [],
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
