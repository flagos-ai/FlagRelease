#!/usr/bin/env python3
"""
flagos_op_config.py — 算子配置应用的统一共享模块（半程收敛）

收敛此前散落在 operator_reduction / operator_expansion / apply_op_config / diagnose_ops /
toggle_flaggems 中的重复实现，作为唯一权威来源：

  1. env 构建与内联字符串：build_op_env() / env_to_inline()
  2. 双路应用（plugin→WHITELIST env / 非plugin→控制文件）：write_op_config()
  3. 环境探测与持久化 helper：is_plugin_env() / persist_env() / clear_env() / env_has()

设计边界（半程收敛，刻意不做）：
  - 不接管 operator_search.py 的 restart_service 与 Layer 1-4 能力探测
    （该链路已经正确且经真机验证，动它是负收益，见 unified-op-config-refactor-plan）
  - 不统一三份 restart 实现，仅供 write/env 侧复用

判定基准（CLAUDE.md 约束26）：
  - plugin 场景（VLLM_FL_PREFER_ENABLED=true）：worker 只读 VLLM_FL_* env，
    控制文件 /root/flaggems_ops_control.json 完全无效，必须写 WHITELIST/BLACKLIST env
  - 非 plugin 场景：控制文件 + FLAGGEMS_CONTROL_MODE 是正确抓手

部署：由 setup_workspace.sh 的 SCRIPT_MAP 部署到容器 /flagos-workspace/scripts/，
同级工具通过 `from flagos_op_config import ...` 引用（扁平目录 import，已有先例）。
"""

import json
import os
from typing import Dict, List, Optional

DEFAULT_CONTROL_FILE = "/root/flaggems_ops_control.json"
ETC_ENVIRONMENT = "/etc/environment"


# =============================================================================
# 算子变体展开（治本：调优阶段就展开，确保测试配置与固化配置一致）
# =============================================================================

def expand_operator_variants(ops: List[str]) -> List[str]:
    """展开算子变体（addmm → addmm + addmm_out + addmm_dtype + addmm_dtype_out）

    Memory 教训（flaggems-blacklist-variant-matching）：
    plugin 模式 VLLM_FL_FLAGOS_BLACKLIST 按函数 __name__ 精确匹配，
    禁 addmm 只禁 base 变体，实际 dispatch 的 addmm_out 仍启用 → 调优空转。

    必须在调优阶段就展开变体，让测试配置与固化配置一致，避免：
    - 调优测试：BLACKLIST=addmm（未展开，无效）
    - 固化/发布：BLACKLIST=addmm,addmm_out,...（自动展开，未经验证）
    → 发布的镜像配置与调优验证的不一致

    Args:
        ops: 算子基础名列表（如 ["addmm", "softmax"]）

    Returns:
        展开后的完整变体列表（如 ["addmm", "addmm_out", "addmm_dtype", ...]）
    """
    if not ops:
        return []

    try:
        import flag_gems
        if not hasattr(flag_gems, 'FULL_CONFIG_BY_FUNC'):
            print("  WARN: flag_gems.FULL_CONFIG_BY_FUNC 不可用，跳过变体展开")
            return sorted(set(ops))

        all_funcs = set(flag_gems.FULL_CONFIG_BY_FUNC.keys())
        expanded = set()

        for op in ops:
            # 查找该 op 的所有变体：op 本身 + op_* 后缀（如 addmm_out, addmm_dtype）
            variants = [f for f in all_funcs if f == op or f.startswith(op + '_')]
            if variants:
                expanded.update(variants)
            else:
                # 保底：即使在 FULL_CONFIG_BY_FUNC 中找不到，也保留原始名
                expanded.add(op)

        result = sorted(expanded)
        if len(result) != len(ops):
            print(f"  算子变体展开: {len(ops)} → {len(result)} 个")
        return result

    except Exception as e:
        print(f"  WARN: 算子变体展开失败，使用原始列表: {e}")
        return sorted(set(ops))


# =============================================================================
# env 构建（唯一实现，替代 4-5 份重复的 env_to_inline / generate）
# =============================================================================

def env_to_inline(env_dict: Dict[str, str]) -> str:
    """将 env dict 转为内联前缀字符串: VAR1=val1 VAR2=val2

    含空格/单引号的值加引号包裹。空值输出 VAR= （显式置空）。
    """
    parts = []
    for k, v in env_dict.items():
        v = "" if v is None else str(v)
        if " " in v or "'" in v:
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def build_op_env(mode: str = "custom",
                 enabled_ops: Optional[List[str]] = None,
                 disabled_ops: Optional[List[str]] = None,
                 oot_blacklist: Optional[List[str]] = None,
                 per_op: Optional[str] = None,
                 expand_variants: bool = True) -> Dict[str, str]:
    """构建 plugin 场景的算子控制 env dict（唯一实现）。

    mode:
      "native" -> USE_FLAGGEMS=0 VLLM_FL_PREFER_ENABLED=false（全关）
      "full"   -> USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true（全量）
      "custom" -> 白名单优先（enabled_ops→WHITELIST），否则黑名单（disabled_ops→BLACKLIST）

    expand_variants: 是否自动展开算子变体（默认 True）
      - True: addmm → addmm + addmm_out + addmm_dtype + addmm_dtype_out
      - False: 保持原样（仅用于特殊场景，如已经展开过的数据）

    仅生成 env，不落盘不重启。调用方决定是内联启动（瞬时）还是 persist_env（持久化）。
    """
    env: Dict[str, str] = {}
    if mode == "native":
        env["USE_FLAGGEMS"] = "0"
        env["VLLM_FL_PREFER_ENABLED"] = "false"
        return env
    if mode == "full":
        env["USE_FLAGGEMS"] = "1"
        env["VLLM_FL_PREFER_ENABLED"] = "true"
        return env
    # custom
    env["USE_FLAGGEMS"] = "1"
    env["VLLM_FL_PREFER_ENABLED"] = "true"
    if oot_blacklist:
        if expand_variants:
            oot_blacklist = expand_operator_variants(oot_blacklist)
        env["VLLM_FL_OOT_BLACKLIST"] = ",".join(sorted(oot_blacklist))
    if enabled_ops:
        # 白名单优先
        if expand_variants:
            enabled_ops = expand_operator_variants(enabled_ops)
        env["VLLM_FL_FLAGOS_WHITELIST"] = ",".join(sorted(enabled_ops))
    elif disabled_ops:
        if expand_variants:
            disabled_ops = expand_operator_variants(disabled_ops)
        env["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(sorted(disabled_ops))
    if per_op:
        env["VLLM_FL_PER_OP"] = per_op
    return env


# =============================================================================
# 环境探测与持久化 helper（唯一实现）
# =============================================================================

def env_has(key: str) -> bool:
    """key 是否存在于 /etc/environment"""
    if not os.path.exists(ETC_ENVIRONMENT):
        return False
    with open(ETC_ENVIRONMENT) as f:
        return any(l.startswith(f"{key}=") for l in f)


def is_plugin_env() -> bool:
    """判断当前是否为 plugin 控制环境（进程 env 或 /etc/environment 有 VLLM_FL_PREFER_ENABLED）"""
    return os.environ.get("VLLM_FL_PREFER_ENABLED") == "true" or env_has("VLLM_FL_PREFER_ENABLED")


def persist_env(key: str, value: str):
    """将环境变量写入 /etc/environment（持久化）并同步当前进程 env"""
    lines = []
    if os.path.exists(ETC_ENVIRONMENT):
        with open(ETC_ENVIRONMENT, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
    lines.append(f"{key}={value}\n")
    with open(ETC_ENVIRONMENT, 'w') as f:
        f.writelines(lines)
    os.environ[key] = value


def clear_env(key: str):
    """从 /etc/environment 移除变量并同步当前进程 env"""
    if os.path.exists(ETC_ENVIRONMENT):
        with open(ETC_ENVIRONMENT, 'r') as f:
            lines = [l for l in f.readlines() if not l.startswith(f"{key}=")]
        with open(ETC_ENVIRONMENT, 'w') as f:
            f.writelines(lines)
    os.environ.pop(key, None)


def load_etc_environment():
    """加载 /etc/environment 中 FlagGems 相关变量到 os.environ"""
    if not os.path.exists(ETC_ENVIRONMENT):
        return
    with open(ETC_ENVIRONMENT, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key.startswith(('USE_FLAGGEMS', 'FLAGGEMS_', 'VLLM_FL_')):
                os.environ[key] = val


# =============================================================================
# 双路应用（唯一实现，替代 reduction/expansion 各自的 write_control_file）
# =============================================================================

def write_op_config(enabled_ops: List[str],
                    control_file: str = DEFAULT_CONTROL_FILE) -> str:
    """按环境类型应用算子白名单配置并持久化，供 start_service.sh 重启后生效。

    plugin 环境：写 VLLM_FL_FLAGOS_WHITELIST 到 /etc/environment（清除冲突的 BLACKLIST）；
                enabled_ops 为空时 USE_FLAGGEMS=0（plugin 仍可独立运行）。
                ⚠ 不写控制文件——plugin 下 VLLM_FL_PREFER_ENABLED=true 使控制文件完全无效，
                误写会导致算子调整静默不生效（历史 bug：v5-operator-expansion-whitelist-bug）。
    非 plugin 环境：写控制文件 {"include": [...]} + FLAGGEMS_CONTROL_MODE=only_enable。

    返回所走路径: "plugin_env" | "control_file"
    """
    if is_plugin_env():
        clear_env("VLLM_FL_FLAGOS_BLACKLIST")
        if enabled_ops:
            persist_env("USE_FLAGGEMS", "1")
            persist_env("VLLM_FL_FLAGOS_WHITELIST", ",".join(sorted(enabled_ops)))
        else:
            persist_env("USE_FLAGGEMS", "0")
            persist_env("VLLM_FL_FLAGOS_WHITELIST", "")
        persist_env("VLLM_FL_PREFER_ENABLED", "true")
        return "plugin_env"
    # 非 plugin：控制文件
    persist_env("USE_FLAGGEMS", "1")
    os.makedirs(os.path.dirname(control_file), exist_ok=True)
    with open(control_file, 'w') as f:
        json.dump({"include": sorted(enabled_ops)}, f, indent=2, ensure_ascii=False)
    persist_env("FLAGGEMS_CONTROL_MODE", "only_enable")
    return "control_file"
