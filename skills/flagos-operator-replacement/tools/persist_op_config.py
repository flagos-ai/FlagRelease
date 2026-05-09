#!/usr/bin/env python3
"""
persist_op_config.py — 算子配置固化工具

将运行时算子配置持久化到容器文件系统中，确保 docker commit 后配置不丢失。
镜像拉下来直接启动服务即为最终达标的算子配置。

固化方式（按 env_type 自动选择）：
- vllm_flaggems：修改源码中 flag_gems.enable() 调用，写入最终算子列表
- vllm_plugin_flaggems：将环境变量持久化到 /etc/environment + /root/.bashrc

两种场景都在 /root/flaggems_op_config.json 记录修改详情。

Usage:
    python persist_op_config.py --auto
    python persist_op_config.py --auto --verify
    python persist_op_config.py --env-type vllm_plugin_flaggems
    python persist_op_config.py --env-type vllm_flaggems --disabled-ops fused_moe,softmax
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CONTEXT_YAML = "/flagos-workspace/shared/context.yaml"
OP_CONFIG_JSON = "/flagos-workspace/results/operator_config.json"
RECORD_FILE = "/root/flaggems_op_config.json"
ETC_ENVIRONMENT = "/etc/environment"
BASHRC = "/root/.bashrc"

FLAGGEMS_BASHRC_MARKER = "# === FlagGems 算子配置（自动生成，勿手动修改）==="
FLAGGEMS_BASHRC_END = "# === FlagGems 算子配置结束 ==="


def load_yaml(path):
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  WARN: 读取 {path} 失败: {e}")
        return {}


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARN: 读取 {path} 失败: {e}")
        return {}


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_runtime_oplist():
    """读取运行时算子列表（权威来源）"""
    candidates = [
        "/tmp/flaggems_enable_oplist.txt",
        "/root/gems.txt",
        "/tmp/gems.txt",
    ]
    # 从 context.yaml 多个可能位置获取 gems_txt_path
    ctx = load_yaml(CONTEXT_YAML)
    for getter in [
        lambda: ctx.get("service", {}).get("gems_txt_path"),
        lambda: ctx.get("environment", {}).get("flaggems_txt_path"),
        lambda: ctx.get("runtime", {}).get("gems_txt_path"),
    ]:
        extra = getter()
        if extra and extra not in candidates:
            candidates.insert(0, extra)

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    ops = [l.strip() for l in f if l.strip()]
                if ops:
                    print(f"  运行时算子列表: {path} ({len(ops)} 个)")
                else:
                    print(f"  运行时算子列表: {path} (空文件，全部算子已禁用)")
                return ops, path
            except Exception:
                continue
    return None, None


def get_disabled_ops_from_config():
    """从 operator_config.json 获取禁用算子列表"""
    state = load_json(OP_CONFIG_JSON)
    disabled = state.get("disabled_ops", [])
    enabled = state.get("enabled_ops", [])
    runtime_ops = state.get("runtime_enabled_ops", [])
    return {
        "disabled": disabled,
        "enabled": enabled,
        "runtime_enabled": runtime_ops,
        "runtime_enabled_count": state.get("runtime_enabled_count"),
    }


def get_env_type():
    """从 context.yaml 获取 env_type"""
    ctx = load_yaml(CONTEXT_YAML)
    return ctx.get("env_type", "")


def get_excluded_ops_from_context():
    """从 context.yaml 获取精度+性能调优禁用的算子"""
    ctx = load_yaml(CONTEXT_YAML)
    eval_section = ctx.get("eval", {})
    perf_section = ctx.get("performance", {})
    opt_section = ctx.get("optimization", {})

    excluded_accuracy = eval_section.get("excluded_ops_accuracy", []) or []
    excluded_perf = opt_section.get("excluded_ops_performance", []) or []

    all_excluded = sorted(set(excluded_accuracy + excluded_perf))
    return all_excluded, excluded_accuracy, excluded_perf


# =========================================================================
# 非 Plugin 场景：修改源码
# =========================================================================

OPS_CONTROL_FILE = "/root/flaggems_ops_control.json"
FLAGGEMS_INJECT_MARKER = "FLAGGEMS_CONTROL_MODE"


def _is_code_injected():
    """检查源码是否已注入环境变量驱动代码"""
    try:
        from toggle_flaggems import find_model_runner_files
    except ImportError:
        sys.path.insert(0, "/flagos-workspace/scripts")
        from toggle_flaggems import find_model_runner_files
    files = find_model_runner_files()
    for f in files:
        try:
            content = Path(f).read_text(encoding='utf-8', errors='ignore')
            if FLAGGEMS_INJECT_MARKER in content:
                return True
        except Exception:
            continue
    return False


def _persist_control_file_and_env(disabled_ops, enabled_ops):
    """已注入场景：持久化控制文件和环境变量"""
    print("\n[环境变量驱动固化] 持久化控制文件和环境变量...")

    # 确定 control_mode：有禁用算子 → only_enable（白名单）；全开 → unused
    if disabled_ops:
        control_mode = "only_enable"
        data = {"include": sorted(enabled_ops) if enabled_ops else []}
    else:
        control_mode = "unused"
        data = {"unused": []}

    # 写入控制文件
    Path(OPS_CONTROL_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OPS_CONTROL_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 控制文件: {OPS_CONTROL_FILE} (mode={control_mode})")

    # 持久化环境变量到 /etc/environment + /root/.bashrc
    env_vars = {
        "USE_FLAGGEMS": "1",
        "FLAGGEMS_CONTROL_MODE": control_mode,
    }
    if disabled_ops:
        env_vars["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(sorted(disabled_ops))
    env_files = []

    try:
        existing = ""
        if os.path.isfile(ETC_ENVIRONMENT):
            with open(ETC_ENVIRONMENT, 'r', encoding='utf-8') as f:
                existing = f.read()
        lines = [l for l in existing.split('\n')
                 if not any(l.startswith(k + "=") for k in env_vars)]
        for k, v in env_vars.items():
            lines.append(f"{k}={v}")
        with open(ETC_ENVIRONMENT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l for l in lines if l is not None) + '\n')
        env_files.append(ETC_ENVIRONMENT)
        print(f"  ✓ {ETC_ENVIRONMENT} 已更新")
    except Exception as e:
        print(f"  WARN: 写入 {ETC_ENVIRONMENT} 失败: {e}")

    try:
        existing = ""
        if os.path.isfile(BASHRC):
            with open(BASHRC, 'r', encoding='utf-8') as f:
                existing = f.read()
        if FLAGGEMS_BASHRC_MARKER in existing:
            pattern = re.compile(
                re.escape(FLAGGEMS_BASHRC_MARKER) + r".*?" + re.escape(FLAGGEMS_BASHRC_END),
                re.DOTALL
            )
            existing = pattern.sub("", existing).strip()
        block = f"\n{FLAGGEMS_BASHRC_MARKER}\n"
        for k, v in env_vars.items():
            block += f"export {k}={v}\n"
        block += f"{FLAGGEMS_BASHRC_END}\n"
        with open(BASHRC, 'w', encoding='utf-8') as f:
            f.write(existing + block)
        env_files.append(BASHRC)
        print(f"  ✓ {BASHRC} 已更新")
    except Exception as e:
        print(f"  WARN: 写入 {BASHRC} 失败: {e}")

    yaml_path = _persist_yaml_config(disabled_ops) if disabled_ops else None

    return {
        "success": len(env_files) > 0,
        "method": "env_control_persist",
        "control_mode": control_mode,
        "control_file": OPS_CONTROL_FILE,
        "env_vars": env_vars,
        "env_files": env_files,
        "yaml_config": yaml_path,
    }


def persist_source_code(disabled_ops, enabled_ops):
    """固化算子配置到源码（或环境变量驱动模式）"""

    # 已注入环境变量驱动代码 → 持久化控制文件和环境变量
    if _is_code_injected():
        return _persist_control_file_and_env(disabled_ops, enabled_ops)

    # 未注入 → 原有源码修改逻辑
    print("\n[源码固化] 修改 flag_gems.enable() 调用...")

    try:
        from toggle_flaggems import modify_enable_call, find_model_runner_files, analyze_flaggems_code
    except ImportError:
        sys.path.insert(0, "/flagos-workspace/scripts")
        from toggle_flaggems import modify_enable_call, find_model_runner_files, analyze_flaggems_code

    files = find_model_runner_files()
    if not files:
        print("  ERROR: 未找到包含 flag_gems 的源码文件")
        return {"success": False, "error": "no_source_files"}

    print(f"  找到 {len(files)} 个包含 flag_gems 的文件")

    # 记录修改前的内容
    before_states = {}
    enable_pattern = re.compile(r"flag_gems\.\w*enable\w*\s*\(")
    for f in files:
        try:
            content = Path(f).read_text(encoding='utf-8', errors='ignore')
            for i, line in enumerate(content.split('\n'), 1):
                if enable_pattern.search(line):
                    before_states[f] = {"line": i, "content": line.strip()}
                    break
        except Exception:
            pass

    result = modify_enable_call(
        files,
        enabled_ops=enabled_ops if enabled_ops else None,
        disabled_ops=disabled_ops,
    )

    modified_files = []
    for r in result.get("results", []):
        entry = {
            "path": r.get("file", ""),
            "method": r.get("method", "unknown"),
            "success": r.get("success", False),
        }
        if r.get("backup"):
            entry["backup"] = r["backup"]
        before = before_states.get(r.get("file", ""))
        if before:
            entry["before"] = before["content"]
        # 读取修改后的内容
        try:
            content = Path(r["file"]).read_text(encoding='utf-8', errors='ignore')
            for line in content.split('\n'):
                if enable_pattern.search(line):
                    entry["after"] = line.strip()
                    break
        except Exception:
            pass
        modified_files.append(entry)
        status = "✓" if r.get("success") else "✗"
        print(f"  {status} {r.get('file', '?')} → {r.get('method', '?')}")

    # 同时尝试 Layer 1 yaml 固化
    yaml_path = _persist_yaml_config(disabled_ops)

    success = any(r.get("success") for r in result.get("results", []))
    return {
        "success": success,
        "method": "source_code_modify",
        "modified_files": modified_files,
        "yaml_config": yaml_path,
        "capabilities": result.get("capabilities", []),
    }


def _persist_yaml_config(disabled_ops):
    """Layer 1 yaml 固化（双重保险）"""
    try:
        import flag_gems
        gems_path = os.path.dirname(flag_gems.__file__)
    except ImportError:
        return None

    config_dirs = []
    for root, dirs, files in os.walk(gems_path):
        if "runtime" in root and "backend" in root:
            config_dirs.append(root)

    if not config_dirs:
        return None

    config_dir = config_dirs[0]
    config_path = os.path.join(config_dir, "enable_configs.yaml")

    content = "exclude:\n"
    for op in sorted(disabled_ops):
        content += f"  - {op}\n"

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Layer 1 yaml 写入: {config_path} ({len(disabled_ops)} 个 exclude)")
        return config_path
    except Exception as e:
        print(f"  WARN: yaml 写入失败: {e}")
        return None


# =========================================================================
# Plugin 场景：持久化环境变量
# =========================================================================

def persist_env_vars(disabled_ops):
    """将环境变量持久化到 /etc/environment 和 /root/.bashrc"""
    print("\n[环境变量固化] 写入持久化环境变量...")

    env_vars = {
        "USE_FLAGGEMS": "1",
        "VLLM_FL_PREFER_ENABLED": "true",
    }
    if disabled_ops:
        env_vars["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(sorted(disabled_ops))

    env_files = []

    # 写入 /etc/environment
    try:
        existing = ""
        if os.path.isfile(ETC_ENVIRONMENT):
            with open(ETC_ENVIRONMENT, 'r', encoding='utf-8') as f:
                existing = f.read()

        # 移除旧的 FlagGems 相关行
        lines = [l for l in existing.split('\n')
                 if not any(l.startswith(k + "=") for k in env_vars)]
        # 追加新的
        for k, v in env_vars.items():
            lines.append(f"{k}={v}")

        with open(ETC_ENVIRONMENT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(l for l in lines if l is not None) + '\n')

        env_files.append(ETC_ENVIRONMENT)
        print(f"  ✓ {ETC_ENVIRONMENT} 已更新")
    except Exception as e:
        print(f"  WARN: 写入 {ETC_ENVIRONMENT} 失败: {e}")

    # 写入 /root/.bashrc
    try:
        existing = ""
        if os.path.isfile(BASHRC):
            with open(BASHRC, 'r', encoding='utf-8') as f:
                existing = f.read()

        # 移除旧的 FlagGems 配置块
        if FLAGGEMS_BASHRC_MARKER in existing:
            pattern = re.compile(
                re.escape(FLAGGEMS_BASHRC_MARKER) + r".*?" + re.escape(FLAGGEMS_BASHRC_END),
                re.DOTALL
            )
            existing = pattern.sub("", existing).strip()

        # 追加新的配置块
        block = f"\n{FLAGGEMS_BASHRC_MARKER}\n"
        for k, v in env_vars.items():
            block += f"export {k}={v}\n"
        block += f"{FLAGGEMS_BASHRC_END}\n"

        with open(BASHRC, 'w', encoding='utf-8') as f:
            f.write(existing + block)

        env_files.append(BASHRC)
        print(f"  ✓ {BASHRC} 已更新")
    except Exception as e:
        print(f"  WARN: 写入 {BASHRC} 失败: {e}")

    success = len(env_files) > 0
    return {
        "success": success,
        "method": "persistent_env_vars",
        "env_vars": env_vars,
        "env_files": env_files,
    }


# =========================================================================
# 记录文件
# =========================================================================

def write_record(env_type, persist_result, enabled_ops, disabled_ops, verified=None, runtime_count=None):
    """写入 /root/flaggems_op_config.json 记录文件"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "env_type": env_type,
        "persist_method": persist_result.get("method", "unknown"),
        "enabled_ops": sorted(enabled_ops) if enabled_ops else [],
        "disabled_ops": sorted(disabled_ops) if disabled_ops else [],
        "enabled_count": len(enabled_ops) if enabled_ops else 0,
        "disabled_count": len(disabled_ops) if disabled_ops else 0,
    }

    if persist_result.get("method") == "source_code_modify":
        record["modified_files"] = persist_result.get("modified_files", [])
        if persist_result.get("yaml_config"):
            record["yaml_config"] = persist_result["yaml_config"]
    elif persist_result.get("method") == "persistent_env_vars":
        record["env_vars"] = persist_result.get("env_vars", {})
        record["env_files"] = persist_result.get("env_files", [])

    if verified is not None:
        record["verified"] = verified
    if runtime_count is not None:
        record["runtime_enabled_count"] = runtime_count

    save_json(record, RECORD_FILE)
    print(f"\n  ✓ 记录文件已写入: {RECORD_FILE}")
    return record


# =========================================================================
# 验证
# =========================================================================

def verify_config(expected_count):
    """重启服务后验证算子数量是否与预期一致"""
    print("\n[验证] 重启服务检查算子配置...")

    # 停止服务
    print("  停止服务...")
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null",
                    shell=True, capture_output=True, timeout=10)
    time.sleep(5)

    # 清除旧的运行时 txt
    for f in ["/tmp/flaggems_enable_oplist.txt", "/tmp/gems.txt"]:
        if os.path.isfile(f):
            os.remove(f)

    # 启动服务（不传任何额外环境变量）
    print("  启动服务（无额外参数）...")
    startup_cmd = "bash /flagos-workspace/scripts/start_service.sh"
    subprocess.run(startup_cmd, shell=True, capture_output=True, timeout=30)

    # 等待服务就绪
    print("  等待服务就绪...")
    wait_result = subprocess.run(
        "bash /flagos-workspace/scripts/wait_for_service.sh --timeout 300",
        shell=True, capture_output=True, text=True, timeout=330
    )
    if wait_result.returncode != 0:
        print("  ✗ 服务启动失败，验证中止")
        return False, None

    # 读取运行时算子列表
    time.sleep(3)
    ops, path = read_runtime_oplist()
    actual_count = len(ops)

    print(f"  运行时算子数: {actual_count}, 预期: {expected_count}")

    if expected_count is not None and actual_count == expected_count:
        print("  ✓ 验证通过：算子数量一致")
        verified, count = True, actual_count
    elif expected_count is None and actual_count > 0:
        print("  ✓ 验证通过：算子列表非空（无预期值可对比）")
        verified, count = True, actual_count
    else:
        print(f"  ✗ 验证失败：算子数量不一致 (实际={actual_count}, 预期={expected_count})")
        verified, count = False, actual_count

    # 停止服务释放 GPU
    subprocess.run("pkill -f 'vllm\\|sglang' 2>/dev/null",
                    shell=True, capture_output=True, timeout=10)
    return verified, count


# =========================================================================
# 主流程
# =========================================================================

def run_persist(env_type=None, disabled_ops_override=None, do_verify=False):
    """执行配置固化"""
    print("=" * 60)
    print("[算子配置固化] persist_op_config.py")
    print("=" * 60)

    # 1. 确定 env_type
    if not env_type:
        env_type = get_env_type()
    print(f"\n  env_type: {env_type}")

    if env_type == "native":
        print("  native 场景无需固化，跳过")
        record = write_record(env_type, {"method": "skip"}, [], [], verified=True)
        print(json.dumps({"success": True, "skipped": True, "reason": "native"}, indent=2))
        return True

    # 2. 获取算子列表
    runtime_ops, runtime_path = read_runtime_oplist()
    config_info = get_disabled_ops_from_config()
    excluded_all, excluded_acc, excluded_perf = get_excluded_ops_from_context()

    # 确定禁用算子列表（优先级：命令行参数 > context.yaml > operator_config.json）
    if disabled_ops_override:
        disabled_ops = disabled_ops_override
    elif excluded_all:
        disabled_ops = excluded_all
    else:
        disabled_ops = config_info.get("disabled", [])

    # 确定启用算子列表
    if runtime_ops is not None:
        enabled_ops = runtime_ops
    else:
        enabled_ops = config_info.get("enabled", []) or config_info.get("runtime_enabled", [])

    # 全关场景：有禁用算子但无启用算子 → 显式设置 enabled_ops = []
    if disabled_ops and not enabled_ops:
        enabled_ops = []

    # 如果没有禁用算子且有运行时算子列表，全量算子已达标，仍需固化 enable(unused=[]) 到源码
    if not disabled_ops and enabled_ops:
        print("  无禁用算子，全量算子已达标，固化 enable(unused=[]) 到源码")

    expected_count = config_info.get("runtime_enabled_count") or (len(enabled_ops) if enabled_ops else None)

    print(f"  启用算子: {len(enabled_ops)} 个")
    print(f"  禁用算子: {len(disabled_ops)} 个")
    if disabled_ops:
        print(f"  禁用列表: {', '.join(sorted(disabled_ops))}")

    # 3. 根据场景执行固化
    if env_type == "vllm_plugin_flaggems":
        persist_result = persist_env_vars(disabled_ops)
    elif env_type == "vllm_flaggems":
        persist_result = persist_source_code(disabled_ops, enabled_ops)
    else:
        # 未知场景，两种都尝试
        print(f"  WARN: 未知 env_type={env_type}，尝试源码修改")
        persist_result = persist_source_code(disabled_ops, enabled_ops)

    if not persist_result.get("success"):
        print("\n  ✗ 配置固化失败")
        record = write_record(env_type, persist_result, enabled_ops, disabled_ops, verified=False)
        print(json.dumps({"success": False, "error": "persist_failed"}, indent=2))
        return False

    # 4. 验证（可选）
    verified = None
    runtime_count = expected_count
    if do_verify:
        verified, runtime_count = verify_config(expected_count)

    # 5. 写入记录文件
    record = write_record(env_type, persist_result, enabled_ops, disabled_ops,
                          verified=verified, runtime_count=runtime_count)

    # 6. 输出结果
    output = {
        "success": True,
        "env_type": env_type,
        "persist_method": persist_result.get("method"),
        "disabled_ops": sorted(disabled_ops),
        "disabled_count": len(disabled_ops),
        "enabled_count": len(enabled_ops),
        "record_file": RECORD_FILE,
        "verified": verified,
    }
    print(f"\n{'=' * 60}")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return True


def main():
    parser = argparse.ArgumentParser(description="算子配置固化工具")
    parser.add_argument("--auto", action="store_true",
                        help="自动检测场景并固化")
    parser.add_argument("--env-type", choices=["vllm_flaggems", "vllm_plugin_flaggems", "native"],
                        help="指定场景（不指定则从 context.yaml 读取）")
    parser.add_argument("--disabled-ops",
                        help="禁用算子列表（逗号分隔，不指定则从 context/config 读取）")
    parser.add_argument("--verify", action="store_true",
                        help="固化后重启服务验证")
    parser.add_argument("--json", action="store_true",
                        help="JSON 输出")

    args = parser.parse_args()

    if not args.auto and not args.env_type:
        parser.print_help()
        sys.exit(1)

    disabled = args.disabled_ops.split(",") if args.disabled_ops else None

    success = run_persist(
        env_type=args.env_type,
        disabled_ops_override=disabled,
        do_verify=args.verify,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
