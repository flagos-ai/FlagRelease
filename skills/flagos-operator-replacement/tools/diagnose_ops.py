#!/usr/bin/env python3
"""
diagnose_ops.py — 算子问题快速定位（三场景诊断）

在搜索之前先"体检"，快速定位问题算子，避免盲搜。

子命令:
    crash-log        从崩溃日志提取问题算子（O(1) 定位）
    accuracy-groups  按功能组生成 blacklist，供逐组精度测试（≤3 轮定位）
    profile          性能热点预扫描，定位耗时最多的算子

Usage:
    python3 diagnose_ops.py crash-log --log-path /flagos-workspace/logs/startup_flagos.log
    python3 diagnose_ops.py accuracy-groups --ops-file /flagos-workspace/results/ops_list.json
    python3 diagnose_ops.py profile --port 8000 --model-name Qwen3-8B

退出码: 0=成功, 1=未找到问题, 2=参数错误
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 共享模块导入（兼容本地开发和容器内扁平部署）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))

from ops_constants import OOT_OPERATORS, OPERATOR_GROUPS


# =============================================================================
# 场景 1：崩溃日志解析
# =============================================================================

# vLLM 日志前缀清理：
#   完整格式: (EngineCore pid=711) ERROR 04-23 03:21:46 [core.py:1108] ...
#   简短格式: (APIServer pid=331) RuntimeError: ...
VLLM_PREFIX_RE = re.compile(
    r'^\([^)]+\)\s+'
    r'(?:(?:ERROR|INFO|WARNING)\s+\S+\s+\S+\s+\[\S+\] ?)?'
)

# 已知错误模式（对应 SKILL.md 5 个已知模式 + 扩展）
CRASH_PATTERNS = [
    # 模式 1: SM 架构不支持
    {
        "pattern": re.compile(
            r"(?:CUDA error|CUDAError).*?no kernel image.*?(?:flag_gems|triton)",
            re.IGNORECASE | re.DOTALL,
        ),
        "type": "sm_unsupported",
        "description": "Triton kernel 未编译对应 SM 架构",
    },
    # 模式 2: 算子参数不匹配
    {
        "pattern": re.compile(
            r"(?:RuntimeError|TypeError).*?(?:got an unexpected keyword argument|"
            r"missing \d+ required|takes \d+ positional)",
            re.IGNORECASE,
        ),
        "type": "signature_mismatch",
        "description": "FlagGems 算子签名与 PyTorch 版本不一致",
    },
    # 模式 3: 精度问题 NaN/Inf
    {
        "pattern": re.compile(
            r"(?:nan|inf).*?(?:detected|found|output)|"
            r"(?:loss|output).*?(?:nan|inf)",
            re.IGNORECASE,
        ),
        "type": "precision_nan",
        "description": "FlagGems 算子在特定输入下精度不足",
    },
    # 模式 4: DeepGemm 兼容性
    {
        "pattern": re.compile(
            r"(?:deep_gemm|deepgemm).*?(?:error|failed|crash)",
            re.IGNORECASE,
        ),
        "type": "deepgemm_compat",
        "description": "DeepGemm 与 FlagGems 算子冲突",
    },
    # 模式 5: Triton 编译失败
    {
        "pattern": re.compile(
            r"(?:triton|CompilationError).*?(?:compile|compilation).*?(?:fail|error)",
            re.IGNORECASE,
        ),
        "type": "triton_compile",
        "description": "Triton kernel 编译失败",
    },
    # 通用 CUDA error
    {
        "pattern": re.compile(
            r"(?:CUDA|cuda)(?:\s+)?(?:error|Error|ERROR)(?:\s*:)?\s*(.+)",
        ),
        "type": "cuda_error",
        "description": "CUDA 运行时错误",
    },
    # 通用 RuntimeError
    {
        "pattern": re.compile(
            r"RuntimeError:\s*(.+)",
        ),
        "type": "runtime_error",
        "description": "Python 运行时错误",
    },
]

# 从 stack trace / 错误信息中提取算子名的模式
OP_EXTRACT_PATTERNS = [
    # flag_gems.ops.softmax / flag_gems/ops/softmax.py
    re.compile(r"flag_gems[./]ops[./](\w+)"),
    # flag_gems.runtime.backend ... op_name
    re.compile(r"flag_gems.*?(?:backend|runtime).*?[/.](\w+)\.py"),
    # triton kernel: xxx_kernel / xxx_jit
    re.compile(r"triton.*?(\w+?)(?:_kernel|_jit)"),
    # vllm_fl.ops.oot.xxx
    re.compile(r"vllm_fl[./]ops[./](?:oot[./])?(\w+)"),
    # "Error in operator: xxx" / "failed op: xxx"
    re.compile(r"(?:operator|op|算子)[:\s]+['\"]?(\w+)['\"]?", re.IGNORECASE),
    # File "xxx/flag_gems/xxx/softmax.py"
    re.compile(r'File\s+"[^"]*flag_gems[^"]*?/(\w+)\.py"'),
    # Triton 内联代码中的 FlagGems 函数调用: xxx_func_tensor_scalar / xxx_func
    re.compile(r"(\w+?)_func(?:_tensor_scalar|_scalar_tensor|_tensor_tensor)?\s*\("),
]


def extract_ops_from_text(text: str, known_ops: set) -> List[str]:
    """从一段文本中提取所有可能的算子名"""
    found = set()
    for pattern in OP_EXTRACT_PATTERNS:
        for match in pattern.finditer(text):
            op_name = match.group(1).lower().strip("_")
            if op_name in known_ops:
                found.add(op_name)
            op_raw = match.group(1)
            if op_raw in known_ops:
                found.add(op_raw)
            # fallback: xxx_func_tensor_scalar 等 FlagGems 特征模式，即使不在 known_ops 中也提取
            # 仅限带 _tensor_scalar/_scalar_tensor/_tensor_tensor 后缀的匹配，避免 compile_func 等误报
            if "_func" in match.re.pattern and op_name and len(op_name) > 2:
                full_match = match.group(0)
                if any(s in full_match for s in ("_tensor_scalar", "_scalar_tensor", "_tensor_tensor")):
                    found.add(op_name)
    return sorted(found)


def analyze_crash_log(log_path: str, ops_file: Optional[str] = None) -> Dict[str, Any]:
    """分析崩溃日志，提取问题算子"""
    # 加载已知算子列表
    known_ops = set()
    all_known = set()
    if ops_file and os.path.isfile(ops_file):
        with open(ops_file, "r") as f:
            ops_data = json.load(f)
            if isinstance(ops_data, list):
                all_known = set(ops_data)
            elif isinstance(ops_data, dict) and "ops" in ops_data:
                all_known = set(ops_data["ops"])
    # 加上所有分组中的算子和 OOT 算子
    for group_ops in OPERATOR_GROUPS.values():
        all_known.update(group_ops)
    all_known.update(OOT_OPERATORS)
    known_ops = {op.lower() for op in all_known} | all_known

    # 读取日志
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_content = f.read()
    except OSError as e:
        return {"error": f"无法读取日志: {e}", "crashed_ops": [], "evidence": []}

    log_lines = log_content.split("\n")

    # 提取 traceback 块（支持 vLLM 带前缀的日志格式）
    # RuntimeError 的多行内容也纳入 traceback 块
    traceback_blocks = []
    current_tb = []
    in_traceback = False
    in_error_body = False
    for i, line in enumerate(log_lines):
        stripped = VLLM_PREFIX_RE.sub('', line)
        is_tb_start = "Traceback" in stripped and "most recent call" in stripped.lower()
        # 新 traceback 开始时，先保存当前块
        if is_tb_start:
            if current_tb:
                traceback_blocks.append(current_tb)
            in_traceback = True
            in_error_body = False
            current_tb = [(i + 1, stripped)]
        elif in_traceback:
            current_tb.append((i + 1, stripped))
            if stripped and not stripped.startswith(" ") and not stripped.startswith("\t") and "Error" in stripped:
                in_traceback = False
                in_error_body = True
        elif in_error_body:
            # RuntimeError 多行内容：缩进行、空行、带引号的内联代码
            if not stripped or stripped.startswith(" ") or stripped.startswith("\t") or stripped.startswith("'"):
                current_tb.append((i + 1, stripped))
            else:
                traceback_blocks.append(current_tb)
                current_tb = []
                in_error_body = False
        if (in_traceback or in_error_body) and len(current_tb) > 100:
            traceback_blocks.append(current_tb)
            current_tb = []
            in_traceback = False
            in_error_body = False
    if current_tb:
        traceback_blocks.append(current_tb)

    # 分析每个 traceback 块
    evidence = []
    crashed_ops = set()

    for tb in traceback_blocks:
        tb_text = "\n".join(line for _, line in tb)
        error_line = tb[-1][1] if tb else ""

        # 匹配已知错误模式
        matched_type = "unknown"
        matched_desc = "未知错误"
        for cp in CRASH_PATTERNS:
            if cp["pattern"].search(tb_text):
                matched_type = cp["type"]
                matched_desc = cp["description"]
                break

        # 提取算子名
        ops_found = extract_ops_from_text(tb_text, known_ops)
        crashed_ops.update(ops_found)

        if ops_found or matched_type != "unknown":
            evidence.append({
                "ops": ops_found,
                "line_start": tb[0][0],
                "line_end": tb[-1][0],
                "error_type": matched_type,
                "error_description": matched_desc,
                "error_message": error_line.strip()[:200],
            })

    # 也扫描非 traceback 区域的单行错误
    for i, line in enumerate(log_lines):
        stripped = VLLM_PREFIX_RE.sub('', line)
        for cp in CRASH_PATTERNS:
            if cp["pattern"].search(stripped):
                ops_found = extract_ops_from_text(stripped, known_ops)
                if ops_found:
                    crashed_ops.update(ops_found)
                    already_covered = any(
                        e["line_start"] <= i + 1 <= e["line_end"] for e in evidence
                    )
                    if not already_covered:
                        evidence.append({
                            "ops": ops_found,
                            "line_start": i + 1,
                            "line_end": i + 1,
                            "error_type": cp["type"],
                            "error_description": cp["description"],
                            "error_message": stripped.strip()[:200],
                        })

    return {
        "log_path": log_path,
        "log_lines": len(log_lines),
        "traceback_count": len(traceback_blocks),
        "crashed_ops": sorted(crashed_ops),
        "evidence": evidence,
        "suggestion": _crash_suggestion(crashed_ops, evidence),
    }


def _crash_suggestion(crashed_ops: set, evidence: list) -> str:
    """根据崩溃分析结果给出建议"""
    if not crashed_ops and not evidence:
        return "未从日志中识别出明确的算子问题，建议人工检查日志"
    if not crashed_ops and evidence:
        return "检测到错误但无法定位到具体算子，建议查看 evidence 中的错误行"
    ops_str = ", ".join(sorted(crashed_ops))
    return f"建议禁用以下算子后重启: {ops_str}"


# =============================================================================
# 场景 2：精度分组测试
# =============================================================================

def generate_accuracy_groups(
    ops_file: str,
    plugin_mode: bool = True,
) -> Dict[str, Any]:
    """按功能组生成累积禁用配置，供逐组精度测试

    测试策略：全量启用 → 逐组累积禁用 → 累积禁用后精度恢复即达标
    算子控制方式：与性能调优(operator_search.py)一致，写白名单控制文件
    """
    # 加载算子列表
    all_ops = []
    if os.path.isfile(ops_file):
        with open(ops_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_ops = data
            elif isinstance(data, dict) and "ops" in data:
                all_ops = data["ops"]

    if not all_ops:
        return {"error": "算子列表为空", "groups": []}

    all_ops_set = set(all_ops)

    # 将实际算子按组分类
    raw_groups = []
    classified_ops = set()

    # OOT 组（仅 plugin 模式）
    if plugin_mode:
        oot_actual = [op for op in OOT_OPERATORS if op in all_ops_set]
        if oot_actual:
            raw_groups.append({
                "name": "oot",
                "description": "OOT 高层融合算子（silu_and_mul, rms_norm 等）",
                "ops": oot_actual,
                "ops_count": len(oot_actual),
                "test_env": _build_group_env(
                    disable_group=oot_actual,
                    all_ops=all_ops,
                    plugin_mode=True,
                ),
            })
            classified_ops.update(oot_actual)

    # 功能组
    for group_name, group_template in OPERATOR_GROUPS.items():
        group_actual = [op for op in group_template if op in all_ops_set and op not in classified_ops]
        if not group_actual:
            continue
        raw_groups.append({
            "name": group_name,
            "description": _group_description(group_name),
            "ops": group_actual,
            "ops_count": len(group_actual),
            "test_env": _build_group_env(
                disable_group=group_actual,
                all_ops=all_ops,
                plugin_mode=plugin_mode,
            ),
        })
        classified_ops.update(group_actual)

    # 未分组算子
    unclassified = sorted(all_ops_set - classified_ops)
    if unclassified:
        raw_groups.append({
            "name": "other",
            "description": "未分类算子",
            "ops": unclassified,
            "ops_count": len(unclassified),
            "test_env": _build_group_env(
                disable_group=unclassified,
                all_ops=all_ops,
                plugin_mode=plugin_mode,
            ),
        })

    # 生成累积禁用配置（每轮在上一轮基础上追加禁用）
    cumulative_disabled = []
    groups = []
    for g in raw_groups:
        cumulative_disabled.extend(g["ops"])
        cumulative_enabled = sorted(all_ops_set - set(cumulative_disabled))
        if plugin_mode:
            cum_env = _build_group_env(
                disable_group=cumulative_disabled,
                all_ops=all_ops,
                plugin_mode=True,
            )
            cum_env["control_file"] = {"include": cumulative_enabled}
            cum_env["control_mode"] = "only_enable"
        else:
            cum_env = {
                "enable_ops": cumulative_enabled,
                "blacklist_ops": sorted(cumulative_disabled),
                "control_file": {"include": cumulative_enabled},
                "control_mode": "only_enable",
            }
        g["cumulative_test_env"] = cum_env
        g["cumulative_disabled_ops"] = sorted(cumulative_disabled)
        g["cumulative_disabled_count"] = len(cumulative_disabled)
        groups.append(g)

    # 基线 = 全量启用（即步骤4的 V2 配置，已有精度数据）
    baseline_env = {
        "USE_FLAGGEMS": "1",
        "VLLM_FL_PREFER_ENABLED": "true",
    }

    OPS_CONTROL_FILE = "/root/flaggems_ops_control.json"

    return {
        "total_ops": len(all_ops),
        "groups_count": len(groups),
        "groups": groups,
        "baseline_env": baseline_env,
        "baseline_env_inline": "USE_FLAGGEMS=1 VLLM_FL_PREFER_ENABLED=true",
        "test_procedure": (
            "1. baseline: 全量启用（V2 配置）→ 已有步骤4的 V2 精度数据\n"
            "2. 逐组累积禁用: 第1轮禁用组A，第2轮禁用组A+B，第3轮禁用组A+B+C → 每轮 fast_gpqa.py 评测\n"
            "3. 某轮累积禁用后精度恢复（下降 ≤5%）→ 达标即停，保留所有已累积禁用的算子\n"
            "4. 问题组内逐个算子排查（可选，缩小禁用范围）"
        ),
        "apply_method": (
            f"每轮使用 cumulative_test_env.control_file 写入 {OPS_CONTROL_FILE}，"
            "设置 FLAGGEMS_CONTROL_MODE=only_enable，通过 start_service.sh 启动服务"
            "（与性能调优 operator_search.py 一致的白名单控制路径）"
        ),
        "control_file_path": OPS_CONTROL_FILE,
    }


def _group_description(name: str) -> str:
    descs = {
        "compute": "计算算子（addmm, mm, matmul 等矩阵运算）",
        "memory": "内存算子（copy_, clone, zeros 等张量创建/复制）",
        "math": "数学算子（cos, sin, gelu, silu 等逐元素运算）",
        "index": "索引算子（gather, scatter, embedding 等）",
        "reduce": "归约算子（softmax, layer_norm, sum, mean 等）",
    }
    return descs.get(name, name)


def _build_group_env(
    disable_group: List[str],
    all_ops: List[str],
    plugin_mode: bool,
) -> Dict[str, str]:
    """生成"禁用指定组、其余全开"的单组独立配置

    注意：此函数生成单组独立配置（test_env），累积禁用配置由
    accuracy_groups() 的 cumulative_test_env 字段提供。
    """
    blacklist = sorted(disable_group)

    if plugin_mode:
        # 分 OOT 和 flagos 两层
        oot_blacklist = [op for op in blacklist if op in set(OOT_OPERATORS)]
        flagos_blacklist = [op for op in blacklist if op not in set(OOT_OPERATORS)]

        env = {
            "USE_FLAGGEMS": "1",
            "VLLM_FL_PREFER_ENABLED": "true",
        }
        if oot_blacklist:
            env["VLLM_FL_OOT_BLACKLIST"] = ",".join(oot_blacklist)
        if flagos_blacklist:
            env["VLLM_FL_FLAGOS_BLACKLIST"] = ",".join(flagos_blacklist)

        env["env_inline"] = " ".join(f"{k}={v}" for k, v in env.items() if k != "env_inline")
        return env

    # 非 plugin：返回 blacklist 列表（供 Layer 1-4 使用）
    all_ops_set = set(all_ops)
    enable_ops = sorted(all_ops_set - set(disable_group))
    return {
        "enable_ops": enable_ops,
        "blacklist_ops": blacklist,
    }


# =============================================================================
# 场景 3：性能 Profiling 预扫描
# =============================================================================

def run_profile(
    port: int,
    model_name: str,
    num_requests: int = 10,
    max_tokens: int = 64,
    baseline_log: Optional[str] = None,
    profiler_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """通过 torch.profiler 采集算子级耗时，定位性能热点

    方案：设置 VLLM_TORCH_PROFILER_DIR 后重启服务，
    发送请求触发 profiling，然后解析 trace 文件。
    如果 profiler 不可用，fallback 到请求级延迟对比。
    """
    result = {
        "method": "",
        "hotspots": [],
        "total_requests": num_requests,
        "suggestion": "",
    }

    # 方案 A：解析已有的 profiler trace
    if profiler_dir and os.path.isdir(profiler_dir):
        hotspots = _parse_profiler_traces(profiler_dir)
        if hotspots:
            result["method"] = "torch_profiler"
            result["hotspots"] = hotspots
            result["suggestion"] = _profile_suggestion(hotspots)
            return result

    # 方案 B：用 torch.profiler 直接采集
    try:
        hotspots = _collect_with_torch_profiler(port, model_name, num_requests, max_tokens)
        if hotspots:
            result["method"] = "torch_profiler_inline"
            result["hotspots"] = hotspots
            result["suggestion"] = _profile_suggestion(hotspots)
            return result
    except Exception as e:
        result["profiler_error"] = str(e)

    # 方案 C：如果有 baseline（native profile），做差值对比
    if baseline_log and os.path.isfile(baseline_log):
        try:
            with open(baseline_log, "r") as f:
                baseline = json.load(f)
            result["method"] = "baseline_comparison"
            result["note"] = "无法采集 profiler 数据，需配合 VLLM_TORCH_PROFILER_DIR 使用"
            return result
        except (json.JSONDecodeError, OSError):
            pass

    # 方案 D：Fallback — 提供配置指引
    result["method"] = "manual_setup_required"
    result["setup_instructions"] = {
        "step1": "设置环境变量 VLLM_TORCH_PROFILER_DIR=/flagos-workspace/traces/profiler",
        "step2": "重启服务",
        "step3": "发送少量请求触发 profiling",
        "step4": f"重新运行: python3 diagnose_ops.py profile --profiler-dir /flagos-workspace/traces/profiler --port {port} --model-name {model_name}",
    }
    result["suggestion"] = (
        "当前无法直接采集 profiler 数据。"
        "请按 setup_instructions 配置后重新运行。"
    )
    return result


def _parse_profiler_traces(profiler_dir: str) -> List[Dict[str, Any]]:
    """解析 torch.profiler 生成的 JSON trace 文件"""
    hotspots = defaultdict(lambda: {"total_us": 0, "calls": 0, "name": ""})

    # 查找最新的 trace 文件
    trace_files = []
    for f in os.listdir(profiler_dir):
        if f.endswith(".json") or f.endswith(".json.gz"):
            trace_files.append(os.path.join(profiler_dir, f))

    if not trace_files:
        return []

    # 按修改时间排序，取最新的
    trace_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    trace_file = trace_files[0]

    try:
        import gzip
        if trace_file.endswith(".gz"):
            with gzip.open(trace_file, "rt") as f:
                trace_data = json.load(f)
        else:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    # Chrome trace format: {"traceEvents": [...]}
    events = trace_data.get("traceEvents", [])
    if not events:
        return []

    # 汇总 GPU kernel 耗时
    for event in events:
        if event.get("cat") in ("kernel", "gpu_memcpy", "cuda_runtime"):
            name = event.get("name", "")
            dur = event.get("dur", 0)  # microseconds

            # 尝试匹配到 FlagGems 算子
            op_name = _trace_event_to_op(name)
            if op_name:
                hotspots[op_name]["total_us"] += dur
                hotspots[op_name]["calls"] += 1
                hotspots[op_name]["name"] = op_name

    # 转为排序列表
    result = []
    for op, data in hotspots.items():
        if data["calls"] > 0:
            result.append({
                "op": op,
                "avg_ms": round(data["total_us"] / data["calls"] / 1000, 3),
                "calls": data["calls"],
                "total_ms": round(data["total_us"] / 1000, 1),
            })

    result.sort(key=lambda x: x["total_ms"], reverse=True)
    return result[:20]  # Top 20


def _trace_event_to_op(kernel_name: str) -> Optional[str]:
    """将 GPU kernel 名称映射到 FlagGems 算子名"""
    name_lower = kernel_name.lower()

    # 直接匹配已知算子名
    all_known = set()
    for group_ops in OPERATOR_GROUPS.values():
        all_known.update(group_ops)
    all_known.update(OOT_OPERATORS)

    for op in all_known:
        if op.lower() in name_lower:
            return op

    # flag_gems / triton 标识
    if "flag_gems" in name_lower or "flaggems" in name_lower:
        for pattern in OP_EXTRACT_PATTERNS:
            m = pattern.search(kernel_name)
            if m:
                return m.group(1)

    return None


def _collect_with_torch_profiler(
    port: int, model_name: str, num_requests: int, max_tokens: int,
) -> List[Dict[str, Any]]:
    """尝试用 torch.profiler 直接采集（需在服务进程内运行）

    这里使用 vLLM 的 /start_profile 和 /stop_profile API（如果可用）
    """
    import urllib.request

    base_url = f"http://localhost:{port}"

    # 尝试 vLLM profiler API
    try:
        req = urllib.request.Request(f"{base_url}/start_profile", method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        raise RuntimeError("服务不支持 /start_profile API，需手动配置 VLLM_TORCH_PROFILER_DIR")

    # 发送请求
    for i in range(num_requests):
        payload = json.dumps({
            "model": model_name,
            "messages": [{"role": "user", "content": f"Count from 1 to {i+1}"}],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=60)
        except Exception:
            pass

    # 停止 profiling
    try:
        req = urllib.request.Request(f"{base_url}/stop_profile", method="POST")
        urllib.request.urlopen(req, timeout=10)
        time.sleep(2)  # 等待 trace 写入
    except Exception:
        pass

    # 查找生成的 trace 文件
    default_dir = "/flagos-workspace/traces/profiler"
    if os.path.isdir(default_dir):
        return _parse_profiler_traces(default_dir)

    # 尝试其他常见路径
    for d in ["/tmp/vllm_profile", "/tmp/profile", "/root/profile"]:
        if os.path.isdir(d):
            return _parse_profiler_traces(d)

    return []


def _profile_suggestion(hotspots: List[Dict]) -> str:
    """根据 profiling 结果给出建议"""
    if not hotspots:
        return "未检测到明显热点"

    # 取总耗时 top 3
    top3 = hotspots[:3]
    ops = [h["op"] for h in top3]
    detail = "; ".join(
        f"{h['op']}({h['total_ms']:.0f}ms, {h['calls']}次)" for h in top3
    )
    return (
        f"性能热点 Top3: {detail}。"
        f"建议优先搜索这些算子: {', '.join(ops)}"
    )


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="算子问题快速定位（三场景诊断）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="诊断场景")

    # 场景 1: crash-log
    crash_parser = subparsers.add_parser(
        "crash-log", help="从崩溃日志提取问题算子",
    )
    crash_parser.add_argument("--log-path", required=True, help="崩溃日志路径")
    crash_parser.add_argument("--ops-file", default=None, help="算子列表 JSON（提高匹配准确率）")
    crash_parser.add_argument("--json", action="store_true", help="JSON 输出")

    # 场景 2: accuracy-groups
    acc_parser = subparsers.add_parser(
        "accuracy-groups", help="按功能组生成精度测试配置",
    )
    acc_parser.add_argument("--ops-file", required=True, help="算子列表 JSON")
    acc_parser.add_argument("--plugin-mode", action="store_true", default=True,
                            help="Plugin 模式（默认启用）")
    acc_parser.add_argument("--no-plugin", dest="plugin_mode", action="store_false",
                            help="非 Plugin 模式")
    acc_parser.add_argument("--json", action="store_true", help="JSON 输出")

    # 场景 3: profile
    prof_parser = subparsers.add_parser(
        "profile", help="性能热点预扫描",
    )
    prof_parser.add_argument("--port", type=int, default=8000, help="服务端口")
    prof_parser.add_argument("--model-name", required=True, help="模型名称")
    prof_parser.add_argument("--num-requests", type=int, default=10, help="请求数量")
    prof_parser.add_argument("--max-tokens", type=int, default=64, help="每请求最大 token")
    prof_parser.add_argument("--baseline-log", default=None, help="Native profiling 基线文件")
    prof_parser.add_argument("--profiler-dir", default=None, help="已有的 profiler trace 目录")
    prof_parser.add_argument("--json", action="store_true", help="JSON 输出")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(2)

    if args.command == "crash-log":
        result = analyze_crash_log(args.log_path, args.ops_file)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_crash_report(result)
        sys.exit(0 if result["crashed_ops"] else 1)

    elif args.command == "accuracy-groups":
        result = generate_accuracy_groups(args.ops_file, args.plugin_mode)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_accuracy_report(result)
        sys.exit(0 if result.get("groups") else 1)

    elif args.command == "profile":
        result = run_profile(
            port=args.port,
            model_name=args.model_name,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            baseline_log=args.baseline_log,
            profiler_dir=args.profiler_dir,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_profile_report(result)
        sys.exit(0 if result.get("hotspots") else 1)


def _print_crash_report(result: Dict):
    """打印崩溃分析报告"""
    print("=" * 60)
    print("崩溃日志分析报告")
    print("=" * 60)
    print(f"日志文件: {result.get('log_path', 'N/A')}")
    print(f"日志行数: {result.get('log_lines', 0)}")
    print(f"Traceback 数量: {result.get('traceback_count', 0)}")
    print()

    ops = result.get("crashed_ops", [])
    if ops:
        print(f"问题算子 ({len(ops)} 个): {', '.join(ops)}")
    else:
        print("未识别出问题算子")

    print()
    for i, ev in enumerate(result.get("evidence", []), 1):
        print(f"  [{i}] 行 {ev['line_start']}-{ev['line_end']}: "
              f"{ev['error_type']} — {ev['error_description']}")
        if ev.get("ops"):
            print(f"      算子: {', '.join(ev['ops'])}")
        print(f"      {ev.get('error_message', '')[:100]}")
        print()

    print(f"建议: {result.get('suggestion', '')}")
    print("=" * 60)


def _print_accuracy_report(result: Dict):
    """打印精度分组报告"""
    print("=" * 60)
    print("精度分组测试配置")
    print("=" * 60)
    print(f"总算子数: {result.get('total_ops', 0)}")
    print(f"分组数: {result.get('groups_count', 0)}")
    print()

    for g in result.get("groups", []):
        env = g.get("test_env", {})
        inline = env.get("env_inline", "")
        print(f"  [{g['name']}] {g['description']} ({g['ops_count']} 个)")
        print(f"    算子: {', '.join(g['ops'][:10])}" + ("..." if len(g['ops']) > 10 else ""))
        if inline:
            print(f"    env: {inline[:120]}" + ("..." if len(inline) > 120 else ""))
        print()

    print("测试流程:")
    print(result.get("test_procedure", ""))
    print("=" * 60)


def _print_profile_report(result: Dict):
    """打印性能热点报告"""
    print("=" * 60)
    print("性能热点预扫描")
    print("=" * 60)
    print(f"采集方式: {result.get('method', 'N/A')}")
    print()

    hotspots = result.get("hotspots", [])
    if hotspots:
        print(f"{'排名':<4} {'算子':<20} {'平均(ms)':<10} {'调用次数':<8} {'总耗时(ms)':<12}")
        print(f"{'-'*4} {'-'*20} {'-'*10} {'-'*8} {'-'*12}")
        for i, h in enumerate(hotspots[:15], 1):
            print(f"{i:<4} {h['op']:<20} {h['avg_ms']:<10.3f} {h['calls']:<8} {h['total_ms']:<12.1f}")
    else:
        print("未检测到算子级耗时数据")

    setup = result.get("setup_instructions")
    if setup:
        print("\n配置指引:")
        for k, v in setup.items():
            print(f"  {k}: {v}")

    print(f"\n{result.get('suggestion', '')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
