#!/usr/bin/env python3
"""
generate_report.py — FlagOS 迁移流程报告生成工具

从 context.yaml / traces / results / logs 汇总生成迁移报告。
流程完成或中途均可调用，缺失数据自动跳过对应段落。

Usage:
    python3 generate_report.py                          # 文本报告输出到 stdout
    python3 generate_report.py --json                   # JSON 报告输出到 stdout
    python3 generate_report.py --summary                # 精简摘要（含调优过程）输出到 stdout
    python3 generate_report.py --output report.md       # 文本报告写入文件
    python3 generate_report.py --json --output report.json
    python3 generate_report.py --workspace /flagos-workspace

退出码: 0=成功, 1=无数据（context.yaml 不存在）
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_yaml(path: str) -> Optional[dict]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def read_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    except Exception:
        return []


def read_csv_table(path: str) -> Optional[str]:
    """读取 CSV 并转为 markdown 表格。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return None
        # performance_compare.py 可能直接输出 markdown 表格
        if content.startswith("|"):
            return content
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if len(rows) < 2:
            return None
        header = rows[0]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
    except Exception:
        return None


def parse_issue_md(content: str) -> Dict[str, str]:
    """从 issue markdown 提取标题、类型、URL 等。"""
    result = {"title": "", "type": "", "steps": "", "description": "", "actual": "", "url": "", "repo": ""}

    # 从 HTML 注释提取 type
    m = re.search(r'<!--\s*Type:\s*(\S+)\s*-->', content)
    if m:
        result["type"] = m.group(1)

    # 从 HTML 注释提取 repo
    m = re.search(r'<!--\s*Repo:\s*(\S+)\s*-->', content)
    if m:
        result["repo"] = m.group(1)

    # 从 HTML 注释或正文提取 GitHub issue URL
    m = re.search(r'(https://github\.com/[^\s)]+/issues/\d+)', content)
    if m:
        result["url"] = m.group(1)

    # 提取 ## Bug Report: xxx 标题
    m = re.search(r'## Bug Report:\s*(.+)', content)
    if m:
        result["title"] = m.group(1).strip()

    # 按 ### 分段提取
    sections = re.split(r'^### ', content, flags=re.MULTILINE)
    for sec in sections:
        if sec.startswith("Steps to Reproduce"):
            result["steps"] = sec.split("\n", 1)[1].strip() if "\n" in sec else ""
        elif sec.startswith("Description"):
            result["description"] = sec.split("\n", 1)[1].strip() if "\n" in sec else ""
        elif sec.startswith("Actual Behavior"):
            result["actual"] = sec.split("\n", 1)[1].strip() if "\n" in sec else ""

    return result


# =============================================================================
# 数据收集
# =============================================================================

def _ops_from_context(ctx: Optional[dict], ver: str) -> List[str]:
    """从 context.yaml 回退提取某版本的启用算子列表。

    results/operator_config*.json 依赖 docker cp 同步，缺失时会导致报告算子栏全空。
    context.yaml 由各段可靠同步（context_snapshot.yaml），作为回退数据源。
    返回启用算子列表（可能为空）。
    """
    if not isinstance(ctx, dict):
        return []
    versions = ctx.get("versions", {}) or {}
    vd = versions.get(ver, {}) if isinstance(versions, dict) else {}
    if isinstance(vd, dict):
        for key in ("enabled_ops", "current_enabled_ops", "kept_ops", "final_enabled_ops"):
            ops = vd.get(key)
            if isinstance(ops, list) and ops:
                return list(ops)
    # v2/v3 兜底：optimization.enabled_ops（减去 disabled）
    if ver in ("v2", "v3"):
        opt = ctx.get("optimization", {}) or {}
        if isinstance(opt, dict):
            enabled = opt.get("enabled_ops")
            if isinstance(enabled, list) and enabled:
                return list(enabled)
            initial = (ctx.get("service", {}) or {}).get("initial_operator_list", [])
            disabled = set(opt.get("disabled_ops", []) or [])
            if isinstance(initial, list) and initial:
                return [op for op in initial if op not in disabled]
    return []


class ReportData:
    """从工作目录收集所有可用数据。"""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.context: Optional[dict] = None
        self.gpqa_result: Optional[dict] = None
        self.native_perf: Optional[dict] = None
        self.flagos_perf: Optional[dict] = None
        self.optimized_perf: Optional[dict] = None
        self.perf_compare_table: Optional[str] = None
        self.traces: Dict[str, dict] = {}
        self.issues: Dict[str, List[str]] = {}
        self.issue_files: List[Dict[str, str]] = []
        self.oplists: Dict[str, List[str]] = {}
        self.op_config: Optional[dict] = None
        self.ops_control_initial: Optional[dict] = None
        self.workflow_complete = False
        # 多版本数据（新增）
        self.gpqa_versions: Dict[str, Optional[dict]] = {}   # {v1: gpqa_json, v2: ..., v3: ..., v5: ...}
        self.perf_versions: Dict[str, Optional[dict]] = {}   # {v1: perf_json, v2: ..., v3: ..., v5: ...}
        self.op_config_v3: Optional[dict] = None
        self.op_config_v4: Optional[dict] = None
        self.op_config_v5: Optional[dict] = None
        self.nv_baseline: Optional[dict] = None

    def collect(self) -> bool:
        """收集数据，返回 False 表示无 context.yaml。"""
        ctx_path = os.path.join(self.workspace, "shared", "context.yaml")
        self.context = read_yaml(ctx_path)
        if not self.context:
            # fallback: config/context_snapshot.yaml
            self.context = read_yaml(os.path.join(self.workspace, "config", "context_snapshot.yaml"))
        if not self.context:
            return False

        wf = self.context.get("workflow", {})
        self.workflow_complete = wf.get("all_done", False) is True

        r = os.path.join(self.workspace, "results")
        self.gpqa_result = read_json(os.path.join(r, "gpqa_result.json"))
        self.native_perf = read_json(os.path.join(r, "native_performance.json"))
        self.flagos_perf = read_json(os.path.join(r, "flagos_performance.json"))
        self.optimized_perf = read_json(os.path.join(r, "flagos_optimized.json"))
        self.perf_compare_table = read_csv_table(os.path.join(r, "performance_compare.csv"))

        # 多版本精度结果（优先新命名，fallback 旧命名）
        self.gpqa_versions["v1"] = read_json(os.path.join(r, "gpqa_v1.json")) or read_json(os.path.join(r, "gpqa_native.json"))
        self.gpqa_versions["v2"] = (
            read_json(os.path.join(r, "gpqa_v2.json"))
            or read_json(os.path.join(r, "gpqa_flagos_optimized.json"))
            or read_json(os.path.join(r, "gpqa_flagos.json"))
        )
        self.gpqa_versions["v3"] = read_json(os.path.join(r, "gpqa_v3.json")) or read_json(os.path.join(r, "gpqa_plugin.json"))
        self.gpqa_versions["v4"] = read_json(os.path.join(r, "gpqa_v4.json"))
        self.gpqa_versions["v5"] = read_json(os.path.join(r, "gpqa_v5.json"))

        # 多版本性能结果
        self.perf_versions["v1"] = read_json(os.path.join(r, "v1_performance.json")) or self.native_perf
        self.perf_versions["v2"] = (
            read_json(os.path.join(r, "v2_performance.json"))
            or self.optimized_perf
            or self.flagos_perf
        )
        self.perf_versions["v3"] = read_json(os.path.join(r, "v3_performance.json"))
        self.perf_versions["v4"] = read_json(os.path.join(r, "v4_performance.json"))
        self.perf_versions["v5"] = read_json(os.path.join(r, "v5_performance.json"))

        # V3/V4/V5 算子配置
        self.op_config_v3 = read_json(os.path.join(r, "operator_config_v3.json"))
        self.op_config_v4 = read_json(os.path.join(r, "operator_config_v4.json"))
        self.op_config_v5 = read_json(os.path.join(r, "operator_config_v5.json"))

        # NV 基线（无独立 V1 时精度基线回退来源）
        self.nv_baseline = (
            read_json(os.path.join(r, "nv_baseline.json"))
            or read_yaml(os.path.join(self.workspace, "shared", "nv_baseline.yaml"))
        )

        # traces
        traces_dir = os.path.join(self.workspace, "traces")
        if os.path.isdir(traces_dir):
            for f in sorted(Path(traces_dir).glob("*.json")):
                data = read_json(str(f))
                if data:
                    self.traces[f.stem] = data

        # issue logs
        for name in ("issues_startup", "issues_accuracy", "issues_performance"):
            lines = read_lines(os.path.join(self.workspace, "logs", f"{name}.log"))
            if lines:
                self.issues[name] = lines

        # issue markdown files (含复现步骤)
        # 排除 issue_report_/issue_data_ 中间文件，只读最终的 issue_{type}_{repo}_{ts}.md
        if os.path.isdir(r):
            for f in sorted(Path(r).glob("issue_*.md")):
                if f.name.startswith(("issue_report_", "issue_data_")):
                    continue
                content = read_text(str(f))
                if content:
                    self.issue_files.append(parse_issue_md(content))

        # oplists
        for name in ("initial_oplist", "accuracy_tuned_oplist", "final_oplist", "v4_oplist", "v5_oplist"):
            lines = read_lines(os.path.join(r, f"{name}.txt"))
            if lines:
                self.oplists[name] = lines

        # operator config (search log from operator_search.py)
        self.op_config = read_json(os.path.join(r, "operator_config.json"))

        # 初始控制文件（start_service.sh 保存的副本）
        self.ops_control_initial = read_json(os.path.join(r, "ops_control_initial.json"))

        return True

    # helpers
    def get(self, *keys, default=None):
        """Nested dict get from context."""
        d = self.context
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d

    def ledger_steps(self) -> List[dict]:
        steps = self.get("workflow_ledger", "steps", default=[])
        if not steps:
            return []

        # 步骤编号到名称的映射
        step_names = {
            "1": "容器准备", "2": "环境检测", "3": "服务启动",
            "4": "精度评测", "5": "精度算子调优", "6": "性能评测",
            "7": "性能算子调优", "8": "自动发布",
            "9": "Plugin 安装", "10": "Plugin 服务启动",
            "11": "Plugin 精度评测", "12": "Plugin 性能评测", "13": "Plugin 发布",
        }

        # 兼容 dict 格式（key=step_number）和 list 格式
        if isinstance(steps, dict):
            converted = []
            for k, v in sorted(steps.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 99):
                if isinstance(v, dict):
                    v.setdefault("step", str(k))
                    v.setdefault("name", step_names.get(str(k), f"步骤{k}"))
                    converted.append(v)
            steps = converted
        elif not isinstance(steps, list):
            return []

        for step in steps:
            if not isinstance(step, dict):
                continue
            step_key = step.get("step", "")
            # 补充 pending 步骤：从 traces fallback
            if step.get("status") == "pending" and step_key in self.traces:
                trace = self.traces[step_key]
                if isinstance(trace, dict) and trace.get("status"):
                    step["status"] = trace["status"]
                    step["duration_seconds"] = trace.get("duration_seconds", 0)
                    if trace.get("timestamp_start"):
                        step["started_at"] = trace["timestamp_start"]
                    if trace.get("timestamp_end"):
                        step["finished_at"] = trace["timestamp_end"]
            # 修正 duration=0 但有时间戳的情况
            if step.get("status") == "success" and not step.get("duration_seconds"):
                if step.get("started_at") and step.get("finished_at"):
                    try:
                        sa = step["started_at"].replace("Z", "+00:00")
                        ea = step["finished_at"].replace("Z", "+00:00")
                        start = datetime.fromisoformat(sa)
                        end = datetime.fromisoformat(ea)
                        step["duration_seconds"] = int((end - start).total_seconds())
                    except (ValueError, TypeError):
                        pass
                elif step_key in self.traces:
                    trace = self.traces[step_key]
                    if isinstance(trace, dict) and trace.get("duration_seconds"):
                        step["duration_seconds"] = trace["duration_seconds"]
        return steps


# =============================================================================
# 文本报告生成
# =============================================================================

def format_duration(seconds) -> str:
    if not seconds or not isinstance(seconds, (int, float)):
        return "-"
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _parse_oplist_txt(lines: List[str]) -> List[str]:
    """从 oplist txt 提取算子名（GEMS 名称）。
    格式: [DEBUG] flag_gems.ops.<module>.<func>: GEMS <NAME>
    返回: ['ZEROS', 'FULL', 'ZERO_', ...] 按出现顺序，不去重
    """
    ops = []
    for line in lines:
        m = re.match(r'\[DEBUG\] flag_gems\.\S+:\s*GEMS\s+(.+)', line)
        if m:
            ops.append(m.group(1).strip())
        elif not line.startswith('[DEBUG]') and line.strip():
            ops.append(line.strip())
    return ops


def _parse_oplist_to_func_names(lines: List[str]) -> List[str]:
    """从 oplist txt 提取小写函数名列表。
    格式: [DEBUG] flag_gems.ops.<module>.<func>: GEMS <NAME>
    或: [DEBUG] flag_gems.runtime.backend.<vendor>.<arch>.ops.<module>.<func>: GEMS <NAME>
    返回: ['zeros', 'arange', 'div', ...] 去重
    """
    funcs = set()
    for line in lines:
        m = re.match(
            r'\[DEBUG\] flag_gems\.(?:ops|runtime\.backend\.\w+\.\w+\.ops)\.(\w+)(?:\.(\w+))?:\s*GEMS\s+',
            line
        )
        if m:
            # 优先取第二级（func），没有则取第一级（module）
            func_name = m.group(2) or m.group(1)
            funcs.add(func_name.lower())
        elif not line.startswith('[DEBUG]') and line.strip():
            funcs.add(line.strip().lower())
    return sorted(funcs)


def _render_ops_comparison(config_ops: List[str], txt_lines: List[str], stage_label: str) -> List[str]:
    """生成配置 vs 运行时 txt 的并排对比文本行。

    config_ops: 配置文件中的算子列表（小写简化名，如 ['add', 'bitwise', ...])
    txt_lines: oplist txt 的原始行
    stage_label: 阶段标签（如 '初始启动'）
    """
    txt_gems_names = _parse_oplist_txt(txt_lines)

    lines = []
    lines.append(f"  ┌─ {stage_label} 算子对比 (配置: {len(config_ops)} 个 | 运行时 txt: {len(txt_gems_names)} 条) ─┐")
    lines.append(f"  │ {'配置文件 (include/enabled)':<34}│ {'运行时 txt (GEMS)':<34}│")
    lines.append(f"  │{'─' * 34}┼{'─' * 34}│")

    # 解析每条 txt 行的函数名和 GEMS 名
    # 格式: [DEBUG] flag_gems.ops.<module>.<func>: GEMS <NAME>
    # 或:   [DEBUG] flag_gems.runtime.backend.<vendor>.<arch>.ops.<module>.<func>: GEMS <NAME>
    txt_entries = []  # [(func_name, module_name, gems_name, line_index)]
    for i, line in enumerate(txt_lines):
        m = re.match(
            r'\[DEBUG\] flag_gems\.(?:ops|runtime\.backend\.\w+\.\w+\.ops)\.(\w+)\.(\w+):\s*GEMS\s+(.+)',
            line
        )
        if m:
            txt_entries.append((m.group(2).lower(), m.group(1).lower(), m.group(3).strip(), i))

    # 建立配置名 → txt GEMS 名的映射
    # 匹配规则：配置名与 txt 函数名或模块名匹配
    config_to_txt = {}  # config_name -> [GEMS names]
    matched_txt_indices = set()

    # 匹配时按名称长度降序（更具体的先匹配，避免 'cumsum' 抢走 'cumsum_out' 的条目）
    config_set_lower = {op.lower() for op in config_ops}
    for cfg_op in sorted(config_ops, key=len, reverse=True):
        matched_gems = []
        cfg_lower = cfg_op.lower()
        for func_name, mod_name, gems_name, idx in txt_entries:
            if idx in matched_txt_indices:
                continue
            # 精确匹配（函数名或模块名）
            if func_name == cfg_lower or mod_name == cfg_lower:
                matched_gems.append(gems_name)
                matched_txt_indices.add(idx)
                continue
            # 前缀匹配（如 bitwise → bitwise_and, bitwise_not; constant → constant_pad_nd）
            cfg_base = cfg_lower.rstrip('_')
            if func_name.startswith(cfg_base):
                rest = func_name[len(cfg_base):]
                if not rest or rest[0] in ('_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
                    # 跳过：如果 func_name 本身精确匹配另一个配置名
                    if func_name in config_set_lower and func_name != cfg_lower:
                        continue
                    matched_gems.append(gems_name)
                    matched_txt_indices.add(idx)
        config_to_txt[cfg_op] = matched_gems

    # 未匹配的 txt 行
    unmatched_txt = []
    for func_name, mod_name, gems_name, idx in txt_entries:
        if idx not in matched_txt_indices:
            unmatched_txt.append(gems_name)

    # 输出并排对比
    for cfg_op in sorted(config_ops):
        gems_list = config_to_txt.get(cfg_op, [])
        if gems_list:
            txt_col = ", ".join(gems_list)
            lines.append(f"  │ {cfg_op:<34}│ {txt_col:<34}│")
        else:
            lines.append(f"  │ {cfg_op:<34}│ {'(未加载)':<34}│ ←")

    # txt 中有但配置中没有的
    if unmatched_txt:
        lines.append(f"  │{'─' * 34}┼{'─' * 34}│")
        for gems_name in unmatched_txt:
            lines.append(f"  │ {'(无对应配置)':<34}│ {gems_name:<34}│ ←")

    lines.append(f"  └{'─' * 34}┴{'─' * 34}┘")

    # 差异摘要
    no_load = [op for op in sorted(config_ops) if not config_to_txt.get(op)]
    if no_load or unmatched_txt:
        diff_parts = []
        if no_load:
            diff_parts.append(f"配置有但未加载: {', '.join(no_load)}")
        if unmatched_txt:
            diff_parts.append(f"txt 有但配置无: {', '.join(unmatched_txt)}")
        lines.append(f"  差异: {'; '.join(diff_parts)}")
    else:
        lines.append(f"  差异: 无（完全匹配）")

    return lines


def _resolve_disabled_ops(data: ReportData) -> tuple:
    """解析禁用算子列表，支持多数据源 fallback。
    返回 (all_disabled, excluded_acc_list, excluded_perf_list, search_log, optimized_ratio)
    """
    optimization = data.get("optimization", default={}) or {}
    eval_sec = data.get("eval", default={}) or {}

    # 精度禁用算子
    excluded_acc = eval_sec.get("excluded_ops_accuracy") or (
        optimization.get("excluded_ops_accuracy") if isinstance(optimization, dict) else None
    )
    excluded_acc_list = excluded_acc if isinstance(excluded_acc, list) else (
        [excluded_acc] if excluded_acc else []
    )

    # 全部禁用算子 — fallback 链
    all_disabled = optimization.get("disabled_ops", []) if isinstance(optimization, dict) else []

    # Fallback 1: operator_config.json
    if not all_disabled and data.op_config and isinstance(data.op_config, dict):
        all_disabled = data.op_config.get("disabled_ops", [])

    # Fallback 2: perf.disabled_ops（可能是字符串）
    if not all_disabled:
        perf_disabled = data.get("perf", "disabled_ops", default=None)
        if perf_disabled:
            if isinstance(perf_disabled, list):
                all_disabled = perf_disabled
            elif isinstance(perf_disabled, str):
                all_disabled = [op.strip() for op in perf_disabled.split(",") if op.strip()]

    # 确保是 list
    if isinstance(all_disabled, str):
        all_disabled = [op.strip() for op in all_disabled.split(",") if op.strip()]

    excluded_perf_list = [op for op in all_disabled if op not in excluded_acc_list] if all_disabled else []

    # 搜索日志
    search_log = []
    if data.op_config and isinstance(data.op_config, dict):
        search_log = data.op_config.get("search_log", [])

    # 优化后 ratio
    optimized_ratio = None
    if data.op_config and isinstance(data.op_config, dict) and data.op_config.get("current_ratio"):
        r = data.op_config["current_ratio"]
        optimized_ratio = r * 100 if r < 2 else r
    elif isinstance(optimization, dict) and optimization.get("current_ratio"):
        r = optimization["current_ratio"]
        optimized_ratio = r * 100 if r < 2 else r
    elif data.get("perf", "optimized_ratio_pct", default=None):
        optimized_ratio = data.get("perf", "optimized_ratio_pct")

    return all_disabled, excluded_acc_list, excluded_perf_list, search_log, optimized_ratio


# =============================================================================
# GPU TFLOPS 查表 (BF16 peak TFLOPS)
# =============================================================================

GPU_TFLOPS_MAP = {
    # NVIDIA
    "A100": 312, "A100-80GB": 312, "A100-SXM": 312, "A100-PCIE": 312,
    "A800": 312, "A800-80GB": 312,
    "H100": 989, "H100-SXM": 989, "H100-PCIE": 756,
    "H800": 989, "H20": 296,
    "L40S": 366, "L40": 181, "L20": 239,
    "B200": 2250, "B100": 1750, "GB200": 2250,
    "4090": 165, "4080": 97, "3090": 71,
    # Ascend (华为)
    "910B": 296, "910ProB": 296, "910C": 320, "910A": 256,
    # Hygon DCU (海光)
    "Z100": 96, "Z100L": 96, "K100": 128, "K100_AI": 128,
    # Moore Threads (摩尔线程)
    "S4000": 96, "S80": 59,
    # MetaX (沐曦)
    "C500": 128, "N100": 96,
    # Cambricon (寒武纪)
    "MLU590": 96, "MLU370": 48,
}


def lookup_tflops(gpu_type: str) -> Optional[float]:
    """从 GPU 型号模糊匹配 TFLOPS 值。"""
    if not gpu_type:
        return None
    # 精确匹配
    normalized = gpu_type.strip().upper().replace(" ", "")
    for key, val in GPU_TFLOPS_MAP.items():
        if key.upper().replace(" ", "").replace("-", "") in normalized.replace("-", ""):
            return val
    # 数字子串匹配（如 "NVIDIA A100-SXM4-80GB" → 找到 "A100"）
    for key, val in sorted(GPU_TFLOPS_MAP.items(), key=lambda x: -len(x[0])):
        if key.upper() in normalized:
            return val
    return None


# =============================================================================
# 版本配置标签
# =============================================================================

VERSION_LABELS = {
    "v1": ("V1", "-", "基础版(FlagTree only)"),
    "v2": ("V2", "Pro", "gems+tree达标版"),
    "v3": ("V3", "Max", "gems+tree+plugin达标版"),
    "v4": ("V4", "Flag-express", "减算子提性能版(≥V3,近/超V1)"),
    "v5": ("V5", "Royal Megamaster", "最大化算子版"),
}


# =============================================================================
# 性能数据提取
# =============================================================================

def _extract_perf_metrics(perf_json: Optional[dict], concurrency: str = "64") -> Dict[str, Any]:
    """从 benchmark JSON 提取指定并发下的性能指标。"""
    if not perf_json or not isinstance(perf_json, dict):
        return {}
    # benchmark_runner 输出格式: {test_case_name: {concurrency: {metric: value}}}
    # 或 quick 模式: {concurrency: {metric: value}}
    for key, val in perf_json.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict):
            # 检查是否直接是 {concurrency: metrics}
            metrics = val.get(concurrency) or val.get(str(concurrency))
            if metrics and isinstance(metrics, dict):
                return metrics
            # 可能是 test_case 层：{tc_name: {conc: metrics}}
            for tc_key, tc_val in val.items():
                if isinstance(tc_val, dict):
                    m = tc_val.get(concurrency) or tc_val.get(str(concurrency))
                    if m and isinstance(m, dict):
                        return m
    # fallback: 找第一个包含 throughput 的 dict
    for key, val in perf_json.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and "Output token throughput (tok/s)" in val:
            return val
    return {}


def _count_ops_from_oplist(oplist_lines: List[str]) -> int:
    """从 oplist txt 行数统计算子数。"""
    return len([l for l in oplist_lines if l.strip() and not l.startswith("#")])


def generate_text_report(data: ReportData) -> str:
    """按照 FlagOS 标准报告模板生成 Markdown 报告。"""
    lines: List[str] = []

    # ── 版本定义说明 ──
    lines.append("> **自动化流程产出镜像版本：**")
    lines.append("> - V1：tree版本=基础版：只带flagtree不开启任何flagos组件")
    lines.append("> - V2：tree+gems=Pro版：开启flaggems且性能达到V1的80%，与V1的精度误差在5%以内")
    lines.append("> - V3：tree+gems+plugin=Max版：在V2的基础上安装使用plugin，且性能达到V1的80%，与V1的精度误差在5%以内")
    lines.append("> - V4：tree+gems+plugin=Flag-express版：在V3的基础上，性能表现超过V1版本")
    lines.append("> - V5：tree+gems(应开尽开)+plugin=Royal Megamaster交付版本：携带了所有的FlagOS组件，所有算子能开尽开，只要服务能够顺利启动就ok")
    lines.append("")

    # ── 上下文数据 ──
    ctx = data.context or {}
    wf = ctx.get("workflow", {}) or {}
    eval_sec = ctx.get("eval", {}) or {}
    insp = ctx.get("inspection", {}) or {}
    env = ctx.get("environment", {}) or {}
    gpu = ctx.get("gpu", {}) or {}
    model = ctx.get("model", {}) or {}
    runtime = ctx.get("runtime", {}) or {}
    timing = ctx.get("timing", {}) or {}
    release = ctx.get("release", {}) or {}
    optimization = ctx.get("optimization", {}) or {}
    plugin_wf = ctx.get("plugin_workflow", {}) or {}
    v5_exp = ctx.get("v5_expansion", {}) or {}
    container = ctx.get("container", {}) or {}
    core_pkgs = insp.get("core_packages", {}) or {}
    flag_pkgs = insp.get("flag_packages", {}) or {}

    # ── TFLOPS ──
    gpu_type = gpu.get("type", "")
    gpu_count = gpu.get("count", 0)
    tflops_per_gpu = lookup_tflops(gpu_type)
    tflops_str = str(tflops_per_gpu) if tflops_per_gpu else "-"
    total_tflops = tflops_per_gpu * gpu_count if tflops_per_gpu and gpu_count else None
    total_tflops_str = str(int(total_tflops)) if total_tflops else "-"

    # ── 算子列表 ──
    initial_oplist = data.oplists.get("initial_oplist", [])
    final_oplist = data.oplists.get("final_oplist", [])
    v5_oplist = data.oplists.get("v5_oplist", [])
    disabled_ops = optimization.get("disabled_ops", [])
    if isinstance(disabled_ops, str):
        disabled_ops = [op.strip() for op in disabled_ops.split(",") if op.strip()]

    initial_ops_control = data.ops_control_initial or {}
    include_list = initial_ops_control.get("include", [])

    # 发布用 oplist（优先 final，其次 initial）
    publish_oplist = final_oplist or initial_oplist

    # 步骤完成时间
    steps_timing = timing.get("steps", {}) or {}

    # ═══════════════════════════════════════════════
    # 基本信息
    # ═══════════════════════════════════════════════
    lines.append("# 基本信息")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|------|------|")
    project_name = ctx.get("project", {}).get("name", "") or model.get("project_name", "")
    lines.append(f"| 项目名称 | {project_name or '-'} |")
    lines.append(f"| 开始时间 | {timing.get('workflow_start', '-')} |")

    # gems+tree 上传时间 = 步骤8完成时间
    release_step = _find_ledger_step(data, "08_release")
    v2_upload_time = release_step.get("finished_at", "-") if release_step else "-"
    lines.append(f"| gems+tree版本上传时间 | {v2_upload_time} |")

    # plugin 上传时间 = 步骤13完成时间
    plugin_release_step = _find_ledger_step(data, "13_plugin_release")
    v3_upload_time = plugin_release_step.get("finished_at", "-") if plugin_release_step else "-"
    lines.append(f"| plugin上传时间 | {v3_upload_time} |")

    lines.append(f"| 模型 | {model.get('name', '-')} |")
    lines.append(f"| 模型类型 | {model.get('model_type', '') or '文本生成'} |")
    lines.append(f"| 权重来源 | {model.get('url', '') or model.get('name', '-')} |")
    lines.append(f"| 权重数制 | {model.get('dtype', '-')} |")
    lines.append(f"| 计算数制（默认权重数制） | {model.get('dtype', '-')} |")
    lines.append(f"| 推理框架后端 | {runtime.get('framework', 'vllm')} |")
    lines.append(f"| 推理框架后端版本 | {core_pkgs.get('vllm', '-')} |")
    lines.append(f"| 推理框架插件plugin-FL | {flag_pkgs.get('vllm_plugin', '-')} |")
    lines.append(f"| FlagGems版本 | {flag_pkgs.get('flaggems', '-')} |")
    lines.append(f"| Flagtree版本 | {env.get('flagtree_version', flag_pkgs.get('flagtree', '-'))} |")
    lines.append(f"| FlagCX版本 | {flag_pkgs.get('flagcx', '-')} |")
    lines.append(f"| 厂商 | {gpu.get('vendor', '-')} |")
    mem_gb = gpu.get("memory_gb", ctx.get("gpu", {}).get("memory_gb", "-"))
    lines.append(f"| GPU | {gpu_type} : {gpu_count} x {mem_gb}GB |")
    lines.append(f"| 容器 | {container.get('name', '-')} |")
    lines.append(f"| release自动化工具版本 | v0.1.0 |")

    # ═══════════════════════════════════════════════
    # 算子替换列表（按版本展示）
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("# 算子替换列表")

    # 各版本的算子数据
    # V1: 不开启 FlagGems → 无算子白名单
    # V2: 调优后的最终算子集 (final_oplist / operator_config)
    # V3: Plugin 调优后 (operator_config_v3)
    # V5: 扩展后 (v5_oplist / operator_config_v5)
    version_ops_data = {}

    # V1 — 无 FlagGems
    version_ops_data["v1"] = {"whitelist": [], "txt": []}

    # V0/V2 — 使用 final oplist（调优后）或 initial oplist
    v2_whitelist = []
    v2_txt_ops = []
    if data.op_config and isinstance(data.op_config, dict):
        v2_whitelist = data.op_config.get("enabled_ops", [])
    elif include_list:
        # 从 include 减去 disabled
        v2_whitelist = [op for op in include_list if op not in set(disabled_ops)]
    if not v2_whitelist and publish_oplist:
        # fallback: 从 final txt 解析函数名
        v2_whitelist = [l.strip() for l in publish_oplist if l.strip() and not l.startswith("[DEBUG]")]
        if not v2_whitelist:
            v2_whitelist = _parse_oplist_to_func_names(publish_oplist)
    if not v2_whitelist:
        # fallback: context.yaml（results 产物缺失时的可靠回退源）
        v2_whitelist = _ops_from_context(data.context, "v2")
    v2_txt_ops = _parse_oplist_to_func_names(publish_oplist) if publish_oplist else v2_whitelist
    version_ops_data["v2"] = {"whitelist": v2_whitelist, "txt": v2_txt_ops}

    # V3 — Plugin 调优后
    v3_whitelist = []
    if data.op_config_v3 and isinstance(data.op_config_v3, dict):
        v3_whitelist = data.op_config_v3.get("enabled_ops", [])
    if not v3_whitelist:
        v3_whitelist = _ops_from_context(data.context, "v3")
    if not v3_whitelist:
        v3_whitelist = v2_whitelist  # fallback: 与 V2 相同
    version_ops_data["v3"] = {"whitelist": v3_whitelist, "txt": v3_whitelist}

    # V4 — 减算子后（operator_reduction.py 产出 kept_ops）；无真实数据时留空（如实显示无数据，不套用 V3）
    v4_whitelist = []
    if data.op_config_v4 and isinstance(data.op_config_v4, dict):
        # operator_reduction 状态文件 current_enabled_ops，或结果 kept_ops
        v4_whitelist = (
            data.op_config_v4.get("current_enabled_ops")
            or data.op_config_v4.get("kept_ops")
            or []
        )
    if not v4_whitelist:
        # fallback: context.yaml versions.v4（results 产物缺失时）；仍为空则如实显示无数据
        v4_whitelist = _ops_from_context(data.context, "v4")
    v4_oplist = data.oplists.get("v4_oplist", [])
    v4_txt_ops = _parse_oplist_to_func_names(v4_oplist) if v4_oplist else v4_whitelist
    version_ops_data["v4"] = {"whitelist": v4_whitelist, "txt": v4_txt_ops}

    # V5 — 扩展后
    v5_whitelist = []
    v5_txt_ops = []
    if data.op_config_v5 and isinstance(data.op_config_v5, dict):
        v5_whitelist = data.op_config_v5.get("current_enabled_ops", [])
    elif v5_oplist:
        v5_whitelist = _parse_oplist_to_func_names(v5_oplist)
    if not v5_whitelist:
        v5_whitelist = _ops_from_context(data.context, "v5")
    if not v5_whitelist:
        v5_whitelist = v3_whitelist  # fallback
    v5_txt_ops = _parse_oplist_to_func_names(v5_oplist) if v5_oplist else v5_whitelist
    version_ops_data["v5"] = {"whitelist": v5_whitelist, "txt": v5_txt_ops}

    # 输出各版本
    for ver_key in ["v1", "v2", "v3", "v4", "v5"]:
        ver_label = VERSION_LABELS[ver_key][0]
        ops_data = version_ops_data[ver_key]
        lines.append("")
        lines.append(f"## {ver_label}")

        # 算子白名单
        lines.append("### 算子白名单")
        wl = ops_data["whitelist"]
        if wl:
            lines.append("```json")
            lines.append('"include": [')
            for i, op in enumerate(sorted(wl)):
                comma = "," if i < len(wl) - 1 else ""
                lines.append(f'    "{op}"{comma}')
            lines.append("]")
            lines.append("```")
        else:
            if ver_key == "v1":
                lines.append("（V1 不开启 FlagGems，无算子白名单）")
            else:
                lines.append("（无数据）")

        # 算子替换列表（txt）
        lines.append("### 算子替换列表（txt）")
        txt = ops_data["txt"]
        if txt:
            lines.append("```json")
            lines.append("[")
            for i, op in enumerate(sorted(txt)):
                comma = "," if i < len(txt) - 1 else ""
                lines.append(f'    "{op}"{comma}')
            lines.append("]")
            lines.append("```")
        else:
            if ver_key == "v1":
                lines.append("（V1 不开启 FlagGems，无算子替换）")
            else:
                lines.append("（无数据）")

    # ═══════════════════════════════════════════════
    # 评测结果 — 精度评测
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("# 评测结果")
    lines.append("")
    lines.append("## 精度评测")

    # V1-V5 版本的精度表
    for ver_key in ["v1", "v2", "v3", "v4", "v5"]:
        ver_label, config_label, _ = VERSION_LABELS.get(ver_key, (ver_key.upper(), "-", ""))
        gpqa = data.gpqa_versions.get(ver_key)
        lines.append("")
        lines.append(f"### {ver_label}")
        lines.append("| 数据集 | 评测条数 | 正确率(%) | 开启算子数 | FlagOS配置 |")
        lines.append("|--------|---------|-----------|-----------|-----------|")

        if gpqa:
            score = gpqa.get("score", "-")
            total = gpqa.get("total_questions", "-")
            # 算子数：V1 无 FlagGems，V2 从 final_oplist，V3 从 v3 config，V4 减算子后，V5 从 v5_expansion
            op_count = "-"
            if ver_key == "v1":
                op_count = "-"
                config_label = "-"
            elif ver_key == "v2":
                op_count = str(_count_ops_from_oplist(publish_oplist)) if publish_oplist else "-"
            elif ver_key == "v3":
                if data.op_config_v3:
                    op_count = str(len(data.op_config_v3.get("enabled_ops", [])))
                elif publish_oplist:
                    op_count = str(_count_ops_from_oplist(publish_oplist))
            elif ver_key == "v4":
                if data.op_config_v4:
                    v4_ops = (
                        data.op_config_v4.get("current_enabled_ops")
                        or data.op_config_v4.get("kept_ops")
                        or []
                    )
                    op_count = str(len(v4_ops)) if v4_ops else "-"
            elif ver_key == "v5":
                if v5_exp.get("final_enabled_count"):
                    op_count = str(v5_exp["final_enabled_count"])
                elif v5_oplist:
                    op_count = str(_count_ops_from_oplist(v5_oplist))
            lines.append(f"| GPQA_Diamond | {total} | {score} | {op_count} | {config_label} |")
        else:
            lines.append(f"| GPQA_Diamond | - | - | - | - |")

    # 精度结果对比
    lines.append("")
    lines.append("### 结果对比")
    lines.append("| 对比项 | 结果 |")
    lines.append("|--------|------|")
    v1_gpqa = data.gpqa_versions.get("v1")
    v1_score = v1_gpqa.get("score", 0) if v1_gpqa else (eval_sec.get("v1_score") or 0)
    for cmp_ver in ["v2", "v3", "v4", "v5"]:
        cmp_gpqa = data.gpqa_versions.get(cmp_ver)
        ver_label = VERSION_LABELS[cmp_ver][0]
        if not cmp_gpqa:
            lines.append(f"| V1 VS {ver_label} | - |")
        else:
            cmp_score = cmp_gpqa.get("score", 0)
            if v1_score and cmp_score:
                diff = abs(float(v1_score) - float(cmp_score))
                lines.append(f"| V1 VS {ver_label} | 精度偏差 {diff:.1f}% |")
            else:
                lines.append(f"| V1 VS {ver_label} | - |")
    # V2 VS V3
    v2_gpqa = data.gpqa_versions.get("v2")
    v3_gpqa = data.gpqa_versions.get("v3")
    if v2_gpqa and v3_gpqa:
        diff = abs(float(v2_gpqa.get("score", 0)) - float(v3_gpqa.get("score", 0)))
        lines.append(f"| V2 VS V3 | 精度偏差 {diff:.1f}% |")
    else:
        lines.append(f"| V2 VS V3 | - |")

    # ═══════════════════════════════════════════════
    # 评测结果 — 性能评测
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("## 性能评测")

    # 合成基线标注：无 V1 场景下 native_performance.json 由 synthesize_perf_baseline.py 生成
    _np_meta = (data.native_perf or {}).get("_meta", {})
    if _np_meta.get("synthetic"):
        lines.append("")
        lines.append(f"> ⚠️ **性能基线为合成值，非实测 V1**：V2 初始性能 ×{_np_meta.get('factor', 1.5)}"
                     f"（baseline_source: {_np_meta.get('baseline_source', 'v2_initial_x1.5')}）。"
                     f"本报告所有以 V1 为基准的性能比均基于该合成基线。")

    model_name = model.get("name", "-")
    vendor = gpu.get("vendor", "-")

    for ver_key in ["v1", "v2", "v3", "v4", "v5"]:
        ver_label, config_label, _ = VERSION_LABELS.get(ver_key, (ver_key.upper(), "-", ""))
        perf = data.perf_versions.get(ver_key)
        metrics = _extract_perf_metrics(perf) if perf else {}
        lines.append("")
        lines.append(f"### {ver_label}")
        lines.append("| 模型名 | 厂商 | TFLOPS（单卡） | 卡数 | TFLOPS（单卡） × 卡数 | 4k-1k 64并发 - mean TTFT（ms） | 4k-1k 64并发 - P99 TTFT（ms） | 4k-1k 64并发 - output toks/s | 4k-1k 64并发 - total tok/s | 4k-1k 64并发 - Mean TPOT (ms) | 开算子数 | FlagOS配置 | 单算力吞吐 |")
        lines.append("|--------|------|---------------|------|---------------------|------|------|------|------|------|------|------|------|")

        if not metrics:
            lines.append(f"| {model_name} | {vendor} | {tflops_str} | {gpu_count} | {total_tflops_str} | - | - | - | - | - | - | {config_label} | - |")
        else:
            ttft = metrics.get("Mean TTFT (ms)", "-")
            p99_ttft = metrics.get("P99 TTFT (ms)", "-")
            output_tps = metrics.get("Output token throughput (tok/s)", "-")
            total_tps_val = metrics.get("Total token throughput (tok/s)", "-")
            tpot = metrics.get("Mean TPOT (ms)", "-")

            # 算子数
            op_count = "-"
            if ver_key == "v1":
                op_count = "0"
                config_label = "-"
            elif ver_key == "v2":
                op_count = str(_count_ops_from_oplist(publish_oplist)) if publish_oplist else "-"
            elif ver_key == "v3":
                if data.op_config_v3:
                    op_count = str(len(data.op_config_v3.get("enabled_ops", [])))
                elif publish_oplist:
                    op_count = str(_count_ops_from_oplist(publish_oplist))
            elif ver_key == "v4":
                if data.op_config_v4:
                    v4_ops = (
                        data.op_config_v4.get("current_enabled_ops")
                        or data.op_config_v4.get("kept_ops")
                        or []
                    )
                    op_count = str(len(v4_ops)) if v4_ops else "-"
            elif ver_key == "v5":
                if v5_exp.get("final_enabled_count"):
                    op_count = str(v5_exp["final_enabled_count"])

            # 单算力吞吐
            throughput_per_tflops = "-"
            try:
                if total_tflops and total_tps_val and total_tps_val != "-":
                    throughput_per_tflops = f"{float(total_tps_val) / total_tflops:.6f}"
            except (ValueError, TypeError, ZeroDivisionError):
                pass

            lines.append(f"| {model_name} | {vendor} | {tflops_str} | {gpu_count} | {total_tflops_str} | {ttft} | {p99_ttft} | {output_tps} | {total_tps_val} | {tpot} | {op_count} | {config_label} | {throughput_per_tflops} |")

    # 性能结果对比
    lines.append("")
    lines.append("### 结果对比")
    lines.append("| 对比项 | 结果 |")
    lines.append("|--------|------|")
    v1_perf = data.perf_versions.get("v1")
    v1_metrics = _extract_perf_metrics(v1_perf) if v1_perf else {}
    v1_total = v1_metrics.get("Total token throughput (tok/s)", 0) if v1_metrics else 0
    for cmp_ver in ["v2", "v3", "v4", "v5"]:
        ver_label = VERSION_LABELS[cmp_ver][0]
        cmp_perf = data.perf_versions.get(cmp_ver)
        cmp_metrics = _extract_perf_metrics(cmp_perf) if cmp_perf else {}
        if not cmp_metrics:
            lines.append(f"| V1 VS {ver_label} | - |")
        else:
            cmp_total = cmp_metrics.get("Total token throughput (tok/s)", 0)
            try:
                if v1_total and cmp_total and float(v1_total) > 0:
                    ratio = float(cmp_total) / float(v1_total) * 100
                    lines.append(f"| V1 VS {ver_label} | 性能比 {ratio:.1f}% |")
                else:
                    lines.append(f"| V1 VS {ver_label} | - |")
            except (ValueError, TypeError):
                lines.append(f"| V1 VS {ver_label} | - |")
    # V2 VS V3
    v2_perf_m = _extract_perf_metrics(data.perf_versions.get("v2")) if data.perf_versions.get("v2") else {}
    v3_perf_m = _extract_perf_metrics(data.perf_versions.get("v3")) if data.perf_versions.get("v3") else {}
    if v2_perf_m and v3_perf_m:
        try:
            v2_t = float(v2_perf_m.get("Total token throughput (tok/s)", 0))
            v3_t = float(v3_perf_m.get("Total token throughput (tok/s)", 0))
            if v2_t > 0:
                ratio = v3_t / v2_t * 100
                lines.append(f"| V2 VS V3 | 性能比 {ratio:.1f}% |")
            else:
                lines.append(f"| V2 VS V3 | - |")
        except (ValueError, TypeError):
            lines.append(f"| V2 VS V3 | - |")
    else:
        lines.append(f"| V2 VS V3 | - |")

    # ═══════════════════════════════════════════════
    # 流程耗时与消费
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("# 流程耗时与消费")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|------|------|")
    total_dur = timing.get("total_duration_seconds", 0)
    lines.append(f"| 流程耗时 | {format_duration(total_dur) if total_dur else '-'} |")
    lines.append(f"| 流程消费 | — |")

    # ═══════════════════════════════════════════════
    # 发布信息
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("# 发布信息")
    lines.append("")
    lines.append("- Harbor 镜像")

    # V1（手动发布，自动化不产出）
    lines.append("  - V1：（阶段一手动发布）")

    # V2
    v2_harbor = release.get("v2_harbor_image", "") or release.get("harbor_image", "")
    image_reg = ctx.get("image", {}).get("registry_url", "")
    if not v2_harbor and image_reg and "-v2" in image_reg:
        v2_harbor = image_reg
    elif not v2_harbor and image_reg and "-v3" not in image_reg and "-v5" not in image_reg and "-plugin" not in image_reg:
        v2_harbor = image_reg  # 旧格式无后缀 = V2
    lines.append(f"  - V2：{v2_harbor or '-'}")

    # V3
    v3_harbor = release.get("v3_harbor_image", "") or plugin_wf.get("plugin_image_url", "")
    lines.append(f"  - V3：{v3_harbor or '-'}")

    # V4
    versions_ctx = ctx.get("versions", {}) or {}
    v4_harbor = (
        release.get("v4_harbor_image", "")
        or (versions_ctx.get("v4", {}) or {}).get("harbor_image", "")
    )
    lines.append(f"  - V4：{v4_harbor or '-'}")

    # V5
    v5_harbor = (
        release.get("v5_harbor_image", "")
        or v5_exp.get("image_url", "")
        or (versions_ctx.get("v5", {}) or {}).get("harbor_image", "")
    )
    lines.append(f"  - V5：{v5_harbor or '-'}")

    lines.append("")
    ms_url = release.get("modelscope_url", "")
    hf_url = release.get("huggingface_url", "")
    lines.append(f"- ModelScope: {ms_url or '-'}")
    lines.append(f"- HuggingFace: {hf_url or '-'}")

    # ═══════════════════════════════════════════════
    # 结论
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("# 结论")
    lines.append("")

    qualified = wf.get("qualified")
    acc_ok = wf.get("accuracy_ok")
    perf_ok = wf.get("performance_ok")
    svc_ok = wf.get("service_ok")

    if qualified is True:
        lines.append("- 流自动化程结论：✅ 流程已达标")
    elif qualified is False:
        reasons = []
        if not svc_ok:
            reasons.append("服务启动失败")
        if not acc_ok:
            reasons.append("精度不达标")
        if not perf_ok:
            reasons.append("性能不达标")
        lines.append(f"- 流自动化程结论：❌ 未达标 ({', '.join(reasons) if reasons else '未知原因'})")
    elif not data.workflow_complete:
        lines.append("- 流自动化程结论：⏳ 流程未完成")
    else:
        lines.append("- 流自动化程结论：N/A")

    # gems+tree 上传
    if v2_harbor:
        lines.append(f"- gems+tree上传正常：✅")
    else:
        lines.append(f"- gems+tree上传正常：❌")

    # Plugin 上传
    plugin_triggered = plugin_wf.get("triggered", False)
    plugin_qualified = plugin_wf.get("qualified")
    plugin_crash = plugin_wf.get("crash_stopped", False)
    if not plugin_triggered:
        lines.append(f"- Plugin 上传正常：⏭ 未触发")
    elif plugin_crash:
        lines.append(f"- Plugin 上传正常：❌ 崩溃中止")
    elif plugin_qualified is True:
        lines.append(f"- Plugin 上传正常：✅")
    elif plugin_qualified is False:
        lines.append(f"- Plugin 上传正常：❌ 不合格")
    else:
        lines.append(f"- Plugin 上传正常：-")

    lines.append(f"- 该模型目前是否达到正常模型发布标准（是否安装plugin成功）：{'是' if plugin_wf.get('service_ok') else '否'}")

    # ═══════════════════════════════════════════════
    # 提交的 Issue
    # ═══════════════════════════════════════════════
    submitted_issues = data.get("issues", "submitted", default=[]) or []
    type_label = {
        "operator-crash": "算子崩溃",
        "accuracy-zero": "精度归零",
        "accuracy-degraded": "精度下降",
        "performance-degraded": "性能下降",
        "flagtree-error": "FlagTree 错误",
        "plugin-error": "Plugin 错误",
    }

    if data.issue_files or submitted_issues:
        lines.append("")
        lines.append("# 提交的 Issue")

        # 合并两个来源：issue_files（本地 md 解析）+ submitted_issues（context.yaml 记录）
        # 构建 issue 列表，优先从 submitted_issues 获取 URL
        submitted_url_map: Dict[str, str] = {}  # title -> url
        if submitted_issues and isinstance(submitted_issues, list):
            for iss in submitted_issues:
                if isinstance(iss, dict):
                    t = iss.get("title", "")
                    u = iss.get("url", "")
                    if t and u:
                        submitted_url_map[t] = u

        issue_idx = 0
        for issue in data.issue_files:
            issue_idx += 1
            itype = type_label.get(issue["type"], issue["type"])
            title = issue["title"] or "未知问题"
            # URL 优先级：issue md 内提取 > submitted_issues 记录
            url = issue.get("url", "") or submitted_url_map.get(title, "")
            if url:
                lines.append(f"{issue_idx}. [{itype}] {title}  ")
                lines.append(f"   {url}")
            else:
                lines.append(f"{issue_idx}. [{itype}] {title}")

        # 补充 submitted_issues 中有 URL 但 issue_files 中没有的条目
        seen_titles = {iss["title"] for iss in data.issue_files if iss.get("title")}
        for iss in (submitted_issues or []):
            if isinstance(iss, dict):
                title = iss.get("title", "")
                url = iss.get("url", "")
                if title and title not in seen_titles and url:
                    issue_idx += 1
                    itype_raw = iss.get("type", "")
                    itype = type_label.get(itype_raw, itype_raw)
                    lines.append(f"{issue_idx}. [{itype}] {title}  ")
                    lines.append(f"   {url}")
            elif isinstance(iss, str) and iss not in seen_titles:
                issue_idx += 1
                lines.append(f"{issue_idx}. {iss}")

    # ═══════════════════════════════════════════════
    # 发布的 README
    # ═══════════════════════════════════════════════
    readme_path = os.path.join(data.workspace, "results", "README.md")
    readme_content = read_text(readme_path)
    if readme_content:
        lines.append("")
        lines.append("# 发布的README")
        lines.append("")
        lines.append(readme_content)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"报告生成时间：{datetime.now().strftime('%Y.%m.%d')}")

    return "\n".join(lines)


def _find_ledger_step(data: ReportData, step_key: str) -> Dict[str, Any]:
    """从 ledger 中找到指定步骤。"""
    steps = data.ledger_steps()
    for s in steps:
        if s.get("step", "").startswith(step_key) or step_key in s.get("step", ""):
            return s
    return {}




# =============================================================================
# JSON 报告辅助函数
# =============================================================================

def _build_operator_tuning_json(data: ReportData, wf: dict, eval_sec: dict) -> dict:
    all_disabled, excluded_acc_list, excluded_perf_list, search_log, optimized_ratio = _resolve_disabled_ops(data)

    search_log_summary = []
    for i, entry in enumerate(search_log, 1):
        disabled_op = entry.get("op", entry.get("disabled_op", entry.get("tested_op", "")))
        ratio = entry.get("ratio") or entry.get("min_ratio")
        if ratio is not None and ratio < 2:
            ratio = ratio * 100
        passed = entry.get("passed", entry.get("met_target", False))
        if not passed and ratio is not None:
            target = data.op_config.get("target_ratio", 0.8) if data.op_config else 0.8
            target_pct = target * 100 if target < 1 else target
            passed = ratio >= target_pct
        search_log_summary.append({
            "round": i,
            "disabled_op": disabled_op,
            "ratio": ratio,
            "passed": passed,
        })

    return {
        "accuracy_tuning_triggered": bool(excluded_acc_list),
        "accuracy_tuning_ok": wf.get("accuracy_ok", False),
        "excluded_accuracy": excluded_acc_list,
        "performance_tuning_triggered": bool(excluded_perf_list),
        "performance_tuning_ok": wf.get("performance_ok", False),
        "excluded_performance": excluded_perf_list,
        "optimized_ratio": optimized_ratio,
        "search_log_summary": search_log_summary,
        "initial_oplist": data.oplists.get("initial_oplist", []),
        "ops_control_initial": data.ops_control_initial.get("include", []) if data.ops_control_initial else [],
        "accuracy_tuned_oplist": data.oplists.get("accuracy_tuned_oplist", []),
        "final_oplist": data.oplists.get("final_oplist", []),
        "config_persisted": wf.get("config_persisted", False),
    }


def _build_service_crash_json(data: ReportData, wf: dict) -> dict:
    service_ok = wf.get("service_ok")
    startup_trace = data.traces.get("03_service_startup")
    crashed_ops = []
    recovery = None

    if startup_trace and isinstance(startup_trace, dict):
        for a in startup_trace.get("actions", []):
            action_str = str(a.get("action", "")).lower()
            if "crash" in action_str or "diagnose" in action_str:
                summary = a.get("output_summary", "")
                if isinstance(summary, str) and summary:
                    crashed_ops.append(summary)
                if a.get("status") == "success":
                    recovery = "success"
                elif a.get("status") == "failed":
                    recovery = "failed"

    if service_ok is False and recovery is None:
        recovery = "failed"
    elif service_ok is True and crashed_ops and recovery is None:
        recovery = "success"

    return {
        "crashed": bool(crashed_ops) or service_ok is False,
        "crashed_ops": crashed_ops,
        "recovery": recovery,
        "service_ok": service_ok,
    }


def _build_plugin_json(data: ReportData) -> dict:
    plugin_wf = data.get("plugin_workflow", default={}) or {}
    if not plugin_wf.get("triggered", False):
        return {"triggered": False}
    plugin_install = data.get("plugin_install", default={}) or {}
    eval_sec = data.get("eval", default={}) or {}
    v1_score = eval_sec.get("v1_score")
    return {
        "triggered": True,
        "install": {
            "version": plugin_install.get("version", ""),
            "method": plugin_install.get("install_method", ""),
            "success": plugin_install.get("success", plugin_install.get("installed", False)),
        },
        "accuracy": {
            "plugin_score": plugin_wf.get("plugin_score"),
            "v1_score": v1_score,
            "diff": plugin_wf.get("accuracy_diff"),
            "ok": plugin_wf.get("accuracy_ok"),
        },
        "performance": {
            "ratio": plugin_wf.get("performance_ratio"),
            "ok": plugin_wf.get("performance_ok"),
        },
        "release": {
            "plugin_image": plugin_wf.get("plugin_image", ""),
            "released": plugin_wf.get("released", False),
            "qualified": plugin_wf.get("qualified"),
        },
        "service_ok": plugin_wf.get("service_ok"),
    }


# =============================================================================
# JSON 报告生成
# =============================================================================

def generate_json_report(data: ReportData) -> dict:
    wf = data.get("workflow", default={}) or {}
    eval_sec = data.get("eval", default={}) or {}
    perf = data.get("perf", default={}) or {}
    release = data.get("release", default={}) or {}
    optimization = data.get("optimization", default={}) or {}

    # 精度数据（context 优先，gpqa_result.json 补充）
    v1_score = eval_sec.get("v1_score")
    v2_score = eval_sec.get("v2_score")
    deviation = eval_sec.get("deviation") or eval_sec.get("accuracy_diff")
    if data.gpqa_result and v1_score is None:
        v1_score = data.gpqa_result.get("v1_score") or data.gpqa_result.get("native_score")
        v2_score = data.gpqa_result.get("v2_score") or data.gpqa_result.get("flagos_score")
        deviation = data.gpqa_result.get("deviation")

    # 步骤状态
    steps_summary = []
    for s in data.ledger_steps():
        steps_summary.append({
            "step": s.get("step", ""),
            "name": s.get("name", ""),
            "status": s.get("status", "pending"),
            "duration_seconds": s.get("duration_seconds", 0),
            "notes": s.get("notes", ""),
            "fail_reason": s.get("fail_reason", ""),
            "skip_reason": s.get("skip_reason", ""),
        })

    # 问题统计
    issues_summary = {}
    for key, entries in data.issues.items():
        count = sum(1 for e in entries if e.startswith("["))
        issues_summary[key] = count

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "workflow_complete": data.workflow_complete,
        "model": {
            "name": data.get("model", "name", default=""),
            "container_path": data.get("model", "container_path", default=""),
        },
        "container": {
            "name": data.get("container", "name", default=""),
        },
        "gpu": {
            "count": data.get("gpu", "count", default=0),
            "type": data.get("gpu", "type", default=""),
            "vendor": data.get("gpu", "vendor", default=""),
        },
        "environment": {
            "env_type": data.get("environment", "env_type", default=""),
        },
        "accuracy": {
            "v1_score": v1_score,
            "v2_score": v2_score,
            "deviation": deviation,
            "threshold": eval_sec.get("accuracy_threshold", 5.0),
            "ok": wf.get("accuracy_ok"),
        },
        "performance": {
            "min_ratio": perf.get("ratio_pct") if perf.get("ratio_pct") is not None else perf.get("min_ratio"),
            "optimized_ratio": perf.get("optimized_ratio_pct"),
            "target_ratio": optimization.get("target_ratio", 80.0),
            "ok": wf.get("performance_ok"),
        },
        "operator_tuning": _build_operator_tuning_json(data, wf, eval_sec),
        "service_crash": _build_service_crash_json(data, wf),
        "release": {
            "qualified": wf.get("qualified"),
            "harbor_image": release.get("harbor_image", ""),
            "modelscope_url": release.get("modelscope_url", ""),
            "huggingface_url": release.get("huggingface_url", ""),
        },
        "plugin": _build_plugin_json(data),
        "steps": steps_summary,
        "issues": {
            "summary": issues_summary,
            "submitted": data.get("issues", "submitted", default=[]) or [],
            "details": [
                {
                    "title": issue["title"],
                    "type": issue["type"],
                    "description": issue["description"].split("\n")[0].strip() if issue["description"] else "",
                    "steps_to_reproduce": issue["steps"],
                    "actual_behavior": issue["actual"],
                }
                for issue in data.issue_files
            ],
        },
        "_meta": {
            "generated_at": "报告生成时间 (ISO 8601)",
            "workflow_complete": "全流程是否已完成",
            "accuracy.ok": "主流程精度是否达标（含调优后结果）",
            "performance.ok": "主流程性能是否达标（含调优后结果）",
            "operator_tuning": "算子调优详情（含搜索过程、各阶段算子列表）",
            "service_crash": "服务崩溃诊断（崩溃算子、恢复状态）",
            "issues.submitted": "已提交到 GitHub 的 issue 列表（含 URL）",
            "release.qualified": "主流程综合判定 = service_ok AND accuracy_ok AND performance_ok",
            "plugin": "Plugin 流程独立判定，不影响主流程 release.qualified",
            "steps[].status": "pending / in_progress / success / failed / skipped",
        },
    }
    return report




# =============================================================================
# Summary 报告生成（精简版，含调优过程详情）
# =============================================================================

def generate_summary(data: ReportData) -> str:
    lines: List[str] = []
    lines.append("═══ FlagOS 迁移摘要 ═══")

    model = data.get("model", "name", default="N/A")
    gpu_count = data.get("gpu", "count", default="?")
    gpu_type = data.get("gpu", "type", default="?")
    env_type = data.get("environment", "env_type", default="N/A")
    plugin_triggered = data.get("plugin_workflow", "triggered", default=False)
    if plugin_triggered and "plugin" in str(env_type):
        main_env = env_type.replace("_plugin_", "_").replace("vllm_plugin_flaggems", "vllm_flaggems")
        env_display = f"{main_env} (主流程) / {env_type} (Plugin)"
    else:
        env_display = env_type

    lines.append(f"模型: {model} | GPU: {gpu_count}x{gpu_type} | 环境: {env_display}")
    lines.append("")

    wf = data.get("workflow", default={}) or {}
    eval_sec = data.get("eval", default={}) or {}
    all_disabled, excluded_acc_list, excluded_perf_list, search_log, optimized_ratio = _resolve_disabled_ops(data)

    # 精度评测
    v1_score = eval_sec.get("v1_score")
    v2_score = eval_sec.get("v2_score")
    deviation = eval_sec.get("deviation") or eval_sec.get("accuracy_diff")
    if data.gpqa_result and v1_score is None:
        v1_score = data.gpqa_result.get("v1_score") or data.gpqa_result.get("native_score")
        v2_score = data.gpqa_result.get("v2_score") or data.gpqa_result.get("flagos_score")
        deviation = data.gpqa_result.get("deviation")

    acc_ok = wf.get("accuracy_ok")
    lines.append("精度评测:")
    if v1_score is not None or v2_score is not None:
        ok_str = "ok" if acc_ok else "不达标"
        dev_str = f" 偏差={deviation}%" if deviation is not None else ""
        lines.append(f"  V1={v1_score}% V2={v2_score}%{dev_str} → {ok_str}")
    else:
        lines.append("  无数据")

    # 精度调优
    acc_triggered = bool(excluded_acc_list)
    if acc_triggered:
        v3_score = eval_sec.get("v3_score")
        lines.append(f"  调优: 触发 → 禁用 {', '.join(excluded_acc_list)}")
        if v3_score is not None:
            lines.append(f"    V3={v3_score}%")
    else:
        lines.append("  调优: 未触发")
    lines.append("")

    # 性能评测
    perf_data = data.get("perf", default={}) or {}
    v1_tps = perf_data.get("v1_output_tps") or (data.get("native_perf", "output_throughput", default=None))
    v2_tps = perf_data.get("v2_output_tps") or (data.get("flagos_full_perf", "output_throughput", default=None))
    ratio_pct = perf_data.get("ratio_pct")
    perf_ok = wf.get("performance_ok")

    lines.append("性能评测:")
    if v1_tps is not None or v2_tps is not None:
        ok_str = "达标" if perf_ok else "不达标"
        ratio_str = f"ratio={ratio_pct}%" if ratio_pct is not None else ""
        lines.append(f"  V1={v1_tps} V2={v2_tps} tok/s, {ratio_str} → {ok_str}")
    else:
        lines.append("  无数据")

    # 性能调优
    perf_triggered = bool(excluded_perf_list)
    if perf_triggered:
        status = "达标" if perf_ok else "未达标"
        ratio_info = f" ({optimized_ratio:.1f}%)" if optimized_ratio else ""
        lines.append(f"  调优: 触发 → {status}{ratio_info}")
        if search_log:
            lines.append(f"    搜索过程 ({len(search_log)} 轮):")
            for i, entry in enumerate(search_log, 1):
                disabled_op = entry.get("op", entry.get("disabled_op", entry.get("tested_op", "?")))
                ratio = entry.get("ratio") or entry.get("min_ratio")
                if ratio is not None and ratio < 2:
                    ratio = ratio * 100
                passed = entry.get("passed", entry.get("met_target", False))
                if not passed and ratio is not None:
                    target = data.op_config.get("target_ratio", 0.8) if data.op_config else 0.8
                    target_pct = target * 100 if target < 1 else target
                    passed = ratio >= target_pct
                ratio_str = f"{ratio:.1f}%" if ratio is not None else "N/A"
                result_str = "达标" if passed else "未达标"
                lines.append(f"      第{i}轮: 禁用 {disabled_op} → ratio {ratio_str} ({result_str})")
        lines.append(f"    最终禁用: {', '.join(excluded_perf_list)}")
        if data.op_config and isinstance(data.op_config, dict):
            enabled_count = len(data.op_config.get("enabled_ops", []))
            if enabled_count:
                lines.append(f"    启用算子数: {enabled_count} 个")
    else:
        lines.append("  调优: 未触发")
    lines.append("")

    # 算子配置 vs 运行时 txt 完整对比
    initial_txt = data.oplists.get("initial_oplist", [])
    acc_tuned_txt = data.oplists.get("accuracy_tuned_oplist", [])
    final_txt = data.oplists.get("final_oplist", [])

    initial_config_ops = None
    if data.ops_control_initial and isinstance(data.ops_control_initial, dict):
        initial_config_ops = data.ops_control_initial.get("include", [])
    if not initial_config_ops and data.op_config and isinstance(data.op_config, dict):
        all_ops = data.op_config.get("all_ops", [])
        if all_ops:
            initial_config_ops = list(all_ops)

    lines.append("算子配置 vs 运行时 txt 对比:")
    if initial_config_ops and initial_txt:
        lines.extend(_render_ops_comparison(initial_config_ops, initial_txt, "初始启动"))

    if acc_tuned_txt and excluded_acc_list:
        acc_config_ops = [op for op in (initial_config_ops or []) if op not in excluded_acc_list]
        if not acc_config_ops and data.op_config and isinstance(data.op_config, dict):
            acc_config_ops = data.op_config.get("enabled_ops", [])
        if acc_config_ops:
            lines.append("")
            lines.extend(_render_ops_comparison(acc_config_ops, acc_tuned_txt, "精度调优后"))

    if final_txt and excluded_perf_list:
        perf_config_ops = []
        if data.op_config and isinstance(data.op_config, dict):
            perf_config_ops = data.op_config.get("enabled_ops", [])
        if perf_config_ops:
            lines.append("")
            lines.extend(_render_ops_comparison(perf_config_ops, final_txt, "性能调优后"))

    if not initial_config_ops and not initial_txt:
        oplist_count = data.get("service", "enable_oplist_count", default=None)
        if oplist_count is not None:
            lines.append(f"  初始算子数: {oplist_count} 个 (无详细对比数据)")
    lines.append("")

    # Issue
    if data.issue_files:
        lines.append("Issue:")
        type_label = {
            "operator-crash": "算子崩溃", "accuracy-zero": "精度归零",
            "accuracy-degraded": "精度下降", "performance-degraded": "性能下降",
            "flagtree-error": "FlagTree 错误", "plugin-error": "Plugin 错误",
        }
        for i, issue in enumerate(data.issue_files, 1):
            itype = type_label.get(issue["type"], issue["type"])
            lines.append(f"  [{i}] {issue['title'] or '未知问题'} ({itype})")
            if issue["steps"]:
                lines.append(f"      复现: {issue['steps'].splitlines()[0].strip()}" if issue["steps"].strip() else "")
        lines.append("")

    # 发布（主流程）
    qualified = wf.get("qualified")
    release = data.get("release", default={}) or {}
    if qualified is not None:
        status = "达标" if qualified else "未达标"
        lines.append(f"发布: qualified={qualified} ({status}, 私有)")
        if release.get("harbor_image"):
            lines.append(f"  Harbor: {release['harbor_image']}")
        if release.get("modelscope_url"):
            lines.append(f"  ModelScope: {release['modelscope_url']}")
        if release.get("huggingface_url"):
            lines.append(f"  HuggingFace: {release['huggingface_url']}")

    # Plugin 区块
    plugin_wf = data.get("plugin_workflow", default={}) or {}
    if plugin_wf.get("triggered", False):
        lines.append("")
        lines.append("── Plugin ──")
        plugin_score = plugin_wf.get("plugin_score")
        plugin_acc_ok = plugin_wf.get("accuracy_ok")
        plugin_acc_diff = plugin_wf.get("accuracy_diff")
        v1_ref = eval_sec.get("v1_score") if isinstance(eval_sec, dict) else None
        if plugin_score is not None:
            ok_str = "ok" if plugin_acc_ok else "不达标"
            v1_str = f" vs V1={v1_ref}%" if v1_ref is not None else ""
            diff_str = f", diff={plugin_acc_diff}%" if plugin_acc_diff is not None else ""
            lines.append(f"精度: Plugin={plugin_score}%{v1_str}{diff_str} → {ok_str}")
        plugin_perf_ratio = plugin_wf.get("performance_ratio")
        plugin_perf_ok = plugin_wf.get("performance_ok")
        if plugin_perf_ratio is not None:
            ok_str = "ok" if plugin_perf_ok else "不达标"
            lines.append(f"性能: ratio={plugin_perf_ratio}% → {ok_str}")
        plugin_qualified = plugin_wf.get("qualified")
        if plugin_qualified is not None:
            lines.append(f"发布: qualified={plugin_qualified}")

    lines.append("═══════════════════════")
    return "\n".join(lines)


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FlagOS 迁移流程报告生成")
    parser.add_argument("--workspace", default="/flagos-workspace", help="工作目录路径")
    parser.add_argument("--json", action="store_true", dest="json_mode", help="JSON 格式输出")
    parser.add_argument("--summary", action="store_true", help="精简摘要（含调优过程详情）")
    parser.add_argument("--output", "-o", help="输出文件路径（不指定则输出到 stdout）")
    args = parser.parse_args()

    data = ReportData(args.workspace)
    if not data.collect():
        if args.output and os.path.exists(args.output):
            print("警告: 无数据，保留已有报告不覆盖", file=sys.stderr)
        else:
            print("错误: 未找到 context.yaml，无法生成报告", file=sys.stderr)
            print(f"  已检查: {args.workspace}/shared/context.yaml", file=sys.stderr)
            print(f"  已检查: {args.workspace}/config/context_snapshot.yaml", file=sys.stderr)
        sys.exit(1)

    if args.summary:
        output = generate_summary(data)
    elif args.json_mode:
        report = generate_json_report(data)
        output = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        output = generate_text_report(data)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        if os.path.exists(args.output):
            base, ext = os.path.splitext(args.output)
            backup_path = f"{base}_prev{ext}"
            shutil.copy2(args.output, backup_path)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"报告已写入: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
