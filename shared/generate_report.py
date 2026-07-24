#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        self.gpqa_versions: Dict[str, Optional[dict]] = {}   # {v1: gpqa_json, v2: ..., v3: ..., v4: ...}
        self.perf_versions: Dict[str, Optional[dict]] = {}   # {v1: perf_json, v2: ..., v3: ..., v4: ...}
        self.op_config_v3: Optional[dict] = None
        self.op_config_v4: Optional[dict] = None
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

        # 多版本性能结果
        self.perf_versions["v1"] = read_json(os.path.join(r, "v1_performance.json")) or self.native_perf
        self.perf_versions["v2"] = (
            read_json(os.path.join(r, "v2_performance.json"))
            or self.optimized_perf
            or self.flagos_perf
        )
        self.perf_versions["v3"] = read_json(os.path.join(r, "v3_performance.json"))
        self.perf_versions["v4"] = read_json(os.path.join(r, "v4_performance.json"))

        # V3/V4 算子配置
        self.op_config_v3 = read_json(os.path.join(r, "operator_config_v3.json"))
        self.op_config_v4 = read_json(os.path.join(r, "operator_config_v4.json"))

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
        # 同一 issue 常生成多个时间戳副本，按标题去重（保留首个）。
        if os.path.isdir(r):
            _seen_issue_titles = set()
            for f in sorted(Path(r).glob("issue_*.md")):
                if f.name.startswith(("issue_report_", "issue_data_")):
                    continue
                content = read_text(str(f))
                if not content:
                    continue
                parsed = parse_issue_md(content)
                title = (parsed.get("title") or "").strip()
                if title and title in _seen_issue_titles:
                    continue
                if title:
                    _seen_issue_titles.add(title)
                self.issue_files.append(parsed)

        # oplists
        for name in ("initial_oplist", "accuracy_tuned_oplist", "final_oplist", "v4_oplist"):
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


# ======================================================================

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
            "issues.submitted": "context 记录的 issue 条目（GitHub 自动上传未实现，报告仅按标题列出本地 issue_*.md，不含 URL）",
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
        ext = ".json" if args.json_mode else ".md"
        out_path = args.output
        # --output 指向目录（已存在的目录，或以 / 结尾）时，按 厂商_模型名_时间戳 自动命名
        if out_path.endswith(os.sep) or os.path.isdir(out_path):
            out_path = os.path.join(out_path, build_report_basename(data, ext))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if os.path.exists(out_path):
            base, e = os.path.splitext(out_path)
            backup_path = f"{base}_prev{e}"
            shutil.copy2(out_path, backup_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"报告已写入: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
