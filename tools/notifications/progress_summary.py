#!/usr/bin/env python3
"""Build deterministic FlagRelease progress summaries and optionally notify Feishu."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1
DEFAULT_WORKSPACE = Path("/data/flagos-workspace")
FEISHU_NOTIFY = Path(__file__).with_name("feishu_notify.py")
TERMINAL_OUTCOMES = {"success", "failed", "timeout", "skipped"}

VENDOR_ALIASES = {
    "nvidia": "nvidia",
    "huawei": "huawei",
    "ascend": "huawei",
    "hygon": "hygon",
    "dcu": "hygon",
    "metax": "metax",
    "cambricon": "cambricon",
    "mlu": "cambricon",
    "mthreads": "mthreads",
    "musa": "mthreads",
    "mtt": "mthreads",
    "kunlunxin": "kunlunxin",
    "kunlun": "kunlunxin",
    "xpu": "kunlunxin",
    "iluvatar": "iluvatar",
    "tianshu": "iluvatar",
}
VENDOR_TOKEN_RE = re.compile(r"(?:^|[/:_.@-])([a-z0-9]+)(?=$|[/:_.@-])", re.IGNORECASE)
# 镜像命名常把厂商名与编号连写(如 metax001),识别时剥掉 token 尾部数字再查表。
VENDOR_TRAILING_DIGITS_RE = re.compile(r"\d+$")


def canonical_vendor_for_token(token: str) -> Optional[str]:
    """Map one lowercased token to a canonical vendor, tolerating trailing digits."""
    if token in VENDOR_ALIASES:
        return VENDOR_ALIASES[token]
    stripped = VENDOR_TRAILING_DIGITS_RE.sub("", token)
    if stripped != token and stripped in VENDOR_ALIASES:
        return VENDOR_ALIASES[stripped]
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def epoch_iso(value: Optional[int]) -> str:
    if value is None:
        return now_iso()
    return datetime.fromtimestamp(max(0, value), tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def vendor_from_target(target: str) -> str:
    """Infer one canonical vendor from delimiter-bounded tokens in a task target."""
    matches = set()
    for token in VENDOR_TOKEN_RE.findall(str(target or "").lower()):
        vendor = canonical_vendor_for_token(token)
        if vendor:
            matches.add(vendor)
    return next(iter(matches)) if len(matches) == 1 else "unknown"


def aggregate_vendors(vendors: Iterable[str]) -> str:
    normalized = [str(item or "unknown").strip() or "unknown" for item in vendors]
    if not normalized or all(item == "unknown" for item in normalized):
        return "unknown"
    unique = set(normalized)
    return next(iter(unique)) if len(unique) == 1 else "mixed"


def nested_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nonnegative_float(value: Any) -> Optional[float]:
    number = as_float(value)
    return number if number is not None and number >= 0 else None


def integer_value(value: Any) -> Optional[int]:
    number = as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def model_from_result_file(
    result_file: Path,
    *,
    model_name: str = "",
    target: str = "",
    task_index: Optional[int] = None,
    outcome_override: str = "",
    exit_code_override: Optional[int] = None,
    elapsed_override: Optional[int] = None,
    vendor_override: str = "",
) -> Dict[str, Any]:
    """Load the authoritative per-model analysis result and flatten it for cards."""
    data = load_json(result_file)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"单模型分析结果不存在或格式错误: {result_file}")
    if data.get("analysis_status") not in {"success", "failed"}:
        raise RuntimeError(f"单模型分析结果 analysis_status 无效: {result_file}")
    pipeline = nested_dict(data.get("pipeline"))
    cost = nested_dict(data.get("cost"))
    delivery = nested_dict(data.get("delivery"))
    analysis = nested_dict(data.get("analysis"))
    evidence = nested_dict(data.get("evidence"))
    notification = nested_dict(data.get("notification"))

    if not pipeline or not cost or not delivery or not notification or not evidence:
        raise RuntimeError(f"单模型分析结果缺少必要字段: {result_file}")
    if not isinstance(cost.get("complete"), bool):
        raise RuntimeError(f"单模型分析结果 cost.complete 无效: {result_file}")
    migration_cost = nonnegative_float(cost.get("migration_cost_usd"))
    if cost.get("migration_cost_usd") is not None and migration_cost is None:
        raise RuntimeError(f"单模型分析结果 migration_cost_usd 无效: {result_file}")
    if cost.get("complete") is True and migration_cost is None:
        raise RuntimeError(f"单模型分析结果费用标记为完整但金额缺失: {result_file}")
    if any(
        not isinstance(evidence.get(key), list)
        or not all(isinstance(item, str) for item in evidence.get(key, []))
        for key in ("accuracy", "upload", "cost")
    ):
        raise RuntimeError(f"单模型分析结果 evidence 无效: {result_file}")
    if not isinstance(notification.get("result_label"), str) or not isinstance(notification.get("lead"), str):
        raise RuntimeError(f"单模型分析结果 notification 无效: {result_file}")
    for key in ("summary", "warning"):
        if notification.get(key) is not None and not isinstance(notification.get(key), str):
            raise RuntimeError(f"单模型分析结果 notification.{key} 无效: {result_file}")

    outcome = outcome_override or str(pipeline.get("outcome") or "")
    if outcome not in TERMINAL_OUTCOMES:
        raise RuntimeError(f"单模型分析结果 outcome 无效: {result_file}")
    pipeline_exit = integer_value(pipeline.get("exit_code"))
    pipeline_elapsed = integer_value(pipeline.get("elapsed_seconds"))
    if pipeline_exit is None or pipeline_elapsed is None or pipeline_elapsed < 0:
        raise RuntimeError(f"单模型分析结果 pipeline 执行事实无效: {result_file}")
    raw_version = delivery.get("version")
    if raw_version is not None and not isinstance(raw_version, str):
        raise RuntimeError(f"单模型分析结果 delivery.version 无效: {result_file}")
    version = str(raw_version or "").lower() or None
    # 新流程 v3.1：交付版本为 V3 Max，或其上的 V4 Express 减算子优化；已无 V5。
    if version not in {None, "v3", "v4"}:
        raise RuntimeError(f"单模型分析结果 delivery.version 无效: {result_file}")
    accuracy_ok = delivery.get("accuracy_ok") if isinstance(delivery.get("accuracy_ok"), bool) else None
    uploaded = delivery.get("uploaded") if isinstance(delivery.get("uploaded"), bool) else None
    if delivery.get("accuracy_ok") is not None and accuracy_ok is None:
        raise RuntimeError(f"单模型分析结果 accuracy_ok 无效: {result_file}")
    if delivery.get("uploaded") is not None and uploaded is None:
        raise RuntimeError(f"单模型分析结果 uploaded 无效: {result_file}")
    qualified = version is not None and accuracy_ok is True and uploaded is True
    elapsed = elapsed_override if elapsed_override is not None else pipeline.get("elapsed_seconds")
    exit_code = exit_code_override if exit_code_override is not None else pipeline.get("exit_code")
    parsed_elapsed = integer_value(elapsed)
    parsed_exit_code = integer_value(exit_code)
    if parsed_elapsed is None or parsed_elapsed < 0 or parsed_exit_code is None:
        raise RuntimeError(f"单模型分析结果执行覆盖参数无效: {result_file}")
    vendor = vendor_override.strip() or str(data.get("vendor") or "unknown").strip() or "unknown"

    return {
        "task_index": task_index,
        "model": model_name or str(data.get("model") or ""),
        "target": target or str(data.get("target") or ""),
        "vendor": vendor,
        "gpu_type": "",
        "gpu_count": 0,
        "outcome": outcome,
        "exit_code": parsed_exit_code,
        "elapsed_seconds": parsed_elapsed,
        "cost_usd": migration_cost,
        "cost_components": list(evidence.get("cost") or []),
        "delivery_version": version,
        "delivery_label": str(delivery.get("label") or ("V4 Express" if version == "v4" else "V3 Max" if version == "v3" else "")),
        "accuracy_ok": accuracy_ok,
        "uploaded": uploaded,
        "qualified_uploaded": qualified,
        "analysis_status": str(data.get("analysis_status") or "failed"),
        "analysis_error": data.get("analysis_error"),
        "analysis_elapsed_seconds": nonnegative_float(analysis.get("elapsed_seconds")),
        "analysis_cost_usd": nonnegative_float(analysis.get("cost_usd")),
        "result_file": str(result_file),
        "evidence": evidence,
        "notification": notification,
        "finished_at": str(data.get("analyzed_at") or now_iso()),
        "analyzed_at": str(data.get("analyzed_at") or ""),
    }


def parse_task_file(path: Path) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        target, separator, model = line.partition("|")
        if not separator:
            target, model = line, ""
        clean_target = target.strip()
        tasks.append(
            {
                "task_index": len(tasks) + 1,
                "target": clean_target,
                "model": model.strip(),
                "vendor": vendor_from_target(clean_target),
            }
        )
    return tasks


def load_state(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"批次状态不存在或格式错误: {path}")
    return data


def state_path(args: argparse.Namespace) -> Path:
    if args.state_file:
        return args.state_file
    if not args.batch_id:
        raise ValueError("必须提供 --state-file 或 --batch-id")
    return args.workspace / f"batch_{args.batch_id}_progress.json"


def update_batch_vendor(state: Dict[str, Any]) -> None:
    task_vendors = [str(item.get("vendor") or "unknown") for item in state.get("tasks", [])]
    if task_vendors:
        state["vendor"] = aggregate_vendors(task_vendors)
        return
    model_vendors = [str(item.get("vendor") or "unknown") for item in state.get("models", [])]
    current = nested_dict(state.get("current_model"))
    if current:
        model_vendors.append(str(current.get("vendor") or "unknown"))
    state["vendor"] = aggregate_vendors(model_vendors)


def batch_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    models = [item for item in state.get("models", []) if item.get("outcome") in TERMINAL_OUTCOMES]
    successful = [item for item in models if item.get("outcome") == "success"]
    elapsed_values = [float(item["elapsed_seconds"]) for item in successful if as_float(item.get("elapsed_seconds")) is not None]
    cost_values = [float(item["cost_usd"]) for item in successful if as_float(item.get("cost_usd")) is not None]
    all_cost_values = [float(item["cost_usd"]) for item in models if as_float(item.get("cost_usd")) is not None]
    qualified = [item for item in models if item.get("qualified_uploaded") is True]
    v4_count = sum(1 for item in qualified if item.get("delivery_version") == "v4")
    v3_count = sum(1 for item in qualified if item.get("delivery_version") == "v3")
    processed = len(models)
    return {
        "processed_models": processed,
        "total_models": int(state.get("total_models") or 0),
        "qualified_uploaded": len(qualified),
        "v4_qualified_uploaded": v4_count,
        "v3_qualified_uploaded": v3_count,
        "success_rate_pct": round(len(qualified) / processed * 100.0, 1) if processed else 0.0,
        "average_elapsed_seconds": sum(elapsed_values) / len(elapsed_values) if elapsed_values else None,
        "average_cost_usd": sum(cost_values) / len(cost_values) if cost_values else None,
        "total_cost_usd": sum(all_cost_values) if all_cost_values else None,
        "failed_models": sum(1 for item in models if item.get("outcome") == "failed"),
        "timeout_models": sum(1 for item in models if item.get("outcome") == "timeout"),
        "skipped_models": sum(1 for item in models if item.get("outcome") == "skipped"),
    }


def format_duration(value: Any) -> str:
    seconds_value = as_float(value)
    if seconds_value is None:
        return "未知"
    seconds = max(0, int(round(seconds_value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_cost(value: Any) -> str:
    number = as_float(value)
    return "未知" if number is None else f"${number:.2f}"


def batch_elapsed_seconds(state: Dict[str, Any]) -> Optional[float]:
    """Return wall-clock seconds from batch start to the latest recorded batch time."""
    explicit = nonnegative_float(state.get("batch_elapsed_seconds"))
    if explicit is not None:
        return explicit
    started_at = str(state.get("started_at") or "").strip()
    finished_at = str(state.get("ended_at") or state.get("updated_at") or "").strip()
    if not started_at or not finished_at:
        return None
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0.0, (finished - started).total_seconds())


def format_device(gpu_type: str, gpu_count: int) -> str:
    """Format only the accelerator specification; vendor belongs in the card title."""
    text = str(gpu_type or "").strip()
    if gpu_count > 0:
        text = f"{text} × {gpu_count}" if text else f"{gpu_count} 卡"
    return text or "未识别"


def escape_table_cell(value: Any) -> str:
    return str(value if value not in (None, "") else "-").replace("|", "\\|").replace("\n", "<br>")


def delivery_display_label(model: Dict[str, Any]) -> str:
    label = str(model.get("delivery_label") or "")
    if model.get("qualified_uploaded") is True or model.get("uploaded") is True or model.get("accuracy_ok") is not None:
        return label or "已生成"
    if model.get("outcome") == "success" and label:
        return f"{label}（待确认）"
    return "未确认"


def compact_model_result(model: Dict[str, Any]) -> str:
    outcome = model.get("outcome")
    if outcome == "timeout":
        return "⏱️ 超时"
    if outcome == "failed":
        return "❌ 流程失败"
    if outcome == "skipped":
        return "⏭️ 跳过"
    if model.get("analysis_status") == "failed":
        return "⚠️ 分析失败"
    delivery = str(model.get("delivery_label") or "").strip()
    if model.get("qualified_uploaded") is True:
        return f"✅ {delivery} 达标" if delivery else "✅ 达标"
    if model.get("accuracy_ok") is None:
        return f"❔ {delivery} 待确认" if delivery else "❔ 待确认"
    if model.get("accuracy_ok") is True and model.get("uploaded") is not True:
        return f"⚠️ {delivery} 未上传" if delivery else "⚠️ 未上传"
    return f"⚠️ {delivery} 未达标" if delivery else "⚠️ 未达标"


def _completed_by_task(state: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    completed: Dict[int, Dict[str, Any]] = {}
    tasks = state.get("tasks", [])
    unassigned = list(state.get("models", []))
    for model in list(unassigned):
        index = int(as_float(model.get("task_index")) or 0)
        if index > 0:
            completed[index] = model
            unassigned.remove(model)
    for task in tasks:
        index = int(as_float(task.get("task_index")) or 0)
        if index in completed:
            continue
        for model in list(unassigned):
            if model.get("model") == task.get("model") and model.get("target") == task.get("target"):
                completed[index] = model
                unassigned.remove(model)
                break
    return completed


def build_task_queue_markdown(state: Dict[str, Any], max_items: int = 5) -> str:
    tasks = list(state.get("tasks", []))
    if not tasks:
        return ""
    visible = tasks if max_items <= 0 else tasks[:max_items]
    show_vendor = str(state.get("vendor") or "").lower() == "mixed"
    lines = ["### 任务队列"]
    for position, task in enumerate(visible, 1):
        index = int(as_float(task.get("task_index")) or position)
        model = task.get("model") or "-"
        if show_vendor:
            lines.append(f"{index}. {task.get('vendor') or 'unknown'} · {model}")
        else:
            lines.append(f"{index}. {model}")
    omitted = len(tasks) - len(visible)
    if omitted:
        lines.extend(["", f"> 另有 {omitted} 个模型"])
    return "\n".join(lines)


def build_models_markdown_table(
    state: Dict[str, Any], max_rows: Optional[int] = None, show_all: bool = False
) -> str:
    tasks = state.get("tasks", [])
    if not tasks:
        return ""
    if max_rows is None:
        try:
            max_rows = int(os.getenv("FLAGOS_FEISHU_TABLE_MAX_ROWS", "20"))
        except ValueError:
            max_rows = 20
    completed = _completed_by_task(state)
    current = nested_dict(state.get("current_model"))
    current_index = int(as_float(current.get("task_index")) or 0)
    task_by_index = {int(as_float(task.get("task_index")) or pos): task for pos, task in enumerate(tasks, 1)}
    all_indexes = list(task_by_index)
    ended = bool(state.get("ended_at"))

    if ended or show_all:
        candidates = all_indexes
    else:
        candidates = sorted(set(completed) | ({current_index} if current_index else set()))
    if not candidates:
        return ""

    original_count = len(candidates)
    if max_rows > 0 and len(candidates) > max_rows:
        if ended:
            problems = [
                index
                for index in candidates
                if index not in completed or not compact_model_result(completed[index]).startswith("✅")
            ]
            priority = problems + sorted(completed, reverse=True) + candidates
        else:
            priority = ([current_index] if current_index else []) + sorted(completed, reverse=True)
            if show_all:
                priority = priority + all_indexes
        selected: List[int] = []
        for index in priority:
            if index in candidates and index not in selected:
                selected.append(index)
            if len(selected) >= max_rows:
                break
        candidates = sorted(selected)
    omitted = original_count - len(candidates)

    show_vendor = str(state.get("vendor") or "").lower() == "mixed"
    lines = ["### 最终模型明细" if ended else "### 模型进度"]
    if show_vendor:
        lines.extend(
            [
                "| # | 厂商 / 模型 | 结果 | 耗时 / 费用 |",
                "| ---: | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| # | 模型 | 结果 | 耗时 / 费用 |",
                "| ---: | --- | --- | --- |",
            ]
        )
    for index in candidates:
        task = task_by_index[index]
        model = completed.get(index)
        if model:
            vendor = str(model.get("vendor") or "unknown")
            result = compact_model_result(model)
            duration_cost = f"{format_duration(model.get('elapsed_seconds'))} / {format_cost(model.get('cost_usd'))}"
        elif index == current_index:
            vendor = str(current.get("vendor") or state.get("vendor") or "unknown")
            result = "⛔ 未完成" if ended else "🔄 运行中"
            duration_cost = "-"
        else:
            vendor = str(task.get("vendor") or state.get("vendor") or "待识别")
            result = "⛔ 未完成" if ended else "⏳ 待执行"
            duration_cost = "-"
        model_name = str(task.get("model") or "-")
        model_cell = f"{vendor}<br>{model_name}" if show_vendor else model_name
        lines.append(
            "| " + " | ".join(
                escape_table_cell(value)
                for value in (index, model_cell, result, duration_cost)
            ) + " |"
        )

    if not ended:
        pending = [index for index in all_indexes if index not in completed and index != current_index]
        if pending:
            next_model = task_by_index[pending[0]].get("model") or "-"
            lines.extend(["", f"> 待执行：{len(pending)} 个 · 下一模型：{next_model}"])
    else:
        unfinished = sum(1 for index in all_indexes if index not in completed)
        if unfinished:
            lines.extend(["", f"> 未完成：{unfinished} 个"])
    if omitted:
        lines.extend(["", f"> 另有 {omitted} 个已处理模型未展示"])
    return "\n".join(lines)


def model_result_label(model: Dict[str, Any]) -> str:
    outcome = model.get("outcome")
    if outcome == "timeout":
        return "⏱️ 超时"
    if outcome == "failed":
        return "❌ 流程失败"
    if outcome == "skipped":
        return "⏭️ 已跳过"
    if model.get("analysis_status") == "failed":
        return "⚠️ 结果分析失败"
    if model.get("qualified_uploaded") is True:
        return f"✅ {model.get('delivery_label', '-')} 达标上传"
    if model.get("accuracy_ok") is None:
        delivery = str(model.get("delivery_label") or "").strip()
        return f"❔ {delivery} 待确认" if delivery else "❔ 待确认"
    if model.get("accuracy_ok") is True and model.get("uploaded") is not True:
        delivery = str(model.get("delivery_label") or "").strip()
        return f"⚠️ {delivery} 未上传" if delivery else "⚠️ 未上传"
    return "⚠️ 完成但未达标"


def status_for_model(model: Dict[str, Any]) -> str:
    if model.get("outcome") in {"failed", "timeout"}:
        return "failed"
    if model.get("qualified_uploaded") is True:
        return "success"
    if model.get("outcome") == "skipped":
        return "skipped"
    return "warning"


def model_result_summary(model: Dict[str, Any]) -> str:
    notification = nested_dict(model.get("notification"))
    value = notification.get("summary") or notification.get("warning")
    return str(value or "").strip()


def summary_fields(state: Dict[str, Any], latest: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Build the compact metric grid for a per-model batch progress card."""
    metrics = batch_metrics(state)
    fields: List[Tuple[str, str]] = [
        ("当前总耗时", format_duration(batch_elapsed_seconds(state))),
        ("累计费用", format_cost(metrics["total_cost_usd"])),
    ]
    if latest:
        fields.extend(
            [
                ("本模型耗时", format_duration(latest.get("elapsed_seconds"))),
                ("本模型费用", format_cost(latest.get("cost_usd"))),
            ]
        )
    fields.extend(
        [
            (
                "达标上传",
                f"{metrics['qualified_uploaded']} / {metrics['processed_models']} · {metrics['success_rate_pct']:.1f}%"
                f"<br>V3 Max {metrics['v3_qualified_uploaded']} · V4 Express {metrics['v4_qualified_uploaded']}",
            ),
            ("成功模型均值", success_average_label(metrics)),
        ]
    )
    return fields


def success_average_label(metrics: Dict[str, Any]) -> str:
    return f"{format_duration(metrics['average_elapsed_seconds'])} · {format_cost(metrics['average_cost_usd'])}"


def batch_progress_note(state: Dict[str, Any]) -> str:
    return f"批次：{state.get('batch_id') or '-'}"


def batch_end_fields(state: Dict[str, Any], metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [
        ("批次总耗时", format_duration(batch_elapsed_seconds(state))),
        ("累计费用", format_cost(metrics["total_cost_usd"])),
        (
            "达标上传",
            f"{metrics['qualified_uploaded']} / {metrics['processed_models']} · {metrics['success_rate_pct']:.1f}%"
            f"<br>V3 Max {metrics['v3_qualified_uploaded']} · V4 Express {metrics['v4_qualified_uploaded']}",
        ),
        ("成功模型均值", success_average_label(metrics)),
    ]


def batch_end_lead(metrics: Dict[str, Any]) -> str:
    processed = int(metrics["processed_models"])
    total = int(metrics["total_models"])
    qualified = int(metrics["qualified_uploaded"])
    failed = int(metrics["failed_models"])
    timeout = int(metrics["timeout_models"])
    skipped = int(metrics["skipped_models"])
    other = max(0, processed - qualified - failed - timeout - skipped)
    unfinished = max(0, total - processed)
    if processed and processed == total and qualified == processed:
        return f"✅ 批次完成：{qualified} / {processed} 达标上传"
    parts = [f"{qualified} 达标"]
    if other:
        parts.append(f"{other} 未达标/待确认")
    if failed:
        parts.append(f"{failed} 失败")
    if timeout:
        parts.append(f"{timeout} 超时")
    if skipped:
        parts.append(f"{skipped} 跳过")
    if unfinished:
        parts.append(f"{unfinished} 未完成")
    return "⚠️ 批次结束：" + " · ".join(parts)


def single_start_fields(args: argparse.Namespace, hardware: Dict[str, Any]) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    device = format_device(hardware["gpu_type"], hardware["gpu_count"])
    if device != "未识别":
        fields.append(("设备", device))
    if args.target:
        fields.append(("镜像", args.target))
    return fields


def single_end_fields(model: Dict[str, Any]) -> List[Tuple[str, str]]:
    fields = [
        ("总耗时", format_duration(model.get("elapsed_seconds"))),
        ("总费用", format_cost(model.get("cost_usd"))),
    ]
    if model.get("outcome") in {"failed", "timeout"} and model.get("exit_code") is not None:
        fields.append(("退出码", str(model["exit_code"])))
    return fields


def run_feishu(
    event: str,
    title: str,
    status: str,
    fields: Iterable[Tuple[str, str]],
    text: str,
    dry_run: bool,
    lead: str = "",
    note: str = "",
) -> int:
    command = [
        sys.executable,
        str(FEISHU_NOTIFY),
        "--event",
        event,
        "--title",
        title,
        "--status",
        status,
        "--skip-if-unconfigured",
    ]
    if lead:
        command.extend(["--lead", lead])
    if note:
        command.extend(["--note", note])
    for key, value in fields:
        command.extend(["--field", f"{key}={value}"])
    if text:
        command.extend(["--text", text])
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def emit_or_notify(
    payload: Dict[str, Any],
    event: str,
    title: str,
    status: str,
    fields: List[Tuple[str, str]],
    text: str,
    notify: bool,
    dry_run: bool,
    lead: str = "",
    note: str = "",
) -> int:
    if notify or dry_run:
        return run_feishu(event, title, status, fields, text, dry_run, lead, note)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_batch_start(args: argparse.Namespace) -> int:
    if not args.task_file:
        raise ValueError("batch-start 必须提供 --task-file")
    tasks = parse_task_file(args.task_file)
    # 批量厂商只由 tasks.txt 第一列解析，避免宿主机或外部环境覆盖任务事实。
    batch_vendor = aggregate_vendors(task["vendor"] for task in tasks)
    path = state_path(args)
    state = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": args.batch_id or path.stem,
        "workspace": str(args.workspace),
        "task_file": str(args.task_file),
        "initial_vendor": batch_vendor,
        "vendor": batch_vendor,
        "gpu_type": "",
        "gpu_count": 0,
        "total_models": len(tasks),
        "tasks": tasks,
        "models": [],
        "current_model": None,
        "started_at": epoch_iso(args.started_at_epoch),
        "ended_at": None,
    }
    atomic_write_json(path, state)
    fields: List[Tuple[str, str]] = [("任务规模", f"{state['total_models']} 个模型")]
    title = f"{state['vendor']}｜{state['total_models']} 个模型"
    lead = "⏳ 批量任务已启动"
    # 批量开始时展示完整模型列表（含厂商/序号），而非仅前 5 个队列摘要。
    queue = build_task_queue_markdown(state, max_items=0)
    note = f"批次：{state['batch_id']}"
    return emit_or_notify(
        state, "task_start", title, "running", fields, queue, args.notify, args.dry_run, lead, note
    )


def command_model_start(args: argparse.Namespace) -> int:
    if not args.model:
        raise ValueError("model-start 必须提供 --model")
    path = state_path(args)
    loaded = load_json(path)
    state = loaded if isinstance(loaded, dict) else {}
    if not state:
        # batch-start 事件可能尚未落地；用最小骨架承载单模型开始通知。
        state = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": args.batch_id or path.stem,
            "workspace": str(args.workspace),
            "vendor": args.vendor.strip() or vendor_from_target(args.target or "") or "unknown",
            "total_models": args.task_index or 1,
            "tasks": [],
            "models": [],
            "current_model": None,
            "ended_at": None,
        }
    task_index = resolve_task_index(state, args.task_index, args.model, args.target or "")
    task = next(
        (
            item
            for item in state.get("tasks", [])
            if int(as_float(item.get("task_index")) or 0) == int(task_index or 0)
        ),
        {},
    )
    vendor = args.vendor.strip() or str(task.get("vendor") or "") or vendor_from_target(args.target or "") or "unknown"
    state["current_model"] = {
        "task_index": task_index,
        "model": args.model,
        "target": args.target or task.get("target") or "",
        "vendor": vendor,
        "started_at": epoch_iso(args.event_time_epoch),
    }
    if args.batch_elapsed_seconds is not None:
        state["batch_elapsed_seconds"] = max(0, args.batch_elapsed_seconds)
    atomic_write_json(path, state)
    metrics = batch_metrics(state)
    total = metrics["total_models"] or state.get("total_models") or 0
    position = int(task_index or metrics["processed_models"] + 1)
    fields: List[Tuple[str, str]] = [("当前进度", f"第 {position} / {total} 个")]
    if args.target:
        fields.append(("镜像", args.target))
    title = f"{state.get('vendor', 'unknown')}｜{position} / {total}"
    lead = f"🚀 开始迁移 · {args.model}"
    note = batch_progress_note(state)
    table = build_models_markdown_table(state, show_all=True)
    payload = {"state": state, "metrics": metrics}
    return emit_or_notify(
        payload, "model_start", title, "running", fields, table, args.notify, args.dry_run, lead, note
    )


def resolve_task_index(state: Dict[str, Any], task_index: Optional[int], model: str, target: str) -> Optional[int]:
    if task_index is not None and task_index > 0:
        return task_index
    completed = _completed_by_task(state)
    for position, task in enumerate(state.get("tasks", []), 1):
        index = int(as_float(task.get("task_index")) or position)
        if index in completed:
            continue
        if task.get("model") == model and (not target or task.get("target") == target):
            return index
    return None


def command_model_finish(args: argparse.Namespace) -> int:
    if not args.model or not args.result_file:
        raise ValueError("model-finish 必须提供 --model 和 --result-file")
    path = state_path(args)
    state = load_state(path)
    task_index = resolve_task_index(state, args.task_index, args.model, args.target or "")
    task = next(
        (
            item
            for item in state.get("tasks", [])
            if int(as_float(item.get("task_index")) or 0) == int(task_index or 0)
        ),
        {},
    )
    model = model_from_result_file(
        args.result_file,
        model_name=args.model,
        target=args.target or "",
        task_index=task_index,
        outcome_override=args.outcome or "",
        exit_code_override=args.exit_code,
        elapsed_override=args.elapsed_seconds,
        vendor_override=args.vendor.strip() or str(task.get("vendor") or ""),
    )
    models = state.setdefault("models", [])
    if task_index is not None:
        models[:] = [item for item in models if int(as_float(item.get("task_index")) or 0) != task_index]
    models.append(model)
    current = nested_dict(state.get("current_model"))
    if not current or task_index is None or int(as_float(current.get("task_index")) or 0) == task_index:
        state["current_model"] = None
    state["updated_at"] = epoch_iso(args.event_time_epoch)
    if args.batch_elapsed_seconds is not None:
        state["batch_elapsed_seconds"] = max(0, args.batch_elapsed_seconds)
    update_batch_vendor(state)
    atomic_write_json(path, state)
    metrics = batch_metrics(state)
    fields = summary_fields(state, model)
    title = f"{state.get('vendor', 'unknown')}｜{metrics['processed_models']} / {metrics['total_models']}"
    lead = f"{model_result_label(model)} · {model['model']}"
    note = batch_progress_note(state)
    payload = {"state": state, "latest": model, "metrics": metrics}
    table = build_models_markdown_table(state, show_all=True)
    result_summary = model_result_summary(model)
    if result_summary:
        table = f"**结果说明**：{result_summary}\n\n{table}" if table else f"**结果说明**：{result_summary}"
    return emit_or_notify(
        payload,
        "result_summary",
        title,
        status_for_model(model),
        fields,
        table,
        args.notify,
        args.dry_run,
        lead,
        note,
    )


def command_batch_end(args: argparse.Namespace) -> int:
    path = state_path(args)
    state = load_state(path)
    state["ended_at"] = state.get("ended_at") or epoch_iso(args.event_time_epoch)
    state["batch_exit_code"] = args.exit_code
    if args.batch_elapsed_seconds is not None:
        state["batch_elapsed_seconds"] = max(0, args.batch_elapsed_seconds)
    update_batch_vendor(state)
    atomic_write_json(path, state)
    latest = state.get("models", [])[-1] if state.get("models") else None
    metrics = batch_metrics(state)
    status = "failed" if args.exit_code not in (None, 0) else "success"
    if status == "success" and (
        metrics["processed_models"] < metrics["total_models"]
        or metrics["failed_models"]
        or metrics["timeout_models"]
    ):
        status = "warning"
    fields = batch_end_fields(state, metrics)
    title_total = metrics["total_models"] or metrics["processed_models"]
    title = (
        f"{state.get('vendor', 'unknown')}｜"
        f"{metrics['qualified_uploaded']} / {title_total} 达标"
    )
    lead = batch_end_lead(metrics)
    note = f"批次：{state.get('batch_id') or '-'}"
    payload = {"state": state, "latest": latest, "metrics": metrics}
    table = build_models_markdown_table(state)
    return emit_or_notify(
        payload, "task_end", title, status, fields, table, args.notify, args.dry_run, lead, note
    )


def single_state(model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": None,
        "vendor": model["vendor"],
        "gpu_type": model["gpu_type"],
        "gpu_count": model["gpu_count"],
        "total_models": 1,
        "models": [model],
    }


def command_single_start(args: argparse.Namespace) -> int:
    if not args.model:
        raise ValueError("single-start 必须提供 --model")
    vendor = args.vendor.strip() or vendor_from_target(args.target or "")
    hardware = {"vendor": vendor, "gpu_type": "", "gpu_count": 0}
    state = {
        "schema_version": SCHEMA_VERSION,
        "vendor": hardware["vendor"],
        "gpu_type": hardware["gpu_type"],
        "gpu_count": hardware["gpu_count"],
        "total_models": 1,
        "models": [],
        "started_at": now_iso(),
    }
    fields = single_start_fields(args, hardware)
    title = f"{hardware['vendor']}｜{args.model}"
    return emit_or_notify(
        state,
        "task_start",
        title,
        "running",
        fields,
        "",
        args.notify,
        args.dry_run,
        "⏳ 正在执行",
    )


def command_single_end(args: argparse.Namespace) -> int:
    if not args.model or not args.result_file:
        raise ValueError("single-end 必须提供 --model 和 --result-file")
    model = model_from_result_file(
        args.result_file,
        model_name=args.model,
        target=args.target or "",
        outcome_override=args.outcome or "",
        exit_code_override=args.exit_code,
        elapsed_override=args.elapsed_seconds,
        vendor_override=args.vendor.strip() or vendor_from_target(args.target or ""),
    )
    state = single_state(model)
    fields = single_end_fields(model)
    title = f"{model['vendor']}｜{model['model']}"
    payload = {"state": state, "latest": model, "metrics": batch_metrics(state)}
    return emit_or_notify(
        payload,
        "task_end",
        title,
        status_for_model(model),
        fields,
        model_result_summary(model),
        args.notify,
        args.dry_run,
        model_result_label(model),
    )


def command_validate_result(args: argparse.Namespace) -> int:
    if not args.result_file:
        raise ValueError("validate-result 必须提供 --result-file")
    model_from_result_file(args.result_file)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 FlagRelease 精简进度汇总并发送飞书通知")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
        subparser.add_argument("--state-file", type=Path)
        subparser.add_argument("--batch-id", default="")
        subparser.add_argument("--task-file", type=Path)
        subparser.add_argument("--model", default="")
        subparser.add_argument("--target", default="")
        subparser.add_argument("--vendor", default="")
        subparser.add_argument("--outcome", choices=sorted(TERMINAL_OUTCOMES))
        subparser.add_argument("--exit-code", type=int)
        subparser.add_argument("--elapsed-seconds", type=int)
        subparser.add_argument("--batch-elapsed-seconds", type=int)
        subparser.add_argument("--started-at-epoch", type=int)
        subparser.add_argument("--event-time-epoch", type=int)
        subparser.add_argument("--task-index", type=int)
        subparser.add_argument("--result-file", type=Path)
        subparser.add_argument("--notify", action="store_true")
        subparser.add_argument("--dry-run", action="store_true")

    commands = {
        "batch-start": command_batch_start,
        "model-start": command_model_start,
        "model-finish": command_model_finish,
        "batch-end": command_batch_end,
        "single-start": command_single_start,
        "single-end": command_single_end,
        "validate-result": command_validate_result,
    }
    for name, handler in commands.items():
        subparser = subparsers.add_parser(name)
        add_common(subparser)
        subparser.set_defaults(handler=handler)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
