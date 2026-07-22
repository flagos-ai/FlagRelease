#!/usr/bin/env python3
"""Run a read-only Claude analysis for one model and persist a validated result."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA_VERSION = 1
TERMINAL_OUTCOMES = {"success", "failed", "timeout", "skipped"}
DELIVERY_VERSIONS = {"v3", "v5"}

CLAUDE_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cost": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "migration_cost_usd": {"type": ["number", "null"], "minimum": 0},
                "complete": {"type": "boolean"},
            },
            "required": ["migration_cost_usd", "complete"],
        },
        "delivery": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "version": {
                    "anyOf": [
                        {"type": "string", "enum": ["v3", "v5"]},
                        {"type": "null"},
                    ]
                },
                "accuracy_ok": {"type": ["boolean", "null"]},
                "uploaded": {"type": ["boolean", "null"]},
            },
            "required": ["version", "accuracy_ok", "uploaded"],
        },
        "notification": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "result_label": {"type": "string"},
                "lead": {"type": "string"},
                "summary": {"type": ["string", "null"]},
                "warning": {"type": ["string", "null"]},
            },
            "required": ["result_label", "lead", "summary", "warning"],
        },
        "evidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "accuracy": {"type": "array", "items": {"type": "string"}},
                "upload": {"type": "array", "items": {"type": "string"}},
                "cost": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["accuracy", "upload", "cost"],
        },
    },
    "required": ["cost", "delivery", "notification", "evidence"],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def finite_nonnegative(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def string_or_none(value: Any, max_chars: int = 500) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_chars] if text else None


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = string_or_none(item, 500)
        if text and text not in result:
            result.append(text)
    return result


def default_notification(
    outcome: str,
    qualified: bool,
    accuracy_ok: Any,
    delivery_label: Optional[str] = None,
) -> Dict[str, Any]:
    if outcome == "failed":
        return {"result_label": "流程失败", "lead": "❌ 流程失败", "summary": None, "warning": None}
    if outcome == "timeout":
        return {"result_label": "超时", "lead": "⏱️ 任务超时", "summary": None, "warning": None}
    if outcome == "skipped":
        return {"result_label": "已跳过", "lead": "⏭️ 已跳过", "summary": None, "warning": None}
    if qualified:
        lead = f"✅ {delivery_label} 达标上传" if delivery_label else "✅ 达标上传"
        return {"result_label": "达标上传", "lead": lead, "summary": None, "warning": None}
    if accuracy_ok is False:
        lead = f"⚠️ {delivery_label} 完成但未达标" if delivery_label else "⚠️ 完成但未达标"
        return {"result_label": "完成但未达标", "lead": lead, "summary": None, "warning": None}
    lead = f"⚠️ {delivery_label} 结果无法确认" if delivery_label else "⚠️ 结果无法确认"
    return {"result_label": "无法确认", "lead": lead, "summary": None, "warning": None}


def failed_result(args: argparse.Namespace, reason: str, analysis_elapsed: int, analysis_cost: Optional[float] = None) -> Dict[str, Any]:
    notification = default_notification(args.outcome, False, None)
    notification.update(
        {
            "result_label": "无法确认" if args.outcome == "success" else notification["result_label"],
            "lead": "⚠️ 流程已结束，结果分析失败" if args.outcome == "success" else notification["lead"],
            "warning": "单模型结果分析失败",
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "failed",
        "analysis_error": reason[:1000],
        "model": args.model_name,
        "target": args.target,
        "vendor": args.vendor or "unknown",
        "pipeline": {
            "outcome": args.outcome,
            "exit_code": args.exit_code,
            "elapsed_seconds": args.elapsed_seconds,
        },
        "cost": {"migration_cost_usd": None, "complete": False},
        "delivery": {
            "version": None,
            "label": None,
            "accuracy_ok": None,
            "uploaded": None,
            "qualified_uploaded": False,
        },
        "notification": notification,
        "evidence": {"accuracy": [], "upload": [], "cost": []},
        "analysis": {"elapsed_seconds": analysis_elapsed, "cost_usd": analysis_cost},
        "analyzed_at": now_iso(),
    }


def normalize_success(args: argparse.Namespace, raw: Dict[str, Any], analysis_elapsed: int, analysis_cost: Optional[float]) -> Dict[str, Any]:
    cost_raw = raw.get("cost") if isinstance(raw.get("cost"), dict) else {}
    delivery_raw = raw.get("delivery") if isinstance(raw.get("delivery"), dict) else {}
    notification_raw = raw.get("notification") if isinstance(raw.get("notification"), dict) else {}
    evidence_raw = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}

    version = delivery_raw.get("version")
    if version not in DELIVERY_VERSIONS:
        version = None
    accuracy_ok = delivery_raw.get("accuracy_ok") if isinstance(delivery_raw.get("accuracy_ok"), bool) else None
    uploaded = delivery_raw.get("uploaded") if isinstance(delivery_raw.get("uploaded"), bool) else None
    label = "V5" if version == "v5" else ("V3 Max" if version == "v3" else None)
    qualified = version is not None and accuracy_ok is True and uploaded is True

    # 状态标签由可信的执行事实和结构化交付结论确定；Claude 只提供简短说明。
    defaults = default_notification(args.outcome, qualified, accuracy_ok, label)
    notification = {
        "result_label": defaults["result_label"],
        "lead": defaults["lead"],
        "summary": string_or_none(notification_raw.get("summary"), 500),
        "warning": string_or_none(notification_raw.get("warning"), 300),
    }

    migration_cost = finite_nonnegative(cost_raw.get("migration_cost_usd"))
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "success",
        "analysis_error": None,
        "model": args.model_name,
        "target": args.target,
        "vendor": args.vendor or "unknown",
        "pipeline": {
            "outcome": args.outcome,
            "exit_code": args.exit_code,
            "elapsed_seconds": args.elapsed_seconds,
        },
        "cost": {
            "migration_cost_usd": migration_cost,
            "complete": bool(cost_raw.get("complete")) and migration_cost is not None,
        },
        "delivery": {
            "version": version,
            "label": label,
            "accuracy_ok": accuracy_ok,
            "uploaded": uploaded,
            "qualified_uploaded": qualified,
        },
        "notification": notification,
        "evidence": {
            "accuracy": string_list(evidence_raw.get("accuracy")),
            "upload": string_list(evidence_raw.get("upload")),
            "cost": string_list(evidence_raw.get("cost")),
        },
        "analysis": {"elapsed_seconds": analysis_elapsed, "cost_usd": analysis_cost},
        "analyzed_at": now_iso(),
    }


def extract_structured_output(stdout: str) -> tuple[Dict[str, Any], Optional[float]]:
    envelope = json.loads(stdout)
    if not isinstance(envelope, dict):
        raise ValueError("Claude 输出不是 JSON 对象")
    analysis_cost = finite_nonnegative(envelope.get("total_cost_usd"))
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured, analysis_cost
    result = envelope.get("result")
    if isinstance(result, dict):
        return result, analysis_cost
    if isinstance(result, str):
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, analysis_cost
    raise ValueError("Claude 输出缺少 structured_output")


def validate_structured_output(raw: Dict[str, Any]) -> None:
    expected_sections = {"cost", "delivery", "notification", "evidence"}
    if set(raw) != expected_sections:
        missing = sorted(expected_sections - set(raw))
        extra = sorted(set(raw) - expected_sections)
        raise ValueError(f"结构化结果字段不完整: missing={missing}, extra={extra}")

    cost = raw.get("cost")
    if not isinstance(cost, dict) or set(cost) != {"migration_cost_usd", "complete"}:
        raise ValueError("cost 字段不符合 schema")
    if cost["migration_cost_usd"] is not None and finite_nonnegative(cost["migration_cost_usd"]) is None:
        raise ValueError("migration_cost_usd 必须为非负数或 null")
    if not isinstance(cost["complete"], bool):
        raise ValueError("cost.complete 必须为 boolean")

    delivery = raw.get("delivery")
    if not isinstance(delivery, dict) or set(delivery) != {"version", "accuracy_ok", "uploaded"}:
        raise ValueError("delivery 字段不符合 schema")
    if delivery["version"] not in DELIVERY_VERSIONS | {None}:
        raise ValueError("delivery.version 无效")
    for key in ("accuracy_ok", "uploaded"):
        if delivery[key] is not None and not isinstance(delivery[key], bool):
            raise ValueError(f"delivery.{key} 必须为 boolean 或 null")

    notification = raw.get("notification")
    notification_keys = {"result_label", "lead", "summary", "warning"}
    if not isinstance(notification, dict) or set(notification) != notification_keys:
        raise ValueError("notification 字段不符合 schema")
    for key in ("result_label", "lead"):
        if not isinstance(notification[key], str):
            raise ValueError(f"notification.{key} 必须为 string")
    for key in ("summary", "warning"):
        if notification[key] is not None and not isinstance(notification[key], str):
            raise ValueError(f"notification.{key} 必须为 string 或 null")

    evidence = raw.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {"accuracy", "upload", "cost"}:
        raise ValueError("evidence 字段不符合 schema")
    for key in ("accuracy", "upload", "cost"):
        if not isinstance(evidence[key], list) or not all(isinstance(item, str) for item in evidence[key]):
            raise ValueError(f"evidence.{key} 必须为 string 数组")


def build_prompt(template: str, args: argparse.Namespace) -> str:
    replacements = {
        "{{MODEL_DIR}}": str(args.model_dir),
        "{{MODEL_NAME}}": args.model_name,
        "{{TARGET}}": args.target,
        "{{VENDOR}}": args.vendor or "unknown",
        "{{OUTCOME}}": args.outcome,
        "{{EXIT_CODE}}": str(args.exit_code),
        "{{ELAPSED_SECONDS}}": str(args.elapsed_seconds),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        prompt = build_prompt(args.prompt_file.read_text(encoding="utf-8"), args)
        command = [
            args.claude_command,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(CLAUDE_RESULT_SCHEMA, ensure_ascii=False, separators=(",", ":")),
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Read,Glob,Grep",
            "--max-turns",
            "100",
            "--no-session-persistence",
            "--add-dir",
            str(args.model_dir),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.timeout_seconds,
        )
        elapsed = max(0, int(round(time.monotonic() - started)))
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}"
            return failed_result(args, f"Claude 分析失败: {detail}", elapsed)
        structured, analysis_cost = extract_structured_output(completed.stdout)
        validate_structured_output(structured)
        return normalize_success(args, structured, elapsed, analysis_cost)
    except subprocess.TimeoutExpired:
        elapsed = max(0, int(round(time.monotonic() - started)))
        return failed_result(args, f"Claude 分析超时（>{args.timeout_seconds}s）", elapsed)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        elapsed = max(0, int(round(time.monotonic() - started)))
        return failed_result(args, f"Claude 分析结果无效: {exc}", elapsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析单模型产物并保存结构化进度结果")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--vendor", default="unknown")
    parser.add_argument("--outcome", choices=sorted(TERMINAL_OUTCOMES), required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--claude-command", default="claude")
    parser.add_argument("--failure-reason", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.failure_reason:
        result = failed_result(args, args.failure_reason, 0)
        atomic_write_json(args.result_json, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    if not args.model_dir.is_dir():
        result = failed_result(args, f"模型目录不存在: {args.model_dir}", 0)
        atomic_write_json(args.result_json, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    if not args.prompt_file.is_file():
        result = failed_result(args, f"提示词文件不存在: {args.prompt_file}", 0)
        atomic_write_json(args.result_json, result)
        print(json.dumps(result, ensure_ascii=False))
        return 1
    result = analyze(args)
    atomic_write_json(args.result_json, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("analysis_status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
