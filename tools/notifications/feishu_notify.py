#!/usr/bin/env python3
"""Send FlagRelease lifecycle and compact summary notifications to a Feishu custom bot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

EVENTS = {
    "task_start": {"keyword": "任务开始", "template": "blue"},
    "task_end": {"keyword": "任务结束", "template": "green"},
    "result_summary": {"keyword": "结果汇总", "template": "turquoise"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过飞书自定义机器人发送 FlagRelease 任务通知",
    )
    parser.add_argument("--event", required=True, choices=EVENTS, help="通知事件")
    parser.add_argument("--title", required=True, help="消息标题（关键词会自动添加）")
    parser.add_argument("--lead", default="", help="卡片顶部的一句话结论")
    parser.add_argument("--text", default="", help="正文文本或 Markdown")
    parser.add_argument("--note", default="", help="卡片底部的次要说明")
    parser.add_argument("--file", type=Path, help="将文件内容追加到正文")
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="追加结构化字段，可重复使用",
    )
    parser.add_argument(
        "--status",
        choices=("success", "failed", "warning", "running", "skipped"),
        help="任务状态；失败时卡片自动使用红色",
    )
    parser.add_argument(
        "--format",
        choices=("interactive", "post", "text"),
        default=os.getenv("FLAGOS_FEISHU_MESSAGE_FORMAT", "interactive"),
        help="飞书消息格式（默认 interactive）",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("FLAGOS_FEISHU_MAX_CHARS", "6000")),
        help="正文最大字符数（默认 6000）",
    )
    parser.add_argument(
        "--webhook-url",
        default=os.getenv("FEISHU_WEBHOOK_URL", ""),
        help="Webhook 地址；默认读取 FEISHU_WEBHOOK_URL",
    )
    parser.add_argument(
        "--skip-if-unconfigured",
        action="store_true",
        help="未配置 Webhook 或通知已禁用时静默跳过",
    )
    parser.add_argument("--dry-run", action="store_true", help="只输出 JSON，不发送")
    return parser.parse_args()


def notifications_enabled(event: str) -> bool:
    enabled = os.getenv("FLAGOS_FEISHU_ENABLED", "auto").strip().lower()
    if enabled in {"0", "false", "no", "off", "disabled"}:
        return False

    configured_events = os.getenv("FLAGOS_FEISHU_NOTIFY_EVENTS", "").strip()
    if not configured_events:
        return True
    allowed = {item.strip() for item in configured_events.split(",") if item.strip()}
    return event in allowed


def parse_fields(items: Iterable[str]) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"字段格式错误（应为 KEY=VALUE）: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"字段名不能为空: {item}")
        fields.append((key, value.strip()))
    return fields


def read_body(args: argparse.Namespace) -> str:
    if args.max_chars <= 0:
        raise ValueError("--max-chars 必须大于 0")
    sections: List[str] = []
    if args.text.strip():
        sections.append(args.text.strip())
    if args.file:
        try:
            sections.append(args.file.read_text(encoding="utf-8").strip())
        except OSError as exc:
            raise RuntimeError(f"无法读取通知文件 {args.file}: {exc}") from exc

    body = "\n\n".join(section for section in sections if section)
    if len(body) > args.max_chars:
        omitted = len(body) - args.max_chars
        body = f"{body[:args.max_chars].rstrip()}\n\n> 内容过长，已截断 {omitted} 个字符；完整内容请查看报告文件。"
    return body


def build_markdown(lead: str, body: str, note: str, fields: List[Tuple[str, str]]) -> str:
    """Build compact fallback Markdown for text/post message formats."""
    lines: List[str] = []
    if lead.strip():
        lines.append(f"**{lead.strip()}**")
    for key, value in fields:
        lines.append(f"**{key}**：{value or '-'}")
    if body:
        lines.extend(["", body])
    if note.strip():
        lines.extend(["", note.strip()])
    return "\n".join(lines)


def markdown_element(content: str, text_size: str = "normal_v2") -> Dict[str, Any]:
    return {
        "tag": "markdown",
        "content": content,
        "text_align": "left",
        "text_size": text_size,
        "margin": "0px 0px 0px 0px",
    }


def field_column(key: str, value: str) -> Dict[str, Any]:
    return {
        "tag": "column",
        "width": "weighted",
        "weight": 1,
        "vertical_align": "top",
        "elements": [markdown_element(f"**{key}**\n{value or '-'}")],
    }


def build_interactive_elements(
    keyword: str,
    lead: str,
    body: str,
    note: str,
    fields: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Build a conclusion-first card body with paired metric columns."""
    elements: List[Dict[str, Any]] = []
    if lead.strip():
        elements.append(markdown_element(f"### {lead.strip()}", "heading"))

    for offset in range(0, len(fields), 2):
        pair = fields[offset : offset + 2]
        if len(pair) == 1:
            key, value = pair[0]
            elements.append(markdown_element(f"**{key}**\n{value or '-'}"))
            continue
        elements.append(
            {
                "tag": "column_set",
                "flex_mode": "bisect",
                "horizontal_spacing": "12px",
                "horizontal_align": "left",
                "columns": [field_column(key, value) for key, value in pair],
                "margin": "4px 0px 4px 0px",
            }
        )

    if body.strip():
        if elements:
            elements.append({"tag": "hr", "margin": "8px 0px 8px 0px"})
        elements.append(markdown_element(body.strip()))
    if note.strip():
        elements.append(markdown_element(f"> {note.strip()}"))
    if not elements:
        elements.append(markdown_element(keyword))
    return elements


def build_payload(args: argparse.Namespace, fields: List[Tuple[str, str]], body: str) -> Dict[str, Any]:
    config = EVENTS[args.event]
    keyword = config["keyword"]
    title = args.title.strip()
    if keyword not in title:
        title = f"{keyword}｜{title}"

    lead = str(getattr(args, "lead", "") or "")
    note = str(getattr(args, "note", "") or "")
    markdown = build_markdown(lead, body, note, fields)

    if args.format == "text":
        content = f"{title}\n\n{markdown}" if markdown else title
        return {"msg_type": "text", "content": {"text": content}}

    if args.format == "post":
        paragraphs = []
        for line in markdown.splitlines() or [keyword]:
            paragraphs.append([{"tag": "text", "text": line or " "}])
        return {
            "msg_type": "post",
            "content": {
                "post": {
                    "zh_cn": {
                        "title": title,
                        "content": paragraphs,
                    }
                }
            },
        }

    template = config["template"]
    if args.status == "failed":
        template = "red"
    elif args.status == "warning":
        template = "orange"
    elif args.status == "skipped":
        template = "grey"

    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"update_multi": True},
            "body": {
                "direction": "vertical",
                "padding": "12px 12px 12px 12px",
                "elements": build_interactive_elements(keyword, lead, body, note, fields),
            },
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": template,
                "padding": "12px 12px 12px 12px",
            },
        },
    }


def response_succeeded(raw: str) -> Tuple[bool, str]:
    if not raw.strip():
        return True, ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return True, raw.strip()

    code = data.get("code", data.get("StatusCode"))
    message = str(data.get("msg", data.get("StatusMessage", "")))
    if code is None:
        return True, message
    return str(code) == "0", message or str(code)


def send_with_curl(webhook_url: str, payload: Dict[str, Any]) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise RuntimeError("未找到 curl 命令")

    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    command = [
        curl,
        "--silent",
        "--show-error",
        "--fail",
        "--connect-timeout",
        "5",
        "--max-time",
        "15",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-d",
        payload_json,
        webhook_url,
    ]

    last_error = ""
    for attempt in range(1, 4):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            ok, message = response_succeeded(completed.stdout)
            if ok:
                return
            last_error = f"飞书返回失败: {message}"
        else:
            last_error = completed.stderr.strip() or f"curl exit={completed.returncode}"
        if attempt < 3:
            time.sleep(attempt)
    raise RuntimeError(last_error)


def main() -> int:
    args = parse_args()
    try:
        fields = parse_fields(args.field)
        body = read_body(args)
        payload = build_payload(args, fields, body)
    except (ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not notifications_enabled(args.event):
        if not args.skip_if_unconfigured:
            print(f"飞书通知事件已禁用: {args.event}", file=sys.stderr)
        return 0

    if not args.webhook_url:
        if args.skip_if_unconfigured:
            return 0
        print("错误: 未配置 FEISHU_WEBHOOK_URL", file=sys.stderr)
        return 2

    try:
        send_with_curl(args.webhook_url, payload)
    except RuntimeError as exc:
        print(f"飞书通知发送失败: {exc}", file=sys.stderr)
        return 1

    print(f"✓ 飞书通知已发送: {EVENTS[args.event]['keyword']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
