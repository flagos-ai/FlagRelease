#!/usr/bin/env python3
"""stream_to_debug_log.py — 将 claude --output-format stream-json 事件流转为人可读的全量 debug 日志。

用法:
    claude -p "..." --output-format stream-json | tee >(python3 stream_to_debug_log.py > debug.log) | stream_filter.py

输出格式:
    [HH:MM:SS] ══ ASSISTANT TEXT ══
    ...文本内容...

    [HH:MM:SS] ▶ Bash
      command: docker inspect ...
      ── result (exit 0) ──
      ...完整输出...

    [HH:MM:SS] ══ SESSION SUMMARY ══
      duration: 15m 0s
      tool_calls: 47
      cost: $1.23
"""

import sys
import json
from datetime import datetime

# 统计
stats = {
    "tool_calls": 0,
    "errors": 0,
    "start_time": None,
}


def ts():
    """当前时间戳"""
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(msg, flush=True)


def format_tool_use(block):
    """格式化 tool_use 块为可读文本"""
    name = block.get("name", "unknown")
    inp = block.get("input", {})
    stats["tool_calls"] += 1

    lines = [f"[{ts()}] ▶ {name}"]

    if name == "Bash":
        cmd = inp.get("command", "")
        lines.append(f"  command: {cmd}")
        if inp.get("timeout"):
            lines.append(f"  timeout: {inp['timeout']}ms")
    elif name in ("Read", "Write"):
        lines.append(f"  file: {inp.get('file_path', '')}")
        if "content" in inp:
            content = inp["content"]
            n = content.count("\n") + 1
            lines.append(f"  content: ({n} lines, {len(content)} chars)")
    elif name == "Edit":
        lines.append(f"  file: {inp.get('file_path', '')}")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        lines.append(f"  old: ({len(old)} chars) {old[:120]}{'...' if len(old) > 120 else ''}")
        lines.append(f"  new: ({len(new)} chars) {new[:120]}{'...' if len(new) > 120 else ''}")
    elif name == "Glob":
        lines.append(f"  pattern: {inp.get('pattern', '')}")
        if inp.get("path"):
            lines.append(f"  path: {inp['path']}")
    elif name == "Grep":
        lines.append(f"  pattern: {inp.get('pattern', '')}")
        if inp.get("path"):
            lines.append(f"  path: {inp['path']}")
        if inp.get("glob"):
            lines.append(f"  glob: {inp['glob']}")
    elif name == "Agent":
        lines.append(f"  description: {inp.get('description', '')[:200]}")
    else:
        # 通用：打印所有 input 字段
        for k, v in inp.items():
            s = str(v)
            lines.append(f"  {k}: {s[:200]}{'...' if len(s) > 200 else ''}")

    return "\n".join(lines)


def format_tool_result(event):
    """格式化 tool_result"""
    result = event.get("tool_use_result", {})

    if isinstance(result, str):
        stdout = result
        is_error = False
    elif isinstance(result, dict):
        stdout = result.get("stdout", "")
        is_error = result.get("is_error", False)
    else:
        return None

    if is_error:
        stats["errors"] += 1

    if not stdout:
        return f"  ── result {'(ERROR)' if is_error else '(empty)'} ──"

    header = f"  ── result {'(ERROR)' if is_error else '(ok)'} ──"
    # 缩进输出内容
    indented = "\n".join(f"  {line}" for line in stdout.split("\n"))
    return f"{header}\n{indented}"


def process_event(event):
    if not isinstance(event, dict):
        return

    etype = event.get("type", "")

    if stats["start_time"] is None:
        stats["start_time"] = datetime.now()

    if etype == "assistant":
        message = event.get("message")
        if not isinstance(message, dict):
            return
        for block in message.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                text = block["text"]
                if text.strip():
                    log(f"\n[{ts()}] ══ ASSISTANT TEXT ══")
                    log(text)
            elif btype == "tool_use":
                log(f"\n{format_tool_use(block)}")

    elif etype == "user":
        formatted = format_tool_result(event)
        if formatted:
            log(formatted)

    elif etype == "result":
        dur_ms = event.get("duration_ms", 0) or 0
        cost = event.get("total_cost_usd", 0) or 0
        dur_s = dur_ms / 1000
        minutes = int(dur_s // 60)
        seconds = int(dur_s % 60)

        log(f"\n[{ts()}] ══ SESSION SUMMARY ══")
        log(f"  duration: {minutes}m {seconds}s")
        log(f"  tool_calls: {stats['tool_calls']}")
        log(f"  errors: {stats['errors']}")
        log(f"  cost: ${cost:.2f}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            log(f"[{ts()}] [RAW] {line}")
            continue
        process_event(event)


if __name__ == "__main__":
    main()
