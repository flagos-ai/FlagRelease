#!/usr/bin/env python3
"""
update_context.py — context.yaml 结构化更新工具

避免编排层每次手写 Python 脚本更新 context.yaml 被 sandbox 拦截。
支持嵌套字段设置、数组追加、workflow_ledger 步骤状态更新。

用法:
  python3 update_context.py --set container.name=xxx --set gpu.count=8
  python3 update_context.py --json-set 'service={"port":8001,"healthy":true}'
  python3 update_context.py --ledger-update 01_container_preparation --ledger-status success --ledger-notes "容器就绪"
  python3 update_context.py --append issues.submitted=/path/to/issue.md
  python3 update_context.py --set-timing workflow_start=2026-04-17T11:34:00
  python3 update_context.py --set-timing steps.container_preparation=171
"""

import argparse
import json
import sys
import datetime

try:
    import yaml
except ImportError:
    print("[ERROR] pyyaml 未安装，请执行: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

DEFAULT_CONTEXT = "/flagos-workspace/shared/context.yaml"


def parse_value(val_str):
    """自动类型推断: true/false→bool, 数字→int/float, 其余→str"""
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
    if val_str.lower() in ("null", "none", "~"):
        return None
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def set_nested(d, key_path, value):
    """通过点号分隔路径设置嵌套字段，自动创建中间层"""
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def get_nested(d, key_path, default=None):
    """通过点号分隔路径获取嵌套字段"""
    keys = key_path.split(".")
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def append_nested(d, key_path, value):
    """追加到嵌套数组字段"""
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    last = keys[-1]
    if last not in d or not isinstance(d[last], list):
        d[last] = []
    d[last].append(value)


def update_ledger(ctx, step_id, status, notes=None, fail_reason=None, skip_reason=None):
    """更新 workflow_ledger 中指定步骤的状态"""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    ledger = ctx.get("workflow_ledger", {}).get("steps", [])
    for s in ledger:
        if s.get("step") == step_id:
            s["status"] = status
            if status == "in_progress":
                s["started_at"] = now
            elif status in ("success", "failed", "skipped"):
                s["finished_at"] = now
                if s.get("started_at"):
                    try:
                        start = datetime.datetime.fromisoformat(s["started_at"])
                        end = datetime.datetime.fromisoformat(now)
                        s["duration_seconds"] = int((end - start).total_seconds())
                    except (ValueError, TypeError):
                        pass
            if notes is not None:
                s["notes"] = notes
            if fail_reason is not None:
                s["fail_reason"] = fail_reason
            if skip_reason is not None:
                s["skip_reason"] = skip_reason
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="context.yaml 结构化更新工具")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="context.yaml 路径")
    parser.add_argument("--set", action="append", dest="sets", metavar="KEY=VALUE",
                        help="设置嵌套字段（点号分隔路径），自动类型推断")
    parser.add_argument("--json-set", action="append", dest="json_sets", metavar="KEY=JSON",
                        help="设置复杂 JSON 值")
    parser.add_argument("--append", action="append", dest="appends", metavar="KEY=VALUE",
                        help="追加到数组字段")
    parser.add_argument("--ledger-update", dest="ledger_step", help="要更新的 workflow_ledger 步骤 ID")
    parser.add_argument("--ledger-status", dest="ledger_status",
                        choices=["pending", "in_progress", "success", "failed", "skipped"])
    parser.add_argument("--ledger-notes", dest="ledger_notes")
    parser.add_argument("--ledger-fail-reason", dest="ledger_fail_reason")
    parser.add_argument("--ledger-skip-reason", dest="ledger_skip_reason")
    parser.add_argument("--set-timing", action="append", dest="timings", metavar="KEY=VALUE",
                        help="设置 timing 字段（如 steps.container_preparation=171）")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    if not any([args.sets, args.json_sets, args.appends, args.ledger_step, args.timings]):
        parser.print_help()
        sys.exit(1)

    try:
        with open(args.context, "r") as f:
            ctx = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[ERROR] {args.context} 不存在", file=sys.stderr)
        sys.exit(1)

    changes = []

    if args.sets:
        for item in args.sets:
            if "=" not in item:
                print(f"[ERROR] --set 格式错误: {item}，应为 KEY=VALUE", file=sys.stderr)
                sys.exit(1)
            key, val = item.split("=", 1)
            parsed = parse_value(val)
            set_nested(ctx, key, parsed)
            changes.append({"op": "set", "key": key, "value": parsed})

    if args.json_sets:
        for item in args.json_sets:
            if "=" not in item:
                print(f"[ERROR] --json-set 格式错误: {item}", file=sys.stderr)
                sys.exit(1)
            key, val = item.split("=", 1)
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON 解析失败: {e}", file=sys.stderr)
                sys.exit(1)
            set_nested(ctx, key, parsed)
            changes.append({"op": "json-set", "key": key})

    if args.appends:
        for item in args.appends:
            if "=" not in item:
                print(f"[ERROR] --append 格式错误: {item}", file=sys.stderr)
                sys.exit(1)
            key, val = item.split("=", 1)
            parsed = parse_value(val)
            append_nested(ctx, key, parsed)
            changes.append({"op": "append", "key": key, "value": parsed})

    if args.ledger_step:
        if not args.ledger_status:
            print("[ERROR] --ledger-update 需要 --ledger-status", file=sys.stderr)
            sys.exit(1)
        found = update_ledger(ctx, args.ledger_step, args.ledger_status,
                              notes=args.ledger_notes,
                              fail_reason=args.ledger_fail_reason,
                              skip_reason=args.ledger_skip_reason)
        if found:
            changes.append({"op": "ledger", "step": args.ledger_step, "status": args.ledger_status})
        else:
            print(f"[WARN] workflow_ledger 中未找到步骤: {args.ledger_step}", file=sys.stderr)

    if args.timings:
        if "timing" not in ctx:
            ctx["timing"] = {}
        for item in args.timings:
            if "=" not in item:
                print(f"[ERROR] --set-timing 格式错误: {item}", file=sys.stderr)
                sys.exit(1)
            key, val = item.split("=", 1)
            parsed = parse_value(val)
            set_nested(ctx["timing"], key, parsed)
            changes.append({"op": "timing", "key": key, "value": parsed})

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    set_nested(ctx, "metadata.updated_at", now)

    with open(args.context, "w") as f:
        yaml.dump(ctx, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if args.json:
        print(json.dumps({"success": True, "changes": changes, "updated_at": now}, ensure_ascii=False))
    else:
        print(f"✓ context.yaml 已更新 ({len(changes)} 项变更)")


if __name__ == "__main__":
    main()
