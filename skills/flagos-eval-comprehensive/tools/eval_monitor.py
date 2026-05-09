#!/usr/bin/env python3
"""
评测监控脚本 — 提交 + 轮询 + 结果获取一体化

将远端评测的 提交→轮询→获取结果 封装为一次脚本调用，
避免 Claude Code 在轮询循环中消耗思考 token。

Usage:
    # 提交新评测并自动轮询到完成
    python eval_monitor.py submit --params params.json

    # 查询已有任务进度
    python eval_monitor.py poll --request-id <id> --domain NLP

    # 获取结果
    python eval_monitor.py result --request-id <id>

    # 停止评测
    python eval_monitor.py stop --request-id <id>

    # 恢复评测
    python eval_monitor.py resume --request-id <id>
"""

import sys

# IO 缓冲修复
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    import functools
    print = functools.partial(print, flush=True)

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# error_writer 集成
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from error_writer import write_last_error, write_checkpoint
except ImportError:
    def write_last_error(*a, **kw): pass
    def write_checkpoint(*a, **kw): pass

# =============================================================================
# 配置
# =============================================================================

DEFAULT_PLATFORM = os.environ.get("FLAGOS_EVAL_PLATFORM", "http://110.43.160.159:5050")

# 轮询策略: (次数区间, 间隔秒数)
# 优化：缩短间隔提升感知速度，同时增加轮询次数保持总覆盖时间
POLL_STRATEGY = [
    (5, 30),     # 第 1-5 次: 每 30 秒（快速确认任务已启动）
    (10, 60),    # 第 6-15 次: 每 60 秒
    (15, 120),   # 第 16-30 次: 每 2 分钟
]
# 进度 > 80% 时切换到密集轮询（30s），见 poll_progress()
FINISHING_INTERVAL = 30
MAX_POLLS = 30
MAX_NETWORK_FAILURES = 3


# =============================================================================
# HTTP 工具
# =============================================================================

def api_request(url: str, method: str = "GET", data: Optional[Dict] = None,
                timeout: int = 30) -> Dict[str, Any]:
    """发送 API 请求"""
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode('utf-8') if data else None

    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}", "err_code": -1}
    except URLError as e:
        return {"error": f"连接失败: {e.reason}", "err_code": -2}
    except Exception as e:
        return {"error": str(e), "err_code": -3}


# =============================================================================
# 提交评测
# =============================================================================

def submit_evaluation(params_file: str, platform: str = DEFAULT_PLATFORM,
                      auto_poll: bool = True) -> Dict[str, Any]:
    """
    提交评测任务。

    params_file: JSON 文件，包含完整的提交参数
    auto_poll: 提交成功后自动轮询到完成
    """
    params_path = Path(params_file)
    if not params_path.exists():
        print(f"ERROR: 参数文件不存在: {params_file}")
        return {"error": "params file not found"}

    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    domain = params.get("domain", "NLP")
    print(f"[提交评测] domain={domain}, platform={platform}")
    print(f"  eval_model: {params.get('eval_infos', [{}])[0].get('eval_model', '?')}")

    monitor_start = time.time()

    # 提交
    url = f"{platform}/evaluation"
    resp = api_request(url, method="POST", data=params)

    if resp.get("err_code", -1) != 0:
        print(f"ERROR: 提交失败 - {resp.get('err_msg', resp.get('error', '未知错误'))}")
        return resp

    request_id = resp.get("request_id", "")
    print(f"[提交成功] request_id: {request_id}")

    submit_time = datetime.now().isoformat()

    # 保存 request_id
    result = {
        "request_id": request_id,
        "domain": domain,
        "submit_time": submit_time,
        "status": "submitted",
        "eval_tasks": resp.get("eval_tasks", []),
    }

    # 自动轮询
    poll_count = 0
    if auto_poll and request_id:
        poll_result = poll_progress(request_id, domain, platform)
        result.update(poll_result)
        poll_count = poll_result.get("poll_count", 0)

        # 如果已完成，获取结果
        if result.get("finished"):
            eval_result = get_result(request_id, platform)
            result["eval_results"] = eval_result.get("eval_results", {})

    result["timing"] = {
        "total_seconds": round(time.time() - monitor_start),
        "submit_time": submit_time,
        "finish_time": datetime.now().isoformat(),
        "poll_count": poll_count,
    }

    return result


# =============================================================================
# 轮询进度
# =============================================================================

def get_poll_interval(poll_count: int) -> int:
    """根据轮询次数返回等待间隔"""
    accumulated = 0
    for count, interval in POLL_STRATEGY:
        accumulated += count
        if poll_count <= accumulated:
            return interval
    return POLL_STRATEGY[-1][1]


def poll_progress(request_id: str, domain: str = "NLP",
                  platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    """轮询评测进度直到完成或超出最大次数"""
    print(f"\n[轮询进度] request_id: {request_id}, 最多 {MAX_POLLS} 次")

    url = f"{platform}/evaluation_progress"
    last_progress = ""
    network_failures = 0

    for i in range(1, MAX_POLLS + 1):
        interval = get_poll_interval(i)
        print(f"\n  [{i}/{MAX_POLLS}] 查询中... (间隔 {interval}s)")

        resp = api_request(url, method="POST", data={
            "request_id": request_id,
            "domain": domain,
        })

        if resp.get("err_code", 0) < 0:
            network_failures += 1
            print(f"    网络错误 ({network_failures}/{MAX_NETWORK_FAILURES}): {resp.get('error', '?')}")
            if network_failures >= MAX_NETWORK_FAILURES:
                print(f"\n[停止轮询] 连续 {MAX_NETWORK_FAILURES} 次网络失败")
                return {"finished": False, "status": "network_error",
                        "request_id": request_id, "poll_count": i}
            time.sleep(interval)
            continue

        network_failures = 0  # 重置

        finished = resp.get("finished", False)
        status = resp.get("status", "unknown")
        datasets_progress = resp.get("datasets_progress", "")
        running_dataset = resp.get("running_dataset", "")
        running_progress = resp.get("running_progress", "")

        current_progress = f"{datasets_progress}|{running_dataset}|{running_progress}"

        # 只在进度变化时输出
        if current_progress != last_progress:
            print(f"    状态: {status}")
            if datasets_progress:
                print(f"    总进度: {datasets_progress}")
            if running_dataset:
                print(f"    当前: {running_dataset} ({running_progress})")
            last_progress = current_progress

        if finished:
            print(f"\n[评测完成] status={status}")
            return {"finished": True, "status": status, "request_id": request_id,
                    "poll_count": i}

        # 自适应：进度 > 80% 时切换到密集轮询，快速感知完成
        actual_interval = interval
        if datasets_progress:
            try:
                pct = float(datasets_progress.strip('%').split('/')[-1]) if '/' in datasets_progress else float(datasets_progress.strip('%'))
                if pct > 80:
                    actual_interval = FINISHING_INTERVAL
                    print(f"    (进度>80%, 切换密集轮询 {FINISHING_INTERVAL}s)")
            except (ValueError, IndexError):
                pass

        if i < MAX_POLLS:
            time.sleep(actual_interval)

    print(f"\n[超出轮询上限] {MAX_POLLS} 次未完成，请稍后手动查询")
    return {"finished": False, "status": "polling_timeout",
            "request_id": request_id, "poll_count": MAX_POLLS}


# =============================================================================
# 获取结果
# =============================================================================

def get_result(request_id: str, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    """获取评测结果"""
    print(f"[获取结果] request_id: {request_id}")

    url = f"{platform}/evaldiffs"
    resp = api_request(url, method="GET", data={"request_id": request_id})

    if resp.get("err_code", -1) != 0:
        print(f"ERROR: 获取结果失败 - {resp.get('err_msg', resp.get('error', '?'))}")
        return resp

    eval_results = resp.get("eval_results", {})
    print(f"[结果获取成功]")

    for model_name, model_result in eval_results.items():
        status = model_result.get("status", "?")
        details = model_result.get("details", [])
        print(f"\n  模型: {model_name} (status={status})")
        for d in details:
            ds = d.get("dataset", "?")
            acc = d.get("accuracy", "?")
            diff = d.get("diff", "?")
            ds_status = d.get("status", "?")
            print(f"    {ds}: accuracy={acc}, diff={diff}, status={ds_status}")

    return resp


# =============================================================================
# 停止/恢复
# =============================================================================

def stop_evaluation(request_id: str, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    """停止评测"""
    print(f"[停止评测] request_id: {request_id}")
    url = f"{platform}/stop_evaluation"
    return api_request(url, method="POST", data={"request_id": request_id})


def resume_evaluation(request_id: str, platform: str = DEFAULT_PLATFORM) -> Dict[str, Any]:
    """恢复评测"""
    print(f"[恢复评测] request_id: {request_id}")
    url = f"{platform}/resume_evaluation"
    return api_request(url, method="POST", data={"request_id": request_id})


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="评测监控 — 提交/轮询/结果一体化")
    subparsers = parser.add_subparsers(dest="command", help="操作命令")

    # submit
    submit_parser = subparsers.add_parser("submit", help="提交评测并自动轮询")
    submit_parser.add_argument("--params", required=True, help="评测参数 JSON 文件")
    submit_parser.add_argument("--platform", default=DEFAULT_PLATFORM, help="评测平台地址")
    submit_parser.add_argument("--no-poll", action="store_true", help="仅提交，不轮询")
    submit_parser.add_argument("--output", help="结果输出 JSON 文件路径")

    # poll
    poll_parser = subparsers.add_parser("poll", help="轮询已有任务进度")
    poll_parser.add_argument("--request-id", required=True, help="评测任务 ID")
    poll_parser.add_argument("--domain", default="NLP", help="评测域 (NLP/MM)")
    poll_parser.add_argument("--platform", default=DEFAULT_PLATFORM)

    # result
    result_parser = subparsers.add_parser("result", help="获取评测结果")
    result_parser.add_argument("--request-id", required=True, help="评测任务 ID")
    result_parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    result_parser.add_argument("--output", help="结果输出 JSON 文件路径")

    # stop
    stop_parser = subparsers.add_parser("stop", help="停止评测")
    stop_parser.add_argument("--request-id", required=True)
    stop_parser.add_argument("--platform", default=DEFAULT_PLATFORM)

    # resume
    resume_parser = subparsers.add_parser("resume", help="恢复评测")
    resume_parser.add_argument("--request-id", required=True)
    resume_parser.add_argument("--platform", default=DEFAULT_PLATFORM)

    args = parser.parse_args()

    if args.command == "submit":
        result = submit_evaluation(
            args.params, args.platform,
            auto_poll=not args.no_poll,
        )
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        print(f"\n{output_json}")
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"\n结果已保存: {args.output}")

    elif args.command == "poll":
        result = poll_progress(args.request_id, args.domain, args.platform)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "result":
        result = get_result(args.request_id, args.platform)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存: {args.output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "stop":
        result = stop_evaluation(args.request_id, args.platform)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "resume":
        result = resume_evaluation(args.request_id, args.platform)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        write_checkpoint("04_accuracy_eval", "远端评测", "running_eval_monitor",
                         action_detail=" ".join(sys.argv))
        main()
    except Exception as e:
        write_last_error(
            tool="eval_monitor.py",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=traceback.format_exc(),
        )
        print(f"[FATAL] eval_monitor.py 异常退出: {e}")
        traceback.print_exc()
        sys.exit(1)
