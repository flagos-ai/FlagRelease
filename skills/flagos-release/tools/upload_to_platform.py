#!/usr/bin/env python3
"""
upload_to_platform.py — 将 FlagOS 流水线结果上传到 FlagRelease 平台

从 context_final.yaml + results/ 目录读取数据，构造 payload 并调用 API。
流程完成后最多上传 3 条记录：native / gt (gems+tree) / pl (plugin)。

Usage:
    python3 upload_to_platform.py --context /path/to/context_final.yaml --results-dir /path/to/results/
    python3 upload_to_platform.py --context /path/to/context_final.yaml --results-dir /path/to/results/ --dry-run

环境变量:
    FLAGRELEASE_API_TOKEN — API 认证 token（必需，除非 --dry-run）

退出码: 0=成功, 1=数据缺失, 2=API调用失败
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import yaml
except ImportError:
    print("错误: 缺少 pyyaml，请安装: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


DEFAULT_API_URL = "https://flagrelease.flagos.net/api/v1/offline/import/insert"


def read_yaml_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def read_json_file(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_chip_name(gpu_type: str, vendor: str) -> str:
    """去掉厂商前缀，如 'NVIDIA H20-3e' → 'H20-3e'"""
    prefixes = ["NVIDIA", "AMD", "Intel", "Ascend", "MetaX", "MThreads", "Iluvatar", "Hygon", "Cambricon"]
    for prefix in prefixes:
        if gpu_type.upper().startswith(prefix.upper()):
            return gpu_type[len(prefix):].strip()
    return gpu_type


def extract_model_short(model_name: str) -> str:
    """'google/gemma-4-E2B-it' → 'gemma-4-E2B-it'"""
    return model_name.split("/")[-1] if "/" in model_name else model_name


def format_datetime(ts: str) -> str:
    """ISO 8601 或各种格式 → 'YYYY-MM-DD HH:MM:SS'"""
    if not ts:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts = ts.replace("T", " ").replace("Z", "")
    if len(ts) > 19:
        ts = ts[:19]
    return ts


def format_date(ts: str) -> str:
    """提取日期部分 'YYYY-MM-DD'"""
    if not ts:
        return datetime.now().strftime("%Y-%m-%d")
    return ts[:10]


def extract_performance(perf_json: dict) -> Optional[Dict[str, Any]]:
    """从性能 JSON 提取数据，支持扁平格式和嵌套格式"""
    # 扁平格式: {test_case, concurrency, request_throughput, ...}
    if "test_case" in perf_json:
        concurrency = int(perf_json.get("concurrency", 64))
        num_prompts = int(perf_json.get("num_prompts", concurrency))
        input_len = int(perf_json.get("input_len", 4096))
        output_len = int(perf_json.get("output_len", 1024))
        return {
            "prefill_length": input_len,
            "decode_length": output_len,
            "max_concurrency": concurrency,
            "completed": num_prompts,
            "failed": 0,
            "duration": perf_json.get("duration", 0),
            "total_input_tokens": num_prompts * input_len,
            "total_output_tokens": num_prompts * output_len,
            "request_throughput": perf_json.get("request_throughput", 0),
            "output_throughput": perf_json.get("output_token_throughput", 0),
            "max_output_tokens_per_s": None,
            "max_concurrent_requests": None,
            "total_token_throughput": perf_json.get("total_token_throughput", 0),
            "mean_ttft_ms": perf_json.get("mean_ttft_ms", 0),
            "median_ttft_ms": perf_json.get("median_ttft_ms", 0),
            "p99_ttft_ms": perf_json.get("p99_ttft_ms", 0),
            "mean_tpot_ms": perf_json.get("mean_tpot_ms", 0),
            "median_tpot_ms": perf_json.get("median_tpot_ms", 0),
            "p99_tpot_ms": perf_json.get("p99_tpot_ms", 0),
            "mean_itl_ms": perf_json.get("mean_itl_ms", 0),
            "median_itl_ms": perf_json.get("median_itl_ms", 0),
            "p99_itl_ms": perf_json.get("p99_itl_ms", 0),
        }

    # 嵌套格式: {4k_input_1k_output: {64: {...}}}
    data = perf_json.get("4k_input_1k_output", {})
    concurrency_key = "64"
    if concurrency_key not in data:
        keys = [k for k in data.keys() if k != "_meta"]
        if not keys:
            return None
        concurrency_key = keys[0]

    metrics = data[concurrency_key]
    return {
        "prefill_length": 4096,
        "decode_length": 1024,
        "max_concurrency": int(concurrency_key),
        "completed": int(metrics.get("Successful requests", 0)),
        "failed": int(metrics.get("Failed requests", 0)),
        "duration": metrics.get("Benchmark duration (s)", 0),
        "total_input_tokens": int(metrics.get("Total input tokens", 0)),
        "total_output_tokens": int(metrics.get("Total generated tokens", 0)),
        "request_throughput": metrics.get("Request throughput (req/s)", 0),
        "output_throughput": metrics.get("Output token throughput (tok/s)", 0),
        "max_output_tokens_per_s": None,
        "max_concurrent_requests": None,
        "total_token_throughput": metrics.get("Total token throughput (tok/s)", 0),
        "mean_ttft_ms": metrics.get("Mean TTFT (ms)", 0),
        "median_ttft_ms": metrics.get("Median TTFT (ms)", 0),
        "p99_ttft_ms": metrics.get("P99 TTFT (ms)", 0),
        "mean_tpot_ms": metrics.get("Mean TPOT (ms)", 0),
        "median_tpot_ms": metrics.get("Median TPOT (ms)", 0),
        "p99_tpot_ms": metrics.get("P99 TPOT (ms)", 0),
        "mean_itl_ms": metrics.get("Mean ITL (ms)", 0),
        "median_itl_ms": metrics.get("Median ITL (ms)", 0),
        "p99_itl_ms": metrics.get("P99 ITL (ms)", 0),
    }


def find_latest_search_step(results_dir: str) -> Optional[str]:
    """查找 results 目录中编号最大的 search_step_N.json，返回完整路径"""
    import glob
    pattern = os.path.join(results_dir, "search_step_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_num(path):
        basename = os.path.basename(path)
        try:
            return int(basename.replace("search_step_", "").replace(".json", ""))
        except ValueError:
            return -1
    files.sort(key=extract_num)
    return files[-1]


def extract_eval_result(gpqa_json: dict) -> Dict[str, Any]:
    """从 GPQA JSON 构造 eval_result"""
    score = gpqa_json.get("score", 0)
    benchmark = gpqa_json.get("benchmark", "gpqa_diamond")
    return {
        "result": [
            {
                "status": "finished",
                "dataset": benchmark,
                "accuracy": score,
            }
        ],
        "isCredible": True,
    }


def build_item(
    name: str,
    software_eco: str,
    full_model_name: str,
    vendor_ename: str,
    chip_name: str,
    used_cards: str,
    framework: str,
    data_format: str,
    cudagraph_mode: str,
    operator_mode: str,
    create_datetime: str,
    release_time: str,
    model_owner: str,
    eval_tag: List[str],
    performance_result: Optional[Dict[str, Any]],
    eval_result: Optional[Dict[str, Any]],
    model_name: str = "",
    gems_op_list: Optional[List[str]] = None,
    enable_op_count: Optional[int] = None,
    total_op_count: Optional[int] = None,
    op_replace_rate: Optional[str] = None,
    operator_version: str = "",
    compiler_version: str = "",
    framework_version: str = "",
    comm_lib_version: str = "",
) -> Dict[str, Any]:
    item = {
        "description": None,
        "update_datetime": None,
        "create_datetime": create_datetime,
        "name": name,
        "model_name": model_name or None,
        "software_eco": software_eco,
        "performance_result": [performance_result] if performance_result else [],
        "gems_op_list": gems_op_list if isinstance(gems_op_list, list) else [],
        "enable_op_count": enable_op_count,
        "total_op_count": total_op_count,
        "op_replace_rate": op_replace_rate,
        "data_format": data_format,
        "start_time": None,
        "release_time": release_time,
        "used_cards": used_cards,
        "framework": framework,
        "full_model_name": full_model_name,
        "vendor_ename": vendor_ename,
        "chip_name": chip_name,
        "eval_result": eval_result or {"result": [], "isCredible": False},
        "model_owner": model_owner,
        "eval_status": "1",
        "eval_tag": eval_tag,
        "cudagraph_mode": cudagraph_mode,
        "operator_mode": operator_mode,
        "operator_version": operator_version or None,
        "compiler_version": compiler_version or None,
        "framework_version": framework_version or None,
        "comm_lib_version": comm_lib_version or None,
    }
    return item


def call_api(api_url: str, token: str, items: List[Dict[str, Any]]) -> dict:
    """调用 FlagRelease API（insert 或 update）"""
    payload = json.dumps({"items": items}, ensure_ascii=False).encode("utf-8")
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "lang": "zh-CN",
        "timezone": "Asia/Shanghai",
    }
    req = Request(api_url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return json.loads(body)
        except Exception:
            return {"success": False, "error": f"HTTP {e.code}: {body}"}
    except URLError as e:
        return {"success": False, "error": f"网络错误: {e.reason}"}


def main():
    parser = argparse.ArgumentParser(description="上传 FlagOS 流水线结果到 FlagRelease 平台")
    parser.add_argument("--context", required=True, help="context_final.yaml 路径")
    parser.add_argument("--results-dir", required=True, help="results/ 目录路径")
    parser.add_argument("--logs-dir", default="", help="logs/ 目录路径（可选，用于解析额外信息）")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API 地址")
    parser.add_argument("--model-owner", default="自动化", help="模型负责人")
    parser.add_argument("--dry-run", action="store_true", help="只打印 payload 不发送")
    args = parser.parse_args()

    # 读取 context
    ctx = read_yaml_file(args.context)
    if not ctx:
        # 尝试 context_snapshot.yaml
        alt = args.context.replace("context_final.yaml", "context_snapshot.yaml")
        ctx = read_yaml_file(alt)
        if not ctx:
            print(f"错误: 无法读取 context 文件: {args.context}", file=sys.stderr)
            sys.exit(1)
        print(f"  使用备选文件: {alt}")

    # 提取公共字段
    model_name = ctx.get("model", {}).get("name", "")
    model_short = extract_model_short(model_name)
    vendor = ctx.get("gpu", {}).get("vendor", "nvidia")
    gpu_type = ctx.get("gpu", {}).get("type", "")
    chip_name = extract_chip_name(gpu_type, vendor)
    tp_size = ctx.get("runtime", {}).get("tp_size", 1)
    framework = (ctx.get("runtime", {}).get("framework", "vllm") or "vllm").upper()
    workflow_start = ctx.get("timing", {}).get("workflow_start", "")
    release_ts = ctx.get("release", {}).get("timestamp", "")
    create_dt = format_datetime(workflow_start)
    release_date = format_date(release_ts) if release_ts else format_date(workflow_start)

    workflow = ctx.get("workflow", {})
    plugin_wf = ctx.get("plugin_workflow", {})
    optimization = ctx.get("optimization", {})

    # 新增平台字段数据源
    inspection = ctx.get("inspection", {})
    flag_packages = inspection.get("flag_packages", {})
    core_packages = inspection.get("core_packages", {})
    environment = ctx.get("environment", {})
    service = ctx.get("service", {})
    operator_replacement = ctx.get("operator_replacement", {})

    flaggems_ver = flag_packages.get("flaggems", "")
    flagtree_ver = environment.get("flagtree_version", "")
    vllm_ver = core_packages.get("vllm", "")
    plugin_ver = flag_packages.get("vllm_plugin", "")
    flagcx_ver = flag_packages.get("flagcx", "")
    framework_ver = vllm_ver or ""

    # 算子列表和数量
    gems_op_list = optimization.get("enabled_ops", []) or service.get("initial_operator_list", [])
    if isinstance(gems_op_list, str):
        try:
            gems_op_list = json.loads(gems_op_list)
        except (json.JSONDecodeError, TypeError):
            gems_op_list = []
    available_ops = operator_replacement.get("available_ops", [])
    enable_op_count = service.get("enable_oplist_count", 0) or len(gems_op_list)
    total_op_count = len(available_ops) if available_ops else enable_op_count

    # 时间戳后缀（从 workflow_start 提取 YYYYMMDD_HHMM）
    ts_suffix = ""
    if workflow_start:
        dt_str = workflow_start.replace("T", " ").replace("Z", "")[:16]
        date_part = dt_str[:10].replace("-", "")
        time_part = dt_str[11:16].replace(":", "")
        ts_suffix = f"_{date_part}_{time_part}"

    items = []
    results_dir = args.results_dir.rstrip("/")

    # --- native 记录 ---
    native_perf_json = read_json_file(os.path.join(results_dir, "native_performance.json"))
    native_gpqa_json = read_json_file(os.path.join(results_dir, "gpqa_native.json"))
    if native_perf_json:
        perf_data = extract_performance(native_perf_json)
        eval_data = extract_eval_result(native_gpqa_json) if native_gpqa_json else {"result": [], "isCredible": False}
        items.append(build_item(
            name=f"{model_short}_{vendor}_native{ts_suffix}",
            software_eco="native",
            full_model_name=model_name,
            vendor_ename=vendor,
            chip_name=chip_name,
            used_cards=str(tp_size),
            framework=framework,
            data_format="BF16",
            cudagraph_mode="FULL",
            operator_mode="python",
            create_datetime=create_dt,
            release_time=release_date,
            model_owner=args.model_owner,
            eval_tag=["自动发布", "原生基线"],
            performance_result=perf_data,
            eval_result=eval_data,
            model_name=model_short,
            gems_op_list=[],
            enable_op_count=0,
            total_op_count=0,
            op_replace_rate=None,
            operator_version="",
            compiler_version="",
            framework_version=framework_ver,
            comm_lib_version="",
        ))
        print(f"  ✓ native 记录已构造 ({model_short}_{vendor}_native{ts_suffix})")
    else:
        print(f"  ⚠ 跳过 native 记录（native_performance.json 不存在）")

    # --- gt 记录 (gems+tree) ---
    # 优先级: flagos_optimized.json > search_step_N.json(最大N) > flagos_performance.json
    gt_perf_file = None
    gt_perf_json = read_json_file(os.path.join(results_dir, "flagos_optimized.json"))
    if gt_perf_json:
        gt_perf_file = "flagos_optimized.json"
    else:
        latest_step = find_latest_search_step(results_dir)
        if latest_step:
            gt_perf_json = read_json_file(latest_step)
            gt_perf_file = os.path.basename(latest_step)
        if not gt_perf_json:
            gt_perf_json = read_json_file(os.path.join(results_dir, "flagos_performance.json"))
            if gt_perf_json:
                gt_perf_file = "flagos_performance.json"
    gt_gpqa_json = read_json_file(os.path.join(results_dir, "gpqa_flagos.json"))

    if gt_perf_json:
        perf_data = extract_performance(gt_perf_json)
        eval_data = extract_eval_result(gt_gpqa_json) if gt_gpqa_json else {"result": [], "isCredible": False}
        perf_ok = workflow.get("performance_ok", False)
        eval_tag = ["自动发布", "性能达标" if perf_ok else "性能不达标"]
        items.append(build_item(
            name=f"{model_short}_{vendor}_flagos{ts_suffix}",
            software_eco="flagos",
            full_model_name=model_name,
            vendor_ename=vendor,
            chip_name=chip_name,
            used_cards=str(tp_size),
            framework=framework,
            data_format="BF16",
            cudagraph_mode="FULL",
            operator_mode="python",
            create_datetime=create_dt,
            release_time=release_date,
            model_owner=args.model_owner,
            eval_tag=eval_tag,
            performance_result=perf_data,
            eval_result=eval_data,
            model_name=model_short,
            gems_op_list=gems_op_list,
            enable_op_count=enable_op_count,
            total_op_count=total_op_count,
            op_replace_rate=None,
            operator_version=flaggems_ver,
            compiler_version=flagtree_ver,
            framework_version=framework_ver,
            comm_lib_version=flagcx_ver,
        ))
        print(f"  ✓ gt 记录已构造 ({model_short}_{vendor}_flagos{ts_suffix}, {'达标' if perf_ok else '不达标'}, 数据源: {gt_perf_file})")
    else:
        print(f"  ⚠ 跳过 gt 记录（flagos 性能数据不存在）")

    # --- pl 记录 (plugin) ---
    plugin_triggered = plugin_wf.get("triggered", False)
    if plugin_triggered:
        pl_perf_json = read_json_file(os.path.join(results_dir, "plugin_performance.json"))
        pl_gpqa_json = read_json_file(os.path.join(results_dir, "gpqa_plugin.json"))
        if pl_perf_json:
            perf_data = extract_performance(pl_perf_json)
            eval_data = extract_eval_result(pl_gpqa_json) if pl_gpqa_json else {"result": [], "isCredible": False}
            pl_perf_ok = plugin_wf.get("performance_ok", False)
            eval_tag = ["自动发布", "性能达标" if pl_perf_ok else "性能不达标"]
            items.append(build_item(
                name=f"{model_short}_{vendor}_plugin{ts_suffix}",
                software_eco="flagos",
                full_model_name=model_name,
                vendor_ename=vendor,
                chip_name=chip_name,
                used_cards=str(tp_size),
                framework=framework,
                data_format="BF16",
                cudagraph_mode="FULL",
                operator_mode="python",
                create_datetime=create_dt,
                release_time=release_date,
                model_owner=args.model_owner,
                eval_tag=eval_tag,
                performance_result=perf_data,
                eval_result=eval_data,
                model_name=model_short,
                gems_op_list=gems_op_list,
                enable_op_count=enable_op_count,
                total_op_count=total_op_count,
                op_replace_rate=None,
                operator_version=flaggems_ver,
                compiler_version=flagtree_ver,
                framework_version=framework_ver,
                comm_lib_version=flagcx_ver,
            ))
            print(f"  ✓ pl 记录已构造 ({model_short}_{vendor}_plugin{ts_suffix}, {'达标' if pl_perf_ok else '不达标'})")
        else:
            print(f"  ⚠ 跳过 pl 记录（plugin_performance.json 不存在）")
    else:
        print(f"  - 跳过 pl 记录（plugin 流程未触发）")

    if not items:
        print("错误: 无任何可上传的记录", file=sys.stderr)
        sys.exit(1)

    # 输出或上传
    payload = {"items": items}

    if args.dry_run:
        print("\n[dry-run] 将上传以下 payload:\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n共 {len(items)} 条记录")
        sys.exit(0)

    # 检查 token
    token = os.environ.get("FLAGRELEASE_API_TOKEN", "")
    if not token:
        print("错误: 未设置环境变量 FLAGRELEASE_API_TOKEN", file=sys.stderr)
        sys.exit(2)

    # 先尝试 insert，如果有记录已存在则拆分处理
    insert_url = args.api_url
    update_url = args.api_url.replace("/insert", "/update")

    print(f"\n  上传 {len(items)} 条记录到 {insert_url} ...")
    result = call_api(insert_url, token, items)

    if result.get("success"):
        data_results = result.get("result", {}).get("data_results", [])
        print(f"  ✓ 上传成功 (insert): {result['result'].get('success_count', 0)}/{result['result'].get('total_count', 0)}")
        for dr in data_results:
            print(f"    {dr.get('record_id', '?')}: task_id={dr.get('mig_task_id', 'N/A')}")
    else:
        errors = result.get("errors", {})
        detail = str(errors.get("detail", "")) if isinstance(errors, dict) else str(errors)
        if "已存在" in detail:
            # 解析哪些记录已存在，拆分为 insert 和 update 两批
            existing_names = set()
            for item in items:
                if item["name"] in detail:
                    existing_names.add(item["name"])

            new_items = [it for it in items if it["name"] not in existing_names]
            update_items = [it for it in items if it["name"] in existing_names]

            success_count = 0
            total_count = len(items)

            # insert 新记录
            if new_items:
                print(f"  → insert {len(new_items)} 条新记录...")
                r = call_api(insert_url, token, new_items)
                if r.get("success"):
                    sc = r.get("result", {}).get("success_count", 0)
                    success_count += sc
                    print(f"    ✓ insert 成功: {sc} 条")
                else:
                    e = r.get("errors", {})
                    print(f"    ⚠ insert 失败: {e.get('detail', e) if isinstance(e, dict) else e}")

            # update 已存在记录
            if update_items:
                print(f"  → update {len(update_items)} 条已存在记录...")
                r = call_api(update_url, token, update_items)
                if r.get("success"):
                    sc = r.get("result", {}).get("success_count", 0)
                    success_count += sc
                    print(f"    ✓ update 成功: {sc} 条")
                else:
                    e = r.get("errors", {})
                    print(f"    ⚠ update 失败: {e.get('detail', e) if isinstance(e, dict) else e}")

            if success_count == total_count:
                print(f"  ✓ 全部完成: {success_count}/{total_count}")
            elif success_count > 0:
                print(f"  ⚠ 部分完成: {success_count}/{total_count}")
            else:
                print(f"  ✗ 全部失败", file=sys.stderr)
                sys.exit(2)
        else:
            print(f"  ✗ 上传失败: {detail or errors}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
