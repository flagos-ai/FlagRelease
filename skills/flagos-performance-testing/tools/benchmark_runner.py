#!/usr/bin/env python3
"""
vLLM 性能基准测试工具

两档测试策略:
- quick: 只跑 4k_input_1k_output 并发 64，预热后单次测试（主流程默认）
- comprehensive: 所有用例，所有并发全跑

Usage:
    python benchmark_runner.py --config config.yaml --strategy quick
    python benchmark_runner.py --config config.yaml --strategy comprehensive
    python benchmark_runner.py --output-name native_performance
    python benchmark_runner.py --test-case 1k_input_1k_output
    python benchmark_runner.py --dry-run
"""

import sys

# IO 缓冲修复：确保容器内实时输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    import functools
    print = functools.partial(print, flush=True)

import argparse
import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# error_writer 集成
sys.path.insert(0, str(Path(__file__).resolve().parent))
# service_monitor: 性能测试期间服务活性监控（容器内 scripts/ 同目录，repo 内跨 skill）
_this_dir_bm = Path(__file__).resolve().parent
for _p in [_this_dir_bm, _this_dir_bm.parent.parent.parent / "flagos-service-startup" / "tools"]:
    if (_p / "service_monitor.py").is_file():
        sys.path.insert(0, str(_p)); break
try:
    from error_writer import write_last_error, write_checkpoint
except ImportError:
    def write_last_error(*a, **kw): pass
    def write_checkpoint(*a, **kw): pass

try:
    from service_monitor import ServiceMonitor, find_latest_startup_log
except ImportError:
    ServiceMonitor = None
    find_latest_startup_log = None

# =============================================================================
# 配置加载
# =============================================================================

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "perf_config.yaml"

DEFAULT_TIMEOUT = 1800  # 30 minutes


def load_yaml(path: Path) -> Dict[str, Any]:
    """加载 YAML 文件"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件，缺失字段自动从 context.yaml 回填"""
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_yaml(cfg_path)

    if "server" not in config:
        config["server"] = {"host": "", "port": 8000}
    if "model" not in config:
        config["model"] = {"name": "", "tokenizer_path": ""}

    # Fallback: 从 context.yaml 补充缺失的 server.host / model.tokenizer_path
    if not config["server"].get("host") or not config["model"].get("tokenizer_path"):
        ctx_path = Path("/flagos-workspace/shared/context.yaml")
        if ctx_path.exists():
            try:
                ctx = load_yaml(ctx_path)
                svc = ctx.get("service", {})
                if not config["server"].get("host"):
                    config["server"]["host"] = svc.get("host", "127.0.0.1")
                if not config["server"].get("port") or config["server"]["port"] == 8000:
                    port = svc.get("port")
                    if port:
                        config["server"]["port"] = port
                if not config["model"].get("tokenizer_path"):
                    config["model"]["tokenizer_path"] = ctx.get("model", {}).get("container_path", "")
                if not config["model"].get("name"):
                    config["model"]["name"] = ctx.get("model", {}).get("name", "")
                print(f"[INFO] 从 context.yaml 补充了缺失配置")
            except Exception as e:
                print(f"[WARN] 读取 context.yaml 失败: {e}")

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置完整性"""
    errors = []

    if not config.get("server", {}).get("host"):
        errors.append("server.host 未配置 (检查 /flagos-workspace/shared/context.yaml)")
    if not config.get("model", {}).get("tokenizer_path"):
        errors.append("model.tokenizer_path 未配置 (检查 /flagos-workspace/shared/context.yaml)")
    if not config.get("test_matrix"):
        errors.append("test_matrix 为空")
    if not config.get("concurrency", {}).get("levels"):
        errors.append("concurrency.levels 未配置")

    for err in errors:
        print(f"ERROR: {err}")

    return len(errors) == 0


# =============================================================================
# 输出解析
# =============================================================================

METRIC_PATTERNS = {
    'Successful requests': r'Successful requests:\s+(\d+)',
    'Failed requests': r'Failed requests:\s+(\d+)',
    'Benchmark duration (s)': r'Benchmark duration \(s\):\s+([\d.]+)',
    'Total input tokens': r'Total input tokens:\s+(\d+)',
    'Total generated tokens': r'Total generated tokens:\s+(\d+)',
    'Request throughput (req/s)': r'Request throughput \(req/s\):\s+([\d.]+)',
    'Output token throughput (tok/s)': r'Output token throughput \(tok/s\):\s+([\d.]+)',
    'Total token throughput (tok/s)': r'Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)',
    'Peak output token throughput (tok/s)': r'Peak output token throughput \(tok/s\):\s+([\d.]+)',
    'Peak concurrent requests': r'Peak concurrent requests:\s+(\d+)',
    'Mean TTFT (ms)': r'Mean TTFT \(ms\):\s+([\d.]+)',
    'Median TTFT (ms)': r'Median TTFT \(ms\):\s+([\d.]+)',
    'P99 TTFT (ms)': r'P99 TTFT \(ms\):\s+([\d.]+)',
    'Mean TPOT (ms)': r'Mean TPOT \(ms\):\s+([\d.]+)',
    'Median TPOT (ms)': r'Median TPOT \(ms\):\s+([\d.]+)',
    'P99 TPOT (ms)': r'P99 TPOT \(ms\):\s+([\d.]+)',
    'Mean ITL (ms)': r'Mean ITL \(ms\):\s+([\d.]+)',
    'Median ITL (ms)': r'Median ITL \(ms\):\s+([\d.]+)',
    'P99 ITL (ms)': r'P99 ITL \(ms\):\s+([\d.]+)',
}


def parse_output(output: str) -> Dict[str, Any]:
    """从 vllm bench 输出中提取指标"""
    metrics = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            val = match.group(1)
            metrics[key] = float(val) if '.' in val else int(val)
        else:
            metrics[key] = None
    return metrics


# =============================================================================
# 基准测试执行
# =============================================================================

def build_command(config: Dict[str, Any], test_case: Dict[str, Any]) -> List[str]:
    """构建 vllm bench 命令"""
    server = config["server"]
    model = config["model"]
    bench = config.get("benchmark", {})

    cmd = [
        "vllm", "bench", "serve",
        "--host", server["host"],
        "--port", str(server["port"]),
        "--model", model["name"],
        "--tokenizer", model["tokenizer_path"],
        "--dataset-name", bench.get("dataset_name", "random"),
        "--random-input-len", str(test_case["input_len"]),
        "--random-output-len", str(test_case["output_len"]),
        "--endpoint", bench.get("endpoint", "/v1/completions"),
    ]

    if bench.get("ignore_eos", True):
        cmd.append("--ignore-eos")
    if bench.get("trust_remote_code", True):
        cmd.append("--trust-remote-code")

    return cmd


def run_benchmark(cmd: List[str], num_prompts: int, max_concurrency: Optional[int] = None,
                  dry_run: bool = False) -> Dict[str, Any]:
    """执行单次基准测试"""
    full_cmd = cmd + ["--num-prompts", str(num_prompts)]
    if max_concurrency:
        full_cmd += ["--max-concurrency", str(max_concurrency)]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(full_cmd)}")
        return {"dry_run": True}

    conc_str = f"concurrency={max_concurrency}" if max_concurrency else "unlimited"
    print(f"  Running: num_prompts={num_prompts}, {conc_str}")

    try:
        proc = subprocess.Popen(
            full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout_lines = []
        import threading, time

        # 后台线程读取 stderr 防止死锁
        stderr_lines = []
        def read_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)
        t = threading.Thread(target=read_stderr, daemon=True)
        t.start()

        # 实时逐行读取 stdout
        for line in proc.stdout:
            stdout_lines.append(line)
            stripped = line.strip()
            if stripped:
                print(f"    | {stripped}")

        proc.wait()
        t.join(timeout=5)
        full_stdout = "".join(stdout_lines)
        full_stderr = "".join(stderr_lines)

        if proc.returncode != 0:
            print(f"    FAILED (rc={proc.returncode}): {full_stderr[:200]}")
            return {"error": full_stderr}

        metrics = parse_output(full_stdout)
        throughput = metrics.get('Output token throughput (tok/s)', 'N/A')
        total_tp = metrics.get('Total token throughput (tok/s)', 'N/A')
        failed = metrics.get('Failed requests', 0)
        print(f"    OK - output={throughput} tok/s, total={total_tp} tok/s, failed={failed}")
        return metrics

    except Exception as e:
        print(f"    ERROR: {e}")
        return {"error": str(e)}


# =============================================================================
# 测试策略
# =============================================================================

# Quick 模式固定并发
QUICK_CONCURRENCY = 64

# Quick 模式多轮测试：跑 5 轮，丢弃第 1 轮（自然预热），取后 4 轮均值
QUICK_TOTAL_ROUNDS = 5
QUICK_DISCARD_ROUNDS = 1

# 预热请求数（comprehensive 模式使用）
WARMUP_NUM_PROMPTS = 2
WARMUP_CONCURRENCY = 2

# Quick 模式硬编码用例名
QUICK_TEST_CASE_NAME = "4k_input_1k_output"


def run_test_case(config: Dict[str, Any], test_case: Dict[str, Any],
                  dry_run: bool = False, strategy: str = "quick") -> Dict[str, Any]:
    """运行单个测试用例的所有并发级别，返回结果中包含 _elapsed_seconds"""
    tc_start = time.time()
    base_cmd = build_command(config, test_case)

    levels = config["concurrency"]["levels"]

    if strategy == "quick":
        results = run_quick_test(base_cmd, dry_run)
    else:
        # comprehensive: 所有并发全跑，不早停
        results = run_comprehensive_test(base_cmd, levels, dry_run)

    results["_elapsed_seconds"] = round(time.time() - tc_start, 1)
    return results


def run_comprehensive_test(base_cmd: List[str], levels: List[int],
                           dry_run: bool = False) -> Dict[str, Any]:
    """
    Comprehensive 模式：所有并发级别全跑，不早停。

    num_prompts = concurrency，逐级测试。
    """
    results = {}
    best_throughput = 0.0
    best_concurrency = levels[0]

    # 预热
    if not dry_run:
        print(f"  [WARMUP] Sending {WARMUP_NUM_PROMPTS} warmup requests (concurrency={WARMUP_CONCURRENCY}) ...")
        run_benchmark(base_cmd, WARMUP_NUM_PROMPTS, WARMUP_CONCURRENCY, dry_run=False)
        print(f"  [WARMUP] Done, starting benchmark")

    print(f"  [COMPREHENSIVE MODE] levels={levels}, num_prompts=concurrency")

    for conc in levels:
        metrics = run_benchmark(base_cmd, conc, conc, dry_run)
        results[str(conc)] = metrics

        if dry_run or "error" in metrics:
            if "error" in metrics and not dry_run:
                print(f"    Error at concurrency={conc}: {metrics['error'][:100]}")
            continue

        current_throughput = metrics.get('Output token throughput (tok/s)', 0) or 0
        if current_throughput > best_throughput:
            best_throughput = current_throughput
            best_concurrency = conc

    results["_search_meta"] = {
        "best_concurrency": best_concurrency,
        "best_throughput": best_throughput,
        "tested_levels": levels,
        "all_levels_tested": True,
    }

    return results


def average_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多轮 metrics 取均值，数值型字段平均，非数值取最后一轮"""
    if not metrics_list:
        return {}
    if len(metrics_list) == 1:
        return metrics_list[0]

    averaged = {}
    for key in METRIC_PATTERNS:
        values = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if values and all(isinstance(v, (int, float)) for v in values):
            averaged[key] = round(sum(values) / len(values), 1)
        else:
            averaged[key] = metrics_list[-1].get(key)

    for key in metrics_list[-1]:
        if key not in averaged:
            averaged[key] = metrics_list[-1][key]

    return averaged


def run_quick_test(base_cmd: List[str],
                   dry_run: bool = False) -> Dict[str, Any]:
    """
    Quick 模式：跑 5 轮，第 1 轮作为自然预热丢弃，取后 4 轮均值。

    每轮 concurrency=64, num_prompts=64。
    """
    total_rounds = QUICK_TOTAL_ROUNDS
    discard = QUICK_DISCARD_ROUNDS
    used_rounds = total_rounds - discard

    print(f"  [QUICK MODE] concurrency={QUICK_CONCURRENCY}, rounds={total_rounds} (discard first {discard}, average last {used_rounds})")

    if dry_run:
        for i in range(1, total_rounds + 1):
            run_benchmark(base_cmd, QUICK_CONCURRENCY, QUICK_CONCURRENCY, dry_run=True)
        return {str(QUICK_CONCURRENCY): {"dry_run": True}, "_search_meta": {"quick_mode": True}}

    all_round_metrics = []
    for i in range(1, total_rounds + 1):
        if i <= discard:
            print(f"\n  [WARMUP ROUND {i}/{total_rounds}]")
        else:
            print(f"\n  [ROUND {i}/{total_rounds}]")

        metrics = run_benchmark(base_cmd, QUICK_CONCURRENCY, QUICK_CONCURRENCY, dry_run=False)
        all_round_metrics.append(metrics)

    valid_metrics = [m for m in all_round_metrics[discard:] if "error" not in m]

    if not valid_metrics:
        final_metrics = all_round_metrics[-1]
    else:
        final_metrics = average_metrics(valid_metrics)

    results = {str(QUICK_CONCURRENCY): final_metrics}

    throughput = 0
    if "error" not in final_metrics:
        throughput = final_metrics.get('Output token throughput (tok/s)', 0) or 0

    per_round_tps = []
    for m in all_round_metrics[discard:]:
        if "error" not in m:
            per_round_tps.append(m.get('Output token throughput (tok/s)', 0) or 0)
        else:
            per_round_tps.append(None)

    results["_search_meta"] = {
        "best_concurrency": QUICK_CONCURRENCY,
        "best_throughput": throughput,
        "tested_levels": [QUICK_CONCURRENCY],
        "all_levels_tested": True,
        "quick_mode": True,
        "rounds_total": total_rounds,
        "rounds_used": len(valid_metrics),
        "per_round_throughputs": per_round_tps,
    }

    return results


# =============================================================================
# 结果保存
# =============================================================================

def save_results(results: Dict[str, Any], config: Dict[str, Any],
                 output_name: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 mode: Optional[str] = None,
                 timing: Optional[Dict[str, Any]] = None) -> str:
    """保存测试结果到 JSON 文件（扁平格式，不含 metadata 包装和 _search_meta）"""
    output_cfg = config.get("output", {})

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(output_cfg.get("dir", "./output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_name:
        filepath = out_dir / f"{output_name}.json"
    else:
        filepath = out_dir / f"benchmark_{timestamp}.json"

    # 扁平格式：直接输出 {tc_name: {concurrency: metrics, ...}, ...}
    # 排除内部使用的 _search_meta
    data = {}
    for tc_name, tc_results in results.items():
        if not isinstance(tc_results, dict):
            data[tc_name] = tc_results
            continue
        cleaned = {k: v for k, v in tc_results.items() if not k.startswith("_")}
        if cleaned:
            data[tc_name] = cleaned

    # 添加 _meta 说明
    data["_meta"] = {
        "说明": "vLLM 性能基准测试结果",
        "格式": "{test_case: {concurrency: {metric: value}}}",
        "关键指标": "Output token throughput (tok/s) 和 Total token throughput (tok/s)",
        "mode": mode or "default",
        "timestamp": datetime.now().isoformat(),
    }
    if timing:
        data["_meta"]["timing"] = timing

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return str(filepath)


# =============================================================================
# 摘要输出
# =============================================================================

def print_summary(results: Dict[str, Any], mode: str = "default"):
    """打印测试结果摘要（逐并发级别）"""
    print(f"\n{'='*60}")
    print(f"性能测试摘要 (mode: {mode})")
    print(f"{'='*60}")

    for tc_name, tc_results in results.items():
        print(f"\n{tc_name}:")

        if not isinstance(tc_results, dict):
            continue

        # 逐并发级别打印结果
        conc_keys = sorted(
            [k for k in tc_results if not k.startswith("_") and isinstance(tc_results[k], dict) and "error" not in tc_results[k]],
            key=lambda x: int(x) if x.isdigit() else 0
        )

        for key in conc_keys:
            metrics = tc_results[key]
            output_tp = metrics.get('Output token throughput (tok/s)', 'N/A')
            total_tp = metrics.get('Total token throughput (tok/s)', 'N/A')
            ttft = metrics.get('Mean TTFT (ms)', 'N/A')
            tpot = metrics.get('Mean TPOT (ms)', 'N/A')
            print(f"  concurrency={key}: output={output_tp} tok/s, total={total_tp} tok/s, TTFT={ttft}ms, TPOT={tpot}ms")

        # 显示搜索元信息
        meta = tc_results.get("_search_meta")
        if meta:
            tested = meta.get('tested_levels', [])
            if not meta.get('all_levels_tested', True):
                print(f"  Tested levels:     {tested} (early-stopped)")


# =============================================================================
# Strategy 解析
# =============================================================================

STRATEGY_CHOICES = ['quick', 'comprehensive']


def resolve_strategy(args) -> str:
    """
    解析 strategy，优先级：--strategy > --quick > 默认 quick
    """
    if args.strategy:
        return args.strategy
    if args.quick:
        return "quick"
    return "quick"


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="vLLM 性能基准测试 (重构版)")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--test-case", help="运行指定测试用例")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令")
    parser.add_argument("--strategy", choices=STRATEGY_CHOICES,
                        help="测试策略: quick(只跑4k_input_1k_output并发64,默认) / comprehensive(全跑)")
    # 向后兼容别名
    parser.add_argument("--quick", action="store_true",
                        help="(向后兼容) 等同于 --strategy quick")
    parser.add_argument("--output-name", help="输出文件名（不含扩展名）")
    parser.add_argument("--output-dir", help="输出目录路径（默认 /flagos-workspace/results/）",
                        default=None)
    parser.add_argument("--mode", help="测试模式标记 (native/flagos_initial/flagos_optimized)",
                        default="default")
    args = parser.parse_args()

    # 解析 strategy
    strategy = resolve_strategy(args)

    # 加载配置
    print("加载配置...")
    config = load_config(args.config)

    if not validate_config(config):
        sys.exit(1)

    # 筛选测试用例
    test_matrix = [tc for tc in config["test_matrix"] if tc.get("enabled", True)]

    if args.test_case:
        test_matrix = [tc for tc in test_matrix if tc["name"] == args.test_case]
        if not test_matrix:
            print(f"ERROR: 测试用例 '{args.test_case}' 不存在或未启用")
            sys.exit(1)

    # quick 模式只跑 4k_input_1k_output
    if strategy == "quick":
        test_matrix = [tc for tc in test_matrix if tc["name"] == QUICK_TEST_CASE_NAME]
        if not test_matrix:
            print(f"ERROR: quick 模式需要 '{QUICK_TEST_CASE_NAME}' 用例，但未找到或未启用")
            sys.exit(1)

    print(f"\n策略: {strategy}")
    print(f"测试用例: {[tc['name'] for tc in test_matrix]}")
    print(f"模式: {args.mode}")

    # 启动服务活性监控
    monitor = None
    if ServiceMonitor is not None:
        log_path = find_latest_startup_log() if find_latest_startup_log else None
        monitor = ServiceMonitor(log_path=log_path)
        monitor.start()
        if log_path:
            print(f"[MONITOR] 服务活性监控已启动 (日志: {log_path})")
        else:
            print(f"[MONITOR] 服务活性监控已启动 (仅进程检测)")

    # 执行测试
    all_results = {}
    tc_timings = {}
    total_start = time.time()
    service_crashed = False
    for tc in test_matrix:
        # 每个用例前检查服务状态
        if monitor and monitor.is_dead():
            reason = monitor.death_reason()
            print(f"\n[MONITOR] 服务崩溃: {reason.get('detail', '未知')}")
            if reason.get('log_line'):
                print(f"[MONITOR] 日志: {reason['log_line']}")
            print(f"[MONITOR] 跳过剩余测试用例")
            service_crashed = True
            break
        print(f"\n{'='*50}")
        print(f"测试用例: {tc['name']} (input={tc['input_len']}, output={tc['output_len']})")
        print('='*50)
        all_results[tc["name"]] = run_test_case(
            config, tc, args.dry_run,
            strategy=strategy
        )
        tc_timings[tc["name"]] = all_results[tc["name"]].get("_elapsed_seconds", 0)
    total_elapsed = round(time.time() - total_start, 1)

    if monitor:
        if not service_crashed and monitor.is_dead():
            reason = monitor.death_reason()
            print(f"\n[MONITOR] ⚠ 测试期间服务崩溃: {reason.get('detail', '未知')}")
            if reason.get('log_line'):
                print(f"[MONITOR] 日志: {reason['log_line']}")
            print("[MONITOR] 性能测试结果可能不完整")
            service_crashed = True
        monitor.stop()

    # 打印摘要
    if not args.dry_run:
        print_summary(all_results, args.mode)

    # 保存结果
    if not args.dry_run:
        timing = {
            "total_seconds": total_elapsed,
            "per_test_case": tc_timings,
            "timestamp_start": datetime.fromtimestamp(total_start).isoformat(),
            "timestamp_end": datetime.now().isoformat(),
        }
        if service_crashed and monitor:
            timing["service_crashed"] = True
            timing["crash_reason"] = monitor.death_reason()
        output_path = save_results(
            all_results, config,
            output_name=args.output_name,
            output_dir=args.output_dir,
            mode=args.mode,
            timing=timing,
        )
        print(f"\n结果已保存: {output_path} (耗时 {total_elapsed}s)")

    return all_results


if __name__ == "__main__":
    try:
        step_id = os.environ.get("FLAGOS_STEP_ID", "06_quick_performance")
        step_title = os.environ.get("FLAGOS_STEP_TITLE", "性能评测")
        write_checkpoint(step_id, step_title, "running_benchmark",
                         action_detail=" ".join(sys.argv))
        main()
    except Exception as e:
        write_last_error(
            tool="benchmark_runner.py",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=traceback.format_exc(),
        )
        print(f"[FATAL] benchmark_runner.py 异常退出: {e}")
        traceback.print_exc()
        sys.exit(1)
