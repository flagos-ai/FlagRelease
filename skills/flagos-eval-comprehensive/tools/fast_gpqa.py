#!/usr/bin/env python3
"""
GPQA Diamond 快速精度评测脚本

自动适配所有模型（thinking/non-thinking），自动探测吞吐选并发，一条命令跑完。

用法:
  python fast_gpqa.py --config config.yaml
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
import yaml

# error_writer 集成（容器内: eval/ 目录，error_writer 在 scripts/ 目录）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# service_monitor: 评测期间服务活性监控（容器内 scripts/ 同目录，repo 内跨 skill）
_this_dir = Path(__file__).resolve().parent
for _p in [_this_dir, _this_dir.parent / "scripts",
           _this_dir.parent.parent.parent / "flagos-service-startup" / "tools"]:
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
# Thinking 模型检测
# =============================================================================

THINKING_PATTERNS = ['qwen3', 'qwq', 'deepseek-r1', 'deepseek-r2']


def detect_thinking(model_name: str) -> bool:
    """根据模型名自动检测是否为 thinking model。"""
    name_lower = model_name.lower()
    return any(p in name_lower for p in THINKING_PATTERNS)


# =============================================================================
# 模型服务查询
# =============================================================================

def query_model_max_len(api_base: str, api_key: str, model_name: str) -> Optional[int]:
    """查询 /v1/models 获取模型的 max_model_len。"""
    try:
        base = api_base.rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3]
        url = f"{base}/v1/models"

        headers = {}
        if api_key and api_key != 'EMPTY':
            headers['Authorization'] = f'Bearer {api_key}'

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for m in data.get('data', []):
            if m.get('id') == model_name:
                val = m.get('max_model_len')
                if val is not None:
                    return int(val)

        # 只有一个模型时直接取
        models = data.get('data', [])
        if len(models) == 1:
            val = models[0].get('max_model_len')
            if val is not None:
                return int(val)
    except Exception as e:
        print(f"[WARN] 查询模型 max_model_len 失败: {e}")

    return None


def auto_max_tokens(api_base: str, api_key: str, model_name: str, is_thinking: bool = False) -> Tuple[int, Optional[int]]:
    """
    自动计算 max_tokens，基于服务端实际 max_model_len。

    thinking 模型：max(max_model_len - 8192, 8192)，无上限 cap
    标准模型：clamp(max_model_len - 8192, 4096, 32768)

    Returns:
        (max_tokens, max_model_len or None)
    """
    max_model_len = query_model_max_len(api_base, api_key, model_name)
    if max_model_len:
        tokens = max_model_len - 8192  # 预留 8K 给 prompt
        if is_thinking:
            tokens = max(tokens, 8192)
        else:
            tokens = max(tokens, 4096)
            tokens = min(tokens, 32768)
        return tokens, max_model_len
    # fallback
    return (16384 if is_thinking else 8192), None


# =============================================================================
# 截断检测
# =============================================================================

GPQA_SAMPLE_QUESTION = (
    "What is the probability that a randomly chosen integer between 1 and 100 "
    "is divisible by both 3 and 7? Show your reasoning step by step."
)


def check_truncation(
    api_base: str,
    api_key: str,
    model_name: str,
    max_tokens: int,
    max_model_len: Optional[int],
) -> Tuple[bool, int]:
    """
    发一条样题检查 finish_reason 是否为 length（截断）。

    如果截断，自动将 max_tokens 翻倍（在 max_model_len 允许范围内）。

    Returns:
        (truncation_detected, adjusted_max_tokens)
    """
    base = api_base.rstrip('/')
    if not base.endswith('/v1'):
        base = base + '/v1'
    url = f"{base}/chat/completions"

    headers = {'Content-Type': 'application/json'}
    if api_key and api_key != 'EMPTY':
        headers['Authorization'] = f'Bearer {api_key}'

    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': GPQA_SAMPLE_QUESTION}],
        'max_tokens': max_tokens,
        'temperature': 0.0,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get('choices', [])
        if choices:
            finish_reason = choices[0].get('finish_reason', '')
            if finish_reason == 'length':
                print(f"[WARN] 截断检测: finish_reason=length, max_tokens={max_tokens} 不足")
                # 尝试翻倍
                new_tokens = max_tokens * 2
                if max_model_len:
                    cap = max_model_len - 2048  # 留 2K 给 prompt
                    new_tokens = min(new_tokens, cap)
                new_tokens = max(new_tokens, max_tokens)  # 至少不降
                if new_tokens > max_tokens:
                    print(f"[WARN] 自动调整 max_tokens: {max_tokens} → {new_tokens}")
                return True, new_tokens
            else:
                print(f"[OK] 截断检测通过: finish_reason={finish_reason}")
    except Exception as e:
        print(f"[WARN] 截断检测请求失败: {e}")

    return False, max_tokens


# =============================================================================
# 探测吞吐 & 选并发
# =============================================================================

def _sanitize_model_id(model_name: str) -> str:
    """将模型名清理为安全的 model_id（不含 / 等特殊字符）。"""
    return model_name.strip('/').split('/')[-1] or model_name


def _preload_dataset(dataset_hub: str, dataset_dir: Optional[str] = None):
    """预加载 gpqa_diamond 数据集到缓存，确保探测阶段计时不含下载时间。"""
    try:
        if dataset_dir:
            # 检查本地缓存是否存在
            import glob as glob_mod
            if glob_mod.glob(os.path.join(dataset_dir, '**', 'gpqa*'), recursive=True):
                return
        if dataset_hub == 'modelscope':
            from modelscope import MsDataset
            MsDataset.load('AI-ModelScope/gpqa_diamond', split='train', trust_remote_code=True)
        else:
            import datasets as hf_datasets
            hf_datasets.load_dataset('Idavidrein/gpqa', name='gpqa_diamond', split='train', trust_remote_code=True)
    except Exception:
        pass  # 预加载失败不影响后续，evalscope 会自行下载


def _probe_single_latency(api_base: str, api_key: str, model_name: str,
                           max_tokens: int, is_thinking: bool) -> float:
    """直接调 OpenAI API 测一条推理的纯推理时间（剥离 evalscope 框架开销）"""
    SAMPLE_QUESTION = (
        "What is the result of the Diels-Alder reaction between cyclopentadiene "
        "and maleic anhydride? Choose the most likely product."
    )
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': SAMPLE_QUESTION}],
        'max_tokens': max_tokens,
        'temperature': 0.6 if is_thinking else 0.0,
    }
    headers = {'Content-Type': 'application/json'}
    if api_key and api_key != 'EMPTY':
        headers['Authorization'] = f'Bearer {api_key}'

    base = api_base.rstrip('/')
    if not base.endswith('/v1'):
        base = base + '/v1'
    url = f"{base}/chat/completions"

    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    latency = time.time() - start
    return latency


def _estimate_concurrency(latency: float, is_thinking: bool) -> list:
    """基于单条延迟估算候选并发范围。thinking 模型输出长度波动大，保守选择。"""
    if is_thinking:
        if latency <= 10:
            return [8, 16, 32]
        elif latency <= 30:
            return [4, 8, 16]
        elif latency <= 60:
            return [2, 4, 8]
        else:
            return [1, 2, 4]
    else:
        if latency <= 3:
            return [16, 32, 64]
        elif latency <= 10:
            return [8, 16, 32]
        elif latency <= 30:
            return [4, 8, 16]
        else:
            return [2, 4, 8]


def _run_concurrent_probe(api_base: str, api_key: str, model_name: str,
                           max_tokens: int, is_thinking: bool,
                           concurrency: int, num_requests: int = 3) -> Tuple[float, int]:
    """并发发 num_requests 个请求，返回 (throughput_rps, error_count)"""
    import concurrent.futures

    SAMPLE_QUESTION = (
        "What is the result of the Diels-Alder reaction between cyclopentadiene "
        "and maleic anhydride? Choose the most likely product."
    )
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': SAMPLE_QUESTION}],
        'max_tokens': max_tokens,
        'temperature': 0.6 if is_thinking else 0.0,
    }
    headers = {'Content-Type': 'application/json'}
    if api_key and api_key != 'EMPTY':
        headers['Authorization'] = f'Bearer {api_key}'

    base = api_base.rstrip('/')
    if not base.endswith('/v1'):
        base = base + '/v1'
    url = f"{base}/chat/completions"

    actual_n = min(concurrency, num_requests)
    errors = 0

    def _send_one():
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=300)
            r.raise_for_status()
            return True
        except Exception:
            return False

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_one) for _ in range(actual_n)]
        for f in concurrent.futures.as_completed(futures):
            if not f.result():
                errors += 1
    elapsed = time.time() - start

    throughput = (actual_n - errors) / elapsed if elapsed > 0 else 0
    return throughput, errors


def _validate_concurrency(api_base: str, api_key: str, model_name: str,
                           candidates: list, max_tokens: int,
                           is_thinking: bool) -> int:
    """对候选并发各跑 3 题，选吞吐最高且无 OOM/超时的。"""
    best_concurrency = candidates[0]
    best_throughput = 0.0

    for c in candidates:
        throughput, errors = _run_concurrent_probe(
            api_base, api_key, model_name, max_tokens, is_thinking,
            concurrency=c, num_requests=3,
        )
        print(f"  并发 {c}: throughput={throughput:.2f} rps, errors={errors}")
        if errors == 0 and throughput > best_throughput:
            best_throughput = throughput
            best_concurrency = c
        elif errors > 0:
            print(f"  并发 {c} 出现错误，跳过更高并发")
            break

    return best_concurrency


def probe_throughput(
    model_name: str,
    api_url: str,
    api_key: str,
    generation_config: Dict,
    dataset_args: Dict,
    evalscope_config: Dict,
) -> Tuple[int, float]:
    """
    三阶段并发探测：
    1. 直接 API 调用测单条推理延迟（剥离 evalscope 框架开销）
    2. 基于延迟 + thinking 模型特性估算候选并发
    3. 快速验证（3 题并发测试，选最优）

    Returns:
        (eval_batch_size, probe_elapsed_seconds)
    """
    is_thinking = detect_thinking(model_name)
    max_tokens = generation_config.get('max_tokens', 4096)

    # 阶段 1: 纯 API 延迟
    print("[PROBE] 阶段1: 测量单条推理延迟（直接 API 调用）...")
    try:
        latency = _probe_single_latency(api_url, api_key, model_name, max_tokens, is_thinking)
        print(f"[PROBE] 单条延迟: {latency:.1f}s (thinking={is_thinking})")
    except Exception as e:
        print(f"[PROBE] 延迟探测失败: {e}")
        print("[PROBE] 使用默认并发 16")
        return 16, 0.0

    # 阶段 2: 估算候选
    candidates = _estimate_concurrency(latency, is_thinking)
    print(f"[PROBE] 候选并发: {candidates}")

    # 阶段 3: 快速验证
    print("[PROBE] 阶段2: 验证候选并发（每档 3 题）...")
    best = _validate_concurrency(api_url, api_key, model_name, candidates, max_tokens, is_thinking)
    print(f"[PROBE] 最终选择并发: {best}")

    return best, latency


# =============================================================================
# 结果解析
# =============================================================================

def parse_result(result: Dict) -> Tuple[Optional[float], Dict]:
    """
    解析 EvalScope run_task 返回的结果。

    Returns:
        (score_percentage, raw_details)
    """
    if not result or 'error' in result:
        return None, result or {}

    for key, val in result.items():
        # Report 对象 → 转 dict
        if hasattr(val, 'to_dict'):
            val_dict = val.to_dict()
            score = val_dict.get('score')
            if score is not None:
                pct = score * 100 if score <= 1.0 else score
                return round(pct, 2), val_dict

        if isinstance(val, dict):
            score = _find_score(val)
            if score is not None:
                pct = score * 100 if score <= 1.0 else score
                return round(pct, 2), val

    return None, dict(result)


def _find_score(d: dict, depth: int = 0) -> Optional[float]:
    """递归查找 score/accuracy 字段。"""
    if depth > 3:
        return None
    for key in ('score', 'accuracy', 'acc', 'mean_acc'):
        if key in d and isinstance(d[key], (int, float)):
            return float(d[key])
    for val in d.values():
        if isinstance(val, dict):
            s = _find_score(val, depth + 1)
            if s is not None:
                return s
    return None


# =============================================================================
# 主流程
# =============================================================================

def run_fast_gpqa(
    model_name: str,
    api_base: str,
    api_key: str = 'EMPTY',
    dataset_dir: Optional[str] = None,
    dataset_hub: str = 'modelscope',
    limit: Optional[int] = 50,
    output_path: Optional[str] = None,
) -> Dict:
    """
    GPQA Diamond 快速评测主流程。

    Returns:
        结果 dict
    """
    from evalscope import TaskConfig, run_task
    from evalscope.constants import EvalType

    total_start = time.time()

    print("=" * 60)
    print("  GPQA Diamond 快速精度评测")
    print("=" * 60)
    print(f"  模型: {model_name}")
    print(f"  API:  {api_base}")

    # Step 1: 检测 thinking 模型
    is_thinking = detect_thinking(model_name)
    mode_str = "thinking" if is_thinking else "standard"
    print(f"  模式: {mode_str}")

    # Step 2: 自动设 max_tokens（基于 max_model_len 动态计算）
    max_tokens, max_model_len = auto_max_tokens(api_base, api_key, model_name, is_thinking)
    if max_model_len:
        print(f"  max_model_len: {max_model_len} (从服务端获取)")
    else:
        print(f"  max_model_len: 未知 (使用 fallback)")
    print(f"  max_tokens: {max_tokens}")

    # Step 3: 截断检测 — 发样题检查 finish_reason
    truncation_detected, max_tokens = check_truncation(
        api_base, api_key, model_name, max_tokens, max_model_len,
    )

    # Step 4: 构建 generation_config
    if is_thinking:
        gen_config = {
            'max_tokens': max_tokens,
            'temperature': 0.6,
            'top_p': 0.95,
            'stream': True,
            'timeout': 120000,
            'n': 1,
        }
    else:
        gen_config = {
            'max_tokens': max_tokens,
            'temperature': 0.0,
            'top_p': 1.0,
            'stream': True,
            'timeout': 120000,
            'n': 1,
        }

    # Step 5: 构建 dataset_args
    dataset_args = {'gpqa_diamond': {'few_shot_num': 0}}
    if is_thinking:
        dataset_args['gpqa_diamond']['filters'] = {'remove_until': '</think>'}

    evalscope_config = {
        'dataset_hub': dataset_hub,
    }
    if dataset_dir:
        evalscope_config['dataset_dir'] = dataset_dir

    # Step 6: 探测吞吐，选并发
    batch_size, probe_time = probe_throughput(
        model_name=model_name,
        api_url=api_base,
        api_key=api_key,
        generation_config=gen_config,
        dataset_args=dataset_args,
        evalscope_config=evalscope_config,
    )

    # Step 7: 正式评测
    print("-" * 60)
    total_questions = limit if limit else 198
    print(f"[EVAL] 正式评测: gpqa_diamond ({total_questions}题, 并发={batch_size})")
    print("-" * 60)

    model_id = _sanitize_model_id(model_name)
    work_dir = f'outputs/gpqa_diamond/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    task_kwargs = dict(
        model=model_name,
        model_id=model_id,
        api_url=api_base,
        api_key=api_key,
        eval_type=EvalType.OPENAI_API,
        datasets=['gpqa_diamond'],
        dataset_args=dataset_args,
        eval_batch_size=batch_size,
        generation_config=gen_config,
        dataset_hub=dataset_hub,
        work_dir=work_dir,
        no_timestamp=True,
    )
    if dataset_dir:
        task_kwargs['dataset_dir'] = dataset_dir
    if limit:
        task_kwargs['limit'] = limit

    task_cfg = TaskConfig(**task_kwargs)

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

    try:
        result = run_task(task_cfg=task_cfg)
    except Exception as e:
        if monitor and monitor.is_dead():
            reason = monitor.death_reason()
            print(f"\n[MONITOR] 服务崩溃: {reason.get('detail', '未知')}")
            if reason.get('log_line'):
                print(f"[MONITOR] 日志: {reason['log_line']}")
            monitor.stop()
            return {'error': f"服务崩溃 ({reason.get('type', 'unknown')}): {reason.get('detail', '')}", 'service_crashed': True, 'crash_reason': reason}
        print(f"[ERROR] 评测失败: {e}")
        traceback.print_exc()
        return {'error': str(e)}
    finally:
        if monitor:
            monitor.stop()

    # 评测完成后检查服务状态
    if monitor and monitor.is_dead():
        reason = monitor.death_reason()
        print(f"\n[MONITOR] ⚠ 评测期间服务崩溃: {reason.get('detail', '未知')}")
        if reason.get('log_line'):
            print(f"[MONITOR] 日志: {reason['log_line']}")
        print("[MONITOR] 评测结果可能不完整")

    # Step 8: 解析结果
    score, raw_details = parse_result(result)
    total_elapsed = round(time.time() - total_start, 2)
    minutes = int(total_elapsed // 60)
    seconds = round(total_elapsed % 60, 1)

    # Step 9: 输出报告
    report = {
        'model': model_name,
        'benchmark': 'gpqa_diamond',
        'mode': mode_str,
        'score': score,
        'total_questions': total_questions,
        'eval_batch_size': batch_size,
        'max_tokens': max_tokens,
        'max_model_len': max_model_len,
        'truncation_detected': truncation_detected,
        'temperature': gen_config['temperature'],
        'probe_time_seconds': probe_time,
        'eval_duration_seconds': round(total_elapsed - probe_time, 2),
        'total_duration_seconds': total_elapsed,
        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'work_dir': work_dir,
    }
    if monitor and monitor.is_dead():
        reason = monitor.death_reason()
        report['service_crashed'] = True
        report['crash_reason'] = reason
    report['_meta'] = {
            'model': '模型名称或路径',
            'benchmark': '评测基准名称（固定 gpqa_diamond）',
            'mode': '评测模式: standard（普通模型）/ thinking（思维链模型）',
            'score': 'GPQA Diamond 正确率百分比',
            'total_questions': '评测题目总数（默认 50 题，--limit 0 为全量 198 题）',
            'eval_batch_size': '评测并发数（自动探测选择）',
            'max_tokens': '单次生成最大 token 数',
            'max_model_len': '模型支持的最大上下文长度',
            'truncation_detected': '是否检测到输出被截断（true 时分数可能偏低）',
            'temperature': '采样温度（0.0=贪心解码）',
            'probe_time_seconds': '并发探测阶段耗时（秒）',
            'eval_duration_seconds': '实际评测阶段耗时（秒）',
            'total_duration_seconds': '总耗时（含探测，秒）',
            'work_dir': 'evalscope 原始输出目录（含预测、报告、日志）',
    }

    # 写 JSON 报告
    report_path = 'gpqa_result.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 如果指定了 output_path，额外写一份到目标路径
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report_path = str(out)

    # 终端打印
    print()
    print("=" * 60)
    print("  GPQA Diamond 快速评测结果")
    print("=" * 60)
    print(f"  模型:     {model_name}")
    print(f"  模式:     {mode_str} (temperature={gen_config['temperature']}, max_tokens={max_tokens})")
    print(f"  并发:     {batch_size}")
    print(f"  题数:     {total_questions}")
    if score is not None:
        print(f"  得分:     {score:.2f}%")
    else:
        print(f"  得分:     解析失败 (查看 {work_dir} 原始输出)")
    print(f"  耗时:     {minutes}m {seconds}s")
    print(f"  报告:     {report_path}")
    print("=" * 60)

    return report


# =============================================================================
# CLI 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPQA Diamond 快速精度评测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python fast_gpqa.py --config config.yaml
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1
        """,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--model-name', type=str, default=None,
                        help='模型名称 (覆盖 config)')
    parser.add_argument('--api-base', type=str, default=None,
                        help='API 地址 (覆盖 config)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API 密钥 (覆盖 config)')
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='数据集缓存目录 (覆盖 config)')
    parser.add_argument('--limit', type=int, default=50,
                        help='限制评测题数（默认 50 题，传 0 或 198 为全量）')
    parser.add_argument('--output', type=str, default=None,
                        help='结果 JSON 输出路径（如 /flagos-workspace/results/gpqa_native.json）')
    args = parser.parse_args()

    # 加载配置
    config = {}
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[ERROR] 加载配置失败: {e}")
            sys.exit(1)

    model_cfg = config.get('model', {})

    # CLI 参数优先级 > config
    model_name = args.model_name or model_cfg.get('name', '')
    api_base = args.api_base or model_cfg.get('api_base', '')
    api_key = args.api_key or model_cfg.get('api_key', 'EMPTY')
    dataset_dir = args.dataset_dir or config.get('dataset_dir', '') or None
    dataset_hub = config.get('dataset_hub', 'modelscope')

    if not model_name:
        print("[ERROR] 必须指定模型名称: --model-name 或 config.yaml 中 model.name")
        sys.exit(1)
    if not api_base:
        print("[ERROR] 必须指定 API 地址: --api-base 或 config.yaml 中 model.api_base")
        sys.exit(1)

    # 验证 API 可达
    try:
        base = api_base.rstrip('/')
        if base.endswith('/v1'):
            base = base[:-3]
        resp = requests.get(f"{base}/v1/models", timeout=10)
        resp.raise_for_status()
        print(f"[OK] API 连通性检查通过")
    except Exception as e:
        print(f"[ERROR] API 不可达 ({api_base}): {e}")
        sys.exit(1)

    # 检查 evalscope
    try:
        import evalscope
        print(f"[OK] evalscope {getattr(evalscope, '__version__', 'unknown')} 已安装")
    except ImportError:
        print("[ERROR] evalscope 未安装，请执行: pip install evalscope")
        sys.exit(1)

    # 运行
    try:
        step_id = os.environ.get("FLAGOS_STEP_ID", "04_accuracy_eval")
        step_title = os.environ.get("FLAGOS_STEP_TITLE", "精度评测")
        write_checkpoint(step_id, step_title, "running_fast_gpqa",
                         action_detail=f"fast_gpqa.py --model-name {model_name} --api-base {api_base}")
        report = run_fast_gpqa(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            dataset_dir=dataset_dir,
            dataset_hub=dataset_hub,
            limit=args.limit or None,
            output_path=args.output,
        )
        sys.exit(0 if report.get('score') is not None else 1)
    except Exception as e:
        write_last_error(
            tool="fast_gpqa.py",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=traceback.format_exc(),
            context={"model": model_name, "api_base": api_base},
        )
        print(f"[FATAL] fast_gpqa.py 异常退出: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
