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
import zlib
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

THINKING_PATTERNS = ['qwen3', 'qwq', 'deepseek-r1', 'deepseek-r2', 'mimo', 'hunyuan']

# thinking 模型 max_tokens 上限。正常思考链输出一般几千~一万多 token 封顶，
# 真正会吃满上限的几乎必然是 runaway 复读死循环（Qwen3-30B 事故: 24576 token
# 复读在慢速芯片上拖了 70 分钟卡死评测收尾）。cap 20000 不截断正常答案，
# 只收窄 runaway 的复读窗口。check_truncation 的翻倍重试同样受此约束
# （防止翻倍后窗口回到 max_model_len 级）。
THINKING_MAX_TOKENS_CAP = 20000


# =============================================================================
# 评测数据集配置（--dataset 参数）
# =============================================================================
# 每个数据集的 evalscope 注册名、全量题数、默认 few-shot 与预加载信息。
# 注意：mmlu 是 evalscope reformat 模式，limit 为 per-subset 语义——
# limit=100 表示 57 个子集各取 100 题（共 5700 题，全量 14042 的 40%）。
DATASET_CONFIG = {
    'gpqa_diamond': {
        'full_count': 198,          # limit=0 全量题数
        'few_shot_num': 0,
        'default_limit': 50,        # 默认题数（--limit 未指定时）
        'preload': {                # 预加载（探测阶段计时不含下载）：hub -> (repo, subset, split)
            'modelscope': ('AI-ModelScope/gpqa_diamond', None, 'train'),
            'huggingface': ('Idavidrein/gpqa', 'gpqa_diamond', 'train'),
        },
        'benchmark_name': 'GPQA Diamond',
    },
    'mmlu': {
        'full_count': 14042,
        'few_shot_num': 5,
        'default_limit': 20,        # 2026-08-14 定稿：20/子集 = 1140 题（降采样实测 gap -0.4pt、95% ±2.05pt，见记忆 eval-sampling-gap-measured）
        'per_subset_limit': True,   # limit 应用在每个子集（57 子集 × limit 题）
        'preload': {
            'modelscope': ('AI-ModelScope/mmlu', None, 'test'),
            'huggingface': ('cais/mmlu', None, 'test'),
        },
        'benchmark_name': 'MMLU',
    },
    'math_500': {
        'full_count': 500,
        'few_shot_num': 0,
        'default_limit': 40,        # 2026-08-14 定稿：40/等级 × 5 = 200 题（--limit 0 仍为全量 500）
        'per_subset_limit': True,   # limit 应用在每个子集（5 子集 Level 1-5 × limit 题，实测 1.5.1 行为）
        'preload': {
            'modelscope': ('AI-ModelScope/MATH-500', None, 'test'),
            'huggingface': ('HuggingFaceH4/MATH-500', None, 'test'),
        },
        'benchmark_name': 'MATH-500',
    },
    # MMStar：多模态（VLM）视觉必答 4 选一 MCQ。evalscope 注册名 mm_star，
    # VisionLanguageAdapter（图文输入，图片以 base64 内嵌 chat message），
    # eval_split=val，6 子集各 250 题（reformat_subset=True → per-subset 语义），
    # metric=acc，0-shot。⚠ 仅对多模态模型有意义：纯文本 LLM 收到图片会报错/胡答，
    # 且 vLLM 须以支持图像输入的方式起服务（见 multimodal 门控）。
    'mm_star': {
        'full_count': 1500,
        'few_shot_num': 0,
        'default_limit': 0,         # 默认全量 1500 题（6 子集 × 250）。多模态采样偏最难子集会显著偏低
                                    # （300 题实测 44.67% vs 全量 62.6%），故默认全量；如需降采样显式传 --limit N/子集
        'per_subset_limit': True,   # reformat_subset=True，limit 应用到每个子集
        'multimodal': True,         # 多模态标志：驱动 prompt 预留放大 / 并发保守 / 门控
        'preload': {
            'modelscope': ('evalscope/MMStar', None, 'val'),
            'huggingface': ('Lin-Chen/MMStar', None, 'val'),
        },
        'benchmark_name': 'MMStar',
    },
}


def detect_thinking(model_name: str) -> bool:
    """根据模型名或 context.yaml 检测是否为 thinking model。"""
    name_lower = model_name.lower()
    if any(p in name_lower for p in THINKING_PATTERNS):
        return True
    # 从 context.yaml 读取 thinking_model 字段（优先级最高）
    try:
        import yaml as _yaml
        ctx_path = "/flagos-workspace/shared/context.yaml"
        with open(ctx_path, "r") as f:
            ctx = _yaml.safe_load(f) or {}
        if ctx.get("runtime", {}).get("thinking_model") or ctx.get("model", {}).get("thinking_model"):
            return True
    except Exception:
        pass
    return False


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


# 多模态 prompt 预留：单张高分辨率图经 vision encoder 可占数千 image token，
# 文本流程固定预留 8K 对 VLM 不够，prompt(文本+image tokens) 可能超 max_model_len
# 被服务拒绝（错误文本当输出 → 全题 0 分）。MMStar 图分辨率跨度大（160×160~1350×1015），
# 保守预留 16K 给 prompt+image，给正常图文请求足够余量。
MM_PROMPT_RESERVE = 16384
# 多模态 max_tokens 硬上限。MMStar 等 MCQ 任务正常答案很短（几十~几百 token），
# 超大窗口只会让复读死循环吃满 token（实测 temp=0 也复读到 8~9 万字符），
# 收窄到 4096 强制模型早收尾给答案，避免 runaway 白跑且必 0 分。
MM_MAX_TOKENS_CAP = 4096


def auto_max_tokens(api_base: str, api_key: str, model_name: str, is_thinking: bool = False,
                    is_multimodal: bool = False) -> Tuple[int, Optional[int]]:
    """
    自动计算 max_tokens，基于服务端实际 max_model_len。

    thinking 模型：max(max_model_len - 8192, 8192)，cap THINKING_MAX_TOKENS_CAP
       —— runaway 复读死循环会吃满 max_tokens，无上限会让一轮复读拖数十分钟
       （慢速芯片 24576 token ≈ 70min）；正常思考链回答一般几千~一万多 token
       封顶，cap 20000 不会截断正常答案，只收窄 runaway 复读窗口。
    标准模型：clamp(max_model_len - 8192, 4096, 32768)
    多模态模型：prompt 预留放大到 MM_PROMPT_RESERVE（图片 token 占用大）。

    Returns:
        (max_tokens, max_model_len or None)
    """
    max_model_len = query_model_max_len(api_base, api_key, model_name)
    reserve = MM_PROMPT_RESERVE if is_multimodal else 8192
    if max_model_len:
        tokens = max_model_len - reserve  # 预留给 prompt（多模态含 image token）
        if is_thinking:
            tokens = max(tokens, 8192)
            tokens = min(tokens, THINKING_MAX_TOKENS_CAP)
        else:
            tokens = max(tokens, 4096)
            tokens = min(tokens, 32768)
        # 防御: 小上下文服务(max_model_len<=16384)下 max_tokens 收敛到上下文一半,
        # 给 prompt 留余量——vllm 以 max_model_len 同时约束 prompt+output 总长,
        # thinking 下限 8192 + 任意非空 prompt 必超限被拒(错误文本当输出, 全题 0 分)。
        # 真实流水线服务 max_model_len>=32768 不触发此分支, 行为不变。
        # 多模态：预留 16K 后再取半，避免 image token 撑破上下文。
        small_ctx = 32768 if is_multimodal else 16384
        if max_model_len <= small_ctx:
            tokens = min(tokens, max_model_len // 2)
        # 多模态硬上限：MCQ 答案短，收窄窗口逼模型早收尾，避免 runaway 复读吃满窗口
        if is_multimodal:
            tokens = min(tokens, MM_MAX_TOKENS_CAP)
        return tokens, max_model_len
    # fallback
    if is_multimodal:
        return MM_MAX_TOKENS_CAP, None
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
    max_tokens_cap: Optional[int] = None,
) -> Tuple[bool, int]:
    """
    发一条样题检查 finish_reason 是否为 length（截断）。

    如果截断，自动将 max_tokens 翻倍（在 max_model_len 允许范围内），
    并遵守 max_tokens_cap（如 thinking 的 runaway 窗口 cap，防止翻倍
    把防线1的窗口上限破坏掉）。

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
                if max_tokens_cap:
                    # 翻倍也遵守 runaway 窗口 cap（防线1）
                    new_tokens = min(new_tokens, max_tokens_cap)
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
# runaway 复读检测（防线2：内容判别，判别性不干扰生成）
# =============================================================================

# 复读判定阈值（保守，宁漏勿误杀正常长推理）
_RUNAWAY_MIN_TEXT_LEN = 500      # 低于此长度的文本不做判别（短答无统计意义）
_RUNAWAY_NGRAM = 3               # n-gram 粒度
_RUNAWAY_MAX_DIVERSITY = 0.10    # 3-gram 去重后独占比低于此值 → 高度重复
_RUNAWAY_MAX_COMPRESS_RATIO = 0.15  # zlib 压缩比低于此值 → 高度可压缩（复读特征）
# 注：diversity 对超长文本（>50K 字符）自然衰减（3-gram 空间饱和，真实长文本
# 实测可低至 0.07），只适合中等长度区分；compress_ratio 对长度稳定（真实文本
# 0.25~0.41，复读 <0.05），是唯一可靠的截断判据。故条件2（length 截断）只用
# compress_ratio，不用 diversity。


def _ngram_diversity(text: str, n: int = _RUNAWAY_NGRAM) -> float:
    """n-gram 去重后独占比：复读文本的 grams 几乎全是同一个，独占比趋近 0。"""
    if len(text) < n:
        return 1.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 1.0
    return len(set(grams)) / len(grams)


def _zlib_compress_ratio(text: str) -> float:
    """zlib 压缩比：压缩后体积 / 原文体积。复读文本高度可压缩，比值趋近 0。"""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, level=6)) / len(raw)


def _content_to_text(content) -> str:
    """归一化 message.content 为纯文本。

    普通模型 content 为 str；reasoning 模型（Qwen3/MiMo/QwQ/DeepSeek-R1 等）
    的 content 是结构化 block 列表，长推理链在 'reasoning' 键、最终答案在 'text' 键
    （亦见 'content' 键）。runaway 复读主要发生在长推理链，故三键全部拼接，
    避免只取答案漏检 CoT 复读。非 str/list 一律 str() 兜底，杜绝 .strip() 崩溃。
    """
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                parts.append(str(blk.get("reasoning") or blk.get("text") or blk.get("content") or ""))
            elif blk is not None:
                parts.append(str(blk))
        return " ".join(parts)
    return str(content) if content is not None else ""


def detect_runaway(text: str, finish_reason: str = "") -> Tuple[bool, Dict]:
    """判别回答是否为 runaway（垃圾复读死循环）。

    判别特征（组合，非单一指标）：
    1. 3-gram 去重后独占比 < 0.10 → 高度重复
    2. zlib 压缩比 < 0.15 → 高度可压缩（复读文本高度冗余）
    3. finish_reason=length + 独占比 < 0.15 → 截断且高重复（撞 max_tokens 的复读）

    保守设计：任一单指标都可能误伤正常长推理（思考链也带一定重复度），
    必须"双指标同时成立"或"length+高重复"才判定 runaway。

    Returns:
        (is_runaway, evidence_dict)
        evidence = {"diversity": float, "compress_ratio": float, "text_len": int,
                    "finish_reason": str, "reason": str}
    """
    text = _content_to_text(text).strip()
    evidence = {
        "diversity": 1.0,
        "compress_ratio": 1.0,
        "text_len": len(text),
        "finish_reason": finish_reason or "",
        "reason": "",
    }
    if len(text) < _RUNAWAY_MIN_TEXT_LEN:
        evidence["reason"] = "text_too_short"
        return False, evidence

    diversity = _ngram_diversity(text)
    compress_ratio = _zlib_compress_ratio(text)
    evidence["diversity"] = round(diversity, 4)
    evidence["compress_ratio"] = round(compress_ratio, 4)

    # 条件1: 双指标同时成立 → 复读
    if diversity < _RUNAWAY_MAX_DIVERSITY and compress_ratio < _RUNAWAY_MAX_COMPRESS_RATIO:
        evidence["reason"] = "high_repeat_and_compressible"
        return True, evidence
    # 条件2: 撞 max_tokens 截断 + 高度可压缩 → runaway 截断（正常答案截断压缩比
    # 仍 >0.25，复读截断 <0.05；不用 diversity——长文本 diversity 自然衰减会误判）
    if finish_reason == "length" and compress_ratio < _RUNAWAY_MAX_COMPRESS_RATIO:
        evidence["reason"] = "length_truncated_high_repeat"
        return True, evidence

    evidence["reason"] = "normal"
    return False, evidence


def analyze_predictions_runaway(work_dir: str, model_id: str, dataset: str = 'gpqa_diamond') -> Dict:
    """评测完成后扫描 predictions jsonl，标记 runaway 题。

    evalscope 是黑盒（生成+判分在其内部完成），fast_gpqa 拿不到逐题文本；
    本题逐题内容判别只能在评测完成后从 work_dir/predictions/ 后处理实现。
    用途：标记 runaway 题 → 报告提示该轮分数可能被垃圾复读污染（不改分，
    仅数据真实性提示；真正止血靠防线3 进程外看门狗）。

    多数据集：主预测文件（{dataset}_default.jsonl）不存在时（mmlu 等
    reformat 模式按子集输出多个文件），扫描 predictions 目录下全部 jsonl。

    Returns:
        {"checked": int, "runaway_count": int, "runaway_indices": [...]}
        runaway_indices = [{"index": int, "reason": str, "diversity": float,
                            "compress_ratio": float, "finish_reason": str}]
    """
    result = {"checked": 0, "runaway_count": 0, "runaway_indices": []}
    pred_dir = os.path.join(work_dir, "predictions", model_id)
    pred_paths = [os.path.join(pred_dir, f"{dataset}_default.jsonl")]
    if not os.path.isfile(pred_paths[0]):
        # mmlu 等 reformat 模式：按子集输出多文件，全部扫描
        pred_paths = sorted(
            os.path.join(pred_dir, f) for f in os.listdir(pred_dir)
            if f.endswith('.jsonl')
        ) if os.path.isdir(pred_dir) else []
    if not pred_paths:
        return result
    for pred_path in pred_paths:
        if not os.path.isfile(pred_path):
            continue
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    result["checked"] += 1
                    choices = (rec.get("model_output") or {}).get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("message") or {}).get("content") or ""
                    stop_reason = choices[0].get("stop_reason") or ""
                    is_ra, ev = detect_runaway(content, stop_reason)
                    if is_ra:
                        result["runaway_count"] += 1
                        result["runaway_indices"].append({
                            "index": rec.get("index"),
                            "reason": ev["reason"],
                            "diversity": ev["diversity"],
                            "compress_ratio": ev["compress_ratio"],
                            "finish_reason": ev["finish_reason"],
                        })
        except OSError:
            pass
    return result


# =============================================================================
# 探测吞吐 & 选并发
# =============================================================================

def _sanitize_model_id(model_name: str) -> str:
    """将模型名清理为安全的 model_id（不含 / 等特殊字符）。"""
    return model_name.strip('/').split('/')[-1] or model_name


def _preload_dataset(dataset_hub: str, dataset_dir: Optional[str] = None,
                     dataset: str = 'gpqa_diamond'):
    """预加载数据集到缓存，确保探测阶段计时不含下载时间。"""
    cfg = DATASET_CONFIG.get(dataset, DATASET_CONFIG['gpqa_diamond'])
    preload = cfg.get('preload', {})
    repo, subset, split = preload.get(dataset_hub, (None, None, None))
    if not repo:
        return
    try:
        if dataset_dir:
            # 检查本地缓存是否存在（按数据集名前缀匹配）
            import glob as glob_mod
            prefix = dataset.split('_')[0]  # gpqa/mmlu/math
            if glob_mod.glob(os.path.join(dataset_dir, '**', f'{prefix}*'), recursive=True):
                return
        if dataset_hub == 'modelscope':
            from modelscope import MsDataset
            MsDataset.load(repo, split=split, trust_remote_code=True)
        else:
            import datasets as hf_datasets
            hf_datasets.load_dataset(repo, name=subset or None, split=split,
                                     trust_remote_code=True)
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
    is_multimodal: bool = False,
) -> Tuple[int, float]:
    """
    三阶段并发探测：
    1. 直接 API 调用测单条推理延迟（剥离 evalscope 框架开销）
    2. 基于延迟 + thinking 模型特性估算候选并发
    3. 快速验证（3 题并发测试，选最优）

    多模态：探测样题是纯文本，无法反映图文请求的真实（更高）延迟与显存占用，
    会高估并发导致正式评测 OOM。故 is_multimodal=True 时将候选并发档位对半下调
    （下限 1），偏保守。真正的并发上限仍由阶段3 的 3 题并发验证（遇错即止）兜底。

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
    if is_multimodal:
        # 图文请求实际更慢 + 显存占用更高，纯文本探测偏乐观 → 候选对半下调
        candidates = sorted({max(1, c // 2) for c in candidates})
        print(f"[PROBE] 多模态：候选并发下调为保守档位")
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
# 采样参数决策（模型配置驱动 + 失败回退默认）
# =============================================================================

# 从模型 generation_config.json 采纳的采样字段白名单。
# 已在真实容器验证 evalscope GenerateConfig 原生支持这些键（不支持也会被
# evalscope 自动挪入 extra_body 透传 vllm，不会报错）。刻意不含 max_tokens：
# 模型配置里的 max_tokens 常偏小（如 1024）会导致截断、精度暴跌，max_tokens
# 仍由 auto_max_tokens 基于 max_model_len 自适应计算 + 截断检测决定。
_GEN_PARAM_WHITELIST = ("temperature", "top_p", "top_k", "repetition_penalty")


def _resolve_model_dir(model_path: Optional[str] = None) -> Optional[str]:
    """定位模型目录（用于读取 generation_config.json）。

    候选来源按优先级：
      1. 传入的 model_path 若本身是存在的目录（独立裸跑场景：--model-name 直接是本地路径）
      2. context.yaml 的 model.local_path / model.container_path（流水线场景兜底）
    全程 best-effort，任何异常都返回 None。
    """
    import os as _os
    # 1) --model-name 本身就是本地目录
    if model_path and _os.path.isdir(model_path):
        return model_path
    # 2) 流水线 context.yaml 兜底
    try:
        import yaml as _yaml
        ctx_path = "/flagos-workspace/shared/context.yaml"
        with open(ctx_path) as f:
            ctx = _yaml.safe_load(f) or {}
        model_sec = ctx.get("model", {}) or {}
        mp = model_sec.get("local_path") or model_sec.get("container_path")
        if mp and _os.path.isdir(mp):
            return mp
    except Exception:
        pass
    return None


def resolve_gen_params(is_thinking: bool, max_tokens: int,
                       model_path: Optional[str] = None) -> Dict:
    """构建 generation_config：优先采用模型自带 generation_config.json 的采样参数，
    读取失败/文件缺失/字段非法时无声回退现有默认。

    设计为纯增强层：任何一步异常都不 raise，最坏情况等价于改动前的固定默认，
    确保绝不因本逻辑新增精度评测报错点（关系整体流程成功率）。

    优先级：模型 generation_config.json 白名单字段 > 现有默认（标准 0.0 / thinking 0.6）。
    模型目录定位：--model-name 本地路径优先，context.yaml 兜底（见 _resolve_model_dir）。
    stream/timeout/n 属评测框架控制项，不受模型配置影响。
    """
    # 1) 现有默认（回退基线，与改动前完全一致）
    if is_thinking:
        cfg = {'max_tokens': max_tokens, 'temperature': 0.6, 'top_p': 0.95,
               'stream': True, 'timeout': 120000, 'n': 1}
    else:
        cfg = {'max_tokens': max_tokens, 'temperature': 0.0, 'top_p': 1.0,
               'stream': True, 'timeout': 120000, 'n': 1}

    # 2) best-effort 读模型配置覆盖采样字段（全程不 raise）
    try:
        import json as _json
        import os as _os
        mp = _resolve_model_dir(model_path)
        if mp:
            gc_path = _os.path.join(mp, "generation_config.json")
            if _os.path.isfile(gc_path):
                with open(gc_path) as f:
                    gc = _json.load(f) or {}
                applied = {}
                for k in _GEN_PARAM_WHITELIST:
                    v = gc.get(k)
                    # 仅采纳合法非负数值，避免字符串/None/负值污染
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                        cfg[k] = v
                        applied[k] = v

                # 修复 do_sample=true 但无显式温度时掉入贪心的问题：
                # 模型声明 do_sample=true 意图是"用采样解码"，此时若白名单未捕获温度
                # （模型配置里没写 temperature 字段），补一个合理采样默认，避免
                # temperature=0.0 贪心导致小模型循环到 max_tokens、精度崩溃。
                # 判断依据：temperature 不在 applied（模型未显式配置）且当前为默认 0.0。
                if gc.get("do_sample") is True and "temperature" not in applied and cfg.get("temperature", 0.0) == 0.0:
                    cfg["temperature"] = 0.7
                    applied["temperature"] = 0.7  # 标记为已应用，避免重复打印
                    if "top_p" not in applied:
                        cfg["top_p"] = 0.95
                        applied["top_p"] = 0.95
                    if "top_k" not in applied:
                        cfg["top_k"] = 64
                        applied["top_k"] = 64

                if applied:
                    print(f"  [gen] 采用模型 generation_config.json 采样参数: {applied}")
                else:
                    print("  [gen] generation_config.json 无可用采样字段，沿用默认")
            else:
                print("  [gen] 未找到模型 generation_config.json，沿用默认采样参数")
        else:
            print("  [gen] 未定位到模型目录（--model-name 非本地路径且 context.yaml 无路径），沿用默认采样参数")
    except Exception as e:  # 任何异常都静默回退，绝不影响评测
        print(f"  [gen] 读取模型配置失败（{e}），回退默认采样参数")

    return cfg


# =============================================================================
# 主流程
# =============================================================================

def run_fast_gpqa(
    model_name: str,
    api_base: str,
    api_key: str = 'EMPTY',
    dataset_dir: Optional[str] = None,
    dataset_hub: str = 'modelscope',
    dataset: str = 'gpqa_diamond',
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    快速精度评测主流程（GPQA Diamond / MMLU / MATH-500）。

    Args:
        dataset: 数据集名（DATASET_CONFIG 的 key）
        limit: 题数上限。None → 数据集默认题数；0 → 全量。
               mmlu 为 per-subset 语义（每个子集各取 limit 题）。
    Returns:
        结果 dict
    """
    from evalscope import TaskConfig, run_task
    from evalscope.constants import EvalType

    cfg = DATASET_CONFIG.get(dataset, DATASET_CONFIG['gpqa_diamond'])
    if limit is None:
        limit = cfg['default_limit']

    total_start = time.time()

    print("=" * 60)
    print(f"  {cfg['benchmark_name']} 快速精度评测")
    print("=" * 60)
    print(f"  模型: {model_name}")
    print(f"  API:  {api_base}")

    # Step 1: 检测 thinking 模型 + 多模态数据集
    is_thinking = detect_thinking(model_name)
    is_multimodal = bool(cfg.get('multimodal'))
    mode_str = "thinking" if is_thinking else "standard"
    print(f"  模式: {mode_str}{' + 多模态(VLM)' if is_multimodal else ''}")

    # Step 2: 自动设 max_tokens（基于 max_model_len 动态计算；多模态放大 prompt 预留）
    max_tokens, max_model_len = auto_max_tokens(api_base, api_key, model_name, is_thinking,
                                                is_multimodal=is_multimodal)
    if max_model_len:
        print(f"  max_model_len: {max_model_len} (从服务端获取)")
    else:
        print(f"  max_model_len: 未知 (使用 fallback)")
    print(f"  max_tokens: {max_tokens}")

    # Step 3: 截断检测 — 发样题检查 finish_reason
    # thinking 模型翻倍重试受 THINKING_MAX_TOKENS_CAP 约束（防线1）
    # 多模态：截断翻倍上限锁死 MM_MAX_TOKENS_CAP，避免样题撞 length 后翻倍破窗
    if is_multimodal:
        trunc_cap = MM_MAX_TOKENS_CAP
    elif is_thinking:
        trunc_cap = THINKING_MAX_TOKENS_CAP
    else:
        trunc_cap = None
    truncation_detected, max_tokens = check_truncation(
        api_base, api_key, model_name, max_tokens, max_model_len,
        max_tokens_cap=trunc_cap,
    )

    # Step 4: 构建 generation_config（优先采用模型自带 generation_config.json 的采样参数，
    # 读取失败/缺失时无声回退现有默认；纯增强层，绝不新增评测报错点）
    gen_config = resolve_gen_params(is_thinking, max_tokens, model_path=model_name)

    # Step 5: 构建 dataset_args（few-shot 按数据集配置；thinking 模型加 remove_until 过滤）
    dataset_args = {dataset: {'few_shot_num': cfg['few_shot_num']}}
    if is_thinking:
        dataset_args[dataset]['filters'] = {'remove_until': '</think>'}

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
        is_multimodal=is_multimodal,
    )

    # Step 7: 正式评测
    print("-" * 60)
    total_questions = limit if limit else cfg['full_count']
    limit_note = f"（每子集 {limit} 题）" if limit and cfg.get('per_subset_limit') else ""
    print(f"[EVAL] 正式评测: {dataset}{limit_note} ({total_questions}题, 并发={batch_size})")
    print("-" * 60)

    model_id = _sanitize_model_id(model_name)
    work_dir = f'outputs/{dataset}/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    task_kwargs = dict(
        model=model_name,
        model_id=model_id,
        api_url=api_base,
        api_key=api_key,
        eval_type=EvalType.OPENAI_API,
        datasets=[dataset],
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
    # 以 evalscope 报告的实际评测数为准（mmlu per-subset 时为 57×limit 而非 limit）
    for _k in ('metrics', 'metric'):
        for _m in (raw_details.get(_k) or []) if isinstance(raw_details, dict) else []:
            if isinstance(_m, dict) and _m.get('num'):
                total_questions = int(_m['num'])
                break
        else:
            continue
        break
    total_elapsed = round(time.time() - total_start, 2)
    minutes = int(total_elapsed // 60)
    seconds = round(total_elapsed % 60, 1)

    # Step 8.5: runaway 复读检测（防线2，评测后逐题内容判别）
    runaway_analysis = analyze_predictions_runaway(work_dir, model_id, dataset=dataset)
    if runaway_analysis.get("runaway_count", 0) > 0:
        print(f"[WARN] 检测到 {runaway_analysis['runaway_count']}/{runaway_analysis.get('checked', 0)} 题 runaway 复读: "
              f"{[r['index'] for r in runaway_analysis['runaway_indices']]}")
        print(f"[WARN] 这些题的截断垃圾可能污染分数，建议关注精度结果可信度")

    # Step 9: 输出报告
    report = {
        '_producer': 'fast_gpqa.py',
        'model': model_name,
        'benchmark': dataset,
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
        'runaway_detection': runaway_analysis,
    }
    if monitor and monitor.is_dead():
        reason = monitor.death_reason()
        report['service_crashed'] = True
        report['crash_reason'] = reason
    report['_meta'] = {
            'model': '模型名称或路径',
            'benchmark': '评测基准名称（gpqa_diamond / mmlu / math_500 / mm_star[多模态]）',
            'mode': '评测模式: standard（普通模型）/ thinking（思维链模型）',
            'score': f'{cfg["benchmark_name"]} 正确率百分比',
            'total_questions': f'实际评测题数（evalscope 报告为准；mmlu/math_500 为 per-subset：mmlu --limit 100 = 5700 题（57 子集×100）、math_500 --limit 10 = 50 题（5 子集×10），--limit 0 为全量 {cfg["full_count"]} 题）',
            'eval_batch_size': '评测并发数（自动探测选择）',
            'max_tokens': '单次生成最大 token 数',
            'max_model_len': '模型支持的最大上下文长度',
            'truncation_detected': '是否检测到输出被截断（true 时分数可能偏低）',
            'temperature': '采样温度（0.0=贪心解码）',
            'probe_time_seconds': '并发探测阶段耗时（秒）',
            'eval_duration_seconds': '实际评测阶段耗时（秒）',
            'total_duration_seconds': '总耗时（含探测，秒）',
            'work_dir': 'evalscope 原始输出目录（含预测、报告、日志）',
            'runaway_detection': '防线2 复读检测：评测完成后扫描逐题回答，标记 runaway 复读题（checked/runaway_count/runaway_indices）；runaway_count>0 说明该轮分数可能被复读污染',
    }

    # 写 JSON 报告
    report_path = f'{dataset}_result.json'
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
    print(f"  {cfg['benchmark_name']} 快速评测结果")
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
        description='快速精度评测（GPQA Diamond / MMLU / MATH-500 / MMStar[多模态]）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python fast_gpqa.py --config config.yaml
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 --dataset mmlu --limit 100
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 --dataset math_500 --limit 0
  # MMStar（多模态/VLM，仅对视觉语言模型有意义；6 子集 × limit，默认 50/子集=300 题，--limit 0 全量 1500）
  python fast_gpqa.py --model-name Qwen2.5-VL-7B --api-base http://localhost:8000/v1 --dataset mm_star
  python fast_gpqa.py --model-name Qwen2.5-VL-7B --api-base http://localhost:8000/v1 --dataset mm_star --limit 0
  # 多数据集（空格或逗号分隔；多数据集时 --output 为目录，每数据集写 {dataset}_result.json）
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 --dataset mmlu math_500 --output /flagos-workspace/results/multi
  python fast_gpqa.py --model-name Qwen3-8B --api-base http://localhost:8000/v1 --dataset mmlu,math_500
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
    parser.add_argument('--dataset', type=str, default=None, nargs='+',
                        help=f'评测数据集，可多个（空格或逗号分隔）: {" / ".join(DATASET_CONFIG)}'
                             f'（默认 gpqa_diamond，可被 config 覆盖；多数据集时 --output 视为目录，每数据集写 {{dataset}}_result.json）')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制评测题数（None=数据集默认，0=全量；mmlu/math_500 为每子集题数，如 mmlu 100=57子集各100题、math_500 10=5子集各10题）')
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

    # 数据集解析：--dataset 支持空格/逗号多值，config dataset 支持逗号分隔，均可被 --dataset 覆盖
    def _split_datasets(raw):
        parts = raw if isinstance(raw, list) else [raw]
        return [p.strip() for part in parts for p in str(part).split(',') if p.strip()]

    datasets = _split_datasets(args.dataset) or _split_datasets(config.get('dataset', 'gpqa_diamond'))
    invalid = [d for d in datasets if d not in DATASET_CONFIG]
    if invalid:
        print(f"[ERROR] 未知数据集: {invalid}（可选: {list(DATASET_CONFIG)}）")
        sys.exit(1)
    multi_dataset = len(datasets) > 1

    if not model_name:
        # 自动从 /v1/models 探测
        try:
            base = api_base.rstrip('/')
            if base.endswith('/v1'):
                base_url = base
            else:
                base_url = base + '/v1'
            resp = requests.get(f"{base_url}/models", timeout=10)
            resp.raise_for_status()
            models = resp.json().get('data', [])
            if models:
                model_name = models[0].get('id', '')
                print(f"[INFO] 自动探测模型名: {model_name}")
        except Exception:
            pass
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

    # 检查 evalscope（统一版本 1.5.1，见 setup_workspace.sh 预装）
    try:
        import evalscope
        print(f"[OK] evalscope {getattr(evalscope, '__version__', 'unknown')} 已安装")
        if getattr(evalscope, '__version__', None) != "1.5.1":
            print(f"[WARN] evalscope 版本 {getattr(evalscope, '__version__', 'unknown')} != 1.5.1 "
                  f"(统一评测版本)，评测行为可能与验证环境不一致")
    except ImportError:
        print("[ERROR] evalscope 未安装，请执行: pip install 'evalscope==1.5.1'")
        sys.exit(1)

    # 运行（多数据集时循环评测，每数据集独立结果）
    try:
        step_id = os.environ.get("FLAGOS_STEP_ID", "04_accuracy_eval")
        step_title = os.environ.get("FLAGOS_STEP_TITLE", "精度评测")
        write_checkpoint(step_id, step_title, "running_fast_gpqa",
                         action_detail=f"fast_gpqa.py --model-name {model_name} --api-base {api_base} "
                                       f"--dataset {' '.join(datasets)}")

        if multi_dataset and args.output:
            os.makedirs(args.output, exist_ok=True)

        reports = []
        for dataset in datasets:
            print(f"\n{'=' * 60}\n[多数据集 {len(datasets)} 个] 评测 #{datasets.index(dataset) + 1}: {dataset}\n{'=' * 60}")
            output_path = args.output
            if multi_dataset and output_path:
                output_path = os.path.join(output_path, f'{dataset}_result.json')
            report = run_fast_gpqa(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                dataset_dir=dataset_dir,
                dataset_hub=dataset_hub,
                dataset=dataset,
                limit=args.limit,
                output_path=output_path,
            )
            reports.append((dataset, report))

        if multi_dataset:
            print("\n" + "=" * 60)
            print("  多数据集评测汇总")
            print("=" * 60)
            for dataset, report in reports:
                score = report.get('score')
                if score is not None:
                    ok = "✓" if report.get('truncation_detected') is False else "⚠"
                    print(f"  {ok} {dataset:12s} {score:6.2f}%  ({report.get('total_questions')} 题, "
                          f"{report.get('eval_duration_seconds', 0):.0f}s)")
                else:
                    print(f"  ✗ {dataset:12s} 失败: {report.get('error', '未知错误')}")
        all_ok = all(r.get('score') is not None for _, r in reports)
        sys.exit(0 if all_ok else 1)
    except Exception as e:
        write_last_error(
            tool="fast_gpqa.py",
            error_type=type(e).__name__,
            error_message=str(e),
            traceback_str=traceback.format_exc(),
            context={"model": model_name, "api_base": api_base, "datasets": datasets},
        )
        print(f"[FATAL] fast_gpqa.py 异常退出: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
