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

"""芯片厂商×型号统一规范模块。

唯一权威数据源：同目录下 chip_spec.yaml。
检测(detect_gpu.py)、命名(config.py/get_image_name.sh)、报告(generate_report.py)
均通过本模块取值，消灭历史上散落且互相矛盾的厂商写法。

对外函数：
  normalize_vendor(raw)          别名/大小写 → 规范 vendor key（未知返回原值小写；
                                 含历史脏值兜底：中文名/括号 display 变体 → 规范 key）
  canonical_chip(vendor, model)  原始 smi 型号 → 规范显示名（未命中返回原值）
  canonical_chip_with_flag(vendor, model)
                                 (规范显示名, 是否命中)——含跨厂商唯一命中兜底
  naming_suffix(vendor)          命名后缀 xxx-{suffix}-FlagOS
  vendor_display(vendor)         "中文名(英文名)" 展示（issue/平台上传等场景；
                                 报告展示请用 vendor_en 纯英文，见 generate_report.py）
  vendor_en(vendor) / vendor_cn(vendor)
  valid_vendor_keys()            全部规范 vendor key 列表
"""
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chip_spec.yaml")


@lru_cache(maxsize=1)
def _load_spec() -> Dict[str, dict]:
    """加载规范表。yaml 缺失或解析失败时返回空 dict（调用方自带兜底，不抛错中断流程）。"""
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(_SPEC_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _alias_index() -> Dict[str, str]:
    """构建 alias（含 key 自身）→ 规范 vendor key 的反查表，全部小写。"""
    idx: Dict[str, str] = {}
    for key, spec in _load_spec().items():
        k = key.strip().lower()
        idx[k] = k
        for alias in (spec.get("aliases") or []):
            a = str(alias).strip().lower()
            if a:
                idx[a] = k
    return idx


def normalize_vendor(raw: Optional[str]) -> str:
    """别名/大小写归一到规范 vendor key。未知则返回原值小写去空格。

    兜底识别历史脏值变体（2026-08 报告兜底规范化需求，合法输入行为不变）：
      1. 括号内英文提取——"英伟达(Nvidia)"、"沐曦(Metax"（括号未闭合）→ 括号内查别名
      2. 中文名子串——"英伟达"、"海光" 等 vendor_cn 直接命中
      3. 英文名子串——"Nvidia 官方" 等包含 vendor_en 的写法
    """
    if not raw:
        return ""
    raw_s = str(raw).strip()
    v = raw_s.lower()
    hit = _alias_index().get(v)
    if hit:
        return hit
    # 1) 括号内英文提取（含未闭合括号兜底）
    inner = _bracket_inner(raw_s)
    if inner:
        hit = _alias_index().get(inner.lower())
        if hit:
            return hit
    # 2) 中文名子串
    v_low = raw_s.lower()
    for key, spec in _load_spec().items():
        cn = (spec.get("vendor_cn") or "").strip().lower()
        if cn and cn in v_low:
            return key
    # 3) 英文名子串
    for key, spec in _load_spec().items():
        en = (spec.get("vendor_en") or "").strip().lower()
        if en and en in v_low:
            return key
    return v


def _bracket_inner(raw: str) -> str:
    """提取括号内内容（display 变体 "中文(英文)" 的英文部分）。"""
    m = re.search(r"\(([^()]+)\)", raw)
    if m:
        return m.group(1)
    # 未闭合括号兜底：如 "沐曦(Metax"（历史脏值）
    m = re.search(r"\(([^()]+)\)?$", raw)
    if m:
        return m.group(1)
    return ""


def valid_vendor_keys() -> List[str]:
    """全部规范 vendor key。"""
    return list(_load_spec().keys())


def naming_suffix(vendor: Optional[str]) -> str:
    """命名后缀：xxx-{suffix}-FlagOS。未知厂商回退归一化后的 key。"""
    key = normalize_vendor(vendor)
    spec = _load_spec().get(key, {})
    return spec.get("naming_suffix", key) or key


def vendor_en(vendor: Optional[str]) -> str:
    """报告用英文名。未知回退归一化 key。"""
    key = normalize_vendor(vendor)
    spec = _load_spec().get(key, {})
    return spec.get("vendor_en", key) or key


def vendor_cn(vendor: Optional[str]) -> str:
    """厂商中文名。未知回退空。"""
    key = normalize_vendor(vendor)
    return _load_spec().get(key, {}).get("vendor_cn", "") or ""


def vendor_display(vendor: Optional[str]) -> str:
    """报告展示：有中文名则 "中文名(英文名)"，否则仅英文名。"""
    key = normalize_vendor(vendor)
    if not key:
        return "-"
    en = vendor_en(key)
    cn = vendor_cn(key)
    return f"{cn}({en})" if cn else en


def canonical_chip(vendor: Optional[str], gpu_model: Optional[str]) -> str:
    """原始采集型号 → 规范显示名。

    按 match 关键字长度降序匹配（长的优先，H20-3e 先于 H20）。
    未命中返回原始 gpu_model（保持信息不丢）。命中判定含跨厂商兜底，
    见 canonical_chip_with_flag。
    """
    display, _ = canonical_chip_with_flag(vendor, gpu_model)
    return display


def canonical_chip_with_flag(vendor: Optional[str], gpu_model: Optional[str]) -> Tuple[str, bool]:
    """原始采集型号 → (规范显示名, 是否命中规范表)。

    vendor-scoped 匹配失败时跨厂商全局搜索兜底：仅当全局唯一命中才归一
    （如 vendor 缺失/错误时的裸型号 "C550" → "Metax C550"、"S5000" → "MTT S5000"，
    对应 2026-08 历史报告 card_model 修复口径）。多义/未命中返回 (原始型号, False)。
    """
    if not gpu_model:
        return (gpu_model or "", False)
    key = normalize_vendor(vendor)
    spec = _load_spec().get(key, {})
    norm = str(gpu_model).lower().replace(" ", "").replace("_", "")

    # vendor-scoped 匹配（match 关键字长度降序，H20-3e 先于 H20）
    for m, display in _chips_candidates(spec):
        if m and m in norm:
            return (display, True)

    # 跨厂商兜底：全局搜索，按关键字长度降序取最长命中（H20-3e 先于 H20，
    # 与 vendor-scoped 同规则）；仅当最长命中的 display 唯一才归一，多义返回原值
    global_cands = []
    for k, s in _load_spec().items():
        for m, display in _chips_candidates(s):
            if m and m in norm:
                global_cands.append((len(m), display))
    if global_cands:
        best_len = max(length for length, _ in global_cands)
        best_displays = {d for length, d in global_cands if length == best_len}
        if len(best_displays) == 1:
            return (best_displays.pop(), True)
    return (str(gpu_model), False)


def _chips_candidates(spec: dict) -> List[Tuple[str, str]]:
    """收集 (归一化match关键字, display)，按关键字长度降序，保证最具体的型号先命中。"""
    candidates = []
    for chip in (spec.get("chips") or []):
        display = chip.get("display", "")
        for m in (chip.get("match") or []):
            candidates.append((str(m).lower().replace(" ", "").replace("_", ""), display))
    candidates.sort(key=lambda x: -len(x[0]))
    return candidates
