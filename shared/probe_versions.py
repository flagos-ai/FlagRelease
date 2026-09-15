#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_versions.py — 独立组件版本探测工具

职责单一：在**当前环境**现场探测 flag 生态与核心组件的真实版本，输出结构化 JSON。
不依赖 context.yaml 传递、不依赖包的 __version__（源码 pip install . 装的常缺失），
以 `pip list` 的元数据为权威来源，与发布侧 get_image_name.sh 的 semver 口径一致。

设计要点
--------
- **现场探测**：本脚本部署在容器 /flagos-workspace/scripts 下，报告生成时同容器运行，
  `pip list` 即当前环境，无需 docker exec。
- **成败无关**：任何单项探测失败都退化为 null，绝不抛错中断——保证流程收尾（无论
  成功失败）都能拿到尽可能完整的版本信息，缺的显示 null 由消费方兜底为 "-"。
- **口径统一**：semver() 保留 x.y.z、去掉 +local/.devN 后缀，与 get_image_name.sh 一致。

用法
----
    python3 probe_versions.py                         # 打印全部版本 JSON 到 stdout
    python3 probe_versions.py --output versions.json  # 同时落盘
    python3 probe_versions.py --field flaggems        # 只打某一项（shell 取值用）
    python3 probe_versions.py --ops                   # 追加当前生效算子集（现态）

探测项（key → pip 包名，含厂商特例回退）
    flaggems      flag_gems
    flagtree      flagtree
    flagcx        flagcx
    vllm          vllm（ascend 场景优先 vllm-ascend）
    plugin_fl     vllm-plugin-fl → vllm_fl
    torch         torch（ascend 场景优先 torch_npu）
    python        python3 --version
"""
import argparse
import json
import os
import re
import subprocess
import sys


# ── pip 包名 → 报告字段名 ──────────────────────────────
# 值为候选包名列表，按序取首个命中（兼容改名 / 厂商特例）。
_PKG_MAP = {
    "flaggems": ["flag_gems", "flaggems"],
    "flagtree": ["flagtree"],
    "flagcx": ["flagcx"],
    "vllm": ["vllm"],
    "plugin_fl": ["vllm-plugin-fl", "vllm_fl", "vllm-plugin-FL"],
    "torch": ["torch"],
}


def _run(cmd, timeout=60):
    """跑命令拿 stdout；失败返回空串（绝不抛错）。"""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        return r.stdout or ""
    except Exception:
        return ""


def _pip_list_map():
    """`pip list` → {规范化小写包名: 版本}。失败返回 {}。

    规范化：下划线/连字符统一为连字符、转小写，兼容 flag_gems / flag-gems。
    """
    out = _run([sys.executable, "-m", "pip", "list", "--format=freeze"], timeout=90)
    if not out.strip():
        # fallback: 纯 pip list（列宽格式）
        out = _run([sys.executable, "-m", "pip", "list"], timeout=90)
        table = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1][0:1].isdigit():
                table[_norm(parts[0])] = parts[1]
        return table
    table = {}
    for line in out.splitlines():
        if "==" not in line:
            continue
        name, _, ver = line.partition("==")
        if name and ver:
            table[_norm(name)] = ver.strip()
    return table


def _norm(name):
    return re.sub(r"[-_]+", "-", name.strip()).lower()


def semver(v):
    """保留 x.y.z 粒度、去掉本地/dev 后缀。与 get_image_name.sh semver() 一致。

    "0.3.0+metax0425" → "0.3.0" ; "0.8.5.dev123" → "0.8.5" ; None/"" → None
    """
    if not v:
        return None
    m = re.match(r"^\s*v?(\d+\.\d+(?:\.\d+)?)", str(v))
    return m.group(1) if m else str(v).strip() or None


def _python_version():
    out = _run([sys.executable, "--version"], timeout=15)
    m = re.search(r"(\d+\.\d+\.\d+)", out)
    if m:
        return m.group(1)
    return "{}.{}.{}".format(*sys.version_info[:3])


def probe(with_ops=False):
    """现场探测所有版本，返回 dict。任何单项失败 → null。"""
    pip_map = _pip_list_map()
    result = {}

    for field, candidates in _PKG_MAP.items():
        raw = None
        for pkg in candidates:
            raw = pip_map.get(_norm(pkg))
            if raw:
                break
        result[field] = semver(raw)

    # 厂商特例：ascend 用 vllm-ascend / torch_npu 覆盖
    vllm_ascend = pip_map.get(_norm("vllm-ascend"))
    if vllm_ascend:
        result["vllm"] = semver(vllm_ascend)
        result["vllm_backend"] = "vllm-ascend"
    torch_npu = pip_map.get(_norm("torch_npu")) or pip_map.get(_norm("torch-npu"))
    if torch_npu:
        result["torch"] = semver(torch_npu)
        result["torch_backend"] = "torch_npu"

    result["python"] = _python_version()

    if with_ops:
        result["current_ops"] = _probe_current_ops()

    return result


def _probe_current_ops():
    """现场读取当前生效的 FlagGems 算子集（现态，非历史版本）。

    优先级：控制文件 /root/flaggems_ops_control.json 的 include →
    启动实际生效清单 flaggems_enable_oplist.txt（DEBUG 全路径，取函数名）。
    仅反映"最后一次启动"的算子集，历史 V2/V3/V4 各自算子集仍需从 results 读。
    """
    # 1) 控制文件 include
    ctrl = "/root/flaggems_ops_control.json"
    try:
        if os.path.isfile(ctrl):
            with open(ctrl, encoding="utf-8") as f:
                d = json.load(f)
            inc = d.get("include") if isinstance(d, dict) else None
            if inc:
                return sorted(set(inc))
    except Exception:
        pass

    # 2) 启动实际生效清单（DEBUG 全路径 → 末段函数名）
    for path in (
        "/flagos-workspace/results/flaggems_enable_oplist.txt",
        "/tmp/flaggems_enable_oplist.txt",
    ):
        try:
            if os.path.isfile(path):
                ops = set()
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # 形如 [DEBUG] flag_gems.ops.add.add: GEMS ...
                        m = re.search(r"flag_gems\.ops\.[\w.]*?\.(\w+)\s*:", line)
                        if m:
                            ops.add(m.group(1))
                        elif not line.startswith("["):
                            ops.add(line)
                if ops:
                    return sorted(ops)
        except Exception:
            continue

    return []


def main():
    ap = argparse.ArgumentParser(description="现场组件版本探测（独立工具）")
    ap.add_argument("--output", "-o", help="同时写入 JSON 文件路径")
    ap.add_argument("--field", help="只打印某一字段的值（缺失打印空串）")
    ap.add_argument("--ops", action="store_true", help="追加当前生效算子集 current_ops")
    args = ap.parse_args()

    data = probe(with_ops=args.ops)

    if args.field:
        val = data.get(args.field)
        print(val if val is not None else "")
        return

    js = json.dumps(data, indent=2, ensure_ascii=False)
    print(js)
    if args.output:
        try:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(js)
            print(f"已保存: {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"写入失败（不阻断）: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
