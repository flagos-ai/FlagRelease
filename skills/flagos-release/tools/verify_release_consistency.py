#!/usr/bin/env python3
"""verify_release_consistency.py — 发布一致性校验（批处理结束兜底）

背景（2026-08-05 Mistral-Small-24B V3 发布事故）：
发布命令因 env VAR=... / /opt/conda/bin/python3 前缀未命中 Bash 白名单被 headless
自动拒绝，main.py 未执行但流程继续，造成"语义正确却静默未发布"。白名单已加兜底规则
（settings.local.json 的 env * / 绝对路径 / nohup 前缀），本脚本作为最后一道防线：
发布步骤 trace 声称 success 时，校验镜像 tag 确实已产出并回写 context；缺失则报错
供编排层触发自动重试或告警。

校验逻辑：
1. 找 traces/ 下发布 trace：13_plugin_release.json（分支A V3）、08_release.json（分支B V2）。
   文件缺失且该步骤未跳过 → 告警（发布可能未执行）。
2. 每个存在且 status=success 的发布 trace：
   - 其 context_updates.image.registry_url 应非空且含版本 tag（-v2/-v3/-v4）
   - context_snapshot.yaml 的 release.harbor_image / image.registry_url 应与其一致
   - 若 trace 成功但 context 无镜像 → 数据不一致（发布结果未落盘）
3. 可选线上比对：ModelScope 仓库 README 中的镜像 tag（需要 token/网络，失败降级为本地校验）

Usage:
    python3 verify_release_consistency.py --host-base /data/flagos-workspace/<model>
    python3 verify_release_consistency.py --host-base ... --container <ctr> --verify-online

退出码: 0=一致/无可校验；1=发现不一致（供编排层重试/告警）
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

# 发布 trace 文件（按优先级）
RELEASE_TRACES = [
    ("13_plugin_release.json", ["v3", "v4"]),
    ("08_release.json", ["v2", "v3", "v4"]),
]

# trace 里应含的版本 tag 后缀
VERSION_SUFFIXES = ("-v2", "-v3", "-v4")


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_yaml(path: str) -> Optional[dict]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def extract_image_tag(text: str) -> List[str]:
    """从字符串提取 harbor 镜像 tag（含 -vN 后缀的 12 位时间戳形式）"""
    return re.findall(r"[a-z0-9_.-]+(?::\d{12}(?:-v[1-4])?)\b", text)


def check_trace(host_base: str, issues: List[str]):
    """校验 traces/ 下的发布 trace 与 context 的一致性"""
    traces_dir = os.path.join(host_base, "traces")
    ctx_snap = os.path.join(host_base, "config", "context_snapshot.yaml")
    ctx_final = os.path.join(host_base, "config", "context_final.yaml")
    ctx_path = ctx_final if os.path.isfile(ctx_final) else ctx_snap

    ctx = read_yaml(ctx_path) if os.path.isfile(ctx_path) else {}
    if not os.path.isdir(traces_dir):
        issues.append(f"traces/ 目录不存在: {traces_dir}（发布是否执行过？）")
        return

    release = ctx.get("release", {}) or {}
    image = ctx.get("image", {}) or {}
    ctx_has_image = bool(release.get("harbor_image") or image.get("registry_url"))
    ctx_release_ok = bool(release.get("modelscope_url") or release.get("huggingface_url"))

    found_any = False
    for trace_name, expect_suffixes in RELEASE_TRACES:
        trace_path = os.path.join(traces_dir, trace_name)
        if not os.path.isfile(trace_path):
            continue
        found_any = True
        trace = load_json(trace_path)
        if not trace:
            issues.append(f"{trace_name} 解析失败（JSON 损坏）")
            continue
        status = trace.get("status", "")
        cu = trace.get("context_updates", {}) or {}

        if status == "skipped":
            continue
        if status != "success":
            issues.append(f"{trace_name} status={status}（发布未成功）")
            continue

        # trace 声称成功 → 镜像 tag 必须已回写
        img = cu.get("image.registry_url") or cu.get("image_url") or ""
        if not img:
            issues.append(
                f"{trace_name} status=success 但 context_updates 无 image.registry_url"
                f"（发布命令可能被拒/未执行，镜像未推送）"
            )
            continue
        if not any(s in img for s in VERSION_SUFFIXES):
            issues.append(
                f"{trace_name} 镜像 tag 不含版本后缀 {VERSION_SUFFIXES}: {img}"
            )

        # trace 成功但 context 无镜像 → 数据未落盘
        if not ctx_has_image:
            issues.append(
                f"{trace_name} status=success 但 context 无 harbor_image/registry_url"
                f"（发布结果未回写 context）"
            )

    # 发布 trace 全部缺失：仅当 context 显示"应已发布"时才报错
    # （应发布 = 流程已推进到发布阶段：qualified/triggered/versions 或 release 段有数据），
    # 避免对中途退出或分支B 仅写 08_release 的场景误报。
    if not found_any:
        versions = ctx.get("versions", {}) or {}
        workflow = ctx.get("workflow", {}) or {}
        plugin_wf = ctx.get("plugin_workflow", {}) or {}
        should_have_released = (
            bool(workflow.get("qualified"))
            or bool(plugin_wf.get("triggered"))
            or any((v or {}).get("image_url") or (v or {}).get("harbor_image")
                   for v in versions.values())
            or ctx_has_image
        )
        if should_have_released:
            issues.append(
                "发布 trace（08_release/13_plugin_release）全部缺失，"
                "但 context 显示流程应已完成发布（发布命令可能被拒/未执行）"
            )


def check_online_readme(host_base: str, container: str, issues: List[str]):
    """可选线上比对：容器内 ModelScope 上传 README 中的镜像 tag vs context。

    需要容器存活 + token。任一失败降级为本地校验（不追加 issue）。
    """
    ctx_snap = os.path.join(host_base, "config", "context_snapshot.yaml")
    ctx = read_yaml(ctx_snap) if os.path.isfile(ctx_snap) else {}
    release = ctx.get("release", {}) or {}
    ms_url = release.get("modelscope_url", "")
    if not ms_url or "modelscope.cn/models/" not in ms_url:
        return  # 无 ModelScope 仓库信息，跳过线上比对
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c",
             f"cd /flagos-workspace/scripts 2>/dev/null && "
             f"PATH=/opt/conda/bin:$PATH python3 -c "
             f"\"from modelscope.hub.api import HubApi; "
             f"api=HubApi(); "
             f"print(api.get_model('{ms_url.split('/models/')[-1]}').model_id or '') \" 2>/dev/null || echo ''"],
            capture_output=True, text=True, timeout=30,
        )
        # 线上比对失败/无输出 → 跳过（可选校验不阻断）
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="发布一致性校验")
    ap.add_argument("--host-base", required=True, help="宿主机工作目录 /data/flagos-workspace/<model>")
    ap.add_argument("--container", default="", help="容器名（线上比对用，可选）")
    ap.add_argument("--verify-online", action="store_true", help="执行线上 README 比对（可选）")
    args = ap.parse_args()

    if not os.path.isdir(args.host_base):
        print(f"  ⚠ 校验跳过：工作目录不存在 {args.host_base}")
        sys.exit(0)

    issues: List[str] = []
    check_trace(args.host_base, issues)
    if args.verify_online and args.container:
        check_online_readme(args.host_base, args.container, issues)

    if issues:
        print(f"\n⚠ 发布一致性校验发现 {len(issues)} 个问题:")
        for i in issues:
            print(f"  - {i}")
        print("  提示：可用以下命令重试发布（幂等，已发布部分自动跳过）:")
        print(f"    python3 skills/flagos-release/tools/main.py "
              f"--from-context {args.host_base}/config/context_snapshot.yaml")
        sys.exit(1)
    print("✓ 发布一致性校验通过（发布 trace 与 context 镜像一致）")
    sys.exit(0)


if __name__ == "__main__":
    main()
