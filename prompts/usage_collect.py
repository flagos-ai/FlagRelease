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

"""usage_collect.py — 采集每次 claude -p 调用的 usage 快照（缓存命中 / 输入输出 token）。

背景：流式 assistant 事件的 usage 不完整（实测 output_tokens 恒为占位 0/1、input 恒 0），
只有最终 result 事件权威，而进程被 timeout 杀掉时 result 永不出现。但 Claude Code CLI
会把**每个已完成轮次的完整 usage 增量写入会话 transcript**，进程被杀后已完成轮次仍在。
故本脚本以 transcript 为 token 权威源（实测按 message.id 去重求和 == result.usage 逐字段相等），
配合既有产物完成归段与成本合并。

数据来源（全部只读、均为既有产物，本脚本不修改任何既有文件）：
  1. <log-dir>/claude_pipeline_*.log 的 system/init 事件 → session_id 枚举（含 cwd/model）。
     注：实测该日志的 result 事件可能缺失（一次真实运行 5 次调用只留 3 个），故 result
     仅作迟采兜底（transcript 已被 30 天清理时仍可拿到总量）。
  2. <projects-root>/<slug>/<session_id>.jsonl（Claude Code transcript）→ 逐轮 usage。
  3. <log-dir>/seg*_cost.txt → 各段成本 + 段结束时刻（mtime）→ 归段。

产物：
  <log-dir>/<label>_usage.json   每次调用一个（label = seg1/seg2_step7/... 或 inv01）
  <log-dir>/usage_summary.json   本模型汇总（分段 + 总计 + by_model + 命中率）

用法：
  python3 prompts/usage_collect.py --log-dir /data/flagos-workspace/<model>/logs --cwd <repo>
  python3 prompts/usage_collect.py --log-dir <logs> --force     # 忽略幂等快进
  python3 prompts/usage_collect.py --log-dir <logs> --cwd <repo> --scan-transcripts  # 兜底扫描补会话

约束：只读既有产物；任何异常都不阻断调用方（退出码恒 0）；可重复运行（幂等）。
"""

import argparse
import datetime
import json
import os
import re
import sys
import tempfile
from pathlib import Path

FIELDS = ("input_tokens", "output_tokens",
          "cache_read_input_tokens", "cache_creation_input_tokens")
_SID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")
# cost 文件 mtime 与 transcript 末次写入的最大允许偏差（秒）：超过则视为归段不可信
_MATCH_TOLERANCE_S = 600.0
# cost 文件写入滞后于最后一次 transcript 写入的宽限（秒）
_WINDOW_TAIL_GRACE_S = 120.0


def _atomic_write_json(path: str, obj) -> None:
    """原子写 JSON（tmp + rename），避免被杀/并发时留下半截文件。"""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _iso_to_epoch(ts: str):
    """ISO8601 → epoch 秒；解析失败返回 None。"""
    if not ts or not isinstance(ts, str):
        return None
    s = ts.strip().replace("Z", "+00:00").replace("z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _first_cwd(path: Path):
    """取 transcript 中首个 cwd 字段（用于运行归属过滤）。"""
    try:
        with open(path, errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 200:
                    break
                s = line.strip()
                if not s.startswith("{"):
                    continue
                try:
                    ev = json.loads(s)
                except Exception:
                    continue
                if isinstance(ev, dict) and ev.get("cwd"):
                    return ev["cwd"]
    except OSError:
        pass
    return None


def _run_window_end(log_files, cost_files) -> float:
    """本轮运行的结束时刻（产物最后写入时间）；兜底扫描的窗口上界。"""
    anchors = [p.stat().st_mtime for p in log_files] + [mt for _, mt, _ in cost_files]
    return (max(anchors) + 600.0) if anchors else float("inf")


def _run_window_start(log_dir: Path, log_files, cost_files) -> float:
    """本轮运行的起始时刻：优先取 context 的 timing.workflow_start，否则按产物 mtime 粗估。"""
    for cand in (log_dir.parent / "config" / "context_snapshot.yaml",
                 log_dir.parent / "config" / "context_final.yaml"):
        try:
            txt = cand.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        m = re.search(r"workflow_start\s*:\s*['\"]?([0-9Tt:+\-.Zz]+)", txt)
        if m:
            ep = _iso_to_epoch(m.group(1))
            if ep:
                return ep - 600.0
    anchors = [p.stat().st_mtime for p in log_files] + [mt for _, mt, _ in cost_files]
    return (min(anchors) - 24 * 3600) if anchors else 0.0


def parse_stream_logs(paths):
    """扫 raw 日志：收集 init 事件（session 枚举主源）与 result 事件（迟采兜底）。"""
    sessions, results, unparsable = {}, {}, 0
    for p in paths:
        try:
            with open(p, errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or not s.startswith("{"):
                        continue          # 非 JSON 行是 stderr 混流（如工具脚本输出），不计失败
                    try:
                        ev = json.loads(s)
                    except Exception:
                        unparsable += 1
                        continue
                    if not isinstance(ev, dict):
                        continue
                    sid = ev.get("session_id")
                    etype = ev.get("type")
                    if etype == "system" and ev.get("subtype") == "init" and sid:
                        sessions.setdefault(sid, {
                            "session_id": sid,
                            "cwd": ev.get("cwd", "") or "",
                            "model": ev.get("model", "") or "",
                            "source": "stream_log_init",
                        })
                    elif etype == "result" and sid:
                        u = ev.get("usage") or {}
                        rec = results.setdefault(sid, {})
                        for k in FIELDS:
                            v = u.get(k)
                            if isinstance(v, (int, float)) and v >= 0:
                                rec[k] = v
                        rec["total_cost_usd"] = ev.get("total_cost_usd")
                        rec["num_turns"] = ev.get("num_turns")
                        rec["models"] = sorted((ev.get("modelUsage") or {}).keys())
        except OSError:
            unparsable += 1
    # result 里出现但 init 缺失的会话（日志可能丢失行）也纳入枚举
    for sid in results:
        sessions.setdefault(sid, {
            "session_id": sid, "cwd": "", "model": "",
            "source": "stream_log_result",
        })
    return sessions, results, unparsable


def scan_transcripts(projects_root: Path, cwd_filter: str, since_ts: float,
                     until_ts: float, known_sids):
    """按 cwd + 时间窗兜底补充 session（raw 日志丢 init 行时仍有救）。"""
    found, skipped_cwd = {}, 0
    try:
        candidates = list(projects_root.glob("*/*.jsonl"))
    except OSError:
        return found, skipped_cwd
    for p in candidates:
        sid = p.stem
        if sid in known_sids or not _SID_RE.match(sid):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime < since_ts or st.st_mtime > until_ts:
            continue
        c = _first_cwd(p)
        if cwd_filter and c:
            try:
                if os.path.realpath(c) != os.path.realpath(cwd_filter):
                    skipped_cwd += 1
                    continue
            except OSError:
                skipped_cwd += 1
                continue
        found[sid] = {
            "session_id": sid, "cwd": c or "", "model": "",
            "source": "window_scan",
        }
    return found, skipped_cwd


def sum_transcript(path: Path):
    """按 message.id 去重求和 transcript 的逐轮 usage。

    同一 message.id 会出现多条重复条目（实测值相同），必须去重；取最后一条
    （先写占位值再补全的场景下后者更完整，实测两者等价）。
    """
    seen, order = {}, []
    duplicate_entries, unparsable = 0, 0
    tmin = tmax = None
    cwd = ""
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        return {"ok": False, "error": f"transcript 打开失败: {e}"}
    with fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("{"):
                unparsable += 1
                continue
            try:
                ev = json.loads(s)
            except Exception:
                unparsable += 1          # 尾部可能被信号截断成半行
                continue
            if not isinstance(ev, dict):
                continue
            if not cwd and ev.get("cwd"):
                cwd = ev["cwd"]
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message") or {}
            mid, usage = msg.get("id"), msg.get("usage")
            if not mid or not isinstance(usage, dict):
                continue
            ts = ev.get("timestamp")
            if isinstance(ts, str):
                if tmin is None or ts < tmin:
                    tmin = ts
                if tmax is None or ts > tmax:
                    tmax = ts
            if mid in seen:
                duplicate_entries += 1
            else:
                order.append(mid)
            seen[mid] = {
                "usage": usage,
                "model": msg.get("model") or "unknown",
                "is_side": bool(ev.get("isSidechain")),
                "ts": ts if isinstance(ts, str) else None,
            }

    totals = {k: 0 for k in FIELDS}
    side_totals = {k: 0 for k in FIELDS}
    by_model, turns = {}, []
    for mid in order:
        rec = seen[mid]
        u = rec["usage"]
        vals = {}
        for k in FIELDS:
            v = u.get(k) or 0
            if not isinstance(v, (int, float)) or v < 0:
                v = 0
            vals[k] = v
            totals[k] += v
        bm = by_model.setdefault(rec["model"], {**{k: 0 for k in FIELDS}, "messages": 0})
        for k in FIELDS:
            bm[k] += vals[k]
        bm["messages"] += 1
        if rec["is_side"]:
            for k in FIELDS:
                side_totals[k] += vals[k]
        turns.append({
            "ts": rec["ts"], "model": rec["model"],
            "in": vals["input_tokens"], "out": vals["output_tokens"],
            "cr": vals["cache_read_input_tokens"], "cc": vals["cache_creation_input_tokens"],
        })

    cc_5m = cc_1h = 0
    for mid in order:
        cc = seen[mid]["usage"].get("cache_creation")
        if isinstance(cc, dict):
            cc_5m += cc.get("ephemeral_5m_input_tokens", 0) or 0
            cc_1h += cc.get("ephemeral_1h_input_tokens", 0) or 0

    denom = totals["input_tokens"] + totals["cache_read_input_tokens"] + totals["cache_creation_input_tokens"]
    return {
        "ok": True,
        "totals": totals,
        "by_model": by_model,
        "turns": turns,
        "messages": len(order),
        "duplicate_entries": duplicate_entries,
        "unparsable_lines": unparsable,
        "sidechain_totals": side_totals,
        "sidechain_messages": sum(1 for mid in order if seen[mid]["is_side"]),
        "cache_hit_rate": round(totals["cache_read_input_tokens"] / denom, 4) if denom else None,
        "cache_creation_breakdown": {
            "ephemeral_5m_input_tokens": cc_5m,
            "ephemeral_1h_input_tokens": cc_1h,
        },
        "first_ts": tmin, "last_ts": tmax, "cwd": cwd,
    }


def read_cost_files(log_dir: Path):
    """读 seg*_cost.txt → [(stem, mtime, value)]，并按 mtime 升序。"""
    out = []
    for p in sorted(log_dir.glob("seg*_cost.txt")):
        try:
            val = float(p.read_text(encoding="utf-8").strip())
            mt = p.stat().st_mtime
        except (OSError, ValueError):
            continue
        stem = p.stem[:-5] if p.stem.endswith("_cost") else p.stem
        out.append((stem, mt, val))
    out.sort(key=lambda x: x[1])
    return out


def _session_window(s):
    """取会话的 (first_epoch, last_epoch)——时间戳在 transcript 子字典里。"""
    tr = s.get("transcript") or {}
    f = _iso_to_epoch(tr.get("first_ts"))
    l = _iso_to_epoch(tr.get("last_ts"))
    return f, l


def match_segments(cost_files, sessions):
    """把 cost 文件按 mtime 归入 session 时间窗；返回 (assigns, unattributed)。"""
    assigns, unattributed = {}, list(cost_files)
    for stem, mt, val in cost_files:
        best = None
        for sid, s in sessions.items():
            if sid in assigns:
                continue
            f, l = _session_window(s)
            if f is None and l is None:
                continue
            in_window = (f is not None and l is not None
                         and f <= mt <= l + _WINDOW_TAIL_GRACE_S)
            ref = l if l is not None else f
            delta = mt - ref
            if best is None:
                best = (in_window, abs(delta), sid, delta)
            else:
                # 优先同窗口，其次更近
                if (in_window, -abs(delta)) > (best[0], -best[1]):
                    best = (in_window, abs(delta), sid, delta)
        if best and (best[0] or best[1] <= _MATCH_TOLERANCE_S):
            _, _, sid, delta = best
            assigns[sid] = {
                "label": stem,
                "label_source": "cost_mtime_in_window" if best[0] else "cost_mtime_nearest",
                "match_delta_seconds": round(delta, 1),
                "cost_usd": round(val, 6),
                "cost_source": f"{stem}_cost.txt",
            }
            unattributed = [c for c in unattributed if c[0] != stem]
    return assigns, unattributed


_META_PER_SESSION = {
    "label": "本次调用归属的段名（seg1/seg2_step7/...）；无法归段时为 invNN（按开始时间排序）",
    "label_source": "归段依据：cost_mtime_in_window=段成本文件 mtime 落在会话时间窗内；"
                    "cost_mtime_nearest=取最近会话；cost_mtime_ordinal=无时间窗（transcript 已清理）"
                    "时按调用顺序与 cost 文件 mtime 顺序配对；ordinal=按序编号（归段失败或被杀段）",
    "match_delta_seconds": "cost 文件 mtime 与会话末次写入的偏差（秒），越小越可信",
    "session_id": "Claude Code 会话 id（transcript 文件名即此 id）",
    "session_source": "会话来源：stream_log_init=raw 日志的 init 事件（常规）；"
                      "stream_log_result=仅 result 事件中出现（init 行丢失）；"
                      "window_scan=--scan-transcripts 兜底扫描所得（未经日志佐证，请留意）",
    "status": "complete=该次调用正常结束（有 cost 文件或 result 事件）；"
              "partial=进程被终止，仅含已完成轮次的 usage（生成中的一轮尚未产生 usage）",
    "tokens_source": "token 来源：transcript=CLI 增量持久化的会话记录（权威）；"
                     "stream_log_result=raw 日志的 result 事件（transcript 已被清理时的兜底）；"
                     "unavailable=两者都不可用",
    "usage": "四类 token 累加值：input_tokens=未命中缓存的新输入；"
             "cache_read_input_tokens=缓存命中读取；cache_creation_input_tokens=缓存写入；"
             "output_tokens=输出",
    "cache_hit_rate": "缓存命中率 = cache_read / (input + cache_read + cache_creation)，"
                      "仅覆盖输入侧；口径可按需自行用原始数字重算",
    "by_model": "按模型拆分的 token 与消息数（同一次调用可能用到多个模型）",
    "turns": "逐轮明细（紧凑键：ts/model/in/out/cr/cc），按会话内首次出现顺序",
    "duplicate_entries": "transcript 中重复的 message.id 条目数（已去重，仅诊断用）",
    "unparsable_lines": "无法解析的行数（多为进程被杀导致的半行）",
    "sidechain_usage": "子代理（Task/sidechain）部分，已含在上方 usage 总计内",
    "cache_creation_breakdown": "缓存写入按 TTL 拆分（1h 单价更高），便于计费",
    "cost_usd": "该次调用成本（美元）；来自归属段的 segN_cost.txt，缺失时为 null",
    "cost_source": "成本来源文件或 result_event；null 表示无成本数据",
    "cross_check": "若 raw 日志中存在该会话的 result 事件，此处给出其 usage 以便比对",
}

_META_SUMMARY = {
    "model": "模型标识（命令行 --model 传入，仅供标注）",
    "generated_at": "本汇总生成时间（ISO 8601）",
    "log_dir": "被扫描的模型 logs 目录",
    "sessions": "每次 claude -p 调用的明细条目（与 <label>_usage.json 同源）",
    "totals": "全部调用的 token 累加值",
    "by_model": "全部调用按模型合并的 token 与消息数",
    "cache_hit_rate": "全部调用的加权缓存命中率（口径同单次）",
    "total_cost_usd": "全部 seg*_cost.txt 之和（含未归段的成本文件，最接近实际账单）",
    "attributed_cost_usd": "已归段会话的成本之和（与 total 的差额来自未归段 cost 文件）",
    "unattributed_cost_files": "存在成本文件但未能归入任何会话（如该段 claude 会话无 transcript）",
    "errors": "采集过程中的非致命错误（有值即说明部分数据缺失）",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="采集每次 claude -p 的 usage 快照（缓存命中/输入输出 token）")
    ap.add_argument("--log-dir", required=True, help="模型 logs 目录（含 claude_pipeline_*.log / seg*_cost.txt）")
    ap.add_argument("--model", default="", help="模型标识（仅写入产物便于标注）")
    ap.add_argument("--cwd", default="", help="运行 claude 时的工作目录（repo 根），用于兜底扫描归属过滤")
    ap.add_argument("--projects-root", default="", help="Claude Code projects 根目录（默认 $CLAUDE_CONFIG_DIR 或 ~/.claude 下 projects/）")
    ap.add_argument("--scan-transcripts", action="store_true",
                    help="兜底：按 cwd+时间窗扫描 transcript 补会话（raw 日志丢行时用；需同时给 --cwd，"
                         "否则会误纳同 cwd 下的交互会话）")
    ap.add_argument("--force", action="store_true", help="忽略 mtime 幂等快进")
    args = ap.parse_args()

    errors = []
    try:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"[usage] 无 logs 目录，跳过: {log_dir}")
            return 0

        projects_root = Path(args.projects_root) if args.projects_root else (
            Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude")) / "projects"
        )

        log_files = sorted(log_dir.glob("claude_pipeline_*.log"))
        cost_files = read_cost_files(log_dir)
        if not log_files and not cost_files:
            print(f"[usage] 无 claude 运行产物（无 claude_pipeline_*.log / seg*_cost.txt），跳过")
            return 0

        # ---- 幂等快进 ----
        summary_path = log_dir / "usage_summary.json"
        if not args.force and summary_path.exists():
            try:
                out_mt = summary_path.stat().st_mtime
                deps = list(log_files) + [log_dir / f"{s}_cost.txt" for s, _, _ in cost_files]
                deps_mt = [p.stat().st_mtime for p in deps if p.exists()]
                if deps_mt and max(deps_mt) < out_mt:
                    print(f"[usage] 产物已是最新，跳过（--force 可强制重采）")
                    return 0
            except OSError:
                pass

        # ---- 1. 枚举 session ----
        sessions, results, unparsable_log = parse_stream_logs(log_files)
        if unparsable_log:
            errors.append(f"raw 日志有 {unparsable_log} 行无法解析（不影响 init/result 提取）")

        if not sessions:
            errors.append("raw 日志中未找到 init/result 事件，无法枚举会话"
                          "（可加 --scan-transcripts --cwd <repo> 兜底扫描）")

        # ---- 兜底扫描（默认关闭）----
        # 主枚举来自 raw 日志的 init/result 事件（实测 5/5 完整）。按 cwd+时间窗扫描
        # transcript 是"日志丢行"时的补救手段，但会误纳同一 cwd 下的交互会话（把无关
        # token 计入该模型），故默认关闭、需显式 --scan-transcripts 且必须给 --cwd。
        if args.scan_transcripts:
            if not args.cwd:
                errors.append("--scan-transcripts 需同时提供 --cwd（防误纳其它会话），已跳过兜底扫描")
            else:
                since = _run_window_start(log_dir, log_files, cost_files)
                until = _run_window_end(log_files, cost_files)
                scanned, skipped_cwd = scan_transcripts(
                    projects_root, args.cwd, since, until, set(sessions))
                for sid, rec in scanned.items():
                    sessions.setdefault(sid, rec)
                if skipped_cwd:
                    errors.append(f"兜底扫描跳过 {skipped_cwd} 个 cwd 不匹配的 transcript")
                if scanned:
                    errors.append(f"兜底扫描补入 {len(scanned)} 个未验证来源的会话（source=window_scan）")

        # ---- 2. 解析 transcript ----
        for sid, rec in sessions.items():
            tpath = None
            if _SID_RE.match(sid):
                hits = sorted(projects_root.glob(f"*/{sid}.jsonl"),
                              key=lambda p: safe_mtime(p), reverse=True)
                if hits:
                    tpath = hits[0]
            if tpath is None:
                rec["transcript"] = None
                if sid not in results:
                    errors.append(f"会话 {sid[:8]} 无 transcript 且无 result 事件，token 不可得")
                continue
            summed = sum_transcript(tpath)
            if not summed.get("ok"):
                rec["transcript"] = None
                errors.append(f"会话 {sid[:8]}: {summed.get('error')}")
                continue
            rec["transcript"] = summed
            rec["transcript_path"] = str(tpath)
            if summed["messages"] == 0 and sid in results:
                # transcript 为空但 raw 日志有 result → 用 result 兜底
                rec["transcript"] = None

        for sid, rec in sessions.items():
            if not rec.get("model") and rec.get("transcript"):
                models = list((rec["transcript"]["by_model"] or {}).keys())
                rec["model"] = models[0] if len(models) == 1 else (",".join(models) if models else "")

        # ---- 3. 归段 + 成本 ----
        assigns, unattributed = match_segments(cost_files, sessions)

        # 归段兜底：transcript 已被清理（迟采）时没有时间窗，改用顺序配对——
        # pipeline 串行执行，故"调用顺序（raw 日志 init 出现顺序）"与
        # "cost 文件 mtime 顺序"一致。仅当剩余两侧数量相等时才配对，避免被杀的
        # 调用（无 cost 文件）导致整体错位。
        matched_stems = {a["label"] for a in assigns.values()}
        rest_costs = [c for c in cost_files if c[0] not in matched_stems]
        rest_sids = [sid for sid in sessions if sid not in assigns]
        if rest_costs and len(rest_costs) == len(rest_sids):
            for (stem, _mt, val), sid in zip(rest_costs, rest_sids):
                assigns[sid] = {
                    "label": stem,
                    "label_source": "cost_mtime_ordinal",
                    "match_delta_seconds": None,
                    "cost_usd": round(val, 6),
                    "cost_source": f"{stem}_cost.txt",
                }
            unattributed = [c for c in unattributed if c[0] not in {s for s, _, _ in rest_costs}]

        def _sort_key(r):
            f, _ = _session_window(r)
            return f if f is not None else float("inf")

        ordered = sorted(sessions.values(), key=_sort_key)
        inv_no = 0
        for rec in ordered:
            sid = rec["session_id"]
            if sid in assigns:
                rec.update(assigns[sid])
            else:
                inv_no += 1
                rec.update({
                    "label": f"inv{inv_no:02d}",
                    "label_source": "ordinal",
                    "match_delta_seconds": None,
                    "cost_usd": None,
                    "cost_source": None,
                })
            res = results.get(sid) or {}
            if rec.get("cost_usd") is None and isinstance(res.get("total_cost_usd"), (int, float)):
                rec["cost_usd"] = round(res["total_cost_usd"], 6)
                rec["cost_source"] = "result_event"
            rec["status"] = "complete" if (rec.get("cost_source") or res) else "partial"

        # ---- 4. 落盘 ----
        written = []
        agg_totals = {k: 0 for k in FIELDS}
        agg_by_model, agg_side_total = {}, 0
        agg_side_msgs = 0
        attributed_cost = 0.0
        for rec in ordered:
            sid = rec["session_id"]
            tr = rec.get("transcript")
            res = results.get(sid) or {}
            if tr:
                usage, tokens_source = tr["totals"], "transcript"
                by_model, turns = tr["by_model"], tr["turns"]
                hit = tr["cache_hit_rate"]
                cc_break = tr["cache_creation_breakdown"]
                messages, dup, unp = tr["messages"], tr["duplicate_entries"], tr["unparsable_lines"]
                first_ts, last_ts, cwd = tr["first_ts"], tr["last_ts"], tr["cwd"]
                side_usage, side_msgs = tr["sidechain_totals"], tr["sidechain_messages"]
            elif res:
                usage = {k: int(res[k]) if isinstance(res.get(k), (int, float)) else None
                         for k in FIELDS}
                if any(v is None for v in usage.values()):
                    errors.append(f"会话 {sid[:8]}: result 事件 usage 字段不全")
                usage = {k: (v or 0) for k, v in usage.items()}
                tokens_source, by_model, turns, hit, cc_break = "stream_log_result", {}, [], None, {}
                messages = res.get("num_turns") or 0
                dup = unp = 0
                first_ts = last_ts = cwd = None
                side_usage, side_msgs = {}, 0
            else:
                usage = {k: None for k in FIELDS}
                tokens_source, by_model, turns, hit, cc_break = "unavailable", {}, [], None, {}
                messages = dup = unp = 0
                first_ts = last_ts = cwd = None
                side_usage, side_msgs = {}, 0

            if cwd is None:
                cwd = rec.get("cwd") or None

            obj = {
                "_meta": _META_PER_SESSION,
                "label": rec["label"],
                "label_source": rec["label_source"],
                "session_source": rec.get("source") or "stream_log_init",
                "match_delta_seconds": rec.get("match_delta_seconds"),
                "session_id": sid,
                "status": rec["status"],
                "tokens_source": tokens_source,
                "model": rec.get("model") or None,
                "cwd": cwd,
                "messages": messages,
                "duplicate_entries": dup,
                "unparsable_lines": unp,
                "sidechain_messages": side_msgs,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "usage": usage,
                "cache_hit_rate": hit,
                "cache_creation_breakdown": cc_break,
                "sidechain_usage": side_usage,
                "by_model": by_model,
                "cost_usd": rec.get("cost_usd"),
                "cost_source": rec.get("cost_source"),
                "turns": turns,
            }
            if res:
                obj["cross_check"] = {
                    "result_usage": {k: res.get(k) for k in FIELDS},
                    "matches": (tokens_source != "transcript" or all(
                        usage[k] == (res.get(k) or 0) for k in FIELDS)),
                }
            out = log_dir / f"{rec['label']}_usage.json"
            try:
                _atomic_write_json(str(out), obj)
                written.append(out.name)
            except OSError as e:
                errors.append(f"写入 {out.name} 失败: {e}")
                continue
            if tokens_source != "unavailable":
                for k in FIELDS:
                    agg_totals[k] += usage[k] or 0
                for m, bm in by_model.items():
                    dst = agg_by_model.setdefault(m, {**{k: 0 for k in FIELDS}, "messages": 0})
                    for k in FIELDS:
                        dst[k] += bm.get(k, 0)
                    dst["messages"] += bm.get("messages", 0)
                agg_side_total += sum((side_usage or {}).values())
                agg_side_msgs += side_msgs
            if isinstance(rec.get("cost_usd"), (int, float)):
                attributed_cost += rec["cost_usd"]

        total_cost = sum(v for _, _, v in cost_files)
        denom = (agg_totals["input_tokens"] + agg_totals["cache_read_input_tokens"]
                 + agg_totals["cache_creation_input_tokens"])
        summary = {
            "_meta": _META_SUMMARY,
            "model": args.model or None,
            "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "log_dir": str(log_dir),
            "sessions": [{
                "label": r["label"], "label_source": r["label_source"],
                "session_id": r["session_id"], "status": r["status"],
                "model": r.get("model") or None,
                "usage": (r["transcript"]["totals"] if r.get("transcript")
                          else ({k: results.get(r["session_id"], {}).get(k) for k in FIELDS}
                                if r["session_id"] in results else None)),
                "cache_hit_rate": (r["transcript"]["cache_hit_rate"] if r.get("transcript") else None),
                "cost_usd": r.get("cost_usd"),
                "first_ts": (r["transcript"]["first_ts"] if r.get("transcript") else None),
                "last_ts": (r["transcript"]["last_ts"] if r.get("transcript") else None),
                "match_delta_seconds": r.get("match_delta_seconds"),
                "file": f"{r['label']}_usage.json",
            } for r in ordered],
            "totals": agg_totals,
            "by_model": agg_by_model,
            "cache_hit_rate": round(agg_totals["cache_read_input_tokens"] / denom, 4) if denom else None,
            "sidechain_usage": {"messages": agg_side_msgs, "total_tokens": agg_side_total},
            "total_cost_usd": round(total_cost, 6) if cost_files else None,
            "attributed_cost_usd": round(attributed_cost, 6),
            "unattributed_cost_files": [
                {"segment": s, "cost_usd": round(v, 6)} for s, _, v in unattributed
            ],
            "errors": errors,
        }
        try:
            _atomic_write_json(str(summary_path), summary)
        except OSError as e:
            print(f"[usage] 写入 usage_summary.json 失败: {e}")
            return 0

        # ---- 5. 打印 ----
        def _fmt(n):
            return f"{n:,}" if isinstance(n, int) else "-"
        print(f"[usage] {args.model or log_dir.name}: "
              f"{len([r for r in ordered if r['status'] == 'complete'])}/{len(ordered)} 段完成"
              f"{'（有被杀段）' if any(r['status'] == 'partial' for r in ordered) else ''}")
        print(f"  {'段':<14}{'状态':<9}{'输入':>12}{'缓存读':>13}{'缓存写':>13}{'输出':>11}{'命中率':>8}{'成本$':>9}")
        for r in ordered:
            u = (r["transcript"]["totals"] if r.get("transcript")
                 else {k: results.get(r["session_id"], {}).get(k) for k in FIELDS})
            hit = r["transcript"]["cache_hit_rate"] if r.get("transcript") else None
            cost = r.get("cost_usd")
            print(f"  {r['label']:<14}{r['status']:<9}{_fmt(u.get('input_tokens')):>12}"
                  f"{_fmt(u.get('cache_read_input_tokens')):>13}{_fmt(u.get('cache_creation_input_tokens')):>13}"
                  f"{_fmt(u.get('output_tokens')):>11}"
                  f"{(f'{hit * 100:.1f}%' if isinstance(hit, (int, float)) else '-'):>8}"
                  f"{(f'{cost:.2f}' if isinstance(cost, (int, float)) else '-'):>9}")
        hit_txt = (f"{summary['cache_hit_rate'] * 100:.1f}%"
                   if isinstance(summary.get("cache_hit_rate"), (int, float)) else "-")
        cost_txt = f" | 成本 ${total_cost:.2f}" if cost_files else ""
        print(f"  合计: 输入 {_fmt(agg_totals['input_tokens'])} / 缓存读 {_fmt(agg_totals['cache_read_input_tokens'])}"
              f" / 缓存写 {_fmt(agg_totals['cache_creation_input_tokens'])} / 输出 {_fmt(agg_totals['output_tokens'])}"
              f" | 命中率 {hit_txt}{cost_txt}")
        print(f"  产物: {', '.join(written[:4])}{' …' if len(written) > 4 else ''}, usage_summary.json")
        if summary["unattributed_cost_files"]:
            print(f"  ⚠ 未归段成本文件: {[u['segment'] for u in summary['unattributed_cost_files']]}")
        if errors:
            print(f"  ⚠ {len(errors)} 条非致命问题（详见 usage_summary.json 的 errors）")
        return 0
    except Exception as e:                      # 采集器绝不阻断调用方
        print(f"[usage] 采集异常（已忽略，不影响主流程）: {type(e).__name__}: {e}")
        return 0


def safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


if __name__ == "__main__":
    sys.exit(main())
