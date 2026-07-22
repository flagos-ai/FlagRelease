#!/usr/bin/env python3
"""Detached event queue and worker for FlagRelease progress notifications.

This module never launches or controls the migration pipeline.  The pipeline only
starts ``progress_runner.sh emit`` as a detached best-effort process; all file IO,
Claude analysis, state updates and Feishu requests happen here.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import progress_summary as summary


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SUMMARY_SCRIPT = SCRIPT_DIR / "progress_summary.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "tools" / "batch_summarize" / "summarize.sh"

DEFAULT_WORKSPACE = Path("/data/flagos-workspace")
DEFAULT_ROOT = DEFAULT_WORKSPACE / ".flagrelease-progress"
EVENT_TYPES = {
    "batch-start",
    "model-finish",
    "skip-model",
    "batch-end",
    "single-start",
    "single-finish",
}
TERMINAL_OUTCOMES = {"success", "failed", "timeout", "skipped"}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_component(value: str, fallback: str = "unknown") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return cleaned[:160] or fallback


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def progress_root(raw: Optional[Path]) -> Path:
    if raw:
        return raw
    configured = os.environ.get("FLAGOS_PROGRESS_ROOT", "").strip()
    return Path(configured) if configured else DEFAULT_ROOT


def state_path(root: Path, stream_id: str) -> Path:
    return root / "states" / f"batch_{safe_component(stream_id)}_progress.json"


def event_sequence(event_type: str, task_index: Optional[int]) -> int:
    if event_type in {"batch-start", "single-start"}:
        return 0
    if event_type == "single-finish":
        return 10
    if event_type in {"model-finish", "skip-model"}:
        return max(1, int(task_index or 0)) * 10
    return 999_999_999


def event_filename(event: Dict[str, Any]) -> str:
    sequence = int(event["sequence"])
    event_type = safe_component(str(event["event_type"]))
    task_index = int(event.get("task_index") or 0)
    return f"{sequence:09d}_{event_type}_{task_index:06d}.json"


def append_log(root: Path, message: str) -> None:
    try:
        path = root / "logs" / "progress_worker.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now_iso()}] {message}\n")
    except OSError:
        pass


def summary_command(root: Path, stream_id: str, command: str, args: Iterable[str]) -> None:
    full_command = [sys.executable, str(SUMMARY_SCRIPT), command, *map(str, args)]
    if os.environ.get("FLAGOS_PROGRESS_NOTIFY", "1") != "0":
        full_command.append("--notify")
    if os.environ.get("FLAGOS_PROGRESS_DRY_RUN", "0") == "1":
        full_command.append("--dry-run")
    completed = subprocess.run(full_command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"progress_summary {command} failed: exit={completed.returncode}")


def event_result_path(root: Path, event: Dict[str, Any]) -> Path:
    stream = safe_component(str(event["stream_id"]))
    task_index = int(event.get("task_index") or 1)
    model = safe_component(str(event.get("model") or "model"), "model")
    return root / "results" / stream / f"{task_index:06d}_{model}.json"


def result_is_valid(path: Path) -> bool:
    try:
        summary.model_from_result_file(path)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def failed_analysis_result(event: Dict[str, Any], reason: str) -> Dict[str, Any]:
    outcome = str(event.get("outcome") or "failed")
    if outcome == "success":
        result_label = "无法确认"
        lead = "⚠️ 流程已结束，结果分析失败"
    elif outcome == "timeout":
        result_label = "超时"
        lead = "⏱️ 任务超时"
    elif outcome == "skipped":
        result_label = "已跳过"
        lead = "⏭️ 已跳过"
    else:
        result_label = "流程失败"
        lead = "❌ 流程失败"
    return {
        "schema_version": 1,
        "analysis_status": "failed",
        "analysis_error": reason[:1000],
        "model": str(event.get("model") or ""),
        "target": str(event.get("target") or ""),
        "vendor": str(event.get("vendor") or "unknown"),
        "pipeline": {
            "outcome": outcome,
            "exit_code": int(event.get("exit_code") or 0),
            "elapsed_seconds": max(0, int(event.get("elapsed_seconds") or 0)),
        },
        "cost": {"migration_cost_usd": None, "complete": False},
        "delivery": {
            "version": None,
            "label": None,
            "accuracy_ok": None,
            "uploaded": None,
            "qualified_uploaded": False,
        },
        "notification": {
            "result_label": result_label,
            "lead": lead,
            "summary": None,
            "warning": "单模型结果分析失败",
        },
        "evidence": {"accuracy": [], "upload": [], "cost": []},
        "analysis": {"elapsed_seconds": 0, "cost_usd": None},
        "analyzed_at": now_iso(),
    }


def reusable_result(root: Path, event: Dict[str, Any], destination: Path) -> bool:
    candidates_with_mtime = []
    for item in root.glob("results/*/*.json"):
        try:
            candidates_with_mtime.append((item.stat().st_mtime, item))
        except OSError:
            continue
    candidates = [item for _, item in sorted(candidates_with_mtime, reverse=True)]
    for candidate in candidates:
        if candidate == destination or not result_is_valid(candidate):
            continue
        data = load_json(candidate)
        if str(data.get("model") or "") != str(event.get("model") or ""):
            continue
        if str(data.get("target") or "") != str(event.get("target") or ""):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(candidate, temp)
            os.replace(temp, destination)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        return True
    return False


def analyze_model(root: Path, event: Dict[str, Any]) -> Path:
    destination = event_result_path(root, event)
    if event["event_type"] == "skip-model" and reusable_result(root, event, destination):
        return destination

    try:
        destination.unlink()
    except FileNotFoundError:
        pass
    destination.parent.mkdir(parents=True, exist_ok=True)

    workspace = Path(str(event.get("workspace") or DEFAULT_WORKSPACE))
    model = str(event.get("model") or "")
    timeout_seconds = max(1, int(os.environ.get("FLAGOS_MODEL_ANALYSIS_TIMEOUT_SECONDS", "1800")))
    command = [
        "bash",
        str(ANALYZE_SCRIPT),
        "--model",
        str(workspace / model),
        "--model-name",
        model,
        "--result-json",
        str(destination),
        "--target",
        str(event.get("target") or ""),
        "--vendor",
        str(event.get("vendor") or "unknown"),
        "--outcome",
        str(event.get("outcome") or "failed"),
        "--exit-code",
        str(int(event.get("exit_code") or 0)),
        "--elapsed-seconds",
        str(max(0, int(event.get("elapsed_seconds") or 0))),
    ]
    ionice = shutil.which("ionice")
    if ionice and os.environ.get("FLAGOS_PROGRESS_IONICE", "1") != "0":
        command = [ionice, "-c", "3", *command]

    analysis_env = os.environ.copy()
    analysis_env["CUDA_VISIBLE_DEVICES"] = ""
    analysis_env["NVIDIA_VISIBLE_DEVICES"] = "void"
    config_dir = analysis_env.get("FLAGOS_PROGRESS_CLAUDE_CONFIG_DIR", "").strip()
    if config_dir:
        analysis_env["CLAUDE_CONFIG_DIR"] = config_dir

    # 明确指定 CWD 为项目根目录，确保 Claude CLI 能加载
    # .claude/settings.local.json 中的工具权限白名单。
    # 不设置 CLAUDE_CONFIG_DIR（会覆盖 ~/.claude/ 导致认证失败），
    # 而是通过 CWD 让 Claude 自动发现项目级 settings.local.json。
    analysis_cwd = str(PROJECT_ROOT)

    reason = "单模型结果分析未生成合法结果"
    try:
        completed = subprocess.run(
            command,
            check=False,
            env=analysis_env,
            stdin=subprocess.DEVNULL,
            cwd=analysis_cwd,
            timeout=timeout_seconds + 60,
        )
        if completed.returncode != 0:
            reason = f"单模型结果分析失败（exit={completed.returncode}）"
    except subprocess.TimeoutExpired:
        reason = f"单模型结果分析超时（>{timeout_seconds + 60}s）"
    except OSError as exc:
        reason = f"单模型结果分析无法启动: {exc}"

    if not result_is_valid(destination):
        atomic_write_json(destination, failed_analysis_result(event, reason))
    return destination


def task_snapshot(root: Path, event: Dict[str, Any]) -> Path:
    stream = safe_component(str(event["stream_id"]))
    snapshot = root / "metadata" / stream / "tasks.txt"
    content = event.get("task_file_content")
    if isinstance(content, str):
        atomic_write_text(snapshot, content)
        return snapshot
    original = Path(str(event.get("task_file") or ""))
    if not original.is_file():
        raise RuntimeError("batch-start task file is unavailable")
    atomic_write_text(snapshot, original.read_text(encoding="utf-8"))
    return snapshot


def common_summary_args(root: Path, event: Dict[str, Any]) -> list[str]:
    return [
        "--workspace",
        str(event.get("workspace") or DEFAULT_WORKSPACE),
        "--state-file",
        str(state_path(root, str(event["stream_id"]))),
        "--batch-id",
        str(event["stream_id"]),
    ]


def process_event(root: Path, event: Dict[str, Any]) -> None:
    event_type = str(event["event_type"])
    stream_id = str(event["stream_id"])
    common = common_summary_args(root, event)

    if event_type == "batch-start":
        snapshot = task_snapshot(root, event)
        args = [*common, "--task-file", str(snapshot)]
        if event.get("batch_started_at") is not None:
            args.extend(["--started-at-epoch", str(int(event["batch_started_at"]))])
        summary_command(root, stream_id, "batch-start", args)
        return

    if event_type in {"model-finish", "skip-model"}:
        result_file = analyze_model(root, event)
        args = [
            *common,
            "--task-index",
            str(int(event.get("task_index") or 0)),
            "--target",
            str(event.get("target") or ""),
            "--model",
            str(event.get("model") or ""),
            "--vendor",
            str(event.get("vendor") or "unknown"),
            "--result-file",
            str(result_file),
            "--outcome",
            str(event.get("outcome") or "skipped"),
            "--exit-code",
            str(int(event.get("exit_code") or 0)),
            "--elapsed-seconds",
            str(max(0, int(event.get("elapsed_seconds") or 0))),
        ]
        if event.get("run_ended_at") is not None:
            args.extend(["--event-time-epoch", str(int(event["run_ended_at"]))])
        if event.get("batch_elapsed_seconds") is not None:
            args.extend(["--batch-elapsed-seconds", str(int(event["batch_elapsed_seconds"]))])
        summary_command(root, stream_id, "model-finish", args)
        return

    if event_type == "batch-end":
        args = [*common, "--exit-code", str(int(event.get("exit_code") or 0))]
        if event.get("run_ended_at") is not None:
            args.extend(["--event-time-epoch", str(int(event["run_ended_at"]))])
        if event.get("elapsed_seconds") is not None:
            args.extend(["--batch-elapsed-seconds", str(int(event["elapsed_seconds"]))])
        summary_command(root, stream_id, "batch-end", args)
        return

    if event_type == "single-start":
        summary_command(
            root,
            stream_id,
            "single-start",
            [
                "--workspace",
                str(event.get("workspace") or DEFAULT_WORKSPACE),
                "--target",
                str(event.get("target") or ""),
                "--model",
                str(event.get("model") or ""),
                "--vendor",
                str(event.get("vendor") or "unknown"),
            ],
        )
        return

    if event_type == "single-finish":
        result_file = analyze_model(root, event)
        summary_command(
            root,
            stream_id,
            "single-end",
            [
                "--workspace",
                str(event.get("workspace") or DEFAULT_WORKSPACE),
                "--target",
                str(event.get("target") or ""),
                "--model",
                str(event.get("model") or ""),
                "--vendor",
                str(event.get("vendor") or "unknown"),
                "--result-file",
                str(result_file),
                "--outcome",
                str(event.get("outcome") or "failed"),
                "--exit-code",
                str(int(event.get("exit_code") or 0)),
                "--elapsed-seconds",
                str(max(0, int(event.get("elapsed_seconds") or 0))),
            ],
        )
        return

    raise RuntimeError(f"unsupported event type: {event_type}")


def processed_path(root: Path, stream_id: str, filename: str) -> Path:
    return root / "processed" / safe_component(stream_id) / filename


def recover_interrupted_events(root: Path, stream: str) -> None:
    """Return unacknowledged processing files to the queue after a worker crash."""
    processing_dir = root / "processing" / stream
    queue_dir = root / "queue" / stream
    if not processing_dir.is_dir():
        return
    queue_dir.mkdir(parents=True, exist_ok=True)
    for processing in sorted(processing_dir.glob("*.json")):
        completed = root / "processed" / stream / processing.name
        if completed.exists():
            processing.unlink(missing_ok=True)
            continue
        try:
            os.replace(processing, queue_dir / processing.name)
        except FileNotFoundError:
            continue


def known_model_event_count(root: Path, stream: str) -> int:
    names = set()
    for area in ("queue", "processing", "processed", "dead-letter"):
        directory = root / area / stream
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            if "_model-finish_" in path.name or "_skip-model_" in path.name:
                names.add(path.name)
    return len(names)


def settle_before_batch_end(root: Path, stream: str, event: Dict[str, Any]) -> None:
    """Give detached model emitters time to land before consuming batch-end."""
    expected = max(0, int(event.get("processed") or 0))
    if known_model_event_count(root, stream) >= expected:
        return

    stable_seconds = max(0.0, float(os.environ.get("FLAGOS_PROGRESS_END_SETTLE_SECONDS", "2")))
    max_wait_seconds = max(stable_seconds, float(os.environ.get("FLAGOS_PROGRESS_END_MAX_WAIT_SECONDS", "30")))
    if max_wait_seconds <= 0:
        return

    deadline = time.monotonic() + max_wait_seconds
    last_count = known_model_event_count(root, stream)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))
        current = known_model_event_count(root, stream)
        if current >= expected:
            return
        if current != last_count:
            last_count = current
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= stable_seconds:
            return


def run_worker(root: Path, stream_id: str, start_only: bool = False) -> int:
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass

    stream = safe_component(stream_id)
    lock_path = root / "locks" / f"{stream}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        queue_dir = root / "queue" / stream
        processing_dir = root / "processing" / stream
        dead_dir = root / "dead-letter" / stream
        max_attempts = max(1, int(os.environ.get("FLAGOS_PROGRESS_MAX_ATTEMPTS", "3")))
        recover_interrupted_events(root, stream)

        while True:
            pending = sorted(queue_dir.glob("*.json")) if queue_dir.is_dir() else []
            if not pending:
                return 0
            progressed = False
            for queued in pending:
                event = load_json(queued)
                if not event:
                    dead_dir.mkdir(parents=True, exist_ok=True)
                    os.replace(queued, dead_dir / queued.name)
                    append_log(root, f"dead-letter invalid event: {queued}")
                    progressed = True
                    continue
                event_type = str(event.get("event_type") or "")
                if event_type in {"model-finish", "skip-model", "batch-end"}:
                    start_done = any((root / "processed" / stream).glob("*_batch-start_*.json"))
                    if not start_done:
                        # A detached batch-start emitter may simply have been
                        # scheduled later. Leave all later events untouched; the
                        # start emitter will launch a worker when it lands.
                        return 0
                if start_only and event.get("event_type") != "batch-start":
                    return 0
                if event.get("event_type") == "batch-end":
                    settle_before_batch_end(root, stream, event)
                    # Detached emitters may have added lower-sequence events while
                    # settling. Rescan from the beginning before finalizing.
                    refreshed = sorted(queue_dir.glob("*.json")) if queue_dir.is_dir() else []
                    if refreshed and refreshed[0].name != queued.name:
                        progressed = True
                        break
                completed = processed_path(root, stream_id, queued.name)
                if completed.exists():
                    queued.unlink(missing_ok=True)
                    progressed = True
                    continue

                processing_dir.mkdir(parents=True, exist_ok=True)
                processing = processing_dir / queued.name
                try:
                    os.replace(queued, processing)
                except FileNotFoundError:
                    continue

                try:
                    process_event(root, event)
                except Exception as exc:  # noqa: BLE001 - worker must contain every failure
                    event["attempts"] = int(event.get("attempts") or 0) + 1
                    event["last_error"] = str(exc)[:1000]
                    event["last_failed_at"] = now_iso()
                    atomic_write_json(processing, event)
                    if event["attempts"] >= max_attempts:
                        dead_dir.mkdir(parents=True, exist_ok=True)
                        os.replace(processing, dead_dir / processing.name)
                        append_log(root, f"dead-letter {processing.name}: {exc}")
                    else:
                        queue_dir.mkdir(parents=True, exist_ok=True)
                        os.replace(processing, queue_dir / processing.name)
                        append_log(root, f"event retry deferred {processing.name}: {exc}")
                    return 1
                else:
                    completed.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(processing, completed)
                    progressed = True
                    if start_only:
                        end_is_waiting = any(queue_dir.glob("*_batch-end_*.json"))
                        if end_is_waiting:
                            start_only = False
                        else:
                            return 0
            if not progressed:
                return 0


def spawn_worker(root: Path, stream_id: str, start_only: bool = False) -> None:
    log_path = root / "logs" / f"worker_{safe_component(stream_id)}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--progress-root", str(root), "--batch-id", stream_id]
    if start_only:
        command.append("--start-only")
    with log_path.open("ab", buffering=0) as log_handle:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            env=os.environ.copy(),
        )


def command_emit(args: argparse.Namespace) -> int:
    if args.event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported event type: {args.event_type}")
    root = progress_root(args.progress_root)
    stream_id = str(args.batch_id or "").strip()
    if not stream_id:
        raise ValueError("emit requires --batch-id")

    outcome = args.outcome
    if args.event_type == "skip-model":
        outcome = "skipped"
    if args.event_type in {"model-finish", "single-finish"} and outcome not in TERMINAL_OUTCOMES:
        raise ValueError("finish event requires --outcome")

    event: Dict[str, Any] = {
        "schema_version": 1,
        "event_type": args.event_type,
        "stream_id": stream_id,
        "batch_id": stream_id,
        "sequence": event_sequence(args.event_type, args.task_index),
        "task_index": args.task_index,
        "workspace": str(args.workspace),
        "task_file": str(args.task_file) if args.task_file else "",
        "total_models": args.total_models,
        "batch_started_at": args.batch_started_at,
        "target": args.target,
        "model": args.model,
        "vendor": summary.vendor_from_target(args.target or ""),
        "outcome": outcome,
        "exit_code": args.exit_code,
        "elapsed_seconds": args.elapsed_seconds,
        "batch_elapsed_seconds": args.batch_elapsed_seconds,
        "run_started_at": args.run_started_at or args.started_at,
        "run_ended_at": args.run_ended_at,
        "processed": args.processed,
        "passed": args.passed,
        "failed": args.failed,
        "skipped": args.skipped,
        "attempts": 0,
        "emitted_at": now_iso(),
    }
    if args.task_file:
        try:
            event["task_file_content"] = args.task_file.read_text(encoding="utf-8")
        except OSError:
            pass

    stream = safe_component(stream_id)
    filename = event_filename(event)
    queue_path = root / "queue" / stream / filename
    atomic_write_json(queue_path, event)

    mode = os.environ.get("FLAGOS_PROGRESS_WORKER_MODE", "after-batch").strip().lower()
    if mode not in {"live", "after-batch", "external"}:
        mode = "after-batch"
    synchronous = os.environ.get("FLAGOS_PROGRESS_SYNC_WORKER", "0") == "1"
    should_start = mode == "live" or args.event_type in {"batch-start", "batch-end", "single-start", "single-finish"}
    start_only = mode == "after-batch" and args.event_type == "batch-start"
    if mode != "external" and should_start:
        if synchronous:
            return run_worker(root, stream_id, start_only=start_only)
        spawn_worker(root, stream_id, start_only=start_only)
    return 0


def command_worker(args: argparse.Namespace) -> int:
    return run_worker(progress_root(args.progress_root), args.batch_id, start_only=args.start_only)


def command_retry_dead_letter(args: argparse.Namespace) -> int:
    root = progress_root(args.progress_root)
    stream = safe_component(args.batch_id)
    dead_dir = root / "dead-letter" / stream
    queue_dir = root / "queue" / stream
    queue_dir.mkdir(parents=True, exist_ok=True)
    for dead in sorted(dead_dir.glob("*.json")) if dead_dir.is_dir() else []:
        event = load_json(dead)
        event["attempts"] = 0
        event.pop("last_error", None)
        event.pop("last_failed_at", None)
        atomic_write_json(dead, event)
        os.replace(dead, queue_dir / dead.name)
    if os.environ.get("FLAGOS_PROGRESS_SYNC_WORKER", "0") == "1":
        return run_worker(root, args.batch_id)
    spawn_worker(root, args.batch_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlagRelease detached progress event queue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit")
    emit.add_argument("event_type", choices=sorted(EVENT_TYPES))
    emit.add_argument("--progress-root", type=Path)
    emit.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    emit.add_argument("--batch-id", required=True)
    emit.add_argument("--task-file", type=Path)
    emit.add_argument("--total-models", type=int)
    emit.add_argument("--batch-started-at", type=int)
    emit.add_argument("--task-index", type=int)
    emit.add_argument("--target", default="")
    emit.add_argument("--model", default="")
    emit.add_argument("--outcome", choices=sorted(TERMINAL_OUTCOMES))
    emit.add_argument("--exit-code", type=int)
    emit.add_argument("--elapsed-seconds", type=int)
    emit.add_argument("--batch-elapsed-seconds", type=int)
    emit.add_argument("--run-started-at", type=int)
    emit.add_argument("--run-ended-at", type=int)
    emit.add_argument("--started-at", type=int)
    emit.add_argument("--processed", type=int)
    emit.add_argument("--passed", type=int)
    emit.add_argument("--failed", type=int)
    emit.add_argument("--skipped", type=int)
    emit.set_defaults(handler=command_emit)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--progress-root", type=Path)
    worker.add_argument("--batch-id", required=True)
    worker.add_argument("--start-only", action="store_true")
    worker.set_defaults(handler=command_worker)

    retry = subparsers.add_parser("retry-dead-letter")
    retry.add_argument("--progress-root", type=Path)
    retry.add_argument("--batch-id", required=True)
    retry.set_defaults(handler=command_retry_dead_letter)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"progress worker error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
