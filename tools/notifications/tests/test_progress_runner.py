import importlib.util
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "tools" / "notifications" / "progress_runner.sh"
WORKER_PY = REPO_ROOT / "tools" / "notifications" / "progress_worker.py"


def _load_worker_module():
    import sys
    pkg_dir = str(WORKER_PY.parent)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    spec = importlib.util.spec_from_file_location("progress_worker_undertest", WORKER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pw = _load_worker_module()


class ProgressQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.workspace = self.base / "workspace"
        self.progress_root = self.base / "progress"
        self.workspace.mkdir()
        (self.workspace / "model-a" / "logs").mkdir(parents=True)
        self.task_file = self.base / "tasks.txt"
        self.task_file.write_text("repo/metax-model:v1 | model-a\n", encoding="utf-8")
        self.claude = self.base / "claude"
        self.write_claude(success=True)
        self.env = os.environ.copy()
        self.env.pop("FEISHU_WEBHOOK_URL", None)
        self.env.update(
            {
                "CLAUDE_COMMAND": str(self.claude),
                "FLAGOS_PROGRESS_ROOT": str(self.progress_root),
                "FLAGOS_PROGRESS_NOTIFY": "0",
                "FLAGOS_PROGRESS_SYNC_WORKER": "1",
                "FLAGOS_PROGRESS_WORKER_MODE": "live",
                "FLAGOS_MODEL_ANALYSIS_TIMEOUT_SECONDS": "5",
                "FLAGOS_PROGRESS_END_SETTLE_SECONDS": "0",
                "FLAGOS_PROGRESS_END_MAX_WAIT_SECONDS": "0",
            }
        )

    def tearDown(self):
        self.tmp.cleanup()

    def write_claude(self, success=True, marker=None, delay_seconds=0):
        if success:
            structured = {
                "cost": {"migration_cost_usd": 4.26, "complete": True},
                "delivery": {"version": "v3", "accuracy_ok": True, "uploaded": True},
                "notification": {
                    "result_label": "达标上传",
                    "lead": "✅ V3 Max 达标上传",
                    "summary": "精度达标且已上传",
                    "warning": None,
                },
                "evidence": {"accuracy": [], "upload": [], "cost": ["logs/seg1_cost.txt"]},
            }
            envelope = json.dumps({"structured_output": structured, "total_cost_usd": 0.08})
            marker_line = f"echo called >> '{marker}'\n" if marker else ""
            delay_line = f"sleep {delay_seconds}\n" if delay_seconds else ""
            text = f"#!/bin/sh\n{marker_line}{delay_line}printf '%s\\n' '{envelope}'\n"
        else:
            text = "#!/bin/sh\necho mock-claude-failure >&2\nexit 9\n"
        self.claude.write_text(text, encoding="utf-8")
        self.claude.chmod(self.claude.stat().st_mode | stat.S_IXUSR)

    def run_runner(self, *args, expected=0, env=None):
        completed = subprocess.run(
            [str(RUNNER), *map(str, args)],
            cwd=REPO_ROOT,
            env=env or self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(completed.returncode, expected, completed.stderr + completed.stdout)
        return completed

    def emit_batch_start(self, batch_id="test"):
        return self.run_runner(
            "emit",
            "batch-start",
            "--batch-id",
            batch_id,
            "--workspace",
            self.workspace,
            "--task-file",
            self.task_file,
            "--total-models",
            "1",
            "--batch-started-at",
            "1000",
        )

    def emit_model(self, batch_id="test", outcome="success", exit_code=0):
        return self.run_runner(
            "emit",
            "model-finish",
            "--batch-id",
            batch_id,
            "--workspace",
            self.workspace,
            "--task-index",
            "1",
            "--total-models",
            "1",
            "--target",
            "repo/metax-model:v1",
            "--model",
            "model-a",
            "--outcome",
            outcome,
            "--exit-code",
            str(exit_code),
            "--elapsed-seconds",
            "42",
            "--batch-elapsed-seconds",
            "50",
            "--run-ended-at",
            "1050",
        )

    def emit_model_start(self, batch_id="test"):
        return self.run_runner(
            "emit",
            "model-start",
            "--batch-id",
            batch_id,
            "--workspace",
            self.workspace,
            "--task-index",
            "1",
            "--total-models",
            "1",
            "--target",
            "repo/metax-model:v1",
            "--model",
            "model-a",
            "--run-started-at",
            "1005",
        )

    def result_path(self, batch_id="test"):
        return self.progress_root / "results" / batch_id / "000001_model-a.json"

    def state_path(self, batch_id="test"):
        return self.progress_root / "states" / f"batch_{batch_id}_progress.json"

    def test_live_worker_analyzes_to_sidecar_and_updates_state(self):
        self.emit_batch_start()
        self.emit_model(exit_code=0)
        result = json.loads(self.result_path().read_text(encoding="utf-8"))
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        self.assertEqual(result["pipeline"]["exit_code"], 0)
        self.assertEqual(result["vendor"], "metax")
        self.assertEqual(state["models"][0]["elapsed_seconds"], 42)
        self.assertEqual(state["batch_elapsed_seconds"], 50)
        self.assertFalse((self.workspace / "model-a" / "reports" / "progress_result.json").exists())

    def test_analysis_failure_is_contained_in_failed_result(self):
        self.emit_batch_start()
        self.write_claude(success=False)
        self.emit_model(outcome="success", exit_code=0)
        result = json.loads(self.result_path().read_text(encoding="utf-8"))
        self.assertEqual(result["analysis_status"], "failed")
        self.assertEqual(result["pipeline"]["outcome"], "success")
        self.assertEqual(result["pipeline"]["exit_code"], 0)
        self.assertIsNone(result["cost"]["migration_cost_usd"])

    def test_model_start_updates_current_model_in_live_mode(self):
        self.emit_batch_start()
        self.emit_model_start()
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        self.assertIsNotNone(state.get("current_model"))
        self.assertEqual(state["current_model"]["model"], "model-a")
        processed = list((self.progress_root / "processed" / "test").glob("*model-start*.json"))
        self.assertEqual(len(processed), 1)

    def test_after_batch_mode_notifies_model_start_but_defers_finish(self):
        self.env["FLAGOS_PROGRESS_WORKER_MODE"] = "after-batch"
        self.emit_batch_start()
        self.emit_model_start()
        # model-start 即时消费（进 processed，state 更新），model-finish 仍延后
        self.assertTrue(
            list((self.progress_root / "processed" / "test").glob("*model-start*.json"))
        )
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        self.assertEqual(state["current_model"]["model"], "model-a")
        self.emit_model()
        self.assertFalse(self.result_path().exists())
        self.assertTrue(
            list((self.progress_root / "queue" / "test").glob("*model-finish*.json"))
        )

    def test_after_batch_mode_queues_models_until_batch_end(self):
        self.env["FLAGOS_PROGRESS_WORKER_MODE"] = "after-batch"
        self.emit_batch_start()
        self.emit_model()
        self.assertFalse(self.result_path().exists())
        self.run_runner(
            "emit",
            "batch-end",
            "--batch-id",
            "test",
            "--workspace",
            self.workspace,
            "--exit-code",
            "0",
            "--elapsed-seconds",
            "60",
            "--processed",
            "1",
        )
        self.assertTrue(self.result_path().is_file())
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        self.assertEqual(len(state["models"]), 1)
        self.assertEqual(state["batch_elapsed_seconds"], 60)

    def test_live_emit_does_not_wait_for_slow_claude_worker(self):
        self.emit_batch_start()
        self.write_claude(success=True, delay_seconds=1.2)
        self.env["FLAGOS_PROGRESS_SYNC_WORKER"] = "0"
        started = time.monotonic()
        self.emit_model()
        emit_elapsed = time.monotonic() - started
        self.assertLess(emit_elapsed, 0.6)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not self.result_path().is_file():
            time.sleep(0.05)
        self.assertTrue(self.result_path().is_file())

    def test_external_mode_only_writes_queue_until_worker_is_run(self):
        self.env["FLAGOS_PROGRESS_WORKER_MODE"] = "external"
        self.emit_batch_start()
        self.assertFalse(self.state_path().exists())
        self.assertEqual(len(list((self.progress_root / "queue" / "test").glob("*.json"))), 1)
        self.run_runner("worker", "--batch-id", "test")
        self.assertTrue(self.state_path().is_file())

    def test_duplicate_event_is_deduplicated_without_second_analysis(self):
        marker = self.base / "claude-called"
        self.write_claude(success=True, marker=marker)
        self.emit_batch_start()
        self.emit_model()
        self.emit_model()
        self.assertEqual(marker.read_text(encoding="utf-8").splitlines(), ["called"])
        state = json.loads(self.state_path().read_text(encoding="utf-8"))
        self.assertEqual(len(state["models"]), 1)

    def test_skip_reuses_sidecar_from_another_batch(self):
        self.emit_batch_start("first")
        self.emit_model("first")
        marker = self.base / "claude-called-again"
        self.write_claude(success=True, marker=marker)
        self.emit_batch_start("second")
        self.run_runner(
            "emit",
            "skip-model",
            "--batch-id",
            "second",
            "--workspace",
            self.workspace,
            "--task-index",
            "1",
            "--total-models",
            "1",
            "--target",
            "repo/metax-model:v1",
            "--model",
            "model-a",
        )
        self.assertFalse(marker.exists())
        state = json.loads(self.state_path("second").read_text(encoding="utf-8"))
        self.assertEqual(state["models"][0]["outcome"], "skipped")

    def test_invalid_event_moves_to_dead_letter_at_retry_limit(self):
        self.env["FLAGOS_PROGRESS_MAX_ATTEMPTS"] = "1"
        missing = self.base / "missing-tasks.txt"
        self.run_runner(
            "emit",
            "batch-start",
            "--batch-id",
            "broken",
            "--workspace",
            self.workspace,
            "--task-file",
            missing,
            "--total-models",
            "1",
            expected=1,
        )
        dead = list((self.progress_root / "dead-letter" / "broken").glob("*.json"))
        self.assertEqual(len(dead), 1)
        event = json.loads(dead[0].read_text(encoding="utf-8"))
        self.assertEqual(event["attempts"], 1)

    def test_worker_recovers_event_left_in_processing_after_crash(self):
        self.env["FLAGOS_PROGRESS_WORKER_MODE"] = "external"
        self.emit_batch_start("recover")
        queued = next((self.progress_root / "queue" / "recover").glob("*.json"))
        processing_dir = self.progress_root / "processing" / "recover"
        processing_dir.mkdir(parents=True)
        queued.rename(processing_dir / queued.name)
        self.run_runner("worker", "--batch-id", "recover")
        self.assertTrue(self.state_path("recover").is_file())
        self.assertFalse(list(processing_dir.glob("*.json")))

    def test_batch_end_settles_and_rescans_for_late_model_event(self):
        self.env["FLAGOS_PROGRESS_WORKER_MODE"] = "external"
        self.emit_batch_start("race")
        self.run_runner("worker", "--batch-id", "race")
        self.run_runner(
            "emit",
            "batch-end",
            "--batch-id",
            "race",
            "--workspace",
            self.workspace,
            "--exit-code",
            "0",
            "--elapsed-seconds",
            "60",
            "--processed",
            "1",
        )
        worker_env = self.env.copy()
        worker_env["FLAGOS_PROGRESS_END_SETTLE_SECONDS"] = "0.5"
        worker_env["FLAGOS_PROGRESS_END_MAX_WAIT_SECONDS"] = "2"

        def emit_late_model():
            time.sleep(0.15)
            self.emit_model("race")

        thread = threading.Thread(target=emit_late_model)
        thread.start()
        self.run_runner("worker", "--batch-id", "race", env=worker_env)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        state = json.loads(self.state_path("race").read_text(encoding="utf-8"))
        self.assertEqual(len(state["models"]), 1)
        self.assertIsNotNone(state["ended_at"])

    def test_late_batch_start_unblocks_already_queued_model_and_end(self):
        self.env["FLAGOS_PROGRESS_WORKER_MODE"] = "after-batch"
        self.emit_model("late-start")
        self.run_runner(
            "emit",
            "batch-end",
            "--batch-id",
            "late-start",
            "--workspace",
            self.workspace,
            "--exit-code",
            "0",
            "--elapsed-seconds",
            "60",
            "--processed",
            "1",
        )
        self.assertFalse(self.state_path("late-start").exists())
        self.emit_batch_start("late-start")
        state = json.loads(self.state_path("late-start").read_text(encoding="utf-8"))
        self.assertEqual(len(state["models"]), 1)
        self.assertIsNotNone(state["ended_at"])


class EventSequenceInvariantTests(unittest.TestCase):
    """model-start 的排序号必须夹在上一个模型的 finish 与自身 finish 之间。"""

    def test_model_start_sits_between_previous_and_own_finish(self):
        for idx in range(1, 6):
            start = pw.event_sequence("model-start", idx)
            own_finish = pw.event_sequence("model-finish", idx)
            self.assertLess(start, own_finish, f"model{idx} start 应早于自身 finish")
            if idx > 1:
                prev_finish = pw.event_sequence("model-finish", idx - 1)
                self.assertLess(prev_finish, start, f"model{idx-1} finish 应早于 model{idx} start")

    def test_full_batch_sequence_is_strictly_interleaved(self):
        events = [("batch-start", None)]
        for idx in range(1, 4):
            events.append(("model-start", idx))
            events.append(("model-finish", idx))
        ordered = sorted(events, key=lambda e: pw.event_sequence(e[0], e[1]))
        # 期望：batch-start → (start1, finish1) → (start2, finish2) → (start3, finish3)
        self.assertEqual(
            ordered,
            [
                ("batch-start", None),
                ("model-start", 1), ("model-finish", 1),
                ("model-start", 2), ("model-finish", 2),
                ("model-start", 3), ("model-finish", 3),
            ],
        )


class OutOfOrderArrivalTests(unittest.TestCase):
    """慢分析导致的乱序落盘：worker 仍按 sequence 处理，不因落盘时刻颠倒。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.workspace = self.base / "workspace"
        self.progress_root = self.base / "progress"
        for name in ("model-a", "model-b"):
            (self.workspace / name / "logs").mkdir(parents=True)
        self.task_file = self.base / "tasks.txt"
        self.task_file.write_text(
            "repo/metax-a:v1 | model-a\nrepo/metax-b:v1 | model-b\n", encoding="utf-8"
        )
        self.claude = self.base / "claude"
        envelope = json.dumps({
            "structured_output": {
                "cost": {"migration_cost_usd": 1.0, "complete": True},
                "delivery": {"version": "v3", "accuracy_ok": True, "uploaded": True},
                "notification": {"result_label": "达标上传", "lead": "✅", "summary": "ok", "warning": None},
                "evidence": {"accuracy": [], "upload": [], "cost": []},
            },
            "total_cost_usd": 0.01,
        })
        self.claude.write_text(f"#!/bin/sh\nprintf '%s\\n' '{envelope}'\n", encoding="utf-8")
        self.claude.chmod(self.claude.stat().st_mode | stat.S_IXUSR)
        self.env = os.environ.copy()
        self.env.pop("FEISHU_WEBHOOK_URL", None)
        self.env.update({
            "CLAUDE_COMMAND": str(self.claude),
            "FLAGOS_PROGRESS_ROOT": str(self.progress_root),
            "FLAGOS_PROGRESS_NOTIFY": "0",
            "FLAGOS_PROGRESS_SYNC_WORKER": "1",
            "FLAGOS_PROGRESS_WORKER_MODE": "external",  # 只入队，稍后统一起 worker
            "FLAGOS_MODEL_ANALYSIS_TIMEOUT_SECONDS": "5",
            "FLAGOS_PROGRESS_END_SETTLE_SECONDS": "0",
            "FLAGOS_PROGRESS_END_MAX_WAIT_SECONDS": "0",
        })

    def tearDown(self):
        self.tmp.cleanup()

    def _emit(self, *args):
        subprocess.run([str(RUNNER), "emit", *map(str, args)], cwd=REPO_ROOT,
                       env=self.env, capture_output=True, text=True, timeout=15, check=True)

    def test_late_finish_still_processed_before_next_model_start(self):
        bid = "ooo"
        self._emit("batch-start", "--batch-id", bid, "--workspace", self.workspace,
                   "--task-file", self.task_file, "--total-models", "2", "--batch-started-at", "1000")
        # 模拟：model-a 的 finish 分析慢，导致 model-b 的 start 先落盘
        self._emit("model-start", "--batch-id", bid, "--workspace", self.workspace,
                   "--task-index", "2", "--total-models", "2", "--target", "repo/metax-b:v1",
                   "--model", "model-b", "--run-started-at", "1200")
        self._emit("model-finish", "--batch-id", bid, "--workspace", self.workspace,
                   "--task-index", "1", "--total-models", "2", "--target", "repo/metax-a:v1",
                   "--model", "model-a", "--outcome", "success", "--exit-code", "0",
                   "--elapsed-seconds", "100", "--run-ended-at", "1150")
        # 起 worker 统一消费
        subprocess.run([str(RUNNER), "worker", "--batch-id", bid], cwd=REPO_ROOT,
                       env=self.env, capture_output=True, text=True, timeout=30, check=True)
        processed = sorted((self.progress_root / "processed" / bid).glob("*.json"))
        names = [p.name for p in processed]
        # 文件名按 sequence 前缀排序：model-a finish(020) 必在 model-b start(015)... 校验实际处理顺序号
        def seq(n):
            return int(n.split("_", 1)[0])
        order = [(seq(n), n.split("_")[1]) for n in names]
        # model-a finish (task1 → 10) 必排在 model-b start (task2 → 15) 之前，
        # 即使 model-b start 先落盘。这就是"慢分析不颠倒顺序"的核心保证。
        self.assertEqual(
            order,
            [(0, "batch-start"), (10, "model-finish"), (15, "model-start")],
        )


if __name__ == "__main__":
    unittest.main()
