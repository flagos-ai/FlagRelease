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


if __name__ == "__main__":
    unittest.main()
