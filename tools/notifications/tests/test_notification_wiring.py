import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FEISHU_NOTIFY = REPO_ROOT / "tools" / "notifications" / "feishu_notify.py"
RUN_BATCH = REPO_ROOT / "prompts" / "run_batch.sh"
RUN_PIPELINE = REPO_ROOT / "prompts" / "run_pipeline.sh"
SUMMARIZE = REPO_ROOT / "tools" / "batch_summarize" / "summarize.sh"
PROGRESS_RUNNER = REPO_ROOT / "tools" / "notifications" / "progress_runner.sh"
PROGRESS_WORKER = REPO_ROOT / "tools" / "notifications" / "progress_worker.py"
PROGRESS_SUMMARY = REPO_ROOT / "tools" / "notifications" / "progress_summary.py"


class FeishuNotifyFailureTests(unittest.TestCase):
    def test_failed_curl_is_retried_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            attempts = tmp / "attempts.log"
            curl = tmp / "curl"
            curl.write_text(
                "#!/bin/sh\n"
                f"echo attempt >> {attempts}\n"
                "echo simulated-network-error >&2\n"
                "exit 7\n",
                encoding="utf-8",
            )
            curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
            env = os.environ.copy()
            env["PATH"] = f"{tmp}{os.pathsep}{env.get('PATH', '')}"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(FEISHU_NOTIFY),
                    "--event",
                    "result_summary",
                    "--title",
                    "测试",
                    "--text",
                    "结果汇总",
                    "--webhook-url",
                    "https://example.invalid/hook",
                ],
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("飞书通知发送失败", completed.stderr)
            self.assertEqual(attempts.read_text(encoding="utf-8").count("attempt"), 3)

    def test_dry_run_injects_required_keyword(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(FEISHU_NOTIFY),
                "--event",
                "task_end",
                "--title",
                "metax｜Qwen3-8B",
                "--lead",
                "✅ V3 Max 达标上传",
                "--field",
                "总耗时=2h18m",
                "--field",
                "总费用=$4.26",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        title = payload["card"]["header"]["title"]["content"]
        self.assertIn("结果汇总", title)
        self.assertNotIn("模型完成", title)
        elements = payload["card"]["body"]["elements"]
        self.assertTrue(any(element.get("tag") == "column_set" for element in elements))
        body = json.dumps(elements, ensure_ascii=False)
        self.assertIn("✅ V3 Max 达标上传", body)
        self.assertNotIn("时间：", body)
        self.assertNotIn("状态：", body)


class NotificationWiringTests(unittest.TestCase):
    def test_batch_runs_pipeline_directly_and_only_emits_detached_events(self):
        batch = RUN_BATCH.read_text(encoding="utf-8")
        self.assertIn("timeout --signal=TERM --kill-after=60", batch)
        self.assertIn("bash prompts/run_pipeline.sh", batch)
        self.assertIn("FLAGOS_BATCH_MODE=1", batch)
        self.assertIn("progress_emit_detached model-finish", batch)
        self.assertIn("progress_emit_detached model-start", batch)
        self.assertNotIn("run_progress run-model", batch)
        self.assertNotIn("run_progress_safe", batch)
        self.assertNotIn("--state-file", batch)

    def test_batch_emits_model_start_before_pipeline_invocation(self):
        batch = RUN_BATCH.read_text(encoding="utf-8")
        start_pos = batch.index("progress_emit_detached model-start")
        invoke_pos = batch.index("bash prompts/run_pipeline.sh")
        self.assertLess(start_pos, invoke_pos)

    def test_batch_defaults_worker_mode_live(self):
        batch = RUN_BATCH.read_text(encoding="utf-8")
        self.assertIn(
            'export FLAGOS_PROGRESS_WORKER_MODE="${FLAGOS_PROGRESS_WORKER_MODE:-live}"',
            batch,
        )

    def test_batch_preserves_original_archive_checkpoint_and_stop_order(self):
        batch = RUN_BATCH.read_text(encoding="utf-8")
        loop_start = batch.index("# ========== 逐个执行 ==========")
        loop_end = batch.index('done < "$TASK_FILE"', loop_start)
        loop = batch[loop_start:loop_end]
        self.assertLess(loop.index("# 归档旧数据"), loop.index("# 断点续跑"))
        self.assertNotIn("STOP_AFTER_REPORT", batch)
        self.assertEqual(loop.count("if $STOP_ON_ERROR; then"), 2)
        self.assertEqual(loop.count("⚠ --stop-on-error 已启用，终止批量执行"), 2)

    def test_pipeline_has_no_self_wrapper_or_added_timeout(self):
        pipeline = RUN_PIPELINE.read_text(encoding="utf-8")
        self.assertIn("progress_emit_detached single-start", pipeline)
        self.assertIn("progress_emit_detached single-finish", pipeline)
        self.assertNotIn("exec \"${PROGRESS_RUNNER}\"", pipeline)
        self.assertNotIn("single-run", pipeline)
        self.assertNotIn("FLAGOS_PROGRESS_WRAPPED", pipeline)
        self.assertNotIn("FLAGOS_SINGLE_MODEL_TIMEOUT_SECONDS", pipeline)

    def test_exit_cleanup_order_is_upload_then_gpu_cleanup_then_emit(self):
        pipeline = RUN_PIPELINE.read_text(encoding="utf-8")
        function = pipeline[pipeline.index("pipeline_on_exit()") : pipeline.index("trap 'pipeline_on_exit")]
        self.assertLess(function.index("upload_to_platform_on_exit"), function.index("cleanup_gpu_services"))
        self.assertLess(function.index("cleanup_gpu_services"), function.index("progress_emit_detached single-finish"))

    def test_runner_and_worker_never_control_pipeline(self):
        runner = PROGRESS_RUNNER.read_text(encoding="utf-8")
        worker = PROGRESS_WORKER.read_text(encoding="utf-8")
        combined = runner + worker
        for token in (
            "run_pipeline.sh",
            "run-model",
            "single-run",
            "ACTIVE_PID",
            "forward_signal",
            "os.kill(",
            ".wait(",
        ):
            self.assertNotIn(token, combined)
        self.assertNotIn("timeout --signal", combined)

    def test_sidecar_never_writes_model_report_paths_or_rescans_model_outputs(self):
        worker = PROGRESS_WORKER.read_text(encoding="utf-8")
        summary = PROGRESS_SUMMARY.read_text(encoding="utf-8")
        for token in ("progress_result.json", '"reports"', "'/reports/'", '"/reports/"'):
            self.assertNotIn(token, worker)
        for token in (
            "context_final.yaml",
            "context_snapshot.yaml",
            "detect_gpu.py",
            "def read_context",
            "def collect_cost",
            "def inspect_model",
            "def delivery_result",
        ):
            self.assertNotIn(token, summary)

    def test_detached_hooks_close_standard_fds_and_never_wait(self):
        for path in (RUN_BATCH, RUN_PIPELINE):
            text = path.read_text(encoding="utf-8")
            start = text.index("progress_emit_detached()")
            function = text[start : text.index("\n}", start) + 2]
            self.assertIn("</dev/null", function)
            self.assertIn(">/dev/null", function)
            self.assertIn("2>&1 &", function)
            self.assertIn("return 0", function)
            self.assertNotIn("wait", function)
            self.assertNotIn("test -x", function)
            self.assertNotIn("mkdir", function)

    def test_detailed_summarizer_has_no_feishu_integration(self):
        text = SUMMARIZE.read_text(encoding="utf-8")
        for token in ("--notify", "--notify-title", "--notify-max-chars", "feishu_notify.py"):
            self.assertNotIn(token, text)

    def test_detached_hook_preserves_all_business_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            runner = tmp / "runner"
            runner.write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
            runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
            for exit_code in (0, 1, 7, 124, 130, 143):
                script = (
                    "progress_emit_detached() {\n"
                    f"  nohup '{runner}' emit \"$@\" </dev/null >/dev/null 2>&1 &\n"
                    "  return 0\n"
                    "}\n"
                    "progress_emit_detached model-finish || :\n"
                    f"exit {exit_code}\n"
                )
                completed = subprocess.run(["bash", "-c", script], check=False, timeout=2)
                self.assertEqual(completed.returncode, exit_code)

    def test_pipeline_exit_trap_preserves_status_even_when_sidecar_fails(self):
        pipeline = RUN_PIPELINE.read_text(encoding="utf-8")
        start = pipeline.index("pipeline_on_exit()")
        end = pipeline.index("\n}\ntrap 'pipeline_on_exit", start) + 3
        function = pipeline[start:end]
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace = Path(tmp_dir) / "trace.log"
            for exit_code in (0, 1, 7, 124, 130, 143):
                trace.unlink(missing_ok=True)
                script = (
                    f"TRACE='{trace}'\n"
                    "upload_to_platform_on_exit() { echo upload >> \"$TRACE\"; return 31; }\n"
                    "cleanup_gpu_services() { echo cleanup >> \"$TRACE\"; return 32; }\n"
                    "progress_emit_detached() { echo emit >> \"$TRACE\"; return 33; }\n"
                    "FLAGOS_BATCH_MODE=0\n"
                    "PIPELINE_START_TS=$(date +%s)\n"
                    "FLAGOS_PROGRESS_RUN_ID=test\n"
                    "FLAGOS_WORKSPACE=/tmp\n"
                    "TARGET=target\n"
                    "MODEL=model\n"
                    f"{function}\n"
                    "trap 'pipeline_on_exit \"$?\"' EXIT\n"
                    f"exit {exit_code}\n"
                )
                completed = subprocess.run(["bash", "-c", script], check=False, timeout=2)
                self.assertEqual(completed.returncode, exit_code)
                self.assertEqual(trace.read_text(encoding="utf-8").splitlines(), ["upload", "cleanup", "emit"])

    def test_missing_unexecutable_broken_and_hanging_runner_are_nonblocking(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cases = {}
            cases["missing"] = tmp / "missing"

            unexecutable = tmp / "unexecutable"
            unexecutable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cases["unexecutable"] = unexecutable

            broken = tmp / "broken"
            broken.write_text("#!/bin/sh\nif then\n", encoding="utf-8")
            broken.chmod(broken.stat().st_mode | stat.S_IXUSR)
            cases["broken"] = broken

            hanging = tmp / "hanging"
            hanging.write_text("#!/bin/sh\nsleep 0.5\n", encoding="utf-8")
            hanging.chmod(hanging.stat().st_mode | stat.S_IXUSR)
            cases["hanging"] = hanging

            for name, runner in cases.items():
                script = (
                    "progress_emit_detached() {\n"
                    f"  nohup '{runner}' emit \"$@\" </dev/null >/dev/null 2>&1 &\n"
                    "  return 0\n"
                    "}\n"
                    "progress_emit_detached task || :\n"
                    "exit 7\n"
                )
                started = time.monotonic()
                completed = subprocess.run(["bash", "-c", script], check=False, timeout=2)
                elapsed = time.monotonic() - started
                self.assertEqual(completed.returncode, 7, name)
                self.assertLess(elapsed, 0.4, name)


if __name__ == "__main__":
    unittest.main()
