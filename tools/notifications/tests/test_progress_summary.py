import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "progress_summary.py"
SPEC = importlib.util.spec_from_file_location("progress_summary", MODULE_PATH)
ps = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ps)


class ProgressSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_result(
        self,
        model,
        *,
        target="",
        vendor="unknown",
        outcome="success",
        exit_code=0,
        elapsed_seconds=0,
        cost_usd=None,
        version=None,
        accuracy_ok=None,
        uploaded=None,
        analysis_status="success",
    ):
        path = self.workspace / "progress-results" / f"{model}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        label = "V5" if version == "v5" else "V3 Max" if version == "v3" else None
        data = {
            "schema_version": 1,
            "analysis_status": analysis_status,
            "analysis_error": None if analysis_status == "success" else "mock failure",
            "model": model,
            "target": target,
            "vendor": vendor,
            "pipeline": {
                "outcome": outcome,
                "exit_code": exit_code,
                "elapsed_seconds": elapsed_seconds,
            },
            "cost": {"migration_cost_usd": cost_usd, "complete": cost_usd is not None},
            "delivery": {
                "version": version,
                "label": label,
                "accuracy_ok": accuracy_ok,
                "uploaded": uploaded,
                "qualified_uploaded": accuracy_ok is True and uploaded is True,
            },
            "notification": {
                "result_label": "达标上传" if accuracy_ok is True and uploaded is True else "无法确认",
                "lead": "mock",
                "summary": None,
                "warning": None,
            },
            "evidence": {"accuracy": [], "upload": [], "cost": []},
            "analysis": {"elapsed_seconds": 2, "cost_usd": 0.01},
            "analyzed_at": "2026-07-22T12:00:00+08:00",
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_cli(self, *args):
        env = os.environ.copy()
        env.pop("FEISHU_WEBHOOK_URL", None)
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), *map(str, args)],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def card_text(self, payload):
        parts = []

        def collect(value):
            if isinstance(value, dict):
                if value.get("tag") == "markdown":
                    parts.append(str(value.get("content") or ""))
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(payload["card"]["body"]["elements"])
        return "\n".join(parts)

    def card_title(self, payload):
        return payload["card"]["header"]["title"]["content"]


    def test_batch_elapsed_uses_updated_time_and_freezes_at_end(self):
        running = {
            "started_at": "2026-07-22T08:00:00+08:00",
            "updated_at": "2026-07-22T09:01:01+08:00",
        }
        self.assertEqual(ps.batch_elapsed_seconds(running), 3661)
        self.assertEqual(ps.format_duration(ps.batch_elapsed_seconds(running)), "1h01m")

        finished = {
            **running,
            "updated_at": "2026-07-22T10:00:00+08:00",
            "ended_at": "2026-07-22T09:30:00+08:00",
        }
        self.assertEqual(ps.batch_elapsed_seconds(finished), 5400)

    def test_explicit_batch_elapsed_excludes_delayed_worker_time(self):
        state = {
            "batch_elapsed_seconds": 125,
            "started_at": "2026-07-22T08:00:00+08:00",
            "updated_at": "2026-07-22T12:00:00+08:00",
        }
        self.assertEqual(ps.batch_elapsed_seconds(state), 125)
        self.assertEqual(ps.format_duration(ps.batch_elapsed_seconds(state)), "2m05s")

    def test_success_average_excludes_failed_timeout_and_missing_values(self):
        state = {
            "total_models": 5,
            "models": [
                {"outcome": "success", "elapsed_seconds": 100, "cost_usd": 2, "qualified_uploaded": True, "delivery_version": "v5"},
                {"outcome": "success", "elapsed_seconds": 300, "cost_usd": None, "qualified_uploaded": False, "delivery_version": "v3"},
                {"outcome": "failed", "elapsed_seconds": 1000, "cost_usd": 20, "qualified_uploaded": False, "delivery_version": "v3"},
                {"outcome": "timeout", "elapsed_seconds": 2000, "cost_usd": 30, "qualified_uploaded": True, "delivery_version": "v3"},
                {"outcome": "skipped", "elapsed_seconds": None, "cost_usd": None, "qualified_uploaded": False, "delivery_version": "v3"},
            ],
        }
        metrics = ps.batch_metrics(state)
        self.assertEqual(metrics["average_elapsed_seconds"], 200)
        self.assertEqual(metrics["average_cost_usd"], 2)
        self.assertEqual(metrics["processed_models"], 5)
        self.assertEqual(metrics["qualified_uploaded"], 2)
        self.assertEqual(metrics["success_rate_pct"], 40)
        self.assertEqual(metrics["total_cost_usd"], 52)

    def test_batch_total_cost_is_unknown_when_all_costs_are_missing(self):
        metrics = ps.batch_metrics(
            {
                "total_models": 2,
                "models": [
                    {"outcome": "success", "cost_usd": None, "qualified_uploaded": False},
                    {"outcome": "failed", "qualified_uploaded": False},
                ],
            }
        )
        self.assertIsNone(metrics["total_cost_usd"])
        self.assertEqual(ps.format_cost(metrics["total_cost_usd"]), "未知")

    def test_mixed_vendor_is_derived_from_task_and_result_snapshots(self):
        state = {"initial_vendor": "metax", "models": [{"vendor": "metax"}, {"vendor": "hygon"}]}
        ps.update_batch_vendor(state)
        self.assertEqual(state["vendor"], "mixed")

    def test_target_vendor_aliases_boundaries_and_ambiguity(self):
        cases = {
            "registry/NVIDIA/model:v1": "nvidia",
            "repo/ascend_model:v1": "huawei",
            "repo/hygon-dcu:model": "hygon",
            "repo/METAX/model:v1": "metax",
            "repo/cambricon-mlu:v1": "cambricon",
            "repo/musa.mtt:v1": "mthreads",
            "repo/kunlun-xpu:v1": "kunlunxin",
            "repo/tianshu:model": "tianshu",
            "repo/notmetaxmodel:v1": "unknown",
            "repo/metax-hygon:model": "unknown",
        }
        for target, expected in cases.items():
            with self.subTest(target=target):
                self.assertEqual(ps.vendor_from_target(target), expected)

    def test_result_validation_rejects_unknown_delivery_version(self):
        result_file = self.write_result(
            "bad-version",
            target="registry/hygon/model:latest",
            vendor="hygon",
            version="v4",
            accuracy_ok=True,
            uploaded=True,
        )
        with self.assertRaisesRegex(RuntimeError, "delivery.version"):
            ps.model_from_result_file(result_file)

    def test_batch_vendor_comes_only_from_task_targets(self):
        task_file = self.workspace / "tasks.txt"
        task_file.write_text(
            "repo/metax/model:v1 | model-a\nrepo/unknown/model:v1 | model-b\n",
            encoding="utf-8",
        )
        state_file = self.workspace / "batch_progress.json"
        state = self.run_cli(
            "batch-start", "--workspace", self.workspace, "--state-file", state_file,
            "--batch-id", "batch", "--task-file", task_file, "--vendor", "nvidia",
        )
        self.assertEqual(state["vendor"], "mixed")
        self.assertEqual([task["vendor"] for task in state["tasks"]], ["metax", "unknown"])

    def test_delivery_display_does_not_claim_unverified_version_for_failure(self):
        self.assertEqual(
            ps.delivery_display_label(
                {
                    "outcome": "failed",
                    "delivery_label": "V3 Max",
                    "uploaded": False,
                    "accuracy_ok": None,
                    "qualified_uploaded": False,
                }
            ),
            "未确认",
        )
        self.assertEqual(
            ps.delivery_display_label(
                {
                    "outcome": "success",
                    "delivery_label": "V3 Max",
                    "uploaded": False,
                    "accuracy_ok": None,
                    "qualified_uploaded": False,
                }
            ),
            "V3 Max（待确认）",
        )

    def test_result_labels(self):
        self.assertIn("达标上传", ps.model_result_label({"outcome": "success", "qualified_uploaded": True, "delivery_label": "V3 Max"}))
        self.assertIn("流程失败", ps.model_result_label({"outcome": "failed"}))
        self.assertIn("超时", ps.model_result_label({"outcome": "timeout"}))
        self.assertIn("已跳过", ps.model_result_label({"outcome": "skipped"}))
        self.assertIn("待确认", ps.model_result_label({"outcome": "success", "qualified_uploaded": False, "accuracy_ok": None}))

    def test_batch_end_lead_reports_unfinished_models(self):
        lead = ps.batch_end_lead(
            {
                "processed_models": 1,
                "total_models": 3,
                "qualified_uploaded": 1,
                "failed_models": 0,
                "timeout_models": 0,
                "skipped_models": 0,
            }
        )
        self.assertIn("1 达标", lead)
        self.assertIn("2 未完成", lead)
        self.assertNotIn("✅ 批次完成", lead)

    def test_batch_cli_progress_and_keywords(self):
        task_file = self.workspace / "tasks.txt"
        task_file.write_text("repo/metax-image:a | model-a\nrepo/hygon-image:b | model-b\n", encoding="utf-8")
        state_file = self.workspace / "batch_demo_progress.json"

        start = self.run_cli(
            "batch-start", "--workspace", self.workspace, "--state-file", state_file,
            "--batch-id", "demo", "--task-file", task_file, "--dry-run",
        )
        self.assertIn("任务开始", self.card_title(start))
        self.assertIn("mixed｜2 个模型", self.card_title(start))
        self.assertIn("⏳ 批量任务已启动", self.card_text(start))
        self.assertIn("**任务规模**\n2 个模型", self.card_text(start))
        self.assertIn("### 任务队列", self.card_text(start))
        self.assertIn("model-a", self.card_text(start))
        self.assertNotIn("未知", self.card_text(start))
        self.assertNotIn("未识别", self.card_text(start))
        self.assertNotIn("成功率", self.card_text(start))

        result_a = self.write_result(
            "model-a", target="repo/metax-image:a", vendor="metax", elapsed_seconds=120,
            version="v3", accuracy_ok=True, uploaded=True,
        )
        model_finish = self.run_cli(
            "model-finish", "--workspace", self.workspace, "--state-file", state_file,
            "--model", "model-a", "--target", "repo/metax-image:a", "--result-file", result_a,
            "--outcome", "success",
            "--exit-code", "0", "--elapsed-seconds", "120", "--dry-run",
        )
        self.assertIn("结果汇总", self.card_title(model_finish))
        self.assertIn("mixed｜1 / 2", self.card_title(model_finish))
        self.assertIn("✅ V3 Max 达标上传 · model-a", self.card_text(model_finish))
        self.assertIn("**当前总耗时**", self.card_text(model_finish))
        self.assertIn("**累计费用**", self.card_text(model_finish))
        self.assertIn("**达标上传**\n1 / 1 · 100.0%", self.card_text(model_finish))
        self.assertIn("**成功模型均值**", self.card_text(model_finish))

        result_b = self.write_result(
            "model-b", target="repo/hygon-image:b", vendor="hygon", outcome="failed",
            exit_code=1, elapsed_seconds=60, version="v3", accuracy_ok=False, uploaded=False,
        )
        self.run_cli(
            "model-finish", "--workspace", self.workspace, "--state-file", state_file,
            "--model", "model-b", "--target", "repo/hygon-image:b", "--result-file", result_b,
            "--outcome", "failed",
            "--exit-code", "1", "--elapsed-seconds", "60", "--dry-run",
        )
        final = self.run_cli(
            "batch-end", "--workspace", self.workspace, "--state-file", state_file,
            "--exit-code", "1", "--dry-run",
        )
        self.assertIn("任务结束", self.card_title(final))
        self.assertNotIn("结果汇总", self.card_title(final))
        self.assertIn("mixed｜1 / 2 达标", self.card_title(final))
        self.assertIn("批次结束", self.card_text(final))
        self.assertIn("**批次总耗时**", self.card_text(final))
        self.assertIn("### 最终模型明细", self.card_text(final))
        state = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertEqual(state["vendor"], "mixed")
        self.assertEqual(len(state["models"]), 2)

    def test_batch_cards_include_model_detail_table(self):
        task_file = self.workspace / "tasks.txt"
        task_file.write_text("image-a | model-a\nimage-b | model-b\n", encoding="utf-8")
        state_file = self.workspace / "batch_table_progress.json"

        start = self.run_cli(
            "batch-start", "--workspace", self.workspace, "--state-file", state_file,
            "--batch-id", "table", "--task-file", task_file, "--vendor", "metax", "--dry-run",
        )
        start_text = self.card_text(start)
        self.assertIn("### 任务队列", start_text)
        self.assertIn("1. model-a", start_text)
        self.assertIn("2. model-b", start_text)
        self.assertNotIn("| # |", start_text)
        self.assertNotIn("⏳ 待执行", start_text)

        result_a = self.write_result(
            "model-a", target="image-a", vendor="metax", elapsed_seconds=120,
            version="v3", accuracy_ok=True, uploaded=True,
        )
        finished = self.run_cli(
            "model-finish", "--workspace", self.workspace, "--state-file", state_file,
            "--model", "model-a", "--target", "image-a", "--task-index", "1",
            "--result-file", result_a, "--outcome", "success", "--exit-code", "0",
            "--elapsed-seconds", "120", "--dry-run",
        )
        finished_text = self.card_text(finished)
        self.assertIn("✅ V3 Max 达标", finished_text)
        self.assertIn("2m00s", finished_text)
        self.assertIn("| # | 模型 | 结果 | 耗时 / 费用 |", finished_text)
        self.assertIn("待执行：1 个 · 下一模型：model-b", finished_text)

    def test_abnormal_batch_end_marks_current_model_unfinished(self):
        state = {
            "total_models": 2,
            "tasks": [
                {"task_index": 1, "target": "a", "model": "m1"},
                {"task_index": 2, "target": "b", "model": "m2"},
            ],
            "models": [],
            "current_model": {"task_index": 1, "model": "m1", "vendor": "metax"},
            "ended_at": "2026-07-21T00:00:00+08:00",
            "vendor": "metax",
        }
        table = ps.build_models_markdown_table(state)
        self.assertIn("⛔ 未完成", table)
        self.assertIn("未完成：2 个", table)

    def test_large_batch_table_is_bounded_and_reports_omitted_rows(self):
        state = {
            "total_models": 25,
            "vendor": "metax",
            "tasks": [
                {"task_index": index, "target": f"image-{index}", "model": f"model-{index}"}
                for index in range(1, 26)
            ],
            "models": [
                {
                    "task_index": index,
                    "model": f"model-{index}",
                    "target": f"image-{index}",
                    "vendor": "metax",
                    "outcome": "success",
                    "elapsed_seconds": index,
                    "cost_usd": index / 10,
                    "delivery_label": "V3 Max",
                    "qualified_uploaded": True,
                    "accuracy_ok": True,
                    "uploaded": True,
                }
                for index in range(1, 26)
            ],
            "current_model": None,
        }
        table = ps.build_models_markdown_table(state, max_rows=5)
        self.assertEqual(table.count("✅ V3 Max 达标"), 5)
        self.assertIn("另有 20 个已处理模型未展示", table)

    def test_single_cli_keywords(self):
        result = self.write_result(
            "model-a", target="repo/metax-model:latest", vendor="metax", elapsed_seconds=12,
            version="v3", accuracy_ok=True, uploaded=True,
        )
        start = self.run_cli(
            "single-start", "--workspace", self.workspace, "--model", "model-a",
            "--target", "repo/metax-model:latest", "--dry-run",
        )
        end = self.run_cli(
            "single-end", "--workspace", self.workspace, "--model", "model-a",
            "--target", "repo/metax-model:latest", "--result-file", result,
            "--outcome", "success", "--exit-code", "0", "--elapsed-seconds", "12", "--dry-run",
        )
        self.assertIn("任务开始", self.card_title(start))
        self.assertIn("metax", self.card_title(start))
        self.assertIn("model-a", self.card_title(start))
        self.assertIn("⏳ 正在执行", self.card_text(start))
        self.assertNotIn("**设备**", self.card_text(start))
        self.assertNotIn("执行环境", self.card_text(start))
        self.assertNotIn("整体进度", self.card_text(start))
        self.assertNotIn("未知", self.card_text(start))
        self.assertIn("任务结束", self.card_title(end))
        self.assertNotIn("结果汇总", self.card_title(end))
        self.assertIn("✅ V3 Max 达标上传", self.card_text(end))
        self.assertIn("**总耗时**\n12s", self.card_text(end))
        self.assertIn("**总费用**\n未知", self.card_text(end))
        self.assertNotIn("执行环境", self.card_text(end))
        self.assertNotIn("设备**", self.card_text(end))
        self.assertNotIn("成功率", self.card_text(end))
        self.assertNotIn("成功模型平均", self.card_text(end))
        self.assertNotIn("累计达标", self.card_text(end))


if __name__ == "__main__":
    unittest.main()
