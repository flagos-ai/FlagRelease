import argparse
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ANALYZER_PATH = Path(__file__).resolve().parents[2] / "batch_summarize" / "model_result_analyzer.py"
SPEC = importlib.util.spec_from_file_location("model_result_analyzer", ANALYZER_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyzer)


class ModelResultAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model_dir = self.root / "model-a"
        self.model_dir.mkdir()
        self.prompt = self.root / "prompt.md"
        self.prompt.write_text("analyze {{MODEL_NAME}}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self):
        return argparse.Namespace(
            model_dir=self.model_dir,
            model_name="model-a",
            result_json=self.model_dir / "reports" / "progress_result.json",
            prompt_file=self.prompt,
            target="repo/metax-model:v1",
            vendor="metax",
            outcome="success",
            exit_code=0,
            elapsed_seconds=123,
            timeout_seconds=10,
            claude_command="claude",
        )

    @staticmethod
    def structured():
        return {
            "cost": {"migration_cost_usd": 4.26, "complete": True},
            "delivery": {"version": "v3", "accuracy_ok": True, "uploaded": True},
            "notification": {
                "result_label": "达标上传",
                "lead": "✅ V3 Max 达标上传",
                "summary": "精度达标且已上传",
                "warning": None,
            },
            "evidence": {
                "accuracy": ["results/gpqa_v3.json"],
                "upload": ["traces/13_v3_release.json"],
                "cost": ["logs/seg1_cost.txt"],
            },
        }

    def test_valid_output_keeps_trusted_runtime_facts_and_separates_analysis_cost(self):
        envelope = {"structured_output": self.structured(), "total_cost_usd": 0.08}
        completed = subprocess.CompletedProcess([], 0, json.dumps(envelope), "")
        with mock.patch.object(analyzer.subprocess, "run", return_value=completed):
            result = analyzer.analyze(self.args())
        self.assertEqual(result["analysis_status"], "success")
        self.assertEqual(result["vendor"], "metax")
        self.assertEqual(result["pipeline"]["elapsed_seconds"], 123)
        self.assertEqual(result["cost"]["migration_cost_usd"], 4.26)
        self.assertEqual(result["analysis"]["cost_usd"], 0.08)
        self.assertTrue(result["delivery"]["qualified_uploaded"])

    def test_claude_cannot_override_deterministic_result_label(self):
        structured = self.structured()
        structured["notification"]["result_label"] = "流程失败"
        structured["notification"]["lead"] = "伪造状态"
        completed = subprocess.CompletedProcess(
            [], 0, json.dumps({"structured_output": structured}), ""
        )
        with mock.patch.object(analyzer.subprocess, "run", return_value=completed):
            result = analyzer.analyze(self.args())
        self.assertEqual(result["notification"]["result_label"], "达标上传")
        self.assertEqual(result["notification"]["lead"], "✅ V3 Max 达标上传")

    def test_delivery_without_version_cannot_be_qualified(self):
        structured = self.structured()
        structured["delivery"]["version"] = None
        completed = subprocess.CompletedProcess(
            [], 0, json.dumps({"structured_output": structured}), ""
        )
        with mock.patch.object(analyzer.subprocess, "run", return_value=completed):
            result = analyzer.analyze(self.args())
        self.assertFalse(result["delivery"]["qualified_uploaded"])
        self.assertEqual(result["notification"]["result_label"], "无法确认")

    def test_code_fenced_result_is_supported(self):
        body = "```json\n" + json.dumps(self.structured()) + "\n```"
        structured, cost = analyzer.extract_structured_output(
            json.dumps({"result": body, "total_cost_usd": 0.03})
        )
        self.assertEqual(structured["delivery"]["version"], "v3")
        self.assertEqual(cost, 0.03)

    def test_missing_schema_field_produces_legal_failed_result(self):
        invalid = self.structured()
        invalid.pop("evidence")
        completed = subprocess.CompletedProcess([], 0, json.dumps({"structured_output": invalid}), "")
        with mock.patch.object(analyzer.subprocess, "run", return_value=completed):
            result = analyzer.analyze(self.args())
        self.assertEqual(result["analysis_status"], "failed")
        self.assertEqual(result["pipeline"]["outcome"], "success")
        self.assertIsNone(result["cost"]["migration_cost_usd"])
        self.assertFalse(result["delivery"]["qualified_uploaded"])
        self.assertIn("字段不完整", result["analysis_error"])

    def test_timeout_produces_failed_result(self):
        with mock.patch.object(
            analyzer.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10),
        ):
            result = analyzer.analyze(self.args())
        self.assertEqual(result["analysis_status"], "failed")
        self.assertIn("分析超时", result["analysis_error"])


if __name__ == "__main__":
    unittest.main()
