"""Regression tests for read-only diagnostics and opt-in Rig Reports."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _llmrig import cli
from _llmrig.guidance import diagnosis, next_steps
from _llmrig.rig_report import build_share_report, format_markdown


def receipt():
    return {
        "schema_version": "0.9",
        "receipt_id": "receipt-demo",
        "model": "org/model",
        "logical_model_id": "org/model",
        "status": "completed",
        "candidate": {
            "runtime": "mlx-lm",
            "artifact_id": "hf://org/model/mlx",
            "artifact_format": "MLX",
            "quantization": "4-bit",
            "artifact_revision": "0123456789abcdef",
            "context_tokens": 4096,
        },
        "endpoint": "http://127.0.0.1:8080/v1",
        "private_path": "/Users/private-user/test",
        "secret": "do-not-publish",
        "verification": {
            "method_version": "race-v2",
            "measured_runs": 2,
            "generation_tps": 42.1,
            "prompt_eval_tps": None,
            "total_latency_s": 1.2,
        },
    }


class RigReportTests(unittest.TestCase):
    def test_report_whitelists_fields_and_omits_private_data(self):
        report = build_share_report(receipt())
        payload = json.dumps(report)
        self.assertEqual(report["model"], "org/model")
        self.assertEqual(report["verification"]["generation_tps"], 42.1)
        self.assertNotIn("private-user", payload)
        self.assertNotIn("do-not-publish", payload)
        self.assertNotIn("127.0.0.1", payload)
        self.assertNotIn("Apple M4", payload)
        markdown = format_markdown(report)
        self.assertIn("not independently attested", markdown)
        self.assertIn("Machine hardware is intentionally omitted", markdown)

    def test_report_rejects_private_model_identity(self):
        unsafe = receipt()
        unsafe["model"] = "/Users/private-user/model"
        unsafe["logical_model_id"] = None
        with self.assertRaises(ValueError):
            build_share_report(unsafe)

    def test_report_rejects_invalid_measurements(self):
        for invalid in (float("nan"), float("inf"), -1, True):
            bad = receipt()
            bad["verification"]["generation_tps"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                build_share_report(bad)

    def test_report_command_requires_existing_receipt_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "report.md"
            args = argparse.Namespace(receipt="latest", json=False, share=True, output=str(path))
            with mock.patch.object(cli, "load_receipt", return_value=receipt()):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.command_report(args), 0)
            content = path.read_text(encoding="utf-8")
            self.assertIn("# LLMRig Rig Report", content)
            with mock.patch.object(cli, "load_receipt", return_value=receipt()):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.command_report(args), 2)
            self.assertEqual(content, path.read_text(encoding="utf-8"))


class GuidanceTests(unittest.TestCase):
    def test_incompatible_candidate_has_specific_next_steps(self):
        candidate = mock.Mock()
        candidate.blockers = ("Memory budget exceeded",)
        candidate.unknowns = ("runtime not installed",)
        plan = mock.Mock()
        plan.selected_candidate_id = None
        plan.candidates = (candidate,)
        plan.blockers = ("incompatible",)
        plan.blocked = True
        plan.model = "org/model"
        plan.plan_id = "plan-1"
        steps = next_steps(plan)
        self.assertTrue(any("smaller" in step for step in steps))
        self.assertTrue(any("runtimes" in step for step in steps))
        self.assertEqual(diagnosis(plan)["status"], "blocked")

    def test_doctor_command_is_read_only(self):
        plan = mock.Mock()
        plan.model = "org/model"
        plan.logical_model_id = "org/model"
        plan.plan_id = "plan-1"
        plan.selected_candidate_id = None
        plan.candidates = ()
        plan.blockers = ("no supported path",)
        plan.blocked = True
        args = argparse.Namespace(model="org/model", context=None, json=True)

        class Legacy:
            class SolveInputError(Exception):
                pass

            class SolveEngineError(Exception):
                pass

        with mock.patch.object(cli, "_build_plan", return_value=plan), \
                mock.patch.object(cli, "apply_autopilot_plan") as apply, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.command_doctor(args, Legacy), 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "blocked")
        apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
