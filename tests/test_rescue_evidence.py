"""Offline compaction evidence assessment tests: matching, safety and privacy."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from _llmrig import cli, rescue_agent, rescue_evidence


def receipt(*, disable=False, status="failed", failure="agent_timeout",
            model="qwen3.8:27b-mlx", context=4096, task=None, steps=None):
    checks = {key: True for key in rescue_evidence.REQUIRED_CHECKS}
    if status == "failed":
        checks["process_exit_zero"] = False
    evaluation = {
        "task_id": task or rescue_agent.TASK_ID,
        "agent": "opencode",
        "model": model,
        "status": status,
        "agent_steps_limit": steps,
        "auto_compaction_disabled": disable,
        "fixture_sha256": rescue_evidence.FIXTURE_SHA256,
        "checks": checks,
        "failure_category": failure if status == "failed" else None,
        "event_sequence": {"notes": "/Users/private/secret"},
    }
    return {
        "mode": "measured_agent_task",
        "agent": "opencode",
        "model": model,
        "context_tokens": context,
        "agent_evaluation": evaluation,
        "secret_path": "/Users/private/token",
    }


class OfflineCompactionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def save(self, filename, body):
        path = self.root / filename
        path.write_text(json.dumps(body), encoding="utf-8")
        return str(path)

    def representative(self):
        baseline = [
            self.save("baseline-90.json", receipt()),
            self.save("baseline-180.json", receipt()),
        ]
        candidate = [
            self.save("candidate-1.json", receipt(disable=True, status="passed")),
            self.save("candidate-2.json", receipt(disable=True, status="passed")),
            self.save("candidate-3.json", receipt(disable=True, status="passed")),
        ]
        return baseline, candidate

    def test_observed_repeated_mitigation_and_no_untrusted_private_values(self):
        baseline, candidate = self.representative()
        result = rescue_evidence.assess(baseline, candidate)
        self.assertEqual(result["mode"], "offline_existing_receipts_only")
        self.assertEqual(result["assessment"],
                         "repeated_mitigation_observed_not_causally_verified")
        self.assertEqual(result["baseline"]["sessions"], 2)
        self.assertEqual(result["baseline"]["strict_task_passes"], 0)
        self.assertEqual(result["baseline"]["agent_timeouts_after_fix_evidence"], 2)
        self.assertEqual(result["candidate"]["strict_task_passes"], 3)
        self.assertEqual(result["candidate"]["agent_timeouts_after_fix_evidence"], 0)
        self.assertEqual(result["recommendation"], "no_automatic_or_global_compaction_change")
        result_json = json.dumps(result)
        self.assertNotIn("/Users/private", result_json)
        self.assertNotIn("baseline-90", result_json)
        self.assertNotIn("candidate-1", result_json)
        self.assertNotIn("secret_path", result_json)
        self.assertIn("not a verified causal repair",
                      rescue_evidence.format_terminal(result))

    def test_one_baseline_is_inconclusive_by_defined_triage_threshold(self):
        baseline, candidate = self.representative()
        result = rescue_evidence.assess(baseline[:1], candidate)
        self.assertEqual(result["assessment"], "inconclusive_or_mixed_evidence")

    def test_mismatched_context_and_model_rejected(self):
        baseline, candidate = self.representative()
        mismatch_context = self.save(
            "candidate-different-context.json",
            receipt(disable=True, status="passed", context=8192)
        )
        with self.assertRaisesRegex(ValueError, "same model and declared context"):
            rescue_evidence.assess(baseline, candidate[:2] + [mismatch_context])
        mismatch_model = self.save(
            "candidate-different-model.json",
            receipt(disable=True, status="passed", model="another:tag")
        )
        with self.assertRaises(ValueError):
            rescue_evidence.assess(baseline, candidate[:2] + [mismatch_model])

    def test_wrong_compaction_policy_and_step_caps_rejected(self):
        baseline, candidate = self.representative()
        with self.assertRaisesRegex(ValueError, "baseline"):
            rescue_evidence.assess(candidate[:1], candidate[1:])
        with self.assertRaisesRegex(ValueError, "candidate"):
            rescue_evidence.assess(baseline, baseline[:1])
        with self.assertRaisesRegex(ValueError, "step-limit"):
            rescue_evidence.assess(
                [self.save("steps-limited.json", receipt(steps=4))], candidate
            )

    def test_wrong_task_missing_checks_and_forged_pass_rejected(self):
        baseline, candidate = self.representative()
        wrong_task = self.save("wrong-task.json", receipt(task="not-our-fixture"))
        with self.assertRaises(ValueError):
            rescue_evidence.assess([wrong_task], candidate)
        broken = receipt()
        del broken["agent_evaluation"]["checks"]["fixture_unchanged"]
        with self.assertRaisesRegex(ValueError, "task check"):
            rescue_evidence.assess([self.save("missing-check.json", broken)], candidate)
        forged = receipt(disable=True, status="passed")
        forged["agent_evaluation"]["checks"]["process_exit_zero"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent pass"):
            rescue_evidence.assess(baseline, [self.save("bad-pass.json", forged)])

    def test_reject_duplicate_report_and_oversized_input(self):
        baseline, candidate = self.representative()
        with self.assertRaisesRegex(ValueError, "unique"):
            rescue_evidence.assess(baseline + baseline[:1], candidate)
        path = self.root / "oversize.json"
        path.write_bytes(b"x" * (rescue_evidence.MAX_REPORT_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "size cap"):
            rescue_evidence.assess([str(path)], candidate)

    def test_cli_never_invokes_agent_or_ollama_and_saves_new_file_only(self):
        baseline, candidate = self.representative()
        output = self.root / "assessment.json"
        args = ["rescue-evidence"]
        for path in baseline:
            args.extend(("--baseline", path))
        for path in candidate:
            args.extend(("--candidate", path))
        args.extend(("--output", str(output)))
        with mock.patch.object(rescue_agent, "evaluate") as agent, \
             mock.patch("urllib.request.urlopen") as net, \
             contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(cli.main(args), 0)
            agent.assert_not_called()
            net.assert_not_called()
        self.assertIn("repeated_mitigation_observed", printed.getvalue())
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["baseline"]["sessions"], 2)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args), 2)  # no overwrite

    def test_bad_json_and_wrong_mode_fail_closed_without_raw_data(self):
        path = self.root / "invalid.json"
        path.write_text("not JSON secret token", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON") as err:
            rescue_evidence.assess([str(path)], [str(path) + "2"])
        self.assertNotIn("secret token", str(err.exception))
        wrong = self.save("preflight.json", {"mode": "preflight"})
        candidate = self.save("candidate.json", receipt(disable=True, status="passed"))
        with self.assertRaisesRegex(ValueError, "only single-task"):
            rescue_evidence.assess([wrong], [candidate])


if __name__ == "__main__":
    unittest.main()
