"""Conservative diagnostic classification for controlled OpenCode compaction A/B."""
import unittest
from unittest import mock

from _llmrig import rescue_ab, rescue_compaction_ab


def trial(model, context, task, seconds, disable_auto_compaction=False):
    checks = {
        "read_tool_completed": True, "bug_explained": True,
        "correct_return_proposed": True,
        "no_disallowed_tool_observed": True, "no_error_event": True,
        "valid_event_stream": True, "process_exit_zero": disable_auto_compaction,
        "fixture_unchanged": True,
    }
    return {
        "task_id": task.task_id, "context_tokens": context,
        "auto_compaction_disabled": disable_auto_compaction,
        "status": "passed" if disable_auto_compaction else "failed",
        "checks": checks,
        "failure_category": None if disable_auto_compaction else "agent_timeout",
    }


class DiagnosticSignatureTests(unittest.TestCase):
    def run_trial(self, override=None):
        def run(model, context, task, seconds, disable_auto_compaction=False):
            result = trial(model, context, task, seconds, disable_auto_compaction)
            if override is not None:
                return override(result, disable_auto_compaction, task)
            return result
        with mock.patch.object(rescue_compaction_ab.shutil, "which", return_value="opencode"), \
             mock.patch.object(rescue_ab, "_trial", side_effect=run):
            return rescue_compaction_ab.compare_compaction(
                "qwen3.8:27b-mlx", 4096, repeats=1,
                seconds=90, budget_seconds=600,
            )

    def test_repeated_post_answer_timeouts_yield_scoped_signature(self):
        result = self.run_trial()
        self.assertEqual(result["status"], "completed")
        diagnosis = result["diagnosis"]
        self.assertEqual(
            diagnosis["code"],
            "suspected_compaction_associated_post_answer_continuation",
        )
        self.assertEqual(diagnosis["supporting_default_timeouts_after_fix"], 3)
        self.assertEqual(diagnosis["supporting_candidate_strict_passes"], 3)
        self.assertIs(diagnosis["real_configuration_recommended"], False)
        self.assertEqual(result["recommendation"], "no_automatic_or_global_compaction_change")
        self.assertIn("No direct OpenCode compaction event",
                      diagnosis["known_unknowns"][0])

    def test_mixed_results_must_not_claim_signature(self):
        def mixed(r, off, task):
            if off and task.task_id == "parity-v1":
                r["status"] = "failed"
                r["failure_category"] = "task_requirements_not_met"
                r["checks"]["bug_explained"] = False
            return r
        result = self.run_trial(mixed)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["diagnosis"]["code"],
                         "mixed_or_insufficient_compaction_evidence")
        self.assertIs(result["diagnosis"]["real_configuration_recommended"], False)

    def test_disallowed_tool_stops_incomplete_without_signature(self):
        def bad(r, off, task):
            r["checks"]["no_disallowed_tool_observed"] = False
            r["failure_category"] = "test_boundary_violation"
            r["status"] = "failed"
            return r
        result = self.run_trial(bad)
        self.assertEqual(result["status"], "stopped_on_boundary_violation")
        self.assertEqual(result["diagnosis"]["code"], "insufficient_comparable_evidence")
        self.assertIs(result["diagnosis"]["real_configuration_recommended"], False)
        self.assertIsNone(result["observed_comparison"])


if __name__ == "__main__":
    unittest.main()
