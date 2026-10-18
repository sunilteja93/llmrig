"""Exercise Rescue's privacy-safe event timeline on bounded synthetic streams."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from _llmrig import rescue_agent, rescue_ab, rescue_trace


def event(kind, *, part=None):
    return json.dumps({"type": kind, "part": part or {}}).encode("utf-8") + b"\n"


def sequence():
    return b"".join((
        event("step_start", part={"type": "step-start"}),
        event("tool_use", part={"tool": "read", "state": {
            "status": "completed", "input": {"filePath": "math_ops.py"},
        }}),
        event("text", part={"text": "n % 2 == 1 tests odd; return n % 2 == 0"}),
        event("step_start", part={"type": "step-start"}),
        event("tool_use", part={"tool": "read", "state": {
            "status": "completed", "input": {"filePath": "math_ops.py"},
        }}),
        event("step_finish", part={"type": "step-finish"}),
    ))


class RescueTraceTests(unittest.TestCase):
    def test_first_fix_and_following_events_are_counted_without_content(self):
        trace = rescue_trace.summarize(sequence(), rescue_agent.FIX)
        self.assertEqual(trace["events_seen"], 6)
        self.assertEqual(trace["first_fix_evidence_event_index"], 3)
        self.assertEqual(trace["events_after_first_fix_evidence"], 3)
        self.assertEqual(trace["event_type_counts"]["step_start"], 2)
        self.assertEqual(trace["tool_category_counts"]["read"], 2)
        self.assertEqual(trace["post_fix_event_type_counts"]["step_start"], 1)
        self.assertEqual(trace["post_fix_event_type_counts"]["tool_use"], 1)
        self.assertEqual(trace["post_fix_event_type_counts"]["step_finish"], 1)
        self.assertEqual(trace["last_event_type_category"], "step_finish")
        self.assertFalse(trace["stream_truncated_or_malformed"])
        self.assertEqual(trace["interpretation"],
                         "provisional_fix_evidence_not_final_agent_completion")
        self.assertNotIn("n % 2", json.dumps(trace))
        self.assertNotIn("math_ops.py", json.dumps(trace))

    def test_untrusted_event_and_tool_names_are_not_exposed(self):
        secret = "/Users/secret/private/tokenXYZ"
        raw = (
            event(secret, part={"text": secret})
            + event("tool_use", part={"tool": secret, "args": {"path": secret}})
            + event("text", part={"text": secret})
        )
        trace = rescue_trace.summarize(raw, rescue_agent.FIX)
        self.assertEqual(trace["event_type_counts"]["other"], 1)
        self.assertEqual(trace["tool_category_counts"]["other"], 1)
        self.assertEqual(trace["first_fix_evidence_event_index"], None)
        self.assertEqual(trace["last_event_type_category"], "text")
        self.assertNotIn(secret, json.dumps(trace))

    def test_unparseable_json_terminates_trace_without_leaking(self):
        trace = rescue_trace.summarize(sequence() + b"not-json /private\n", rescue_agent.FIX)
        self.assertTrue(trace["stream_truncated_or_malformed"])
        self.assertEqual(trace["events_seen"], 6)
        self.assertNotIn("private", json.dumps(trace))

    def test_one_agent_timeout_stays_failed_despite_detected_correction(self):
        raw = sequence()
        def invoke(cmd, cwd, env, seconds):
            self.assertEqual(seconds, 90)
            self.assertIn("ollama/qwen3.8:27b-mlx", cmd)
            return ("timeout", -9, raw, True)
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke", side_effect=invoke):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 4096, seconds=90)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_category"], "agent_timeout")
        self.assertEqual(result["completion_diagnostic"],
                         "provisional_fix_observed_before_agent_timeout")
        trace = result["event_sequence"]
        self.assertEqual(trace["first_fix_evidence_event_index"], 3)
        self.assertEqual(trace["events_after_first_fix_evidence"], 3)
        self.assertFalse(result["checks"]["process_exit_zero"])
        self.assertTrue(result["checks"]["correct_return_proposed"])
        self.assertNotIn("n % 2", json.dumps(result))

    def test_ab_task_trace_is_versioned_and_cannot_change_failure(self):
        task = rescue_ab.TASKS[0]
        with tempfile.TemporaryDirectory() as temp:
            # Unit checks use the real report-producing trial and mocked executor.
            def invoke(cmd, cwd, env, seconds):
                path = cwd / task.filename
                self.assertTrue(path.exists())
                # A full mock stream with an absolute fixture path.
                raw = b"".join((
                    event("tool_use", part={"tool": "read", "state": {
                        "status": "completed", "input": {"filePath": str(path)},
                    }}),
                    event("text", part={
                        "text": "n % 2 == 1 tests odd numbers; return n % 2 == 0",
                    }),
                    event("step_start", part={}),
                ))
                return ("timeout", -9, raw, True)
            with mock.patch.object(rescue_ab.shutil, "which", return_value="/usr/bin/opencode"), \
                 mock.patch.object(rescue_ab, "_invoke", side_effect=invoke):
                result = rescue_ab._trial("qwen3.8:27b-mlx", 4096, task, seconds=90)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_category"], "agent_timeout")
        self.assertEqual(result["completion_diagnostic"],
                         "provisional_fix_observed_before_agent_timeout")
        self.assertEqual(result["event_sequence"]["events_after_first_fix_evidence"], 1)
        self.assertTrue(result["checks"]["correct_return_proposed"])
        self.assertNotIn(temp, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
