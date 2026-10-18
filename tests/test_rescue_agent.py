"""Mocked end-to-end evaluation contract for the first Rescue coding task."""
import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock

from _llmrig import cli, rescue, rescue_agent


def _stream(*, read=True, correct=True, explanation=True, extra_tool=None):
    events = [{"type": "step_start", "part": {"type": "step-start"}}]
    if read:
        events.append({
            "type": "tool_use",
            "part": {
                "tool": "read",
                "state": {
                    "status": "completed", "input": {"filePath": "math_ops.py"},
                    "output": "def is_even(n):\n    return n % 2 == 1",
                },
            },
        })
    if extra_tool:
        events.append({
            "type": "tool_use",
            "part": {"tool": extra_tool, "state": {"status": "completed", "input": {}}},
        })
    answer = (
        ("n % 2 == 1 checks odd numbers. " if explanation else "The code has an issue. ")
        + ("return n % 2 == 0" if correct else "return n % 2 == 1")
    )
    events.append({"type": "text", "part": {"type": "text", "text": answer}})
    events.append({"type": "step_finish", "part": {"type": "step-finish"}})
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode("utf-8")


class RescueAgentTests(unittest.TestCase):
    def test_fixture_and_permissions_are_correct(self):
        config = rescue_agent._configuration("qwen3.8:27b-mlx", 8192)
        self.assertEqual(config["enabled_providers"], ["ollama"])
        self.assertEqual(config["provider"]["ollama"]["options"]["baseURL"],
                         "http://127.0.0.1:11434/v1")
        self.assertEqual(config["permission"]["*"], "deny")
        self.assertEqual(config["permission"]["bash"], "deny" if "bash" in config["permission"] else "deny")
        self.assertEqual(config["permission"]["read"], "allow")
        self.assertEqual(config["permission"]["external_directory"], "deny")
        self.assertEqual(config["agent"]["rescue-readonly"]["mode"], "primary")
        self.assertEqual(config["share"], "disabled")

    def test_completed_read_and_real_correction_pass(self):
        def fake_invoke(cmd, cwd, env, seconds):
            self.assertEqual(cmd[1:4], ["run", "--format", "json"])
            self.assertEqual(cmd[cmd.index("--agent") + 1], "rescue-readonly")
            self.assertEqual(cmd[cmd.index("--model") + 1], "ollama/qwen3.8:27b-mlx")
            self.assertEqual(seconds, 120)
            self.assertIn("def is_even(n)", (cwd / "math_ops.py").read_text())
            self.assertEqual(json.loads((cwd / "opencode.json").read_text())["permission"]["*"], "deny")
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("HTTP_PROXY", env)
            self.assertNotEqual(env["HOME"], os.environ.get("HOME"))
            return ("exited", 0, _stream(), False)

        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke", side_effect=fake_invoke):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["failure_category"], None)
        self.assertEqual(result["task_id"], rescue_agent.TASK_ID)
        self.assertNotIn("n % 2 == 1", json.dumps(result))
        self.assertNotIn("Users/", json.dumps(result))
        self.assertFalse(result["raw_agent_output_included"])

    def test_no_read_is_not_success(self):
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke",
                               return_value=("exited", 0, _stream(read=False), False)):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["read_tool_completed"])
        self.assertTrue(result["checks"]["correct_return_proposed"])
        self.assertEqual(result["failure_category"], "task_requirements_not_met")

    def test_wrong_fix_is_not_success(self):
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke",
                               return_value=("exited", 0, _stream(correct=False), False)):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["checks"]["correct_return_proposed"])

    def test_disallowed_tool_event_fails_closed(self):
        checks, _ = rescue_agent._read_events(_stream(extra_tool="bash"))
        self.assertFalse(checks["no_disallowed_tool_observed"])
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke",
                               return_value=("exited", 0, _stream(extra_tool="bash"), False)):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
        self.assertEqual(result["failure_category"], "test_boundary_violation")

    def test_file_mutation_fails_closed(self):
        def changed(cmd, cwd, env, seconds):
            (cwd / "math_ops.py").write_text("modified")
            return ("exited", 0, _stream(), False)
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke", side_effect=changed):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
        self.assertFalse(result["checks"]["fixture_unchanged"])
        self.assertEqual(result["failure_category"], "test_boundary_violation")

    def test_agent_errors_are_failures_without_raw_leak(self):
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke",
                               return_value=("timeout", -9, b"secret /Users/jane", True)):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
        self.assertEqual(result["failure_category"], "agent_timeout")
        self.assertNotIn("jane", json.dumps(result))
        self.assertEqual(result["status"], "failed")

    def test_bad_json_is_not_success(self):
        checks, count = rescue_agent._read_events(b"not json\n")
        self.assertFalse(checks["valid_event_stream"])
        self.assertEqual(count, 1)

    def test_absent_opencode_fails_before_process(self):
        with mock.patch.object(rescue_agent.shutil, "which", return_value=None), \
             mock.patch.object(rescue_agent, "_invoke") as process:
            with self.assertRaises(ValueError):
                rescue_agent.evaluate("qwen3.8:27b-mlx", 8192)
            process.assert_not_called()

    def test_cli_agent_test_requires_explicit_opt_in(self):
        report = {
            "status": "ready_for_probe", "agent_executable_detected": True,
            "agent": "opencode", "model": "qwen3.8:27b-mlx", "context_tokens": 8192,
            "blockers": [], "unknowns": [], "probes": [], "comparison": None,
        }
        with mock.patch.object(rescue, "preflight", return_value=report), \
             mock.patch.object(rescue, "local_host", return_value="http://127.0.0.1:11434"), \
             mock.patch.object(rescue_agent, "evaluate", return_value={
                 "status": "passed", "checks": {"read_tool_completed": True},
                 "failure_category": None,
             }) as evaluator:
            ordinary = rescue.rescue("opencode", "qwen3.8:27b-mlx", 8192, 3, None, False)
            self.assertEqual(ordinary["status"], "ready_for_probe")
            evaluator.assert_not_called()
            tested = rescue.rescue(
                "opencode", "qwen3.8:27b-mlx", 8192, 3, None, False,
                agent_test=True
            )
            self.assertEqual(tested["status"], "agent_task_passed")
            evaluator.assert_called_once()
        with self.assertRaises(rescue.RescueError):
            rescue.rescue("opencode", "qwen3.8:27b-mlx", 8192, 3, 4096, False,
                          agent_test=True)

    def test_cli_invocation_reports_agent_result(self):
        fake = {
            "status": "agent_task_passed", "agent": "opencode",
            "agent_executable_detected": True,
            "model": "qwen3.8:27b-mlx", "context_tokens": 8192,
            "blockers": [], "unknowns": [], "probes": [], "comparison": None,
            "agent_evaluation": {
                "status": "passed", "checks": {"read_tool_completed": True},
                "failure_category": None
            }
        }
        with mock.patch.object(rescue, "rescue", return_value=fake), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main([
                "rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                "--context", "8192", "--agent-test"
            ]), 0)
        self.assertIn("Agent coding task: passed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
