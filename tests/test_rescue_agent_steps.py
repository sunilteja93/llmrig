"""Explicit, isolated OpenCode steps cap for reproducing post-answer loops."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

from _llmrig import cli, rescue, rescue_agent


def _events():
    return (b'{"type":"tool_use","part":{"tool":"read","state":'
            b'{"status":"completed","input":{"filePath":"math_ops.py"}}}}\n'
            b'{"type":"text","part":{"text":"n % 2 == 1 tests odd numbers. '
            b'return n % 2 == 0"}}\n'
            b'{"type":"step_finish","part":{"type":"step-finish"}}\n')


class RescueStepLimitTests(unittest.TestCase):
    def test_default_fixture_is_unmodified_and_cap_is_temporary(self):
        ordinary = rescue_agent._configuration("qwen3.8:27b-mlx", 4096)
        limited = rescue_agent._configuration("qwen3.8:27b-mlx", 4096, agent_steps=4)
        self.assertNotIn("steps", ordinary["agent"]["rescue-readonly"])
        self.assertEqual(limited["agent"]["rescue-readonly"]["steps"], 4)
        self.assertEqual(ordinary["permission"], limited["permission"])
        self.assertEqual(limited["permission"]["bash"], "deny")
        self.assertEqual(limited["permission"]["edit"], "deny")
        self.assertEqual(limited["provider"]["ollama"]["options"]["baseURL"],
                         "http://127.0.0.1:11434/v1")

    def test_bounded_agent_invokes_only_disposable_config_and_reports_limit(self):
        def fake_invoke(cmd, cwd, env, seconds):
            self.assertEqual(seconds, 90)
            self.assertIn("--agent", cmd)
            self.assertEqual(cmd[cmd.index("--agent") + 1], "rescue-readonly")
            conf = json.loads((cwd / "opencode.json").read_text(encoding="utf-8"))
            self.assertEqual(conf["agent"]["rescue-readonly"]["steps"], 4)
            self.assertEqual(conf["permission"]["*"], "deny")
            self.assertEqual((cwd / "math_ops.py").read_text(encoding="utf-8"),
                             rescue_agent.SOURCE)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("HTTP_PROXY", env)
            self.assertEqual(Path(env["HOME"]).parent.name, Path(cwd).parent.name)
            return ("exited", 0, _events(), False)

        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke", side_effect=fake_invoke):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 4096, seconds=90, agent_steps=4)
        self.assertEqual(result["agent_steps_limit"], 4)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["failure_category"], None)
        self.assertTrue(result["checks"]["correct_return_proposed"])
        self.assertTrue(result["checks"]["fixture_unchanged"])
        self.assertNotIn("math_ops.py", json.dumps(result))

    def test_steps_do_not_reclassify_a_timed_out_agent_as_passed(self):
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke",
                               return_value=("timeout", -9, _events(), True)):
            result = rescue_agent.evaluate("qwen3.8:27b-mlx", 4096, agent_steps=4)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["agent_steps_limit"], 4)
        self.assertEqual(result["failure_category"], "agent_timeout")
        self.assertFalse(result["checks"]["process_exit_zero"])
        self.assertTrue(result["checks"]["correct_return_proposed"])

    def test_steps_explicit_opt_in_and_validation_before_process(self):
        for bad in (1, 13, True, "4"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rescue_agent.evaluate("qwen3.8:27b-mlx", 4096, agent_steps=bad)

        with self.assertRaises(rescue.RescueError):
            rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, None, False,
                          agent_steps=4)
        with self.assertRaises(rescue.RescueError):
            rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, 8192, False,
                          ab_test=True, agent_steps=4)
        with self.assertRaises(rescue.RescueError):
            rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, None, False,
                          agent_test=True, agent_steps=13)

    def test_agent_steps_forwarded_to_evaluator_and_cli(self):
        report = {
            "status": "ready_for_probe",
            "mode": "preflight",
            "agent_executable_detected": True,
            "agent": "opencode", "model": "qwen3.8:27b-mlx",
            "context_tokens": 4096,
            "blockers": [], "unknowns": [], "probes": [], "comparison": None,
        }
        observed = {
            "status": "passed", "agent_steps_limit": 4,
            "checks": {"read_tool_completed": True},
            "failure_category": None,
        }
        with mock.patch.object(rescue, "local_host", return_value="http://127.0.0.1:11434"), \
             mock.patch.object(rescue, "preflight", return_value=report), \
             mock.patch.object(rescue_agent, "evaluate", return_value=observed) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main([
                    "rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                    "--context", "4096", "--agent-test", "--agent-steps", "4",
                ]), 0)
            run.assert_called_once_with(
                "qwen3.8:27b-mlx", 4096, seconds=120, agent_steps=4
            )
            self.assertIn("Configured agent step limit: 4", output.getvalue())
            self.assertIn("temporary, not a process timeout", output.getvalue())


if __name__ == "__main__":
    unittest.main()
