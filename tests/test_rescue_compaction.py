"""Safety and opt-in contract for an isolated OpenCode compaction hypothesis."""
import contextlib
import io
import json
import unittest
from unittest import mock

from _llmrig import cli, rescue, rescue_agent


def _events():
    return (
        b'{"type":"tool_use","part":{"tool":"read","state":'
        b'{"status":"completed","input":{"filePath":"math_ops.py"}}}}\n'
        b'{"type":"text","part":{"text":"n % 2 == 1 checks odd numbers; '
        b'return n % 2 == 0"}}\n'
        b'{"type":"step_finish","part":{"type":"step-finish"}}\n'
    )


class CompactionHypothesisTests(unittest.TestCase):
    def test_default_never_changes_compaction(self):
        ordinary = rescue_agent._configuration("qwen3.8:27b-mlx", 4096)
        disabled = rescue_agent._configuration(
            "qwen3.8:27b-mlx", 4096, disable_auto_compaction=True
        )
        self.assertNotIn("compaction", ordinary)
        self.assertEqual(disabled["compaction"], {"auto": False})
        self.assertEqual(ordinary["permission"], disabled["permission"])
        self.assertEqual(ordinary["provider"], disabled["provider"])
        self.assertEqual(ordinary["agent"], disabled["agent"])
        self.assertEqual(disabled["permission"]["edit"], "deny")
        self.assertEqual(disabled["permission"]["bash"], "deny")
        self.assertEqual(disabled["permission"]["external_directory"], "deny")

    def test_disposable_fixture_only_and_no_persistent_settings(self):
        def fake_invoke(cmd, cwd, env, seconds):
            self.assertEqual(seconds, 90)
            self.assertIn("--model", cmd)
            self.assertEqual(cmd[cmd.index("--agent") + 1], "rescue-readonly")
            local = json.loads((cwd / "opencode.json").read_text(encoding="utf-8"))
            self.assertEqual(local["compaction"], {"auto": False})
            self.assertEqual(json.loads(env["OPENCODE_CONFIG_CONTENT"])["compaction"],
                             {"auto": False})
            self.assertEqual(local["permission"]["*"], "deny")
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("HTTP_PROXY", env)
            self.assertEqual((cwd / "math_ops.py").read_text(encoding="utf-8"),
                             rescue_agent.SOURCE)
            return ("exited", 0, _events(), False)

        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke", side_effect=fake_invoke):
            result = rescue_agent.evaluate(
                "qwen3.8:27b-mlx", 4096, seconds=90, disable_auto_compaction=True
            )
        self.assertEqual(result["status"], "passed")
        self.assertIs(result["auto_compaction_disabled"], True)
        self.assertEqual(result["checks"]["process_exit_zero"], True)
        self.assertTrue(result["checks"]["fixture_unchanged"])
        self.assertFalse(result["raw_agent_output_included"])
        self.assertNotIn("n % 2", json.dumps(result))

    def test_disabling_compaction_does_not_override_timeout_failure(self):
        with mock.patch.object(rescue_agent.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_agent, "_invoke",
                               return_value=("timeout", -9, _events(), True)):
            result = rescue_agent.evaluate(
                "qwen3.8:27b-mlx", 4096, disable_auto_compaction=True
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_category"], "agent_timeout")
        self.assertTrue(result["checks"]["correct_return_proposed"])
        self.assertFalse(result["checks"]["process_exit_zero"])

    def test_opt_in_required_before_network_or_agent_run(self):
        with self.assertRaisesRegex(rescue.RescueError, "requires explicit --agent-test"):
            rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, None, False,
                          agent_no_auto_compact=True)
        with self.assertRaisesRegex(rescue.RescueError, "requires explicit --agent-test"):
            rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 3, 8192, False,
                          ab_test=True, agent_no_auto_compact=True)
        with self.assertRaises(ValueError):
            rescue_agent.evaluate("qwen3.8:27b-mlx", 4096,
                                  disable_auto_compaction="no")

    def test_cli_passes_only_explicit_request(self):
        preflight = {
            "status": "ready_for_probe", "mode": "preflight",
            "agent_executable_detected": True,
            "agent": "opencode", "model": "qwen3.8:27b-mlx",
            "context_tokens": 4096, "blockers": [], "unknowns": [],
            "probes": [], "comparison": None,
        }
        evaluation = {
            "status": "passed", "auto_compaction_disabled": True,
            "checks": {"read_tool_completed": True},
            "failure_category": None,
        }
        with mock.patch.object(rescue, "local_host", return_value="http://127.0.0.1:11434"), \
             mock.patch.object(rescue, "preflight", return_value=preflight), \
             mock.patch.object(rescue_agent, "evaluate", return_value=evaluation) as run, \
             contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main([
                "rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                "--context", "4096", "--agent-test",
                "--agent-no-auto-compact", "--agent-timeout", "90",
            ])
        self.assertEqual(code, 0)
        run.assert_called_once_with(
            "qwen3.8:27b-mlx", 4096, seconds=90,
            disable_auto_compaction=True,
        )
        self.assertIn("Automatic context compaction: disabled", out.getvalue())


if __name__ == "__main__":
    unittest.main()
