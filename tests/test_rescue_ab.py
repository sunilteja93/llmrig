"""Tests for deterministic A/B task grading, scoping, budgets and user opt-in."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from _llmrig import cli, rescue, rescue_ab


def _events(task, workspace, *, correct=True, tool="read", path=None):
    responses = {
        "parity-v1": "n % 2 == 1 tests odd numbers. return n % 2 == 0",
        "nonnegative-clamp-v1": "max(0, -n) negates input. return max(0, n)",
        "prefix-v1": "endswith checks the end, not the prefix. return text.startswith('a')",
    }
    text = responses[task.task_id] if correct else "The code is fine; no fix needed"
    events = [
        {
            "type": "tool_use",
            "part": {
                "tool": tool, "state": {
                    "status": "completed",
                    "input": {"filePath": str(path or (workspace / task.filename))},
                },
            },
        },
        {"type": "text", "part": {"type": "text", "text": text}},
    ]
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")


class RescueAbTests(unittest.TestCase):
    def test_three_distinct_objective_tasks_grade_positive(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for task in rescue_ab.TASKS:
                with self.subTest(task=task.task_id):
                    checks, count = rescue_ab._grade_events(
                        _events(task, directory), task, directory
                    )
                    self.assertEqual(count, 2)
                    self.assertTrue(all(checks.values()), checks)

    def test_answer_without_real_fixture_read_fails(self):
        with tempfile.TemporaryDirectory() as name:
            workspace = Path(name)
            task = rescue_ab.TASKS[0]
            wrong_path = workspace.parent / "math_ops.py"
            checks, _ = rescue_ab._grade_events(
                _events(task, workspace, path=wrong_path), task, workspace
            )
            self.assertFalse(checks["read_tool_completed"])
            self.assertTrue(checks["correct_return_proposed"])

    def test_wrong_answer_and_unsafe_tool_do_not_pass(self):
        with tempfile.TemporaryDirectory() as name:
            workspace = Path(name)
            for task in rescue_ab.TASKS:
                bad, _ = rescue_ab._grade_events(
                    _events(task, workspace, correct=False, tool="bash"), task, workspace
                )
                self.assertFalse(bad["correct_return_proposed"])
                self.assertFalse(bad["no_disallowed_tool_observed"])

    def test_no_raw_tool_output_is_accepted_as_model_answer(self):
        with tempfile.TemporaryDirectory() as name:
            workspace = Path(name)
            task = rescue_ab.TASKS[0]
            stream = (
                json.dumps({
                    "type": "tool_use",
                    "part": {
                        "tool": "read",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": str(workspace / task.filename)},
                            "output": "n % 2 == 1 is odd. return n % 2 == 0",
                        },
                    },
                }) + "\n"
            ).encode("utf-8")
            checks, _ = rescue_ab._grade_events(stream, task, workspace)
            self.assertTrue(checks["read_tool_completed"])
            self.assertFalse(checks["correct_return_proposed"])
            self.assertFalse(checks["bug_explained"])

    def test_real_fixtures_are_isolated_and_model_answer_graded(self):
        def fake_invoke(cmd, cwd, env, seconds):
            self.assertIn("--agent", cmd)
            self.assertIn("--model", cmd)
            self.assertEqual(cmd[cmd.index("--agent") + 1], "rescue-readonly")
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("HTTP_PROXY", env)
            config = json.loads((cwd / "opencode.json").read_text(encoding="utf-8"))
            self.assertEqual(config["permission"]["*"], "deny")
            self.assertEqual(config["permission"]["bash"], "deny")
            filename = next(p.name for p in cwd.iterdir() if p.name != "opencode.json")
            task = next(t for t in rescue_ab.TASKS if t.filename == filename)
            self.assertEqual((cwd / task.filename).read_text(encoding="utf-8"), task.source)
            return ("exited", 0, _events(task, cwd), False)

        with mock.patch.object(rescue_ab.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_invoke", side_effect=fake_invoke):
            result = rescue_ab.compare("qwen3.8:27b-mlx", 4096, 8192, repeats=2)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["planned_agent_sessions"], 12)
        self.assertEqual(result["completed_agent_sessions"], 12)
        self.assertEqual(result["summary"]["baseline"]["observed_passes"], 6)
        self.assertEqual(result["summary"]["candidate"]["observed_passes"], 6)
        self.assertEqual(result["observed_comparison"]["assessment"], "no_observed_difference")
        self.assertEqual(result["recommendation"], "no_verified_configuration_improvement")
        self.assertEqual(result["observed_comparison"]["evidence_strength"],
                         "exploratory_not_statistically_validated")
        for key in ("Users/", "def is_even", "n % 2", "secret /"):
            self.assertNotIn(key, json.dumps(result))

    def test_one_candidate_failure_is_observation_not_recommendation(self):
        def fake_invoke(cmd, cwd, env, seconds):
            filename = next(p.name for p in cwd.iterdir() if p.name != "opencode.json")
            task = next(t for t in rescue_ab.TASKS if t.filename == filename)
            config = json.loads((cwd / "opencode.json").read_text(encoding="utf-8"))
            ctx = config["provider"]["ollama"]["models"]["qwen3.8:27b-mlx"]["limit"]["context"]
            failed = ctx == 8192 and task.task_id == "prefix-v1"
            return ("exited", 0, _events(task, cwd, correct=not failed), False)

        with mock.patch.object(rescue_ab.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_invoke", side_effect=fake_invoke):
            result = rescue_ab.compare("qwen3.8:27b-mlx", 4096, 8192, repeats=1)
        self.assertEqual(result["summary"]["baseline"]["observed_passes"], 3)
        self.assertEqual(result["summary"]["candidate"]["observed_passes"], 2)
        self.assertEqual(result["observed_comparison"]["candidate_minus_baseline_pass_rate"], -0.3333)
        self.assertEqual(result["recommendation"], "no_verified_configuration_improvement")

    def test_violation_stops_remaining_agent_sessions(self):
        calls = []
        def violation(model, context, task, seconds):
            calls.append((context, task.task_id))
            return {
                "task_id": task.task_id,
                "context_tokens": context,
                "status": "failed",
                "failure_category": "test_boundary_violation",
            }

        with mock.patch.object(rescue_ab.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_trial", side_effect=violation):
            result = rescue_ab.compare("qwen3.8:27b-mlx", 4096, 8192, repeats=3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "stopped_on_boundary_violation")
        self.assertIsNone(result["observed_comparison"])

    def test_invalid_budget_and_repeats_rejected(self):
        for overrides in ({"repeats": 0}, {"repeats": 4}, {"seconds": 0},
                          {"budget_seconds": 60}, {"budget_seconds": 1500}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                rescue_ab.compare("qwen3.8:27b-mlx", 4096, 8192, **overrides)

    def test_separate_opt_in_and_preflight_not_overstated(self):
        preflight = {
            "status": "ready_for_probe", "mode": "preflight", "agent_executable_detected": True,
            "agent": "opencode", "model": "qwen3.8:27b-mlx",
            "context_tokens": 4096, "blockers": [],
            "unknowns": ["OpenCode tool execution and task completion have not been tested"],
            "probes": [], "comparison": None,
        }
        mock_result = {
            "status": "completed", "summary": {
                "baseline": {"context_tokens": 4096, "observed_passes": 3, "observed_trials": 3},
                "candidate": {"context_tokens": 8192, "observed_passes": 3, "observed_trials": 3},
            },
            "observed_comparison": {"assessment": "no_observed_difference"},
        }
        with mock.patch.object(rescue, "local_host", return_value="http://127.0.0.1:11434"), \
             mock.patch.object(rescue, "preflight", side_effect=lambda *a: dict(preflight, unknowns=list(preflight["unknowns"]))), \
             mock.patch.object(rescue_ab, "compare", return_value=mock_result) as run:
            not_invoked = rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 2, None, False)
            self.assertEqual(not_invoked["status"], "ready_for_probe")
            run.assert_not_called()
            with self.assertRaises(rescue.RescueError):
                rescue.rescue("opencode", "qwen3.8:27b-mlx", 4096, 2, None, False, ab_test=True)
            measured = rescue.rescue(
                "opencode", "qwen3.8:27b-mlx", 4096, 2, 8192, False,
                ab_test=True, ab_repeats=1
            )
            self.assertEqual(measured["status"], "agent_ab_completed")
            self.assertEqual(measured["mode"], "measured_agent_ab")
            self.assertNotIn("have not been tested", json.dumps(measured))
            self.assertIn("exploratory", rescue.format_terminal(measured).lower())
            run.assert_called_once()

    def test_cli_ab_report(self):
        fake = {
            "status": "agent_ab_completed", "mode": "measured_agent_ab",
            "agent_executable_detected": True, "agent": "opencode",
            "model": "qwen3.8:27b-mlx", "context_tokens": 4096,
            "blockers": [], "unknowns": [], "probes": [], "comparison": None,
            "agent_ab_evaluation": {
                "status": "completed", "summary": {
                    "baseline": {"context_tokens": 4096, "observed_passes": 3, "observed_trials": 3},
                    "candidate": {"context_tokens": 8192, "observed_passes": 3, "observed_trials": 3},
                },
                "observed_comparison": {"assessment": "no_observed_difference"},
            },
        }
        with mock.patch.object(rescue, "rescue", return_value=fake):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main([
                    "rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                    "--context", "4096", "--compare-context", "8192", "--ab-test",
                    "--ab-repeats", "1", "--ab-budget", "300",
                ]), 0)
            self.assertIn("A/B suite: completed", out.getvalue())
            self.assertIn("No repair was attempted", out.getvalue())


if __name__ == "__main__":
    unittest.main()
