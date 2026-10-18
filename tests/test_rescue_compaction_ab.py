"""Controls for paired OpenCode compaction A/B experiments (no live inference)."""
import contextlib
import io
import json
import unittest
from unittest import mock

from _llmrig import cli, rescue, rescue_ab, rescue_compaction_ab


def trial(model, context, task, seconds, disable_auto_compaction=False):
    assert model == "qwen3.8:27b-mlx"
    assert context == 4096
    assert seconds == 90
    checks = {
        "read_tool_completed": True,
        "bug_explained": True,
        "correct_return_proposed": True,
        "no_disallowed_tool_observed": True,
        "no_error_event": True,
        "valid_event_stream": True,
        "process_exit_zero": disable_auto_compaction,
        "fixture_unchanged": True,
    }
    return {
        "task_id": task.task_id,
        "context_tokens": context,
        "auto_compaction_disabled": disable_auto_compaction,
        "status": "passed" if disable_auto_compaction else "failed",
        "checks": checks,
        "failure_category": None if disable_auto_compaction else "agent_timeout",
    }


class ControlledCompactionAbTests(unittest.TestCase):
    def test_alternating_order_and_matching_task_pairs(self):
        seen = []
        def measure(model, context, task, seconds, disable_auto_compaction=False):
            seen.append((task.task_id, disable_auto_compaction, context, seconds))
            return trial(model, context, task, seconds, disable_auto_compaction)
        with mock.patch.object(rescue_compaction_ab.shutil, "which",
                               return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_trial", side_effect=measure):
            report = rescue_compaction_ab.compare_compaction(
                "qwen3.8:27b-mlx", 4096, repeats=2, seconds=90, budget_seconds=1200
            )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["executed_agent_sessions"], 12)
        self.assertEqual(report["planned_agent_sessions"], 12)
        self.assertEqual(
            [(t["condition"], t["pair_number"]) for t in report["trials"][:6]],
            [("compaction_default", 1), ("compaction_disabled", 1),
             ("compaction_disabled", 2), ("compaction_default", 2),
             ("compaction_default", 3), ("compaction_disabled", 3)],
        )
        self.assertEqual(
            [(t["condition"], t["pair_number"]) for t in report["trials"][6:12]],
            [("compaction_disabled", 4), ("compaction_default", 4),
             ("compaction_default", 5), ("compaction_disabled", 5),
             ("compaction_disabled", 6), ("compaction_default", 6)],
        )
        self.assertEqual({x[2] for x in seen}, {4096})
        self.assertEqual({x[3] for x in seen}, {90})
        b, c = report["groups"]["baseline"], report["groups"]["candidate"]
        self.assertEqual(b["strict_passes"], 0)
        self.assertEqual(b["timeouts_with_fix"], 6)
        self.assertEqual(c["strict_passes"], 6)
        self.assertEqual(c["timeouts"], 0)
        self.assertEqual(
            report["observed_comparison"]["candidate_minus_baseline_strict_pass_rate"], 1.0
        )
        self.assertEqual(report["recommendation"], "no_automatic_or_global_compaction_change")
        self.assertNotIn("Users/", json.dumps(report))

    def test_abort_safely_on_tool_boundary_violation(self):
        calls = []
        def violation(model, context, task, seconds, disable_auto_compaction=False):
            calls.append(task.task_id)
            result = trial(model, context, task, seconds, disable_auto_compaction)
            result["status"] = "failed"
            result["failure_category"] = "test_boundary_violation"
            result["checks"]["no_disallowed_tool_observed"] = False
            return result
        with mock.patch.object(rescue_compaction_ab.shutil, "which",
                               return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_trial", side_effect=violation):
            result = rescue_compaction_ab.compare_compaction(
                "qwen3.8:27b-mlx", 4096, repeats=2, seconds=90, budget_seconds=1200
            )
        self.assertEqual(result["executed_agent_sessions"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "stopped_on_boundary_violation")
        self.assertIsNone(result["observed_comparison"])

    def test_budget_never_runs_a_truncated_pair(self):
        with mock.patch.object(rescue_compaction_ab.shutil, "which",
                               return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_trial") as run:
            result = rescue_compaction_ab.compare_compaction(
                "qwen3.8:27b-mlx", 4096, repeats=1, seconds=90, budget_seconds=180
            )
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["executed_agent_sessions"], 0)
        self.assertIsNone(result["observed_comparison"])
        run.assert_not_called()

    def test_refuse_bad_bounds_and_missing_agent(self):
        with mock.patch.object(rescue_compaction_ab.shutil, "which",
                               return_value="/usr/bin/opencode"):
            for settings in ({"repeats": 0}, {"repeats": 4}, {"seconds": 19},
                             {"seconds": 181}, {"budget_seconds": 119},
                             {"budget_seconds": 1201}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    rescue_compaction_ab.compare_compaction(
                        "qwen3.8:27b-mlx", 4096, **settings
                    )
        with mock.patch.object(rescue_compaction_ab.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "not found"):
                rescue_compaction_ab.compare_compaction("qwen3.8:27b-mlx", 4096)

    def test_real_trial_config_only_differs_by_compaction(self):
        calls = []
        def invoke(command, cwd, env, seconds):
            cfg = json.loads((cwd / "opencode.json").read_text(encoding="utf-8"))
            calls.append(cfg)
            self.assertEqual(cfg["provider"]["ollama"]["models"]
                             ["qwen3.8:27b-mlx"]["limit"]["context"], 4096)
            self.assertEqual(cfg["permission"]["bash"], "deny")
            self.assertEqual(cfg["permission"]["edit"], "deny")
            self.assertEqual(seconds, 90)
            self.assertNotIn("OPENAI_API_KEY", env)
            fixture = next(x for x in rescue_ab.TASKS if (cwd / x.filename).exists())
            stream = (
                json.dumps({"type": "tool_use", "part": {
                    "tool": "read", "state": {
                        "status": "completed",
                        "input": {"filePath": str(cwd / fixture.filename)}
                    }
                }}) + "\n" +
                json.dumps({"type": "text", "part": {"text": {
                    "parity-v1": "n % 2 == 1 checks odd numbers; return n % 2 == 0",
                    "nonnegative-clamp-v1": "max(0, -n) negates sign; return max(0, n)",
                    "prefix-v1": "endswith checks the end; return text.startswith('a')",
                }[fixture.task_id]}}) + "\n"
            ).encode("utf-8")
            return ("exited", 0, stream, False)
        with mock.patch.object(rescue_ab.shutil, "which", return_value="/usr/bin/opencode"), \
             mock.patch.object(rescue_ab, "_invoke", side_effect=invoke):
            first = rescue_ab._trial("qwen3.8:27b-mlx", 4096, rescue_ab.TASKS[0], 90)
            second = rescue_ab._trial(
                "qwen3.8:27b-mlx", 4096, rescue_ab.TASKS[0], 90,
                disable_auto_compaction=True,
            )
        self.assertEqual(first["status"], "passed")
        self.assertEqual(second["status"], "passed")
        self.assertEqual(first["auto_compaction_disabled"], False)
        self.assertEqual(second["auto_compaction_disabled"], True)
        self.assertNotIn("compaction", calls[0])
        self.assertEqual(calls[1]["compaction"], {"auto": False})
        without = dict(calls[1]); without.pop("compaction")
        self.assertEqual(without, calls[0])

    def test_cli_explicit_opt_in_and_report_display(self):
        preflight = {
            "status": "ready_for_probe",
            "agent_executable_detected": True,
            "agent": "opencode", "model": "qwen3.8:27b-mlx",
            "context_tokens": 4096, "unknowns": [
                "OpenCode tool execution and task completion have not been tested"
            ], "blockers": [], "probes": [], "comparison": None,
        }
        run_result = {
            "status": "completed",
            "groups": {
                "baseline": {"strict_passes": 0, "trials": 3, "process_exits": 0,
                             "timeouts": 3, "fix_evidence": 3,
                             "explanation_only_rubric_misses": 0},
                "candidate": {"strict_passes": 3, "trials": 3, "process_exits": 3,
                              "timeouts": 0, "fix_evidence": 3,
                              "explanation_only_rubric_misses": 0},
            },
            "observed_comparison": {
                "direction": "compaction_disabled_observed_more_passes"
            },
        }
        with mock.patch.object(rescue, "preflight", return_value=preflight), \
             mock.patch.object(rescue, "local_host",
                               return_value="http://127.0.0.1:11434"), \
             mock.patch.object(rescue_compaction_ab, "compare_compaction",
                               return_value=run_result) as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = cli.main([
                "rescue", "--agent", "opencode", "--model", "qwen3.8:27b-mlx",
                "--context", "4096", "--compaction-ab", "--compaction-repeats",
                "1", "--agent-timeout", "90", "--ab-budget", "600",
            ])
        self.assertEqual(code, 0)
        run.assert_called_once_with(
            "qwen3.8:27b-mlx", 4096, repeats=1,
            seconds=90, budget_seconds=600
        )
        self.assertIn("Compaction default: 0/3", output.getvalue())
        self.assertIn("Compaction disabled: 3/3", output.getvalue())
        self.assertIn("not causal", output.getvalue())
        self.assertNotIn("have not been tested", output.getvalue())

    def test_compaction_ab_mutually_exclusive_with_other_modes(self):
        for options in [
            {"probe": True},
            {"agent_test": True},
            {"ab_test": True},
            {"compare_context": 8192},
            {"agent_steps": 4},
            {"agent_no_auto_compact": True},
        ]:
            with self.subTest(options=options), self.assertRaises(rescue.RescueError):
                rescue.rescue(
                    "opencode", "qwen3.8:27b-mlx", 4096, 3,
                    options.get("compare_context"), options.get("probe", False),
                    compaction_ab=True, **{
                        k: v for k, v in options.items()
                        if k not in ("probe", "compare_context")
                    },
                )


if __name__ == "__main__":
    unittest.main()
