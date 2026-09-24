import argparse
import unittest
from unittest import mock

from _llmrig import cli
from _llmrig.autopilot_actions import ActionKind, PlannedAction
from _llmrig.autopilot_plan import AutopilotCandidate, AutopilotExecutionPlan


class _Legacy:
    RACE_CONTEXT = 4096

    class SolveInputError(Exception):
        pass

    class SolveEngineError(Exception):
        pass


class AutopilotVerifyCliTests(unittest.TestCase):
    def candidate(self):
        return AutopilotCandidate(
            candidate_id="candidate-1",
            runtime="omlx",
            artifact_id="hf://org/model/mlx",
            artifact_format="MLX",
            quantization="4-bit",
            context_tokens=32768,
            discovery="discovered",
            compatibility="compatible",
            runtime_availability="available",
            local_availability="available",
            execution="executable",
            measurement_capability="measurable",
            recipe_status="ready",
            recipe_steps=(),
            blockers=(),
            unknowns=(),
        )

    def plan(self, *, selected=True, actions=None):
        candidate = self.candidate()
        if actions is None:
            actions = (
                PlannedAction(
                    "a04-verify",
                    ActionKind.VERIFY,
                    candidate.runtime,
                    "Verify the exact prior configuration.",
                    False,
                    ("verification-only regression",),
                ),
            )
        return AutopilotExecutionPlan(
            plan_id="plan-current",
            model="org/model",
            logical_model_id="org/model",
            machine=(("os", "Darwin"), ("arch", "arm64"), ("cpu", "Apple M4 Max"), ("ram_gib", 48.0)),
            candidates=(candidate,),
            recommendation_status="ready" if selected else "inconclusive",
            selected_candidate_id=candidate.candidate_id if selected else None,
            recommendation_reason="test",
            actions=actions,
            blockers=() if selected else ("no unique evidenced candidate is selected for apply",),
        )

    def receipt(self):
        return {
            "schema_version": "0.9",
            "model": "org/model",
            "candidate": {
                "candidate_id": "candidate-1",
                "runtime": "omlx",
                "artifact_id": "hf://org/model/mlx",
                "artifact_format": "MLX",
                "quantization": "4-bit",
            },
        }

    def test_verify_reobserves_and_reuses_only_exact_unique_candidate(self):
        args = argparse.Namespace(receipt="latest", json=True)
        plan = self.plan()
        with mock.patch.object(cli, "load_receipt", return_value=self.receipt()), mock.patch.object(
            cli, "_build_plan", return_value=plan
        ) as build, mock.patch.object(cli, "_apply", return_value=0) as apply:
            self.assertEqual(cli.command_verify(args, _Legacy), 0)
        build.assert_called_once_with("org/model", None, _Legacy)
        applied_plan = apply.call_args.args[0]
        self.assertFalse(applied_plan.has_mutations)
        self.assertEqual(
            [action.kind.value for action in applied_plan.actions],
            ["verify"],
        )
        selected = next(
            item
            for item in applied_plan.candidates
            if item.candidate_id == applied_plan.selected_candidate_id
        )
        self.assertEqual(selected.context_tokens, 4096)
        apply.assert_called_once_with(applied_plan, args, _Legacy, False)
        self.assertTrue(args.yes)

    def test_verify_fails_closed_when_current_selection_is_not_unique(self):
        args = argparse.Namespace(receipt="latest", json=True)
        with mock.patch.object(cli, "load_receipt", return_value=self.receipt()), mock.patch.object(
            cli, "_build_plan", return_value=self.plan(selected=False)
        ), mock.patch.object(cli, "_apply") as apply:
            self.assertEqual(cli.command_verify(args, _Legacy), 2)
        apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
