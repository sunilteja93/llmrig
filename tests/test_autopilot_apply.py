import unittest
from unittest import mock

from _llmrig import autopilot_apply
from _llmrig.acquisition import AcquisitionError, AcquisitionRecord, AcquiredArtifact
from _llmrig.autopilot_plan import build_autopilot_plan


class _Legacy:
    def validated_local_locator(self, runtime, locator):
        if runtime != "mlx-lm":
            raise ValueError("unexpected runtime")
        return locator


class AutopilotApplyTests(unittest.TestCase):
    def plan(self):
        candidate = {
            "candidate_id": "mlx-lm:hf://org/model/mlx",
            "artifact_id": "hf://org/model/mlx",
            "runtime": "mlx-lm",
            "artifact_format": "MLX",
            "quantization": "4-bit",
            "context_tokens": 32768,
            "assessments": {
                "discovery": {"state": "discovered", "blockers": [], "unknowns": []},
                "compatibility": {"state": "compatible", "blockers": [], "unknowns": []},
                "runtime_availability": {"state": "available", "blockers": [], "unknowns": []},
                "local_availability": {"state": "not_available", "blockers": [], "unknowns": []},
                "execution": {"state": "not_executable", "blockers": [], "unknowns": []},
                "measurement_capability": {"state": "measurable", "blockers": [], "unknowns": []},
                "measurement": {"state": "not_requested", "blockers": [], "unknowns": []},
                "recommendation": {"state": "inconclusive", "blockers": [], "unknowns": []},
            },
            "recipe": {"status": "unknown", "steps": [], "blockers": []},
        }
        payload = {
            "status": "analyzed",
            "request": {"model": "org/model"},
            "resolution": {"logical_model_id": "org/model"},
            "candidates": [candidate],
            "plan": {
                "recommendation_status": "inconclusive",
                "recommended_candidate_id": None,
                "reason": "No configuration is currently known to be runnable.",
                "unknowns": [],
            },
            "unknowns": [],
        }
        return build_autopilot_plan(
            payload,
            {"os": "Darwin", "arch": "arm64", "cpu": "Apple M4 Max", "ram_gib": 48.0},
        )

    def acquired(self):
        record = AcquisitionRecord(
            artifact_id="hf://org/model/mlx",
            repository_id="org/model",
            revision="a" * 40,
            runtime="mlx-lm",
            acquisition_kind="huggingface-snapshot",
            status="completed",
            completed_at="2026-09-24T15:00:00+00:00",
        )
        return AcquiredArtifact(record, "/private/local/model")

    def test_mutating_plan_requires_explicit_approval(self):
        plan = self.plan()
        self.assertTrue(plan.has_mutations)
        with self.assertRaises(PermissionError):
            autopilot_apply.apply_autopilot_plan(
                plan,
                _Legacy(),
                explicit_user_approval=False,
                persist_receipt=False,
            )

    def test_successful_apply_receipt_never_serializes_private_locator(self):
        verification = autopilot_apply.VerificationRecord(
            "race-v2", 31.0, 100.0, 4.2, 2, ("performance is not model quality",)
        )
        with mock.patch.object(
            autopilot_apply,
            "acquire_huggingface_artifact",
            return_value=self.acquired(),
        ), mock.patch.object(
            autopilot_apply,
            "_verify_candidate",
            return_value=verification,
        ):
            receipt = autopilot_apply.apply_autopilot_plan(
                self.plan(),
                _Legacy(),
                explicit_user_approval=True,
                persist_receipt=False,
            )

        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.verification, verification)
        self.assertTrue(all(action.success for action in receipt.actions))
        rendered = str(receipt.to_dict())
        self.assertNotIn("/private/local/model", rendered)
        self.assertNotIn("locator", rendered.lower())
        self.assertIn("a" * 40, rendered)

    def test_failed_acquisition_stops_all_later_actions(self):
        with mock.patch.object(
            autopilot_apply,
            "acquire_huggingface_artifact",
            side_effect=AcquisitionError("private failure detail"),
        ):
            receipt = autopilot_apply.apply_autopilot_plan(
                self.plan(),
                _Legacy(),
                explicit_user_approval=True,
                persist_receipt=False,
            )

        self.assertEqual(receipt.status, "blocked")
        self.assertEqual(receipt.actions[0].status, "blocked")
        self.assertTrue(all(item.status == "skipped" for item in receipt.actions[1:]))
        self.assertNotIn("private failure detail", str(receipt.to_dict()))


if __name__ == "__main__":
    unittest.main()
