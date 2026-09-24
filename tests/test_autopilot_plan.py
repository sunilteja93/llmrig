import unittest

from _llmrig.autopilot_plan import build_autopilot_plan


class AutopilotPlanTests(unittest.TestCase):
    def machine(self):
        return {
            "os": "Darwin",
            "arch": "arm64",
            "cpu": "Apple M4 Max",
            "ram_gib": 48.0,
        }

    def candidate(self, **overrides):
        value = {
            "candidate_id": "candidate-omlx",
            "logical_model_id": "org/model",
            "artifact_id": "hf://org/model/mlx",
            "runtime": "omlx",
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
                "recommendation": {"state": "recommended", "blockers": [], "unknowns": []},
            },
            "recipe": {
                "status": "known_setup_path",
                "steps": ["Acquire the exact repository artifact."],
                "blockers": [],
                "private_locator_required": False,
            },
        }
        value.update(overrides)
        return value

    def solve_payload(self, *, recommended=True, candidate=None):
        candidate = candidate or self.candidate()
        return {
            "schema_version": "1.1",
            "status": "analyzed",
            "request": {
                "model": "org/model",
                "context_tokens": 32768,
                "local_artifacts": [],
                "verify": False,
            },
            "resolution": {
                "status": "resolved",
                "confidence": "high",
                "logical_model_id": "org/model",
                "model_name": "model",
            },
            "candidates": [candidate],
            "plan": {
                "recommendation_status": "recommended" if recommended else "inconclusive",
                "recommended_candidate_id": candidate["candidate_id"] if recommended else None,
                "reason": (
                    "One evidenced candidate is currently preferred."
                    if recommended
                    else "Multiple viable candidates remain."
                ),
                "unknowns": [],
            },
            "verification": {"status": "not_requested"},
            "unknowns": [],
            "error": None,
        }

    def test_plan_is_deterministic_and_read_only(self):
        first = build_autopilot_plan(self.solve_payload(), self.machine())
        second = build_autopilot_plan(self.solve_payload(), self.machine())

        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.selected_candidate_id, "candidate-omlx")
        self.assertFalse(first.blocked)
        self.assertTrue(first.has_mutations)
        self.assertEqual(
            [action.kind.value for action in first.actions],
            ["acquire_artifact", "load_model", "verify"],
        )
        self.assertTrue(first.actions[0].mutating)
        self.assertFalse(first.actions[-1].mutating)

    def test_inconclusive_solve_does_not_invent_selection(self):
        plan = build_autopilot_plan(
            self.solve_payload(recommended=False), self.machine()
        )

        self.assertIsNone(plan.selected_candidate_id)
        self.assertEqual(plan.actions, ())
        self.assertTrue(plan.blocked)
        self.assertIn(
            "no unique evidenced candidate is selected for apply", plan.blockers
        )

    def test_non_hf_missing_artifact_fails_closed_for_acquisition(self):
        candidate = self.candidate(artifact_id="artifact-without-evidenced-source")
        plan = build_autopilot_plan(
            self.solve_payload(candidate=candidate), self.machine()
        )

        self.assertTrue(plan.blocked)
        acquire = plan.actions[0]
        self.assertEqual(acquire.kind.value, "acquire_artifact")
        self.assertIn(
            "artifact acquisition requires an exact evidenced source",
            acquire.blockers,
        )

    def test_ollama_runtime_native_acquisition_is_evidenced(self):
        candidate = self.candidate(
            candidate_id="candidate-ollama",
            artifact_id="qwen3:8b",
            runtime="ollama",
            artifact_format="Ollama",
        )
        plan = build_autopilot_plan(
            self.solve_payload(candidate=candidate), self.machine()
        )

        self.assertFalse(plan.blocked)
        acquire = plan.actions[0]
        self.assertEqual(acquire.kind.value, "acquire_artifact")
        self.assertEqual(acquire.blockers, ())
        self.assertIn("runtime-native", acquire.evidence[0])

    def test_unavailable_cli_runtime_fails_closed_instead_of_inventing_start(self):
        candidate = self.candidate(
            candidate_id="candidate-mlx",
            runtime="mlx-lm",
            assessments={
                **self.candidate()["assessments"],
                "runtime_availability": {
                    "state": "unavailable",
                    "blockers": ["runtime is not currently available"],
                    "unknowns": [],
                },
            },
        )
        plan = build_autopilot_plan(
            self.solve_payload(candidate=candidate), self.machine()
        )

        start = next(action for action in plan.actions if action.kind.value == "start_runtime")
        self.assertIn(
            "no automatic start path",
            start.blockers[0],
        )
        self.assertTrue(plan.blocked)

    def test_private_artifact_path_is_rejected(self):
        candidate = self.candidate(artifact_id="/Users/private/model.gguf")
        with self.assertRaises(ValueError):
            build_autopilot_plan(
                self.solve_payload(candidate=candidate), self.machine()
            )


if __name__ == "__main__":
    unittest.main()
