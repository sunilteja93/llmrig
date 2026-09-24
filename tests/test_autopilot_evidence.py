import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _llmrig.autopilot_apply import (
    ActionExecutionReceipt,
    AutopilotReceipt,
    VerificationRecord,
)
from _llmrig.autopilot_evidence import record_receipt_evidence
from _llmrig.autopilot_plan import AutopilotCandidate, AutopilotExecutionPlan


class AutopilotEvidenceTests(unittest.TestCase):
    def test_measured_receipt_is_persisted_without_inventing_prediction(self):
        candidate = AutopilotCandidate(
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
        plan = AutopilotExecutionPlan(
            plan_id="plan-abc",
            model="org/model",
            logical_model_id="org/model",
            machine=(("os", "Darwin"), ("arch", "arm64"), ("cpu", "Apple M4 Max"), ("ram_gib", 48.0)),
            candidates=(candidate,),
            recommendation_status="setup_selected",
            selected_candidate_id="candidate-1",
            recommendation_reason="test",
            actions=(),
        )
        verification = VerificationRecord(
            method_version="race-v2",
            generation_tps=31.0,
            prompt_eval_tps=100.0,
            total_latency_s=4.0,
            measured_runs=2,
        )
        action = ActionExecutionReceipt(
            action_id="a04-verify",
            kind="verify",
            runtime="omlx",
            status="completed",
            detail="verified",
            started_at="2026-09-24T16:00:00+00:00",
            completed_at="2026-09-24T16:00:05+00:00",
            verification=verification,
        )
        receipt = AutopilotReceipt(
            receipt_id="receipt-abc",
            plan_id="plan-abc",
            model="org/model",
            logical_model_id="org/model",
            candidate_id="candidate-1",
            runtime="omlx",
            artifact_id="hf://org/model/mlx",
            artifact_format="MLX",
            quantization="4-bit",
            status="completed",
            started_at="2026-09-24T16:00:00+00:00",
            completed_at="2026-09-24T16:00:05+00:00",
            actions=(action,),
            endpoint="http://127.0.0.1:8000/v1",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "_llmrig.riggraph.evidence_dir", return_value=Path(directory)
        ):
            record = record_receipt_evidence(plan, receipt)
        self.assertIsNotNone(record)
        payload = record.to_dict()
        self.assertEqual(payload["measurement"]["generation_tps"], 31.0)
        self.assertIsNone(payload["prediction"]["generation_tps"])
        self.assertIsNone(payload["calibration"]["generation_tps_delta"])


if __name__ == "__main__":
    unittest.main()
