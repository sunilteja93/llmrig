import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _llmrig.riggraph import (
    CalibrationRecord,
    RigGraphEvidenceRecord,
    calibration_delta,
    load_evidence_records,
    save_evidence_record,
)


class RigGraphV09Tests(unittest.TestCase):
    def test_calibration_delta_requires_comparable_values(self):
        self.assertEqual(calibration_delta(30.0, 33.0), 3.0)
        self.assertIsNone(calibration_delta(None, 33.0))
        self.assertIsNone(calibration_delta(30.0, None))

    def test_evidence_record_separates_prediction_measurement_and_calibration(self):
        record = RigGraphEvidenceRecord.build(
            observed_at="2026-09-24T16:00:00+00:00",
            machine={"os": "Darwin", "arch": "arm64", "cpu": "Apple M4 Max", "ram_gib": 48.0},
            model="mlx-community/Qwen3.5-27B-4bit",
            logical_model_id="mlx-community/Qwen3.5-27B-4bit",
            runtime="omlx",
            artifact_id="hf://mlx-community/Qwen3.5-27B-4bit/model.safetensors",
            artifact_format="MLX",
            quantization="4-bit",
            context_tokens=32768,
            prediction={"generation_tps": 30.0},
            measurement={"generation_tps": 33.0, "total_latency_s": 4.5},
            receipt_id="receipt-abc",
            plan_id="plan-abc",
        )
        payload = record.to_dict()
        self.assertEqual(payload["prediction"]["generation_tps"], 30.0)
        self.assertEqual(payload["measurement"]["generation_tps"], 33.0)
        self.assertEqual(payload["calibration"]["generation_tps_delta"], 3.0)
        self.assertIsNone(payload["calibration"]["prompt_eval_tps_delta"])
        self.assertNotIn("locator", json.dumps(payload).lower())

    def test_persistence_is_local_and_privacy_safe(self):
        record = RigGraphEvidenceRecord.build(
            observed_at="2026-09-24T16:00:00+00:00",
            machine={"os": "Darwin", "arch": "arm64", "cpu": "Apple M4 Max", "ram_gib": 48.0},
            model="org/model",
            logical_model_id="org/model",
            runtime="ollama",
            artifact_id="org/model:latest",
            artifact_format="Ollama",
            quantization=None,
            context_tokens=8192,
            prediction={},
            measurement={"generation_tps": 20.0},
            receipt_id="receipt-def",
            plan_id="plan-def",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "_llmrig.riggraph.evidence_dir", return_value=Path(directory)
        ):
            save_evidence_record(record)
            records = load_evidence_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_id"], record.record_id)


if __name__ == "__main__":
    unittest.main()
