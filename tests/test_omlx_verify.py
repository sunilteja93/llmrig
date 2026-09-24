from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

import llmrig
from _llmrig import cli, runtime_bridge
from _llmrig.inventory import InventoryRecord, InventoryTarget
from _llmrig.omlx_api import OmlxHfDownloadRecord, OmlxMeasurement, OmlxModelStatus
from _llmrig.omlx_verify import (
    OmlxExecutionAdapter,
    inventory_targets_from_statuses,
)
from _llmrig.runtime_adapters import RuntimeEvidence, RuntimeProbe


class OmlxVerifyTests(unittest.TestCase):
    def mac_profile(self):
        return {
            "os": "Darwin",
            "arch": "arm64",
            "cpu": "Apple M4 Max",
            "ram_gib": 48.0,
            "gpus": [
                {
                    "name": "Apple M4 Max",
                    "vram_gb": None,
                    "backend": "Metal / unified memory",
                }
            ],
            "disk": {"free_gib": 500.0},
        }

    def probe(
        self,
        runtime,
        *,
        service_available,
        version,
        cli_path,
    ):
        return RuntimeProbe(
            runtime=runtime,
            installed=True,
            service_available=service_available,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=("MLX",),
            supported_platforms=("Darwin",),
            supported_architectures=("arm64", "aarch64"),
            execution_api=(
                "OpenAI-compatible /v1" if runtime == "omlx" else "local CLI"
            ),
            evidence=(
                RuntimeEvidence(
                    "verified-local-runtime",
                    "test probe",
                    "the runtime was observed in the test fixture",
                ),
            ),
        )

    def resolution(self):
        model = llmrig.Model("org/model", "Model", 7.0, ("text",))
        artifact = llmrig.ModelArtifact(
            "hf://org/model/mlx",
            model.model_id,
            None,
            "MLX",
            4.0,
            32768,
            ("Darwin",),
            size_bytes=4_000_000_000,
            evidence=(
                llmrig.RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face repository metadata",
                    "the repository explicitly identifies an MLX artifact",
                ),
            ),
        )
        return llmrig.ModelResolution(
            "resolved",
            model,
            (artifact,),
            llmrig.Confidence.HIGH,
            evidence=(
                llmrig.RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face repository metadata",
                    "the repository metadata resolved successfully",
                ),
            ),
        )

    def test_inventory_accepts_exact_runtime_reported_source_repo(self):
        statuses = (
            OmlxModelStatus(
                "org--model", "org/model", "hf_cache", True, 32768
            ),
        )
        targets = inventory_targets_from_statuses(llmrig, "org/model", statuses)
        self.assertEqual(len(targets), 1)
        record = targets[0].record
        self.assertEqual(record.runtime, "omlx")
        self.assertEqual(record.public_artifact_id, "org--model")
        self.assertEqual(record.logical_model_id, "org/model")
        self.assertEqual(record.artifact_format, "MLX")
        self.assertEqual(record.association_kind, "runtime_reported_hf_source")
        self.assertFalse(record.identity_attested)
        self.assertEqual(record.identity_confidence, llmrig.Confidence.HIGH)
        serialized = json.dumps(record.to_dict())
        self.assertNotIn("model_path", serialized)
        self.assertNotIn("/Users/", serialized)

    def test_inventory_rejects_display_id_without_source_provenance(self):
        statuses = (
            OmlxModelStatus("model", None, "local", True, 32768),
        )
        self.assertEqual(
            inventory_targets_from_statuses(llmrig, "org/model", statuses), ()
        )

    def test_inventory_accepts_unique_completed_dashboard_download_provenance(self):
        statuses = (
            OmlxModelStatus("model", None, "local", False, 32768),
        )
        downloads = (
            OmlxHfDownloadRecord("org/model", "completed"),
        )
        targets = inventory_targets_from_statuses(
            llmrig,
            "org/model",
            statuses,
            downloads,
        )
        self.assertEqual(len(targets), 1)
        record = targets[0].record
        self.assertEqual(record.public_artifact_id, "model")
        self.assertEqual(record.logical_model_id, "org/model")
        self.assertEqual(record.association_kind, "runtime_reported_hf_download")
        self.assertEqual(record.identity_confidence, llmrig.Confidence.HIGH)
        self.assertTrue(
            any(
                item.source == "oMLX completed Hugging Face download registry"
                for item in record.identity_evidence
            )
        )
        serialized = json.dumps(record.to_dict())
        self.assertNotIn("model_path", serialized)
        self.assertNotIn("/Users/", serialized)

    def test_inventory_rejects_incomplete_dashboard_download(self):
        statuses = (
            OmlxModelStatus("model", None, "local", False, 32768),
        )
        downloads = (
            OmlxHfDownloadRecord("org/model", "downloading"),
        )
        self.assertEqual(
            inventory_targets_from_statuses(
                llmrig,
                "org/model",
                statuses,
                downloads,
            ),
            (),
        )

    def test_inventory_rejects_ambiguous_completed_download_leaf(self):
        statuses = (
            OmlxModelStatus("model", None, "local", False, 32768),
        )
        downloads = (
            OmlxHfDownloadRecord("org/model", "completed"),
            OmlxHfDownloadRecord("other/model", "completed"),
        )
        self.assertEqual(
            inventory_targets_from_statuses(
                llmrig,
                "org/model",
                statuses,
                downloads,
            ),
            (),
        )

    def test_inventory_fails_closed_when_source_maps_to_multiple_server_ids(self):
        statuses = (
            OmlxModelStatus("model-a", "org/model", "hf_cache", True, 32768),
            OmlxModelStatus("model-b", "org/model", "hf_cache", False, 32768),
        )
        self.assertEqual(
            inventory_targets_from_statuses(llmrig, "org/model", statuses), ()
        )

    def test_execution_adapter_uses_server_metrics_and_preserves_warning(self):
        configuration = llmrig.RaceConfiguration(
            "org/model",
            "omlx",
            "org--model",
            "MLX",
            None,
            "omlx 1.2.3",
            True,
        )
        target = llmrig.ExecutionTarget(configuration, "org--model")
        workload = llmrig.RaceWorkload(
            prompt="deterministic prompt",
            context=4096,
            num_predict=16,
            runs=2,
            warmup_runs=1,
            warmup_num_predict=4,
        )

        def measured(model_id, prompt, max_tokens, **kwargs):
            self.assertEqual(model_id, "org--model")
            self.assertEqual(prompt, "deterministic prompt")
            return OmlxMeasurement(
                model_id="org--model",
                prompt_tokens=20,
                completion_tokens=max_tokens,
                generation_tps=50.0,
                prompt_tps=120.0,
                total_time_s=2.6,
                prompt_eval_duration_s=0.5,
                generation_duration_s=2.0,
                time_to_first_token_s=0.6,
            )

        with mock.patch("_llmrig.omlx_verify.measure_completion", side_effect=measured):
            competitor = OmlxExecutionAdapter(llmrig).benchmark(target, workload)

        self.assertEqual(competitor.execution_status, "success")
        self.assertEqual(competitor.generation_tps, 50.0)
        self.assertEqual(competitor.prompt_eval_tps, 120.0)
        self.assertEqual(competitor.total_latency_s, 2.5)
        self.assertEqual(competitor.generated_tokens, 32)
        self.assertEqual(competitor.measured_runs, 2)
        self.assertTrue(
            any("per-request KV-context" in warning for warning in competitor.warnings)
        )
        self.assertTrue(
            all(
                sample.get("measurement_source") == "omlx-server-usage"
                for sample in competitor.raw_samples
            )
        )

    def test_execution_adapter_rejects_response_identity_change(self):
        configuration = llmrig.RaceConfiguration(
            "org/model", "omlx", "org--model", "MLX", None, None, True
        )
        target = llmrig.ExecutionTarget(configuration, "org--model")
        workload = llmrig.RaceWorkload(
            prompt="prompt", num_predict=8, runs=1, warmup_runs=0
        )
        measurement = OmlxMeasurement(
            model_id="different-model",
            prompt_tokens=10,
            completion_tokens=8,
            generation_tps=40.0,
            prompt_tps=100.0,
            total_time_s=1.0,
            prompt_eval_duration_s=0.2,
            generation_duration_s=0.8,
            time_to_first_token_s=0.25,
        )
        with mock.patch(
            "_llmrig.omlx_verify.measure_completion", return_value=measurement
        ):
            with self.assertRaisesRegex(RuntimeError, "identity did not match"):
                OmlxExecutionAdapter(llmrig).benchmark(target, workload)

    def test_solve_verify_measures_exact_provenance_omlx_candidate(self):
        omlx_probe = self.probe(
            "omlx",
            service_available=True,
            version="omlx 1.2.3",
            cli_path="~/.omlx/bin/omlx",
        )
        mlx_probe = self.probe(
            "mlx-lm",
            service_available=None,
            version="0.30.0",
            cli_path="/usr/local/bin/mlx_lm.generate",
        )
        omlx_target = inventory_targets_from_statuses(
            llmrig,
            "org/model",
            (
                OmlxModelStatus(
                    "org--model", "org/model", "hf_cache", True, 32768
                ),
            ),
        )[0]
        association = llmrig.RecommendationEvidence(
            "user-supplied-local-association",
            "test fixture",
            "the test associates this private MLX target with the requested model",
        )
        local_structure = llmrig.RecommendationEvidence(
            "verified-local-artifact-structure",
            "test fixture",
            "the test fixture represents a structurally valid local MLX artifact",
        )
        mlx_record = InventoryRecord(
            runtime="mlx-lm",
            public_artifact_id="local:mlx-lm:user-supplied",
            artifact_fingerprint=None,
            logical_model_id="org/model",
            artifact_format="MLX",
            quantization=None,
            association_kind="user_supplied",
            identity_attested=False,
            identity_confidence=llmrig.Confidence.UNKNOWN,
            identity_evidence=(association,),
            evidence=(local_structure,),
            unknowns=(
                "artifact content identity is not independently verified",
                "artifact content digest is unknown",
                "quantization is unknown",
            ),
        )
        mlx_target = InventoryTarget(mlx_record, "/private/test/mlx-model")

        class FakeMlxExecutionAdapter:
            runtime = "mlx-lm"

            def benchmark(self, target, workload):
                configuration = target.configuration
                return llmrig.RaceCompetitor(
                    logical_model_id=configuration.logical_model_id,
                    runtime=configuration.runtime,
                    artifact_id=configuration.artifact_id,
                    artifact_fingerprint=configuration.artifact_fingerprint,
                    artifact_format=configuration.artifact_format,
                    quantization=configuration.quantization,
                    runtime_version=configuration.runtime_version,
                    execution_status="success",
                    generation_tps=40.0,
                    prompt_eval_tps=100.0,
                    total_latency_s=3.5,
                    generated_tokens=256,
                    measured_runs=2,
                    generation_samples=2,
                    prompt_eval_samples=2,
                    latency_samples=2,
                    timestamp="2026-09-24T00:00:00+00:00",
                    warnings=("performance measurement does not establish model quality",),
                )

        def measured(model_id, prompt, max_tokens, **kwargs):
            return OmlxMeasurement(
                model_id=model_id,
                prompt_tokens=20,
                completion_tokens=max_tokens,
                generation_tps=55.0,
                prompt_tps=130.0,
                total_time_s=2.6,
                prompt_eval_duration_s=0.4,
                generation_duration_s=2.0,
                time_to_first_token_s=0.5,
            )

        output = io.StringIO()
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ), mock.patch.object(
            llmrig.HF_SOURCE, "resolve", return_value=self.resolution()
        ), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=[]
        ), mock.patch.object(
            llmrig,
            "_autopilot_explicit_native_inventory",
            return_value=(mlx_target,),
        ), mock.patch.object(
            llmrig, "MlxExecutionAdapter", FakeMlxExecutionAdapter
        ), mock.patch.object(
            runtime_bridge,
            "probe_runtimes",
            return_value=(omlx_probe, mlx_probe),
        ), mock.patch(
            "_llmrig.omlx_verify.observe_omlx_inventory",
            return_value=(omlx_target,),
        ), mock.patch(
            "_llmrig.omlx_verify.measure_completion", side_effect=measured
        ), contextlib.redirect_stdout(output):
            code = cli.main(["solve", "org/model", "--verify", "--json"])

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["verification"]["status"], "completed")
        self.assertGreaterEqual(len(payload["verification"]["candidate_ids"]), 2)
        omlx_candidates = [
            candidate
            for candidate in payload["candidates"]
            if candidate["runtime"] == "omlx"
            and candidate["artifact_id"] == "org--model"
        ]
        self.assertEqual(len(omlx_candidates), 1)
        candidate = omlx_candidates[0]
        self.assertEqual(
            candidate["assessments"]["local_availability"]["state"], "available"
        )
        self.assertEqual(
            candidate["assessments"]["execution"]["state"], "executable"
        )
        self.assertEqual(
            candidate["assessments"]["measurement_capability"]["state"],
            "measurable",
        )
        self.assertEqual(candidate["assessments"]["measurement"]["state"], "measured")
        self.assertEqual(candidate["measurement_result"]["generation_tps"], 55.0)
        self.assertEqual(
            payload["verification"]["recommendation"]["recommended_candidate_id"],
            candidate["candidate_id"],
        )


if __name__ == "__main__":
    unittest.main()
