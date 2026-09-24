from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

import llmrig
from _llmrig import cli, runtime_bridge
from _llmrig.runtime_adapters import RuntimeEvidence, RuntimeProbe


class OmlxSolveCliTests(unittest.TestCase):
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

    def runtime_probe(
        self,
        runtime,
        *,
        installed,
        service_available,
        version=None,
        cli_path=None,
        formats=(),
        api=None,
    ):
        return RuntimeProbe(
            runtime=runtime,
            installed=installed,
            service_available=service_available,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=formats,
            supported_platforms=("Darwin",),
            supported_architectures=("arm64", "aarch64"),
            execution_api=api,
            evidence=(
                RuntimeEvidence(
                    "verified-local-runtime",
                    "test runtime probe",
                    "runtime state was observed",
                ),
            ),
        )

    def mlx_resolution(self):
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

    def test_mlx_repository_surfaces_omlx_without_claiming_local(self):
        omlx = self.runtime_probe(
            "omlx",
            installed=False,
            service_available=False,
            formats=("MLX",),
            api="OpenAI-compatible /v1",
        )
        mlx_lm = self.runtime_probe(
            "mlx-lm",
            installed=False,
            service_available=None,
            formats=("MLX",),
            api="local CLI",
        )
        output = io.StringIO()

        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ), mock.patch.object(
            llmrig.HF_SOURCE, "resolve", return_value=self.mlx_resolution()
        ), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=[]
        ), mock.patch.object(
            runtime_bridge, "probe_runtimes", return_value=(omlx, mlx_lm)
        ), mock.patch(
            "_llmrig.omlx_verify.observe_omlx_inventory", return_value=()
        ), contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["solve", "org/model", "--json"]), 0)

        payload = json.loads(output.getvalue())
        by_runtime = {
            candidate["runtime"]: candidate for candidate in payload["candidates"]
        }
        self.assertIn("omlx", by_runtime)
        candidate = by_runtime["omlx"]
        self.assertEqual(candidate["artifact_format"], "MLX")
        self.assertEqual(
            candidate["assessments"]["runtime_availability"]["state"],
            "unavailable",
        )
        self.assertEqual(
            candidate["assessments"]["local_availability"]["state"],
            "unknown",
        )
        self.assertEqual(
            candidate["assessments"]["measurement_capability"]["state"],
            "measurable",
        )
        self.assertEqual(
            candidate["assessments"]["execution"]["state"],
            "not_executable",
        )
        self.assertNotEqual(
            payload["plan"]["recommended_candidate_id"], candidate["candidate_id"]
        )

    def test_generic_safetensors_still_has_no_omlx_runtime_match(self):
        probe = self.runtime_probe(
            "omlx",
            installed=True,
            service_available=True,
            version="omlx 1.0",
            cli_path="~/.omlx/bin/omlx",
            formats=("MLX",),
            api="OpenAI-compatible /v1",
        )
        capabilities = runtime_bridge.capabilities_from_probes(
            llmrig, self.mac_profile(), (probe,)
        )
        artifact = llmrig.ModelArtifact(
            "hf://org/model/model.safetensors",
            "org/model",
            None,
            "safetensors",
            4.0,
            None,
            (),
            size_bytes=4_000_000_000,
        )
        candidates = llmrig.runtime_candidates_for_artifacts(
            (artifact,), capabilities, self.mac_profile()
        )
        self.assertEqual(candidates, ())


if __name__ == "__main__":
    unittest.main()
