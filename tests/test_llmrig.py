import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "llmrig.py"
spec = importlib.util.spec_from_file_location("llmrig", MODULE_PATH)
llmrig = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = llmrig
spec.loader.exec_module(llmrig)


class LLMRigTests(unittest.TestCase):
    def mac_profile(self, ram_gib=48.0):
        return {
            "os": "Darwin",
            "ram_gib": ram_gib,
            "gpus": [{"name": "Apple M4 Max", "vram_gb": None, "backend": "Metal"}],
            "disk": {"free_gib": 500.0},
        }

    def linux_cpu_profile(self, ram_gib=16.0):
        return {
            "os": "Linux",
            "ram_gib": ram_gib,
            "gpus": [],
            "disk": {"free_gib": 500.0},
        }

    def runtime_capability(
        self, runtime, artifact_format, installed=True, available=True
    ):
        return llmrig.RuntimeCapability(
            runtime=runtime,
            installed=installed,
            available=available,
            version="1.0" if installed else None,
            supported_artifact_formats=(artifact_format,),
            supported_platforms=llmrig.ALL_PLATFORMS,
            supported_architectures=(),
            runtime_execution_capable=True,
            llmrig_installation_supported=False,
            llmrig_execution_supported=False,
            llmrig_benchmark_supported=False,
            confidence=llmrig.Confidence.HIGH,
            evidence=(
                llmrig.RecommendationEvidence(
                    "verified-local-runtime", "test runtime", "runtime was detected"
                ),
            ),
            unknowns=("successful inference is unknown",),
        )

    def generic_resolution(self, artifact_format="GGUF", size_bytes=4_000_000_000):
        model = llmrig.Model("org/model", "model", None, ("text",))
        artifact = llmrig.ModelArtifact(
            f"hf://org/model/{artifact_format.lower()}",
            model.model_id,
            None,
            artifact_format,
            size_bytes / 1_000_000_000 if size_bytes else None,
            None,
            (),
            size_bytes=size_bytes,
        )
        return llmrig.ModelResolution(
            "resolved", model, (artifact,), llmrig.Confidence.HIGH
        )

    def multi_artifact_resolution(self, sizes):
        model = llmrig.Model("org/model", "model", None, ("text",))
        artifacts = tuple(
            llmrig.ModelArtifact(
                f"hf://org/model/artifact-{index}.gguf",
                model.model_id,
                None,
                "GGUF",
                size / 1_000_000_000 if size else None,
                None,
                (),
                size_bytes=size,
            )
            for index, size in enumerate(sizes)
        )
        return llmrig.ModelResolution(
            "resolved", model, artifacts, llmrig.Confidence.HIGH
        )

    def race_configuration(
        self,
        runtime="ollama",
        artifact="model:a",
        quantization="Q4_K_M",
        fingerprint=None,
    ):
        return llmrig.RaceConfiguration(
            logical_model_id="logical/model",
            runtime=runtime,
            artifact_id=artifact,
            artifact_format="GGUF",
            quantization=quantization,
            runtime_version="1.2.3",
            eligible=True,
            artifact_fingerprint=fingerprint,
        )

    def race_competitor(
        self,
        runtime="ollama",
        artifact="model:a",
        quantization="Q4_K_M",
        generation=50.0,
        prompt=100.0,
        latency=2.0,
        runs=2,
    ):
        return llmrig.RaceCompetitor(
            logical_model_id="logical/model",
            runtime=runtime,
            artifact_id=artifact,
            artifact_fingerprint=None,
            artifact_format="GGUF",
            quantization=quantization,
            runtime_version="1.2.3",
            execution_status="success",
            generation_tps=generation,
            prompt_eval_tps=prompt,
            total_latency_s=latency,
            generated_tokens=256,
            measured_runs=runs,
            generation_samples=runs if generation is not None else 0,
            prompt_eval_samples=runs if prompt is not None else 0,
            latency_samples=runs if latency is not None else 0,
            timestamp="2026-01-01T00:00:00+00:00",
            evidence=(
                llmrig.RecommendationEvidence(
                    "measured", "mock runtime", "two timed runs completed"
                ),
            ),
        )

    def race_workload(self):
        return llmrig.RaceWorkload("deterministic prompt", runs=2)

    def test_project_branding(self):
        self.assertEqual(llmrig.PROJECT_NAME, "LLMRig")
        self.assertEqual(llmrig.PROJECT_SLUG, "llmrig")
        self.assertEqual(llmrig.VERSION, "0.4.1")

    def test_catalog_is_valid(self):
        self.assertEqual(llmrig.validate_curated_catalog(), [])

    def test_curated_spec_separates_model_from_artifact(self):
        spec = llmrig.resolve_curated_model("qwen3.8:27b-mlx")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.model.model_id, "qwen3.8")
        self.assertEqual(spec.model.modalities, ("text", "image"))
        self.assertEqual(spec.artifact.artifact_id, spec.ollama)
        self.assertEqual(spec.artifact.runtime, "ollama")
        self.assertEqual(spec.artifact.format, "MLX")
        self.assertIsNone(spec.artifact.quantization)
        quantized = llmrig.resolve_curated_model("qwen3.8:27b-q4_K_M").artifact
        self.assertEqual(quantized.format, "Unknown")
        self.assertEqual(quantized.quantization, "Q4_K_M")
        self.assertIn("artifact file format is unknown", quantized.unknowns)

    def test_model_and_artifact_reject_invalid_known_values(self):
        with self.assertRaises(ValueError):
            llmrig.Model("model", "Model", 0, ("text",))
        with self.assertRaises(ValueError):
            llmrig.ModelArtifact(
                "artifact", "model", "ollama", "", None, None, ()
            )

    def test_evidence_requires_provenance_and_serializes(self):
        evidence = llmrig.RecommendationEvidence(
            "curated-metadata", "snapshot", "artifact size is known"
        )
        self.assertEqual(
            evidence.to_dict(),
            {
                "kind": "curated-metadata",
                "source": "snapshot",
                "detail": "artifact size is known",
            },
        )
        with self.assertRaises(ValueError):
            llmrig.RecommendationEvidence("estimate", "", "detail")

    def test_compatibility_result_serialization_is_explicit(self):
        result = llmrig.CompatibilityResult(
            model_id="qwen3.8",
            artifact_id="qwen3.8:27b-mlx",
            runtime="ollama",
            status=llmrig.CompatibilityStatus.GOOD,
            confidence=llmrig.Confidence.HIGH,
            evidence=(
                llmrig.RecommendationEvidence(
                    "curated-metadata", "snapshot", "artifact is known"
                ),
            ),
            model_name="Qwen3.8 27B MLX",
            can_run=True,
            artifact_format="MLX",
            estimated_model_memory_gb=18.0,
            planning_budget_gb=28.8,
            memory_headroom_gb=10.8,
            recommended_context=32768,
            unknowns=("generation performance is not measured",),
        )
        self.assertEqual(
            result.to_dict(),
            {
                "model_id": "qwen3.8",
                "artifact_id": "qwen3.8:27b-mlx",
                "runtime": "ollama",
                "status": "good",
                "confidence": "high",
                "evidence": [
                    {
                        "kind": "curated-metadata",
                        "source": "snapshot",
                        "detail": "artifact is known",
                    }
                ],
                "reason": None,
                "model_name": "Qwen3.8 27B MLX",
                "can_run": True,
                "artifact_format": "MLX",
                "estimated_model_memory_gb": 18.0,
                "planning_budget_gb": 28.8,
                "memory_headroom_gb": 10.8,
                "recommended_context": 32768,
                "unknowns": ["generation performance is not measured"],
                "alternatives": [],
            },
        )

    def test_unknown_hardware_stays_unknown(self):
        spec = llmrig.resolve_curated_model("qwen3:8b")
        result = llmrig.assess_curated_compatibility(spec, {})
        self.assertEqual(result.status, llmrig.CompatibilityStatus.UNKNOWN)
        self.assertEqual(result.confidence, llmrig.Confidence.UNKNOWN)

    def test_known_compatibility_requires_evidence(self):
        with self.assertRaises(ValueError):
            llmrig.CompatibilityResult(
                "qwen3",
                "qwen3:8b",
                "ollama",
                llmrig.CompatibilityStatus.GOOD,
                llmrig.Confidence.HIGH,
            )

    def test_fit_label_preserves_existing_output_through_domain_result(self):
        spec = llmrig.resolve_curated_model("qwen3.8:27b-mlx")
        result = llmrig.assess_curated_compatibility(spec, self.mac_profile())
        self.assertEqual(result.status, llmrig.CompatibilityStatus.EXCELLENT)
        self.assertEqual(llmrig.fit_label(spec, self.mac_profile()), "excellent")
        self.assertEqual(result.artifact_id, spec.artifact.artifact_id)
        self.assertTrue(result.evidence)

    def test_can_known_supported_model_has_practical_configuration(self):
        result = llmrig.compatibility_for_identifier(
            "qwen3.8:27b-mlx", self.mac_profile()
        )
        self.assertTrue(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.EXCELLENT)
        self.assertEqual(result.confidence, llmrig.Confidence.HIGH)
        self.assertEqual(result.runtime, "ollama")
        self.assertEqual(result.artifact_format, "MLX")
        self.assertEqual(result.recommended_context, 32768)
        self.assertEqual(result.estimated_model_memory_gb, 18.0)
        self.assertEqual(result.memory_headroom_gb, 10.8)
        self.assertTrue(result.evidence)

    def test_can_model_that_does_not_fit_has_no_configuration(self):
        result = llmrig.compatibility_for_identifier(
            "qwen3.8:27b-bf16", self.mac_profile()
        )
        self.assertFalse(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.TOO_LARGE)
        self.assertIsNone(result.recommended_context)
        self.assertLess(result.memory_headroom_gb, 0)
        self.assertIn("qwen3.8:27b-mlx", result.alternatives)
        self.assertNotIn("qwen3.8:27b-bf16", result.alternatives)

    def test_can_non_native_artifact_is_not_supported_cross_platform(self):
        result = llmrig.compatibility_for_identifier(
            "qwen3.8:27b-mlx", self.linux_cpu_profile(64.0)
        )
        self.assertFalse(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.NOT_NATIVE)
        self.assertIsNone(result.recommended_context)

    def test_can_insufficient_metadata_remains_unknown(self):
        model = llmrig.ModelSpec(
            "Unknown-size model",
            "example:unknown",
            "test",
            llmrig.OFFICIAL,
            7.0,
            None,
            "unknown",
            llmrig.ALL_PLATFORMS,
            32768,
            "text",
            1,
            "Test-only incomplete metadata.",
        )
        result = llmrig.assess_curated_compatibility(model, self.mac_profile())
        self.assertIsNone(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.UNKNOWN)
        self.assertEqual(result.confidence, llmrig.Confidence.UNKNOWN)
        self.assertIn("artifact memory estimate is unknown", result.unknowns)

    def test_can_unknown_model_is_explicit_and_unresolved(self):
        result = llmrig.compatibility_for_identifier(
            "not-in-curated-catalog", self.mac_profile()
        )
        self.assertIsNone(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.UNKNOWN)
        self.assertEqual(result.confidence, llmrig.Confidence.UNKNOWN)
        self.assertIsNone(result.artifact_id)
        self.assertTrue(result.unknowns)

    def test_can_json_uses_domain_result_and_contains_only_json(self):
        args = llmrig.argparse.Namespace(model="qwen3.8:27b-mlx", json=True)
        output = io.StringIO()
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ), contextlib.redirect_stdout(output):
            self.assertEqual(llmrig.command_can(args), 0)

        payload = json.loads(output.getvalue())
        expected = llmrig.compatibility_for_identifier(
            "qwen3.8:27b-mlx", self.mac_profile()
        ).to_dict()
        self.assertEqual(payload, expected)
        self.assertNotIn("private-user", output.getvalue())

    def test_can_unknown_model_json_is_valid_and_returns_not_found(self):
        args = llmrig.argparse.Namespace(model="unknown/model", json=True)
        output = io.StringIO()
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ), mock.patch.object(
            llmrig.HF_SOURCE,
            "resolve",
            return_value=llmrig.ModelResolution("not_found_or_inaccessible", None),
        ), contextlib.redirect_stdout(output):
            self.assertEqual(llmrig.command_can(args), 2)

        payload = json.loads(output.getvalue())
        self.assertIsNone(payload["can_run"])
        self.assertEqual(payload["status"], "unknown")
        self.assertIsNone(payload["artifact_id"])

    def test_can_exit_codes_are_three_state_predicate_in_both_modes(self):
        cases = (
            ("qwen3.8:27b-mlx", 0, True),
            ("qwen3.8:27b-bf16", 1, False),
            ("unknown/model", 2, None),
        )
        for json_mode in (False, True):
            for model, expected_code, expected_can_run in cases:
                with self.subTest(json=json_mode, model=model):
                    args = llmrig.argparse.Namespace(model=model, json=json_mode)
                    output = io.StringIO()
                    with mock.patch.object(
                        llmrig, "hardware_profile", return_value=self.mac_profile()
                    ), mock.patch.object(
                        llmrig.HF_SOURCE,
                        "resolve",
                        return_value=llmrig.ModelResolution("not_found_or_inaccessible", None),
                    ), contextlib.redirect_stdout(output):
                        self.assertEqual(llmrig.command_can(args), expected_code)
                    if json_mode:
                        self.assertIs(
                            json.loads(output.getvalue())["can_run"], expected_can_run
                        )

    def test_can_human_output_renders_confidence_evidence_and_unknowns(self):
        args = llmrig.argparse.Namespace(model="qwen3.8:27b-mlx", json=False)
        output = io.StringIO()
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ), contextlib.redirect_stdout(output):
            self.assertEqual(llmrig.command_can(args), 0)

        rendered = output.getvalue()
        self.assertIn("Fit:         YES", rendered)
        self.assertIn("Confidence:  HIGH", rendered)
        self.assertIn("Runtime:     Ollama", rendered)
        self.assertIn("Build:       qwen3.8:27b-mlx", rendered)
        self.assertIn("Format:      MLX", rendered)
        self.assertNotIn("Artifact:", rendered)
        self.assertIn("Evidence", rendered)
        self.assertIn("Unknown", rendered)

    def test_alignment_aliases(self):
        self.assertEqual(llmrig.normalize_category("official"), llmrig.OFFICIAL)
        self.assertEqual(llmrig.normalize_category("restricted"), llmrig.OFFICIAL)
        self.assertEqual(
            llmrig.normalize_category("uncensored"),
            llmrig.REDUCED_REFUSAL,
        )
        self.assertEqual(
            llmrig.normalize_category("unrestricted"),
            llmrig.REDUCED_REFUSAL,
        )

    def test_parameter_parser(self):
        self.assertEqual(llmrig.parse_total_params_b("Qwen/Qwen3.8-27B"), 27.0)
        self.assertEqual(llmrig.parse_total_params_b("Qwen/Qwen3.8-2.4T-A95B"), 2400.0)
        self.assertIsNone(llmrig.parse_total_params_b("Qwen/Qwen-VL"))

    def test_model_name_matching_is_exact(self):
        self.assertTrue(llmrig.model_name_matches("qwen3:8b", "qwen3:8b"))
        self.assertFalse(llmrig.model_name_matches("qwen3:8b", "qwen3.8:27b-mlx"))
        self.assertTrue(llmrig.model_name_matches("qwen3.8", "qwen3.8:latest"))

    def test_48gb_mac_balanced_official_recommendation(self):
        model = llmrig.recommend_model(
            self.mac_profile(), llmrig.OFFICIAL, "balanced"
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.ollama, "qwen3.8:27b-mlx")

    def test_48gb_mac_reduced_refusal_recommendation(self):
        model = llmrig.recommend_model(
            self.mac_profile(), llmrig.REDUCED_REFUSAL, "balanced"
        )
        self.assertIsNotNone(model)
        self.assertEqual(
            model.ollama,
            "huihui_ai/Qwen3.8-abliterated:27b-q6_K",
        )

    def test_16gb_cpu_linux_recommendation(self):
        model = llmrig.recommend_model(
            self.linux_cpu_profile(), llmrig.OFFICIAL, "balanced"
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.ollama, "qwen3:8b")

    def test_48gb_18gb_model_starts_at_32k_context(self):
        model = llmrig.resolve_curated_model("qwen3.8:27b-mlx")
        self.assertIsNotNone(model)
        self.assertEqual(
            llmrig.recommended_context(self.mac_profile(), model),
            32768,
        )

    def test_64gb_18gb_model_can_start_at_64k_context(self):
        model = llmrig.resolve_curated_model("qwen3.8:27b-mlx")
        self.assertIsNotNone(model)
        self.assertEqual(
            llmrig.recommended_context(self.mac_profile(64.0), model),
            65536,
        )

    def test_huihui_default_alias_resolves(self):
        model = llmrig.resolve_curated_model(
            "huihui_ai/Qwen3.8-abliterated:27b"
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.quant, "Q4_K_M")

    def test_speed_metrics(self):
        response = {
            "eval_count": 500,
            "eval_duration": 10_000_000_000,
            "prompt_eval_count": 100,
            "prompt_eval_duration": 2_000_000_000,
            "load_duration": 500_000_000,
            "total_duration": 12_500_000_000,
        }
        metrics = llmrig.speed_metrics(response)
        self.assertEqual(metrics["generation_tps"], 50.0)
        self.assertEqual(metrics["prompt_tps"], 50.0)
        self.assertEqual(metrics["load_duration_s"], 0.5)


    def test_live_llm_candidate_accepts_text_generation(self):
        item = {"id": "Qwen/Qwen3.8-27B", "pipeline_tag": "image-text-to-text"}
        self.assertTrue(llmrig.is_live_llm_candidate(item))

    def test_live_llm_candidate_excludes_benchmark_repo(self):
        item = {"id": "Qwen/Qwen-Image-Bench", "pipeline_tag": "image-text-to-text"}
        # Known benchmark/artifact naming hints are excluded from the default
        # local-LLM discovery view even if metadata exposes an inference-like tag.
        self.assertFalse(llmrig.is_live_llm_candidate(item))

    def test_live_llm_candidate_excludes_sae_without_task(self):
        item = {"id": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", "pipeline_tag": None}
        self.assertFalse(llmrig.is_live_llm_candidate(item))

    def test_live_rows_do_not_claim_unverified_fit(self):
        rows = llmrig.live_rows([{"id": "Qwen/Qwen3.8-27B-FP8", "pipeline_tag": "image-text-to-text"}], llmrig.OFFICIAL)
        self.assertEqual(rows[0]["status"], "discovery only")
        self.assertNotIn("fit", rows[0])
        self.assertNotIn("q4_est", rows[0])

    def test_live_llm_candidate_excludes_asr(self):
        item = {
            "id": "Qwen/Qwen3-ASR-1.7B-hf",
            "pipeline_tag": "automatic-speech-recognition",
        }
        self.assertFalse(llmrig.is_live_llm_candidate(item))

    def test_hf_gguf_artifact_recognition_and_provenance(self):
        payload = {
            "id": "org/model-GGUF",
            "pipeline_tag": "text-generation",
            "config": {"max_position_embeddings": 8192},
            "siblings": [
                {"rfilename": "model-Q4_K_M.gguf", "size": 4_000_000_000}
            ],
        }
        with mock.patch.object(llmrig, "http_json", return_value=(payload, {})):
            resolution = llmrig.HuggingFaceModelSource().resolve("org/model-GGUF")
        artifact = resolution.artifacts[0]
        self.assertEqual(artifact.format, "GGUF")
        self.assertEqual(artifact.quantization, "Q4_K_M")
        self.assertEqual(artifact.size_bytes, 4_000_000_000)
        self.assertEqual(artifact.context_max, 8192)
        self.assertIsNone(artifact.runtime)
        self.assertIn("runtime compatibility is unknown", artifact.unknowns)
        self.assertEqual(
            {item.kind for item in artifact.evidence},
            {"deterministic-inference", "verified-metadata"},
        )

    def test_hf_mlx_recognition_uses_explicit_repository_metadata(self):
        payload = {
            "id": "org/model-mlx",
            "library_name": "mlx",
            "config": {
                "max_position_embeddings": 32768,
                "quantization": {"bits": 4},
            },
            "siblings": [
                {"rfilename": "model-00001-of-00002.safetensors", "size": 100},
                {"rfilename": "model-00002-of-00002.safetensors", "size": 200},
            ],
        }
        artifacts = llmrig.hf_artifacts_from_metadata(
            llmrig.hf_model_from_metadata("org/model-mlx", payload), payload
        )
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].format, "MLX")
        self.assertEqual(artifacts[0].quantization, "4-bit")
        self.assertEqual(artifacts[0].size_bytes, 300)
        self.assertEqual(artifacts[0].evidence[0].kind, "verified-metadata")

    def test_hf_safetensors_recognition(self):
        payload = {
            "id": "org/native-model",
            "pipeline_tag": "text-generation",
            "siblings": [{"rfilename": "model.safetensors", "size": 1234}],
        }
        artifacts = llmrig.hf_artifacts_from_metadata(
            llmrig.hf_model_from_metadata("org/native-model", payload), payload
        )
        self.assertEqual(artifacts[0].format, "Safetensors")
        self.assertEqual(artifacts[0].size_bytes, 1234)
        self.assertIsNone(artifacts[0].quantization)

    def test_hf_repository_can_expose_multiple_artifact_types(self):
        payload = {
            "id": "org/mixed-model",
            "siblings": [
                {"rfilename": "model-Q8_0.gguf", "size": 800},
                {"rfilename": "model.safetensors", "size": 1600},
            ],
        }
        artifacts = llmrig.hf_artifacts_from_metadata(
            llmrig.hf_model_from_metadata("org/mixed-model", payload), payload
        )
        self.assertEqual([item.format for item in artifacts], ["GGUF", "Safetensors"])

    def test_hf_explicit_base_model_separates_logical_model_from_repository(self):
        payload = {
            "id": "converter/model-GGUF",
            "cardData": {"base_model": "upstream/model"},
            "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 400}],
        }
        with mock.patch.object(llmrig, "http_json", return_value=(payload, {})):
            resolution = llmrig.HuggingFaceModelSource().resolve(
                "converter/model-GGUF"
            )
        self.assertEqual(resolution.model.model_id, "upstream/model")
        self.assertEqual(
            resolution.artifacts[0].artifact_id,
            "hf://converter/model-GGUF/model-Q4_K_M.gguf",
        )
        self.assertEqual(resolution.artifacts[0].model_id, "upstream/model")
        self.assertTrue(
            any("base_model" in item.source for item in resolution.evidence)
        )

    def test_hf_missing_file_metadata_stays_unknown(self):
        payload = {
            "id": "org/missing-metadata",
            "siblings": [{"rfilename": "weights.gguf"}],
        }
        artifact = llmrig.hf_artifacts_from_metadata(
            llmrig.hf_model_from_metadata("org/missing-metadata", payload), payload
        )[0]
        self.assertIsNone(artifact.size_bytes)
        self.assertIsNone(artifact.quantization)
        self.assertIn("artifact size is unknown", artifact.unknowns)
        self.assertIn("quantization is unknown", artifact.unknowns)

    def test_hf_ambiguous_quantization_filename_stays_unknown(self):
        quantization, status = llmrig.infer_gguf_quantization(
            "model-Q4_K_M-Q5_K_M.gguf"
        )
        self.assertIsNone(quantization)
        self.assertEqual(status, "ambiguous")

    def test_hf_ambiguous_safetensors_group_does_not_invent_size(self):
        payload = {
            "id": "org/ambiguous-weights",
            "siblings": [
                {"rfilename": "model.safetensors", "size": 100},
                {"rfilename": "alternate.safetensors", "size": 200},
            ],
        }
        artifact = llmrig.hf_artifacts_from_metadata(
            llmrig.hf_model_from_metadata("org/ambiguous-weights", payload), payload
        )[0]
        self.assertIsNone(artifact.size_bytes)
        self.assertIn(
            "artifact size is unknown because weight-file grouping is ambiguous",
            artifact.unknowns,
        )

    def test_hf_unknown_repository_is_distinct_from_network_failure(self):
        for status_code in (401, 404):
            unavailable = llmrig.urllib.error.HTTPError(
                "https://huggingface.co/api/models/org/missing",
                status_code,
                "unavailable",
                {},
                None,
            )
            with mock.patch.object(llmrig, "http_json", side_effect=unavailable):
                missing = llmrig.HuggingFaceModelSource().resolve("org/missing")
            self.assertEqual(missing.status, "not_found_or_inaccessible")
        with mock.patch.object(llmrig, "http_json", side_effect=OSError("offline")):
            failed = llmrig.HuggingFaceModelSource().resolve("org/model")
        self.assertEqual(failed.status, "network_error")
        self.assertNotIn("offline", failed.message)

    def test_generic_can_json_is_deterministic_and_privacy_safe(self):
        payload = {
            "id": "org/model",
            "pipeline_tag": "text-generation",
            "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 4000}],
        }
        args = llmrig.argparse.Namespace(model="org/model", json=True)
        outputs = []
        with mock.patch.object(llmrig, "http_json", return_value=(payload, {})), mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ):
            for _ in range(2):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(llmrig.command_can(args), 2)
                outputs.append(output.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        parsed = json.loads(outputs[0])
        self.assertEqual(parsed["compatibility_status"], "unknown")
        self.assertEqual(parsed["compatibility_confidence"], "unknown")
        self.assertEqual(parsed["resolution_status"], "resolved")
        self.assertEqual(parsed["resolution_confidence"], "high")
        self.assertIsNone(parsed["can_run"])
        self.assertEqual(parsed["artifacts"][0]["format"], "GGUF")
        self.assertNotIn("private-user", outputs[0])

    def test_generic_can_human_output_distinguishes_artifact_from_runtime(self):
        payload = {
            "id": "org/model",
            "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 4000}],
        }
        args = llmrig.argparse.Namespace(model="org/model", json=False)
        output = io.StringIO()
        with mock.patch.object(llmrig, "http_json", return_value=(payload, {})), mock.patch.object(
            llmrig, "hardware_profile", return_value=self.mac_profile()
        ), contextlib.redirect_stdout(output):
            self.assertEqual(llmrig.command_can(args), 2)
        rendered = output.getvalue()
        self.assertIn("Compatibility:            unknown", rendered)
        self.assertIn("Compatibility confidence: UNKNOWN", rendered)
        self.assertIn("Resolution status:         resolved", rendered)
        self.assertIn("Resolution confidence:     HIGH", rendered)
        self.assertIn("Detected artifacts", rendered)
        self.assertIn("Build: hf://org/model/model-Q4_K_M.gguf", rendered)
        self.assertIn("Format: GGUF", rendered)
        self.assertIn("Quantization: Q4_K_M", rendered)
        self.assertIn("Runtime: unknown", rendered)

    def test_ollama_capability_distinguishes_installed_from_available(self):
        provider = llmrig.OllamaRuntimeProvider()
        with mock.patch.object(
            provider, "info", return_value={"installed": True, "version": "0.12.0"}
        ), mock.patch.object(provider, "is_available", return_value=False):
            capability = provider.capability(self.mac_profile())
        self.assertTrue(capability.installed)
        self.assertFalse(capability.available)
        self.assertTrue(capability.runtime_execution_capable)
        self.assertTrue(capability.llmrig_execution_supported)
        self.assertTrue(capability.llmrig_benchmark_supported)
        self.assertEqual(capability.supported_artifact_formats, ("Ollama",))

    def test_ollama_capability_reports_uninstalled(self):
        provider = llmrig.OllamaRuntimeProvider()
        with mock.patch.object(
            provider, "info", return_value={"installed": False, "version": None}
        ):
            capability = provider.capability(self.mac_profile())
        self.assertFalse(capability.installed)
        self.assertFalse(capability.available)

    def test_llama_cpp_capability_detects_versioned_executable(self):
        process = mock.Mock(returncode=0, stdout="llama.cpp version 999\n", stderr="")
        with mock.patch.object(
            llmrig.shutil, "which", side_effect=lambda name: "/bin/llama-cli" if name == "llama-cli" else None
        ), mock.patch.object(llmrig, "run_cmd", return_value=process):
            capability = llmrig.LlamaCppRuntimeProvider().capability(
                self.linux_cpu_profile()
            )
        self.assertTrue(capability.installed)
        self.assertTrue(capability.available)
        self.assertEqual(capability.supported_artifact_formats, ("GGUF",))
        self.assertTrue(capability.runtime_execution_capable)
        self.assertFalse(capability.llmrig_execution_supported)
        self.assertFalse(capability.llmrig_benchmark_supported)

    def test_llama_cpp_capability_reports_unavailable_without_executable(self):
        with mock.patch.object(llmrig.shutil, "which", return_value=None):
            capability = llmrig.LlamaCppRuntimeProvider().capability(
                self.linux_cpu_profile()
            )
        self.assertFalse(capability.installed)
        self.assertFalse(capability.available)

    def test_runtime_command_on_path_without_successful_probe_is_unavailable(self):
        process = mock.Mock(returncode=1, stdout="", stderr="probe failed")
        with mock.patch.object(
            llmrig.shutil, "which", return_value="/private/path/llama-cli"
        ), mock.patch.object(llmrig, "run_cmd", return_value=process):
            capability = llmrig.LlamaCppRuntimeProvider().capability(
                self.linux_cpu_profile()
            )
        self.assertTrue(capability.installed)
        self.assertFalse(capability.available)
        self.assertIsNone(capability.version)

    def test_runtime_version_output_hides_home_path(self):
        process = mock.Mock(
            returncode=0,
            stdout=str(Path.home() / "private-build" / "llama.cpp 1.0"),
            stderr="",
        )
        with mock.patch.object(llmrig, "run_cmd", return_value=process):
            version = llmrig.detected_runtime_version("llama-cli")
        self.assertIsNotNone(version)
        self.assertNotIn(str(Path.home()), version)
        self.assertTrue(version.startswith("~"))

    def test_mlx_capability_is_available_only_on_supported_platform(self):
        provider = llmrig.MlxRuntimeProvider()
        with mock.patch.object(
            provider,
            "info",
            return_value={
                "installed": True,
                "command_installed": True,
                "command_available": True,
                "version": "0.29.0",
            },
        ):
            mac = provider.capability({**self.mac_profile(), "arch": "arm64"})
            linux = provider.capability({**self.linux_cpu_profile(), "arch": "x86_64"})
        self.assertTrue(mac.available)
        self.assertFalse(linux.available)
        self.assertIn("MLX", mac.supported_artifact_formats)
        self.assertTrue(mac.runtime_execution_capable)
        self.assertFalse(mac.llmrig_execution_supported)
        self.assertFalse(mac.llmrig_benchmark_supported)

    def test_mlx_capability_reports_uninstalled(self):
        provider = llmrig.MlxRuntimeProvider()
        with mock.patch.object(
            provider,
            "info",
            return_value={
                "installed": False,
                "command_installed": False,
                "command_available": False,
                "version": None,
            },
        ):
            capability = provider.capability({**self.mac_profile(), "arch": "arm64"})
        self.assertFalse(capability.installed)
        self.assertFalse(capability.available)

    def test_gguf_with_available_llama_cpp_produces_practical_candidate(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(), self.mac_profile()
            )
        self.assertTrue(result.can_run)
        self.assertEqual(result.runtime, "llama.cpp")
        self.assertEqual(result.runtime_candidates[0].support_status, "available")
        self.assertEqual(result.runtime_candidates[0].fit_result, "fits")
        self.assertIsNone(result.recommended_context)

    def test_mlx_with_available_mlx_lm_produces_runtime_candidate(self):
        capability = self.runtime_capability("mlx-lm", "MLX")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution("MLX"), self.mac_profile()
            )
        self.assertTrue(result.can_run)
        self.assertEqual(result.runtime, "mlx-lm")

    def test_safetensors_alone_has_no_runtime_candidate(self):
        capabilities = (
            self.runtime_capability("llama.cpp", "GGUF"),
            self.runtime_capability("mlx-lm", "MLX"),
        )
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=capabilities):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution("Safetensors"), self.mac_profile()
            )
        self.assertIsNone(result.can_run)
        self.assertEqual(result.runtime_candidates, ())

    def test_recognized_artifact_with_unavailable_runtime_stays_unknown(self):
        capability = self.runtime_capability(
            "llama.cpp", "GGUF", installed=False, available=False
        )
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(), self.mac_profile()
            )
        self.assertIsNone(result.can_run)
        self.assertEqual(
            result.runtime_candidates[0].support_status, "runtime_not_installed"
        )

    def test_available_runtime_with_oversized_artifact_returns_no(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(size_bytes=80_000_000_000), self.mac_profile()
            )
        self.assertFalse(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.TOO_LARGE)

    def test_available_runtime_with_unknown_artifact_size_stays_unknown(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(size_bytes=None), self.mac_profile()
            )
        self.assertIsNone(result.can_run)
        self.assertIn("artifact size", result.unknowns[0])

    def test_aggregate_yes_wins_over_unknown_candidate(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.multi_artifact_resolution((4_000_000_000, None)),
                self.mac_profile(),
            )
        self.assertTrue(result.can_run)
        self.assertEqual(
            [candidate.fit_result for candidate in result.runtime_candidates],
            ["fits", "unknown"],
        )

    def test_aggregate_yes_wins_over_oversized_candidate(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.multi_artifact_resolution((4_000_000_000, 80_000_000_000)),
                self.mac_profile(),
            )
        self.assertTrue(result.can_run)

    def test_aggregate_all_oversized_candidates_returns_no(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.multi_artifact_resolution((70_000_000_000, 80_000_000_000)),
                self.mac_profile(),
            )
        self.assertFalse(result.can_run)
        self.assertTrue(
            all(candidate.fit_result == "too_large" for candidate in result.runtime_candidates)
        )

    def test_aggregate_oversized_plus_unknown_size_returns_unknown(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.multi_artifact_resolution((80_000_000_000, None)),
                self.mac_profile(),
            )
        self.assertIsNone(result.can_run)

    def test_known_platform_incompatibility_can_rule_out_every_path(self):
        capability = llmrig.replace(
            self.runtime_capability("mlx-lm", "MLX"),
            supported_platforms=("Darwin",),
            supported_architectures=("arm64",),
        )
        profile = {**self.linux_cpu_profile(), "arch": "x86_64"}
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution("MLX", size_bytes=None), profile
            )
        self.assertFalse(result.can_run)
        self.assertEqual(result.status, llmrig.CompatibilityStatus.NOT_NATIVE)
        self.assertEqual(
            result.runtime_candidates[0].support_status, "platform_incompatible"
        )

    def test_installed_but_unavailable_runtime_remains_unknown(self):
        capability = self.runtime_capability(
            "llama.cpp", "GGUF", installed=True, available=False
        )
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(), self.mac_profile()
            )
        self.assertIsNone(result.can_run)
        self.assertEqual(
            result.runtime_candidates[0].support_status, "runtime_unavailable"
        )

    def test_runtime_and_compatibility_confidence_remain_separate(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            result = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(size_bytes=None), self.mac_profile()
            )
        payload = result.to_dict()
        self.assertEqual(payload["resolution_confidence"], "high")
        self.assertEqual(payload["runtime_candidates"][0]["confidence"], "high")
        self.assertEqual(payload["compatibility_confidence"], "unknown")

    def test_runtime_candidate_json_is_deterministic_and_privacy_safe(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)):
            first = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(), self.mac_profile()
            ).to_dict()
            second = llmrig.assess_generic_runtime_compatibility(
                self.generic_resolution(), self.mac_profile()
            ).to_dict()
        self.assertEqual(first, second)
        serialized = json.dumps(first, sort_keys=True)
        self.assertNotIn("private-user", serialized)
        self.assertNotIn("/bin/", serialized)

    def test_runtime_capability_json_names_external_and_llmrig_support_explicitly(self):
        capability = self.runtime_capability("llama.cpp", "GGUF")
        payload = capability.to_dict()
        self.assertTrue(payload["runtime_execution_capable"])
        self.assertFalse(payload["llmrig_execution_supported"])
        self.assertFalse(payload["llmrig_benchmark_supported"])
        self.assertNotIn("execution_supported", payload)
        self.assertNotIn("benchmark_supported", payload)
        self.assertNotIn("installation_supported", payload)

    def test_curated_can_does_not_call_generic_source(self):
        with mock.patch.object(llmrig.HF_SOURCE, "resolve") as generic_resolve:
            result = llmrig.compatibility_for_identifier(
                "qwen3.8:27b-mlx", self.mac_profile()
            )
        self.assertTrue(result.can_run)
        generic_resolve.assert_not_called()


    def test_unload_model_uses_keep_alive_zero(self):
        with mock.patch.object(llmrig, "http_json", return_value=({}, {})) as http_mock, \
             mock.patch.object(llmrig, "running_ollama_models", return_value=[]):
            self.assertTrue(
                llmrig.unload_ollama_model(
                    "http://127.0.0.1:11434", "qwen3.8:27b-mlx", wait_seconds=0
                )
            )
        _, kwargs = http_mock.call_args
        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["payload"]["keep_alive"], 0)
        self.assertEqual(kwargs["payload"]["model"], "qwen3.8:27b-mlx")

    def test_benchmark_isolation_unloads_resident_models(self):
        with mock.patch.object(
            llmrig,
            "running_ollama_models",
            return_value=["model-a:latest", "model-b:latest"],
        ), mock.patch.object(llmrig, "unload_ollama_model", return_value=True) as unload:
            isolated = llmrig.isolate_ollama_for_benchmark("http://127.0.0.1:11434")
        self.assertEqual(isolated, ["model-a:latest", "model-b:latest"])
        self.assertEqual(unload.call_count, 2)

    def test_race_with_zero_eligible_configurations_is_unavailable(self):
        adapter = mock.Mock(runtime="ollama")
        result = llmrig.execute_race(
            "logical/model",
            (),
            (),
            (adapter,),
            self.race_workload(),
            {},
            timestamp="fixed",
        )
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(llmrig.race_exit_code(result), 2)
        adapter.benchmark.assert_not_called()

    def test_race_with_one_eligible_configuration_does_not_benchmark(self):
        adapter = mock.Mock(runtime="ollama")
        result = llmrig.execute_race(
            "logical/model",
            (self.race_configuration(),),
            (),
            (adapter,),
            self.race_workload(),
            {},
            timestamp="fixed",
        )
        self.assertEqual(result.status, "unavailable")
        self.assertIn("at least 2", result.reason)
        adapter.benchmark.assert_not_called()

    def test_race_aliases_with_same_artifact_fingerprint_count_once(self):
        configurations = (
            self.race_configuration(artifact="model:alias-a", fingerprint="digest-1"),
            self.race_configuration(artifact="model:alias-b", fingerprint="digest-1"),
        )
        adapter = mock.Mock(runtime="ollama")
        result = llmrig.execute_race(
            "logical/model",
            configurations,
            (),
            (adapter,),
            self.race_workload(),
            {},
            timestamp="fixed",
        )
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(len(result.eligible_configurations), 1)
        self.assertEqual(result.eligible_configurations[0].artifact_id, "model:alias-a")
        adapter.benchmark.assert_not_called()

    def test_race_with_two_successful_configurations_has_metric_winners(self):
        configurations = (
            self.race_configuration(artifact="model:b"),
            self.race_configuration(artifact="model:a"),
        )
        results = {
            "model:a": self.race_competitor(
                artifact="model:a", generation=60.0, prompt=90.0, latency=2.0
            ),
            "model:b": self.race_competitor(
                artifact="model:b", generation=40.0, prompt=120.0, latency=3.0
            ),
        }
        adapter = mock.Mock(runtime="ollama")
        adapter.benchmark.side_effect = lambda config, workload: results[config.artifact_id]
        result = llmrig.execute_race(
            "logical/model",
            configurations,
            (),
            (adapter,),
            self.race_workload(),
            {},
            timestamp="fixed",
        )
        winners = result.to_dict()["winners"]
        self.assertEqual(result.status, "completed")
        self.assertEqual(llmrig.race_exit_code(result), 0)
        self.assertEqual(winners["fastest_generation"]["artifact_id"], "model:a")
        self.assertEqual(
            winners["fastest_prompt_evaluation"]["artifact_id"], "model:b"
        )
        self.assertEqual(winners["lowest_latency"]["artifact_id"], "model:a")

    def test_race_one_competitor_failure_invalidates_comparison(self):
        configurations = (
            self.race_configuration(artifact="model:a"),
            self.race_configuration(artifact="model:b"),
        )
        adapter = mock.Mock(runtime="ollama")
        adapter.benchmark.side_effect = [
            self.race_competitor(artifact="model:a"),
            RuntimeError("private token should not leak"),
        ]
        result = llmrig.execute_race(
            "logical/model", configurations, (), (adapter,), self.race_workload(), {}, timestamp="fixed"
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(llmrig.race_exit_code(result), 1)
        self.assertNotIn("private token", json.dumps(result.to_dict()))
        self.assertEqual(result.competitors[1].failure, "benchmark execution failed")

    def test_race_two_successes_plus_one_failure_retains_all_without_winners(self):
        configurations = tuple(
            self.race_configuration(artifact=f"model:{name}") for name in ("a", "b", "c")
        )
        adapter = mock.Mock(runtime="ollama")
        adapter.benchmark.side_effect = [
            self.race_competitor(artifact="model:a", generation=60.0),
            self.race_competitor(artifact="model:b", generation=50.0),
            RuntimeError("failed"),
        ]
        result = llmrig.execute_race(
            "logical/model", configurations, (), (adapter,), self.race_workload(), {}, timestamp="fixed"
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(llmrig.race_exit_code(result), 1)
        self.assertEqual(len(result.competitors), 3)
        self.assertEqual(
            [item.execution_status for item in result.competitors],
            ["success", "success", "failed"],
        )
        self.assertEqual(result.winners, ())

    def test_race_both_competitors_failure_is_structured(self):
        configurations = (
            self.race_configuration(artifact="model:a"),
            self.race_configuration(artifact="model:b"),
        )
        adapter = mock.Mock(runtime="ollama")
        adapter.benchmark.side_effect = RuntimeError("failure")
        result = llmrig.execute_race(
            "logical/model", configurations, (), (adapter,), self.race_workload(), {}, timestamp="fixed"
        )
        self.assertEqual(result.status, "failed")
        self.assertTrue(all(item.execution_status == "failed" for item in result.competitors))

    def test_race_timeout_is_reported_without_exception_details(self):
        configurations = (
            self.race_configuration(artifact="model:a"),
            self.race_configuration(artifact="model:b"),
        )
        adapter = mock.Mock(runtime="ollama")
        adapter.benchmark.side_effect = subprocess.TimeoutExpired(["safe"], 120)
        result = llmrig.execute_race(
            "logical/model", configurations, (), (adapter,), self.race_workload(), {}, timestamp="fixed"
        )
        self.assertTrue(all(item.failure == "benchmark timed out" for item in result.competitors))

    def test_race_competitor_and_json_ordering_are_deterministic(self):
        configurations = (
            self.race_configuration(runtime="z-runtime", artifact="z"),
            self.race_configuration(runtime="a-runtime", artifact="a"),
        )
        adapters = []
        for configuration in configurations:
            adapter = mock.Mock(runtime=configuration.runtime)
            adapter.benchmark.return_value = self.race_competitor(
                runtime=configuration.runtime, artifact=configuration.artifact_id
            )
            adapters.append(adapter)
        kwargs = dict(
            logical_model_id="logical/model",
            ineligible=(),
            adapters=adapters,
            workload=self.race_workload(),
            hardware={"os": "Test"},
            timestamp="fixed",
        )
        first = llmrig.execute_race(eligible=configurations, **kwargs).to_dict()
        second = llmrig.execute_race(eligible=tuple(reversed(configurations)), **kwargs).to_dict()
        self.assertEqual(first, second)
        self.assertEqual([item["runtime"] for item in first["competitors"]], ["a-runtime", "z-runtime"])

    def test_race_ties_and_single_run_are_inconclusive(self):
        close = (
            self.race_competitor(artifact="a", generation=100.0),
            self.race_competitor(artifact="b", generation=103.0),
        )
        self.assertEqual(
            llmrig.metric_winner(close, "generation_tps", True)["status"],
            "inconclusive",
        )
        one_run = tuple(
            llmrig.replace(item, measured_runs=1, generation_samples=1)
            for item in close
        )
        self.assertIn(
            "fewer than two valid samples",
            llmrig.metric_winner(one_run, "generation_tps", True)["reason"],
        )
        self.assertEqual(
            llmrig.metric_winner(tuple(reversed(close)), "generation_tps", True),
            llmrig.metric_winner(close, "generation_tps", True),
        )

    def test_missing_prompt_metrics_do_not_block_generation_winner(self):
        competitors = (
            self.race_competitor(artifact="a", generation=70.0, prompt=None),
            self.race_competitor(artifact="b", generation=50.0, prompt=100.0),
        )
        generation = llmrig.metric_winner(competitors, "generation_tps", True)
        prompt = llmrig.metric_winner(competitors, "prompt_eval_tps", True)
        self.assertEqual(generation["status"], "winner")
        self.assertEqual(generation["artifact_id"], "a")
        self.assertEqual(prompt["status"], "inconclusive")

    def test_race_warns_when_quantizations_differ(self):
        configurations = (
            self.race_configuration(artifact="a", quantization="Q4"),
            self.race_configuration(artifact="b", quantization="Q8"),
        )
        adapter = mock.Mock(runtime="ollama")
        adapter.benchmark.side_effect = [
            self.race_competitor(artifact="a", quantization="Q4"),
            self.race_competitor(artifact="b", quantization="Q8", generation=70.0),
        ]
        result = llmrig.execute_race(
            "logical/model", configurations, (), (adapter,), self.race_workload(), {}, timestamp="fixed"
        )
        self.assertTrue(any("different quantizations" in item for item in result.warnings))

    def test_race_warns_when_artifact_formats_differ(self):
        configurations = (
            self.race_configuration(runtime="runtime-a", artifact="a"),
            llmrig.replace(
                self.race_configuration(runtime="runtime-b", artifact="b"),
                artifact_format="MLX",
            ),
        )
        adapters = []
        for configuration in configurations:
            adapter = mock.Mock(runtime=configuration.runtime)
            adapter.benchmark.return_value = llmrig.replace(
                self.race_competitor(
                    runtime=configuration.runtime, artifact=configuration.artifact_id
                ),
                artifact_format=configuration.artifact_format,
            )
            adapters.append(adapter)
        result = llmrig.execute_race(
            "logical/model", configurations, (), adapters, self.race_workload(), {}, timestamp="fixed"
        )
        self.assertTrue(any("different artifact formats" in item for item in result.warnings))

    def test_race_command_json_one_competitor_returns_two_without_execution(self):
        configuration = self.race_configuration()
        args = llmrig.argparse.Namespace(
            model="logical/model",
            json=True,
            context=4096,
            runs=2,
            num_predict=128,
            host="http://127.0.0.1:11434",
        )
        output = io.StringIO()
        with mock.patch.object(llmrig, "hardware_profile", return_value=self.mac_profile()), mock.patch.object(
            llmrig, "race_configurations", return_value=("logical/model", (configuration,), (), None)
        ), mock.patch.object(llmrig.OllamaExecutionAdapter, "benchmark") as benchmark, contextlib.redirect_stdout(output):
            self.assertEqual(llmrig.command_race(args), 2)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(len(payload["eligible_configurations"]), 1)
        benchmark.assert_not_called()

    def test_existing_bench_command_still_uses_existing_benchmark_path(self):
        args = llmrig.argparse.Namespace(
            host="http://127.0.0.1:11434",
            all_installed=False,
            model="qwen3:8b",
            context=4096,
            runs=1,
            output_dir=None,
        )
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "ensure_available", return_value=True), mock.patch.object(
            llmrig, "run_benchmark", return_value={"aggregate": {}}
        ) as benchmark:
            self.assertEqual(llmrig.command_bench(args), 0)
        benchmark.assert_called_once_with(
            "qwen3:8b", 4096, 1, args.host, None
        )

    def test_ollama_execution_adapter_reuses_bounded_generation_and_cleans_up(self):
        response = {
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 1_000_000_000,
            "total_duration": 3_000_000_000,
        }
        adapter = llmrig.OllamaExecutionAdapter("http://127.0.0.1:11434")
        configuration = self.race_configuration()
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "is_available", return_value=True), mock.patch.object(
            llmrig, "isolate_ollama_for_benchmark", return_value=[]
        ), mock.patch.object(
            llmrig, "ollama_generate", return_value=response
        ) as generate, mock.patch.object(
            llmrig, "unload_ollama_model", return_value=True
        ) as unload:
            result = adapter.benchmark(configuration, self.race_workload())
        self.assertEqual(result.execution_status, "success")
        self.assertEqual(result.generation_tps, 50.0)
        self.assertEqual(result.generated_tokens, 200)
        self.assertEqual(generate.call_count, 3)
        self.assertTrue(all(call.kwargs["timeout"] == 120 for call in generate.call_args_list))
        unload.assert_called_once_with(adapter.host, configuration.artifact_id)

    def test_ollama_warmup_failure_is_not_a_successful_competitor(self):
        adapter = llmrig.OllamaExecutionAdapter("http://127.0.0.1:11434")
        configurations = (
            self.race_configuration(artifact="model:a"),
            self.race_configuration(artifact="model:b"),
        )
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "is_available", return_value=True), mock.patch.object(
            llmrig, "isolate_ollama_for_benchmark", return_value=[]
        ), mock.patch.object(
            llmrig, "ollama_generate", side_effect=RuntimeError("warmup failed")
        ), mock.patch.object(llmrig, "unload_ollama_model", return_value=True) as unload:
            result = llmrig.execute_race(
                "logical/model", configurations, (), (adapter,), self.race_workload(), {}, timestamp="fixed"
            )
        self.assertEqual(result.status, "failed")
        self.assertTrue(all(item.execution_status == "failed" for item in result.competitors))
        self.assertEqual(unload.call_count, 2)

    def test_invalid_generation_metrics_fail_competitor_and_cleanup(self):
        adapter = llmrig.OllamaExecutionAdapter("http://127.0.0.1:11434")
        configuration = self.race_configuration()
        response = {"eval_count": 0, "eval_duration": 0}
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "is_available", return_value=True), mock.patch.object(
            llmrig, "isolate_ollama_for_benchmark", return_value=[]
        ), mock.patch.object(
            llmrig, "ollama_generate", return_value=response
        ), mock.patch.object(llmrig, "unload_ollama_model", return_value=True) as unload:
            with self.assertRaises(RuntimeError):
                adapter.benchmark(configuration, self.race_workload())
        unload.assert_called_once_with(adapter.host, configuration.artifact_id)

    def test_race_eligibility_never_pulls_or_downloads_models(self):
        adapter = mock.Mock(runtime="ollama")
        capability = llmrig.replace(
            self.runtime_capability("ollama", "Ollama"),
            llmrig_execution_supported=True,
            llmrig_benchmark_supported=True,
        )
        with mock.patch.object(llmrig, "runtime_capabilities", return_value=(capability,)), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=[]
        ), mock.patch.object(llmrig, "pull_model") as pull:
            _, eligible, ineligible, _ = llmrig.race_configurations(
                "qwen3.8:27b-mlx", self.mac_profile(), (adapter,)
            )
        self.assertEqual(eligible, ())
        self.assertTrue(ineligible)
        pull.assert_not_called()

    def test_race_json_is_privacy_safe_and_records_runtime_version(self):
        configuration = self.race_configuration()
        result = llmrig.execute_race(
            "logical/model",
            (configuration,),
            (),
            (),
            self.race_workload(),
            {"cpu": "Test CPU"},
            timestamp="fixed",
        )
        serialized = json.dumps(result.to_dict())
        self.assertIn("1.2.3", serialized)
        self.assertNotIn(str(Path.home()), serialized)

    def test_race_runtime_version_rejects_local_paths(self):
        self.assertIsNone(
            llmrig.race_safe_runtime_version("build /Users/private-user/llama")
        )
        self.assertIsNone(llmrig.race_safe_runtime_version("build C:\\private\\llama"))
        self.assertEqual(llmrig.race_safe_runtime_version("runtime 1.2.3"), "runtime 1.2.3")

    def test_run_cmd_never_enables_shell(self):
        with mock.patch.object(llmrig.subprocess, "run") as run:
            llmrig.run_cmd(["safe-command", "--version"])
        self.assertFalse(run.call_args.kwargs.get("shell", False))


    def test_shareable_hardware_profile_removes_model_store(self):
        with mock.patch.object(
            llmrig,
            "hardware_profile",
            return_value={"cpu": "Apple M4 Max", "model_store": "/Users/example/.ollama/models"},
        ):
            profile = llmrig.shareable_hardware_profile()
        self.assertEqual(profile["cpu"], "Apple M4 Max")
        self.assertNotIn("model_store", profile)

    def test_shareable_ollama_info_removes_executable_path(self):
        with mock.patch.object(
            llmrig,
            "ollama_cli_info",
            return_value={"installed": True, "path": "/Users/example/bin/ollama", "version": "0.32.13"},
        ):
            info = llmrig.shareable_ollama_info()
        self.assertEqual(info["version"], "0.32.13")
        self.assertNotIn("path", info)

    def test_default_model_store_is_displayed_relative_to_home(self):
        home = Path("/Users/private-user")
        with mock.patch.object(llmrig.Path, "home", return_value=home), mock.patch.dict(
            os.environ, {"OLLAMA_MODELS": ""}
        ):
            model_store = llmrig.ollama_model_store_path()
            self.assertEqual(
                llmrig.privacy_safe_path(model_store),
                str(Path("~") / ".ollama" / "models"),
            )

    def test_custom_model_store_outside_home_remains_meaningful(self):
        home = Path("/Users/private-user")
        custom = home.parent / "shared-ollama-models"
        with mock.patch.object(llmrig.Path, "home", return_value=home), mock.patch.dict(
            os.environ, {"OLLAMA_MODELS": str(custom)}
        ):
            model_store = llmrig.ollama_model_store_path()
            self.assertEqual(model_store, custom)
            self.assertEqual(llmrig.privacy_safe_path(model_store), str(custom))

    def test_doctor_json_hides_home_directory_identity(self):
        home = Path("/Users/private-user")
        profile = {
            "model_store": str(home / ".ollama" / "models"),
            "ram_gib": 48.0,
            "os": "Darwin",
            "gpus": [],
        }
        output = io.StringIO()
        with mock.patch.object(llmrig.Path, "home", return_value=home), mock.patch.object(
            llmrig.OLLAMA_RUNTIME, "info", return_value={"installed": False}
        ), mock.patch.object(
            llmrig.OLLAMA_RUNTIME, "is_available", return_value=False
        ), contextlib.redirect_stdout(output):
            llmrig.print_doctor(profile, json_mode=True)

        serialized = output.getvalue()
        payload = json.loads(serialized)
        self.assertEqual(payload["model_store"], str(Path("~") / ".ollama" / "models"))
        self.assertNotIn("private-user", serialized)

    def passport_fixture(self, timestamp="2026-08-27T00:00:00Z"):
        return llmrig._passport_payload(
            timestamp=timestamp,
            logical_model_id="org/model",
            artifact_id="model:q4",
            artifact_fingerprint="sha256:abc",
            artifact_format="GGUF",
            quantization="Q4_K_M",
            artifact_size_bytes=4_000_000_000,
            runtime="ollama",
            runtime_version="0.32.13",
            adapter="ollama-race-adapter",
            hardware={"os": "Darwin", "architecture": "arm64", "cpu_or_chip": "Apple M4", "ram_gib": 48.0, "accelerator": "Apple Metal"},
            workload={"prompt": "fixed", "requested_context": 4096, "num_predict": 128, "temperature": 0, "seed": 42, "think": False, "warmup_count": 1, "warmup_token_cap": 32, "measured_run_count": 2, "timeout_s": 120, "other_generation_settings": {}},
            raw_samples=(
                {"generation_tps": 10.0, "prompt_tps": 20.0, "wall_seconds": 2.0, "total_duration_s": 1.9, "eval_count": 100},
                {"generation_tps": 12.0, "prompt_tps": None, "wall_seconds": 2.2, "total_duration_s": 2.1, "eval_count": 100},
            ),
        )

    def test_successful_passport_raw_samples_reproduce_aggregates(self):
        payload = self.passport_fixture().to_dict()
        self.assertEqual(llmrig.validate_passport(payload), [])
        self.assertEqual(payload["measurement"]["aggregates"]["generation_tps"], 11.0)
        self.assertEqual(payload["measurement"]["aggregates"]["prompt_evaluation_tps"], 20.0)
        self.assertEqual(payload["measurement"]["aggregates"]["generated_tokens"], 200)
        self.assertEqual(len(payload["measurement"]["raw_samples"]), 2)

    def test_warmups_are_metadata_not_measured_samples(self):
        payload = self.passport_fixture().to_dict()
        self.assertEqual(payload["workload"]["warmup_count"], 1)
        self.assertEqual(payload["measurement"]["measured_run_count"], 2)
        self.assertNotIn("warmup", json.dumps(payload["measurement"]))

    def test_canonical_serialization_ignores_dictionary_order(self):
        self.assertEqual(
            llmrig.sha256_identity({"a": 1, "b": 2}),
            llmrig.sha256_identity({"b": 2, "a": 1}),
        )

    def test_passport_id_changes_but_configuration_stays_for_distinct_runs(self):
        first = self.passport_fixture("one").to_dict()
        second = self.passport_fixture("two").to_dict()
        self.assertNotEqual(first["passport_id"], second["passport_id"])
        self.assertEqual(first["configuration_fingerprint"], second["configuration_fingerprint"])

    def test_passport_id_hash_explicitly_excludes_stored_id(self):
        payload = self.passport_fixture().to_dict()
        stored = payload["passport_id"]
        without_id = dict(payload)
        without_id.pop("passport_id")
        self.assertEqual(stored, llmrig.sha256_identity(without_id))
        payload["passport_id"] = "f" * 64
        self.assertEqual(stored, llmrig.passport_id_for_document(payload))
        self.assertIn("passport ID mismatch", llmrig.validate_passport(payload))

    def test_measurements_do_not_change_configuration_fingerprint(self):
        first = self.passport_fixture().to_dict()
        changed = self.passport_fixture().to_dict()
        changed["measurement"]["raw_samples"][0]["generation_tps"] = 999.0
        changed["measurement"]["aggregates"]["generation_tps"] = 505.5
        self.assertEqual(first["configuration_fingerprint"], changed["configuration_fingerprint"])

    def test_configuration_fingerprint_changes_for_material_dimensions(self):
        original = self.passport_fixture().to_dict()
        for section, key, value in (
            ("model", "artifact_or_build", "other"),
            ("runtime", "runtime_version", "different"),
            ("hardware", "ram_gib", 64.0),
            ("workload", "requested_context", 8192),
        ):
            changed = json.loads(json.dumps(original))
            changed[section][key] = value
            config = {"model": changed["model"], "runtime": changed["runtime"], "hardware": changed["hardware"], "workload": changed["workload"], "benchmark_method": changed["identity"]["benchmark_method"]}
            self.assertNotEqual(original["configuration_fingerprint"], llmrig.sha256_identity(config))
        changed_method = json.loads(json.dumps(original))
        changed_method["identity"]["benchmark_method"] = "new-method"
        config = {"model": changed_method["model"], "runtime": changed_method["runtime"], "hardware": changed_method["hardware"], "workload": changed_method["workload"], "benchmark_method": "new-method"}
        self.assertNotEqual(original["configuration_fingerprint"], llmrig.sha256_identity(config))

    def test_validation_rejects_schema_ids_missing_fields_and_impossible_success(self):
        for mutation, expected in (
            (lambda p: p.update(schema_version="9"), "unsupported schema version"),
            (lambda p: p.update(passport_id="bad"), "passport_id must be"),
            (lambda p: p.update(configuration_fingerprint="bad"), "configuration_fingerprint must be"),
            (lambda p: p.pop("runtime"), "missing required fields"),
            (lambda p: p["measurement"].update(raw_samples=[], measured_run_count=0), "successful benchmark requires"),
        ):
            payload = self.passport_fixture().to_dict()
            mutation(payload)
            self.assertTrue(any(expected in error for error in llmrig.validate_passport(payload)))

    def test_failed_passport_cannot_masquerade_as_measurement(self):
        payload = self.passport_fixture().to_dict()
        payload["measurement"]["execution_status"] = "failed"
        self.assertTrue(any("failed benchmark" in error for error in llmrig.validate_passport(payload)))

    def test_passport_privacy_rejects_paths_identity_and_credentials(self):
        for field, value in (
            ("runtime_version", "/Users/private/bin/ollama"),
            ("local_path", "/tmp/model.gguf"),
            ("hostname", "private-mac"),
            ("failure", "Bearer hf_abcdefghijklmnop"),
        ):
            payload = self.passport_fixture().to_dict()
            payload["runtime"][field] = value
            self.assertTrue(llmrig.passport_privacy_issues(payload))
        with mock.patch.dict(os.environ, {"USER": "private-user"}), mock.patch.object(
            llmrig.platform, "node", return_value="private-host"
        ):
            self.assertIsNone(llmrig.passport_safe_runtime_version("tool by private-user"))
            self.assertIsNone(llmrig.passport_safe_runtime_version("tool on private-host"))
            self.assertIsNone(llmrig.passport_safe_runtime_version("Bearer hf_abcdefghijklmnop"))

    def test_comparison_classifications(self):
        exact = self.passport_fixture().to_dict()
        self.assertEqual(llmrig.compare_passports(exact, dict(exact))["classification"], "exact")
        quant = json.loads(json.dumps(exact))
        quant["model"]["quantization"] = "Q8_0"
        quant["configuration_fingerprint"] = "different"
        self.assertEqual(llmrig.compare_passports(exact, quant)["classification"], "comparable_with_warnings")
        workload = json.loads(json.dumps(exact))
        workload["workload"]["prompt"] = "other"
        workload["configuration_fingerprint"] = "different"
        self.assertEqual(llmrig.compare_passports(exact, workload)["classification"], "not_comparable")
        hardware = json.loads(json.dumps(exact))
        hardware["hardware"]["ram_gib"] = 64
        hardware["configuration_fingerprint"] = "different"
        comparison = llmrig.compare_passports(exact, hardware)
        self.assertEqual(comparison["classification"], "comparable_with_warnings")
        self.assertIn("different hardware", comparison["reasons"])

    def test_passport_verify_is_read_only_and_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passport.json"
            llmrig.write_passport(self.passport_fixture(), path)
            args = llmrig.argparse.Namespace(file=str(path))
            with mock.patch.object(llmrig, "http_json") as network, mock.patch.object(llmrig.HF_SOURCE, "resolve") as huggingface, mock.patch.object(llmrig, "ollama_generate") as inference, mock.patch.object(llmrig, "run_cmd") as subprocess_call, mock.patch.object(llmrig, "pull_model") as pull, mock.patch.object(llmrig, "install_ollama_help") as install, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(llmrig.command_passport_verify(args), 0)
            network.assert_not_called()
            huggingface.assert_not_called()
            inference.assert_not_called()
            subprocess_call.assert_not_called()
            pull.assert_not_called()
            install.assert_not_called()

    def test_passport_validation_accepts_reformatted_json_and_rejects_tamper(self):
        payload = self.passport_fixture().to_dict()
        reordered = json.loads(json.dumps(payload, sort_keys=True))
        self.assertEqual(llmrig.validate_passport(reordered), [])
        reordered["measurement"]["raw_samples"][0]["generation_tps"] = 999
        errors = llmrig.validate_passport(reordered)
        self.assertIn("measurement aggregates do not match raw samples", errors)
        self.assertIn("passport ID mismatch", errors)

    def test_legacy_benchmark_conversion_retains_unknowns_without_fabrication(self):
        result = {"timestamp": "fixed", "model": "unknown/model", "context": 4096, "hardware": {"os": "Darwin", "arch": "arm64", "cpu": "Apple", "ram_gib": 48}, "ollama": {"version": "1.0"}, "running_model": {}, "throughput_runs": [{"generation_tps": 4.0, "prompt_tps": None, "wall_seconds": 1.0, "total_duration_s": 0.9, "eval_count": 4}]}
        payload = llmrig.passport_from_benchmark_result(result).to_dict()
        self.assertIsNone(payload["model"]["artifact_format"])
        self.assertIsNone(payload["model"]["quantization"])
        self.assertIsNone(payload["model"]["artifact_size_bytes"])
        self.assertIsNone(payload["measurement"]["aggregates"]["prompt_evaluation_tps"])
        self.assertEqual(llmrig.validate_passport(payload), [])

    def test_execution_adapter_retains_only_measured_raw_samples(self):
        response = {"eval_count": 10, "eval_duration": 1_000_000_000, "prompt_eval_count": 2, "prompt_eval_duration": 1_000_000_000, "total_duration": 2_000_000_000}
        adapter = llmrig.OllamaExecutionAdapter()
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "is_available", return_value=True), mock.patch.object(llmrig, "isolate_ollama_for_benchmark", return_value=[]), mock.patch.object(llmrig, "ollama_generate", return_value=response), mock.patch.object(llmrig, "unload_ollama_model", return_value=True):
            competitor = adapter.benchmark(self.race_configuration(), self.race_workload())
        self.assertEqual(len(competitor.raw_samples), 2)

    def test_distinct_bench_and_race_samples_survive_without_reconstruction(self):
        raw = (
            {"generation_tps": 40.0, "prompt_tps": 80.0, "wall_seconds": 2.0, "total_duration_s": 1.9, "eval_count": 100, "eval_duration_s": 1.5},
            {"generation_tps": 60.0, "prompt_tps": None, "wall_seconds": 1.5, "total_duration_s": 1.4, "eval_count": 90, "eval_duration_s": 1.0},
        )
        bench = {"timestamp": "fixed", "model": "unknown/model", "context": 4096, "hardware": {}, "ollama": {}, "running_model": {}, "throughput_runs": list(raw)}
        bench_samples = llmrig.passport_from_benchmark_result(bench).to_dict()["measurement"]["raw_samples"]
        self.assertEqual([item["generation_tps"] for item in bench_samples], [40.0, 60.0])
        competitor = llmrig.replace(self.race_competitor(), raw_samples=raw)
        race = llmrig.RaceResult("completed", "logical/model", None, llmrig.RACE_METHOD_VERSION, "fixed", self.race_workload(), {}, (), (), (competitor,))
        race_samples = llmrig.passport_from_race_competitor(race, competitor).to_dict()["measurement"]["raw_samples"]
        self.assertEqual([item["generation_tps"] for item in race_samples], [40.0, 60.0])
        self.assertEqual([item["generated_tokens"] for item in race_samples], [100, 90])

    def test_failed_race_exports_no_standalone_success_passports(self):
        success = llmrig.replace(self.race_competitor(), raw_samples=({"generation_tps": 40.0, "prompt_tps": 80.0, "wall_seconds": 2.0, "total_duration_s": 1.9, "eval_count": 100},))
        failed = llmrig.replace(self.race_competitor(artifact="model:failed"), execution_status="failed", generation_tps=None, prompt_eval_tps=None, total_latency_s=None, generated_tokens=None, measured_runs=0, generation_samples=0, prompt_eval_samples=0, latency_samples=0, failure="benchmark execution failed")
        race = llmrig.RaceResult("failed", "logical/model", "incomplete", llmrig.RACE_METHOD_VERSION, "fixed", self.race_workload(), {}, (), (), (success, failed))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "passports"
            self.assertEqual(llmrig.export_race_passports(race, output), ())
            self.assertFalse(output.exists())

    def test_bench_passport_requires_one_model_before_execution(self):
        args = llmrig.argparse.Namespace(host="host", all_installed=True, model=None, context=4096, runs=1, output_dir=None, passport="out.json")
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "ensure_available", return_value=True), mock.patch.object(llmrig, "installed_ollama_models", return_value=[{"name": "qwen:a", "id": "a"}, {"name": "qwen:b", "id": "b"}]), mock.patch.object(llmrig, "run_benchmark") as benchmark:
            self.assertEqual(llmrig.command_bench(args), 2)
        benchmark.assert_not_called()

    def test_bench_passport_export_reuses_benchmark_and_local_digest(self):
        result = {"timestamp": "fixed", "model": "qwen3.8:27b-mlx", "context": 4096, "hardware": {"os": "Darwin", "arch": "arm64", "cpu": "Apple", "ram_gib": 48}, "ollama": {"version": "1.0"}, "running_model": {}, "throughput_runs": [{"generation_tps": 4.0, "prompt_tps": 5.0, "wall_seconds": 1.0, "total_duration_s": 0.9, "eval_count": 4}], "aggregate": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.json"
            args = llmrig.argparse.Namespace(host="host", all_installed=False, model="qwen3.8:27b-mlx", context=4096, runs=1, output_dir=None, passport=str(path))
            with mock.patch.object(llmrig.OLLAMA_RUNTIME, "ensure_available", return_value=True), mock.patch.object(llmrig, "installed_ollama_models", return_value=[{"name": "qwen3.8:27b-mlx", "id": "digest"}]), mock.patch.object(llmrig, "run_benchmark", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(llmrig.command_bench(args), 0)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["model"]["artifact_fingerprint"], "digest")
            self.assertEqual(llmrig.validate_passport(payload), [])

    def test_bench_passport_timeout_matches_actual_measured_execution(self):
        response = {
            "response": "391",
            "eval_count": 40,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500_000_000,
            "total_duration": 2_000_000_000,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            llmrig, "ensure_ollama_service", return_value=True
        ), mock.patch.object(
            llmrig, "isolate_ollama_for_benchmark", return_value=[]
        ), mock.patch.object(
            llmrig, "ollama_generate", return_value=response
        ) as generate, mock.patch.object(
            llmrig, "unload_ollama_model", return_value=True
        ), mock.patch.object(
            llmrig, "memory_snapshot", return_value={}
        ), mock.patch.object(
            llmrig, "running_model_details", return_value={}
        ), mock.patch.object(
            llmrig, "shareable_hardware_profile", return_value={}
        ), mock.patch.object(
            llmrig, "shareable_ollama_info", return_value={}
        ), contextlib.redirect_stdout(io.StringIO()):
            result = llmrig.run_benchmark(
                "qwen3.8:27b-mlx", 4096, 1, output_dir=Path(directory)
            )
        passport = llmrig.passport_from_benchmark_result(result).to_dict()
        timed_call = next(
            call for call in generate.call_args_list if call.args[2] == llmrig.SPEED_PROMPT
        )
        self.assertEqual(
            timed_call.kwargs["timeout"], passport["workload"]["timeout_s"]
        )
        self.assertEqual(passport["workload"]["timeout_s"], llmrig.BENCH_REQUEST_TIMEOUT)

    def test_exported_race_passport_normalizes_and_validates_phase5_hardware(self):
        raw = (
            {"generation_tps": 40.0, "prompt_tps": 80.0, "wall_seconds": 2.0, "total_duration_s": 1.9, "eval_count": 100},
            {"generation_tps": 60.0, "prompt_tps": 90.0, "wall_seconds": 1.5, "total_duration_s": 1.4, "eval_count": 100},
        )
        competitor = llmrig.replace(self.race_competitor(), raw_samples=raw)
        hardware = {
            "os": "Darwin", "arch": "arm64", "cpu": "Apple Test Chip",
            "ram_gib": 48.0, "accelerator": "Metal test accelerator",
        }
        race = llmrig.RaceResult(
            "completed", "logical/model", None, llmrig.RACE_METHOD_VERSION,
            "fixed", self.race_workload(), hardware, (), (), (competitor,),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = llmrig.export_race_passports(race, Path(directory))
            self.assertEqual(len(paths), 1)
            document = json.loads(paths[0].read_text())
        self.assertEqual(
            document["hardware"],
            {"os": "Darwin", "architecture": "arm64", "cpu_or_chip": "Apple Test Chip", "ram_gib": 48.0, "accelerator": "Metal test accelerator"},
        )
        self.assertEqual(llmrig.validate_passport(document), [])

if __name__ == "__main__":
    unittest.main()
