import contextlib
import importlib.util
import io
import json
import os
import sys
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

if __name__ == "__main__":
    unittest.main()
