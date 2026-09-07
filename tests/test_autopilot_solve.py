import ast
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import llmrig
from _llmrig import (
    CompatibilityState,
    ExecutionState,
    LocalAvailabilityState,
    MeasurementState,
    RecommendationState,
    RuntimeAvailabilityState,
    SOLVE_SCHEMA_VERSION,
    SolveRequest,
)


class AutopilotSolveTests(unittest.TestCase):
    def profile(self, ram_gib=48.0):
        return {
            "os": "Darwin",
            "arch": "arm64",
            "ram_gib": ram_gib,
            "gpus": [{"name": "Apple", "vram_gb": None, "backend": "Metal"}],
            "disk": {"free_gib": 500.0},
        }

    def capability(self, runtime="ollama", available=True):
        formats = {
            "ollama": ("Ollama",),
            "llama.cpp": ("GGUF",),
            "mlx-lm": ("MLX",),
        }[runtime]
        return llmrig.RuntimeCapability(
            runtime=runtime,
            installed=available,
            available=available,
            version="1.0" if available else None,
            supported_artifact_formats=formats,
            supported_platforms=llmrig.ALL_PLATFORMS,
            supported_architectures=(),
            runtime_execution_capable=True,
            llmrig_installation_supported=False,
            llmrig_execution_supported=True,
            llmrig_benchmark_supported=True,
            confidence=llmrig.Confidence.HIGH,
            evidence=(
                llmrig.RecommendationEvidence(
                    "verified-local-runtime", "test", "runtime state was observed"
                ),
            ),
        )

    def installed(self, name="qwen3:0.6b"):
        return [{"name": name, "id": "abcdef123456", "raw": "ignored"}]

    def generic_resolution(self, context_max=32768):
        model = llmrig.Model("org/model", "model", None, ("text",))
        artifact = llmrig.ModelArtifact(
            "hf://org/model/model-Q4_K_M.gguf",
            model.model_id,
            None,
            "GGUF",
            4.0,
            context_max,
            (),
            quantization="Q4_K_M",
            size_bytes=4_000_000_000,
            evidence=(
                llmrig.RecommendationEvidence(
                    "verified-metadata", "test", "artifact metadata was resolved"
                ),
            ),
        )
        return llmrig.ModelResolution(
            "resolved",
            model,
            (artifact,),
            llmrig.Confidence.HIGH,
            (
                llmrig.RecommendationEvidence(
                    "verified-metadata", "test", "repository was resolved"
                ),
            ),
        )

    def solve(self, model="qwen3:0.6b", inventory=None, capabilities=None, context=None, local=(), verify=False):
        inventory = [] if inventory is None else inventory
        capabilities = capabilities or (self.capability(),)
        request = SolveRequest(
            model,
            context,
            tuple(runtime for runtime, _ in llmrig.parse_local_artifact_values(local)),
            verify,
        )
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.profile()
        ), mock.patch.object(
            llmrig, "runtime_capabilities", return_value=capabilities
        ), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=inventory
        ):
            return llmrig.solve_for_request(request, local)

    def command(self, model="qwen3:0.6b", json_mode=True, context=None, local=(), inventory=None, capabilities=None, verify=False):
        output = io.StringIO()
        args = llmrig.argparse.Namespace(
            model=model,
            json=json_mode,
            context=context,
            local_artifact=list(local),
            verify=verify,
        )
        inventory = [] if inventory is None else inventory
        capabilities = capabilities or (self.capability(),)
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.profile()
        ), mock.patch.object(
            llmrig, "runtime_capabilities", return_value=capabilities
        ), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=inventory
        ), contextlib.redirect_stdout(output):
            code = llmrig.command_solve(args)
        return code, output.getvalue()

    def measured_competitor(self, target, workload, generation=100.0, prompt=200.0, latency=1.0):
        configuration = getattr(target, "configuration", target)
        sample = {
            "generation_tps": generation,
            "prompt_tps": prompt,
            "wall_seconds": latency,
            "eval_count": workload.num_predict,
            "prompt_eval_count": 16,
        }
        return llmrig.RaceCompetitor(
            configuration.logical_model_id,
            configuration.runtime,
            configuration.artifact_id,
            configuration.artifact_fingerprint,
            configuration.artifact_format,
            configuration.quantization,
            configuration.runtime_version,
            "success",
            generation,
            prompt,
            latency,
            workload.num_predict * 2,
            2,
            2,
            2,
            2,
            "2026-01-01T00:00:00+00:00",
            evidence=(
                llmrig.RecommendationEvidence(
                    "measured", "test race-v2", "two timed samples completed"
                ),
            ),
            warnings=("performance measurement does not establish model quality",),
            raw_samples=(sample, sample),
        )

    def verified_with_native(
        self,
        native_value,
        native_runtime="llama.cpp",
        rates=None,
        failing_runtime=None,
        context=None,
    ):
        rates = rates or {
            "ollama": (100.0, 200.0, 1.0),
            native_runtime: (70.0, 150.0, 2.0),
        }

        def benchmark(_adapter, target, workload):
            configuration = getattr(target, "configuration", target)
            if configuration.runtime == failing_runtime:
                raise RuntimeError("private runtime failure")
            return self.measured_competitor(
                target, workload, *rates[configuration.runtime]
            )

        capabilities = (self.capability(), self.capability(native_runtime))
        with mock.patch.object(
            llmrig.OllamaExecutionAdapter, "benchmark", autospec=True, side_effect=benchmark
        ), mock.patch.object(
            llmrig.LlamaCppExecutionAdapter, "benchmark", autospec=True, side_effect=benchmark
        ), mock.patch.object(
            llmrig.MlxExecutionAdapter, "benchmark", autospec=True, side_effect=benchmark
        ):
            return self.solve(
                inventory=self.installed(),
                capabilities=capabilities,
                context=context,
                local=(native_value,),
                verify=True,
            )

    def test_curated_model_solve_can_recommend_one_observed_runnable_candidate(self):
        result = self.solve(inventory=self.installed())
        candidate = result.candidates[0]
        self.assertEqual(result.status, "analyzed")
        self.assertEqual(candidate.assessments.compatibility.state, CompatibilityState.COMPATIBLE)
        self.assertEqual(candidate.assessments.local_availability.state, LocalAvailabilityState.AVAILABLE)
        self.assertEqual(candidate.assessments.execution.state, ExecutionState.EXECUTABLE)
        self.assertEqual(candidate.assessments.measurement.state, MeasurementState.NOT_REQUESTED)
        self.assertEqual(candidate.assessments.recommendation.state, RecommendationState.RECOMMENDED)
        self.assertEqual(candidate.recipe.status, "already_runnable")

    def test_generic_model_resolution_reuses_artifact_and_runtime_assessment(self):
        resolution = self.generic_resolution()
        capabilities = (self.capability("llama.cpp"),)
        with mock.patch.object(llmrig.HF_SOURCE, "resolve", return_value=resolution):
            result = self.solve("org/model", capabilities=capabilities)
        self.assertEqual(result.resolution_status, "resolved")
        self.assertEqual(result.logical_model_id, "org/model")
        self.assertEqual(result.candidates[0].configuration.runtime, "llama.cpp")
        self.assertEqual(
            result.candidates[0].assessments.compatibility.state,
            CompatibilityState.COMPATIBLE,
        )

    def test_compatible_but_not_local_remains_orthogonal(self):
        result = self.solve(inventory=self.installed("other/model:latest"))
        candidate = result.candidates[0]
        self.assertEqual(candidate.assessments.compatibility.state, CompatibilityState.COMPATIBLE)
        self.assertEqual(candidate.assessments.local_availability.state, LocalAvailabilityState.NOT_AVAILABLE)
        self.assertEqual(candidate.assessments.execution.state, ExecutionState.NOT_EXECUTABLE)
        self.assertEqual(candidate.recipe.status, "known_setup_path")

    def test_local_but_runtime_unavailable_does_not_change_compatibility(self):
        result = self.solve(
            inventory=self.installed(), capabilities=(self.capability(available=False),)
        )
        candidate = result.candidates[0]
        self.assertEqual(candidate.assessments.compatibility.state, CompatibilityState.COMPATIBLE)
        self.assertEqual(candidate.assessments.local_availability.state, LocalAvailabilityState.AVAILABLE)
        self.assertEqual(candidate.assessments.runtime_availability.state, RuntimeAvailabilityState.UNAVAILABLE)
        self.assertEqual(candidate.assessments.execution.state, ExecutionState.NOT_EXECUTABLE)

    def test_runtime_available_but_observed_artifact_absent(self):
        result = self.solve(inventory=self.installed("other/model:latest"))
        candidate = result.candidates[0]
        self.assertEqual(candidate.assessments.runtime_availability.state, RuntimeAvailabilityState.AVAILABLE)
        self.assertEqual(candidate.assessments.local_availability.state, LocalAvailabilityState.NOT_AVAILABLE)

    def test_empty_ollama_inventory_preserves_unknown_absence(self):
        result = self.solve(inventory=[])
        candidate = result.candidates[0]
        self.assertEqual(candidate.assessments.local_availability.state, LocalAvailabilityState.UNKNOWN)
        self.assertEqual(candidate.assessments.execution.state, ExecutionState.UNKNOWN)
        self.assertIn("empty Ollama observation", candidate.assessments.local_availability.unknowns[0])

    def test_ollama_positive_inventory_observation_is_not_identity_attestation(self):
        candidate = self.solve(inventory=self.installed()).candidates[0]
        self.assertIs(candidate.local_identity_attested, False)
        self.assertEqual(candidate.local_identity_confidence, llmrig.Confidence.HIGH)

    def test_explicit_unresolved_alternative_prevents_practical_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            result = self.solve(
                inventory=self.installed(),
                local=(f"llama.cpp={artifact}",),
                capabilities=(self.capability(), self.capability("llama.cpp")),
            )
        self.assertEqual(result.plan.recommendation_status, "inconclusive")
        self.assertTrue(
            all(
                item.assessments.recommendation.state == RecommendationState.INCONCLUSIVE
                for item in result.candidates
            )
        )

    def test_explicit_gguf_is_private_unattested_and_quantization_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "secret-model-Q4.gguf"
            artifact.write_bytes(b"weights")
            value = f"llama.cpp={artifact}"
            result = self.solve(
                local=(value,),
                capabilities=(self.capability(), self.capability("llama.cpp")),
            )
            candidate = next(item for item in result.candidates if item.configuration.runtime == "llama.cpp")
            public = json.dumps(result.to_dict(), sort_keys=True)
            self.assertEqual(candidate.configuration.quantization, None)
            self.assertIs(candidate.local_identity_attested, False)
            self.assertEqual(candidate.local_identity_confidence, llmrig.Confidence.UNKNOWN)
            self.assertTrue(candidate.recipe.private_locator_required)
            self.assertNotIn(str(artifact), public)
            self.assertNotIn(artifact.name, public)

    def test_explicit_mlx_is_private_and_human_output_has_no_path(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "private-mlx"
            artifact.mkdir()
            (artifact / "config.json").write_text("{}", encoding="utf-8")
            (artifact / "model.safetensors").write_bytes(b"weights")
            value = f"mlx-lm={artifact}"
            code, output = self.command(
                json_mode=False,
                local=(value,),
                capabilities=(self.capability(), self.capability("mlx-lm")),
            )
            self.assertEqual(code, 0)
            self.assertIn("local:mlx-lm:user-supplied", output)
            self.assertIn("Local identity attested: no", output)
            self.assertNotIn(str(artifact), output)
            self.assertNotIn(artifact.name, output)

    def test_context_is_recorded_and_excess_is_incompatible(self):
        result = self.solve(context=1_000_000, inventory=self.installed())
        candidate = result.candidates[0]
        self.assertEqual(candidate.context_tokens, 1_000_000)
        self.assertEqual(candidate.assessments.compatibility.state, CompatibilityState.INCOMPATIBLE)
        self.assertEqual(candidate.assessments.execution.state, ExecutionState.NOT_EXECUTABLE)

    def test_invalid_context_and_unresolvable_model_exit_two(self):
        code, output = self.command(context=0)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output)["status"], "invalid_request")
        code, output = self.command(model="not-a-repository")
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output)["resolution"]["status"], "invalid_identifier")

    def test_private_path_shaped_model_identifier_is_rejected_without_leakage(self):
        private = str(Path.home() / "Secret Models" / "model.gguf")
        code, output = self.command(model=private)
        self.assertEqual(code, 2)
        self.assertNotIn(private, output)
        self.assertNotIn("Secret Models", output)

    def test_successful_incompatibility_exits_zero(self):
        code, output = self.command(
            model="qwen3.8:27b-bf16", inventory=self.installed("other/model:latest")
        )
        self.assertEqual(code, 0)
        payload = json.loads(output)
        self.assertEqual(payload["status"], "analyzed")
        self.assertEqual(payload["candidates"][0]["assessments"]["compatibility"]["state"], "incompatible")

    def test_json_is_deterministic_schema_versioned_and_domain_derived(self):
        first_code, first = self.command(inventory=self.installed())
        second_code, second = self.command(inventory=self.installed())
        self.assertEqual((first_code, first), (second_code, second))
        payload = json.loads(first)
        self.assertEqual(payload["schema_version"], SOLVE_SCHEMA_VERSION)
        self.assertEqual(payload["status"], "analyzed")

    def test_generic_network_failure_is_operational_exit_one(self):
        failure = llmrig.ModelResolution("network_error", None)
        with mock.patch.object(llmrig.HF_SOURCE, "resolve", return_value=failure):
            code, output = self.command(model="org/model")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output)["status"], "engine_failure")

    def test_solve_parser_adds_verify_without_unrelated_objectives(self):
        parser = llmrig.build_parser()
        args = parser.parse_args(
            ["solve", "org/model", "--json", "--context", "4096", "--verify", "--local-artifact", "llama.cpp=/tmp/a.gguf"]
        )
        self.assertEqual(args.command, "solve")
        self.assertEqual(args.context, 4096)
        self.assertTrue(args.verify)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["solve", "org/model", "--objective", "speed"])

    def test_solve_does_not_mutate_execute_benchmark_or_scan(self):
        with mock.patch.object(llmrig, "hardware_profile", return_value=self.profile()), mock.patch.object(
            llmrig, "runtime_capabilities", return_value=(self.capability(),)
        ), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=self.installed()
        ) as inventory, mock.patch.object(
            llmrig.OLLAMA_RUNTIME, "ensure_available"
        ) as service_start, mock.patch.object(
            llmrig, "pull_model"
        ) as pull, mock.patch.object(
            llmrig, "install_ollama_help"
        ) as install, mock.patch.object(
            llmrig, "ollama_generate"
        ) as inference, mock.patch.object(
            llmrig, "run_benchmark"
        ) as benchmark, mock.patch.object(
            subprocess, "Popen"
        ) as popen, mock.patch(
            "os.walk"
        ) as walk, mock.patch.object(
            Path, "rglob"
        ) as recursive_glob:
            result = llmrig.solve_for_request(SolveRequest("qwen3:0.6b"), ())
        self.assertEqual(result.status, "analyzed")
        inventory.assert_called_once()
        service_start.assert_not_called()
        pull.assert_not_called()
        install.assert_not_called()
        inference.assert_not_called()
        benchmark.assert_not_called()
        popen.assert_not_called()
        walk.assert_not_called()
        recursive_glob.assert_not_called()

    def test_verify_happy_path_reuses_race_v2_and_balanced_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            with mock.patch.object(
                llmrig, "execute_race", wraps=llmrig.execute_race
            ) as race, mock.patch.object(
                llmrig, "analyze_decision", wraps=llmrig.analyze_decision
            ) as decision:
                result = self.verified_with_native(f"llama.cpp={artifact}")
        self.assertEqual(result.verification.status, "completed")
        self.assertEqual(result.verification.method_version, llmrig.RACE_METHOD_VERSION)
        self.assertEqual(len(result.verification.candidate_ids), 2)
        self.assertTrue(all(
            item.assessments.measurement.state == MeasurementState.MEASURED
            for item in result.candidates
        ))
        race.assert_called_once()
        decision.assert_called_once()
        self.assertEqual(decision.call_args.args[1], "balanced")

    def test_verify_with_fewer_than_two_candidates_is_structured_unavailable(self):
        with mock.patch.object(
            llmrig.OllamaExecutionAdapter, "benchmark"
        ) as benchmark:
            result = self.solve(inventory=self.installed(), verify=True)
        self.assertEqual(result.status, "analyzed")
        self.assertEqual(result.verification.status, "unavailable")
        self.assertIn("at least 2", result.verification.reason)
        self.assertEqual(llmrig.solve_exit_code(result), 2)
        self.assertEqual(
            result.candidates[0].assessments.measurement.state,
            MeasurementState.VERIFICATION_UNAVAILABLE,
        )
        benchmark.assert_not_called()

    def test_competitor_failure_invalidates_verification_without_incompatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            result = self.verified_with_native(
                f"llama.cpp={artifact}", failing_runtime="llama.cpp"
            )
        states = {item.configuration.runtime: item.assessments for item in result.candidates}
        self.assertEqual(result.verification.status, "failed")
        self.assertEqual(llmrig.solve_exit_code(result), 1)
        self.assertEqual(states["ollama"].measurement.state, MeasurementState.MEASURED)
        self.assertEqual(states["llama.cpp"].measurement.state, MeasurementState.FAILED)
        self.assertEqual(states["llama.cpp"].compatibility.state, CompatibilityState.COMPATIBLE)
        self.assertEqual(result.verification.recommendation.recommendation_status, "inconclusive")

    def test_five_percent_threshold_and_pareto_tradeoffs_remain_inconclusive(self):
        cases = (
            {
                "ollama": (100.0, 200.0, 1.0),
                "llama.cpp": (96.0, 192.0, 1.04),
            },
            {
                "ollama": (100.0, 100.0, 2.0),
                "llama.cpp": (70.0, 180.0, 1.0),
            },
        )
        for rates in cases:
            with self.subTest(rates=rates), tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory) / "model.gguf"
                artifact.write_bytes(b"weights")
                result = self.verified_with_native(
                    f"llama.cpp={artifact}", rates=rates
                )
                self.assertEqual(result.verification.status, "completed")
                self.assertEqual(
                    result.verification.recommendation.recommendation_status,
                    "inconclusive",
                )
                self.assertTrue(all(
                    item.assessments.recommendation.state
                    == RecommendationState.INCONCLUSIVE
                    for item in result.candidates
                ))

    def test_measured_pareto_leader_supersedes_but_does_not_erase_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            result = self.verified_with_native(f"llama.cpp={artifact}")
        self.assertEqual(result.plan.recommendation_status, "inconclusive")
        self.assertEqual(result.verification.recommendation.recommendation_status, "recommended")
        self.assertIn("ollama", result.verification.recommendation.recommended_candidate_id)
        selected = next(
            item for item in result.candidates
            if item.candidate_id == result.verification.recommendation.recommended_candidate_id
        )
        self.assertEqual(
            selected.assessments.recommendation.state, RecommendationState.RECOMMENDED
        )

    def test_unsupported_verify_context_preserves_compatibility_and_runs_no_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            with mock.patch.object(
                llmrig.OllamaExecutionAdapter, "benchmark"
            ) as ollama_benchmark, mock.patch.object(
                llmrig.LlamaCppExecutionAdapter, "benchmark"
            ) as native_benchmark:
                result = self.solve(
                    inventory=self.installed(),
                    capabilities=(self.capability(), self.capability("llama.cpp")),
                    context=32769,
                    local=(f"llama.cpp={artifact}",),
                    verify=True,
                )
        self.assertEqual(result.verification.status, "unavailable")
        self.assertIn("unsupported", result.verification.reason)
        self.assertTrue(any(
            item.assessments.compatibility.state == CompatibilityState.COMPATIBLE
            for item in result.candidates
        ))
        self.assertTrue(all(
            item.assessments.measurement_capability.state.value == "not_measurable"
            for item in result.candidates
            if item.candidate_id in result.verification.candidate_ids
        ))
        ollama_benchmark.assert_not_called()
        native_benchmark.assert_not_called()

    def test_private_gguf_and_mlx_locators_are_handed_off_only_internally(self):
        for runtime in ("llama.cpp", "mlx-lm"):
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if runtime == "llama.cpp":
                    artifact = root / "secret-model.gguf"
                    artifact.write_bytes(b"weights")
                else:
                    artifact = root / "secret-mlx"
                    artifact.mkdir()
                    (artifact / "config.json").write_text("{}", encoding="utf-8")
                    (artifact / "model.safetensors").write_bytes(b"weights")
                captured = []

                def benchmark(_adapter, target, workload):
                    if getattr(target, "configuration", target).runtime == runtime:
                        captured.append(target)
                    rate = 70.0 if getattr(target, "configuration", target).runtime == runtime else 100.0
                    return self.measured_competitor(target, workload, rate, rate * 2, 2.0 if rate == 70 else 1.0)

                adapter = (
                    llmrig.LlamaCppExecutionAdapter
                    if runtime == "llama.cpp"
                    else llmrig.MlxExecutionAdapter
                )
                with mock.patch.object(
                    llmrig.OllamaExecutionAdapter, "benchmark", autospec=True, side_effect=benchmark
                ), mock.patch.object(
                    adapter, "benchmark", autospec=True, side_effect=benchmark
                ):
                    result = self.solve(
                        inventory=self.installed(),
                        capabilities=(self.capability(), self.capability(runtime)),
                        local=(f"{runtime}={artifact}",),
                        verify=True,
                    )
                self.assertEqual(len(captured), 1)
                self.assertIsInstance(captured[0], llmrig.ExecutionTarget)
                public = json.dumps(result.to_dict()) + repr(result) + repr(captured[0])
                self.assertNotIn(str(artifact), public)
                self.assertNotIn(artifact.name, public)

    def test_private_path_is_absent_from_failed_verify_json_and_human_output(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "private-model.gguf"
            artifact.write_bytes(b"weights")

            def benchmark(_adapter, target, workload):
                configuration = getattr(target, "configuration", target)
                if configuration.runtime == "llama.cpp":
                    raise RuntimeError(str(artifact))
                return self.measured_competitor(target, workload)

            args = llmrig.argparse.Namespace(
                model="qwen3:0.6b", json=False, context=None,
                local_artifact=[f"llama.cpp={artifact}"], verify=True,
            )
            output = io.StringIO()
            with mock.patch.object(
                llmrig, "hardware_profile", return_value=self.profile()
            ), mock.patch.object(
                llmrig, "runtime_capabilities",
                return_value=(self.capability(), self.capability("llama.cpp")),
            ), mock.patch.object(
                llmrig, "installed_ollama_models", return_value=self.installed()
            ), mock.patch.object(
                llmrig.OllamaExecutionAdapter, "benchmark", autospec=True, side_effect=benchmark
            ), mock.patch.object(
                llmrig.LlamaCppExecutionAdapter, "benchmark", autospec=True, side_effect=benchmark
            ), contextlib.redirect_stdout(output):
                code = llmrig.command_solve(args)
            public = output.getvalue()
        self.assertEqual(code, 1)
        self.assertNotIn(str(artifact), public)
        self.assertNotIn(artifact.name, public)
        self.assertIn("benchmark execution failed", public)

    def test_sdk_read_only_has_no_print_exit_or_inference(self):
        output = io.StringIO()
        with mock.patch.object(
            llmrig, "hardware_profile", return_value=self.profile()
        ), mock.patch.object(
            llmrig, "runtime_capabilities", return_value=(self.capability(),)
        ), mock.patch.object(
            llmrig, "installed_ollama_models", return_value=self.installed()
        ), mock.patch.object(
            llmrig, "execute_race"
        ) as race, mock.patch.object(
            llmrig.sys, "exit"
        ) as sys_exit, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = llmrig.solve("qwen3:0.6b")
        self.assertIsInstance(result, llmrig.SolveResult)
        self.assertEqual(result.verification.status, "not_requested")
        self.assertEqual(output.getvalue(), "")
        race.assert_not_called()
        sys_exit.assert_not_called()

    def test_sdk_verify_uses_same_engine_and_returns_result(self):
        expected = self.solve(inventory=self.installed())
        with mock.patch.object(
            llmrig, "solve_for_request", return_value=expected
        ) as engine:
            result = llmrig.solve("qwen3:0.6b", verify=True)
        self.assertIs(result, expected)
        self.assertTrue(engine.call_args.args[0].verify)

    def test_sdk_verify_executes_existing_race_without_printing(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")

            def benchmark(_adapter, target, workload):
                configuration = getattr(target, "configuration", target)
                rate = 100.0 if configuration.runtime == "ollama" else 70.0
                return self.measured_competitor(
                    target, workload, rate, rate * 2, 1.0 if rate == 100 else 2.0
                )

            output = io.StringIO()
            with mock.patch.object(
                llmrig, "hardware_profile", return_value=self.profile()
            ), mock.patch.object(
                llmrig, "runtime_capabilities",
                return_value=(self.capability(), self.capability("llama.cpp")),
            ), mock.patch.object(
                llmrig, "installed_ollama_models", return_value=self.installed()
            ), mock.patch.object(
                llmrig.OllamaExecutionAdapter, "benchmark", autospec=True,
                side_effect=benchmark,
            ), mock.patch.object(
                llmrig.LlamaCppExecutionAdapter, "benchmark", autospec=True,
                side_effect=benchmark,
            ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = llmrig.solve(
                    "qwen3:0.6b",
                    local_artifacts=(f"llama.cpp={artifact}",),
                    verify=True,
                )
        self.assertEqual(result.verification.status, "completed")
        self.assertEqual(output.getvalue(), "")

    def test_cli_delegates_to_public_sdk_function(self):
        expected = self.solve(inventory=self.installed())
        args = llmrig.argparse.Namespace(
            model="qwen3:0.6b", json=True, context=None,
            local_artifact=[], verify=False,
        )
        output = io.StringIO()
        with mock.patch.object(llmrig, "solve", return_value=expected) as sdk, \
             contextlib.redirect_stdout(output):
            code = llmrig.command_solve(args)
        self.assertEqual(code, 0)
        sdk.assert_called_once_with(
            "qwen3:0.6b", context=None, local_artifacts=[], verify=False
        )
        self.assertEqual(json.loads(output.getvalue()), expected.to_dict())

    def test_sdk_invalid_input_raises_deliberate_exception_without_exit_or_print(self):
        output = io.StringIO()
        with mock.patch.object(llmrig.sys, "exit") as sys_exit, \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with self.assertRaises(llmrig.SolveInputError):
                llmrig.solve("", context=0)
        self.assertEqual(output.getvalue(), "")
        sys_exit.assert_not_called()

    def test_verify_does_not_pull_install_start_service_or_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            with mock.patch.object(
                llmrig, "execute_race",
                wraps=llmrig.execute_race,
            ), mock.patch.object(
                llmrig.OllamaExecutionAdapter, "benchmark", autospec=True,
                side_effect=lambda _self, target, workload: self.measured_competitor(target, workload),
            ), mock.patch.object(
                llmrig.LlamaCppExecutionAdapter, "benchmark", autospec=True,
                side_effect=lambda _self, target, workload: self.measured_competitor(target, workload, 70, 150, 2),
            ), mock.patch.object(
                llmrig.OLLAMA_RUNTIME, "ensure_available"
            ) as service_start, mock.patch.object(
                llmrig, "pull_model"
            ) as pull, mock.patch.object(
                llmrig, "install_ollama_help"
            ) as install, mock.patch("os.walk") as walk, mock.patch.object(
                Path, "rglob"
            ) as recursive_glob:
                result = self.solve(
                    inventory=self.installed(),
                    capabilities=(self.capability(), self.capability("llama.cpp")),
                    local=(f"llama.cpp={artifact}",),
                    verify=True,
                )
        self.assertEqual(result.verification.status, "completed")
        service_start.assert_not_called()
        pull.assert_not_called()
        install.assert_not_called()
        walk.assert_not_called()
        recursive_glob.assert_not_called()

    def test_verification_serialization_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.gguf"
            artifact.write_bytes(b"weights")
            result = self.verified_with_native(f"llama.cpp={artifact}")
        self.assertEqual(result.to_dict(), result.to_dict())
        self.assertEqual(
            json.dumps(result.to_dict(), sort_keys=True),
            json.dumps(result.to_dict(), sort_keys=True),
        )

    def test_python_39_syntax_compatibility(self):
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "llmrig.py", "_llmrig/solve.py", "_llmrig/inventory.py",
            "_llmrig/planning.py", "tests/test_autopilot_solve.py",
        ):
            ast.parse(
                (root / relative).read_text(encoding="utf-8"),
                filename=relative,
                feature_version=(3, 9),
            )


if __name__ == "__main__":
    unittest.main()
