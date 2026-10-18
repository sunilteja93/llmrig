"""Regression tests for pathological Ollama prompt-throughput denominators."""
import unittest
from unittest import mock

import llmrig


class OllamaPromptTimingTests(unittest.TestCase):
    @staticmethod
    def target():
        configuration = llmrig.RaceConfiguration(
            logical_model_id="qwen3.8",
            runtime="ollama",
            artifact_id="qwen3.8:27b-mlx",
            artifact_format="MLX",
            quantization=None,
            runtime_version="0.33.3",
            eligible=True,
        )
        return llmrig.ExecutionTarget(configuration, "qwen3.8:27b-mlx")

    def measure(self, prompt_duration_ns, prompt_count=25):
        response = {
            "eval_count": 128,
            "eval_duration": 2500000000,
            "prompt_eval_count": prompt_count,
            "prompt_eval_duration": prompt_duration_ns,
            "load_duration": 0,
            "total_duration": 2500000000 + prompt_duration_ns,
        }
        with mock.patch.object(llmrig.OLLAMA_RUNTIME, "is_available", return_value=True), \
             mock.patch.object(llmrig, "isolate_ollama_for_benchmark"), \
             mock.patch.object(llmrig, "unload_ollama_model"), \
             mock.patch.object(llmrig, "ollama_generate", return_value=response):
            return llmrig.OllamaExecutionAdapter().benchmark(
                self.target(),
                llmrig.RaceWorkload(prompt="small repeatable prompt", warmup_runs=0, runs=2),
            )

    def test_microsecond_prompt_phase_is_not_an_advertised_throughput(self):
        result = self.measure(5000)
        self.assertEqual(result.execution_status, "success")
        self.assertEqual(result.measured_runs, 2)
        self.assertIsNone(result.prompt_eval_tps)
        self.assertAlmostEqual(result.generation_tps, 51.2, places=1)
        self.assertTrue(any("under 1 ms" in warning for warning in result.warnings))
        self.assertTrue(all(sample["prompt_tps"] is None for sample in result.raw_samples))
        self.assertTrue(all(sample["eval_count"] == 128 for sample in result.raw_samples))

    def test_sufficient_prompt_timing_retains_measured_throughput(self):
        result = self.measure(50000000)
        self.assertEqual(result.execution_status, "success")
        self.assertEqual(result.prompt_eval_tps, 500.0)
        self.assertFalse(any("under 1 ms" in warning for warning in result.warnings))

    def test_submillisecond_boundary_is_not_rounded_into_valid_timing(self):
        result = self.measure(500000)
        self.assertIsNone(result.prompt_eval_tps)
        self.assertEqual(result.measured_runs, 2)


if __name__ == "__main__":
    unittest.main()
