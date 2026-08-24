import importlib.util
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

if __name__ == "__main__":
    unittest.main()
