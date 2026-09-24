import json
import unittest
from unittest import mock

from _llmrig.omlx_api import (
    OmlxApiError,
    list_model_statuses,
    list_models,
    measure_completion,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, limit=-1):
        if limit is None or limit < 0:
            return self._body
        return self._body[:limit]


class OmlxApiTests(unittest.TestCase):
    def test_list_models_is_read_only_deterministic_and_path_free(self):
        payload = {
            "object": "list",
            "data": [
                {
                    "id": "qwen-b",
                    "owned_by": "omlx",
                    "max_model_len": 32768,
                },
                {
                    "id": "qwen-a",
                    "owned_by": "omlx",
                    "max_model_len": 131072,
                },
                {"id": "qwen-b", "owned_by": "omlx"},
                {"id": ""},
                {"unexpected": "record"},
            ],
        }
        with mock.patch(
            "_llmrig.omlx_api.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ) as urlopen:
            result = list_models("http://127.0.0.1:8000")

        self.assertEqual([item.model_id for item in result], ["qwen-a", "qwen-b"])
        self.assertEqual(result[0].max_model_len, 131072)
        self.assertEqual(result[0].owned_by, "omlx")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertTrue(request.full_url.endswith("/v1/models"))
        self.assertNotIn("model_path", result[0].to_dict())

    def test_model_status_keeps_provenance_but_discards_private_path(self):
        payload = {
            "models": [
                {
                    "id": "mlx-community--Qwen3.5-27B-4bit",
                    "model_path": "/Users/private/models/qwen",
                    "source_type": "hf_cache",
                    "source_repo_id": "mlx-community/Qwen3.5-27B-4bit",
                    "loaded": True,
                    "max_context_window": 131072,
                },
                {
                    "id": "local-only",
                    "model_path": "/secret/local-only",
                    "source_type": "local",
                    "source_repo_id": None,
                    "loaded": False,
                    "model_context_length": 32768,
                },
            ]
        }
        with mock.patch(
            "_llmrig.omlx_api.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ) as urlopen:
            result = list_model_statuses("http://127.0.0.1:8000")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].model_id, "local-only")
        self.assertEqual(result[0].max_model_len, 32768)
        self.assertFalse(result[0].loaded)
        matched = result[1]
        self.assertEqual(
            matched.source_repo_id, "mlx-community/Qwen3.5-27B-4bit"
        )
        self.assertEqual(matched.source_type, "hf_cache")
        self.assertTrue(matched.loaded)
        self.assertEqual(matched.max_model_len, 131072)
        self.assertNotIn("model_path", matched.to_dict())
        self.assertNotIn("/Users/private", json.dumps(matched.to_dict()))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertTrue(request.full_url.endswith("/v1/models/status"))

    def test_measure_completion_uses_server_metrics_for_race_sample(self):
        payload = {
            "model": "Qwen3.5-27B-4bit",
            "choices": [{"text": "ok", "index": 0}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 128,
                "total_tokens": 228,
                "total_time": 3.1,
                "prompt_eval_duration": 0.6,
                "generation_duration": 2.4,
                "time_to_first_token": 0.7,
                "prompt_tokens_per_second": 166.67,
                "generation_tokens_per_second": 53.33,
            },
        }
        with mock.patch(
            "_llmrig.omlx_api.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ) as urlopen:
            result = measure_completion(
                "Qwen3.5-27B-4bit",
                "deterministic prompt",
                128,
                base_url="http://127.0.0.1:8000",
            )

        self.assertTrue(result.race_ready)
        self.assertAlmostEqual(result.race_latency_s, 3.0)
        self.assertEqual(
            result.race_sample(),
            {
                "generation_tps": 53.33,
                "prompt_tps": 166.67,
                "wall_seconds": 3.0,
                "prompt_tokens": 100,
                "generated_tokens": 128,
                "measurement_source": "omlx-server-usage",
            },
        )
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        submitted = json.loads(request.data.decode("utf-8"))
        self.assertEqual(submitted["temperature"], 0)
        self.assertEqual(submitted["max_tokens"], 128)
        self.assertEqual(submitted["seed"], 42)
        self.assertFalse(submitted["stream"])

    def test_measurement_falls_back_to_server_total_time(self):
        payload = {
            "model": "model-a",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_time": 1.25,
                "prompt_tokens_per_second": 80.0,
                "generation_tokens_per_second": 16.0,
            },
        }
        with mock.patch(
            "_llmrig.omlx_api.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = measure_completion("model-a", "prompt", 20)

        self.assertEqual(result.race_latency_s, 1.25)
        self.assertTrue(result.race_ready)

    def test_incomplete_server_metrics_do_not_become_race_ready(self):
        payload = {
            "model": "model-a",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_time": 1.25,
                "generation_tokens_per_second": 16.0,
            },
        }
        with mock.patch(
            "_llmrig.omlx_api.urllib.request.urlopen",
            return_value=FakeResponse(payload),
        ):
            result = measure_completion("model-a", "prompt", 20)

        self.assertFalse(result.race_ready)
        with self.assertRaises(OmlxApiError):
            result.race_sample()

    def test_invalid_limits_fail_before_network_access(self):
        with mock.patch("_llmrig.omlx_api.urllib.request.urlopen") as urlopen:
            with self.assertRaises(ValueError):
                measure_completion("model-a", "prompt", 0)
            with self.assertRaises(ValueError):
                measure_completion("model-a", "prompt", 513)
            urlopen.assert_not_called()

    def test_missing_usage_is_explicit_failure(self):
        with mock.patch(
            "_llmrig.omlx_api.urllib.request.urlopen",
            return_value=FakeResponse({"model": "model-a", "choices": []}),
        ):
            with self.assertRaisesRegex(OmlxApiError, "missing usage metrics"):
                measure_completion("model-a", "prompt", 16)


if __name__ == "__main__":
    unittest.main()
