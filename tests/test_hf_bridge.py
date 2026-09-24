from __future__ import annotations

import unittest
from unittest import mock

import llmrig

from _llmrig.hf_bridge import hf_metadata_for_legacy


class HuggingFaceBridgeTests(unittest.TestCase):
    def resolve(self, payload):
        with hf_metadata_for_legacy(llmrig), mock.patch.object(
            llmrig, "http_json", return_value=(payload, {})
        ):
            return llmrig.HuggingFaceModelSource().resolve(
                str(payload.get("id") or "owner/model")
            )

    def test_tag_only_mlx_hint_does_not_promote_safetensors_format(self):
        result = self.resolve(
            {
                "id": "owner/model",
                "tags": ["mlx"],
                "siblings": [{"rfilename": "model.safetensors", "size": 100}],
            }
        )
        self.assertEqual(result.artifacts[0].format, "Safetensors")
        self.assertIn(
            "generic safetensors do not establish MLX/oMLX compatibility",
            result.artifacts[0].unknowns,
        )

    def test_explicit_library_metadata_promotes_mlx_format(self):
        result = self.resolve(
            {
                "id": "owner/model",
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
        )
        artifact = result.artifacts[0]
        self.assertEqual(artifact.format, "MLX")
        self.assertEqual(artifact.quantization, "4-bit")
        self.assertEqual(artifact.context_max, 32768)
        self.assertEqual(artifact.size_bytes, 300)

    def test_conflicting_context_stays_unknown_in_legacy_artifact(self):
        result = self.resolve(
            {
                "id": "owner/model",
                "config": {
                    "max_position_embeddings": 32768,
                    "seq_length": 65536,
                },
                "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 100}],
            }
        )
        artifact = result.artifacts[0]
        self.assertIsNone(artifact.context_max)
        self.assertIn("context limit metadata is conflicting", artifact.unknowns)

    def test_ambiguous_base_model_does_not_change_logical_identity(self):
        result = self.resolve(
            {
                "id": "converter/model",
                "cardData": {"base_model": ["upstream/a", "upstream/b"]},
                "siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 100}],
            }
        )
        self.assertEqual(result.model.model_id, "converter/model")
        self.assertEqual(result.artifacts[0].model_id, "converter/model")
        self.assertIn("base model association is ambiguous", result.artifacts[0].unknowns)

    def test_bridge_restores_legacy_functions(self):
        original_artifacts = llmrig.hf_artifacts_from_metadata
        original_base_model = llmrig.hf_base_model_id
        with hf_metadata_for_legacy(llmrig):
            self.assertIsNot(llmrig.hf_artifacts_from_metadata, original_artifacts)
            self.assertIsNot(llmrig.hf_base_model_id, original_base_model)
        self.assertIs(llmrig.hf_artifacts_from_metadata, original_artifacts)
        self.assertIs(llmrig.hf_base_model_id, original_base_model)


if __name__ == "__main__":
    unittest.main()
