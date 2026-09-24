import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from _llmrig import acquisition


class _FakeApi:
    def __init__(self, calls):
        self.calls = calls

    def model_info(self, repository_id, files_metadata=False):
        self.calls.append(("model_info", repository_id, files_metadata))
        return SimpleNamespace(sha="a" * 40)


class _FakeHub:
    def __init__(self):
        self.calls = []

    def HfApi(self):
        return _FakeApi(self.calls)

    def snapshot_download(self, **kwargs):
        self.calls.append(("snapshot_download", kwargs))
        return "/private/models/org/model"

    def hf_hub_download(self, **kwargs):
        self.calls.append(("hf_hub_download", kwargs))
        return "/private/cache/model-Q4_K_M.gguf"


class AcquisitionTests(unittest.TestCase):
    def test_parse_exact_hf_artifact(self):
        source = acquisition.parse_hf_artifact_id("hf://org/model/sub/model-Q4_K_M.gguf")
        self.assertEqual(source.repository_id, "org/model")
        self.assertEqual(source.artifact_path, "sub/model-Q4_K_M.gguf")
        self.assertTrue(source.is_gguf_file)
        self.assertFalse(source.is_snapshot)

    def test_parse_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            acquisition.parse_hf_artifact_id("hf://org/model/../secret.gguf")

    def test_snapshot_is_pinned_and_private_locator_is_not_public(self):
        hub = _FakeHub()
        with mock.patch.object(acquisition, "_hub_module", return_value=hub), mock.patch.object(
            acquisition, "record_acquisition"
        ) as record, mock.patch.object(
            acquisition, "_omlx_destination", return_value=Path("/private/omlx/org/model")
        ):
            acquired = acquisition.acquire_huggingface_artifact(
                "hf://org/model/mlx", "omlx"
            )

        self.assertEqual(acquired.record.revision, "a" * 40)
        self.assertEqual(acquired.record.repository_id, "org/model")
        self.assertEqual(acquired.record.runtime, "omlx")
        public = acquired.record.to_dict()
        self.assertNotIn("/private", str(public))
        self.assertNotIn("locator", public)
        self.assertNotIn("/private", repr(acquired))
        with self.assertRaises(TypeError):
            copy.copy(acquired)
        with self.assertRaises(TypeError):
            copy.deepcopy(acquired)
        record.assert_called_once_with(acquired.record)

        snapshot = next(call for call in hub.calls if call[0] == "snapshot_download")
        self.assertEqual(snapshot[1]["revision"], "a" * 40)
        self.assertEqual(
            Path(snapshot[1]["local_dir"]), Path("/private/omlx/org/model")
        )

    def test_gguf_download_uses_exact_file_and_revision(self):
        hub = _FakeHub()
        with mock.patch.object(acquisition, "_hub_module", return_value=hub), mock.patch.object(
            acquisition, "record_acquisition"
        ):
            acquired = acquisition.acquire_huggingface_artifact(
                "hf://org/model/model-Q4_K_M.gguf", "llama.cpp"
            )

        call = next(call for call in hub.calls if call[0] == "hf_hub_download")
        self.assertEqual(call[1]["filename"], "model-Q4_K_M.gguf")
        self.assertEqual(call[1]["revision"], "a" * 40)
        self.assertEqual(acquired.record.acquisition_kind, "huggingface-file")

    def test_registry_persists_only_path_free_records(self):
        record = acquisition.AcquisitionRecord(
            artifact_id="hf://org/model/mlx",
            repository_id="org/model",
            revision="b" * 40,
            runtime="omlx",
            acquisition_kind="huggingface-snapshot",
            status="completed",
            completed_at="2026-09-24T15:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acquisitions.json"
            with mock.patch.object(acquisition, "acquisition_registry_file", return_value=path):
                acquisition.record_acquisition(record)
                loaded = acquisition.completed_acquisitions(
                    runtime="omlx", repository_id="org/model"
                )
                raw = path.read_text(encoding="utf-8")

        self.assertEqual(loaded, (record,))
        self.assertNotIn("/Users/", raw)
        self.assertNotIn("private", raw.lower())


if __name__ == "__main__":
    unittest.main()
