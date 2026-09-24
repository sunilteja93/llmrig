from __future__ import annotations

import unittest
from unittest import mock

from _llmrig.privacy import validate_public_text
from _llmrig.runtime_adapters import (
    LlamaCppRuntimeAdapter,
    MlxLmRuntimeAdapter,
    OmlxRuntimeAdapter,
    OllamaRuntimeAdapter,
    RuntimeProbe,
    build_execution_adapters,
    probe_capability_available,
    probe_runtimes,
)


class RuntimeProbeTests(unittest.TestCase):
    def test_locally_usable_requires_install_and_no_blocker(self) -> None:
        probe = RuntimeProbe(
            runtime="test",
            installed=True,
            service_available=None,
            version="1",
            cli_path="/tmp/test",
            supported_artifact_formats=(),
            supported_platforms=(),
            supported_architectures=(),
            execution_api=None,
        )
        self.assertTrue(probe.locally_usable)
        self.assertFalse(
            RuntimeProbe(
                runtime="test",
                installed=True,
                service_available=None,
                version="1",
                cli_path="/tmp/test",
                supported_artifact_formats=(),
                supported_platforms=(),
                supported_architectures=(),
                execution_api=None,
                blockers=("blocked",),
            ).locally_usable
        )

    def test_unknown_availability_policy_fails_closed(self) -> None:
        class Adapter:
            availability_policy = "unsupported-policy"

        probe = RuntimeProbe(
            runtime="test",
            installed=True,
            service_available=True,
            version="1",
            cli_path="/tmp/test",
            supported_artifact_formats=(),
            supported_platforms=(),
            supported_architectures=(),
            execution_api=None,
        )
        self.assertFalse(probe_capability_available(Adapter(), probe))

    @mock.patch("_llmrig.runtime_adapters._json_endpoint_alive", return_value=True)
    @mock.patch("_llmrig.runtime_adapters._run_version", return_value="omlx 1.2.3")
    @mock.patch.object(OmlxRuntimeAdapter, "_cli_path", return_value="/usr/local/bin/omlx")
    @mock.patch("_llmrig.runtime_adapters.platform.machine", return_value="arm64")
    @mock.patch("_llmrig.runtime_adapters.platform.system", return_value="Darwin")
    def test_omlx_probe_detects_apple_silicon_service(
        self,
        _system: mock.Mock,
        _machine: mock.Mock,
        _cli: mock.Mock,
        _version: mock.Mock,
        _alive: mock.Mock,
    ) -> None:
        adapter = OmlxRuntimeAdapter()
        probe = adapter.probe()
        self.assertEqual(probe.runtime, "omlx")
        self.assertTrue(probe.installed)
        self.assertTrue(probe.service_available)
        self.assertTrue(probe.locally_usable)
        self.assertTrue(probe_capability_available(adapter, probe))
        self.assertEqual(adapter.availability_policy, "service")
        self.assertTrue(adapter.llmrig_execution_supported)
        self.assertTrue(adapter.llmrig_benchmark_supported)
        self.assertEqual(probe.version, "omlx 1.2.3")
        self.assertIn("MLX", probe.supported_artifact_formats)
        self.assertEqual(probe.execution_api, "OpenAI-compatible /v1")
        self.assertFalse(probe.blockers)
        self.assertIn("oMLX model inventory API", [item.source for item in probe.evidence])
        for item in probe.evidence:
            validate_public_text(item.kind, "runtime evidence kind")
            validate_public_text(item.source, "runtime evidence source")
            validate_public_text(item.detail, "runtime evidence detail")

    @mock.patch.dict("os.environ", {"OMLX_API_KEY": "test-secret"}, clear=False)
    @mock.patch("_llmrig.runtime_adapters._json_endpoint_alive", return_value=True)
    @mock.patch("_llmrig.runtime_adapters._run_version", return_value="omlx 1.2.3")
    @mock.patch.object(OmlxRuntimeAdapter, "_cli_path", return_value="/usr/local/bin/omlx")
    @mock.patch("_llmrig.runtime_adapters.platform.machine", return_value="arm64")
    @mock.patch("_llmrig.runtime_adapters.platform.system", return_value="Darwin")
    def test_omlx_probe_uses_configured_api_key_without_exposing_it(
        self,
        _system: mock.Mock,
        _machine: mock.Mock,
        _cli: mock.Mock,
        _version: mock.Mock,
        _alive: mock.Mock,
    ) -> None:
        probe = OmlxRuntimeAdapter().probe()
        _alive.assert_called_once_with(
            "http://127.0.0.1:8000/v1/models",
            headers={"Authorization": "Bearer test-secret"},
        )
        self.assertTrue(probe.service_available)
        self.assertNotIn("test-secret", repr(probe))
        self.assertNotIn("test-secret", str(probe.to_dict()))

    @mock.patch("_llmrig.runtime_adapters._json_endpoint_alive", return_value=False)
    @mock.patch("_llmrig.runtime_adapters._run_version", return_value="omlx 1.2.3")
    @mock.patch.object(OmlxRuntimeAdapter, "_cli_path", return_value="/usr/local/bin/omlx")
    @mock.patch("_llmrig.runtime_adapters.platform.machine", return_value="x86_64")
    @mock.patch("_llmrig.runtime_adapters.platform.system", return_value="Linux")
    def test_omlx_probe_keeps_platform_blockers_explicit(
        self,
        _system: mock.Mock,
        _machine: mock.Mock,
        _cli: mock.Mock,
        _version: mock.Mock,
        _alive: mock.Mock,
    ) -> None:
        adapter = OmlxRuntimeAdapter()
        probe = adapter.probe()
        self.assertTrue(probe.installed)
        self.assertFalse(probe.locally_usable)
        self.assertFalse(probe_capability_available(adapter, probe))
        self.assertIn("oMLX requires macOS", probe.blockers)
        self.assertIn("oMLX requires Apple Silicon", probe.blockers)

    @mock.patch("_llmrig.runtime_adapters._json_endpoint_alive", return_value=False)
    @mock.patch("_llmrig.runtime_adapters._path_or_none", return_value=None)
    def test_ollama_absence_is_not_runtime_availability(
        self, _path: mock.Mock, _alive: mock.Mock
    ) -> None:
        adapter = OllamaRuntimeAdapter()
        probe = adapter.probe()
        self.assertFalse(probe.installed)
        self.assertFalse(probe.locally_usable)
        self.assertFalse(probe_capability_available(adapter, probe))
        self.assertEqual(adapter.availability_policy, "service")
        self.assertTrue(adapter.llmrig_execution_supported)
        self.assertTrue(adapter.llmrig_benchmark_supported)

    @mock.patch("_llmrig.runtime_adapters._run_version", return_value="version 1")
    @mock.patch(
        "_llmrig.runtime_adapters._path_or_none",
        side_effect=lambda name: "/usr/bin/llama-cli" if name == "llama-cli" else None,
    )
    def test_llama_cpp_probe_reports_gguf(
        self, _path: mock.Mock, _version: mock.Mock
    ) -> None:
        adapter = LlamaCppRuntimeAdapter()
        probe = adapter.probe()
        self.assertTrue(probe.installed)
        self.assertTrue(probe_capability_available(adapter, probe))
        self.assertEqual(adapter.availability_policy, "cli-version")
        self.assertTrue(adapter.llmrig_execution_supported)
        self.assertTrue(adapter.llmrig_benchmark_supported)
        self.assertEqual(probe.supported_artifact_formats, ("GGUF",))

    @mock.patch("_llmrig.runtime_adapters.platform.machine", return_value="arm64")
    @mock.patch("_llmrig.runtime_adapters.platform.system", return_value="Darwin")
    @mock.patch("_llmrig.runtime_adapters._path_or_none", return_value="/usr/bin/mlx_lm.generate")
    @mock.patch("_llmrig.runtime_adapters.importlib.metadata.version", return_value="0.32.0")
    @mock.patch("_llmrig.runtime_adapters.importlib.util.find_spec", return_value=object())
    def test_mlx_lm_probe_is_usable_on_apple_silicon(
        self,
        _find: mock.Mock,
        _metadata: mock.Mock,
        _path: mock.Mock,
        _system: mock.Mock,
        _machine: mock.Mock,
    ) -> None:
        adapter = MlxLmRuntimeAdapter()
        probe = adapter.probe()
        self.assertTrue(probe.installed)
        self.assertTrue(probe.locally_usable)
        self.assertTrue(probe_capability_available(adapter, probe))
        self.assertEqual(adapter.availability_policy, "cli-version")
        self.assertTrue(adapter.llmrig_execution_supported)
        self.assertTrue(adapter.llmrig_benchmark_supported)
        self.assertEqual(probe.version, "0.32.0")

    def test_probe_runtimes_preserves_adapter_order(self) -> None:
        class Adapter:
            def __init__(self, name: str) -> None:
                self.name = name

            def probe(self) -> RuntimeProbe:
                return RuntimeProbe(
                    runtime=self.name,
                    installed=False,
                    service_available=None,
                    version=None,
                    cli_path=None,
                    supported_artifact_formats=(),
                    supported_platforms=(),
                    supported_architectures=(),
                    execution_api=None,
                )

        probes = probe_runtimes((Adapter("a"), Adapter("b")))
        self.assertEqual([item.runtime for item in probes], ["a", "b"])

    def test_execution_adapters_follow_registry_order(self) -> None:
        class FakeExecutionAdapter:
            def __init__(self, runtime: str) -> None:
                self.runtime = runtime

        class FakeOllamaExecutionAdapter(FakeExecutionAdapter):
            def __init__(self, host: str) -> None:
                self.host = host
                super().__init__("ollama")

        class FakeLlamaCppExecutionAdapter(FakeExecutionAdapter):
            def __init__(self) -> None:
                super().__init__("llama.cpp")

        class FakeMlxExecutionAdapter(FakeExecutionAdapter):
            def __init__(self) -> None:
                super().__init__("mlx-lm")

        class Legacy:
            DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
            OllamaExecutionAdapter = FakeOllamaExecutionAdapter
            LlamaCppExecutionAdapter = FakeLlamaCppExecutionAdapter
            MlxExecutionAdapter = FakeMlxExecutionAdapter

        adapters = build_execution_adapters(Legacy)
        self.assertEqual(
            [item.runtime for item in adapters],
            ["omlx", "ollama", "mlx-lm", "llama.cpp"],
        )
        self.assertEqual(adapters[1].host, Legacy.DEFAULT_OLLAMA_HOST)


if __name__ == "__main__":
    unittest.main()
