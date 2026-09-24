from __future__ import annotations

import unittest
from unittest import mock

import llmrig
from _llmrig import cli, runtime_bridge
from _llmrig.runtime_adapters import RuntimeEvidence, RuntimeProbe


class RuntimeBridgeTests(unittest.TestCase):
    def probe(
        self,
        runtime,
        *,
        installed,
        service_available,
        version=None,
        cli_path=None,
        formats=(),
        platforms=("Darwin",),
        architectures=("arm64",),
        api=None,
        availability_policy="local",
        runtime_execution_capable=True,
        installation_supported=False,
        execution_supported=True,
        benchmark_supported=True,
        blockers=(),
        unknowns=(),
    ):
        return RuntimeProbe(
            runtime=runtime,
            installed=installed,
            service_available=service_available,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=formats,
            supported_platforms=platforms,
            supported_architectures=architectures,
            execution_api=api,
            availability_policy=availability_policy,
            runtime_execution_capable=runtime_execution_capable,
            llmrig_installation_supported=installation_supported,
            llmrig_execution_supported=execution_supported,
            llmrig_benchmark_supported=benchmark_supported,
            evidence=(RuntimeEvidence("verified", "test probe", "observed runtime state"),),
            blockers=blockers,
            unknowns=unknowns,
        )

    def test_omlx_service_is_runtime_available_and_llmrig_measurable(self):
        probe = self.probe(
            "omlx",
            installed=True,
            service_available=True,
            version="omlx 1.2.3",
            cli_path="~/.omlx/bin/omlx",
            formats=("MLX",),
            api="OpenAI-compatible /v1",
            availability_policy="service",
        )
        capability = runtime_bridge.capabilities_from_probes(
            llmrig, {"os": "Darwin", "arch": "arm64"}, (probe,)
        )[0]

        self.assertTrue(capability.installed)
        self.assertTrue(capability.available)
        self.assertTrue(capability.runtime_execution_capable)
        self.assertTrue(capability.llmrig_execution_supported)
        self.assertTrue(capability.llmrig_benchmark_supported)
        self.assertEqual(capability.supported_artifact_formats, ("MLX",))

    def test_installed_omlx_without_service_is_not_available(self):
        probe = self.probe(
            "omlx",
            installed=True,
            service_available=False,
            version="omlx 1.2.3",
            cli_path="~/.omlx/bin/omlx",
            formats=("MLX",),
            api="OpenAI-compatible /v1",
            availability_policy="service",
            unknowns=("oMLX is installed but its default local service is not responding",),
        )
        capability = runtime_bridge.capabilities_from_probes(llmrig, {}, (probe,))[0]
        self.assertTrue(capability.installed)
        self.assertFalse(capability.available)
        self.assertTrue(capability.llmrig_execution_supported)
        self.assertTrue(capability.llmrig_benchmark_supported)
        self.assertIn("service is not responding", " ".join(capability.unknowns))

    def test_native_cli_requires_version_evidence_for_availability(self):
        probe = self.probe(
            "llama.cpp",
            installed=True,
            service_available=None,
            version=None,
            cli_path="/usr/local/bin/llama-cli",
            formats=("GGUF",),
            platforms=("Darwin", "Linux", "Windows"),
            architectures=(),
            api="local CLI",
            availability_policy="cli-version",
            unknowns=("runtime version is unknown",),
        )
        capability = runtime_bridge.capabilities_from_probes(llmrig, {}, (probe,))[0]
        self.assertFalse(capability.available)
        self.assertIn("supported architectures are unknown", capability.unknowns)

    def test_bridge_does_not_infer_capability_policy_from_runtime_name(self):
        probe = self.probe(
            "future-runtime",
            installed=True,
            service_available=False,
            version="9.9",
            cli_path="/usr/local/bin/future-runtime",
            availability_policy="service",
            runtime_execution_capable=False,
            installation_supported=True,
            execution_supported=False,
            benchmark_supported=False,
        )
        capability = runtime_bridge.capabilities_from_probes(llmrig, {}, (probe,))[0]
        self.assertFalse(capability.available)
        self.assertFalse(capability.runtime_execution_capable)
        self.assertTrue(capability.llmrig_installation_supported)
        self.assertFalse(capability.llmrig_execution_supported)
        self.assertFalse(capability.llmrig_benchmark_supported)

    def test_solve_cli_temporarily_uses_adapter_capabilities(self):
        omlx = self.probe(
            "omlx",
            installed=False,
            service_available=False,
            formats=("MLX",),
            api="OpenAI-compatible /v1",
            availability_policy="service",
        )
        ollama = self.probe(
            "ollama",
            installed=True,
            service_available=True,
            version="ollama version is 0.33.3",
            cli_path="/usr/local/bin/ollama",
            formats=("Ollama",),
            platforms=("Darwin", "Linux", "Windows"),
            architectures=(),
            api="Ollama HTTP API",
            availability_policy="service",
        )
        original = llmrig.runtime_capabilities
        seen = []

        def fake_main(argv):
            capabilities = llmrig.runtime_capabilities(
                {"os": "Darwin", "arch": "arm64"}
            )
            seen.extend(item.runtime for item in capabilities)
            return 23

        with mock.patch.object(
            runtime_bridge, "probe_runtimes", return_value=(omlx, ollama)
        ), mock.patch.object(llmrig, "main", side_effect=fake_main):
            self.assertEqual(cli.main(["solve", "org/model"]), 23)

        self.assertEqual(seen, ["omlx", "ollama"])
        self.assertIs(llmrig.runtime_capabilities, original)

    def test_non_solve_cli_does_not_replace_legacy_capabilities(self):
        original = llmrig.runtime_capabilities

        def fake_main(argv):
            self.assertIs(llmrig.runtime_capabilities, original)
            return 17

        with mock.patch.object(llmrig, "main", side_effect=fake_main):
            self.assertEqual(cli.main(["check"]), 17)


if __name__ == "__main__":
    unittest.main()
