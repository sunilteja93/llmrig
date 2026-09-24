"""Bridge v0.8 runtime probes into the established solve capability contract.

This module keeps runtime observation in one adapter registry while preserving the
legacy :class:`llmrig.RuntimeCapability` schema consumed by compatibility and solve.
It never installs, starts, or downloads a runtime or model. Execution support flags
only describe code paths that LLMRig can use when the user explicitly requests a
measured verification.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from .runtime_adapters import RuntimeProbe, probe_runtimes


_LLMRIG_EXECUTION_SUPPORTED = {
    "ollama": True,
    "llama.cpp": True,
    "mlx-lm": True,
    "omlx": True,
}

_LLMRIG_BENCHMARK_SUPPORTED = {
    "ollama": True,
    "llama.cpp": True,
    "mlx-lm": True,
    "omlx": True,
}


def _probe_available(probe: RuntimeProbe) -> bool:
    """Translate observational probe state without overstating readiness."""
    if not probe.installed or probe.blockers:
        return False

    if probe.runtime in {"omlx", "ollama"}:
        # Service-backed runtimes are available only when their local endpoint
        # actually responded. Installation alone is not enough.
        return probe.service_available is True

    if probe.runtime in {"mlx-lm", "llama.cpp"}:
        # Preserve the established native-runtime health boundary: a detected
        # command without version evidence is not treated as available yet.
        return probe.cli_path is not None and probe.version is not None

    return probe.locally_usable


def capabilities_from_probes(
    legacy: Any,
    profile: Any,
    probes: Optional[Sequence[RuntimeProbe]] = None,
) -> Tuple[Any, ...]:
    """Create established RuntimeCapability values from v0.8 adapter probes.

    ``profile`` remains part of the bridge signature because it is part of the
    stable capability-provider contract. Platform/architecture facts are already
    represented explicitly by each probe and are not re-inferred here.
    """
    del profile
    observed = tuple(probe_runtimes() if probes is None else probes)
    capabilities = []

    for probe in observed:
        evidence = tuple(
            legacy.RecommendationEvidence(item.kind, item.source, item.detail)
            for item in probe.evidence
        )
        unknowns = list(probe.unknowns)
        if not probe.supported_architectures:
            unknowns.append("supported architectures are unknown")

        capabilities.append(
            legacy.RuntimeCapability(
                runtime=probe.runtime,
                installed=probe.installed,
                available=_probe_available(probe),
                version=probe.version,
                supported_artifact_formats=probe.supported_artifact_formats,
                supported_platforms=probe.supported_platforms,
                supported_architectures=probe.supported_architectures,
                runtime_execution_capable=True,
                llmrig_installation_supported=False,
                llmrig_execution_supported=_LLMRIG_EXECUTION_SUPPORTED.get(
                    probe.runtime, False
                ),
                llmrig_benchmark_supported=_LLMRIG_BENCHMARK_SUPPORTED.get(
                    probe.runtime, False
                ),
                confidence=legacy.Confidence.HIGH,
                evidence=evidence,
                unknowns=tuple(dict.fromkeys(unknowns)),
            )
        )

    return tuple(capabilities)


@contextmanager
def adapter_capabilities_for_legacy(legacy: Any) -> Iterator[None]:
    """Temporarily route legacy solve capability reads through v0.8 adapters."""
    original = legacy.runtime_capabilities

    def bridged(profile: Any) -> Tuple[Any, ...]:
        return capabilities_from_probes(legacy, profile)

    legacy.runtime_capabilities = bridged
    try:
        yield
    finally:
        legacy.runtime_capabilities = original
