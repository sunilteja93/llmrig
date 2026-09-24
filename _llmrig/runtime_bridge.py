"""Bridge v0.8 runtime probes into the established solve capability contract.

This module preserves the legacy :class:`llmrig.RuntimeCapability` schema consumed
by compatibility and solve while treating adapter-produced probes as the single
source of runtime readiness and LLMRig support facts. It never installs, starts,
or downloads a runtime or model.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from .runtime_adapters import RuntimeProbe, probe_runtimes


def capabilities_from_probes(
    legacy: Any,
    profile: Any,
    probes: Optional[Sequence[RuntimeProbe]] = None,
) -> Tuple[Any, ...]:
    """Create established RuntimeCapability values from v0.8 adapter probes.

    ``profile`` remains part of the bridge signature because it is part of the
    stable capability-provider contract. Platform/architecture and readiness facts
    are already represented explicitly by each probe and are not re-inferred here.
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
                available=probe.capability_available,
                version=probe.version,
                supported_artifact_formats=probe.supported_artifact_formats,
                supported_platforms=probe.supported_platforms,
                supported_architectures=probe.supported_architectures,
                runtime_execution_capable=probe.runtime_execution_capable,
                llmrig_installation_supported=probe.llmrig_installation_supported,
                llmrig_execution_supported=probe.llmrig_execution_supported,
                llmrig_benchmark_supported=probe.llmrig_benchmark_supported,
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
