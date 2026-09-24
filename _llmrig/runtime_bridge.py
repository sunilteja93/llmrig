"""Bridge v0.8 runtime probes into the established solve capability contract.

This module preserves the legacy :class:`llmrig.RuntimeCapability` schema consumed
by compatibility and solve while treating the runtime adapter registry as the
single source of readiness and LLMRig support facts. It never installs, starts,
or downloads a runtime or model.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from .runtime_adapters import (
    DEFAULT_RUNTIME_ADAPTERS,
    RuntimeAdapter,
    RuntimeProbe,
    probe_capability_available,
    probe_runtimes,
    runtime_adapter_for,
)


def capabilities_from_probes(
    legacy: Any,
    profile: Any,
    probes: Optional[Sequence[RuntimeProbe]] = None,
    adapters: Sequence[RuntimeAdapter] = DEFAULT_RUNTIME_ADAPTERS,
) -> Tuple[Any, ...]:
    """Create established RuntimeCapability values from adapter observations.

    ``profile`` remains part of the bridge signature because it is part of the
    stable capability-provider contract. Platform/architecture facts stay on the
    probe; readiness and LLMRig support facts come from the matching registry
    adapter. Unknown/unregistered runtimes fail closed for LLMRig execution support.
    """
    del profile
    observed = tuple(probe_runtimes(adapters) if probes is None else probes)
    capabilities = []

    for probe in observed:
        adapter = runtime_adapter_for(probe.runtime, adapters)
        evidence = tuple(
            legacy.RecommendationEvidence(item.kind, item.source, item.detail)
            for item in probe.evidence
        )
        unknowns = list(probe.unknowns)
        if not probe.supported_architectures:
            unknowns.append("supported architectures are unknown")

        if adapter is None:
            available = probe.locally_usable
            runtime_execution_capable = True
            installation_supported = False
            execution_supported = False
            benchmark_supported = False
        else:
            available = probe_capability_available(adapter, probe)
            runtime_execution_capable = adapter.runtime_execution_capable
            installation_supported = adapter.llmrig_installation_supported
            execution_supported = adapter.llmrig_execution_supported
            benchmark_supported = adapter.llmrig_benchmark_supported

        capabilities.append(
            legacy.RuntimeCapability(
                runtime=probe.runtime,
                installed=probe.installed,
                available=available,
                version=probe.version,
                supported_artifact_formats=probe.supported_artifact_formats,
                supported_platforms=probe.supported_platforms,
                supported_architectures=probe.supported_architectures,
                runtime_execution_capable=runtime_execution_capable,
                llmrig_installation_supported=installation_supported,
                llmrig_execution_supported=execution_supported,
                llmrig_benchmark_supported=benchmark_supported,
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
