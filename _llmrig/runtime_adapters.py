"""Runtime adapter foundation for LLMRig v0.8.

The current public CLI still owns the stable behavior in :mod:`llmrig`. This
module introduces a deliberately small, stdlib-only adapter boundary that can be
adopted incrementally without changing existing solve semantics.

A :class:`RuntimeProbe` contains observation only. Runtime-specific readiness,
LLMRig support facts, and verification construction live on the registered adapter.
This keeps the registry as the single capability source while preserving existing
probe fixtures and public runtime output.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class RuntimeEvidence:
    kind: str
    source: str
    detail: str


@dataclass(frozen=True)
class RuntimeProbe:
    """Read-only observation of one runtime on the current machine."""

    runtime: str
    installed: bool
    service_available: Optional[bool]
    version: Optional[str]
    cli_path: Optional[str]
    supported_artifact_formats: Tuple[str, ...]
    supported_platforms: Tuple[str, ...]
    supported_architectures: Tuple[str, ...]
    execution_api: Optional[str]
    evidence: Tuple[RuntimeEvidence, ...] = ()
    blockers: Tuple[str, ...] = ()
    unknowns: Tuple[str, ...] = ()

    @property
    def locally_usable(self) -> bool:
        if not self.installed or self.blockers:
            return False
        return self.service_available is not False

    def to_dict(self) -> dict:
        public_cli_path = (
            self.cli_path.replace(str(Path.home()), "~") if self.cli_path else None
        )
        return {
            "runtime": self.runtime,
            "installed": self.installed,
            "service_available": self.service_available,
            "version": self.version,
            "cli_path": public_cli_path,
            "supported_artifact_formats": list(self.supported_artifact_formats),
            "supported_platforms": list(self.supported_platforms),
            "supported_architectures": list(self.supported_architectures),
            "execution_api": self.execution_api,
            "locally_usable": self.locally_usable,
            "evidence": [item.__dict__ for item in self.evidence],
            "blockers": list(self.blockers),
            "unknowns": list(self.unknowns),
        }


class RuntimeAdapter(Protocol):
    """Capability, observation, and execution-routing contract for one runtime."""

    name: str
    availability_policy: str
    runtime_execution_capable: bool
    llmrig_installation_supported: bool
    llmrig_execution_supported: bool
    llmrig_benchmark_supported: bool

    def probe(self) -> RuntimeProbe: ...

    def build_execution_adapter(self, legacy: Any) -> Any: ...


def _run_version(command: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    if not lines:
        return None
    return lines[0][:200].replace(str(Path.home()), "~")


def _json_endpoint_alive(
    url: str,
    timeout: float = 1.5,
    headers: Optional[Mapping[str, str]] = None,
) -> bool:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "llmrig-runtime-probe",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(1024 * 1024)
            if response.status < 200 or response.status >= 300:
                return False
            if payload:
                json.loads(payload.decode("utf-8"))
            return True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _path_or_none(command: str) -> Optional[str]:
    value = shutil.which(command)
    return str(value) if value else None


def probe_capability_available(adapter: RuntimeAdapter, probe: RuntimeProbe) -> bool:
    """Apply the registered adapter's readiness policy without runtime-name checks."""
    if not probe.installed or probe.blockers:
        return False
    if adapter.availability_policy == "service":
        return probe.service_available is True
    if adapter.availability_policy == "cli-version":
        return probe.cli_path is not None and probe.version is not None
    if adapter.availability_policy == "local":
        return probe.locally_usable
    return False


class OmlxRuntimeAdapter:
    """Detect and route oMLX without starting its server."""

    name = "omlx"
    availability_policy = "service"
    runtime_execution_capable = True
    llmrig_installation_supported = False
    llmrig_execution_supported = True
    llmrig_benchmark_supported = True
    default_endpoint = os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

    @staticmethod
    def _cli_path() -> Optional[str]:
        path = _path_or_none("omlx")
        if path:
            return path
        # The oMLX macOS application installs this lightweight shim.
        shim = Path.home() / ".omlx" / "bin" / "omlx"
        return str(shim) if shim.is_file() and os.access(shim, os.X_OK) else None

    def build_execution_adapter(self, legacy: Any) -> Any:
        # Lazy import avoids a module cycle: omlx_verify imports this adapter for
        # inventory observation, while the execution class itself lives there.
        from .omlx_verify import OmlxExecutionAdapter

        return OmlxExecutionAdapter(legacy)

    def probe(self) -> RuntimeProbe:
        cli_path = self._cli_path()
        installed = cli_path is not None
        version = _run_version([cli_path, "--version"]) if cli_path else None
        system = platform.system()
        architecture = platform.machine().lower()
        platform_ok = system == "Darwin"
        architecture_ok = architecture in {"arm64", "aarch64"}
        api_key = os.environ.get("OMLX_API_KEY")
        auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        service_available = _json_endpoint_alive(
            f"{self.default_endpoint}/v1/models",
            headers=auth_headers,
        )

        blockers = []
        if installed and not platform_ok:
            blockers.append("oMLX requires macOS")
        if installed and not architecture_ok:
            blockers.append("oMLX requires Apple Silicon")

        evidence = [
            RuntimeEvidence(
                "deterministic-runtime-knowledge",
                "oMLX runtime contract",
                "oMLX serves MLX-compatible local models on Apple Silicon through an OpenAI-compatible API",
            ),
            RuntimeEvidence(
                "verified-local-runtime",
                "oMLX CLI detection",
                "a local oMLX command was detected"
                if installed
                else "no local oMLX command was detected",
            ),
        ]
        if service_available:
            evidence.append(
                RuntimeEvidence(
                    "verified-local-runtime",
                    "oMLX model inventory API",
                    "the local OpenAI-compatible oMLX endpoint responded with JSON",
                )
            )

        unknowns = []
        if installed and version is None:
            unknowns.append("runtime version is unknown")
        if installed and not service_available:
            unknowns.append("oMLX is installed but its default local service is not responding")

        return RuntimeProbe(
            runtime=self.name,
            installed=installed,
            service_available=service_available,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=("MLX",),
            supported_platforms=("Darwin",),
            supported_architectures=("arm64", "aarch64"),
            execution_api="OpenAI-compatible /v1",
            evidence=tuple(evidence),
            blockers=tuple(blockers),
            unknowns=tuple(unknowns),
        )


class OllamaRuntimeAdapter:
    name = "ollama"
    availability_policy = "service"
    runtime_execution_capable = True
    llmrig_installation_supported = False
    llmrig_execution_supported = True
    llmrig_benchmark_supported = True
    default_endpoint = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    def build_execution_adapter(self, legacy: Any) -> Any:
        return legacy.OllamaExecutionAdapter(legacy.DEFAULT_OLLAMA_HOST)

    def probe(self) -> RuntimeProbe:
        cli_path = _path_or_none("ollama")
        installed = cli_path is not None
        version = _run_version([cli_path, "--version"]) if cli_path else None
        available = _json_endpoint_alive(f"{self.default_endpoint}/api/version")
        evidence = (
            RuntimeEvidence(
                "verified-local-runtime",
                "Ollama CLI detection",
                "a local Ollama command was detected"
                if installed
                else "no local Ollama command was detected",
            ),
        )
        unknowns = () if version else (("runtime version is unknown",) if installed else ())
        return RuntimeProbe(
            runtime=self.name,
            installed=installed,
            service_available=available,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=("Ollama",),
            supported_platforms=("Darwin", "Linux", "Windows"),
            supported_architectures=(),
            execution_api="Ollama HTTP API",
            evidence=evidence,
            unknowns=unknowns,
        )


class LlamaCppRuntimeAdapter:
    name = "llama.cpp"
    availability_policy = "cli-version"
    runtime_execution_capable = True
    llmrig_installation_supported = False
    llmrig_execution_supported = True
    llmrig_benchmark_supported = True

    def build_execution_adapter(self, legacy: Any) -> Any:
        return legacy.LlamaCppExecutionAdapter()

    def probe(self) -> RuntimeProbe:
        cli_path = next(
            (path for name in ("llama-cli", "llama.cpp") if (path := _path_or_none(name))),
            None,
        )
        version = _run_version([cli_path, "--version"]) if cli_path else None
        installed = cli_path is not None
        return RuntimeProbe(
            runtime=self.name,
            installed=installed,
            service_available=None,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=("GGUF",),
            supported_platforms=("Darwin", "Linux", "Windows"),
            supported_architectures=(),
            execution_api="local CLI",
            evidence=(
                RuntimeEvidence(
                    "deterministic-runtime-knowledge",
                    "llama.cpp runtime contract",
                    "llama.cpp consumes GGUF artifacts",
                ),
            ),
            unknowns=("runtime version is unknown",) if installed and version is None else (),
        )


class MlxLmRuntimeAdapter:
    name = "mlx-lm"
    availability_policy = "cli-version"
    runtime_execution_capable = True
    llmrig_installation_supported = False
    llmrig_execution_supported = True
    llmrig_benchmark_supported = True

    def build_execution_adapter(self, legacy: Any) -> Any:
        return legacy.MlxExecutionAdapter()

    def probe(self) -> RuntimeProbe:
        try:
            package_installed = importlib.util.find_spec("mlx_lm") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            package_installed = False
        cli_path = _path_or_none("mlx_lm.generate")
        installed = package_installed or cli_path is not None
        version = None
        if package_installed:
            try:
                version = importlib.metadata.version("mlx-lm")
            except importlib.metadata.PackageNotFoundError:
                pass
        system = platform.system()
        architecture = platform.machine().lower()
        blockers = []
        if installed and system != "Darwin":
            blockers.append("MLX-LM requires macOS")
        if installed and architecture not in {"arm64", "aarch64"}:
            blockers.append("MLX-LM requires Apple Silicon")
        if package_installed and cli_path is None:
            blockers.append("the MLX-LM generation command was not detected")
        return RuntimeProbe(
            runtime=self.name,
            installed=installed,
            service_available=None,
            version=version,
            cli_path=cli_path,
            supported_artifact_formats=("MLX",),
            supported_platforms=("Darwin",),
            supported_architectures=("arm64", "aarch64"),
            execution_api="local CLI",
            evidence=(
                RuntimeEvidence(
                    "deterministic-runtime-knowledge",
                    "MLX-LM runtime contract",
                    "MLX-LM executes models packaged for the MLX ecosystem",
                ),
            ),
            blockers=tuple(blockers),
            unknowns=("runtime version is unknown",) if installed and version is None else (),
        )


DEFAULT_RUNTIME_ADAPTERS: Tuple[RuntimeAdapter, ...] = (
    OmlxRuntimeAdapter(),
    OllamaRuntimeAdapter(),
    MlxLmRuntimeAdapter(),
    LlamaCppRuntimeAdapter(),
)


def runtime_adapter_for(
    runtime: str,
    adapters: Sequence[RuntimeAdapter] = DEFAULT_RUNTIME_ADAPTERS,
) -> Optional[RuntimeAdapter]:
    """Return the unique registered adapter for a runtime, otherwise fail closed."""
    matches = tuple(adapter for adapter in adapters if adapter.name == runtime)
    return matches[0] if len(matches) == 1 else None


def probe_runtimes(
    adapters: Sequence[RuntimeAdapter] = DEFAULT_RUNTIME_ADAPTERS,
) -> Tuple[RuntimeProbe, ...]:
    """Probe runtimes in stable order without mutating the machine."""

    return tuple(adapter.probe() for adapter in adapters)


def build_execution_adapters(
    legacy: Any,
    adapters: Sequence[RuntimeAdapter] = DEFAULT_RUNTIME_ADAPTERS,
) -> Tuple[Any, ...]:
    """Build supported verification adapters in the same stable registry order."""

    return tuple(
        adapter.build_execution_adapter(legacy)
        for adapter in adapters
        if adapter.llmrig_execution_supported and adapter.llmrig_benchmark_supported
    )
