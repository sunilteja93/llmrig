"""Explicit runtime mutations used only by approved Autopilot apply flows."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, Tuple

from .runtime_adapters import OmlxRuntimeAdapter, runtime_adapter_for


@dataclass(frozen=True)
class RuntimeActionOutcome:
    status: str
    detail: str
    blockers: Tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.status in {"completed", "already_satisfied"}


class RuntimeActionAdapter(Protocol):
    name: str
    start_supported: bool
    native_acquisition_supported: bool

    def start(self, legacy: Any) -> RuntimeActionOutcome: ...

    def acquire_native(self, legacy: Any, artifact_id: str) -> RuntimeActionOutcome: ...

    def refresh_after_acquisition(self, legacy: Any) -> RuntimeActionOutcome: ...


class _BaseActions:
    name = ""
    start_supported = False
    native_acquisition_supported = False

    def start(self, legacy: Any) -> RuntimeActionOutcome:
        return RuntimeActionOutcome(
            "blocked",
            f"LLMRig does not automatically start {self.name}.",
            ("runtime start is not supported by this action adapter",),
        )

    def acquire_native(self, legacy: Any, artifact_id: str) -> RuntimeActionOutcome:
        return RuntimeActionOutcome(
            "blocked",
            f"LLMRig has no runtime-native acquisition path for {self.name}.",
            ("runtime-native artifact acquisition is not supported",),
        )

    def refresh_after_acquisition(self, legacy: Any) -> RuntimeActionOutcome:
        return RuntimeActionOutcome(
            "already_satisfied",
            f"{self.name} does not require a persistent refresh after acquisition.",
        )


class OmlxRuntimeActions(_BaseActions):
    name = "omlx"
    start_supported = True

    def _command(self) -> Optional[str]:
        return OmlxRuntimeAdapter._cli_path()

    def start(self, legacy: Any) -> RuntimeActionOutcome:
        probe = OmlxRuntimeAdapter().probe()
        if not probe.installed or not probe.cli_path:
            return RuntimeActionOutcome(
                "blocked",
                "oMLX is not installed.",
                ("LLMRig does not install oMLX automatically",),
            )
        if probe.service_available:
            return RuntimeActionOutcome("already_satisfied", "oMLX is already running.")
        try:
            completed = subprocess.run(
                [probe.cli_path, "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return RuntimeActionOutcome("failed", "oMLX failed to start.")
        if completed.returncode != 0:
            return RuntimeActionOutcome("failed", "oMLX start command returned an error.")
        for _ in range(40):
            if OmlxRuntimeAdapter().probe().service_available:
                return RuntimeActionOutcome("completed", "oMLX started successfully.")
            time.sleep(0.25)
        return RuntimeActionOutcome("failed", "oMLX did not become ready after start.")

    def refresh_after_acquisition(self, legacy: Any) -> RuntimeActionOutcome:
        probe = OmlxRuntimeAdapter().probe()
        if not probe.installed or not probe.cli_path:
            return RuntimeActionOutcome(
                "blocked",
                "oMLX is not installed.",
                ("LLMRig does not install oMLX automatically",),
            )
        command = "restart" if probe.service_available else "start"
        try:
            completed = subprocess.run(
                [probe.cli_path, command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return RuntimeActionOutcome("failed", "oMLX model refresh failed.")
        if completed.returncode != 0:
            return RuntimeActionOutcome("failed", "oMLX model refresh command returned an error.")
        for _ in range(40):
            if OmlxRuntimeAdapter().probe().service_available:
                return RuntimeActionOutcome(
                    "completed",
                    "oMLX refreshed its model inventory after acquisition.",
                )
            time.sleep(0.25)
        return RuntimeActionOutcome("failed", "oMLX did not become ready after refresh.")


class OllamaRuntimeActions(_BaseActions):
    name = "ollama"
    start_supported = True
    native_acquisition_supported = True

    def start(self, legacy: Any) -> RuntimeActionOutcome:
        try:
            ready = bool(legacy.OLLAMA_RUNTIME.ensure_available())
        except Exception:
            ready = False
        return RuntimeActionOutcome(
            "completed" if ready else "failed",
            "Ollama is ready." if ready else "Ollama could not be started.",
        )

    def acquire_native(self, legacy: Any, artifact_id: str) -> RuntimeActionOutcome:
        command = getattr(legacy, "command_exists", None)
        if callable(command) and not command("ollama"):
            return RuntimeActionOutcome(
                "blocked",
                "Ollama is not installed.",
                ("LLMRig does not install Ollama automatically",),
            )
        readiness = self.start(legacy)
        if not readiness.success:
            return RuntimeActionOutcome(
                readiness.status,
                "Ollama must be ready before its runtime-native artifact can be acquired.",
                readiness.blockers,
            )
        try:
            completed = subprocess.run(
                ["ollama", "pull", artifact_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return RuntimeActionOutcome("failed", "Ollama artifact acquisition failed.")
        if completed.returncode != 0:
            return RuntimeActionOutcome(
                "failed", "Ollama pull returned an error for the selected artifact."
            )
        return RuntimeActionOutcome(
            "completed",
            "Ollama acquired the selected runtime-native artifact after confirming service readiness.",
        )


class MlxLmRuntimeActions(_BaseActions):
    name = "mlx-lm"


class LlamaCppRuntimeActions(_BaseActions):
    name = "llama.cpp"


DEFAULT_RUNTIME_ACTIONS: Tuple[RuntimeActionAdapter, ...] = (
    OmlxRuntimeActions(),
    OllamaRuntimeActions(),
    MlxLmRuntimeActions(),
    LlamaCppRuntimeActions(),
)


def runtime_actions_for(
    runtime: str,
    adapters: Sequence[RuntimeActionAdapter] = DEFAULT_RUNTIME_ACTIONS,
) -> Optional[RuntimeActionAdapter]:
    # Mutation is allowed only for runtimes that also have exactly one registered
    # observation/execution adapter. This prevents action-only runtime aliases.
    if runtime_adapter_for(runtime) is None:
        return None
    matches = tuple(adapter for adapter in adapters if adapter.name == runtime)
    return matches[0] if len(matches) == 1 else None
