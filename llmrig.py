#!/usr/bin/env python3
"""LLMRig: cross-platform local LLM readiness, discovery, setup, and benchmarking."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import importlib.metadata
import importlib.util
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import webbrowser
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, TypeVar

PROJECT_NAME = "LLMRig"
PROJECT_SLUG = "llmrig"
VERSION = "0.5.1"
CURATED_SNAPSHOT_DATE = "2026-08-19"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
HF_MODELS_API = "https://huggingface.co/api/models"
CACHE_TTL_SECONDS = 6 * 60 * 60
RACE_METHOD_VERSION = "race-v2"
RACE_CONTEXT = 4096
RACE_NUM_PREDICT = 128
RACE_NOISE_THRESHOLD = 0.05
PASSPORT_SCHEMA_VERSION = "1.0"
BENCHMARK_METHOD_VERSION = "ollama-bench-v1"
BENCH_REQUEST_TIMEOUT = 600

OFFICIAL = "official"
REDUCED_REFUSAL = "community-reduced-refusal"
REDUCED_TERMS = ("abliterat", "uncensor", "unrestrict", "heretic", "refusal")
LIVE_LLM_TASKS = {"text-generation", "image-text-to-text"}
LIVE_NON_LLM_HINTS = (
    "-bench",
    "bench-",
    "sae-",
    "forcedaligner",
    "forced-aligner",
    "asr-",
    "-asr",
    "tts-",
    "-tts",
    "embedding",
    "reranker",
)


class Confidence(str, Enum):
    """Categorical confidence for compatibility and recommendation facts."""

    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecommendationEvidence:
    """One attributable fact used to make a recommendation."""

    kind: str
    source: str
    detail: str

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("evidence kind must not be empty")
        if not self.source.strip():
            raise ValueError("evidence source must not be empty")
        if not self.detail.strip():
            raise ValueError("evidence detail must not be empty")

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "source": self.source, "detail": self.detail}


@dataclass(frozen=True)
class Model:
    """A logical model, independent of any runnable packaging."""

    model_id: str
    name: str
    params_b: Optional[float]
    modalities: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.name.strip():
            raise ValueError("model id and name must not be empty")
        if self.params_b is not None and self.params_b <= 0:
            raise ValueError("model parameter count must be positive when known")
        if not self.modalities or any(not item.strip() for item in self.modalities):
            raise ValueError("model modalities must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "params_b": self.params_b,
            "modalities": list(self.modalities),
        }


@dataclass(frozen=True)
class ModelArtifact:
    """A concrete model package intended for a particular runtime."""

    artifact_id: str
    model_id: str
    runtime: Optional[str]
    format: str
    size_gb: Optional[float]
    context_max: Optional[int]
    platforms: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()
    quantization: Optional[str] = None
    size_bytes: Optional[int] = None
    evidence: Tuple[RecommendationEvidence, ...] = ()
    unknowns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.model_id.strip():
            raise ValueError("artifact id and model id must not be empty")
        if not self.format.strip():
            raise ValueError("artifact format must not be empty")
        if self.runtime is not None and not self.runtime.strip():
            raise ValueError("artifact runtime must not be empty when known")
        if self.size_gb is not None and self.size_gb <= 0:
            raise ValueError("artifact size must be positive when known")
        if self.context_max is not None and self.context_max <= 0:
            raise ValueError("artifact context must be positive when known")
        if self.size_bytes is not None and self.size_bytes <= 0:
            raise ValueError("artifact byte size must be positive when known")
        if self.artifact_id in self.aliases or len(set(self.aliases)) != len(self.aliases):
            raise ValueError("artifact aliases must be unique and exclude the primary id")

    @property
    def ids(self) -> Tuple[str, ...]:
        return (self.artifact_id,) + self.aliases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "model_id": self.model_id,
            "format": self.format,
            "quantization": self.quantization,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "runtime": self.runtime,
            "context_max": self.context_max,
            "platforms": list(self.platforms),
            "evidence": [item.to_dict() for item in self.evidence],
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class RuntimeCapability:
    """Read-only description of a locally detectable runtime's capabilities."""

    runtime: str
    installed: bool
    available: bool
    version: Optional[str]
    supported_artifact_formats: Tuple[str, ...]
    supported_platforms: Tuple[str, ...]
    supported_architectures: Tuple[str, ...]
    runtime_execution_capable: bool
    llmrig_installation_supported: bool
    llmrig_execution_supported: bool
    llmrig_benchmark_supported: bool
    confidence: Confidence
    evidence: Tuple[RecommendationEvidence, ...] = ()
    unknowns: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime,
            "installed": self.installed,
            "available": self.available,
            "version": self.version,
            "supported_artifact_formats": list(self.supported_artifact_formats),
            "supported_platforms": list(self.supported_platforms),
            "supported_architectures": list(self.supported_architectures),
            "runtime_execution_capable": self.runtime_execution_capable,
            "llmrig_installation_supported": self.llmrig_installation_supported,
            "llmrig_execution_supported": self.llmrig_execution_supported,
            "llmrig_benchmark_supported": self.llmrig_benchmark_supported,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class RuntimeCandidate:
    """An evidenced runtime/artifact match, not a promise of successful inference."""

    runtime: str
    artifact_id: str
    artifact_format: str
    support_status: str
    fit_result: str
    confidence: Confidence
    evidence: Tuple[RecommendationEvidence, ...]
    blockers: Tuple[str, ...] = ()
    unknowns: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runtime": self.runtime,
            "artifact_id": self.artifact_id,
            "artifact_format": self.artifact_format,
            "support_status": self.support_status,
            "fit_result": self.fit_result,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "blockers": list(self.blockers),
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class RaceWorkload:
    """Small deterministic workload shared by every eligible competitor."""

    prompt: str
    context: int = RACE_CONTEXT
    num_predict: int = RACE_NUM_PREDICT
    runs: int = 2
    warmup_runs: int = 1
    warmup_num_predict: int = 32
    request_timeout_s: int = 120
    temperature: int = 0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "context": self.context,
            "num_predict": self.num_predict,
            "runs": self.runs,
            "warmup_runs": self.warmup_runs,
            "warmup_num_predict": self.warmup_num_predict,
            "request_timeout_s": self.request_timeout_s,
            "temperature": self.temperature,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RaceConfiguration:
    """One local or theoretical runtime/artifact configuration considered for a race."""

    logical_model_id: str
    runtime: str
    artifact_id: str
    artifact_format: str
    quantization: Optional[str]
    runtime_version: Optional[str]
    eligible: bool
    artifact_fingerprint: Optional[str] = None
    blockers: Tuple[str, ...] = ()
    evidence: Tuple[RecommendationEvidence, ...] = ()

    @property
    def identity(self) -> Tuple[str, str, str, str]:
        return (
            self.logical_model_id,
            self.runtime,
            self.artifact_fingerprint or self.artifact_id,
            self.quantization or "unknown",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_model_id": self.logical_model_id,
            "runtime": self.runtime,
            "artifact_id": self.artifact_id,
            "artifact_fingerprint": self.artifact_fingerprint,
            "artifact_format": self.artifact_format,
            "quantization": self.quantization,
            "runtime_version": self.runtime_version,
            "eligible": self.eligible,
            "blockers": list(self.blockers),
            "evidence": [item.to_dict() for item in self.evidence],
        }


class ExecutionTarget:
    """Private, deliberately non-serializable runtime invocation state."""

    __slots__ = ("configuration", "_locator")

    def __init__(self, configuration: RaceConfiguration, locator: str) -> None:
        self.configuration = configuration
        self._locator = locator

    def __repr__(self) -> str:
        return f"ExecutionTarget(configuration={self.configuration!r})"

    @staticmethod
    def _serialization_error() -> TypeError:
        return TypeError("private execution targets cannot be serialized or copied")

    def __reduce__(self) -> Any:
        raise self._serialization_error()

    def __reduce_ex__(self, protocol: int) -> Any:
        raise self._serialization_error()

    def __getstate__(self) -> Any:
        raise self._serialization_error()


@dataclass(frozen=True)
class RaceCompetitor:
    """Measured execution result for one eligible configuration."""

    logical_model_id: str
    runtime: str
    artifact_id: str
    artifact_fingerprint: Optional[str]
    artifact_format: str
    quantization: Optional[str]
    runtime_version: Optional[str]
    execution_status: str
    generation_tps: Optional[float]
    prompt_eval_tps: Optional[float]
    total_latency_s: Optional[float]
    generated_tokens: Optional[int]
    measured_runs: int
    generation_samples: int
    prompt_eval_samples: int
    latency_samples: int
    timestamp: str
    evidence: Tuple[RecommendationEvidence, ...] = ()
    warnings: Tuple[str, ...] = ()
    failure: Optional[str] = None
    raw_samples: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_model_id": self.logical_model_id,
            "runtime": self.runtime,
            "artifact_id": self.artifact_id,
            "artifact_fingerprint": self.artifact_fingerprint,
            "artifact_format": self.artifact_format,
            "quantization": self.quantization,
            "runtime_version": self.runtime_version,
            "execution_status": self.execution_status,
            "generation_tps": self.generation_tps,
            "prompt_eval_tps": self.prompt_eval_tps,
            "total_latency_s": self.total_latency_s,
            "generated_tokens": self.generated_tokens,
            "measured_runs": self.measured_runs,
            "generation_samples": self.generation_samples,
            "prompt_eval_samples": self.prompt_eval_samples,
            "latency_samples": self.latency_samples,
            "timestamp": self.timestamp,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
            "failure": self.failure,
            "raw_samples": [dict(item) for item in self.raw_samples],
        }


@dataclass(frozen=True)
class RaceResult:
    """Structured measured comparison; winners are metric-specific only."""

    status: str
    logical_model_id: str
    reason: Optional[str]
    method_version: str
    timestamp: str
    workload: RaceWorkload
    hardware: Dict[str, Any]
    eligible_configurations: Tuple[RaceConfiguration, ...]
    ineligible_configurations: Tuple[RaceConfiguration, ...]
    competitors: Tuple[RaceCompetitor, ...] = ()
    warnings: Tuple[str, ...] = ()
    winners: Tuple[Tuple[str, Dict[str, Any]], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "logical_model_id": self.logical_model_id,
            "reason": self.reason,
            "method_version": self.method_version,
            "timestamp": self.timestamp,
            "workload": self.workload.to_dict(),
            "hardware": dict(self.hardware),
            "eligible_configurations": [
                item.to_dict() for item in self.eligible_configurations
            ],
            "ineligible_configurations": [
                item.to_dict() for item in self.ineligible_configurations
            ],
            "competitors": [item.to_dict() for item in self.competitors],
            "warnings": list(self.warnings),
            "winners": {metric: value for metric, value in self.winners},
        }


@dataclass(frozen=True)
class BenchmarkPassport:
    """Portable record of one measured execution, not an attestation."""

    schema_version: str
    passport_id: str
    configuration_fingerprint: str
    identity: Dict[str, Any]
    model: Dict[str, Any]
    runtime: Dict[str, Any]
    hardware: Dict[str, Any]
    workload: Dict[str, Any]
    measurement: Dict[str, Any]
    evidence: Dict[str, Any]
    calibration: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passport_id": self.passport_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "identity": dict(self.identity),
            "model": dict(self.model),
            "runtime": dict(self.runtime),
            "hardware": dict(self.hardware),
            "workload": dict(self.workload),
            "measurement": dict(self.measurement),
            "evidence": dict(self.evidence),
            "calibration": dict(self.calibration),
        }


class CompatibilityStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    POSSIBLE = "possible/spill"
    NOT_NATIVE = "not native"
    TOO_LARGE = "too large"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompatibilityResult:
    """Evidence-aware compatibility assessment; unknown remains explicit."""

    model_id: str
    artifact_id: Optional[str]
    runtime: Optional[str]
    status: CompatibilityStatus
    confidence: Confidence
    evidence: Tuple[RecommendationEvidence, ...] = ()
    reason: Optional[str] = None
    model_name: Optional[str] = None
    can_run: Optional[bool] = None
    artifact_format: Optional[str] = None
    estimated_model_memory_gb: Optional[float] = None
    planning_budget_gb: Optional[float] = None
    memory_headroom_gb: Optional[float] = None
    recommended_context: Optional[int] = None
    unknowns: Tuple[str, ...] = ()
    alternatives: Tuple[str, ...] = ()
    artifacts: Tuple[ModelArtifact, ...] = ()
    resolution_status: Optional[str] = None
    resolution_confidence: Optional[Confidence] = None
    resolved_model: Optional[Model] = None
    runtime_candidates: Tuple[RuntimeCandidate, ...] = ()
    runtime_capabilities: Tuple[RuntimeCapability, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("compatibility result model id must not be empty")
        status_unknown = self.status == CompatibilityStatus.UNKNOWN
        confidence_unknown = self.confidence == Confidence.UNKNOWN
        if status_unknown != confidence_unknown:
            raise ValueError("unknown status and confidence must be represented together")
        if not status_unknown and not self.evidence:
            raise ValueError("known compatibility status requires evidence")
        expected_can_run = {
            CompatibilityStatus.EXCELLENT: True,
            CompatibilityStatus.GOOD: True,
            CompatibilityStatus.POSSIBLE: None,
            CompatibilityStatus.NOT_NATIVE: False,
            CompatibilityStatus.TOO_LARGE: False,
            CompatibilityStatus.UNKNOWN: None,
        }[self.status]
        if self.can_run is not expected_can_run:
            raise ValueError("can_run must agree with compatibility status")
        if self.recommended_context is not None and self.can_run is not True:
            raise ValueError("context can only be recommended for a practical fit")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "model_id": self.model_id,
            "artifact_id": self.artifact_id,
            "runtime": self.runtime,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "reason": self.reason,
            "model_name": self.model_name,
            "can_run": self.can_run,
            "artifact_format": self.artifact_format,
            "estimated_model_memory_gb": self.estimated_model_memory_gb,
            "planning_budget_gb": self.planning_budget_gb,
            "memory_headroom_gb": self.memory_headroom_gb,
            "recommended_context": self.recommended_context,
            "unknowns": list(self.unknowns),
            "alternatives": list(self.alternatives),
        }
        if self.resolution_status is not None:
            payload["compatibility_status"] = self.status.value
            payload["compatibility_confidence"] = self.confidence.value
            payload["resolution_status"] = self.resolution_status
            payload["resolution_confidence"] = (
                self.resolution_confidence.value if self.resolution_confidence else None
            )
            payload["model"] = self.resolved_model.to_dict() if self.resolved_model else None
            payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
            payload["runtime_candidates"] = [
                candidate.to_dict() for candidate in self.runtime_candidates
            ]
            payload["runtime_capabilities"] = [
                capability.to_dict() for capability in self.runtime_capabilities
            ]
        return payload


class RuntimeProvider(Protocol):
    """Minimum runtime boundary used by current readiness/setup behavior."""

    name: str

    def info(self) -> Dict[str, Any]: ...

    def is_available(self, endpoint: Optional[str] = None) -> bool: ...

    def ensure_available(self, endpoint: Optional[str] = None) -> bool: ...

    def capability(self, profile: Dict[str, Any]) -> RuntimeCapability: ...


class ExecutionAdapter(Protocol):
    """Boundary for runtime invocation implemented and controlled by LLMRig."""

    runtime: str

    def benchmark(
        self, target: ExecutionTarget, workload: RaceWorkload
    ) -> RaceCompetitor: ...


ResolvedType = TypeVar("ResolvedType", covariant=True)


class ModelSource(Protocol[ResolvedType]):
    """Minimum source boundary used by the curated catalog."""

    name: str

    def resolve(self, identifier: str) -> Optional[ResolvedType]: ...


@dataclass(frozen=True)
class ModelSpec:
    """Backward-compatible curated entry joining a logical model and artifact."""

    name: str
    ollama: str
    origin: str
    category: str
    params_b: float
    size_gb: float
    quant: str
    platforms: Tuple[str, ...]
    context_max: int
    modalities: str
    tier: int
    notes: str
    recommendable: bool = True
    aliases: Tuple[str, ...] = ()

    @property
    def model(self) -> Model:
        # The family-level identifier deliberately excludes the Ollama tag while
        # retaining the namespace for community derivatives.
        model_id = self.ollama.split(":", 1)[0]
        return Model(
            model_id=model_id,
            name=self.name,
            params_b=self.params_b,
            modalities=tuple(item.strip() for item in self.modalities.split(",")),
        )

    @property
    def artifact(self) -> ModelArtifact:
        label = self.quant.strip()
        if "MLX" in label.upper():
            artifact_format = "MLX"
            quantization = re.sub(r"\bMLX\b", "", label, flags=re.I).strip()
            quantization = quantization or None
        else:
            artifact_format = "Unknown"
            quantization = label
        return ModelArtifact(
            artifact_id=self.ollama,
            model_id=self.model.model_id,
            runtime="ollama",
            format=artifact_format,
            size_gb=self.size_gb,
            context_max=self.context_max,
            platforms=self.platforms,
            aliases=self.aliases,
            quantization=quantization,
            unknowns=(
                ("artifact file format is unknown",)
                if artifact_format == "Unknown"
                else ()
            ),
        )

    @property
    def ids(self) -> Tuple[str, ...]:
        return self.artifact.ids


ALL_PLATFORMS = ("Darwin", "Linux", "Windows")
MAC_ONLY = ("Darwin",)

# Curated entries are the only identifiers LLMRig may auto-pull.
# The live Hugging Face catalog is broader and future-facing, but informational only.
#
# Qwen3.8 tags are a verified snapshot of the Ollama pages on CURATED_SNAPSHOT_DATE.
# Alias tags are represented in ModelSpec.aliases instead of duplicate rows.
CURATED_MODELS: List[ModelSpec] = [
    # ----- Official Qwen3.8 -----
    ModelSpec(
        "Qwen3.8 27B MLX",
        "qwen3.8:27b-mlx",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        18.0,
        "MLX",
        MAC_ONLY,
        262_144,
        "text,image",
        102,
        "Apple Silicon optimized local build.",
    ),
    ModelSpec(
        "Qwen3.8 27B Q4_K_M",
        "qwen3.8:27b-q4_K_M",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        18.0,
        "Q4_K_M",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        100,
        "Portable efficiency-focused build.",
    ),
    ModelSpec(
        "Qwen3.8 27B Q8_0",
        "qwen3.8:27b-q8_0",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        30.0,
        "Q8_0",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        107,
        "Higher precision; needs substantially more memory headroom.",
    ),
    ModelSpec(
        "Qwen3.8 27B BF16",
        "qwen3.8:27b-bf16",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        56.0,
        "BF16",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        110,
        "Near-original precision; workstation/server memory class.",
        recommendable=False,
    ),
    ModelSpec(
        "Qwen3.8 27B MTP Q4_K_M",
        "qwen3.8:27b-mtp-q4_K_M",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        18.0,
        "MTP Q4_K_M",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        99,
        "MTP-enabled build; current :27b and :latest aliases point here.",
        recommendable=False,
        aliases=("qwen3.8:27b", "qwen3.8:latest"),
    ),
    ModelSpec(
        "Qwen3.8 27B MTP Q8_0",
        "qwen3.8:27b-mtp-q8_0",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        30.0,
        "MTP Q8_0",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        106,
        "MTP-enabled higher-precision build.",
        recommendable=False,
    ),
    ModelSpec(
        "Qwen3.8 27B MTP BF16",
        "qwen3.8:27b-mtp-bf16",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        56.0,
        "MTP BF16",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        109,
        "MTP-enabled BF16 build.",
        recommendable=False,
    ),
    ModelSpec(
        "Qwen3.8 27B MXFP8",
        "qwen3.8:27b-mxfp8",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        32.0,
        "MXFP8 MLX",
        MAC_ONLY,
        262_144,
        "text,image",
        106,
        "Apple Silicon MLX precision variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Qwen3.8 27B NVFP4",
        "qwen3.8:27b-nvfp4",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        18.0,
        "NVFP4 MLX",
        MAC_ONLY,
        262_144,
        "text,image",
        100,
        "Apple Silicon MLX 4-bit-class variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Qwen3.8 27B MLX BF16",
        "qwen3.8:27b-mlx-bf16",
        "Qwen / Ollama",
        OFFICIAL,
        27.0,
        56.0,
        "MLX BF16",
        MAC_ONLY,
        262_144,
        "text,image",
        110,
        "Apple Silicon BF16 build; requires very large unified memory.",
        recommendable=False,
    ),

    # ----- Official Qwen3 fallbacks for lower-memory systems -----
    ModelSpec(
        "Qwen3 0.6B",
        "qwen3:0.6b",
        "Qwen / Ollama",
        OFFICIAL,
        0.6,
        0.52,
        "Q4 class",
        ALL_PLATFORMS,
        40_960,
        "text",
        15,
        "Tiny footprint; useful for testing and constrained systems.",
    ),
    ModelSpec(
        "Qwen3 1.7B",
        "qwen3:1.7b",
        "Qwen / Ollama",
        OFFICIAL,
        1.7,
        1.4,
        "Q4 class",
        ALL_PLATFORMS,
        40_960,
        "text",
        25,
        "Small local assistant.",
    ),
    ModelSpec(
        "Qwen3 4B",
        "qwen3:4b",
        "Qwen / Ollama",
        OFFICIAL,
        4.0,
        2.5,
        "Q4 class",
        ALL_PLATFORMS,
        262_144,
        "text",
        42,
        "Fast small model; current default tag has extended context.",
    ),
    ModelSpec(
        "Qwen3 8B",
        "qwen3:8b",
        "Qwen / Ollama",
        OFFICIAL,
        8.0,
        5.2,
        "Q4_K_M",
        ALL_PLATFORMS,
        40_960,
        "text",
        55,
        "Good low-memory balance.",
    ),
    ModelSpec(
        "Qwen3 14B",
        "qwen3:14b",
        "Qwen / Ollama",
        OFFICIAL,
        14.0,
        9.3,
        "Q4_K_M",
        ALL_PLATFORMS,
        40_960,
        "text",
        70,
        "Strong mid-size fallback.",
    ),
    ModelSpec(
        "Qwen3 30B-A3B",
        "qwen3:30b-a3b",
        "Qwen / Ollama",
        OFFICIAL,
        30.0,
        19.0,
        "Q4 class MoE",
        ALL_PLATFORMS,
        262_144,
        "text",
        89,
        "Mixture-of-experts model; useful alternative to dense 27B.",
    ),
    ModelSpec(
        "Qwen3 32B",
        "qwen3:32b",
        "Qwen / Ollama",
        OFFICIAL,
        32.0,
        20.0,
        "Q4_K_M",
        ALL_PLATFORMS,
        40_960,
        "text",
        88,
        "Dense 32B fallback for machines with enough memory.",
        recommendable=False,
    ),

    # ----- Community Qwen3.8 reduced-refusal / 'uncensored' derivatives -----
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q4_K_M",
        "huihui_ai/Qwen3.8-abliterated:27b-q4_K_M",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        18.0,
        "Q4_K_M",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        100,
        "Community-modified Qwen3.8 with reduced refusal behavior.",
        aliases=(
            "huihui_ai/Qwen3.8-abliterated:27b",
            "huihui_ai/Qwen3.8-abliterated:latest",
        ),
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q6_K",
        "huihui_ai/Qwen3.8-abliterated:27b-q6_K",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        23.0,
        "Q6_K",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        104,
        "Higher-precision reduced-refusal option.",
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q8_0",
        "huihui_ai/Qwen3.8-abliterated:27b-q8_0",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        30.0,
        "Q8_0",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        107,
        "High-precision reduced-refusal option.",
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q2_K",
        "huihui_ai/Qwen3.8-abliterated:27b-q2_K",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        12.0,
        "Q2_K",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        70,
        "Very aggressive quantization.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q2_K_L",
        "huihui_ai/Qwen3.8-abliterated:27b-q2_K_L",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        18.0,
        "Q2_K_L",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        72,
        "Community large-tensor Q2 variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q3_K",
        "huihui_ai/Qwen3.8-abliterated:27b-q3_K",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        14.0,
        "Q3_K",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        80,
        "Low-memory community quantization.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q3_K_L",
        "huihui_ai/Qwen3.8-abliterated:27b-q3_K_L",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        20.0,
        "Q3_K_L",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        82,
        "Community large-tensor Q3 variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q4_K_L",
        "huihui_ai/Qwen3.8-abliterated:27b-q4_K_L",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        22.0,
        "Q4_K_L",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        98,
        "Community large-tensor Q4 variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q5_K",
        "huihui_ai/Qwen3.8-abliterated:27b-q5_K",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        20.0,
        "Q5_K",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        101,
        "Community Q5 variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q5_K_L",
        "huihui_ai/Qwen3.8-abliterated:27b-q5_K_L",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        24.0,
        "Q5_K_L",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        102,
        "Community large-tensor Q5 variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q6_K_L",
        "huihui_ai/Qwen3.8-abliterated:27b-q6_K_L",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        26.0,
        "Q6_K_L",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        105,
        "Community large-tensor Q6 variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated Q8_0_L",
        "huihui_ai/Qwen3.8-abliterated:27b-q8_0_L",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        40.0,
        "Q8_0_L",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        108,
        "Large-tensor Q8 community variant.",
        recommendable=False,
    ),
    ModelSpec(
        "Huihui Qwen3.8 27B Abliterated BF16",
        "huihui_ai/Qwen3.8-abliterated:27b-bf16",
        "Huihui AI / community",
        REDUCED_REFUSAL,
        27.0,
        56.0,
        "BF16",
        ALL_PLATFORMS,
        262_144,
        "text,image",
        110,
        "BF16 community derivative; very high memory requirement.",
        recommendable=False,
    ),
]

SPEED_PROMPT = (
    "Explain how a transformer neural network works in approximately 450 words. "
    "Cover tokenization, positional information, self-attention, multi-head attention, "
    "feed-forward layers, residual connections, normalization, and autoregressive generation. "
    "Be technically precise."
)

SMOKE_TESTS = [
    (
        "arithmetic",
        "Compute 17 * 23. Return only the integer and nothing else.",
        "391",
        lambda text: bool(re.search(r"\b391\b", text)),
    ),
    (
        "decimal-ordering",
        "Which number is larger, 9.11 or 9.9? Return only the larger number.",
        "9.9",
        lambda text: "9.9" in text.strip(),
    ),
    (
        "syllogism",
        "All flarns are trels. No trels are bims. Can any flarn be a bim? Answer only YES or NO.",
        "NO",
        lambda text: text.strip().upper().startswith("NO"),
    ),
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def run_cmd(cmd: Sequence[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def normalize_category(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {
        "unrestricted",
        "uncensored",
        "abliterated",
        "reduced",
        "reduced-refusal",
        REDUCED_REFUSAL,
    }:
        return REDUCED_REFUSAL
    if normalized in {"standard", "official", "restricted"}:
        return OFFICIAL
    if normalized == "all":
        return "all"
    raise ValueError(f"Unknown model category: {value}")


def clip(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(1, width - 1)] + "…"


def print_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str, int]]) -> None:
    if not rows:
        print("(none)")
        return

    widths: Dict[str, int] = {}
    for key, title, max_width in columns:
        value_width = max(len(str(row.get(key, ""))) for row in rows)
        widths[key] = min(max(len(title), value_width), max_width)

    print("  ".join(title.ljust(widths[key]) for key, title, _ in columns))
    print("  ".join("-" * widths[key] for key, _, _ in columns))
    for row in rows:
        print(
            "  ".join(
                clip(row.get(key, ""), widths[key]).ljust(widths[key])
                for key, _, _ in columns
            )
        )


def model_name_matches(requested: str, actual: str) -> bool:
    """Compare Ollama model names without risky substring matching."""

    def normalize(name: str) -> str:
        name = name.strip()
        if name.endswith(":latest"):
            return name[: -len(":latest")]
        return name

    return normalize(requested) == normalize(actual)


def validate_curated_catalog() -> List[str]:
    """Return catalog validation errors. An empty list means the catalog is sane."""
    errors: List[str] = []
    seen_primary: set[str] = set()
    seen_any: set[str] = set()

    for model in CURATED_MODELS:
        if model.category not in {OFFICIAL, REDUCED_REFUSAL}:
            errors.append(f"{model.ollama}: invalid category {model.category}")
        if model.size_gb <= 0:
            errors.append(f"{model.ollama}: non-positive size")
        if model.context_max <= 0:
            errors.append(f"{model.ollama}: non-positive context")
        if not model.platforms:
            errors.append(f"{model.ollama}: no supported platforms")
        if model.ollama in seen_primary:
            errors.append(f"duplicate primary model id: {model.ollama}")
        seen_primary.add(model.ollama)

        for identifier in model.ids:
            if identifier in seen_any:
                errors.append(f"duplicate model/alias id: {identifier}")
            seen_any.add(identifier)

    return errors


# ---------------------------------------------------------------------------
# Hardware discovery
# ---------------------------------------------------------------------------


def get_total_ram_bytes() -> int:
    system = platform.system()

    if system == "Darwin":
        try:
            return int(run_cmd(["sysctl", "-n", "hw.memsize"]).stdout.strip())
        except Exception:
            pass

    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)

    return 0


def get_available_ram_bytes() -> int:
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    if system == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)

    if system == "Darwin":
        # Approximation from vm_stat. macOS can also reclaim compressed/purgeable
        # memory, so this is intentionally shown as an approximate current value.
        try:
            text = run_cmd(["vm_stat"]).stdout
            page_size = 4096
            first_line = text.splitlines()[0] if text else ""
            match = re.search(r"page size of (\d+) bytes", first_line)
            if match:
                page_size = int(match.group(1))

            values: Dict[str, int] = {}
            for line in text.splitlines()[1:]:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                digits = re.sub(r"[^0-9]", "", raw_value)
                if digits:
                    values[key.strip()] = int(digits)

            pages = (
                values.get("Pages free", 0)
                + values.get("Pages inactive", 0)
                + values.get("Pages speculative", 0)
            )
            return pages * page_size
        except Exception:
            pass

    return 0


def get_swap_used_bytes() -> Optional[int]:
    system = platform.system()

    if system == "Linux":
        try:
            total_kb = 0
            free_kb = 0
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("SwapTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("SwapFree:"):
                        free_kb = int(line.split()[1])
            return max(0, total_kb - free_kb) * 1024
        except Exception:
            return None

    if system == "Darwin":
        try:
            text = run_cmd(["sysctl", "vm.swapusage"]).stdout
            match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", text, re.I)
            if not match:
                return None
            value = float(match.group(1))
            unit = match.group(2).upper()
            multiplier = {
                "K": 1024,
                "M": 1024**2,
                "G": 1024**3,
                "T": 1024**4,
                "P": 1024**5,
            }[unit]
            return int(value * multiplier)
        except Exception:
            return None

    # Windows does not expose a directly comparable swap-used value through the
    # lightweight standard-library path used here.
    return None


def get_cpu_name() -> str:
    system = platform.system()

    if system == "Darwin":
        try:
            data = json.loads(
                run_cmd(["system_profiler", "SPHardwareDataType", "-json"], timeout=30).stdout
            )
            items = data.get("SPHardwareDataType", [])
            if items:
                for key in ("chip_type", "cpu_type", "_name"):
                    if items[0].get(key):
                        return str(items[0][key])
        except Exception:
            pass

    if system == "Windows":
        try:
            output = run_cmd(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ]
            ).stdout.strip()
            if output:
                return output
        except Exception:
            pass

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    return platform.processor() or platform.machine() or "Unknown CPU"


def get_gpu_info() -> List[Dict[str, Any]]:
    system = platform.system()
    gpus: List[Dict[str, Any]] = []

    if command_exists("nvidia-smi"):
        try:
            output = run_cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ).stdout
            for line in output.splitlines():
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.split(",")]
                vram_gb = safe_float(parts[1]) / 1024 if len(parts) > 1 else 0.0
                gpus.append(
                    {
                        "name": parts[0],
                        "vram_gb": round(vram_gb, 1),
                        "backend": "CUDA",
                    }
                )
            if gpus:
                return gpus
        except Exception:
            pass

    if system == "Darwin":
        try:
            data = json.loads(
                run_cmd(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30).stdout
            )
            for item in data.get("SPDisplaysDataType", []):
                name = (
                    item.get("sppci_model")
                    or item.get("spdisplays_chipset-model")
                    or item.get("_name")
                    or "Apple GPU"
                )
                gpus.append(
                    {
                        "name": str(name),
                        "vram_gb": None,
                        "backend": "Metal / unified memory",
                    }
                )
        except Exception:
            pass

        if not gpus and platform.machine().lower() in {"arm64", "aarch64"}:
            gpus.append(
                {
                    "name": "Apple Silicon GPU",
                    "vram_gb": None,
                    "backend": "Metal / unified memory",
                }
            )
        return gpus

    if system == "Windows":
        try:
            ps = (
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
            )
            payload = json.loads(run_cmd(["powershell", "-NoProfile", "-Command", ps]).stdout)
            if isinstance(payload, dict):
                payload = [payload]
            for item in payload or []:
                raw = item.get("AdapterRAM")
                vram = gib(int(raw)) if raw else None
                gpus.append(
                    {
                        "name": item.get("Name", "GPU"),
                        "vram_gb": round(vram, 1) if vram else None,
                        "backend": "Windows GPU",
                    }
                )
        except Exception:
            pass

    if system == "Linux" and not gpus and command_exists("lspci"):
        try:
            for line in run_cmd(["lspci"]).stdout.splitlines():
                lower = line.lower()
                if "vga compatible controller" in lower or "3d controller" in lower:
                    gpus.append(
                        {
                            "name": line.split(":", 2)[-1].strip(),
                            "vram_gb": None,
                            "backend": "Linux GPU",
                        }
                    )
        except Exception:
            pass

    return gpus


def ollama_model_store_path() -> Path:
    custom = os.environ.get("OLLAMA_MODELS")
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".ollama" / "models"


def nearest_existing_path(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def privacy_safe_path(path: Path) -> str:
    """Hide the current home directory identity while preserving a useful path."""
    expanded = path.expanduser()
    try:
        relative = expanded.relative_to(Path.home())
    except ValueError:
        return str(expanded)
    return str(Path("~") / relative)


def hardware_profile() -> Dict[str, Any]:
    total = get_total_ram_bytes()
    available = get_available_ram_bytes()
    model_store = ollama_model_store_path()
    usage = shutil.disk_usage(str(nearest_existing_path(model_store)))
    swap = get_swap_used_bytes()

    return {
        "timestamp": now_iso(),
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu": get_cpu_name(),
        "ram_gib": round(gib(total), 1) if total else 0.0,
        "available_ram_gib": round(gib(available), 1) if available else None,
        "swap_used_gib": round(gib(swap), 2) if swap is not None else None,
        "gpus": get_gpu_info(),
        "model_store": str(model_store),
        "disk": {
            "total_gib": round(gib(usage.total), 1),
            "used_gib": round(gib(usage.used), 1),
            "free_gib": round(gib(usage.free), 1),
        },
    }


def accelerator_summary(profile: Dict[str, Any]) -> str:
    gpus = profile.get("gpus") or []
    if not gpus:
        return "No GPU detected (CPU inference may still work)"

    parts = []
    for gpu in gpus:
        vram = gpu.get("vram_gb")
        suffix = (
            f", {vram:.1f} GiB VRAM"
            if isinstance(vram, (int, float)) and vram
            else ""
        )
        parts.append(f"{gpu.get('name', 'GPU')} [{gpu.get('backend', '')}{suffix}]")
    return "; ".join(parts)


def model_budget_gb(profile: Dict[str, Any]) -> float:
    """Return a conservative model-weight budget, not a hard technical limit."""
    ram = safe_float(profile.get("ram_gib"))
    system = profile.get("os")
    gpus = profile.get("gpus") or []

    if system == "Darwin" and gpus:
        # Keep about 40% unified memory free for macOS, runtime, KV cache, and apps.
        return max(0.0, ram * 0.60)

    discrete_vram = max(
        [safe_float(gpu.get("vram_gb")) for gpu in gpus if gpu.get("vram_gb")] or [0.0]
    )
    if discrete_vram > 0:
        # Favor mostly/full GPU residency. CPU offload can work, but may be much slower.
        return discrete_vram * 0.88

    # CPU-only path: reserve more system memory for OS/runtime.
    return max(0.0, ram * 0.55)


def viability_label(profile: Dict[str, Any]) -> str:
    ram = safe_float(profile.get("ram_gib"))
    gpus = profile.get("gpus") or []

    if ram < 8:
        return "Limited: only very small models are practical."
    if not gpus:
        if ram < 16:
            return "Possible but CPU-bound: small models only; expect slower inference."
        return "Possible but CPU-bound: models may fit, but benchmark before committing to large downloads."
    if ram < 16:
        return "Basic: small 0.6B-4B/8B quantized models are the practical target."
    if ram < 24:
        return "Good: 8B-14B quantized models are practical."
    if ram < 32:
        return "Very good: 14B models are comfortable; some ~27B Q4-class builds may fit."
    if ram < 48:
        return "Excellent: ~27B/30B efficient quantizations are a realistic local target."
    if ram < 64:
        return "Excellent: ~27B/30B models are a strong local target with context headroom."
    return "Workstation class: larger quantizations and higher precision become practical."


# ---------------------------------------------------------------------------
# HTTP + Ollama
# ---------------------------------------------------------------------------


def http_json(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Tuple[Any, Dict[str, str]]:
    data = None
    headers = {
        "User-Agent": f"{PROJECT_SLUG}/{VERSION}",
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        parsed = json.loads(body.decode("utf-8")) if body else None
        return parsed, dict(response.headers.items())


def ollama_cli_info() -> Dict[str, Any]:
    path = shutil.which("ollama")
    result: Dict[str, Any] = {"installed": bool(path), "path": path, "version": None}
    if not path:
        return result

    try:
        process = run_cmd(["ollama", "--version"], timeout=10)
        result["version"] = (process.stdout or process.stderr).strip()
    except Exception:
        result["version"] = "unknown"
    return result


def ollama_api_alive(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    try:
        http_json(f"{host}/api/version", timeout=3)
        return True
    except Exception:
        return False


def ensure_ollama_service(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    if ollama_api_alive(host):
        return True
    if not command_exists("ollama"):
        return False

    try:
        kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except Exception:
        return False

    for _ in range(20):
        if ollama_api_alive(host):
            return True
        time.sleep(0.5)
    return False


class OllamaRuntimeProvider:
    """Runtime adapter preserving the current Ollama implementation."""

    name = "ollama"

    def info(self) -> Dict[str, Any]:
        return ollama_cli_info()

    def is_available(self, endpoint: Optional[str] = None) -> bool:
        return ollama_api_alive(endpoint or DEFAULT_OLLAMA_HOST)

    def ensure_available(self, endpoint: Optional[str] = None) -> bool:
        return ensure_ollama_service(endpoint or DEFAULT_OLLAMA_HOST)

    def capability(self, profile: Dict[str, Any]) -> RuntimeCapability:
        info = self.info()
        installed = bool(info.get("installed"))
        available = installed and self.is_available()
        evidence = []
        evidence.append(
            RecommendationEvidence(
                "verified-local-runtime",
                "Ollama CLI detection",
                (
                    "the Ollama command is installed locally"
                    if installed
                    else "the Ollama command was not detected locally"
                ),
            )
        )
        if available:
            evidence.append(
                RecommendationEvidence(
                    "verified-local-runtime",
                    "Ollama version API",
                    "the local Ollama service responded to its version endpoint",
                )
            )
        version = info.get("version")
        if isinstance(version, str) and version.startswith("unknown ("):
            version = None
        unknowns = () if version else ("runtime version is unknown",)
        return RuntimeCapability(
            runtime=self.name,
            installed=installed,
            available=available,
            version=version,
            supported_artifact_formats=("Ollama",),
            supported_platforms=ALL_PLATFORMS,
            supported_architectures=(),
            runtime_execution_capable=True,
            llmrig_installation_supported=False,
            llmrig_execution_supported=True,
            llmrig_benchmark_supported=True,
            confidence=Confidence.HIGH,
            evidence=tuple(evidence),
            unknowns=unknowns + ("supported architectures are unknown",),
        )


def detected_runtime_version(executable: str) -> Optional[str]:
    """Return a bounded, privacy-safe version line without exposing command errors."""
    try:
        process = run_cmd([executable, "--version"], timeout=5)
    except Exception:
        return None
    if process.returncode != 0:
        return None
    text = (process.stdout or process.stderr).strip().splitlines()
    if not text:
        return None
    return text[0][:200].replace(str(Path.home()), "~")


def llama_cpp_executable() -> Optional[str]:
    """Return the first recognized local llama.cpp CLI locator."""
    return next(
        (candidate for candidate in ("llama-cli", "llama.cpp") if shutil.which(candidate)),
        None,
    )


class LlamaCppRuntimeProvider:
    """Safe local capability detection for llama.cpp; never executes weights."""

    name = "llama.cpp"

    def info(self) -> Dict[str, Any]:
        executable = llama_cpp_executable()
        return {
            "installed": executable is not None,
            "version": detected_runtime_version(executable) if executable else None,
        }

    def is_available(self, endpoint: Optional[str] = None) -> bool:
        info = self.info()
        return bool(info["installed"] and info["version"])

    def ensure_available(self, endpoint: Optional[str] = None) -> bool:
        return self.is_available()

    def capability(self, profile: Dict[str, Any]) -> RuntimeCapability:
        info = self.info()
        installed = bool(info["installed"])
        available = installed and info["version"] is not None
        evidence = (
            RecommendationEvidence(
                "deterministic-runtime-knowledge",
                "llama.cpp GGUF runtime contract",
                "llama.cpp uses GGUF model artifacts",
            ),
            RecommendationEvidence(
                "verified-local-runtime",
                "llama.cpp executable detection",
                (
                    "a recognized llama.cpp command is installed locally"
                    if installed
                    else "no recognized llama.cpp command was detected locally"
                ),
            ),
        )
        if available:
            evidence += (
                RecommendationEvidence(
                    "verified-local-runtime",
                    "llama.cpp version command",
                    "the detected llama.cpp command returned version information",
                ),
            )
        unknowns = () if info["version"] else ("runtime version is unknown",)
        return RuntimeCapability(
            runtime=self.name,
            installed=installed,
            available=available,
            version=info["version"],
            supported_artifact_formats=("GGUF",),
            supported_platforms=ALL_PLATFORMS,
            supported_architectures=(),
            runtime_execution_capable=True,
            llmrig_installation_supported=False,
            llmrig_execution_supported=True,
            llmrig_benchmark_supported=True,
            confidence=Confidence.HIGH,
            evidence=evidence,
            unknowns=unknowns + ("supported architectures are unknown",),
        )


class MlxRuntimeProvider:
    """Detect an optional local MLX-LM installation without importing it."""

    name = "mlx-lm"

    def info(self) -> Dict[str, Any]:
        try:
            installed = importlib.util.find_spec("mlx_lm") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        version = None
        if installed:
            try:
                version = importlib.metadata.version("mlx-lm")
            except importlib.metadata.PackageNotFoundError:
                pass
        command_installed = shutil.which("mlx_lm.generate") is not None
        command_available = False
        if installed and command_installed:
            try:
                command_available = (
                    run_cmd(["mlx_lm.generate", "--help"], timeout=5).returncode == 0
                )
            except Exception:
                pass
        return {
            "installed": installed,
            "command_installed": command_installed,
            "command_available": command_available,
            "version": version,
        }

    def is_available(self, endpoint: Optional[str] = None) -> bool:
        info = self.info()
        return bool(
            info["installed"]
            and info["command_installed"]
            and info["command_available"]
            and platform.system() == "Darwin"
            and platform.machine().lower() in {"arm64", "aarch64"}
        )

    def ensure_available(self, endpoint: Optional[str] = None) -> bool:
        return self.is_available()

    def capability(self, profile: Dict[str, Any]) -> RuntimeCapability:
        info = self.info()
        installed = bool(info["installed"])
        system = str(profile.get("os") or platform.system())
        architecture = str(profile.get("arch") or platform.machine()).lower()
        platform_supported = system == "Darwin"
        architecture_supported = architecture in {"arm64", "aarch64"}
        command_installed = bool(info.get("command_installed"))
        command_available = bool(info.get("command_available"))
        available = (
            installed
            and command_installed
            and command_available
            and platform_supported
            and architecture_supported
        )
        evidence = (
            RecommendationEvidence(
                "deterministic-runtime-knowledge",
                "MLX-LM package contract",
                "MLX-LM executes repositories packaged for the MLX ecosystem",
            ),
            RecommendationEvidence(
                "verified-local-runtime",
                "Python package and command detection",
                (
                    "the optional mlx-lm package and generation command are installed locally"
                    if installed and command_installed
                    else "a complete local MLX-LM package and command installation was not detected"
                ),
            ),
        )
        blockers = []
        if installed and not platform_supported:
            blockers.append("MLX-LM is supported on macOS")
        if installed and not architecture_supported:
            blockers.append("MLX-LM requires an Apple Silicon architecture")
        if installed and not command_installed:
            blockers.append("the MLX-LM generation command was not detected")
        elif installed and not command_available:
            blockers.append("the MLX-LM generation command did not pass its health probe")
        return RuntimeCapability(
            runtime=self.name,
            installed=installed,
            available=available,
            version=info["version"],
            supported_artifact_formats=("MLX",),
            supported_platforms=("Darwin",),
            supported_architectures=("arm64", "aarch64"),
            runtime_execution_capable=True,
            llmrig_installation_supported=False,
            llmrig_execution_supported=True,
            llmrig_benchmark_supported=True,
            confidence=Confidence.HIGH,
            evidence=evidence,
            unknowns=tuple(blockers)
            + (() if info["version"] else ("runtime version is unknown",)),
        )


OLLAMA_RUNTIME: RuntimeProvider = OllamaRuntimeProvider()
LLAMA_CPP_RUNTIME: RuntimeProvider = LlamaCppRuntimeProvider()
MLX_RUNTIME: RuntimeProvider = MlxRuntimeProvider()
RUNTIME_PROVIDERS: Tuple[RuntimeProvider, ...] = (
    OLLAMA_RUNTIME,
    LLAMA_CPP_RUNTIME,
    MLX_RUNTIME,
)


def runtime_capabilities(profile: Dict[str, Any]) -> Tuple[RuntimeCapability, ...]:
    """Detect providers in stable display/serialization order."""
    return tuple(provider.capability(profile) for provider in RUNTIME_PROVIDERS)


def installed_ollama_models() -> List[Dict[str, str]]:
    if not command_exists("ollama"):
        return []
    try:
        lines = [
            line
            for line in run_cmd(["ollama", "list"], timeout=15).stdout.splitlines()
            if line.strip()
        ]
    except Exception:
        return []

    output: List[Dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        output.append(
            {
                "name": parts[0],
                "id": parts[1] if len(parts) > 1 else "",
                "raw": line,
            }
        )
    return output


class CuratedModelSource:
    """Trusted local snapshot; the only source currently eligible for auto-pull."""

    name = "curated"

    def __init__(self, specs: Sequence[ModelSpec]) -> None:
        self._specs = tuple(specs)

    def list_specs(self) -> Sequence[ModelSpec]:
        return self._specs

    def resolve(self, identifier: str) -> Optional[ModelSpec]:
        for model in self._specs:
            if identifier in model.artifact.ids:
                return model
        return None


CURATED_SOURCE: ModelSource = CuratedModelSource(CURATED_MODELS)


def resolve_curated_model(identifier: str) -> Optional[ModelSpec]:
    return CURATED_SOURCE.resolve(identifier)


def installed_id_for_spec(model: ModelSpec, installed_names: Iterable[str]) -> Optional[str]:
    names = list(installed_names)
    for identifier in model.ids:
        for installed in names:
            if model_name_matches(identifier, installed):
                return installed
    return None


def running_model_details(host: str, model_name: str) -> Dict[str, Any]:
    try:
        payload, _ = http_json(f"{host}/api/ps", timeout=5)
    except Exception:
        return {}

    for item in (payload or {}).get("models", []):
        actual_names = [str(item.get("name", "")), str(item.get("model", ""))]
        if not any(model_name_matches(model_name, actual) for actual in actual_names if actual):
            continue

        size = safe_float(item.get("size"))
        size_vram = safe_float(item.get("size_vram"))
        accelerator_percent = (size_vram / size * 100.0) if size > 0 else None
        return {
            "name": item.get("name") or item.get("model"),
            "size_bytes": int(size) if size else None,
            "size_vram_bytes": int(size_vram) if size_vram else None,
            "accelerator_percent": (
                round(accelerator_percent, 1) if accelerator_percent is not None else None
            ),
            "context_length": item.get("context_length"),
            "expires_at": item.get("expires_at"),
        }
    return {}



def running_ollama_models(host: str) -> List[str]:
    """Return model names currently resident in Ollama memory."""
    try:
        payload, _ = http_json(f"{host}/api/ps", timeout=5)
    except Exception:
        return []

    names: List[str] = []
    for item in (payload or {}).get("models", []):
        name = str(item.get("name") or item.get("model") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def unload_ollama_model(host: str, model: str, wait_seconds: float = 10.0) -> bool:
    """Ask Ollama to unload a model and wait briefly for it to leave memory."""
    try:
        http_json(
            f"{host}/api/generate",
            method="POST",
            payload={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
            timeout=60,
        )
    except Exception:
        return False

    deadline = time.time() + max(0.0, wait_seconds)
    while time.time() < deadline:
        if not any(model_name_matches(model, name) for name in running_ollama_models(host)):
            return True
        time.sleep(0.25)
    return not any(model_name_matches(model, name) for name in running_ollama_models(host))


def isolate_ollama_for_benchmark(host: str) -> List[str]:
    """Unload currently resident Ollama models before a benchmark.

    Benchmarking one model while another stays resident can create memory pressure,
    swap, and misleading throughput results on unified-memory and VRAM-limited systems.
    Returns the names that were requested to unload.
    """
    loaded = running_ollama_models(host)
    for name in loaded:
        unload_ollama_model(host, name)
    return loaded


def install_ollama_help(open_page: bool = False) -> None:
    system = platform.system()
    if system == "Darwin":
        url = "https://ollama.com/download/mac"
        message = "macOS: install the official Ollama app, launch it once, then rerun LLMRig."
    elif system == "Windows":
        url = "https://ollama.com/download/windows"
        message = "Windows: install the official Ollama setup app, then rerun LLMRig."
    elif system == "Linux":
        url = "https://ollama.com/download/linux"
        message = "Linux: use Ollama's official Linux installation instructions, then rerun LLMRig."
    else:
        url = "https://ollama.com/download"
        message = "Install Ollama for your platform, then rerun LLMRig."

    print("\nOllama is not installed.")
    print(message)
    print(f"Official download: {url}")
    if open_page:
        try:
            webbrowser.open(url)
        except Exception:
            pass


def pull_model(model: str) -> int:
    print(f"\nPulling model: {model}\n")
    try:
        return int(subprocess.run(["ollama", "pull", model]).returncode)
    except KeyboardInterrupt:
        eprint("\nDownload interrupted.")
        return 130
    except Exception as exc:
        eprint(f"Could not run ollama pull: {exc}")
        return 1


# ---------------------------------------------------------------------------
# Live Hugging Face discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelResolution:
    """A logical model and the artifacts evidenced by one upstream source."""

    status: str
    model: Optional[Model]
    artifacts: Tuple[ModelArtifact, ...] = ()
    confidence: Confidence = Confidence.UNKNOWN
    evidence: Tuple[RecommendationEvidence, ...] = ()
    message: Optional[str] = None


GGUF_QUANT_PATTERN = re.compile(
    r"(?:^|[-_.])((?:IQ(?:1_[SM]|2_(?:XXS|XS|S|M)|3_(?:XXS|XS|S|M)|4_(?:NL|XS)))|"
    r"(?:Q(?:2_K(?:_[SL])?|3_K(?:_[SML]|_XL)?|4_(?:0(?:_[48]_[48])?|1|K(?:_[SML]|_XL)?)|"
    r"5_(?:0|1|K(?:_[SML]|_XL)?)|6_K(?:_[SL])?|8_0)))(?=$|[-_.])",
    re.I,
)


def infer_gguf_quantization(filename: str) -> Tuple[Optional[str], str]:
    """Infer only conventional GGUF quant tokens, never arbitrary name fragments."""
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    matches = {match.upper() for match in GGUF_QUANT_PATTERN.findall(stem)}
    if len(matches) == 1:
        return next(iter(matches)), "inferred"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "unknown"


def hf_sibling_size(item: Dict[str, Any]) -> Optional[int]:
    raw_size = item.get("size")
    if raw_size is None and isinstance(item.get("lfs"), dict):
        raw_size = item["lfs"].get("size")
    try:
        size = int(raw_size)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


SAFETENSORS_SHARD_PATTERN = re.compile(
    r"^(.*)-(\d{5})-of-(\d{5})\.safetensors$", re.I
)


def hf_weight_set_size(files: Sequence[Dict[str, Any]]) -> Tuple[Optional[int], str]:
    """Sum only a single file or one complete, unambiguous Safetensors shard set."""
    if not files:
        return None, "missing"
    if len(files) == 1:
        size = hf_sibling_size(files[0])
        return (size, "verified") if size is not None else (None, "missing")

    matches = [
        SAFETENSORS_SHARD_PATTERN.match(str(item.get("rfilename") or ""))
        for item in files
    ]
    if not all(matches):
        return None, "ambiguous"
    shard_keys = {(match.group(1), int(match.group(3))) for match in matches if match}
    if len(shard_keys) != 1:
        return None, "ambiguous"
    _, total = next(iter(shard_keys))
    indices = {int(match.group(2)) for match in matches if match}
    if len(files) != total or indices != set(range(1, total + 1)):
        return None, "ambiguous"
    sizes = [hf_sibling_size(item) for item in files]
    if any(size is None for size in sizes):
        return None, "missing"
    return sum(size for size in sizes if size is not None), "verified"


def hf_context_max(payload: Dict[str, Any]) -> Optional[int]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        try:
            value = int(config.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def hf_explicit_quantization(payload: Dict[str, Any]) -> Optional[str]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    quant_config = config.get("quantization_config")
    if isinstance(quant_config, dict) and isinstance(quant_config.get("quant_method"), str):
        return str(quant_config["quant_method"])
    mlx_quant = config.get("quantization")
    if isinstance(mlx_quant, dict):
        try:
            bits = int(mlx_quant.get("bits"))
        except (TypeError, ValueError):
            return None
        if bits > 0:
            return f"{bits}-bit"
    return None


def hf_base_model_id(payload: Dict[str, Any]) -> Optional[str]:
    candidates = []
    card_data = payload.get("cardData")
    if isinstance(card_data, dict):
        base_model = card_data.get("base_model")
        if isinstance(base_model, str):
            candidates.append(base_model)
        elif isinstance(base_model, list):
            candidates.extend(item for item in base_model if isinstance(item, str))
    for tag in payload.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("base_model:"):
            candidates.append(tag.split(":", 1)[1])
    unique = tuple(dict.fromkeys(item.strip() for item in candidates if item.strip()))
    return unique[0] if len(unique) == 1 else None


def hf_model_from_metadata(identifier: str, payload: Dict[str, Any]) -> Model:
    repository_id = str(payload.get("id") or payload.get("modelId") or identifier)
    model_id = hf_base_model_id(payload) or repository_id

    task = str(payload.get("pipeline_tag") or "").lower()
    if task == "image-text-to-text":
        modalities = ("text", "image")
    elif task in LIVE_LLM_TASKS:
        modalities = ("text",)
    else:
        modalities = ("unknown",)
    return Model(model_id, model_id.rsplit("/", 1)[-1], None, modalities)


def hf_artifacts_from_metadata(
    model: Model, payload: Dict[str, Any]
) -> Tuple[ModelArtifact, ...]:
    repository_id = str(payload.get("id") or payload.get("modelId") or model.model_id)
    siblings = [item for item in (payload.get("siblings") or []) if isinstance(item, dict)]
    siblings.sort(key=lambda item: str(item.get("rfilename") or ""))
    context_max = hf_context_max(payload)
    explicit_quant = hf_explicit_quantization(payload)
    artifacts: List[ModelArtifact] = []

    for item in siblings:
        filename = str(item.get("rfilename") or "")
        if not filename.lower().endswith(".gguf"):
            continue
        size_bytes = hf_sibling_size(item)
        quantization, quant_status = infer_gguf_quantization(filename)
        evidence = [
            RecommendationEvidence(
                "deterministic-inference",
                "GGUF .gguf extension rule",
                f"{filename} is recognized as a GGUF artifact from its file extension",
            )
        ]
        unknowns = ["runtime compatibility is unknown"]
        if size_bytes is not None:
            evidence.append(
                RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face repository file metadata",
                    f"artifact byte size is reported for {filename}",
                )
            )
        else:
            unknowns.append("artifact size is unknown")
        if quant_status == "inferred":
            evidence.append(
                RecommendationEvidence(
                    "deterministic-inference",
                    "conventional GGUF quantization filename rule",
                    f"quantization {quantization} is inferred from {filename}",
                )
            )
        elif quant_status == "ambiguous":
            unknowns.append("quantization is ambiguous")
        else:
            unknowns.append("quantization is unknown")
        if context_max is None:
            unknowns.append("context limit is unknown")
        else:
            evidence.append(
                RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face model config metadata",
                    "context limit is reported by repository metadata",
                )
            )
        artifacts.append(
            ModelArtifact(
                artifact_id=f"hf://{repository_id}/{filename}",
                model_id=model.model_id,
                runtime=None,
                format="GGUF",
                size_gb=size_bytes / 1_000_000_000 if size_bytes else None,
                context_max=context_max,
                platforms=(),
                quantization=quantization,
                size_bytes=size_bytes,
                evidence=tuple(evidence),
                unknowns=tuple(unknowns),
            )
        )

    weight_files = [
        item
        for item in siblings
        if str(item.get("rfilename") or "").lower().endswith(".safetensors")
    ]
    tags = {str(tag).lower() for tag in (payload.get("tags") or [])}
    is_mlx = str(payload.get("library_name") or "").lower() == "mlx" or "mlx" in tags
    if weight_files or is_mlx:
        size_bytes, size_status = hf_weight_set_size(weight_files)
        artifact_format = "MLX" if is_mlx else "Safetensors"
        format_source = (
            "Hugging Face library/tag metadata"
            if is_mlx
            else "Safetensors .safetensors extension rule"
        )
        format_kind = "verified-metadata" if is_mlx else "deterministic-inference"
        evidence = [
            RecommendationEvidence(
                format_kind,
                format_source,
                f"repository weights are recognized as {artifact_format}",
            )
        ]
        unknowns = ["runtime compatibility is unknown"]
        if size_bytes is not None:
            evidence.append(
                RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face repository file metadata",
                    "artifact size is the sum of all listed weight-file sizes",
                )
            )
        elif size_status == "ambiguous":
            unknowns.append("artifact size is unknown because weight-file grouping is ambiguous")
        else:
            unknowns.append("artifact size is unknown")
        if explicit_quant is not None:
            evidence.append(
                RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face model config metadata",
                    f"quantization {explicit_quant} is explicitly reported",
                )
            )
        else:
            unknowns.append("quantization is unknown")
        if context_max is None:
            unknowns.append("context limit is unknown")
        else:
            evidence.append(
                RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face model config metadata",
                    "context limit is reported by repository metadata",
                )
            )
        artifacts.append(
            ModelArtifact(
                artifact_id=f"hf://{repository_id}/{artifact_format.lower()}",
                model_id=model.model_id,
                runtime=None,
                format=artifact_format,
                size_gb=size_bytes / 1_000_000_000 if size_bytes else None,
                context_max=context_max,
                platforms=(),
                quantization=explicit_quant,
                size_bytes=size_bytes,
                evidence=tuple(evidence),
                unknowns=tuple(unknowns),
            )
        )
    return tuple(artifacts)


class HuggingFaceModelSource:
    """Generic, read-only Hugging Face metadata resolver with no weight downloads."""

    name = "hugging-face"

    def resolve(self, identifier: str) -> ModelResolution:
        encoded = urllib.parse.quote(identifier.strip(), safe="/")
        url = f"{HF_MODELS_API}/{encoded}?blobs=true"
        try:
            payload, _ = http_json(url, timeout=20)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 404}:
                return ModelResolution(
                    "not_found_or_inaccessible",
                    None,
                    message="repository was not found or is inaccessible",
                )
            return ModelResolution("network_error", None, message="Hugging Face request failed")
        except Exception:
            return ModelResolution("network_error", None, message="Hugging Face is unavailable")
        if not isinstance(payload, dict):
            return ModelResolution("network_error", None, message="unexpected Hugging Face response")

        model = hf_model_from_metadata(identifier, payload)
        artifacts = hf_artifacts_from_metadata(model, payload)
        evidence = (
            RecommendationEvidence(
                "verified-metadata",
                "Hugging Face model API",
                "repository identity and metadata were returned by Hugging Face",
            ),
        )
        base_model = hf_base_model_id(payload)
        if base_model:
            evidence += (
                RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face base_model metadata",
                    f"repository explicitly links its artifacts to logical model {base_model}",
                ),
            )
        confidence = Confidence.HIGH if artifacts else Confidence.MEDIUM
        return ModelResolution("resolved", model, artifacts, confidence, evidence)


HF_SOURCE: ModelSource[ModelResolution] = HuggingFaceModelSource()


def cache_dir() -> Path:
    if platform.system() == "Windows" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    path = base / PROJECT_SLUG
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_file() -> Path:
    return cache_dir() / "catalog.json"


def load_cache() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(cache_file().read_text(encoding="utf-8")) if cache_file().exists() else None
    except Exception:
        return None


def write_cache(payload: Dict[str, Any]) -> None:
    try:
        cache_file().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def next_link(header: str) -> Optional[str]:
    for part in (header or "").split(","):
        if 'rel="next"' in part:
            match = re.search(r"<([^>]+)>", part)
            if match:
                return match.group(1)
    return None


def hf_list_models(params: Dict[str, Any], max_pages: int = 20) -> List[Dict[str, Any]]:
    url: Optional[str] = f"{HF_MODELS_API}?{urllib.parse.urlencode(params, doseq=True)}"
    output: List[Dict[str, Any]] = []
    pages = 0

    while url and pages < max_pages:
        payload, headers = http_json(url, timeout=20)
        if not isinstance(payload, list):
            break
        output.extend(payload)
        pages += 1
        url = next_link(headers.get("Link", "") or headers.get("link", ""))

    return output


def fetch_live_catalog(force: bool = False) -> Dict[str, Any]:
    cached = load_cache()
    if (
        cached
        and not force
        and time.time() - safe_float(cached.get("fetched_epoch")) < CACHE_TTL_SECONDS
    ):
        return cached

    try:
        official = hf_list_models(
            {
                "author": "Qwen",
                "sort": "lastModified",
                "direction": "-1",
                "limit": 1000,
            }
        )

        community: Dict[str, Dict[str, Any]] = {}
        for term in ("Qwen abliterated", "Qwen uncensored", "Qwen heretic"):
            try:
                found = hf_list_models(
                    {
                        "search": term,
                        "sort": "lastModified",
                        "direction": "-1",
                        "limit": 100,
                    },
                    max_pages=2,
                )
            except Exception:
                found = []

            for item in found:
                model_id = str(item.get("id") or item.get("modelId") or "")
                lowered = model_id.lower()
                if model_id.startswith("Qwen/"):
                    continue
                if "qwen" not in lowered:
                    continue
                if not any(term_part in lowered for term_part in REDUCED_TERMS):
                    continue
                community[model_id] = item

        payload = {
            "fetched_at": now_iso(),
            "fetched_epoch": time.time(),
            "official": official,
            "community_reduced_refusal": list(community.values()),
        }
        write_cache(payload)
        return payload
    except Exception as exc:
        if cached:
            cached["warning"] = f"Live refresh failed; using cached catalog: {exc}"
            return cached
        return {
            "fetched_at": None,
            "fetched_epoch": 0,
            "official": [],
            "community_reduced_refusal": [],
            "warning": f"Live model discovery unavailable: {exc}",
        }


def parse_total_params_b(model_id: str) -> Optional[float]:
    """Infer total parameter count from common model-name patterns."""
    matches = list(
        re.finditer(r"(?<![A-Za-z])(\d+(?:\.\d+)?)([BT])(?:-|$)", model_id, re.I)
    )
    if not matches:
        return None
    value = float(matches[0].group(1))
    return value * 1000 if matches[0].group(2).upper() == "T" else value


def q4_estimate_gb(params_b: Optional[float]) -> Optional[float]:
    """Rough Q4 planning estimate only; never used as a live repository fit claim."""
    return params_b * 0.67 if params_b is not None else None


def is_live_llm_candidate(item: Dict[str, Any]) -> bool:
    """Best-effort filter for repositories that look like runnable language/multimodal LLMs.

    Hugging Face organizations can contain benchmarks, interpretability artifacts,
    speech models, and other repositories. LLMRig keeps those available under
    ``models --all`` but does not mix them into the default local-LLM discovery view.
    """
    model_id = str(item.get("id") or item.get("modelId") or "")
    lowered = model_id.lower()
    task = str(item.get("pipeline_tag") or "").strip().lower()

    if any(hint in lowered for hint in LIVE_NON_LLM_HINTS):
        return False
    if task in LIVE_LLM_TASKS:
        return True
    if task:
        return False

    # Some fresh or community repositories have no pipeline tag yet. Keep a
    # conservative name-based fallback for parameterized Qwen model repos.
    return "qwen" in lowered and parse_total_params_b(model_id) is not None


def live_rows(
    items: Iterable[Dict[str, Any]],
    category: str,
) -> List[Dict[str, Any]]:
    """Render discovery metadata without pretending an unverified repo has a known fit.

    A Hugging Face repository name is not enough to know its actual local package
    size, quantization, backend compatibility, or runtime memory use. Hardware-fit
    labels therefore belong only to the curated catalog.
    """
    rows: List[Dict[str, Any]] = []

    for item in items:
        model_id = str(item.get("id") or item.get("modelId") or "")
        params = parse_total_params_b(model_id)
        updated = str(item.get("lastModified") or "-")
        if "T" in updated:
            updated = updated.split("T", 1)[0]

        rows.append(
            {
                "category": "official" if category == OFFICIAL else "community reduced-refusal",
                "model": model_id,
                "task": str(item.get("pipeline_tag") or "-"),
                "params": f"{params:.1f}B" if params is not None else "-",
                "updated": updated,
                "status": "discovery only",
            }
        )
    return rows


def fit_label(model: ModelSpec, profile: Dict[str, Any]) -> str:
    return assess_curated_compatibility(model, profile).status.value


def assess_curated_compatibility(
    model: ModelSpec, profile: Dict[str, Any]
) -> CompatibilityResult:
    """Assess a verified curated artifact with conservative deterministic heuristics."""
    budget = model_budget_gb(profile)
    ram = safe_float(profile.get("ram_gib"))
    artifact = model.artifact
    curated_evidence = RecommendationEvidence(
        "curated-metadata",
        f"curated snapshot {CURATED_SNAPSHOT_DATE}",
        "artifact identity and available metadata come from the curated catalog",
    )
    estimate_evidence = RecommendationEvidence(
        "deterministic-estimate",
        "local hardware profile",
        "model-weight budget is calculated with conservative headroom",
    )
    runtime_evidence = RecommendationEvidence(
        "runtime-support",
        "LLMRig Ollama runtime provider",
        "LLMRig implements curated artifact execution through Ollama",
    )
    base_unknowns = (
        "runtime memory overhead is not measured",
        "generation performance is not predicted or measured",
    )
    if not profile.get("os") or not ram or artifact.size_gb is None:
        missing = []
        if not profile.get("os"):
            missing.append("hardware platform is unknown")
        if not ram:
            missing.append("total system memory is unknown")
        if artifact.size_gb is None:
            missing.append("artifact memory estimate is unknown")
        return CompatibilityResult(
            model.model.model_id,
            artifact.artifact_id,
            artifact.runtime,
            CompatibilityStatus.UNKNOWN,
            Confidence.UNKNOWN,
            (curated_evidence,),
            "; ".join(missing),
            model.name,
            None,
            model.quant,
            artifact.size_gb,
            round(budget, 1) if budget else None,
            None,
            None,
            tuple(missing) + base_unknowns,
        )

    evidence = (
        curated_evidence,
        estimate_evidence,
        runtime_evidence,
    )
    headroom = round(budget - artifact.size_gb, 1)
    context: Optional[int] = None
    confidence = Confidence.HIGH
    reason: Optional[str] = None
    can_run: Optional[bool]
    if profile.get("os") not in artifact.platforms:
        status = CompatibilityStatus.NOT_NATIVE
        can_run = False
        reason = "the curated artifact does not support this operating system"
    elif artifact.size_gb <= budget * 0.75:
        status = CompatibilityStatus.EXCELLENT
        can_run = True
    elif artifact.size_gb <= budget:
        status = CompatibilityStatus.GOOD
        can_run = True
    elif artifact.size_gb <= ram * 0.80:
        status = CompatibilityStatus.POSSIBLE
        can_run = None
        confidence = Confidence.MEDIUM
        reason = "the artifact may require memory spill beyond the conservative budget"
    else:
        status = CompatibilityStatus.TOO_LARGE
        can_run = False
        reason = "the artifact leaves insufficient memory for the OS, runtime, and context"

    if can_run:
        context = recommended_context(profile, model)
        evidence += (
            RecommendationEvidence(
                "context-heuristic",
                "curated context limit and local hardware profile",
                "practical starting context is selected by a deterministic heuristic",
            ),
        )

    return CompatibilityResult(
        model.model.model_id,
        artifact.artifact_id,
        artifact.runtime,
        status,
        confidence,
        evidence,
        reason,
        model.name,
        can_run,
        model.quant,
        round(artifact.size_gb, 1),
        round(budget, 1),
        headroom,
        context,
        base_unknowns,
    )


def print_curated(profile: Optional[Dict[str, Any]] = None) -> None:
    rows: List[Dict[str, Any]] = []
    for model in CURATED_SOURCE.list_specs():
        aliases = ", ".join(model.aliases) if model.aliases else "-"
        rows.append(
            {
                "class": "official" if model.category == OFFICIAL else "reduced-refusal",
                "model": model.ollama,
                "aliases": aliases,
                "quant": model.quant,
                "size": f"{model.size_gb:.1f} GB",
                "ctx": f"{int(model.context_max / 1024)}K",
                "input": model.modalities,
                "fit": fit_label(model, profile) if profile else "-",
                "rec": "yes" if model.recommendable else "manual",
            }
        )

    print_table(
        rows,
        [
            ("class", "Class", 16),
            ("model", "Ollama model", 54),
            ("aliases", "Aliases", 38),
            ("quant", "Format", 15),
            ("size", "Size", 10),
            ("ctx", "Max ctx", 9),
            ("input", "Input", 12),
            ("fit", "This machine", 15),
            ("rec", "Auto-rec", 8),
        ],
    )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def candidate_models(profile: Dict[str, Any], category: str) -> List[ModelSpec]:
    system = profile.get("os")
    ram = safe_float(profile.get("ram_gib"))
    budget = model_budget_gb(profile)
    disk_free = safe_float((profile.get("disk") or {}).get("free_gib"))

    output: List[ModelSpec] = []
    for model in CURATED_SOURCE.list_specs():
        if not model.recommendable:
            continue
        if model.category != category:
            continue
        if system not in model.platforms:
            continue
        if disk_free and disk_free < model.size_gb * 1.25:
            continue
        if model.size_gb > budget:
            continue
        output.append(model)
    return output


def recommend_model(
    profile: Dict[str, Any],
    category: str = OFFICIAL,
    preference: str = "balanced",
) -> Optional[ModelSpec]:
    category = normalize_category(category)
    if category == "all":
        category = OFFICIAL

    candidates = candidate_models(profile, category)
    if not candidates:
        return None

    ram = safe_float(profile.get("ram_gib"))
    system = profile.get("os")

    preferred: List[str]
    if category == OFFICIAL:
        if preference == "speed":
            preferred = (
                [
                    "qwen3.8:27b-mlx" if system == "Darwin" else "qwen3.8:27b-q4_K_M",
                    "qwen3:14b",
                    "qwen3:8b",
                ]
                if ram >= 24
                else ["qwen3:8b", "qwen3:4b", "qwen3:1.7b"]
            )
        elif preference == "quality":
            if ram >= 64:
                preferred = [
                    "qwen3.8:27b-q8_0",
                    "qwen3.8:27b-mlx" if system == "Darwin" else "qwen3.8:27b-q4_K_M",
                    "qwen3:30b-a3b",
                ]
            elif ram >= 32:
                preferred = [
                    "qwen3.8:27b-mlx" if system == "Darwin" else "qwen3.8:27b-q4_K_M",
                    "qwen3:30b-a3b",
                    "qwen3:14b",
                ]
            else:
                preferred = ["qwen3:14b", "qwen3:8b"]
        else:
            preferred = (
                [
                    "qwen3.8:27b-mlx" if system == "Darwin" else "qwen3.8:27b-q4_K_M",
                    "qwen3:30b-a3b",
                    "qwen3:14b",
                ]
                if ram >= 32
                else ["qwen3:14b", "qwen3:8b", "qwen3:4b"]
                if ram >= 16
                else ["qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"]
            )
    else:
        if preference == "speed":
            preferred = ["huihui_ai/Qwen3.8-abliterated:27b-q4_K_M"]
        elif preference == "quality" and ram >= 64:
            preferred = [
                "huihui_ai/Qwen3.8-abliterated:27b-q8_0",
                "huihui_ai/Qwen3.8-abliterated:27b-q6_K",
                "huihui_ai/Qwen3.8-abliterated:27b-q4_K_M",
            ]
        elif ram >= 40:
            preferred = [
                "huihui_ai/Qwen3.8-abliterated:27b-q6_K",
                "huihui_ai/Qwen3.8-abliterated:27b-q4_K_M",
            ]
        else:
            preferred = ["huihui_ai/Qwen3.8-abliterated:27b-q4_K_M"]

    by_id = {model.ollama: model for model in candidates}
    for identifier in preferred:
        if identifier in by_id:
            return by_id[identifier]

    # Generic fallback: highest tier among candidates that fit.
    return max(candidates, key=lambda model: model.tier)


def recommended_context(profile: Dict[str, Any], model: ModelSpec) -> int:
    ram = safe_float(profile.get("ram_gib"))
    max_context = int(model.context_max or 32_768)

    if ram >= 96 and model.size_gb <= ram * 0.40:
        return min(131_072, max_context)
    if ram >= 64 and model.size_gb <= ram * 0.45:
        return min(65_536, max_context)
    if ram >= 48 and model.size_gb <= ram * 0.25:
        return min(65_536, max_context)
    return min(32_768, max_context)


def runtime_candidates_for_artifacts(
    artifacts: Sequence[ModelArtifact],
    capabilities: Sequence[RuntimeCapability],
    profile: Dict[str, Any],
) -> Tuple[RuntimeCandidate, ...]:
    """Match recognized artifacts to runtime capabilities without executing them."""
    candidates = []
    system = str(profile.get("os") or "")
    architecture = str(profile.get("arch") or "").lower()
    budget = model_budget_gb(profile)
    for artifact in artifacts:
        for capability in capabilities:
            if artifact.format not in capability.supported_artifact_formats:
                continue
            blockers = []
            support_status = "available"
            if system and capability.supported_platforms and system not in capability.supported_platforms:
                support_status = "platform_incompatible"
                blockers.append(f"runtime does not support platform {system}")
            elif not system and capability.supported_platforms:
                support_status = "platform_unknown"
                blockers.append("machine platform is unknown")
            elif (
                architecture
                and capability.supported_architectures
                and architecture not in capability.supported_architectures
            ):
                support_status = "architecture_incompatible"
                blockers.append(f"runtime does not support architecture {architecture}")
            elif not architecture and capability.supported_architectures:
                support_status = "architecture_unknown"
                blockers.append("machine architecture is unknown")
            elif not capability.runtime_execution_capable:
                support_status = "execution_unsupported"
                blockers.append("runtime execution capability is unsupported")
            elif not capability.installed:
                support_status = "runtime_not_installed"
                blockers.append("runtime is not installed")
            elif not capability.available:
                support_status = "runtime_unavailable"
                blockers.append("runtime is installed but is not currently available")
            if artifact.size_bytes is None or budget <= 0:
                fit_result = "unknown"
            elif gib(artifact.size_bytes) <= budget:
                fit_result = "fits"
            else:
                fit_result = "too_large"
            unknowns = [
                "successful model inference has not been verified",
                "runtime memory overhead is unknown",
            ]
            unknowns.extend(capability.unknowns)
            match_evidence = RecommendationEvidence(
                "deterministic-runtime-match",
                f"{capability.runtime} artifact capability",
                f"{capability.runtime} declares support for {artifact.format} artifacts",
            )
            candidates.append(
                RuntimeCandidate(
                    runtime=capability.runtime,
                    artifact_id=artifact.artifact_id,
                    artifact_format=artifact.format,
                    support_status=support_status,
                    fit_result=fit_result,
                    confidence=Confidence.HIGH,
                    evidence=capability.evidence + (match_evidence,),
                    blockers=tuple(blockers),
                    unknowns=tuple(dict.fromkeys(unknowns)),
                )
            )
    return tuple(sorted(candidates, key=lambda item: (item.artifact_id, item.runtime)))


def assess_generic_runtime_compatibility(
    resolution: ModelResolution,
    profile: Dict[str, Any],
) -> CompatibilityResult:
    """Combine artifact, runtime, and conservative machine-fit evidence."""
    assert resolution.model is not None
    capabilities = runtime_capabilities(profile)
    candidates = runtime_candidates_for_artifacts(
        resolution.artifacts, capabilities, profile
    )
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in resolution.artifacts}
    budget = model_budget_gb(profile)
    fitting = [
        (candidate, artifacts_by_id[candidate.artifact_id])
        for candidate in candidates
        if candidate.support_status == "available" and candidate.fit_result == "fits"
    ]
    common = {
        "model_id": resolution.model.model_id,
        "model_name": resolution.model.name,
        "artifacts": resolution.artifacts,
        "resolution_status": resolution.status,
        "resolution_confidence": resolution.confidence,
        "resolved_model": resolution.model,
        "runtime_candidates": candidates,
        "runtime_capabilities": capabilities,
    }
    base_unknowns = (
        "practical context is unknown without measured runtime memory behavior",
        "generation performance is not predicted or measured",
    )
    if fitting:
        candidate, artifact = min(
            fitting,
            key=lambda item: (int(item[1].size_bytes or 0), item[1].artifact_id, item[0].runtime),
        )
        size_gib = gib(int(artifact.size_bytes or 0))
        memory_evidence = RecommendationEvidence(
            "deterministic-estimate",
            "verified artifact size and local hardware profile",
            "artifact size is within the conservative local model-weight budget",
        )
        return CompatibilityResult(
            artifact_id=artifact.artifact_id,
            runtime=candidate.runtime,
            status=CompatibilityStatus.GOOD,
            confidence=Confidence.HIGH,
            evidence=resolution.evidence + candidate.evidence + (memory_evidence,),
            reason="an available runtime supports the artifact and its verified size fits the conservative memory budget",
            can_run=True,
            artifact_format=artifact.format,
            estimated_model_memory_gb=round(size_gib, 1),
            planning_budget_gb=round(budget, 1),
            memory_headroom_gb=round(budget - size_gib, 1),
            unknowns=base_unknowns + ("successful inference has not been verified",),
            **common,
        )
    definitively_ruled_out = [
        candidate
        for candidate in candidates
        if candidate.fit_result == "too_large"
        or candidate.support_status
        in {"platform_incompatible", "architecture_incompatible"}
    ]
    if candidates and len(definitively_ruled_out) == len(candidates):
        candidate_artifacts = [
            (candidate, artifacts_by_id[candidate.artifact_id])
            for candidate in candidates
        ]
        candidate, artifact = min(
            candidate_artifacts,
            key=lambda item: (
                item[0].fit_result != "too_large",
                int(item[1].size_bytes or 0),
                item[1].artifact_id,
                item[0].runtime,
            ),
        )
        size_gib = gib(artifact.size_bytes) if artifact.size_bytes is not None else None
        if candidate.fit_result == "too_large":
            exclusion_evidence = RecommendationEvidence(
                "deterministic-estimate",
                "verified artifact size and local hardware profile",
                "artifact size exceeds the conservative local model-weight budget",
            )
            status = CompatibilityStatus.TOO_LARGE
        else:
            exclusion_evidence = RecommendationEvidence(
                "deterministic-runtime-match",
                f"{candidate.runtime} platform capability",
                "the runtime candidate is incompatible with the detected platform or architecture",
            )
            status = CompatibilityStatus.NOT_NATIVE
        return CompatibilityResult(
            artifact_id=artifact.artifact_id,
            runtime=candidate.runtime,
            status=status,
            confidence=Confidence.HIGH,
            evidence=resolution.evidence + candidate.evidence + (exclusion_evidence,),
            reason="every supported runtime path is definitively ruled out by machine fit or platform compatibility",
            can_run=False,
            artifact_format=artifact.format,
            estimated_model_memory_gb=round(size_gib, 1) if size_gib is not None else None,
            planning_budget_gb=round(budget, 1),
            memory_headroom_gb=(
                round(budget - size_gib, 1) if size_gib is not None else None
            ),
            unknowns=base_unknowns,
            **common,
        )

    if not resolution.artifacts:
        artifact_unknown = "no GGUF, MLX, or Safetensors artifacts were recognized"
    elif not candidates:
        artifact_unknown = "recognized artifacts have no supported execution runtime"
    elif all(candidate.fit_result == "unknown" for candidate in candidates):
        artifact_unknown = "artifact size or machine planning budget is unknown"
    else:
        artifact_unknown = "at least one plausible runtime path remains unresolved"
    return CompatibilityResult(
        artifact_id=None,
        runtime=None,
        status=CompatibilityStatus.UNKNOWN,
        confidence=Confidence.UNKNOWN,
        evidence=resolution.evidence,
        reason="repository resolved, but practical execution compatibility is unknown",
        can_run=None,
        unknowns=(artifact_unknown,) + base_unknowns,
        **common,
    )


def compatibility_for_identifier(
    identifier: str, profile: Dict[str, Any]
) -> CompatibilityResult:
    """Resolve curated IDs first, then generic Hugging Face repository metadata."""
    model = CURATED_SOURCE.resolve(identifier)
    if model is None:
        parts = identifier.strip().split("/")
        if len(parts) != 2 or not all(parts):
            return CompatibilityResult(
                model_id=identifier,
                artifact_id=None,
                runtime=None,
                status=CompatibilityStatus.UNKNOWN,
                confidence=Confidence.UNKNOWN,
                reason="identifier is not a curated model or Hugging Face repository ID",
                unknowns=(
                    "logical model metadata is unknown",
                    "runnable artifact is unknown",
                    "runtime compatibility is unknown",
                    "memory requirement is unknown",
                    "practical context is unknown",
                    "generation performance is not predicted or measured",
                ),
            )
        resolution = HF_SOURCE.resolve(identifier)
        if resolution.status == "resolved" and resolution.model is not None:
            return assess_generic_runtime_compatibility(resolution, profile)
        reason = (
            "Hugging Face repository was not found or is inaccessible"
            if resolution.status == "not_found_or_inaccessible"
            else "Hugging Face metadata request failed"
        )
        return CompatibilityResult(
            model_id=identifier,
            artifact_id=None,
            runtime=None,
            status=CompatibilityStatus.UNKNOWN,
            confidence=Confidence.UNKNOWN,
            reason=reason,
            unknowns=(
                "logical model metadata is unknown",
                "artifact metadata is unknown",
                "runtime compatibility is unknown",
                "memory requirement is unknown",
                "practical context is unknown",
            ),
            resolution_status=resolution.status,
            resolution_confidence=resolution.confidence,
        )
    result = assess_curated_compatibility(model, profile)
    alternatives = []
    for candidate in CURATED_SOURCE.list_specs():
        if candidate.ollama == model.ollama or not candidate.recommendable:
            continue
        if candidate.model.model_id != model.model.model_id:
            continue
        candidate_result = assess_curated_compatibility(candidate, profile)
        if candidate_result.can_run is True:
            alternatives.append(candidate.artifact.artifact_id)
    return replace(result, alternatives=tuple(alternatives))


def compatibility_exit_code(result: CompatibilityResult) -> int:
    """Map the three-state compatibility predicate to its CLI exit status."""
    if result.can_run is True:
        return 0
    if result.can_run is False:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def ollama_generate(
    host: str,
    model: str,
    prompt: str,
    context: int,
    num_predict: int,
    think: bool = False,
    timeout: int = BENCH_REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_ctx": int(context),
            "num_predict": int(num_predict),
            "seed": 42,
        },
    }
    response, _ = http_json(
        f"{host}/api/generate",
        method="POST",
        payload=payload,
        timeout=timeout,
    )
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected response from Ollama")
    return response


def speed_metrics(response: Dict[str, Any]) -> Dict[str, Any]:
    eval_count = safe_float(response.get("eval_count"))
    eval_duration_ns = safe_float(response.get("eval_duration"))
    prompt_count = safe_float(response.get("prompt_eval_count"))
    prompt_duration_ns = safe_float(response.get("prompt_eval_duration"))
    total_duration_ns = safe_float(response.get("total_duration"))
    load_duration_ns = safe_float(response.get("load_duration"))

    generation_tps = (
        eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else None
    )
    prompt_tps = (
        prompt_count / (prompt_duration_ns / 1e9)
        if prompt_duration_ns > 0
        else None
    )

    return {
        "eval_count": int(eval_count),
        "eval_duration_s": round(eval_duration_ns / 1e9, 4),
        "generation_tps": round(generation_tps, 2) if generation_tps is not None else None,
        "prompt_eval_count": int(prompt_count),
        "prompt_eval_duration_s": round(prompt_duration_ns / 1e9, 4),
        "prompt_tps": round(prompt_tps, 2) if prompt_tps is not None else None,
        "load_duration_s": round(load_duration_ns / 1e9, 4),
        "total_duration_s": round(total_duration_ns / 1e9, 4),
    }


def benchmark_markdown(result: Dict[str, Any]) -> str:
    hardware = result["hardware"]
    aggregate = result["aggregate"]
    running = result.get("running_model") or {}
    memory_before = result.get("memory_before") or {}
    memory_after = result.get("memory_after") or {}
    memory_after_unload = result.get("memory_after_unload") or {}

    lines = [
        f"# {PROJECT_NAME} benchmark: `{result['model']}`",
        "",
        f"- Timestamp: `{result['timestamp']}`",
        f"- Tool version: `{result['tool_version']}`",
        f"- OS: `{hardware.get('platform')}`",
        f"- CPU/chip: `{hardware.get('cpu')}`",
        f"- RAM: `{hardware.get('ram_gib')} GiB`",
        f"- Accelerator: `{accelerator_summary(hardware)}`",
        f"- Context allocation: `{result['context']}` tokens",
        f"- Average generation speed: **{aggregate.get('generation_tps_avg')} tokens/sec**",
        f"- Average prompt-eval speed: **{aggregate.get('prompt_tps_avg')} tokens/sec**",
        f"- Smoke tests: **{aggregate.get('smoke_passed')}/{aggregate.get('smoke_total')} passed**",
    ]

    if running.get("accelerator_percent") is not None:
        lines.append(
            f"- Model resident on accelerator: `~{running.get('accelerator_percent')}%`"
        )
    if memory_before.get("swap_used_gib") is not None:
        lines.append(f"- Swap before benchmark: `{memory_before.get('swap_used_gib')} GiB`")
    if memory_after.get("swap_used_gib") is not None:
        lines.append(f"- Swap while model was loaded: `{memory_after.get('swap_used_gib')} GiB`")
    if memory_after_unload.get("swap_used_gib") is not None:
        lines.append(f"- Swap snapshot after unload: `{memory_after_unload.get('swap_used_gib')} GiB`")

    lines += [
        "",
        "## Throughput runs",
        "",
        "| Run | Generation tok/s | Prompt tok/s | Total seconds |",
        "|---:|---:|---:|---:|",
    ]
    for index, run in enumerate(result.get("throughput_runs") or [], start=1):
        lines.append(
            f"| {index} | {run.get('generation_tps')} | {run.get('prompt_tps')} | "
            f"{run.get('total_duration_s')} |"
        )

    lines += [
        "",
        "## Correctness smoke tests",
        "",
        "| Test | Result | Expected | Model response |",
        "|---|---|---|---|",
    ]
    for item in result.get("smoke_tests") or []:
        response = str(item.get("response", "")).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item.get('name')} | {'PASS' if item.get('passed') else 'FAIL'} | "
            f"`{item.get('expected')}` | `{clip(response, 140)}` |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This is a local throughput and smoke-test report. It is not an academic model-quality benchmark. "
        "Memory values are snapshots, not peak-memory measurements.",
        "",
    ]
    return "\n".join(lines)


def memory_snapshot() -> Dict[str, Any]:
    total = get_total_ram_bytes()
    available = get_available_ram_bytes()
    swap = get_swap_used_bytes()
    return {
        "ram_gib": round(gib(total), 1) if total else None,
        "available_ram_gib": round(gib(available), 2) if available else None,
        "swap_used_gib": round(gib(swap), 2) if swap is not None else None,
    }



def shareable_hardware_profile() -> Dict[str, Any]:
    """Return benchmark hardware metadata without user-specific filesystem paths."""
    profile = hardware_profile()
    profile.pop("model_store", None)
    return profile


def shareable_ollama_info() -> Dict[str, Any]:
    """Return benchmark Ollama metadata without local executable paths."""
    info = ollama_cli_info()
    info.pop("path", None)
    return info


def canonical_json(value: Any) -> str:
    """Serialize identity-bearing data independently of display formatting."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def passport_id_for_document(document: Dict[str, Any]) -> str:
    """Hash canonical passport content with the stored passport_id excluded."""
    content = dict(document)
    content.pop("passport_id", None)
    return sha256_identity(content)


def configuration_fingerprint_content(
    model: Dict[str, Any],
    runtime: Dict[str, Any],
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    benchmark_method: Any,
) -> Dict[str, Any]:
    """Exact canonical field set for benchmark-configuration identity."""
    return {
        "model": model,
        "runtime": runtime,
        "hardware": hardware,
        "workload": workload,
        "benchmark_method": benchmark_method,
    }


def passport_hardware(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Select only benchmark-relevant, privacy-safe hardware facts."""
    return {
        "os": profile.get("os") or profile.get("platform"),
        "architecture": profile.get("architecture") or profile.get("arch"),
        "cpu_or_chip": profile.get("cpu_or_chip") or profile.get("cpu"),
        "ram_gib": profile.get("ram_gib"),
        "accelerator": profile.get("accelerator") or accelerator_summary(profile),
    }


def passport_safe_runtime_version(value: Optional[str]) -> Optional[str]:
    """Retain a useful version only when it contains no local identity or secret."""
    text = race_safe_runtime_version(value)
    if not text:
        return None
    private_values = {
        str(os.environ.get("USER") or "").strip(),
        str(os.environ.get("USERNAME") or "").strip(),
        str(platform.node() or "").strip(),
    }
    if any(item and re.search(rf"(?<!\w){re.escape(item)}(?!\w)", text) for item in private_values):
        return None
    if passport_privacy_issues({"runtime_version": text}):
        return None
    return text


def _mean(values: Sequence[Any], digits: int) -> Optional[float]:
    usable = [float(value) for value in values if value is not None]
    return round(sum(usable) / len(usable), digits) if usable else None


def canonical_passport_samples(
    raw_samples: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Preserve actual per-run runtime measurements in their execution order."""
    return [
        {
            "run": index,
            "generation_tps": sample.get("generation_tps"),
            "prompt_evaluation_tps": sample.get("prompt_tps"),
            "latency_s": sample.get("wall_seconds"),
            "runtime_total_duration_s": sample.get("total_duration_s"),
            "generated_tokens": sample.get("eval_count"),
            "generation_duration_s": sample.get("eval_duration_s"),
            "prompt_evaluation_tokens": sample.get("prompt_eval_count"),
            "prompt_evaluation_duration_s": sample.get("prompt_eval_duration_s"),
            "load_duration_s": sample.get("load_duration_s"),
        }
        for index, sample in enumerate(raw_samples, start=1)
    ]


def aggregate_passport_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Canonical passport aggregation: means rounded half-even by Python round."""
    return {
        "generation_tps": _mean([item.get("generation_tps") for item in samples], 2),
        "prompt_evaluation_tps": _mean(
            [item.get("prompt_evaluation_tps") for item in samples], 2
        ),
        "latency_s": _mean([item.get("latency_s") for item in samples], 4),
        "generated_tokens": (
            sum(int(item.get("generated_tokens") or 0) for item in samples)
            if samples
            else None
        ),
    }


def _passport_payload(
    *,
    timestamp: str,
    logical_model_id: str,
    artifact_id: str,
    artifact_fingerprint: Optional[str],
    artifact_format: Optional[str],
    quantization: Optional[str],
    artifact_size_bytes: Optional[int],
    runtime: str,
    runtime_version: Optional[str],
    adapter: str,
    hardware: Dict[str, Any],
    workload: Dict[str, Any],
    raw_samples: Sequence[Dict[str, Any]],
    execution_status: str = "success",
    failure_category: Optional[str] = None,
    warnings: Sequence[str] = (),
    benchmark_method: str = BENCHMARK_METHOD_VERSION,
) -> BenchmarkPassport:
    samples = canonical_passport_samples(raw_samples)
    successful = execution_status == "success"
    measurement = {
        "execution_status": execution_status,
        "measured_run_count": len(samples),
        "raw_samples": samples,
        "aggregates": aggregate_passport_samples(samples)
        if successful
        else {
            "generation_tps": None,
            "prompt_evaluation_tps": None,
            "latency_s": None,
            "generated_tokens": None,
        },
        "warnings": sorted(dict.fromkeys(warnings)),
        "failure_category": failure_category,
    }
    identity = {
        "tool": PROJECT_SLUG,
        "llmrig_version": VERSION,
        "benchmark_method": benchmark_method,
        "timestamp": timestamp,
    }
    model = {
        "logical_model_id": logical_model_id,
        "artifact_or_build": artifact_id,
        "artifact_fingerprint": artifact_fingerprint,
        "artifact_format": artifact_format,
        "quantization": quantization,
        "artifact_size_bytes": artifact_size_bytes,
    }
    runtime_data = {
        "runtime": runtime,
        "runtime_version": passport_safe_runtime_version(runtime_version),
        "llmrig_execution_adapter": adapter,
        "execution_evidence_level": "measured" if successful else "failed",
    }
    config = configuration_fingerprint_content(
        model, runtime_data, hardware, workload, benchmark_method
    )
    fingerprint = sha256_identity(config)
    body = {
        "schema_version": PASSPORT_SCHEMA_VERSION,
        "configuration_fingerprint": fingerprint,
        "identity": identity,
        "model": model,
        "runtime": runtime_data,
        "hardware": hardware,
        "workload": workload,
        "measurement": measurement,
        "evidence": {
            "measurements": "measured by the named local execution adapter",
            "configuration": "deterministic from recorded benchmark inputs",
            "metadata": "verified where present; null means unknown",
            "unknown_fields": sorted(
                f"model.{key}" for key, value in model.items() if value is None
            )
            + sorted(
                f"runtime.{key}" for key, value in runtime_data.items() if value is None
            )
            + sorted(
                f"hardware.{key}" for key, value in hardware.items() if value is None
            ),
            "independent_attestation": False,
        },
        "calibration": {
            "status": "unavailable",
            "reason": "no comparable measured peak-memory observation is available",
        },
    }
    return BenchmarkPassport(passport_id=passport_id_for_document(body), **body)


def passport_from_benchmark_result(result: Dict[str, Any]) -> BenchmarkPassport:
    """Create a passport from the existing Ollama benchmark result structure."""
    spec = resolve_curated_model(str(result.get("model") or ""))
    artifact = spec.artifact if spec else None
    workload = {
        "prompt": SPEED_PROMPT,
        "requested_context": result.get("context"),
        "num_predict": 640,
        "temperature": 0,
        "seed": 42,
        "think": False,
        "warmup_count": 1,
        "warmup_token_cap": 64,
        "measured_run_count": len(result.get("throughput_runs") or []),
        "timeout_s": BENCH_REQUEST_TIMEOUT,
        "other_generation_settings": {},
    }
    ollama = result.get("ollama") or {}
    running = result.get("running_model") or {}
    return _passport_payload(
        timestamp=str(result.get("timestamp") or now_iso()),
        logical_model_id=artifact.model_id if artifact else str(result.get("model")),
        artifact_id=str(result.get("model")),
        artifact_fingerprint=result.get("artifact_fingerprint") or running.get("digest"),
        artifact_format=artifact.format if artifact else None,
        quantization=artifact.quantization if artifact else None,
        artifact_size_bytes=None,
        runtime="ollama",
        runtime_version=ollama.get("version"),
        adapter="ollama-existing-benchmark",
        hardware=passport_hardware(result.get("hardware") or {}),
        workload=workload,
        raw_samples=result.get("throughput_runs") or [],
        warnings=("performance measurements do not establish model quality",),
    )


def passport_from_race_competitor(
    result: RaceResult, competitor: RaceCompetitor
) -> BenchmarkPassport:
    workload = result.workload.to_dict()
    workload.update(
        {
            "requested_context": workload.pop("context"),
            "measured_run_count": workload.pop("runs"),
            "timeout_s": workload.pop("request_timeout_s"),
            "warmup_count": workload.pop("warmup_runs"),
            "warmup_token_cap": workload.pop("warmup_num_predict"),
            "think": False,
            "other_generation_settings": {},
        }
    )
    return _passport_payload(
        timestamp=competitor.timestamp,
        logical_model_id=competitor.logical_model_id,
        artifact_id=competitor.artifact_id,
        artifact_fingerprint=competitor.artifact_fingerprint,
        artifact_format=competitor.artifact_format,
        quantization=competitor.quantization,
        artifact_size_bytes=None,
        runtime=competitor.runtime,
        runtime_version=competitor.runtime_version,
        adapter=f"{competitor.runtime}-race-adapter",
        hardware=passport_hardware(result.hardware),
        workload=workload,
        raw_samples=competitor.raw_samples,
        execution_status=competitor.execution_status,
        failure_category=competitor.failure,
        warnings=competitor.warnings,
        benchmark_method=result.method_version,
    )


def write_passport(passport: BenchmarkPassport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(passport.to_dict(), indent=2) + "\n", encoding="utf-8")


def passport_privacy_issues(value: Any, path: str = "$") -> List[str]:
    issues: List[str] = []
    forbidden_keys = {
        "username", "hostname", "serial_number", "mac_address", "model_store",
        "home_directory", "local_path", "temporary_path", "api_key",
        "authorization", "bearer_token", "environment",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden_keys:
                issues.append(f"privacy-sensitive field: {path}.{key}")
            issues.extend(passport_privacy_issues(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(passport_privacy_issues(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        home = str(Path.home())
        if home and home in value:
            issues.append(f"home path at {path}")
        if re.search(r"(?:^|\s)(?:/Users/[^/]+|/home/[^/]+|/tmp/|/private/(?:tmp|var)/|[A-Za-z]:\\\\Users\\\\)", value):
            issues.append(f"local path at {path}")
        if re.search(r"(?i)(?:bearer\s+|hf_[A-Za-z0-9]{12,}|api[_-]?key\s*[=:])", value):
            issues.append(f"credential-like value at {path}")
    return issues


def validate_passport(document: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required = {
        "schema_version", "passport_id", "configuration_fingerprint", "identity",
        "model", "runtime", "hardware", "workload", "measurement", "evidence", "calibration",
    }
    missing = sorted(required - set(document))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
        return errors
    if document.get("schema_version") != PASSPORT_SCHEMA_VERSION:
        errors.append("unsupported schema version")
    for key in ("passport_id", "configuration_fingerprint"):
        if not isinstance(document.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(document.get(key) or "")
        ):
            errors.append(f"{key} must be a SHA-256 hexadecimal digest")
    for key in ("identity", "model", "runtime", "hardware", "workload", "measurement", "evidence", "calibration"):
        if not isinstance(document.get(key), dict):
            errors.append(f"{key} must be an object")
    if errors:
        return errors
    nested_required = {
        "identity": {"tool", "llmrig_version", "benchmark_method", "timestamp"},
        "model": {"logical_model_id", "artifact_or_build", "artifact_fingerprint", "artifact_format", "quantization", "artifact_size_bytes"},
        "runtime": {"runtime", "runtime_version", "llmrig_execution_adapter", "execution_evidence_level"},
        "hardware": {"os", "architecture", "cpu_or_chip", "ram_gib", "accelerator"},
        "workload": {"prompt", "requested_context", "num_predict", "temperature", "seed", "think", "warmup_count", "warmup_token_cap", "measured_run_count", "timeout_s", "other_generation_settings"},
        "measurement": {"execution_status", "measured_run_count", "raw_samples", "aggregates", "warnings", "failure_category"},
    }
    for section, keys in nested_required.items():
        absent = sorted(keys - set(document[section]))
        if absent:
            errors.append(f"{section} missing required fields: " + ", ".join(absent))
    if errors:
        return errors
    config = configuration_fingerprint_content(
        document["model"], document["runtime"], document["hardware"],
        document["workload"], document["identity"].get("benchmark_method"),
    )
    if document.get("configuration_fingerprint") != sha256_identity(config):
        errors.append("configuration fingerprint mismatch")
    if document.get("passport_id") != passport_id_for_document(document):
        errors.append("passport ID mismatch")
    measurement = document["measurement"]
    samples = measurement.get("raw_samples")
    if not isinstance(samples, list):
        errors.append("measurement.raw_samples must be an array")
        samples = []
    invalid_sample = False
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"measurement sample {index + 1} must be an object")
            invalid_sample = True
            continue
        for key in (
            "generation_tps", "prompt_evaluation_tps", "latency_s",
            "runtime_total_duration_s", "generation_duration_s",
            "prompt_evaluation_duration_s", "load_duration_s",
        ):
            value = sample.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value < 0):
                errors.append(f"measurement sample {index + 1} has invalid {key}")
                invalid_sample = True
        tokens = sample.get("generated_tokens")
        if tokens is not None and (not isinstance(tokens, int) or tokens < 0):
            errors.append(f"measurement sample {index + 1} has invalid generated_tokens")
            invalid_sample = True
        prompt_tokens = sample.get("prompt_evaluation_tokens")
        if prompt_tokens is not None and (not isinstance(prompt_tokens, int) or prompt_tokens < 0):
            errors.append(f"measurement sample {index + 1} has invalid prompt_evaluation_tokens")
            invalid_sample = True
    status = measurement.get("execution_status")
    aggregates = measurement.get("aggregates")
    if not isinstance(aggregates, dict):
        errors.append("measurement.aggregates must be an object")
        aggregates = {}
    if status == "success":
        if not samples or measurement.get("measured_run_count") != len(samples):
            errors.append("successful benchmark requires matching nonzero measured runs")
        if document["workload"].get("measured_run_count") != len(samples):
            errors.append("workload measured-run count does not match raw samples")
        if not invalid_sample:
            expected = aggregate_passport_samples(samples)
            if aggregates != expected:
                errors.append("measurement aggregates do not match raw samples")
    elif status == "failed":
        if any(value is not None for value in aggregates.values()):
            errors.append("failed benchmark cannot contain success aggregates")
    else:
        errors.append("execution_status must be success or failed")
    errors.extend(passport_privacy_issues(document))
    return errors


def compare_passports(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    same_passport = left.get("passport_id") == right.get("passport_id")
    if left.get("configuration_fingerprint") == right.get("configuration_fingerprint"):
        return {"same_passport": same_passport, "classification": "exact", "reasons": []}
    reasons = []
    if left.get("model", {}).get("logical_model_id") != right.get("model", {}).get("logical_model_id"):
        reasons.append("different logical model")
    if left.get("workload") != right.get("workload"):
        reasons.append("different workload")
    if reasons:
        return {"same_passport": same_passport, "classification": "not_comparable", "reasons": reasons}
    for label, section, key in (
        ("artifact", "model", "artifact_or_build"), ("quantization", "model", "quantization"),
        ("artifact format", "model", "artifact_format"),
        ("runtime", "runtime", "runtime"), ("runtime version", "runtime", "runtime_version"),
        ("hardware", "hardware", None),
    ):
        a = left.get(section) if key is None else left.get(section, {}).get(key)
        b = right.get(section) if key is None else right.get(section, {}).get(key)
        if a != b:
            reasons.append(f"different {label}")
    return {"same_passport": same_passport, "classification": "comparable_with_warnings", "reasons": reasons}


def run_benchmark(
    model: str,
    context: int,
    runs: int,
    host: str = DEFAULT_OLLAMA_HOST,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if not ensure_ollama_service(host):
        raise RuntimeError("Ollama service is not available.")

    print(f"\nBenchmarking {model}")
    print(f"Context allocation: {context:,} tokens")
    print("Thinking disabled for throughput comparability.")
    print("Isolating benchmark memory by unloading currently resident Ollama models...")
    isolated_models = isolate_ollama_for_benchmark(host)

    before = memory_snapshot()
    timed: List[Dict[str, Any]] = []
    smoke: List[Dict[str, Any]] = []
    running: Dict[str, Any] = {}
    loaded_after: Dict[str, Any] = {}

    try:
        print("Running warm-up request...")
        ollama_generate(
            host,
            model,
            "In one sentence, define a transformer neural network.",
            context,
            64,
            think=False,
            timeout=BENCH_REQUEST_TIMEOUT,
        )

        for index in range(max(1, runs)):
            print(f"Throughput run {index + 1}/{max(1, runs)}...")
            started = time.perf_counter()
            response = ollama_generate(
                host,
                model,
                SPEED_PROMPT,
                context,
                640,
                think=False,
                timeout=BENCH_REQUEST_TIMEOUT,
            )
            metrics = speed_metrics(response)
            metrics["wall_seconds"] = round(time.perf_counter() - started, 4)
            timed.append(metrics)

        print("Running lightweight correctness smoke tests...")
        for name, prompt, expected, check in SMOKE_TESTS:
            response = ollama_generate(
                host,
                model,
                prompt,
                min(context, 32_768),
                48,
                think=False,
                timeout=BENCH_REQUEST_TIMEOUT,
            )
            text = str(response.get("response") or "").strip()
            passed = bool(check(text))
            smoke.append(
                {
                    "name": name,
                    "passed": passed,
                    "expected": expected,
                    "response": text,
                }
            )
            print(f"  {'PASS' if passed else 'FAIL'}  {name}: {clip(text, 80)}")

        running = running_model_details(host, model)
        loaded_after = memory_snapshot()
    finally:
        # A benchmark must not leave its model resident when another model may be
        # benchmarked next. Ollama keep_alive=0 is the documented immediate-unload path.
        unload_ollama_model(host, model)

    after_unload = memory_snapshot()

    generation_values = [
        item["generation_tps"] for item in timed if item.get("generation_tps") is not None
    ]
    prompt_values = [
        item["prompt_tps"] for item in timed if item.get("prompt_tps") is not None
    ]
    total_values = [
        item["total_duration_s"] for item in timed if item.get("total_duration_s") is not None
    ]

    aggregate = {
        "generation_tps_avg": (
            round(sum(generation_values) / len(generation_values), 2)
            if generation_values
            else None
        ),
        "generation_tps_min": round(min(generation_values), 2) if generation_values else None,
        "generation_tps_max": round(max(generation_values), 2) if generation_values else None,
        "prompt_tps_avg": (
            round(sum(prompt_values) / len(prompt_values), 2) if prompt_values else None
        ),
        "total_duration_avg_s": (
            round(sum(total_values) / len(total_values), 3) if total_values else None
        ),
        "smoke_passed": sum(1 for item in smoke if item["passed"]),
        "smoke_total": len(smoke),
    }

    result = {
        "tool": PROJECT_SLUG,
        "tool_version": VERSION,
        "timestamp": now_iso(),
        "model": model,
        "context": context,
        "runs": runs,
        "hardware": shareable_hardware_profile(),
        "ollama": shareable_ollama_info(),
        "running_model": running,
        "memory_before": before,
        # Kept for backwards compatibility: this is the snapshot while the model
        # is still loaded at the end of the benchmark workload.
        "memory_after": loaded_after,
        "memory_after_unload": after_unload,
        "isolated_models_before_benchmark": isolated_models,
        "aggregate": aggregate,
        "throughput_runs": timed,
        "smoke_tests": smoke,
        "notes": [
            "Local throughput/smoke benchmark, not an academic model-quality benchmark.",
            "Thinking disabled for throughput and smoke tests for comparability.",
            "Benchmark isolation unloads resident Ollama models before the run and unloads the target afterward.",
            "Memory values are snapshots, not peak-memory measurements.",
            "macOS may retain allocated swap space after memory pressure has ended, so swap usage is not expected to return immediately to zero.",
            "Longer context allocations can increase memory use and change performance.",
        ],
    }

    output = output_dir or Path.cwd() / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    json_path = output / f"{safe_name}_{stamp}.json"
    markdown_path = output / f"{safe_name}_{stamp}.md"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path.write_text(benchmark_markdown(result), encoding="utf-8")
    result["files"] = {"json": str(json_path), "markdown": str(markdown_path)}

    print("\nBenchmark result")
    print("----------------")
    print(f"Generation: {aggregate['generation_tps_avg']} tokens/sec average")
    print(f"Prompt eval: {aggregate['prompt_tps_avg']} tokens/sec average")
    print(
        f"Correctness smoke tests: {aggregate['smoke_passed']}/{aggregate['smoke_total']} passed"
    )
    if running.get("accelerator_percent") is not None:
        print(f"Model resident on accelerator during benchmark: ~{running['accelerator_percent']}%")
    if running.get("context_length"):
        print(f"Ollama-reported context: {running['context_length']:,}")
    if loaded_after.get("swap_used_gib") is not None:
        print(f"Swap while model was loaded: {loaded_after['swap_used_gib']} GiB")
    if after_unload.get("swap_used_gib") is not None:
        print(f"Swap snapshot after unload: {after_unload['swap_used_gib']} GiB")
    print("Model unloaded after benchmark: yes")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return result


class OllamaExecutionAdapter:
    """Measured Ollama execution using existing bounded generation primitives."""

    runtime = "ollama"

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.host = host

    def benchmark(
        self, target: ExecutionTarget, workload: RaceWorkload
    ) -> RaceCompetitor:
        configuration = target.configuration if isinstance(target, ExecutionTarget) else target
        if not OLLAMA_RUNTIME.is_available(self.host):
            raise RuntimeError("Ollama service is unavailable")
        isolate_ollama_for_benchmark(self.host)
        timed = []
        try:
            for _ in range(workload.warmup_runs):
                ollama_generate(
                    self.host,
                    configuration.artifact_id,
                    workload.prompt,
                    workload.context,
                    min(workload.warmup_num_predict, workload.num_predict),
                    think=False,
                    timeout=workload.request_timeout_s,
                )
            for _ in range(workload.runs):
                started = time.perf_counter()
                response = ollama_generate(
                    self.host,
                    configuration.artifact_id,
                    workload.prompt,
                    workload.context,
                    workload.num_predict,
                    think=False,
                    timeout=workload.request_timeout_s,
                )
                metrics = speed_metrics(response)
                if metrics.get("generation_tps") is None or not metrics.get("eval_count"):
                    raise RuntimeError("Ollama response did not contain valid generation metrics")
                metrics["process_wall_seconds"] = round(time.perf_counter() - started, 4)
                prompt_duration = metrics.get("prompt_eval_duration_s")
                generation_duration = metrics.get("eval_duration_s")
                metrics["wall_seconds"] = (
                    round(prompt_duration + generation_duration, 4)
                    if prompt_duration is not None and generation_duration is not None
                    else None
                )
                timed.append(metrics)
        finally:
            unload_ollama_model(self.host, configuration.artifact_id)

        if not timed:
            raise RuntimeError("no measured benchmark runs completed")
        generation = [
            item["generation_tps"]
            for item in timed
            if item.get("generation_tps") is not None
        ]
        prompt = [
            item["prompt_tps"] for item in timed if item.get("prompt_tps") is not None
        ]
        latency = [item["wall_seconds"] for item in timed if item.get("wall_seconds") is not None]
        generated_tokens = sum(int(item.get("eval_count") or 0) for item in timed)
        evidence = (
            RecommendationEvidence(
                "measured",
                f"Ollama {configuration.runtime_version or 'version unknown'} local execution",
                f"{workload.runs} timed run(s) completed with deterministic generation settings",
            ),
        )
        return RaceCompetitor(
            logical_model_id=configuration.logical_model_id,
            runtime=configuration.runtime,
            artifact_id=configuration.artifact_id,
            artifact_fingerprint=configuration.artifact_fingerprint,
            artifact_format=configuration.artifact_format,
            quantization=configuration.quantization,
            runtime_version=configuration.runtime_version,
            execution_status="success",
            generation_tps=(
                round(sum(generation) / len(generation), 2) if generation else None
            ),
            prompt_eval_tps=round(sum(prompt) / len(prompt), 2) if prompt else None,
            total_latency_s=round(sum(latency) / len(latency), 4) if latency else None,
            generated_tokens=generated_tokens,
            measured_runs=len(timed),
            generation_samples=len(generation),
            prompt_eval_samples=len(prompt),
            latency_samples=len(latency),
            timestamp=now_iso(),
            evidence=evidence,
            warnings=("performance measurement does not establish model quality",),
            raw_samples=tuple(dict(item) for item in timed),
        )


_METRIC_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?"
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_LLAMA_PREFIX = r"(?:llama_perf_context_print|llama_print_timings)"


def _positive_metric(value: str) -> float:
    number = float(value.replace(",", ""))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("metric must be positive")
    return number


def parse_llama_cpp_metrics(output: str) -> Dict[str, Any]:
    """Parse exactly one complete, explicitly prefixed llama.cpp timing block."""
    clean = _ANSI_ESCAPE.sub("", output).replace("\r", "")
    header = re.compile(
        rf"(?im)^\s*(?P<source>{_LLAMA_PREFIX}):\s*(?P<kind>prompt eval|eval) time\s*=",
    )
    headers = header.findall(clean)
    if sum(kind.lower().startswith("prompt") for _, kind in headers) != 1 or sum(
        kind.lower() == "eval" for _, kind in headers
    ) != 1:
        raise ValueError("llama.cpp timing output is incomplete or ambiguous")
    if len({source.lower() for source, _ in headers}) != 1:
        raise ValueError("llama.cpp timing output mixes incompatible timing prefixes")
    pattern = re.compile(
        rf"(?im)^\s*(?P<source>{_LLAMA_PREFIX}):\s*(?P<kind>prompt eval|eval) time\s*=\s*"
        rf"(?P<ms>{_METRIC_NUMBER})\s*ms\s*/\s*(?P<tokens>\d+)\s+"
        rf"(?:tokens?|runs?)\s*\(\s*(?:(?:{_METRIC_NUMBER})\s*ms\s+per\s+token\s*,\s*)?"
        rf"(?P<tps>{_METRIC_NUMBER})\s+tokens?\s+per\s+second\s*\)",
    )
    matches = list(pattern.finditer(clean))
    if len(matches) != 2:
        raise ValueError("llama.cpp output did not contain required measured timing metrics")
    values: Dict[str, Any] = {}
    for match in matches:
        prefix = "prompt_eval" if match.group("kind").lower().startswith("prompt") else "eval"
        count = int(match.group("tokens"))
        if count <= 0:
            raise ValueError("metric count must be positive")
        values[f"{prefix}_count"] = count
        values[f"{prefix}_duration_s"] = round(_positive_metric(match.group("ms")) / 1000, 6)
        values["prompt_tps" if prefix == "prompt_eval" else "generation_tps"] = _positive_metric(match.group("tps"))
    values["wall_seconds"] = round(values["prompt_eval_duration_s"] + values["eval_duration_s"], 4)
    values["total_duration_s"] = values["wall_seconds"]
    return values


def parse_mlx_metrics(output: str) -> Dict[str, Any]:
    """Parse one MLX-LM metric pair after the final framework separator."""
    lines = output.replace("\r", "").splitlines()
    separators = [index for index, line in enumerate(lines) if line.strip() == "=========="]
    if not separators:
        raise ValueError("MLX-LM output did not contain the framework metric separator")
    suffix = "\n".join(lines[separators[-1] + 1 :])
    pattern = re.compile(
        rf"^\s*(Prompt|Generation)\s*:\s*(\d+)\s+tokens?\s*,\s*"
        rf"({_METRIC_NUMBER})\s+tokens?(?:-|\s+)per(?:-|\s+)sec\s*$",
        re.I | re.M,
    )
    matches = pattern.findall(suffix)
    if sum(kind.lower() == "prompt" for kind, _, _ in matches) != 1 or sum(
        kind.lower() == "generation" for kind, _, _ in matches
    ) != 1:
        raise ValueError("MLX-LM output metrics are incomplete or ambiguous")
    values: Dict[str, Any] = {}
    for kind, count, rate in matches:
        tokens = int(count)
        tps = _positive_metric(rate)
        if tokens <= 0:
            raise ValueError("metric count must be positive")
        prefix = "prompt_eval" if kind.lower() == "prompt" else "eval"
        values[f"{prefix}_count"] = tokens
        values[f"{prefix}_duration_s"] = round(tokens / tps, 6)
        values["prompt_tps" if prefix == "prompt_eval" else "generation_tps"] = tps
    values["wall_seconds"] = round(values["prompt_eval_duration_s"] + values["eval_duration_s"], 4)
    values["total_duration_s"] = values["wall_seconds"]
    return values


def native_race_competitor(
    configuration: RaceConfiguration,
    samples: Sequence[Dict[str, Any]],
    workload: RaceWorkload,
) -> RaceCompetitor:
    generation = [float(item["generation_tps"]) for item in samples]
    prompt = [float(item["prompt_tps"]) for item in samples]
    latency = [float(item["wall_seconds"]) for item in samples]
    return RaceCompetitor(
        configuration.logical_model_id, configuration.runtime, configuration.artifact_id,
        configuration.artifact_fingerprint, configuration.artifact_format,
        configuration.quantization, configuration.runtime_version, "success",
        round(sum(generation) / len(generation), 2),
        round(sum(prompt) / len(prompt), 2),
        round(sum(latency) / len(latency), 4),
        sum(int(item["eval_count"]) for item in samples), len(samples), len(samples),
        len(samples), len(samples), now_iso(),
        evidence=(RecommendationEvidence(
            "measured", f"{configuration.runtime} local execution",
            f"{workload.runs} timed run(s) produced runtime-reported inference metrics",
        ),),
        warnings=("performance measurement does not establish model quality",),
        raw_samples=tuple(dict(item) for item in samples),
    )


class LlamaCppExecutionAdapter:
    runtime = "llama.cpp"

    def benchmark(self, target: ExecutionTarget, workload: RaceWorkload) -> RaceCompetitor:
        if not isinstance(target, ExecutionTarget):
            raise RuntimeError("llama.cpp execution requires a private local target")
        executable = llama_cpp_executable()
        if not executable or not LLAMA_CPP_RUNTIME.is_available():
            raise RuntimeError("llama.cpp is unavailable")
        samples = []
        for index in range(workload.warmup_runs + workload.runs):
            tokens = min(workload.warmup_num_predict, workload.num_predict) if index < workload.warmup_runs else workload.num_predict
            command = [executable, "--model", target._locator, "--prompt", workload.prompt,
                       "--ctx-size", str(workload.context), "--n-predict", str(tokens),
                       "--temp", str(workload.temperature), "--seed", str(workload.seed)]
            process = run_cmd(command, timeout=workload.request_timeout_s)
            if process.returncode != 0:
                raise RuntimeError("llama.cpp local execution failed")
            try:
                metrics = parse_llama_cpp_metrics(process.stderr or "")
            except ValueError:
                raise RuntimeError("llama.cpp timing metrics are unavailable or ambiguous")
            if index >= workload.warmup_runs:
                samples.append(metrics)
        return native_race_competitor(target.configuration, samples, workload)


class MlxExecutionAdapter:
    runtime = "mlx-lm"

    def benchmark(self, target: ExecutionTarget, workload: RaceWorkload) -> RaceCompetitor:
        if not isinstance(target, ExecutionTarget):
            raise RuntimeError("MLX-LM execution requires a private local target")
        executable = shutil.which("mlx_lm.generate")
        if not executable or not MLX_RUNTIME.is_available():
            raise RuntimeError("MLX-LM is unavailable")
        samples = []
        for index in range(workload.warmup_runs + workload.runs):
            tokens = min(workload.warmup_num_predict, workload.num_predict) if index < workload.warmup_runs else workload.num_predict
            command = [executable, "--model", target._locator, "--prompt", workload.prompt,
                       "--max-tokens", str(tokens), "--temp", str(workload.temperature),
                       "--seed", str(workload.seed), "--max-kv-size", str(workload.context)]
            process = run_cmd(command, timeout=workload.request_timeout_s)
            if process.returncode != 0:
                raise RuntimeError("MLX-LM local execution failed")
            metrics = parse_mlx_metrics(process.stdout or "")
            if index >= workload.warmup_runs:
                samples.append(metrics)
        return native_race_competitor(target.configuration, samples, workload)


def race_hardware_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal privacy-safe hardware facts relevant to a local comparison."""
    return {
        "os": profile.get("os"),
        "arch": profile.get("arch"),
        "cpu": profile.get("cpu"),
        "ram_gib": profile.get("ram_gib"),
        "accelerator": accelerator_summary(profile),
    }


def race_safe_runtime_version(value: Optional[str]) -> Optional[str]:
    """Keep useful version text while excluding anything path-like from race output."""
    if not value:
        return None
    text = " ".join(value.strip().splitlines())[:200]
    if str(Path.home()) in text or re.search(r"(?:^|\s)(?:[/~]|[A-Za-z]:\\)", text):
        return None
    return text


def parse_local_artifact_values(values: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    """Parse repeatable runtime=locator inputs without returning locators in errors."""
    parsed = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError("local artifact must use runtime=local-path syntax")
        runtime, locator = value.split("=", 1)
        runtime = runtime.strip().lower()
        if runtime not in {"llama.cpp", "mlx-lm"}:
            if runtime == "ollama":
                raise ValueError("explicit Ollama local artifacts are not supported; use existing local Ollama discovery")
            raise ValueError("the supplied local artifact uses an unsupported runtime")
        if not locator or not locator.strip():
            raise ValueError("local artifact must use runtime=local-path syntax")
        if runtime in seen:
            raise ValueError("only one explicit local artifact per native runtime is supported")
        seen.add(runtime)
        parsed.append((runtime, locator))
    return tuple(parsed)


def local_artifact_id(runtime: str) -> str:
    """Return a path-independent ID for the race's single user-supplied target."""
    return f"local:{runtime}:user-supplied"


def validated_local_locator(runtime: str, supplied: str) -> str:
    """Validate structural local evidence and return only private normalized state."""
    path = Path(supplied).expanduser()
    try:
        if runtime == "llama.cpp":
            valid = (
                path.is_file()
                and path.suffix.lower() == ".gguf"
                and path.stat().st_size > 0
            )
        else:
            config = path / "config.json"
            weights = (
                item
                for item in path.iterdir()
                if item.is_file()
                and item.name.startswith("model")
                and item.name.endswith(".safetensors")
                and item.stat().st_size > 0
            ) if path.is_dir() else ()
            valid = path.is_dir() and config.is_file() and config.stat().st_size > 0 and any(weights)
        if valid:
            return str(path.resolve(strict=True))
    except (OSError, RuntimeError):
        pass
    kind = (
        "non-empty local GGUF file"
        if runtime == "llama.cpp"
        else "local MLX directory with config.json and local weights"
    )
    raise ValueError(f"the supplied {runtime} local artifact is not a {kind}")


def local_execution_targets(
    identifier: str,
    values: Sequence[str],
    profile: Dict[str, Any],
    adapters: Sequence[ExecutionAdapter],
    compatibility: Optional[CompatibilityResult] = None,
) -> Tuple[ExecutionTarget, ...]:
    """Validate explicit native artifacts and create private invocation targets."""
    capabilities = {item.runtime: item for item in runtime_capabilities(profile)}
    adapter_names = {adapter.runtime for adapter in adapters}
    compatibility = compatibility or compatibility_for_identifier(identifier, profile)
    logical_id = compatibility.model_id
    targets = []
    for runtime, supplied in parse_local_artifact_values(values):
        locator = validated_local_locator(runtime, supplied)
        capability = capabilities.get(runtime)
        blockers = []
        if capability is None:
            blockers.append("runtime capability is unknown")
        else:
            if not capability.installed:
                blockers.append("runtime is not installed")
            if not capability.available:
                blockers.append("runtime is not currently available")
            if not capability.llmrig_execution_supported:
                blockers.append("LLMRig has no execution adapter for this runtime")
            if not capability.llmrig_benchmark_supported:
                blockers.append("LLMRig has no benchmark adapter for this runtime")
        if runtime not in adapter_names:
            blockers.append("LLMRig execution adapter is unavailable")
        if compatibility.can_run is False:
            blockers.append(compatibility.reason or "static machine compatibility is incompatible")
        elif compatibility.can_run is not True:
            blockers.append("static machine compatibility is unresolved")
        evidence = (
            RecommendationEvidence(
                "user-supplied-local-association", "race command input",
                "the user associated this private local execution target with the requested logical model; model identity was not independently attested",
            ),
            RecommendationEvidence(
                "estimated", "privacy-safe local target identity",
                "the public local target ID denotes the race's user-supplied target slot and is not a model-content fingerprint or artifact attestation",
            ),
        )
        configuration = RaceConfiguration(
            logical_id, runtime, local_artifact_id(runtime),
            "GGUF" if runtime == "llama.cpp" else "MLX", None,
            race_safe_runtime_version(capability.version) if capability else None,
            not blockers, None, tuple(dict.fromkeys(blockers)), evidence,
        )
        targets.append(ExecutionTarget(configuration, locator))
    return tuple(sorted(targets, key=lambda item: (item.configuration.runtime, item.configuration.artifact_id)))


def race_configurations(
    identifier: str,
    profile: Dict[str, Any],
    adapters: Sequence[ExecutionAdapter],
    compatibility: Optional[CompatibilityResult] = None,
) -> Tuple[str, Tuple[RaceConfiguration, ...], Tuple[RaceConfiguration, ...], Optional[str]]:
    """Resolve deterministic local eligibility without pulling or downloading anything."""
    adapter_names = {adapter.runtime for adapter in adapters}
    capabilities = {item.runtime: item for item in runtime_capabilities(profile)}
    eligible = []
    ineligible = []
    curated = CURATED_SOURCE.resolve(identifier)
    if curated is not None:
        logical_model_id = curated.model.model_id
        installed_items = tuple(installed_ollama_models())
        installed_names = tuple(item["name"] for item in installed_items)
        for spec in CURATED_SOURCE.list_specs():
            if spec.model.model_id != logical_model_id:
                continue
            artifact = spec.artifact
            capability = capabilities.get(artifact.runtime or "")
            installed_id = installed_id_for_spec(spec, installed_names)
            installed_fingerprint = next(
                (
                    item.get("id") or None
                    for item in installed_items
                    if installed_id and item.get("name") == installed_id
                ),
                None,
            )
            blockers = []
            if capability is None:
                blockers.append("runtime capability is unknown")
            else:
                if not capability.installed:
                    blockers.append("runtime is not installed")
                if not capability.available:
                    blockers.append("runtime is not currently available")
                if not capability.llmrig_execution_supported:
                    blockers.append("LLMRig has no execution adapter for this runtime")
                if not capability.llmrig_benchmark_supported:
                    blockers.append("LLMRig has no benchmark adapter for this runtime")
            if artifact.runtime not in adapter_names:
                blockers.append("LLMRig execution adapter is unavailable")
            if installed_id is None:
                blockers.append("artifact is not installed locally")
            static = assess_curated_compatibility(spec, profile)
            if static.can_run is not True:
                blockers.append(static.reason or "static machine compatibility is unresolved")
            configuration = RaceConfiguration(
                logical_model_id=logical_model_id,
                runtime=artifact.runtime or "unknown",
                artifact_id=installed_id or artifact.artifact_id,
                artifact_format=artifact.format,
                quantization=artifact.quantization,
                runtime_version=(
                    race_safe_runtime_version(capability.version) if capability else None
                ),
                eligible=not blockers,
                artifact_fingerprint=installed_fingerprint,
                blockers=tuple(dict.fromkeys(blockers)),
                evidence=static.evidence,
            )
            (eligible if configuration.eligible else ineligible).append(configuration)
        reason = None
    else:
        compatibility = compatibility or compatibility_for_identifier(identifier, profile)
        logical_model_id = compatibility.model_id
        artifacts = {artifact.artifact_id: artifact for artifact in compatibility.artifacts}
        for candidate in compatibility.runtime_candidates:
            artifact = artifacts[candidate.artifact_id]
            capability = capabilities.get(candidate.runtime)
            blockers = list(candidate.blockers)
            if candidate.runtime not in adapter_names:
                blockers.append("LLMRig execution adapter is unavailable")
            if capability is None or not capability.llmrig_execution_supported:
                blockers.append("LLMRig execution is not implemented for this runtime")
            blockers.append("artifact is not available through a local execution adapter")
            ineligible.append(
                RaceConfiguration(
                    logical_model_id=logical_model_id,
                    runtime=candidate.runtime,
                    artifact_id=candidate.artifact_id,
                    artifact_format=artifact.format,
                    quantization=artifact.quantization,
                    runtime_version=(
                        race_safe_runtime_version(capability.version)
                        if capability
                        else None
                    ),
                    eligible=False,
                    blockers=tuple(dict.fromkeys(blockers)),
                    evidence=candidate.evidence,
                )
            )
        if compatibility.artifacts and not compatibility.runtime_candidates:
            for artifact in compatibility.artifacts:
                ineligible.append(
                    RaceConfiguration(
                        logical_model_id=logical_model_id,
                        runtime="unknown",
                        artifact_id=artifact.artifact_id,
                        artifact_format=artifact.format,
                        quantization=artifact.quantization,
                        runtime_version=None,
                        eligible=False,
                        blockers=("no LLMRig execution adapter supports this artifact",),
                        evidence=artifact.evidence,
                    )
                )
        reason = compatibility.reason if compatibility.resolution_status != "resolved" else None

    return (
        logical_model_id,
        unique_race_configurations(eligible),
        unique_race_configurations(ineligible),
        reason,
    )


def unique_race_configurations(
    configurations: Sequence[RaceConfiguration],
) -> Tuple[RaceConfiguration, ...]:
    """Deterministically collapse aliases for the same actual execution identity."""
    unique: Dict[Tuple[str, str, str, str], RaceConfiguration] = {}
    for configuration in sorted(
        configurations, key=lambda item: (item.runtime, item.artifact_id)
    ):
        unique.setdefault(configuration.identity, configuration)
    return tuple(sorted(unique.values(), key=lambda item: (item.runtime, item.artifact_id)))


def metric_winner(
    competitors: Sequence[RaceCompetitor],
    attribute: str,
    higher_is_better: bool,
) -> Dict[str, Any]:
    measured = [item for item in competitors if getattr(item, attribute) is not None]
    if len(measured) < 2:
        return {"status": "inconclusive", "reason": "fewer than two measured values"}
    sample_attribute = {
        "generation_tps": "generation_samples",
        "prompt_eval_tps": "prompt_eval_samples",
        "total_latency_s": "latency_samples",
    }[attribute]
    if any(getattr(item, sample_attribute) < 2 for item in measured):
        return {
            "status": "inconclusive",
            "reason": "fewer than two valid samples per competitor",
        }
    ordered = sorted(
        measured,
        key=lambda item: (
            -float(getattr(item, attribute))
            if higher_is_better
            else float(getattr(item, attribute)),
            item.runtime,
            item.artifact_id,
        ),
    )
    best, second = ordered[0], ordered[1]
    best_value = float(getattr(best, attribute))
    second_value = float(getattr(second, attribute))
    scale = max(abs(best_value), abs(second_value), 1e-12)
    if abs(best_value - second_value) / scale <= RACE_NOISE_THRESHOLD:
        return {
            "status": "inconclusive",
            "reason": "leading results are within the 5% noise threshold",
        }
    return {
        "status": "winner",
        "runtime": best.runtime,
        "artifact_id": best.artifact_id,
        "value": getattr(best, attribute),
    }


def execute_race(
    logical_model_id: str,
    eligible: Sequence[RaceConfiguration],
    ineligible: Sequence[RaceConfiguration],
    adapters: Sequence[ExecutionAdapter],
    workload: RaceWorkload,
    hardware: Dict[str, Any],
    resolution_reason: Optional[str] = None,
    timestamp: Optional[str] = None,
    execution_targets: Sequence[ExecutionTarget] = (),
) -> RaceResult:
    """Measure eligible configurations or return a structured unavailable result."""
    ordered_eligible = unique_race_configurations(eligible)
    ordered_ineligible = unique_race_configurations(ineligible)
    started_at = timestamp or now_iso()
    if len(ordered_eligible) < 2:
        return RaceResult(
            "unavailable",
            logical_model_id,
            resolution_reason
            or f"{len(ordered_eligible)} executable configuration(s) found; at least 2 are required",
            RACE_METHOD_VERSION,
            started_at,
            workload,
            hardware,
            ordered_eligible,
            ordered_ineligible,
        )

    adapter_by_runtime = {adapter.runtime: adapter for adapter in adapters}
    target_by_identity = {target.configuration.identity: target for target in execution_targets}
    competitors = []
    for configuration in ordered_eligible:
        try:
            adapter = adapter_by_runtime[configuration.runtime]
            target = target_by_identity.get(configuration.identity, configuration)
            competitors.append(adapter.benchmark(target, workload))
        except subprocess.TimeoutExpired:
            competitors.append(failed_race_competitor(configuration, "benchmark timed out"))
        except Exception:
            competitors.append(failed_race_competitor(configuration, "benchmark execution failed"))
    ordered_competitors = tuple(
        sorted(competitors, key=lambda item: (item.runtime, item.artifact_id))
    )
    failures = [item for item in ordered_competitors if item.execution_status != "success"]
    known_quantizations = {
        item.quantization for item in ordered_competitors if item.quantization is not None
    }
    formats = {item.artifact_format for item in ordered_competitors}
    warnings = []
    if any(item.quantization is None for item in ordered_competitors):
        warnings.append("one or more competitor quantizations are unknown; artifact equivalence cannot be established")
    if len(known_quantizations) > 1:
        warnings.append("competitors use different quantizations; speed results are not artifact-equivalent")
    if len(formats) > 1:
        warnings.append("competitors use different artifact formats; results compare executable configurations, not identical artifacts")
    if any(
        evidence.kind == "user-supplied-local-association"
        for configuration in ordered_eligible for evidence in configuration.evidence
    ):
        warnings.append("local native artifact identity is user-associated with the logical model and is not independently attested")
    prompt_counts = {
        int(sample["prompt_eval_count"])
        for item in ordered_competitors for sample in item.raw_samples
        if sample.get("prompt_eval_count") is not None
    }
    if len(prompt_counts) > 1:
        warnings.append("prompt token counts differ across runtimes; prompt-evaluation throughput is not tokenization-equivalent")
    generated_counts = [
        int(sample["eval_count"])
        for item in ordered_competitors for sample in item.raw_samples
        if sample.get("eval_count") is not None
    ]
    if generated_counts and min(generated_counts) < max(generated_counts) * 0.95:
        warnings.append("generated token counts materially differ or early EOS occurred; throughput compares observed execution, not identical token sequences")
    if failures:
        return RaceResult(
            "failed",
            logical_model_id,
            "one or more benchmark executions failed; comparison is invalid",
            RACE_METHOD_VERSION,
            started_at,
            workload,
            hardware,
            ordered_eligible,
            ordered_ineligible,
            ordered_competitors,
            tuple(warnings),
        )
    winners = (
        ("fastest_generation", metric_winner(ordered_competitors, "generation_tps", True)),
        ("fastest_prompt_evaluation", metric_winner(ordered_competitors, "prompt_eval_tps", True)),
        ("lowest_latency", metric_winner(ordered_competitors, "total_latency_s", False)),
    )
    return RaceResult(
        "completed",
        logical_model_id,
        None,
        RACE_METHOD_VERSION,
        started_at,
        workload,
        hardware,
        ordered_eligible,
        ordered_ineligible,
        ordered_competitors,
        tuple(warnings),
        winners,
    )


def failed_race_competitor(
    configuration: RaceConfiguration, failure: str
) -> RaceCompetitor:
    return RaceCompetitor(
        configuration.logical_model_id,
        configuration.runtime,
        configuration.artifact_id,
        configuration.artifact_fingerprint,
        configuration.artifact_format,
        configuration.quantization,
        configuration.runtime_version,
        "failed",
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
        now_iso(),
        failure=failure,
    )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def print_doctor(profile: Dict[str, Any], json_mode: bool = False) -> None:
    ollama = OLLAMA_RUNTIME.info()
    raw_model_store = profile.get("model_store")
    model_store = privacy_safe_path(Path(str(raw_model_store))) if raw_model_store else ""

    if json_mode:
        payload = dict(profile)
        payload["model_store"] = model_store
        payload["ollama"] = ollama
        payload["ollama_api_alive"] = OLLAMA_RUNTIME.is_available()
        payload["viability"] = viability_label(profile)
        payload["model_weight_budget_gb"] = round(model_budget_gb(profile), 1)
        print(json.dumps(payload, indent=2))
        return

    print(f"\n{PROJECT_NAME} - system doctor")
    print("-" * (len(PROJECT_NAME) + 16))
    print(f"OS:          {profile.get('platform')}")
    print(f"CPU/chip:    {profile.get('cpu')}")
    print(f"RAM:         {profile.get('ram_gib')} GiB")
    if profile.get("available_ram_gib") is not None:
        print(f"RAM avail.:  ~{profile.get('available_ram_gib')} GiB now")
    if profile.get("swap_used_gib") is not None:
        print(f"Swap used:   {profile.get('swap_used_gib')} GiB")
    print(f"Accelerator: {accelerator_summary(profile)}")
    print(f"Model store: {model_store}")
    print(f"Free disk:   {profile.get('disk', {}).get('free_gib')} GiB")
    print(
        f"Ollama:      {ollama.get('version') if ollama.get('installed') else 'not installed'}"
    )
    print(f"Ollama API:  {'ready' if OLLAMA_RUNTIME.is_available() else 'not responding'}")
    print(f"\nAssessment:  {viability_label(profile)}")
    print(
        "Planning model-weight budget: "
        f"~{model_budget_gb(profile):.1f} GB "
        "(conservative headroom target)"
    )

    print("\nRecommended local models")
    for category, label in (
        (OFFICIAL, "Official/standard"),
        (REDUCED_REFUSAL, "Community reduced-refusal"),
    ):
        model = recommend_model(profile, category, "balanced")
        if model:
            print(
                f"- {label}: {model.ollama} "
                f"({model.size_gb:.0f} GB, {model.quant}, "
                f"start at {recommended_context(profile, model) // 1024}K context)"
            )
        else:
            print(f"- {label}: no curated model fits the conservative budget.")


def command_models(args: argparse.Namespace) -> int:
    profile = hardware_profile() if args.fit else None

    print("\nCurated local-ready Qwen models")
    print("===============================")
    print(f"Snapshot date: {CURATED_SNAPSHOT_DATE}")
    print("OFFICIAL = upstream Qwen/Ollama model")
    print(
        "REDUCED-REFUSAL = third-party Qwen derivative often described by its author as "
        "uncensored/abliterated"
    )
    print_curated(profile)

    if args.offline:
        return 0

    print("\nRefreshing live Hugging Face catalog...")
    catalog = fetch_live_catalog(force=args.refresh)
    if catalog.get("warning"):
        print(f"Warning: {catalog['warning']}")
    if catalog.get("fetched_at"):
        print(f"Catalog fetched: {catalog['fetched_at']}")

    official = catalog.get("official") or []
    community = catalog.get("community_reduced_refusal") or []

    official_candidates = [item for item in official if is_live_llm_candidate(item)]
    community_candidates = [item for item in community if is_live_llm_candidate(item)]

    if args.all:
        official_view = official
        community_view = community
        official_count = len(official)
        community_count = len(community)
        official_label = "Official Qwen Hugging Face repositories"
        community_label = "Best-effort community reduced-refusal repositories"
    else:
        official_view = official_candidates[: args.latest]
        community_view = community_candidates[: min(args.latest, 40)]
        official_count = len(official_candidates)
        community_count = len(community_candidates)
        official_label = "Official Qwen LLM/multimodal inference candidates"
        community_label = "Best-effort community reduced-refusal inference candidates"

    print(f"\n{official_label}: {official_count:,} discovered")
    if not args.all:
        candidate_count = official_count
        if candidate_count > len(official_view):
            print(
                f"Showing newest {len(official_view)} of {candidate_count}. "
                "Use `models --all` to inspect every Qwen organization repository, including non-LLM artifacts."
            )
    print_table(
        live_rows(official_view, OFFICIAL),
        [
            ("category", "Class", 18),
            ("model", "Model", 66),
            ("task", "Task", 20),
            ("params", "Params*", 10),
            ("updated", "Updated", 12),
            ("status", "Install", 14),
        ],
    )

    print(f"\n{community_label}: {community_count:,} discovered")
    print("These are third-party repositories, not official Qwen releases. Review each model card.")
    print_table(
        live_rows(community_view, REDUCED_REFUSAL),
        [
            ("category", "Class", 18),
            ("model", "Model", 70),
            ("task", "Task", 20),
            ("params", "Params*", 10),
            ("updated", "Updated", 12),
            ("status", "Install", 14),
        ],
    )

    print("\n* Parameter counts are inferred from common model-name patterns when possible.")
    print(
        "  Live discovery is informational. Repository names do not provide enough verified "
        "information to claim package size, hardware fit, or Ollama compatibility."
    )
    print(
        "  Hardware-fit labels are shown only for the curated local-ready catalog above."
    )
    print(
        "  Community discovery is best-effort and cannot guarantee every third-party reduced-refusal variant."
    )
    print("  Only curated Ollama IDs are eligible for automatic setup.")
    return 0


def command_recommend(args: argparse.Namespace) -> int:
    profile = hardware_profile()
    category = normalize_category(args.category)
    print_doctor(profile)
    print("\nRecommendation")
    print("--------------")

    categories = [OFFICIAL, REDUCED_REFUSAL] if category == "all" else [category]
    for current in categories:
        model = recommend_model(profile, current, args.preference)
        label = "official" if current == OFFICIAL else "community reduced-refusal"
        if not model:
            print(f"{label}: no curated recommendation")
            continue
        print(f"{label}:")
        print(f"  model:    {model.ollama}")
        print(f"  size:     ~{model.size_gb} GB")
        print(f"  format:   {model.quant}")
        print(f"  context:  start at {recommended_context(profile, model):,}")
        print(f"  reason:   {model.notes}")
    return 0


def command_can(args: argparse.Namespace) -> int:
    result = compatibility_for_identifier(args.model, hardware_profile())
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return compatibility_exit_code(result)

    fit = "YES" if result.can_run is True else "NO" if result.can_run is False else "UNKNOWN"
    print(f"\n{PROJECT_NAME} compatibility")
    print("--------------------")
    print(f"Model:       {result.model_name or result.model_id}")
    if result.model_name and result.model_name != result.model_id:
        print(f"Logical ID:  {result.model_id}")
    print(f"Fit:         {fit}")
    if result.resolution_status is not None:
        print(f"Compatibility:            {result.status.value}")
        print(f"Compatibility confidence: {result.confidence.value.upper()}")
        resolution_confidence = (
            result.resolution_confidence.value.upper()
            if result.resolution_confidence
            else "UNKNOWN"
        )
        print(f"Resolution status:         {result.resolution_status}")
        print(f"Resolution confidence:     {resolution_confidence}")
        if result.resolved_model:
            params = (
                f"{result.resolved_model.params_b:.3f}B"
                if result.resolved_model.params_b is not None
                else "unknown"
            )
            print(f"Parameters:  {params}")
            print(f"Modalities:  {', '.join(result.resolved_model.modalities)}")
    else:
        print(f"Class:       {result.status.value}")
        print(f"Confidence:  {result.confidence.value.upper()}")

    if result.can_run is True:
        print("\nPractical configuration")
        runtime = result.runtime.capitalize() if result.runtime else "unknown"
        print(f"Runtime:     {runtime}")
        print(f"Build:       {result.artifact_id}")
        print(f"Format:      {result.artifact_format}")
        if result.recommended_context is not None:
            print(f"Context:     {result.recommended_context:,} tokens")
        else:
            print("Context:     unknown")

    if result.artifacts:
        print("\nDetected artifacts")
        for artifact in result.artifacts:
            print(f"- Build: {artifact.artifact_id}")
            print(f"  Format: {artifact.format}")
            print(f"  Size: {artifact.size_bytes} bytes" if artifact.size_bytes else "  Size: unknown")
            print(f"  Quantization: {artifact.quantization or 'unknown'}")
            print(f"  Runtime: {artifact.runtime or 'unknown'}")
            print(
                f"  Context: {artifact.context_max:,} tokens"
                if artifact.context_max
                else "  Context: unknown"
            )
            for item in artifact.evidence:
                print(f"  Evidence: {item.detail} ({item.kind}; {item.source})")
            for unknown in artifact.unknowns:
                print(f"  Unknown: {unknown}")

    if result.runtime_candidates:
        print("\nRuntime candidates")
        for candidate in result.runtime_candidates:
            print(f"- Runtime: {candidate.runtime}")
            print(f"  Build: {candidate.artifact_id}")
            print(f"  Support: {candidate.support_status}")
            print(f"  Fit: {candidate.fit_result}")
            print(f"  Runtime confidence: {candidate.confidence.value.upper()}")
            for blocker in candidate.blockers:
                print(f"  Blocker: {blocker}")
            for unknown in candidate.unknowns:
                print(f"  Unknown: {unknown}")

    if result.estimated_model_memory_gb is not None:
        print("\nMemory planning")
        print(f"Model:       ~{result.estimated_model_memory_gb:.1f} GB")
        if result.planning_budget_gb is not None:
            print(f"Budget:      ~{result.planning_budget_gb:.1f} GB")
        if result.memory_headroom_gb is not None:
            print(f"Headroom:    ~{result.memory_headroom_gb:.1f} GB")

    if result.reason:
        print(f"\nReason: {result.reason}")
    if result.evidence:
        print("\nEvidence")
        for item in result.evidence:
            print(f"- {item.detail} ({item.source})")
    if result.alternatives:
        print("\nCurated practical alternatives")
        for alternative in result.alternatives:
            print(f"- {alternative}")
    if result.unknowns:
        print("\nUnknown")
        for unknown in result.unknowns:
            print(f"- {unknown}")
    return compatibility_exit_code(result)


def command_setup(args: argparse.Namespace) -> int:
    profile = hardware_profile()
    print_doctor(profile)
    ollama = OLLAMA_RUNTIME.info()

    if not ollama.get("installed"):
        install_ollama_help(args.open_installer)
        return 2
    if not OLLAMA_RUNTIME.ensure_available(args.host):
        eprint(
            "Ollama is installed but its localhost API is unavailable. "
            "Launch Ollama or run `ollama serve`."
        )
        return 2

    if args.model:
        selected = resolve_curated_model(args.model)
        if not selected:
            eprint(
                "Automatic setup only accepts curated model IDs or known aliases. "
                "Run `models --offline` to see them."
            )
            return 2
    else:
        selected = recommend_model(
            profile,
            normalize_category(args.category),
            args.preference,
        )

    if not selected:
        eprint("No curated model fits the conservative recommendation.")
        return 2

    context = min(
        int(args.context or recommended_context(profile, selected)),
        int(selected.context_max or 32_768),
    )

    print("\nSelected setup")
    print("--------------")
    print(f"Model:      {selected.ollama}")
    print(f"Class:      {selected.category}")
    print(f"Package:    ~{selected.size_gb} GB")
    print(f"Format:     {selected.quant}")
    print(f"Context:    {context:,} tokens")
    print(f"Notes:      {selected.notes}")

    if selected.category == REDUCED_REFUSAL:
        print(
            "\nCommunity model notice: reduced-refusal/abliterated models are third-party "
            "derivatives, not official Qwen releases. Reduced refusals do not imply better "
            "reasoning, accuracy, or security."
        )

    if not args.yes:
        answer = input("\nPull this model if needed and benchmark it? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    installed_names = [item["name"] for item in installed_ollama_models()]
    runtime_model = installed_id_for_spec(selected, installed_names)
    if runtime_model:
        print(f"\nAlready installed: {runtime_model}")
    else:
        pull_id = args.model if args.model in selected.ids else selected.ollama
        rc = pull_model(pull_id)
        if rc:
            return rc
        runtime_model = pull_id

    if args.no_benchmark:
        print("\nModel is ready.")
        print(f"Chat: ollama run {runtime_model}")
        print(f"API:  {args.host}")
        return 0

    try:
        result = run_benchmark(
            runtime_model,
            context,
            args.runs,
            args.host,
            Path(args.output_dir) if args.output_dir else None,
        )
    except Exception as exc:
        eprint(f"Benchmark failed: {exc}")
        return 1

    print("\nLocal LLM is ready.")
    print(f"Interactive chat: ollama run {runtime_model}")
    print(f"Native Ollama API: {args.host}/api")
    print(f"OpenAI-compatible base URL: {args.host}/v1/")
    print(f"Benchmark: {result.get('files', {}).get('markdown')}")
    return 0


def command_bench(args: argparse.Namespace) -> int:
    if not OLLAMA_RUNTIME.ensure_available(args.host):
        eprint("Ollama API is not available.")
        return 2

    if args.all_installed:
        models = []
        seen_ids: set[str] = set()
        for item in installed_ollama_models():
            if "qwen" not in item["name"].lower():
                continue
            digest = item.get("id", "")
            if digest and digest in seen_ids:
                continue
            if digest:
                seen_ids.add(digest)
            models.append(item["name"])
    else:
        models = [args.model] if args.model else []

    if not models:
        eprint("No installed Qwen model selected or found.")
        return 2
    passport_path = getattr(args, "passport", None)
    if passport_path and len(models) != 1:
        eprint("bench --passport requires exactly one selected model")
        return 2
    passport_inventory = installed_ollama_models() if passport_path else []

    failures = 0
    results: List[Dict[str, Any]] = []
    for model in models:
        try:
            results.append(
                run_benchmark(
                    model,
                    args.context,
                    args.runs,
                    args.host,
                    Path(args.output_dir) if args.output_dir else None,
                )
            )
            if passport_path:
                for installed in passport_inventory:
                    if model_name_matches(model, installed.get("name", "")):
                        results[-1]["artifact_fingerprint"] = installed.get("id") or None
                        break
                write_passport(
                    passport_from_benchmark_result(results[-1]), Path(passport_path)
                )
                print(f"Passport: {passport_path}")
        except Exception as exc:
            failures += 1
            eprint(f"Benchmark failed for {model}: {exc}")

    if len(results) > 1:
        rows: List[Dict[str, Any]] = []
        for result in results:
            aggregate = result["aggregate"]
            rows.append(
                {
                    "model": result["model"],
                    "gen": aggregate.get("generation_tps_avg"),
                    "prompt": aggregate.get("prompt_tps_avg"),
                    "smoke": f"{aggregate.get('smoke_passed')}/{aggregate.get('smoke_total')}",
                    "ctx": result["context"],
                }
            )
        print("\nComparison")
        print("----------")
        print_table(
            rows,
            [
                ("model", "Model", 60),
                ("gen", "Gen tok/s", 12),
                ("prompt", "Prompt tok/s", 14),
                ("smoke", "Smoke", 8),
                ("ctx", "Context", 10),
            ],
        )
    return 1 if failures else 0


def race_exit_code(result: RaceResult) -> int:
    return 0 if result.status == "completed" else 1 if result.status == "failed" else 2


def print_race_result(result: RaceResult) -> None:
    print(f"\n{PROJECT_NAME} runtime race")
    print("-------------------")
    print(f"Model:  {result.logical_model_id}")
    print(f"Status: {result.status}")
    if result.reason:
        print(f"Reason: {result.reason}")

    print(f"\nExecutable configurations: {len(result.eligible_configurations)}")
    for item in result.eligible_configurations:
        print(f"- Runtime: {item.runtime}")
        print(f"  Build: {item.artifact_id}")
        print(f"  Format: {item.artifact_format}")
        print(f"  Quantization: {item.quantization or 'unknown'}")
        print(f"  Runtime version: {item.runtime_version or 'unknown'}")

    if result.ineligible_configurations:
        print("\nIneligible/theoretical configurations")
        for item in result.ineligible_configurations:
            print(f"- {item.runtime}: {item.artifact_id}")
            for blocker in item.blockers:
                print(f"  Blocker: {blocker}")

    if result.competitors:
        print("\nMeasured competitors")
        for item in result.competitors:
            print(f"- {item.runtime}: {item.artifact_id}")
            print(f"  Execution: {item.execution_status}")
            if item.execution_status == "success":
                print(f"  Generation: {item.generation_tps} tok/s")
                print(f"  Prompt evaluation: {item.prompt_eval_tps} tok/s")
                print(f"  Mean normalized inference latency: {item.total_latency_s} s")
                print(f"  Generated tokens: {item.generated_tokens}")
            elif item.failure:
                print(f"  Failure: {item.failure}")
    for warning in result.warnings:
        print(f"\nWarning: {warning}")
    if result.winners:
        print("\nMetric results")
        for metric, winner in result.winners:
            label = metric.replace("_", " ").capitalize()
            if winner.get("status") == "winner":
                print(
                    f"- {label}: {winner['runtime']} / {winner['artifact_id']} "
                    f"({winner['value']})"
                )
            else:
                print(f"- {label}: inconclusive ({winner.get('reason')})")


def export_race_passports(result: RaceResult, output: Path) -> Tuple[Path, ...]:
    """Export only a completed race; failed comparisons produce no passports."""
    if result.status != "completed":
        return ()
    written = []
    for competitor in result.competitors:
        if competitor.execution_status != "success" or not competitor.raw_samples:
            continue
        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+", "_", f"{competitor.runtime}_{competitor.artifact_id}"
        )
        path = output / f"{safe_name}.passport.json"
        write_passport(passport_from_race_competitor(result, competitor), path)
        written.append(path)
    return tuple(written)


def command_race(args: argparse.Namespace) -> int:
    if args.runs < 1 or args.runs > 5:
        eprint("race --runs must be between 1 and 5")
        return 2
    if args.context < 1 or args.context > 32_768:
        eprint("race --context must be between 1 and 32768")
        return 2
    if args.num_predict < 1 or args.num_predict > 512:
        eprint("race --num-predict must be between 1 and 512")
        return 2

    local_values = tuple(getattr(args, "local_artifact", ()) or ())
    try:
        parse_local_artifact_values(local_values)
    except ValueError as error:
        eprint(str(error))
        return 2
    profile = hardware_profile()
    adapters: Tuple[ExecutionAdapter, ...] = (
        OllamaExecutionAdapter(args.host), LlamaCppExecutionAdapter(), MlxExecutionAdapter()
    )
    compatibility = compatibility_for_identifier(args.model, profile) if local_values else None
    try:
        native_targets = local_execution_targets(
            args.model, local_values, profile, adapters, compatibility
        )
    except ValueError as error:
        eprint(str(error))
        return 2
    logical_id, eligible, ineligible, resolution_reason = race_configurations(
        args.model, profile, adapters, compatibility
    )
    native_eligible = tuple(target.configuration for target in native_targets if target.configuration.eligible)
    native_ineligible = tuple(target.configuration for target in native_targets if not target.configuration.eligible)
    eligible = unique_race_configurations(tuple(eligible) + native_eligible)
    ineligible = unique_race_configurations(tuple(ineligible) + native_ineligible)
    workload = RaceWorkload(
        prompt=SPEED_PROMPT,
        context=args.context,
        num_predict=args.num_predict,
        runs=args.runs,
    )
    result = execute_race(
        logical_id,
        eligible,
        ineligible,
        adapters,
        workload,
        race_hardware_summary(profile),
        resolution_reason,
        execution_targets=native_targets,
    )
    passport_dir = getattr(args, "passport_dir", None)
    if passport_dir:
        export_race_passports(result, Path(passport_dir))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_race_result(result)
    return race_exit_code(result)


def command_passport_verify(args: argparse.Namespace) -> int:
    """Validate a local passport without network or inference."""
    try:
        document = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("Passport validation: FAIL")
        print("- file is not readable valid JSON")
        return 1
    if not isinstance(document, dict):
        print("Passport validation: FAIL")
        print("- passport root must be an object")
        return 1
    errors = validate_passport(document)
    if errors:
        print("Passport validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Passport validation: PASS")
    print("The passport is internally valid; this is not independent proof of the measurement.")
    return 0


def command_check(args: argparse.Namespace) -> int:
    """Run project sanity checks without downloading a model."""
    errors = validate_curated_catalog()
    if errors:
        print("Catalog validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Catalog validation: PASS")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Curated models: {len(CURATED_MODELS)}")

    profile = hardware_profile()
    if not profile.get("ram_gib"):
        print("Hardware RAM detection: FAIL")
        return 1
    print("Hardware detection: PASS")

    if args.online:
        catalog = fetch_live_catalog(force=True)
        if catalog.get("warning") or not catalog.get("official"):
            print(f"Live catalog: FAIL - {catalog.get('warning') or 'no official models returned'}")
            return 1
        print(f"Live catalog: PASS ({len(catalog.get('official') or [])} official Qwen repos)")

    print("LLMRig checks: PASS")
    return 0


def interactive() -> int:
    print(f"\n{PROJECT_NAME}")
    print("=" * len(PROJECT_NAME))
    print(
        "Inspect this machine, choose a practical Qwen model, pull it with Ollama, "
        "and save benchmark results.\n"
    )

    profile = hardware_profile()
    print_doctor(profile)

    if not ollama_cli_info().get("installed"):
        install_ollama_help(open_page=True)
        return 2

    print("\nWhich model class do you want?")
    print("  1. Official / standard alignment")
    print("  2. Community reduced-refusal (often called uncensored/abliterated)")
    raw = input("Choice [1]: ").strip() or "1"
    category = REDUCED_REFUSAL if raw == "2" else OFFICIAL

    selected = recommend_model(profile, category, "balanced")
    if not selected:
        eprint("No curated recommendation was found.")
        return 2

    context = recommended_context(profile, selected)
    print(
        f"\nRecommended: {selected.ollama} "
        f"(~{selected.size_gb} GB, {selected.quant}, start at {context // 1024}K context)"
    )
    if input("Pull it if needed and run the benchmark? [Y/n]: ").strip().lower() in {
        "n",
        "no",
    }:
        return 0

    args = argparse.Namespace(
        model=selected.ollama,
        category=category,
        preference="balanced",
        context=context,
        yes=True,
        no_benchmark=False,
        runs=2,
        host=DEFAULT_OLLAMA_HOST,
        output_dir=None,
        open_installer=True,
    )
    return command_setup(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-platform local LLM readiness with Qwen-first discovery, "
            "Ollama setup, and benchmarking."
        )
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Inspect hardware and Ollama readiness.")
    doctor.add_argument("--json", action="store_true")

    models = subparsers.add_parser("models", help="Show curated + live model catalogs (Qwen-first).")
    models.add_argument("--all", action="store_true", help="Show every discovered repository, including non-LLM artifacts.")
    models.add_argument("--latest", type=int, default=40, help="Newest live entries to show.")
    models.add_argument("--refresh", action="store_true", help="Ignore the six-hour cache.")
    models.add_argument("--offline", action="store_true", help="Show only the curated snapshot.")
    models.add_argument("--fit", action="store_true", help="Estimate fit for curated local-ready models on this machine.")

    recommend = subparsers.add_parser("recommend", help="Recommend a model for this machine.")
    recommend.add_argument(
        "--category",
        "--alignment",
        dest="category",
        default="all",
        choices=[
            "all",
            "official",
            "standard",
            "restricted",
            "unrestricted",
            "uncensored",
            "reduced-refusal",
        ],
    )
    recommend.add_argument(
        "--preference",
        choices=["balanced", "speed", "quality"],
        default="balanced",
    )

    can = subparsers.add_parser(
        "can", help="Check a curated model or inspect Hugging Face artifact metadata."
    )
    can.add_argument("model", help="Curated ID/alias or owner/repository Hugging Face ID.")
    can.add_argument("--json", action="store_true", help="Emit deterministic JSON.")

    setup = subparsers.add_parser(
        "setup",
        help="Pull a recommended curated model and optionally benchmark it.",
    )
    setup.add_argument(
        "--category",
        "--alignment",
        dest="category",
        default="official",
        choices=[
            "official",
            "standard",
            "restricted",
            "unrestricted",
            "uncensored",
            "reduced-refusal",
        ],
    )
    setup.add_argument(
        "--preference",
        choices=["balanced", "speed", "quality"],
        default="balanced",
    )
    setup.add_argument("--model", help="Exact curated Ollama model ID or known alias.")
    setup.add_argument("--context", type=int, help="Context allocation in tokens.")
    setup.add_argument("--runs", type=int, default=2, help="Timed throughput runs.")
    setup.add_argument("--yes", action="store_true", help="Skip confirmation.")
    setup.add_argument("--no-benchmark", action="store_true", help="Prepare the model only.")
    setup.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama base URL.")
    setup.add_argument("--output-dir", help="Benchmark output directory.")
    setup.add_argument(
        "--open-installer",
        action="store_true",
        help="Open the official Ollama download page if Ollama is missing.",
    )

    bench = subparsers.add_parser("bench", help="Benchmark an installed supported Ollama model.")
    bench.add_argument("--model", help="Installed Ollama model ID.")
    bench.add_argument(
        "--all-installed",
        action="store_true",
        help="Benchmark every installed model whose name contains 'qwen'.",
    )
    bench.add_argument("--context", type=int, default=32_768)
    bench.add_argument("--runs", type=int, default=2)
    bench.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    bench.add_argument("--output-dir")
    bench.add_argument(
        "--passport",
        help="Write a privacy-safe benchmark passport for a single selected model.",
    )

    race = subparsers.add_parser(
        "race", help="Compare at least two locally executable configurations."
    )
    race.add_argument("model", help="Curated ID/alias or owner/repository identifier.")
    race.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    race.add_argument("--context", type=int, default=RACE_CONTEXT)
    race.add_argument("--runs", type=int, default=2)
    race.add_argument("--num-predict", type=int, default=RACE_NUM_PREDICT)
    race.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    race.add_argument(
        "--local-artifact", action="append", default=[], metavar="RUNTIME=PATH",
        help="Use an explicit already-local MLX-LM directory or llama.cpp GGUF file.",
    )
    race.add_argument(
        "--passport-dir",
        help="Write one passport per successfully measured competitor.",
    )

    passport = subparsers.add_parser(
        "passport", help="Inspect local benchmark passports without execution or network access."
    )
    passport_subparsers = passport.add_subparsers(dest="passport_command")
    passport_verify = passport_subparsers.add_parser(
        "verify", help="Verify passport schema, integrity, aggregates, and privacy."
    )
    passport_verify.add_argument("file")

    check = subparsers.add_parser("check", help="Run project sanity checks without a model download.")
    check.add_argument(
        "--online",
        action="store_true",
        help="Also verify live Hugging Face discovery.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        try:
            return interactive()
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 130

    if args.command == "doctor":
        print_doctor(hardware_profile(), args.json)
        return 0
    if args.command == "models":
        return command_models(args)
    if args.command == "recommend":
        return command_recommend(args)
    if args.command == "can":
        return command_can(args)
    if args.command == "setup":
        return command_setup(args)
    if args.command == "bench":
        return command_bench(args)
    if args.command == "race":
        return command_race(args)
    if args.command == "passport":
        if args.passport_command == "verify":
            return command_passport_verify(args)
        parser.parse_args(["passport", "--help"])
        return 0
    if args.command == "check":
        return command_check(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
