#!/usr/bin/env python3
"""LLMRig: cross-platform local LLM readiness, discovery, setup, and benchmarking."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

PROJECT_NAME = "LLMRig"
PROJECT_SLUG = "llmrig"
VERSION = "0.4.1"
CURATED_SNAPSHOT_DATE = "2026-08-19"
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
HF_MODELS_API = "https://huggingface.co/api/models"
CACHE_TTL_SECONDS = 6 * 60 * 60

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


@dataclass(frozen=True)
class ModelArtifact:
    """A concrete model package intended for a particular runtime."""

    artifact_id: str
    model_id: str
    runtime: str
    format: str
    size_gb: Optional[float]
    context_max: Optional[int]
    platforms: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id.strip() or not self.model_id.strip():
            raise ValueError("artifact id and model id must not be empty")
        if not self.runtime.strip() or not self.format.strip():
            raise ValueError("artifact runtime and format must not be empty")
        if self.size_gb is not None and self.size_gb <= 0:
            raise ValueError("artifact size must be positive when known")
        if self.context_max is not None and self.context_max <= 0:
            raise ValueError("artifact context must be positive when known")
        if not self.platforms:
            raise ValueError("artifact must support at least one platform")
        if self.artifact_id in self.aliases or len(set(self.aliases)) != len(self.aliases):
            raise ValueError("artifact aliases must be unique and exclude the primary id")

    @property
    def ids(self) -> Tuple[str, ...]:
        return (self.artifact_id,) + self.aliases


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

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("compatibility result model id must not be empty")
        status_unknown = self.status == CompatibilityStatus.UNKNOWN
        confidence_unknown = self.confidence == Confidence.UNKNOWN
        if status_unknown != confidence_unknown:
            raise ValueError("unknown status and confidence must be represented together")
        if not status_unknown and not self.evidence:
            raise ValueError("known compatibility status requires evidence")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "artifact_id": self.artifact_id,
            "runtime": self.runtime,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
            "reason": self.reason,
        }


class RuntimeProvider(Protocol):
    """Minimum runtime boundary used by current readiness/setup behavior."""

    name: str

    def info(self) -> Dict[str, Any]: ...

    def is_available(self, endpoint: Optional[str] = None) -> bool: ...

    def ensure_available(self, endpoint: Optional[str] = None) -> bool: ...


class ModelSource(Protocol):
    """Minimum source boundary used by the curated catalog."""

    name: str

    def list_specs(self) -> Sequence["ModelSpec"]: ...

    def resolve(self, identifier: str) -> Optional["ModelSpec"]: ...


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
        return ModelArtifact(
            artifact_id=self.ollama,
            model_id=self.model.model_id,
            runtime="ollama",
            format=self.quant,
            size_gb=self.size_gb,
            context_max=self.context_max,
            platforms=self.platforms,
            aliases=self.aliases,
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
    except Exception as exc:
        result["version"] = f"unknown ({exc})"
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


OLLAMA_RUNTIME: RuntimeProvider = OllamaRuntimeProvider()


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
    evidence = (
        RecommendationEvidence(
            "curated-metadata",
            f"curated snapshot {CURATED_SNAPSHOT_DATE}",
            "artifact size, runtime, platform, and context are curated",
        ),
        RecommendationEvidence(
            "deterministic-estimate",
            "local hardware profile",
            "model-weight budget is calculated with conservative headroom",
        ),
    )
    if not profile.get("os") or not ram:
        return CompatibilityResult(
            model.model.model_id,
            artifact.artifact_id,
            artifact.runtime,
            CompatibilityStatus.UNKNOWN,
            Confidence.UNKNOWN,
            evidence[:1],
            "hardware platform or memory is unknown",
        )
    if profile.get("os") not in artifact.platforms:
        status = CompatibilityStatus.NOT_NATIVE
    elif model.size_gb <= budget * 0.75:
        status = CompatibilityStatus.EXCELLENT
    elif model.size_gb <= budget:
        status = CompatibilityStatus.GOOD
    elif model.size_gb <= ram * 0.80:
        status = CompatibilityStatus.POSSIBLE
    else:
        status = CompatibilityStatus.TOO_LARGE
    return CompatibilityResult(
        model.model.model_id,
        artifact.artifact_id,
        artifact.runtime,
        status,
        Confidence.HIGH,
        evidence,
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
    timeout: int = 600,
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
    if args.command == "setup":
        return command_setup(args)
    if args.command == "bench":
        return command_bench(args)
    if args.command == "check":
        return command_check(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
