"""Conservative Hugging Face artifact classification for LLMRig.

This module intentionally separates *what a repository contains* from *what can
run*.  File names and Hub metadata can establish packaging evidence, but they do
not establish runtime compatibility, model quality, or installation trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


_GGUF_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:IQ|Q)\d(?:_[A-Z0-9]+)+|F16|F32|BF16)(?:[-_.]|$)", re.I
)


@dataclass(frozen=True)
class HubArtifact:
    model_id: str
    path: str
    format: str
    size_bytes: Optional[int]
    quantization: Optional[str]
    runtime_hints: Tuple[str, ...]
    evidence: Tuple[str, ...]
    unknowns: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": self.path,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "quantization": self.quantization,
            "runtime_hints": list(self.runtime_hints),
            "evidence": list(self.evidence),
            "unknowns": list(self.unknowns),
        }


def _safe_size(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _gguf_quantization(path: str) -> Optional[str]:
    match = _GGUF_QUANT_RE.search(path.upper())
    return match.group(1).upper() if match else None


def _runtime_hints(path: str, format_name: str, tags: Sequence[str]) -> Tuple[str, ...]:
    lower_path = path.lower()
    lower_tags = {item.lower() for item in tags}
    hints = []
    if format_name == "GGUF":
        hints.append("llama.cpp")
    if "mlx" in lower_path or "mlx" in lower_tags or "mlx-community" in lower_path:
        hints.extend(("mlx-lm", "omlx"))
    # Do not infer a runtime from generic safetensors: many unrelated runtimes
    # use the same container format.
    return tuple(dict.fromkeys(hints))


def classify_hub_file(
    model_id: str,
    sibling: Mapping[str, Any],
    *,
    tags: Sequence[str] = (),
) -> Optional[HubArtifact]:
    """Classify one Hub sibling using only observable repository metadata."""

    path = str(sibling.get("rfilename") or sibling.get("path") or "").strip()
    if not path:
        return None
    lower = path.lower()
    size = _safe_size(sibling.get("size"))

    if lower.endswith(".gguf"):
        format_name = "GGUF"
        quantization = _gguf_quantization(path)
        evidence = ("file extension identifies a GGUF artifact",)
        unknowns = () if quantization else ("GGUF quantization is not encoded recognizably in the file name",)
    elif lower.endswith(".safetensors"):
        format_name = "safetensors"
        quantization = None
        evidence = ("file extension identifies a safetensors shard or artifact",)
        unknowns = ("safetensors alone does not establish runtime compatibility or quantization",)
    elif lower.endswith((".bin", ".pt", ".pth")):
        format_name = "PyTorch"
        quantization = None
        evidence = ("file extension identifies a PyTorch-family weight artifact",)
        unknowns = ("runtime compatibility and quantization are unknown",)
    else:
        return None

    return HubArtifact(
        model_id=model_id,
        path=path,
        format=format_name,
        size_bytes=size,
        quantization=quantization,
        runtime_hints=_runtime_hints(path, format_name, tags),
        evidence=evidence,
        unknowns=unknowns,
    )


def classify_hub_siblings(
    model_id: str,
    siblings: Iterable[Mapping[str, Any]],
    *,
    tags: Sequence[str] = (),
) -> Tuple[HubArtifact, ...]:
    """Return stable, deduplicated artifact observations for a Hub repository."""

    by_path = {}
    for sibling in siblings:
        artifact = classify_hub_file(model_id, sibling, tags=tags)
        if artifact is not None:
            by_path.setdefault(artifact.path, artifact)
    return tuple(by_path[path] for path in sorted(by_path))


def summarize_hub_artifacts(artifacts: Sequence[HubArtifact]) -> dict:
    """Summarize packaging evidence without turning hints into recommendations."""

    formats = tuple(sorted({item.format for item in artifacts}))
    runtime_hints = tuple(sorted({hint for item in artifacts for hint in item.runtime_hints}))
    quantizations = tuple(sorted({item.quantization for item in artifacts if item.quantization}))
    known_size = sum(item.size_bytes or 0 for item in artifacts)
    all_sizes_known = bool(artifacts) and all(item.size_bytes is not None for item in artifacts)
    return {
        "artifact_count": len(artifacts),
        "formats": list(formats),
        "runtime_hints": list(runtime_hints),
        "quantizations": list(quantizations),
        "known_size_bytes": known_size if known_size else None,
        "all_artifact_sizes_known": all_sizes_known,
        "warning": "runtime hints are packaging evidence, not proof of executability",
    }
