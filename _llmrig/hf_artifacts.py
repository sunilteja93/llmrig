"""Conservative Hugging Face artifact classification for LLMRig.

This module separates *what repository metadata proves* from *what may run*.
Hub file names, tags, and config fields are useful evidence, but they do not
establish installation trust, local availability, or successful execution.
Conflicting or incomplete metadata fails closed instead of being guessed through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


_GGUF_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:IQ|Q)\d(?:_[A-Z0-9]+)+|F16|F32|BF16)(?:[-_.]|$)", re.I
)
_SAFETENSORS_SHARD_RE = re.compile(
    r"^(.*)-(\d{5})-of-(\d{5})\.safetensors$", re.I
)
_CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
)


@dataclass(frozen=True)
class HubEvidence:
    kind: str
    source: str
    detail: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "source": self.source, "detail": self.detail}


@dataclass(frozen=True)
class HubArtifact:
    model_id: str
    path: str
    format: str
    size_bytes: Optional[int]
    quantization: Optional[str]
    runtime_hints: Tuple[str, ...]
    evidence: Tuple[HubEvidence, ...]
    context_max: Optional[int] = None
    members: Tuple[str, ...] = ()
    unknowns: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "path": self.path,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "quantization": self.quantization,
            "runtime_hints": list(self.runtime_hints),
            "context_max": self.context_max,
            "members": list(self.members),
            "evidence": [item.to_dict() for item in self.evidence],
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class HubRepositoryClassification:
    requested_id: str
    repository_id: str
    logical_model_id: str
    base_model_id: Optional[str]
    base_model_status: str
    context_max: Optional[int]
    context_status: str
    explicit_quantization: Optional[str]
    quantization_status: str
    artifacts: Tuple[HubArtifact, ...]
    evidence: Tuple[HubEvidence, ...]
    unknowns: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "requested_id": self.requested_id,
            "repository_id": self.repository_id,
            "logical_model_id": self.logical_model_id,
            "base_model_id": self.base_model_id,
            "base_model_status": self.base_model_status,
            "context_max": self.context_max,
            "context_status": self.context_status,
            "explicit_quantization": self.explicit_quantization,
            "quantization_status": self.quantization_status,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "evidence": [item.to_dict() for item in self.evidence],
            "unknowns": list(self.unknowns),
        }


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def sibling_size(sibling: Mapping[str, Any]) -> Optional[int]:
    raw = sibling.get("size")
    lfs = sibling.get("lfs")
    if raw is None and isinstance(lfs, Mapping):
        raw = lfs.get("size")
    return _positive_int(raw)


def _gguf_quantization(path: str) -> Tuple[Optional[str], str]:
    stem = path.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    matches = {match.upper() for match in _GGUF_QUANT_RE.findall(stem.upper())}
    if len(matches) == 1:
        return next(iter(matches)), "inferred"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "unknown"


def _runtime_hints(path: str, format_name: str, tags: Sequence[str]) -> Tuple[str, ...]:
    lower_path = path.lower()
    lower_tags = {str(item).lower() for item in tags}
    hints = []
    if format_name == "GGUF":
        hints.append("llama.cpp")
    if (
        format_name == "MLX"
        or "mlx" in lower_path
        or "mlx" in lower_tags
        or "mlx-community" in lower_path
    ):
        hints.extend(("mlx-lm", "omlx"))
    return tuple(dict.fromkeys(hints))


def base_model_metadata(payload: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    candidates = []
    card_data = payload.get("cardData")
    if isinstance(card_data, Mapping):
        value = card_data.get("base_model")
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, (list, tuple)):
            candidates.extend(item for item in value if isinstance(item, str))
    for tag in payload.get("tags") or ():
        if isinstance(tag, str) and tag.startswith("base_model:"):
            candidates.append(tag.split(":", 1)[1])
    unique = tuple(dict.fromkeys(item.strip() for item in candidates if item.strip()))
    if len(unique) == 1:
        return unique[0], "verified"
    if len(unique) > 1:
        return None, "ambiguous"
    return None, "unknown"


def context_metadata(payload: Mapping[str, Any]) -> Tuple[Optional[int], str, Tuple[str, ...]]:
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    containers = [("config", config)]
    text_config = config.get("text_config") if isinstance(config, Mapping) else None
    if isinstance(text_config, Mapping):
        containers.append(("config.text_config", text_config))

    observations = []
    for prefix, container in containers:
        for key in _CONTEXT_KEYS:
            value = _positive_int(container.get(key))
            if value is not None:
                observations.append((f"{prefix}.{key}", value))
    values = {value for _, value in observations}
    sources = tuple(source for source, _ in observations)
    if len(values) == 1:
        return next(iter(values)), "verified", sources
    if len(values) > 1:
        return None, "ambiguous", sources
    return None, "unknown", ()


def explicit_quantization_metadata(
    payload: Mapping[str, Any],
) -> Tuple[Optional[str], str, Tuple[str, ...]]:
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    quant_config = config.get("quantization_config") if isinstance(config, Mapping) else None
    mlx_quant = config.get("quantization") if isinstance(config, Mapping) else None

    method = None
    quant_bits = None
    mlx_bits = None
    sources = []
    if isinstance(quant_config, Mapping):
        raw_method = quant_config.get("quant_method")
        if isinstance(raw_method, str) and raw_method.strip():
            method = raw_method.strip()
            sources.append("config.quantization_config.quant_method")
        quant_bits = _positive_int(quant_config.get("bits"))
        if quant_bits is not None:
            sources.append("config.quantization_config.bits")
    if isinstance(mlx_quant, Mapping):
        mlx_bits = _positive_int(mlx_quant.get("bits"))
        if mlx_bits is not None:
            sources.append("config.quantization.bits")

    if quant_bits is not None and mlx_bits is not None and quant_bits != mlx_bits:
        return None, "ambiguous", tuple(sources)
    bits = quant_bits if quant_bits is not None else mlx_bits
    if method is not None and bits is not None:
        return f"{method}-{bits}-bit", "verified", tuple(sources)
    if method is not None:
        return method, "verified", tuple(sources)
    if bits is not None:
        return f"{bits}-bit", "verified", tuple(sources)
    return None, "unknown", ()


def safetensors_weight_set(
    siblings: Iterable[Mapping[str, Any]],
) -> Tuple[Tuple[str, ...], Optional[int], str]:
    by_path = {}
    for item in siblings:
        path = str(item.get("rfilename") or item.get("path") or "").strip()
        if path.lower().endswith(".safetensors"):
            by_path.setdefault(path, item)
    paths = tuple(sorted(by_path))
    ordered = [by_path[path] for path in paths]
    if not ordered:
        return (), None, "missing"

    if len(ordered) == 1:
        shard = _SAFETENSORS_SHARD_RE.match(paths[0])
        if shard is not None and (int(shard.group(2)) != 1 or int(shard.group(3)) != 1):
            return paths, None, "ambiguous"
        size = sibling_size(ordered[0])
        return paths, size, "verified" if size is not None else "missing_size"

    matches = [_SAFETENSORS_SHARD_RE.match(path) for path in paths]
    if not all(matches):
        return paths, None, "ambiguous"
    shard_keys = {(match.group(1), int(match.group(3))) for match in matches if match}
    if len(shard_keys) != 1:
        return paths, None, "ambiguous"
    _, total = next(iter(shard_keys))
    indices = {int(match.group(2)) for match in matches if match}
    if len(paths) != total or indices != set(range(1, total + 1)):
        return paths, None, "ambiguous"
    sizes = [sibling_size(item) for item in ordered]
    if any(size is None for size in sizes):
        return paths, None, "missing_size"
    return paths, sum(size for size in sizes if size is not None), "verified"


def classify_hub_file(
    model_id: str,
    sibling: Mapping[str, Any],
    *,
    tags: Sequence[str] = (),
    library_name: Optional[str] = None,
    context_max: Optional[int] = None,
) -> Optional[HubArtifact]:
    """Classify one Hub sibling using only observable repository metadata."""

    path = str(sibling.get("rfilename") or sibling.get("path") or "").strip()
    if not path:
        return None
    lower = path.lower()
    size = sibling_size(sibling)
    evidence = []
    unknowns = []

    if lower.endswith(".gguf"):
        format_name = "GGUF"
        quantization, quant_status = _gguf_quantization(path)
        evidence.append(
            HubEvidence(
                "deterministic-inference",
                "GGUF .gguf extension rule",
                "file extension identifies a GGUF artifact",
            )
        )
        if quant_status == "inferred":
            evidence.append(
                HubEvidence(
                    "deterministic-inference",
                    "conventional GGUF quantization filename rule",
                    f"quantization {quantization} is encoded unambiguously in the file name",
                )
            )
        elif quant_status == "ambiguous":
            unknowns.append("GGUF quantization filename contains conflicting recognized tokens")
        else:
            unknowns.append("GGUF quantization is not encoded recognizably in the file name")
    elif lower.endswith(".safetensors"):
        explicit_mlx = str(library_name or "").strip().lower() == "mlx"
        format_name = "MLX" if explicit_mlx else "safetensors"
        quantization = None
        if explicit_mlx:
            evidence.append(
                HubEvidence(
                    "verified-metadata",
                    "Hugging Face library metadata",
                    "library_name=mlx explicitly identifies MLX packaging",
                )
            )
        else:
            evidence.append(
                HubEvidence(
                    "deterministic-inference",
                    "Safetensors .safetensors extension rule",
                    "file extension identifies a safetensors shard or artifact",
                )
            )
            unknowns.append(
                "safetensors alone does not establish runtime compatibility or MLX packaging"
            )
    elif lower.endswith((".bin", ".pt", ".pth")):
        format_name = "PyTorch"
        quantization = None
        evidence.append(
            HubEvidence(
                "deterministic-inference",
                "PyTorch-family extension rule",
                "file extension identifies a PyTorch-family weight artifact",
            )
        )
        unknowns.append("runtime compatibility and quantization are unknown")
    else:
        return None

    if size is not None:
        evidence.append(
            HubEvidence(
                "verified-metadata",
                "Hugging Face repository file metadata",
                "artifact byte size is reported by repository metadata",
            )
        )
    else:
        unknowns.append("artifact size is unknown")

    return HubArtifact(
        model_id=model_id,
        path=path,
        format=format_name,
        size_bytes=size,
        quantization=quantization,
        runtime_hints=_runtime_hints(path, format_name, tags),
        evidence=tuple(evidence),
        context_max=context_max,
        members=(path,),
        unknowns=tuple(dict.fromkeys(unknowns)),
    )


def classify_hub_siblings(
    model_id: str,
    siblings: Iterable[Mapping[str, Any]],
    *,
    tags: Sequence[str] = (),
    library_name: Optional[str] = None,
) -> Tuple[HubArtifact, ...]:
    """Return stable, deduplicated per-file artifact observations."""

    by_path = {}
    for sibling in siblings:
        artifact = classify_hub_file(
            model_id,
            sibling,
            tags=tags,
            library_name=library_name,
        )
        if artifact is not None:
            by_path.setdefault(artifact.path, artifact)
    return tuple(by_path[path] for path in sorted(by_path))


def _context_evidence(value: int, sources: Sequence[str]) -> HubEvidence:
    return HubEvidence(
        "verified-metadata",
        "Hugging Face model config metadata",
        "context limit is consistently reported by " + ", ".join(sources),
    )


def _quantization_evidence(value: str, sources: Sequence[str]) -> HubEvidence:
    return HubEvidence(
        "verified-metadata",
        "Hugging Face model config metadata",
        f"quantization {value} is explicitly reported by " + ", ".join(sources),
    )


def classify_hub_repository(
    requested_id: str,
    payload: Mapping[str, Any],
) -> HubRepositoryClassification:
    """Classify one Hub repository without inventing missing or conflicting facts."""

    requested = str(requested_id or "").strip()
    repository_id = str(payload.get("id") or payload.get("modelId") or requested).strip()
    if not repository_id:
        repository_id = requested

    base_model_id, base_status = base_model_metadata(payload)
    logical_model_id = base_model_id or repository_id
    context_max, context_status, context_sources = context_metadata(payload)
    explicit_quant, quant_status, quant_sources = explicit_quantization_metadata(payload)
    tags = tuple(str(item) for item in (payload.get("tags") or ()) if isinstance(item, str))
    library_name = str(payload.get("library_name") or "").strip() or None
    siblings = tuple(item for item in (payload.get("siblings") or ()) if isinstance(item, Mapping))

    context_evidence = (
        _context_evidence(context_max, context_sources)
        if context_status == "verified" and context_max is not None
        else None
    )
    quant_evidence = (
        _quantization_evidence(explicit_quant, quant_sources)
        if quant_status == "verified" and explicit_quant is not None
        else None
    )

    repository_evidence = [
        HubEvidence(
            "verified-metadata",
            "Hugging Face model API",
            "repository identity and metadata were returned by Hugging Face",
        )
    ]
    repository_unknowns = []
    if base_status == "verified":
        repository_evidence.append(
            HubEvidence(
                "verified-metadata",
                "Hugging Face base_model metadata",
                f"repository explicitly links its artifacts to logical model {base_model_id}",
            )
        )
    elif base_status == "ambiguous":
        repository_unknowns.append("base model association is ambiguous")
    if context_evidence is not None:
        repository_evidence.append(context_evidence)
    elif context_status == "ambiguous":
        repository_unknowns.append("context limit metadata is conflicting")
    if quant_evidence is not None:
        repository_evidence.append(quant_evidence)
    elif quant_status == "ambiguous":
        repository_unknowns.append("quantization metadata is conflicting")

    artifacts = []
    for item in sorted(
        siblings,
        key=lambda value: str(value.get("rfilename") or value.get("path") or ""),
    ):
        path = str(item.get("rfilename") or item.get("path") or "")
        if not path.lower().endswith(".gguf"):
            continue
        artifact = classify_hub_file(
            logical_model_id,
            item,
            tags=tags,
            library_name=library_name,
            context_max=context_max,
        )
        if artifact is None:
            continue
        evidence = list(artifact.evidence)
        unknowns = list(artifact.unknowns)
        if context_evidence is not None:
            evidence.append(context_evidence)
        elif context_status == "ambiguous":
            unknowns.append("context limit metadata is conflicting")
        else:
            unknowns.append("context limit is unknown")
        artifacts.append(
            replace(
                artifact,
                evidence=tuple(dict.fromkeys(evidence)),
                unknowns=tuple(dict.fromkeys(unknowns)),
            )
        )

    members, size_bytes, size_status = safetensors_weight_set(siblings)
    if members:
        explicit_mlx = str(library_name or "").lower() == "mlx"
        format_name = "MLX" if explicit_mlx else "safetensors"
        evidence = []
        unknowns = ["runtime compatibility is unknown"]
        if explicit_mlx:
            evidence.append(
                HubEvidence(
                    "verified-metadata",
                    "Hugging Face library metadata",
                    "library_name=mlx explicitly identifies MLX packaging",
                )
            )
        else:
            evidence.append(
                HubEvidence(
                    "deterministic-inference",
                    "Safetensors .safetensors extension rule",
                    "repository contains safetensors weight files",
                )
            )
            unknowns.append("generic safetensors do not establish MLX/oMLX compatibility")

        if size_status == "verified" and size_bytes is not None:
            evidence.append(
                HubEvidence(
                    "verified-metadata",
                    "Hugging Face repository file metadata",
                    "artifact size is the sum of the complete listed weight set",
                )
            )
        elif size_status == "ambiguous":
            unknowns.append("artifact size is unknown because weight-file grouping is ambiguous")
        else:
            unknowns.append("artifact size is unknown")

        quantization = explicit_quant if quant_evidence is not None else None
        if quant_evidence is not None:
            evidence.append(quant_evidence)
        elif quant_status == "ambiguous":
            unknowns.append("quantization metadata is conflicting")
        else:
            unknowns.append("quantization is unknown")

        if context_evidence is not None:
            evidence.append(context_evidence)
        elif context_status == "ambiguous":
            unknowns.append("context limit metadata is conflicting")
        else:
            unknowns.append("context limit is unknown")

        artifacts.append(
            HubArtifact(
                model_id=logical_model_id,
                path="mlx" if explicit_mlx else "safetensors",
                format=format_name,
                size_bytes=size_bytes,
                quantization=quantization,
                runtime_hints=_runtime_hints("/".join(members), format_name, tags),
                evidence=tuple(dict.fromkeys(evidence)),
                context_max=context_max,
                members=members,
                unknowns=tuple(dict.fromkeys(unknowns)),
            )
        )

    return HubRepositoryClassification(
        requested_id=requested,
        repository_id=repository_id,
        logical_model_id=logical_model_id,
        base_model_id=base_model_id,
        base_model_status=base_status,
        context_max=context_max,
        context_status=context_status,
        explicit_quantization=explicit_quant,
        quantization_status=quant_status,
        artifacts=tuple(artifacts),
        evidence=tuple(repository_evidence),
        unknowns=tuple(dict.fromkeys(repository_unknowns)),
    )


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
