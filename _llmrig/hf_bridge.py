"""Hugging Face metadata bridge for the stable legacy resolver.

The top-level :mod:`llmrig` module still owns the public v0.7-compatible model
and artifact classes.  This bridge lets the v0.8 CLI use the hardened Hub
classifier without rewriting that stable surface.  It is read-only and performs
no downloads, filesystem scans, installs, or runtime actions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence, Tuple

from .hf_artifacts import base_model_metadata, classify_hub_repository


def _legacy_evidence(legacy: Any, evidence: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(
        legacy.RecommendationEvidence(item.kind, item.source, item.detail)
        for item in evidence
    )


def _artifact_id(repository_id: str, artifact: Any) -> str:
    if artifact.format == "GGUF":
        return f"hf://{repository_id}/{artifact.path}"
    if artifact.format == "MLX":
        return f"hf://{repository_id}/mlx"
    if artifact.format == "safetensors":
        return f"hf://{repository_id}/safetensors"
    return f"hf://{repository_id}/{artifact.path}"


def _legacy_format(format_name: str) -> str:
    return "Safetensors" if format_name == "safetensors" else format_name


def artifacts_from_metadata(legacy: Any, model: Any, payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Convert hardened repository classification into stable legacy artifacts."""

    requested = str(payload.get("id") or payload.get("modelId") or model.model_id)
    classification = classify_hub_repository(requested, payload)
    output = []
    for artifact in classification.artifacts:
        evidence = _legacy_evidence(legacy, artifact.evidence)
        unknowns = tuple(
            dict.fromkeys((*artifact.unknowns, *classification.unknowns))
        )
        size_bytes = artifact.size_bytes
        output.append(
            legacy.ModelArtifact(
                artifact_id=_artifact_id(classification.repository_id, artifact),
                model_id=classification.logical_model_id,
                runtime=None,
                format=_legacy_format(artifact.format),
                size_gb=size_bytes / 1_000_000_000 if size_bytes else None,
                context_max=artifact.context_max,
                platforms=(),
                quantization=artifact.quantization,
                size_bytes=size_bytes,
                evidence=evidence,
                unknowns=unknowns,
            )
        )
    return tuple(output)


def hardened_base_model_id(payload: Mapping[str, Any]) -> Any:
    base_model, status = base_model_metadata(payload)
    return base_model if status == "verified" else None


@contextmanager
def hf_metadata_for_legacy(legacy: Any) -> Iterator[None]:
    """Temporarily use hardened Hub metadata semantics for legacy CLI calls."""

    original_artifacts = legacy.hf_artifacts_from_metadata
    original_base_model = legacy.hf_base_model_id

    def artifacts(model: Any, payload: Mapping[str, Any]) -> Tuple[Any, ...]:
        return artifacts_from_metadata(legacy, model, payload)

    legacy.hf_artifacts_from_metadata = artifacts
    legacy.hf_base_model_id = hardened_base_model_id
    try:
        yield
    finally:
        legacy.hf_base_model_id = original_base_model
        legacy.hf_artifacts_from_metadata = original_artifacts
