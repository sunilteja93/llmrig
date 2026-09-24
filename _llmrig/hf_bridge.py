"""Hugging Face metadata bridge for the stable legacy resolver.

The top-level :mod:`llmrig` module still owns the public v0.7-compatible model
and artifact classes. This bridge lets the v0.8 CLI use the hardened Hub
classifier without rewriting that stable surface. It is read-only and performs
no weight downloads, filesystem scans, installs, or runtime actions.
"""

from __future__ import annotations

import urllib.parse
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator, Mapping, Sequence, Tuple

from .hf_artifacts import (
    base_model_metadata,
    classify_hub_repository,
    context_metadata,
    explicit_quantization_metadata,
)


_CONFIG_HINT_KEYS = {
    "architectures",
    "max_position_embeddings",
    "model_type",
    "quantization",
    "quantization_config",
    "text_config",
}
_MLX_QUANT_MODES = {"affine", "mxfp4", "nvfp4", "mxfp8"}


def _legacy_evidence(legacy: Any, evidence: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(
        legacy.RecommendationEvidence(item.kind, item.source, item.detail)
        for item in evidence
    )


def _artifact_id(repository_id: str, artifact: Any, *, format_name: str) -> str:
    if format_name == "GGUF":
        return f"hf://{repository_id}/{artifact.path}"
    if format_name == "MLX":
        return f"hf://{repository_id}/mlx"
    if format_name == "safetensors":
        return f"hf://{repository_id}/safetensors"
    return f"hf://{repository_id}/{artifact.path}"


def _legacy_format(format_name: str) -> str:
    return "Safetensors" if format_name == "safetensors" else format_name


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _looks_like_model_config(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(_CONFIG_HINT_KEYS.intersection(value))


def _has_root_config_sibling(payload: Mapping[str, Any]) -> bool:
    return any(
        isinstance(item, Mapping)
        and str(item.get("rfilename") or item.get("path") or "") == "config.json"
        for item in (payload.get("siblings") or ())
    )


def _needs_exact_config(payload: Mapping[str, Any]) -> bool:
    _, context_status, _ = context_metadata(payload)
    _, quant_status, _ = explicit_quantization_metadata(payload)
    return context_status == "unknown" or quant_status == "unknown"


def _enrich_exact_config(legacy: Any, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fetch only exact ``config.json`` metadata when the model API omits it.

    The request is pinned to the Hub commit SHA when available. Failure is
    non-fatal: the existing model-API payload remains the only evidence.
    """

    if not _has_root_config_sibling(payload) or not _needs_exact_config(payload):
        return payload

    repository_id = str(payload.get("id") or payload.get("modelId") or "").strip()
    if not repository_id:
        return payload
    revision = str(payload.get("sha") or "main").strip() or "main"
    repo = urllib.parse.quote(repository_id, safe="/")
    ref = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/{repo}/resolve/{ref}/config.json"
    try:
        config_payload, _ = legacy.http_json(url, timeout=20)
    except Exception:
        return payload
    if not _looks_like_model_config(config_payload):
        return payload

    enriched = dict(payload)
    enriched["config"] = dict(config_payload)
    return enriched


def _strong_structured_mlx_evidence(payload: Mapping[str, Any]) -> bool:
    """Recognize MLX packaging only from a tag plus MLX-LM config structure.

    An ``mlx`` tag by itself remains only a hint. The additional structured
    quantization contract mirrors the fields MLX-LM consumes for quantized
    models and is therefore stronger packaging evidence.
    """

    tags = {str(item).strip().lower() for item in (payload.get("tags") or ())}
    if "mlx" not in tags:
        return False
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    quant = config.get("quantization") if isinstance(config, Mapping) else None
    if not isinstance(quant, Mapping):
        return False
    bits = _positive_int(quant.get("bits"))
    group_size = _positive_int(quant.get("group_size"))
    mode = str(quant.get("mode") or "").strip().lower()
    if bits is None or group_size is None or mode not in _MLX_QUANT_MODES:
        return False

    legacy_quant = config.get("quantization_config")
    if isinstance(legacy_quant, Mapping):
        legacy_bits = _positive_int(legacy_quant.get("bits"))
        legacy_group = _positive_int(legacy_quant.get("group_size"))
        legacy_mode = str(legacy_quant.get("mode") or "").strip().lower()
        if legacy_bits is not None and legacy_bits != bits:
            return False
        if legacy_group is not None and legacy_group != group_size:
            return False
        if legacy_mode and legacy_mode != mode:
            return False
    return True


def artifacts_from_metadata(legacy: Any, model: Any, payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Convert hardened repository classification into stable legacy artifacts."""

    payload = _enrich_exact_config(legacy, payload)
    requested = str(payload.get("id") or payload.get("modelId") or model.model_id)
    classification = classify_hub_repository(requested, payload)
    structured_mlx = _strong_structured_mlx_evidence(payload)
    output = []
    for artifact in classification.artifacts:
        effective_format = artifact.format
        extra_evidence = ()
        extra_unknowns = ()
        if artifact.format == "safetensors" and structured_mlx:
            effective_format = "MLX"
            extra_evidence = (
                legacy.RecommendationEvidence(
                    "verified-metadata",
                    "Hugging Face mlx tag and MLX-LM quantization config contract",
                    "the repository combines an mlx tag with structured bits, group_size, and mode metadata used by MLX-LM quantized packaging",
                ),
            )
            extra_unknowns = (
                "MLX packaging is established from structured repository metadata rather than library_name",
            )

        evidence = _legacy_evidence(legacy, artifact.evidence) + extra_evidence
        unknowns = tuple(
            dict.fromkeys((*artifact.unknowns, *classification.unknowns, *extra_unknowns))
        )
        if effective_format == "MLX":
            unknowns = tuple(
                item
                for item in unknowns
                if item
                != "generic safetensors do not establish MLX/oMLX compatibility"
            )
        size_bytes = artifact.size_bytes
        output.append(
            legacy.ModelArtifact(
                artifact_id=_artifact_id(
                    classification.repository_id,
                    artifact,
                    format_name=effective_format,
                ),
                model_id=classification.logical_model_id,
                runtime=None,
                format=_legacy_format(effective_format),
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
