"""Fail-closed oMLX provenance fallback for artifacts acquired by LLMRig."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence, Tuple

from .acquisition import AcquisitionRecord, completed_acquisitions
from .inventory import InventoryRecord, InventoryTarget
from .omlx_api import (
    DEFAULT_OMLX_BASE_URL,
    OmlxApiError,
    OmlxModelStatus,
    list_model_statuses,
)


def inventory_targets_from_llmrig_acquisitions(
    legacy: Any,
    requested_logical_model_id: str,
    statuses: Sequence[OmlxModelStatus],
    acquisitions: Sequence[AcquisitionRecord],
) -> Tuple[InventoryTarget, ...]:
    requested = str(requested_logical_model_id or "").strip()
    if not requested or "/" not in requested:
        return ()

    completed = tuple(
        item
        for item in acquisitions
        if item.status == "completed" and item.runtime == "omlx"
    )
    if not any(item.repository_id == requested for item in completed):
        return ()

    requested_leaf = requested.rsplit("/", 1)[-1]
    colliding_repositories = {
        item.repository_id
        for item in completed
        if item.repository_id.rsplit("/", 1)[-1] == requested_leaf
    }
    if colliding_repositories != {requested}:
        return ()

    matches = tuple(
        item
        for item in statuses
        if item.source_repo_id is None
        and (item.source_type or "").strip().lower() == "local"
        and item.model_id.strip() == requested_leaf
    )
    if len(matches) != 1:
        return ()
    selected = matches[0]

    acquisition = max(
        (item for item in completed if item.repository_id == requested),
        key=lambda item: (item.completed_at, item.revision),
    )
    provenance = legacy.RecommendationEvidence(
        "verified-runtime-provenance",
        "LLMRig completed Hugging Face acquisition record",
        (
            "LLMRig previously completed an exact Hugging Face acquisition for "
            "the requested repository at a pinned commit, and the oMLX API-visible "
            "local model ID is its unique repository leaf"
        ),
    )
    observed = legacy.RecommendationEvidence(
        "verified-local-inventory",
        "oMLX model-status API",
        "the model is present in the local oMLX API-visible inventory",
    )
    record = InventoryRecord(
        runtime="omlx",
        public_artifact_id=f"hf://{requested}/mlx",
        artifact_fingerprint=None,
        logical_model_id=requested,
        artifact_format="MLX",
        quantization=None,
        association_kind="llmrig_hf_acquisition",
        identity_attested=False,
        identity_confidence=legacy.Confidence.HIGH,
        identity_evidence=(provenance,),
        evidence=(observed, provenance),
        unknowns=(
            "artifact content digest is unknown",
            "quantization is unknown",
            "oMLX model-status source repository metadata is unavailable; association uses LLMRig acquisition provenance",
            f"acquired Hugging Face revision: {acquisition.revision}",
        ),
    )
    return (InventoryTarget(record, selected.model_id),)


def observe_llmrig_acquired_omlx_inventory(
    legacy: Any,
    requested_logical_model_id: str,
    *,
    base_url: str = DEFAULT_OMLX_BASE_URL,
) -> Tuple[InventoryTarget, ...]:
    try:
        statuses = list_model_statuses(base_url=base_url)
    except OmlxApiError:
        return ()
    acquisitions = completed_acquisitions(runtime="omlx")
    return inventory_targets_from_llmrig_acquisitions(
        legacy,
        requested_logical_model_id,
        statuses,
        acquisitions,
    )


@contextmanager
def omlx_acquisition_for_legacy(legacy: Any) -> Iterator[None]:
    """Add LLMRig acquisition provenance only when stronger oMLX evidence is absent."""

    original_inventory = legacy._autopilot_explicit_native_inventory

    def inventory(logical_model_id: str, values: Sequence[str]) -> Tuple[Any, ...]:
        observed = tuple(original_inventory(logical_model_id, values))
        if any(item.record.runtime == "omlx" for item in observed):
            return observed
        fallback = observe_llmrig_acquired_omlx_inventory(legacy, logical_model_id)
        return tuple(
            sorted(
                observed + fallback,
                key=lambda item: (
                    item.record.runtime,
                    item.record.public_artifact_id,
                ),
            )
        )

    legacy._autopilot_explicit_native_inventory = inventory
    try:
        yield
    finally:
        legacy._autopilot_explicit_native_inventory = original_inventory
