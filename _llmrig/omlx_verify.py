"""Focused oMLX inventory and solve-verification bridge.

This module is intentionally scoped to the v0.8 CLI front controller. It lets the
stable legacy solve engine consume provenance-safe oMLX inventory and execute an
oMLX race-v2 measurement only when the user explicitly requests ``solve --verify``.
No install, service-start, download, or filesystem-scan behavior lives here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Iterator, Optional, Sequence, Tuple

from .inventory import InventoryRecord, InventoryTarget
from .omlx_api import (
    DEFAULT_OMLX_BASE_URL,
    OmlxApiError,
    OmlxModelStatus,
    list_model_statuses,
    measure_completion,
)
from .runtime_adapters import OmlxRuntimeAdapter


def inventory_targets_from_statuses(
    legacy: Any,
    requested_logical_model_id: str,
    statuses: Sequence[OmlxModelStatus],
) -> Tuple[InventoryTarget, ...]:
    """Create one exact-provenance oMLX target or fail closed.

    The public model-list display IDs may be aliases, so they are not sufficient to
    assert Hugging Face identity. The detailed status endpoint exposes
    ``source_repo_id``; LLMRig accepts only an exact source-repository match.
    Multiple matching server IDs are treated as ambiguous rather than silently
    choosing one.
    """
    requested = str(requested_logical_model_id or "").strip()
    if not requested:
        return ()
    matches = tuple(
        item
        for item in statuses
        if item.source_repo_id is not None
        and item.source_repo_id.strip() == requested
        and item.model_id.strip()
    )
    unique_ids = tuple(sorted({item.model_id.strip() for item in matches}))
    if len(unique_ids) != 1:
        return ()

    selected = next(item for item in matches if item.model_id.strip() == unique_ids[0])
    observed = legacy.RecommendationEvidence(
        "verified-local-inventory",
        "oMLX model-status API",
        "the model is present in the local oMLX API-visible inventory",
    )
    provenance = legacy.RecommendationEvidence(
        "verified-runtime-provenance",
        "oMLX model-status source repository metadata",
        "oMLX reports an exact Hugging Face source repository match for the requested logical model",
    )
    unknowns = [
        "artifact content digest is unknown",
        "quantization is unknown",
    ]
    if selected.loaded is False:
        unknowns.append("the oMLX model is API-visible but is not currently resident in memory")

    record = InventoryRecord(
        runtime="omlx",
        public_artifact_id=selected.model_id,
        artifact_fingerprint=None,
        logical_model_id=requested,
        artifact_format="MLX",
        quantization=None,
        association_kind="runtime_reported_hf_source",
        identity_attested=False,
        identity_confidence=legacy.Confidence.HIGH,
        identity_evidence=(provenance,),
        evidence=(observed, provenance),
        unknowns=tuple(unknowns),
    )
    # The model ID is an API routing identifier, not a filesystem path.
    return (InventoryTarget(record, selected.model_id),)


def observe_omlx_inventory(
    legacy: Any,
    requested_logical_model_id: str,
    *,
    base_url: str = DEFAULT_OMLX_BASE_URL,
) -> Tuple[InventoryTarget, ...]:
    """Observe exact-provenance oMLX inventory without filesystem access."""
    if OmlxRuntimeAdapter._cli_path() is None:
        return ()
    try:
        statuses = list_model_statuses(base_url=base_url)
    except OmlxApiError:
        return ()
    return inventory_targets_from_statuses(legacy, requested_logical_model_id, statuses)


class OmlxExecutionAdapter:
    """Bounded race-v2 execution over oMLX's OpenAI-compatible API."""

    runtime = "omlx"

    def __init__(
        self,
        legacy: Any,
        base_url: str = DEFAULT_OMLX_BASE_URL,
    ) -> None:
        self._legacy = legacy
        self.base_url = base_url.rstrip("/")

    def benchmark(self, target: Any, workload: Any) -> Any:
        if not isinstance(target, self._legacy.ExecutionTarget):
            raise RuntimeError("oMLX execution requires a private local target")
        model_id = str(getattr(target, "_locator", "") or "").strip()
        if not model_id:
            raise RuntimeError("oMLX execution target is missing its private API model id")

        samples = []
        total_runs = workload.warmup_runs + workload.runs
        for index in range(total_runs):
            tokens = (
                min(workload.warmup_num_predict, workload.num_predict)
                if index < workload.warmup_runs
                else workload.num_predict
            )
            try:
                measurement = measure_completion(
                    model_id,
                    workload.prompt,
                    tokens,
                    seed=workload.seed,
                    base_url=self.base_url,
                    timeout=workload.request_timeout_s,
                )
            except (OmlxApiError, ValueError) as error:
                raise RuntimeError("oMLX measured execution failed") from error
            if measurement.model_id != model_id:
                raise RuntimeError("oMLX response model identity did not match the requested API model id")
            if index < workload.warmup_runs:
                continue
            if measurement.completion_tokens is None or measurement.completion_tokens <= 0:
                raise RuntimeError("oMLX response did not report generated-token count")
            try:
                sample = measurement.race_sample()
            except OmlxApiError as error:
                raise RuntimeError("oMLX server metrics are incomplete for race-v2") from error
            sample["eval_count"] = measurement.completion_tokens
            if measurement.prompt_tokens is not None:
                sample["prompt_eval_count"] = measurement.prompt_tokens
            samples.append(sample)

        if not samples:
            raise RuntimeError("no measured oMLX benchmark runs completed")
        competitor = self._legacy.native_race_competitor(
            target.configuration, samples, workload
        )
        return replace(
            competitor,
            warnings=tuple(
                dict.fromkeys(
                    competitor.warnings
                    + (
                        "oMLX OpenAI API does not expose a per-request KV-context allocation; the measurement uses the server's active context configuration",
                    )
                )
            ),
        )


def _verification_configurations_with_omlx(
    legacy: Any,
    original_configurations: Any,
    result: Any,
    inputs: Any,
    ollama_targets: Sequence[Any],
    native_targets: Sequence[Any],
    adapters: Sequence[Any],
) -> Tuple[Any, Any, Any, Any]:
    eligible, ineligible, execution_targets, candidate_map = original_configurations(
        result, inputs, ollama_targets, native_targets, adapters
    )
    targets = list(execution_targets)
    known_identities = {
        (item.configuration.runtime, item.configuration.artifact_id) for item in targets
    }
    for configuration in eligible:
        if configuration.runtime != "omlx":
            continue
        identity = (configuration.runtime, configuration.artifact_id)
        if identity in known_identities:
            continue
        inventory_target = next(
            (
                item
                for item in native_targets
                if item.record.runtime == "omlx"
                and item.record.public_artifact_id == configuration.artifact_id
            ),
            None,
        )
        if inventory_target is None:
            continue
        targets.append(
            legacy._execution_target_from_inventory(inventory_target, configuration)
        )
        known_identities.add(identity)
    return eligible, ineligible, tuple(targets), candidate_map


def verify_solve_result_with_omlx(
    legacy: Any,
    original_configurations: Any,
    result: Any,
    inputs: Any,
    profile: Any,
    ollama_targets: Sequence[Any],
    native_targets: Sequence[Any],
) -> Any:
    """Run the existing verification pipeline with an additional oMLX adapter."""
    from .solve import apply_verification

    adapters = (
        legacy.OllamaExecutionAdapter(legacy.DEFAULT_OLLAMA_HOST),
        legacy.LlamaCppExecutionAdapter(),
        legacy.MlxExecutionAdapter(),
        OmlxExecutionAdapter(legacy),
    )
    eligible, ineligible, execution_targets, candidate_map = (
        _verification_configurations_with_omlx(
            legacy,
            original_configurations,
            result,
            inputs,
            ollama_targets,
            native_targets,
            adapters,
        )
    )
    context = inputs.request.context_tokens or legacy.RACE_CONTEXT
    workload = legacy.RaceWorkload(
        prompt=legacy.SPEED_PROMPT,
        context=context,
        num_predict=legacy.RACE_NUM_PREDICT,
    )
    validation_error = legacy.race_workload_error(
        workload.context, workload.num_predict, workload.runs
    )
    if validation_error is not None:
        race = legacy.RaceResult(
            "unavailable",
            inputs.logical_model_id,
            "requested verification workload is unsupported: %s" % validation_error,
            legacy.RACE_METHOD_VERSION,
            legacy.now_iso(),
            workload,
            legacy.race_hardware_summary(profile),
            eligible,
            ineligible,
        )
    else:
        race = legacy.execute_race(
            inputs.logical_model_id,
            eligible,
            ineligible,
            adapters,
            workload,
            legacy.race_hardware_summary(profile),
            execution_targets=execution_targets,
        )
    decision = legacy.analyze_decision(race, "balanced")
    return apply_verification(
        result,
        race,
        decision,
        candidate_map,
        legacy.RecommendationEvidence,
        legacy.Confidence.HIGH,
        measurement_capability_unavailable=validation_error is not None,
    )


@contextmanager
def omlx_verify_for_legacy(legacy: Any) -> Iterator[None]:
    """Temporarily add oMLX inventory and explicit verification to legacy solve."""
    original_inventory = legacy._autopilot_explicit_native_inventory
    original_verify = legacy._verify_solve_result
    original_configurations = legacy._solve_verification_configurations

    def inventory(logical_model_id: str, values: Sequence[str]) -> Tuple[Any, ...]:
        native = tuple(original_inventory(logical_model_id, values))
        omlx = observe_omlx_inventory(legacy, logical_model_id)
        return tuple(
            sorted(
                native + omlx,
                key=lambda item: (
                    item.record.runtime,
                    item.record.public_artifact_id,
                ),
            )
        )

    def verify(
        result: Any,
        inputs: Any,
        profile: Any,
        ollama_targets: Sequence[Any],
        native_targets: Sequence[Any],
    ) -> Any:
        return verify_solve_result_with_omlx(
            legacy,
            original_configurations,
            result,
            inputs,
            profile,
            ollama_targets,
            native_targets,
        )

    legacy._autopilot_explicit_native_inventory = inventory
    legacy._verify_solve_result = verify
    try:
        yield
    finally:
        legacy._autopilot_explicit_native_inventory = original_inventory
        legacy._verify_solve_result = original_verify
