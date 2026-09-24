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
    OmlxHfDownloadRecord,
    OmlxModelStatus,
    list_hf_download_records,
    list_model_statuses,
    measure_completion,
)
from .runtime_adapters import OmlxRuntimeAdapter, build_execution_adapters


def _canonical_mlx_artifact_id(requested: str) -> str:
    return f"hf://{requested}/mlx"


def _inventory_target_for_status(
    legacy: Any,
    requested: str,
    selected: OmlxModelStatus,
    *,
    association_kind: str,
    provenance_source: str,
    provenance_detail: str,
    extra_unknowns: Sequence[str] = (),
) -> InventoryTarget:
    observed = legacy.RecommendationEvidence(
        "verified-local-inventory",
        "oMLX model-status API",
        "the model is present in the local oMLX API-visible inventory",
    )
    provenance = legacy.RecommendationEvidence(
        "verified-runtime-provenance",
        provenance_source,
        provenance_detail,
    )
    unknowns = [
        "artifact content digest is unknown",
        "quantization is unknown",
        *extra_unknowns,
    ]
    if selected.loaded is False:
        unknowns.append(
            "the oMLX model is API-visible but is not currently resident in memory"
        )

    record = InventoryRecord(
        runtime="omlx",
        public_artifact_id=_canonical_mlx_artifact_id(requested),
        artifact_fingerprint=None,
        logical_model_id=requested,
        artifact_format="MLX",
        quantization=None,
        association_kind=association_kind,
        identity_attested=False,
        identity_confidence=legacy.Confidence.HIGH,
        identity_evidence=(provenance,),
        evidence=(observed, provenance),
        unknowns=tuple(dict.fromkeys(unknowns)),
    )
    # The API model ID stays private as the execution locator; public solve output
    # uses the canonical resolved MLX artifact identity instead.
    return InventoryTarget(record, selected.model_id)


def inventory_targets_from_statuses(
    legacy: Any,
    requested_logical_model_id: str,
    statuses: Sequence[OmlxModelStatus],
    download_records: Sequence[OmlxHfDownloadRecord] = (),
) -> Tuple[InventoryTarget, ...]:
    """Create one defensible oMLX target or fail closed.

    Preferred evidence is exact ``source_repo_id`` from the detailed status API.
    oMLX 0.6.x dashboard downloads can lose that field and appear as generic local
    directories. For that case only, a completed Hugging Face downloader record may
    establish provenance when the requested repository is exact and its leaf maps
    uniquely to one API-visible local model. Display IDs alone are never accepted.
    """
    requested = str(requested_logical_model_id or "").strip()
    if not requested:
        return ()

    direct_matches = tuple(
        item
        for item in statuses
        if item.source_repo_id is not None
        and item.source_repo_id.strip() == requested
        and item.model_id.strip()
    )
    if direct_matches:
        unique_ids = tuple(sorted({item.model_id.strip() for item in direct_matches}))
        if len(unique_ids) != 1:
            return ()
        selected = next(
            item for item in direct_matches if item.model_id.strip() == unique_ids[0]
        )
        return (
            _inventory_target_for_status(
                legacy,
                requested,
                selected,
                association_kind="runtime_reported_hf_source",
                provenance_source="oMLX model-status source repository metadata",
                provenance_detail=(
                    "oMLX reports an exact Hugging Face source repository match "
                    "for the requested logical model"
                ),
            ),
        )

    completed_repo_ids = {
        item.repo_id.strip()
        for item in download_records
        if item.status.strip().lower() == "completed" and item.repo_id.strip()
    }
    if requested not in completed_repo_ids:
        return ()

    requested_leaf = requested.rsplit("/", 1)[-1]
    if not requested_leaf:
        return ()
    colliding_downloads = {
        repo_id
        for repo_id in completed_repo_ids
        if repo_id.rsplit("/", 1)[-1] == requested_leaf
    }
    if colliding_downloads != {requested}:
        return ()

    fallback_matches = tuple(
        item
        for item in statuses
        if item.source_repo_id is None
        and (item.source_type or "").strip().lower() == "local"
        and item.model_id.strip() == requested_leaf
    )
    unique_ids = tuple(sorted({item.model_id.strip() for item in fallback_matches}))
    if len(unique_ids) != 1 or len(fallback_matches) != 1:
        return ()

    selected = fallback_matches[0]
    return (
        _inventory_target_for_status(
            legacy,
            requested,
            selected,
            association_kind="runtime_reported_hf_download",
            provenance_source="oMLX completed Hugging Face download registry",
            provenance_detail=(
                "oMLX reports a completed download for the requested Hugging Face "
                "repository and the API-visible local model ID is its unique "
                "repository leaf"
            ),
            extra_unknowns=(
                "model-status source repository metadata is unavailable; association uses completed-download provenance",
            ),
        ),
    )


def observe_omlx_inventory(
    legacy: Any,
    requested_logical_model_id: str,
    *,
    base_url: str = DEFAULT_OMLX_BASE_URL,
) -> Tuple[InventoryTarget, ...]:
    """Observe exact or defensible oMLX provenance without filesystem access."""
    if OmlxRuntimeAdapter._cli_path() is None:
        return ()
    try:
        statuses = list_model_statuses(base_url=base_url)
    except OmlxApiError:
        return ()

    direct = inventory_targets_from_statuses(
        legacy,
        requested_logical_model_id,
        statuses,
    )
    if direct:
        return direct

    try:
        download_records = list_hf_download_records(base_url=base_url)
    except OmlxApiError:
        return ()
    return inventory_targets_from_statuses(
        legacy,
        requested_logical_model_id,
        statuses,
        download_records,
    )


def _merge_omlx_inventory_candidate(legacy: Any, result: Any) -> Any:
    """Join canonical compatibility facts with provenance-backed local oMLX facts.

    The generic solve engine intentionally treats every native inventory record as a
    separate candidate because user-supplied artifacts are not automatically the same
    thing as a resolved Hub artifact. oMLX is different here: this bridge only emits
    a native record after runtime-reported provenance has defensibly associated the
    local model with the exact requested Hugging Face repository. In that narrow
    case, merge the duplicate native/canonical candidates instead of weakening the
    generic rule.
    """
    logical_id = str(getattr(result, "logical_model_id", "") or "").strip()
    if not logical_id:
        return result
    candidate_id = f"omlx:{_canonical_mlx_artifact_id(logical_id)}"
    matches = tuple(
        candidate
        for candidate in result.candidates
        if candidate.candidate_id == candidate_id
    )
    if len(matches) != 2:
        return result

    canonical = next(
        (
            candidate
            for candidate in matches
            if getattr(candidate.assessments.compatibility.state, "value", None)
            == "compatible"
        ),
        None,
    )
    local = next(
        (
            candidate
            for candidate in matches
            if getattr(candidate.assessments.local_availability.state, "value", None)
            == "available"
            and candidate.local_identity_evidence
        ),
        None,
    )
    if canonical is None or local is None or canonical is local:
        return result
    if (
        getattr(canonical.assessments.runtime_availability.state, "value", None)
        != "available"
    ):
        return result

    execution_evidence = (
        legacy.RecommendationEvidence(
            "deterministic-execution-state",
            "LLMRig oMLX provenance join",
            "execution is enabled because the resolved MLX artifact is compatible, the oMLX runtime is available, and runtime-reported provenance establishes the exact local model association",
        ),
    )
    execution = replace(
        canonical.assessments.execution,
        state=type(canonical.assessments.execution.state).EXECUTABLE,
        confidence=legacy.Confidence.HIGH,
        evidence=execution_evidence,
        blockers=(),
        unknowns=(),
    )
    assessments = replace(
        canonical.configuration.assessments,
        local_availability=local.assessments.local_availability,
        execution=execution,
    )
    configuration = replace(canonical.configuration, assessments=assessments)
    recipe = type(canonical.recipe)(
        "already_runnable",
        (
            "The runtime and exact local artifact are available; use --verify to measure this configuration.",
        ),
        (),
        True,
    )
    merged = replace(
        canonical,
        configuration=configuration,
        local_identity_attested=local.local_identity_attested,
        local_identity_confidence=local.local_identity_confidence,
        local_identity_evidence=local.local_identity_evidence,
        recipe=recipe,
    )
    remaining = [
        candidate
        for candidate in result.candidates
        if candidate.candidate_id != candidate_id
    ]
    remaining.append(merged)
    return replace(
        result,
        candidates=tuple(sorted(remaining, key=lambda candidate: candidate.candidate_id)),
    )


def _aggregate_omlx_samples(legacy: Any, configuration: Any, samples: Sequence[Any]) -> Any:
    """Aggregate available oMLX server metrics without inventing missing values."""
    generation = [
        float(item["generation_tps"])
        for item in samples
        if item.get("generation_tps") is not None
    ]
    prompt = [
        float(item["prompt_tps"])
        for item in samples
        if item.get("prompt_tps") is not None
    ]
    latency = [
        float(item["wall_seconds"])
        for item in samples
        if item.get("wall_seconds") is not None
    ]
    if not latency:
        raise RuntimeError("oMLX server did not report usable inference latency")

    warnings = [
        "performance measurement does not establish model quality",
        "oMLX OpenAI API does not expose a per-request KV-context allocation; the measurement uses the server's active context configuration",
    ]
    if len(generation) != len(samples) or len(prompt) != len(samples):
        warnings.append(
            "oMLX omitted per-phase timing metrics for one or more measured runs; latency uses server-reported total_time where needed and unavailable throughput dimensions remain unmeasured"
        )

    return legacy.RaceCompetitor(
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
        prompt_eval_tps=(round(sum(prompt) / len(prompt), 2) if prompt else None),
        total_latency_s=round(sum(latency) / len(latency), 4),
        generated_tokens=sum(int(item["eval_count"]) for item in samples),
        measured_runs=len(samples),
        generation_samples=len(generation),
        prompt_eval_samples=len(prompt),
        latency_samples=len(latency),
        timestamp=legacy.now_iso(),
        evidence=(
            legacy.RecommendationEvidence(
                "measured",
                "oMLX server usage",
                f"{len(samples)} timed run(s) produced server-reported inference metrics",
            ),
        ),
        warnings=tuple(dict.fromkeys(warnings)),
        raw_samples=tuple(dict(item) for item in samples),
    )


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
                raise RuntimeError(
                    "oMLX response model identity did not match the requested API model id"
                )
            if index < workload.warmup_runs:
                continue
            if measurement.completion_tokens is None or measurement.completion_tokens <= 0:
                raise RuntimeError("oMLX response did not report generated-token count")

            if measurement.race_ready:
                sample = measurement.race_sample()
            else:
                latency = measurement.race_latency_s
                if latency is None or latency <= 0:
                    raise RuntimeError(
                        "oMLX server metrics are incomplete for race-v2"
                    )
                sample = {
                    "generation_tps": measurement.generation_tps,
                    "prompt_tps": measurement.prompt_tps,
                    "wall_seconds": latency,
                    "prompt_tokens": measurement.prompt_tokens,
                    "generated_tokens": measurement.completion_tokens,
                    "measurement_source": "omlx-server-usage",
                }
            sample["eval_count"] = measurement.completion_tokens
            if measurement.prompt_tokens is not None:
                sample["prompt_eval_count"] = measurement.prompt_tokens
            samples.append(sample)

        if not samples:
            raise RuntimeError("no measured oMLX benchmark runs completed")
        return _aggregate_omlx_samples(self._legacy, target.configuration, samples)


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
    """Run the existing verification pipeline with registry-provided adapters."""
    from .solve import apply_verification

    adapters = build_execution_adapters(legacy)
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
    from . import solve as solve_module

    original_inventory = legacy._autopilot_explicit_native_inventory
    original_verify = legacy._verify_solve_result
    original_configurations = legacy._solve_verification_configurations
    original_construct = solve_module.solve

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

    def construct(inputs: Any) -> Any:
        return _merge_omlx_inventory_candidate(legacy, original_construct(inputs))

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
    solve_module.solve = construct
    try:
        yield
    finally:
        solve_module.solve = original_construct
        legacy._autopilot_explicit_native_inventory = original_inventory
        legacy._verify_solve_result = original_verify
