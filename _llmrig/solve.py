"""Read-only Autopilot solve domain and candidate-state construction."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .planning import (
    CandidateAssessment,
    CandidateAssessments,
    CompatibilityState,
    DiscoveryState,
    ExecutionState,
    LocalAvailabilityState,
    MeasurementCapabilityState,
    MeasurementState,
    PlanningCandidate,
    RecommendationState,
    RuntimeAvailabilityState,
    _confidence_value,
    _evidence_payloads,
)
from .privacy import validate_public_text


SOLVE_SCHEMA_VERSION = "1.1"
_RECIPE_STATUSES = {
    "already_runnable",
    "known_setup_path",
    "manual_action_required",
    "unsupported_setup_path",
    "unknown",
}


@dataclass(frozen=True)
class SolveRequest:
    """Privacy-safe public portion of a solve request."""

    model: Optional[str]
    context_tokens: Optional[int] = None
    local_artifact_runtimes: Tuple[str, ...] = ()
    verify: bool = False

    def __post_init__(self) -> None:
        if self.model is not None:
            validate_public_text(self.model, "solve model identifier", identity=True)
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("context tokens must be a positive integer")
        runtimes = tuple(sorted(dict.fromkeys(self.local_artifact_runtimes)))
        for runtime in runtimes:
            validate_public_text(runtime, "solve local artifact runtime")
        object.__setattr__(self, "local_artifact_runtimes", runtimes)
        if not isinstance(self.verify, bool):
            raise ValueError("solve verify must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "context_tokens": self.context_tokens,
            "local_artifacts": [
                {"runtime": runtime, "private_locator_required": True}
                for runtime in self.local_artifact_runtimes
            ],
            "verify": self.verify,
        }


@dataclass(frozen=True)
class ReproducibleRecipe:
    """Read-only instructions for reaching or using a candidate state."""

    status: str
    steps: Tuple[str, ...]
    blockers: Tuple[str, ...] = ()
    private_locator_required: bool = False

    def __post_init__(self) -> None:
        if self.status not in _RECIPE_STATUSES:
            raise ValueError("solve recipe uses an unsupported status")
        for label, values in (("steps", self.steps), ("blockers", self.blockers)):
            normalized = tuple(dict.fromkeys(item.strip() for item in values))
            if label == "blockers":
                normalized = tuple(sorted(normalized))
            if any(not item for item in normalized):
                raise ValueError(f"solve recipe {label} must be non-empty strings")
            for item in normalized:
                validate_public_text(item, f"solve recipe {label}")
            object.__setattr__(self, label, normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "steps": list(self.steps),
            "blockers": list(self.blockers),
            "private_locator_required": self.private_locator_required,
        }


@dataclass(frozen=True)
class SolveCandidate:
    """A configuration candidate with independent truth dimensions."""

    candidate_id: str
    configuration: PlanningCandidate
    context_tokens: Optional[int]
    local_identity_attested: Optional[bool]
    local_identity_confidence: Any
    local_identity_evidence: Tuple[Any, ...]
    recipe: ReproducibleRecipe
    measurement_result: Optional["SolveMeasurement"] = None

    def __post_init__(self) -> None:
        validate_public_text(self.candidate_id, "solve candidate id", identity=True)
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("solve candidate context must be positive")
        _confidence_value(self.local_identity_confidence)
        _evidence_payloads(self.local_identity_evidence)

    @property
    def assessments(self) -> CandidateAssessments:
        return self.configuration.assessments

    def to_dict(self) -> Dict[str, Any]:
        payload = self.configuration.to_dict()
        payload.update(
            {
                "candidate_id": self.candidate_id,
                "context_tokens": self.context_tokens,
                "local_identity": {
                    "attested": self.local_identity_attested,
                    "confidence": _confidence_value(self.local_identity_confidence),
                    "evidence": list(_evidence_payloads(self.local_identity_evidence)),
                },
                "recipe": self.recipe.to_dict(),
                "measurement_result": (
                    self.measurement_result.to_dict()
                    if self.measurement_result is not None
                    else None
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class SolveMeasurement:
    """Privacy-safe measured race evidence for one solve candidate."""

    method_version: str
    generation_tps: Optional[float]
    prompt_eval_tps: Optional[float]
    total_latency_s: Optional[float]
    measured_runs: int
    timestamp: str
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_public_text(self.method_version, "solve measurement method")
        validate_public_text(self.timestamp, "solve measurement timestamp")
        if self.measured_runs < 1:
            raise ValueError("solve measurement requires at least one measured run")
        for value in (
            self.generation_tps,
            self.prompt_eval_tps,
            self.total_latency_s,
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(
                    "solve measurement metrics must be positive and finite when known"
                )
        normalized = tuple(sorted(dict.fromkeys(item.strip() for item in self.warnings)))
        for item in normalized:
            validate_public_text(item, "solve measurement warning")
        object.__setattr__(self, "warnings", normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_version": self.method_version,
            "generation_tps": self.generation_tps,
            "prompt_eval_tps": self.prompt_eval_tps,
            "total_latency_s": self.total_latency_s,
            "measured_runs": self.measured_runs,
            "timestamp": self.timestamp,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SolvePlan:
    """Evidence-bounded recommendation over the currently known candidates."""

    recommendation_status: str
    recommended_candidate_id: Optional[str]
    reason: str
    unknowns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.recommendation_status not in {"recommended", "inconclusive"}:
            raise ValueError("solve plan recommendation status is invalid")
        if self.recommendation_status == "recommended" and not self.recommended_candidate_id:
            raise ValueError("recommended solve plan requires a candidate id")
        if self.recommendation_status == "inconclusive" and self.recommended_candidate_id:
            raise ValueError("inconclusive solve plan cannot identify a recommended candidate")
        validate_public_text(self.recommended_candidate_id, "recommended candidate id")
        validate_public_text(self.reason, "solve plan reason")
        normalized = tuple(sorted(dict.fromkeys(item.strip() for item in self.unknowns)))
        for item in normalized:
            validate_public_text(item, "solve plan unknown")
        object.__setattr__(self, "unknowns", normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_status": self.recommendation_status,
            "recommended_candidate_id": self.recommended_candidate_id,
            "reason": self.reason,
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class SolveVerification:
    """Outcome of the explicitly requested race-v2 verification step."""

    status: str
    reason: Optional[str]
    method_version: Optional[str]
    context_tokens: Optional[int]
    num_predict: Optional[int]
    candidate_ids: Tuple[str, ...]
    recommendation: SolvePlan
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"not_requested", "unavailable", "completed", "failed"}:
            raise ValueError("solve verification status is invalid")
        validate_public_text(self.reason, "solve verification reason")
        validate_public_text(self.method_version, "solve verification method")
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("solve verification context must be positive")
        if self.num_predict is not None and self.num_predict <= 0:
            raise ValueError("solve verification token count must be positive")
        candidate_ids = tuple(sorted(dict.fromkeys(self.candidate_ids)))
        for item in candidate_ids:
            validate_public_text(item, "solve verification candidate id", identity=True)
        warnings = tuple(sorted(dict.fromkeys(item.strip() for item in self.warnings)))
        for item in warnings:
            validate_public_text(item, "solve verification warning")
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "method_version": self.method_version,
            "workload": {
                "context_tokens": self.context_tokens,
                "num_predict": self.num_predict,
            },
            "candidate_ids": list(self.candidate_ids),
            "recommendation": self.recommendation.to_dict(),
            "warnings": list(self.warnings),
        }


def verification_not_requested() -> SolveVerification:
    return SolveVerification(
        "not_requested",
        None,
        None,
        None,
        None,
        (),
        SolvePlan(
            "inconclusive",
            None,
            "Measured run recommendation was not requested.",
            ("performance has not been measured",),
        ),
    )


@dataclass(frozen=True)
class SolveResult:
    """Schema-versioned result shared by JSON and human rendering."""

    status: str
    request: SolveRequest
    resolution_status: str
    resolution_confidence: Any
    logical_model_id: Optional[str]
    model_name: Optional[str]
    candidates: Tuple[SolveCandidate, ...]
    plan: SolvePlan
    unknowns: Tuple[str, ...] = ()
    error: Optional[str] = None
    schema_version: str = SOLVE_SCHEMA_VERSION
    verification: Optional[SolveVerification] = None

    def __post_init__(self) -> None:
        if self.status not in {"analyzed", "invalid_request", "engine_failure"}:
            raise ValueError("solve result status is invalid")
        validate_public_text(self.resolution_status, "solve resolution status")
        _confidence_value(self.resolution_confidence)
        validate_public_text(self.logical_model_id, "solve logical model id", identity=True)
        validate_public_text(self.model_name, "solve model name", identity=True)
        validate_public_text(self.error, "solve error")
        normalized = tuple(sorted(dict.fromkeys(item.strip() for item in self.unknowns)))
        for item in normalized:
            validate_public_text(item, "solve result unknown")
        object.__setattr__(self, "unknowns", normalized)
        if self.verification is None:
            object.__setattr__(self, "verification", verification_not_requested())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "request": self.request.to_dict(),
            "resolution": {
                "status": self.resolution_status,
                "confidence": _confidence_value(self.resolution_confidence),
                "logical_model_id": self.logical_model_id,
                "model_name": self.model_name,
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "plan": self.plan.to_dict(),
            "verification": self.verification.to_dict(),
            "unknowns": list(self.unknowns),
            "error": self.error,
        }


@dataclass(frozen=True)
class SolveInputs:
    """Resolved read-only observations supplied by the public facade."""

    request: SolveRequest
    logical_model_id: str
    model_name: str
    resolution_status: str
    resolution_confidence: Any
    resolution_evidence: Tuple[Any, ...]
    compatibility: Any
    artifact_compatibilities: Tuple[Any, ...]
    artifacts: Tuple[Any, ...]
    runtime_candidates: Tuple[Any, ...]
    capabilities: Tuple[Any, ...]
    matched_ollama_records: Tuple[Tuple[str, Any], ...]
    ollama_inventory_conclusive: bool
    explicit_native_records: Tuple[Any, ...]
    evidence_factory: Callable[[str, str, str], Any]
    high_confidence: Any
    unknown_confidence: Any


def _known(
    state: Any,
    confidence: Any,
    evidence: Sequence[Any],
    blockers: Sequence[str] = (),
    unknowns: Sequence[str] = (),
) -> CandidateAssessment[Any]:
    return CandidateAssessment(
        state, confidence, tuple(evidence), tuple(blockers), tuple(unknowns)
    )


def _unknown(
    state: Any, confidence: Any, *unknowns: str
) -> CandidateAssessment[Any]:
    return CandidateAssessment(state, confidence, unknowns=tuple(unknowns))


def _capability_for(runtime: str, capabilities: Sequence[Any]) -> Optional[Any]:
    return next((item for item in capabilities if item.runtime == runtime), None)


def _runtime_candidate_for(
    artifact_id: str, runtime: str, values: Sequence[Any]
) -> Optional[Any]:
    return next(
        (
            item
            for item in values
            if item.artifact_id == artifact_id and item.runtime == runtime
        ),
        None,
    )


def _compatibility_assessment(
    inputs: SolveInputs,
    artifact: Any,
    artifact_compatibility: Optional[Any],
    runtime_candidate: Optional[Any],
    *,
    explicit_native: bool,
) -> CandidateAssessment[CompatibilityState]:
    if explicit_native:
        if inputs.request.verify and getattr(inputs.compatibility, "can_run", None) is True:
            evidence = (
                inputs.evidence_factory(
                    "estimated-native-verification-eligibility",
                    "resolved logical-model compatibility and explicit local structure",
                    "the existing race verifier may attempt this user-associated native target; its exact content, size, and quantization remain unverified",
                ),
            )
            unknowns = (
                "native artifact size and content identity are not independently verified",
                "native artifact quantization is unknown",
            )
            if inputs.request.context_tokens is not None:
                unknowns += (
                    "the requested context limit is not independently verified for this native artifact",
                )
            return _known(
                CompatibilityState.COMPATIBLE,
                inputs.high_confidence,
                evidence,
                unknowns=unknowns,
            )
        unknowns = [
            "static compatibility of the user-supplied native artifact is unknown",
            "native artifact size and content identity are not independently verified",
        ]
        if inputs.request.context_tokens is not None:
            unknowns.append("the requested context limit is unknown for this native artifact")
        return _unknown(
            CompatibilityState.UNKNOWN, inputs.unknown_confidence, *unknowns
        )

    context = inputs.request.context_tokens
    context_max = getattr(artifact, "context_max", None)
    if context is not None and context_max is not None and context > context_max:
        evidence = (
            inputs.evidence_factory(
                "deterministic-context-check",
                "resolved artifact context metadata",
                "the requested context exceeds the artifact's reported context limit",
            ),
        )
        return _known(
            CompatibilityState.INCOMPATIBLE,
            inputs.high_confidence,
            evidence,
            ("requested context exceeds the reported artifact limit",),
        )
    if context is not None and context_max is None:
        return _unknown(
            CompatibilityState.UNKNOWN,
            inputs.unknown_confidence,
            "the requested context cannot be checked because the artifact context limit is unknown",
        )

    if runtime_candidate is not None:
        if runtime_candidate.support_status in {
            "platform_incompatible",
            "architecture_incompatible",
        } or runtime_candidate.fit_result == "too_large":
            return _known(
                CompatibilityState.INCOMPATIBLE,
                runtime_candidate.confidence,
                runtime_candidate.evidence,
                runtime_candidate.blockers
                or ("the candidate is incompatible with the detected machine",),
                runtime_candidate.unknowns,
            )
        if runtime_candidate.fit_result == "fits":
            return _known(
                CompatibilityState.COMPATIBLE,
                runtime_candidate.confidence,
                runtime_candidate.evidence,
                unknowns=runtime_candidate.unknowns,
            )
        return _unknown(
            CompatibilityState.UNKNOWN,
            inputs.unknown_confidence,
            "artifact fit is unknown for this machine",
        )

    if getattr(artifact, "runtime", None) and artifact_compatibility is not None:
        if artifact_compatibility.can_run is True:
            return _known(
                CompatibilityState.COMPATIBLE,
                artifact_compatibility.confidence,
                artifact_compatibility.evidence,
                unknowns=artifact_compatibility.unknowns,
            )
        if artifact_compatibility.can_run is False:
            return _known(
                CompatibilityState.INCOMPATIBLE,
                artifact_compatibility.confidence,
                artifact_compatibility.evidence,
                (
                    artifact_compatibility.reason
                    or "static compatibility is incompatible",
                ),
                artifact_compatibility.unknowns,
            )
    return _unknown(
        CompatibilityState.UNKNOWN,
        inputs.unknown_confidence,
        "static compatibility is unresolved for this candidate",
    )


def _runtime_assessment(
    inputs: SolveInputs, capability: Optional[Any]
) -> CandidateAssessment[RuntimeAvailabilityState]:
    if capability is None:
        return _unknown(
            RuntimeAvailabilityState.UNKNOWN,
            inputs.unknown_confidence,
            "runtime availability is unknown",
        )
    state = (
        RuntimeAvailabilityState.AVAILABLE
        if capability.available
        else RuntimeAvailabilityState.UNAVAILABLE
    )
    blockers = () if capability.available else ("runtime is not currently available",)
    return _known(
        state,
        capability.confidence,
        capability.evidence,
        blockers,
        capability.unknowns,
    )


def _local_assessment(
    inputs: SolveInputs,
    runtime: str,
    artifact_id: str,
    *,
    explicit_record: Optional[Any],
) -> Tuple[
    CandidateAssessment[LocalAvailabilityState], Optional[bool], Any, Tuple[Any, ...]
]:
    if explicit_record is not None:
        return (
            _known(
                LocalAvailabilityState.AVAILABLE,
                inputs.high_confidence,
                explicit_record.evidence,
                unknowns=explicit_record.unknowns,
            ),
            explicit_record.identity_attested,
            explicit_record.identity_confidence,
            explicit_record.identity_evidence,
        )
    matched = tuple(
        record
        for matched_artifact_id, record in inputs.matched_ollama_records
        if matched_artifact_id == artifact_id
    )
    if runtime == "ollama" and matched:
        record = matched[0]
        return (
            _known(
                LocalAvailabilityState.AVAILABLE,
                inputs.high_confidence,
                record.evidence,
                unknowns=record.unknowns,
            ),
            record.identity_attested,
            record.identity_confidence,
            record.identity_evidence,
        )
    if runtime == "ollama" and inputs.ollama_inventory_conclusive:
        evidence = (
            inputs.evidence_factory(
                "verified-local-inventory",
                "Ollama installed-model listing",
                "the listing returned installed artifacts but none matched this candidate",
            ),
        )
        return (
            _known(
                LocalAvailabilityState.NOT_AVAILABLE,
                inputs.high_confidence,
                evidence,
                ("the artifact is not present in the observed Ollama inventory",),
            ),
            None,
            inputs.unknown_confidence,
            (),
        )
    unknown = (
        "an empty Ollama observation cannot prove that the artifact is absent"
        if runtime == "ollama"
        else "local availability is unknown because no equivalent explicit native locator was inspected"
    )
    return (
        _unknown(LocalAvailabilityState.UNKNOWN, inputs.unknown_confidence, unknown),
        None,
        inputs.unknown_confidence,
        (),
    )


def _execution_assessment(
    inputs: SolveInputs,
    compatibility: CandidateAssessment[CompatibilityState],
    runtime: CandidateAssessment[RuntimeAvailabilityState],
    local: CandidateAssessment[LocalAvailabilityState],
    capability: Optional[Any],
) -> CandidateAssessment[ExecutionState]:
    evidence = (
        inputs.evidence_factory(
            "deterministic-execution-state",
            "LLMRig solve state composition",
            "execution state is derived from separate compatibility, runtime, local-artifact, and LLMRig support facts",
        ),
    )
    blockers = []
    if compatibility.state == CompatibilityState.INCOMPATIBLE:
        blockers.append("candidate compatibility is incompatible")
    if runtime.state == RuntimeAvailabilityState.UNAVAILABLE:
        blockers.append("runtime is not currently available")
    if local.state == LocalAvailabilityState.NOT_AVAILABLE:
        blockers.append("artifact is not locally available")
    if capability is not None and not capability.llmrig_execution_supported:
        blockers.append("LLMRig does not support execution through this runtime")
    if blockers:
        return _known(
            ExecutionState.NOT_EXECUTABLE,
            inputs.high_confidence,
            evidence,
            tuple(blockers),
        )
    unknowns = []
    if compatibility.state == CompatibilityState.UNKNOWN:
        unknowns.append("candidate compatibility is unknown")
    if runtime.state == RuntimeAvailabilityState.UNKNOWN:
        unknowns.append("runtime availability is unknown")
    if local.state == LocalAvailabilityState.UNKNOWN:
        unknowns.append("local artifact availability is unknown")
    if capability is None:
        unknowns.append("LLMRig runtime execution support is unknown")
    if unknowns:
        return _unknown(ExecutionState.UNKNOWN, inputs.unknown_confidence, *unknowns)
    return _known(
        ExecutionState.EXECUTABLE,
        inputs.high_confidence,
        evidence,
        unknowns=("successful inference has not been verified",),
    )


def _measurement_capability_assessment(
    inputs: SolveInputs, capability: Optional[Any]
) -> CandidateAssessment[MeasurementCapabilityState]:
    if capability is None:
        return _unknown(
            MeasurementCapabilityState.UNKNOWN,
            inputs.unknown_confidence,
            "measurement support is unknown",
        )
    evidence = (
        inputs.evidence_factory(
            "runtime-measurement-support",
            "LLMRig runtime capability provider",
            (
                "LLMRig has benchmark support for this runtime"
                if capability.llmrig_benchmark_supported
                else "LLMRig has no benchmark support for this runtime"
            ),
        ),
    )
    state = (
        MeasurementCapabilityState.MEASURABLE
        if capability.llmrig_benchmark_supported
        else MeasurementCapabilityState.NOT_MEASURABLE
    )
    blockers = (
        ()
        if capability.llmrig_benchmark_supported
        else ("LLMRig benchmark support is unavailable",)
    )
    return _known(state, inputs.high_confidence, evidence, blockers)


def _recipe(
    inputs: SolveInputs,
    artifact_id: str,
    runtime_name: str,
    compatibility: CandidateAssessment[CompatibilityState],
    runtime: CandidateAssessment[RuntimeAvailabilityState],
    local: CandidateAssessment[LocalAvailabilityState],
    execution: CandidateAssessment[ExecutionState],
    capability: Optional[Any],
    *,
    explicit_native: bool,
) -> ReproducibleRecipe:
    private_required = explicit_native
    if execution.state == ExecutionState.EXECUTABLE:
        return ReproducibleRecipe(
            "already_runnable",
            ("Use the already-local artifact with the available runtime.",),
            private_locator_required=private_required,
        )
    if compatibility.state == CompatibilityState.INCOMPATIBLE:
        return ReproducibleRecipe(
            "unsupported_setup_path",
            ("Choose a different compatible artifact or context configuration.",),
            compatibility.blockers,
            private_required,
        )
    if capability is not None and not capability.llmrig_execution_supported:
        return ReproducibleRecipe(
            "unsupported_setup_path",
            ("Use a runtime execution path supported outside this LLMRig solve result.",),
            ("LLMRig does not support execution through this runtime",),
            private_required,
        )
    if runtime.state == RuntimeAvailabilityState.UNAVAILABLE:
        return ReproducibleRecipe(
            "manual_action_required",
            ("Install or start the runtime manually, then run solve again.",),
            runtime.blockers,
            private_required,
        )
    if local.state == LocalAvailabilityState.NOT_AVAILABLE and runtime_name == "ollama":
        return ReproducibleRecipe(
            "known_setup_path",
            (
                f"Pull {artifact_id} manually with Ollama; LLMRig has not downloaded it.",
                "Run solve again after the artifact appears in the local inventory.",
            ),
            local.blockers,
            private_required,
        )
    steps = (
        "Supply or verify the unresolved local/runtime facts, then run solve again.",
    )
    if explicit_native:
        steps += (
            "Retain the private locator locally; public solve output intentionally omits it.",
        )
    return ReproducibleRecipe(
        "unknown",
        steps,
        tuple(compatibility.blockers + runtime.blockers + local.blockers),
        private_required,
    )


def _candidate(
    inputs: SolveInputs,
    logical_model_id: str,
    artifact_id: str,
    artifact_format: str,
    quantization: Optional[str],
    runtime_name: str,
    artifact: Any,
    artifact_compatibility: Optional[Any],
    runtime_candidate: Optional[Any],
    explicit_record: Optional[Any] = None,
) -> SolveCandidate:
    explicit_native = explicit_record is not None
    capability = _capability_for(runtime_name, inputs.capabilities)
    discovery_evidence = (
        tuple(explicit_record.evidence)
        if explicit_record is not None
        else tuple(getattr(artifact, "evidence", ())) or inputs.resolution_evidence
    )
    discovery = _known(
        DiscoveryState.DISCOVERED,
        inputs.high_confidence,
        discovery_evidence,
        unknowns=tuple(getattr(artifact, "unknowns", ())),
    )
    compatibility = _compatibility_assessment(
        inputs,
        artifact,
        artifact_compatibility,
        runtime_candidate,
        explicit_native=explicit_native,
    )
    runtime = _runtime_assessment(inputs, capability)
    local, identity_attested, identity_confidence, identity_evidence = _local_assessment(
        inputs, runtime_name, artifact_id, explicit_record=explicit_record
    )
    execution = _execution_assessment(
        inputs, compatibility, runtime, local, capability
    )
    measurement_capability = _measurement_capability_assessment(inputs, capability)
    measurement_requested = inputs.request.verify
    measurement = _known(
        (
            MeasurementState.NOT_MEASURED
            if measurement_requested
            else MeasurementState.NOT_REQUESTED
        ),
        inputs.high_confidence,
        (
            inputs.evidence_factory(
                "solve-policy",
                "solve verification policy",
                (
                    "benchmark verification was requested but has not yet been performed"
                    if measurement_requested
                    else "benchmark verification was not requested or performed"
                ),
            ),
        ),
        unknowns=("performance has not been measured",),
    )
    recommendation = _known(
        RecommendationState.INCONCLUSIVE,
        inputs.high_confidence,
        (
            inputs.evidence_factory(
                "solve-policy",
                "evidence-bounded recommendation policy",
                "recommendation remains inconclusive until all comparable candidates are resolved",
            ),
        ),
    )
    assessments = CandidateAssessments(
        discovery,
        compatibility,
        runtime,
        local,
        execution,
        measurement_capability,
        measurement,
        recommendation,
    )
    context_tokens = inputs.request.context_tokens
    if context_tokens is None and artifact_compatibility is not None:
        context_tokens = artifact_compatibility.recommended_context
    configuration = PlanningCandidate(
        logical_model_id,
        artifact_id,
        runtime_name,
        artifact_format,
        quantization,
        assessments,
    )
    recipe = _recipe(
        inputs,
        artifact_id,
        runtime_name,
        compatibility,
        runtime,
        local,
        execution,
        capability,
        explicit_native=explicit_native,
    )
    return SolveCandidate(
        f"{runtime_name}:{artifact_id}",
        configuration,
        context_tokens,
        identity_attested,
        identity_confidence,
        tuple(identity_evidence),
        recipe,
    )


def solve(inputs: SolveInputs) -> SolveResult:
    """Construct a read-only solve result from existing repository observations."""
    candidates = []
    for artifact in inputs.artifacts:
        artifact_compatibility = next(
            (
                item
                for item in inputs.artifact_compatibilities
                if item.artifact_id == artifact.artifact_id
            ),
            None,
        )
        if getattr(artifact, "runtime", None):
            runtime_names = (artifact.runtime,)
        else:
            runtime_names = tuple(
                sorted(
                    {
                        item.runtime
                        for item in inputs.runtime_candidates
                        if item.artifact_id == artifact.artifact_id
                    }
                )
            ) or ("unknown",)
        for runtime_name in runtime_names:
            candidates.append(
                _candidate(
                    inputs,
                    inputs.logical_model_id,
                    artifact.artifact_id,
                    artifact.format,
                    artifact.quantization,
                    runtime_name,
                    artifact,
                    artifact_compatibility,
                    _runtime_candidate_for(
                        artifact.artifact_id, runtime_name, inputs.runtime_candidates
                    ),
                )
            )

    for record in inputs.explicit_native_records:
        candidates.append(
            _candidate(
                inputs,
                inputs.logical_model_id,
                record.public_artifact_id,
                record.artifact_format or "unknown",
                record.quantization,
                record.runtime,
                record,
                None,
                None,
                record,
            )
        )

    candidates.sort(key=lambda item: item.candidate_id)
    executable = [
        item
        for item in candidates
        if item.assessments.execution.state == ExecutionState.EXECUTABLE
    ]
    unresolved = [
        item
        for item in candidates
        if item.assessments.execution.state == ExecutionState.UNKNOWN
    ]
    recommendation_evidence = inputs.evidence_factory(
        "solve-policy",
        "evidence-bounded recommendation policy",
        "exactly one candidate is runnable and no comparable unresolved candidate remains",
    )
    recommended_id = None
    if len(executable) == 1 and not unresolved:
        recommended_id = executable[0].candidate_id
        updated = []
        for item in candidates:
            state = (
                RecommendationState.RECOMMENDED
                if item.candidate_id == recommended_id
                else RecommendationState.NOT_RECOMMENDED
            )
            recommendation = _known(
                state, inputs.high_confidence, (recommendation_evidence,)
            )
            updated.append(
                replace(
                    item,
                    configuration=replace(
                        item.configuration,
                        assessments=replace(
                            item.assessments, recommendation=recommendation
                        ),
                    ),
                )
            )
        candidates = updated
        plan = SolvePlan(
            "recommended",
            recommended_id,
            "Exactly one configuration is currently runnable and no comparable unresolved alternative remains.",
            ("performance has not been measured",),
        )
    else:
        reason = (
            "Multiple configurations are runnable, so solve cannot select a best runtime without performance evidence."
            if len(executable) > 1
            else "At least one comparable candidate remains unresolved."
            if unresolved
            else "No configuration is currently known to be runnable."
        )
        plan = SolvePlan(
            "inconclusive",
            None,
            reason,
            (
                "performance has not been measured",
                "no performance-based runtime ranking is available",
            ),
        )

    return SolveResult(
        "analyzed",
        inputs.request,
        inputs.resolution_status,
        inputs.resolution_confidence,
        inputs.logical_model_id,
        inputs.model_name,
        tuple(candidates),
        plan,
        unknowns=(
            ("benchmark verification is pending",)
            if inputs.request.verify
            else ("benchmark verification was not performed",)
        ),
    )


def apply_verification(
    result: SolveResult,
    race: Any,
    decision: Any,
    candidate_by_competitor: Dict[Tuple[str, str], str],
    evidence_factory: Callable[[str, str, str], Any],
    high_confidence: Any,
    measurement_capability_unavailable: bool = False,
) -> SolveResult:
    """Fold one existing race-v2 result into the public solve representation."""
    selected_ids = tuple(sorted(set(candidate_by_competitor.values())))
    competitor_by_candidate = {
        candidate_by_competitor[(item.runtime, item.artifact_id)]: item
        for item in race.competitors
        if (item.runtime, item.artifact_id) in candidate_by_competitor
    }
    measured_evidence = evidence_factory(
        "measured",
        f"{race.method_version} local execution",
        "the candidate was measured by the existing race-v2 workload and execution path",
    )
    failed_evidence = evidence_factory(
        "measured-execution-failure",
        f"{race.method_version} local execution",
        "the candidate's benchmark execution failed and invalidated the comparison",
    )
    unavailable_evidence = evidence_factory(
        "verification-unavailable",
        f"{race.method_version} verification policy",
        "the requested measured comparison could not be performed",
    )

    preferred_identity = None
    if decision is not None and decision.preferred is not None:
        preferred_identity = (decision.preferred.runtime, decision.preferred.artifact_id)
    preferred_candidate_id = (
        candidate_by_competitor.get(preferred_identity)
        if preferred_identity is not None
        else None
    )
    measured_recommendation = bool(
        race.status == "completed"
        and decision is not None
        and decision.status == "recommended"
        and preferred_candidate_id is not None
    )

    candidates = []
    for candidate in result.candidates:
        competitor = competitor_by_candidate.get(candidate.candidate_id)
        if competitor is not None and competitor.execution_status == "success":
            measurement_state = _known(
                MeasurementState.MEASURED,
                high_confidence,
                (measured_evidence,) + tuple(competitor.evidence),
            )
            measurement_result = SolveMeasurement(
                race.method_version,
                competitor.generation_tps,
                competitor.prompt_eval_tps,
                competitor.total_latency_s,
                competitor.measured_runs,
                competitor.timestamp,
                tuple(race.warnings) + tuple(competitor.warnings),
            )
        elif competitor is not None:
            measurement_state = _known(
                MeasurementState.FAILED,
                high_confidence,
                (failed_evidence,),
                (competitor.failure or "benchmark execution failed",),
            )
            measurement_result = None
        elif race.status == "unavailable" and candidate.candidate_id in selected_ids:
            measurement_state = _known(
                MeasurementState.VERIFICATION_UNAVAILABLE,
                high_confidence,
                (unavailable_evidence,),
                (race.reason or "verification is unavailable",),
            )
            measurement_result = None
        else:
            measurement_state = _known(
                MeasurementState.NOT_MEASURED,
                high_confidence,
                (unavailable_evidence,),
                unknowns=("this candidate was not measured by the requested comparison",),
            )
            measurement_result = None

        if measured_recommendation:
            recommendation_state = (
                RecommendationState.RECOMMENDED
                if candidate.candidate_id == preferred_candidate_id
                else RecommendationState.NOT_RECOMMENDED
                if candidate.candidate_id in selected_ids
                else RecommendationState.INCONCLUSIVE
            )
        else:
            recommendation_state = RecommendationState.INCONCLUSIVE
        recommendation_evidence = evidence_factory(
            "measured-decision",
            f"{race.method_version} balanced Pareto analysis",
            (
                "the candidate is the unique measured-performance Pareto leader"
                if recommendation_state == RecommendationState.RECOMMENDED
                else "no measured run recommendation is assigned to this candidate"
            ),
        )
        recommendation = _known(
            recommendation_state,
            high_confidence,
            (recommendation_evidence,),
        )
        measurement_capability = candidate.assessments.measurement_capability
        if (
            measurement_capability_unavailable
            and candidate.candidate_id in selected_ids
        ):
            measurement_capability = _known(
                MeasurementCapabilityState.NOT_MEASURABLE,
                high_confidence,
                (unavailable_evidence,),
                (race.reason or "the requested workload is not measurable",),
            )
        candidates.append(
            replace(
                candidate,
                configuration=replace(
                    candidate.configuration,
                    assessments=replace(
                        candidate.assessments,
                        measurement_capability=measurement_capability,
                        measurement=measurement_state,
                        recommendation=recommendation,
                    ),
                ),
                measurement_result=measurement_result,
            )
        )

    if race.status == "completed" and decision is not None:
        recommendation = SolvePlan(
            "recommended" if measured_recommendation else "inconclusive",
            preferred_candidate_id if measured_recommendation else None,
            (
                "A unique measured-performance Pareto leader was identified by race-v2."
                if measured_recommendation
                else decision.reason or "The measured comparison remains inconclusive."
            ),
            () if measured_recommendation else ("multiple measured tradeoffs remain",),
        )
    else:
        recommendation = SolvePlan(
            "inconclusive",
            None,
            (
                race.reason
                or "The requested measured comparison did not complete successfully."
            ),
            ("no measured run recommendation is available",),
        )
    verification = SolveVerification(
        race.status,
        race.reason,
        race.method_version,
        race.workload.context,
        race.workload.num_predict,
        selected_ids,
        recommendation,
        tuple(race.warnings),
    )
    return replace(
        result,
        candidates=tuple(candidates),
        verification=verification,
        unknowns=(
            ()
            if race.status == "completed"
            else ("requested benchmark verification did not complete",)
        ),
    )


def terminal_solve_result(
    request: SolveRequest,
    status: str,
    resolution_status: str,
    unknown_confidence: Any,
    error: str,
) -> SolveResult:
    """Create a privacy-safe invalid/engine-failure result."""
    return SolveResult(
        status,
        request,
        resolution_status,
        unknown_confidence,
        None,
        None,
        (),
        SolvePlan(
            "inconclusive",
            None,
            "No solve plan was constructed.",
            ("model resolution did not complete",),
        ),
        unknowns=("candidate state was not analyzed",),
        error=error,
    )
