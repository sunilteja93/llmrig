"""Orthogonal, evidence-aware candidate state for the future solve engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, Optional, Protocol, Tuple, TypeVar

from .privacy import validate_public_text


class ConfidenceLike(Protocol):
    """Structural view of ``llmrig.Confidence`` without importing the facade."""

    @property
    def value(self) -> str: ...


class EvidenceLike(Protocol):
    """Structural view of ``llmrig.RecommendationEvidence``."""

    def to_dict(self) -> Dict[str, str]: ...


class DiscoveryState(str, Enum):
    DISCOVERED = "discovered"
    NOT_DISCOVERED = "not_discovered"
    UNKNOWN = "unknown"


class CompatibilityState(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class RuntimeAvailabilityState(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class LocalAvailabilityState(str, Enum):
    AVAILABLE = "available"
    NOT_AVAILABLE = "not_available"
    UNKNOWN = "unknown"


class ExecutionState(str, Enum):
    EXECUTABLE = "executable"
    NOT_EXECUTABLE = "not_executable"
    UNKNOWN = "unknown"


class MeasurementCapabilityState(str, Enum):
    MEASURABLE = "measurable"
    NOT_MEASURABLE = "not_measurable"
    UNKNOWN = "unknown"


class MeasurementState(str, Enum):
    NOT_REQUESTED = "not_requested"
    NOT_MEASURED = "not_measured"
    MEASURED = "measured"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RecommendationState(str, Enum):
    RECOMMENDED = "recommended"
    NOT_RECOMMENDED = "not_recommended"
    INCONCLUSIVE = "inconclusive"
    UNKNOWN = "unknown"


StateType = TypeVar("StateType", bound=Enum)
_CONFIDENCE_VALUES = {"verified", "high", "medium", "low", "unknown"}


def _confidence_value(confidence: ConfidenceLike) -> str:
    value = getattr(confidence, "value", None)
    if value not in _CONFIDENCE_VALUES:
        raise ValueError("candidate assessment requires a categorical confidence")
    return str(value)


def _normalized_messages(values: Tuple[str, ...], label: str) -> Tuple[str, ...]:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"candidate assessment {label} must be non-empty strings")
    normalized = tuple(value.strip() for value in values)
    for value in normalized:
        validate_public_text(value, f"candidate assessment {label}")
    return tuple(sorted(dict.fromkeys(normalized)))


def _evidence_payloads(evidence: Tuple[EvidenceLike, ...]) -> Tuple[Dict[str, str], ...]:
    payloads = []
    for item in evidence:
        to_dict = getattr(item, "to_dict", None)
        if not callable(to_dict):
            raise ValueError("candidate evidence must use RecommendationEvidence semantics")
        payload = to_dict()
        if not isinstance(payload, dict) or any(
            not isinstance(payload.get(key), str) or not payload[key].strip()
            for key in ("kind", "source", "detail")
        ):
            raise ValueError("candidate evidence must have kind, source, and detail")
        payloads.append(
            {
                "kind": payload["kind"],
                "source": payload["source"],
                "detail": payload["detail"],
            }
        )
        for value in payloads[-1].values():
            validate_public_text(value, "candidate evidence")
    return tuple(
        sorted(payloads, key=lambda item: (item["kind"], item["source"], item["detail"]))
    )


@dataclass(frozen=True)
class CandidateAssessment(Generic[StateType]):
    """One independent candidate fact with confidence and provenance."""

    state: StateType
    confidence: ConfidenceLike
    evidence: Tuple[EvidenceLike, ...] = ()
    blockers: Tuple[str, ...] = ()
    unknowns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, Enum):
            raise ValueError("candidate assessment state must be an enum value")
        confidence = _confidence_value(self.confidence)
        _evidence_payloads(self.evidence)
        blockers = _normalized_messages(self.blockers, "blockers")
        unknowns = _normalized_messages(self.unknowns, "unknowns")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "unknowns", unknowns)
        if self.state.value == "unknown":
            if confidence != "unknown":
                raise ValueError("unknown candidate state requires unknown confidence")
            if not unknowns:
                raise ValueError("unknown candidate state requires an explicit unknown")
        elif not self.evidence:
            raise ValueError("known candidate state requires evidence")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "confidence": _confidence_value(self.confidence),
            "evidence": list(_evidence_payloads(self.evidence)),
            "blockers": list(self.blockers),
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class CandidateAssessments:
    """All candidate dimensions, kept separate by construction."""

    discovery: CandidateAssessment[DiscoveryState]
    compatibility: CandidateAssessment[CompatibilityState]
    runtime_availability: CandidateAssessment[RuntimeAvailabilityState]
    local_availability: CandidateAssessment[LocalAvailabilityState]
    execution: CandidateAssessment[ExecutionState]
    measurement_capability: CandidateAssessment[MeasurementCapabilityState]
    measurement: CandidateAssessment[MeasurementState]
    recommendation: CandidateAssessment[RecommendationState]

    def __post_init__(self) -> None:
        expected = (
            (self.discovery, DiscoveryState, "discovery"),
            (self.compatibility, CompatibilityState, "compatibility"),
            (
                self.runtime_availability,
                RuntimeAvailabilityState,
                "runtime availability",
            ),
            (self.local_availability, LocalAvailabilityState, "local availability"),
            (self.execution, ExecutionState, "execution"),
            (
                self.measurement_capability,
                MeasurementCapabilityState,
                "measurement capability",
            ),
            (self.measurement, MeasurementState, "measurement"),
            (self.recommendation, RecommendationState, "recommendation"),
        )
        for assessment, state_type, label in expected:
            if not isinstance(assessment, CandidateAssessment) or not isinstance(
                assessment.state, state_type
            ):
                raise ValueError(f"candidate {label} uses the wrong state type")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery": self.discovery.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "runtime_availability": self.runtime_availability.to_dict(),
            "local_availability": self.local_availability.to_dict(),
            "execution": self.execution.to_dict(),
            "measurement_capability": self.measurement_capability.to_dict(),
            "measurement": self.measurement.to_dict(),
            "recommendation": self.recommendation.to_dict(),
        }


@dataclass(frozen=True)
class PlanningCandidate:
    """Privacy-safe candidate identity plus normalized independent state."""

    logical_model_id: str
    artifact_id: str
    runtime: str
    artifact_format: str
    quantization: Optional[str]
    assessments: CandidateAssessments

    def __post_init__(self) -> None:
        validate_public_text(
            self.logical_model_id, "candidate logical model id", identity=True
        )
        validate_public_text(self.artifact_id, "candidate artifact id", identity=True)
        validate_public_text(self.runtime, "candidate runtime")
        validate_public_text(self.artifact_format, "candidate artifact format")
        validate_public_text(self.quantization, "candidate quantization")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_model_id": self.logical_model_id,
            "artifact_id": self.artifact_id,
            "runtime": self.runtime,
            "artifact_format": self.artifact_format,
            "quantization": self.quantization,
            "assessments": self.assessments.to_dict(),
        }
