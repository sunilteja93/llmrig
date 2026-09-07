"""Private implementation support for LLMRig.

This package is intentionally internal. The deliberate SDK contract lives in the
top-level :mod:`llmrig` module; imports from :mod:`_llmrig` are not public API.
"""

from .inventory import (
    ExplicitNativeInventoryProvider,
    InventoryProvider,
    InventoryRecord,
    InventoryTarget,
    OllamaInventoryProvider,
)
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
)
from .solve import (
    ReproducibleRecipe,
    SOLVE_SCHEMA_VERSION,
    SolveCandidate,
    SolveMeasurement,
    SolvePlan,
    SolveRequest,
    SolveResult,
    SolveVerification,
)

__all__ = [
    "CandidateAssessment",
    "CandidateAssessments",
    "CompatibilityState",
    "DiscoveryState",
    "ExecutionState",
    "ExplicitNativeInventoryProvider",
    "InventoryProvider",
    "InventoryRecord",
    "InventoryTarget",
    "LocalAvailabilityState",
    "MeasurementCapabilityState",
    "MeasurementState",
    "OllamaInventoryProvider",
    "PlanningCandidate",
    "RecommendationState",
    "RuntimeAvailabilityState",
    "ReproducibleRecipe",
    "SOLVE_SCHEMA_VERSION",
    "SolveCandidate",
    "SolveMeasurement",
    "SolvePlan",
    "SolveRequest",
    "SolveResult",
    "SolveVerification",
]
