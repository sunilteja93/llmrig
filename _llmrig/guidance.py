"""Read-only, evidence-respecting guidance for Autopilot plans.

Guidance is not evidence: it never changes candidate states or asserts that a
runtime, artifact, or download will work.
"""
from __future__ import annotations

from typing import Any, Dict, List


def next_steps(plan: Any) -> List[str]:
    """Return short, actionable suggestions without guessing an execution path."""
    selected = getattr(plan, "selected_candidate_id", None)
    candidates = tuple(getattr(plan, "candidates", ()))
    blockers = tuple(getattr(plan, "blockers", ()))
    parts = [str(value).lower() for value in blockers]
    for candidate in candidates:
        parts.extend(str(value).lower() for value in candidate.blockers)
        parts.extend(str(value).lower() for value in candidate.unknowns)
    message = " ".join(parts)
    steps: List[str] = []

    if any(word in message for word in ("runtime", "not installed", "not detected", "not ready", "unavailable")):
        steps.append("Run 'llmrig runtimes' to inspect which supported runtimes are installed and ready.")
    if any(word in message for word in ("memory", "ram", "incompatible", "too large", "capacity", "fit")):
        steps.append("Try a smaller or more heavily quantized artifact and run 'llmrig plan MODEL' again.")
    if any(word in message for word in ("provenance", "artifact", "revision", "unsupported", "unknown", "incomplete")):
        steps.append("Check the exact model repository, available artifact formats, and supported runtime.")
    if not selected:
        steps.append("No unique executable path was selected. Compare the candidates and resolve their blockers before applying.")
    elif getattr(plan, "blocked", False):
        steps.append("Resolve the listed apply blockers, then build a fresh plan before approval.")
    else:
        steps.append("Review the planned actions, then run 'llmrig run MODEL' to approve and verify the setup.")
    return list(dict.fromkeys(steps))[:4]


def diagnosis(plan: Any) -> Dict[str, Any]:
    """Read-only diagnostic summary without changing public plan schema."""
    return {
        "model": str(plan.model),
        "plan_id": str(plan.plan_id),
        "status": "blocked" if plan.blocked else (
            "inconclusive" if not plan.selected_candidate_id else "ready_to_approve"
        ),
        "candidate_count": len(plan.candidates),
        "next_steps": next_steps(plan),
    }
