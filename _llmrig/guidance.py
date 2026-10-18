"""Read-only, evidence-respecting guidance for Autopilot plans.

Guidance is not evidence: it never changes candidate states or asserts that a
runtime, artifact, or download will work.
"""
from __future__ import annotations

from typing import Any, Dict, List


def next_steps(plan: Any) -> List[str]:
    """Suggest actionable steps for the selected path, never unrelated alternatives.

    A successful unique selection must not inherit blockers from unselected
    candidates. This is guidance only and changes no planning evidence.
    """
    selected_id = getattr(plan, "selected_candidate_id", None)
    candidates = tuple(getattr(plan, "candidates", ()))
    blocked = bool(getattr(plan, "blocked", True))
    if selected_id and not blocked:
        return [
            "A supported path has been selected. Review the planned actions before approval.",
            "Run 'llmrig run MODEL' to approve and verify this setup. No action has been taken.",
        ]

    # Where a selected path is blocked, only its own blockers and the plan's
    # apply blockers affect remediation. Where no path exists, include all
    # candidates to explain why the planner remains inconclusive.
    relevant = (
        tuple(c for c in candidates if c.candidate_id == selected_id)
        if selected_id else candidates
    )
    parts = [str(v).lower() for v in getattr(plan, "blockers", ())]
    for candidate in relevant:
        parts.extend(str(v).lower() for v in getattr(candidate, "blockers", ()))
        parts.extend(str(v).lower() for v in getattr(candidate, "unknowns", ()))
    message = " ".join(parts)
    steps: List[str] = []
    if any(word in message for word in ("runtime", "not installed", "not detected", "not ready", "unavailable")):
        steps.append("Run 'llmrig runtimes' to inspect which supported runtimes are installed and ready.")
    if any(word in message for word in ("memory", "ram", "incompatible", "too large", "capacity", "fit")):
        steps.append("Try a smaller or more heavily quantized artifact and run 'llmrig plan MODEL' again.")
    if any(word in message for word in ("provenance", "artifact", "revision", "unsupported", "unknown", "incomplete")):
        steps.append("Check the exact model repository, available artifact formats, and supported runtime.")
    if not selected_id:
        steps.append("No unique executable path was selected. Compare the candidates and resolve their blockers before applying.")
    else:
        steps.append("Resolve the selected path's apply blockers, then build a fresh plan before approval.")
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
