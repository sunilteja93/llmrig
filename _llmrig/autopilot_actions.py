"""Plan/apply primitives for future mutating Autopilot stages.

Nothing in this module performs side effects.  It exists so later acquisition,
configuration, and launch implementations have a common safety contract from day
one instead of growing ad-hoc confirmation checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ActionKind(str, Enum):
    ACQUIRE_ARTIFACT = "acquire_artifact"
    INSTALL_RUNTIME = "install_runtime"
    CONFIGURE_RUNTIME = "configure_runtime"
    START_RUNTIME = "start_runtime"
    LOAD_MODEL = "load_model"
    VERIFY = "verify"


@dataclass(frozen=True)
class PlannedAction:
    action_id: str
    kind: ActionKind
    runtime: Optional[str]
    description: str
    mutating: bool
    evidence: Tuple[str, ...] = ()
    blockers: Tuple[str, ...] = ()

    @property
    def executable(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "kind": self.kind.value,
            "runtime": self.runtime,
            "description": self.description,
            "mutating": self.mutating,
            "executable": self.executable,
            "evidence": list(self.evidence),
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class AutopilotPlan:
    plan_id: str
    actions: Tuple[PlannedAction, ...]
    recommendation_basis: str

    @property
    def has_mutations(self) -> bool:
        return any(item.mutating for item in self.actions)

    @property
    def blocked(self) -> bool:
        return any(not item.executable for item in self.actions)

    def assert_apply_allowed(self, *, explicit_user_approval: bool) -> None:
        if self.blocked:
            raise RuntimeError("Autopilot plan contains blocked actions")
        if self.has_mutations and not explicit_user_approval:
            raise PermissionError(
                "mutating Autopilot actions require explicit user approval"
            )

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "recommendation_basis": self.recommendation_basis,
            "has_mutations": self.has_mutations,
            "blocked": self.blocked,
            "actions": [item.to_dict() for item in self.actions],
        }
