"""Deterministic, read-only Autopilot plan construction.

This module converts the existing privacy-safe solve result into a stable plan
contract. It performs no downloads, runtime changes, model loads, or inference.
Future apply/receipt stages consume this contract rather than recomputing intent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .autopilot_actions import ActionKind, PlannedAction
from .privacy import validate_public_text
from .runtime_actions import runtime_actions_for


AUTOPILOT_PLAN_SCHEMA_VERSION = "0.9"


def _state(candidate: Mapping[str, Any], name: str) -> str:
    assessments = candidate.get("assessments")
    if not isinstance(assessments, Mapping):
        return "unknown"
    assessment = assessments.get(name)
    if not isinstance(assessment, Mapping):
        return "unknown"
    value = assessment.get("state")
    return str(value) if isinstance(value, str) and value else "unknown"


def _strings(values: Any) -> Tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    result = []
    for value in values:
        if isinstance(value, str) and value.strip():
            text = value.strip()
            validate_public_text(text, "Autopilot plan text")
            result.append(text)
    return tuple(sorted(dict.fromkeys(result)))


@dataclass(frozen=True)
class AutopilotCandidate:
    candidate_id: str
    runtime: str
    artifact_id: str
    artifact_format: str
    quantization: Optional[str]
    context_tokens: Optional[int]
    discovery: str
    compatibility: str
    runtime_availability: str
    local_availability: str
    execution: str
    measurement_capability: str
    recipe_status: str
    recipe_steps: Tuple[str, ...]
    blockers: Tuple[str, ...]
    unknowns: Tuple[str, ...]
    artifact_revision: Optional[str] = None

    def __post_init__(self) -> None:
        validate_public_text(self.candidate_id, "Autopilot candidate id", identity=True)
        validate_public_text(self.runtime, "Autopilot runtime")
        validate_public_text(self.artifact_id, "Autopilot artifact id", identity=True)
        validate_public_text(self.artifact_format, "Autopilot artifact format")
        validate_public_text(self.quantization, "Autopilot quantization")
        validate_public_text(self.artifact_revision, "Autopilot artifact revision")
        validate_public_text(self.recipe_status, "Autopilot recipe status")
        if self.context_tokens is not None and self.context_tokens <= 0:
            raise ValueError("Autopilot candidate context must be positive")
        for value in (
            self.discovery,
            self.compatibility,
            self.runtime_availability,
            self.local_availability,
            self.execution,
            self.measurement_capability,
        ):
            validate_public_text(value, "Autopilot candidate state")
        for value in self.recipe_steps + self.blockers + self.unknowns:
            validate_public_text(value, "Autopilot candidate detail")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "runtime": self.runtime,
            "artifact_id": self.artifact_id,
            "artifact_format": self.artifact_format,
            "quantization": self.quantization,
            "context_tokens": self.context_tokens,
            "states": {
                "discovery": self.discovery,
                "compatibility": self.compatibility,
                "runtime_availability": self.runtime_availability,
                "local_availability": self.local_availability,
                "execution": self.execution,
                "measurement_capability": self.measurement_capability,
            },
            "recipe": {
                "status": self.recipe_status,
                "steps": list(self.recipe_steps),
            },
            "blockers": list(self.blockers),
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class AutopilotExecutionPlan:
    plan_id: str
    model: str
    logical_model_id: Optional[str]
    machine: Tuple[Tuple[str, Any], ...]
    candidates: Tuple[AutopilotCandidate, ...]
    recommendation_status: str
    selected_candidate_id: Optional[str]
    recommendation_reason: str
    actions: Tuple[PlannedAction, ...]
    blockers: Tuple[str, ...] = ()
    unknowns: Tuple[str, ...] = ()
    schema_version: str = AUTOPILOT_PLAN_SCHEMA_VERSION

    @property
    def has_mutations(self) -> bool:
        return any(action.mutating for action in self.actions)

    @property
    def blocked(self) -> bool:
        return bool(self.blockers) or any(not action.executable for action in self.actions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "model": self.model,
            "logical_model_id": self.logical_model_id,
            "machine": dict(self.machine),
            "recommendation": {
                "status": self.recommendation_status,
                "selected_candidate_id": self.selected_candidate_id,
                "reason": self.recommendation_reason,
            },
            "has_mutations": self.has_mutations,
            "blocked": self.blocked,
            "actions": [action.to_dict() for action in self.actions],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "blockers": list(self.blockers),
            "unknowns": list(self.unknowns),
        }


def _candidate_from_payload(payload: Mapping[str, Any]) -> AutopilotCandidate:
    recipe = payload.get("recipe")
    if not isinstance(recipe, Mapping):
        recipe = {}
    candidate_id = str(payload.get("candidate_id") or "")
    runtime = str(payload.get("runtime") or "unknown")
    artifact_id = str(payload.get("artifact_id") or "")
    artifact_format = str(payload.get("artifact_format") or "Unknown")
    quantization = payload.get("quantization")
    if quantization is not None:
        quantization = str(quantization)
    context_tokens = payload.get("context_tokens")
    if not isinstance(context_tokens, int):
        context_tokens = None

    candidate_blockers = list(_strings(payload.get("blockers")))
    candidate_unknowns = list(_strings(payload.get("unknowns")))
    assessments = payload.get("assessments")
    if isinstance(assessments, Mapping):
        for assessment in assessments.values():
            if not isinstance(assessment, Mapping):
                continue
            candidate_blockers.extend(_strings(assessment.get("blockers")))
            candidate_unknowns.extend(_strings(assessment.get("unknowns")))
    candidate_blockers.extend(_strings(recipe.get("blockers")))

    return AutopilotCandidate(
        candidate_id=candidate_id,
        runtime=runtime,
        artifact_id=artifact_id,
        artifact_format=artifact_format,
        quantization=quantization,
        context_tokens=context_tokens,
        discovery=_state(payload, "discovery"),
        compatibility=_state(payload, "compatibility"),
        runtime_availability=_state(payload, "runtime_availability"),
        local_availability=_state(payload, "local_availability"),
        execution=_state(payload, "execution"),
        measurement_capability=_state(payload, "measurement_capability"),
        recipe_status=str(recipe.get("status") or "unknown"),
        recipe_steps=_strings(recipe.get("steps")),
        blockers=tuple(sorted(dict.fromkeys(candidate_blockers))),
        unknowns=tuple(sorted(dict.fromkeys(candidate_unknowns))),
    )


def _actions_for(candidate: AutopilotCandidate) -> Tuple[PlannedAction, ...]:
    actions = []
    action_adapter = runtime_actions_for(candidate.runtime)

    if candidate.local_availability != "available":
        source_is_exact_hf = candidate.artifact_id.startswith("hf://")
        runtime_native = bool(
            action_adapter is not None
            and action_adapter.native_acquisition_supported
            and not source_is_exact_hf
        )
        if source_is_exact_hf:
            evidence = (candidate.artifact_id,)
            blockers = ()
        elif runtime_native:
            evidence = (
                "the selected runtime exposes an explicit native acquisition path",
            )
            blockers = ()
        else:
            evidence = ()
            blockers = ("artifact acquisition requires an exact evidenced source",)
        actions.append(
            PlannedAction(
                "a01-acquire-artifact",
                ActionKind.ACQUIRE_ARTIFACT,
                candidate.runtime,
                "Acquire or resolve the selected model artifact from its evidenced source.",
                True,
                evidence,
                blockers,
            )
        )

    if candidate.runtime_availability == "unknown":
        actions.append(
            PlannedAction(
                "a02-prepare-runtime",
                ActionKind.START_RUNTIME,
                candidate.runtime,
                "Resolve the selected local runtime before execution.",
                True,
                (),
                ("runtime availability is unknown",),
            )
        )
    elif candidate.runtime_availability == "unavailable":
        start_supported = bool(
            action_adapter is not None and action_adapter.start_supported
        )
        actions.append(
            PlannedAction(
                "a02-prepare-runtime",
                ActionKind.START_RUNTIME,
                candidate.runtime,
                "Prepare and start the selected local runtime.",
                True,
                ("runtime availability was observed as unavailable",),
                ()
                if start_supported
                else (
                    "the runtime is unavailable and LLMRig has no automatic start path for it",
                ),
            )
        )

    if candidate.execution != "executable":
        load_supported = bool(
            candidate.runtime in {"omlx", "ollama", "mlx-lm", "llama.cpp"}
        )
        actions.append(
            PlannedAction(
                "a03-load-model",
                ActionKind.LOAD_MODEL,
                candidate.runtime,
                "Load or register the selected artifact with the runtime.",
                True,
                ("candidate is not currently evidenced as executable",),
                ()
                if load_supported
                else ("LLMRig has no model-load path for this runtime",),
            )
        )

    actions.append(
        PlannedAction(
            "a04-verify",
            ActionKind.VERIFY,
            candidate.runtime,
            "Verify the resulting configuration with LLMRig's deterministic workload.",
            False,
            ("verification is defined before apply",),
        )
    )
    return tuple(actions)


def _setup_viable(candidate: AutopilotCandidate) -> bool:
    if candidate.discovery != "discovered" or candidate.compatibility != "compatible":
        return False
    actions = _actions_for(candidate)
    return bool(actions) and not any(action.blockers for action in actions)


def _setup_mutation_cost(candidate: AutopilotCandidate) -> int:
    """Count only explicit local-state changes; verification is not a mutation."""
    return sum(1 for action in _actions_for(candidate) if action.mutating)


def _machine_tuple(machine: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    allowed = ("os", "arch", "cpu", "ram_gib")
    values = []
    for key in allowed:
        value = machine.get(key)
        if isinstance(value, str):
            validate_public_text(value, "Autopilot machine field")
        elif value is not None and not isinstance(value, (int, float, bool)):
            value = str(value)
            validate_public_text(value, "Autopilot machine field")
        values.append((key, value))
    return tuple(values)


def build_autopilot_plan(
    solve_payload: Mapping[str, Any], machine: Mapping[str, Any]
) -> AutopilotExecutionPlan:
    """Build a deterministic plan from an already-computed read-only solve result."""

    request = solve_payload.get("request")
    if not isinstance(request, Mapping):
        request = {}
    model = request.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Autopilot planning requires a model identifier")
    model = model.strip()
    validate_public_text(model, "Autopilot model", identity=True)

    resolution = solve_payload.get("resolution")
    if not isinstance(resolution, Mapping):
        resolution = {}
    logical_model_id = resolution.get("logical_model_id")
    if logical_model_id is not None:
        logical_model_id = str(logical_model_id)
        validate_public_text(logical_model_id, "Autopilot logical model id", identity=True)

    raw_candidates = solve_payload.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raw_candidates = ()
    candidates = tuple(
        sorted(
            (
                _candidate_from_payload(item)
                for item in raw_candidates
                if isinstance(item, Mapping)
            ),
            key=lambda item: item.candidate_id,
        )
    )

    solve_plan = solve_payload.get("plan")
    if not isinstance(solve_plan, Mapping):
        solve_plan = {}
    recommendation_status = str(solve_plan.get("recommendation_status") or "inconclusive")
    selected_candidate_id = solve_plan.get("recommended_candidate_id")
    if selected_candidate_id is not None:
        selected_candidate_id = str(selected_candidate_id)
        validate_public_text(selected_candidate_id, "Autopilot selected candidate", identity=True)
    recommendation_reason = str(
        solve_plan.get("reason") or "No evidenced candidate recommendation is available."
    )
    validate_public_text(recommendation_reason, "Autopilot recommendation reason")

    blockers = []
    if solve_payload.get("status") != "analyzed":
        blockers.append("solve analysis did not complete successfully")
    if not candidates:
        blockers.append("no execution candidates were discovered")

    selected = next(
        (candidate for candidate in candidates if candidate.candidate_id == selected_candidate_id),
        None,
    )
    if selected_candidate_id and selected is None:
        blockers.append("the recommended candidate is not present in the candidate set")
        selected_candidate_id = None

    # v0.8 solve answers which configuration is runnable *now*. Autopilot needs a
    # separate setup-path decision: if exactly one compatible candidate can be
    # made runnable through fully supported actions, select that path without
    # claiming it is a measured-performance winner.
    if selected_candidate_id is None and solve_payload.get("status") == "analyzed":
        setup_candidates = tuple(candidate for candidate in candidates if _setup_viable(candidate))
        if len(setup_candidates) == 1:
            selected = setup_candidates[0]
            selected_candidate_id = selected.candidate_id
            recommendation_status = "setup_selected"
            recommendation_reason = (
                "Exactly one compatible candidate has a complete, supported Autopilot setup path. "
                "This is a setup selection, not a performance ranking."
            )
        elif len(setup_candidates) > 1:
            minimum_mutations = min(
                _setup_mutation_cost(candidate)
                for candidate in setup_candidates
            )
            minimal_change_candidates = tuple(
                candidate
                for candidate in setup_candidates
                if _setup_mutation_cost(candidate) == minimum_mutations
            )

            if len(minimal_change_candidates) == 1:
                selected = minimal_change_candidates[0]
                selected_candidate_id = selected.candidate_id
                recommendation_status = "setup_selected"
                recommendation_reason = (
                    "Multiple compatible Autopilot setup paths exist, but exactly one "
                    "requires the fewest mutating actions. LLMRig selected the "
                    "minimal-change setup path. This is not a performance ranking."
                )
            else:
                recommendation_status = "inconclusive"
                recommendation_reason = (
                    "Multiple compatible Autopilot setup paths remain with equal "
                    "minimal-change cost; LLMRig will not rank them without additional "
                    "evidence or measurement."
                )
        else:
            recommendation_status = "inconclusive"

    if selected_candidate_id is None:
        blockers.append("no unique evidenced candidate is selected for apply")

    actions = _actions_for(selected) if selected is not None and selected_candidate_id else ()
    unknowns = list(_strings(solve_payload.get("unknowns")))
    unknowns.extend(_strings(solve_plan.get("unknowns")))

    body = {
        "schema_version": AUTOPILOT_PLAN_SCHEMA_VERSION,
        "model": model,
        "logical_model_id": logical_model_id,
        "machine": dict(_machine_tuple(machine)),
        "recommendation": {
            "status": recommendation_status,
            "selected_candidate_id": selected_candidate_id,
            "reason": recommendation_reason,
        },
        "actions": [action.to_dict() for action in actions],
        "candidates": [candidate.to_dict() for candidate in candidates],
        "blockers": sorted(dict.fromkeys(blockers)),
        "unknowns": sorted(dict.fromkeys(unknowns)),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    plan_id = "plan-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    return AutopilotExecutionPlan(
        plan_id=plan_id,
        model=model,
        logical_model_id=logical_model_id,
        machine=_machine_tuple(machine),
        candidates=candidates,
        recommendation_status=recommendation_status,
        selected_candidate_id=selected_candidate_id,
        recommendation_reason=recommendation_reason,
        actions=actions,
        blockers=tuple(sorted(dict.fromkeys(blockers))),
        unknowns=tuple(sorted(dict.fromkeys(unknowns))),
    )
