"""Approved Autopilot apply execution and privacy-safe receipts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .acquisition import (
    AcquisitionError,
    AcquiredArtifact,
    acquire_huggingface_artifact,
    acquisition_registry_file,
    parse_hf_artifact_id,
)
from .autopilot_actions import ActionKind, PlannedAction
from .autopilot_plan import AutopilotCandidate, AutopilotExecutionPlan
from .inventory import _execution_locator
from .privacy import validate_public_text
from .runtime_actions import RuntimeActionOutcome, runtime_actions_for
from .runtime_adapters import runtime_adapter_for


APPLY_RECEIPT_SCHEMA_VERSION = "0.9"


class ApplyError(RuntimeError):
    """Raised before mutation when an Autopilot plan cannot be safely applied."""


@dataclass(frozen=True)
class VerificationRecord:
    method_version: str
    generation_tps: Optional[float]
    prompt_eval_tps: Optional[float]
    total_latency_s: Optional[float]
    measured_runs: int
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_public_text(self.method_version, "verification method")
        if self.measured_runs < 1:
            raise ValueError("verification requires at least one measured run")
        for warning in self.warnings:
            validate_public_text(warning, "verification warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_version": self.method_version,
            "generation_tps": self.generation_tps,
            "prompt_eval_tps": self.prompt_eval_tps,
            "total_latency_s": self.total_latency_s,
            "measured_runs": self.measured_runs,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ActionExecutionReceipt:
    action_id: str
    kind: str
    runtime: Optional[str]
    status: str
    detail: str
    started_at: str
    completed_at: str
    evidence: Tuple[str, ...] = ()
    verification: Optional[VerificationRecord] = None

    def __post_init__(self) -> None:
        if self.status not in {
            "completed",
            "already_satisfied",
            "blocked",
            "failed",
            "skipped",
        }:
            raise ValueError("unsupported apply action status")
        for label, value in (
            ("action id", self.action_id),
            ("action kind", self.kind),
            ("runtime", self.runtime),
            ("detail", self.detail),
            ("started at", self.started_at),
            ("completed at", self.completed_at),
        ):
            validate_public_text(value, f"apply {label}")
        for value in self.evidence:
            validate_public_text(value, "apply action evidence")

    @property
    def success(self) -> bool:
        return self.status in {"completed", "already_satisfied"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "runtime": self.runtime,
            "status": self.status,
            "detail": self.detail,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "evidence": list(self.evidence),
            "verification": (
                self.verification.to_dict() if self.verification is not None else None
            ),
        }


@dataclass(frozen=True)
class AutopilotReceipt:
    receipt_id: str
    plan_id: str
    model: str
    logical_model_id: Optional[str]
    candidate_id: str
    runtime: str
    artifact_id: str
    artifact_format: str
    quantization: Optional[str]
    status: str
    started_at: str
    completed_at: str
    actions: Tuple[ActionExecutionReceipt, ...]
    endpoint: Optional[str] = None
    schema_version: str = APPLY_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "blocked"}:
            raise ValueError("unsupported apply receipt status")
        for label, value in (
            ("receipt id", self.receipt_id),
            ("plan id", self.plan_id),
            ("model", self.model),
            ("logical model", self.logical_model_id),
            ("candidate id", self.candidate_id),
            ("runtime", self.runtime),
            ("artifact id", self.artifact_id),
            ("artifact format", self.artifact_format),
            ("quantization", self.quantization),
            ("started at", self.started_at),
            ("completed at", self.completed_at),
            ("endpoint", self.endpoint),
        ):
            validate_public_text(
                value,
                f"apply receipt {label}",
                identity=label in {"model", "logical model", "candidate id", "artifact id"},
            )

    @property
    def verification(self) -> Optional[VerificationRecord]:
        for action in reversed(self.actions):
            if action.verification is not None:
                return action.verification
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "model": self.model,
            "logical_model_id": self.logical_model_id,
            "candidate": {
                "candidate_id": self.candidate_id,
                "runtime": self.runtime,
                "artifact_id": self.artifact_id,
                "artifact_format": self.artifact_format,
                "quantization": self.quantization,
            },
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "endpoint": self.endpoint,
            "actions": [action.to_dict() for action in self.actions],
            "verification": (
                self.verification.to_dict() if self.verification is not None else None
            ),
        }


class _PrivateApplyState:
    __slots__ = ("acquired", "locator")

    def __init__(self) -> None:
        self.acquired: Optional[AcquiredArtifact] = None
        self.locator: Optional[str] = None

    def __repr__(self) -> str:
        return "_PrivateApplyState()"

    def __reduce__(self) -> Any:
        raise TypeError("private apply state cannot be serialized")


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _selected_candidate(plan: AutopilotExecutionPlan) -> AutopilotCandidate:
    if not plan.selected_candidate_id:
        raise ApplyError("Autopilot plan has no selected candidate")
    selected = tuple(
        item for item in plan.candidates if item.candidate_id == plan.selected_candidate_id
    )
    if len(selected) != 1:
        raise ApplyError("Autopilot selected candidate is missing or ambiguous")
    return selected[0]


def _public_endpoint(runtime: str) -> Optional[str]:
    if runtime == "omlx":
        from .omlx_api import DEFAULT_OMLX_BASE_URL

        return DEFAULT_OMLX_BASE_URL + "/v1"
    if runtime == "ollama":
        import os

        return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    return None


def _runtime_outcome_receipt(
    action: PlannedAction,
    outcome: RuntimeActionOutcome,
    started_at: str,
) -> ActionExecutionReceipt:
    return ActionExecutionReceipt(
        action.action_id,
        action.kind.value,
        action.runtime,
        outcome.status,
        outcome.detail,
        started_at,
        _now_iso(),
        tuple(action.evidence),
    )


def _resolve_omlx_locator(legacy: Any, candidate: AutopilotCandidate) -> Optional[str]:
    from .omlx_acquisition_provenance import observe_llmrig_acquired_omlx_inventory
    from .omlx_verify import observe_omlx_inventory

    requested = candidate.artifact_id
    try:
        requested = parse_hf_artifact_id(candidate.artifact_id).repository_id
    except ValueError:
        pass
    targets = observe_omlx_inventory(legacy, requested)
    if not targets:
        targets = observe_llmrig_acquired_omlx_inventory(legacy, requested)
    if len(targets) != 1:
        return None
    return _execution_locator(targets[0])


def _validate_acquired_locator(
    legacy: Any, candidate: AutopilotCandidate, state: _PrivateApplyState
) -> RuntimeActionOutcome:
    if candidate.runtime == "omlx":
        actions = runtime_actions_for("omlx")
        if actions is None:
            return RuntimeActionOutcome("blocked", "oMLX action adapter is unavailable.")
        refresh = actions.refresh_after_acquisition(legacy)
        if not refresh.success:
            return refresh
        locator = _resolve_omlx_locator(legacy, candidate)
        if locator is None:
            return RuntimeActionOutcome(
                "failed",
                "oMLX became ready but exact acquired-model provenance could not be re-observed.",
            )
        state.locator = locator
        return RuntimeActionOutcome(
            "completed", "oMLX refreshed and exposed the exact acquired model."
        )

    if candidate.runtime == "ollama":
        state.locator = candidate.artifact_id
        return RuntimeActionOutcome(
            "already_satisfied", "Ollama acquisition registered the selected artifact."
        )

    if state.acquired is None:
        return RuntimeActionOutcome(
            "blocked",
            "The native runtime requires a private local artifact locator that is not available.",
            ("re-run apply from an exact acquisition plan",),
        )
    try:
        state.locator = legacy.validated_local_locator(
            candidate.runtime, state.acquired.locator
        )
    except Exception:
        return RuntimeActionOutcome(
            "failed", "The acquired native artifact failed structural validation."
        )
    return RuntimeActionOutcome(
        "completed", "The acquired native artifact passed local structural validation."
    )


def _verify_candidate(
    legacy: Any,
    plan: AutopilotExecutionPlan,
    candidate: AutopilotCandidate,
    state: _PrivateApplyState,
) -> VerificationRecord:
    adapter = runtime_adapter_for(candidate.runtime)
    if adapter is None or not adapter.llmrig_execution_supported:
        raise ApplyError("LLMRig execution support is unavailable for the selected runtime")

    if candidate.runtime == "omlx" and state.locator is None:
        state.locator = _resolve_omlx_locator(legacy, candidate)
    elif candidate.runtime == "ollama" and state.locator is None:
        state.locator = candidate.artifact_id
    if state.locator is None:
        raise ApplyError("verification requires a private local execution locator")

    probe = adapter.probe()
    configuration = legacy.RaceConfiguration(
        logical_model_id=plan.logical_model_id or plan.model,
        runtime=candidate.runtime,
        artifact_id=candidate.artifact_id,
        artifact_format=candidate.artifact_format,
        quantization=candidate.quantization,
        runtime_version=probe.version,
        eligible=True,
    )
    target = legacy.ExecutionTarget(configuration, state.locator)
    workload = legacy.RaceWorkload(
        prompt=legacy.SPEED_PROMPT,
        context=candidate.context_tokens or legacy.RACE_CONTEXT,
        num_predict=legacy.RACE_NUM_PREDICT,
    )
    validation_error = legacy.race_workload_error(
        workload.context, workload.num_predict, workload.runs
    )
    if validation_error is not None:
        raise ApplyError("the planned verification workload is not supported")

    executor = adapter.build_execution_adapter(legacy)
    competitor = executor.benchmark(target, workload)
    if competitor.execution_status != "success" or competitor.measured_runs < 1:
        raise ApplyError("selected runtime did not complete measured verification")
    return VerificationRecord(
        method_version=legacy.RACE_METHOD_VERSION,
        generation_tps=competitor.generation_tps,
        prompt_eval_tps=competitor.prompt_eval_tps,
        total_latency_s=competitor.total_latency_s,
        measured_runs=competitor.measured_runs,
        warnings=tuple(dict.fromkeys(competitor.warnings)),
    )


def _execute_action(
    legacy: Any,
    plan: AutopilotExecutionPlan,
    candidate: AutopilotCandidate,
    action: PlannedAction,
    state: _PrivateApplyState,
) -> ActionExecutionReceipt:
    started = _now_iso()
    if action.blockers:
        return ActionExecutionReceipt(
            action.action_id,
            action.kind.value,
            action.runtime,
            "blocked",
            "The planned action is blocked by unresolved prerequisites.",
            started,
            _now_iso(),
            tuple(action.evidence),
        )

    runtime_actions = runtime_actions_for(candidate.runtime)
    if runtime_actions is None:
        return ActionExecutionReceipt(
            action.action_id,
            action.kind.value,
            action.runtime,
            "blocked",
            "No registered runtime action adapter is available.",
            started,
            _now_iso(),
            tuple(action.evidence),
        )

    if action.kind == ActionKind.ACQUIRE_ARTIFACT:
        if candidate.artifact_id.startswith("hf://"):
            try:
                state.acquired = acquire_huggingface_artifact(
                    candidate.artifact_id, candidate.runtime
                )
                state.locator = state.acquired.locator
            except AcquisitionError:
                return ActionExecutionReceipt(
                    action.action_id,
                    action.kind.value,
                    action.runtime,
                    "blocked",
                    "Exact Hugging Face acquisition could not be completed.",
                    started,
                    _now_iso(),
                    tuple(action.evidence),
                )
            return ActionExecutionReceipt(
                action.action_id,
                action.kind.value,
                action.runtime,
                "completed",
                "Acquired the exact Hugging Face artifact at a pinned repository revision.",
                started,
                _now_iso(),
                tuple(action.evidence)
                + (
                    f"Hugging Face revision: {state.acquired.record.revision}",
                ),
            )
        outcome = runtime_actions.acquire_native(legacy, candidate.artifact_id)
        if outcome.success and candidate.runtime == "ollama":
            state.locator = candidate.artifact_id
        return _runtime_outcome_receipt(action, outcome, started)

    if action.kind == ActionKind.START_RUNTIME:
        return _runtime_outcome_receipt(action, runtime_actions.start(legacy), started)

    if action.kind == ActionKind.LOAD_MODEL:
        return _runtime_outcome_receipt(
            action,
            _validate_acquired_locator(legacy, candidate, state),
            started,
        )

    if action.kind == ActionKind.VERIFY:
        try:
            verification = _verify_candidate(legacy, plan, candidate, state)
        except (ApplyError, RuntimeError, ValueError):
            return ActionExecutionReceipt(
                action.action_id,
                action.kind.value,
                action.runtime,
                "failed",
                "The selected configuration did not complete verification.",
                started,
                _now_iso(),
                tuple(action.evidence),
            )
        return ActionExecutionReceipt(
            action.action_id,
            action.kind.value,
            action.runtime,
            "completed",
            "The selected configuration completed deterministic measured verification.",
            started,
            _now_iso(),
            tuple(action.evidence),
            verification,
        )

    return ActionExecutionReceipt(
        action.action_id,
        action.kind.value,
        action.runtime,
        "blocked",
        "This Autopilot action kind is not implemented.",
        started,
        _now_iso(),
        tuple(action.evidence),
    )


def _receipt_id(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "receipt-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def receipts_dir() -> Path:
    return acquisition_registry_file().parent / "receipts"


def save_receipt(receipt: AutopilotReceipt) -> None:
    directory = receipts_dir()
    directory.mkdir(parents=True, exist_ok=True)
    payload = receipt.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = directory / f"{receipt.receipt_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    latest = directory / "latest.json"
    latest_tmp = directory / "latest.tmp"
    latest_tmp.write_text(text, encoding="utf-8")
    latest_tmp.replace(latest)


def load_receipt(receipt_id: str = "latest") -> Dict[str, Any]:
    validate_public_text(receipt_id, "receipt id", identity=True)
    path = (
        receipts_dir() / "latest.json"
        if receipt_id == "latest"
        else receipts_dir() / f"{receipt_id}.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ApplyError("the requested Autopilot receipt could not be loaded") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != APPLY_RECEIPT_SCHEMA_VERSION:
        raise ApplyError("the requested Autopilot receipt uses an unsupported schema")
    return payload


def apply_autopilot_plan(
    plan: AutopilotExecutionPlan,
    legacy: Any,
    *,
    explicit_user_approval: bool,
    persist_receipt: bool = True,
) -> AutopilotReceipt:
    """Execute exactly the approved deterministic plan, stopping on first failure."""

    if plan.blocked:
        raise ApplyError("Autopilot plan is blocked and cannot be applied")
    if plan.has_mutations and not explicit_user_approval:
        raise PermissionError("mutating Autopilot actions require explicit user approval")
    candidate = _selected_candidate(plan)
    started_at = _now_iso()
    state = _PrivateApplyState()
    receipts = []
    stopped = False

    for action in plan.actions:
        if stopped:
            now = _now_iso()
            receipts.append(
                ActionExecutionReceipt(
                    action.action_id,
                    action.kind.value,
                    action.runtime,
                    "skipped",
                    "Skipped because an earlier action did not complete successfully.",
                    now,
                    now,
                    tuple(action.evidence),
                )
            )
            continue
        receipt = _execute_action(legacy, plan, candidate, action, state)
        receipts.append(receipt)
        if not receipt.success:
            stopped = True

    completed_at = _now_iso()
    if any(item.status == "blocked" for item in receipts):
        status = "blocked"
    elif any(item.status == "failed" for item in receipts):
        status = "failed"
    elif receipts and all(item.success for item in receipts):
        status = "completed"
    else:
        status = "failed"

    body = {
        "schema_version": APPLY_RECEIPT_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "model": plan.model,
        "logical_model_id": plan.logical_model_id,
        "candidate_id": candidate.candidate_id,
        "runtime": candidate.runtime,
        "artifact_id": candidate.artifact_id,
        "artifact_format": candidate.artifact_format,
        "quantization": candidate.quantization,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "actions": [item.to_dict() for item in receipts],
        "endpoint": _public_endpoint(candidate.runtime) if status == "completed" else None,
    }
    receipt = AutopilotReceipt(
        receipt_id=_receipt_id(body),
        plan_id=plan.plan_id,
        model=plan.model,
        logical_model_id=plan.logical_model_id,
        candidate_id=candidate.candidate_id,
        runtime=candidate.runtime,
        artifact_id=candidate.artifact_id,
        artifact_format=candidate.artifact_format,
        quantization=candidate.quantization,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        actions=tuple(receipts),
        endpoint=body["endpoint"],
    )
    if persist_receipt:
        save_receipt(receipt)
    return receipt
