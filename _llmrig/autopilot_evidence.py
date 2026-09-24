"""Bridge Autopilot receipts into privacy-safe RigGraph evidence."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .autopilot_apply import AutopilotReceipt
from .autopilot_plan import AutopilotExecutionPlan
from .riggraph import RigGraphEvidenceRecord, save_evidence_record


def record_receipt_evidence(
    plan: AutopilotExecutionPlan,
    receipt: AutopilotReceipt,
    *,
    prediction: Optional[Mapping[str, Any]] = None,
) -> Optional[RigGraphEvidenceRecord]:
    """Persist measured receipt evidence without inventing unavailable predictions."""

    verification = receipt.verification
    if verification is None:
        return None
    selected = next(
        (item for item in plan.candidates if item.candidate_id == receipt.candidate_id),
        None,
    )
    if selected is None:
        return None
    measurement = {
        "generation_tps": verification.generation_tps,
        "prompt_eval_tps": verification.prompt_eval_tps,
        "total_latency_s": verification.total_latency_s,
    }
    record = RigGraphEvidenceRecord.build(
        observed_at=receipt.completed_at,
        machine=dict(plan.machine),
        model=receipt.model,
        logical_model_id=receipt.logical_model_id,
        runtime=receipt.runtime,
        artifact_id=receipt.artifact_id,
        artifact_format=receipt.artifact_format,
        quantization=receipt.quantization,
        context_tokens=receipt.context_tokens,
        prediction=prediction or {},
        measurement=measurement,
        receipt_id=receipt.receipt_id,
        plan_id=receipt.plan_id,
    )
    save_evidence_record(record)
    return record
