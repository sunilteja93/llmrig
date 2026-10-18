"""Offline, privacy-filtered review of opt-in Rescue JSON receipts.

Assess *observed* mitigation evidence; never execute agents, contact Ollama,
load source files, or edit a real OpenCode configuration. Reports supplied by
the caller are self-reported observations, not cryptographic attestations.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

from .rescue_agent import SOURCE, TASK_ID, _sha

MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_GROUP = 10
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,199}$")
REQUIRED_CHECKS = (
    "read_tool_completed",
    "bug_explained",
    "correct_return_proposed",
    "no_disallowed_tool_observed",
    "no_error_event",
    "valid_event_stream",
    "process_exit_zero",
    "fixture_unchanged",
)
FIXTURE_SHA256 = _sha(SOURCE.encode("utf-8"))


def _load(path: str) -> Dict[str, Any]:
    """Read only a caller-selected report with a strict bounded byte count."""
    try:
        with Path(path).open("rb") as src:
            data = src.read(MAX_REPORT_BYTES + 1)
    except OSError:
        raise ValueError("could not read an explicitly selected Rescue receipt") from None
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("Rescue receipt exceeds 2 MiB size cap")
    try:
        report = json.loads(data)
    except (ValueError, UnicodeError):
        raise ValueError("invalid JSON in selected Rescue receipt") from None
    if not isinstance(report, dict):
        raise ValueError("Rescue receipt root must be an object")
    return report


def _normalize(report: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist and validate all fields used in the assessment."""
    if report.get("mode") != "measured_agent_task" or report.get("agent") != "opencode":
        raise ValueError("only single-task OpenCode Rescue receipts are supported")
    model = report.get("model")
    context = report.get("context_tokens")
    if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model):
        raise ValueError("invalid or unsupported model tag in receipt")
    if isinstance(context, bool) or not isinstance(context, int) or not 256 <= context <= 262144:
        raise ValueError("invalid declared context in receipt")
    result = report.get("agent_evaluation")
    if not isinstance(result, dict):
        raise ValueError("receipt does not contain a single-task agent evaluation")
    if result.get("agent") != "opencode" or result.get("model") != model:
        raise ValueError("agent or model identity mismatch in receipt")
    if result.get("task_id") != TASK_ID or result.get("fixture_sha256") != FIXTURE_SHA256:
        raise ValueError("the fixture or task version differs from the fixed Rescue task")
    if result.get("agent_steps_limit") is not None:
        raise ValueError("step-limit experiments are not comparable in this analysis")
    # Receipts predating the opt-in compaction switch had no such field.
    # Accept them as *possible* default-compaction baselines only, and
    # explicitly disclose that the setting was not recorded in the receipt.
    legacy_policy_not_recorded = "auto_compaction_disabled" not in result
    compaction_disabled = result.get("auto_compaction_disabled", False)
    if type(compaction_disabled) is not bool:
        raise ValueError("receipt contains an invalid compaction policy")
    status = result.get("status")
    if status not in ("passed", "failed"):
        raise ValueError("receipt has an unsupported task status")
    checks = result.get("checks")
    if not isinstance(checks, dict) or not all(type(checks.get(key)) is bool for key in REQUIRED_CHECKS):
        raise ValueError("receipt is missing structured task check results")
    category = result.get("failure_category")
    if category is not None and (not isinstance(category, str) or category not in (
        "agent_timeout", "output_limit", "agent_nonzero_exit",
        "invalid_agent_event_stream", "test_boundary_violation",
        "task_requirements_not_met", "agent_process_failed_to_start",
    )):
        raise ValueError("receipt contains an unsupported failure category")
    if status == "passed" and (category is not None or not all(checks[key] for key in REQUIRED_CHECKS)):
        raise ValueError("inconsistent pass and task checks in receipt")
    # A strict successful completion requires both status and all independently
    # reported checks. A timeout with a fix never qualifies as success.
    strict_pass = status == "passed" and category is None and all(
        checks[key] for key in REQUIRED_CHECKS
    )
    correction_evidence = all(checks[key] for key in (
        "read_tool_completed", "correct_return_proposed",
        "valid_event_stream", "no_disallowed_tool_observed", "fixture_unchanged",
    ))
    timed_out_with_fix = category == "agent_timeout" and correction_evidence
    return {
        "model": model,
        "declared_context": context,
        "auto_compaction_disabled": compaction_disabled,
        "legacy_compaction_policy_not_recorded": legacy_policy_not_recorded,
        "strict_pass": strict_pass,
        "correction_evidence": correction_evidence,
        "process_exit": checks["process_exit_zero"],
        "timeout": category == "agent_timeout",
        "timeout_with_fix": timed_out_with_fix,
        "boundary_violation": category == "test_boundary_violation"
            or not checks["no_disallowed_tool_observed"] or not checks["fixture_unchanged"],
    }


def _group(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(observations)
    return {
        "sessions": count,
        "legacy_compaction_policy_not_recorded": sum(
            x["legacy_compaction_policy_not_recorded"] for x in observations
        ),
        "strict_task_passes": sum(x["strict_pass"] for x in observations),
        "provisional_correct_fixes": sum(x["correction_evidence"] for x in observations),
        "successful_process_exits": sum(x["process_exit"] for x in observations),
        "agent_timeouts": sum(x["timeout"] for x in observations),
        "agent_timeouts_after_fix_evidence": sum(x["timeout_with_fix"] for x in observations),
        "boundary_violations": sum(x["boundary_violation"] for x in observations),
    }


def assess(baselines: Sequence[str], candidates: Sequence[str]) -> Dict[str, Any]:
    """Analyze only existing reports. No I/O beyond explicitly chosen files."""
    if (not 1 <= len(baselines) <= MAX_GROUP or not 1 <= len(candidates) <= MAX_GROUP):
        raise ValueError("supply between 1 and 10 receipts for each configuration")
    paths = list(baselines) + list(candidates)
    # Refuse simple accidental double-counting of the same receipt.
    if len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("each receipt path must be unique")
    original = [_normalize(_load(path)) for path in baselines]
    modified = [_normalize(_load(path)) for path in candidates]
    all_reports = original + modified
    model = original[0]["model"]
    context = original[0]["declared_context"]
    if any(r["model"] != model or r["declared_context"] != context for r in all_reports):
        raise ValueError("all receipts must use the same model and declared context")
    if any(r["auto_compaction_disabled"] for r in original):
        raise ValueError("baseline receipts must have default automatic compaction")
    if any(not r["auto_compaction_disabled"] or r["legacy_compaction_policy_not_recorded"]
           for r in modified):
        raise ValueError("candidate receipts must explicitly disable auto compaction")

    before = _group(original)
    after = _group(modified)
    boundaries = before["boundary_violations"] or after["boundary_violations"]
    # Triage threshold ONLY. This is not a significance test, and no single
    # small task is sufficient to recommend disabling compaction globally.
    candidate_repeatable = (
        len(original) >= 2 and len(modified) >= 3
        and before["strict_task_passes"] == 0
        and before["agent_timeouts_after_fix_evidence"] == len(original)
        and after["strict_task_passes"] == len(modified)
        and not boundaries
    )
    return {
        "schema_version": "rescue-mitigation-evidence-v1",
        "mode": "offline_existing_receipts_only",
        "task_id": TASK_ID,
        "model": model,
        "declared_context_tokens": context,
        "variable_compared": "opencode_compaction_auto",
        "baseline_auto_compaction": "default_or_legacy_policy_not_recorded",
        "candidate_auto_compaction": "explicitly_disabled",
        "baseline": before,
        "candidate": after,
        "assessment": (
            "repeated_mitigation_observed_not_causally_verified" if candidate_repeatable
            else "inconclusive_or_mixed_evidence"
        ),
        "assessment_basis": (
            "Exploratory triage: at least 2 default-compaction timeouts with "
            "provisional fixes, and 3 explicitly-disabled completed passes "
            "at identical declared model, task, and context, with no integrity violations."
        ),
        "recommendation": "no_automatic_or_global_compaction_change",
        "warnings": [
            "Receipts are user-selected self-reported JSON, not independently signed execution evidence.",
            "The reported OpenCode configuration was constructed in a disposable workspace.",
            "Older baseline receipts may lack an explicit auto-compaction field; "
            "those are treated as possible defaults, not proof of the actual setting.",
            "The declared context is not verified as Ollama's effective per-request context.",
            "Uncontrolled differences in execution order, timeouts, machine load and runtime state remain.",
            "One tiny read-only task and few runs cannot establish causal or statistical reliability.",
            "Disabling automatic compaction globally risks context-window exhaustion on longer tasks.",
            "No model was invoked, user files inspected, or configuration modified by this assessment.",
        ],
    }


def format_terminal(report: Dict[str, Any]) -> str:
    baseline, candidate = report["baseline"], report["candidate"]
    return "\n".join((
        "LLMRig Rescue | Offline compaction mitigation evidence",
        "======================================================",
        "Model: " + report["model"],
        "Declared context: " + str(report["declared_context_tokens"]) + " tokens",
        "Legacy baselines without recorded compaction policy: " +
            str(baseline["legacy_compaction_policy_not_recorded"]),
        "Baseline: " + str(baseline["strict_task_passes"]) + "/" +
            str(baseline["sessions"]) + " strict passes; " +
            str(baseline["agent_timeouts_after_fix_evidence"]) + " timeouts with provisional fixes",
        "Compaction disabled: " + str(candidate["strict_task_passes"]) + "/" +
            str(candidate["sessions"]) + " strict passes; " +
            str(candidate["agent_timeouts_after_fix_evidence"]) + " timeouts with provisional fixes",
        "Assessment: " + report["assessment"],
        "Recommendation: " + report["recommendation"],
        "Evidence: exploratory, self-reported, not a verified causal repair",
        "No inference or persistent configuration changes were performed.",
        "",
    ))
