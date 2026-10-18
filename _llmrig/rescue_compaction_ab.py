"""Opt-in controlled compaction on/off OpenCode A/B experiment.

Only compaction.auto differs within paired disposable fixture sessions.
No outside user project or persistent OpenCode configuration is modified.
This remains exploratory and not a statistically verified repair.
"""
from __future__ import annotations

import shutil
import time
from typing import Any, Dict, List

from . import rescue_ab


def _summary(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks = [t.get("checks") or {} for t in trials]
    n = len(trials)
    successes = sum(t.get("status") == "passed" for t in trials)
    return {
        "trials": n,
        "strict_passes": successes,
        "strict_pass_rate": round(successes / n, 4) if n else None,
        "fix_evidence": sum(all(c.get(k) is True for k in (
            "read_tool_completed", "correct_return_proposed",
            "valid_event_stream", "no_disallowed_tool_observed",
            "fixture_unchanged",
        )) for c in checks),
        "process_exits": sum(c.get("process_exit_zero") is True for c in checks),
        "timeouts": sum(t.get("failure_category") == "agent_timeout" for t in trials),
        "timeouts_with_fix": sum(
            t.get("failure_category") == "agent_timeout"
            and all(c.get(k) is True for k in (
                "read_tool_completed", "correct_return_proposed",
                "valid_event_stream", "no_disallowed_tool_observed", "fixture_unchanged"
            ))
            for t, c in zip(trials, checks)
        ),
        "explanation_only_rubric_misses": sum(
            t.get("failure_category") == "task_requirements_not_met"
            and c.get("bug_explained") is False
            and all(c.get(k) is True for k in (
                "read_tool_completed", "correct_return_proposed", "process_exit_zero",
                "no_disallowed_tool_observed", "no_error_event",
                "valid_event_stream", "fixture_unchanged",
            ))
            for t, c in zip(trials, checks)
        ),
    }


def compare_compaction(model: str, context: int, repeats: int = 1,
                       seconds: int = 90, budget_seconds: int = 600
                       ) -> Dict[str, Any]:
    """Run paired tasks, reversing condition order across tasks and repeats.

    Budget reserves full per-run timeout for BOTH sides before launching a
    pair. If a full pair cannot fit, stop rather than truncate one side.
    Failure of an individual agent task is recorded, not hidden as success.
    """
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("--compaction-repeats must be between 1 and 3")
    if type(seconds) is not int or not 20 <= seconds <= 180:
        raise ValueError("--agent-timeout must be between 20 and 180 seconds")
    if type(budget_seconds) is not int or not 120 <= budget_seconds <= 1200:
        raise ValueError("--ab-budget must be between 120 and 1200 seconds")
    if not shutil.which("opencode"):
        raise ValueError("OpenCode executable was not found")

    tasks = rescue_ab.TASKS
    result: Dict[str, Any] = {
        "schema_version": "rescue-compaction-ab-v1",
        "status": "incomplete",
        "agent": "opencode",
        "model": model,
        "declared_context_tokens": context,
        "task_suite_version": "rescue-readonly-fixed-v1",
        "task_ids": [t.task_id for t in tasks],
        "condition_variable": "opencode_compaction_auto",
        "baseline_policy": "unspecified_default",
        "candidate_policy": "explicit_auto_false",
        "repeats_per_task": repeats,
        "timeout_seconds_per_run": seconds,
        "total_budget_seconds": budget_seconds,
        "planned_agent_sessions": len(tasks) * repeats * 2,
        "executed_agent_sessions": 0,
        "trials": [],
        "groups": None,
        "observed_comparison": None,
        "diagnosis": {
            "code": "insufficient_comparable_evidence",
            "confidence": "unverified",
            "candidate_mitigation": "disable_auto_compaction_in_disposable_test_only",
            "real_configuration_recommended": False,
        },
        "recommendation": "no_automatic_or_global_compaction_change",
        "limitations": [
            "Task suite is tiny and synthetic, not representative of coding agents.",
            "Reported context is an OpenCode declaration; per-request Ollama context is unverified.",
            "Equal timeouts and alternating order reduce some confounds but do not prove causality.",
            "Compaction disabled can cause context exhaustion on larger real tasks.",
            "This executes third-party OpenCode under current OS user; not a kernel-enforced sandbox.",
            "No user project opened and no persistent user configuration intentionally changed.",
            "Raw agent messages, paths and arguments are not retained in evidence.",
        ],
    }
    deadline = time.monotonic() + budget_seconds
    halted = False
    stopped_for_budget = False
    pair_number = 0
    for repetition in range(repeats):
        for task_index, task in enumerate(tasks):
            # Do not penalize one condition by giving it less time than the other.
            if deadline - time.monotonic() < (2 * seconds + 2):
                stopped_for_budget = True
                break
            pair_number += 1
            order = (False, True)
            if (repetition + task_index) % 2:
                order = (True, False)
            for disabled in order:
                trial = rescue_ab._trial(
                    model, context, task, seconds,
                    disable_auto_compaction=disabled,
                )
                trial["pair_number"] = pair_number
                trial["repeat_index"] = repetition + 1
                trial["condition"] = (
                    "compaction_disabled" if disabled else "compaction_default"
                )
                result["trials"].append(trial)
                if (trial.get("failure_category") == "test_boundary_violation"
                        or not (trial.get("checks") or {}).get("fixture_unchanged", True)
                        or not (trial.get("checks") or {}).get("no_disallowed_tool_observed", True)):
                    halted = True
                    break
            if halted:
                break
        if halted or stopped_for_budget:
            break

    trials = result["trials"]
    result["executed_agent_sessions"] = len(trials)
    full = not halted and not stopped_for_budget and len(trials) == result["planned_agent_sessions"]
    result["status"] = (
        "completed" if full else
        "stopped_on_boundary_violation" if halted else
        "budget_exhausted"
    )
    baseline = [t for t in trials if t.get("condition") == "compaction_default"]
    candidate = [t for t in trials if t.get("condition") == "compaction_disabled"]
    base, modified = _summary(baseline), _summary(candidate)
    result["groups"] = {"baseline": base, "candidate": modified}

    # Only complete, equally-timed paired schedules can have a comparison.
    if full:
        rate = round(modified["strict_pass_rate"] - base["strict_pass_rate"], 4)
        result["observed_comparison"] = {
            "candidate_minus_baseline_strict_pass_rate": rate,
            "candidate_minus_baseline_successful_exits":
                modified["process_exits"] - base["process_exits"],
            "candidate_minus_baseline_fix_evidence":
                modified["fix_evidence"] - base["fix_evidence"],
            "direction": (
                "compaction_disabled_observed_more_passes" if rate > 0
                else "compaction_disabled_observed_fewer_passes" if rate < 0
                else "no_observed_strict_pass_difference"
            ),
            "evidence_strength": "exploratory_small_sample_not_causally_verified",
        }
    if full:
        # This is a reproducible *signature* of a suspected compaction
        # interaction, not proof that a compaction event actually occurred.
        # An explanation-only rubric miss does not invalidate correct-fix
        # evidence, but it cannot count as a strict completed task pass.
        repeatable_signature = (
            len(baseline) == len(candidate) and len(baseline) >= 3
            and base["timeouts_with_fix"] == len(baseline)
            and base["process_exits"] == 0
            and modified["strict_passes"] == len(candidate)
            and modified["timeouts"] == 0
            and all(t.get("failure_category") != "test_boundary_violation"
                    and (t.get("checks") or {}).get("fixture_unchanged") is True
                    and (t.get("checks") or {}).get("no_disallowed_tool_observed") is True
                    for t in trials)
        )
        result["diagnosis"] = {
            "code": (
                "suspected_compaction_associated_post_answer_continuation"
                if repeatable_signature else "mixed_or_insufficient_compaction_evidence"
            ),
            "confidence": "exploratory_controlled_observation_not_root_cause",
            "candidate_mitigation": "disable_auto_compaction_in_disposable_test_only",
            "real_configuration_recommended": False,
            "supporting_default_timeouts_after_fix": base["timeouts_with_fix"],
            "supporting_candidate_strict_passes": modified["strict_passes"],
            "known_unknowns": [
                "No direct OpenCode compaction event is verified.",
                "Effective per-request Ollama context is unverified.",
                "A single small task suite cannot prove general reliability.",
            ],
        }
    return result
