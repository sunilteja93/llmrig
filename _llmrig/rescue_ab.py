"""Exploratory A/B comparison on fixed, disposable OpenCode coding tasks.

This is NOT a repair engine, a representative coding benchmark, or an OS
sandbox. No model or agent settings are changed outside each temporary run.
Never persist or return raw agent messages, shell output, paths, or secrets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .rescue_agent import (
    MAX_EVENTS, SAFE_TOOLS, _configuration, _environment, _invoke, _sha,
)

SCHEMA = "rescue-ab-v1"
MIN_RUN_BUDGET = 20


class Task(NamedTuple):
    task_id: str
    filename: str
    source: str
    prompt: str
    bug_expression: str
    explanation_pattern: str
    correction_pattern: str


TASKS = (
    Task(
        "parity-v1", "math_ops.py",
        "def is_even(n):\n    return n % 2 == 1\n",
        "Use the read tool to inspect math_ops.py. Explain why is_even is "
        "incorrect and show the exact corrected return statement. Do not edit files.",
        "n % 2 == 1", r"\bodd\b", r"\breturn\s+n\s*%\s*2\s*==\s*0\b",
    ),
    Task(
        "nonnegative-clamp-v1", "bounds.py",
        "def nonnegative(n):\n    return max(0, -n)\n",
        "Use the read tool to inspect bounds.py. Explain the logic bug in "
        "nonnegative and show the exact corrected return statement. Do not edit files.",
        "max(0, -n)", r"\bnegat\w*\b|\bsign\b", r"\breturn\s+max\s*\(\s*0\s*,\s*n\s*\)",
    ),
    Task(
        "prefix-v1", "text_ops.py",
        "def starts_with_a(text):\n    return text.endswith('a')\n",
        "Use the read tool to inspect text_ops.py. Explain the logic bug in "
        "starts_with_a and show the exact corrected return statement. Do not edit files.",
        "endswith", r"\bend\b|\bsuffix\b|\blast\b",
        r"\breturn\s+text\.startswith\s*\(\s*['\"]a['\"]\s*\)",
    ),
)


def _grade_events(raw: bytes, task: Task, workspace: Path
                  ) -> Tuple[Dict[str, bool], int]:
    """Check only recognized JSONL fields, never include raw event content."""
    count = 0
    malformed = False
    error = False
    read_completed = False
    observed_tools: List[str] = []
    text_parts: List[str] = []
    fixture_path = (workspace / task.filename).resolve()

    for line in raw.splitlines():
        if not line.strip():
            continue
        count += 1
        if count > MAX_EVENTS:
            malformed = True
            break
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            malformed = True
            break
        if not isinstance(event, dict):
            malformed = True
            break
        kind = event.get("type")
        if kind == "error":
            error = True
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        if kind == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"][:16000])
        if kind != "tool_use":
            continue
        tool = part.get("tool")
        if not isinstance(tool, str):
            malformed = True
            continue
        observed_tools.append(tool.lower())
        state = part.get("state")
        if tool.lower() == "read" and isinstance(state, dict) and state.get("status") == "completed":
            arguments = state.get("input")
            if isinstance(arguments, dict):
                target = arguments.get("filePath", arguments.get("path"))
                if isinstance(target, str):
                    candidate = Path(target)
                    # Ensure a read of an unrelated private file cannot pass.
                    if not candidate.is_absolute():
                        candidate = workspace / candidate
                    if candidate.resolve() == fixture_path:
                        read_completed = True

    combined = "\n".join(text_parts)
    # A narrow, fixed heuristic, not an LLM-as-judge.
    # Correct code must be in model answer text, not tool output.
    return {
        "read_tool_completed": read_completed,
        "bug_explained": bool(task.bug_expression in combined and re.search(
            task.explanation_pattern, combined, re.IGNORECASE)),
        "correct_return_proposed": bool(re.search(task.correction_pattern, combined)),
        "no_disallowed_tool_observed": all(t in SAFE_TOOLS for t in observed_tools),
        "no_error_event": not error,
        "valid_event_stream": not malformed and count > 0,
    }, count


def _trial(model: str, context: int, task: Task, seconds: int
           ) -> Dict[str, Any]:
    """Run a single fixed task with a fresh project and fresh HOME."""
    outcome: Dict[str, Any] = {
        "task_id": task.task_id,
        "context_tokens": context,
        "status": "failed",
        "checks": {},
        "events_seen": 0,
        "duration_seconds": None,
        "failure_category": None,
        "fixture_sha256": _sha(task.source.encode("utf-8")),
    }
    with tempfile.TemporaryDirectory(prefix="llmrig-rescue-ab-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        home = root / "home"
        workspace.mkdir()
        home.mkdir()
        (home / "empty-opencode").mkdir()
        fixture = workspace / task.filename
        fixture.write_text(task.source, encoding="utf-8")
        config = _configuration(model, context)
        (workspace / "opencode.json").write_text(json.dumps(config), encoding="utf-8")
        original = _sha(fixture.read_bytes())
        opencode = shutil.which("opencode")
        if not opencode:
            outcome["failure_category"] = "opencode_missing"
            return outcome
        command = [
            opencode, "run", "--format", "json", "--agent", "rescue-readonly",
            "--model", "ollama/" + model, task.prompt,
        ]
        start = time.monotonic()
        try:
            process_status, code, output, timed_out = _invoke(
                command, workspace, _environment(home, workspace, config), seconds
            )
        except OSError:
            outcome["failure_category"] = "agent_process_failed_to_start"
            return outcome
        outcome["duration_seconds"] = round(time.monotonic() - start, 2)
        checks, events = _grade_events(output, task, workspace)
        checks["process_exit_zero"] = code == 0 and not timed_out and process_status == "exited"
        checks["fixture_unchanged"] = fixture.is_file() and _sha(fixture.read_bytes()) == original
        outcome["checks"] = checks
        outcome["events_seen"] = events
        if timed_out:
            outcome["failure_category"] = "agent_timeout"
        elif process_status == "output_limit":
            outcome["failure_category"] = "event_output_limit"
        elif code != 0:
            outcome["failure_category"] = "agent_nonzero_exit"
        elif not checks["valid_event_stream"]:
            outcome["failure_category"] = "invalid_agent_event_stream"
        elif not checks["no_disallowed_tool_observed"] or not checks["fixture_unchanged"]:
            outcome["failure_category"] = "test_boundary_violation"
        elif not all(checks.values()):
            outcome["failure_category"] = "task_requirements_not_met"
        else:
            outcome["status"] = "passed"
    return outcome


def compare(model: str, baseline_context: int, candidate_context: int,
            repeats: int = 2, seconds: int = 90, budget_seconds: int = 600
            ) -> Dict[str, Any]:
    """Bounded alternating-order A/B trial with no automatic recommendation.

    Runs 3 tasks x 2 contexts x repeats (default 12 agent sessions).
    The same model is used on both sides, changing only declared context.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 3:
        raise ValueError("--ab-repeats must be between 1 and 3")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 20 <= seconds <= 180:
        raise ValueError("--agent-timeout must be between 20 and 180 seconds")
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, int) or not 120 <= budget_seconds <= 1200:
        raise ValueError("--ab-budget must be between 120 and 1200 seconds")
    if baseline_context == candidate_context:
        raise ValueError("A/B context values must differ")
    if not shutil.which("opencode"):
        raise ValueError("OpenCode executable not found")

    result: Dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "incomplete",
        "model": model,
        "agent": "opencode",
        "baseline_context_tokens": baseline_context,
        "candidate_context_tokens": candidate_context,
        "repeats_per_task": repeats,
        "task_ids": [task.task_id for task in TASKS],
        "suite_version": "rescue-readonly-fixed-v1",
        "planned_agent_sessions": len(TASKS) * 2 * repeats,
        "completed_agent_sessions": 0,
        "trials": [],
        "summary": None,
        "observed_comparison": None,
        "limitations": [
            "Only requested context allocation changes; prompts are short and do not stress context capacity.",
            "Exploratory, small synthetic task suite; not a representative coding benchmark.",
            "Results do not establish statistically reliable improvement or autonomous repair.",
            "Temp HOME and OpenCode tool policy are not a kernel-enforced sandbox.",
            "Raw agent output is not retained in this report.",
        ],
        "recommendation": "no_verified_configuration_improvement",
    }
    deadline = time.monotonic() + budget_seconds
    halted_for_safety = False
    # A/B, B/A alternation reduces systematic cold-start and order bias.
    for repeat_index in range(repeats):
        for task in TASKS:
            configs = (baseline_context, candidate_context)
            if (repeat_index + TASKS.index(task)) % 2:
                configs = tuple(reversed(configs))
            for context in configs:
                remaining = deadline - time.monotonic()
                if remaining < MIN_RUN_BUDGET:
                    break
                one = _trial(model, context, task, min(seconds, int(remaining)))
                one["repeat_index"] = repeat_index + 1
                result["trials"].append(one)
                if one["failure_category"] == "test_boundary_violation":
                    halted_for_safety = True
                    break
            if halted_for_safety or deadline - time.monotonic() < MIN_RUN_BUDGET:
                break
        if halted_for_safety or deadline - time.monotonic() < MIN_RUN_BUDGET:
            break

    trials = result["trials"]
    result["completed_agent_sessions"] = len(trials)
    completed = len(trials) == result["planned_agent_sessions"] and not halted_for_safety
    result["status"] = "completed" if completed else (
        "stopped_on_boundary_violation" if halted_for_safety else "budget_exhausted"
    )
    def _solution_evidence(trial: Dict[str, Any]) -> bool:
        """Observed answer evidence, even if the agent later timed out.

        This is explicitly NOT a completed or independently executed fix.
        Keep syntax graders separate from final agent process status.
        """
        checks = trial.get("checks") or {}
        return all(checks.get(key) is True for key in (
            "read_tool_completed", "correct_return_proposed",
            "valid_event_stream", "no_disallowed_tool_observed",
            "fixture_unchanged",
        ))

    def _explanation_only_miss(trial: Dict[str, Any]) -> bool:
        checks = trial.get("checks") or {}
        return (
            trial.get("failure_category") == "task_requirements_not_met"
            and checks.get("bug_explained") is False
            and all(checks.get(key) is True for key in (
                "read_tool_completed", "correct_return_proposed",
                "no_disallowed_tool_observed", "no_error_event",
                "valid_event_stream", "process_exit_zero", "fixture_unchanged",
            ))
        )

    group = []
    for context in (baseline_context, candidate_context):
        subset = [t for t in trials if t["context_tokens"] == context]
        successes = sum(t["status"] == "passed" for t in subset)
        group.append({
            "context_tokens": context,
            "observed_passes": successes,
            "observed_trials": len(subset),
            "observed_pass_rate": round(successes / len(subset), 4) if subset else None,
            # Independent facets of the same sessions. A timeout may preserve
            # correct answer evidence, but can never become a strict task pass.
            "solution_evidence_seen": sum(_solution_evidence(t) for t in subset),
            "process_exits_successful": sum(
                (t.get("checks") or {}).get("process_exit_zero") is True for t in subset
            ),
            "agent_timeouts": sum(t.get("failure_category") == "agent_timeout" for t in subset),
            "explanation_rubric_only_misses": sum(_explanation_only_miss(t) for t in subset),
            "failures_by_category": {
                reason: sum(t["failure_category"] == reason for t in subset)
                for reason in sorted(set(t["failure_category"] for t in subset if t["failure_category"]))
            },
        })
    result["summary"] = {"baseline": group[0], "candidate": group[1]}
    if completed:
        a, b = group
        delta = round(b["observed_pass_rate"] - a["observed_pass_rate"], 4)
        result["observed_comparison"] = {
            "candidate_minus_baseline_pass_rate": delta,
            "comparison_metric": "strict_task_passes_including_successful_agent_exit",
            "candidate_minus_baseline_successful_exits": (
                b["process_exits_successful"] - a["process_exits_successful"]
            ),
            "candidate_minus_baseline_solution_evidence": (
                b["solution_evidence_seen"] - a["solution_evidence_seen"]
            ),
            "assessment": (
                "candidate_observed_more_passes" if delta > 0 else
                "candidate_observed_fewer_passes" if delta < 0 else
                "no_observed_difference"
            ),
            "evidence_strength": "exploratory_not_statistically_validated",
        }
    return result
