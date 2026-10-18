"""Opt-in, evidence-scoped OpenCode coding-task evaluation on a disposable fixture.

This is an agent *test*, not a machine-level sandbox or automatic repair.
It never inspects the user's own project; raw agent output is never published.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

TASK_ID = "python-parity-readonly-v1"
SOURCE = "def is_even(n):\n    return n % 2 == 1\n"
PROMPT = (
    "Use the read tool to inspect math_ops.py in this project. "
    "Identify the logical bug in is_even and give the exact corrected "
    "return statement. Do not edit any files, run shell commands, "
    "or access other directories."
)
MAX_CAPTURE = 1024 * 1024
MAX_EVENTS = 512
FIX = re.compile(r"return\s+n\s*%\s*2\s*==\s*0\b")
WRONG = re.compile(r"n\s*%\s*2\s*==\s*1\b")
SAFE_TOOLS = frozenset(("read", "glob", "grep", "list"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _configuration(model: str, context: int, agent_steps: Optional[int] = None,
                   disable_auto_compaction: bool = False) -> Dict[str, Any]:
    """Explicit local provider plus deny-by-default read-only agent."""
    permissions = {
        "*": "deny",
        "bash": "deny",
        "edit": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "task": "deny",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "external_directory": "deny",
    }
    config = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "enabled_providers": ["ollama"],
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Local Ollama",
                "options": {"baseURL": "http://127.0.0.1:11434/v1"},
                "models": {
                    model: {
                        "name": "Rescue local model",
                        "limit": {"context": context, "output": 1024},
                    }
                },
            }
        },
        "permission": permissions,
        "agent": {
            "rescue-readonly": {
                "mode": "primary",
                "permission": permissions,
            }
        },
    }
    if agent_steps is not None:
        # OpenCode's documented step budget applies only to this disposable
        # read-only agent. It is not a process timeout or persistent setting.
        config["agent"]["rescue-readonly"]["steps"] = agent_steps
    if disable_auto_compaction:
        # Explicit isolated hypothesis test: never alter the user's configs.
        # Turning off auto compaction could produce a context-limit error.
        config["compaction"] = {"auto": False}
    return config


def _environment(home: Path, project: Path, config: Dict[str, Any]) -> Dict[str, str]:
    # Do not inherit API keys, proxy credentials, arbitrary OpenCode instructions,
    # or installed plugins from the normal home directory.
    allow = ("PATH", "SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "LC_CTYPE")
    env = {key: os.environ[key] for key in allow if key in os.environ}
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["XDG_CONFIG_HOME"] = str(home / "config")
    env["XDG_DATA_HOME"] = str(home / "data")
    env["XDG_CACHE_HOME"] = str(home / "cache")
    env["XDG_STATE_HOME"] = str(home / "state")
    env["OPENCODE_CONFIG"] = str(project / "opencode.json")
    env["OPENCODE_CONFIG_DIR"] = str(home / "empty-opencode")
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, separators=(",", ":"))
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "0"
    env["CI"] = "1"
    return env


def _read_events(raw: bytes) -> Tuple[Dict[str, bool], int]:
    """Parse OpenCode JSONL without storing user-visible raw text or paths."""
    tools_seen: List[str] = []
    completed_read = False
    text_parts: List[str] = []
    error_event = False
    malformed_event = False
    event_count = 0

    for line in raw.splitlines():
        if not line.strip():
            continue
        event_count += 1
        if event_count > MAX_EVENTS:
            malformed_event = True
            break
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            malformed_event = True
            break
        if not isinstance(event, dict):
            malformed_event = True
            break
        kind = event.get("type")
        if kind == "error":
            error_event = True
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        if kind == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"][:16000])
        if kind == "tool_use":
            tool = part.get("tool")
            if not isinstance(tool, str):
                malformed_event = True
                continue
            tools_seen.append(tool.lower())
            state = part.get("state")
            if tool.lower() == "read" and isinstance(state, dict):
                args = state.get("input")
                # The tool must have successfully read the exact fixture.
                if state.get("status") == "completed" and isinstance(args, dict):
                    target = args.get("filePath", args.get("path"))
                    if isinstance(target, str) and Path(target).name == "math_ops.py":
                        completed_read = True

    combined = "\n".join(text_parts)
    return {
        "read_tool_completed": completed_read,
        "bug_explained": bool(WRONG.search(combined) and "odd" in combined.lower()),
        "correct_return_proposed": bool(FIX.search(combined)),
        "no_disallowed_tool_observed": all(tool in SAFE_TOOLS for tool in tools_seen),
        "no_error_event": not error_event,
        "valid_event_stream": not malformed_event and event_count > 0,
    }, event_count


def _invoke(cmd: List[str], cwd: Path, env: Dict[str, str], seconds: int
            ) -> Tuple[str, int, bytes, bool]:
    """Run once with a timeout; raw stdout/stderr stay in disposable files."""
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        kwargs: Dict[str, Any] = {
            "cwd": str(cwd), "env": env, "stdin": subprocess.DEVNULL,
            "stdout": output, "stderr": errors,
        }
        if os.name != "nt":
            kwargs["start_new_session"] = True
        process = subprocess.Popen(cmd, **kwargs)
        timed_out = False
        try:
            returncode = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            returncode = process.returncode
        output.seek(0, os.SEEK_END)
        length = output.tell()
        output.seek(0)
        bounded = output.read(MAX_CAPTURE + 1)
        oversized = length > MAX_CAPTURE or len(bounded) > MAX_CAPTURE
        return ("timeout" if timed_out else "output_limit" if oversized else "exited",
                returncode, bounded[:MAX_CAPTURE], timed_out)


def evaluate(model: str, context: int, seconds: int = 120,
             agent_steps: Optional[int] = None,
             disable_auto_compaction: bool = False) -> Dict[str, Any]:
    """Require opt-in from the caller. No network except the selected local model is intended.

    OpenCode is third-party executable code with OS user permissions; this
    temporary HOME and config reduce exposure but are NOT an OS sandbox.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, int) or not 20 <= seconds <= 180:
        raise ValueError("agent timeout must be between 20 and 180 seconds")
    if agent_steps is not None and (
        isinstance(agent_steps, bool) or not isinstance(agent_steps, int)
        or not 2 <= agent_steps <= 12
    ):
        raise ValueError("--agent-steps must be an integer between 2 and 12")
    if not isinstance(disable_auto_compaction, bool):
        raise ValueError("--agent-no-auto-compact must be boolean")
    opencode = shutil.which("opencode")
    if not opencode:
        raise ValueError("OpenCode is not on PATH")
    report: Dict[str, Any] = {
        "task_id": TASK_ID,
        "agent_steps_limit": agent_steps,
        "auto_compaction_disabled": disable_auto_compaction,
        "status": "failed",
        "model": model,
        "agent": "opencode",
        "fixture_sha256": _sha(SOURCE.encode("utf-8")),
        "agent_invoked": False,
        "checks": {},
        "json_events_seen": 0,
        "event_sequence": None,
        "completion_diagnostic": None,
        "elapsed_seconds": None,
        "failure_category": None,
        "raw_agent_output_included": False,
        "limitations": [
            "Temporary HOME and tool permissions are not an operating-system sandbox.",
            "Single simple debugging task does not establish general agent reliability.",
            "OpenCode may create isolated session state in the temporary directory.",
        ],
    }

    with tempfile.TemporaryDirectory(prefix="llmrig-rescue-agent-") as directory:
        root = Path(directory)
        project = root / "workspace"
        home = root / "home"
        project.mkdir()
        home.mkdir()
        (home / "empty-opencode").mkdir()
        fixture = project / "math_ops.py"
        fixture.write_text(SOURCE, encoding="utf-8")
        config = _configuration(
            model, context, agent_steps=agent_steps,
            disable_auto_compaction=disable_auto_compaction,
        )
        (project / "opencode.json").write_text(json.dumps(config), encoding="utf-8")
        before = _sha(fixture.read_bytes())
        cmd = [
            opencode, "run", "--format", "json",
            "--agent", "rescue-readonly",
            "--model", "ollama/" + model,
            PROMPT,
        ]
        start = time.monotonic()
        try:
            status, code, raw, timed_out = _invoke(
                cmd, project, _environment(home, project, config), seconds
            )
        except OSError:
            report["failure_category"] = "agent_process_failed_to_start"
            return report
        report["elapsed_seconds"] = round(time.monotonic() - start, 2)
        report["agent_invoked"] = True
        checks, count = _read_events(raw)
        from .rescue_trace import summarize
        report["event_sequence"] = summarize(raw, FIX)
        checks["process_exit_zero"] = code == 0 and not timed_out and status == "exited"
        checks["fixture_unchanged"] = fixture.is_file() and _sha(fixture.read_bytes()) == before
        report["checks"] = checks
        report["json_events_seen"] = count
        if timed_out:
            report["failure_category"] = "agent_timeout"
            report["completion_diagnostic"] = (
                "provisional_fix_observed_before_agent_timeout"
                if checks["correct_return_proposed"] and checks["valid_event_stream"]
                else "agent_timeout_without_verified_fix_evidence"
            )
        elif status == "output_limit":
            report["failure_category"] = "event_output_limit"
        elif code != 0:
            report["failure_category"] = "agent_nonzero_exit"
        elif not checks["valid_event_stream"]:
            report["failure_category"] = "invalid_agent_event_stream"
        elif not checks["no_disallowed_tool_observed"] or not checks["fixture_unchanged"]:
            report["failure_category"] = "test_boundary_violation"
        elif not all(checks.values()):
            report["failure_category"] = "task_requirements_not_met"
        else:
            report["status"] = "passed"
    return report
