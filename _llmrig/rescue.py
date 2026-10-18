"""Local-only Rescue preflight and opt-in Ollama tool-call protocol canaries.

This checks one component of an agent stack. It does NOT run OpenCode, edit its
configuration, certify agent task success, or silently download any weights.
Only synthetic canary input is ever sent to the user's loopback Ollama API.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "rescue-preflight-v1"
CANARY = "rescue-constant-7"
TOOL = "llmrig_rescue_echo"
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,190}(?::[A-Za-z0-9][A-Za-z0-9._-]{0,100})?\Z")
_PUBLIC = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}\Z")
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class RescueError(ValueError):
    """Error without disclosing HTTP response bodies or local file paths."""


def validate_model(model: str) -> str:
    if not isinstance(model, str) or not _MODEL.fullmatch(model):
        raise RescueError("model must be an exact public Ollama tag (no paths or URLs)")
    repository = model.split(":", 1)[0]
    if any(part in ("", ".", "..") for part in repository.split("/")):
        raise RescueError("model contains an invalid tag component")
    return model


def local_host() -> str:
    """Require a loopback-only API; never send even synthetic probes to a remote host."""
    raw = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urllib.parse.urlsplit(raw)
        hostname = parsed.hostname
        port = parsed.port
        if parsed.scheme != "http" or not hostname or parsed.username or parsed.password:
            raise ValueError()
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment or not port:
            raise ValueError()
        if hostname != "localhost" and not ipaddress.ip_address(hostname).is_loopback:
            raise ValueError()
    except (ValueError, TypeError):
        raise RescueError("Rescue requires a local HTTP Ollama endpoint with an explicit port")
    return raw.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise RescueError("local Ollama request redirected; refusing to follow")


def _http(base: str, endpoint: str, payload: Optional[Dict[str, Any]] = None,
          timeout: int = 8) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        base + endpoint, data=data, method="POST" if payload is not None else "GET",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        # Do not route even synthetic agent-protocol probes via HTTP_PROXY.
        # The endpoint is validated as loopback by local_host().
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise RescueError("Ollama response exceeded size limit")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise RescueError("Ollama returned a non-object response")
        return result
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RescueError("local Ollama request failed; check that the service is ready")
    except (ValueError, UnicodeError):
        raise RescueError("local Ollama returned invalid JSON")


def _safe_metadata(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _PUBLIC.fullmatch(value) else None


def preflight(agent: str, model: str, context: int, base: str) -> Dict[str, Any]:
    """Only local read-only metadata requests; no agent invocation or inference."""
    installed = shutil.which("opencode") is not None
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "preflight",
        "scope": "Ollama protocol preflight, NOT end-to-end agent evaluation",
        "agent": agent,
        "agent_executable_detected": installed,
        "runtime": "ollama",
        "model": model,
        "context_tokens": context,
        "status": "blocked",
        "blockers": [],
        "unknowns": ["OpenCode tool execution and task completion have not been tested"],
        "model_metadata": {
            "architecture": None,
            "quantization": None,
            "parameters": None,
            "tools_capability": "unknown",
        },
        "probes": [],
        "comparison": None,
        "local_changes": "none (preflight only)",
    }
    if not installed:
        result["unknowns"].append("OpenCode executable not detected; agent integration not validated")
    try:
        _http(base, "/api/version")
        tags = _http(base, "/api/tags")
    except RescueError:
        result["blockers"].append("Ollama local HTTP API is unavailable")
        return result
    models = tags.get("models")
    if not isinstance(models, list) or not any(
        isinstance(item, dict) and model in (item.get("name"), item.get("model"))
        for item in models
    ):
        result["blockers"].append("Exact Ollama model tag is not installed; no download attempted")
        return result
    try:
        details = _http(base, "/api/show", {"model": model})
    except RescueError:
        result["blockers"].append("Installed model metadata could not be inspected")
        return result
    metadata = result["model_metadata"]
    raw_details = details.get("details")
    if isinstance(raw_details, dict):
        metadata["quantization"] = _safe_metadata(raw_details.get("quantization_level"))
        metadata["parameters"] = _safe_metadata(raw_details.get("parameter_size"))
    model_info = details.get("model_info")
    if isinstance(model_info, dict):
        metadata["architecture"] = _safe_metadata(model_info.get("general.architecture"))
    capabilities = details.get("capabilities")
    if isinstance(capabilities, list) and all(isinstance(item, str) for item in capabilities):
        metadata["tools_capability"] = "declared" if "tools" in capabilities else "not_declared"
    if metadata["tools_capability"] != "declared":
        result["unknowns"].append("Tool-call capability has not been positively declared by Ollama")
    result["status"] = "ready_for_probe"
    return result


def _probe_payload(model: str, context: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call the supplied llmrig_rescue_echo tool with code "
                    "'rescue-constant-7'. Do not answer with natural-language text."
                ),
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": TOOL,
                "description": "Return a fixed public diagnostic token; never runs real code.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "enum": [CANARY]}},
                    "required": ["code"],
                },
            },
        }],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": context, "num_predict": 96},
    }


def _classify(response: Dict[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, dict):
        return "malformed_response"
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return "no_single_tool_call"
    call = calls[0]
    fn = call.get("function") if isinstance(call, dict) else None
    if not isinstance(fn, dict):
        return "malformed_response"
    if fn.get("name") != TOOL:
        return "wrong_tool"
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return "wrong_arguments"
    if not isinstance(args, dict) or args.get("code") != CANARY:
        return "wrong_arguments"
    return "passed"


def measure(base: str, model: str, context: int, runs: int,
            timeout: int = 60) -> Dict[str, Any]:
    """Bounded, opt-in transport canary; no external tool function is executed."""
    outcomes: List[str] = []
    latencies: List[float] = []
    for _ in range(runs):
        start = time.monotonic()
        try:
            response = _http(base, "/api/chat", _probe_payload(model, context), timeout=timeout)
            outcome = _classify(response)
        except RescueError:
            outcome = "transport_error"
        latency = (time.monotonic() - start) * 1000
        outcomes.append(outcome)
        latencies.append(round(latency, 1))
    passed = outcomes.count("passed")
    return {
        "context_tokens": context,
        "runs": runs,
        "tool_call_passes": passed,
        "tool_call_failures": runs - passed,
        "pass_rate": round(passed / runs, 4),
        "observed_wall_latency_ms": [round(value, 1) for value in latencies],
        "outcomes": outcomes,
    }


def rescue(agent: str, model: str, context: int, runs: int,
           compare_context: Optional[int], probe: bool) -> Dict[str, Any]:
    if agent != "opencode":
        raise RescueError("only the opencode preflight adapter is supported")
    model = validate_model(model)
    if isinstance(context, bool) or not isinstance(context, int) or not 256 <= context <= 262144:
        raise RescueError("--context must be between 256 and 262144")
    if isinstance(runs, bool) or not isinstance(runs, int) or not 1 <= runs <= 5:
        raise RescueError("--runs must be between 1 and 5")
    if compare_context is not None and (
        isinstance(compare_context, bool) or not isinstance(compare_context, int)
        or not 256 <= compare_context <= 262144 or compare_context == context
    ):
        raise RescueError("--compare-context must differ and be between 256 and 262144")
    if compare_context is not None and not probe:
        raise RescueError("--compare-context requires explicit --probe")
    base = local_host()
    report = preflight(agent, model, context, base)
    if not probe or report["status"] != "ready_for_probe":
        return report
    report["mode"] = "measured_protocol_probe"
    report["local_changes"] = "Ollama inference/model memory use only; no agent config writes"
    report["probes"].append(measure(base, model, context, runs))
    if compare_context is not None:
        report["probes"].append(measure(base, model, compare_context, runs))
        first, second = report["probes"]
        delta = second["tool_call_passes"] - first["tool_call_passes"]
        report["comparison"] = {
            "baseline_context_tokens": context,
            "candidate_context_tokens": compare_context,
            "observed_pass_delta": delta,
            "assessment": (
                "candidate_observed_more_passes" if delta > 0 else
                "candidate_observed_fewer_passes" if delta < 0 else
                "no_observed_pass_difference"
            ),
            "confidence": "insufficient_to_establish_agent_improvement",
        }
    report["status"] = "measured"
    report["unknowns"].append(
        "A synthetic tool-call canary does not measure OpenCode task success or prove a repair"
    )
    report["unknowns"].append(
        "A few repeated trials cannot establish statistical reliability or model quality"
    )
    return report


def format_terminal(report: Dict[str, Any]) -> str:
    lines = [
        "LLMRig Rescue | Ollama protocol preflight",
        "======================================",
        "Agent: " + report["agent"] + " (executable " +
        ("found" if report["agent_executable_detected"] else "not detected") + ")",
        "Model: " + report["model"],
        "Context: " + str(report["context_tokens"]) + " tokens",
        "Status: " + report["status"],
    ]
    for blocker in report["blockers"]:
        lines.append("Blocker: " + blocker)
    for item in report["probes"]:
        lines.append(
            "Context " + str(item["context_tokens"]) + ": " +
            str(item["tool_call_passes"]) + "/" + str(item["runs"]) +
            " synthetic tool calls passed"
        )
        lines.append("Outcomes: " + ", ".join(item["outcomes"]))
    if report["comparison"]:
        lines.append("Comparison: " + report["comparison"]["assessment"])
        lines.append("Confidence: insufficient to establish agent improvement")
    for item in report["unknowns"]:
        lines.append("Unknown: " + item)
    lines.append("No OpenCode configuration was read or changed.")
    return "\n".join(lines) + "\n"
