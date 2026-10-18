"""Explicitly shareable, privacy-filtered Markdown/JSON reports from receipts.

Share reports are generated only when requested. No network upload is performed.
We never serialize raw receipt payloads, local paths, secrets or endpoints.
"""
from __future__ import annotations

import html
import json
import math
import re
from typing import Any, Dict, Mapping, Optional

from .privacy import validate_public_text


REPORT_SCHEMA_VERSION = "1.0"


def _public_string(value: Any, label: str, *, identity: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid " + label)
    validate_public_text(value, "report " + label, identity=identity)
    # Prevent newlines and inline Markdown from becoming executable markup.
    cleaned = re.sub(r"[\x00-\x1f\x7f\x60]", " ", value).strip()
    if not cleaned or len(cleaned) > 256:
        raise ValueError("invalid " + label)
    return cleaned


def _number(payload: Mapping[str, Any], key: str) -> Optional[float]:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("invalid measurement: " + key)
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("invalid measurement: " + key)
    return round(number, 4)


def build_share_report(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Whitelist only public, historically recorded facts from a persisted receipt."""
    if not isinstance(receipt, Mapping):
        raise ValueError("invalid receipt")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("receipt is missing candidate evidence")
    status = _public_string(receipt.get("status"), "status")
    if status not in ("completed", "blocked", "failed"):
        raise ValueError("invalid receipt status")
    model = _public_string(receipt.get("logical_model_id") or receipt.get("model"), "model", identity=True)
    runtime = _public_string(candidate.get("runtime"), "runtime")
    fmt = _public_string(candidate.get("artifact_format"), "artifact format")
    quantization = candidate.get("quantization")
    if quantization is not None:
        quantization = _public_string(quantization, "quantization")
    artifact_revision = candidate.get("artifact_revision")
    if artifact_revision is not None:
        artifact_revision = _public_string(artifact_revision, "artifact revision", identity=True)
    context = candidate.get("context_tokens")
    if context is not None and (
        isinstance(context, bool) or not isinstance(context, int) or context <= 0
    ):
        raise ValueError("invalid context")
    verified = receipt.get("verification")
    metrics = None
    if verified is not None:
        if not isinstance(verified, Mapping):
            raise ValueError("invalid verification")
        runs = verified.get("measured_runs")
        if isinstance(runs, bool) or not isinstance(runs, int) or runs <= 0:
            raise ValueError("invalid verification run count")
        metrics = {
            "method": _public_string(verified.get("method_version"), "verification method"),
            "runs": runs,
            "generation_tps": _number(verified, "generation_tps"),
            "prompt_eval_tps": _number(verified, "prompt_eval_tps"),
            "total_latency_s": _number(verified, "total_latency_s"),
        }
    return {
        "report_schema": REPORT_SCHEMA_VERSION,
        "project": "LLMRig",
        "model": model,
        "runtime": runtime,
        "artifact_format": fmt,
        "quantization": quantization,
        "artifact_revision": artifact_revision,
        "context_tokens": context,
        "status": status,
        "verification": metrics,
        "source": "local Autopilot receipt (not independently attested)",
        "project_url": "https://github.com/sunilteja93/llmrig",
    }


def _markdown_value(value: Any) -> str:
    """Avoid treating receipt-controlled text as Markdown or HTML markup."""
    escaped = html.escape(str(value), quote=True)
    return re.sub(r"([\\\x60*_{}\[\]()#+.!|])", r"\\\1", escaped)


def format_markdown(report: Mapping[str, Any]) -> str:
    """Render only whitelisted report fields, with no private machine claims."""
    lines = [
        "# LLMRig Rig Report",
        "",
        "**Model:** " + _markdown_value(report["model"]),
        "**Runtime:** " + _markdown_value(report["runtime"]),
        "**Format:** " + _markdown_value(report["artifact_format"]),
        "**Status:** " + _markdown_value(report["status"]),
    ]
    for key, title in (
        ("quantization", "Quantization"),
        ("artifact_revision", "Artifact revision"),
        ("context_tokens", "Context tokens"),
    ):
        if report.get(key) is not None:
            lines.append("**" + title + ":** " + _markdown_value(report[key]))
    lines.append("")
    verification = report.get("verification")
    if verification:
        lines.extend([
            "## Measured verification",
            "",
            "**Method:** " + _markdown_value(verification["method"]),
            "**Measured runs:** " + str(verification["runs"]),
        ])
        for key, label, unit in (
            ("generation_tps", "Generation", " tokens/s"),
            ("prompt_eval_tps", "Prompt evaluation", " tokens/s"),
            ("total_latency_s", "Total latency", " s"),
        ):
            value = verification.get(key)
            if value is not None:
                lines.append("**" + label + ":** " + str(value) + unit)
        missing = [
            label for key, label in (
                ("generation_tps", "generation speed"),
                ("prompt_eval_tps", "prompt evaluation speed"),
            )
            if verification.get(key) is None
        ]
        if missing:
            lines.append(
                "**Unavailable throughput:** "
                + ", ".join(missing)
                + " not recorded in this receipt. "
                "Tokens/s is not inferred from total latency."
            )
    else:
        lines.extend(["## Verification", "", "No measurements recorded."])
    lines.extend([
        "",
        "Source: local Autopilot receipt. Self-reported, not independently attested.",
        "Performance measurements do not establish model quality.",
        "Machine hardware is intentionally omitted because receipts do not record it.",
        "",
        "https://github.com/sunilteja93/llmrig",
        "",
    ])
    return "\n".join(lines)


def format_terminal(report: Mapping[str, Any]) -> str:
    """Human-friendly local summary, distinct from opt-in Markdown sharing."""
    lines = [
        "LLMRig Rig Report",
        "=================",
        "Model:   " + str(report["model"]),
        "Runtime: " + str(report["runtime"]),
        "Status:  " + str(report["status"]),
    ]
    measurements = report.get("verification")
    if measurements:
        lines.append("Measured runs: " + str(measurements["runs"]))
        if measurements.get("generation_tps") is not None:
            lines.append("Generation: " + str(measurements["generation_tps"]) + " tokens/s")
        if measurements.get("prompt_eval_tps") is not None:
            lines.append("Prompt: " + str(measurements["prompt_eval_tps"]) + " tokens/s")
        if measurements.get("total_latency_s") is not None:
            lines.append("Reported latency: " + str(measurements["total_latency_s"]) + " s")
        if measurements.get("generation_tps") is None or measurements.get("prompt_eval_tps") is None:
            lines.append("Throughput: one or more speeds not recorded; not inferred from latency.")
    else:
        lines.append("Measured verification: not recorded")
    lines.extend([
        "",
        "Self-reported local evidence; not a model-quality claim.",
        "To create a portable Markdown card: llmrig report --share",
        "No data was uploaded.",
        "",
    ])
    return "\n".join(lines)


def format_json(report: Mapping[str, Any]) -> str:
    return json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
