"""LLMRig CLI front controller.

The established command surface continues to delegate to the stable top-level
``llmrig`` CLI. Adapter-backed v0.8/v0.9 commands live here so Autopilot can
evolve without destabilizing legacy behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .autopilot_plan import AutopilotExecutionPlan, build_autopilot_plan
from .hf_bridge import hf_metadata_for_legacy
from .omlx_verify import omlx_verify_for_legacy
from .runtime_adapters import RuntimeProbe, probe_runtimes
from .runtime_bridge import adapter_capabilities_for_legacy


RUNTIME_SCHEMA_VERSION = "0.1"


def _status(probe: RuntimeProbe) -> str:
    if not probe.installed:
        return "not detected"
    if probe.blockers:
        return "blocked"
    if probe.service_available is False and probe.execution_api in {
        "OpenAI-compatible /v1",
        "Ollama HTTP API",
    }:
        return "stopped"
    return "ready"


def _runtime_payload() -> dict:
    # Import lazily so the adapter package remains independently testable and
    # so all existing public llmrig imports keep their current behavior.
    import llmrig as legacy

    profile = legacy.hardware_profile()
    probes = probe_runtimes()
    rows = []
    for probe in probes:
        item = probe.to_dict()
        item["status"] = _status(probe)
        rows.append(item)

    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "machine": {
            "os": profile.get("os"),
            "arch": profile.get("arch"),
            "cpu": profile.get("cpu"),
            "ram_gib": profile.get("ram_gib"),
        },
        "summary": {
            "known_runtimes": len(rows),
            "installed": sum(1 for item in rows if item["installed"]),
            "ready": sum(1 for item in rows if item["status"] == "ready"),
        },
        "runtimes": rows,
    }


def _clip(value: object, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(1, width - 1)] + "…"


def _print_runtime_table(payload: dict) -> None:
    machine = payload["machine"]
    print("\nLLMRig Runtime Intelligence")
    print("===========================")
    cpu = machine.get("cpu") or "Unknown CPU"
    ram = machine.get("ram_gib")
    ram_text = f"{ram} GiB" if ram else "RAM unknown"
    print(
        f"Machine: {cpu} · {ram_text} · "
        f"{machine.get('os') or 'Unknown OS'} / {machine.get('arch') or 'unknown'}"
    )
    print()

    headers = ("Runtime", "Installed", "Status", "Formats", "Interface")
    widths = (12, 10, 12, 16, 24)
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join(("-" * width) for width in widths))
    for item in payload["runtimes"]:
        values = (
            item["runtime"],
            "yes" if item["installed"] else "no",
            item["status"],
            ", ".join(item["supported_artifact_formats"]) or "unknown",
            item["execution_api"] or "unknown",
        )
        print(
            "  ".join(
                _clip(value, width).ljust(width)
                for value, width in zip(values, widths)
            )
        )

    summary = payload["summary"]
    print(
        f"\n{summary['installed']} of {summary['known_runtimes']} runtimes detected; "
        f"{summary['ready']} currently ready."
    )
    print("\nEvidence notes")
    for item in payload["runtimes"]:
        if not item["installed"] and not item["blockers"] and not item["unknowns"]:
            continue
        print(f"- {item['runtime']}:")
        if item["version"]:
            print(f"  version: {item['version']}")
        for blocker in item["blockers"]:
            print(f"  blocker: {blocker}")
        for unknown in item["unknowns"]:
            print(f"  unknown: {unknown}")


def command_runtimes(args: argparse.Namespace) -> int:
    payload = _runtime_payload()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_runtime_table(payload)
    return 0


def _runtimes_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrig runtimes",
        description="Inspect local inference runtimes without installing, starting, or executing them.",
    )
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    return parser


def _plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrig plan",
        description=(
            "Build a deterministic Autopilot plan from current machine, model, "
            "artifact, and runtime evidence. This command is read-only."
        ),
    )
    parser.add_argument("model", help="Curated model name or exact Hugging Face repository.")
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Requested context length in tokens.",
    )
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    return parser


def _print_autopilot_plan(plan: AutopilotExecutionPlan) -> None:
    payload = plan.to_dict()
    machine = payload["machine"]
    cpu = machine.get("cpu") or "Unknown CPU"
    ram = machine.get("ram_gib")
    ram_text = f"{ram} GiB" if ram else "RAM unknown"

    print("\nLLMRig Autopilot Plan")
    print("=====================")
    print(f"Plan:    {plan.plan_id}")
    print(f"Machine: {cpu} · {ram_text}")
    print(f"Model:   {plan.logical_model_id or plan.model}")
    print()

    if plan.candidates:
        headers = ("Runtime", "Format", "Quant", "Local", "Executable")
        widths = (12, 14, 12, 12, 12)
        print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
        print("  ".join(("-" * width) for width in widths))
        for candidate in plan.candidates:
            values = (
                candidate.runtime,
                candidate.artifact_format,
                candidate.quantization or "unknown",
                candidate.local_availability,
                candidate.execution,
            )
            print(
                "  ".join(
                    _clip(value, width).ljust(width)
                    for value, width in zip(values, widths)
                )
            )
        print()

    if plan.selected_candidate_id:
        selected = next(
            item for item in plan.candidates if item.candidate_id == plan.selected_candidate_id
        )
        print(f"Selected: {selected.runtime} · {selected.artifact_format}")
        print(f"Reason:   {plan.recommendation_reason}")
        print("\nPlanned actions")
        for index, action in enumerate(plan.actions, 1):
            mutation = "changes local state" if action.mutating else "verification"
            print(f"{index}. {action.description} [{mutation}]")
            for blocker in action.blockers:
                print(f"   blocker: {blocker}")
    else:
        print("Selection: inconclusive")
        print(f"Reason:    {plan.recommendation_reason}")

    if plan.blockers:
        print("\nApply blockers")
        for blocker in plan.blockers:
            print(f"- {blocker}")
    if plan.unknowns:
        print("\nUnknowns")
        for unknown in plan.unknowns:
            print(f"- {unknown}")

    print("\nNo action has been taken.")


def command_plan(args: argparse.Namespace, legacy: object) -> int:
    if args.context is not None and args.context <= 0:
        print("llmrig plan: --context must be a positive integer", file=sys.stderr)
        return 2

    try:
        result = legacy.solve(args.model, context=args.context, verify=False)
    except legacy.SolveInputError as exc:
        print(f"llmrig plan: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveEngineError as exc:
        print(f"llmrig plan: {exc}", file=sys.stderr)
        return 1

    profile = legacy.hardware_profile()
    machine = {
        "os": profile.get("os"),
        "arch": profile.get("arch"),
        "cpu": profile.get("cpu"),
        "ram_gib": profile.get("ram_gib"),
    }
    plan = build_autopilot_plan(result.to_dict(), machine)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        _print_autopilot_plan(plan)
    return 0


def _print_augmented_help() -> int:
    import llmrig as legacy

    legacy.build_parser().print_help()
    print("\nv0.8 runtime intelligence:")
    print("  runtimes            Inspect oMLX, Ollama, MLX-LM, and llama.cpp readiness.")
    print("\nv0.9 Autopilot:")
    print("  plan MODEL          Build a deterministic, read-only local-AI execution plan.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "runtimes":
        args = _runtimes_parser().parse_args(values[1:])
        return command_runtimes(args)
    if values in (["--help"], ["-h"]):
        return _print_augmented_help()

    import llmrig as legacy

    # Hub metadata hardening is read-only and applies to every command that
    # resolves Hugging Face metadata. It never downloads or mutates models.
    with hf_metadata_for_legacy(legacy):
        if values and values[0] == "plan":
            args = _plan_parser().parse_args(values[1:])
            # Planning needs the same provenance-safe runtime/inventory view as
            # solve, but explicitly disables verification and performs no actions.
            with adapter_capabilities_for_legacy(legacy), omlx_verify_for_legacy(legacy):
                return command_plan(args, legacy)

        if values and values[0] == "solve":
            # Keep the established solve implementation and schema, but feed its
            # capability, inventory, and explicit verification boundaries from the
            # adapter layer. These patches exist only for this command call.
            with adapter_capabilities_for_legacy(legacy), omlx_verify_for_legacy(legacy):
                return legacy.main(values)

        return legacy.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
