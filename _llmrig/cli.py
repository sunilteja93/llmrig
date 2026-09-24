"""LLMRig CLI front controller.

The established command surface continues to delegate to the stable top-level
``llmrig`` CLI. Adapter-backed v0.8/v0.9 commands live here so Autopilot can
evolve without destabilizing legacy behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence, Tuple

from .autopilot_apply import (
    ApplyError,
    AutopilotReceipt,
    apply_autopilot_plan,
)
from .autopilot_plan import AutopilotExecutionPlan, build_autopilot_plan
from .hf_bridge import hf_metadata_for_legacy
from .omlx_acquisition_provenance import omlx_acquisition_for_legacy
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


def _add_model_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Curated model name or exact Hugging Face repository.")
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Requested context length in tokens.",
    )


def _plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrig plan",
        description=(
            "Build a deterministic Autopilot plan from current machine, model, "
            "artifact, and runtime evidence. This command is read-only."
        ),
    )
    _add_model_context_arguments(parser)
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    return parser


def _apply_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrig apply",
        description=(
            "Recompute and apply an approved Autopilot plan. The supplied plan ID "
            "must exactly match current evidence before any mutation occurs."
        ),
    )
    _add_model_context_arguments(parser)
    parser.add_argument(
        "--plan-id",
        required=True,
        help="Exact plan ID produced by `llmrig plan`.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly approve all mutating actions in the matching plan.",
    )
    parser.add_argument("--json", action="store_true", help="Emit receipt JSON.")
    return parser


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrig run",
        description=(
            "Plan, explicitly approve if needed, apply, and verify one local-AI setup."
        ),
    )
    _add_model_context_arguments(parser)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly approve all mutating actions without an interactive prompt.",
    )
    parser.add_argument("--json", action="store_true", help="Emit final receipt JSON.")
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
        label = "Setup path" if plan.recommendation_status == "setup_selected" else "Selected"
        print(f"{label}: {selected.runtime} · {selected.artifact_format}")
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


def _print_receipt(receipt: AutopilotReceipt) -> None:
    print("\nLLMRig Autopilot Receipt")
    print("========================")
    print(f"Receipt: {receipt.receipt_id}")
    print(f"Plan:    {receipt.plan_id}")
    print(f"Status:  {receipt.status}")
    print(f"Model:   {receipt.logical_model_id or receipt.model}")
    print(f"Runtime: {receipt.runtime}")
    if receipt.endpoint:
        print(f"Endpoint: {receipt.endpoint}")
    print("\nActions")
    for action in receipt.actions:
        print(f"- {action.kind}: {action.status} — {action.detail}")
    if receipt.verification is not None:
        measured = receipt.verification
        print("\nMeasured verification")
        if measured.generation_tps is not None:
            print(f"Generation: {measured.generation_tps} tok/s")
        if measured.prompt_eval_tps is not None:
            print(f"Prompt:     {measured.prompt_eval_tps} tok/s")
        if measured.total_latency_s is not None:
            print(f"Latency:    {measured.total_latency_s} s")
        print(f"Runs:       {measured.measured_runs}")
    print("\nReceipt contains no private filesystem locator or secret.")


def _build_plan(model: str, context: Optional[int], legacy: object) -> AutopilotExecutionPlan:
    if context is not None and context <= 0:
        raise ValueError("--context must be a positive integer")
    result = legacy.solve(model, context=context, verify=False)
    profile = legacy.hardware_profile()
    machine = {
        "os": profile.get("os"),
        "arch": profile.get("arch"),
        "cpu": profile.get("cpu"),
        "ram_gib": profile.get("ram_gib"),
    }
    return build_autopilot_plan(result.to_dict(), machine)


def command_plan(args: argparse.Namespace, legacy: object) -> int:
    try:
        plan = _build_plan(args.model, args.context, legacy)
    except ValueError as exc:
        print(f"llmrig plan: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveInputError as exc:
        print(f"llmrig plan: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveEngineError as exc:
        print(f"llmrig plan: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        _print_autopilot_plan(plan)
    return 0


def _approve(plan: AutopilotExecutionPlan, *, yes: bool, json_mode: bool) -> bool:
    if not plan.has_mutations:
        return True
    if yes:
        return True
    if json_mode or not sys.stdin.isatty():
        return False
    try:
        answer = input("\nApply this plan? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _apply_current_plan(
    plan: AutopilotExecutionPlan,
    args: argparse.Namespace,
    legacy: object,
    *,
    show_plan: bool,
) -> int:
    if show_plan and not args.json:
        _print_autopilot_plan(plan)
    if plan.blocked:
        if args.json:
            print(json.dumps({"status": "blocked", "plan": plan.to_dict()}, indent=2))
        else:
            print("\nAutopilot cannot apply this plan until its blockers are resolved.")
        return 2

    approved = _approve(plan, yes=args.yes, json_mode=args.json)
    if plan.has_mutations and not approved:
        print(
            "llmrig: mutating Autopilot actions require interactive approval or --yes",
            file=sys.stderr,
        )
        return 2

    try:
        receipt = apply_autopilot_plan(
            plan,
            legacy,
            explicit_user_approval=approved,
            persist_receipt=True,
        )
    except (ApplyError, PermissionError) as exc:
        print(f"llmrig: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(receipt.to_dict(), indent=2))
    else:
        _print_receipt(receipt)
    return 0 if receipt.status == "completed" else 1


def command_apply(args: argparse.Namespace, legacy: object) -> int:
    try:
        plan = _build_plan(args.model, args.context, legacy)
    except ValueError as exc:
        print(f"llmrig apply: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveInputError as exc:
        print(f"llmrig apply: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveEngineError as exc:
        print(f"llmrig apply: {exc}", file=sys.stderr)
        return 1

    if args.plan_id != plan.plan_id:
        print(
            "llmrig apply: plan drift detected; current evidence no longer matches the approved plan ID",
            file=sys.stderr,
        )
        if not args.json:
            print(f"Current plan: {plan.plan_id}", file=sys.stderr)
        return 2
    return _apply_current_plan(plan, args, legacy, show_plan=True)


def command_run(args: argparse.Namespace, legacy: object) -> int:
    try:
        plan = _build_plan(args.model, args.context, legacy)
    except ValueError as exc:
        print(f"llmrig run: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveInputError as exc:
        print(f"llmrig run: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveEngineError as exc:
        print(f"llmrig run: {exc}", file=sys.stderr)
        return 1
    return _apply_current_plan(plan, args, legacy, show_plan=True)


def _print_augmented_help() -> int:
    import llmrig as legacy

    legacy.build_parser().print_help()
    print("\nv0.8 runtime intelligence:")
    print("  runtimes            Inspect oMLX, Ollama, MLX-LM, and llama.cpp readiness.")
    print("\nv0.9 Autopilot:")
    print("  plan MODEL          Build a deterministic, read-only execution plan.")
    print("  apply MODEL         Apply an exact matching plan after explicit approval.")
    print("  run MODEL           Plan, approve, apply, and verify in one workflow.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "runtimes":
        args = _runtimes_parser().parse_args(values[1:])
        return command_runtimes(args)
    if values in (["--help"], ["-h"]):
        return _print_augmented_help()

    import llmrig as legacy

    with hf_metadata_for_legacy(legacy):
        if values and values[0] in {"plan", "apply", "run", "solve"}:
            with (
                adapter_capabilities_for_legacy(legacy),
                omlx_verify_for_legacy(legacy),
                omlx_acquisition_for_legacy(legacy),
            ):
                if values[0] == "plan":
                    return command_plan(_plan_parser().parse_args(values[1:]), legacy)
                if values[0] == "apply":
                    return command_apply(_apply_parser().parse_args(values[1:]), legacy)
                if values[0] == "run":
                    return command_run(_run_parser().parse_args(values[1:]), legacy)
                return legacy.main(values)

        return legacy.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
