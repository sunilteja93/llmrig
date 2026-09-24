"""LLMRig CLI front controller for runtime intelligence and Autopilot."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Optional, Sequence

from .autopilot_apply import (
    ApplyError,
    AutopilotReceipt,
    apply_autopilot_plan,
    load_receipt,
)
from .autopilot_evidence import record_receipt_evidence
from .autopilot_plan import AutopilotExecutionPlan, build_autopilot_plan
from .hf_bridge import hf_metadata_for_legacy
from .omlx_acquisition_provenance import omlx_acquisition_for_legacy
from .omlx_verify import omlx_verify_for_legacy
from .runtime_adapters import RuntimeProbe, probe_runtimes
from .runtime_bridge import adapter_capabilities_for_legacy


RUNTIME_SCHEMA_VERSION = "0.1"


def _runtime_status(probe: RuntimeProbe) -> str:
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
    rows = []
    for probe in probe_runtimes():
        item = probe.to_dict()
        item["status"] = _runtime_status(probe)
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
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _print_runtime_table(payload: dict) -> None:
    machine = payload["machine"]
    cpu = machine.get("cpu") or "Unknown CPU"
    ram = machine.get("ram_gib")
    print("\nLLMRig Runtime Intelligence")
    print("===========================")
    print(
        f"Machine: {cpu} · {ram if ram else 'RAM unknown'}"
        + (" GiB" if ram else "")
        + f" · {machine.get('os') or 'Unknown OS'} / {machine.get('arch') or 'unknown'}"
    )
    print()
    headers = ("Runtime", "Installed", "Status", "Formats", "Interface")
    widths = (12, 10, 12, 16, 24)
    print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
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


def _model_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("model", help="Curated model name or exact Hugging Face repository.")
    parser.add_argument("--context", type=int, default=None, help="Requested context length.")
    return parser


def _runtimes_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmrig runtimes")
    parser.add_argument("--json", action="store_true")
    return parser


def _plan_parser() -> argparse.ArgumentParser:
    parser = _model_parser(
        "llmrig plan",
        "Build a deterministic, read-only Autopilot plan from current evidence.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _apply_parser() -> argparse.ArgumentParser:
    parser = _model_parser(
        "llmrig apply",
        "Recompute and apply an exact matching Autopilot plan.",
    )
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _run_parser() -> argparse.ArgumentParser:
    parser = _model_parser(
        "llmrig run",
        "Plan, explicitly approve if needed, apply, and verify a local-AI setup.",
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _verify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrig verify",
        description="Re-observe current evidence and re-verify a prior Autopilot receipt.",
    )
    parser.add_argument("receipt", nargs="?", default="latest")
    parser.add_argument("--json", action="store_true")
    return parser


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


def _print_plan(plan: AutopilotExecutionPlan) -> None:
    machine = dict(plan.machine)
    ram = machine.get("ram_gib")
    print("\nLLMRig Autopilot Plan")
    print("=====================")
    print(f"Plan:    {plan.plan_id}")
    print(
        f"Machine: {machine.get('cpu') or 'Unknown CPU'} · "
        f"{ram if ram else 'RAM unknown'}" + (" GiB" if ram else "")
    )
    print(f"Model:   {plan.logical_model_id or plan.model}")
    if plan.candidates:
        print("\nRuntime       Format          Quant         Local         Executable")
        print("------------  --------------  ------------  ------------  ------------")
        for item in plan.candidates:
            values = (
                item.runtime,
                item.artifact_format,
                item.quantization or "unknown",
                item.local_availability,
                item.execution,
            )
            widths = (12, 14, 12, 12, 12)
            print(
                "  ".join(
                    _clip(value, width).ljust(width)
                    for value, width in zip(values, widths)
                )
            )
    if plan.selected_candidate_id:
        selected = next(
            item for item in plan.candidates if item.candidate_id == plan.selected_candidate_id
        )
        label = "Setup path" if plan.recommendation_status == "setup_selected" else "Selected"
        print(f"\n{label}: {selected.runtime} · {selected.artifact_format}")
        print(f"Reason: {plan.recommendation_reason}")
        print("\nPlanned actions")
        for index, action in enumerate(plan.actions, 1):
            mode = "changes local state" if action.mutating else "verification"
            print(f"{index}. {action.description} [{mode}]")
            for blocker in action.blockers:
                print(f"   blocker: {blocker}")
    else:
        print("\nSelection: inconclusive")
        print(f"Reason: {plan.recommendation_reason}")
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
    if receipt.verification:
        result = receipt.verification
        print("\nMeasured verification")
        if result.generation_tps is not None:
            print(f"Generation: {result.generation_tps} tok/s")
        if result.prompt_eval_tps is not None:
            print(f"Prompt:     {result.prompt_eval_tps} tok/s")
        if result.total_latency_s is not None:
            print(f"Latency:    {result.total_latency_s} s")
        print(f"Runs:       {result.measured_runs}")
    print("\nReceipt contains no private filesystem locator or secret.")


def command_runtimes(args: argparse.Namespace) -> int:
    payload = _runtime_payload()
    print(json.dumps(payload, indent=2)) if args.json else _print_runtime_table(payload)
    return 0


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
    print(json.dumps(plan.to_dict(), indent=2)) if args.json else _print_plan(plan)
    return 0


def _approval(plan: AutopilotExecutionPlan, args: argparse.Namespace) -> bool:
    if not plan.has_mutations or args.yes:
        return True
    if args.json or not sys.stdin.isatty():
        return False
    try:
        return input("\nApply this plan? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _apply(
    plan: AutopilotExecutionPlan,
    args: argparse.Namespace,
    legacy: object,
    show_plan: bool,
) -> int:
    if show_plan and not args.json:
        _print_plan(plan)
    if plan.blocked:
        if args.json:
            print(json.dumps({"status": "blocked", "plan": plan.to_dict()}, indent=2))
        else:
            print("\nAutopilot cannot apply this plan until its blockers are resolved.")
        return 2
    approved = _approval(plan, args)
    if plan.has_mutations and not approved:
        print(
            "llmrig: mutating actions require interactive approval or --yes",
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
    if receipt.verification is not None:
        try:
            record_receipt_evidence(plan, receipt)
        except OSError as exc:
            print(f"llmrig: warning: RigGraph evidence could not be persisted: {exc}", file=sys.stderr)
    print(json.dumps(receipt.to_dict(), indent=2)) if args.json else _print_receipt(receipt)
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
    return _apply(plan, args, legacy, True)


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
    return _apply(plan, args, legacy, True)


def command_verify(args: argparse.Namespace, legacy: object) -> int:
    try:
        previous = load_receipt(args.receipt)
    except ApplyError as exc:
        print(f"llmrig verify: {exc}", file=sys.stderr)
        return 2
    model = previous.get("model")
    candidate_payload = previous.get("candidate")
    if not isinstance(model, str) or not isinstance(candidate_payload, dict):
        print("llmrig verify: receipt is missing model/candidate identity", file=sys.stderr)
        return 2
    receipt_context = candidate_payload.get("context_tokens")
    if (
        not isinstance(receipt_context, int)
        or isinstance(receipt_context, bool)
        or receipt_context <= 0
    ):
        receipt_context = None
    try:
        plan = _build_plan(model, receipt_context, legacy)
    except (ValueError, legacy.SolveInputError) as exc:
        print(f"llmrig verify: {exc}", file=sys.stderr)
        return 2
    except legacy.SolveEngineError as exc:
        print(f"llmrig verify: {exc}", file=sys.stderr)
        return 1

    expected_runtime = candidate_payload.get("runtime")
    expected_artifact = candidate_payload.get("artifact_id")
    selected = next(
        (
            item
            for item in plan.candidates
            if item.runtime == expected_runtime and item.artifact_id == expected_artifact
        ),
        None,
    )
    if selected is None:
        print(
            "llmrig verify: current evidence no longer exposes the receipt's exact runtime/artifact candidate",
            file=sys.stderr,
        )
        return 2
    if plan.selected_candidate_id != selected.candidate_id:
        print(
            "llmrig verify: current evidence no longer uniquely selects the receipt's configuration",
            file=sys.stderr,
        )
        return 2
    expected_format = candidate_payload.get("artifact_format")
    expected_quantization = candidate_payload.get("quantization")
    if (
        selected.artifact_format != expected_format
        or selected.quantization != expected_quantization
    ):
        print(
            "llmrig verify: current evidence no longer matches the receipt's exact artifact configuration",
            file=sys.stderr,
        )
        return 2

    artifact_revision = candidate_payload.get("artifact_revision")
    if artifact_revision is not None and not isinstance(artifact_revision, str):
        print("llmrig verify: receipt artifact revision is invalid", file=sys.stderr)
        return 2

    verified_candidate = replace(
        selected,
        context_tokens=receipt_context or legacy.RACE_CONTEXT,
        artifact_revision=artifact_revision,
    )
    verify_actions = tuple(
        action for action in plan.actions if action.kind.value == "verify"
    )
    if len(verify_actions) != 1:
        print(
            "llmrig verify: current plan does not expose exactly one verification action",
            file=sys.stderr,
        )
        return 2
    verification_candidates = tuple(
        verified_candidate if item.candidate_id == selected.candidate_id else item
        for item in plan.candidates
    )
    verification_plan = AutopilotExecutionPlan(
        plan_id=str(previous.get("plan_id") or plan.plan_id),
        model=plan.model,
        logical_model_id=plan.logical_model_id,
        machine=plan.machine,
        candidates=verification_candidates,
        recommendation_status="verification_selected",
        selected_candidate_id=verified_candidate.candidate_id,
        recommendation_reason=(
            "The receipt's exact configuration was re-observed for verification only."
        ),
        actions=verify_actions,
        blockers=(),
        unknowns=plan.unknowns,
    )
    args.yes = True
    return _apply(verification_plan, args, legacy, False)


def _print_augmented_help() -> int:
    import llmrig as legacy

    legacy.build_parser().print_help()
    print("\nRuntime intelligence:")
    print("  runtimes            Inspect oMLX, Ollama, MLX-LM, and llama.cpp readiness.")
    print("\nv0.9 Autopilot:")
    print("  plan MODEL          Build a deterministic, read-only execution plan.")
    print("  apply MODEL         Apply an exact matching plan after explicit approval.")
    print("  verify [RECEIPT]    Re-observe and re-verify current state from a receipt.")
    print("  run MODEL           Plan, approve, apply, and verify in one workflow.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "runtimes":
        return command_runtimes(_runtimes_parser().parse_args(values[1:]))
    if values in (["--help"], ["-h"]):
        return _print_augmented_help()

    import llmrig as legacy

    with hf_metadata_for_legacy(legacy):
        if values and values[0] in {"plan", "apply", "verify", "run", "solve"}:
            with adapter_capabilities_for_legacy(legacy), omlx_verify_for_legacy(
                legacy
            ), omlx_acquisition_for_legacy(legacy):
                if values[0] == "plan":
                    return command_plan(_plan_parser().parse_args(values[1:]), legacy)
                if values[0] == "apply":
                    return command_apply(_apply_parser().parse_args(values[1:]), legacy)
                if values[0] == "verify":
                    return command_verify(_verify_parser().parse_args(values[1:]), legacy)
                if values[0] == "run":
                    return command_run(_run_parser().parse_args(values[1:]), legacy)
                return legacy.main(values)
        return legacy.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
