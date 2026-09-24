"""Thin v0.8 CLI front controller.

Existing commands continue to delegate to the stable top-level ``llmrig`` CLI.
New adapter-backed commands can be introduced here incrementally without
rewriting or destabilizing the legacy command surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

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
    print(f"Machine: {cpu} · {ram_text} · {machine.get('os') or 'Unknown OS'} / {machine.get('arch') or 'unknown'}")
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


def _print_augmented_help() -> int:
    import llmrig as legacy

    legacy.build_parser().print_help()
    print("\nv0.8 runtime intelligence:")
    print("  runtimes            Inspect oMLX, Ollama, MLX-LM, and llama.cpp readiness.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "runtimes":
        args = _runtimes_parser().parse_args(values[1:])
        return command_runtimes(args)
    if values in (["--help"], ["-h"]):
        return _print_augmented_help()

    import llmrig as legacy

    if values and values[0] == "solve":
        # Keep the established solve implementation and schema, but feed its
        # capability, inventory, and explicit verification boundaries from the
        # v0.8 adapter layer. These patches exist only for this command call.
        with adapter_capabilities_for_legacy(legacy), omlx_verify_for_legacy(legacy):
            return legacy.main(values)

    return legacy.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
