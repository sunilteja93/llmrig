from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import llmrig
from _llmrig import cli
from _llmrig.runtime_adapters import RuntimeEvidence, RuntimeProbe


def _probe(
    runtime: str,
    *,
    installed: bool,
    service_available: Optional[bool],
    version: Optional[str] = None,
    cli_path: Optional[str] = None,
    formats: Tuple[str, ...] = (),
    api: Optional[str] = None,
    blockers: Tuple[str, ...] = (),
    unknowns: Tuple[str, ...] = (),
) -> RuntimeProbe:
    return RuntimeProbe(
        runtime=runtime,
        installed=installed,
        service_available=service_available,
        version=version,
        cli_path=cli_path,
        supported_artifact_formats=formats,
        supported_platforms=("Darwin",),
        supported_architectures=("arm64",),
        execution_api=api,
        evidence=(RuntimeEvidence("test", "unit", "fixture"),),
        blockers=blockers,
        unknowns=unknowns,
    )


def test_runtime_probe_public_path_hides_home(monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: Path("/Users/private-user"))
    probe = _probe(
        "omlx",
        installed=True,
        service_available=True,
        cli_path="/Users/private-user/.omlx/bin/omlx",
        formats=("MLX",),
        api="OpenAI-compatible /v1",
    )
    assert probe.to_dict()["cli_path"] == "~/.omlx/bin/omlx"


def test_runtimes_json_reports_adapter_state(monkeypatch, capsys):
    monkeypatch.setattr(
        llmrig,
        "hardware_profile",
        lambda: {
            "os": "Darwin",
            "arch": "arm64",
            "cpu": "Apple M4 Max",
            "ram_gib": 48.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "probe_runtimes",
        lambda: (
            _probe(
                "omlx",
                installed=True,
                service_available=True,
                version="omlx 1.0",
                formats=("MLX",),
                api="OpenAI-compatible /v1",
            ),
            _probe(
                "ollama",
                installed=True,
                service_available=False,
                version="ollama 0.1",
                formats=("Ollama",),
                api="Ollama HTTP API",
                unknowns=("service is not responding",),
            ),
        ),
    )

    assert cli.main(["runtimes", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "0.1"
    assert payload["machine"]["cpu"] == "Apple M4 Max"
    assert payload["summary"] == {"known_runtimes": 2, "installed": 2, "ready": 1}
    assert payload["runtimes"][0]["runtime"] == "omlx"
    assert payload["runtimes"][0]["status"] == "ready"
    assert payload["runtimes"][1]["status"] == "stopped"


def test_runtimes_human_output_is_compact(monkeypatch, capsys):
    monkeypatch.setattr(
        llmrig,
        "hardware_profile",
        lambda: {
            "os": "Darwin",
            "arch": "arm64",
            "cpu": "Apple M4 Max",
            "ram_gib": 48.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "probe_runtimes",
        lambda: (
            _probe(
                "omlx",
                installed=True,
                service_available=True,
                version="omlx 1.0",
                formats=("MLX",),
                api="OpenAI-compatible /v1",
            ),
            _probe(
                "llama.cpp",
                installed=False,
                service_available=None,
                formats=("GGUF",),
                api="local CLI",
            ),
        ),
    )

    assert cli.main(["runtimes"]) == 0
    output = capsys.readouterr().out
    assert "LLMRig Runtime Intelligence" in output
    assert "Apple M4 Max" in output
    assert "omlx" in output
    assert "llama.cpp" in output
    assert "1 of 2 runtimes detected; 1 currently ready." in output


def test_existing_commands_delegate_to_legacy_main(monkeypatch):
    seen = []
    monkeypatch.setattr(llmrig, "main", lambda argv: seen.append(argv) or 17)
    assert cli.main(["check"]) == 17
    assert seen == [["check"]]
