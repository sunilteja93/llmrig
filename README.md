<h1 align="center">LLMRig</h1>

<p align="center"><strong>Evidence-driven control plane for local AI.</strong></p>

<p align="center">Detect runtimes. Resolve artifacts. Decide from evidence. Run locally. Verify with measurement.</p>

<p align="center">
  <a href="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <a href="https://pypi.org/project/llmrig/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/llmrig"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sunilteja93/llmrig/main/assets/llmrig-terminal.svg" alt="LLMRig v0.8 runtime intelligence and evidence-driven solve flow" width="100%" />
</p>

LLMRig answers a deceptively hard local-AI question: **what can this machine actually run, through which runtime, with what evidence?**

It keeps hardware, model identity, artifact format, quantization, runtime capability,
local availability, execution state, context, and measured performance separate so
missing evidence never silently becomes a confident claim.

```text
unknown != false
compatible != local
local != executable
executable != measurable
measurable != measured
measured performance != model quality
```

## What v0.8 adds

LLMRig v0.8 turns the v0.7 Autopilot foundation into a runtime-aware local-AI
control plane foundation:

- a common runtime-adapter registry for **oMLX, Ollama, MLX-LM, and llama.cpp**
- `llmrig runtimes` for read-only runtime readiness and capability inspection
- Hugging Face-first artifact resolution with conservative format, quantization,
  context, and provenance rules
- first-class oMLX inventory/provenance and explicit `solve --verify` execution
- authenticated local oMLX support through `OMLX_API_KEY` without exposing secrets
- cross-runtime `race-v2` measurement that leaves unavailable metrics unknown
- no arbitrary filesystem scans and no surprise model/runtime installation

The longer-term Autopilot direction is **Detect → Decide → Configure → Run → Verify**.
v0.8 deliberately keeps configuration and acquisition actions read-only/manual;
mutating plan/apply actions are the next roadmap stage.

## Quick start

Install the isolated CLI:

```bash
pipx install llmrig
llmrig --version
```

Inspect the local runtime surface:

```bash
llmrig runtimes
```

Then analyze a curated model or an exact Hugging Face repository:

```bash
llmrig solve qwen3:0.6b
llmrig solve mlx-community/Qwen3.5-27B-4bit
```

Nothing is downloaded or executed by default.

## Runtime intelligence

```bash
llmrig runtimes
llmrig runtimes --json
```

LLMRig reports runtime installation/readiness separately from model locality and
execution support. The v0.8 registry currently covers:

| Runtime | Detection | LLMRig execution / measurement | Primary artifact evidence |
|---|---|---|---|
| oMLX | Yes | Yes, for provenance-backed local models | MLX |
| Ollama | Yes | Yes | Ollama-managed artifacts |
| MLX-LM | Yes | Yes, for explicit local artifacts | MLX |
| llama.cpp | Yes | Yes, for explicit local artifacts | GGUF |

Runtime detection is read-only. LLMRig does not install these runtimes or start
services while probing them.

## Solve first

```bash
llmrig solve MODEL
llmrig solve MODEL --json
llmrig solve MODEL --context 32768
```

`solve` constructs independent evidence dimensions for each candidate:

```text
discovery
compatibility
runtime availability
local availability
execution
measurement capability
measurement
recommendation
```

A candidate can therefore be compatible but not local, local but not executable,
or measurable but not measured. An inconclusive result is a valid result.

LLMRig does not calculate a universal score or infer model quality from throughput.
Planning recommendations only appear when the available evidence supports them.

### Explicit local artifacts

LLMRig never scans arbitrary directories for native model files. Supply a locator
explicitly when you want a native local artifact considered:

```bash
llmrig solve MODEL --local-artifact llama.cpp=/path/to/model.gguf
llmrig solve MODEL --local-artifact mlx-lm=/path/to/model-directory
```

A user-supplied path establishes a local association, not independent content
attestation. LLMRig checks bounded structure, keeps the private locator out of
public solve output, and leaves unknown identity/quantization facts unknown.

## Hugging Face resolution

Exact `owner/repository` identifiers are resolved through read-only Hugging Face
metadata. LLMRig may read the repository's small `config.json` when necessary for
structured format, quantization, or context evidence; it does **not** download model
weights during resolution.

Important evidence rules include:

- `.gguf` establishes GGUF packaging; filename quantization is accepted only when
  exactly one recognized token is present
- generic `.safetensors` does not by itself establish MLX/oMLX compatibility
- an `mlx` tag/path hint alone is not proof of MLX packaging
- MLX packaging requires stronger structured evidence, such as explicit
  `library_name=mlx` or an MLX hint backed by the MLX-LM quantization contract
- conflicting context, quantization, base-model provenance, or incomplete shard
  groupings fail closed instead of being guessed through

Discovery metadata is evidence, not installation trust.

## oMLX in v0.8

LLMRig can observe an API-visible oMLX model and associate it with an exact Hugging
Face repository only when provenance is defensible:

1. oMLX directly reports the exact source repository; or
2. the completed oMLX Hugging Face download registry reports the exact repository
   and the API-visible local model mapping is unique and unambiguous.

Display names alone are not accepted as provenance.

Authenticated local endpoints are supported through:

```bash
export OMLX_API_KEY="..."
llmrig runtimes
```

The API key, admin session cookie, filesystem paths, and private execution locator
are never serialized into public solve output.

## Verify explicitly

Default solve is read-only. Measurement requires explicit intent:

```bash
llmrig solve MODEL \
  --local-artifact mlx-lm=/path/to/model-directory \
  --verify
```

`solve --verify` requires at least two comparable, already-local, executable,
measurable candidates and reuses the deterministic `race-v2` workload.

For oMLX, LLMRig prefers server-reported prompt/generation timing and throughput.
If the server reports only `total_time`, LLMRig records **latency only** and leaves
throughput unmeasured. It does not synthesize tokens/second from incomparable data.

Balanced decisions remain inconclusive when too few comparable measured dimensions
remain or multiple Pareto tradeoffs survive.

Solve exit codes:

- `0` — analysis completed, including completed-but-inconclusive verification
- `1` — operational solve or attempted verification failure
- `2` — invalid/unresolvable input or unavailable requested verification

## Minimal Python SDK

```python
import llmrig

result = llmrig.solve("mlx-community/Qwen3.5-27B-4bit")
print(result.plan.recommendation_status)
```

Stable entry point:

```python
llmrig.solve(
    model,
    *,
    context=None,
    local_artifacts=(),
    verify=False,
) -> llmrig.SolveResult
```

The SDK prints nothing and never exits the process. Invalid/unresolvable input raises
`SolveInputError`; operational failures raise `SolveEngineError`. `SolveResult` and
`SolveCandidate` are deliberate public contracts. `_llmrig` is private implementation.

## Measurement commands

| Command | Purpose | Decision semantics |
|---|---|---|
| `solve --verify` | Verify a solve candidate set | balanced Pareto or inconclusive |
| `race` | Compare local configurations | metric-specific generation/prompt/latency results |
| `choose` | Explain one measured objective | generation, prompt, latency, or balanced |
| `optimize` | Expose measured tradeoffs | unranked noise-aware Pareto frontier |
| `bench` | Full Ollama benchmark | throughput, residency when available, smoke tests |

Example native race:

```bash
llmrig race MODEL \
  --local-artifact llama.cpp=/path/to/model.gguf \
  --local-artifact mlx-lm=/path/to/model-directory
```

Results within the configured 5% race threshold are inconclusive. Performance
measurement does not establish model quality.

## Installation

[`pipx`](https://pipx.pypa.io/) is recommended:

```bash
pipx install llmrig
pipx upgrade llmrig
```

Or use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install llmrig
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install llmrig
```

LLMRig supports Python 3.9+ on macOS, Linux, and Windows and has no third-party
Python runtime dependencies. A source checkout can also run `python3 llmrig.py`.

## Architecture direction

```text
                     LLMRig
                  ┌───────────┐
                  │ RigGraph  │
                  └─────┬─────┘
                        │
             ┌──────────┴──────────┐
             │   Autopilot Engine  │
             └──────────┬──────────┘
                        │
     ┌──────────┬───────┼───────┬──────────┐
     ↓          ↓       ↓       ↓          ↓
   oMLX       Ollama  MLX-LM llama.cpp   future
```

v0.8 establishes the adapter/evidence foundation. v0.9 is planned to add explicit
plan/apply actions for acquisition, configuration, and launch. See
[`ROADMAP_V1.md`](https://github.com/sunilteja93/llmrig/blob/main/ROADMAP_V1.md).

## Project docs

- [`CHANGELOG.md`](https://github.com/sunilteja93/llmrig/blob/main/CHANGELOG.md) — canonical release history
- [`ROADMAP_V1.md`](https://github.com/sunilteja93/llmrig/blob/main/ROADMAP_V1.md) — product direction
- [`RELEASING.md`](https://github.com/sunilteja93/llmrig/blob/main/RELEASING.md) — GitHub Release and PyPI process
- [`SECURITY.md`](https://github.com/sunilteja93/llmrig/blob/main/SECURITY.md) — security policy and trust boundaries
- [`CONTRIBUTING.md`](https://github.com/sunilteja93/llmrig/blob/main/CONTRIBUTING.md) — contribution guide

## License

MIT
