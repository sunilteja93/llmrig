<h1 align="center">LLMRig</h1>

<p align="center"><strong>Know what your rig can run.</strong></p>

<p align="center">The open compatibility and performance intelligence layer for local AI.</p>

<p align="center">
  <a href="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <a href="https://pypi.org/project/llmrig/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/llmrig"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sunilteja93/llmrig/main/assets/llmrig-terminal.svg" alt="LLMRig v0.7 Autopilot foundation: detect, resolve, assess, solve, and verify flow" width="100%" />
</p>

LLMRig determines viable ways to run AI models on real hardware. It keeps model,
artifact, quantization, runtime, context, and measured performance separate so
missing evidence does not become a confident recommendation.

## Autopilot foundation

LLMRig v0.7 introduces the reasoning and verification foundation for an Autopilot
for local AI. `solve` reasons across hardware × model × artifact × quantization ×
runtime × context × measured performance:

```bash
llmrig solve MODEL
llmrig solve MODEL --verify
```

v0.7 ships the reasoning and verification foundation for Autopilot, not a fully
autonomous operator: the current workflow does not automatically acquire models,
install or start runtimes, configure the system, or perform other action stages.

Default `solve` is read-only. It does not install runtimes, download models, start
services, scan arbitrary filesystem locations, execute inference, or write
benchmark output. A planning recommendation is distinct from a measured
recommendation. Verification is explicit and requires at least two comparable,
already-local, executable, and measurable configurations. Benchmark evidence
outranks heuristics, and unknown stays unknown.

Install the isolated CLI and run the safest useful first command:

```bash
pipx install llmrig
llmrig solve qwen3:0.6b
```

```text
hardware
  → model and artifact resolution
  → runtime compatibility and local availability
  → read-only solve
  → optional measured verification
  → reproducible evidence
```

LLMRig is Qwen-first today. Curated setup uses Ollama. Local race measurement also
supports llama.cpp and MLX-LM when compatible artifacts and runtimes are already
present; LLMRig does not install those native runtimes or download their artifacts.

## v0.8 runtime intelligence (development)

The v0.8 development branch introduces a read-only runtime adapter registry and
first-class oMLX observation alongside Ollama, MLX-LM, and llama.cpp:

```bash
llmrig runtimes
llmrig runtimes --json
```

The v0.8 CLI `solve` path consumes those adapter observations when matching resolved
artifacts to runtime candidates. oMLX is currently surfaced as an MLX-capable runtime
path without treating installation as model availability. LLMRig does not yet claim
oMLX execution or benchmark support; that remains a separate measured-integration
step. Generic `safetensors` files do not imply MLX/oMLX compatibility.

## Install

[`pipx`](https://pipx.pypa.io/) is recommended for the CLI:

```bash
pipx install llmrig
pipx upgrade llmrig
llmrig --version
```

If `pipx` is unavailable, use a virtual environment. This also avoids PEP 668
errors from system-managed Python installations.

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install llmrig
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install llmrig
```

LLMRig supports Python 3.9+ on macOS, Linux, and Windows and has no third-party
Python runtime dependencies. It can also run directly from a source checkout with
`python3 llmrig.py`.

## Autopilot: solve first

Analyze a curated model or exact `owner/repository` Hugging Face identifier:

```bash
llmrig solve MODEL
llmrig solve MODEL --json
llmrig solve MODEL --context 32768
llmrig solve MODEL --local-artifact llama.cpp=/path/to/model.gguf
llmrig solve MODEL --local-artifact mlx-lm=/path/to/model-directory
```

Hugging Face resolution reads metadata and file listings but does not download
weights. Native artifacts are inspected only at paths explicitly supplied with
`--local-artifact`; LLMRig does not scan arbitrary filesystem locations.

Solve reports these evidence dimensions independently:

```text
compatible != local
local != executable
executable != measurable
measurable != measured
measured != recommended
```

Unknown is not false. An observed Ollama name is local evidence, not independent
content attestation. A native model association is supplied by the user: LLMRig
checks its local structure but does not attest its identity, weights, quality, or
quantization. Private native paths stay out of public solve results.

Solve does not calculate a universal score, infer quality from throughput, or
promise a winner. It recommends a planning candidate only when the available
evidence supports exactly one runnable configuration with no comparable unresolved
alternative. An inconclusive result is valid.

### Verify explicitly

```bash
llmrig solve MODEL \
  --local-artifact llama.cpp=/path/to/model.gguf \
  --verify
```

`solve --verify` executes only compatible, already-local, executable, measurable
candidates. It requires at least two comparable configurations and reuses the
unchanged `race-v2` workload, two-sample rule, 5% comparison threshold, and balanced
Pareto decision. It does not pull models, install or start runtimes, scan for
artifacts, or write passports. Fewer than two candidates makes verification
unavailable without executing inference; a competitor failure invalidates the
comparison; multiple Pareto members remain inconclusive.

Solve exit codes are:

- `0`: analysis completed, including a completed but inconclusive verification
- `1`: an operational solve or attempted verification failure
- `2`: invalid/unresolvable input or unavailable requested verification

## Minimal Python SDK

```python
import llmrig

result = llmrig.solve("qwen3:0.6b")
print(result.plan.recommendation_status)
```

The stable entry point is:

```python
llmrig.solve(
    model,
    *,
    context=None,
    local_artifacts=(),
    verify=False,
) -> llmrig.SolveResult
```

It prints nothing and never exits the process. `local_artifacts` accepts
`RUNTIME=PATH` strings. Invalid or unresolvable input raises `SolveInputError`;
operational failures raise `SolveEngineError`. `SolveResult` and `SolveCandidate`
are the deliberate public result contracts. `_llmrig` is private implementation.

## Measurement commands

The measurement commands share local execution machinery but answer different
questions:

| Command | Purpose | Result rule | Writes passports |
|---|---|---|---|
| `solve --verify` | Verify a solve candidate set | Balanced Pareto decision, or inconclusive | No |
| `race` | Measure at least two local configurations | Metric-specific generation, prompt, and latency results | Optional |
| `choose` | Explain one measured objective | `generation`, `prompt`, `latency`, or balanced | No |
| `optimize` | Expose measured tradeoffs | Unranked noise-aware Pareto frontier | No |
| `bench` | Run the full Ollama benchmark | Throughput, residency data when available, and smoke tests | Optional |

Race two already-local native configurations, or combine one with an installed
equivalent Ollama build:

```bash
llmrig race MODEL \
  --local-artifact llama.cpp=/path/to/model.gguf \
  --local-artifact mlx-lm=/path/to/model-directory
```

`race` never installs or downloads. At most one explicit artifact per native
runtime is accepted. A GGUF target must be a non-empty `.gguf` file. An MLX-LM
target must have an immediate non-empty `config.json` and at least one immediate
non-empty `model*.safetensors` file. These structural checks do not prove the
runtime can load the artifact or that differently packaged artifacts have identical
weights or quality.

Race exit `0` means at least two competitors were measured successfully, `1` means
execution failed and invalidated the comparison, and `2` means the race is
unavailable or unresolved. Results within 5% are inconclusive. There is no
composite result or model-quality claim.

Choose an explicit objective from the same measured race path:

```bash
llmrig choose MODEL --objective generation
llmrig choose MODEL --objective prompt --json
llmrig choose MODEL --objective latency
llmrig choose MODEL --objective balanced
```