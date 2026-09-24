<h1 align="center">LLMRig</h1>

<p align="center"><strong>Autopilot for local AI.</strong></p>

<p align="center">Give LLMRig a model and a machine. It plans the evidenced local execution path, changes nothing without permission, applies the approved setup, verifies reality, and records what happened.</p>

<p align="center">
  <a href="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <a href="https://pypi.org/project/llmrig/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/llmrig"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sunilteja93/llmrig/main/assets/llmrig-terminal.svg" alt="LLMRig runtime intelligence and evidence-driven local AI flow" width="100%" />
</p>

LLMRig sits above local inference runtimes. **oMLX, Ollama, MLX-LM, and llama.cpp are execution paths; LLMRig decides what the current evidence supports for this model on this machine.**

It deliberately keeps facts separate:

```text
unknown != false
compatible != local
local != executable
executable != measurable
measurable != measured
measured != recommended
measured performance != model quality
discovery metadata != installation trust
```

## The v0.9 Autopilot flow

### 1. Plan — read only

```bash
llmrig plan mlx-community/Qwen3.5-27B-4bit
```

A plan can include:

- machine and exact model identity
- viable runtime/artifact candidates
- artifact format, quantization, and context evidence
- required acquisition/runtime/model-load actions
- blockers and unknowns
- a verification step defined **before** mutation
- a deterministic plan ID

`plan` performs no artifact download, runtime start, model load, or inference.

Example shape:

```text
LLMRig Autopilot Plan
=====================
Plan:    plan-...
Machine: Apple M4 Max · 48 GiB
Model:   mlx-community/Qwen3.5-27B-4bit

Runtime       Format          Quant         Local         Executable
------------  --------------  ------------  ------------  ------------
omlx          MLX             4-bit         not_available not_executable
mlx-lm        MLX             4-bit         not_available not_executable

Setup path: omlx · MLX

Planned actions
1. Acquire the selected artifact [changes local state]
2. Load/register it with the runtime [changes local state]
3. Verify with the deterministic workload [verification]

No action has been taken.
```

If multiple setup paths remain valid, LLMRig stays inconclusive rather than inventing a winner.

### 2. Apply — explicit intent

```bash
llmrig apply mlx-community/Qwen3.5-27B-4bit \
  --plan-id plan-...
```

LLMRig recomputes the plan immediately before apply. If current evidence no longer produces the approved plan ID, apply fails closed with **plan drift detected**.

Mutating actions require either interactive approval or an explicit non-interactive approval flag:

```bash
llmrig apply MODEL --plan-id plan-... --yes
```

Exact Hugging Face acquisition is pinned to the repository revision returned by the Hub. Generic discovery metadata never becomes installation trust.

Approved Hugging Face acquisition uses the optional `huggingface_hub` package. For a pipx installation:

```bash
pipx inject llmrig huggingface_hub
```

### 3. Verify — measure current reality

Apply includes measured verification when the selected runtime is executable. You can also re-observe and re-verify a prior receipt:

```bash
llmrig verify
llmrig verify receipt-...
```

`verify` does not trust an old success record. It recomputes current evidence, requires the exact runtime/artifact candidate to remain uniquely evidenced, refuses any mutation, and then runs the deterministic measurement workload.

Unavailable metrics remain unknown. LLMRig never synthesizes missing throughput from incomparable timing data.

### 4. Run — the convenience workflow

```bash
llmrig run MODEL
```

`run` is the product shortcut for plan → explicit approval → apply → verify. It does **not** bypass the approval boundary. For automation, intent must still be explicit:

```bash
llmrig run MODEL --yes
```

## Action receipts

Every applied workflow produces a privacy-safe receipt containing:

```text
plan ID
receipt ID
model / artifact / runtime identity
actions attempted
status and public evidence
verification measurements
public endpoint when applicable
timestamps
```

Private filesystem locators, API keys, session cookies, and secrets are not serialized into receipts.

## RigGraph — local evidence that can learn from reality

Successful measured verification is persisted locally as graph-shaped evidence across:

```text
machine
  × model
  × artifact
  × quantization
  × runtime
  × context
  × measurement
```

Prediction and measurement are separate facts. Calibration deltas are computed only when the same metric exists on both sides.

```text
prediction: generation_tps = unknown
measurement: generation_tps = 31.2
calibration: generation_tps_delta = unknown
```

No anonymous/community upload occurs by default in v0.9.

## Runtime intelligence

```bash
llmrig runtimes
llmrig runtimes --json
```

The adapter registry currently covers:

| Runtime | Readiness detection | LLMRig execution / measurement | Primary artifact evidence |
|---|---|---|---|
| oMLX | Yes | Yes, for provenance-backed local models | MLX |
| Ollama | Yes | Yes | Ollama-managed artifacts |
| MLX-LM | Yes | Yes, for explicit/evidenced local artifacts | MLX |
| llama.cpp | Yes | Yes, for explicit/evidenced local artifacts | GGUF |

Runtime probing remains read-only. Mutation support is adapter-specific and unsupported actions remain explicit blockers rather than being silently emulated.

## Solve — inspect evidence without Autopilot actions

```bash
llmrig solve MODEL
llmrig solve MODEL --json
llmrig solve MODEL --context 32768
```

`solve` constructs orthogonal evidence dimensions for each candidate:

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

A candidate can therefore be compatible but not local, local but not executable, executable but not measured, or measured without being recommendable.

### Explicit local artifacts

LLMRig does not scan arbitrary directories for native model files. Supply a locator explicitly when you want a native local artifact considered:

```bash
llmrig solve MODEL --local-artifact llama.cpp=/path/to/model.gguf
llmrig solve MODEL --local-artifact mlx-lm=/path/to/model-directory
```

The private locator stays at the execution seam and is excluded from public solve output.

## Hugging Face-native resolution

Exact `owner/repository` identifiers are resolved through read-only Hub metadata. LLMRig may read the repository's small `config.json` when needed for structured format, quantization, or context evidence; it does not download weights during resolution.

Evidence rules include:

- `.gguf` establishes GGUF packaging; filename quantization is accepted only when exactly one recognized token is present
- generic `.safetensors` does not itself establish MLX/oMLX compatibility
- an `mlx` tag/path hint alone is not proof of MLX packaging
- MLX packaging requires stronger structured evidence such as explicit `library_name=mlx` or an MLX hint backed by the MLX-LM quantization contract
- conflicting context, quantization, provenance, or incomplete shard groupings fail closed

## oMLX

LLMRig can associate an API-visible oMLX model with an exact Hugging Face repository only when provenance is defensible: exact runtime source metadata, or an exact completed-download record whose mapping is unique and unambiguous.

Authenticated local endpoints are supported through:

```bash
export OMLX_API_KEY="..."
llmrig runtimes
```

The API key, admin-session cookie, filesystem model path, and private execution locator are never serialized into public results.

## Measurement and comparison

Existing measured-analysis commands remain available:

| Command | Purpose |
|---|---|
| `solve --verify` | verify a comparable solve candidate set |
| `race` | compare local configurations |
| `choose` | explain one measured objective |
| `optimize` | expose an unranked noise-aware Pareto frontier |
| `bench` | full Ollama benchmark |

Performance measurement does not establish model quality. Results inside the configured 5% race threshold remain inconclusive.

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

The SDK prints nothing and never exits the process. `SolveResult` and `SolveCandidate` are deliberate public contracts; `_llmrig` remains private implementation.

## Installation

[`pipx`](https://pipx.pypa.io/) is recommended:

```bash
pipx install llmrig
llmrig --version
```

Upgrade:

```bash
pipx upgrade llmrig
```

Or use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install llmrig
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install llmrig
```

LLMRig supports Python 3.9+ on macOS, Linux, and Windows and has no mandatory third-party Python runtime dependency.

## Architecture

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

The north star is **Detect → Decide → Configure → Run → Verify**, with explicit intent at every mutation boundary.

See [`ROADMAP_V1.md`](https://github.com/sunilteja93/llmrig/blob/main/ROADMAP_V1.md).

## Project docs

- [`CHANGELOG.md`](https://github.com/sunilteja93/llmrig/blob/main/CHANGELOG.md) — canonical release history
- [`ROADMAP_V1.md`](https://github.com/sunilteja93/llmrig/blob/main/ROADMAP_V1.md) — product direction
- [`RELEASING.md`](https://github.com/sunilteja93/llmrig/blob/main/RELEASING.md) — GitHub Release and PyPI process
- [`SECURITY.md`](https://github.com/sunilteja93/llmrig/blob/main/SECURITY.md) — security policy and trust boundaries
- [`CONTRIBUTING.md`](https://github.com/sunilteja93/llmrig/blob/main/CONTRIBUTING.md) — contribution guide

## License

MIT
