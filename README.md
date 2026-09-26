<h1 align="center">LLMRig</h1>

<p align="center"><strong>Stop guessing how to run local models.</strong></p>

<p align="center">LLMRig is an evidence-driven Autopilot for local AI. Give it a model and a machine: it inspects the available evidence, builds a deterministic local execution plan, and tells you what is known, blocked, or unknown before changing anything. Approve a supported plan and LLMRig can apply the setup, verify the result, and record a privacy-safe receipt.</p>

<p align="center">
  <a href="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <a href="https://pypi.org/project/llmrig/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/llmrig"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sunilteja93/llmrig/main/assets/llmrig-terminal.svg" alt="LLMRig runtime intelligence and evidence-driven local AI flow" width="100%" />
</p>

## See it in 30 seconds

```bash
pipx install llmrig
llmrig plan mlx-community/Qwen3.5-27B-4bit
```

Representative output:

```text
LLMRig Autopilot Plan
=====================
Machine: Apple M4 Max · 48 GiB
Model:   mlx-community/Qwen3.5-27B-4bit

Runtime       Format   Quant    Local      Executable
------------  -------  -------  ---------  ------------
omlx          MLX      4-bit    unknown    not_executable
mlx-lm        MLX      4-bit    unknown    not_executable

Setup path: omlx · MLX
Reason: exactly one compatible candidate has a complete, supported
Autopilot setup path. This is a setup selection, not a performance ranking.

Planned actions
1. Acquire the selected artifact
2. Prepare the selected runtime
3. Load/register the artifact
4. Verify with LLMRig's deterministic workload

No action has been taken.
```

`plan` is read-only. If the evidence is insufficient, LLMRig stays inconclusive instead of inventing a winner.

Ready to continue? Run the workflow and approve any mutation explicitly:

```bash
llmrig run mlx-community/Qwen3.5-27B-4bit
```

## Why LLMRig

Running models locally still involves too much guesswork:

- Large weights can be downloaded before you discover the intended setup is not viable on your machine.
- Runtime, format, context, and quantization advice is fragmented across model cards, forums, and machine-specific anecdotes.
- oMLX, Ollama, MLX-LM, and llama.cpp expose different artifacts and capabilities, so a model name alone is not an execution plan.

**Ollama runs models. LLMRig sits above runtimes and determines what the current evidence supports for this model on this machine** — then, with approval, it can apply the supported setup and verify what actually happened.

And when the evidence is not there, LLMRig says `unknown` instead of converting absence into certainty:

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

## Quickstart

```bash
pipx install llmrig          # or: python -m pip install llmrig
llmrig solve MODEL           # inspect evidence; change nothing
llmrig plan MODEL            # build a deterministic read-only plan
llmrig run MODEL             # approve → apply → verify
```

`solve` and `plan` are read-only. Downloads, runtime starts, model loads, and other mutations require explicit approval. For non-interactive automation, use `--yes` only when that intent is deliberate.

## Measured, not guessed

Planning evidence and measured performance are separate facts.

Before verification, unavailable performance stays `unknown`. When a supported run is actually verified, LLMRig records the measured result in a privacy-safe receipt containing the model, artifact, runtime, configuration, workload version, and observed metrics.

```text
Receipt: receipt-...
Status:  completed
Verification:
  method_version: race-v2
  generation_tps: <measured value>
  prompt_eval_tps: <measured value>
  measured_runs: 2
```

Public, privacy-safe examples from the v0.9 Apple Silicon smoke are available in the [Benchmark Passports dataset](https://huggingface.co/datasets/sunilvadlamani/llmrig-benchmark-passports). Performance measurements are evidence about that exact configuration and workload; they are not claims about model quality.

## The Autopilot flow

Four commands, one principle: **explicit intent at every mutation boundary.**

### 1. Plan — read only

```bash
llmrig plan mlx-community/Qwen3.5-27B-4bit
```

Produces a deterministic plan ID covering machine identity, viable runtime/artifact candidates, required actions, blockers, and a verification step defined *before* anything changes. Downloads nothing, starts nothing.

If several setup paths stay valid, LLMRig stays inconclusive rather than inventing a winner.

### 2. Apply — explicit intent

```bash
llmrig apply mlx-community/Qwen3.5-27B-4bit --plan-id plan-... --yes
```

LLMRig recomputes the plan immediately before applying. If the evidence no longer reproduces the approved plan ID, apply fails closed with **plan drift detected**. Hugging Face downloads are pinned to the exact repository revision resolved by the Hub.

### 3. Verify — measure current reality

```bash
llmrig verify
llmrig verify receipt-...
```

Recomputes evidence from scratch, refuses any mutation, and runs the deterministic measurement workload. Old success records are never trusted; missing metrics stay `unknown` — never synthesized.

### 4. Run — the convenience workflow

```bash
llmrig run MODEL --yes
```

Plan → approval → apply → verify in one shortcut. It does **not** bypass the approval boundary.

## Action receipts

Every applied workflow emits a privacy-safe receipt: plan ID, receipt ID, model/artifact/runtime identity, actions attempted, status, verification measurements, and timestamps. Filesystem paths, API keys, cookies, and secrets are never serialized.

## RigGraph — evidence that learns from reality

Verified measurements persist locally as graph-shaped evidence across machine × model × artifact × quantization × runtime × context × measurement. Predictions and measurements are stored as separate facts; calibration deltas are computed only when the same metric exists on both sides. No community upload happens by default.

## Runtime intelligence

```bash
llmrig runtimes
llmrig runtimes --json
```

| Runtime | Readiness detection | LLMRig execution / measurement | Primary artifact evidence |
|---|---|---|---|
| oMLX | Yes | Yes, for provenance-backed local models | MLX |
| Ollama | Yes | Yes | Ollama-managed artifacts |
| MLX-LM | Yes | Yes, for explicit/evidenced local artifacts | MLX |
| llama.cpp | Yes | Yes, for explicit/evidenced local artifacts | GGUF |

Runtime probing is read-only. Unsupported actions surface as explicit blockers, never silent emulation.

LLMRig resolves exact `owner/repository` identifiers through read-only Hugging Face Hub metadata (no weight downloads during resolution), with strict evidence rules — e.g. an `mlx` tag alone is not proof of MLX packaging, and conflicting context, quantization, or provenance evidence fails closed.

## Hugging Face

- [LLMRig Autopilot Space](https://huggingface.co/spaces/sunilvadlamani/llmrig-autopilot) — evidence-first walkthrough of the Autopilot flow
- [Benchmark Passports dataset](https://huggingface.co/datasets/sunilvadlamani/llmrig-benchmark-passports) — privacy-safe RigGraph measurement samples
- [LLMRig collection](https://huggingface.co/collections/sunilvadlamani/llmrig-autopilot-for-local-ai-6ab56f4bc732154111d04d9b) — Space, dataset, and referenced model grouped together
- [Launch discussion](https://huggingface.co/spaces/sunilvadlamani/llmrig-autopilot/discussions/1) — reproducible launch notes and evidence

The Space is intentionally a static explainer: it does **not** inspect a visitor's machine. Run LLMRig locally for real discovery, planning, execution, and verification.

## Measurement and comparison

| Command | Purpose |
|---|---|
| `solve --verify` | verify a comparable solve candidate set |
| `race` | compare local configurations |
| `choose` | explain one measured objective |
| `optimize` | expose an unranked noise-aware Pareto frontier |
| `bench` | full Ollama benchmark |

Measurement never establishes model quality, and results inside the configured 5% race threshold stay inconclusive.

## Minimal Python SDK

```python
import llmrig

result = llmrig.solve("mlx-community/Qwen3.5-27B-4bit")
print(result.plan.recommendation_status)
```

```python
llmrig.solve(
    model,
    *,
    context=None,
    local_artifacts=(),
    verify=False,
) -> llmrig.SolveResult
```

The SDK prints nothing and never exits the process. `SolveResult` and `SolveCandidate` are the public contracts; `_llmrig` stays private.

## Installation

[`pipx`](https://pipx.pypa.io/) is recommended:

```bash
pipx install llmrig
llmrig --version
```

Upgrade with `pipx upgrade llmrig`, or use a virtual environment:

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

Python 3.9+ on macOS, Linux, and Windows. Hugging Face Hub support is included for exact, revision-pinned artifact acquisition.

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

The north star is **Detect → Decide → Configure → Run → Verify**, with explicit intent at every mutation boundary. See [`ROADMAP_V1.md`](https://github.com/sunilteja93/llmrig/blob/main/ROADMAP_V1.md).

## Project docs

- [`CHANGELOG.md`](https://github.com/sunilteja93/llmrig/blob/main/CHANGELOG.md) — canonical release history
- [`ROADMAP_V1.md`](https://github.com/sunilteja93/llmrig/blob/main/ROADMAP_V1.md) — product direction
- [`RELEASING.md`](https://github.com/sunilteja93/llmrig/blob/main/RELEASING.md) — GitHub Release and PyPI process
- [`SECURITY.md`](https://github.com/sunilteja93/llmrig/blob/main/SECURITY.md) — security policy and trust boundaries
- [`CONTRIBUTING.md`](https://github.com/sunilteja93/llmrig/blob/main/CONTRIBUTING.md) — contribution guide

## License

MIT