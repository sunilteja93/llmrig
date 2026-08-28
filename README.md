<h1 align="center">LLMRig</h1>

<p align="center">
  <strong>Know what your rig can run.</strong>
</p>

<p align="center">
  The open compatibility and performance intelligence layer for local AI.
</p>

<p align="center">
  <a href="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/sunilteja93/llmrig/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg">
  <a href="https://pypi.org/project/llmrig/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/llmrig"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/sunilteja93/llmrig/main/assets/llmrig-terminal.svg" alt="LLMRig local model fit flow" width="100%" />
</p>

LLMRig helps answer one practical question:

> **Which local LLM can this machine actually run well?**

It inspects hardware, resolves logical models and runnable artifacts, identifies runtime paths, estimates compatibility, measures local execution, compares configurations, and preserves reproducible benchmark evidence. Curated setup and measured execution currently use Ollama.

### Install

Install LLMRig from PyPI:

```bash
python3 -m pip install llmrig
```

Then use LLMRig from anywhere:

```bash
llmrig doctor
llmrig can qwen3.8:27b-mlx
llmrig recommend
llmrig models --fit
```

Or run directly from source without installing:

```bash
python3 llmrig.py
```

**LLMRig has no third-party Python runtime dependencies.** The CLI uses only the Python standard library.

```text
detect hardware
→ resolve model + artifact
→ identify runtime paths
→ estimate compatibility
→ execute locally
→ measure
→ compare
→ preserve evidence
```

LLMRig is currently **Qwen-first**. The architecture is intended to expand to additional model families, runtimes, GPUs, and platforms without changing the core workflow.

## What LLMRig does

- Detects privacy-safe hardware facts across **macOS, Windows, and Linux**.
- Separates logical models from GGUF, MLX, Safetensors, and curated Ollama artifacts, with generic read-only Hugging Face resolution.
- Analyzes compatibility with explicit confidence, evidence provenance, practical context, and unknown handling.
- Detects Ollama, llama.cpp, and MLX-LM runtime capabilities without implying an execution adapter exists.
- Recommends and sets up only curated, verified model identifiers through Ollama.
- Measures local generation, prompt evaluation, latency, residency, and lightweight correctness through the Ollama benchmark adapter.
- Races at least two unique executable configurations with metric-specific, non-composite results.
- Exports privacy-safe benchmark passports containing raw per-run evidence and deterministic identities.
- Verifies passport structure, integrity, aggregates, and privacy entirely offline without inference.

## Current scope

LLMRig remains **Qwen-first**. Its curated catalog supports practical recommendations and setup, while generic Hugging Face resolution inspects repository metadata without downloading model weights. Automatic installation remains limited to manually verified curated identifiers.

Ollama is currently LLMRig's implemented setup, execution, and benchmark backend. LLMRig can detect and reason about llama.cpp and MLX-LM capabilities, but execution adapters for those runtimes are not yet implemented.

The project name is intentionally broader than Qwen because the long-term direction is to support additional model families and runtimes without changing the user experience:

```text
hardware → model + artifact → runtime paths → compatibility → measurement → evidence
```

If you want to add support for another model family, runtime, GPU vendor, or operating system, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Requirements

- Python **3.9+**
- macOS, Windows, or Linux
- Ollama for automatic model setup and benchmarking
- Internet access for live discovery and model downloads
- No third-party Python runtime dependencies

## Quick start

Run the interactive wizard:

```bash
llmrig
```

The wizard inspects the machine, recommends a supported model, pulls it if necessary, benchmarks it, and prints the local chat/API details.

## Commands

### Inspect the machine

```bash
llmrig doctor
```

Machine-readable output:

```bash
llmrig doctor --json
```

### Check whether a curated model can run

```bash
llmrig can qwen3.8:27b-mlx
```

Machine-readable compatibility, confidence, configuration, and evidence:

```bash
llmrig can qwen3.8:27b-mlx --json
```

Curated identifiers receive the existing practical compatibility analysis. An
`owner/repository` Hugging Face ID is resolved through read-only metadata and file
listings, without downloading weights. LLMRig recognizes evidenced GGUF, MLX, and
Safetensors artifacts. For generic GGUF and MLX artifacts it reports matching
llama.cpp or MLX-LM capability candidates and whether those runtimes are locally
available; Safetensors alone does not imply a runnable path. Runtime installation,
current availability, format support, and total machine compatibility remain separate.
Even a detected candidate is not a claim that inference was tested. Memory results
are conservative planning estimates, and unmeasured runtime overhead, practical
context, and performance remain explicitly unknown.

`llmrig can` also behaves as a three-state Unix predicate in both human and JSON
modes: exit `0` means the model can run, exit `1` means it cannot run, and exit
`2` means compatibility is unknown or the identifier cannot be analyzed.

### Race locally executable configurations

```bash
llmrig race qwen3.8:27b-mlx
llmrig race qwen3.8:27b-mlx --json
```

`race` measures only configurations that are already local, currently available,
and backed by an LLMRig execution/benchmark adapter. It never installs runtimes or
downloads artifacts. At least two executable configurations are required; otherwise
the command reports the eligible and blocked alternatives without running a benchmark.
Runtime race execution currently supports Ollama. llama.cpp and MLX-LM are detected
as capability providers but do not yet have LLMRig execution adapters.

Exit `0` means at least two competitors were measured successfully. Exit `1` means
an attempted execution failed and invalidated the comparison. Exit `2` means the race
is unavailable or the model could not be resolved. Winners are reported separately
for measured generation throughput, prompt-evaluation throughput, and latency; there
is no composite score or model-quality claim. Results within 5% are treated as
inconclusive, and at least two timed runs per competitor are required for a winner.

### List models

Show curated local-ready models plus the newest live Qwen LLM/multimodal candidates:

```bash
llmrig models --fit
```

Force live refresh:

```bash
llmrig models --refresh --fit
```

Show the full Qwen Hugging Face organization catalog, including non-LLM artifacts:

```bash
llmrig models --all --fit
```

Use only the built-in curated snapshot:

```bash
llmrig models --offline --fit
```

### Get a recommendation

```bash
llmrig recommend
```

Official models only:

```bash
llmrig recommend --category official
```

Community reduced-refusal models only:

```bash
llmrig recommend --category unrestricted
```

Prioritize quality:

```bash
llmrig recommend \
  --category official \
  --preference quality
```

For CLI convenience, `unrestricted`, `uncensored`, and `reduced-refusal` map to the community reduced-refusal category. `restricted` is accepted as an alias for the official category. LLMRig uses **official** and **reduced-refusal** in its output because those labels are more precise.

### Set up a model

```bash
llmrig setup --category official
```

Or choose an exact curated model:

```bash
llmrig setup \
  --model qwen3.8:27b-mlx \
  --context 32768
```

If a known alias of the selected curated build is already installed, LLMRig reuses it when possible.

### Benchmark installed models

One model:

```bash
llmrig bench \
  --model qwen3.8:27b-mlx \
  --context 32768 \
  --runs 2 \
  --passport benchmark.passport.json
```

All installed supported Qwen models:

```bash
llmrig bench \
  --all-installed \
  --context 32768 \
  --runs 2
```

LLMRig deduplicates installed aliases that resolve to the same Ollama model ID.

### Benchmark passports

A benchmark passport is a versioned, privacy-safe JSON record of one measured
execution configuration. It records the exact model/build, runtime, safe hardware
summary, applied workload, individual timed samples, reproducible aggregates, and
evidence provenance. Use `bench --passport FILE` for a single model, or
`race --passport-dir DIR` to export each successfully measured race competitor.

Validate a passport locally without network access or inference:

```bash
llmrig passport verify benchmark.passport.json
```

Verification checks the schema, SHA-256 identity and configuration fingerprints,
raw-sample aggregates, impossible states, and privacy constraints. It establishes
only that the document is internally consistent according to LLMRig's schema. The
hashes are identifiers and integrity checks, not signatures, independent proof, or
benchmark certification.

`passport_id` is the SHA-256 hash of canonical passport content with `passport_id`
itself excluded. It identifies the exact record, including its timestamp and
measurements. `configuration_fingerprint` hashes the logical model, exact
artifact/build, artifact digest, format, quantization, runtime and version, execution
adapter, privacy-safe hardware facts, exact workload and generation settings, and
benchmark method version. It excludes the passport ID, timestamp, measurements,
aggregates, run-only warnings, and output path. The LLMRig tool version is record
metadata; the benchmark method version is the compatibility boundary and must change
when the procedure changes.

Two passports are `exact` when their configuration fingerprints match, even though
their passport IDs and measured results may differ. `comparable_with_warnings` means
the logical model and workload match but artifact, format, quantization, runtime,
runtime version, or hardware differs. `not_comparable` means the logical model or
workload—including context—differs. These classifications do not rank results.

Race passports are exported only when the overall race completes successfully. If
any intended competitor fails, the race remains failed and `--passport-dir` writes
no standalone competitor passports that could hide the incomplete comparison.

Passport aggregates are derived only from the recorded timed-run samples. Throughput
means are rounded to two decimal places, wall-latency means to four decimal places,
using Python's deterministic `round` behavior; generated tokens are the sum of the
runtime-reported per-run counts. Warmups are recorded as policy metadata and never
enter samples or aggregates.

### Run project checks

Offline:

```bash
llmrig check
```

Include live Hugging Face discovery:

```bash
llmrig check --online
```

## Model discovery and safety

LLMRig deliberately separates **discovery** from **automatic installation**.

The curated catalog contains local model identifiers that LLMRig may pull automatically. Live discovery queries the official Qwen organization on Hugging Face so new releases can appear without requiring an immediate LLMRig release.

A newly discovered repository is **discovery only**. LLMRig does not infer package size, hardware fit, or Ollama compatibility from a repository name. A model becomes eligible for one-command setup only after its identifier, package size, context capability, backend support, and provenance are verified and added to the curated catalog.

Community reduced-refusal discovery is best-effort and is not an authoritative registry. Third-party models should be reviewed before use.

## Official vs reduced-refusal models

**Official** entries are upstream Qwen models distributed through the selected local backend.

**Community reduced-refusal** entries are third-party derivatives that modify model behavior to reduce refusals. Their authors may use terms such as `uncensored`, `unrestricted`, or `abliterated`.

Reduced refusal behavior does not imply better reasoning, accuracy, safety, or reliability. Review model provenance, licensing, and benchmark results before relying on a derivative for important work.

## Hardware fit and context

LLMRig deliberately leaves headroom for the operating system, inference runtime, KV cache, other applications, and GPU/runtime buffers.

For Apple Silicon, CPU and GPU share unified memory. For discrete GPUs, LLMRig favors configurations likely to stay mostly or fully on the accelerator when VRAM can be detected.

The model's advertised maximum context is not automatically used as the default. Longer context allocations consume more memory, so LLMRig starts conservatively and increases context only when there is comfortable headroom.

Hardware-fit results are estimates. **The benchmark on the user's actual machine is the final check.**

## Benchmarking

LLMRig unloads currently resident Ollama models before a benchmark and unloads the target model afterward. This reduces cross-model memory contamination and makes comparisons more reproducible.

Each benchmark records:

- generation tokens/second
- prompt-evaluation tokens/second
- load and total duration
- Ollama-reported context length
- accelerator residency when available
- RAM/swap snapshots when available
- three lightweight deterministic correctness smoke tests

Reports are written under `benchmarks/` as JSON and Markdown. That directory is ignored by Git by default so local benchmark data is not published accidentally. Review any benchmark before sharing it.

These are local performance/configuration checks, **not academic model-quality benchmarks**. Memory values are snapshots rather than peak-memory measurements.

## Contributing

LLMRig is open source and contributions are welcome.

Good first contribution areas include:

- additional model families
- new Ollama/local-backend model mappings
- AMD, Intel, and NVIDIA GPU detection improvements
- Windows and Linux hardware testing
- reproducible benchmark improvements
- new hardware profiles and recommendation rules
- documentation and usability improvements
- tests for new operating systems and model variants

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Bug reports, feature ideas, model-support requests, and benchmark improvements are all welcome through GitHub issues.

## Development

Run the full local validation set before opening a pull request:

```bash
python3 -m py_compile llmrig.py
python3 -m unittest discover -s tests -v
python3 llmrig.py check
python3 llmrig.py models --offline --fit
```

With internet access:

```bash
python3 llmrig.py check --online
```

GitHub Actions also runs compile, unit-test, and sanity-check jobs on Linux, macOS, and Windows.

## Repository layout

```text
llmrig/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── publish-to-pypi.yml
│   │   └── refresh-profile-on-release.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── assets/
│   └── llmrig-terminal.svg
├── tests/
│   ├── __init__.py
│   └── test_llmrig.py
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
├── SECURITY.md
└── llmrig.py
```

## Roadmap

The roadmap is intentionally community-driven. Credible next directions include additional model families; llama.cpp, MLX-LM, and other execution adapters; richer GPU and runtime support; explainable comparison and decision intelligence; Pareto-style configuration optimization; and standardized privacy-safe benchmark sharing.

The rule for new functionality is simple: **be useful, be reproducible, and do not turn unverified discovery metadata into an automatic install decision.**

## License

LLMRig is released under the [MIT License](LICENSE).

## Primary references

- Qwen official Hugging Face organization: `https://huggingface.co/Qwen`
- Qwen3.8 official repository: `https://github.com/QwenLM/Qwen3.8`
- Hugging Face Hub API: `https://huggingface.co/docs/huggingface_hub/package_reference/hf_api`
- Ollama documentation: `https://docs.ollama.com/`
- Ollama generate API: `https://docs.ollama.com/api/generate`
- Ollama running-model API: `https://docs.ollama.com/api/ps`
- Qwen3.8 Ollama tags: `https://ollama.com/library/qwen3.8/tags`
