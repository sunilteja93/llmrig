<h1 align="center">LLMRig</h1>

<p align="center">
  <strong>Know what your rig can run.</strong>
</p>

<p align="center">
  Hardware-aware local LLM selection, setup, and benchmarking.
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

LLMRig answers one practical question:

> **Which local LLM can this machine actually run well?**

It detects the hardware you actually have, estimates a conservative model budget, recommends a practical model and context configuration, can set it up through Ollama, and benchmarks the result on the machine itself.

### Install

Install LLMRig from PyPI:

```bash
python3 -m pip install llmrig
```

Then use LLMRig from anywhere:

```bash
llmrig doctor
llmrig recommend
llmrig models --fit
```

Or run directly from source without installing:

```bash
python3 llmrig.py
```

**LLMRig has no third-party Python runtime dependencies.** The CLI uses only the Python standard library.

```text
detect hardware → estimate fit → recommend → setup → benchmark → compare
```

LLMRig is currently **Qwen-first**. The architecture is intended to expand to additional model families, runtimes, GPUs, and platforms without changing the core workflow.

## What LLMRig does

- Runs on **macOS, Windows, and Linux**
- Detects CPU/chip, architecture, RAM, free disk space, and GPU/VRAM when available
- Respects `OLLAMA_MODELS` when checking model-storage capacity
- Checks whether Ollama and its local API are available
- Gives a conservative local-LLM readiness assessment
- Recommends a model for `balanced`, `speed`, or `quality` priorities
- Separates official Qwen models from community reduced-refusal derivatives
- Discovers newly published official Qwen repositories from Hugging Face at runtime
- Auto-pulls only curated, verified Ollama identifiers
- Benchmarks generation speed, prompt-evaluation speed, accelerator residency, and lightweight correctness checks
- Isolates benchmark runs by unloading resident Ollama models before and after each test
- Saves shareable benchmark reports as JSON and Markdown
- Uses only the Python standard library

## Current scope

LLMRig currently has a **Qwen-first curated catalog** and uses Ollama as its local inference backend. Live discovery tracks likely Qwen LLM/multimodal inference repositories, while automatic installation is limited to model identifiers that have been manually verified.

The project name is intentionally broader than Qwen because the long-term direction is to support additional model families and runtimes without changing the user experience:

```text
hardware → discover → recommend → setup → benchmark → compare
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
  --runs 2
```

All installed supported Qwen models:

```bash
llmrig bench \
  --all-installed \
  --context 32768 \
  --runs 2
```

LLMRig deduplicates installed aliases that resolve to the same Ollama model ID.

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

The roadmap is intentionally community-driven. Likely directions include support for more model families, multiple local inference runtimes, richer GPU detection, benchmark leaderboards, and standardized community hardware reports.

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
