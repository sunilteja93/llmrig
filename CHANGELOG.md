# Changelog

All notable changes to LLMRig will be documented here.

## 0.6.0 - 2026-08-27

### Added

- Add native cross-runtime execution and benchmarking for explicit already-local artifacts across Ollama, llama.cpp, and MLX-LM.
- Add repeatable `--local-artifact RUNTIME=PATH` support for race and analysis commands.
- Add `race-v2` cross-runtime measurement with generation throughput, prompt-evaluation throughput, and normalized inference latency.
- Add benchmark passport support for successfully measured native race competitors.
- Add `llmrig choose <model>` for explainable objective-specific generation, prompt, latency, and balanced decisions.
- Add `llmrig optimize <model>` for noise-aware measured-performance Pareto analysis.

### Decision intelligence

- Generation, prompt, and latency decisions reuse the existing metric-specific race winners.
- A balanced recommendation exists only when one unique measured-performance Pareto leader remains; multiple Pareto frontier members remain explicitly unresolved.
- Decisions use no composite or universal score and do not infer model quality from speed.

### Reliability and privacy

- Preserve the 5% race noise threshold for winner and dominance decisions.
- Omit Pareto dimensions globally when values are missing, invalid, or non-finite; unknown values are never coerced to zero.
- Sanitize private filesystem-shaped artifact IDs at the analysis boundary so local paths are not exposed in Decision or Pareto results.
- Trust llama.cpp stderr diagnostics, rather than generated stdout, for benchmark metrics.
- Keep native execution targets private and non-serializable.
- Do not automatically download native models or install llama.cpp or MLX-LM.

### Compatibility

- Support Python 3.9+ on macOS, Linux, and Windows with no third-party Python runtime dependencies.
- Ollama, llama.cpp, and MLX-LM have LLMRig execution and benchmark support where compatible artifacts and runtimes are already available locally; LLMRig does not install or fully manage llama.cpp or MLX-LM, and compatibility is specific to each model, artifact, and runtime combination.

## 0.5.1 - 2026-08-27

### Fixed / Documentation

- Recommend `pipx` as the primary CLI installation method.
- Document a virtual-environment installation fallback.
- Avoid PEP 668 externally-managed-environment failures on system-managed Python installations.

## 0.5.0 - 2026-08-27

### Added

- Add `llmrig can <model>` with human and JSON output, three-state compatibility and exit semantics, categorical confidence, and evidence provenance.
- Resolve generic Hugging Face repositories through read-only metadata and identify evidenced GGUF, MLX, Safetensors, and curated Ollama artifacts without downloading weights.
- Detect runtime capabilities for Ollama, llama.cpp, and MLX-LM while keeping runtime-native capability separate from LLMRig execution and benchmark support.
- Add `llmrig race <model>` for measured comparison of unique local executable configurations, with deterministic competitor identity, alias de-duplication, and metric-specific results.
- Add versioned benchmark passports with genuine per-run samples, passport IDs, deterministic configuration fingerprints, comparison classifications, bench/race export, and offline `llmrig passport verify` validation.

### Reliability and safety

- Keep missing compatibility, artifact, runtime, and measurement information explicitly unknown rather than inventing metadata or performance.
- Prevent surprise artifact or runtime downloads during `can`, `race`, and passport verification.
- Require at least two unique executable configurations before a race runs; incomplete races produce no official winners or exported competitor passports.
- Export privacy-safe benchmark metadata. Passport hashes provide identity and integrity checks, not signatures, independent attestation, or benchmark certification.
- Keep passport verification read-only and offline: it does not execute inference or prove that a claimed measurement is true.

### Compatibility

- Supports Python 3.9+ on macOS, Linux, and Windows with no third-party Python runtime dependencies.
- Ollama is the implemented setup, execution, and benchmark adapter. llama.cpp and MLX-LM support is capability detection only in this release.

## 0.4.1 - 2026-08-24

### Fixed

- Use an absolute URL for the terminal graphic so it renders correctly on PyPI.
- Refresh README installation and runtime-dependency wording.
- Update the repository layout to reflect packaging, assets, and publishing workflows.

## 0.4.0 - 2026-08-23

### Added

- Installable Python command-line interface with the `llmrig` command.
- Python packaging through `pyproject.toml`.
- Cross-platform CI validation of the installed CLI on Linux, macOS, and Windows.
- CI coverage for Python 3.9 and Python 3.13.
- Self-hosted terminal-style README hero.

### Usage

LLMRig can now be installed from a source checkout and used as a system command:

```bash
python3 -m pip install .
llmrig doctor
llmrig recommend
llmrig models --fit
llmrig setup
llmrig bench
```

The original source invocation remains supported:

```bash
python3 llmrig.py
```

### Compatibility

- Python 3.9+
- macOS
- Linux
- Windows
- No third-party Python runtime dependencies
