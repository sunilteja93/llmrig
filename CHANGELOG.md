# Changelog

All notable changes to LLMRig will be documented here.

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
