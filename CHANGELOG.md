# Changelog

All notable changes to LLMRig will be documented here.

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
