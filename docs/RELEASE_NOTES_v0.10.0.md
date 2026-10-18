## LLMRig v0.10.0: Verified Autopilot and Rig Reports

### End-to-end execution validated
- Completed local Autopilot verification with **oMLX 0.6.4** and `mlx-community/Qwen3-0.6B-4bit` on an Apple M4 Max (48 GiB): 2 measured race-v2 runs, 0.29 s reported latency; throughput not recorded.
- Completed local verification with **Ollama 0.33.3** and installed `qwen3.8:27b-mlx`: 2 measured race-v2 runs, 49.93 generation tok/s, 2.5861 s reported latency. These different models are not a controlled runtime comparison.
- Added reproducible [field-run notes](https://github.com/sunilteja93/llmrig/blob/main/docs/VERIFIED_DEMOS.md) and an illustrative evidence card.

### New workflows
- Model-specific read-only `llmrig doctor MODEL`, actionable guidance, offline JSON Plan Explorer.
- Privacy-filtered `llmrig report` output in terminal, Markdown, JSON, plus an offline local Rig Report viewer.
- Safer Hugging Face artifact acquisition, provenance, and informative runtime failure receipts.
- Suppress potentially misleading Ollama prompt-throughput estimates from sub-millisecond phases.
- Exact, sanitized Ollama artifact tag in shared reports; catalog format clearly distinguished from verified runtime metadata.
- Improved clean installation documentation and independent wheel/`pipx` smoke coverage.

### Boundaries
- Mutating actions remain approval-gated; read-only planning never downloads weights.
- Verification numbers are local, self-reported observations, not independent benchmarks or hardware attestations.
- Ongoing Ollama catalog metadata and oMLX HTTP 401 classification limitations should not be mistaken for fixed bugs.
- Python 3.9+ and the existing optional local runtime requirements are unchanged.

See [README](https://github.com/sunilteja93/llmrig/blob/main/README.md) and [Installation](https://github.com/sunilteja93/llmrig/blob/main/docs/INSTALL.md).
