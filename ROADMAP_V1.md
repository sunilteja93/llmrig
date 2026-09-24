# LLMRig v1 Direction

LLMRig is evolving from a compatibility CLI into an evidence-driven Autopilot for local AI.

## North star

Given a model and a machine, LLMRig should determine viable execution paths across model artifact, quantization, runtime, context, and measured performance; explain its evidence; and, only with explicit permission, carry out the selected setup.

## Product pillars

1. **Runtime adapters** — Ollama, MLX-LM, oMLX, llama.cpp, and future runtimes expose a common capability, discovery, launch, and measurement interface.
2. **Hugging Face-native resolution** — exact Hub model IDs resolve available formats and quantizations without turning metadata into installation trust.
3. **Autopilot** — read-only planning remains the default; opt-in actions may acquire, configure, launch, and verify.
4. **RigGraph** — machine × model × artifact × quantization × runtime × context × measured-performance relationships without collapsing unknowns into guesses.
5. **Benchmark Passports** — reproducible, privacy-filtered measurement records that can later power community evidence and calibration.
6. **Prediction + calibration** — predictions are explicitly separated from measurements and checked against comparable real benchmark evidence.
7. **Public ecosystem** — integrations and adapters should make LLMRig useful to runtime and model communities instead of competing with them.

## v0.8 — Ecosystem foundation — shipped

- Common runtime-adapter contract across oMLX, Ollama, MLX-LM, and llama.cpp.
- First-class oMLX capability detection, provenance-safe local inventory, and explicit measured verification.
- `llmrig runtimes` for read-only runtime capability and readiness evidence.
- Hardened exact Hugging Face artifact classification for format, quantization, context, and provenance.
- Adapter-backed `solve` runtime candidate construction while keeping default solve read-only.
- Explicit `solve --verify` measurement without surprise installation, download, or filesystem scans.

## v0.9 — Autopilot actions — release scope complete

- Deterministic `llmrig plan MODEL` with stable plan IDs and explicit blockers/unknowns.
- Explicit `apply` approval boundary with plan-drift detection.
- Opt-in exact Hugging Face artifact acquisition pinned to repository revision.
- Adapter-owned runtime actions across oMLX, Ollama, MLX-LM, and llama.cpp where safely supported.
- `llmrig run MODEL` convenience flow without bypassing mutation approval.
- Standalone `llmrig verify [RECEIPT]` that re-observes current evidence and refuses mutation.
- Privacy-safe action receipts.
- Local RigGraph evidence persistence after measured verification.
- Prediction and measurement stored separately; calibration deltas exist only for comparable metrics.
- No anonymous/community evidence upload by default.

## v1.0 — Public evidence ecosystem

- Import/export RigGraph evidence and Benchmark Passports through stable public schemas.
- Add opt-in anonymous community evidence submission.
- Calibrate predictions from sufficiently comparable accumulated measurements.
- Expose a stable runtime adapter/plugin API.
- Expand the runtime ecosystem without making LLMRig runtime-specific.
- Publish Hugging Face Space/Dataset/Collection surfaces for reproducible public discovery and evidence.

## Non-negotiable invariants

- Unknown is not false.
- Compatible is not local.
- Local is not executable.
- Executable is not measurable.
- Measurable is not measured.
- Measured is not recommended.
- Measured performance is not model quality.
- Discovery metadata is not installation trust.
- Mutating actions require explicit user intent.
- Private filesystem paths never appear in public results.
- Secrets never appear in plans, receipts, or RigGraph exports.
