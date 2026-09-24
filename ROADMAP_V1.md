# LLMRig v1 Direction

LLMRig is evolving from a compatibility CLI into an evidence-driven Autopilot for local AI.

## North star

Given a model and a machine, LLMRig should determine viable execution paths across model artifact, quantization, runtime, context, and measured performance; explain its evidence; and, only with explicit permission, carry out the selected setup.

## Product pillars

1. **Runtime adapters** — Ollama, MLX-LM, oMLX, llama.cpp, and future runtimes expose a common capability, discovery, launch, and measurement interface.
2. **Hugging Face-native resolution** — exact Hub model IDs resolve available formats and quantizations without turning metadata into installation trust.
3. **Autopilot** — read-only planning remains the default; opt-in actions may acquire, configure, launch, and verify.
4. **RigGraph** — model hardware × model × artifact × quantization × runtime × context × measured-performance relationships without collapsing unknowns into guesses.
5. **Benchmark Passports** — reproducible, privacy-filtered measurement records that can later power community evidence and calibration.
6. **Prediction + calibration** — predictions are explicitly separated from measurements and continuously checked against real benchmark evidence.
7. **Public ecosystem** — integrations and adapters should make LLMRig useful to runtime and model communities instead of competing with them.

## v0.8 — Ecosystem foundation

- Introduce a stable runtime-adapter contract.
- Add first-class oMLX capability detection.
- Move existing Ollama / MLX-LM / llama.cpp knowledge behind adapters incrementally.
- Improve exact Hugging Face model resolution and artifact classification.
- Add `llmrig runtimes` to explain local runtime capabilities and evidence.
- Keep `solve` read-only unless the user explicitly chooses an action.

## v0.9 — Autopilot actions

- Add explicit plan/apply separation.
- Support opt-in artifact acquisition for trusted, evidenced sources.
- Configure and launch supported runtimes.
- Verify selected configurations with unchanged, comparable workloads.
- Emit machine-readable action receipts.

## v1.0 — RigGraph

- Persist compatibility and performance evidence as graph-shaped records.
- Import/export Benchmark Passports.
- Add opt-in anonymous community evidence submission.
- Add prediction-versus-measurement calibration.
- Expose a stable SDK and adapter/plugin API.

## Non-negotiable invariants

- Unknown is not false.
- Compatible is not local.
- Local is not executable.
- Executable is not measured.
- Measured performance is not model quality.
- Discovery metadata is not installation trust.
- Mutating actions require explicit user intent.
- Private filesystem paths never appear in public results.
