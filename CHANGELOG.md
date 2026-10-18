# Changelog

All notable changes to LLMRig will be documented here.

## 0.11.0 - 2026-10-18

### LLMRig Rescue (experimental)
- Ship `llmrig rescue` OpenCode + Ollama preflight with explicitly opted-in
  synthetic tool-call probes and read-only coding-agent tasks in disposable
  workspaces. No existing user project is opened or modified.
- Add privacy-filtered OpenCode JSONL event-sequence diagnostics distinguishing
  observed correct-fix text from agent completion, nonzero exits, timeouts,
  explanation-rubric misses and test-boundary violations.
- Add versioned three-task, paired A/B tests for context declarations and
  automatic-compaction policies with alternating order, equal per-session
  timeout budgets, integrity checks and explicit noncausal limitations.
- Add `rescue-evidence` for offline comparison of explicitly selected JSON
  receipts. Support legacy baseline receipts with missing policy fields
  while disclosing that uncertainty. Never send reports remotely.
- Emit a conservative diagnosis for the repeated pattern of correct-fix
  evidence followed by timeouts under default compaction with successful
  independent compaction-disabled fixture runs. **Never** recommend
  automatically or globally disabling automatic compaction.

### Field evidence and scope
- On one Apple M4 Max 48 GiB with OpenCode 1.19.3, Ollama 0.33.3 and
  local `qwen3.8:27b-mlx`, the controlled 3-task 4K-*declared*-context
  experiment recorded 0/3 strict passes with default compaction (3 timeouts
  after correction evidence), versus 3/3 strict passes with compaction
  explicitly disabled (all 6 contained correct-fix evidence).
- Separately, 3/3 repeated compaction-disabled parity tasks passed in
  28.43 s, 22.36 s and 10.21 s. All observations were reported by one
  machine/user; no independent attestation or statistical claim.
- Effective Ollama per-request context, direct compaction events,
  broader generalization, and a safe persistent real-world mitigation
  remain unverified. Disposable HOME and tool restrictions are **not**
  an operating-system sandbox.
- Preserve Autopilot capabilities, Python 3.9+ support and existing
  opt-in mutation boundaries. No new dependency or silent downloads.

## 0.10.0 - 2026-10-18

### Real Autopilot validation
- Complete end-to-end oMLX and Ollama verification workflows backed by real
  local receipts and versioned public field-run notes. Retain missing metric
  and catalog provenance caveats; these are different models, not a runtime race.
- Provide model-specific read-only Doctor guidance and truthful failure diagnoses.
- Add privacy-filtered Rig Reports (terminal, Markdown, JSON) and a local-only
  interactive viewer, plus the evidence-backed offline Plan Explorer.
- Display the exact whitelisted Ollama artifact tag for reproducibility.

### Reliability and measurement
- Improve safe, revision-pinned Hugging Face acquisition and apply receipts.
- Suppress misleading Ollama prompt-throughput values from sub-millisecond
  evaluations instead of advertising cache-inflated rates.
- Clarify oMLX startup errors, approval boundaries and local authentication.
- Document installation recovery and honest unverified metrics.

### Distribution and documentation
- Add illustrated, clearly labeled real-run README walkthrough.
- Cover isolated pipx installs, clean virtual environments, macOS, Linux and
  Windows, and distinguish published packages from unreleased source.
- Smoke-test real pipx wheel installs from outside the checkout and maintain
  the Python 3.9/3.14 cross-platform CI matrix.
- No new runtime dependencies, automatic uploads or silent installations.

## 0.9.3 - 2026-10-16

### GGUF planning and compatibility

- Treat split GGUF shard sets as one logical model artifact instead of one
  candidate per shard.
- Require complete shard sets before trusting aggregate artifact size; incomplete
  sets stay explicitly unknown.
- Use aggregate GGUF size in the existing conservative machine-fit reasoning so
  oversized configurations can be ruled out before model-weight download.
- Exclude GGUF imatrix calibration artifacts from runnable model candidates.
- Preserve the memory-fit blocker independently of runtime installation state.

### Autopilot output

- Add a `Fit` column to `llmrig plan` so compatibility is visible alongside
  local availability and executability.
- Explain plans where every discovered configuration exceeds the conservative
  model-weight budget instead of reducing the result to a generic inconclusive
  message.

### Validation

- Add regression coverage for complete and incomplete split GGUF sets, shard-set
  acquisition, imatrix filtering, compatibility rendering, and memory-fit plan
  explanations.
- Validate the Beam 501B A23B GGUF repository on an Apple M4 Max with 48 GiB
  unified memory without downloading model weights.
- 328 local tests passed, with cross-platform CI and CodeQL green before release.


## 0.9.2 - 2026-09-25

### Documentation / Release

- Tighten the README opening so new users can understand LLMRig's role faster.
- Add real v0.9 Apple Silicon verification measurements from the public
  Benchmark Passports evidence instead of placeholder benchmark values.
- Republish the improved README as the PyPI long description.
- No runtime, planning, execution, verification, RigGraph, or benchmark
  behavior changes.

## 0.9.1 - 2026-09-24

### Release / Documentation

- Add the public Hugging Face Autopilot Space, Benchmark Passports dataset,
  Collection, and launch discussion to the README and package metadata.
- Align the PyPI-facing README and project metadata with the public v0.9
  Hugging Face launch surfaces.
- Update PyPI release smoke validation for the bundled `huggingface_hub`
  dependency and retain manual recovery support for an existing release tag.
- No runtime, planning, execution, verification, RigGraph, or benchmark
  behavior changes.

## 0.9.0 - 2026-09-24

### Autopilot

- Add deterministic `llmrig plan MODEL` with stable plan IDs, blockers,
  unknowns, candidate execution paths, and planned verification.
- Add explicit `llmrig apply MODEL --plan-id ...` with interactive or
  explicit non-interactive approval and fail-closed plan-drift detection.
- Add `llmrig run MODEL` as the convenience plan → approve → apply → verify
  workflow without bypassing mutation approval.
- Add standalone `llmrig verify [RECEIPT]` that re-observes current evidence,
  refuses mutation, and performs fresh measured verification.

### Acquisition and runtime actions

- Add opt-in exact Hugging Face artifact acquisition pinned to the resolved
  repository revision.
- Keep private download/execution locators out of public plans and receipts.
- Add adapter-owned runtime action capabilities for oMLX, Ollama, MLX-LM,
  and llama.cpp where safely supported.
- Leave unsupported mutation explicit instead of silently emulating it.

### Receipts, RigGraph, and calibration

- Emit privacy-safe deterministic action receipts for applied workflows.
- Persist successful measured verification into a local RigGraph evidence
  store across machine, model, artifact, quantization, runtime, context,
  and measurement dimensions.
- Keep prediction and measurement as separate facts.
- Compute calibration deltas only when predicted and measured dimensions
  are genuinely comparable; missing metrics remain unknown.

### Safety and evidence

- Preserve read-only planning as the default.
- Preserve the evidence boundaries: unknown is not false; compatible is not
  local; local is not executable; executable is not measurable; measurable
  is not measured; measured is not recommended; measured performance is not
  model quality.
- Require explicit user intent for every mutating workflow.
- Keep private filesystem paths and secrets out of public plans, receipts,
  and RigGraph evidence.


## 0.8.1 - 2026-09-24

### Fixed / Documentation

- Replace repository-relative README documentation links with absolute GitHub URLs so the PyPI long description no longer sends users to `pypi.org/...` 404 pages.
- Add regression coverage that rejects non-portable relative Markdown and HTML links in the PyPI-facing README.
- No runtime, solve, benchmark, evidence, or Autopilot behavior changes.

## 0.8.0 - 2026-09-24

### Runtime intelligence

- Add `llmrig runtimes` and `llmrig runtimes --json` for read-only inspection of
  oMLX, Ollama, MLX-LM, and llama.cpp installation, readiness, formats,
  interfaces, blockers, and unknowns.
- Unify runtime readiness, capability, and verification routing behind the v0.8
  runtime-adapter registry instead of duplicating runtime-specific support tables.
- Keep probing read-only: LLMRig does not install runtimes, start services,
  download model weights, or scan arbitrary filesystem locations.

### oMLX

- Add first-class authenticated oMLX observation and execution through its
  OpenAI-compatible local API, honoring `OMLX_API_KEY` without serializing
  credentials into public output.
- Associate an API-visible oMLX model with an exact Hugging Face repository only
  from defensible provenance: exact runtime source metadata, or an exact and
  unambiguous completed oMLX Hugging Face download record.
- Join provenance-backed local oMLX inventory with the canonical resolved MLX
  candidate while keeping the private runtime model locator out of public output.
- Extend `solve --verify` to measure already-local oMLX candidates with `race-v2`.
  Prefer server-reported per-phase timing and throughput; when oMLX reports only
  server `total_time`, record latency only and leave unavailable throughput
  dimensions unknown instead of synthesizing values.

### Hugging Face evidence hardening

- Harden exact Hub artifact classification for GGUF, MLX, and generic
  safetensors without turning discovery metadata into installation trust.
- Keep an `mlx` tag or path hint by itself as a hint, not proof of MLX packaging.
  Promote safetensors packaging to MLX only from stronger structured evidence,
  such as explicit `library_name=mlx` or an MLX hint backed by the MLX-LM
  quantization contract.
- Preserve 4-bit and context metadata from exact repository configuration when
  the model API omits needed structured fields, without downloading weights.
- Fail closed on conflicting context, quantization, base-model provenance, or
  incomplete/ambiguous safetensors shard sets.
- Keep generic safetensors runtime compatibility unknown unless stronger
  evidence exists.

### Evidence, privacy, and reliability

- Preserve the evidence invariants: unknown is not false; compatible is not
  local; local is not executable; executable is not measurable; measurable is
  not measured; measured performance is not model quality.
- Keep private filesystem paths, API keys, session cookies, and private execution
  locators out of public solve and runtime output.
- Preserve deterministic, privacy-safe solve schemas while expanding runtime and
  Hub evidence behind the existing CLI and SDK surfaces.
- Validate the release across macOS, Linux, and Windows on Python 3.9 and 3.14,
  including wheel/sdist installation and CodeQL checks.

## 0.7.0 - 2026-09-06

### Added

- Add the Autopilot foundation with orthogonal discovery, compatibility,
  runtime-availability, local-availability, execution, measurement, and
  recommendation states; unknown facts remain explicit.
- Add read-only local inventory and `llmrig solve MODEL` planning. Default solve
  does not download models, install or start runtimes, execute inference, or
  write benchmark output.
- Add explicit `llmrig solve MODEL --verify`, which measures at least two
  comparable, already-local candidates through the unchanged `race-v2`
  workload and balanced Pareto decision semantics.
- Add a minimal Python SDK centered on `llmrig.solve(...) -> SolveResult`, with
  deliberate input and engine exception contracts.

### Evidence, privacy, and release quality

- Keep public solve recipes and results free of native artifact paths while
  retaining private execution locators only at the inventory-to-runtime seam.
- Distinguish planning recommendations from measured verification outcomes,
  including unavailable, failed, measured, and inconclusive states with
  provenance.
- Harden release packaging, CI artifact installation checks, publication
  version gates, security guidance, and project metadata for the 0.7.0 release.

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
